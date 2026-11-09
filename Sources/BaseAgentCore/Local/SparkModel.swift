import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXLLM

// 参考实现见仓库根目录 `spark/SparkModel.swift`；本文件是编入
// BaseAgentCore 的正式版本，由 `MLXLLMService` 在加载容器前调用
// `SparkModel.register()` 注册 `spark2_5` 架构（Spark-X2.6 4B MLX）。

// MARK: - Configuration

/// Mirrors the `config.json` of the custom `spark2_5` (Spark2_5ForCausalLM) architecture.
///
/// Only the fields the model needs are decoded; everything else in `config.json`
/// (e.g. `transformers_version`, `use_cache`) is ignored by `Codable`.
public struct Spark2_5Configuration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var tieWordEmbeddings: Bool
    var slidingWindow: Int
    var layerTypes: [String]
    var ropeParameters: [String: RoPEParameters]
    var padTokenId: Int?
    var bosTokenId: Int?
    var eosTokenId: Int?
    var maxPositionEmbeddings: Int?

    public struct RoPEParameters: Codable, Sendable {
        var partialRotaryFactor: Double
        var ropeTheta: Double

        enum CodingKeys: String, CodingKey {
            case partialRotaryFactor = "partial_rotary_factor"
            case ropeTheta = "rope_theta"
        }
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case padTokenId = "pad_token_id"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case maxPositionEmbeddings = "max_position_embeddings"
    }
}

// MARK: - Attention

class SparkAttention: Module {
    let args: Spark2_5Configuration
    let scale: Float
    let qDim: Int
    let kvDim: Int
    let isSliding: Bool

    @ModuleInfo(key: "q_k_v_proj") var qkvProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let rope: RoPELayer

    public init(_ args: Spark2_5Configuration, layerType: String) {
        self.args = args
        self.scale = pow(Float(args.headDim), -0.5)
        self.qDim = args.attentionHeads * args.headDim
        self.kvDim = args.kvHeads * args.headDim
        self.isSliding = (layerType == "sliding_attention")

        _qkvProj.wrappedValue = Linear(args.hiddenSize, qDim + 2 * kvDim, bias: false)
        // g_proj is intentionally left unquantized in the checkpoint (no .scales/.biases),
        // so declare it as a plain Linear; the loader keeps it as-is.
        _gProj.wrappedValue = Linear(args.hiddenSize, args.attentionHeads, bias: false)
        _outProj.wrappedValue = Linear(args.attentionHeads * args.headDim, args.hiddenSize, bias: false)

        // Per-layer RoPE: sliding layers rotate the full head_dim, full layers only a
        // fraction (partial_rotary_factor) with a different base. MLX.RoPE rotates the
        // first `dimensions` feature dims and leaves the rest unchanged, which is exactly
        // the Python partial rotary embedding.
        let ropeParams = args.ropeParameters[layerType] ?? args.ropeParameters["full_attention"]!
        let ropeHeadDim = Int(Double(args.headDim) * ropeParams.partialRotaryFactor)
        let base = Float(ropeParams.ropeTheta)
        self.rope = RoPE(dimensions: ropeHeadDim, traditional: false, base: base, scale: 1.0)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        // Headwise sigmoid gating is computed from the (already normalized) input and
        // applied to the attention output *before* the output projection.
        let gateScore = gProj(x)  // (B, L, n_heads)

        // Fused QKV projection.
        let qkv = qkvProj(x)  // (B, L, qDim + 2*kvDim)
        let q = qkv[.ellipsis, 0 ..< qDim]
        let k = qkv[.ellipsis, qDim ..< qDim + kvDim]
        let v = qkv[.ellipsis, qDim + kvDim ..< qDim + 2 * kvDim]

        var queries = q.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        var keys = k.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        let values = v.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let attn = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )  // (B, n_heads, L, head_dim)

        let g = sigmoid(gateScore.asType(.float32))
            .reshaped(B, L, args.attentionHeads, 1)
            .transposed(0, 2, 1, 3)  // (B, n_heads, L, 1)
        let gated = attn * g

        let output = gated.transposed(0, 2, 1, 3).reshaped(B, L, -1)  // (B, L, n_heads*head_dim)
        return outProj(output)
    }
}

// MARK: - MLP (GEGLU: exact erf gelu)

class SparkMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(gelu(gate(x)) * up(x))
    }
}

// MARK: - Decoder layer

class SparkDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SparkAttention
    let mlp: SparkMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let isSliding: Bool
    let slidingWindow: Int

    public init(_ args: Spark2_5Configuration, layerType: String) {
        self.isSliding = (layerType == "sliding_attention")
        self.slidingWindow = args.slidingWindow
        _attention.wrappedValue = SparkAttention(args, layerType: layerType)
        self.mlp = SparkMLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        // Sliding-window layers use a windowed causal mask; full-attention layers use a
        // plain causal mask. The sliding cache is created accordingly in `newCache`.
        let windowSize = isSliding ? slidingWindow : nil
        let mask = createAttentionMask(h: x, cache: cache, windowSize: windowSize)

        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }
}

// MARK: - Inner model (everything under the `model.` prefix in the checkpoint)

public class SparkModelInner: Module {
    @ModuleInfo(key: "embedding") var embedding: QuantizedEmbedding
    fileprivate let layers: [SparkDecoderLayer]
    let norm: RMSNorm

    public init(_ args: Spark2_5Configuration) {
        // The embedding is shipped int4-quantized (it carries .scales/.biases), so it must
        // be declared as QuantizedEmbedding. The loader fills weight/scales/biases from the
        // checkpoint; it will not be re-quantized because it is already a Quantized layer.
        let embedding = Embedding(embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        _embedding.wrappedValue = QuantizedEmbedding(embedding, groupSize: 64, bits: 4, mode: .affine)

        self.layers = (0 ..< args.hiddenLayers).map { i in
            SparkDecoderLayer(args, layerType: args.layerTypes[i])
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedding(inputs).asType(.float32)
        for (i, layer) in layers.enumerated() {
            h = layer(h, cache: cache?[i])
        }
        return norm(h)
    }
}

// MARK: - Top-level model (LLMModel)

/// `spark2_5` / `Spark2_5ForCausalLM` — a custom architecture not shipped with mlx-swift-lm.
///
/// Registered at runtime via ``register()`` so `loadModelContainer` can build it. The root
/// module exposes a single `model` child (matching the `model.*` checkpoint prefix); the
/// tied LM head reuses the embedding via `asLinear`, so there is no `lm_head` module.
public class SparkModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let configuration: Spark2_5Configuration

    @ModuleInfo(key: "model") var model: SparkModelInner

    public init(_ args: Spark2_5Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        _model.wrappedValue = SparkModelInner(args)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let h = model(inputs, cache: cache)
        // Tied word embeddings: logits = hidden @ embedding.weightᵀ.
        return model.embedding.asLinear(h)
    }

    /// Sliding-window layers get a windowed KV cache; full-attention layers get a standard one.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try configuration.layerTypes.map { layerType in
            if layerType == "sliding_attention" {
                return try makeSlidingWindowKVCache(
                    parameters: parameters, window: configuration.slidingWindow)
            } else {
                return try makeAttentionKVCache(parameters: parameters)
            }
        }
    }

    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Registration

extension SparkModel {
    /// Register the `spark2_5` model type with the shared model registry so that
    /// `loadModelContainer(from:using:)` can build it. Safe to call repeatedly; it
    /// overwrites the entry for the same model type.
    public static func register() async {
        await LLMTypeRegistry.shared.registerModelType("spark2_5") { data in
            let config = try JSONDecoder().decode(Spark2_5Configuration.self, from: data)
            return SparkModel(config)
        }
    }
}
