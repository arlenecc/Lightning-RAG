import Foundation
import MLX

/// jina-embeddings-v5-text-nano（EuroBERT-210M）的 MLX int4 推理实现。
///
/// 结构对应模型目录里的 `model.py`：量化 embedding → 12 层 pre-norm
/// Transformer（RoPE + 无 bias MHA、SwiGLU MLP、RMSNorm）→ last-token pooling。
/// 与 v3 不同，V5 没有 task LoRA / token_type embedding / q_norm / k_norm，
/// 任务类型完全由输入前缀（`Query: ` / `Document: `）表达。
///
/// 输出按 provider 配置的 Matryoshka 维度截断（默认 128）并重新 L2 归一化。
final class JinaV5MLXModel: @unchecked Sendable {
    // MARK: - 配置

    struct Configuration: Sendable {
        let hiddenSize: Int
        let layerCount: Int
        let headCount: Int
        let keyValueHeadCount: Int
        let intermediateSize: Int
        let headDimension: Int
        let vocabularySize: Int
        let maxPositionEmbeddings: Int
        let ropeTheta: Float
        let rmsNormEps: Float
        let groupSize: Int
        let bits: Int
        let quantizationMode: String

        static func load(from directory: URL) throws -> Configuration {
            guard let config = JinaV5Profile.readConfig(directory: directory) else {
                throw LocalModelError.invalidConfiguration(
                    "Jina v5 模型缺少可解析的 config.json（目录：\(directory.path)）")
            }
            guard JinaV5Profile.isJinaV5(directory: directory) else {
                throw LocalModelError.unsupportedArchitecture(
                    "不是 jina-embeddings-v5 模型目录：\(directory.path)")
            }

            let quantization = config["quantization"]?.objectValue
                ?? config["quantization_config"]?.objectValue
            let hidden = config["hidden_size"]?.intValue ?? JinaV5Profile.nativeDimension
            let heads = config["num_attention_heads"]?.intValue ?? 12
            let headDimension = config["head_dim"]?.intValue
                ?? (heads > 0 ? hidden / heads : 64)

            return Configuration(
                hiddenSize: hidden,
                layerCount: config["num_hidden_layers"]?.intValue ?? 12,
                headCount: heads,
                keyValueHeadCount: config["num_key_value_heads"]?.intValue ?? heads,
                intermediateSize: config["intermediate_size"]?.intValue ?? 3_072,
                headDimension: headDimension,
                vocabularySize: config["vocab_size"]?.intValue ?? 128_256,
                maxPositionEmbeddings: config["max_position_embeddings"]?.intValue ?? 8_192,
                ropeTheta: Float(config["rope_theta"]?.doubleValue ?? 1_000_000),
                rmsNormEps: Float(config["rms_norm_eps"]?.doubleValue ?? 1e-5),
                groupSize: quantization?["group_size"]?.intValue ?? 32,
                bits: quantization?["bits"]?.intValue ?? 4,
                quantizationMode: quantization?["mode"]?.stringValue ?? "affine"
            )
        }
    }

    // MARK: - 权重模块

    /// int4 量化 Linear：权重为 packed U32，scales/biases 为 BF16；
    /// V5 的 attention / MLP 都没有 bias。
    private struct QuantizedLinear: @unchecked Sendable {
        let weight: MLXArray
        let scales: MLXArray
        let biases: MLXArray
        let groupSize: Int
        let bits: Int
        let mode: QuantizationMode

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            quantizedMM(
                x, weight,
                scales: scales, biases: biases,
                transpose: true,
                groupSize: groupSize, bits: bits, mode: mode
            ).asType(.float32)
        }
    }

    private struct Attention: @unchecked Sendable {
        let queryProjection: QuantizedLinear
        let keyProjection: QuantizedLinear
        let valueProjection: QuantizedLinear
        let outputProjection: QuantizedLinear

        func callAsFunction(
            _ x: MLXArray,
            additiveMask: MLXArray,
            configuration: Configuration
        ) -> MLXArray {
            let batch = x.dim(0)
            let sequence = x.dim(1)
            let heads = configuration.headCount
            let kvHeads = configuration.keyValueHeadCount
            let headDimension = configuration.headDimension

            var queries = queryProjection(x)
                .reshaped([batch, sequence, heads, headDimension])
                .transposed(0, 2, 1, 3)
            var keys = keyProjection(x)
                .reshaped([batch, sequence, kvHeads, headDimension])
                .transposed(0, 2, 1, 3)
            let values = valueProjection(x)
                .reshaped([batch, sequence, kvHeads, headDimension])
                .transposed(0, 2, 1, 3)

            // traditional=false 即 rotate_half 约定（与 PyTorch / model.py 一致）。
            queries = RoPE(
                queries, dimensions: headDimension, traditional: false,
                base: configuration.ropeTheta, scale: 1, offset: 0)
            keys = RoPE(
                keys, dimensions: headDimension, traditional: false,
                base: configuration.ropeTheta, scale: 1, offset: 0)

            var expandedKeys = keys
            var expandedValues = values
            if kvHeads != heads, kvHeads > 0 {
                let repetition = heads / kvHeads
                expandedKeys = repeated(expandedKeys, count: repetition, axis: 1)
                expandedValues = repeated(expandedValues, count: repetition, axis: 1)
            }

            let scale = 1 / Float(headDimension).squareRoot()
            var scores = matmul(queries * scale, expandedKeys.transposed(0, 1, 3, 2))
            scores = scores + additiveMask
            let attention = softmax(scores, axis: -1)
            let attended = matmul(attention, expandedValues)
                .transposed(0, 2, 1, 3)
                .reshaped([batch, sequence, heads * headDimension])
            return outputProjection(attended)
        }
    }

    private struct MLP: @unchecked Sendable {
        let gateProjection: QuantizedLinear
        let upProjection: QuantizedLinear
        let downProjection: QuantizedLinear

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            // SwiGLU：silu(gate(x)) * up(x)；silu(v) = v * sigmoid(v)。
            let gate = gateProjection(x)
            let activated = gate * sigmoid(gate)
            return downProjection(activated * upProjection(x))
        }
    }

    private struct Layer: @unchecked Sendable {
        let attention: Attention
        let mlp: MLP
        let inputNormWeight: MLXArray
        let postAttentionNormWeight: MLXArray

        func callAsFunction(
            _ x: MLXArray,
            additiveMask: MLXArray,
            configuration: Configuration
        ) -> MLXArray {
            // pre-norm：x + MHA(RMSNorm(x))，再 x + MLP(RMSNorm(x))。
            let afterAttention = x + attention(
                rmsNorm(x, weight: inputNormWeight, eps: configuration.rmsNormEps),
                additiveMask: additiveMask,
                configuration: configuration
            )
            return afterAttention + mlp(
                rmsNorm(
                    afterAttention,
                    weight: postAttentionNormWeight,
                    eps: configuration.rmsNormEps
                )
            )
        }
    }

    // MARK: - 状态

    let configuration: Configuration

    private let tokenEmbedding: MLXArray
    private let tokenScales: MLXArray
    private let tokenBiases: MLXArray
    private let finalNormWeight: MLXArray
    private let layers: [Layer]

    init(directory: URL) throws {
        let configuration = try Configuration.load(from: directory)
        guard let weightsURL = JinaV5Profile.weightsURL(in: directory) else {
            throw LocalModelError.runtime(
                "Jina v5 缺少 MLX 权重 weights.safetensors / model.safetensors"
                    + "（目录：\(directory.path)）")
        }

        let weights: [String: MLXArray]
        do {
            // safetensors 的 Load primitive 仅在 CPU 实现；默认流就是 CPU。
            weights = try MLX.loadArrays(url: weightsURL)
        } catch {
            throw LocalModelError.runtime(
                "Jina v5 权重加载失败（\(weightsURL.path)）：\(error.localizedDescription)")
        }

        let mode = QuantizationMode(rawValue: configuration.quantizationMode) ?? .affine
        let root = "model"

        self.configuration = configuration
        self.tokenEmbedding = try Self.require(weights, "\(root).embed_tokens.weight")
        self.tokenScales = try Self.require(weights, "\(root).embed_tokens.scales")
        self.tokenBiases = try Self.require(weights, "\(root).embed_tokens.biases")
        self.finalNormWeight = try Self.require(weights, "\(root).norm.weight")
            .asType(.float32)

        self.layers = try (0 ..< configuration.layerCount).map { index in
            let prefix = "\(root).layers.\(index)"
            let attention = prefix + ".self_attn"
            let mlp = prefix + ".mlp"
            func linear(_ name: String) throws -> QuantizedLinear {
                try Self.makeLinear(
                    attention + "." + name, weights: weights,
                    configuration: configuration, mode: mode)
            }
            return Layer(
                attention: Attention(
                    queryProjection: try linear("q_proj"),
                    keyProjection: try linear("k_proj"),
                    valueProjection: try linear("v_proj"),
                    outputProjection: try linear("o_proj")
                ),
                mlp: MLP(
                    gateProjection: try Self.makeLinear(
                        mlp + ".gate_proj", weights: weights,
                        configuration: configuration, mode: mode),
                    upProjection: try Self.makeLinear(
                        mlp + ".up_proj", weights: weights,
                        configuration: configuration, mode: mode),
                    downProjection: try Self.makeLinear(
                        mlp + ".down_proj", weights: weights,
                        configuration: configuration, mode: mode)
                ),
                inputNormWeight: try Self.require(
                    weights, prefix + ".input_layernorm.weight").asType(.float32),
                postAttentionNormWeight: try Self.require(
                    weights, prefix + ".post_attention_layernorm.weight").asType(.float32)
            )
        }
    }

    // MARK: - 推理

    struct JinaV5MLXEncodeResult: Sendable {
        let vectors: [[Float]]
        let nativeVectors: [[Float]]
    }

    /// 双向编码 + last-token pooling，返回前 `dimension` 维、已重新 L2 归一化的向量。
    ///
    /// - Parameters:
    ///   - inputIDs / attentionMask: 等长的 batch（右 padding）。
    ///   - dimension: Matryoshka 输出维度；大于原生维时按原生维处理。
    func encode(inputIDs: [[Int]], attentionMask: [[Int]],
                dimension: Int) -> JinaV5MLXEncodeResult {
        guard let first = inputIDs.first, !first.isEmpty else {
            return JinaV5MLXEncodeResult(vectors: [], nativeVectors: [])
        }
        let batch = inputIDs.count
        let ids = MLXArray(inputIDs.map { row in row.map(Int32.init) })
        let mask = attentionMask.map { row in row.map { Int32($0 == 0 ? 0 : 1) } }

        var hidden = dequantized(
            tokenEmbedding[ids],
            scales: tokenScales[ids], biases: tokenBiases[ids],
            groupSize: configuration.groupSize, bits: configuration.bits,
            mode: QuantizationMode(rawValue: configuration.quantizationMode) ?? .affine,
            dtype: .float32
        )

        // (B,S) 的 0/1 mask → (B,1,1,S) 的加性 mask（与 model.py 的 -1e9 一致）。
        let additive = MLXArray(attentionMask.map { row in
            row.map { Float32($0 == 0 ? -1e9 : 0) }
        }).expandedDimensions(axes: [1, 2])

        for layer in layers {
            hidden = layer(
                hidden, additiveMask: additive, configuration: configuration)
        }
        hidden = rmsNorm(
            hidden, weight: finalNormWeight, eps: configuration.rmsNormEps)

        // last-token pooling：右 padding 下最后一个真实 token 的下标是 mask 之和 - 1。
        let lastIndices = mask.map { row in
            max(0, row.reduce(0) { $0 + $1 } - 1)
        }
        let batchIndices = MLXArray((0 ..< batch).map { Int32($0) })
        let lengthIndices = MLXArray(lastIndices.map { Int32($0) })
        var pooled = hidden[batchIndices, lengthIndices]
        // 二值签名只能从截断前的原生 768 维生成；归一化不改变符号。
        let nativeVectors = pooled.map { $0.asArray(Float.self) }

        // Matryoshka 截断：取前 N 维后必须重新归一化（不是复用 768 维的范数）。
        let target = max(1, min(dimension, pooled.dim(-1)))
        if pooled.dim(-1) > target {
            pooled = pooled[.ellipsis, 0 ..< target]
        }
        let normalized = pooled.l2Normalized(axis: -1, eps: 1e-12)
        normalized.eval()
        return JinaV5MLXEncodeResult(
            vectors: normalized.map { $0.asArray(Float.self) },
            nativeVectors: nativeVectors)
    }

    // MARK: - 权重装配

    private static func require(
        _ weights: [String: MLXArray], _ key: String
    ) throws -> MLXArray {
        guard let value = weights[key] else {
            throw LocalModelError.runtime("Jina v5 权重缺少 `\(key)`")
        }
        return value
    }

    private static func makeLinear(
        _ prefix: String,
        weights: [String: MLXArray],
        configuration: Configuration,
        mode: QuantizationMode
    ) throws -> QuantizedLinear {
        QuantizedLinear(
            weight: try require(weights, prefix + ".weight"),
            scales: try require(weights, prefix + ".scales"),
            biases: try require(weights, prefix + ".biases"),
            groupSize: configuration.groupSize,
            bits: configuration.bits,
            mode: mode
        )
    }
}
