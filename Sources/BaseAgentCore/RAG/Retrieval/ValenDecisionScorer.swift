import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

/// BBox 分段注意力掩码缓存。
///
/// Valen-Decision 的官方推理约定：prefill 时全注意力层使用 bbox_blocks 掩码
/// （prefix 行只见 prefix；后缀行见 prefix + 公共胶水 seg0 + 本段，段内双向）。
/// Python 侧通过 monkeypatch `_create_qwen3_5_attention_mask` 注入；Swift 侧
/// qwen3_5 的掩码来自 `cache.makeMask(...)`（Qwen35 forward 委托
/// `cacheArray?[faIdx]`），因此只需在 faIdx 位置放一个重写 `makeMask` 的缓存。
/// 直接实现 KVCache 协议（BaseKVCache 的 init 为 internal，无法跨模块继承；
/// KVCacheSimple 为 public 非 open，同样不可继承）。
final class BBoxMaskCache: KVCache {
    /// [1, 1, T, T] bool，true = 可见（与 createCausalMask 的语义一致）。
    let mask: MLXArray
    private(set) var offset = 0
    private var keys: MLXArray?
    private var values: MLXArray?

    init(mask: MLXArray) {
        self.mask = mask
    }

    var maxSize: Int? { nil }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if let k = self.keys, let v = self.values {
            self.keys = concatenated([k, keys], axis: 2)
            self.values = concatenated([v, values], axis: 2)
        } else {
            self.keys = keys
            self.values = values
        }
        offset += keys.dim(2)
        return (self.keys!, self.values!)
    }

    var state: [MLXArray] {
        get { [keys, values].compactMap { $0 } }
        set {
            guard newValue.count >= 2 else { return }
            keys = newValue[0]
            values = newValue[1]
            offset = keys!.dim(2)
        }
    }

    var metaState: [String] {
        get { [""] }
        set {}
    }

    var isTrimmable: Bool { false }

    func trim(_ n: Int) -> Int { 0 }

    func copy() -> any KVCache { BBoxMaskCache(mask: mask) }

    func innerState() -> [MLXArray] { [keys, values].compactMap { $0 } }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode {
        .array(mask)
    }
}

/// Valen-Decision-1.5 决策模型打分器。
///
/// 「Qwen3.5 hybrid backbone + fp32 MixerHead（decision_head.safetensors，未量化）」：
/// 从 prompt 中池化 4 类角色向量（context/instruction/candidate/decision），
/// 经 LN+投影+roles 与 2 个 MixerBlock 后读 candidate 行得各候选 logit。
/// 推理时需要 bbox_blocks 分段掩码（经 `BBoxMaskCache` 注入，见上）。
/// 提示格式为官方 compiler schema：chat 模板包 state + `Question 1 / Task /
/// choice / Candidates` 块 + `Question 1 Decision:` 探针。
///
/// Swift 侧用 VLMModelFactory 加载 backbone（model_type=qwen3_5），
/// 通过 mtpEmitFlagKey / mtpLastHiddenStatesKey 取 prefill 全序列 hidden。
/// 验收：与模型自带 baseline.npz 对齐（top-1 一致、概率偏差 < 0.02）。
public final class ValenDecisionScorer: @unchecked Sendable {
    /// Valen-Decision-1.5 的本机默认目录（用户提供的外置 SSD 路径）。
    public static let defaultModelDir =
        "/Volumes/SSD/Decision Model/Valen-Decision-1.5-0.8B-mlx-int4"

    private let lock = NSLock()
    private var container: ModelContainer?
    private var headWeights: [String: MLXArray]?
    private var temperature: Double = 1.0
    private let modelDir: URL

    public init(modelDir: String) {
        self.modelDir = URL(fileURLWithPath: modelDir, isDirectory: true)
    }

    public var isReady: Bool {
        lock.lock(); defer { lock.unlock() }
        return container != nil && headWeights != nil
    }

    /// 决策门专用权重是否齐全（无需加载）：MixerHead 权重文件是 Valen 特有的必需文件。
    public var isModelAvailable: Bool {
        FileManager.default.fileExists(
            atPath: modelDir.appendingPathComponent("decision_head.safetensors").path)
    }

    /// 加载 backbone 与 MixerHead（幂等）。
    public func load() async throws {
        lock.lock()
        if container != nil && headWeights != nil { lock.unlock(); return }
        lock.unlock()

        let headURL = modelDir.appendingPathComponent("decision_head.safetensors")
        guard FileManager.default.fileExists(atPath: headURL.path) else {
            throw AgentError.llm("决策模型缺少 decision_head.safetensors: \(modelDir.path)")
        }
        let weights = try MLX.loadArrays(url: headURL)
        guard weights["head.projection.weight"] != nil,
              weights["head.output.1.weight"] != nil else {
            throw AgentError.llm("决策模型 decision_head 缺少关键张量")
        }
        var loadedTemperature = 1.0
        if let data = try? Data(contentsOf: modelDir.appendingPathComponent("decision_head.json")),
           let cfg = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let t = cfg["temperature"] as? Double, t.isFinite {
            loadedTemperature = min(max(t, 0.01), 10)
        }
        let container = try await VLMModelFactory.shared.loadContainer(
            from: modelDir, using: LocalTokenizerLoader())

        lock.lock()
        self.headWeights = weights
        self.temperature = loadedTemperature
        self.container = container
        lock.unlock()
    }

    // MARK: - 提示构建（复刻 example_decision.py build_prompt 单题路径）

    struct ValenQuestion {
        /// 判定问句（Task: choice 的 instructions）。
        var instructions: String
        /// 候选码与描述，顺序即 logits/probs 顺序（judgeBinary 固定 [false, true]）。
        var keys: [String]
        var descs: [String]
    }

    struct ValenPrompt {
        let tokens: [Int]
        let prefixLen: Int
        /// context 角色池化区间（官方 build_prompt 为 (0, base_len)）。
        let contextSpan: (Int, Int)
        let instructionSpan: (Int, Int)
        let candidateSpans: [(Int, Int)]
        let decisionPosition: Int
    }

    /// 把正文里出现的模板特殊 token 尖括号替换为全角（与 MeteorDecisionScorer 一致）。
    private static func escape(_ text: String) -> String {
        var result = text
        for token in ["<|fim_suffix|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>",
                      "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"] {
            let escaped = token.replacingOccurrences(of: "<", with: "＜")
                               .replacingOccurrences(of: ">", with: "＞")
            result = result.replacingOccurrences(of: token, with: escaped)
        }
        return result
    }

    /// 构建单题 prompt。分片编码在「空白边界」切开（lead 不含尾随空格、desc 以空格
    /// 开头），与 BPE 预分词边界一致，拼接结果与官方联合编码逐 token 等价
    /// （测试 ValenDecisionScorerTests 用 baseline.npz 的 ids 逐位校验）。
    static func buildPrompt(tokenizer: Tokenizer, state: String,
                            question: ValenQuestion) throws -> ValenPrompt {
        // 官方 build_prompt 用 add_generation_prompt=False：不能带 assistant 生成提示，
        // 否则会在后缀前多出一段 "<|im_start|>assistant\n<think>…"。协议只暴露
        // 3 参重载（固定 true），需经 MLXTokenizerBridge 的全量重载显式传 false。
        let base: [Int]
        if let bridge = tokenizer as? MLXTokenizerBridge {
            base = try bridge.applyChatTemplate(
                messages: [["role": "user", "content": state]], tools: nil,
                additionalContext: ["enable_thinking": false],
                addGenerationPrompt: false)
        } else {
            base = try tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": state]], tools: nil,
                additionalContext: ["enable_thinking": false])
        }
        let baseLen = base.count
        var tokens = base
        var segs = base.map { _ in -1 } // base 全部记 -1（视为 prefix，掩码里被 key_prefix 覆盖）

        func append(_ text: String, seg: Int) {
            let ids = tokenizer.encode(text: text, addSpecialTokens: false)
            tokens.append(contentsOf: ids)
            segs.append(contentsOf: Array(repeating: seg, count: ids.count))
        }
        func appendSpan(_ text: String, seg: Int) -> (Int, Int) {
            let start = tokens.count
            append(text, seg: seg)
            return (start, tokens.count)
        }

        append("<|im_start|>user\n", seg: 0)
        // 官方联合编码 "…Question: {instructions}\nCandidates:\n" 的 instructions
        // 字符区间对应此处的中段 token 范围（空格随 instructions 分片）。
        append("Question 1\nTask: choice\nQuestion:", seg: 1)
        let instructionSpan = appendSpan(" \(question.instructions)", seg: 1)
        append("\nCandidates:\n", seg: 1)
        var candidateSpans: [(Int, Int)] = []
        for (key, desc) in zip(question.keys, question.descs) {
            append("\(key):", seg: 1)
            candidateSpans.append(appendSpan(" \(desc)", seg: 1))
            append("\n", seg: 1)
        }
        append("<|im_end|>\n<|im_start|>assistant\n", seg: 0)
        append("Question 1 Decision:", seg: 1)
        let decisionPosition = tokens.count - 1
        append("\n", seg: 1)

        return ValenPrompt(tokens: tokens, prefixLen: baseLen,
                           contextSpan: (0, baseLen),
                           instructionSpan: instructionSpan,
                           candidateSpans: candidateSpans,
                           decisionPosition: decisionPosition)
    }

    /// bbox_blocks 可见性矩阵：[1,1,T,T] bool（行=query、列=key）。
    static func bboxMask(segments: [Int], prefixLen: Int, length: Int) -> MLXArray {
        var visible = [[Int32]]()
        visible.reserveCapacity(length)
        for q in 0..<length {
            var row = [Int32](repeating: 0, count: length)
            if q < prefixLen {
                // prefix 行只见 prefix。
                for k in 0..<prefixLen { row[k] = 1 }
            } else {
                // 后缀行见 prefix + 公共胶水(seg 0) + 本段（段内双向）。
                for k in 0..<length {
                    row[k] = (k < prefixLen || segments[k] == 0 || segments[k] == segments[q]) ? 1 : 0
                }
            }
            visible.append(row)
        }
        // [[Int32]] 构出 [T, T]，用 newAxis 扩成 [1, 1, T, T]（直接多写两个 0...
        // 会在 2 维数组上越界）。
        return MLXArray(visible)[.newAxis, .newAxis, 0..., 0...].asType(.bool)
    }

    /// cache 数组：faIdx 放 BBoxMaskCache（模型从 cache[faIdx].makeMask 取掩码并
    /// 广播到所有全注意力层），其余层放常规 KVCacheSimple 占位（VLM 包装要求非
    /// 可选元素、长度覆盖全部层；线性层对非 MambaCache 会自动降级为无缓存）。
    static func makeCacheArray(prompt: ValenPrompt, layerCount: Int, faIdx: Int,
                               maskOverride: MLXArray? = nil)
        -> [any KVCache]? {
        guard layerCount > 0 else { return nil }
        let mask: MLXArray
        if let maskOverride {
            mask = maskOverride
        } else {
            // segments：prefix 部分记 0（掩码由 key_prefix 覆盖），suffix 题目块记 1。
            let segments = [Int](repeating: 0, count: prompt.prefixLen)
                + [Int](repeating: 1, count: prompt.tokens.count - prompt.prefixLen)
            mask = bboxMask(segments: segments, prefixLen: prompt.prefixLen,
                            length: prompt.tokens.count)
        }
        var caches: [any KVCache] = []
        caches.reserveCapacity(layerCount)
        for i in 0..<layerCount {
            caches.append(i == faIdx ? BBoxMaskCache(mask: mask) : KVCacheSimple())
        }
        return caches
    }

    // MARK: - MixerHead 前向

    /// 手写 layerNorm：torch F.layer_norm 默认 biased variance + eps=1e-5。
    private static func layerNorm(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray) -> MLXArray {
        let mu = x.mean(axis: -1, keepDims: true)
        let centered = x - mu
        let variance = (centered * centered).mean(axis: -1, keepDims: true)
        return centered / sqrt(variance + 1e-5) * w + b
    }

    /// features_from_hidden：池化 4 类角色向量 → [C, 4, D]。
    static func featuresFromHidden(hidden: MLXArray, prompt: ValenPrompt) -> MLXArray {
        let ctx = hidden[prompt.contextSpan.0..<prompt.contextSpan.1].mean(axis: 0)
        let ins = hidden[prompt.instructionSpan.0..<prompt.instructionSpan.1].mean(axis: 0)
        let dec = hidden[prompt.decisionPosition]
        var rows: [MLXArray] = []
        rows.reserveCapacity(prompt.candidateSpans.count)
        for span in prompt.candidateSpans {
            let cand = hidden[span.0..<span.1].mean(axis: 0)
            rows.append(stacked([ctx, ins, cand, dec], axis: 0))
        }
        return stacked(rows, axis: 0) // [C, 4, D]
    }

    /// head_logits：LN → 投影+roles → 2×(token_mlp + channel_mlp 残差块) →
    /// output LN → 读 candidate 行（角色下标 2）。
    static func headLogits(features: MLXArray, weights: [String: MLXArray]) throws -> MLXArray {
        guard let inputNormW = weights["head.input_norm.weight"],
              let inputNormB = weights["head.input_norm.bias"],
              let projW = weights["head.projection.weight"],
              let projB = weights["head.projection.bias"],
              let roles = weights["head.roles"],
              let out0W = weights["head.output.0.weight"],
              let out0B = weights["head.output.0.bias"],
              let out1W = weights["head.output.1.weight"],
              let out1B = weights["head.output.1.bias"] else {
            throw AgentError.llm("决策模型 decision_head 缺少关键张量")
        }
        var x = layerNorm(features, inputNormW, inputNormB)
        x = matmul(x, projW.T) + projB
        x = x + roles
        let width = x.shape[x.shape.count - 1]
        var blockCount = 0
        while weights["head.blocks.\(blockCount).token_norm.weight"] != nil { blockCount += 1 }
        guard blockCount > 0 else {
            throw AgentError.llm("决策模型 decision_head 缺少 blocks 权重")
        }
        for i in 0..<blockCount {
            let p = "head.blocks.\(i)."
            guard let tNormW = weights[p + "token_norm.weight"],
                  let tNormB = weights[p + "token_norm.bias"],
                  let t0W = weights[p + "token_mlp.0.weight"],
                  let t0B = weights[p + "token_mlp.0.bias"],
                  let t2W = weights[p + "token_mlp.2.weight"],
                  let t2B = weights[p + "token_mlp.2.bias"],
                  let cNormW = weights[p + "channel_norm.weight"],
                  let cNormB = weights[p + "channel_norm.bias"],
                  let c0W = weights[p + "channel_mlp.0.weight"],
                  let c0B = weights[p + "channel_mlp.0.bias"],
                  let c2W = weights[p + "channel_mlp.2.weight"],
                  let c2B = weights[p + "channel_mlp.2.bias"] else {
                throw AgentError.llm("决策模型 decision_head block\(i) 权重不完整")
            }
            // token_mlp：沿角色维度 4→16→4（每 token 位置独立），残差回加。
            var t = layerNorm(x, tNormW, tNormB).transposed(0, 2, 1)
            t = gelu(matmul(t, t0W.T) + t0B)
            t = matmul(t, t2W.T) + t2B
            x = x + t.transposed(0, 2, 1)
            // channel_mlp：沿宽度维度 256→1024→256。
            var c = layerNorm(x, cNormW, cNormB)
            c = gelu(matmul(c, c0W.T) + c0B)
            c = matmul(c, c2W.T) + c2B
            x = x + c
        }
        let y = layerNorm(x, out0W, out0B)
        let candidateRows = y[0..., 2, 0..<width]
        return (matmul(candidateRows, out1W.T) + out1B).squeezed(axis: -1) // [C]
    }

    // MARK: - 前向执行

    /// 一次前向：构建 prompt → 注入 bbox 掩码 → 取 hidden → 池化 → MixerHead。
    /// 返回与 question.keys 顺序一致的 softmax 概率。
    private func decideQuestion(state: String, question: ValenQuestion) async throws
        -> [Double] {
        try await load()
        let prompt = try Self.buildPrompt(
            tokenizer: try await LocalTokenizerLoader().load(from: modelDir),
            state: Self.escape(state.count > 3000 ? String(state.prefix(3000)) : state),
            question: ValenQuestion(
                instructions: Self.escape(question.instructions),
                keys: question.keys,
                descs: question.descs))
        guard !prompt.tokens.isEmpty else {
            throw AgentError.llm("判定 prompt tokenize 为空")
        }
        return try await forwardProbs(prompt: prompt)
    }

    /// 对给定 prompt 执行一次前向（bbox 掩码注入 + MixerHead），返回各候选概率。
    /// decideQuestion 的底层入口，测试/校准也直接使用。
    func forwardProbs(prompt: ValenPrompt, cacheArray: [any KVCache]? = nil) async throws -> [Double] {
        try await load()
        let temperature = self.temperature
        let (layerCount, faIdx) = Self.layerLayout(configText: configJSONText())
        let caches = cacheArray ?? Self.makeCacheArray(prompt: prompt,
                                                       layerCount: layerCount, faIdx: faIdx)
        return try await containerPerform { ctx, weights in
            guard !prompt.tokens.isEmpty else {
                throw AgentError.llm("判定 prompt tokenize 为空")
            }

            var state = LMOutput.State()
            state[mtpEmitFlagKey] = true
            let input = LMInput.Text(tokens: MLXArray(prompt.tokens)
                .reshaped([1, prompt.tokens.count]))
            let output = ctx.model(input, cache: caches, state: state)
            guard let hidden = output.state?[mtpLastHiddenStatesKey],
                  hidden.shape.count == 3 else {
                throw AgentError.llm("决策模型未返回有效 hidden states")
            }
            let feats = Self.featuresFromHidden(hidden: hidden[0], prompt: prompt)
            let logits = try Self.headLogits(features: feats, weights: weights)
            let stable = logits - max(logits)
            let tempArray = MLXArray([Float(temperature)])
            let exponent = exp(stable / tempArray)
            let probs = exponent / sum(exponent)
            MLX.eval(probs)
            return probs.asArray(Float.self).map { Double($0) }
        }
    }

    /// 从模型目录 config.json 读取层数与全注意力层下标（faIdx = interval-1）。
    static func layerLayout(configText: String) -> (layerCount: Int, faIdx: Int) {
        var layerCount = 24
        var interval = 4
        if let data = configText.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            let cfg = (obj["text_config"] as? [String: Any]) ?? obj
            if let n = cfg["num_hidden_layers"] as? Int { layerCount = n }
            if let v = cfg["full_attention_interval"] as? Int { interval = v }
        }
        return (layerCount, max(0, interval - 1))
    }

    private func configJSONText() -> String {
        (try? String(contentsOf: modelDir.appendingPathComponent("config.json"),
                     encoding: .utf8)) ?? ""
    }

    // MARK: - 判定入口

    /// 对一批文本切片判定与 query 的相关性分数（0~1，越高越相关）。
    /// Valen 每个切片独立成题（state 各异无法共享），逐条前向。
    public func score(query: String, passages: [String],
                      coreSentenceCount: Int,
                      useCoreSentences: Bool,
                      progress: ((Int, Int) -> Void)? = nil) async throws -> [Double] {
        try await load()
        guard !passages.isEmpty else { return [] }
        var results = [Double](repeating: 0, count: passages.count)
        for (i, passage) in passages.enumerated() {
            if Task.isCancelled { break }
            let text = useCoreSentences
                ? CharacterNgramBM25.coreSentences(query: query, text: passage,
                                                   count: max(1, coreSentenceCount)).joined(separator: " ")
                : passage
            let probs = try await decideQuestion(
                state: text,
                question: ValenQuestion(
                    instructions: "Is the state relevant to the question \"\(query)\"?",
                    keys: ["false", "true"],
                    descs: ["No / false", "Yes / true"]))
            results[i] = probs[1] // keys=[false,true]，取 true 概率
            progress?(i + 1, passages.count)
        }
        return results
    }

    /// 通用二元判定：返回「true 选项」的概率（0~1）。
    public func judgeBinary(state: String, question: String,
                            trueOption: String = "Yes / true",
                            falseOption: String = "No / false",
                            maxStateChars: Int = 3000) async throws -> Double {
        let clipped = state.count > maxStateChars ? String(state.prefix(maxStateChars)) : state
        let probs = try await decideQuestion(
            state: clipped,
            question: ValenQuestion(
                instructions: question,
                keys: ["false", "true"],
                descs: [falseOption, trueOption]))
        return probs[1]
    }

    /// 读取 container/weights 快照后执行闭包（container.perform 内部串行化模型访问）。
    private func containerPerform<T>(_ body: @Sendable (ModelContext, [String: MLXArray]) throws -> T)
        async throws -> T {
        lock.lock()
        let container = self.container
        let weights = self.headWeights
        lock.unlock()
        guard let container, let weights else { throw AgentError.llm("决策模型未加载") }
        return try await container.perform { ctx in try body(ctx, weights) }
    }
}

// MARK: - 协议遵循

extension ValenDecisionScorer: BinaryDecisionScoring {}
