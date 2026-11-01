import Foundation
import MLX
import MLXLMCommon
import MLXNN

// 注意：`MLXFast` 是 MLX 模块内的命名空间（Source/MLX/MLXFast.swift），
// 不是独立模块，因此不能 `import MLXFast`。

/// 左 padding 批量生成的输入规划。
///
/// 一批提示按最长序列左对齐补齐；行序与请求顺序一致，
/// 生成结果按行索引写回。
public struct BatchGenerationPlan: Sendable {
    /// 补齐后的 token 行，每行等长。
    public let tokens: [[Int]]
    /// 每行左侧 pad 数量。
    public let leftPadding: [Int]
    /// 每行原始长度（不含 pad）。
    public let lengths: [Int]

    public init(tokens: [[Int]], leftPadding: [Int], lengths: [Int]) {
        self.tokens = tokens
        self.leftPadding = leftPadding
        self.lengths = lengths
    }

    /// 按最长序列左对齐 pad；空输入返回 nil。
    public static func leftPad(rows: [[Int]], padToken: Int) -> BatchGenerationPlan? {
        guard !rows.isEmpty else { return nil }
        var lengths: [Int] = []
        lengths.reserveCapacity(rows.count)
        var maxLength = 0
        for row in rows {
            lengths.append(row.count)
            maxLength = max(maxLength, row.count)
        }
        var padded: [[Int]] = []
        padded.reserveCapacity(rows.count)
        var leftPadding: [Int] = []
        leftPadding.reserveCapacity(rows.count)
        for (row, length) in zip(rows, lengths) {
            let pads = maxLength - length
            leftPadding.append(pads)
            padded.append(Array(repeating: padToken, count: pads) + row)
        }
        return BatchGenerationPlan(tokens: padded, leftPadding: leftPadding, lengths: lengths)
    }
}

/// 左 padding 感知的批量注意力 KV 缓存。
///
/// - pad 列的 K/V 在写入缓存时清零，等价于在注意力分数上屏蔽 pad 键；
/// - `ropeOffset` 返回逐行偏移 `.batch(offset - leftPadding)`，
///   使每行 RoPE 角度按自身序列位置计算。
public final class BatchedAttentionKVCache: KVCache, CustomDebugStringConvertible {
    private let leftPadding: [Int]?
    public private(set) var offset = 0
    private var keys: MLXArray?
    private var values: MLXArray?
    public var step = 256

    public var maxSize: Int? { nil }

    public init(leftPadding: [Int]? = nil) {
        self.leftPadding = leftPadding
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = offset

        let reset =
            if let currentKeys = self.keys, (previous + keys.dim(2)) > currentKeys.dim(2) {
                true
            } else {
                self.keys == nil
            }
        if reset {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)
            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)
            if var currentKeys = self.keys, var currentValues = self.values {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = MLX.concatenated([currentKeys, newK], axis: 2)
                self.values = MLX.concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        // 左 padding 列清零：本 chunk 内仍处于 pad 区的列，其 K/V 在写入前清零。
        // 之后任何位置的注意力（含 decode 步）都不会读到 pad token 的键值。
        var writeKeys = keys
        var writeValues = values
        if let leftPadding, !leftPadding.isEmpty {
            let threshold = MLXArray(leftPadding) - previous
            let cols = MLXArray(Int32(0) ..< Int32(keys.dim(2)))
            let zeroMask = cols .< threshold[0..., .newAxis]
            let expanded = zeroMask[0..., .newAxis, 0..., .newAxis]
            writeKeys = MLX.where(expanded, MLXArray.zeros(keys.shape, dtype: keys.dtype), keys)
            writeValues = MLX.where(
                expanded, MLXArray.zeros(values.shape, dtype: values.dtype), values)
        }

        offset += writeKeys.dim(2)
        self.keys?[.ellipsis, previous ..< offset, 0...] = writeKeys
        self.values?[.ellipsis, previous ..< offset, 0...] = writeValues

        let returnedKeys = self.keys![.ellipsis, ..<offset, 0...]
        let returnedValues = self.values![.ellipsis, ..<offset, 0...]
        return (returnedKeys, returnedValues)
    }

    public var ropeOffset: RoPEOffset {
        guard let leftPadding, !leftPadding.isEmpty else { return .scalar(offset) }
        return .batch(MLXArray(leftPadding.map { offset - $0 }))
    }

    public var state: [MLXArray] {
        get {
            guard let keys, let values else { return [] }
            if offset == keys.dim(2) {
                return [keys, values]
            }
            return [
                keys[.ellipsis, ..<offset, 0...],
                values[.ellipsis, ..<offset, 0...],
            ]
        }
        set {
            guard newValue.count == 2 else {
                fatalError("BatchedAttentionKVCache state must have exactly 2 arrays (keys, values)")
            }
            keys = newValue[0]
            values = newValue[1]
            offset = self.keys?.dim(2) ?? 0
        }
    }

    public var metaState: [String] {
        get { [""] }
        set {}
    }

    public var isTrimmable: Bool { false }

    @discardableResult
    public func trim(_ n: Int) -> Int { 0 }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard let leftPadding, !leftPadding.isEmpty else {
            if n == 1 { return .none }
            if returnArray || (windowSize != nil && n > windowSize!) {
                return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
            }
            return .causal
        }

        // 左 padding 感知掩码：pad 列不可被注意，其余列维持因果语义。
        // 显式返回数组掩码（也不能走符号 .causal），true 表示可参与注意力。
        // 形状必须为 [B, 1, n, kv]，才能广播到 [B, heads, n, kv] 的注意力分数；
        // 少一维会与 heads 维冲突（broadcast_shapes 报错）。
        let kvLength = offset + n
        let causal = createCausalMask(n: n, offset: offset, windowSize: windowSize)
        let columns = MLXArray(Int32(0) ..< Int32(kvLength))
        let attendable = columns[.newAxis] .>= MLXArray(leftPadding)[0..., .newAxis]
        let mask = causal[.newAxis, .newAxis] & attendable[0..., .newAxis, .newAxis, 0...]
        return .array(mask)
    }

    public func copy() -> any KVCache {
        let new = BatchedAttentionKVCache(leftPadding: leftPadding)
        new.step = step
        let s = state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        return new
    }

    public var debugDescription: String {
        "BatchedAttentionKVCache offset=\(offset)"
    }
}

extension BatchedAttentionKVCache: Evaluatable {
    public func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }
}

/// 单线程批量生成器：一组请求合成一个 batch，
/// 用模型自身的 batching 跑 chunked prefill + 逐步 decode。
///
/// 崩溃根因（同模型多路并发 generation，每路独立 trace 撞上共享权重被
/// 临时替换的 tracer 占位数组）在此不再成立：整个 map 阶段只有一条
/// 单线程生成循环，共享权重只在串行 trace 中读写。
public enum BatchedMLXGenerator {

    /// prefill 每次前向的最大 token 数。
    private static let prefillChunkSize = 256
    /// 温度和为零时的走贪心；高于该阈值启用 categorical 采样。
    private static let greedyThreshold: Float = 0.02

    /// 组装左 padding 感知缓存：SSM 层用 `MambaCache(leftPadding:)`（掩码由
    /// 框架原生处理），注意力层换成 `BatchedAttentionKVCache`（显式 pad 掩码
    /// + 逐行 RoPE 偏移）。
    static func makeCaches(
        model: any LanguageModel, leftPadding: [Int]?
    ) throws -> [KVCache] {
        // 空数组会让框架按「批大小 0」构造掩码，统一归一化为 nil（无 padding）。
        let pads = (leftPadding?.isEmpty ?? true) ? nil : leftPadding
        let baseCaches = try model.newCache(parameters: nil)
        var caches: [KVCache] = []
        caches.reserveCapacity(baseCaches.count)
        for cache in baseCaches {
            if cache is MambaCache {
                caches.append(MambaCache(leftPadding: pads))
            } else {
                caches.append(BatchedAttentionKVCache(leftPadding: pads))
            }
        }
        return caches
    }

    /// 批量生成。行序与 `plan` 输入一致，结果按行索引写回。
    ///
    /// - Parameters:
    ///   - tokens: 左 padding 后的输入规划
    ///   - model: 已加载的语言模型（调用方保证串行访问）
    ///   - stopTokenIds: 停止 token 集合（与 `buildStopTokenIds` 一致：
    ///     model config 的 eos、tokenizer eos、extraEOSTokens）
    ///   - padToken: 左 padding 与已结束行的占位 token
    ///   - maxTokensPerRow: 每行最多的生成 token 数
    ///   - temperature: nil 或 <= 0.02 时贪心采样
    public static func generate(
        tokens plan: BatchGenerationPlan,
        model: any LanguageModel,
        stopTokenIds: Set<Int>,
        padToken: Int,
        maxTokensPerRow: Int,
        temperature: Float? = nil
    ) throws -> [[Int]] {
        let batchSize = plan.tokens.count
        let sequenceLength = plan.tokens[0].count
        precondition(batchSize > 0 && sequenceLength > 0)
        precondition(maxTokensPerRow > 0)
        precondition(!stopTokenIds.isEmpty)

        let caches = try makeCaches(model: model, leftPadding: plan.leftPadding)

        // chunked prefill
        let input = MLXArray(plan.tokens)
        var state: LMOutput.State? = nil
        var position = 0
        var logits: MLXArray?
        while position < sequenceLength {
            if Task.isCancelled { throw CancellationError() }
            let end = min(position + Self.prefillChunkSize, sequenceLength)
            let chunk = input[0..., position ..< end]
            let result = model(LMInput.Text(tokens: chunk), cache: caches, state: state)
            state = result.state
            logits = result.logits
            MLX.asyncEval(chunk)
            position = end
        }
        guard var currentLogits = logits else { return [] }

        // finalize 清掉 SSM 侧的瞬态掩码（leftPadding/lengths），使 decode 阶段
        // createSSMMask 为空、走 decodeLinearLayer/decodeAttentionLayer 编译快路径。
        // 注意力缓存的 pads 常驻，decode 时 ropeOffset/makeMask 仍按行偏移。
        for cache in caches {
            cache.finalize()
        }

        var generated = [[Int]](repeating: [], count: batchSize)
        var active = [Bool](repeating: true, count: batchSize)

        for step in 0 ..< maxTokensPerRow {
            // 协作式取消：协调器 cancel Task 后尽快停下，避免长批次占住模型。
            if Task.isCancelled { throw CancellationError() }
            let sampled = Self.sample(logits: currentLogits, temperature: temperature)
            let sampledTokens = sampled.asArray(Int.self)
            precondition(sampledTokens.count == batchSize)

            var feedTokens = sampledTokens
            var anyActive = false
            for row in 0 ..< batchSize where active[row] {
                anyActive = true
                if stopTokenIds.contains(sampledTokens[row]) {
                    active[row] = false
                    // 已结束的行继续喂占位 token，保持 batch 形状。
                    feedTokens[row] = padToken
                } else {
                    generated[row].append(sampledTokens[row])
                }
            }
            if !anyActive { break }

            let nextInput = MLXArray(feedTokens)[0..., .newAxis]
            let result = model(LMInput.Text(tokens: nextInput), cache: caches, state: state)
            state = result.state
            currentLogits = result.logits
            MLX.asyncEval(nextInput)
            if (step + 1) % 256 == 0 {
                MLX.Memory.clearCache()
            }
        }

        return generated
    }

    /// 逐行采样：贪心或按温度 categorical。
    static func sample(logits: MLXArray, temperature: Float?) -> MLXArray {
        var logits = logits[0..., -1, 0...]
        if let temperature, temperature > greedyThreshold {
            if logits.dtype == .bfloat16 {
                logits = logits.asType(.float32)
            }
            return MLXRandom.categorical(logits * (1.0 / temperature))
        }
        return argMax(logits, axis: -1)
    }
}