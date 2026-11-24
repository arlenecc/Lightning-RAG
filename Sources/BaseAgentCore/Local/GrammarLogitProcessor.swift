import Foundation
import MLX
import MLXLMCommon
import MLXGuidedGeneration

/// 编译好的语法状态机与 tokenizer 在约束组件工厂闭包中共享。
final class UnguardedTokenizer: @unchecked Sendable {
    let tokenizer: any MLXLMCommon.Tokenizer
    init(_ tokenizer: any MLXLMCommon.Tokenizer) { self.tokenizer = tokenizer }
}

/// 复用的 logit 掩码缓冲：每 token 只重置上一轮放行的位置，按位图置位写入，
/// 避免每步遍历整个词表。
private final class GrammarMaskBuffer {
    private var values: [Float]
    private var allowed: [Int] = []
    private let vocabSize: Int

    init(logitDimension: Int, vocabSize: Int) {
        self.vocabSize = vocabSize
        self.values = [Float](repeating: -Float.infinity, count: logitDimension)
    }

    func array(for mask: MaskResult) -> MLXArray {
        for index in allowed { values[index] = -Float.infinity }
        allowed.removeAll(keepingCapacity: true)
        let words = mask.mask
        let wordCount = min(words.count, (min(vocabSize, values.count) + 31) / 32)
        for wordIndex in 0 ..< wordCount {
            var word = UInt32(bitPattern: words[wordIndex])
            while word != 0 {
                let token = wordIndex * 32 + Int(word.trailingZeroBitCount)
                if token < vocabSize && token < values.count {
                    values[token] = 0
                    allowed.append(token)
                }
                word &= word - 1
            }
        }
        return MLXArray(values)
    }
}

/// 与 `GrammarConstraint`（matcher）同生命周期的共享提交计数。
/// MLXLLMService 按 grammar 源码缓存同一个 constraint，却为每一轮生成新建
/// `GrammarProcessorState`；提交数若放在 state 里，新一轮的 `committed` 从 0
/// 开始，`reset()` 就不会回滚上一轮遗留的 matcher 推进，下一次 commit 会被
/// xgrammar 以 `GrammarError.commitFailed`（error 4）拒绝。
final class GuidedConstraintState: @unchecked Sendable {
    let constraint: GrammarConstraint
    private var committed: Int32 = 0
    private var invalid = false
    private let lock = NSLock()

    init(constraint: GrammarConstraint) { self.constraint = constraint }

    var isAvailable: Bool {
        lock.lock(); defer { lock.unlock() }
        return !invalid
    }

    func disable() {
        lock.lock(); invalid = true; lock.unlock()
    }

    /// 回到本轮生成前的 matcher 状态；覆盖上一轮正常完成、被取消或超时的所有路径。
    func rollbackAll() {
        lock.lock()
        defer { lock.unlock() }
        guard !invalid, committed > 0 else { return }
        do {
            try constraint.rollback(committed)
        } catch {
            AgentLog.warning("[guided] 语法状态回滚失败，后续禁用约束：\(error.localizedDescription)")
            invalid = true
        }
        committed = 0
    }

    func commit(tokenID: Int) throws -> CommitResult {
        let result = try constraint.commitToken(Int32(tokenID))
        lock.lock()
        committed += 1 + Int32(result.tokens.count)
        lock.unlock()
        return result
    }
}

/// 约束解码的全部可变状态。官方 API 下 token 由 LogitProcessor.didSample 提交
/// （读一次 GPU 值）；若应用了 patch，也可由 fork 的回调提交。
final class GrammarProcessorState: @unchecked Sendable {
    let constraintState: GuidedConstraintState
    let tokenizer: UnguardedTokenizer
    let vocabSize: Int
    let trigger: String

    private var decoded = ""
    private var armed = false
    private var finished = false
    private var maskBuffer: GrammarMaskBuffer?
    /// rollback 失败时禁用本实例约束，避免陈旧状态污染后续生成。
    private var constraintInvalid = false

    init(constraintState: GuidedConstraintState, tokenizer: UnguardedTokenizer,
         vocabSize: Int, trigger: String) {
        self.constraintState = constraintState
        self.tokenizer = tokenizer
        self.vocabSize = vocabSize
        self.trigger = trigger
    }

    /// 兼容直接构造单个 constraint 的调用方（含单测）。
    convenience init(constraint: GrammarConstraint, tokenizer: UnguardedTokenizer,
                     vocabSize: Int, trigger: String) {
        self.init(constraintState: GuidedConstraintState(constraint: constraint),
                  tokenizer: tokenizer, vocabSize: vocabSize, trigger: trigger)
    }

    func reset() {
        guard !constraintInvalid else { return }
        constraintState.rollbackAll()
        decoded = ""
        armed = false
        finished = false
        maskBuffer = nil
    }

    /// 当前约束是否仍可用；rollback 失败或处于 copy 副本时禁用。
    var isAvailable: Bool { !constraintInvalid && constraintState.isAvailable }

    /// vendored xgrammar 无 clone：copy 出的副本不能安全共享 matcher，直接禁用。
    func makeDisabledCopy() -> GrammarProcessorState {
        let copy = GrammarProcessorState(constraintState: constraintState, tokenizer: tokenizer,
                                         vocabSize: vocabSize, trigger: trigger)
        copy.constraintInvalid = true
        return copy
    }

    func commit(tokenID: Int) {
        guard isAvailable else { return }
        if !armed {
            decoded += tokenizer.tokenizer.decode(tokenIds: [tokenID])
            guard decoded.hasSuffix(trigger) else { return }
            armed = true
            AgentLog.info("[guided] tool-call grammar armed")
            return
        }
        guard !finished else { return }

        decoded += tokenizer.tokenizer.decode(tokenIds: [tokenID])

        do {
            let result = try constraintState.commit(tokenID: tokenID)
            if result.isTerminated {
                finished = true
                AgentLog.info("[guided] tool-call grammar completed")
            }
        } catch {
            AgentLog.warning("[guided] 语法状态机提交失败，本轮停止约束：\(error.localizedDescription)")
            finished = true
        }
    }

    /// 每步重新计算掩码。值区间的允许集合看似恒定，但多字节 UTF-8 token
    /// 的中间状态会改变位图，缓存会导致输出错位，因此不做区间复用。
    func maskArray(logits: MLXArray) -> MLXArray? {
        guard isAvailable, armed, !finished else { return nil }
        guard let mask = try? constraintState.constraint.computeMask(), mask.needsApply else { return nil }
        let dimension = logits.shape[logits.ndim - 1]
        guard dimension > 0 else { return nil }
        if maskBuffer == nil {
            maskBuffer = GrammarMaskBuffer(logitDimension: dimension, vocabSize: vocabSize)
        }
        return maskBuffer!.array(for: mask)
    }
}

/// 把 xgrammar 掩码包装成 mlx-swift-lm 的 `LogitProcessor`：保留原
/// `TokenIterator`、temperature 采样、跨轮 KV 复用与工具调用解析链路，
/// 只在掩码层插入。只有模型自己输出 `<tool_call>` 后才开始约束。
struct GrammarLogitProcessor: LogitProcessor {
    let state: GrammarProcessorState

    func process(logits: MLXArray) -> MLXArray {
        guard let mask = state.maskArray(logits: logits) else { return logits }
        return logits + mask
    }

    mutating func prompt(_ prompt: MLXArray) {
        state.reset()
    }

    mutating func didSample(token: MLXArray) {
        // 官方 API 下这是唯一能拿到当前采样 token 的时点；读取需要一次
        // GPU 同步（约 2x 开销）。若要消除需应用 patches/ 下的 fork patch。
        state.commit(tokenID: token.item(Int.self))
    }

    /// vendored xgrammar 不支持 clone；投机解码等 copy 路径下禁用约束，
    /// 避免两个 processor 共享同一 matcher 造成状态污染。
    func copy() -> GrammarLogitProcessor {
        GrammarLogitProcessor(state: state.makeDisabledCopy())
    }
}
