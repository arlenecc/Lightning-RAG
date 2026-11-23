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

/// 约束解码的全部可变状态。官方 API 下 token 由 LogitProcessor.didSample 提交
/// （读一次 GPU 值）；若应用了 patch，也可由 fork 的回调提交。
final class GrammarProcessorState: @unchecked Sendable {
    let constraint: GrammarConstraint
    let tokenizer: UnguardedTokenizer
    let vocabSize: Int
    let trigger: String

    private var decoded = ""
    private var armed = false
    private var finished = false
    private var committed: Int32 = 0
    private var maskBuffer: GrammarMaskBuffer?
    /// rollback 失败时禁用本实例约束，避免陈旧状态污染后续生成。
    private var constraintInvalid = false

    init(constraint: GrammarConstraint, tokenizer: UnguardedTokenizer,
         vocabSize: Int, trigger: String) {
        self.constraint = constraint
        self.tokenizer = tokenizer
        self.vocabSize = vocabSize
        self.trigger = trigger
    }

    func reset() {
        if committed > 0 {
            do {
                try constraint.rollback(committed)
            } catch {
                AgentLog.warning("[guided] 语法状态回滚失败，本实例后续禁用约束：\(error.localizedDescription)")
                constraintInvalid = true
            }
        }
        committed = 0
        decoded = ""
        armed = false
        finished = false
        maskBuffer = nil
    }

    /// 当前约束是否仍可用；rollback 失败或处于 copy 副本时禁用。
    var isAvailable: Bool { !constraintInvalid }

    /// vendored xgrammar 无 clone：copy 出的副本不能安全共享 matcher，直接禁用。
    func makeDisabledCopy() -> GrammarProcessorState {
        let copy = GrammarProcessorState(constraint: constraint, tokenizer: tokenizer,
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
            let result = try constraint.commitToken(Int32(tokenID))
            committed += 1 + Int32(result.tokens.count)
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
        guard let mask = try? constraint.computeMask(), mask.needsApply else { return nil }
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
