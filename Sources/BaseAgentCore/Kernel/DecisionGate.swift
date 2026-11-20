import Foundation

// MARK: - 内核决策门（complete 收尾拦截 + plan_step 计划评审）

/// 内核决策判定协议。返回 nil 表示判定不可用（模型未配置/加载失败/超时），
/// 内核必须静默放行，绝不能因为决策门故障阻塞主流程。
public protocol KernelDecisionJudge: AnyObject, Sendable {
    /// 计划步骤是否必要合理（true=必要；false=可能不必要/多余；nil=不可用）。
    func stepNecessary(goal: String, step: String) async -> Bool?
    /// 已收集信息是否足以完整回答任务（true=足够；false=不足；nil=不可用）。
    func informationSufficient(task: String, evidence: String) async -> Bool?
}

/// 基于 Meteor 决策模型（readout head）的内核判定实现。
///
/// 两个判定点（模拟实验 6/6 与 3/3 准确）：
/// - complete：收尾时判断「已收集信息是否足以完整回答任务」；
/// - plan_step：计划创建时判断「每个步骤是否必要合理」。
///
/// state 组装遵循 example_decision.py 的 State/Question/Options 模板；
/// 单次前向 ~0.3-0.5s，任何异常/超时返回 nil 由调用方放行。
public final class MeteorKernelJudge: KernelDecisionJudge, @unchecked Sendable {
    private let scorer: MeteorDecisionScorer
    private let timeoutSeconds: Double
    /// 判定为 true 的概率阈值（true 选项 softmax 概率 ≥ threshold 才算通过）。
    /// 来自 config.kernelDecisionGateThreshold（默认 0.6，可调 0.4-0.95）。
    /// 阈值越高拦截越保守（越少误拦，但漏拦增多）。
    private let threshold: Double
    /// 判定失败（模型缺失/加载失败/超时）后整个会话禁用决策门：
    /// 首次打一条 WARN 便于排查，后续静默放行，避免每轮收尾刷同一条告警。
    private let lock = NSLock()
    private var disabled = false

    public init(scorer: MeteorDecisionScorer, timeoutSeconds: Double = 10,
                threshold: Double = 0.6) {
        self.scorer = scorer
        self.timeoutSeconds = timeoutSeconds
        self.threshold = threshold
    }

    /// 统一执行：超时/异常 → nil（静默放行）；首次失败后本次会话不再尝试。
    private func judge(_ label: String,
                       state: String, question: String) async -> Double? {
        lock.lock()
        let isDisabled = disabled
        lock.unlock()
        if isDisabled { return nil }
        do {
            let value = try await withTimeout(seconds: timeoutSeconds) {
                try await self.scorer.judgeBinary(state: state, question: question)
            }
            return value
        } catch {
            lock.lock()
            let firstFailure = !disabled
            disabled = true
            lock.unlock()
            if firstFailure {
                AgentLog.warning("[decision-gate] \(label) 判定失败，决策门已禁用（后续静默放行）："
                    + error.localizedDescription)
            }
            return nil
        }
    }

    public func stepNecessary(goal: String, step: String) async -> Bool? {
        let state = """
        Task goal: \(goal)
        Plan step: \(step)
        """
        guard let prob = await judge("plan_step", state: state, question:
            "Is this plan step necessary and reasonable to achieve the task goal? "
                + "(A: unnecessary or redundant, B: necessary and reasonable)") else {
            return nil
        }
        return prob >= threshold
    }

    public func informationSufficient(task: String, evidence: String) async -> Bool? {
        let state = """
        User task: \(task)

        Collected tool results (evidence):
        \(evidence)
        """
        guard let prob = await judge("complete", state: state, question:
            "Is the collected evidence sufficient to fully answer the user's task? "
                + "(A: insufficient, more information is needed, B: sufficient)") else {
            return nil
        }
        return prob >= threshold
    }
}

/// 带超时的任务执行：超时抛出 `AgentError.llm`。
/// 注意：超时不会真正取消底层前向（模型调用不可取消），只是不再等待其结果，
/// 由调用方按失败处理；底层任务会被等待到自然结束并丢弃结果。
private func withTimeout<T: Sendable>(seconds: Double,
                                      _ body: @escaping @Sendable () async throws -> T) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask { try await body() }
        group.addTask {
            try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            throw AgentError.llm("决策判定超时（\(seconds)s）")
        }
        guard let first = try await group.next() else {
            throw AgentError.llm("决策判定任务组异常结束")
        }
        group.cancelAll()
        return first
    }
}
