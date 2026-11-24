import Foundation

// MARK: - 决策打分日志（供用户对照调整阈值）

/// 每次决策模型打分都输出一条可读日志：分数、生效阈值、判定结果、问句与候选摘要。
/// 用户可在日志里搜索 `[decision]` 直接看到「概率是多少、阈值是多少、为什么放行/拦截」，
/// 从而决定把设置窗口里的哪个阈值调高或调低。
enum DecisionScoreLog {
    /// 截断长文本，避免一条日志把 state/evidence 全文刷出来。
    static func brief(_ text: String, limit: Int = 80) -> String {
        let flat = text.replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return flat.count <= limit ? flat : String(flat.prefix(limit)) + "…"
    }

    /// 二元判定（judgeBinary，返回 true 选项概率 0~1）。
    /// - passed 语义由调用方给出：true=放行/通过，false=拦截。
    static func binary(label: String, prob: Double, threshold: Double,
                        passed: Bool, question: String = "", subject: String = "") {
        var line = "[decision] \(label) prob=\(String(format: "%.3f", prob)) "
            + "thr=\(String(format: "%.2f", threshold)) "
            + "→ \(passed ? "PASS" : "BLOCK")"
        if !subject.isEmpty { line += " target=\(brief(subject, limit: 40))" }
        if !question.isEmpty { line += " q=\"\(brief(question))\"" }
        AgentLog.info(line)
    }

    /// 多候选打分（score，相对相关性分数）。打印每个候选分数与选中项。
    static func ranked(label: String, names: [String], scores: [Double],
                       threshold: Double? = nil, picked: String = "",
                       question: String = "", extra: String = "") {
        let pairs = zip(names, scores).prefix(20).map {
            "\($0.0)=\(String(format: "%.3f", $0.1))"
        }.joined(separator: ", ")
        var line = "[decision] \(label) scores=[\(pairs)]"
        if let threshold { line += " thr=\(String(format: "%.2f", threshold))" }
        if !picked.isEmpty { line += " picked=\(brief(picked, limit: 40))" }
        if !extra.isEmpty { line += " \(extra)" }
        if !question.isEmpty { line += " q=\"\(brief(question))\"" }
        AgentLog.info(line)
    }
}

// MARK: - 内核决策门（complete 收尾拦截 + plan_step 计划评审）

/// 内核决策判定协议。返回 nil 表示判定不可用（模型未配置/加载失败/超时），
/// 内核必须静默放行，绝不能因为决策门故障阻塞主流程。
public protocol KernelDecisionJudge: AnyObject, Sendable {
    /// 计划步骤是否必要合理（true=必要；false=可能不必要/多余；nil=不可用）。
    /// `context` 是内核提供的「已成功获取的工具结果概要」（几百字），
    /// 用于判断信息获取类步骤是否已有对应材料。
    func stepNecessary(goal: String, step: String, context: String) async -> Bool?
    /// 已收集信息是否足以完整回答任务（true=足够；false=不足；nil=不可用）。
    /// `evidence` 是最终交给生成模型的材料（工具结果原文 + 会话状态），
    /// 实现方不得再截断或摘要——被裁掉的部分可能正是判定依据。
    func informationSufficient(task: String, evidence: String) async -> Bool?
    /// 带草稿答案的判定：生成前传空串（只判材料是否足够），生成后兜底时传候选答案。
    func informationSufficient(task: String, evidence: String, draft: String) async -> Bool?
}

extension KernelDecisionJudge {
    /// 默认忽略草稿，兼容只实现两参版本的判定器。
    public func informationSufficient(
        task: String, evidence: String, draft: String
    ) async -> Bool? {
        await informationSufficient(task: task, evidence: evidence)
    }
}

/// 基于决策模型打分器的内核判定实现（Meteor/Valen 任一后端均可注入）。
///
/// 两个判定点（模拟实验 6/6 与 3/3 准确）：
/// - complete：收尾时判断「已收集信息是否足以完整回答任务」；
/// - plan_step：计划创建时判断「每个步骤是否必要合理」。
///
/// state 组装遵循 example_decision.py 的 State/Question/Options 模板；
/// 单次前向 ~0.3-0.5s，任何异常/超时返回 nil 由调用方放行。
public final class MeteorKernelJudge: KernelDecisionJudge, @unchecked Sendable {
    private let scorer: any BinaryDecisionScoring
    private let timeoutSeconds: Double
    /// 计划完整性（步骤评审）阈值：true（必要）概率 < 该值才提示修订。
    /// 来自 config.kernelPlanReviewThreshold（默认 0.5，可调 0.4-0.95）。
    private let planThreshold: Double
    /// 执行完成度（收尾拦截）阈值：true（信息足够）概率 < 该值才拦截。
    /// 来自 config.kernelFinishGateThreshold（默认 0.5，可调 0.4-0.95）。
    private let finishThreshold: Double
    /// 判定失败后的降级策略：
    /// - 超时：本轮放行、下轮重试；连续 3 次才禁用（避免每轮都白等 10s）；
    /// - 模型缺失/加载失败等确定性错误：立即禁用。
    /// 禁用后本会话静默放行，只保留一条 WARN 便于排查。
    private let lock = NSLock()
    private var disabled = false
    private var consecutiveTimeouts = 0
    /// 连续超时达到该次数即禁用，防止 GPU 长期被占用时每轮都阻塞 10s。
    private let maxConsecutiveTimeouts = 3

    /// - Parameters:
    ///   - threshold: 兼容旧调用，作为两个阈值的共同默认；显式传 plan/finish 时以显式值为准。
    public init(scorer: any BinaryDecisionScoring, timeoutSeconds: Double = 10,
                threshold: Double = 0.5,
                planThreshold: Double? = nil, finishThreshold: Double? = nil) {
        self.scorer = scorer
        self.timeoutSeconds = timeoutSeconds
        self.planThreshold = planThreshold ?? threshold
        self.finishThreshold = finishThreshold ?? threshold
    }

    /// 统一执行：异常/超时 → nil（静默放行）。
    /// 超时只在连续达到上限后禁用；确定性错误立即禁用。
    private func judge(_ label: String,
                       state: String, question: String,
                       maxStateChars: Int = 3000) async -> Double? {
        lock.lock()
        let isDisabled = disabled
        lock.unlock()
        if isDisabled { return nil }
        do {
            let value = try await withTimeout(seconds: timeoutSeconds) {
                try await self.scorer.judgeBinary(
                    state: state, question: question,
                    trueOption: "Yes / true", falseOption: "No / false",
                    maxStateChars: maxStateChars)
            }
            lock.lock()
            consecutiveTimeouts = 0
            lock.unlock()
            return value
        } catch {
            lock.lock()
            let wasDisabled = disabled
            var warnTimeout = false
            var disableNow = false
            if error is DecisionTimeoutError {
                consecutiveTimeouts += 1
                if consecutiveTimeouts >= maxConsecutiveTimeouts {
                    disabled = true
                    disableNow = true
                } else {
                    warnTimeout = (consecutiveTimeouts == 1)
                }
            } else {
                disabled = true
                disableNow = true
            }
            lock.unlock()

            if disableNow && !wasDisabled {
                AgentLog.warning("[decision-gate] \(label) 判定失败，决策门已禁用（后续静默放行）："
                    + error.localizedDescription)
            } else if warnTimeout {
                AgentLog.warning("[decision-gate] \(label) 判定超时（\(timeoutSeconds)s），本轮放行并保留决策门，"
                    + "下一轮重试（连续 \(maxConsecutiveTimeouts) 次才禁用）")
            }
            return nil
        }
    }

    public func stepNecessary(goal: String, step: String, context: String) async -> Bool? {
        let trimmedContext = context.trimmingCharacters(in: .whitespacesAndNewlines)
        let state = """
        Task goal: \(goal)
        Plan step: \(step)

        Context gathered so far (briefs of successful tool results):
        \(trimmedContext.isEmpty ? "(none yet — no tool has returned anything)" : trimmedContext)
        """
        // 收紧「跳过信息获取步骤」的语义：只要该步骤是检索/获取来源、
        // 且上下文里还没有对应材料，就必须判为必要，防止计划评审把
        // 唯一的检索步骤判成「多余」。
        guard let prob = await judge("plan_step", state: state, question:
            "Is this plan step necessary and reasonable to achieve the task goal? "
                + "IMPORTANT: if the step gathers information from a source "
                + "(search / outline / web_scan / web_search / reading files) and the context above "
                + "does not yet contain the corresponding material, it MUST be judged necessary. "
                + "(A: unnecessary or redundant, B: necessary and reasonable)") else {
            return nil
        }
        let passed = prob >= planThreshold
        DecisionScoreLog.binary(label: "plan_step", prob: prob, threshold: planThreshold,
                                passed: passed, question: goal, subject: step)
        return passed
    }

    public func informationSufficient(task: String, evidence: String) async -> Bool? {
        await informationSufficient(task: task, evidence: evidence, draft: "")
    }

    public func informationSufficient(
        task: String, evidence: String, draft: String
    ) async -> Bool? {
        let trimmedDraft = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        let state: String
        let question: String
        if trimmedDraft.isEmpty {
            state = """
            User task: \(task)

            Material given to the answer model:
            \(evidence)
            """
            question =
                "Is this material sufficient to fully answer the user's task? "
                + "(A: insufficient, more information is needed, B: sufficient)"
        } else {
            state = """
            User task: \(task)

            Draft answer (candidate response, not yet shown to the user):
            \(trimmedDraft)

            Material given to the answer model:
            \(evidence)
            """
            question =
                "Is the draft answer complete, accurate, and supported by the material for "
                + "the user's task? (A: incomplete or unsupported, needs more work, B: sufficient)"
        }
        // 判定输入就是生成的输入：材料与草稿都必须完整送入，任何截断都会让
        // 「证据是否足够」的结论偏离真实生成条件。
        guard let prob = await judge("complete", state: state, question: question,
                                     maxStateChars: Int.max) else {
            return nil
        }
        let passed = prob >= finishThreshold
        DecisionScoreLog.binary(label: "complete", prob: prob, threshold: finishThreshold,
                                passed: passed, question: task)
        return passed
    }
}

// MARK: - 技能加载判定（检索后、加载使用前）

/// 技能判定协议。nil = 判定不可用（模型缺失/失败/超时），调用方必须静默放行，
/// 绝不因判定故障阻塞技能使用。
public protocol SkillDecisionJudge: AnyObject, Sendable {
    /// 多个候选技能中选一个最合适的；返回候选数组下标，nil = 判定不可用。
    func pickSkill(question: String, conversationState: String,
                   candidates: [(name: String, description: String, tags: [String])]) async -> Int?
    /// 单个技能是否适合当前问句/对话；false = 禁止加载使用，nil = 判定不可用（放行）。
    func isAppropriate(question: String, conversationState: String,
                       skill: (name: String, description: String, tags: [String])) async -> Bool?
}

/// 基于 Meteor 决策模型的技能加载判定。
/// 每次判定先跑 judgeBinary，失败/超时后整个实例禁用（首次 WARN、后续静默）。
public final class MeteorSkillJudge: SkillDecisionJudge, @unchecked Sendable {
    private let scorer: any BinaryDecisionScoring
    private let timeoutSeconds: Double
    private let threshold: Double
    private let lock = NSLock()
    private var disabled = false

    public init(scorer: any BinaryDecisionScoring, timeoutSeconds: Double = 10,
                threshold: Double = 0.5) {
        self.scorer = scorer
        self.timeoutSeconds = timeoutSeconds
        self.threshold = threshold
    }

    private func judge(_ label: String, state: String, question: String) async -> Double? {
        lock.lock(); let off = disabled; lock.unlock()
        if off { return nil }
        do {
            return try await withTimeout(seconds: timeoutSeconds) {
                try await self.scorer.judgeBinary(state: state, question: question)
            }
        } catch {
            lock.lock(); let first = !disabled; disabled = true; lock.unlock()
            if first {
                AgentLog.warning("[skill-judge] \(label) 判定失败，技能判定已禁用（后续静默放行）："
                    + error.localizedDescription)
            }
            return nil
        }
    }

    private func stateText(question: String, conversationState: String) -> String {
        """
        User question:
        \(question)

        Conversation state (plan / recent context):
        \(conversationState.isEmpty ? "(none)" : conversationState)
        """
    }

    public func pickSkill(question: String, conversationState: String,
                          candidates: [(name: String, description: String, tags: [String])]) async -> Int? {
        guard !candidates.isEmpty else { return nil }
        if candidates.count == 1 {
            // 单技能只需判断是否合适；不合适返回 -1 表示"None of them"。
            let ok = await isAppropriate(question: question,
                                         conversationState: conversationState,
                                         skill: candidates[0])
            guard let ok else { return nil }
            return ok ? 0 : -1
        }
        // 多技能：用决策模型的 query↔passage 相关性打分（score）一次性给所有候选排序，
        // 比逐个二元"是否加载"判别更稳（真机验证后者会把 web 技能误判为知识库查询）。
        // passage 包含技能名/描述/tags，query 带问句与对话状态前缀。
        let passages = candidates.map { skill in
            "Skill: \(skill.name)\nPurpose: \(skill.description)\nTags: \(skill.tags.joined(separator: ", "))"
        }
        let rankedQuery = conversationState.isEmpty
            ? question
            : "\(question)\n(Conversation state: \(conversationState))"
        do {
            let scores = try await withTimeout(seconds: timeoutSeconds) {
                try await self.scorer.score(query: rankedQuery, passages: passages,
                                            coreSentenceCount: 0, useCoreSentences: false,
                                            progress: nil)
            }
            guard let best = scores.indices.max(by: { scores[$0] < scores[$1] }) else { return nil }
            // score 是相对相关性分数，用是否显著高于均值判断"至少有一个合适"。
            let mean = scores.reduce(0, +) / Double(scores.count)
            let chosen = scores[best] > 0 && scores[best] >= mean ? best : -1
            DecisionScoreLog.ranked(
                label: "skill_pick", names: candidates.map(\.name), scores: scores,
                picked: chosen >= 0 ? candidates[chosen].name : "(none)",
                question: question,
                extra: "best=\(String(format: "%.3f", scores[best])) "
                    + "mean=\(String(format: "%.3f", mean)) criterion=best>0&best>=mean "
                    + "(skill thr=\(String(format: "%.2f", threshold)) 不作用于多技能选取)")
            return chosen
        } catch {
            lock.lock(); let first = !disabled; disabled = true; lock.unlock()
            if first {
                AgentLog.warning("[skill-judge] pick_skill 打分失败，技能判定已禁用："
                    + error.localizedDescription)
            }
            return nil
        }
    }

    public func isAppropriate(question: String, conversationState: String,
                              skill: (name: String, description: String, tags: [String])) async -> Bool? {
        let state = stateText(question: question, conversationState: conversationState)
            + "\n\nSkill to evaluate:\nName: \(skill.name)\n"
            + "Description: \(skill.description)\nTags: \(skill.tags.joined(separator: ", "))"
        guard let p = await judge("skill_fit", state: state,
                                  question: "Is this skill appropriate and useful for the user question "
                                    + "given the conversation state? "
                                    + "(A: no, not appropriate, B: yes, appropriate)") else { return nil }
        let passed = p >= threshold
        DecisionScoreLog.binary(label: "skill_fit", prob: p, threshold: threshold,
                                passed: passed, question: question, subject: skill.name)
        return passed
    }
}

/// 判定超时：与模型缺失/加载失败区分开——超时可重试，不直接禁用决策门。
struct DecisionTimeoutError: LocalizedError {
    let seconds: Double
    var errorDescription: String? { "决策判定超时（\(seconds)s）" }
}

/// 带超时的任务执行：超时抛出 `DecisionTimeoutError`。
/// 注意：超时不会真正取消底层前向（模型调用不可取消），只是不再等待其结果，
/// 由调用方按失败处理；底层任务会被等待到自然结束并丢弃结果。
private func withTimeout<T: Sendable>(seconds: Double,
                                      _ body: @escaping @Sendable () async throws -> T) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask { try await body() }
        group.addTask {
            try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            throw DecisionTimeoutError(seconds: seconds)
        }
        guard let first = try await group.next() else {
            throw AgentError.llm("决策判定任务组异常结束")
        }
        group.cancelAll()
        return first
    }
}
