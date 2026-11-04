import Foundation

// MARK: - 计划（L2.5：任务级「目标 + 路径」）

/// 计划步骤状态。done/blocked/skipped 都是终态；pending/doing 是非终态。
private let planOpenStatuses: Set<String> = ["pending", "doing"]
private let planStatuses: Set<String> = ["pending", "doing", "done", "blocked", "skipped"]
// 每步标题/备注的长度上限（计划每轮都要回注，不能无限膨胀）。
private let maxPlanSteps = 20
private let maxStepTitle = 120
private let maxStepNote = 200
private let maxPlanGoal = 300
// 状态标记（省 token：一个字符表示一步）。
private let planMarks: [String: String] = [
    "done": "✓", "doing": "▶", "pending": " ", "blocked": "✗", "skipped": "⊘",
]

/// 把一份 plan 渲染成紧凑多行文本（纯函数版，UI 面板可直接使用）。
public func renderPlanDict(_ plan: Plan?) -> String {
    guard let plan else { return "" }
    var lines: [String] = []
    if !plan.goal.isEmpty {
        lines.append("Goal: \(plan.goal)")
    }
    var current = ""
    for step in plan.steps {
        let mark = planMarks[step.status] ?? " "
        var line = "- [\(mark)] \(step.id). \(step.title)"
        if !step.note.isEmpty {
            line += "  # \(step.note)"
        }
        lines.append(line)
        if step.status == "doing" && current.isEmpty {
            current = "\(step.id). \(step.title)"
        }
    }
    if !current.isEmpty {
        lines.append("Now: \(current)")
    }
    let openSteps = plan.steps.filter { planOpenStatuses.contains($0.status) }
    if !openSteps.isEmpty {
        lines.append("Remaining: \(openSteps.count) step(s) not finished")
    }
    return lines.joined(separator: "\n")
}

/// 把模型传来的 plan 归一化为确定结构；非法返回 nil。
///
/// 计划是「模型自述、内核依赖」的数据 —— 内核的「未完成步骤不得收尾」门
/// 要读它，因此必须容忍脏输入。
public func normalizePlan(_ data: JSONValue) -> Plan? {
    guard let dict = data.objectValue else { return nil }
    let goal = String((dict["goal"]?.stringValue ?? "")
        .trimmingCharacters(in: .whitespacesAndNewlines).prefix(maxPlanGoal))

    var rawSteps = dict["steps"]
    // 本地模型常把 steps 作为 JSON 字符串传入（双重编码）；宽容解析。
    if case .string(let encoded)? = rawSteps {
        rawSteps = JSONValue.parse(encoded)
    }
    guard let stepValues = rawSteps?.arrayValue else { return nil }

    var steps: [PlanStep] = []
    for (offset, item) in stepValues.prefix(maxPlanSteps).enumerated() {
        let defaultID = "\(offset + 1)"
        let title: String
        let status: String
        let note: String
        let id: String
        switch item {
        case .object(let obj):
            title = String((obj["title"]?.stringValue ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines).prefix(maxStepTitle))
            status = (obj["status"]?.stringValue ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            note = String((obj["note"]?.stringValue ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines).prefix(maxStepNote))
            id = obj["id"]?.stringValue ?? defaultID
        case .string(let s):
            title = String(s.trimmingCharacters(in: .whitespacesAndNewlines).prefix(maxStepTitle))
            status = ""
            note = ""
            id = defaultID
        default:
            continue
        }
        if title.isEmpty { continue }
        steps.append(PlanStep(
            id: id,
            title: title,
            status: planStatuses.contains(status) ? status : "pending",
            note: note
        ))
    }
    guard !steps.isEmpty else { return nil }
    return Plan(goal: goal, steps: steps)
}

// MARK: - Session

/// 历史容器 + 工作记忆 + L3 折叠。
///
/// 内核不持有 `_history`。每一轮交给 Session，由 Session 决定模型能看到什么：
///
///     [system]
///       + L3 折叠区（旧轮 → 一行摘要，同类项合并）
///       + 最近 K 轮原文（保头保尾，工具输出已过 L1）
///       + 本轮 user 消息（= 锚点提示 + next_prompt）
///
/// 因为轮次是整轮丢弃的，`tool` 消息永远不会与它的 `assistant(tool_calls)`
/// 分离（否则 OpenAI 兼容服务会 400）。
public final class Session: @unchecked Sendable {
    public let budget: ContextBudget
    public let rawTurns: Int
    public let maxDigestLines: Int

    public var turns: [TurnRecord] = []
    /// key_info / checkpoint / 临时事实。
    public var working: [String: JSONValue] = [:]
    /// CONSOLIDATE 态主动折叠后留下的摘要行（这些 TurnRecord 已从 turns 移除）。
    public private(set) var extraDigest: [String] = []

    public private(set) var turnIndex: Int = 0
    /// 已被折叠的轮数（统计用）。
    public var foldedCount: Int = 0
    /// 计划版本号：setPlan / clearPlan 每次变化递增。
    /// Agent 用「本轮前后版本是否变化」判断这次请求是否真正用过计划，
    /// 从而避免给寒暄/简单查询提示固化技能。
    public private(set) var planRevision = 0
    /// 因摘要区超限被真正丢弃的行数（用于省略标记的累计值）。
    private var digestDropped = 0
    /// 上次发出「稳定 raw 窗口越顶」告警的轮次；用于节流，避免段内每轮刷屏。
    private var lastOldSlidingWarnTurn: Int = -1000

    public init(budget: ContextBudget, rawTurns: Int = 2, maxDigestLines: Int = 40) {
        self.budget = budget
        self.rawTurns = max(0, rawTurns)
        self.maxDigestLines = max(5, maxDigestLines)
    }

    // MARK: 轮次生命周期

    public var currentTurn: Int { turnIndex }

    @discardableResult
    public func startTurn(prompt: String = "", images: [Data]? = nil) -> TurnRecord {
        turnIndex += 1
        let rec = TurnRecord(index: turnIndex, prompt: prompt, images: images)
        turns.append(rec)
        return rec
    }

    public func finishTurn(_ rec: TurnRecord,
                           assistantText: String = "",
                           toolCalls: [ToolCall] = [],
                           toolResults: [ToolResultRecord] = [],
                           summary: String = "",
                           rawContent: String = "") {
        rec.assistantText = assistantText
        rec.rawContent = rawContent
        rec.toolCalls = toolCalls
        rec.toolResults = toolResults
        rec.summary = summary
        rec.tokens = TokenEstimator.estimate(
            rec.prompt + rec.assistantText
                + rec.toolResults.map(\.content).joined()
        )
        // 一次性提示到此才真正消费：本轮 turn 已落地，提示不再需要回注。
        // 被 rollback 的轮不会走到这里，提示会继续保留给下一轮。
        working.removeValue(forKey: "__truncated__")
        working.removeValue(forKey: "__plan_boundary__")
    }

    /// 空轮/无效轮不占历史，直接回退。
    public func rollbackTurn() {
        if !turns.isEmpty {
            turns.removeLast()
        }
        turnIndex = max(0, turnIndex - 1)
    }

    // MARK: 工作记忆（L4 的落地位置）

    public func setKeyInfo(_ text: String) {
        working["key_info"] = .string(text)
        working["passed_sessions"] = .int(0)
    }

    public func getKeyInfo() -> String {
        working["key_info"]?.stringValue ?? ""
    }

    /// 追加一条折叠摘要行（兜底收缩 / L4 摘要用）。
    public func appendDigest(_ line: String) {
        extraDigest.append(line)
        compactDigest()
    }

    /// 摘要区超限时丢弃最老的行，并在头部维护一条省略标记。
    ///
    /// 标记本身占一行，因此需要多腾出一行；已有标记时先移除旧标记，
    /// 避免「先删再覆盖」额外丢掉一条真实摘要。
    private func compactDigest() {
        guard extraDigest.count > maxDigestLines else { return }
        let markerPrefix = "[…更早的 "
        let hadMarker = extraDigest.first?.hasPrefix(markerPrefix) == true
        let drop = extraDigest.count - maxDigestLines + 1
        let removingMarker = hadMarker ? 1 : 0
        extraDigest.removeFirst(drop)
        digestDropped += max(0, drop - removingMarker)
        extraDigest.insert("[…更早的 \(digestDropped) 轮已折叠省略]", at: 0)
    }

    public func setFact(_ key: String, _ value: JSONValue) {
        var facts: [String: JSONValue]
        if let obj = working["facts"]?.objectValue {
            facts = obj
        } else {
            facts = [:]
        }
        facts[key] = value
        working["facts"] = .object(facts)
    }

    public func clearWorking() {
        working.removeValue(forKey: "key_info")
        working.removeValue(forKey: "facts")
    }

    // MARK: 计划

    /// 写入计划（已由 normalizePlan 校验）。返回是否接受。
    ///
    /// 与 Python 一致，计划存放在 `working["plan"]`：它与 key_info 同生命周期
    /// （跨 run 存续、reset 清空），并随 anchorPrompt 每轮回注。
    @discardableResult
    public func setPlan(_ data: JSONValue) -> Bool {
        guard let normalized = normalizePlan(data) else { return false }
        working["plan"] = Self.planJSON(normalized)
        planRevision += 1
        return true
    }

    public func setPlan(_ value: Plan) {
        working["plan"] = Self.planJSON(value)
        planRevision += 1
    }

    public func getPlan() -> Plan? {
        guard let stored = working["plan"] else { return nil }
        return normalizePlan(stored)
    }

    public func clearPlan() {
        working.removeValue(forKey: "plan")
        planRevision += 1
    }

    /// 未完成的步骤（pending / doing）——反思门的判据。
    public func planOpenSteps() -> [PlanStep] {
        guard let plan = getPlan() else { return [] }
        return plan.steps.filter { planOpenStatuses.contains($0.status) }
    }

    /// 把计划渲染成紧凑的多行文本（每轮回注 + UI 面板共用）。
    public func renderPlan() -> String {
        renderPlanDict(getPlan())
    }

    private static func planJSON(_ plan: Plan) -> JSONValue {
        .object([
            "goal": .string(plan.goal),
            "steps": .array(plan.steps.map { step in
                .object([
                    "id": .string(step.id),
                    "title": .string(step.title),
                    "status": .string(step.status),
                    "note": .string(step.note),
                ])
            }),
        ])
    }

    // MARK: 锚点提示

    /// 下一轮 user 消息的正文。对应 GenericAgent 的 `_get_anchor_prompt`。
    public func anchorPrompt(extra: String = "", everyN: Int = 8) -> String {
        var parts: [String] = []
        if !extra.isEmpty {
            parts.append(extra.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        parts.append("\n### [WORKING MEMORY]")
        parts.append("Current turn: \(turnIndex + 1)")
        let keyInfo = getKeyInfo()
        if !keyInfo.isEmpty {
            parts.append("<key_info>\(keyInfo)</key_info>")
        }
        // 计划紧随 key_info：这是「防止跑丢」的核心回注。
        let planText = renderPlan()
        if !planText.isEmpty {
            parts.append("### [PLAN]\n\(planText)")
        }
        // 上一轮被轮次上限截断：必须让模型看到，否则它会以为任务已完成。
        if let truncated = working["__truncated__"]?.stringValue, !truncated.isEmpty {
            parts.append("[SYSTEM] \(truncated)")
        }
        // 任务边界提示：让模型在「新任务重建计划 / 继续旧任务」之间显式判断。
        if let boundary = working["__plan_boundary__"]?.stringValue, !boundary.isEmpty {
            parts.append("[SYSTEM] \(boundary)")
        }
        if let facts = working["facts"]?.objectValue, !facts.isEmpty {
            parts.append("<facts>\(compactJSON(.object(facts), maxChars: 600))</facts>")
        }
        // 周期性纪律（对应 GA 的 turn%13 / turn%31 / turn%175 注入）。
        let n = turnIndex + 1
        if n % 13 == 0 {
            parts.append(
                "[TIPS] 已运行多轮：调用 update_working_checkpoint 保存关键上下文；"
                + "停止无效重试，无进展就换策略（1) 探测物理边界 2) 回读相关 SOP/技能）。"
            )
        }
        if n % 31 == 0 {
            parts.append(
                "[TIPS] 把检查点/关键发现/已试过的方案写入**文件**，不要只放在工作记忆里。"
            )
        }
        // 只有真正建过计划的任务才提示固化技能。寒暄/简单查询没有计划，
        // 不应反复诱导模型询问「是否保存为技能」。
        if n % max(1, everyN * 2) == 0, getPlan() != nil {
            parts.append(
                "[TIPS] 长任务：如已有可复用流程，用 skill_save 固化；"
                + "阶段性成果用 start_long_term_update 蒸馏进长期记忆。"
            )
        }
        return parts.joined(separator: "\n")
    }

    // MARK: CONSOLIDATE：主动折叠

    /// 把最老的 n 轮从 `turns` 摘掉，摘要行留进 `extraDigest`。
    ///
    /// 整轮摘除（而不是摘单条消息）保证不会出现悬挂的 `tool` 消息。
    /// 返回实际折叠的轮数。
    @discardableResult
    public func forceFold(_ n: Int = 1) -> Int {
        var folded = 0
        for _ in 0..<max(0, n) {
            if turns.isEmpty { break }
            extraDigest.append(turns.removeFirst().digestLine())
            folded += 1
        }
        compactDigest()
        if folded > 0 {
            foldedCount += folded
        }
        return folded
    }

    /// CONSOLIDATE 的折叠策略：**存储层**落地 —— 把 `turns` 真正压缩进 `extraDigest`，
    /// 折到只剩最近一轮，给下一个周期留满 raw 窗口。
    ///
    /// 与 `windows()`（渲染层，只决定当轮怎么渲染、不改 `turns`）不同，这里是唯一
    /// 会**缩小 `turns` 数组**的地方；不这么做，`turns` 无限增长（内存 + 每轮计算成本），
    /// 且一旦累计轮数超过 `rawTurns`，最老原文轮滑进 `old` 被重新编码 → 逐轮废 KV。
    ///
    /// 为什么折到「只剩最近一轮」而不是只折已摘要区（`old`）？因为段内 `old` 本来就
    /// 为空（`rawTurns` 全覆盖），折 `old` = 折 0，存储永不收敛。把稳定的原文 token
    /// 换成摘要 token 必然改变前缀，但这是**一个周期仅一次**的塌陷；若只折 `old`，
    /// 则 `old` 会无限膨胀、最终逐轮塌陷，比一次重置更糟。
    /// 返回实际折叠的轮数。
    @discardableResult
    public func foldForConsolidation() -> Int {
        guard turns.count > 1 else { return 0 }
        return forceFold(turns.count - 1)
    }

    // MARK: 组装消息

    /// 切分 `(old, raw)` —— L3 的核心，**纯渲染层**决策。
    ///
    /// 本函数**只决定「这一轮发给模型时」哪些轮当摘要、哪些轮当原文**，
    /// **绝不修改存储的 `turns` 数组**：`turns` 逐轮累加，从不被这里裁掉。
    /// 摘要 `<earlier_context>` 是 `l3Fold(extraDigest + old.map{digestLine})`
    /// 每轮现算的：只要 `old` 非空，每轮多吞一轮 → 摘要文本变 → 前缀 token 变 → KV 废。
    ///
    /// `raw` = 最近 `rawTurns` 个**已完成**的轮 + 当前进行中的轮。
    /// `old` = 其余全部，折叠成摘要行。
    ///
    /// 只有真正逼近模型上下文硬顶（侵入 reserve）时才收缩 `raw`。
    /// 早先用「raw 软配额 40%」做收缩阈值，会把 8 轮窗口压到 5~6 轮：
    /// 每新增一轮就有旧轮滑进 `<earlier_context>`（消息索引 1），
    /// 跨轮 KV 前缀每轮都失效、全量 prefill —— 这是「越聊越慢」的根因。
    ///
    /// 段内（两次 CONSOLIDATE 之间）`old` 恒为空、`rawTurns` 覆盖全部已完成轮，
    /// 摘要 = 固定 `extraDigest` → 前缀稳定 → KV 整段复用。把累加的 `turns` 真正
    /// 落地成摘要、让 `old` 清零的是 `foldForConsolidation()`（存储层），不是这里。
    func windows() -> (old: [TurnRecord], raw: [TurnRecord]) {
        guard let current = turns.last else { return ([], []) }
        let completed = Array(turns.dropLast())
        var keep = min(rawTurns, completed.count)
        // 硬顶 = 总预算 − 输出预留 − 其它区实际占用；不再参考 raw 软配额。
        let nonRaw = budget.used("system") + budget.used("earlier") + budget.used("tools")
        let rawCeiling = max(0, budget.total - budget.quota.reserve - nonRaw)
        while keep > 0 {
            let slice = Array(completed.suffix(keep)) + [current]
            if TokenEstimator.estimate(Self.renderRaw(slice)) <= rawCeiling {
                break
            }
            keep -= 1
        }
        let old = keep > 0 ? Array(completed.dropLast(keep)) : completed
        let raw = (keep > 0 ? Array(completed.suffix(keep)) : []) + [current]
        return (old, raw)
    }

    /// 估算原文区大小。必须与实际发出的消息同源：
    /// `tool_calls[*].function.arguments` 的 JSON 也会发出去。
    static func renderRaw(_ turns: [TurnRecord]) -> String {
        turns.map { turn in
            (turn.prompt)
                + (turn.assistantText)
                + turn.toolCalls.map { JSONValue.object($0.arguments).jsonString() }.joined()
                + turn.toolResults.map(\.content).joined()
        }.joined()
    }

    private func turnMessages(_ turn: TurnRecord) -> [ChatMessage] {
        // 进行中的轮（还没产出任何东西）不进消息列表。
        if turn.prompt.isEmpty && turn.assistantText.isEmpty && turn.toolCalls.isEmpty {
            return []
        }
        var out: [ChatMessage] = []
        if !turn.prompt.isEmpty {
            out.append(ChatMessage(role: .user, content: turn.prompt,
                                   images: turn.images))
        }
        if turn.assistantText.isEmpty && turn.toolCalls.isEmpty {
            return out
        }
        // 工具调用参数按 Python 侧行为统一重新序列化（历史里不保留模型原样空白）。
        let wireCalls = turn.toolCalls.map { tc -> ToolCall in
            var copy = tc
            copy.rawArguments = JSONValue.object(copy.arguments).jsonString()
            return copy
        }
        out.append(ChatMessage(role: .assistant, content: turn.assistantText,
                               toolCalls: wireCalls))

        // 每个 tool_call 都必须有一条对应结果，否则 API 400。
        if !turn.toolCalls.isEmpty {
            var byID: [String: ToolResultRecord] = [:]
            for r in turn.toolResults where !r.id.isEmpty {
                byID[r.id] = r
            }
            for tc in turn.toolCalls {
                let content = byID[tc.id]?.content ?? "(no result)"
                out.append(ChatMessage(role: .tool, content: content,
                                       name: tc.name, toolCallID: tc.id))
            }
        }
        return out
    }

    /// 产出本轮发给模型的完整消息列表（已过 L1/L3）。
    public func buildMessages(systemPrompt: String, userText: String = "") -> [ChatMessage] {
        let (old, raw) = windows()
        foldedCount = extraDigest.count + old.count

        // 运行时诊断：段内 `old` 非空 = 稳定 raw 窗口已越顶——raw 被迫滑进
        // `<earlier_context>`，意味着 kernelRawTurns 在当前负载下不足，KV 前缀每轮
        // 失效（越聊越慢）。借此判断 24（或所配值）是否够用：若持续出现该告警，
        // 应调大 kernelMaxContextTokens 抬高 rawCeiling，而非单加 kernelRawTurns。
        // 节流：相邻告警间隔 ≥8 轮，避免工具密集段每轮刷屏。
        if !old.isEmpty, currentTurn - lastOldSlidingWarnTurn >= 8 {
            lastOldSlidingWarnTurn = currentTurn
            let nonRaw = budget.used("system") + budget.used("earlier") + budget.used("tools")
            let rawCeiling = max(0, budget.total - budget.quota.reserve - nonRaw)
            let rawTokens = TokenEstimator.estimate(Self.renderRaw(raw))
            AgentLog.warning(
                "[context] 稳定 raw 窗口已越顶：段内 \(old.count) 轮滑入 <earlier_context>"
                + "（raw \(raw.count) 轮≈\(rawTokens) token / 上限 \(rawCeiling)）。"
                + " kernelRawTurns=\(rawTurns) 在当前负载下不足，"
                + "建议调大 kernelMaxContextTokens 抬高 rawCeiling，或精简单轮 token。"
            )
        }

        // 摘要 = 存储层 `extraDigest`（仅 CONSOLIDATE 时变，稳定）
        //       + 渲染层 `old` 的摘要行（随 `turns` 增长逐轮变）。
        // `old` 为空时（段内）摘要即固定 `extraDigest`，前缀不变 → KV 复用。
        let digest = l3Fold(
            extraDigest + old.map { $0.digestLine() },
            maxLines: maxDigestLines
        )

        var messages: [ChatMessage] = [ChatMessage(role: .system, content: systemPrompt)]
        if !digest.isEmpty {
            messages.append(ChatMessage(
                role: .user,
                content: "<earlier_context>\n\(digest)\n</earlier_context>"
            ))
        }
        for turn in raw {
            messages.append(contentsOf: turnMessages(turn))
        }
        // 当前轮的 prompt 已由 turnMessages 发出，这里只补「额外」的 user 文本。
        let extraUser = !userText.isEmpty && (raw.isEmpty || raw.last?.prompt != userText)
        if extraUser {
            messages.append(ChatMessage(role: .user, content: userText))
        }

        // 记账
        budget.setUsed("system", tokens: TokenEstimator.estimate(systemPrompt))
        budget.setUsed("earlier", tokens: TokenEstimator.estimate(digest))
        budget.setUsed("raw", tokens: TokenEstimator.estimate(
            Self.renderRaw(raw) + (extraUser ? userText : "")
        ))
        return messages
    }

    // MARK: 供工具/UI 观察

    /// 已折叠（L3）区的摘要行。
    public func digestLines() -> [String] {
        let (old, _) = windows()
        return extraDigest + old.map { $0.digestLine() }
    }

    /// 全部摘要行（折叠区 + 原文窗口）—— L4 蒸馏要用完整轨迹。
    public func allDigestLines() -> [String] {
        extraDigest + turns.map { $0.digestLine() }
    }

    public func stats() -> [String: JSONValue] {
        let (_, raw) = windows()
        return [
            "turn": .int(turnIndex),
            "total_turns": .int(turns.count),
            "raw_window": .int(raw.count),
            "folded": .int(foldedCount),
            "budget": .string(budget.snapshot()),
            "key_info_chars": .int(getKeyInfo().count),
        ]
    }

    /// 日志友好的单行统计（`stats()` 的展示形态）。
    public func statsDescription() -> String {
        let s = stats()
        return "turn=\(s["turn"]?.intValue ?? 0) "
            + "total_turns=\(s["total_turns"]?.intValue ?? 0) "
            + "raw_window=\(s["raw_window"]?.intValue ?? 0) "
            + "folded=\(s["folded"]?.intValue ?? 0) "
            + "\(s["budget"]?.stringValue ?? "") "
            + "key_info_chars=\(s["key_info_chars"]?.intValue ?? 0)"
    }

    public func reset() {
        turns.removeAll()
        working.removeAll()
        extraDigest.removeAll()
        turnIndex = 0
        foldedCount = 0
        digestDropped = 0
        planRevision += 1
    }
}
