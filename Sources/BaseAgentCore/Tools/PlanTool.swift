import Foundation

// MARK: - 计划状态文件

/// 计划落盘实现：`<workspace>/.agent/plan.json`。
///
/// Python 版把计划写进 `Session.working["plan"]`；Swift 工具层不持有 Session，
/// 因此以 workspace 文件作为最小可行实现，并在每次变更后回调
/// `AgentCallbacks.onPlanUpdated` 让 UI / 内核同步。
enum PlanStore {
    static func path(_ config: AgentConfig) -> String {
        ToolSupport.stateFile("plan.json", workspace: config.workspace)
    }

    static func load(_ config: AgentConfig) -> Plan? {
        guard let data = FileManager.default.contents(atPath: path(config)) else {
            return nil
        }
        return try? JSONDecoder().decode(Plan.self, from: data)
    }

    static func save(_ plan: Plan, config: AgentConfig) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(plan)
        try AtomicFile.write(data, to: path(config))
    }

    static func clear(_ config: AgentConfig) {
        try? FileManager.default.removeItem(atPath: path(config))
    }
}

private func planJSONValue(_ plan: Plan) -> JSONValue {
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

// MARK: - update_plan

/// `update_plan`：维护结构化执行计划（目标 + 有序步骤）。
///
/// 四个操作与 Python 完全一致：
/// - `create` 新建/整体替换计划，并把第一步标为 doing；
/// - `patch` 更新某步 status/note（标 done 时顺带推进下一个 pending）；
/// - `append` 追加步骤（id 从现有最大数字 id + 1 起，跳过已占用 id）；
/// - `clear` 清空。
public final class UpdatePlanTool: AgentTool {
    public let name = "update_plan"
    public let description =
        "Manage the structured execution plan (goal + ordered steps). "
        + "Call op='create' ONCE BEFORE starting a non-trivial task — the plan "
        + "MUST be written before any actual tool call is issued, so progress is "
        + "tracked from step one; op='patch' to mark a step done/blocked with its "
        + "conclusion, op='append' when you discover a missing step, op='clear' "
        + "when finished. Mark a step 'blocked' only after at least 3 different "
        + "methods have failed (explain why in note) — a single failure is not a "
        + "reason to give up on a step. The plan is re-injected into your context "
        + "every turn — keep it accurate."
    public let destructive = false

    private let config: AgentConfig
    private let callbacks: AgentCallbacks?
    private weak var registry: ToolRegistry?
    private static let statuses: Set<String> = [
        "pending", "doing", "done", "blocked", "skipped",
    ]

    public var parameters: JSONValue {
        ToolSupport.objectSchema(
            properties: [
                "op": ToolSupport.stringProperty(
                    "create=新建/替换; patch=更新步骤; append=追加步骤; clear=清空",
                    enumValues: ["create", "patch", "append", "clear"]),
                "goal": ToolSupport.stringProperty(
                    "一句话目标（create 时必填，≤300 字符）"),
                "steps": ToolSupport.arrayProperty(
                    "create/append 的步骤列表。每项可以是字符串，或 "
                    + "{title, status?, note?}；status ∈ pending/doing/done/blocked/skipped",
                    items: .object(["type": .string("object")])),
                "step_id": ToolSupport.stringProperty(
                    "patch 的目标步骤 id（不填则更新第一个 status 不匹配的步骤）"),
                "status": ToolSupport.stringProperty(
                    "patch 的新状态。blocked 仅限多种方法都确认不可行时使用",
                    enumValues: ["pending", "doing", "done", "blocked", "skipped"]),
                "note": ToolSupport.stringProperty(
                    "patch 的结论/原因（≤200 字符）"),
            ],
            required: ["op"])
    }

    public init(config: AgentConfig, callbacks: AgentCallbacks?, registry: ToolRegistry? = nil) {
        self.config = config
        self.callbacks = callbacks
        self.registry = registry
    }

    /// 计划既要落盘（跨会话），也要同步进内核 Session（本轮内反思门/收尾判定可见）。
    private func syncSession(_ plan: Plan?) {
        if let plan {
            registry?.session?.setPlan(plan)
        } else {
            registry?.session?.clearPlan()
        }
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        do {
            let op = try ToolSupport.requireString(args, "op")
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            let goal = ToolSupport.optionalString(args, "goal") ?? ""
            let stepsValue = parseStepsValue(args["steps"])
            let stepID = (ToolSupport.optionalString(args, "step_id") ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let status = (ToolSupport.optionalString(args, "status") ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            let note = (ToolSupport.optionalString(args, "note") ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)

            switch op {
            case "clear":
                PlanStore.clear(config)
                syncSession(nil)
                callbacks?.onPlanUpdated(nil)
                return .ok("plan cleared")

            case "create":
                return try create(goal: goal, stepsValue: stepsValue)

            case "append":
                return try append(goal: goal, stepsValue: stepsValue)

            case "patch":
                return patch(stepID: stepID, status: status, note: note)

            default:
                return .fail("unknown op: '\(op)'")
            }
        } catch {
            return ToolSupport.failure(error)
        }
    }

    // MARK: create

    private func create(goal: String, stepsValue: JSONValue?) throws -> ToolResult {
        let rawSteps = stepsValue ?? .array([])
        guard var plan = normalizePlan(.object([
            "goal": .string(goal),
            "steps": rawSteps,
        ])) else {
            return .fail("create needs a goal and at least one non-empty step title")
        }

        // 仅当模型没有显式设定任何状态时才把第一步标为 doing。
        if plan.steps.allSatisfy({ $0.status == "pending" }) {
            plan.steps[0].status = "doing"
        }
        plan.createdAt = Date()
        plan.updatedAt = Date()
        try PlanStore.save(plan, config: config)
        syncSession(plan)
        callbacks?.onPlanUpdated(plan)

        var output = "plan created: \(plan.steps.count) step(s)"
        let rawCount = rawSteps.arrayValue?.count ?? 1
        if rawCount > plan.steps.count {
            output += "\n⚠ 步骤数超过上限 \(plan.steps.count)，"
                + "多出的 \(rawCount - plan.steps.count) 步被丢弃"
        }
        output += "\n▶ 现在立即开始执行第 1 步（调用相应工具），不要停下来询问是否开始。"
        return .ok(output + "\n" + renderPlanDict(plan))
    }

    // MARK: append

    private func append(goal: String, stepsValue: JSONValue?) throws -> ToolResult {
        guard let items = appendItems(stepsValue), !items.isEmpty else {
            return .fail("append needs steps")
        }

        let current = PlanStore.load(config)
        let baseGoal = current?.goal ?? goal
        var merged = current?.steps ?? []
        var usedIDs = Set(merged.map(\.id))
        var nextID = (merged.compactMap { Int($0.id) }.max() ?? 0) + 1
        var appended = 0

        for item in items {
            let title: String
            let status: String
            let note: String
            switch item {
            case .object(let object):
                title = (object["title"]?.stringValue ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let rawStatus = (object["status"]?.stringValue ?? "pending")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
                status = Self.statuses.contains(rawStatus) ? rawStatus : "pending"
                note = (object["note"]?.stringValue ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            case .string(let text):
                title = text.trimmingCharacters(in: .whitespacesAndNewlines)
                status = "pending"
                note = ""
            default:
                continue
            }
            guard !title.isEmpty else { continue }

            while usedIDs.contains(String(nextID)) { nextID += 1 }
            merged.append(PlanStep(
                id: String(nextID),
                title: String(title.prefix(120)),
                status: status,
                note: String(note.prefix(200))))
            usedIDs.insert(String(nextID))
            nextID += 1
            appended += 1
        }

        guard appended > 0 else {
            return .fail("append produced no valid step")
        }

        // normalizePlan 会按上限截断（保留最早 20 步）并校验状态。
        guard var plan = normalizePlan(planJSONValue(Plan(goal: baseGoal, steps: merged))) else {
            return .fail("append produced an empty plan")
        }
        plan.createdAt = current?.createdAt ?? plan.createdAt
        plan.updatedAt = Date()
        try PlanStore.save(plan, config: config)
        syncSession(plan)
        callbacks?.onPlanUpdated(plan)

        let dropped = merged.count - plan.steps.count
        var output = "plan now has \(plan.steps.count) step(s)"
        if dropped > 0 {
            output += "\n⚠ 步骤数超过上限（20），最新追加的 \(dropped) 步被丢弃；"
                + "请改用 update_plan(op='create') 重建一个更聚焦的计划"
        }
        return .ok(output + "\n" + renderPlanDict(plan))
    }

    // MARK: patch

    private func patch(stepID: String, status: String, note: String) -> ToolResult {
        guard var plan = PlanStore.load(config) else {
            return .fail("no plan to patch — call op='create' first")
        }

        if !status.isEmpty && !Self.statuses.contains(status) {
            return .fail("invalid status: '\(status)'")
        }
        if status.isEmpty && note.isEmpty {
            return .fail("patch needs status and/or note")
        }

        let target = stepID.trimmingCharacters(in: .whitespacesAndNewlines)
        var indices: [Int] = []
        if !target.isEmpty {
            indices = plan.steps.indices.filter { plan.steps[$0].id == target }
        } else if !status.isEmpty {
            // 无 step_id：更新第一个状态不匹配的步骤。
            if let index = plan.steps.firstIndex(where: { $0.status != status }) {
                indices = [index]
            }
        } else {
            return .fail("patch with only 'note' requires 'step_id'")
        }

        guard !indices.isEmpty else {
            return .fail("no step matched step_id='\(target)'; "
                + "current plan:\n\(renderPlanDict(plan))")
        }

        for index in indices {
            if !status.isEmpty { plan.steps[index].status = status }
            if !note.isEmpty { plan.steps[index].note = String(note.prefix(200)) }
        }
        let touched = indices.count

        // 标 done 时把**其后**第一个 pending 步骤推进为 doing，
        // 让 "Now:" 始终指向当前位置。
        if status == "done", let lastIndex = indices.max() {
            if lastIndex + 1 < plan.steps.count {
                for index in (lastIndex + 1)..<plan.steps.count
                where plan.steps[index].status == "pending" {
                    plan.steps[index].status = "doing"
                    break
                }
            }
        }

        plan.updatedAt = Date()
        do {
            try PlanStore.save(plan, config: config)
        } catch {
            return ToolSupport.failure(error)
        }
        syncSession(plan)
        callbacks?.onPlanUpdated(plan)

        let remaining = plan.steps.filter {
            $0.status == "pending" || $0.status == "doing"
        }.count
        return .ok("patched \(touched) step(s); \(remaining) open step(s) left\n"
            + renderPlanDict(plan))
    }

    // MARK: 参数兼容

    /// append 兼容 Python 的 `steps if isinstance(steps, list) else [steps]`。
    private func appendItems(_ raw: JSONValue?) -> [JSONValue]? {
        guard let raw else { return nil }
        if case .string(let encoded) = raw {
            let trimmed = encoded.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return nil }
            if let parsed = JSONValue.parse(trimmed) {
                return parsed.arrayValue ?? [parsed]
            }
            // 纯文本步骤（非 JSON）：当成单个步骤。
            return [raw]
        }
        if let array = raw.arrayValue { return array }
        if raw.isNull { return nil }
        return [raw]
    }

    /// 本地模型的文本工具调用会把 steps 作为 JSON 字符串传入（双重编码）。
    private func parseStepsValue(_ raw: JSONValue?) -> JSONValue? {
        guard let raw else { return nil }
        guard case .string(let encoded) = raw else { return raw }
        let trimmed = encoded.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return JSONValue.parse(trimmed)
    }
}
