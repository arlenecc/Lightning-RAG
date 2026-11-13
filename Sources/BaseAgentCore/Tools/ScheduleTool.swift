import Foundation

// MARK: - schedule_task

/// `schedule_task`：创建/管理定时任务。
///
/// 用户可以在对话里直接说「每天早上 9 点帮我……」，由模型把自然语言解析成
/// 结构化参数调用本工具；任务到点后由 `ScheduleRunner` 触发 App 执行并
/// 把结果写回对话框。
public final class ScheduleTaskTool: AgentTool, ShutdownableTool {
    public let name = "schedule_task"
    public let description =
        "Create and manage scheduled tasks that run unattended later. "
        + "When the user asks for something recurring ('every morning at 9', "
        + "'every 30 minutes', '每天…'), call op='create' with a self-contained "
        + "`prompt` describing exactly what to do at run time. The prompt is "
        + "executed later by you (the agent) and may call any tool — external MCP "
        + "tools, shell_run, code_run, file_tools, rag_tools, web_scan — so it must "
        + "contain every detail needed without the current conversation. "
        + "Use op='list' to show tasks, op='delete'/'enable'/'disable' to manage, "
        + "op='run_now' to trigger one immediately. "
        + "Do NOT execute the task yourself when creating it, and do not create "
        + "duplicates for the same request. Results of scheduled runs appear in "
        + "the chat automatically."
    public let destructive = true

    private let config: AgentConfig
    private weak var callbacks: AgentCallbacks?
    private weak var registry: ToolRegistry?
    private let store: ScheduleStore
    private let runner: ScheduleRunner

    public init(config: AgentConfig,
                callbacks: AgentCallbacks?,
                registry: ToolRegistry? = nil,
                store: ScheduleStore,
                runner: ScheduleRunner) {
        self.config = config
        self.callbacks = callbacks
        self.registry = registry
        self.store = store
        self.runner = runner
    }

    public var parameters: JSONValue {
        ToolSupport.objectSchema(
            properties: [
                "op": ToolSupport.stringProperty(
                    "create=新建; list=列出; delete=删除; enable=启用; "
                    + "disable=停用; run_now=立即执行一次",
                    enumValues: ["create", "list", "delete", "enable", "disable", "run_now"]),
                "title": ToolSupport.stringProperty(
                    "任务名称（create 时可选，缺省取 prompt 前 24 字）"),
                "prompt": ToolSupport.stringProperty(
                    "到点后交给 Agent 执行的完整指令（create 必填）。必须自包含："
                    + "不依赖当前对话上下文；可以要求调用任何工具完成自定义任务"),
                "every_minutes": ToolSupport.numberProperty(
                    "间隔触发：每多少分钟执行一次（与 time 二选一，范围 1-10080）"),
                "time": ToolSupport.stringProperty(
                    "每天固定时刻触发：24 小时制 \"HH:mm\"（如 \"09:30\"；"
                    + "与 every_minutes 二选一）"),
                "id": ToolSupport.stringProperty(
                    "任务 id 或标题（delete/enable/disable/run_now 必填；"
                    + "标题支持子串匹配）"),
            ],
            required: ["op"])
    }

    /// 只有创建/删除/立即执行会实际改变系统状态，需要用户确认；
    /// 查看与启停不打扰用户。
    public func shouldConfirm(_ args: [String: JSONValue]) -> Bool {
        let op = (args["op"]?.stringValue ?? "").lowercased()
        return op == "create" || op == "delete" || op == "run_now"
    }

    public func shutdown() {
        runner.stop()
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let op = (ToolSupport.optionalString(args, "op") ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        // 无人值守执行中的任务不允许增删改任务：模型可能被网页/文档内容诱导，
        // 每隔一分钟再创建一批任务，形成无法自行停止的自我调度循环。
        if registry?.isUnattended == true, op != "list" {
            return .fail("定时任务执行期间禁止创建/修改/删除定时任务；"
                + "如需调整，请让用户在本轮之外的对话中操作。")
        }
        switch op {
        case "create":
            return create(args)
        case "list":
            return .ok(renderList())
        case "delete":
            return delete(args)
        case "enable":
            return setEnabled(args, enabled: true)
        case "disable":
            return setEnabled(args, enabled: false)
        case "run_now":
            return runNow(args)
        default:
            return .fail("unknown op: '\(op)'（可用：create/list/delete/enable/disable/run_now）")
        }
    }

    // MARK: create

    private func create(_ args: [String: JSONValue]) -> ToolResult {
        let prompt = (ToolSupport.optionalString(args, "prompt") ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            return .fail("create 需要 prompt：到点后要执行的完整指令")
        }
        guard prompt.count <= 8_000 else {
            return .fail("prompt 过长（\(prompt.count) 字符），请压缩到 8000 字符以内")
        }

        let minutes = ToolSupport.optionalInt(args, "every_minutes") ?? 0
        let timeText = (ToolSupport.optionalString(args, "time") ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let rule: ScheduleRule
        if minutes > 0, !timeText.isEmpty {
            return .fail("every_minutes 与 time 只能二选一")
        } else if minutes > 0 {
            guard (1...10_080).contains(minutes) else {
                return .fail("every_minutes 需在 1-10080 之间（最长 7 天）")
            }
            rule = .everyMinutes(minutes)
        } else if !timeText.isEmpty {
            guard let parsed = Self.parseTime(timeText) else {
                return .fail("time 格式应为 24 小时制 \"HH:mm\"，例如 \"09:30\"")
            }
            rule = .daily(hour: parsed.hour, minute: parsed.minute)
        } else {
            return .fail("create 需要触发规则：every_minutes（每 N 分钟）或 time（每天 HH:mm）")
        }

        var title = (ToolSupport.optionalString(args, "title") ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if title.isEmpty {
            title = String(prompt.prefix(24))
        }
        title = String(title.prefix(60))

        do {
            let task = try store.create(title: title, prompt: prompt, rule: rule)
            callbacks?.onLog("[schedule] 已创建定时任务「\(task.title)」（\(rule.summary)）")
            let next = rule.nextFireDate(after: task.createdAt)
            return .ok(
                "已创建定时任务「\(task.title)」(id: \(task.id))\n"
                + "触发规则：\(rule.summary)\n"
                + "下次执行：\(scheduleLocalTimeText(next))\n"
                + "执行方式：到点后自动把你的指令交给 Agent（可调用任意工具）执行，"
                + "结果会直接显示在对话里。\n"
                + "注意：不要在本次对话中立即执行它，也不要为同一需求重复创建任务。")
        } catch {
            return ToolSupport.failure(error)
        }
    }

    // MARK: 其它操作

    private func delete(_ args: [String: JSONValue]) -> ToolResult {
        guard let task = resolve(args) else { return notFound(args) }
        do {
            _ = try store.remove(id: task.id)
            callbacks?.onLog("[schedule] 已删除定时任务「\(task.title)」")
            return .ok("已删除定时任务「\(task.title)」(id: \(task.id))")
        } catch {
            return ToolSupport.failure(error)
        }
    }

    private func setEnabled(_ args: [String: JSONValue], enabled: Bool) -> ToolResult {
        guard let task = resolve(args) else { return notFound(args) }
        do {
            guard let updated = try store.setEnabled(id: task.id, enabled) else {
                return .fail("任务不存在：\(task.id)")
            }
            let state = updated.enabled ? "已启用" : "已停用"
            callbacks?.onLog("[schedule] 定时任务「\(updated.title)」\(state)")
            var output = "\(state)定时任务「\(updated.title)」(id: \(updated.id))"
            if updated.enabled {
                output += "\n下次执行：\(scheduleLocalTimeText(updated.nextRunDate()))"
            }
            return .ok(output)
        } catch {
            return ToolSupport.failure(error)
        }
    }

    private func runNow(_ args: [String: JSONValue]) -> ToolResult {
        guard let task = resolve(args) else { return notFound(args) }
        runner.triggerNow(task)
        return .ok("已触发定时任务「\(task.title)」(id: \(task.id))，"
            + "执行结果会显示在对话里。")
    }

    // MARK: 辅助

    private func resolve(_ args: [String: JSONValue]) -> ScheduledTask? {
        let key = (ToolSupport.optionalString(args, "id") ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { return nil }
        let all = store.all()
        if let exact = all.first(where: { $0.id == key }) { return exact }
        let matches = all.filter {
            $0.title.localizedCaseInsensitiveContains(key)
        }
        return matches.count == 1 ? matches.first : nil
    }

    private func notFound(_ args: [String: JSONValue]) -> ToolResult {
        let key = (ToolSupport.optionalString(args, "id") ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else {
            return .fail("缺少参数 id（任务 id 或标题）；可先用 op='list' 查看")
        }
        let all = store.all()
        let ambiguous = all.filter { $0.title.localizedCaseInsensitiveContains(key) }
        if ambiguous.count > 1 {
            let ids = ambiguous.map { "\($0.id)(\($0.title))" }.joined(separator: ", ")
            return .fail("「\(key)」匹配到多个任务：\(ids)；请改用任务 id")
        }
        return .fail("未找到匹配「\(key)」的任务；当前任务：\n\(renderList())")
    }

    private func renderList() -> String {
        let all = store.all()
        guard !all.isEmpty else { return "当前没有定时任务。" }
        var lines = ["共 \(all.count) 个定时任务："]
        let now = Date()
        for task in all {
            let state = task.enabled ? "启用" : "停用"
            var line = "- [\(task.id)] \(task.title) · \(task.rule.summary) · \(state)"
            if task.enabled {
                let next = task.nextRunDate()
                line += now >= next
                    ? " · 已到期（等待执行）"
                    : " · 下次 \(scheduleLocalTimeText(next))"
            }
            if let last = task.lastRunAt {
                line += " · 上次 \(scheduleLocalTimeText(last))"
            }
            if let result = task.lastResult, !result.isEmpty {
                line += "\n    上次结果：\(result)"
            }
            lines.append(line)
        }
        return lines.joined(separator: "\n")
    }

    /// 解析 "H:mm"/"HH:mm"（也容忍全角冒号）。
    static func parseTime(_ text: String) -> (hour: Int, minute: Int)? {
        let normalized = text
            .replacingOccurrences(of: "：", with: ":")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let parts = normalized.split(separator: ":", omittingEmptySubsequences: false)
        guard parts.count == 2,
              let hour = Int(parts[0]), let minute = Int(parts[1]),
              (0...23).contains(hour), (0...59).contains(minute) else {
            return nil
        }
        return (hour, minute)
    }
}
