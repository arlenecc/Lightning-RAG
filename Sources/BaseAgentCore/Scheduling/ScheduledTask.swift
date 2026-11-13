import Foundation

// MARK: - 触发规则

/// 定时任务的触发规则：每 N 分钟，或每天固定时刻（本地时区）。
///
/// 用扁平字段而不是带关联值的枚举，保证 `schedules.json` 可手工编辑、
/// 字段缺失时也能稳定解码。
public struct ScheduleRule: Codable, Sendable, Equatable {
    public enum Kind: String, Codable, Sendable {
        /// 每 N 分钟。
        case everyMinutes = "every_minutes"
        /// 每天 HH:mm。
        case daily
    }

    public var kind: Kind
    /// `everyMinutes` 的间隔分钟数。
    public var minutes: Int
    /// `daily` 的小时（0-23）。
    public var hour: Int
    /// `daily` 的分钟（0-59）。
    public var minute: Int

    public init(kind: Kind, minutes: Int = 0, hour: Int = 0, minute: Int = 0) {
        self.kind = kind
        self.minutes = minutes
        self.hour = hour
        self.minute = minute
    }

    enum CodingKeys: String, CodingKey {
        case kind, minutes, hour, minute
    }

    /// 容错解码：手工编辑 `schedules.json` 时缺失的数值字段按 0 处理，
    /// 由 `isValid` 决定该规则是否参与调度，而不是让整个文件解析失败。
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        kind = try container.decodeIfPresent(Kind.self, forKey: .kind) ?? .everyMinutes
        minutes = try container.decodeIfPresent(Int.self, forKey: .minutes) ?? 0
        hour = try container.decodeIfPresent(Int.self, forKey: .hour) ?? 0
        minute = try container.decodeIfPresent(Int.self, forKey: .minute) ?? 0
    }

    public static func everyMinutes(_ value: Int) -> ScheduleRule {
        ScheduleRule(kind: .everyMinutes, minutes: max(1, value))
    }

    public static func daily(hour: Int, minute: Int) -> ScheduleRule {
        ScheduleRule(kind: .daily, hour: hour, minute: minute)
    }

    public var isValid: Bool {
        switch kind {
        case .everyMinutes:
            return (1...10_080).contains(minutes)
        case .daily:
            return (0...23).contains(hour) && (0...59).contains(minute)
        }
    }

    /// 人类可读的规则描述（工具输出与侧栏共用）。
    public var summary: String {
        switch kind {
        case .everyMinutes:
            if minutes % 60 == 0 { return "每 \(minutes / 60) 小时" }
            return "每 \(minutes) 分钟"
        case .daily:
            return String(format: "每天 %02d:%02d", hour, minute)
        }
    }

    /// 从 `reference`（上次运行时间，没有则创建时间）之后的下一次触发时刻。
    public func nextFireDate(after reference: Date,
                             calendar: Calendar = .current) -> Date {
        switch kind {
        case .everyMinutes:
            let interval = TimeInterval(max(1, minutes) * 60)
            return reference.addingTimeInterval(interval)
        case .daily:
            let hour = min(max(hour, 0), 23)
            let minute = min(max(minute, 0), 59)
            let day = calendar.startOfDay(for: reference)
            var components = DateComponents()
            components.hour = hour
            components.minute = minute
            // 今天的时刻还没到就用今天，否则顺延到明天。
            if let today = calendar.date(byAdding: components, to: day), today > reference {
                return today
            }
            if let tomorrow = calendar.date(byAdding: .day, value: 1, to: day),
               let next = calendar.date(byAdding: components, to: tomorrow) {
                return next
            }
            return reference.addingTimeInterval(86_400)
        }
    }
}

// MARK: - 定时任务

/// 一条定时任务：到点后把 `prompt` 交给 Agent 执行，结果回写到对话。
///
/// 任务本身不限定执行方式 —— `prompt` 可以要求模型调用任意已注册工具
/// （外部 MCP 工具、shell_run / code_run 等）完成自定义任务。
public struct ScheduledTask: Codable, Sendable, Equatable, Identifiable {
    public var id: String
    public var title: String
    /// 交给 Agent 的执行指令。
    public var prompt: String
    public var rule: ScheduleRule
    public var enabled: Bool
    public var createdAt: Date
    /// 上次触发时间；也作为「每 N 分钟」的下次计时锚点。
    public var lastRunAt: Date?
    /// 上次执行结果的摘要（侧栏展示用，完整结果在对话里）。
    public var lastResult: String?

    public init(id: String = String(UUID().uuidString.prefix(8)),
                title: String,
                prompt: String,
                rule: ScheduleRule,
                enabled: Bool = true,
                createdAt: Date = Date(),
                lastRunAt: Date? = nil,
                lastResult: String? = nil) {
        self.id = id
        self.title = title
        self.prompt = prompt
        self.rule = rule
        self.enabled = enabled
        self.createdAt = createdAt
        self.lastRunAt = lastRunAt
        self.lastResult = lastResult
    }

    /// 下一次触发时刻（不判断是否已过期）。
    public func nextRunDate() -> Date {
        rule.nextFireDate(after: lastRunAt ?? createdAt)
    }

    /// 当前是否已到期。禁用或规则非法时不触发。
    public func isDue(now: Date = Date()) -> Bool {
        enabled && rule.isValid && now >= nextRunDate()
    }

    enum CodingKeys: String, CodingKey {
        case id, title, prompt, rule, enabled, createdAt, lastRunAt, lastResult
    }

    /// 容错解码：单条任务缺字段时按默认值补齐，避免一条坏数据让整份任务列表
    /// 解码失败、所有定时任务一起消失。
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decodeIfPresent(String.self, forKey: .id)
            ?? String(UUID().uuidString.prefix(8))
        title = try container.decodeIfPresent(String.self, forKey: .title) ?? ""
        prompt = try container.decodeIfPresent(String.self, forKey: .prompt) ?? ""
        rule = try container.decodeIfPresent(ScheduleRule.self, forKey: .rule)
            ?? ScheduleRule(kind: .everyMinutes, minutes: 0)
        enabled = try container.decodeIfPresent(Bool.self, forKey: .enabled) ?? true
        createdAt = try container.decodeIfPresent(Date.self, forKey: .createdAt) ?? Date()
        lastRunAt = try container.decodeIfPresent(Date.self, forKey: .lastRunAt)
        lastResult = try container.decodeIfPresent(String.self, forKey: .lastResult)
    }
}

// MARK: - 持久化

/// `schedules.json`（`<workspace>/.agent/schedules.json`）的线程安全读写。
///
/// 工具层（创建/删除/启停）与调度器（更新上次运行时间）共享同一实例，
/// 每次变更都原子落盘，并通过 `onChange` 通知 UI 刷新。
public final class ScheduleStore: @unchecked Sendable {
    private let lock = NSLock()
    private let path: String
    private var tasks: [ScheduledTask] = []
    /// 任务列表发生变化时回调（在写入线程调用，UI 侧自行切主线程）。
    public var onChange: (([ScheduledTask]) -> Void)?

    public init(path: String) {
        self.path = path
        self.tasks = Self.load(path)
    }

    public var filePath: String { path }

    public func all() -> [ScheduledTask] {
        lock.lock(); defer { lock.unlock() }
        return tasks
    }

    @discardableResult
    public func create(title: String,
                       prompt: String,
                       rule: ScheduleRule) throws -> ScheduledTask {
        let task = ScheduledTask(title: title, prompt: prompt, rule: rule)
        lock.lock()
        tasks.append(task)
        do {
            try persistLocked()
        } catch {
            tasks.removeLast()
            lock.unlock()
            throw error
        }
        let snapshot = tasks
        lock.unlock()
        onChange?(snapshot)
        return task
    }

    @discardableResult
    public func setEnabled(id: String, _ enabled: Bool) throws -> ScheduledTask? {
        lock.lock()
        guard let index = tasks.firstIndex(where: { $0.id == id }) else {
            lock.unlock()
            return nil
        }
        let previous = tasks[index]
        tasks[index].enabled = enabled
        do {
            try persistLocked()
        } catch {
            tasks[index] = previous
            lock.unlock()
            throw error
        }
        let updated = tasks[index]
        let snapshot = tasks
        lock.unlock()
        onChange?(snapshot)
        return updated
    }

    @discardableResult
    public func remove(id: String) throws -> ScheduledTask? {
        lock.lock()
        guard let index = tasks.firstIndex(where: { $0.id == id }) else {
            lock.unlock()
            return nil
        }
        let removed = tasks.remove(at: index)
        do {
            try persistLocked()
        } catch {
            tasks.insert(removed, at: index)
            lock.unlock()
            throw error
        }
        let snapshot = tasks
        lock.unlock()
        onChange?(snapshot)
        return removed
    }

    /// 记录一次触发时间。先落盘再执行，避免调度器每秒重复触发同一任务。
    public func markRun(id: String, at date: Date) throws {
        try mutate(id: id) { task in
            task.lastRunAt = date
        }
    }

    /// 记录最近一次执行结果摘要。
    public func setLastResult(id: String, summary: String) throws {
        try mutate(id: id) { task in
            task.lastResult = String(summary.prefix(200))
        }
    }

    private func mutate(id: String,
                        _ body: (inout ScheduledTask) -> Void) throws {
        lock.lock()
        guard let index = tasks.firstIndex(where: { $0.id == id }) else {
            lock.unlock()
            return
        }
        let previous = tasks[index]
        body(&tasks[index])
        do {
            try persistLocked()
        } catch {
            tasks[index] = previous
            lock.unlock()
            throw error
        }
        let snapshot = tasks
        lock.unlock()
        onChange?(snapshot)
    }

    private func persistLocked() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(tasks)
        try AtomicFile.write(data, to: path)
    }

    private static func load(_ path: String) -> [ScheduledTask] {
        guard let data = FileManager.default.contents(atPath: path) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let tasks = try? decoder.decode([ScheduledTask].self, from: data) else {
            AgentLog.warning("[schedule] schedules.json 解析失败，按空任务列表处理：\(path)")
            return []
        }
        return tasks
    }
}

// MARK: - 展示辅助

/// 本地时间文案（与 UI 时区一致）。
public func scheduleLocalTimeText(_ date: Date) -> String {
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "zh_CN")
    formatter.timeZone = .current
    formatter.dateFormat = "yyyy-MM-dd HH:mm"
    return formatter.string(from: date)
}
