import Foundation

// MARK: - JSON

/// 与 Python 侧 `Any` 等价的动态 JSON 值。
/// 工具参数、工具 schema、模型返回值都通过它传递，避免为每个工具定义 DTO。
public enum JSONValue: Codable, Equatable, Sendable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null; return }
        if let b = try? c.decode(Bool.self) { self = .bool(b); return }
        if let i = try? c.decode(Int.self) { self = .int(i); return }
        if let d = try? c.decode(Double.self) { self = .double(d); return }
        if let s = try? c.decode(String.self) { self = .string(s); return }
        if let a = try? c.decode([JSONValue].self) { self = .array(a); return }
        if let o = try? c.decode([String: JSONValue].self) { self = .object(o); return }
        throw DecodingError.dataCorruptedError(in: c, debugDescription: "Unsupported JSON value")
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .int(let i): try c.encode(i)
        case .double(let d): try c.encode(d)
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }

    public var stringValue: String? {
        switch self {
        case .string(let s): return s
        case .int(let i): return String(i)
        case .double(let d): return String(d)
        case .bool(let b): return b ? "true" : "false"
        default: return nil
        }
    }

    public var intValue: Int? {
        switch self {
        case .int(let i): return i
        case .double(let d): return Int(d)
        case .bool(let b): return b ? 1 : 0
        case .string(let s): return Int(s)
        default: return nil
        }
    }

    public var doubleValue: Double? {
        switch self {
        case .double(let d): return d
        case .int(let i): return Double(i)
        case .string(let s): return Double(s)
        default: return nil
        }
    }

    public var boolValue: Bool? {
        switch self {
        case .bool(let b): return b
        case .int(let i): return i != 0
        case .string(let s):
            switch s.lowercased() {
            case "true", "1", "yes", "on": return true
            case "false", "0", "no", "off", "": return false
            default: return nil
            }
        default: return nil
        }
    }

    public var objectValue: [String: JSONValue]? {
        if case .object(let o) = self { return o }
        return nil
    }

    public var arrayValue: [JSONValue]? {
        if case .array(let a) = self { return a }
        return nil
    }

    public subscript(key: String) -> JSONValue? {
        get { objectValue?[key] }
        set {
            guard var o = objectValue else { return }
            o[key] = newValue
            self = .object(o)
        }
    }

    public static func from(_ any: Any) -> JSONValue {
        switch any {
        case is NSNull: return .null
        case let b as Bool: return .bool(b)
        case let i as Int: return .int(i)
        case let d as Double: return .double(d)
        case let s as String: return .string(s)
        case let a as [Any]: return .array(a.map(JSONValue.from))
        case let o as [String: Any]: return .object(o.mapValues(JSONValue.from))
        default: return .null
        }
    }

    public func jsonString(pretty: Bool = false) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = pretty ? [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes] : [.sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(self),
              let s = String(data: data, encoding: .utf8) else { return "null" }
        return s
    }

    public static func parse(_ text: String) -> JSONValue? {
        guard let data = text.data(using: .utf8),
              let value = try? JSONDecoder().decode(JSONValue.self, from: data) else { return nil }
        return value
    }

    public var isNull: Bool {
        if case .null = self { return true }
        return false
    }
}

// MARK: - Chat messages

public struct ChatMessage: Codable, Sendable, Equatable {
    public enum Role: String, Codable, Sendable {
        case system, user, assistant, tool
    }

    public var role: Role
    public var content: String
    public var name: String?
    public var toolCallID: String?
    public var toolCalls: [ToolCall]?
    /// 可选图像附件（JPEG/PNG 字节）。本地多模态后端（llama.cpp + mmproj、
    /// MLX VLM）消费；其它后端与 transcript/wire 序列化忽略此字段。
    public var images: [Data]?

    public init(role: Role, content: String, name: String? = nil,
                toolCallID: String? = nil, toolCalls: [ToolCall]? = nil,
                images: [Data]? = nil) {
        self.role = role
        self.content = content
        self.name = name
        self.toolCallID = toolCallID
        self.toolCalls = toolCalls
        self.images = images
    }

    enum CodingKeys: String, CodingKey {
        case role, content, name, toolCalls
        case toolCallID = "tool_call_id"
    }

    /// 发给 OpenAI 兼容端点的 wire 格式。
    public func wire() -> [String: Any] {
        var m: [String: Any] = ["role": role.rawValue, "content": content]
        if let name { m["name"] = name }
        if let toolCallID { m["tool_call_id"] = toolCallID }
        if let toolCalls, !toolCalls.isEmpty {
            m["tool_calls"] = toolCalls.map { tc in
                [
                    "id": tc.id,
                    "type": "function",
                    "function": ["name": tc.name, "arguments": tc.rawArguments],
                ] as [String: Any]
            }
        }
        return m
    }
}

// MARK: - Tool calls / results

public struct ToolCall: Codable, Sendable, Equatable, Identifiable {
    public var id: String
    public var name: String
    public var arguments: [String: JSONValue]
    public var rawArguments: String

    public init(id: String, name: String, arguments: [String: JSONValue], rawArguments: String = "") {
        self.id = id
        self.name = name
        self.arguments = arguments
        self.rawArguments = rawArguments
    }

    public var compact: String {
        let visible = arguments.filter { !$0.key.hasPrefix("_") }
        let s = JSONValue.object(visible).jsonString()
        return "\(name)(\(s.count > 120 ? String(s.prefix(120)) + "..." : s))"
    }
}

public struct ToolResult: Sendable {
    public var success: Bool
    public var output: String
    public var error: String
    /// true 表示工具要求中断整轮（ask_user / 终止信号）。
    public var shouldExit: Bool

    public init(success: Bool, output: String = "", error: String = "", shouldExit: Bool = false) {
        self.success = success
        self.output = output
        self.error = error
        self.shouldExit = shouldExit
    }

    public static func ok(_ output: String) -> ToolResult { .init(success: true, output: output) }
    public static func fail(_ error: String) -> ToolResult { .init(success: false, error: error) }

    public func toMessage() -> String {
        success ? output : (error.isEmpty ? "ERROR: unknown" : "ERROR: \(error)")
    }
}

// MARK: - Tool protocol

public protocol AgentTool: AnyObject {
    var name: String { get }
    var description: String { get }
    /// JSON Schema 对象（type=object, properties=...）。
    var parameters: JSONValue { get }
    /// 是否可能影响工作区外的系统/文件。
    var destructive: Bool { get }
    func shouldConfirm(_ args: [String: JSONValue]) -> Bool
    func run(_ args: [String: JSONValue]) async throws -> ToolResult
}

public extension AgentTool {
    var destructive: Bool { false }
    func shouldConfirm(_ args: [String: JSONValue]) -> Bool { destructive }
    var schema: JSONValue {
        .object([
            "type": .string("function"),
            "function": .object([
                "name": .string(name),
                "description": .string(description),
                "parameters": parameters,
            ]),
        ])
    }
}

/// 同一实现的另一个名字（对齐 GenericAgent 的原子工具命名）。
public final class AliasTool: AgentTool {
    public let name: String
    public let description: String
    private let targetName: String
    private weak var registry: ToolRegistry?

    public init(targetName: String, alias: String, description: String, registry: ToolRegistry?) {
        self.targetName = targetName
        self.name = alias
        self.description = description
        self.registry = registry
    }

    public var parameters: JSONValue {
        registry?.get(targetName)?.parameters ?? .object(["type": .string("object"), "properties": .object([:])])
    }

    public func shouldConfirm(_ args: [String: JSONValue]) -> Bool {
        registry?.get(targetName)?.shouldConfirm(args) ?? false
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        guard let registry, let target = registry.get(targetName) else {
            return .fail("alias \(name) not bound")
        }
        // 确认由 AgentToolDispatcher 统一完成（shouldConfirm 已转发目标工具），
        // 这里直接执行目标：避免二次弹窗，也避免 headless fail-closed 把
        // 已经确认过的别名调用再次误拒。
        return try await target.run(args)
    }
}

// MARK: - Tool registry

public final class ToolRegistry {
    private var tools: [String: AgentTool] = [:]
    private let lock = NSRecursiveLock()
    public let config: AgentConfig
    public weak var callbacks: AgentCallbacks?
    /// 工具层与内核共享的会话状态（计划、工作记忆、锚点）。
    public var session: Session?
    /// 懒初始化的 RAG 引擎引用，供 rag_* 工具与系统提示复用。
    public var ragEngine: RAGEngine?

    public init(config: AgentConfig, callbacks: AgentCallbacks? = nil) {
        self.config = config
        self.callbacks = callbacks
    }

    public func register(_ tool: AgentTool) {
        lock.lock(); defer { lock.unlock() }
        tools[tool.name] = tool
    }

    public func registerAll(_ list: [AgentTool]) {
        list.forEach(register)
    }

    /// 注销工具（MCP 服务动态刷新工具清单时用）。
    public func unregister(_ name: String) {
        lock.lock(); defer { lock.unlock() }
        tools.removeValue(forKey: name)
    }

    public func get(_ name: String) -> AgentTool? {
        lock.lock(); defer { lock.unlock() }
        return tools[name]
    }

    public var allTools: [AgentTool] {
        lock.lock(); defer { lock.unlock() }
        return Array(tools.values).sorted { $0.name < $1.name }
    }

    public func schemas() -> [JSONValue] {
        allTools.map(\.schema)
    }

    /// 解析模型返回的参数 JSON；失败抛错（不能当 {} 静默误诊）。
    public static func parseArgs(_ raw: String) throws -> [String: JSONValue] {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return [:] }
        guard let value = JSONValue.parse(trimmed) else {
            throw AgentError.invalidArguments("工具参数不是合法 JSON")
        }
        guard let object = value.objectValue else {
            throw AgentError.invalidArguments("工具参数必须是 JSON 对象")
        }
        return object
    }

    public func execute(_ name: String, _ args: [String: JSONValue]) async throws -> ToolResult {
        guard let tool = get(name) else {
            return .fail("unknown tool: \(name)")
        }
        if tool.shouldConfirm(args) {
            // fail-closed：没有确认通道就绝不执行需要人工复核的工具。
            // 旧实现 `if ..., let callbacks` 在 headless 下会静默跳过确认。
            guard let callbacks else {
                return .fail("工具 \(name) 需要人工确认，但当前没有确认通道（headless fail-closed）")
            }
            let ok = callbacks.confirm(ToolSupport.confirmationMessage(name: name, args: args))
            if !ok { return .fail("User denied") }
        }
        return try await tool.run(args)
    }

    public func close() {
        // MCP 等持有子进程/连接的工具在 shutdown 时清理。
        for tool in allTools {
            (tool as? ShutdownableTool)?.shutdown()
        }
    }
}

public protocol ShutdownableTool {
    func shutdown()
}

// MARK: - Callbacks

public protocol AgentCallbacks: AnyObject {
    func isCancelled() -> Bool
    func onContent(_ text: String)
    func onReasoning(_ text: String)
    func onToolStart(_ name: String, _ args: [String: JSONValue])
    func onToolEnd(_ name: String, _ result: ToolResult)
    func onLog(_ line: String)
    func onUsage(_ usage: [String: Int])
    func onTokenSpeed(_ totalTokens: Int, _ speed: Double)
    func onPlanUpdated(_ plan: Plan?)
    func onError(_ message: String)
    func onContextShrunk(_ summary: String, _ reason: String)
    func onSkillSuggested(_ skill: SkillSuggestion)
    /// 检索即展示：rag_search 召回的全部切片（含进上下文的 top-1）推给 UI
    /// 作气泡展示；headless / 无 UI 调用方默认空实现即可。
    func onRetrievedChunks(_ chunks: [RAGCitation])
    func confirm(_ message: String) -> Bool
    func askUser(_ prompt: String) -> String?
    func onFinished()
}

public extension AgentCallbacks {
    func isCancelled() -> Bool { false }
    func onContent(_ text: String) {}
    func onReasoning(_ text: String) {}
    func onToolStart(_ name: String, _ args: [String: JSONValue]) {}
    func onToolEnd(_ name: String, _ result: ToolResult) {}
    func onLog(_ line: String) {}
    func onUsage(_ usage: [String: Int]) {}
    func onTokenSpeed(_ totalTokens: Int, _ speed: Double) {}
    func onPlanUpdated(_ plan: Plan?) {}
    func onError(_ message: String) {}
    func onContextShrunk(_ summary: String, _ reason: String) {}
    func onSkillSuggested(_ skill: SkillSuggestion) {}
    func onRetrievedChunks(_ chunks: [RAGCitation]) {}
    /// 默认拒绝：没有实现人工确认通道的调用方（headless / 测试 / CLI）不得
    /// 静默放行高风险工具。需要放行的场景必须显式使用 `AutoConfirmCallbacks`。
    func confirm(_ message: String) -> Bool { false }
    func askUser(_ prompt: String) -> String? { nil }
    func onFinished() {}
}

/// 没有任何 UI 输出、且**默认拒绝**一切确认请求的回调（fail-closed）。
public final class NoopCallbacks: AgentCallbacks {}

/// 显式自动批准：仅供单元测试或完全受信任的无人值守自动化使用。
/// 生产入口必须提供真实的人工确认实现（见 `AppModel.confirm`）。
public final class AutoConfirmCallbacks: AgentCallbacks {
    public init() {}
    public func confirm(_ message: String) -> Bool { true }
}

// MARK: - Shared lightweight models

public struct PlanStep: Codable, Sendable, Equatable, Identifiable {
    public var id: String
    public var title: String
    public var status: String   // pending / doing / done / blocked / skipped
    public var note: String

    public init(id: String = UUID().uuidString.prefix(8).description,
                title: String, status: String = "pending", note: String = "") {
        self.id = id
        self.title = title
        self.status = status
        self.note = note
    }
}

public struct Plan: Codable, Sendable, Equatable {
    public var goal: String
    public var steps: [PlanStep]
    public var createdAt: Date
    public var updatedAt: Date

    public init(goal: String, steps: [PlanStep] = [],
                createdAt: Date = Date(), updatedAt: Date = Date()) {
        self.goal = goal
        self.steps = steps
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    public var isComplete: Bool {
        steps.allSatisfy { ["done", "blocked", "skipped"].contains($0.status) }
    }

    /// 每轮注入上下文的紧凑视图。
    public func render() -> String {
        let icons = ["pending": "○", "doing": "◐", "done": "●", "blocked": "✕", "skipped": "—"]
        var lines = ["[计划] \(goal)"]
        for s in steps {
            let icon = icons[s.status] ?? "○"
            let note = s.note.isEmpty ? "" : " — \(s.note)"
            lines.append("  \(icon) \(s.id) \(s.title)\(note)")
        }
        return lines.joined(separator: "\n")
    }
}

public struct SkillSuggestion: Sendable, Equatable {
    public var name: String
    public var description: String
    public var score: Double
    public init(name: String, description: String = "", score: Double = 0) {
        self.name = name
        self.description = description
        self.score = score
    }
}

// MARK: - Errors

public enum AgentError: LocalizedError {
    case invalidArguments(String)
    case unsupported(String)
    case cancelled
    case contextTooLong
    case llm(String)
    case storage(String)
    case parse(String)
    case permission(String)
    case notFound(String)
    case internalError(String)
    /// 操作超出时限被主动放弃（知识库查询、打开库等），调用方应跳过而不是重试等待。
    case timeout(String)

    public var errorDescription: String? {
        switch self {
        case .invalidArguments(let s): return "参数错误: \(s)"
        case .unsupported(let s): return "不支持: \(s)"
        case .cancelled: return "已取消"
        case .contextTooLong: return "上下文超出模型窗口"
        case .llm(let s): return "LLM 错误: \(s)"
        case .storage(let s): return "存储错误: \(s)"
        case .parse(let s): return "解析错误: \(s)"
        case .permission(let s): return "权限错误: \(s)"
        case .notFound(let s): return "未找到: \(s)"
        case .internalError(let s): return "内部错误: \(s)"
        case .timeout(let s): return "超时: \(s)"
        }
    }
}

// MARK: - Utilities

public enum TokenEstimator {
    private static let asciiPerToken = 3.5
    private static let cjkPerToken = 1.3

    /// 与 Python 内核一致的粗略估算：CJK 更密，代码/JSON 按 3.5 字符/token。
    public static func estimate(_ text: String) -> Int {
        guard !text.isEmpty else { return 0 }
        if text.unicodeScalars.allSatisfy({ $0.isASCII }) {
            return max(1, Int(Double(text.count) / asciiPerToken))
        }
        var cjk = 0, other = 0
        for scalar in text.unicodeScalars {
            switch scalar.value {
            case 0x4E00...0xA000, 0x3400...0x4DC0, 0x3000...0x3100, 0xFF00...0xFFF0:
                cjk += 1
            default:
                other += 1
            }
        }
        return max(1, Int(Double(cjk) / cjkPerToken + Double(other) / asciiPerToken))
    }
}

public enum AtomicFile {
    public static func write(_ text: String, to path: String) throws {
        guard let data = text.data(using: .utf8) else {
            throw AgentError.internalError("文本无法编码为 UTF-8")
        }
        try write(data, to: path)
    }

    public static func write(_ data: Data, to path: String) throws {
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        // Data.write(.atomic) 使用同目录临时文件 + rename，不会先删除旧文件；
        // 进程在写入途中崩溃时旧文件仍然完整。
        try data.write(to: url, options: .atomic)
    }
}

public func expandPath(_ path: String) -> String {
    (path as NSString).expandingTildeInPath
}

/// 进程内简单日志（UI 通过 AgentCallbacks.onLog 接管展示）。
public enum AgentLog {
    private static let queue = DispatchQueue(label: "base-agent.log")
    /// GUI 启动时 stderr 不落盘，性能问题无法事后定位；所有日志同步追加到
    /// `~/Library/Logs/Lightning-RAG/agent-yyyyMMdd.log`（按天切分）。
    private static let logDirectory: URL = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent("Library/Logs/Lightning-RAG")
    }()
    /// 当前日志文件句柄与日期（只在 queue 内访问，天然串行）。
    nonisolated(unsafe) private static var logHandle: FileHandle?
    nonisolated(unsafe) private static var logDate: String = ""

    private static func appendToFile(_ line: String) {
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd"
        let day = f.string(from: Date())
        if day != logDate || logHandle == nil {
            try? FileManager.default.createDirectory(at: logDirectory,
                                                     withIntermediateDirectories: true)
            let path = logDirectory.appendingPathComponent("agent-\(day).log")
            if !FileManager.default.fileExists(atPath: path.path) {
                FileManager.default.createFile(atPath: path.path, contents: nil)
            }
            try? logHandle?.close()
            logHandle = try? FileHandle(forWritingTo: path)
            logHandle?.seekToEndOfFile()
            logDate = day
        }
        logHandle?.write(Data((line + "\n").utf8))
    }
    /// 进程级兜底 sink：没有任务级 sink 时才生效（App 启动时接一次）。
    nonisolated(unsafe) public static var sink: ((String) -> Void)?

    /// 任务局部 sink，优先于全局 sink：按调用链而不是进程全局路由日志。
    /// `Task` / `async` 子调用会继承，`Task.detached` 不继承 —— 后台任务的
    /// 日志不会再串进前台请求的收集器（全量测试里 KV 日志串扰的根因）。
    @TaskLocal public static var taskSink: ((String) -> Void)?

    /// 在当前任务上下文安装 sink，作用域结束自动失效，不触碰全局状态。
    public static func withTaskSink<T: Sendable>(
        _ handler: @escaping (String) -> Void,
        _ body: () async throws -> T) async rethrows -> T {
        try await $taskSink.withValue(handler, operation: body)
    }

    public static func info(_ message: String) {
        let local = taskSink
        queue.sync {
            let line = "[\(Self.timestamp())] \(message)"
            FileHandle.standardError.write((line + "\n").data(using: .utf8)!)
            Self.appendToFile(line)
            if let local { local(line) } else { sink?(line) }
        }
    }

    public static func warning(_ message: String) { info("WARN \(message)") }
    public static func error(_ message: String) { info("ERROR \(message)") }

    private static func timestamp() -> String {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f.string(from: Date())
    }
}
