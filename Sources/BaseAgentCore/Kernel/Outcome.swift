import Foundation

// MARK: - 三态自动机状态

/// 三态自动机状态。
///
/// - THINK       决策态：把折叠后的上下文交给 LLM，取得下一步意图。
/// - ACT         执行态：分发工具，能力全部在这一态落地。
/// - CONSOLIDATE 沉淀态：不调 LLM，只做信息减熵（L3 折叠 / L4 蒸馏）。
public enum KernelState: String, Sendable, CaseIterable {
    case think = "think"
    case act = "act"
    case consolidate = "consolidate"
}

/// 自动机的终止边标签（对齐 Python `ExitReason`）。
public enum ExitReason: String, Sendable, Equatable {
    case done = "done"
    /// 工具要求中断整轮（保留给未来的阻塞语义）。
    case interrupt = "interrupt"
    case maxTurns = "max_turns"
    case cancelled = "cancelled"
    case error = "error"
}

/// 转移表的目标：进入某个状态，或直接退出。
public enum KernelTransition: Sendable, Equatable {
    case state(KernelState)
    case exit
}

/// (当前状态, 事件) -> 下一状态；未知组合一律终止，避免自动机静默卡死。
private let kernelTransitions: [String: KernelTransition] = [
    "think:has_calls": .state(.act),
    "think:no_calls": .exit,
    "think:retry": .state(.think),
    "think:error": .exit,
    "act:continue": .state(.think),
    "act:consolidate": .state(.consolidate),
    "act:interrupt": .exit,
    "act:done": .exit,
    "consolidate:done": .state(.think),
]

public func nextState(_ current: KernelState, _ event: String) -> KernelTransition {
    kernelTransitions["\(current.rawValue):\(event)"] ?? .exit
}

// MARK: - 内核产物

/// 一次 `KernelLoop.run()` 的产物。
public struct KernelResult: Sendable, Equatable {
    public var text: String
    public var reason: ExitReason
    public var turns: Int
    public var distilled: Bool

    public init(text: String = "", reason: ExitReason = .done,
                turns: Int = 0, distilled: Bool = false) {
        self.text = text
        self.reason = reason
        self.turns = turns
        self.distilled = distilled
    }
}

/// 一次工具分发的产物 —— 直接对应 GenericAgent / Python 的 `StepOutcome`。
///
/// - `data`       回灌给模型的内容（已在 L1 级裁剪过）。
/// - `nextPrompt` 追加到下一轮 user 消息的提示；空串表示任务完成。
/// - `shouldExit` 强制中断整条自动机。
public struct StepOutcome: Sendable {
    public var data: String
    public var nextPrompt: String?
    public var shouldExit: Bool

    public init(data: String = "", nextPrompt: String? = nil, shouldExit: Bool = false) {
        self.data = data
        self.nextPrompt = nextPrompt
        self.shouldExit = shouldExit
    }

    public var finished: Bool {
        shouldExit || (nextPrompt ?? "").isEmpty
    }
}

/// 一次 THINK 的产物。
public struct LLMResponse: Sendable {
    public var content: String
    public var reasoning: String
    public var toolCalls: [ToolCall]
    public var usage: [String: Int]
    /// 模型按协议输出的 `<summary>` 物理快照（L2 级摘要的原料）。
    public var summary: String
    /// 模型原始输出；只有本地 MLX 后端会填，HTTP 后端恒为空。
    public var rawContent: String
    /// 本轮输出被截断（工具调用未生成完整）。
    public var truncated: Bool

    public init(content: String = "", reasoning: String = "",
                toolCalls: [ToolCall] = [], usage: [String: Int] = [:],
                summary: String = "", rawContent: String = "",
                truncated: Bool = false) {
        self.content = content
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.usage = usage
        self.summary = summary
        self.rawContent = rawContent
        self.truncated = truncated
    }

    public var hasCalls: Bool { !toolCalls.isEmpty }
}

/// 工具结果在历史中的物理快照（对应 Python 的 `{id, name, content}`）。
public struct ToolResultRecord: Sendable, Equatable {
    public var id: String
    public var name: String
    public var content: String

    public init(id: String, name: String, content: String) {
        self.id = id
        self.name = name
        self.content = content
    }
}

/// 一轮的完整记录。Session 用它在 L3 阶段折叠历史。
///
/// 「一轮」= 一次 THINK + 随后的 ACT。折叠时作为一个整体丢弃，
/// 因此永远不会产生悬挂的 tool 消息。
///
/// 使用引用类型是为了保留 Python 中 `finish_turn(rec, ...)` 原地写入的语义。
public final class TurnRecord: @unchecked Sendable {
    public var index: Int
    /// 注入的 user 内容（已压缩）。
    public var prompt: String
    /// 模型文本（已压缩）。
    public var assistantText: String
    /// 双视图历史的第二视图：模型原始输出；HTTP 后端为空。
    public var rawContent: String
    public var toolCalls: [ToolCall]
    public var toolResults: [ToolResultRecord]
    /// L2 摘要（本轮一行）。
    public var summary: String
    /// 估算 token 数。
    public var tokens: Int
    /// 可选图像附件（JPEG/PNG 字节），仅首条用户消息携带，用于多模态模型。
    /// 文本/无多模态能力的后端忽略此字段。
    public var images: [Data]?

    public init(index: Int = 0, prompt: String = "", assistantText: String = "",
                rawContent: String = "", toolCalls: [ToolCall] = [],
                toolResults: [ToolResultRecord] = [], summary: String = "",
                tokens: Int = 0, images: [Data]? = nil) {
        self.index = index
        self.prompt = prompt
        self.assistantText = assistantText
        self.rawContent = rawContent
        self.toolCalls = toolCalls
        self.toolResults = toolResults
        self.summary = summary
        self.tokens = tokens
        self.images = images
    }

    /// L2/L3：本轮折叠成的一行摘要。
    public func digestLine() -> String {
        if !summary.isEmpty {
            return "[T\(index)] \(summary)"
        }
        if !toolCalls.isEmpty {
            let names = toolCalls.prefix(4).map(\.name).joined(separator: ",")
            let tail = toolCalls.count > 4 ? "..." : ""
            return "[T\(index)] 调用 \(names)\(tail)"
        }
        if !assistantText.isEmpty {
            return "[T\(index)] \(String(assistantText.prefix(40)))"
        }
        return "[T\(index)] (空轮)"
    }
}

// MARK: - 经验沉淀回调

/// 经验沉淀回调。默认 no-op —— 内核本身不做任何沉淀。
public protocol KernelHooks: AnyObject {
    /// 折叠/沉淀回调。返回 true 表示上下文真的变小了。
    func consolidated(_ session: Session, reason: String) async -> Bool
    /// 折叠边界通知（周期性 / 90% 收缩 / 提前折叠 / 异常重试）。
    /// Agent 在此刷新 system 前缀（L0/L1 变更延迟到边界生效）并预热 KV。
    func didFold(_ session: Session, reason: String, tools: [JSONValue]) async
    func finished(_ session: Session, text: String)
    func failed(_ session: Session, error: String)
}

public extension KernelHooks {
    func consolidated(_ session: Session, reason: String) async -> Bool { false }
    func didFold(_ session: Session, reason: String, tools: [JSONValue]) async {}
    func finished(_ session: Session, text: String) {}
    func failed(_ session: Session, error: String) {}
}

public final class NoopKernelHooks: KernelHooks {
    public init() {}
}

// MARK: - L4：记忆蒸馏的结果结构

public struct DistilledSkill: Sendable, Equatable {
    public var name: String
    public var summary: String
    public var whenToUse: String
    public var steps: String

    public init(name: String = "", summary: String = "",
                whenToUse: String = "", steps: String = "") {
        self.name = name
        self.summary = summary
        self.whenToUse = whenToUse
        self.steps = steps
    }

    public var isEmpty: Bool { name.isEmpty }
}

public struct DistillResult: Sendable, Equatable {
    public var facts: [String]
    public var lessons: [String]
    public var skill: DistilledSkill

    public init(facts: [String] = [], lessons: [String] = [],
                skill: DistilledSkill = DistilledSkill()) {
        self.facts = facts
        self.lessons = lessons
        self.skill = skill
    }
}

// MARK: - L4 JSON 提取 / 归一化

/// 从 LLM 提纯输出里稳健地取出 JSON（容忍 ```json 围栏与尾随文字）。
public func l4ExtractJSON(_ text: String) -> [String: JSONValue] {
    var s = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !s.isEmpty else { return [:] }
    // 去掉开头的 ``` 或 ```json 围栏（只剥第一行）。
    if s.hasPrefix("```") {
        if let newline = s.firstIndex(of: "\n") {
            s = String(s[s.index(after: newline)...])
        }
    }
    if s.hasSuffix("```") {
        s = String(s.dropLast(3))
    }
    s = s.trimmingCharacters(in: .whitespacesAndNewlines)
    if let value = JSONValue.parse(s), let obj = value.objectValue {
        return obj
    }
    // 尾随文字：取第一个 { 到最后一个 } 之间。
    guard let start = s.firstIndex(of: "{"), let end = s.lastIndex(of: "}"),
          start < end else { return [:] }
    let candidate = String(s[start...end])
    if let value = JSONValue.parse(candidate), let obj = value.objectValue {
        return obj
    }
    return [:]
}

/// 归一化提纯结果，所有字段都有确定类型，避免下游 KeyError/TypeError。
public func l4Normalize(_ data: [String: JSONValue]) -> DistillResult {
    func strList(_ v: JSONValue?, limit: Int) -> [String] {
        guard let arr = v?.arrayValue else { return [] }
        var out: [String] = []
        for item in arr {
            let s: String?
            switch item {
            case .string(let x): s = x
            case .int(let x): s = String(x)
            case .double(let x): s = String(x)
            default: s = nil
            }
            if let s, !s.isEmpty { out.append(s) }
            if out.count >= limit { break }
        }
        return Array(out.prefix(limit))
    }

    var name = "", summary = "", whenToUse = "", steps = ""
    if let skill = data["skill"]?.objectValue {
        name = (skill["name"]?.stringValue ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        summary = (skill["summary"]?.stringValue ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        whenToUse = (skill["when_to_use"]?.stringValue ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        steps = (skill["steps"]?.stringValue ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    }
    // skill 名只允许 [a-z0-9-]，防止写穿目录。
    name = KernelRegex.replace("[^a-z0-9\\-]", in: name.lowercased(), with: "-")
        .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
    name = String(name.prefix(64))
    return DistillResult(
        facts: strList(data["facts"], limit: 5),
        lessons: strList(data["lessons"], limit: 3),
        skill: DistilledSkill(
            name: name,
            summary: String(summary.prefix(200)),
            whenToUse: String(whenToUse.prefix(400)),
            steps: String(steps.prefix(6000))
        )
    )
}
