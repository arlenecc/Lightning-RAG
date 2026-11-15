import Foundation

// MARK: - ask_user

/// `ask_user`：暂停自动流程，通过 AgentCallbacks 向用户提问，回答作为工具结果。
///
/// 无 UI callbacks（headless / 未绑定）时返回明确错误，而不是静默成功。
/// 回答必须作为工具结果回注，继续同一轮推理；不能中断整轮，否则用户回答会被丢弃。
public final class AskUserTool: AgentTool {
    public let name = "ask_user"
    public let description =
        "Ask the user a clarifying question or request confirmation/info. "
        + "Use when information is missing or a non-destructive decision is needed."

    private let config: AgentConfig
    private let callbacks: AgentCallbacks?
    private weak var registry: ToolRegistry?

    public var parameters: JSONValue {
        ToolSupport.objectSchema(
            properties: [
                "prompt": ToolSupport.stringProperty("Question to ask the user."),
            ],
            required: ["prompt"])
    }

    public init(config: AgentConfig, callbacks: AgentCallbacks?, registry: ToolRegistry? = nil) {
        self.config = config
        self.callbacks = callbacks
        self.registry = registry
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        do {
            let prompt = try ToolSupport.requireString(args, "prompt")
            guard let callbacks else {
                return .fail(
                    "ask_user is unavailable in this context (no UI callbacks bound).")
            }
            guard let answer = callbacks.askUser(prompt) else {
                return .fail("User dismissed the question.")
            }
            // 用户回答澄清问题本身就是引导；计入技能学习信号（计划需仍在进行）。
            if let session = registry?.session,
               let plan = session.getPlan(),
               !session.planOpenSteps().isEmpty {
                session.noteSkillGuidance(
                    planGoal: plan.goal,
                    corrective: SkillManager.isCorrectiveFeedback(answer),
                    supplemental: true)
            }
            return ToolResult(success: true, output: answer)
        } catch {
            return ToolSupport.failure(error)
        }
    }
}
