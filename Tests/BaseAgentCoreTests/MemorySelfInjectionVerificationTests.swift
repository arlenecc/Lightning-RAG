import XCTest
@testable import BaseAgentCore

/// 端到端验证：L0/L1 默认不常驻时，两个记忆工具自带所需 prompt，
/// 且工具结果按既有 tool role 回灌进下一轮对话（使用时自行注入）。
final class MemorySelfInjectionVerificationTests: XCTestCase {
    private final class CapturingLLM: LLMService, @unchecked Sendable {
        let identifier = "verify-inject"
        let supportsTools = false
        let contextWindow = 8192
        var systems: [String] = []
        var prompts: [[ChatMessage]] = []

        func chatStream(messages: [ChatMessage], tools: [JSONValue],
                        temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
            systems.append(messages.first(where: { $0.role == .system })?.content ?? "")
            prompts.append(messages)
            return AsyncThrowingStream { c in
                c.yield(LLMStreamEvent(kind: .content, content: "收到"))
                c.yield(LLMStreamEvent(kind: .done)); c.finish()
            }
        }
    }

    private func makeAgent(_ llm: CapturingLLM) throws -> (Agent, URL) {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("verify-inject-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("memory"), withIntermediateDirectories: true)
        try "# L1 索引\nL3: verify_sop(验证用)\n".write(
            to: root.appendingPathComponent("memory/global_mem_insight.txt"),
            atomically: true, encoding: .utf8)
        var config = AgentConfig()
        config.workspace = root.path
        config.llmProvider = "api"
        config.maxIterations = 6
        return (Agent(llm: llm, config: config, callbacks: NoopCallbacks()), root)
    }

    /// 1) 默认关闭：所有回合的 system 里都没有 L0/L1 标题。
    func testDefaultSystemNeverContainsL0L1() async throws {
        let llm = CapturingLLM()
        let (agent, root) = try makeAgent(llm)
        defer { try? FileManager.default.removeItem(at: root) }
        _ = await agent.run("请说明当前工作目录的用途")
        XCTAssertFalse(llm.systems.isEmpty, "真实任务必须进入 LLM")
        XCTAssertTrue(llm.systems.allSatisfy {
            !$0.contains("# [L0 META-SOP") && !$0.contains("# [L1 Global Memory Insight]")
        })
    }

    /// 2) start_long_term_update：关闭时常驻时返回合法 JSON，四字段齐全，
    ///    自带分类规则且不引用不可见的 L0，L1 全文仍由工具独立维护。
    func testLongTermUpdatePayloadSelfContainedAndValid() async throws {
        let llm = CapturingLLM()
        let (agent, root) = try makeAgent(llm)
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = ToolRegistry(config: agent.config)
        registry.session = agent.session
        for _ in 0..<6 { _ = agent.session.startTurn(prompt: "t") }
        let tool = StartLongTermUpdateTool(config: agent.config,
                                           callbacks: NoopCallbacks(), registry: registry)
        let result = try await tool.run(["focus": .string("verify")])
        XCTAssertTrue(result.success, result.error)
        let data = try XCTUnwrap(result.output.data(using: .utf8))
        let json = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["status"] as? String, "DISTILL_READY")
        XCTAssertTrue((json["l1_current"] as? String)?.contains("# L1 索引") ?? false)
        let prompt = json["prompt"] as? String ?? ""
        XCTAssertFalse(prompt.contains("按L0提取"), "关闭时常驻时不得引用不可见 L0: \(prompt)")
        XCTAssertTrue(prompt.contains("按下方 next 的分类规则提取"))
        XCTAssertTrue((json["next"] as? String)?.contains("L0 未常驻") ?? false)
        XCTAssertTrue((json["next"] as? String)?.contains("global_mem.txt") ?? false)
        XCTAssertTrue((json["next"] as? String)?.contains("xxx_sop.md") ?? false)
    }

    /// 3) update_working_checkpoint 写入的 key_info 会被 Session.anchorPrompt
    ///    在下一轮回注到 user 消息中（自行注入，不依赖 L0/L1）。
    func testWorkingCheckpointReinjectedNextTurn() async throws {
        let llm = CapturingLLM()
        let (agent, root) = try makeAgent(llm)
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = ToolRegistry(config: agent.config)
        registry.session = agent.session
        let tool = UpdateWorkingCheckpointTool(config: agent.config,
                                                callbacks: NoopCallbacks(), registry: registry)
        let result = try await tool.run(["key_info": .string("验证路径 /tmp/verify-port=8080")])
        XCTAssertTrue(result.success)
        let anchor = agent.session.anchorPrompt(extra: "")
        XCTAssertTrue(anchor.contains("验证路径 /tmp/verify-port=8080"))
        XCTAssertTrue(anchor.contains("### [WORKING MEMORY]"))
    }

    /// 4) 超长 L1 时 compactJSON 字段级裁剪后仍是合法 JSON（不会截成非法串）。
    func testOversizedPayloadStaysValidJSON() async throws {
        let llm = CapturingLLM()
        let (agent, root) = try makeAgent(llm)
        defer { try? FileManager.default.removeItem(at: root) }
        try String(repeating: "L1 索引行很长。", count: 900).write(
            to: root.appendingPathComponent("memory/global_mem_insight.txt"),
            atomically: true, encoding: .utf8)
        let registry = ToolRegistry(config: agent.config)
        registry.session = agent.session
        for _ in 0..<6 { _ = agent.session.startTurn(prompt: "t") }
        let tool = StartLongTermUpdateTool(config: agent.config,
                                           callbacks: NoopCallbacks(), registry: registry)
        let result = try await tool.run([:])
        XCTAssertTrue(result.success)
        let data = try XCTUnwrap(result.output.data(using: .utf8))
        let json = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        for key in ["status", "l1_current", "prompt", "next"] {
            XCTAssertNotNil(json[key], "裁剪后不得丢字段: \(key)")
        }
    }
}

private final class NoopCallbacks: AgentCallbacks {}
