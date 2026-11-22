import XCTest
@testable import BaseAgentCore

/// 记忆管理开关：默认关闭，L0/L1 不常驻 system prompt；记忆工具仍可用。
final class MemoryHierarchyToggleTests: XCTestCase {
    private final class PlainLLM: LLMService {
        let identifier = "mem-toggle"
        let supportsTools = false
        let contextWindow = 4096
        func chatStream(messages: [ChatMessage], tools: [JSONValue],
                        temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
            AsyncThrowingStream { c in c.yield(LLMStreamEvent(kind: .done)); c.finish() }
        }
    }

    private func makeAgent(enabled: Bool? = nil) throws -> (Agent, URL) {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mem-toggle-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var config = AgentConfig()
        config.workspace = root.path
        config.llmProvider = "api"
        if let enabled { config.memoryHierarchyInSystemPrompt = enabled }
        return (Agent(llm: PlainLLM(), config: config, callbacks: NoopCallbacks()), root)
    }

    func testDefaultIsOff() {
        XCTAssertFalse(AgentConfig().memoryHierarchyInSystemPrompt)
    }

    /// 默认（关闭）：system prompt 只有 BASE，不含 L0/L1 标题与文件正文。
    func testOffExcludesL0L1() throws {
        let (agent, root) = try makeAgent()
        defer { try? FileManager.default.removeItem(at: root) }
        let system = agent.systemPrompt()
        XCTAssertTrue(system.hasPrefix("You are base-agent"))
        XCTAssertFalse(system.contains("# [L0 META-SOP: memory_management_sop]"))
        XCTAssertFalse(system.contains("# [L1 Global Memory Insight]"))
    }

    /// 开启：L0/L1 常驻；同一 Agent 运行时切换后必须刷新（签名含开关）。
    func testOnIncludesL0L1AndToggleRefreshes() throws {
        let (agent, root) = try makeAgent(enabled: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let on = agent.systemPrompt()
        XCTAssertTrue(on.contains("# [L0 META-SOP: memory_management_sop]"))
        XCTAssertTrue(on.contains("# [L1 Global Memory Insight]"))

        agent.config.memoryHierarchyInSystemPrompt = false
        let off = agent.systemPrompt()
        XCTAssertFalse(off.contains("# [L0 META-SOP: memory_management_sop]"))
        XCTAssertFalse(off.contains("# [L1 Global Memory Insight]"))
    }

    /// 配置持久化：encode/decode 后开关值保持（默认 false 与新 true 都要验证）。
    func testConfigRoundTrip() throws {
        var config = AgentConfig()
        XCTAssertFalse(config.memoryHierarchyInSystemPrompt)
        config.memoryHierarchyInSystemPrompt = true
        let dict = try XCTUnwrap(config.export() as? [String: Any])
        XCTAssertEqual(dict["memory_hierarchy_in_system_prompt"] as? Bool, true)
    }

    /// 关闭时 start_long_term_update 仍返回 L1 全文与自包含分类指引（不再假设 L0 常驻）。
    func testLongTermUpdateSelfContainedWhenOff() async throws {
        let (agent, root) = try makeAgent()
        defer { try? FileManager.default.removeItem(at: root) }
        try "L1 内容\nL2: 现空\n".write(
            to: root.appendingPathComponent("memory/global_mem_insight.txt"),
            atomically: true, encoding: .utf8)
        let registry = ToolRegistry(config: agent.config)
        registry.session = agent.session
        for _ in 0..<6 { _ = agent.session.startTurn(prompt: "t") }
        let tool = StartLongTermUpdateTool(config: agent.config,
                                           callbacks: NoopCallbacks(),
                                           registry: registry)
        let result = try await tool.run([:])
        XCTAssertTrue(result.success, result.error)
        XCTAssertTrue(result.output.contains("L1 内容"), "仍返回 L1 全文: \(result.output)")
        XCTAssertTrue(result.output.contains("L0 未常驻"), "给出自包含分类指引: \(result.output)")
    }

    /// 关闭时 update_working_checkpoint 仍可写入 session key_info（不依赖 L0/L1）。
    func testWorkingCheckpointWorksWhenOff() async throws {
        let (agent, root) = try makeAgent()
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = ToolRegistry(config: agent.config)
        registry.session = agent.session
        let tool = UpdateWorkingCheckpointTool(config: agent.config,
                                                callbacks: NoopCallbacks(),
                                                registry: registry)
        let result = try await tool.run(["key_info": .string("验证过的路径: /tmp/x")])
        XCTAssertTrue(result.success)
        XCTAssertTrue(result.output.contains("working checkpoint updated"))
    }
}

private final class NoopCallbacks: AgentCallbacks {}
