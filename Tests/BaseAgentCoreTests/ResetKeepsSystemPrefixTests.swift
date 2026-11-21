import XCTest
@testable import BaseAgentCore

/// 清空对话保留预热 system 前缀：冻结前缀不丢、会话历史清空、默认 reset 不受影响。
final class ResetKeepsSystemPrefixTests: XCTestCase {
    private final class PlainLLM: LLMService {
        let identifier = "reset-llm"
        let supportsTools = false
        let contextWindow = 4096
        func chatStream(messages: [ChatMessage], tools: [JSONValue],
                        temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
            AsyncThrowingStream { c in
                c.yield(LLMStreamEvent(kind: .content, content: "ok"))
                c.yield(LLMStreamEvent(kind: .done)); c.finish()
            }
        }
    }

    private func makeAgent() throws -> (Agent, URL) {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("reset-prefix-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var config = AgentConfig()
        config.workspace = root.path
        config.maxIterations = 1
        config.llmProvider = "api"
        let agent = Agent(llm: PlainLLM(), config: config, callbacks: NoopCallbacks())
        return (agent, root)
    }

    /// 段内冻结的 system 前缀在「保留式清空」后必须逐字节不变，
    /// 否则下一轮 prompt 前缀与预热 KV 快照失配，会整段重建。
    func testKeepSystemPrefixKeepsFrozenBytesIdentical() throws {
        let (agent, root) = try makeAgent()
        defer { try? FileManager.default.removeItem(at: root) }
        agent.freezeSystemPrefixForTesting()
        let before = agent.frozenSystemPrefixForTesting()
        XCTAssertNotNil(before)
        XCTAssertFalse(before?.isEmpty ?? true)
        agent.reset(keepSystemPrefix: true)
        XCTAssertEqual(agent.frozenSystemPrefixForTesting(), before,
                       "保留式清空不得丢掉冻结前缀")
        XCTAssertEqual(agent.systemPrompt(), before!, "systemPrompt() 必须返回同一份")
    }

    /// 默认 reset 仍按旧语义清掉冻结前缀（配置重载等路径不受影响）。
    func testDefaultResetClearsFrozenPrefix() throws {
        let (agent, root) = try makeAgent()
        defer { try? FileManager.default.removeItem(at: root) }
        agent.freezeSystemPrefixForTesting()
        XCTAssertNotNil(agent.frozenSystemPrefixForTesting())
        agent.reset()
        XCTAssertNil(agent.frozenSystemPrefixForTesting(),
                     "默认 reset 应保持旧语义：重新读 L0/L1")
        XCTAssertFalse(agent.systemPrompt().isEmpty)
    }

    /// 两种 reset 都必须清空会话历史（只剩 system 一条消息）。
    func testBothResetsClearHistory() throws {
        let (agent, root) = try makeAgent()
        defer { try? FileManager.default.removeItem(at: root) }
        agent.reset(keepSystemPrefix: true)
        XCTAssertEqual(agent.history().count, 1, "清空后只剩 system")
        XCTAssertEqual(agent.history().first?.role, .system)
        agent.reset()
        XCTAssertEqual(agent.history().count, 1)
    }
}

private final class NoopCallbacks: AgentCallbacks {}
