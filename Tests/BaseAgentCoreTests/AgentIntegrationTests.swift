import XCTest
@testable import BaseAgentCore

private final class ScriptedLLM: LLMService, @unchecked Sendable {
    let identifier = "scripted"
    let supportsTools = true
    private let lock = NSLock()
    private var call = 0

    func chatStream(messages: [ChatMessage], tools: [JSONValue],
                    temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        lock.lock()
        let index = call
        call += 1
        lock.unlock()

        return AsyncThrowingStream { continuation in
            if index == 0 {
                let arguments = JSONValue.object([
                    "op": .string("write"),
                    "path": .string("out.txt"),
                    "content": .string("hello from tool"),
                ]).jsonString()
                continuation.yield(LLMStreamEvent(
                    kind: .done,
                    toolCalls: [ToolCall(id: "call_1", name: "file_tools",
                                         arguments: ["op": .string("write"),
                                                      "path": .string("out.txt"),
                                                      "content": .string("hello from tool")],
                                         rawArguments: arguments)]))
            } else {
                continuation.yield(LLMStreamEvent(kind: .content, content: "Done."))
                continuation.yield(LLMStreamEvent(kind: .done,
                                                  usage: ["total_tokens": 12]))
            }
            continuation.finish()
        }
    }
}

/// 记录 reset 是否失效了跨轮 KV cache。
private final class InvalidatingLLM: LLMService, KVCacheInvalidating, @unchecked Sendable {
    let identifier = "invalidating"
    let supportsTools = true
    private let lock = NSLock()
    private var invalidations = 0

    func chatStream(messages: [ChatMessage], tools: [JSONValue],
                    temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }

    func invalidateKVCache() {
        lock.lock()
        invalidations += 1
        lock.unlock()
    }

    var invalidationCount: Int {
        lock.lock(); defer { lock.unlock() }
        return invalidations
    }
}

final class AgentIntegrationTests: XCTestCase {
    func testAgentRunsToolThenFinishes() async throws {
        let workspace = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-agent-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: workspace) }

        var config = AgentConfig()
        config.workspace = workspace.path
        config.llmProvider = "api"
        config.embeddingProvider = "local"
        config.maxIterations = 3

        let agent = Agent(llm: ScriptedLLM(), config: config, callbacks: NoopCallbacks())
        let answer = await agent.run("write file")
        // 收尾自动把代建计划标完成时会追加计划说明，正文前缀不变。
        XCTAssertTrue(answer.hasPrefix("Done."), "实际答复：\(answer)")
        let written = try String(contentsOf: workspace.appendingPathComponent("out.txt"),
                                 encoding: .utf8)
        XCTAssertEqual(written, "hello from tool")
        agent.close()
    }

    func testToolRegistrySchemasContainCoreTools() {
        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory() + "/base-agent-schema-\(UUID().uuidString)"
        let agent = Agent(llm: ScriptedLLM(), config: config, callbacks: NoopCallbacks())
        let names = Set(agent.tools.allTools.map(\.name))
        for expected in ["file_tools", "web_scan", "rag_tools", "long_memory", "skill_tools",
                         "code_run", "shell_run", "ask_user", "work_memory",
                         "update_working_checkpoint", "start_long_term_update", "update_plan"] {
            XCTAssertTrue(names.contains(expected), "缺少工具 \(expected)")
        }
        XCTAssertTrue(agent.dispatcher.schemas().count >= names.count)
        agent.close()
    }

    func testResetInvalidatesKVCache() {
        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory()
            + "/base-agent-reset-\(UUID().uuidString)"
        config.llmProvider = "api"
        let llm = InvalidatingLLM()
        let agent = Agent(llm: llm, config: config, callbacks: NoopCallbacks())
        agent.reset()
        XCTAssertEqual(llm.invalidationCount, 1)
        agent.close()
    }
}
