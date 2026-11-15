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

/// 只创建「全部 done」的计划、从不调用实际工具的模型。
private final class PlanOnlyLLM: LLMService, @unchecked Sendable {
    let identifier = "plan-only"
    let supportsTools = true
    private let lock = NSLock()
    private var _calls = 0

    var calls: Int {
        lock.lock(); defer { lock.unlock() }
        return _calls
    }

    func chatStream(messages: [ChatMessage], tools: [JSONValue],
                    temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        lock.lock()
        let index = _calls
        _calls += 1
        lock.unlock()

        return AsyncThrowingStream { continuation in
            if index == 0 {
                let steps: [JSONValue] = [
                    .object(["title": .string("抓取新闻"), "status": .string("done")]),
                    .object(["title": .string("总结新闻"), "status": .string("done")]),
                ]
                continuation.yield(LLMStreamEvent(
                    kind: .done,
                    toolCalls: [ToolCall(
                        id: "plan_1", name: "update_plan",
                        arguments: ["op": .string("create"),
                                    "goal": .string("获取并总结今日新闻"),
                                    "steps": .array(steps)])]))
            } else {
                continuation.yield(LLMStreamEvent(kind: .content,
                                                  content: "已完成：今日新闻共 3 条。"))
                continuation.yield(LLMStreamEvent(kind: .done))
            }
            continuation.finish()
        }
    }
}

private final class PlanSnapshotCallbacks: AgentCallbacks {
    private let lock = NSLock()
    private var _plans: [Plan] = []

    var plans: [Plan] {
        lock.lock(); defer { lock.unlock() }
        return _plans
    }

    func onPlanUpdated(_ plan: Plan?) {
        guard let plan else { return }
        lock.lock()
        _plans.append(plan)
        lock.unlock()
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

    /// 模型把计划直接写成全部 done、随后只用文字收尾：不能算任务完成。
    /// 内核必须继续要求真实工具；最终把无工具验证的 done 降级为 skipped。
    func testPlanMarkedDoneWithoutToolsIsNotAccepted() async throws {
        let workspace = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-plan-only-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: workspace) }

        var config = AgentConfig()
        config.workspace = workspace.path
        config.llmProvider = "api"
        config.embeddingProvider = "local"
        config.maxIterations = 8

        let llm = PlanOnlyLLM()
        let callbacks = PlanSnapshotCallbacks()
        let agent = Agent(llm: llm, config: config, callbacks: callbacks)
        let answer = await agent.run("请按步骤获取并总结今日新闻")

        XCTAssertGreaterThanOrEqual(llm.calls, 5,
            "只更新计划不能直接收尾，必须多次要求实际工具；实际 \(llm.calls) 次")
        XCTAssertTrue(answer.contains("没有实际工具验证") || answer.contains("skipped"),
                      "最终答复必须标注未验证：\(answer)")
        let last = callbacks.plans.last
        XCTAssertTrue(last?.steps.allSatisfy { $0.status == "skipped" } ?? false,
                      "零工具完成的计划必须降级为 skipped：\(last?.steps ?? [])")
        agent.close()
    }

    func testToolRegistrySchemasContainCoreTools() {
        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory() + "/base-agent-schema-\(UUID().uuidString)"
        let agent = Agent(llm: ScriptedLLM(), config: config, callbacks: NoopCallbacks())
        let names = Set(agent.tools.allTools.map(\.name))
        for expected in ["file_tools", "web_scan", "rag_tools", "long_memory", "skill_tools",
                         "code_run", "shell_run", "ask_user",
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
