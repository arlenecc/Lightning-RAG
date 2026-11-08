import XCTest
@testable import BaseAgentCore

/// P1-a：日志按任务隔离 + 后台取消看门狗。
final class AgentLogIsolationTests: XCTestCase {

    private final class Box: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var lines: [String] = []
        func append(_ line: String) {
            lock.lock(); lines.append(line); lock.unlock()
        }
    }

    private final class LogCallbacks: AgentCallbacks, @unchecked Sendable {
        private let lock = NSLock()
        private(set) var lines: [String] = []
        func onLog(_ line: String) {
            lock.lock(); lines.append(line); lock.unlock()
        }
    }

    private final class LoggingLLM: LLMService, @unchecked Sendable {
        let identifier = "logging"
        let supportsTools = true
        func chatStream(messages: [ChatMessage], tools: [JSONValue],
                        temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
            AsyncThrowingStream { continuation in
                // 模拟 MLX/GGUF 等底层组件经 AgentLog 输出的诊断日志。
                AgentLog.info("LOW_LEVEL_MARKER")
                continuation.yield(LLMStreamEvent(kind: .content, content: "ok"))
                continuation.yield(LLMStreamEvent(kind: .done,
                                                  usage: ["total_tokens": 1]))
                continuation.finish()
            }
        }
    }

    // MARK: 日志隔离

    func testConcurrentTaskSinksDoNotCrossTalk() async {
        let a = Box()
        let b = Box()
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                await AgentLog.withTaskSink({ a.append($0) }) {
                    AgentLog.info("FROM_A")
                    await Task.yield()
                }
            }
            group.addTask {
                await AgentLog.withTaskSink({ b.append($0) }) {
                    AgentLog.info("FROM_B")
                    await Task.yield()
                }
            }
        }
        XCTAssertTrue(a.lines.contains { $0.contains("FROM_A") })
        XCTAssertFalse(a.lines.contains { $0.contains("FROM_B") }, "A 不应收到 B 的日志")
        XCTAssertTrue(b.lines.contains { $0.contains("FROM_B") })
        XCTAssertFalse(b.lines.contains { $0.contains("FROM_A") }, "B 不应收到 A 的日志")
    }

    func testDetachedTaskLogsDoNotLeakIntoForegroundSink() async {
        let box = Box()
        await AgentLog.withTaskSink({ box.append($0) }) {
            let detached = Task.detached {
                AgentLog.info("FROM_DETACHED")
            }
            _ = await detached.value
            AgentLog.info("FROM_SCOPE")
        }
        XCTAssertFalse(box.lines.contains { $0.contains("FROM_DETACHED") },
                       "后台 detached 任务正是原先串扰的来源")
        XCTAssertTrue(box.lines.contains { $0.contains("FROM_SCOPE") })
    }

    func testAgentRunRoutesLowLevelLogsToItsOwnCallbacks() async {
        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory() + "agent-log-\(UUID().uuidString)"
        config.llmProvider = "api"
        config.embeddingProvider = "local"
        config.maxIterations = 1
        let callbacks = LogCallbacks()
        let agent = Agent(llm: LoggingLLM(), config: config, callbacks: callbacks)
        defer { agent.close() }

        // 寒暄现在是本地直答（不创建 LLM stream），必须用普通任务轮才能
        // 覆盖「底层组件日志经 taskSink 回到本次请求 callbacks」的链路。
        _ = await agent.run("请写个脚本")
        XCTAssertTrue(callbacks.lines.contains { $0.contains("LOW_LEVEL_MARKER") },
                      "底层 AgentLog 应经 taskSink 流到本次请求的 callbacks："
                      + "\(callbacks.lines)")
    }

    // MARK: 后台取消看门狗

    func testWaitForTaskExitReturnsTrueForFinishedTask() async {
        let task = Task<Void, Never> {}
        let exited = await Agent.waitForTaskExit(task, timeoutNanos: 1_000_000_000)
        XCTAssertTrue(exited)
    }

    func testWaitForTaskExitTimesOutWithoutHanging() async {
        // 模拟不响应取消、卡在 GPU 上的后台生成。
        let task = Task<Void, Never> {
            try? await Task.sleep(nanoseconds: 3_000_000_000)
        }
        let start = Date()
        let exited = await Agent.waitForTaskExit(task, timeoutNanos: 80_000_000)
        XCTAssertFalse(exited)
        XCTAssertLessThan(Date().timeIntervalSince(start), 1.0,
                          "超时后必须立即返回，不能等后台任务真正结束")
        task.cancel()
    }
}
