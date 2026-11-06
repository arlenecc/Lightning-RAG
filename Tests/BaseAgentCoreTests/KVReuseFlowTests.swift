import XCTest
@testable import BaseAgentCore

/// 用户交互全流程 KV cache 复用实测：
/// 同一 Agent 连续多轮 `run()`，每轮经 Kernel/Session 组装完整消息后交给
/// 真实 MLX 模型。断言全量 prefill 只发生在合法位置（首次构建、截断工具
/// 调用后的保护性重建），其余每轮 LLM 调用都必须命中跨轮 KV
/// （`[mlx-kv] reuse=true cached>0`、prefill 毫秒级）——
/// 若正常轮次也全量重建，即用户感知的「每轮都卡顿」。
final class KVReuseFlowTests: XCTestCase {

    // MARK: - 日志捕获

    private final class LogCollector: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var lines: [String] = []
        func append(_ line: String) {
            lock.lock(); lines.append(line); lock.unlock()
        }
    }

    /// 同时捕获 AgentLog 与 kernel/agent 回调日志。
    private final class CollectingCallbacks: AgentCallbacks {
        let collector: LogCollector
        init(_ collector: LogCollector) { self.collector = collector }
        func onLog(_ line: String) { collector.append(line) }
    }

    // MARK: - 门控

    private func modelDirectory() throws -> URL {
        let env = ProcessInfo.processInfo.environment
        let path = env["MLX_KV_TEST_MODEL_DIR"] ?? "models/Qwen3.5-2B-MLX-4bit"
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(
            atPath: url.appendingPathComponent("model.safetensors").path) else {
            throw XCTSkip("本地缺少 MLX 测试模型：\(path)")
        }
        return url
    }

    // MARK: - 解析

    private func isKVLine(_ line: String) -> Bool {
        line.contains("[mlx-kv] reuse=")
    }
    private func isTruncatedWarning(_ line: String) -> Bool {
        line.contains("工具调用未完成")
    }
    /// 内核主动失效缓存的事件（合法的全量重建触发）：
    /// L3 周期折叠 / 兜底收缩都会改写历史前缀，必须丢弃旧 KV。
    private func isInvalidationEvent(_ line: String) -> Bool {
        line.contains("[kernel] CONSOLIDATE") || line.contains("兜底收缩后重试")
    }
    /// 从 `[mlx-kv] reuse=… cached=N …` 提取 cached 值。
    private func cachedTokens(_ line: String) -> Int? {
        guard let range = line.range(of: "cached=") else { return nil }
        let digits = line[range.upperBound...].prefix { $0.isNumber }
        return Int(digits)
    }
    /// 从 `[mlx-kv] reuse=… prefill_ms=N …` 提取 prefill 毫秒。
    private func prefillMS(_ line: String) -> Int? {
        guard let range = line.range(of: "prefill_ms=") else { return nil }
        let digits = line[range.upperBound...].prefix { $0.isNumber }
        return Int(digits)
    }

    /// 校验调用序列：`reuse=false` 只允许出现在首次调用，以及紧跟
    /// 「截断工具调用→保护性 invalidate」或「周期折叠/兜底收缩→主动失效」
    /// 之后的重建；其余必须是 reuse=true 且 cached>0、prefill 毫秒级。
    /// 返回违规描述列表（空 = 通过）。
    private func violations(in lines: [String]) -> [String] {
        var problems: [String] = []
        var firstSeen = false
        var rebuiltByInvalidation = false
        for line in lines {
            if isTruncatedWarning(line) || isInvalidationEvent(line) {
                rebuiltByInvalidation = true
                continue
            }
            guard isKVLine(line) else { continue }
            let reuse = line.contains("reuse=true")
            if !firstSeen {
                // 首次调用：必须全量构建。
                if reuse {
                    problems.append("首次调用不应有缓存可复用：\(line)")
                }
                firstSeen = true
                rebuiltByInvalidation = false
                continue
            }
            if reuse {
                let cached = cachedTokens(line) ?? 0
                if cached <= 0 {
                    problems.append("reuse=true 但 cached=0，复用形同虚设：\(line)")
                }
                if let ms = prefillMS(line), ms > 5_000 {
                    problems.append("reuse=true 但 prefill 耗时 \(ms)ms，近似全量重扫：\(line)")
                }
                rebuiltByInvalidation = false
            } else if !rebuiltByInvalidation {
                problems.append("正常轮次发生全量重建（每轮卡顿的直接原因）：\(line)")
            }
        }
        return problems
    }

    // MARK: - 测试

    /// 三轮真实交互：第 1 轮全量构建；第 2、3 轮必须命中跨轮 KV，
    /// 全程除首次构建外不允许任何 rebuild。
    func testMultiTurnAgentRunsReuseKVCache() async throws {
        let modelDir = try modelDirectory()

        let service = MLXLLMService(
            directory: modelDir,
            modelName: modelDir.lastPathComponent,
            maxTokens: 128,
            topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false,
            contextWindow: 32_768)
        defer { service.close() }

        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory()
            + "/kv-flow-\(UUID().uuidString)"
        config.maxIterations = 1

        let collector = LogCollector()
        // Agent.run 会把底层 AgentLog 按任务注入到 CollectingCallbacks.onLog；
        // 不再改进程级全局 sink，避免并发测试互相串日志 / 互相抹掉 sink。

        let agent = Agent(llm: service, config: config, callbacks: CollectingCallbacks(collector))
        defer { agent.close() }

        // 不用「介绍你自己」：它会命中能力问询快速通道（toolsEnabled=false），
        // 与预热/跨轮的全量工具前缀不一致，测的就不是 KV 复用了。
        _ = await agent.run("用一句话说明什么是向量检索，不要调用任何工具。")
        _ = await agent.run("接着回答：1+1 等于几？只回答数字。")
        _ = await agent.run("再回答：2+2 等于几？只回答数字。")

        let kvLines = collector.lines.filter { isKVLine($0) || isTruncatedWarning($0) || isInvalidationEvent($0) }
        XCTAssertGreaterThanOrEqual(
            kvLines.count, 3, "应至少有 3 次 LLM 调用，实际：\(kvLines)")

        let problems = violations(in: kvLines)
        XCTAssertTrue(problems.isEmpty,
            "KV 复用异常：\n\(problems.joined(separator: "\n"))\n--- 完整事件序列 ---\n"
            + collector.lines.suffix(40).joined(separator: "\n"))
    }

    /// 工具循环 / 衔接轮：同一次 run 内追加 assistant(tool_calls)+tool 消息
    /// 再次调用 LLM，是 delta 复用路径最容易被破坏（unexpected-assistant /
    /// no-common-prefix）的场景，同样必须走增量追加。
    func testToolLoopTurnsReuseKVCache() async throws {
        let modelDir = try modelDirectory()

        let service = MLXLLMService(
            directory: modelDir,
            modelName: modelDir.lastPathComponent,
            maxTokens: 256,
            topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false,
            contextWindow: 32_768)
        defer { service.close() }

        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory()
            + "/kv-flow-\(UUID().uuidString)"
        config.maxIterations = 3

        let collector = LogCollector()
        // Agent.run 会把底层 AgentLog 按任务注入到 CollectingCallbacks.onLog；
        // 不再改进程级全局 sink，避免并发测试互相串日志 / 互相抹掉 sink。

        let agent = Agent(llm: service, config: config, callbacks: CollectingCallbacks(collector))
        defer { agent.close() }

        _ = await agent.run("请记住数字 7，然后直接告诉我你记住了什么，不要调用任何工具。")
        _ = await agent.run("刚才让你记住的数字是多少？只回答数字，不要调用工具。")

        let kvLines = collector.lines.filter { isKVLine($0) || isTruncatedWarning($0) || isInvalidationEvent($0) }
        XCTAssertGreaterThanOrEqual(
            kvLines.count, 2, "应至少有 2 次 LLM 调用，实际：\(kvLines)")

        let problems = violations(in: kvLines)
        XCTAssertTrue(problems.isEmpty,
            "KV 复用异常：\n\(problems.joined(separator: "\n"))\n--- 完整事件序列 ---\n"
            + collector.lines.suffix(40).joined(separator: "\n"))
    }
}
