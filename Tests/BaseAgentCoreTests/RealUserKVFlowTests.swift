import XCTest
import Foundation
@testable import BaseAgentCore

/// 模拟正式用户的启动与对话序列：
/// - 启动预热（首次 Metal 编译 + 7k token prefill，实测 20~25s）期间立刻发第一句；
/// - 寒暄 / 能力问询本地直答，零 GPU、零延迟，也不该换掉 system/tools 前缀；
/// - 之后每个正式轮（中间继续夹杂寒暄）都必须命中跨轮 KV，不允许全量重建。
final class RealUserKVFlowTests: XCTestCase {

    private final class LogCollector: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var lines: [String] = []
        func append(_ line: String) {
            lock.lock(); lines.append(line); lock.unlock()
        }
    }

    private final class Callbacks: AgentCallbacks {
        let collector: LogCollector
        init(_ collector: LogCollector) { self.collector = collector }
        func onLog(_ line: String) { collector.append(line) }
    }

    private func log(_ text: String) {
        FileHandle.standardError.write(Data((text + "\n").utf8))
    }

    private func modelDirectory() throws -> URL {
        let path = ProcessInfo.processInfo.environment["MLX_KV_TEST_MODEL_DIR"]
            ?? "models/Qwen3.5-2B-MLX-4bit"
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(
            atPath: url.appendingPathComponent("model.safetensors").path) else {
            throw XCTSkip("本地缺少 MLX 测试模型：\(path)")
        }
        return url
    }

    private func cachedTokens(_ line: String) -> Int? {
        guard let range = line.range(of: "cached=") else { return nil }
        return Int(line[range.upperBound...].prefix { $0.isNumber })
    }

    private func prefillMS(_ line: String) -> Int? {
        guard let range = line.range(of: "prefill_ms=") else { return nil }
        return Int(line[range.upperBound...].prefix { $0.isNumber })
    }

    func testWarmupServesFirstTurnAndEverySubsequentTurn() async throws {
        let dir = try modelDirectory()
        let service = MLXLLMService(
            directory: dir, modelName: dir.lastPathComponent,
            maxTokens: 64, topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false, contextWindow: 32_768)
        defer { service.close() }

        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory() + "/kv-real-\(UUID().uuidString)"
        config.maxIterations = 1
        let collector = LogCollector()
        let agent = Agent(llm: service, config: config,
                          callbacks: Callbacks(collector))
        defer { agent.close() }

        // 与 AppModel.warmupMainModel 一致：MCP 连接完成后取 system + 全量工具。
        await agent.connectMCP()
        let systemPrompt = agent.systemPrompt()
        let tools = agent.dispatcher.schemas()

        // 启动即发送：预热在后台跑，用户第一句「你好」与它同时进行。
        let warmupTask = Task {
            try await service.warmup(systemPrompt: systemPrompt, tools: tools)
        }
        let firstStart = Date()
        _ = await agent.run("你好")
        let firstMS = Int(Date().timeIntervalSince(firstStart) * 1000)
        log("[real-user] 启动期首句寒暄 ms=\(firstMS)")
        let kvDuringGreeting = collector.lines.filter {
            $0.contains("[mlx-kv] reuse=")
        }
        XCTAssertTrue(kvDuringGreeting.isEmpty,
            "寒暄应本地直答，不应触发 LLM / 破坏预热缓存：\(kvDuringGreeting)")
        XCTAssertLessThan(firstMS, 3_000,
            "首句寒暄不能等 20s 的预热：\(firstMS)ms")

        let warmupStart = Date()
        try await warmupTask.value
        log("[real-user] warmup_wait_ms="
            + "\(Int(Date().timeIntervalSince(warmupStart) * 1000))")

        // 正式用户序列：中间继续夹杂能力问询与寒暄，验证轻量轮不再打断 KV 轨迹。
        let prompts = [
            "你能干什么",
            "用一句话说明什么是 KV 缓存，不要调用任何工具。",
            "谢谢",
            "接着回答：1+1 等于几？只回答数字。",
            "有什么可用工具",
            "再回答：2+2 等于几？只回答数字。",
            "你好",
            "最后回答：3+3 等于几？只回答数字。",
        ]
        let llmTurns: Set<Int> = [1, 3, 5, 7]   // 夹杂的寒暄/能力问询走本地直答
        var llmTurnMS: [Int] = []
        var localTurnMS: [(Int, Int)] = []
        for (index, prompt) in prompts.enumerated() {
            let start = Date()
            _ = await agent.run(prompt)
            let ms = Int(Date().timeIntervalSince(start) * 1000)
            if llmTurns.contains(index) {
                llmTurnMS.append(ms)
            } else {
                localTurnMS.append((index + 1, ms))
            }
            log("[real-user] turn=\(index + 1) ms=\(ms) prompt=\(prompt)")
        }

        let kv = collector.lines.filter { $0.contains("[mlx-kv]") }
        let warmupLines = collector.lines.filter { $0.contains("[mlx-warmup]") }
        log("===== WARMUP / KV LOGS =====")
        for line in kv { log(line) }
        for line in warmupLines { log(line) }

        let calls = kv.filter { $0.contains("[mlx-kv] reuse=") }
        XCTAssertGreaterThanOrEqual(calls.count, 4,
            "4 个正式轮应各有一次 LLM 调用，实际：\(kv)")
        XCTAssertTrue(calls.allSatisfy { $0.contains("reuse=true") },
            "每个正式轮都必须命中 KV 缓存：\(calls)")
        for line in calls {
            XCTAssertGreaterThan(cachedTokens(line) ?? 0, 3_000,
                "必须复用预热好的 system+tools 前缀：\(line)")
            if let ms = prefillMS(line) {
                XCTAssertLessThan(ms, 3_000, "增量 prefill 不应接近全量耗时：\(line)")
            }
        }
        let rebuilds = kv.filter { $0.contains("[mlx-kv] rebuild reason=") }
        XCTAssertTrue(rebuilds.isEmpty,
            "启动预热后不应出现任何全量重建：\(rebuilds)")
        let toolChanges = kv.filter { $0.contains("tools-changed") }
        XCTAssertTrue(toolChanges.isEmpty,
            "轻量轮不该再换工具清单：\(toolChanges)")

        // 折叠边界不能重跑全量预热：那会让该轮墙钟从 ~2s 突增到 ~24s，
        // 也就是用户感知的「时快时慢」。启动预热不经过 callbacks，因此
        // collector 里出现 prefill tokens= 就一定是折叠期的重复预热。
        let repeatedPrefill = warmupLines.filter {
            $0.contains("[mlx-warmup] prefill tokens=")
                || $0.contains("gpu_ready_ms=")
        }
        XCTAssertTrue(repeatedPrefill.isEmpty,
            "折叠后不得重新 prefill 整段前缀：\(repeatedPrefill)")
        let snapshotReuse = warmupLines.filter {
            $0.contains("[mlx-warmup] 复用已有前缀快照")
        }
        // 轻量轮跳过 CONSOLIDATE 后，短序列可能整段都不折叠（KV 全程
        // 只追加，这是最理想的情况）；一旦发生折叠，就必须复用快照而
        // 不是重跑全量 prefill。
        if !warmupLines.isEmpty {
            XCTAssertGreaterThanOrEqual(snapshotReuse.count, 1,
                "周期性折叠应复用前缀快照：\(warmupLines)")
        }
        // 墙钟只作兜底：全量测试并发跑时内核反思门可能多生成几轮（每轮
        // ~1.5s），绝对阈值不能卡得太紧；真正判断是否重新预热的是上面
        // 的结构性日志断言 + 每次调用的 prefill_ms < 3s。
        for (index, ms) in llmTurnMS.enumerated() {
            XCTAssertLessThan(ms, 20_000,
                "正式轮 \(index + 1) 墙钟 \(ms)ms，疑似重新预热 / 全量 prefill")
        }
        // 轻量轮不得触发周期性折叠（invalidateKV + didFold），否则刚累积的
        // 会话缓存会被一句「你好」白白丢掉。
        for (turn, ms) in localTurnMS {
            XCTAssertLessThan(ms, 3_000,
                "本地直答轮 \(turn) 耗时 \(ms)ms，疑似触发折叠/重新预热")
        }
    }
}
