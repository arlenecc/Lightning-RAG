import XCTest
import Foundation
@testable import BaseAgentCore

/// GGUF（llama.cpp）侧模拟正式用户的启动与对话序列，与
/// `RealUserKVFlowTests`（MLX）对齐：启动预热期间立刻发第一句，之后正式轮
/// 夹杂寒暄/能力问询，检查每轮是否命中 KV、折叠边界是否重复全量预热。
/// 需要 `GGUF_KV_TEST_MODEL` 指向真实 .gguf；未设置时整组跳过。
final class GGUFRealUserFlowTests: XCTestCase {

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

    private func modelURL() throws -> URL {
        let path = ProcessInfo.processInfo.environment["GGUF_KV_TEST_MODEL"] ?? ""
        guard !path.isEmpty, FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("未设置 GGUF_KV_TEST_MODEL，跳过 GGUF 正式用户流程测试")
        }
        return URL(fileURLWithPath: path)
    }

    func testWarmupServesFirstTurnAndEverySubsequentTurn() async throws {
        let url = try modelURL()
        let service = LlamaCppLLMService(
            modelPath: url.path, contextWindow: 32_768, maxTokens: 64,
            topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false, gpuLayers: 99)
        defer { service.close() }

        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory() + "/gguf-real-\(UUID().uuidString)"
        config.maxIterations = 1
        let collector = LogCollector()
        let agent = Agent(llm: service, config: config,
                          callbacks: Callbacks(collector))
        defer { agent.close() }

        await agent.connectMCP()
        let systemPrompt = agent.systemPrompt()
        let tools = agent.dispatcher.schemas()

        let warmupTask = Task {
            try await service.warmup(systemPrompt: systemPrompt, tools: tools)
        }
        let firstStart = Date()
        _ = await agent.run("你好")
        let firstMS = Int(Date().timeIntervalSince(firstStart) * 1000)
        log("[gguf-real] 启动期首句寒暄 ms=\(firstMS)")
        let kvDuringGreeting = collector.lines.filter { $0.contains("[gguf-kv]") }
        XCTAssertTrue(kvDuringGreeting.isEmpty,
            "寒暄应本地直答，不应创建 LLM stream：\(kvDuringGreeting)")
        XCTAssertLessThan(firstMS, 3_000,
            "首句寒暄不能等预热：\(firstMS)ms")
        try await warmupTask.value

        let llmTurns: Set<Int> = [1, 3, 5, 7]
        var llmTurnMS: [Int] = []
        var localTurnMS: [(Int, Int)] = []
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
        for (index, prompt) in prompts.enumerated() {
            let start = Date()
            _ = await agent.run(prompt)
            let ms = Int(Date().timeIntervalSince(start) * 1000)
            if llmTurns.contains(index) {
                llmTurnMS.append(ms)
            } else {
                localTurnMS.append((index + 1, ms))
            }
            log("[gguf-real] turn=\(index + 1) ms=\(ms) prompt=\(prompt)")
        }

        log("===== GGUF WARMUP / KV LOGS =====")
        for line in collector.lines where line.contains("[gguf-") { log(line) }

        let modes = collector.lines.filter { $0.contains("[gguf-kv] mode=") }
        XCTAssertGreaterThanOrEqual(modes.count, 4,
            "4 个正式轮应各有一次 LLM 调用：\(modes)")
        // 启动预热必须真的生效：首轮从预热前缀（约 7.2k token，随提示注入
        // 略有浮动）后增量追加，而不是全量重建。
        let appendFromWarmup = modes.filter {
            $0.contains("mode=append") && $0.contains("tools_match=true")
                && $0.contains("cached=0") == false
        }
        XCTAssertFalse(appendFromWarmup.isEmpty,
            "预热前缀必须被正式请求命中：\(modes)")
        XCTAssertTrue(modes.allSatisfy { $0.contains("tools_match=true") },
            "轻量轮不得换掉工具清单（tools-changed）：\(modes)")

        // GGUF 没有可复制的 KV 快照，折叠边界重跑一次前缀 prefill 是必要的
        // （让下一轮 mode=append 只算增量），但不应失控地反复重跑。
        let warmups = collector.lines.filter {
            $0.contains("[gguf-warmup] prefill tokens=")
        }
        XCTAssertLessThanOrEqual(warmups.count, 2,
            "折叠边界预热不应反复重跑：\(warmups)")

        // generationStart 计时修复的回归：tps 必须是真实的几十~一百多，
        // 旧实现把计时放在生成之后会打出 tps=64000 这类失真值。
        for line in collector.lines where line.contains("[gguf] prompt=") {
            guard let range = line.range(of: "tps=") else { continue }
            let tps = Int(line[range.upperBound...].prefix { $0.isNumber }) ?? 0
            XCTAssertGreaterThan(tps, 0, "生成速度日志异常：\(line)")
            XCTAssertLessThan(tps, 1_000, "tps 疑似计时失真：\(line)")
        }
        // 已知结构性缺口：hybrid/recurrent memory 的 llama_memory_seq_rm
        // 不支持 partial sequence 裁剪，Qwen3.5 GGUF 每轮都退化成整段
        // prefill（7~8s），反思门重试时还会成倍放大。这里不断言墙钟，
        // 只把耗时打进日志；真正的回归保护是上面的预热命中 / 工具前缀 /
        // tps / 轻量轮零预热断言，以及「除首轮外不得出现 tools_match=false」。
        XCTAssertFalse(llmTurnMS.isEmpty, "应记录到正式轮耗时")
        // 轻量轮绝不能触发 CONSOLIDATE → invalidateKV → didFold → warmup：
        // GGUF 重跑一次 7k 前缀就是 7.5s，一句「你好」不该等这么久。
        for (turn, ms) in localTurnMS {
            XCTAssertLessThan(ms, 3_000,
                "本地直答轮 \(turn) 耗时 \(ms)ms，疑似触发折叠/重新预热")
        }
    }
}
