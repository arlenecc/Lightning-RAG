import XCTest
@testable import BaseAgentCore

/// KV cache 按参考实现（preload 预热 + token 级前缀匹配三形态）的全流程效果验证：
///
/// 1. warmup 预热：模型加载后把 system+tools 前缀写入缓存，首轮对话只需
///    预填用户消息部分（对应 AppModel 启动后的真实调用顺序）。
/// 2. token 级复用（可裁剪缓存）：多轮追加只预填增量；中途分叉（历史被
///    改写）裁到分叉点复用；新 prompt 更短（回退重试）裁到公共前缀。
/// 每个用例输出各轮 prefill 耗时与 cached/delta token，用于人工核对效果。
final class KVCacheEffectivenessTests: XCTestCase {

    // MARK: - 日志捕获

    private final class LogCollector: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var lines: [String] = []
        func append(_ line: String) {
            lock.lock(); lines.append(line); lock.unlock()
        }
    }

    private final class CollectingCallbacks: AgentCallbacks {
        let collector: LogCollector
        init(_ collector: LogCollector) { self.collector = collector }
        func onLog(_ line: String) { collector.append(line) }
    }

    private func installSink(_ collector: LogCollector) -> () -> Void {
        let previous = AgentLog.sink
        AgentLog.sink = { collector.append($0) }
        return { AgentLog.sink = previous }
    }

    /// 从 `[mlx-kv] reuse=… cached=N …` 提取 cached 值。
    private func cachedTokens(_ line: String) -> Int? {
        guard let range = line.range(of: "cached=") else { return nil }
        return Int(line[range.upperBound...].prefix { $0.isNumber })
    }
    private func prefillMS(_ line: String) -> Int? {
        guard let range = line.range(of: "prefill_ms=") else { return nil }
        return Int(line[range.upperBound...].prefix { $0.isNumber })
    }
    private func kvLines(_ collector: LogCollector) -> [String] {
        collector.lines.filter { $0.contains("[mlx-kv]") }
    }
    /// 汇总每个调用轮的复用效果，作为测试输出便于人工核对。
    private func printEffectSummary(_ collector: LogCollector, title: String) {
        print("===== \(title) =====")
        for line in kvLines(collector) { print(line) }
    }

    // MARK: - 模型门控

    private func modelDirectory(env: String, fallbacks: [String]) throws -> URL {
        let path = ProcessInfo.processInfo.environment[env] ?? ""
        var candidates = path.isEmpty ? [] : [path]
        candidates.append(contentsOf: fallbacks)
        for candidate in candidates where FileManager.default.fileExists(
            atPath: candidate + "/config.json") {
            return URL(fileURLWithPath: candidate)
        }
        throw XCTSkip("本地缺少测试模型（已检查：\(candidates.joined(separator: ", "))）")
    }

    // MARK: - 1. warmup 预热 + 多轮复用（Qwen3.5，Mamba 混合缓存 → 消息级路径）

    /// 模拟 App 真实顺序：加载 → warmup(system+tools) → 多轮对话。
    /// 断言首轮即命中 warmup 快照（cached≈system+tools 的 ~5k token），
    /// 后续轮次复用跨轮缓存；汇总各轮 prefill 毫秒数验证首句优化生效。
    func testWarmupSeedsFirstTurnAndSubsequentTurnsReuse() async throws {
        // 需要可裁剪缓存：Qwen3.5（Mamba 混合）模板还要求 user 消息，预热会
        // 明确跳过；标准 KV 模型（Qwen2.5 系）走「占位差分 system 段」预热。
        let modelDir = try modelDirectory(
            env: "MLX_KV_TRIM_TEST_MODEL_DIR",
            fallbacks: ["/Volumes/SSD/models/lmstudio-community/"
                + "Qwen2.5-Coder-14B-Instruct-MLX-4bit"])

        let service = MLXLLMService(
            directory: modelDir,
            modelName: modelDir.lastPathComponent,
            maxTokens: 128,
            topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false,
            contextWindow: 32_768)
        defer { service.close() }

        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory() + "/kv-eff-\(UUID().uuidString)"
        config.maxIterations = 1

        let collector = LogCollector()
        let restore = installSink(collector)
        defer { restore() }

        let agent = Agent(llm: service, config: config,
                          callbacks: CollectingCallbacks(collector))
        defer { agent.close() }

        // 与 AppModel 相同的预热调用：system + 全量工具 schema 进缓存。
        try await service.warmup(systemPrompt: agent.systemPrompt(),
                                 tools: agent.dispatcher.schemas())

        // 不用「介绍你自己」：它会命中能力问询快速通道（toolsEnabled=false），
        // 与 warmup 的全量工具前缀不一致，测的就不是 KV 复用了。
        _ = await agent.run("用一句话说明什么是 KV 缓存，不要调用任何工具。")
        _ = await agent.run("接着回答：1+1 等于几？只回答数字。")
        _ = await agent.run("再回答：2+2 等于几？只回答数字。")

        printEffectSummary(collector, title: "warmup + 多轮（Qwen3.5 消息级）")

        let kv = kvLines(collector)
        let calls = kv.filter { $0.contains("[mlx-kv] reuse=") }
        XCTAssertGreaterThanOrEqual(calls.count, 3, "应有 ≥3 次 LLM 调用：\(kv)")

        // 首轮：命中 warmup 快照（形态①的预热版），cached 接近完整 system+tools。
        let first = try XCTUnwrap(calls.first)
        XCTAssertTrue(first.contains("reuse=true"),
            "warmup 预热后首轮就应复用快照，实际：\(first)")
        let firstCached = cachedTokens(first) ?? 0
        XCTAssertGreaterThan(firstCached, 3_000,
            "首轮 cached 应≈system+tools 前缀（>3k token），实际：\(first)")
        if let ms = prefillMS(first) {
            XCTAssertLessThan(ms, 5_000,
                "首轮只应预填用户消息部分（快照命中），实际 prefill \(ms)ms")
        }
        let rebuildLines = kv.filter { $0.contains("[mlx-kv] rebuild reason=") }
        XCTAssertTrue(rebuildLines.isEmpty,
            "warmup 命中时首轮也不应有 rebuild 日志：\(rebuildLines)")

        // 后续轮次：跨轮缓存复用。
        for line in calls.dropFirst() {
            XCTAssertTrue(line.contains("reuse=true"), "后续轮未命中缓存：\(line)")
            XCTAssertGreaterThan(cachedTokens(line) ?? 0, 0, "cached 应 >0：\(line)")
        }
    }

    // MARK: - 2. token 级前缀匹配三形态（标准 KV 模型，缓存可裁剪）

    /// 用可裁剪缓存的真实模型验证形态①②③：
    /// - 调 2（多轮追加）：缓存是新 prompt 的严格前缀 → 只预填增量；
    /// - 调 3（中途分叉）：历史首条 user 被改写 → 裁到分叉点，system 段仍复用；
    /// - 调 4（回退收缩）：新 prompt 更短 → 裁到公共前缀复用。
    func testTokenLevelPrefixMatchOnTrimmableCache() async throws {
        let modelDir = try modelDirectory(
            env: "MLX_KV_TRIM_TEST_MODEL_DIR",
            fallbacks: ["/Volumes/SSD/models/lmstudio-community/"
                + "Qwen2.5-Coder-14B-Instruct-MLX-4bit"])

        let service = MLXLLMService(
            directory: modelDir,
            modelName: modelDir.lastPathComponent,
            maxTokens: 64,
            topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false,
            contextWindow: 32_768)
        defer { service.close() }

        let collector = LogCollector()
        let restore = installSink(collector)
        defer { restore() }

        let system = "你是测试助手，回答必须极简。"
        // 与 App 真实顺序一致：先预热 system 段，再开始对话。
        try await service.warmup(systemPrompt: system, tools: [])
        func systemTokens() -> [ChatMessage] {
            [ChatMessage(role: .system, content: system)]
        }

        func turn(_ messages: [ChatMessage]) async throws -> String {
            var output = ""
            for try await event in service.chatStream(
                messages: messages, tools: [], temperature: 0.1) {
                if event.kind == .content { output += event.content }
                if event.kind == .done { break }
            }
            return output.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // 调 1：全量构建（无缓存）。
        let u1 = ChatMessage(role: .user, content: "北京有什么好吃的？一句话。")
        let a1 = try await turn(systemTokens() + [u1])
        XCTAssertFalse(a1.isEmpty, "首轮应正常回复")

        // 调 2（形态①）：多轮正常追加，缓存是严格前缀 → 只预填新 user。
        let a2 = try await turn(systemTokens() + [
            u1,
            ChatMessage(role: .assistant, content: a1),
            ChatMessage(role: .user, content: "上海呢？一句话。"),
        ])
        XCTAssertFalse(a2.isEmpty)

        // 调 3（形态③）：首条 user 被改写（模拟历史改写/锚点变化），
        // 分叉点在 user 消息处 → 应裁剪到分叉点，system 段仍复用。
        let a3 = try await turn(systemTokens() + [
            ChatMessage(role: .user, content: "介绍上海（改写后的历史）"),
            ChatMessage(role: .assistant, content: "上海。"),
            ChatMessage(role: .user, content: "广州有什么好吃的？一句话。"),
        ])
        XCTAssertFalse(a3.isEmpty)

        // 调 4（形态②）：新 prompt 比缓存短（回退/收缩重试场景）→ 裁到公共前缀。
        let a4 = try await turn(systemTokens() + [
            ChatMessage(role: .user, content: "深圳呢？一句话。"),
        ])
        XCTAssertFalse(a4.isEmpty)

        printEffectSummary(collector, title: "token 级三形态（\(modelDir.lastPathComponent)）")

        // 该用例 system 极短（<32 token），warmup 保护性跳过 → 首轮冷启动
        // 合法；调 2/3/4 全部 token 级复用（形态①②③各一次）。
        let reuse = kvLines(collector).filter { $0.contains("[mlx-kv] token-reuse") }
        XCTAssertEqual(reuse.count, 3,
            "调 2/3/4 都应走 token 级复用，实际：\(kvLines(collector))")

        // 三次复用的 cached 都应 ≥ system 段（分叉/回退场景下 system 仍复用）。
        for line in reuse {
            XCTAssertGreaterThanOrEqual(cachedTokens(line) ?? 0, 5,
                "复用点应至少覆盖 system 段：\(line)")
        }
        // 增量 prefill 应远小于全量（token-reuse 路径 prefill 毫秒级）。
        for line in reuse {
            if let ms = prefillMS(line) {
                XCTAssertLessThan(ms, 3_000, "增量预填不应接近全量耗时：\(line)")
            }
        }
        // 除首轮冷启动（未预热）外不应有任何全量重建。
        let rebuilds = kvLines(collector).filter { $0.contains("rebuild reason=") }
        XCTAssertEqual(rebuilds.count, 1, "只允许首轮冷启动一次 rebuild：\(rebuilds)")
    }
}
