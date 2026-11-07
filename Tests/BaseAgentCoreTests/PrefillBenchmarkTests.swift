import XCTest
@testable import BaseAgentCore

/// prefill 吞吐基准：对比 stepSize / kvInt8 配置对全量 prefill 耗时的影响，
/// 并验证 warmup 快照在 Qwen3.5（Mamba 混合缓存）上的可用性。
///
/// 数据看输出 `[prefill-bench]` 与日志 `[mlx-kv]` / `[mlx-warmup]`。
/// 机器基线：Apple M1 16GB；4B 模型的 7k token prefill 理论下限 ≈ 20s，
/// 2B 模型 ≈ 10s，因此优化目标是压挤出常数开销，而非数量级提升。
final class PrefillBenchmarkTests: XCTestCase {
    private func modelDirectory() throws -> URL {
        let url = URL(fileURLWithPath: "models/Qwen3.5-2B-MLX-4bit")
        guard FileManager.default.fileExists(
            atPath: url.appendingPathComponent("model.safetensors").path) else {
            throw XCTSkip("本地缺少 MLX 测试模型：\(url.path)")
        }
        return url
    }

    private final class LogCollector: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var lines: [String] = []
        func append(_ line: String) { lock.lock(); lines.append(line); lock.unlock() }
    }

    /// system ≈ 3.5k token + user ≈ 1.5k token，接近真实对话形态。
    private let systemPrompt = String(repeating:
        "你是严谨的中文知识助手。回答前先复述问题要点，引用材料时标注来源段落。", count: 60)
    private let userPrompt = String(repeating:
        "背景材料：向量检索把文本映射为高维语义向量，通过近似最近邻索引加速召回，"
            + "并用重排模型精排。请结合材料回答。", count: 60)

    func testPrefillConfigurations() async throws {
        let modelDir = try modelDirectory()
        let messages: [ChatMessage] = [
            ChatMessage(role: .system, content: systemPrompt),
            ChatMessage(role: .user, content: userPrompt + "\n\n用一句话概括上面材料的主题。"),
        ]

        for (label, batchSize, kvInt8) in [
            ("default-512-kvint8", 0, true),
            ("step4096-kvint8", 4_096, true),
            ("step4096-kvfp16", 4_096, false),
        ] {
            let service = MLXLLMService(
                directory: modelDir,
                modelName: "bench-\(label)",
                maxTokens: 8,
                enableThinking: false,
                contextWindow: 32_768,
                promptBatchSize: batchSize,
                kvReuse: false,
                kvInt8: kvInt8)
            defer { service.close() }

            // warmup：与请求使用同一 system 前缀，观测快照是否建立（[mlx-warmup]）。
            try await service.warmup(systemPrompt: systemPrompt, tools: [])

            let collector = LogCollector()
            let clock = Date()
            try await AgentLog.withTaskSink({ collector.append($0) }) {
                for try await _ in service.chatStream(
                    messages: messages, tools: [], temperature: 0.1) {}
            }
            let wall = Int(Date().timeIntervalSince(clock) * 1000)
            let kvLine = collector.lines.first { $0.contains("[mlx-kv] reuse=") } ?? "（无 KV 行）"
            let warmupLine = collector.lines.first { $0.contains("[mlx-warmup]") } ?? "（无 warmup 行）"
            print("[prefill-bench] \(label) wall=\(wall)ms | \(kvLine)")
            print("[prefill-bench] \(label) | \(warmupLine)")
        }
    }

    /// 失效后重建必须命中 warmup 快照（prefix-reuse）：模拟截断/取消/折叠后
    /// 的全量重建路径，断言只 prefill 动态历史而非整段 system。
    func testRebuildAfterInvalidationHitsSnapshot() async throws {
        let modelDir = try modelDirectory()
        let service = MLXLLMService(
            directory: modelDir,
            modelName: "snap-reuse",
            maxTokens: 8,
            enableThinking: false,
            contextWindow: 32_768,
            kvReuse: true,
            kvInt8: true)
        defer { service.close() }
        try await service.warmup(systemPrompt: systemPrompt, tools: [])
        let messages: [ChatMessage] = [
            ChatMessage(role: .system, content: systemPrompt),
            ChatMessage(role: .user, content: userPrompt + "\n\n一句话概括主题。"),
        ]

        func turn(_ collector: LogCollector) async throws {
            try await AgentLog.withTaskSink({ collector.append($0) }) {
                for try await _ in service.chatStream(
                    messages: messages, tools: [], temperature: 0.1) {}
            }
        }

        // 第一轮：从 warmup 快照接续，建立跨轮缓存。
        let first = LogCollector()
        try await turn(first)
        XCTAssertTrue(first.lines.contains { $0.contains("reuse=true") },
            "首轮应命中 warmup 前缀：\(first.lines)")

        // 强制失效（等价于截断保护/取消后的缓存丢失）。
        service.invalidateKVCache()

        // 重建：必须命中快照 prefix-reuse，prefill 只剩动态历史。
        let second = LogCollector()
        try await turn(second)
        let reuseLine = second.lines.first { $0.contains("[mlx-kv] reuse=") } ?? "（无）"
        let snapLine = second.lines.first { $0.contains("prefix-reuse") } ?? "（无）"
        print("[prefill-bench] rebuild-after-invalidate: \(reuseLine)")
        print("[prefill-bench] rebuild-after-invalidate: \(snapLine)")
        XCTAssertTrue(snapLine.contains("prefix-reuse"),
            "失效后重建应命中 warmup 快照：\(second.lines.joined(separator: " | "))")
        if let ms = prefillMS(reuseLine) {
            // 全量 prefill（system+user ≈3.3k token）约 8s；快照命中只 prefill
            // user 段（≈2k token），留 1.5 倍余量。
            XCTAssertLessThan(ms, 6_500,
                "快照命中后重建 prefill 应明显低于全量：\(reuseLine)")
        }
    }

    private func prefillMS(_ line: String) -> Int? {
        guard let range = line.range(of: "prefill_ms=") else { return nil }
        let digits = line[range.upperBound...].prefix { $0.isNumber }
        return Int(digits)
    }
}
