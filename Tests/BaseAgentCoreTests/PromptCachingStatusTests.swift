import XCTest
@testable import BaseAgentCore

/// 按当前默认配置（`llm_kv_reuse=true`、`flash_attention` 默认开启 →
/// KV 不量化）端到端验证 Prefix/Prompt Caching：
/// warmup 快照命中 → 多轮增量复用 → 失效后重建仍命中快照。
final class PromptCachingStatusTests: XCTestCase {
    private final class LogCollector: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var lines: [String] = []
        func append(_ line: String) { lock.lock(); lines.append(line); lock.unlock() }
    }

    private func modelDirectory() throws -> URL {
        let url = URL(fileURLWithPath: "models/Qwen3.5-2B-MLX-4bit")
        guard FileManager.default.fileExists(
            atPath: url.appendingPathComponent("model.safetensors").path) else {
            throw XCTSkip("本地缺少 MLX 测试模型：\(url.path)")
        }
        return url
    }

    private let systemPrompt = String(repeating:
        "你是严谨的中文知识助手，回答时引用材料并标注来源。", count: 60)

    private func metric(_ line: String, _ key: String) -> Int? {
        guard let range = line.range(of: key) else { return nil }
        return Int(line[range.upperBound...].prefix { $0.isNumber })
    }

    func testDefaultConfigPrefixCachingEndToEnd() async throws {
        let modelDir = try modelDirectory()
        let config = AgentConfig()
        // 锁定当前默认：flash 开启 → KV 不量化 → MLX 走标准 KV 缓存。
        XCTAssertTrue(config.flashAttention)
        XCTAssertTrue(config.llmKVReuse)
        XCTAssertFalse(config.effectiveMainKVInt8)

        let service = MLXLLMService(
            directory: modelDir,
            modelName: "prompt-caching-status",
            maxTokens: 32,
            enableThinking: false,
            contextWindow: config.effectiveMainContextTokens,
            promptBatchSize: config.promptBatchSize,
            kvReuse: config.llmKVReuse,
            kvInt8: config.effectiveMainKVInt8)
        defer { service.close() }

        try await service.warmup(systemPrompt: systemPrompt, tools: [])

        var transcript: [ChatMessage] = [ChatMessage(role: .system, content: systemPrompt)]
        for turn in 1...3 {
            transcript.append(ChatMessage(role: .user,
                content: "第 \(turn) 轮：请用一句话说明向量检索的用途。"))
            let collector = LogCollector()
            var reply = ""
            try await AgentLog.withTaskSink({ collector.append($0) }) {
                for try await event in service.chatStream(
                    messages: transcript, tools: [], temperature: 0.1) {
                    if event.kind == .content { reply += event.content }
                }
            }
            let kvLine = collector.lines.first { $0.contains("[mlx-kv] reuse=") } ?? ""
            let cached = metric(kvLine, "cached=") ?? 0
            let delta = metric(kvLine, "delta=") ?? 0
            let prefill = metric(kvLine, "prefill_ms=") ?? 0
            print("[prompt-cache] turn\(turn): \(kvLine)")
            XCTAssertTrue(kvLine.contains("reuse=true"),
                "第 \(turn) 轮应命中 KV 复用：\(collector.lines)")
            XCTAssertGreaterThan(cached, 0, "第 \(turn) 轮应复用已有缓存")
            XCTAssertLessThan(delta, 800,
                "第 \(turn) 轮只应预填增量：\(kvLine)")
            if turn > 1 {
                XCTAssertLessThan(prefill, 3_000,
                    "增量轮 prefill 应远低于全量重建：\(kvLine)")
            }
            if !reply.isEmpty {
                transcript.append(ChatMessage(role: .assistant, content: reply))
            }
        }

        // 失效后重建（折叠/截断/取消后的路径）必须命中 warmup 快照。
        service.invalidateKVCache()
        let collector = LogCollector()
        try await AgentLog.withTaskSink({ collector.append($0) }) {
            for try await _ in service.chatStream(
                messages: transcript, tools: [], temperature: 0.1) {}
        }
        let snap = collector.lines.first { $0.contains("prefix-reuse") }
        print("[prompt-cache] after-invalidate: "
            + (snap ?? collector.lines.joined(separator: " | ")))
        XCTAssertNotNil(snap, "失效后重建应命中 warmup 前缀快照")
    }
}
