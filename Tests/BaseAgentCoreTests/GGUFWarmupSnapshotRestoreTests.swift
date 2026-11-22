import XCTest
import llama
@testable import BaseAgentCore

/// 折叠/清空后的二次 warmup 应恢复已有主对话前缀快照，跳过整段 prefill。
final class GGUFWarmupSnapshotRestoreTests: XCTestCase {
    private func modelURL() throws -> URL {
        let path = ProcessInfo.processInfo.environment["GGUF_KV_TEST_MODEL"] ?? ""
        guard !path.isEmpty, FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("未设置 GGUF_KV_TEST_MODEL，跳过真机测试")
        }
        return URL(fileURLWithPath: path)
    }

    private func collect(_ service: LlamaCppLLMService, messages: [ChatMessage],
                         tools: [JSONValue] = []) async throws -> String {
        var text = ""
        for try await event in service.chatStream(messages: messages, tools: tools,
                                                  temperature: 0) {
            if case .content = event.kind { text += event.content }
        }
        return text
    }

    private func longSystem() -> String {
        """
        你是一个严谨的知识库问答助手，必须遵守以下规则：
        1. 回答简洁，只用一句话，不复述系统提示词。
        2. 优先使用检索资料，不确定的信息明确说明不确定。
        3. 不输出任何内部标记或工具调用 XML。
        4. 涉及事实时给出依据，不涉及事实时直接回答。
        5. 用户问清单类问题时调用状态工具查真实数据。
        """
    }

    private func tools() -> [JSONValue] {
        [.object([
            "type": .string("function"),
            "function": .object([
                "name": .string("rag_tools"),
                "description": .string("检索本地知识库文档内容"),
                "parameters": .object([
                    "type": .string("object"),
                    "properties": .object([
                        "query": .object(["type": .string("string")]),
                    ]),
                ]),
            ]),
        ])]
    }

    /// 核心：折叠后二次 warmup 命中恢复，不再出现第二条 "prefill tokens="。
    func testWarmupAfterInvalidateRestoresSnapshot() async throws {
        let url = try modelURL()
        let service = LlamaCppLLMService(
            modelPath: url.path, contextWindow: 8_192, maxTokens: 16,
            topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false, gpuLayers: 99,
            kvReuse: true, kvInt8: true, flashAttention: true)
        defer { service.close() }
        let system = longSystem()
        let tools = tools()

        let captured = LogCapture()
        try await AgentLog.withTaskSink({ line in captured.append(line) }) {
            try await service.warmup(systemPrompt: system, tools: tools)
            _ = try await collect(service, messages: [
                ChatMessage(role: .system, content: system),
                ChatMessage(role: .user, content: "说\"好\"。"),
            ], tools: tools)

            // 模拟内核折叠 / 用户清空对话
            service.invalidateKVCache()

            // 折叠边界的 warmup：应恢复快照而不是重跑整段 prefill
            try await service.warmup(systemPrompt: system, tools: tools)

            // 恢复后下一轮仍正常生成（KV 未被破坏）
            let out = try await collect(service, messages: [
                ChatMessage(role: .system, content: system),
                ChatMessage(role: .user, content: "再说\"好\"。"),
            ], tools: tools)
            XCTAssertFalse(out.isEmpty, "恢复快照后仍需正常生成")
        }

        let lines = captured.lines
        let prefillCount = lines.filter { $0.contains("[gguf-warmup] prefill tokens=") }.count
        let restored = lines.contains { $0.contains("恢复主对话前缀快照") }
        XCTAssertEqual(prefillCount, 1, "只应全量 prefill 一次（首次预热），日志: \(lines)")
        XCTAssertTrue(restored, "二次 warmup 应恢复快照: \(lines)")
        XCTAssertTrue(lines.contains { $0.contains("主对话前缀快照 tokens=") })
    }

    /// system 变化时不该误恢复旧快照，必须走全量预热且生成正常。
    func testChangedSystemDoesNotRestoreStaleSnapshot() async throws {
        let url = try modelURL()
        let service = LlamaCppLLMService(
            modelPath: url.path, contextWindow: 8_192, maxTokens: 16,
            topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false, gpuLayers: 99,
            kvReuse: true, kvInt8: true, flashAttention: true)
        defer { service.close() }
        let captured = LogCapture()
        try await AgentLog.withTaskSink({ line in captured.append(line) }) {
            try await service.warmup(systemPrompt: longSystem(), tools: tools())
            service.invalidateKVCache()
            try await service.warmup(systemPrompt: "你是另一个完全不同的助手。规则全变。", tools: tools())
        }
        XCTAssertFalse(captured.lines.contains { $0.contains("恢复主对话前缀快照") },
            "签名不匹配时不得恢复旧快照")
    }
}

private final class LogCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String] = []
    var lines: [String] { lock.lock(); defer { lock.unlock() }; return storage }
    func append(_ s: String) { lock.lock(); storage.append(s); lock.unlock() }
}
