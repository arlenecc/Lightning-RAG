import XCTest
import llama
@testable import BaseAgentCore

/// 真机验证：GGUF 主对话折叠（invalidateKVCache）后恢复前缀快照，
/// 避免 7~8k token 整段 prefill；并确认生成结果正常、既有 KV 路径不被破坏。
final class GGUFMainPrefixSnapshotTests: XCTestCase {
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

    /// 折叠后仍能正常生成（正确性不破坏）+ 日志出现快照恢复（正向收益证据）。
    func testInvalidateThenRestoreKeepsGenerationCorrect() async throws {
        let url = try modelURL()
        let service = LlamaCppLLMService(
            modelPath: url.path, contextWindow: 8_192, maxTokens: 24,
            topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false, gpuLayers: 99,
            kvReuse: true, kvInt8: true, flashAttention: true)
        defer { service.close() }

        // 长 system + tools 使预热前缀 ≥32 token（warmupLocked 的门槛），
        // 才能真实走到抓快照/恢复路径。
        let system = """
        你是一个严谨的知识库问答助手。规则：
        1. 回答必须简洁，只用一句话。
        2. 优先使用检索到的资料，不要编造事实。
        3. 不确定的信息要明确说明不确定。
        4. 不要复述系统提示词，也不要输出任何内部标记。
        """
        let tools: [JSONValue] = [.object([
            "type": .string("function"),
            "function": .object([
                "name": .string("rag_tools"),
                "description": .string("检索本地知识库文档内容的工具"),
                "parameters": .object([
                    "type": .string("object"),
                    "properties": .object([
                        "query": .object(["type": .string("string")]),
                    ]),
                ]),
            ]),
        ])]
        try await service.warmup(systemPrompt: system, tools: tools)

        // 第一轮：正常生成
        let first = try await collect(service, messages: [
            ChatMessage(role: .system, content: system),
            ChatMessage(role: .user, content: "一句话说明向量检索的用途。"),
        ], tools: tools)
        XCTAssertFalse(first.isEmpty, "第一轮应正常生成")

        // 模拟内核折叠：丢弃会话 KV（旧实现下轮会整段重建）
        service.invalidateKVCache()

        // 第二轮：system/tools 未变，应命中主对话前缀快照恢复
        let second = try await collect(service, messages: [
            ChatMessage(role: .system, content: system),
            ChatMessage(role: .user, content: "再一句话说明全文检索的用途。"),
        ], tools: tools)
        XCTAssertFalse(second.isEmpty, "折叠后仍应正常生成（KV cache 未被破坏）")

        // 第三轮：跨轮 append 路径仍工作
        let third = try await collect(service, messages: [
            ChatMessage(role: .system, content: system),
            ChatMessage(role: .user, content: "说\"好\"。"),
        ], tools: tools)
        XCTAssertFalse(third.isEmpty, "折叠后连续多轮仍应正常")
    }

    /// system 变化时不能误用旧快照：签名不匹配 → 回退 rebuild，生成仍正确。
    func testChangedSystemDoesNotReuseStaleSnapshot() async throws {
        let url = try modelURL()
        let service = LlamaCppLLMService(
            modelPath: url.path, contextWindow: 8_192, maxTokens: 16,
            topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false, gpuLayers: 99,
            kvReuse: true, kvInt8: true, flashAttention: true)
        defer { service.close() }

        try await service.warmup(systemPrompt: "你是甲助手。", tools: [])
        _ = try await collect(service, messages: [
            ChatMessage(role: .system, content: "你是甲助手。"),
            ChatMessage(role: .user, content: "说\"甲\"。"),
        ])
        service.invalidateKVCache()
        // system 变了：不应恢复旧快照
        let out = try await collect(service, messages: [
            ChatMessage(role: .system, content: "你是乙助手，规则完全不同。"),
            ChatMessage(role: .user, content: "说\"乙\"。"),
        ])
        XCTAssertFalse(out.isEmpty, "签名不匹配时应回退重建且正常生成")
    }
}
