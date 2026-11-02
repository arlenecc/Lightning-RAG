import XCTest
import llama
@testable import BaseAgentCore

/// GGUF（llama.cpp）侧跨轮 KV 复用验证。
///
/// 形态②③靠 `llama_memory_seq_rm` 裁剪 KV 到公共前缀；本机无 GGUF 模型时
/// 集成用例跳过，纯逻辑用例始终运行。
final class GGUFKVReuseTests: XCTestCase {

    // MARK: - 纯逻辑

    func testCommonPrefixForms() {
        // 形态①：缓存是严格前缀（含生成 token 的完整序列与渲染结果对齐）。
        let cached: [llama_token] = [10, 20, 30, 40]
        XCTAssertEqual(LlamaCppContext.commonPrefix(cached + [50], cached), 4)
        // 形态③：中途分叉 → 裁到分叉点。
        XCTAssertEqual(LlamaCppContext.commonPrefix([10, 20, 99], cached), 2)
        // 形态②：新 prompt 更短 → 裁到公共前缀。
        XCTAssertEqual(LlamaCppContext.commonPrefix([10], cached), 1)
        // 无公共前缀 / 空序列。
        XCTAssertEqual(LlamaCppContext.commonPrefix([9, 8], cached), 0)
        XCTAssertEqual(LlamaCppContext.commonPrefix([], cached), 0)
        XCTAssertEqual(LlamaCppContext.commonPrefix(cached, []), 0)
    }

    // MARK: - 集成（需真实 GGUF 模型）

    private func modelURL() throws -> URL {
        let path = ProcessInfo.processInfo.environment["GGUF_KV_TEST_MODEL"] ?? ""
        guard !path.isEmpty, FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("未设置 GGUF_KV_TEST_MODEL，跳过 GGUF KV 复用集成测试")
        }
        return URL(fileURLWithPath: path)
    }

    private func collect(_ service: LlamaCppLLMService,
                         messages: [ChatMessage]) async throws -> String {
        var output = ""
        for try await event in service.chatStream(messages: messages, tools: [],
                                                  temperature: 0.1) {
            if event.kind == .content { output += event.content }
            if event.kind == .done { break }
        }
        return output
    }

    /// 多轮 + 回退 + 分叉：断言除首次构建外全部复用（append/trim），日志里
    /// 不允许 rebuild。
    func testMultiTurnAndTrimReuse() async throws {
        let modelURL = try modelURL()
        let service = LlamaCppLLMService(
            modelPath: modelURL.path,
            contextWindow: 8_192, maxTokens: 48,
            topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false, gpuLayers: 0)
        defer { service.close() }

        let collector = LogBox()
        let previous = AgentLog.sink
        AgentLog.sink = { collector.append($0) }
        defer { AgentLog.sink = previous }

        let system = ChatMessage(role: .system, content: "你是测试助手，回答必须极简。")
        _ = try await collect(service, messages: [
            system, ChatMessage(role: .user, content: "北京有什么好吃的？一句话。")])
        // 形态①：正常追加。
        _ = try await collect(service, messages: [
            system,
            ChatMessage(role: .user, content: "北京有什么好吃的？一句话。"),
            ChatMessage(role: .assistant, content: "烤鸭。"),
            ChatMessage(role: .user, content: "上海呢？一句话。")])
        // 形态②：新 prompt 更短（回退重试）。
        _ = try await collect(service, messages: [
            system, ChatMessage(role: .user, content: "深圳呢？一句话。")])
        // 形态③：历史首条 user 被改写（中途分叉）。
        _ = try await collect(service, messages: [
            system,
            ChatMessage(role: .user, content: "介绍广州（改写后的历史）"),
            ChatMessage(role: .assistant, content: "广州。"),
            ChatMessage(role: .user, content: "成都呢？一句话。")])

        let modes = collector.lines
            .filter { $0.contains("[gguf-kv] mode=") }
            .compactMap { line -> String? in
                guard let range = line.range(of: "mode=") else { return nil }
                return String(line[range.upperBound...].prefix { $0.isLetter })
            }
        XCTAssertEqual(modes.count, 4, "应有 4 次 LLM 调用：\(collector.lines)")
        XCTAssertEqual(modes.first, "rebuild", "首轮全量构建：\(modes)")
        for mode in modes.dropFirst() {
            XCTAssertTrue(mode == "append" || mode == "trim",
                "后续调用应复用（append/trim），实际 \(modes)")
        }
        let rebuilds = collector.lines.filter {
            $0.contains("[gguf-kv] seq_rm") && $0.contains("失败")
        }
        XCTAssertTrue(rebuilds.isEmpty,
            "支持 seq_rm 的模型不应出现裁剪失败：\(rebuilds)")
    }

    private final class LogBox: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var lines: [String] = []
        func append(_ line: String) {
            lock.lock(); lines.append(line); lock.unlock()
        }
    }
}
