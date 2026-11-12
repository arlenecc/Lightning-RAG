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

    /// Flash Attention (ENABLED) 与 q8_0 KV 量化叠加：
    /// llama.cpp 必须能初始化并正常生成，不因组合报错/回退为不可用。
    func testFlashAttentionWithQuantizedKV() async throws {
        let modelURL = try modelURL()
        let service = LlamaCppLLMService(
            modelPath: modelURL.path,
            contextWindow: 8_192, maxTokens: 32,
            topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false, gpuLayers: 99,
            kvInt8: true, flashAttention: true)
        defer { service.close() }

        let output = try await collect(service, messages: [
            ChatMessage(role: .system, content: "你是测试助手，回答必须极简。"),
            ChatMessage(role: .user, content: "一句话说明向量检索的用途。"),
        ])
        XCTAssertFalse(output.isEmpty, "ENABLED + q8_0 KV 组合应能正常生成")
    }

    /// `kv_reuse` 开关关闭（辅助模型默认 `aux_llm_kv_reuse=false`）时，
    /// 隔离请求的稳定前缀复用仍应生效：它与主对话跨轮缓存是两套机制。
    func testIsolatedPrefixReuseIndependentOfKVReuseToggle() async throws {
        let modelURL = try modelURL()
        let service = LlamaCppLLMService(
            modelPath: modelURL.path,
            contextWindow: 8_192, maxTokens: 24,
            topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false, gpuLayers: 99,
            kvReuse: false)
        defer { service.close() }

        let collector = LogBox()
        let system = ChatMessage(role: .system, content: """
        你是知识图谱抽取助手。请从用户提供的文本中抽取实体与关系，\
        只输出 JSON 对象，不要输出解释、前后缀或 Markdown 代码围栏。\
        实体字段为 name、type、description；关系字段为 source、target、\
        keywords、description。实体数量不超过 20 个，关系不超过 50 个，\
        每条描述不超过 100 个字符。若文本中没有可抽取的内容，输出空数组。
        """)
        func isolated(_ user: String) async throws {
            try await AgentLog.withTaskSink({ collector.append($0) }) {
                let stream = service.chatStream(
                    messages: [system, ChatMessage(role: .user, content: user)],
                    tools: [], temperature: 0,
                    options: LLMRequestOptions(enableThinking: false, isolated: true))
                for try await event in stream {
                    if event.kind == .done { break }
                }
            }
        }

        try await isolated("切片一：Alice 在 Acme 公司工作，负责 Aurora 项目。")
        try await isolated("切片二：Bob 在 Beta 公司工作，负责 Delta 项目。")

        let restored = collector.lines.filter { $0.contains("恢复前缀快照") }
        XCTAssertEqual(restored.count, 1,
                       "kv_reuse=false 时隔离请求也应恢复前缀快照：\(collector.lines)")
    }

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
            enableThinking: false, gpuLayers: 99)
        defer { service.close() }

        let collector = LogBox()
        // 日志按任务绑定，不再改进程级全局 sink（并发测试会互相串/互相抹掉）。
        func logged(_ messages: [ChatMessage]) async throws {
            try await AgentLog.withTaskSink({ collector.append($0) }) {
                _ = try await collect(service, messages: messages)
            }
        }

        let system = ChatMessage(role: .system, content: "你是测试助手，回答必须极简。")
        try await logged([
            system, ChatMessage(role: .user, content: "北京有什么好吃的？一句话。")])
        // 形态①：正常追加。
        try await logged([
            system,
            ChatMessage(role: .user, content: "北京有什么好吃的？一句话。"),
            ChatMessage(role: .assistant, content: "烤鸭。"),
            ChatMessage(role: .user, content: "上海呢？一句话。")])
        // 形态②：新 prompt 更短（回退重试）。
        try await logged([
            system, ChatMessage(role: .user, content: "深圳呢？一句话。")])
        // 形态③：历史首条 user 被改写（中途分叉）。
        try await logged([
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
        // llama.h 明确：partial sequence 裁剪在 hybrid/recurrent memory 上
        // 会返回 false（「Removing a whole sequence never fails」）。混合 SSM
        // 模型（如 Qwen3.5 GGUF）改走消息级增量 append，不再整段重建。
        let seqRmFailures = collector.lines.filter {
            $0.contains("[gguf-kv] seq_rm") && $0.contains("失败")
        }
        if seqRmFailures.isEmpty {
            for mode in modes.dropFirst() {
                XCTAssertTrue(mode == "append" || mode == "trim",
                    "支持 seq_rm 的模型后续应复用，实际 \(modes)")
            }
        } else {
            XCTAssertFalse(collector.lines.contains { $0.contains("llama_decode 返回") },
                "裁剪降级后不得出现 decode 失败：\(collector.lines)")
            let deltas = collector.lines.filter {
                $0.contains("[gguf-kv] delta-append")
            }
            XCTAssertFalse(deltas.isEmpty,
                "裁剪失败后必须走消息级增量，避免每轮整段重建 7~8s："
                + "\(collector.lines.filter { $0.contains("[gguf-") })")
        }
    }

    /// 隔离请求（知识图谱抽取）：system 前缀整轮相同，首次预热抓快照，
    /// 之后每次恢复快照，只 prefill 切片正文。
    func testIsolatedRequestsReuseWarmPrefix() async throws {
        let modelURL = try modelURL()
        let service = LlamaCppLLMService(
            modelPath: modelURL.path,
            contextWindow: 8_192, maxTokens: 24,
            topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false, gpuLayers: 99)
        defer { service.close() }

        let collector = LogBox()
        let system = ChatMessage(role: .system, content: """
        你是知识图谱抽取助手。请从用户提供的文本中抽取实体与关系，\
        只输出 JSON 对象，不要输出解释、前后缀或 Markdown 代码围栏。\
        实体字段为 name、type、description；关系字段为 source、target、\
        keywords、description。实体数量不超过 20 个，关系不超过 50 个，\
        每条描述不超过 100 个字符。若文本中没有可抽取的内容，输出空数组。
        """)
        func isolated(_ user: String) async throws {
            try await AgentLog.withTaskSink({ collector.append($0) }) {
                let stream = service.chatStream(
                    messages: [system, ChatMessage(role: .user, content: user)],
                    tools: [], temperature: 0,
                    options: LLMRequestOptions(enableThinking: false, isolated: true))
                for try await event in stream {
                    if event.kind == .done { break }
                }
            }
        }

        try await isolated("切片一：Alice 在 Acme 公司工作，负责 Aurora 项目。")
        try await isolated("切片二：Bob 在 Beta 公司工作，负责 Delta 项目。")
        try await isolated("切片三：Carol 在 Gamma 公司工作，负责 Omega 项目。")

        let restored = collector.lines.filter { $0.contains("恢复前缀快照") }
        XCTAssertEqual(restored.count, 2,
                       "第 2、3 次隔离请求应恢复前缀快照：\(collector.lines)")
        let warmups = collector.lines.filter { $0.contains("[gguf-warmup] prefill") }
        XCTAssertEqual(warmups.count, 1,
                       "同一 system 前缀只应预热一次：\(collector.lines)")
    }

    private final class LogBox: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var lines: [String] = []
        func append(_ line: String) {
            lock.lock(); lines.append(line); lock.unlock()
        }
    }
}
