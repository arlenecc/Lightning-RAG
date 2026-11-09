import XCTest
import Foundation
import llama
@testable import BaseAgentCore

/// Spark-X2.6 GGUF（`/Volumes/SSD/gguf/Spark-X2.6-4B.i1-Q4_K_M.gguf`）真机回归。
///
/// GGUF 元数据里的 `tokenizer.chat_template` 与 MLX 版完全相同，含 HF 专有
/// `tojson(separators=(...))`（swift-jinja 不支持）与 GLM47 的
/// `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>` 方言。
/// 这里覆盖：元数据/路径解析 → 模板补丁 → 加载生成 → 工具调用 →
/// 混合 sliding/full 注意力的跨轮 KV → FA + q8_0 KV 组合。
/// 模型缺失时全部跳过；可用 `SPARK_X26_GGUF_MODEL` 覆盖路径。
final class SparkX26GGUFTests: XCTestCase {

    private func modelPath() throws -> String {
        let path = ProcessInfo.processInfo.environment["SPARK_X26_GGUF_MODEL"]
            ?? "/Volumes/SSD/gguf/Spark-X2.6-4B.i1-Q4_K_M.gguf"
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("本地缺少 Spark-X2.6 GGUF：\(path)")
        }
        return path
    }

    private func makeService(_ path: String,
                             maxTokens: Int = 64,
                             contextWindow: Int = 4096,
                             enableThinking: Bool = false,
                             kvInt8: Bool = true,
                             flashAttention: Bool = true) -> LlamaCppLLMService {
        LlamaCppLLMService(
            modelPath: path,
            contextWindow: contextWindow,
            maxTokens: maxTokens,
            topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: enableThinking,
            gpuLayers: 99,
            kvReuse: true,
            kvInt8: kvInt8,
            flashAttention: flashAttention)
    }

    private func toolSchema() -> BaseAgentCore.JSONValue {
        .object([
            "type": .string("function"),
            "function": .object([
                "name": .string("get_current_time"),
                "description": .string("Return the current local time for a timezone."),
                "parameters": .object([
                    "type": .string("object"),
                    "properties": .object([
                        "timezone": .object([
                            "type": .string("string"),
                            "description": .string("IANA timezone name"),
                        ]),
                    ]),
                    "required": .array([.string("timezone")]),
                ]),
            ]),
        ])
    }

    private final class LogBox: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var lines: [String] = []
        func append(_ line: String) {
            lock.lock(); lines.append(line); lock.unlock()
        }
    }

    private func collect(_ service: LlamaCppLLMService,
                         messages: [ChatMessage],
                         tools: [BaseAgentCore.JSONValue] = [])
        async throws -> (text: String, calls: [BaseAgentCore.ToolCall]) {
        var text = ""
        var calls: [BaseAgentCore.ToolCall] = []
        for try await event in service.chatStream(
            messages: messages, tools: tools, temperature: 0) {
            switch event.kind {
            case .content: text += event.content
            case .reasoning: break
            case .done: calls = event.toolCalls
            }
        }
        return (text, calls)
    }

    // MARK: - 路径 / 元数据 / 模板补丁

    func testExplicitGGUFPathResolves() throws {
        let path = try modelPath()
        let url = try GGUFFileResolver.resolve(
            kind: .llm, value: path, config: AgentConfig())
        XCTAssertEqual(url.standardizedFileURL.path,
                       URL(fileURLWithPath: path).standardizedFileURL.path)
    }

    func testMetadataAndTemplatePatch() throws {
        let path = try modelPath()
        let model = try LlamaCppModel(path: path, gpuLayers: 0)
        XCTAssertEqual(
            LlamaCppModel.metadataString(model.raw, key: "general.architecture"),
            "spark2_5")
        let rawTemplate = try XCTUnwrap(
            LlamaCppModel.metadataString(model.raw, key: "tokenizer.chat_template"))
        XCTAssertTrue(rawTemplate.contains("separators"),
                      "原始 GGUF 模板应含 HF 专有语法，补丁才有意义")
        XCTAssertTrue(rawTemplate.contains("<arg_key>"))

        let patched = try XCTUnwrap(model.chatTemplate)
        XCTAssertFalse(patched.contains("separators"),
                       "加载后模板必须已去掉 tojson 的 separators 参数")
        XCTAssertTrue(patched.contains("tojson"))
        XCTAssertTrue(patched.contains("<tool_call>"))
        XCTAssertGreaterThan(model.contextTrain, 0)
        XCTAssertGreaterThan(model.vocabSize, 0)
    }

    // MARK: - 生成 / 工具调用

    func testLoadsAndGeneratesWithoutTools() async throws {
        let path = try modelPath()
        let service = makeService(path)
        defer { service.close() }
        try await service.preload()
        let result = try await collect(service, messages: [
            ChatMessage(role: .system, content: "You are a helpful assistant."),
            ChatMessage(role: .user, content: "Reply with exactly: SPARK_GGUF_OK"),
        ])
        XCTAssertFalse(result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                       "模型必须产出可见正文")
        XCTAssertTrue(result.calls.isEmpty)
    }

    func testToolCallRendersAndParses() async throws {
        let path = try modelPath()
        let service = makeService(path, maxTokens: 192,
                                 enableThinking: false)
        defer { service.close() }
        try await service.preload()
        let result = try await collect(service, messages: [
            ChatMessage(role: .system,
                        content: "You are a helpful assistant. Always use the provided tool."),
            ChatMessage(role: .user,
                        content: "Call get_current_time with timezone Asia/Shanghai. Output only the tool call."),
        ], tools: [toolSchema()])
        FileHandle.standardError.write(Data(
            ("[spark-gguf][tool] text=\(result.text) calls=\(result.calls)\n").utf8))
        XCTAssertFalse(result.calls.isEmpty,
                       "GLM47 工具调用必须被解析成结构化 ToolCall")
        XCTAssertEqual(result.calls.first?.name, "get_current_time")
        XCTAssertEqual(result.calls.first?.arguments["timezone"]?.stringValue,
                       "Asia/Shanghai")
    }

    func testThinkingModeRendersAndGenerates() async throws {
        let path = try modelPath()
        // 模板在 enable_thinking=true 时以 `<think>` 结尾；验证流式思考切分
        // 与生成前缀不会因非 Qwen 的 assistant 标记而失败。
        let service = makeService(path, maxTokens: 128, enableThinking: true)
        defer { service.close() }
        try await service.preload()
        var sawDone = false
        var reasoning = ""
        for try await event in service.chatStream(
            messages: [
                ChatMessage(role: .system, content: "You are a helpful assistant."),
                ChatMessage(role: .user, content: "Reply with exactly: OK"),
            ], tools: [], temperature: 0) {
            switch event.kind {
            case .reasoning: reasoning += event.content
            case .done: sawDone = true
            case .content: break
            }
        }
        XCTAssertTrue(sawDone, "thinking 模式的流必须正常结束")
        XCTAssertFalse(reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                       "模板以 <think> 开头，应切分出思考内容")
    }

    func testFlashAttentionWithQuantizedKV() async throws {
        let path = try modelPath()
        // 默认组合就是风险最高的：FA 开启 + KV q8_0 量化同时生效。
        let service = makeService(path, maxTokens: 48,
                                  kvInt8: true, flashAttention: true)
        defer { service.close() }
        try await service.preload()
        let result = try await collect(service, messages: [
            ChatMessage(role: .user, content: "Say OK."),
        ])
        XCTAssertFalse(result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    // MARK: - 混合 sliding/full 注意力的跨轮 KV

    func testMultiTurnKVReuse() async throws {
        let path = try modelPath()
        let service = makeService(path, maxTokens: 48)
        defer { service.close() }
        try await service.preload()
        let box = LogBox()
        func logged(_ messages: [ChatMessage]) async throws {
            try await AgentLog.withTaskSink({ box.append($0) }) {
                _ = try await collect(service, messages: messages)
            }
        }
        let system = ChatMessage(role: .system, content: "你是简洁的助手。")
        let first = try await collect(service, messages: [
            system, ChatMessage(role: .user, content: "说 A。"),
        ])
        XCTAssertFalse(first.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        // 再跑两轮并捕获 KV 日志：首轮已在缓存中，后续应复用或安全降级。
        try await logged([
            system,
            ChatMessage(role: .user, content: "说 A。"),
            ChatMessage(role: .assistant, content: "A"),
            ChatMessage(role: .user, content: "说 B。"),
        ])
        try await logged([
            system, ChatMessage(role: .user, content: "说 C。"),
        ])
        let modes = box.lines
            .filter { $0.contains("[gguf-kv] mode=") }
            .compactMap { line -> String? in
                guard let range = line.range(of: "mode=") else { return nil }
                return String(line[range.upperBound...].prefix { $0.isLetter })
            }
        FileHandle.standardError.write(Data(
            ("[spark-gguf][kv] modes=\(modes) lines=\(box.lines.filter { $0.contains("[gguf") })\n").utf8))
        XCTAssertFalse(box.lines.contains { $0.contains("llama_decode 返回") },
                       "KV 裁剪降级不得导致 decode 失败")
        XCTAssertFalse(box.lines.contains { $0.contains("chat template 渲染失败") },
                       "带/不带 tools 的模板都不能再因 separators 报错")
        XCTAssertGreaterThanOrEqual(modes.count, 2,
                                    "至少两次 KV 路径判定：\(modes)")
    }

    // MARK: - 纯逻辑：GGUF/ONNX 共用的工具文本解析

    func testGLM47TextParser() {
        let allowed: Set<String> = ["get_current_time", "rag_status"]
        let withArgs = "<tool_call>get_current_time<arg_key>timezone</arg_key>"
            + "<arg_value>Asia/Shanghai</arg_value></tool_call>"
        let calls = LocalToolCallParser.extract(raw: withArgs, allowedNames: allowed)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.name, "get_current_time")
        XCTAssertEqual(calls.first?.arguments["timezone"]?.stringValue,
                       "Asia/Shanghai")

        let zero = LocalToolCallParser.extract(
            raw: "<tool_call>rag_status</tool_call>", allowedNames: allowed)
        XCTAssertEqual(zero.count, 1, "零参数工具（上游库不支持）必须能兜底")
        XCTAssertEqual(zero.first?.name, "rag_status")
        XCTAssertTrue(zero.first?.arguments.isEmpty ?? false)

        XCTAssertTrue(LocalToolCallParser.extract(
            raw: "<tool_call>evil</tool_call>", allowedNames: allowed).isEmpty)
        XCTAssertEqual(LocalToolCallParser.extract(
            raw: "<tool_call>{\"name\":\"rag_status\",\"arguments\":{}}</tool_call>",
            allowedNames: allowed).count, 1, "JSON 方言仍走原路径")
        XCTAssertEqual(LocalToolCallParser.extract(
            raw: "<tool_call><function=rag_status></function></tool_call>",
            allowedNames: allowed).count, 1, "Qwen XML 方言仍走原路径")
    }
}
