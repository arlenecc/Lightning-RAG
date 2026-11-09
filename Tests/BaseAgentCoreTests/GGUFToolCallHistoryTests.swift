import XCTest
import Foundation
import llama
@testable import BaseAgentCore

/// Spark-X2.6 GGUF 的「历史 tool_calls 回灌」回归。
///
/// 模板要求 `tool_call.function.arguments` 是 mapping，否则直接
/// `raise_exception('... must be a dictionary')`；而 GGUF 渲染层历来把
/// arguments 传成 JSON 字符串（ChatML / Qwen / Llama 模板正是这样用的）。
/// 修复按模板特征只给 Spark/GLM47 传字典，其它模型保持 `.string` 不动。
final class GGUFToolCallHistoryTests: XCTestCase {

    private func sparkPath() throws -> String {
        let path = ProcessInfo.processInfo.environment["SPARK_X26_GGUF_MODEL"]
            ?? "/Volumes/SSD/gguf/Spark-X2.6-4B.i1-Q4_K_M.gguf"
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("本地缺少 Spark-X2.6 GGUF：\(path)")
        }
        return path
    }

    private func controlPath() throws -> String {
        let path = ProcessInfo.processInfo.environment["OTHER_GGUF_MODEL"]
            ?? "/Volumes/SSD/gguf/Qwen3.5-2B-Q4_K_M.gguf"
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("本地缺少对照 GGUF 模型：\(path)")
        }
        return path
    }

    /// 带 assistant.tool_calls + tool 结果的历史：正是因为这一段才触发
    /// 模板里的 `is not mapping` 检查（单轮首调不会触发）。
    private func historyMessages() -> [ChatMessage] {
        let call = BaseAgentCore.ToolCall(
            id: "call_history_1", name: "get_weather",
            arguments: ["city": .string("Beijing")],
            rawArguments: "{\"city\":\"Beijing\"}")
        return [
            ChatMessage(role: .system, content: "You are a helpful assistant."),
            ChatMessage(role: .user, content: "查询北京天气"),
            ChatMessage(role: .assistant, content: "", toolCalls: [call]),
            ChatMessage(role: .tool, content: "晴，26℃",
                        name: "get_weather", toolCallID: "call_history_1"),
            ChatMessage(role: .user, content: "用一句话总结"),
        ]
    }

    private func toolSchema() -> BaseAgentCore.JSONValue {
        .object([
            "type": .string("function"),
            "function": .object([
                "name": .string("get_weather"),
                "description": .string("查询城市天气"),
                "parameters": .object([
                    "type": .string("object"),
                    "properties": .object([
                        "city": .object(["type": .string("string")]),
                    ]),
                    "required": .array([.string("city")]),
                ]),
            ]),
        ])
    }

    // MARK: - 模板特征检测：严格限定在需要 mapping 的模板

    func testTemplateFeatureDetectionIsNarrow() {
        let spark = "{%- if tool_call.function.arguments is not mapping %}"
            + "raise_exception('must be a dictionary'){%- endif %}"
            + "{%- for k, v in args.items() %}<arg_key>{{ k }}</arg_key>"
            + "<arg_value>{{ v }}</arg_value>{%- endfor %}"
        XCTAssertTrue(LlamaCppPromptRenderer.templateNeedsMappingToolArguments(spark))

        let others = [
            // Qwen ChatML：arguments 当作字符串原样输出
            "{% for m in messages %}{{'<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>'}}{% endfor %}{% for tc in message.tool_calls %}{{ tc.function.arguments }}{% endfor %}",
            // Llama 3
            "{{ bos_token }}{% for m in messages %}{{'<|start_header_id|>' + m.role + '<|end_header_id|>'}}{% endfor %}",
            // Gemma
            "{% for m in messages %}{{'<start_of_turn>' + m.role + '\\n'}}{% endfor %}",
            // 普通 JSON / 无工具模板
            "{\"name\":\"foo\"}",
            // 只有 <arg_key> 字样但不要求 mapping：不能误判
            "Use <arg_key> as a literal example in prose.",
            "",
        ]
        for template in others {
            XCTAssertFalse(
                LlamaCppPromptRenderer.templateNeedsMappingToolArguments(template),
                "其它模板必须保持字符串形态：\(template.prefix(100))")
        }
    }

    func testRealSparkTemplateNeedsMapping() throws {
        let model = try LlamaCppModel(path: try sparkPath(), gpuLayers: 0)
        let template = try XCTUnwrap(model.chatTemplate)
        XCTAssertTrue(template.contains("<arg_key>"))
        XCTAssertTrue(LlamaCppPromptRenderer.templateNeedsMappingToolArguments(template))
    }

    // MARK: - 核心复现：含历史 tool_calls 的渲染必须成功且参数被正确展开

    func testSparkToolCallHistoryRenders() throws {
        let model = try LlamaCppModel(path: try sparkPath(), gpuLayers: 0)
        let rendered = try LlamaCppPromptRenderer.render(
            template: model.chatTemplate,
            model: model,
            messages: historyMessages(),
            tools: [toolSchema()],
            enableThinking: false,
            addGenerationPrompt: true)
        XCTAssertFalse(rendered.isEmpty)
        XCTAssertTrue(rendered.contains("<arg_key>city</arg_key>"),
                      "历史工具参数必须被模板按 mapping 展开：\(rendered.suffix(400))")
        XCTAssertTrue(rendered.contains("<arg_value>Beijing</arg_value>"),
                      "参数值必须正确：\(rendered.suffix(400))")
    }

    /// 零参数工具的历史形态：空 mapping 也要能渲染（不触发 is not mapping）。
    func testSparkZeroArgumentHistoryRenders() throws {
        let model = try LlamaCppModel(path: try sparkPath(), gpuLayers: 0)
        let call = BaseAgentCore.ToolCall(
            id: "call_zero", name: "rag_status",
            arguments: [:], rawArguments: "{}")
        let messages: [ChatMessage] = [
            ChatMessage(role: .user, content: "检查知识库"),
            ChatMessage(role: .assistant, content: "", toolCalls: [call]),
            ChatMessage(role: .tool, content: "4056 chunks",
                        name: "rag_status", toolCallID: "call_zero"),
            ChatMessage(role: .user, content: "总结"),
        ]
        let rendered = try LlamaCppPromptRenderer.render(
            template: model.chatTemplate, model: model, messages: messages,
            tools: [], enableThinking: false, addGenerationPrompt: true)
        XCTAssertTrue(rendered.contains("<tool_call>rag_status</tool_call>"),
                      "零参数历史调用应渲染为纯工具名：\(rendered.suffix(400))")
    }

    /// 真机推理闭环：历史含 tool_calls 时不能抛模板异常，且能继续生成。
    func testSparkToolCallHistoryGenerates() async throws {
        let path = try sparkPath()
        let service = LlamaCppLLMService(
            modelPath: path, contextWindow: 4096, maxTokens: 64,
            topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false, gpuLayers: 99)
        defer { service.close() }
        try await service.preload()
        var text = ""
        var sawDone = false
        for try await event in service.chatStream(
            messages: historyMessages(), tools: [toolSchema()], temperature: 0) {
            switch event.kind {
            case .content: text += event.content
            case .done: sawDone = true
            case .reasoning: break
            }
        }
        XCTAssertTrue(sawDone, "含历史工具调用的流必须正常结束")
        XCTAssertFalse(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                       "必须能继续生成正文")
    }

    // MARK: - 对照模型：走原 `.string` 路径，渲染结果不变

    func testControlModelKeepsStringArguments() throws {
        let model = try LlamaCppModel(path: try controlPath(), gpuLayers: 0)
        let template = try XCTUnwrap(model.chatTemplate)
        XCTAssertFalse(
            LlamaCppPromptRenderer.templateNeedsMappingToolArguments(template),
            "对照模型不得进入字典形态分支")
        let rendered = try LlamaCppPromptRenderer.render(
            template: model.chatTemplate, model: model,
            messages: historyMessages(), tools: [toolSchema()],
            enableThinking: false, addGenerationPrompt: true)
        XCTAssertFalse(rendered.isEmpty, "对照模型必须仍能正常渲染")
    }
}
