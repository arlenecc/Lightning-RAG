import XCTest
@testable import BaseAgentCore

/// GGUF 工具调用约束回归。
/// 旧实现用 llama.cpp lazy grammar 的 trigger token/regex 自动触发，
/// 触发串会被 grammar 二次消费，Qwen tool-call 模板下抛 C++ 异常并 abort。
/// 现在改为 app 自己检测 `<tool_call>`，然后换用从 `<function=` 开始的普通 grammar。
final class GgufToolCallRegressionTests: XCTestCase {
    private let modelPath = "/Volumes/SSD/gguf/Qwen3.5-0.8B-Q4_K_M.gguf"

    private func makeTools() -> [JSONValue] {
        [
            .object(["type": .string("function"), "function": .object([
                "name": .string("rag_tools"),
                "description": .string("查询知识库状态"),
                "parameters": .object([
                    "type": .string("object"),
                    "properties": .object([
                        "op": .object(["type": .string("string"),
                                       "enum": .array([.string("status")])]),
                    ]),
                    "required": .array([.string("op")]),
                ]),
            ])]),
        ]
    }

    func testAfterTriggerGrammarStartsAtFunction() throws {
        let grammar = try XCTUnwrap(QwenXMLToolGrammar.makeGBNFAfterTrigger(tools: makeTools()))
        let root = grammar.split(separator: "\n").first.map(String.init) ?? ""
        XCTAssertTrue(root.contains("root ::= ws \"<function=\""),
                      "after-trigger grammar 应从 <function= 开始: \(root)")
        XCTAssertFalse(root.contains("<tool_call>"),
                       "after-trigger grammar 不应再包含触发串: \(root)")
    }

    func testToolCallStreamDoesNotCrash() async throws {
        guard FileManager.default.fileExists(atPath: modelPath) else { throw XCTSkip("无小模型") }
        let service = LlamaCppLLMService(
            modelPath: modelPath, contextWindow: 4096, maxTokens: 64,
            topP: 0.8, minP: 0, topK: 20, repetitionPenalty: 1.0,
            enableThinking: false, gpuLayers: 0, kvReuse: false,
            kvInt8: false, flashAttention: false, guidedToolArguments: true)

        let messages = [
            ChatMessage(role: .system, content: "你必须使用工具回答。"),
            ChatMessage(role: .user, content: "调用 rag_tools(op=\"status\") 查询知识库状态。"),
        ]
        let stream = service.chatStream(messages: messages, tools: makeTools(),
                                        temperature: 0,
                                        options: LLMRequestOptions(enableThinking: false,
                                                                   isolated: true))
        var calls: [ToolCall] = []
        for try await event in stream where event.kind == .done {
            calls.append(contentsOf: event.toolCalls)
        }
        XCTAssertFalse(calls.isEmpty, "模型应至少产出一个工具调用")
        XCTAssertEqual(calls.first?.name, "rag_tools")
        XCTAssertEqual(calls.first?.arguments["op"]?.stringValue, "status")
    }
}
