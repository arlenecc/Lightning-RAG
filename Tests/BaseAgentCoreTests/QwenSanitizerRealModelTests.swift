import XCTest
import Foundation
@testable import BaseAgentCore

/// 真机验证 ContentSanitizer 不影响 Qwen：工具调用被正确解析，正文无 XML 残留、正常输出。
final class QwenSanitizerRealModelTests: XCTestCase {
    private func modelDirectory() throws -> URL {
        let path = ProcessInfo.processInfo.environment["QWEN_SANITIZER_MODEL"]
            ?? "/Users/arlenecc/codebase/agent-app/models/Qwen3.5-2B-MLX-4bit"
        guard FileManager.default.fileExists(atPath: path + "/config.json") else {
            throw XCTSkip("本地缺少 Qwen 模型：\(path)")
        }
        return URL(fileURLWithPath: path)
    }

    func testQwenToolCallAndContentIntact() async throws {
        let dir = try modelDirectory()
        let service = MLXLLMService(
            directory: dir, modelName: dir.lastPathComponent,
            maxTokens: 200, topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false, contextWindow: 4096)
        defer { service.close() }

        let tool: JSONValue = .object([
            "type": .string("function"),
            "function": .object([
                "name": .string("get_weather"),
                "description": .string("Get weather for a city."),
                "parameters": .object([
                    "type": .string("object"),
                    "properties": .object([
                        "city": .object(["type": .string("string")]),
                    ]),
                    "required": .array([.string("city")]),
                ]),
            ]),
        ])
        var text = ""
        var calls: [BaseAgentCore.ToolCall] = []
        for try await event in service.chatStream(
            messages: [
                ChatMessage(role: .system, content: "You are helpful. Use the tool to answer."),
                ChatMessage(role: .user, content: "Use get_weather for Beijing, then tell me if it's cold. Output the tool call first."),
            ],
            tools: [tool], temperature: 0) {
            switch event.kind {
            case .content: text += event.content
            case .done: calls = event.toolCalls
            default: break
            }
        }
        let diag = "[qwen-san] content=[\(text)] calls=\(calls)\n"
        FileHandle.standardError.write(Data(diag.utf8))
        XCTAssertFalse(calls.isEmpty, "Qwen 工具调用必须被解析")
        XCTAssertEqual(calls.first?.name, "get_weather")
        XCTAssertFalse(text.contains("<tool_call"), "正文不得残留 tool_call XML")
    }

    /// 第二轮带工具结果：Qwen 应输出正常中文正文，不能被 sanitizer 吞掉。
    func testQwenPlainAnswerIntact() async throws {
        let dir = try modelDirectory()
        let service = MLXLLMService(
            directory: dir, modelName: dir.lastPathComponent,
            maxTokens: 120, topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false, contextWindow: 4096)
        defer { service.close() }
        var text = ""
        for try await event in service.chatStream(
            messages: [
                ChatMessage(role: .system, content: "用中文简洁回答。"),
                ChatMessage(role: .user, content: "1 和 2 哪个大？直接回答，不要调用工具。"),
            ],
            tools: [], temperature: 0) {
            if case .content = event.kind { text += event.content }
        }
        let diag = "[qwen-plain] content=[\(text)]\n"
        FileHandle.standardError.write(Data(diag.utf8))
        XCTAssertFalse(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                       "普通中文答复不得被吞")
    }
}
