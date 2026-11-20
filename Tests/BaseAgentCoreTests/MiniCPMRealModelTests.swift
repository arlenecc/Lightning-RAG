import XCTest
import Foundation
@testable import BaseAgentCore

/// MiniCPM5 真机测试：实际加载 /Volumes/SSD/MiniCPM5-2B-MLX-int4 生成工具调用，
/// 走完整 chatStream（chat template 渲染 → 模型生成 → 框架解析 → 文本兜底），
/// 验证 App 真实链路能解析 MiniCPM 的裸 <function>/<paramname> 方言。
final class MiniCPMRealModelTests: XCTestCase {
    private func modelDirectory() throws -> URL {
        let path = ProcessInfo.processInfo.environment["MINICPM5_MODEL"]
            ?? "/Volumes/SSD/MiniCPM5-2B-MLX-int4"
        guard FileManager.default.fileExists(atPath: path + "/config.json") else {
            throw XCTSkip("本地缺少 MiniCPM5 模型：\(path)")
        }
        return URL(fileURLWithPath: path)
    }

    func testRealModelToolCall() async throws {
        let dir = try modelDirectory()
        let service = MLXLLMService(
            directory: dir, modelName: dir.lastPathComponent,
            maxTokens: 200, topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: true, contextWindow: 4096)
        defer { service.close() }

        let tool: BaseAgentCore.JSONValue = .object([
            "type": .string("function"),
            "function": .object([
                "name": .string("rag_tools"),
                "description": .string("Query the local knowledge base. Use op=status for statistics."),
                "parameters": .object([
                    "type": .string("object"),
                    "properties": .object([
                        "op": .object([
                            "type": .string("string"),
                            "enum": .array([.string("status"), .string("search")]),
                        ]),
                    ]),
                    "required": .array([.string("op")]),
                ]),
            ]),
        ])

        var text = ""
        var reasoning = ""
        var raw = ""
        var truncated = false
        var calls: [BaseAgentCore.ToolCall] = []
        for try await event in service.chatStream(
            messages: [
                ChatMessage(role: .system,
                            content: "You are a helpful assistant. Use the provided tool when asked about the knowledge base."),
                ChatMessage(role: .user,
                            content: "Use rag_tools with op=status to check the knowledge base. Output only the tool call."),
            ],
            tools: [tool], temperature: 0) {
            switch event.kind {
            case .content: text += event.content
            case .reasoning: reasoning += event.content
            case .done:
                calls = event.toolCalls
                raw = event.rawContent
                truncated = event.truncated
            }
        }
        FileHandle.standardError.write(Data(
            ("[minicpm5][tool] thinking=\(reasoning)\ncontent=\(text)\n"
             + "raw=\(raw)\ntruncated=\(truncated)\ncalls=\(calls)\n").utf8))
        XCTAssertFalse(calls.isEmpty, "MiniCPM5 工具调用必须被解析成结构化 ToolCall")
        XCTAssertEqual(calls.first?.name, "rag_tools")
        XCTAssertEqual(calls.first?.arguments["op"]?.stringValue, "status")
    }
}
