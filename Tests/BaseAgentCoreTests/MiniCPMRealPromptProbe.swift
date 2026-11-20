import XCTest
import Foundation
@testable import BaseAgentCore

/// 真机复现用户场景：MiniCPM5 + 复杂 system prompt + 多工具 + "你知识库里有什么"。
/// 验证工具调用能否被解析、是否会重复空转。
final class MiniCPMRealPromptProbe: XCTestCase {
    private func modelDirectory() throws -> URL {
        let path = ProcessInfo.processInfo.environment["MINICPM5_MODEL"]
            ?? "/Volumes/SSD/MiniCPM5-2B-MLX-int4"
        guard FileManager.default.fileExists(atPath: path + "/config.json") else {
            throw XCTSkip("本地缺少 MiniCPM5 模型：\(path)")
        }
        return URL(fileURLWithPath: path)
    }

    func testKnowledgeBaseQuestionWithManyTools() async throws {
        let dir = try modelDirectory()
        let service = MLXLLMService(
            directory: dir, modelName: dir.lastPathComponent,
            maxTokens: 200, topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: true, contextWindow: 4096)
        defer { service.close() }

        // 模拟真实的多工具 schema（rag_tools 是 composite，含 op 枚举）。
        func tool(_ name: String, _ desc: String,
                  props: [String: JSONValue] = [:]) -> JSONValue {
            .object([
                "type": .string("function"),
                "function": .object([
                    "name": .string(name),
                    "description": .string(desc),
                    "parameters": .object([
                        "type": .string("object"),
                        "properties": .object(props),
                        "required": .array([.string("op")]),
                    ]),
                ]),
            ])
        }
        let ragTools = tool("rag_tools",
            "Local knowledge base (RAG). Prefer this over web for domain questions. "
                + "Search ALL mounted partitions by default. Choose via op: "
                + "search / outline / status / ingest.",
            props: [
                "op": .object(["type": .string("string"),
                               "enum": .array([.string("search"), .string("outline"),
                                               .string("status"), .string("ingest")])]),
                "query": .object(["type": .string("string")]),
                "book_name": .object(["type": .string("string")]),
            ])
        let webScan = tool("web_scan", "Fetch a web page.",
            props: ["url": .object(["type": .string("string")])])
        let webSearch = tool("web_search", "Search the web.",
            props: ["query": .object(["type": .string("string")])])
        let fileTools = tool("file_tools", "File read/write.",
            props: ["op": .object(["type": .string("string")])])

        // 真实风格的 system prompt（含知识库策略）。
        let system = """
        You are base-agent, a local autonomous agent.
        ## 交互协议
        纯寒暄直接回答；多步任务先 <summary>…</summary>。
        你有文件读写、脚本执行、联网抓取权限。
        ## 工具
        - 知识库：领域性问题先 rag_tools(op="search") 跨全部分区检索；问清单用 op="status"。
        - 用户给网址或要实时信息用 web_search + web_scan。
        - file_tools 读写本地文件。
        """

        var text = ""
        var reasoning = ""
        var raw = ""
        var calls: [BaseAgentCore.ToolCall] = []
        for try await event in service.chatStream(
            messages: [
                ChatMessage(role: .system, content: system),
                ChatMessage(role: .user, content: "你知识库里有什么"),
            ],
            tools: [ragTools, webScan, webSearch, fileTools], temperature: 0) {
            switch event.kind {
            case .content: text += event.content
            case .reasoning: reasoning += event.content
            case .done:
                calls = event.toolCalls
                raw = event.rawContent
            }
        }
        let diag = "[real-prompt] reasoning=[\(reasoning)]\ncontent=[\(text)]\n"
            + "calls=\(calls) rawHasFunction=\(raw.contains("<function"))\n"
        FileHandle.standardError.write(Data(diag.utf8))
        XCTAssertFalse(calls.isEmpty, "应解析出工具调用")
        XCTAssertEqual(calls.first?.name, "rag_tools")
        // 工具调用 XML 不得泄漏到任何可见输出（正文或思考）。
        XCTAssertFalse(text.contains("<function"), "正文残留 function XML: \(text)")
        XCTAssertFalse(reasoning.contains("<function"), "思考残留 function XML: \(reasoning)")
    }
}
