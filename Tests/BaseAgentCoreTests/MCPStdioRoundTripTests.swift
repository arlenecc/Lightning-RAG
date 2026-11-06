import MCP
import XCTest
@testable import BaseAgentCore

/// MCP 客户端 ↔ MCP 服务端 真实往返模拟测试：
/// 应用内 `MCPClient`（官方 SDK stdio 客户端封装）以子进程方式拉起
/// 本仓库的 `rag-mcp-stdio`（官方 SDK stdio 服务端），完整验证
/// initialize 握手、tools/list 与 7 个工具的调用与返回。
///
/// 门控：找到 rag-mcp-stdio 可执行文件才运行（`RAG_MCP_STDIO_BIN` 可显式
/// 指定；默认从仓库 .build/debug 推导）。embedding/reranker 走仓库
/// `models/` 目录（通过 `BASE_AGENT_MODELS_DIR` 锚定）。
final class MCPStdioRoundTripTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mcp-roundtrip-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    /// 定位 rag-mcp-stdio：环境变量优先，其次 .build/debug 与
    /// .build/<arch>-apple-macosx/debug（swift test 的默认构建目录）。
    private func stdioBinaryPath() -> String? {
        if let override = ProcessInfo.processInfo.environment["RAG_MCP_STDIO_BIN"],
           FileManager.default.isExecutableFile(atPath: override) {
            return override
        }
        // #filePath = <repo>/Tests/BaseAgentCoreTests/MCPStdioRoundTripTests.swift
        let testsDir = (#filePath as NSString).deletingLastPathComponent
        let repoRoot = ((testsDir as NSString).deletingLastPathComponent as NSString)
            .deletingLastPathComponent
        let buildDirs = ["", ".build/debug", ".build/arm64-apple-macosx/debug",
                         ".build/x86_64-apple-macosx/debug"]
        for relative in buildDirs {
            let candidate = (repoRoot as NSString)
                .appendingPathComponent(relative)
            let binary = (candidate as NSString).appendingPathComponent("rag-mcp-stdio")
            if FileManager.default.isExecutableFile(atPath: binary) {
                return binary
            }
        }
        return nil
    }

    private func modelsDirectory() -> String {
        let testsDir = (#filePath as NSString).deletingLastPathComponent
        let repoRoot = ((testsDir as NSString).deletingLastPathComponent as NSString)
            .deletingLastPathComponent
        let models = (repoRoot as NSString).appendingPathComponent("models")
        return FileManager.default.fileExists(atPath: models) ? models : ""
    }

    func testClientServerRoundTripOverStdio() async throws {
        guard let binary = stdioBinaryPath() else {
            let testsDir = (#filePath as NSString).deletingLastPathComponent
            throw XCTSkip("未找到 rag-mcp-stdio（filePath=\(#filePath)，"
                + "repoRoot=\((testsDir as NSString).deletingLastPathComponent)，"
                + "可先 swift build 或设置 RAG_MCP_STDIO_BIN）")
        }
        let kb = directory.appendingPathComponent("kb").path
        var env: [String: String] = [:]
        let models = modelsDirectory()
        if !models.isEmpty { env["BASE_AGENT_MODELS_DIR"] = models }

        let client = MCPClient(name: "rag-stdio", command: binary,
                               args: ["--kb", kb], env: env, timeout: 120)
        defer { client.shutdown() }

        // 1. 启动子进程 + initialize 握手（官方协议协商）。
        try await client.start()
        XCTAssertTrue(client.isRunning)
        let serverInfo = client.serverInfoValue
        XCTAssertEqual(
            serverInfo.objectValue?["serverInfo"]?.objectValue?["name"]?.stringValue,
            "lightning-rag-mcp",
            "initialize 应返回服务端信息：\(serverInfo.jsonString())")
        XCTAssertNotNil(
            serverInfo.objectValue?["serverInfo"]?.objectValue?["version"]?.stringValue)

        // 2. tools/list：7 个工具，schema 含 properties/required。
        let tools = client.listTools()
        XCTAssertEqual(tools.count, 7, "工具清单：\(tools.map { $0.jsonString() })")
        let names = tools.compactMap { $0.objectValue?["name"]?.stringValue }
        XCTAssertEqual(names, ["parse_and_ingest_document", "delete_document",
                               "forget_context", "list_document",
                               "extract_metadata", "hybrid_search_with_rerank",
                               "get_document"])
        let searchSchema = tools
            .first { $0.objectValue?["name"]?.stringValue == "hybrid_search_with_rerank" }?
            .objectValue?["inputSchema"]?.objectValue
        XCTAssertEqual(searchSchema?["required"]?.arrayValue?.compactMap(\.stringValue),
                       ["query"], "inputSchema 应透传 required")

        // 3. tools/call：入库（触发真实 embedding 模型加载 + 切片 + 向量化）。
        let body = String(repeating: "pineapple cultivation handbook. ", count: 40)
        let ingestOutput = try await client.callTool(
            name: "parse_and_ingest_document",
            arguments: ["filename": .string("pineapple.md"), "content": .string(body)])
        XCTAssertTrue(ingestOutput.contains("入库完成"), ingestOutput)
        XCTAssertTrue(ingestOutput.contains("pineapple.md"), ingestOutput)

        // 4. list_document：清单返回。
        let listOutput = try await client.callTool(name: "list_document", arguments: [:])
        XCTAssertTrue(listOutput.contains("pineapple.md"), listOutput)

        // 5. hybrid_search_with_rerank：检索命中（FTS5 + 向量 + 重排）。
        let searchOutput = try await client.callTool(
            name: "hybrid_search_with_rerank",
            arguments: ["query": .string("pineapple cultivation"), "top_k": .int(3)])
        XCTAssertTrue(searchOutput.contains("pineapple"), "应命中已入库文档：\(searchOutput)")

        // 6. get_document：原文透传。
        let docOutput = try await client.callTool(
            name: "get_document", arguments: ["doc_name": .string("pineapple.md")])
        XCTAssertTrue(docOutput.contains("pineapple"), docOutput)

        // 7. extract_metadata：目录结构。
        let metaOutput = try await client.callTool(
            name: "extract_metadata", arguments: ["doc_name": .string("pineapple.md")])
        XCTAssertTrue(metaOutput.contains("## 目录"), metaOutput)

        // 8. 错误路径：删除后 get_document 返回 isError。
        let deleteOutput = try await client.callTool(
            name: "delete_document", arguments: ["source": .string("pineapple.md")])
        XCTAssertTrue(deleteOutput.contains("已遗忘"), deleteOutput)
        do {
            _ = try await client.callTool(
                name: "get_document", arguments: ["doc_name": .string("pineapple.md")])
            XCTFail("删除后 get_document 应返回 isError")
        } catch let error as MCPClientError {
            XCTAssertTrue(error.message.contains("未找到文档"),
                          "错误应来自工具 isError：\(error.message)")
        }

        // 9. forget_context 别名：可达服务端（未找到文档走 isError 错误路径）。
        _ = try? await client.callTool(
            name: "forget_context", arguments: ["source": .string("不存在的文档")])

        // 10. 未知工具：官方 SDK 客户端收到协议级 -32602。
        do {
            _ = try await client.callTool(name: "no_such_tool", arguments: [:])
            XCTFail("未知工具应抛 -32602")
        } catch let error as MCP.MCPError {
            guard case .invalidParams = error else {
                return XCTFail("未知工具应为协议级 -32602，实际：\(error)")
            }
        }
    }
}
