import MCP
import XCTest
@testable import BaseAgentCore

/// MCP 客户端 HTTP 形态真实往返测试：
/// 应用内 `MCPClient`（Streamable HTTP 直连形态，官方 `HTTPClientTransport`）
/// 连接本仓库 `RAGMCPServer`（真实 NWListener 监听），验证无子进程形态下
/// 握手 / 工具清单 / 调用与返回 / 错误路径 / 关闭语义全部正常。
final class MCPClientHTTPTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mcp-http-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    /// 与 RAGMCPServerTests 相同的关键词基向量 embedding（保证检索可命中）。
    private final class KeywordEmbedding: EmbeddingProvider {
        let identifier = "mcp-http-test"
        let dimension = 8
        private let bases: [String: [Float]] = [
            "apple": [1, 0, 0, 0, 0, 0, 0, 0],
            "volcano": [0, 1, 0, 0, 0, 0, 0, 0],
        ]

        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map { text in
                var vector = [Float](repeating: 0, count: dimension)
                for (word, base) in bases
                where text.lowercased().contains(word) {
                    for i in 0..<dimension { vector[i] += base[i] }
                }
                if vector.allSatisfy({ $0 == 0 }) { vector[7] = 1 }
                return VectorCodec.normalize(vector)
            }
        }

        func embedQuery(_ text: String) async throws -> [Float] {
            try await embedDocuments([text])[0]
        }
    }

    private func makeEngineAndServer() throws -> (RAGEngine, RAGMCPServer) {
        let kb = directory.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        var config = AgentConfig()
        config.workspace = directory.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 60
        config.ragChunkOverlap = 4
        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: KeywordEmbedding(), reranker: nil)
        let server = RAGMCPServer(
            engine: engine,
            options: RAGMCPServerOptions(listenLAN: false, port: 0))
        try server.start()
        return (engine, server)
    }

    private func waitReady(_ server: RAGMCPServer, timeout: TimeInterval = 10) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while !server.isRunning {
            if Date() > deadline {
                XCTFail("MCP 服务 10s 内未就绪")
                return
            }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
    }

    func testMCPClientOverStreamableHTTP() async throws {
        let (engine, server) = try makeEngineAndServer()
        defer { engine.close(); server.stop() }
        try await waitReady(server)
        guard let port = server.boundPort else { return XCTFail("服务未监听") }

        // HTTP 形态客户端：无子进程，直连端点。
        let client = MCPClient(name: "rag-http",
                               url: "http://127.0.0.1:\(port)/mcp",
                               timeout: 30)
        defer { client.shutdown() }
        XCTAssertNotNil(client.remoteURL)

        // 1. initialize 握手（官方协议协商）。
        try await client.start()
        XCTAssertTrue(client.isRunning)
        XCTAssertEqual(
            client.serverInfoValue.objectValue?["serverInfo"]?
                .objectValue?["name"]?.stringValue,
            "lightning-rag-mcp")
        let negotiated = client.serverInfoValue
            .objectValue?["protocolVersion"]?.stringValue ?? ""
        XCTAssertTrue(MCP.Version.supported.contains(negotiated),
                      "协商版本应在支持集合内：\(negotiated)")

        // 2. tools/list：7 个工具（真实 RAG 工具层）。
        let tools = client.listTools()
        XCTAssertEqual(tools.count, 7)
        let names = tools.compactMap { $0.objectValue?["name"]?.stringValue }
        XCTAssertTrue(names.contains("hybrid_search_with_rerank"))
        XCTAssertTrue(names.contains("parse_and_ingest_document"))

        // 3. tools/call：入库（真实切片 + 向量化）。
        let body = String(repeating: "apple orchard manual. ", count: 40)
        let ingestOutput = try await client.callTool(
            name: "parse_and_ingest_document",
            arguments: ["filename": .string("orchard.md"), "content": .string(body)])
        XCTAssertTrue(ingestOutput.contains("入库完成"), ingestOutput)
        XCTAssertTrue(ingestOutput.contains("orchard.md"), ingestOutput)

        // 4. 检索命中。
        let searchOutput = try await client.callTool(
            name: "hybrid_search_with_rerank",
            arguments: ["query": .string("apple orchard"), "top_k": .int(3)])
        XCTAssertTrue(searchOutput.contains("apple"), searchOutput)

        // 5. 原文透传 + 清单。
        let docOutput = try await client.callTool(
            name: "get_document", arguments: ["doc_name": .string("orchard.md")])
        XCTAssertTrue(docOutput.contains("apple"), docOutput)
        let listOutput = try await client.callTool(
            name: "list_document", arguments: [:])
        XCTAssertTrue(listOutput.contains("orchard.md"), listOutput)

        // 6. 未知工具：官方 SDK 抛协议级 -32602。
        do {
            _ = try await client.callTool(name: "no_such_tool", arguments: [:])
            XCTFail("未知工具应抛 -32602")
        } catch let error as MCP.MCPError {
            guard case .invalidParams = error else {
                return XCTFail("应为 -32602，实际：\(error)")
            }
        }

        // 7. isError 路径：删除后 get_document 报错。
        let deleteOutput = try await client.callTool(
            name: "delete_document", arguments: ["source": .string("orchard.md")])
        XCTAssertTrue(deleteOutput.contains("已遗忘"), deleteOutput)
        do {
            _ = try await client.callTool(
                name: "get_document", arguments: ["doc_name": .string("orchard.md")])
            XCTFail("删除后 get_document 应报错")
        } catch let error as MCPClientError {
            XCTAssertTrue(error.message.contains("未找到文档"),
                          "应来自工具 isError：\(error.message)")
        }

        // 8. shutdown 后不可复活。
        client.shutdown()
        XCTAssertFalse(client.isRunning)
        do {
            _ = try await client.callTool(name: "list_document", arguments: [:])
            XCTFail("shutdown 后应抛错")
        } catch let error as MCPClientError {
            XCTAssertTrue(error.message.contains("已关闭"),
                          "应提示已关闭：\(error.message)")
        }
    }

    /// MCPServerConfig 混合形态路由：url 优先，缺省回退 stdio 字段。
    func testMCPServerConfigURLRouting() throws {
        let http = MCPServerConfig(name: "a", command: "", url: "http://127.0.0.1:9/mcp")
        XCTAssertEqual(http.url, "http://127.0.0.1:9/mcp")

        // Codable 往返：url 字段保留。
        let data = try JSONEncoder().encode(http)
        let back = try JSONDecoder().decode(MCPServerConfig.self, from: data)
        XCTAssertEqual(back, http)

        // 旧配置（无 url 键）解码兼容。
        let legacy = Data(#"{"name":"b","command":"npx","args":["-y","x"]}"#.utf8)
        let old = try JSONDecoder().decode(MCPServerConfig.self, from: legacy)
        XCTAssertNil(old.url)
        XCTAssertEqual(old.command, "npx")
    }
}
