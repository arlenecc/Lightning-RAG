import XCTest
@testable import BaseAgentCore

/// 模拟 MCP 客户端对接 RAG MCP 服务（Streamable HTTP）的端到端测试。
/// 覆盖：initialize 握手 / tools/list / 六项能力的 tools/call / 错误路径。
final class RAGMCPServerTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rag-mcp-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    // MARK: 测试用 embedding（关键词基向量，保证检索可命中）

    private final class KeywordEmbedding: EmbeddingProvider {
        let identifier = "rag-mcp-test"
        let dimension = 8
        private let bases: [String: [Float]] = [
            "apple": [1, 0, 0, 0, 0, 0, 0, 0],
            "zebra": [0, 1, 0, 0, 0, 0, 0, 0],
            "volcano": [0, 0, 1, 0, 0, 0, 0, 0],
        ]

        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map { text in
                var vector = [Float](repeating: 0, count: dimension)
                for (word, base) in bases
                where text.lowercased().contains(word) {
                    for i in 0..<dimension { vector[i] += base[i] }
                }
                if vector.allSatisfy({ $0 == 0 }) {
                    // 无关键词时做确定性兜底，保证非零向量。
                    vector[7] = 1
                }
                return VectorCodec.normalize(vector)
            }
        }

        func embedQuery(_ text: String) async throws -> [Float] {
            try await embedDocuments([text])[0]
        }
    }

    // MARK: 客户端辅助

    private func makeServer(_ engine: RAGEngine,
                            port: Int = 0) throws -> RAGMCPServer {
        let server = RAGMCPServer(
            engine: engine,
            options: RAGMCPServerOptions(listenLAN: false, port: port))
        try server.start()
        return server
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

    private func exchange(_ payload: String, port: UInt16,
                          method: String = "POST",
                          session: String? = nil,
                          origin: String? = nil,
                          protocolVersion: String? = nil)
        async throws -> (HTTPURLResponse, String) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/mcp")!)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json, text/event-stream",
                         forHTTPHeaderField: "Accept")
        if let session {
            request.setValue(session, forHTTPHeaderField: "Mcp-Session-Id")
        }
        if let origin {
            request.setValue(origin, forHTTPHeaderField: "Origin")
        }
        if let protocolVersion {
            request.setValue(protocolVersion, forHTTPHeaderField: "MCP-Protocol-Version")
        }
        request.httpBody = payload.data(using: .utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        return (response as! HTTPURLResponse,
                String(data: data, encoding: .utf8) ?? "")
    }

    private func header(_ response: HTTPURLResponse, _ name: String) -> String {
        response.allHeaderFields
            .first { ("\($0.key)" as String).lowercased() == name }
            .map { "\($0.value)" } ?? ""
    }

    private func postJSON(_ payload: [String: JSONValue], port: UInt16,
                          session: String? = nil) async throws -> (Int, JSONValue) {
        let (response, text) = try await exchange(
            JSONValue.object(payload).jsonString(),
            port: port, session: session)
        return (response.statusCode, JSONValue.parse(text) ?? .null)
    }

    private func callTool(_ name: String, _ args: [String: JSONValue],
                          port: UInt16, session: String?) async throws
        -> (isError: Bool, text: String) {
        let (status, value) = try await postJSON([
            "jsonrpc": .string("2.0"),
            "id": .int(42),
            "method": .string("tools/call"),
            "params": .object([
                "name": .string(name),
                "arguments": .object(args),
            ]),
        ], port: port, session: session)
        XCTAssertEqual(status, 200)
        guard let result = value.objectValue?["result"]?.objectValue else {
            XCTFail("tools/call 缺少 result：\(value.jsonString())")
            return (true, "")
        }
        let text = result["content"]?.arrayValue?.first?.objectValue?["text"]?.stringValue ?? ""
        let isError = result["isError"]?.boolValue ?? false
        return (isError, text)
    }

    /// 发送 initialize 并解析响应与 `Mcp-Session-Id`。
    private func initialize(port: UInt16,
                            clientVersion: String = "2025-06-18") async throws
        -> (Int, JSONValue, String) {
        let payload = JSONValue.object([
            "jsonrpc": .string("2.0"),
            "id": .int(1),
            "method": .string("initialize"),
            "params": .object([
                "protocolVersion": .string(clientVersion),
                "capabilities": .object([:]),
                "clientInfo": .object(["name": .string("test-client"),
                                       "version": .string("1.0")]),
            ]),
        ]).jsonString()
        let (response, text) = try await exchange(payload, port: port)
        let session = header(response, "mcp-session-id")
        return (response.statusCode, JSONValue.parse(text) ?? .null, session)
    }

    private func makeEngine(_ name: String) throws -> RAGEngine {
        let kb = directory.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        var config = AgentConfig()
        config.workspace = directory.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 60
        config.ragChunkOverlap = 4
        return RAGEngine(config: config, knowledgeBase: kb.path,
                         embedding: KeywordEmbedding(), reranker: nil)
    }

    // MARK: 全链路

    func testMCPEndToEndAllTools() async throws {
        let engine = try makeEngine("kb")
        defer { engine.close() }

        let server = try makeServer(engine)
        defer { server.stop() }
        try await waitReady(server)
        guard let port = server.boundPort else {
            return XCTFail("服务未监听")
        }

        // 1. initialize：协商版本 + 颁发 session。
        let (initStatus, initValue, initSession) = try await initialize(port: port)
        XCTAssertEqual(initStatus, 200)
        XCTAssertEqual(
            initValue.objectValue?["result"]?.objectValue?["protocolVersion"]?.stringValue,
            "2025-06-18")
        XCTAssertEqual(
            initValue.objectValue?["result"]?.objectValue?["serverInfo"]?
                .objectValue?["name"]?.stringValue,
            "lightning-rag-mcp")
        XCTAssertFalse(initSession.isEmpty, "initialize 必须回 Mcp-Session-Id")

        // notifications/initialized → 202 无响应体。
        let (notifResponse, notifBody) = try await exchange(
            JSONValue.object([
                "jsonrpc": .string("2.0"),
                "method": .string("notifications/initialized"),
            ]).jsonString(), port: port, session: initSession)
        XCTAssertEqual(notifResponse.statusCode, 202)
        XCTAssertTrue(notifBody.isEmpty)

        // 2. tools/list：7 个工具（6 能力 + forget_context 别名）。
        let (_, listValue) = try await postJSON([
            "jsonrpc": .string("2.0"),
            "id": .int(2),
            "method": .string("tools/list"),
        ], port: port, session: initSession)
        let tools = listValue.objectValue?["result"]?.objectValue?["tools"]?
            .arrayValue?.compactMap { $0.objectValue?["name"]?.stringValue } ?? []
        XCTAssertEqual(tools, ["parse_and_ingest_document", "delete_document",
                               "forget_context", "list_document",
                               "extract_metadata", "hybrid_search_with_rerank",
                               "get_document"])

        // 3. parse_and_ingest_document：纯文本入库。
        let docBody = String(repeating: "apple orchard biology notes. ", count: 30)
        let (ingestError, ingestText) = try await callTool(
            "parse_and_ingest_document",
            ["filename": .string("notes.md"), "content": .string(docBody)],
            port: port, session: initSession)
        XCTAssertFalse(ingestError, "入库失败：\(ingestText)")
        XCTAssertTrue(ingestText.contains("notes.md"))
        XCTAssertTrue(ingestText.contains("已处理"), "应包含统计：\(ingestText)")

        // 3b. 二进制 base64 入库（第二个文档，供 forget_context 用）。
        let zipBody = String(repeating: "zebra migration data. ", count: 30)
        let (b64Error, b64Text) = try await callTool(
            "parse_and_ingest_document",
            [
                "filename": .string("migration.md"),
                "content_base64": .string(Data(zipBody.utf8).base64EncodedString()),
            ],
            port: port, session: initSession)
        XCTAssertFalse(b64Error, "base64 入库失败：\(b64Text)")

        // 4. list_document：两个文档都在。
        let (_, listText) = try await callTool("list_document", [:],
                                               port: port, session: initSession)
        XCTAssertTrue(listText.contains("notes.md"))
        XCTAssertTrue(listText.contains("migration.md"))

        // 5. hybrid_search_with_rerank：检索命中 apple 文档。
        let (searchError, searchText) = try await callTool(
            "hybrid_search_with_rerank",
            ["query": .string("apple orchard"), "top_k": .int(3)],
            port: port, session: initSession)
        XCTAssertFalse(searchError, "检索失败：\(searchText)")
        XCTAssertTrue(searchText.contains("apple"), "应命中 apple 文档：\(searchText)")

        // 6. extract_metadata：目录 markdown。
        let (metaError, metaText) = try await callTool(
            "extract_metadata", ["doc_name": .string("notes.md")],
            port: port, session: initSession)
        XCTAssertFalse(metaError, "元数据提取失败：\(metaText)")
        XCTAssertTrue(metaText.contains("## 目录"))

        // 7. get_document：原文透传（清洗后的 markdown）。
        let (docError, docText) = try await callTool(
            "get_document", ["doc_name": .string("notes.md")],
            port: port, session: initSession)
        XCTAssertFalse(docError, "原文获取失败：\(docText)")
        XCTAssertTrue(docText.contains("apple"), "原文应包含正文：\(docText)")

        // 8. delete_document：删除后 get_document 失败。
        let (delError, delText) = try await callTool(
            "delete_document", ["source": .string("notes.md")],
            port: port, session: initSession)
        XCTAssertFalse(delError, "删除失败：\(delText)")
        XCTAssertTrue(delText.contains("已遗忘"))
        let (afterDelError, afterDelText) = try await callTool(
            "get_document", ["doc_name": .string("notes.md")],
            port: port, session: initSession)
        XCTAssertTrue(afterDelError, "删除后获取原文应报错：\(afterDelText)")

        // 9. forget_context：别名能力。
        let (forgetError, forgetText) = try await callTool(
            "forget_context", ["source": .string("migration.md")],
            port: port, session: initSession)
        XCTAssertFalse(forgetError, "遗忘失败：\(forgetText)")
        XCTAssertTrue(forgetText.contains("已遗忘"))

        // 10. 未找到的文档：isError + 明确提示。
        let (nfError, nfText) = try await callTool(
            "delete_document", ["source": .string("missing.md")],
            port: port, session: initSession)
        XCTAssertTrue(nfError)
        XCTAssertTrue(nfText.contains("未找到"))

        // 11. 协议错误：未知方法 -32601；未知 session 404；GET 405。
        let (_, unknownMethod) = try await postJSON([
            "jsonrpc": .string("2.0"),
            "id": .int(9),
            "method": .string("resources/list"),
        ], port: port, session: initSession)
        XCTAssertEqual(
            unknownMethod.objectValue?["error"]?.objectValue?["code"]?.intValue,
            -32601)

        let (badSessionResponse, _) = try await exchange(
            JSONValue.object([
                "jsonrpc": .string("2.0"),
                "id": .int(10),
                "method": .string("tools/list"),
            ]).jsonString(), port: port, session: "not-a-session")
        XCTAssertEqual(badSessionResponse.statusCode, 404)

        let (getResponse, _) = try await exchange("", port: port, method: "GET")
        XCTAssertEqual(getResponse.statusCode, 405)

        // 12. 会话强制：无 session 的非 initialize 请求 → 400（规范）。
        let (noSessionResponse, noSessionText) = try await exchange(
            JSONValue.object([
                "jsonrpc": .string("2.0"),
                "id": .int(11),
                "method": .string("tools/list"),
            ]).jsonString(), port: port)
        XCTAssertEqual(noSessionResponse.statusCode, 400)
        XCTAssertTrue(noSessionText.contains("Mcp-Session-Id"))

        // 13. MUST：Origin 防 DNS rebinding —— 非本机来源 403，本机放行。
        let (evilResponse, _) = try await exchange(
            JSONValue.object([
                "jsonrpc": .string("2.0"),
                "id": .int(12),
                "method": .string("tools/list"),
            ]).jsonString(), port: port, session: initSession,
            origin: "http://evil.example.com")
        XCTAssertEqual(evilResponse.statusCode, 403)
        let (localResponse, _) = try await exchange(
            JSONValue.object([
                "jsonrpc": .string("2.0"),
                "id": .int(13),
                "method": .string("tools/list"),
            ]).jsonString(), port: port, session: initSession,
            origin: "http://localhost:5173")
        XCTAssertEqual(localResponse.statusCode, 200)

        // 14. MUST：不支持的 MCP-Protocol-Version → 400；支持的版本放行。
        let (badVersionResponse, _) = try await exchange(
            JSONValue.object([
                "jsonrpc": .string("2.0"),
                "id": .int(14),
                "method": .string("tools/list"),
            ]).jsonString(), port: port, session: initSession,
            protocolVersion: "1999-01-01")
        XCTAssertEqual(badVersionResponse.statusCode, 400)
        let (okVersionResponse, _) = try await exchange(
            JSONValue.object([
                "jsonrpc": .string("2.0"),
                "id": .int(15),
                "method": .string("tools/list"),
            ]).jsonString(), port: port, session: initSession,
            protocolVersion: "2025-03-26")
        XCTAssertEqual(okVersionResponse.statusCode, 200)

        // 15. 未知工具 → JSON-RPC error -32602（协议级错误）。
        let (_, unknownTool) = try await postJSON([
            "jsonrpc": .string("2.0"),
            "id": .int(16),
            "method": .string("tools/call"),
            "params": .object(["name": .string("no_such_tool"),
                               "arguments": .object([:])]),
        ], port: port, session: initSession)
        XCTAssertEqual(
            unknownTool.objectValue?["error"]?.objectValue?["code"]?.intValue,
            -32602)

        // 16. DELETE 显式终止会话 → 之后同 session 请求 404。
        let (deleteResponse, _) = try await exchange(
            "", port: port, method: "DELETE", session: initSession)
        XCTAssertEqual(deleteResponse.statusCode, 200)
        let (afterDeleteStatus, _) = try await postJSON([
            "jsonrpc": .string("2.0"),
            "id": .int(17),
            "method": .string("tools/list"),
        ], port: port, session: initSession)
        XCTAssertEqual(afterDeleteStatus, 404)
    }

    // MARK: 生命周期

    func testServerLifecycleAndPortValidation() async throws {
        let engine = try makeEngine("kb-lifecycle")
        defer { engine.close() }

        // 端口非法直接抛错。
        let badPort = RAGMCPServer(
            engine: engine,
            options: RAGMCPServerOptions(listenLAN: false, port: 99_999))
        XCTAssertThrowsError(try badPort.start())

        // port 0 自动分配；stop 幂等；重启可再次监听。
        let server = try makeServer(engine)
        defer { server.stop() }
        try await waitReady(server)
        guard let port = server.boundPort, port > 0 else {
            return XCTFail("自动分配端口失败")
        }
        server.stop()
        XCTAssertFalse(server.isRunning)
        server.stop()  // 幂等：重复 stop 不崩。
        try server.start()
        try await waitReady(server)
        // 重启后必须重新监听（端口可能变化：旧 socket 释放与重绑存在时序）。
        XCTAssertNotNil(server.boundPort, "重启后应重新监听")
        XCTAssertGreaterThan(server.boundPort ?? 0, 0)
    }

    func testIngestCopiesOutsideFileIntoKnowledgeBase() async throws {
        let engine = try makeEngine("kb-copy")
        defer { engine.close() }
        let server = try makeServer(engine)
        defer { server.stop() }
        try await waitReady(server)
        guard let port = server.boundPort else { return XCTFail("服务未监听") }

        // 知识库目录外的文件：path 方式入库会复制进去。
        let (_, _, session) = try await initialize(port: port)
        let outside = directory.appendingPathComponent("outside.md")
        try String(repeating: "volcano geology report. ", count: 30)
            .write(to: outside, atomically: true, encoding: .utf8)
        let (isError, text) = try await callTool(
            "parse_and_ingest_document", ["path": .string(outside.path)],
            port: port, session: session)
        XCTAssertFalse(isError, "path 入库失败：\(text)")
        XCTAssertTrue(text.contains("outside.md"), "应返回文档名：\(text)")

        // 检索命中，且知识库目录里出现复制副本。
        let (_, searchText) = try await callTool(
            "hybrid_search_with_rerank", ["query": .string("volcano geology")],
            port: port, session: session)
        XCTAssertTrue(searchText.contains("volcano"), "应命中：\(searchText)")
        let kbContents = try FileManager.default.contentsOfDirectory(
            atPath: engine.knowledgeBase)
        XCTAssertTrue(kbContents.contains { $0.contains("outside") },
                      "文件应被复制进知识库：\(kbContents)")
    }
}
