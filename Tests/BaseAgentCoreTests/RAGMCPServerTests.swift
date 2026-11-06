import MCP
import XCTest
@testable import BaseAgentCore

/// 对接 RAG MCP 服务（官方 MCP Swift SDK，Streamable HTTP）的端到端测试。
/// 覆盖：官方 SDK 客户端全链路 / 原始 HTTP 协议行为 / stdio 共用工具层 / 生命周期。
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

    // MARK: 服务端辅助

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

    // MARK: 原始 HTTP 辅助（协议层行为测试用）

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

    /// StatefulHTTPServerTransport 的 POST 响应是 SSE 流：取最后一个非空
    /// `data:` 行（跳过 priming 空事件）解析为 JSON-RPC 消息。
    private func jsonFromSSE(_ body: String) -> JSONValue? {
        var payload: String?
        for line in body.split(separator: "\n") {
            guard line.hasPrefix("data:") else { continue }
            let text = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
            guard !text.isEmpty else { continue }
            payload = text
        }
        guard let payload else { return nil }
        return JSONValue.parse(payload)
    }

    private func postJSON(_ payload: [String: JSONValue], port: UInt16,
                          session: String? = nil) async throws -> (Int, JSONValue) {
        let (response, text) = try await exchange(
            JSONValue.object(payload).jsonString(),
            port: port, session: session)
        return (response.statusCode, jsonFromSSE(text) ?? .null)
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
        return (response.statusCode, jsonFromSSE(text) ?? .null, session)
    }

    // MARK: 全链路（官方 SDK 客户端）

    func testOfficialSDKClientEndToEnd() async throws {
        let engine = try makeEngine("kb")
        defer { engine.close() }

        let server = try makeServer(engine)
        defer { server.stop() }
        try await waitReady(server)
        guard let port = server.boundPort else {
            return XCTFail("服务未监听")
        }

        // 官方 HTTPClientTransport（SSE 流式）对接自建 HTTP 前端。
        let transport = MCP.HTTPClientTransport(
            endpoint: URL(string: "http://127.0.0.1:\(port)/mcp")!,
            streaming: true)
        let client = MCP.Client(name: "official-client", version: "1.0")
        defer { Task { await client.disconnect() } }
        let info = try await client.connect(transport: transport)
        XCTAssertNotNil(info.capabilities.tools, "应声明 tools 能力")
        XCTAssertTrue(MCP.Version.supported.contains(info.protocolVersion))

        // tools/list：7 个工具（6 能力 + forget_context 别名）。
        let (tools, _) = try await client.listTools()
        XCTAssertEqual(tools.map(\.name),
                       ["parse_and_ingest_document", "delete_document",
                        "forget_context", "list_document",
                        "extract_metadata", "hybrid_search_with_rerank",
                        "get_document"])

        // parse_and_ingest_document：纯文本入库。
        let docBody = String(repeating: "apple orchard biology notes. ", count: 30)
        let (ingestContent, ingestError) = try await client.callTool(
            name: "parse_and_ingest_document",
            arguments: ["filename": .string("notes.md"), "content": .string(docBody)])
        XCTAssertNotEqual(ingestError, true)
        let ingestText = Self.text(of: ingestContent)
        XCTAssertTrue(ingestText.contains("notes.md"), ingestText)
        XCTAssertTrue(ingestText.contains("已处理"), ingestText)

        // list_document：文档在列。
        let (listContent, listError) = try await client.callTool(
            name: "list_document", arguments: [:])
        XCTAssertNotEqual(listError, true)
        XCTAssertTrue(Self.text(of: listContent).contains("notes.md"))

        // hybrid_search_with_rerank：检索命中。
        let (searchContent, searchError) = try await client.callTool(
            name: "hybrid_search_with_rerank",
            arguments: ["query": .string("apple orchard"), "top_k": .int(3)])
        XCTAssertNotEqual(searchError, true)
        XCTAssertTrue(Self.text(of: searchContent).contains("apple"))

        // extract_metadata / get_document。
        let (metaContent, metaError) = try await client.callTool(
            name: "extract_metadata", arguments: ["doc_name": .string("notes.md")])
        XCTAssertNotEqual(metaError, true)
        XCTAssertTrue(Self.text(of: metaContent).contains("## 目录"))

        let (docContent, docError) = try await client.callTool(
            name: "get_document", arguments: ["doc_name": .string("notes.md")])
        XCTAssertNotEqual(docError, true)
        XCTAssertTrue(Self.text(of: docContent).contains("apple"))

        // delete_document 后 get_document 报错。
        let (delContent, delError) = try await client.callTool(
            name: "delete_document", arguments: ["source": .string("notes.md")])
        XCTAssertNotEqual(delError, true)
        XCTAssertTrue(Self.text(of: delContent).contains("已遗忘"))
        let (_, afterDelError) = try await client.callTool(
            name: "get_document", arguments: ["doc_name": .string("notes.md")])
        XCTAssertEqual(afterDelError, true, "删除后获取原文应报错")

        // 未知工具：官方客户端收到协议级 -32602。
        do {
            _ = try await client.callTool(name: "no_such_tool", arguments: [:])
            XCTFail("未知工具应抛错")
        } catch let error as MCP.MCPError {
            guard case .invalidParams = error else {
                return XCTFail("应返回 -32602，实际：\(error)")
            }
        }
    }

    private static func text(of content: [MCP.Tool.Content]) -> String {
        content.compactMap {
            if case .text(let text, _, _) = $0 { return text }
            return nil
        }.joined(separator: "\n")
    }

    // MARK: 原始 HTTP 协议行为（校验器 / 会话 / 通知）

    func testHTTPProtocolBehaviors() async throws {
        let engine = try makeEngine("kb-protocol")
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

        // 2. 未知方法 → -32601。
        let (_, unknownMethod) = try await postJSON([
            "jsonrpc": .string("2.0"),
            "id": .int(9),
            "method": .string("resources/list"),
        ], port: port, session: initSession)
        XCTAssertEqual(
            unknownMethod.objectValue?["error"]?.objectValue?["code"]?.intValue,
            -32601)

        // 3. 未知 session → 404。
        let (badSessionResponse, _) = try await exchange(
            JSONValue.object([
                "jsonrpc": .string("2.0"),
                "id": .int(10),
                "method": .string("tools/list"),
            ]).jsonString(), port: port, session: "not-a-session")
        XCTAssertEqual(badSessionResponse.statusCode, 404)

        // 4. 会话强制：无 session 的非 initialize 请求 → 400（规范）。
        let (noSessionResponse, noSessionText) = try await exchange(
            JSONValue.object([
                "jsonrpc": .string("2.0"),
                "id": .int(11),
                "method": .string("tools/list"),
            ]).jsonString(), port: port)
        XCTAssertEqual(noSessionResponse.statusCode, 400)
        XCTAssertTrue(noSessionText.contains("Mcp-Session-Id"))

        // 5. GET 无 session → 400（有 session 且 Accept SSE 才能建推送流）。
        let (getResponse, _) = try await exchange("", port: port, method: "GET")
        XCTAssertEqual(getResponse.statusCode, 400)

        // 6. MUST：Origin 防 DNS rebinding —— 非本机来源 403，本机放行。
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

        // 7. MUST：不支持的 MCP-Protocol-Version → 400；支持的版本放行。
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

        // 8. 未知工具 → JSON-RPC error -32602（协议级错误）。
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

        // 9. 多会话：第二个客户端可以并行 initialize（各自独立 session）。
        let (_, secondInit, secondSession) = try await initialize(port: port)
        XCTAssertEqual(
            secondInit.objectValue?["result"]?.objectValue?["protocolVersion"]?.stringValue,
            "2025-06-18")
        XCTAssertNotEqual(secondSession, initSession)
        let (secondListStatus, secondListValue) = try await postJSON([
            "jsonrpc": .string("2.0"),
            "id": .int(20),
            "method": .string("tools/list"),
        ], port: port, session: secondSession)
        XCTAssertEqual(secondListStatus, 200)
        XCTAssertNil(secondListValue.objectValue?["error"])

        // 10. DELETE 显式终止会话 → 之后同 session 请求 404。
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

    // MARK: stdio 共用工具层（InMemory 传输）

    func testToolsetOverInMemoryTransport() async throws {
        let engine = try makeEngine("kb-inmemory")
        defer { engine.close() }

        let toolset = RAGMCPToolset(engine: engine)
        let (clientTransport, serverTransport) =
            await MCP.InMemoryTransport.createConnectedPair()
        let server = await toolset.makeServer()
        try await server.start(transport: serverTransport)
        let client = MCP.Client(name: "inmemory-client", version: "1.0")
        defer { Task { await client.disconnect(); await server.stop() } }
        try await client.connect(transport: clientTransport)

        let (tools, _) = try await client.listTools()
        XCTAssertEqual(tools.count, 7)

        let docBody = String(repeating: "volcano geology report. ", count: 30)
        let (ingest, ingestError) = try await client.callTool(
            name: "parse_and_ingest_document",
            arguments: ["filename": .string("geo.md"), "content": .string(docBody)])
        XCTAssertNotEqual(ingestError, true)
        XCTAssertTrue(Self.text(of: ingest).contains("geo.md"))

        let (search, searchError) = try await client.callTool(
            name: "hybrid_search_with_rerank",
            arguments: ["query": .string("volcano geology")])
        XCTAssertNotEqual(searchError, true)
        XCTAssertTrue(Self.text(of: search).contains("volcano"))
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
