import Foundation
import Network

// MARK: - RAG MCP 服务（MCP 2025-06-18 Streamable HTTP）

/// RAG MCP 服务监听配置。`listenLAN=false` 只绑 127.0.0.1（默认），
/// `true` 绑 0.0.0.0 允许局域网访问。
public struct RAGMCPServerOptions: Sendable, Equatable {
    public var listenLAN: Bool
    public var port: Int

    public init(listenLAN: Bool = false, port: Int = 8080) {
        self.listenLAN = listenLAN
        self.port = port
    }
}

/// 把本机知识库的 RAG 能力（入库 / 删除 / 清单 / 目录 / 检索 / 原文透传）
/// 封装成 MCP 工具，通过 Streamable HTTP 对外提供，供其它大模型应用对接。
///
/// 协议要点（MCP 2025-06-18）：
/// - 单一 endpoint，`POST` JSON-RPC 消息；响应为 `application/json`（所有工具
///   都是同步计算，无 server→client 通知，不需要 SSE 流）；
/// - `initialize` 响应回 `Mcp-Session-Id` header；后续请求可选携带（未携带按
///   无状态处理，便于最小客户端对接）；未知 session 返回 404 要求重新握手；
/// - `GET` 不支持（无通知/订阅需求），返回 405；
/// - 通知消息返回 202 Accepted。
///
/// 所有 `tools/call` 在内部串行执行（一次一个），避免并发入库/删除与
/// `RAGEngine` 的 ingest 互斥标志互相踩踏。入库、检索完全复用
/// `RAGEngine` 既有流程，与 App 内功能共享同一个知识库。
public final class RAGMCPServer: @unchecked Sendable {
    public static let supportedProtocolVersions = ["2025-06-18", "2025-03-26"]
    public static let serverName = "lightning-rag-mcp"

    private let engine: RAGEngine
    private let options: RAGMCPServerOptions
    private let log: @Sendable (String) -> Void
    /// tools/call 串行闸门：入库与删除不能并发跑。
    private let gate = CallGate()
    private let stateLock = NSLock()
    private var listener: NWListener?
    private var runningState = false
    private var sessionIDs: Set<String> = []

    public init(engine: RAGEngine, options: RAGMCPServerOptions,
                log: @escaping @Sendable (String) -> Void = { _ in }) {
        self.engine = engine
        self.options = options
        self.log = log
    }

    public var isRunning: Bool {
        stateLock.lock(); defer { stateLock.unlock() }
        return runningState
    }

    /// 实际监听的端口（测试用 port 0 自动分配后由此读取）。
    public var boundPort: UInt16? {
        stateLock.lock(); defer { stateLock.unlock() }
        return listener?.port?.rawValue
    }

    /// Origin 校验（规范 MUST，防 DNS rebinding）：只放行本机来源。
    /// 非浏览器客户端不带 Origin（非必填），直接放行。
    static func isLocalOrigin(_ origin: String) -> Bool {
        guard let url = URL(string: origin),
              let host = url.host?.lowercased(), !host.isEmpty else { return false }
        if ["localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"].contains(host) {
            return true
        }
        let localName = Host.current().localizedName?.lowercased()
        return host == localName || host == "\(localName).local"
    }

    /// 启动监听。端口被占用等错误向上抛出。
    public func start() throws {
        stateLock.lock()
        if runningState {
            stateLock.unlock()
            return
        }
        stateLock.unlock()
        guard options.port == 0 || (1...65535).contains(options.port) else {
            throw AgentError.invalidArguments("RAG MCP 服务端口非法：\(options.port)")
        }
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        let host: NWEndpoint.Host = options.listenLAN
            ? .ipv4(IPv4Address.any)
            : .ipv4(IPv4Address("127.0.0.1")!)
        // port 0 = 系统自动分配（测试用）。
        params.requiredLocalEndpoint = .hostPort(
            host: host, port: NWEndpoint.Port(rawValue: UInt16(options.port))!)
        let listener = try NWListener(using: params)
        let queue = DispatchQueue(label: "base-agent.rag.mcp", qos: .userInitiated)
        listener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection, queue: queue)
        }
        listener.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                self.stateLock.lock()
                self.runningState = true
                self.stateLock.unlock()
                self.log("[rag-mcp] 服务已启动（监听 \(options.listenLAN ? "0.0.0.0" : "127.0.0.1"):\(options.port)）")
            case .failed(let error):
                self.log("[rag-mcp] 监听失败：\(error.localizedDescription)")
                self.stop()
            default:
                break
            }
        }
        stateLock.lock()
        self.listener = listener
        stateLock.unlock()
        listener.start(queue: queue)
    }

    /// 停止监听并关闭活跃连接。幂等。
    public func stop() {
        stateLock.lock()
        let listener = self.listener
        self.listener = nil
        self.runningState = false
        self.sessionIDs.removeAll()
        stateLock.unlock()
        listener?.cancel()
        log("[rag-mcp] 服务已停止")
    }

    // MARK: 连接处理（极简 HTTP/1.1：Content-Length + Connection: close）

    private static let maxBodyBytes = 64 * 1024 * 1024

    private func accept(_ connection: NWConnection, queue: DispatchQueue) {
        connection.stateUpdateHandler = { [weak self] state in
            if case .failed(let error) = state {
                self?.log("[rag-mcp] 连接异常：\(error.localizedDescription)")
            }
        }
        connection.start(queue: queue)
        receiveLoop(connection, buffer: Data())
    }

    private func receiveLoop(_ connection: NWConnection, buffer: Data) {
        guard buffer.count <= Self.maxBodyBytes + 1_048_576 else {
            respond(connection, status: 413,
                    body: mcpErrorJSON(.invalidRequest, "请求体超过上限").data(using: .utf8),
                    sessionID: nil)
            return
        }
        connection.receive(minimumIncompleteLength: 1,
                           maximumLength: 1 << 20) { [weak self] data, _, _, error in
            guard let self else { return }
            if let error {
                self.log("[rag-mcp] 接收中断：\(error.localizedDescription)")
                connection.cancel()
                return
            }
            guard let data else {
                connection.cancel()
                return
            }
            let merged = buffer + data
            if let request = HTTPRequest.parse(merged) {
                self.handle(request: request, connection: connection)
            } else {
                // 头部或 body 还没收全，继续读。
                self.receiveLoop(connection, buffer: merged)
            }
        }
    }

    private func handle(request: HTTPRequest, connection: NWConnection) {
        let sessionID = request.headers["mcp-session-id"]
        switch request.method {
        case "GET":
            // 无 server→client 推送需求：显式拒绝，符合规范（405）。
            respond(connection, status: 405, body: Data(), sessionID: nil)
        case "POST":
            // MUST（规范安全要求）：校验 Origin 防 DNS rebinding —— 浏览器型
            // 客户端会携带 Origin；非本机来源一律拒绝。非浏览器客户端不带
            // Origin，直接放行（Origin 非必填）。
            if let origin = request.headers["origin"], !origin.isEmpty,
               !Self.isLocalOrigin(origin) {
                respond(connection, status: 403,
                        body: mcpErrorJSON(.invalidRequest,
                                           "拒绝跨站来源：\(origin)").data(using: .utf8),
                        sessionID: nil)
                return
            }
            // MUST：客户端必须携带协商确定的 MCP-Protocol-Version 头；
            // 不支持的版本必须返回 400。未携带时按规范假定 2025-03-26（已支持）。
            if let version = request.headers["mcp-protocol-version"],
               !version.isEmpty,
               !Self.supportedProtocolVersions.contains(version) {
                respond(connection, status: 400,
                        body: mcpErrorJSON(.invalidRequest,
                                           "不支持的协议版本：\(version)").data(using: .utf8),
                        sessionID: nil)
                return
            }
            Task {
                let reply = await self.dispatch(request: request, sessionID: sessionID)
                self.respond(connection, status: reply.status,
                             body: reply.body.data(using: .utf8),
                             sessionID: reply.sessionID)
            }
        case "DELETE":
            // 客户端显式终止会话：移除 session 后确认（规范允许 server 接受
            // 终止或回 405；接受更干净）。
            if let sessionID, !sessionID.isEmpty {
                stateLock.lock()
                sessionIDs.remove(sessionID)
                stateLock.unlock()
            }
            respond(connection, status: 200, body: Data(), sessionID: nil)
        default:
            respond(connection, status: 405, body: Data(), sessionID: nil)
        }
    }

    private func respond(_ connection: NWConnection, status: Int,
                         body: Data?, sessionID: String?) {
        var head = "HTTP/1.1 \(status) \(HTTPRequest.statusText(status))\r\n"
        if let body, !body.isEmpty {
            head += "Content-Type: application/json\r\n"
            head += "Content-Length: \(body.count)\r\n"
        } else {
            head += "Content-Length: 0\r\n"
        }
        if let sessionID {
            head += "Mcp-Session-Id: \(sessionID)\r\n"
        }
        head += "Connection: close\r\n\r\n"
        var payload = head.data(using: .utf8) ?? Data()
        if let body, !body.isEmpty { payload += body }
        connection.send(content: payload,
                        completion: .contentProcessed { _ in connection.cancel() })
    }

    // MARK: JSON-RPC 分发

    private struct Reply {
        var status: Int
        var body: String
        var sessionID: String?
    }

    private func dispatch(request: HTTPRequest, sessionID: String?) async -> Reply {
        guard let message = JSONValue.parse(request.body), let object = message.objectValue else {
            return Reply(status: 400,
                         body: mcpErrorJSON(.parseError, "请求不是合法 JSON"),
                         sessionID: nil)
        }
        let id = object["id"]
        let method = object["method"]?.stringValue ?? ""
        let params = object["params"]?.objectValue ?? [:]

        // 会话强制（规范：颁发了 session 的 server，除 initialize 外必须携带
        // 有效 `Mcp-Session-Id`；缺失 → 400，未知 → 404 要求重新 initialize）。
        if method != "initialize" {
            guard let sessionID, !sessionID.isEmpty else {
                return Reply(status: 400,
                             body: mcpErrorJSON(.invalidRequest,
                                                "缺少 Mcp-Session-Id，请先 initialize"),
                             sessionID: nil)
            }
            stateLock.lock()
            let known = sessionIDs.contains(sessionID)
            stateLock.unlock()
            if !known {
                return Reply(status: 404,
                             body: mcpErrorJSON(.invalidRequest,
                                                "未知会话，请重新 initialize"),
                             sessionID: nil)
            }
        }

        // 通知（无 id）：不需要响应体。
        guard id != nil, id!.isNull == false else {
            if method == "notifications/initialized" || method.hasPrefix("notifications/") {
                return Reply(status: 202, body: "", sessionID: nil)
            }
            return Reply(status: 200,
                         body: mcpErrorJSON(.invalidRequest,
                                            "通知消息不应携带响应", id: nil),
                         sessionID: nil)
        }

        switch method {
        case "initialize":
            let clientVersion = params["protocolVersion"]?.stringValue ?? ""
            let negotiated = Self.supportedProtocolVersions.contains(clientVersion)
                ? clientVersion
                : Self.supportedProtocolVersions[0]
            let newSession = UUID().uuidString
            stateLock.lock()
            sessionIDs.insert(newSession)
            stateLock.unlock()
            let result: JSONValue = .object([
                "protocolVersion": .string(negotiated),
                "capabilities": .object(["tools": .object([:])]),
                "serverInfo": .object([
                    "name": .string(Self.serverName),
                    "version": .string(AppVersion.current),
                ]),
            ])
            return Reply(status: 200,
                         body: jsonRPCResult(id: id!, result: result),
                         sessionID: newSession)
        case "ping":
            return Reply(status: 200, body: jsonRPCResult(id: id!, result: .object([:])),
                         sessionID: sessionID)
        case "tools/list":
            return Reply(status: 200,
                         body: jsonRPCResult(id: id!, result: .object([
                             "tools": .array(Self.toolSchemas()),
                         ])),
                         sessionID: sessionID)
        case "tools/call":
            let name = params["name"]?.stringValue ?? ""
            let args = params["arguments"]?.objectValue ?? [:]
            // 未知工具是协议级错误：返回 JSON-RPC error -32602（规范约定），
            // 而不是 isError 结果。工具执行失败才是 isError: true。
            guard Self.toolSchemas().first(where: {
                $0.objectValue?["name"]?.stringValue == name
            }) != nil else {
                return Reply(status: 200,
                             body: mcpErrorJSON(.invalidParams,
                                                "未知工具：\(name)", id: id),
                             sessionID: sessionID)
            }
            let outcome = await gate.run { await self.callTool(name: name, args: args) }
            switch outcome {
            case .success(let text):
                let result: JSONValue = .object([
                    "content": .array([.object(["type": .string("text"),
                                                "text": .string(text)])]),
                    "isError": .bool(false),
                ])
                return Reply(status: 200,
                             body: jsonRPCResult(id: id!, result: result),
                             sessionID: sessionID)
            case .failure(let message):
                let result: JSONValue = .object([
                    "content": .array([.object(["type": .string("text"),
                                                "text": .string(message)])]),
                    "isError": .bool(true),
                ])
                return Reply(status: 200,
                             body: jsonRPCResult(id: id!, result: result),
                             sessionID: sessionID)
            }
        default:
            return Reply(status: 200,
                         body: mcpErrorJSON(.methodNotFound, "未知方法：\(method)", id: id),
                         sessionID: sessionID)
        }
    }

    // MARK: 工具实现

    private func callTool(name: String, args: [String: JSONValue]) async -> RAGMCPToolOutcome {
        switch name {
        case "parse_and_ingest_document":
            return await ingestDocument(args)
        case "delete_document", "forget_context":
            return deleteDocument(args)
        case "list_document":
            return listDocuments(args)
        case "extract_metadata":
            return extractMetadata(args)
        case "hybrid_search_with_rerank":
            return await hybridSearch(args)
        case "get_document":
            return getDocument(args)
        default:
            return .failure("未知工具：\(name)")
        }
    }

    /// `parse_and_ingest_document`：把传入文档落到知识库目录，然后走既有
    /// `ingest()` 增量流程（解析 → 提取 → 清洗 → 切片 → 向量化 → 入库）。
    private func ingestDocument(_ args: [String: JSONValue]) async -> RAGMCPToolOutcome {
        let kb = engine.knowledgeBase
        let fm = FileManager.default
        do {
            try fm.createDirectory(atPath: kb, withIntermediateDirectories: true)
        } catch {
            return .failure("知识库目录不可写（\(kb)）：\(error.localizedDescription)")
        }

        let sourcePath: String
        if let path = args["path"]?.stringValue, !path.isEmpty {
            let standardized = (path as NSString).standardizingPath
            guard fm.fileExists(atPath: standardized) else {
                return .failure("文件不存在：\(standardized)")
            }
            let ext = (standardized as NSString).pathExtension.lowercased()
            guard RAGEngine.supportedExtensions.contains(ext) else {
                return .failure("不支持的文件类型 .\(ext)；支持："
                    + RAGEngine.supportedExtensions.sorted().joined(separator: ", "))
            }
            if (standardized as NSString).deletingPathPrefixIfMatches(kb) {
                // 已在知识库目录内：直接增量入库。
                sourcePath = standardized
            } else {
                let target = uniqueDestination(for: standardized, in: kb)
                do {
                    try fm.copyItem(atPath: standardized, toPath: target)
                } catch {
                    return .failure("复制文件到知识库失败：\(error.localizedDescription)")
                }
                sourcePath = target
            }
        } else if let filename = args["filename"]?.stringValue, !filename.isEmpty {
            let safeName = (filename as NSString).lastPathComponent
            guard !safeName.isEmpty, safeName != "." , safeName != ".." else {
                return .failure("filename 非法：\(filename)")
            }
            let ext = (safeName as NSString).pathExtension.lowercased()
            guard RAGEngine.supportedExtensions.contains(ext) else {
                return .failure("不支持的文件类型 .\(ext)；支持："
                    + RAGEngine.supportedExtensions.sorted().joined(separator: ", "))
            }
            let target = (kb as NSString).appendingPathComponent(safeName)
            let content: Data
            if let base64 = args["content_base64"]?.stringValue,
               let decoded = Data(base64Encoded: base64) {
                content = decoded
            } else if let text = args["content"]?.stringValue {
                content = Data(text.utf8)
            } else {
                return .failure("缺少文档内容：请提供 content_base64 或 content")
            }
            do {
                try content.write(to: URL(fileURLWithPath: target), options: .atomic)
            } catch {
                return .failure("写入知识库失败：\(error.localizedDescription)")
            }
            sourcePath = target
        } else {
            return .failure("缺少文档：请提供 path（本机路径）或 filename + content_base64/content")
        }

        let stats = await engine.ingest(force: false)
        let docName = ((sourcePath as NSString).lastPathComponent)
        return .success("""
        入库完成：\(docName)
        \(stats.summary)
        （来源：\(sourcePath)）
        """)
    }

    /// 同名冲突时生成 `name-1.ext`、`name-2.ext` … 避免覆盖已有文档。
    private func uniqueDestination(for path: String, in kb: String) -> String {
        let fm = FileManager.default
        let name = (path as NSString).lastPathComponent
        let ext = (name as NSString).pathExtension
        let stem = ext.isEmpty ? name : (name as NSString).deletingPathExtension
        var candidate = (kb as NSString).appendingPathComponent(name)
        var counter = 1
        while fm.fileExists(atPath: candidate) {
            let suffixed = ext.isEmpty ? "\(stem)-\(counter)" : "\(stem)-\(counter).\(ext)"
            candidate = (kb as NSString).appendingPathComponent(suffixed)
            counter += 1
        }
        return candidate
    }

    private func deleteDocument(_ args: [String: JSONValue]) -> RAGMCPToolOutcome {
        guard let source = args["source"]?.stringValue ?? args["doc_name"]?.stringValue,
              !source.isEmpty else {
            return .failure("缺少参数 source（文档名或路径）")
        }
        do {
            let outcome = try engine.deleteDocument(source)
            switch outcome {
            case .deleted(let chunks, let docName):
                return .success("已遗忘文档：\(docName)（删除 \(chunks) 个切片、目录与原文缓存）")
            case .ambiguous(let candidates):
                return .failure("匹配到多个文档，请用更精确的名称重试："
                    + candidates.prefix(10).joined(separator: "、"))
            case .notFound:
                return .failure("未找到文档：\(source)")
            }
        } catch {
            return .failure("删除失败：\(error.localizedDescription)")
        }
    }

    private func listDocuments(_ args: [String: JSONValue]) -> RAGMCPToolOutcome {
        let limit = max(1, min(1000, args["limit"]?.intValue ?? 200))
        let docs = engine.listIngestedDocuments(limit: limit)
        if docs.isEmpty {
            return .success("知识库当前为空，尚无已入库文档。")
        }
        var lines: [String] = ["| 文档 | 切片数 | 大小 | 备注 |", "|---|---|---|---|"]
        for doc in docs {
            var note = ""
            if let dup = doc.duplicateOf { note = "重复副本（代表文件：\(dup)）" }
            else if doc.discarded == true { note = "正文过短已丢弃" }
            let chunks = doc.chunkCount.map(String.init) ?? "-"
            let size = doc.sizeBytes.map {
                ByteCountFormatter.string(fromByteCount: $0, countStyle: .file)
            } ?? "-"
            lines.append("| \(doc.docName) | \(chunks) | \(size) | \(note) |")
        }
        lines.append("")
        lines.append("共 \(docs.count) 个文档（上限 \(limit)，如需更多请增大 limit）。")
        return .success(lines.joined(separator: "\n"))
    }

    private func extractMetadata(_ args: [String: JSONValue]) -> RAGMCPToolOutcome {
        guard let docName = args["doc_name"]?.stringValue ?? args["source"]?.stringValue,
              !docName.isEmpty else {
            return .failure("缺少参数 doc_name（文档名或路径）")
        }
        guard let outline = engine.getEnrichedOutline(docName: docName) else {
            return .failure("未找到文档：\(docName)")
        }
        var md: [String] = ["# \(outline.record.docName)", ""]
        md.append("## 目录")
        md.append(outline.repairedTOC)
        if let book = outline.bookSummary, !book.summary.isEmpty {
            md.append("")
            md.append("## 全书摘要")
            md.append(book.summary)
        }
        if !outline.chapterSummaries.isEmpty {
            md.append("")
            md.append("## 章节摘要")
            for chapter in outline.chapterSummaries.prefix(200) {
                let title = chapter.title.isEmpty ? chapter.heading : chapter.title
                md.append("- **\(title)**：\(chapter.summary)")
            }
        }
        return .success(md.joined(separator: "\n"))
    }

    private func hybridSearch(_ args: [String: JSONValue]) async -> RAGMCPToolOutcome {
        guard let query = args["query"]?.stringValue, !query.isEmpty else {
            return .failure("缺少参数 query（检索语句）")
        }
        let topK = max(1, min(20, args["top_k"]?.intValue ?? 5))
        let source = args["source"]?.stringValue
        do {
            let hits = try await engine.search(query: query, topK: topK, source: source)
            if hits.isEmpty {
                return .success("知识库中未检索到与「\(query)」相关的内容。")
            }
            var md: [String] = ["检索到 \(hits.count) 条结果：", ""]
            for (index, hit) in hits.enumerated() {
                md.append("### [\(index + 1)] \(hit.source)（chunk \(hit.chunkIndex)，score \(String(format: "%.4f", hit.score))）")
                if !hit.heading.isEmpty {
                    md.append("标题：\(hit.heading)")
                }
                md.append(hit.text)
                md.append("")
            }
            return .success(md.joined(separator: "\n"))
        } catch {
            return .failure("检索失败：\(error.localizedDescription)")
        }
    }

    private func getDocument(_ args: [String: JSONValue]) -> RAGMCPToolOutcome {
        guard let docName = args["doc_name"]?.stringValue ?? args["source"]?.stringValue,
              !docName.isEmpty else {
            return .failure("缺少参数 doc_name（文档名或路径）")
        }
        let maxChars = max(1_000, min(400_000, args["max_chars"]?.intValue ?? 100_000))
        guard let markdown = engine.readMarkdown(for: docName) else {
            return .failure("未找到文档或解析缓存缺失：\(docName)")
        }
        if markdown.count > maxChars {
            return .success(String(markdown.prefix(maxChars))
                + "\n\n[已截断：全文 \(markdown.count) 字符，仅返回前 \(maxChars)；"
                + "可用 max_chars 参数调整或用 extract_metadata 获取目录概览]")
        }
        return .success(markdown)
    }

    // MARK: 工具 schema

    private static func toolSchemas() -> [JSONValue] {
        func schema(_ name: String, _ description: String,
                    _ properties: [String: JSONValue],
                    required: [String] = []) -> JSONValue {
            .object([
                "name": .string(name),
                "description": .string(description),
                "inputSchema": .object([
                    "type": .string("object"),
                    "properties": .object(properties),
                    "required": .array(required.map(JSONValue.string)),
                ]),
            ])
        }
        return [
            schema(
                "parse_and_ingest_document",
                "解析文档并入库到 RAG 知识库：解析 → 提取 → 清洗 → 切片 → 向量化 → 入库。"
                + "三选一：本机路径 path；或 filename + content_base64（二进制文件，如 PDF/EPUB）；"
                + "或 filename + content（纯文本）。返回入库统计。",
                [
                    "path": .object(["type": .string("string"),
                                     "description": .string("本机文档绝对路径（可选）")]),
                    "filename": .object(["type": .string("string"),
                                         "description": .string("文档文件名（与内容二选一成对提供）")]),
                    "content_base64": .object(["type": .string("string"),
                                               "description": .string("文档内容的 base64 编码")]),
                    "content": .object(["type": .string("string"),
                                        "description": .string("纯文本文档内容")]),
                ]),
            schema(
                "delete_document",
                "知识遗忘/删除：按文档名或路径删除该文档的全部切片向量、目录记录"
                + "与解析后的原始文本缓存。",
                ["source": .object(["type": .string("string"),
                                    "description": .string("文档名、相对路径或完整路径")])],
                required: ["source"]),
            schema(
                "forget_context",
                "知识遗忘：与 delete_document 相同的能力，语义化别名——"
                + "删除指定文档在知识库中的全部记忆（向量 + 原文）。",
                ["source": .object(["type": .string("string"),
                                    "description": .string("文档名、相对路径或完整路径")])],
                required: ["source"]),
            schema(
                "list_document",
                "列出知识库中已入库的文档清单（名称、切片数、大小、重复副本标记）。",
                ["limit": .object(["type": .string("integer"),
                                   "description": .string("最多返回条数，默认 200，上限 1000")])]),
            schema(
                "extract_metadata",
                "智能元数据提取：返回指定文档的章节目录与摘要（markdown 格式，"
                + "含修复后的目录、全书摘要、各章节要点）。",
                ["doc_name": .object(["type": .string("string"),
                                      "description": .string("文档名或路径")])],
                required: ["doc_name"]),
            schema(
                "hybrid_search_with_rerank",
                "混合检索与重排序：对知识库执行 FTS5 + 向量混合检索（RRF 融合），"
                + "按配置执行重排序，返回最相关的切片。",
                [
                    "query": .object(["type": .string("string"),
                                      "description": .string("检索查询语句")]),
                    "top_k": .object(["type": .string("integer"),
                                      "description": .string("返回条数，默认 5，上限 20")]),
                    "source": .object(["type": .string("string"),
                                       "description": .string("可选：限定文档名/路径子串")]),
                ],
                required: ["query"]),
            schema(
                "get_document",
                "原始文件透传：返回文档经解析、提取、清洗之后形成的 markdown 原文。",
                [
                    "doc_name": .object(["type": .string("string"),
                                         "description": .string("文档名或路径")]),
                    "max_chars": .object(["type": .string("integer"),
                                          "description": .string("最多返回字符数，默认 100000")]),
                ],
                required: ["doc_name"]),
        ]
    }

    // MARK: JSON-RPC 辅助

    private enum MCPErrorCode: Int {
        case parseError = -32700
        case invalidRequest = -32600
        case methodNotFound = -32601
        case invalidParams = -32602
    }

    private func jsonRPCResult(id: JSONValue, result: JSONValue) -> String {
        JSONValue.object([
            "jsonrpc": .string("2.0"),
            "id": id,
            "result": result,
        ]).jsonString()
    }

    private func mcpErrorJSON(_ code: MCPErrorCode, _ message: String,
                              id: JSONValue? = nil) -> String {
        JSONValue.object([
            "jsonrpc": .string("2.0"),
            "id": id ?? .null,
            "error": .object([
                "code": .int(code.rawValue),
                "message": .string(message),
            ]),
        ]).jsonString()
    }
}

/// 单个工具调用的结果：成功文本 / 失败原因（进入 MCP content，isError 区分）。
enum RAGMCPToolOutcome {
    case success(String)
    case failure(String)
}

/// 串行执行 tools/call：入库与删除不允许并发。
private actor CallGate {
    func run<T: Sendable>(_ body: @Sendable () async throws -> T) async rethrows -> T {
        try await body()
    }
}

// MARK: - 极简 HTTP/1.1 请求解析

struct HTTPRequest {
    var method: String
    var path: String
    var headers: [String: String]   // 全小写 key
    var body: String

    /// 解析完整请求（头部 + 按 Content-Length 的 body）；数据不完整返回 nil。
    static func parse(_ data: Data) -> HTTPRequest? {
        guard let text = String(data: data, encoding: .utf8),
              let headerRange = text.range(of: "\r\n\r\n") else { return nil }
        let head = String(text[..<headerRange.lowerBound])
        var lines = head.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else { return nil }
        lines.removeFirst()
        let parts = requestLine.split(separator: " ")
        guard parts.count >= 3 else { return nil }
        var headers: [String: String] = [:]
        for line in lines {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let key = String(line[..<colon]).trimmingCharacters(in: .whitespaces)
                .lowercased()
            let value = String(line[line.index(after: colon)...])
                .trimmingCharacters(in: .whitespaces)
            headers[key] = value
        }
        let contentLength = headers["content-length"].flatMap { Int($0) } ?? 0
        let remaining = String(text[headerRange.upperBound...])
        guard remaining.utf8.count >= contentLength else { return nil }
        // 按字节截取（Content-Length 是字节数；中文多字节字符不能按 Character 数）。
        let body = String(decoding: remaining.utf8.prefix(contentLength), as: UTF8.self)
        return HTTPRequest(method: String(parts[0]).uppercased(),
                           path: String(parts[1]),
                           headers: headers,
                           body: body)
    }

    static func statusText(_ code: Int) -> String {
        switch code {
        case 200: return "OK"
        case 202: return "Accepted"
        case 400: return "Bad Request"
        case 404: return "Not Found"
        case 405: return "Method Not Allowed"
        case 413: return "Payload Too Large"
        default: return "Unknown"
        }
    }
}

// MARK: - 小工具

extension NSString {
    /// 判断 path 是否位于 root 内（标准路径前缀比对）。
    fileprivate func deletingPathPrefixIfMatches(_ root: String) -> Bool {
        let standard = standardizingPath
        let rootStandard = (root as NSString).standardizingPath
        return standard.hasPrefix(rootStandard)
    }
}

/// App 版本号（Info.plist 优先，回退脚本维护的发布版本）。
enum AppVersion {
    static var current: String = "1.2.17"
}
