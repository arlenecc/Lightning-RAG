import XCTest
import Foundation
@testable import BaseAgentCore
@testable import ColBERTReader

/// 知识图谱多源抽取回归：ColBERT document 表、跨分区切片聚合、外部源替代本地 chunks。
final class KnowledgeGraphMultiSourceTests: XCTestCase {

    // MARK: - ColBERT document 表枚举

    func testColbertListAllChunkTexts() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("kg-colbert-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbPath = dir.appendingPathComponent("colbert.db").path
        let engine = try ColBERTReaderEngine(dbPath: dbPath, modelDir: dir.path)
        let db = try SQLiteDB(path: dbPath)
        // document(rowid, doc_id, title, source, content)
        try db.execute("INSERT INTO document(rowid,doc_id,title,source,content) VALUES(?,?,?,?,?)",
                       [.int(1), .text("doc#0000"), .text("第一章"), .text("a.md"),
                        .text("寄生虫改变宿主行为。")])
        try db.execute("INSERT INTO document(rowid,doc_id,title,source,content) VALUES(?,?,?,?,?)",
                       [.int(2), .text("doc#0001"), .text("第二章"), .text("b.md"),
                        .text("病毒驱动基因转移。")])
        let rows = try engine.listAllChunkTexts()
        XCTAssertEqual(rows.count, 2)
        XCTAssertEqual(rows.first?.id, "doc#0000")
        XCTAssertEqual(rows.first?.heading, "第一章")
        XCTAssertEqual(rows.last?.text, "病毒驱动基因转移。")
    }

    /// RAGEngine 在极速模式下应从 colbert.db（而非空 chunks 表）抽取图谱源切片。
    func testEngineSourceChunksInColbertMode() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("kg-engine-colbert-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = root.appendingPathComponent("kb").path
        config.colbertReaderEnabled = true
        let engine = RAGEngine(config: config, knowledgeBase: config.knowledgeBase,
                               embedding: StubKGEmbedding(), reranker: nil)
        // 建 colbert schema 并写一片正文（不加载模型）
        let colbertEngine = try ColBERTReaderEngine(dbPath: engine.colbertDBPath,
                                                    modelDir: root.path)
        _ = colbertEngine
        let db = try SQLiteDB(path: engine.colbertDBPath)
        try db.execute("INSERT INTO document(rowid,doc_id,title,source,content) VALUES(?,?,?,?,?)",
                       [.int(1), .text("x#0000"), .text(" parasite 章"), .text("自然/寄生虫星球.md"),
                        .text("寄生虫塑造宿主生态。")])
        let chunks = engine.knowledgeGraphSourceChunks()
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks.first?.id, "x#0000")
        XCTAssertEqual(chunks.first?.text, "寄生虫塑造宿主生态。")
        XCTAssertEqual(chunks.first?.heading, " parasite 章")
    }

    // MARK: - builder 外部源替代本地 chunks

    func testBuilderUsesExternalChunksWhenProvided() async throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("kg-external-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbPath = dir.appendingPathComponent("index.sqlite").path
        let store = try KnowledgeGraphStore(dbPath: dbPath, embedding: StubKGEmbedding())
        let graph = KnowledgeGraph(snapshotPath: dir.appendingPathComponent("kg.json").path,
                                   revisionProvider: { try? store.revision() })
        let llm = StubKGLLM()
        let builder = KnowledgeGraphBuilder(store: store, graph: graph, llm: llm,
                                            limits: KGExtractionLimits(),
                                            format: .delimited, maxConcurrency: 1)
        let external = [
            KnowledgeGraphStore.ChunkListItem(id: "zoneA|x", text: "外部切片正文甲。",
                                              source: "社会/a.md", heading: "章一"),
            KnowledgeGraphStore.ChunkListItem(id: "zoneB|y", text: "外部切片正文乙。",
                                              source: "自然/b.md", heading: "章二"),
        ]
        let stats = try await builder.build(extraChunkRefs: external)
        XCTAssertEqual(stats.chunksProcessed, 2, "只应抽取外部源切片")
        XCTAssertTrue(llm.seenText.contains("外部切片正文甲"))
        XCTAssertTrue(llm.seenText.contains("外部切片正文乙"))
    }
}

private final class StubKGEmbedding: EmbeddingProvider {
    let identifier = "stub-kg"
    let dimension = 4
    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map { _ in [1, 0, 0, 0] }
    }
    func embedQuery(_ text: String) async throws -> [Float] { [1, 0, 0, 0] }
}

private final class StubKGLLM: LLMService, @unchecked Sendable {
    let identifier = "stub-kg-llm"
    let supportsTools = false
    let contextWindow = 4096
    private let lock = NSLock()
    private var _seenText = ""
    var seenText: String { lock.lock(); defer { lock.unlock() }; return _seenText }

    func chatStream(messages: [ChatMessage], tools: [JSONValue],
                    temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let prompt = messages.last?.content ?? ""
            lock.lock(); _seenText += prompt; lock.unlock()
            // delimited 格式：entity<|#|>名字<|#|>type<|#|>desc
            continuation.yield(LLMStreamEvent(kind: .content,
                content: "entity<|#|>测试实体<|#|>concept<|#|>来自外部源\n"))
            continuation.yield(LLMStreamEvent(kind: .done, toolCalls: [],
                usage: [:], rawContent: prompt, truncated: false))
            continuation.finish()
        }
    }
}
