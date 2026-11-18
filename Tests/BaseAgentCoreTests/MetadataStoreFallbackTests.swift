import XCTest
import Foundation
@testable import BaseAgentCore

/// 方案 A 回归：检索阅读模式下目录/章节摘要/全书摘要统一存 index.sqlite，
/// 读写都走不依赖 embedding 的 metadata store，切回普通模式仍读同一份库。
final class MetadataStoreFallbackTests: XCTestCase {
    private var root: URL!
    private var kb: URL!
    private var config: AgentConfig!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("metadata-store-\(UUID().uuidString)")
        kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.colbertReaderEnabled = true
        self.config = config
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    /// 首次同步时 index.sqlite 尚不存在：openForMetadata 必须能建库并登记目录。
    /// 旧实现用只读连接打开，会因 SQLITE_OPEN_READONLY 无法建库而失败。
    func testOpenForMetadataCreatesMissingDBAndWritesDigest() throws {
        let dbPath = root.appendingPathComponent("rag/index.sqlite").path
        XCTAssertFalse(FileManager.default.fileExists(atPath: dbPath))

        let store = try VectorStore.openForMetadata(dbPath: dbPath)
        defer { store.close() }
        XCTAssertTrue(FileManager.default.fileExists(atPath: dbPath))

        let mdPath = kb.appendingPathComponent("book.md").path
        try store.upsertDocument(docName: "book.md", digest: "# 第一章 概述",
                                 chapters: "[{\"heading\":\"第一章 概述\"}]",
                                 mdPath: mdPath, contentHash: "hash-1")
        let record = try XCTUnwrap(store.getDocumentDigest(docName: "book.md"))
        XCTAssertEqual(record.digest, "# 第一章 概述")
        XCTAssertEqual(record.mdPath, mdPath)
        XCTAssertFalse(record.docID.isEmpty)
    }

    /// openForMetadata 必须继承库内已存的维度/元素类型/指纹，
    /// 否则元数据连接会误判配置变化、刷 rebuild 告警。
    func testOpenForMetadataInheritsStoredConfig() throws {
        let dbPath = root.appendingPathComponent("rag/index.sqlite").path
        let writer = try VectorStore(dbPath: dbPath, dimension: 32,
                                     enableSQLiteVec: false, elementType: .fp16,
                                     indexFingerprint: "unit-test-fp16-32")
        writer.close()

        let reader = try VectorStore.openForMetadata(dbPath: dbPath)
        defer { reader.close() }
        XCTAssertEqual(reader.dimension, 32)
        XCTAssertEqual(reader.elementType, .fp16)
        XCTAssertEqual(reader.indexFingerprint, "unit-test-fp16-32")
        XCTAssertNil(reader.rebuildReason)
    }

    /// colbert 模式下 rag_outline 的读取路径直接命中 documents/summary 表，
    /// 不走 manifest 兜底（兜底记录 docID 为空、无章节摘要）。
    func testColbertModeOutlineReadsFromMetadataStore() throws {
        let engine = RAGEngine(config: config, knowledgeBase: kb.path)
        XCTAssertTrue(config.colbertReaderEnabled)

        let mdPath = kb.appendingPathComponent("book.md").path
        let writer = try VectorStore.openForMetadata(dbPath: engine.indexDBPath)
        try writer.upsertDocument(docName: "book.md", digest: "# 第一章 概述",
                                  chapters: "[]", mdPath: mdPath,
                                  contentHash: "hash-1")
        let stored = try XCTUnwrap(writer.getDocumentDigest(docName: "book.md"))
        try writer.upsertChapterSummary(ChapterSummaryRecord(
            docID: stored.docID, sectionIndex: 0, heading: "第一章 概述",
            title: "", level: 1, summary: "本章讲概述。", contentHash: "sec-1"))
        writer.close()

        let record = engine.getDocumentOutline(docName: "book.md")
        let resolved = try XCTUnwrap(record)
        XCTAssertFalse(resolved.docID.isEmpty, "应命中 documents 表而不是 manifest 兜底")
        XCTAssertEqual(resolved.digest, "# 第一章 概述")

        let enriched = engine.getEnrichedOutline(for: resolved)
        let outline = try XCTUnwrap(enriched)
        XCTAssertEqual(outline.chapterSummaries.first?.summary, "本章讲概述。")

        let (names, total) = engine.listDocuments(limit: 10)
        XCTAssertEqual(names, ["book.md"])
        XCTAssertEqual(total, 1)
    }
}
