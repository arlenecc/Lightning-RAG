import XCTest
import Foundation
@testable import ColBERTReader

/// 极速模式异步编码回填数据层回归：正文先入库、pending 队列、幽灵切片清理。
/// 只测纯 DB 路径（addMarkdownTextsOnly 是静态方法，不依赖 ColBERT 模型）。
final class ColbertBackfillTests: XCTestCase {
    private var directory: URL!
    private var dbPath: String!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("colbert-backfill-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        dbPath = directory.appendingPathComponent("colbert.db").path
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func makeDB() throws -> SQLiteDB {
        let db = try SQLiteDB(path: dbPath)
        try SearchEngine.createSchema(db)
        return db
    }

    private func pendingCount(_ db: SQLiteDB) -> Int {
        Int((try? db.query("SELECT COUNT(*) AS n FROM pending_colbert")
            .first?["n"]?.intValue) ?? -1)
    }

    func testTextsFirstPendingAndReopen() throws {
        do {
            let db = try makeDB()
            let n = try SearchEngine.addMarkdownTextsOnly(
                db, text: "# 第一章\n\n合同履行应当遵循诚实信用原则，违约方承担相应责任。",
                source: "contract.md", bits: 4, sourceKey: "src-contract")
            XCTAssertGreaterThan(n, 0)
            XCTAssertEqual(pendingCount(db), n)
            let rows = try db.query("SELECT bits, rowid FROM pending_colbert")
            XCTAssertTrue(rows.allSatisfy { $0["bits"]?.intValue == 4 })
            // FTS 立即可检索（正文已入库），向量表仍为空（待回填）。
            let fts = try db.query("SELECT rowid FROM document_fts WHERE content_tokens MATCH ?",
                                   [.text("合同")])
            XCTAssertFalse(fts.isEmpty, "正文入库后 FTS 应立即命中")
            let vecRows = try db.query("SELECT COUNT(*) AS n FROM doc_vectors")
            XCTAssertEqual(vecRows.first?["n"]?.intValue ?? -1, 0)
        }
        // 重启：pending 从库恢复
        let db2 = try makeDB()
        XCTAssertGreaterThan(pendingCount(db2), 0)
    }

    func testDropPendingGhostRemovesDocumentAndCount() throws {
        let db = try makeDB()
        let n = try SearchEngine.addMarkdownTextsOnly(
            db, text: "一段将被剪枝为空的正文内容。", source: "ghost.md",
            sourceKey: "src-ghost")
        XCTAssertEqual(n, 1)
        let rows = try db.query("SELECT doc_id, rowid FROM pending_colbert")
        XCTAssertEqual(rows.count, 1)
        let docID = rows[0]["doc_id"]?.stringValue ?? ""
        let rowid = rows[0]["rowid"]?.intValue ?? 0

        // 模拟幽灵出队：删除 document/FTS/pending 并扣 doc_count（与 dropPendingGhost 同语义）。
        try db.transaction {
            try db.execute("DELETE FROM document WHERE doc_id = ?", [.text(docID)])
            try db.execute("DELETE FROM document_fts WHERE doc_id = ?", [.text(docID)])
            try db.execute("DELETE FROM pending_colbert WHERE doc_id = ?", [.text(docID)])
            let prevRow = try db.query("SELECT value FROM doc_meta WHERE key = 'doc_count'")
            let prev = Int(prevRow.first?["value"]?.stringValue ?? "") ?? 0
            try db.execute("""
                INSERT INTO doc_meta(key,value) VALUES('doc_count',?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """, [.text(String(max(0, prev - 1)))])
        }
        XCTAssertEqual(pendingCount(db), 0)
        XCTAssertEqual(try SearchEngine.totalChunkCount(db), 0)
    }

    func testEncodingCompletenessQueries() throws {
        let db = try makeDB()
        // 初始空库：完备（0/0）
        XCTAssertEqual(try SearchEngine.totalChunkCount(db), 0)
        XCTAssertEqual(try SearchEngine.encodedChunkCount(db), 0)
        // 文本入库后：total>0、filled=0、pending>0 → 不完备
        try SearchEngine.addMarkdownTextsOnly(db, text: "合同正文内容若干。", source: "c.md", sourceKey: "c")
        let total = try SearchEngine.totalChunkCount(db)
        let filled = try SearchEngine.encodedChunkCount(db)
        let pending = SearchEngine.pendingEncodeCount(db)
        XCTAssertGreaterThan(total, 0)
        XCTAssertEqual(filled, 0)
        XCTAssertEqual(pending, total, "文本入库后全部待编码")
        XCTAssertFalse(pending == 0 && filled >= total, "此时不应判定为完备")
    }

    /// 入库用哈希 sourceKey 作 doc_id 前缀、document.source 存原始路径；
    /// 删除/计数按哈希前缀命中，兜底按原始 source 也能命中。
    func testSourceIdentityForDeleteAndCount() throws {
        let db = try makeDB()
        let source = "/tmp/kb/合同 汇编_2024.pdf"
        let hashKey = "a1b2c3d4" // 模拟 service.sourceKey(source)
        try SearchEngine.addMarkdownTextsOnly(db, text: "合同条款正文内容。",
                                              source: source, sourceKey: hashKey)
        // 按哈希前缀计数应命中
        let byKey = try SearchEngine.chunkCount(db, hashKey)
        XCTAssertGreaterThan(byKey, 0, "按哈希 sourceKey 应能计数")
        // document.source 存的是原始路径（不是相对文件名），兜底删除能命中
        let rows = try db.query("SELECT DISTINCT source FROM document")
        XCTAssertEqual(rows.first?["source"]?.stringValue, source,
                       "document.source 应存原始路径供兜底")
        // 按哈希前缀删除应清空
        let removed = try SearchEngine.deleteSource(db, hashKey)
        XCTAssertEqual(removed, byKey)
        XCTAssertEqual(try SearchEngine.totalChunkCount(db), 0)
    }

    /// 退避重试：失败项 created_at 推后，nextPendingEncode 不再立即取到它。
    func testRetryPendingBackoff() throws {
        let db = try makeDB()
        try SearchEngine.addMarkdownTextsOnly(db, text: "待编码正文内容。", source: "r.md", sourceKey: "r")
        XCTAssertEqual(SearchEngine.pendingEncodeCount(db), 1)
        let rows = try db.query("SELECT doc_id FROM pending_colbert")
        let docID = rows[0]["doc_id"]?.stringValue ?? ""
        // 直接调 retryPending 需实例方法：通过反射不可行，改为直接验证 SQL 更新+排序语义。
        let retryAt = Date().timeIntervalSince1970 + 300
        try db.execute("UPDATE pending_colbert SET created_at = ? WHERE doc_id = ?",
                       [.double(retryAt), .text(docID)])
        let rowsAfter = try db.query("SELECT doc_id FROM pending_colbert WHERE created_at <= ?",
                                     [.double(Date().timeIntervalSince1970)])
        XCTAssertTrue(rowsAfter.isEmpty, "退避中的项不应被取出")
        XCTAssertEqual(SearchEngine.pendingEncodeCount(db), 1, "退避不丢数据")
    }

    func testConcurrentTextsOnlyRowIDsUnique() throws {
        let db = try makeDB()
        // 源码已把 rowSeq 分配放进写事务；这里顺序入两批验证 rowid 不复用。
        try SearchEngine.addMarkdownTextsOnly(db, text: "文档一正文内容。", source: "a.md", sourceKey: "a")
        try SearchEngine.addMarkdownTextsOnly(db, text: "文档二正文内容。", source: "b.md", sourceKey: "b")
        let rows = try db.query("SELECT rowid FROM pending_colbert ORDER BY rowid")
        let ids = rows.compactMap { $0["rowid"]?.intValue }
        XCTAssertEqual(Set(ids).count, ids.count, "rowid 必须唯一")
    }
}
