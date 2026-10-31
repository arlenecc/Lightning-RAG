import XCTest
@testable import BaseAgentCore

final class VectorStoreBinaryTests: XCTestCase {
    private var directory: URL!
    private let dimension = 16
    private let binaryDimension = 16

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-vector-bin-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func dbPath(_ name: String) -> String {
        directory.appendingPathComponent("\(name).sqlite").path
    }

    private func oneHot(_ index: Int) -> [Float] {
        var vector = [Float](repeating: 0, count: dimension)
        vector[index] = 1
        return vector
    }

    private func chunks(_ count: Int) -> [Chunk] {
        (0..<count).map { Chunk(text: "doc \($0)", source: "s\($0).md", chunkIndex: 0) }
    }

    /// 16 bit 签名固定 2 字节。
    private func signature(_ value: UInt8) -> Data {
        Data([value, value &+ 1])
    }

    private func binaryTableSQL(_ store: VectorStore) throws -> String? {
        try store.db.scalar(
            "SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
            [.text(VectorStore.binaryVectorTable)]).stringValue
    }

    private func meta(_ store: VectorStore, _ key: String) throws -> String? {
        try store.db.scalar("SELECT value FROM meta WHERE key = ?", [.text(key)]).stringValue
    }

    func testCreatesBinaryTableForSupportedConfiguration() throws {
        let store = try VectorStore(
            dbPath: dbPath("supported"), dimension: dimension,
            elementType: .int8, binaryDimension: binaryDimension,
            binarySource: "sig-native", indexFingerprint: "fp")
        let status = store.binaryIndexStatus()
        XCTAssertTrue(status.storageEnabled)
        XCTAssertEqual(status.dimension, binaryDimension)
        XCTAssertEqual(status.rows, 0)
        XCTAssertEqual(status.chunkRows, 0)
        XCTAssertEqual(status.coverage, 0)
        XCTAssertFalse(status.isComplete)

        let sql = try XCTUnwrap(binaryTableSQL(store))
        XCTAssertTrue(sql.contains("USING vec0"))
        XCTAssertTrue(sql.contains("bit[16]"))
        XCTAssertEqual(try meta(store, VectorStore.binaryDimensionKey), "16")
        XCTAssertEqual(try meta(store, VectorStore.binarySourceKey), "sig-native")
        XCTAssertEqual(try meta(store, VectorStore.binaryQuantizerVersionKey), "1")
        XCTAssertEqual(try meta(store, VectorStore.binaryRowsKey), "0")
        store.close()
    }

    func testDisablesBinaryStorageWithoutProviderOrSQLiteVec() throws {
        let noProvider = try VectorStore(
            dbPath: dbPath("no-provider"), dimension: dimension,
            elementType: .int8, binaryDimension: 0, binarySource: "",
            indexFingerprint: "no-provider")
        let noProviderStatus = noProvider.binaryIndexStatus()
        XCTAssertFalse(noProviderStatus.storageEnabled)
        XCTAssertTrue(noProviderStatus.reason?.contains("不支持") == true,
                      noProviderStatus.reason ?? "")
        XCTAssertNil(try binaryTableSQL(noProvider), "不支持时不应创建 bin 表")
        noProvider.close()

        let noVec = try VectorStore(
            dbPath: dbPath("no-vec"), dimension: dimension,
            enableSQLiteVec: false, elementType: .int8,
            binaryDimension: binaryDimension, binarySource: "sig",
            indexFingerprint: "no-vec")
        let noVecStatus = noVec.binaryIndexStatus()
        XCTAssertFalse(noVecStatus.storageEnabled)
        XCTAssertTrue(noVecStatus.reason?.contains("sqlite-vec") == true,
                      noVecStatus.reason ?? "")
        XCTAssertNil(try binaryTableSQL(noVec))
        noVec.close()

        let binPrimary = try VectorStore(
            dbPath: dbPath("bin-primary"), dimension: dimension,
            elementType: .bin, binaryDimension: binaryDimension,
            binarySource: "sig", indexFingerprint: "bin-primary")
        let binStatus = binPrimary.binaryIndexStatus()
        XCTAssertFalse(binStatus.storageEnabled)
        XCTAssertTrue(binStatus.reason?.contains("bin 主存储") == true,
                      binStatus.reason ?? "")
        XCTAssertNil(try binaryTableSQL(binPrimary))
        binPrimary.close()
    }

    /// 维度或来源变化时，bin 表必须 DROP 并重建为空表，不能复用旧签名。
    func testMetaMismatchDropsBinaryRowsAndRecreatesEmptyTable() throws {
        let path = dbPath("meta-mismatch")
        do {
            let store = try VectorStore(
                dbPath: path, dimension: dimension, elementType: .int8,
                binaryDimension: binaryDimension, binarySource: "source-a",
                indexFingerprint: "fp")
            _ = try store.add(chunks: chunks(2), vectors: [oneHot(0), oneHot(1)])
            try store.db.execute(
                "INSERT INTO \(VectorStore.binaryVectorTable)(rowid, embedding) "
                + "VALUES(1, vec_bit(?))", [.blob(signature(0x0F))])
            try store.db.execute(
                "UPDATE meta SET value = '1' WHERE key = ?",
                [.text(VectorStore.binaryRowsKey)])
            store.close()
        }

        let reopened = try VectorStore(
            dbPath: path, dimension: dimension, elementType: .int8,
            binaryDimension: 24, binarySource: "source-b",
            indexFingerprint: "fp")
        let status = reopened.binaryIndexStatus()
        XCTAssertTrue(status.storageEnabled)
        XCTAssertEqual(status.rows, 0)
        XCTAssertEqual(status.chunkRows, 2)
        XCTAssertEqual(status.coverage, 0)
        XCTAssertFalse(status.isComplete)
        let sql = try XCTUnwrap(binaryTableSQL(reopened))
        XCTAssertTrue(sql.contains("bit[24]"), sql)
        XCTAssertEqual(
            try reopened.db.scalar(
                "SELECT COUNT(*) FROM \(VectorStore.binaryVectorTable)").intValue, 0)
        XCTAssertEqual(try meta(reopened, VectorStore.binarySourceKey), "source-b")
        XCTAssertEqual(try meta(reopened, VectorStore.binaryDimensionKey), "24")
        reopened.close()
    }

    /// meta 一致但表被删掉时，重开应补建空表而不是报错。
    func testMissingBinaryTableIsRecreatedWhenMetaMatches() throws {
        let path = dbPath("missing-table")
        do {
            let store = try VectorStore(
                dbPath: path, dimension: dimension, elementType: .int8,
                binaryDimension: binaryDimension, binarySource: "sig",
                indexFingerprint: "fp")
            try store.db.execute("DROP TABLE \(VectorStore.binaryVectorTable)")
            store.close()
        }

        let reopened = try VectorStore(
            dbPath: path, dimension: dimension, elementType: .int8,
            binaryDimension: binaryDimension, binarySource: "sig",
            indexFingerprint: "fp")
        let status = reopened.binaryIndexStatus()
        XCTAssertTrue(status.storageEnabled)
        XCTAssertEqual(status.rows, 0)
        XCTAssertNotNil(try binaryTableSQL(reopened))
        reopened.close()
    }

    /// 覆盖率只读计数器：rows == chunkRows 才算完整。这里用 meta + 手写行
    /// 模拟 Task 8 写入后的状态，验证重开能恢复计数器。
    func testStatusUsesCountersAndBecomesCompleteWhenRowsMatchChunks() throws {
        let path = dbPath("coverage")
        do {
            let store = try VectorStore(
                dbPath: path, dimension: dimension, elementType: .int8,
                binaryDimension: binaryDimension, binarySource: "sig",
                indexFingerprint: "fp")
            _ = try store.add(chunks: chunks(2), vectors: [oneHot(0), oneHot(1)])
            let partial = store.binaryIndexStatus()
            XCTAssertEqual(partial.rows, 0)
            XCTAssertEqual(partial.chunkRows, 2)
            XCTAssertEqual(partial.coverage, 0)
            XCTAssertFalse(partial.isComplete)

            for rowID in [Int64(1), Int64(2)] {
                try store.db.execute(
                    "INSERT INTO \(VectorStore.binaryVectorTable)(rowid, embedding) "
                    + "VALUES(?, vec_bit(?))",
                    [.int(rowID), .blob(signature(UInt8(rowID)))])
            }
            try store.db.execute(
                "UPDATE meta SET value = '2' WHERE key = ?",
                [.text(VectorStore.binaryRowsKey)])
            store.close()
        }

        let reopened = try VectorStore(
            dbPath: path, dimension: dimension, elementType: .int8,
            binaryDimension: binaryDimension, binarySource: "sig",
            indexFingerprint: "fp")
        let complete = reopened.binaryIndexStatus()
        XCTAssertEqual(complete.rows, 2)
        XCTAssertEqual(complete.chunkRows, 2)
        XCTAssertEqual(complete.coverage, 1)
        XCTAssertTrue(complete.isComplete)
        reopened.close()
    }

    func testAddWritesBinaryRowsWithSameRowIDsAndPersistsCoverage() throws {
        let path = dbPath("write")
        let store = try VectorStore(
            dbPath: path, dimension: dimension, elementType: .int8,
            binaryDimension: binaryDimension, binarySource: "sig",
            indexFingerprint: "fp")
        let input = chunks(3)
        let signatures = [UInt8(1), 2, 3].map(signature)
        XCTAssertEqual(
            try store.add(chunks: input,
                          vectors: [oneHot(0), oneHot(1), oneHot(2)],
                          binarySignatures: signatures), 3)

        let status = store.binaryIndexStatus()
        XCTAssertEqual(status.rows, 3)
        XCTAssertEqual(status.chunkRows, 3)
        XCTAssertEqual(status.coverage, 1)
        XCTAssertTrue(status.isComplete)

        let chunkIDs = try store.db.query("SELECT rowid FROM chunks ORDER BY rowid")
            .compactMap { $0["rowid"]?.intValue }
        let binIDs = try store.db.query(
            "SELECT rowid FROM \(VectorStore.binaryVectorTable) ORDER BY rowid")
            .compactMap { $0["rowid"]?.intValue }
        XCTAssertEqual(binIDs, chunkIDs, "bin 行 rowid 必须与 chunks 严格一致")
        XCTAssertEqual(try meta(store, VectorStore.binaryRowsKey), "3")
        store.close()

        let reopened = try VectorStore(
            dbPath: path, dimension: dimension, elementType: .int8,
            binaryDimension: binaryDimension, binarySource: "sig",
            indexFingerprint: "fp")
        XCTAssertTrue(reopened.binaryIndexStatus().isComplete,
                      "覆盖率计数器必须跨重开保持")
        reopened.close()
    }

    /// 没有签名时主向量照常写入，但覆盖率下降；补齐签名后同一 source
    /// 会被 upsert 覆盖，覆盖率恢复完整。
    func testAddWithoutSignaturesKeepsMainVectorsAndCoverageFalls() throws {
        let store = try VectorStore(
            dbPath: dbPath("partial"), dimension: dimension, elementType: .int8,
            binaryDimension: binaryDimension, binarySource: "sig",
            indexFingerprint: "fp")
        XCTAssertEqual(
            try store.add(chunks: chunks(2), vectors: [oneHot(0), oneHot(1)]), 2)
        let partial = store.binaryIndexStatus()
        XCTAssertEqual(partial.rows, 0)
        XCTAssertEqual(partial.chunkRows, 2)
        XCTAssertEqual(partial.coverage, 0)
        XCTAssertFalse(partial.isComplete)
        XCTAssertFalse(try store.search(vector: oneHot(0), topK: 1).isEmpty,
                       "无签名不应影响主向量写入与检索")

        XCTAssertEqual(
            try store.add(chunks: chunks(2), vectors: [oneHot(0), oneHot(1)],
                          binarySignatures: [signature(1), signature(2)]), 2)
        XCTAssertTrue(store.binaryIndexStatus().isComplete, "重同步后覆盖率应恢复")
        store.close()
    }

    func testSignatureCountMismatchRollsBackMainAndBinaryRows() throws {
        let store = try VectorStore(
            dbPath: dbPath("count-mismatch"), dimension: dimension, elementType: .int8,
            binaryDimension: binaryDimension, binarySource: "sig",
            indexFingerprint: "fp")
        _ = try store.add(
            chunks: [Chunk(text: "keep", source: "keep.md", chunkIndex: 0)],
            vectors: [oneHot(0)], binarySignatures: [signature(1)])
        XCTAssertEqual(store.count, 1)
        XCTAssertEqual(store.binaryIndexStatus().rows, 1)

        XCTAssertThrowsError(
            try store.add(chunks: chunks(2), vectors: [oneHot(1), oneHot(2)],
                          binarySignatures: [signature(2)])
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("签名数量"),
                          error.localizedDescription)
        }
        XCTAssertEqual(store.count, 1, "失败必须回滚主向量")
        XCTAssertEqual(store.binaryIndexStatus().rows, 1, "失败必须回滚 bin 计数")
        XCTAssertEqual(try store.db.query("SELECT text FROM chunks").count, 1)
        store.close()
    }

    func testSignatureLengthMismatchRollsBackWholeTransaction() throws {
        let store = try VectorStore(
            dbPath: dbPath("length-mismatch"), dimension: dimension, elementType: .int8,
            binaryDimension: binaryDimension, binarySource: "sig",
            indexFingerprint: "fp")
        XCTAssertThrowsError(
            try store.add(chunks: chunks(2), vectors: [oneHot(0), oneHot(1)],
                          binarySignatures: [Data([0x01]), signature(2)])
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("字节数"),
                          error.localizedDescription)
        }
        XCTAssertEqual(store.count, 0)
        XCTAssertEqual(store.binaryIndexStatus().rows, 0)
        XCTAssertEqual(
            try store.db.scalar(
                "SELECT COUNT(*) FROM \(VectorStore.binaryVectorTable)").intValue, 0)
        store.close()
    }

    func testDeleteClearAndRebuildKeepBinaryCounterInSync() throws {
        let store = try VectorStore(
            dbPath: dbPath("lifecycle"), dimension: dimension, elementType: .int8,
            binaryDimension: binaryDimension, binarySource: "sig",
            indexFingerprint: "fp")
        _ = try store.add(chunks: chunks(2), vectors: [oneHot(0), oneHot(1)],
                          binarySignatures: [signature(1), signature(2)])
        let ids = try store.db.query("SELECT rowid FROM chunks ORDER BY rowid")
            .compactMap { $0["rowid"]?.intValue }
        XCTAssertEqual(try store.deleteRowIDs([ids[0]]), 1)

        let afterDelete = store.binaryIndexStatus()
        XCTAssertEqual(afterDelete.rows, 1)
        XCTAssertEqual(afterDelete.chunkRows, 1)
        XCTAssertTrue(afterDelete.isComplete)
        XCTAssertEqual(
            try store.db.scalar(
                "SELECT COUNT(*) FROM \(VectorStore.binaryVectorTable)").intValue, 1)

        _ = try store.add(
            chunks: [Chunk(text: "new", source: "new.md", chunkIndex: 0)],
            vectors: [oneHot(2)], binarySignatures: [signature(3)])
        XCTAssertEqual(store.binaryIndexStatus().rows, 2)

        try store.clear()
        let cleared = store.binaryIndexStatus()
        XCTAssertEqual(cleared.rows, 0)
        XCTAssertEqual(cleared.chunkRows, 0)
        XCTAssertEqual(cleared.coverage, 0)
        XCTAssertEqual(
            try store.db.scalar(
                "SELECT COUNT(*) FROM \(VectorStore.binaryVectorTable)").intValue, 0)

        _ = try store.add(chunks: chunks(1), vectors: [oneHot(0)],
                          binarySignatures: [signature(1)])
        XCTAssertEqual(store.binaryIndexStatus().rows, 1)
        _ = try store.rebuildForCurrentConfiguration()
        let rebuilt = store.binaryIndexStatus()
        XCTAssertEqual(rebuilt.rows, 0)
        XCTAssertEqual(rebuilt.chunkRows, 0)
        XCTAssertEqual(
            try store.db.scalar(
                "SELECT COUNT(*) FROM \(VectorStore.binaryVectorTable)").intValue, 0)
        store.close()
    }

    /// 空库按真实维度重建主表时，残留的 bin 行也必须一并清掉。
    func testDimensionRebuildClearsStaleBinaryRows() throws {
        let store = try VectorStore(
            dbPath: dbPath("dimension-rebuild"), dimension: dimension,
            elementType: .int8, binaryDimension: binaryDimension,
            binarySource: "sig", indexFingerprint: "fp")
        try store.db.execute(
            "INSERT INTO \(VectorStore.binaryVectorTable)(rowid, embedding) "
            + "VALUES(1, vec_bit(?))", [.blob(signature(1))])
        try store.db.execute(
            "UPDATE meta SET value = '1' WHERE key = ?",
            [.text(VectorStore.binaryRowsKey)])

        let eightDimensions = [Float](repeating: 0, count: 8)
        _ = try store.add(
            chunks: [Chunk(text: "new", source: "new.md", chunkIndex: 0)],
            vectors: [eightDimensions])

        XCTAssertEqual(store.dimension, 8)
        XCTAssertEqual(store.binaryIndexStatus().rows, 0)
        XCTAssertEqual(
            try store.db.scalar(
                "SELECT COUNT(*) FROM \(VectorStore.binaryVectorTable)").intValue, 0)
        store.close()
    }
}
