import XCTest
@testable import BaseAgentCore

/// 异步向量回填数据层回归：切片文本先入库、向量后台补齐、FTS 立即可检索。
final class VectorBackfillTests: XCTestCase {
    private var directory: URL!
    private var dbPath: String!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("backfill-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        dbPath = directory.appendingPathComponent("index.sqlite").path
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func makeStore() throws -> VectorStore {
        try VectorStore(dbPath: dbPath, dimension: 3, enableSQLiteVec: true)
    }

    func testTextsFirstThenVectorBackfill() throws {
        let store = try makeStore()
        let chunks = [
            Chunk(text: "apple banana fruit", source: "bookA.pdf", chunkIndex: 0),
            Chunk(text: "engine turbine machine", source: "bookB.pdf", chunkIndex: 0),
        ]
        let rowids = try store.addTextsOnly(chunks: chunks)
        XCTAssertEqual(rowids.count, 2)
        XCTAssertEqual(store.count, 2, "正文应立即入库")
        XCTAssertEqual(store.pendingVectorCount(), 2, "两条都应进入待回填队列")

        // 回填前：FTS 可搜到（正文已入库），向量路搜不到（向量缺行）。
        XCTAssertEqual(try store.searchFTS(query: "apple", topK: 5).count, 1)
        XCTAssertTrue(try store.search(vector: [1, 0, 0], topK: 5).isEmpty,
                      "向量未回填前向量检索不应命中这些切片")

        // 回填第一批。
        let pending = store.nextPendingTexts(limit: 10)
        XCTAssertEqual(pending.count, 2)
        for item in pending {
            let vector: [Float] = item.text.hasPrefix("apple") ? [1, 0, 0] : [0, 1, 0]
            try store.fillPendingVector(rowid: item.rowid, vector: vector)
        }
        XCTAssertEqual(store.pendingVectorCount(), 0, "回填后队列应清空")

        // 回填后：向量检索命中，且 idempotent 再填不报错。
        let hits = try store.search(vector: [1, 0, 0], topK: 5)
        XCTAssertEqual(hits.first?.text, "apple banana fruit")
        if let first = pending.first {
            try store.fillPendingVector(rowid: first.rowid, vector: [1, 0, 0])
        }
        XCTAssertEqual(store.count, 2, "重复回填不应产生重复行")
    }

    /// 回归：int8 模式下 vec0 二值表不支持 rowid 点查，签名齐全的库曾被
    /// enqueueMissingVectors 全部误判为缺失、启动即全量回填抢占 GPU。
    /// 签名覆盖率计数齐全时不得入队任何回填任务。
    func testEnqueueSkipsWhenBinarySignaturesComplete() throws {
        let path = directory.appendingPathComponent("int8-index.sqlite").path
        let makeInt8Store = {
            try VectorStore(dbPath: path, dimension: 4, enableSQLiteVec: true,
                            elementType: .int8, binaryDimension: 8,
                            binarySource: "test-signing", indexFingerprint: "fp-int8")
        }
        do {
            let store = try makeInt8Store()
            let chunks = (0..<3).map {
                Chunk(text: "doc \($0) content", source: "book.pdf", chunkIndex: $0)
            }
            let vectors = (0..<3).map { _ in [Float](repeating: 0.5, count: 4) }
            let signatures = (0..<3).map { _ in Data([0b1010_1010]) }
            try store.add(chunks: chunks, vectors: vectors, binarySignatures: signatures)
            XCTAssertEqual(store.binaryIndexStatus().rows, 3)
        }
        // 重开（模拟启动）：签名齐全，不应误判入队。
        let reopened = try makeInt8Store()
        XCTAssertEqual(reopened.count, 3)
        XCTAssertEqual(reopened.enqueueMissingVectors(limit: 100), 0,
                       "签名齐全时不应把已有切片误判为缺向量/缺签名")
        XCTAssertEqual(reopened.pendingVectorCount(), 0)
    }

    func testPendingSurvivesReopen() throws {
        do {
            let store = try makeStore()
            try store.addTextsOnly(chunks: [
                Chunk(text: "hello world", source: "doc.md", chunkIndex: 0)
            ])
            XCTAssertEqual(store.pendingVectorCount(), 1)
        }
        // 重开数据库（模拟 App 重启）：pending 仍在，可续做。
        let reopened = try makeStore()
        XCTAssertEqual(reopened.pendingVectorCount(), 1, "重启后待回填项应从数据库恢复")
        let pending = reopened.nextPendingTexts(limit: 5)
        XCTAssertEqual(pending.count, 1)
        XCTAssertEqual(pending.first?.text, "hello world")
        try reopened.fillPendingVector(rowid: pending[0].rowid, vector: [0, 0, 1])
        XCTAssertEqual(reopened.pendingVectorCount(), 0)
    }

    /// 模拟决策模式入库（不入队）后切回向量模式：应能扫描补齐缺失向量。
    func testEnqueueMissingVectorsAfterDecisionMode() throws {
        let store = try makeStore()
        // 决策模式入库：正文入 chunks、明确不入 pending。
        try store.addTextsOnly(chunks: [
            Chunk(text: "决策模式下入库的正文一", source: "decision.md", chunkIndex: 0),
            Chunk(text: "决策模式下入库的正文二", source: "decision.md", chunkIndex: 1),
        ], enqueuePending: false)
        XCTAssertEqual(store.pendingVectorCount(), 0, "决策模式不应入向量队列")
        XCTAssertEqual(store.count, 2)

        // 切回向量模式：扫描补齐
        let added = store.enqueueMissingVectors()
        XCTAssertEqual(added, 2)
        XCTAssertEqual(store.pendingVectorCount(), 2)
        // 重复调用不重复入队
        XCTAssertEqual(store.enqueueMissingVectors(), 0)
        XCTAssertEqual(store.pendingVectorCount(), 2)

        // 回填一条后再扫描：只剩一条缺向量
        let pending = store.nextPendingTexts(limit: 5)
        try store.fillPendingVector(rowid: pending[0].rowid, vector: [1, 0, 0])
        XCTAssertEqual(store.enqueueMissingVectors(), 0, "已在 pending 的不重复入队")
        XCTAssertEqual(store.pendingVectorCount(), 1)
        try store.fillPendingVector(rowid: pending[1].rowid, vector: [0, 1, 0])
        XCTAssertEqual(store.enqueueMissingVectors(), 0, "全部回填后无缺失")
        XCTAssertEqual(store.pendingVectorCount(), 0)
    }

    /// 决策模式用「仅元数据」store（enableSQLiteVec:false）也能做 FTS 粗筛。
    func testFTSWorksOnMetadataOnlyStore() throws {
        let metaStore = try VectorStore.openForMetadata(dbPath: dbPath)
        try metaStore.addTextsOnly(chunks: [
            Chunk(text: "合同违约责任条款正文内容", source: "law.md", chunkIndex: 0),
        ], enqueuePending: false)
        let hits = try metaStore.searchFTS(query: "合同 违约", topK: 10)
        XCTAssertEqual(hits.count, 1, "决策模式不依赖 embedding，FTS 应能独立工作")
        XCTAssertTrue(hits[0].text.contains("违约"))
    }

    /// 失败退避：retry 后该切片不在下一批出现，但仍在 pending 中（不丢数据）。
    func testRetryPendingBackoff() throws {
        let store = try makeStore()
        try store.addTextsOnly(chunks: [
            Chunk(text: "退避测试正文", source: "retry.md", chunkIndex: 0),
        ])
        let pending = store.nextPendingTexts(limit: 5)
        XCTAssertEqual(pending.count, 1)
        store.retryPendingVector(rowid: pending[0].rowid, backoffSeconds: 300)
        XCTAssertTrue(store.nextPendingTexts(limit: 5).isEmpty, "退避中的项不应被取出")
        XCTAssertEqual(store.pendingVectorCount(), 1, "退避不丢数据")
    }

    func testDeletingSourceClearsPending() throws {
        let store = try makeStore()
        try store.addTextsOnly(chunks: [
            Chunk(text: "one", source: "gone.md", chunkIndex: 0),
            Chunk(text: "two", source: "kept.md", chunkIndex: 0),
        ])
        XCTAssertEqual(store.pendingVectorCount(), 2)
        XCTAssertEqual(try store.deleteBySource("gone.md"), 1)
        XCTAssertEqual(store.pendingVectorCount(), 1, "删除来源应同步清掉其 pending 行")
        XCTAssertEqual(store.nextPendingTexts(limit: 5).first?.text, "two")
    }
}
