import XCTest
@testable import BaseAgentCore

/// 引擎与 ANN 索引的集成：自动构建开关、状态字段、检索走索引/回退。
final class RAGEngineANNTests: XCTestCase {
    private var root: URL!
    private var kb: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ann-engine-\(UUID().uuidString)")
        kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    // MARK: - 夹具

    private final class FakeEmbedding: EmbeddingProvider {
        let identifier = "ann-engine-fake"
        let dimension = 8

        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map { text in
                var vector = [Float](repeating: 0, count: dimension)
                for (index, scalar) in text.unicodeScalars.enumerated() {
                    vector[Int(scalar.value + UInt32(index)) % dimension] += 1
                }
                return VectorCodec.normalize(vector)
            }
        }

        func embedQuery(_ text: String) async throws -> [Float] {
            try await embedDocuments([text])[0]
        }
    }

    private func makeConfig() -> AgentConfig {
        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 5
        config.ragANNClusters = 4
        config.ragANNClusterSize = 8
        return config
    }

    private func writeDocs(_ count: Int) throws {
        for index in 1...count {
            let text = """
            # 文档 \(index)

            Apple banana orange fruit \(index). 苹果 香蕉 橙子 水果 \(index)。
            涡轮 发动机 机械 设备 编号 \(index)。turbine engine machine \(index)。
            Additional sentences make the body long enough for indexing, so the document is kept in the knowledge base.
            """
            try text.write(to: kb.appendingPathComponent("doc\(index).md"),
                           atomically: true, encoding: .utf8)
        }
    }

    private func makeEngine(_ config: AgentConfig) -> RAGEngine {
        RAGEngine(config: config, knowledgeBase: kb.path,
                  embedding: FakeEmbedding(), reranker: nil)
    }

    // MARK: - 用例

    /// 状态快照里必须能读到 ANN 索引状态（未构建 / 已就绪）。
    func testStatusExposesANNIndexState() async throws {
        try writeDocs(2)
        var config = makeConfig()
        // 关闭自动构建：先验证「未构建」状态。
        config.ragANNAutoBuild = false
        let engine = makeEngine(config)
        _ = await engine.ingest()

        let before = engine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(before["ann_index"]?.stringValue, "ANN 索引未构建")
        XCTAssertEqual(before["ann_usable"]?.boolValue, false)

        let built = try XCTUnwrap(engine.buildANNIndex())
        XCTAssertFalse(built.cancelled)
        XCTAssertGreaterThan(built.indexedRows, 0)

        let after = engine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(after["ann_usable"]?.boolValue, true)
        XCTAssertEqual(after["ann_needs_rebuild"]?.boolValue, false)
        XCTAssertTrue(after["ann_index"]?.stringValue?.contains("分块 IVF") == true)
        XCTAssertEqual(after["ann_stale_rows"]?.intValue, 0)
        XCTAssertEqual(after["ann_effective_coverage"]?.doubleValue ?? 0, 1.0, accuracy: 0.01)
        XCTAssertEqual(after["ann_building"]?.boolValue, false)
        engine.close()
    }

    /// 未达到自动构建的切片数下限时不建索引（小库精确扫描足够快）。
    func testAutoBuildHonoursMinChunks() async throws {
        try writeDocs(3)
        var config = makeConfig()
        config.ragANNMinChunks = 1_000_000
        let engine = makeEngine(config)
        _ = await engine.ingest()
        // 自动构建在同步返回后的后台任务里执行，先等它跑完再读状态。
        await engine.awaitANNAutoBuild()

        let status = try XCTUnwrap(engine.annIndexStatus())
        XCTAssertEqual(status.blocks, 0)
        engine.close()
    }

    /// 达到下限后同步结束会自动构建，检索随即走索引。
    func testIngestAutoBuildsIndexAndSearchUsesIt() async throws {
        try writeDocs(3)
        var config = makeConfig()
        config.ragANNMinChunks = 1
        let engine = makeEngine(config)
        // 断言失败时也要关闭引擎：泄漏的后台回填服务（共享全局 embedding gate）
        // 会干扰后续用例的回填时序。
        defer { engine.close() }
        let stats = await engine.ingest()
        XCTAssertGreaterThan(stats.chunks, 0)
        // 向量回填全部完成且服务排空（约 6s 空闲判定）后才会调度自动构建：
        // 先等向量补齐，再等服务退出，最后等构建任务收尾。
        try await awaitVectorsFilled(engine)
        try await awaitBackfillIdle(engine)
        await engine.awaitANNAutoBuild()

        let status = try XCTUnwrap(engine.annIndexStatus())
        XCTAssertTrue(status.isReady, status.summary)
        XCTAssertGreaterThan(status.indexedRows, 0)
        XCTAssertFalse(engine.annAutoBuildState.isRunning)
        let snapshot = engine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(snapshot["ann_building"]?.boolValue, false)
        XCTAssertEqual(snapshot["ann_build_progress"]?.doubleValue ?? 0, 0)

        let hits = try await engine.search(query: "apple fruit turbine", topK: 3)
        XCTAssertFalse(hits.isEmpty)
        XCTAssertTrue(hits[0].source.hasSuffix(".md"))

        // 关闭 ANN 后结果仍然可用（回退精确检索）。ANN 是近似检索（nprobe=1），
        // topK 边界上的排序可能与精确扫描不同，因此只要求两者至少命中同一片段。
        var offConfig = config
        offConfig.ragANNEnabled = false
        let offEngine = makeEngine(offConfig)
        defer { offEngine.close() }
        let offHits = try await offEngine.search(query: "apple fruit turbine", topK: 3)
        XCTAssertFalse(offHits.isEmpty)
        XCTAssertTrue(offHits.contains { hit in hits.contains { $0.rowID == hit.rowID } },
                      "ANN=\(hits.map(\.rowID)) 精确=\(offHits.map(\.rowID))")
        offEngine.close()
        engine.close()
    }

    /// 索引已就绪时再次构建是空操作（水位在末尾，不会重复写块）。
    func testSecondBuildIsNoopWhenUpToDate() async throws {
        try writeDocs(2)
        var config = makeConfig()
        config.ragANNMinChunks = 1
        let engine = makeEngine(config)
        _ = await engine.ingest()
        await engine.awaitANNAutoBuild()
        // 手动构建需要完整向量（文本先行入库），等后台回填结束再建。
        try await awaitVectorsFilled(engine)

        let first = try XCTUnwrap(engine.buildANNIndex())
        XCTAssertGreaterThan(first.blocks, 0)
        let second = try XCTUnwrap(engine.buildANNIndex())
        XCTAssertEqual(second.blocks, first.blocks)
        XCTAssertEqual(second.indexedRows, first.indexedRows)
        XCTAssertEqual(second.tailRows, 0)
        XCTAssertFalse(second.cancelled)
        engine.close()
    }

    /// 删除来源后旧 rowid 留在 ANN 块内：状态要展示陈旧行与实际覆盖率。
    func testStatusShowsStaleRowsAfterSourceDeletion() async throws {
        try writeDocs(3)
        var config = makeConfig()
        config.ragANNAutoBuild = false
        let engine = makeEngine(config)
        _ = await engine.ingest()
        // ANN 构建需要完整向量（文本先行入库），等后台回填结束再建。
        try await awaitVectorsFilled(engine)
        _ = try XCTUnwrap(engine.buildANNIndex())

        // 新增来源形成尾块，删除后的实际覆盖率才会低于 100%。
        let extra = """
        # 文档 4

        Apple banana orange fruit 4. 苹果 香蕉 橙子 水果 4。
        Additional sentences make the body long enough for indexing, so the document is kept in the knowledge base.
        """
        try extra.write(to: kb.appendingPathComponent("doc4.md"),
                        atomically: true, encoding: .utf8)
        _ = await engine.ingest()
        XCTAssertGreaterThan(try XCTUnwrap(engine.annIndexStatus()).tailRows, 0)

        try FileManager.default.removeItem(at: kb.appendingPathComponent("doc1.md"))
        let stats = await engine.ingest()
        XCTAssertEqual(stats.removed, 1)

        let status = try XCTUnwrap(engine.annIndexStatus())
        XCTAssertGreaterThan(status.staleRows, 0)
        XCTAssertLessThan(status.effectiveCoverage, 1.0)
        let snapshot = engine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(snapshot["ann_stale_rows"]?.intValue, status.staleRows)
        XCTAssertEqual(snapshot["ann_effective_coverage"]?.doubleValue ?? -1,
                       status.effectiveCoverage, accuracy: 0.0001)
        engine.close()
    }

    /// force 同步会先重置 ANN：旧 rowid 全部失效，续建没有意义。
    func testForceIngestResetsANNIndex() async throws {
        try writeDocs(3)
        var config = makeConfig()
        config.ragANNAutoBuild = false
        let engine = makeEngine(config)
        _ = await engine.ingest()
        _ = try XCTUnwrap(engine.buildANNIndex())
        XCTAssertTrue(try XCTUnwrap(engine.annIndexStatus()).isReady)

        _ = await engine.ingest(force: true)
        let status = try XCTUnwrap(engine.annIndexStatus())
        XCTAssertEqual(status.blocks, 0)
        XCTAssertEqual(status.indexedRows, 0)
        XCTAssertEqual(engine.status(sourceLimit: 0, maxAge: 0)["ann_index"]?.stringValue,
                       "ANN 索引未构建")
        engine.close()
    }

    /// 手动构建支持进度与取消；取消后已写完的块保留，再次构建从断点续建。
    func testManualBuildCanBeCancelledAndResumed() async throws {
        try writeDocs(12)
        var config = makeConfig()
        config.ragANNAutoBuild = false
        // 2 簇 × 4 行 = 8 行/块，确保至少两块。
        config.ragANNClusters = 2
        config.ragANNClusterSize = 4
        let engine = makeEngine(config)
        _ = await engine.ingest()
        // 取消/续建验证需要完整向量（文本先行入库），等后台回填结束再建。
        try await awaitVectorsFilled(engine)
        let chunks = engine.status(sourceLimit: 0, maxAge: 0)["chunks_stored"]?.intValue ?? 0
        XCTAssertGreaterThan(chunks, 8, "至少需要两块才能验证取消")

        var stages: [String] = []
        var didCancel = false
        let partial = try XCTUnwrap(engine.buildANNIndex(progress: { update in
            stages.append(update.stage)
            if update.stage == "blocks", !didCancel {
                didCancel = true
                engine.cancel()
            }
        }))
        XCTAssertTrue(partial.cancelled)
        XCTAssertGreaterThan(partial.indexedRows, 0)
        XCTAssertLessThan(partial.indexedRows, chunks)
        XCTAssertTrue(stages.contains("start"))
        XCTAssertTrue(stages.contains("blocks"))

        let resumed = try XCTUnwrap(engine.buildANNIndex())
        XCTAssertFalse(resumed.cancelled)
        let status = try XCTUnwrap(engine.annIndexStatus())
        XCTAssertEqual(status.indexedRows, chunks)
        XCTAssertTrue(status.isReady, status.summary)
        engine.close()
    }

    // MARK: 回填等待辅助

    /// 等待全部切片向量回填完成（文本先行入库，ANN 构建与断言需要完整向量）。
    private func awaitVectorsFilled(_ engine: RAGEngine) async throws {
        for _ in 0..<300 {
            if engine.vectorCompleteness().pending == 0 { return }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
    }

    /// 等待后台回填服务排空退出（约 6 秒空闲判定；排空回调才会调度自动构建）。
    private func awaitBackfillIdle(_ engine: RAGEngine) async throws {
        for _ in 0..<120 {
            if !engine.backfillStatus().running { return }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
    }
}