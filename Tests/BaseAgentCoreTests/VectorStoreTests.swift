import XCTest
@testable import BaseAgentCore

final class VectorStoreTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-vector-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    /// Weighted RRF：向量路命中 vs BM25 路命中，权重 0.7 时向量优先、0.3 时反转。
    func testHybridSearchWeightedRRF() throws {
        let store = try VectorStore(dbPath: directory.appendingPathComponent("wrrf.sqlite").path,
                                    dimension: 3, enableSQLiteVec: true)
        // A 只被向量路命中（query 向量 [1,0,0]）；B 只被 BM25 命中（词 zebraquokka）。
        let chunks = [
            Chunk(text: "alpha general text", source: "a.md", chunkIndex: 0),
            Chunk(text: "zebraquokka rareword", source: "b.md", chunkIndex: 0),
        ]
        XCTAssertEqual(try store.add(chunks: chunks,
                                     vectors: [[1, 0, 0], [0, 1, 0]]), 2)
        // 向量权重 0.7：向量命中的 a 应排前
        let high = try store.hybridSearch(vector: [1, 0, 0], query: "zebraquokka",
                                          topK: 2, rrfK: 60, vectorWeight: 0.7)
        // a 只被向量路命中（BM25 词只在 b 中），其 rrfScore 应严格等于 w/(k+1)：
        // 权重 0.7 → 0.7/61，权重 0.3 → 0.3/61，证明权重真正作用于融合公式。
        let aHigh = high.first(where: { $0.source.hasPrefix("a.md") })
        let low = try store.hybridSearch(vector: [1, 0, 0], query: "zebraquokka",
                                         topK: 2, rrfK: 60, vectorWeight: 0.3)
        let aLow = low.first(where: { $0.source.hasPrefix("a.md") })
        XCTAssertNotNil(aHigh)
        XCTAssertNotNil(aLow)
        XCTAssertEqual(aHigh?.rrfScore ?? 0, 0.7 / 61.0, accuracy: 1e-4)
        XCTAssertEqual(aLow?.rrfScore ?? 0, 0.3 / 61.0, accuracy: 1e-4)
    }

    func testSQLiteVecTransactionalWriteTwoStageKNNAndHybridSearch() throws {
        let store = try VectorStore(dbPath: directory.appendingPathComponent("index.sqlite").path,
                                    dimension: 3, enableSQLiteVec: true)
        XCTAssertTrue(store.vectorEnabled, "sqlite-vec 应成功静态注册")

        let chunks = [
            Chunk(text: "apple banana fruit", source: "bookA.pdf", chunkIndex: 0, heading: "第一章"),
            Chunk(text: "orange grape fruit", source: "bookA.pdf", chunkIndex: 1),
            Chunk(text: "engine turbine machine", source: "bookB.pdf", chunkIndex: 0),
        ]
        let vectors: [[Float]] = [[1, 0, 0], [0.8, 0.2, 0], [0, 1, 0]]
        XCTAssertEqual(try store.add(chunks: chunks, vectors: vectors), 3)
        XCTAssertEqual(store.count, 3)

        // 两段式 KNN：先向量表取 rowid，再主表取正文。
        let hits = try store.search(vector: [1, 0, 0], topK: 2)
        XCTAssertEqual(hits.count, 2)
        XCTAssertEqual(hits.first?.text, "apple banana fruit")
        // hydrate 改用 stream + 列序号读取，heading/chunkIndex 不能错位。
        XCTAssertEqual(hits.first?.heading, "第一章")
        XCTAssertEqual(hits.first?.chunkIndex, 0)

        // over-fetch + 第二阶段过滤。
        let filtered = try store.search(vector: [1, 0, 0], topK: 2, sourceContains: "bookB")
        XCTAssertEqual(filtered.count, 1)
        XCTAssertEqual(filtered.first?.source, "bookB.pdf")

        // FTS5 关键词检索。
        let fts = try store.searchFTS(query: "apple", topK: 3)
        XCTAssertEqual(fts.count, 1)
        XCTAssertEqual(fts.first?.chunkIndex, 0)

        // RRF 融合：向量与 FTS 都命中 apple。
        let hybrid = try store.hybridSearch(vector: [1, 0, 0], query: "apple", topK: 3)
        XCTAssertFalse(hybrid.isEmpty)
        XCTAssertEqual(hybrid.first?.rowID, hits.first?.rowID)

        // 单事务删除。
        XCTAssertEqual(try store.deleteBySource("bookA.pdf"), 2)
        XCTAssertEqual(store.count, 1)
        XCTAssertTrue(try store.searchFTS(query: "apple", topK: 3).isEmpty)

        // 写入失败必须回滚（包括其前置删除）。
        XCTAssertThrowsError(try store.add(
            chunks: [Chunk(text: "bad", source: "bookB.pdf", chunkIndex: 0)],
            vectors: [[1, 0]])
        )
        XCTAssertEqual(store.count, 1)
    }

    func testDeduplicateTextChunksKeepsEarliestCopy() throws {
        let store = try VectorStore(dbPath: directory.appendingPathComponent("dedupe.sqlite").path,
                                    dimension: 2)
        let chunks = [
            Chunk(text: "重复正文", source: "a.md", chunkIndex: 0),
            Chunk(text: "唯一 A", source: "a.md", chunkIndex: 1),
            Chunk(text: "重复正文", source: "b.md", chunkIndex: 0),
            Chunk(text: "重复正文", source: "c.md", chunkIndex: 0),
            Chunk(text: "唯一 B", source: "b.md", chunkIndex: 1),
        ]
        _ = try store.add(chunks: chunks,
                          vectors: [[1, 0], [0, 1], [1, 0], [1, 0], [0, 1]])
        // 入库时间：a.md 最早，b.md 次之，c.md 最晚。
        for (source, created) in [("a.md", 100.0), ("b.md", 200.0), ("c.md", 300.0)] {
            try store.db.execute("UPDATE chunks SET created_at = ? WHERE source = ?",
                                 [.double(created), .text(source)])
        }

        let preview = try store.previewDuplicateTextChunks(pageSize: 2)
        XCTAssertEqual(preview.scannedRows, 5)
        XCTAssertEqual(preview.duplicateRows, 2, "预览只统计重复，不删除")
        XCTAssertEqual(preview.removedRows, 0)
        XCTAssertEqual(store.count, 5)

        let result = try store.deduplicateTextChunks(pageSize: 2)
        XCTAssertEqual(result.removedRows, 2)
        XCTAssertEqual(result.removedBySource["b.md"], 1)
        XCTAssertEqual(result.removedBySource["c.md"], 1)
        XCTAssertEqual(result.canonicalBySource["b.md"], "a.md")
        XCTAssertEqual(result.canonicalBySource["c.md"], "a.md")
        XCTAssertEqual(store.chunkCount(source: "a.md"), 2, "最早入库的一份必须保留")
        XCTAssertEqual(store.chunkCount(source: "b.md"), 1)
        XCTAssertEqual(store.chunkCount(source: "c.md"), 0)
        XCTAssertEqual(store.count, 3)
        XCTAssertEqual(store.sourceCount(), 2, "c.md 的切片被清空后不再计入来源数")
        XCTAssertEqual(store.listSources(limit: 2).count, 2)
        XCTAssertEqual(try store.searchFTS(query: "重复正文", topK: 3).count, 1)

        // 重复执行幂等。
        XCTAssertEqual(try store.deduplicateTextChunks().removedRows, 0)
        store.close()
    }

    /// 精确来源过滤必须在该来源内部做向量检索：全局 KNN 的前 topK 被其他
    /// 来源占满时，定向检索不能返回空。
    func testExactSourceFilterUsesScopedVectorSearch() throws {
        let store = try VectorStore(dbPath: directory.appendingPathComponent("scoped.sqlite").path,
                                    dimension: 2, enableSQLiteVec: true)
        XCTAssertTrue(store.vectorEnabled, "本测试需要 vec0 全局 KNN 才能复现召回截断")

        // 旧实现只做全局 KNN 并按倍数放大 k（20 → 40 → … → maxVectorK=4096）。
        // 其他来源的切片超过 4096 条时，目标块永远进不了候选池，过滤后为空。
        let others = 4_100
        var chunks: [Chunk] = []
        var vectors: [[Float]] = []
        chunks.reserveCapacity(others + 1)
        vectors.reserveCapacity(others + 1)
        for index in 0..<others {
            chunks.append(Chunk(text: "apple \(index)", source: "other.md", chunkIndex: index))
            vectors.append([1, 0])
        }
        chunks.append(Chunk(text: "zebra quokka", source: "target.md", chunkIndex: 0))
        vectors.append([0.2, 0.98])
        XCTAssertEqual(try store.add(chunks: chunks, vectors: vectors), others + 1)

        // 目标块全局排名在第 4101 位，超出精确 KNN 的候选上限。
        let hits = try store.search(vector: [1, 0], topK: 1, sourceFilter: "target.md")
        XCTAssertEqual(hits.count, 1, "精确来源检索必须在该来源内召回")
        XCTAssertEqual(hits.first?.source, "target.md")
        store.close()
    }

    func testCJKFTSTokenization() throws {
        let store = try VectorStore(dbPath: directory.appendingPathComponent("cjk.sqlite").path,
                                    dimension: 2)
        let chunks = [
            Chunk(text: "人工智能技术正在快速发展", source: "zh.md", chunkIndex: 0),
            Chunk(text: "今天天气很好", source: "zh.md", chunkIndex: 1),
        ]
        _ = try store.add(chunks: chunks, vectors: [[1, 0], [0, 1]])
        let hits = try store.searchFTS(query: "人工智能", topK: 3)
        XCTAssertEqual(hits.count, 1)
        XCTAssertTrue(hits[0].text.contains("人工智能"))
    }

    /// FTS 先按显著 token 的 AND 召回；AND 无结果才退化为 OR。
    /// 否则「apple banana」会被 OR 稀释，只含 apple 的切片也会排在前面。
    func testFTSPrefersANDThenFallsBackToOR() throws {
        let store = try VectorStore(dbPath: directory.appendingPathComponent("fts-and.sqlite").path,
                                    dimension: 2)
        let chunks = [
            Chunk(text: "apple banana fruit", source: "a.md", chunkIndex: 0),
            Chunk(text: "apple only", source: "b.md", chunkIndex: 0),
            Chunk(text: "banana only", source: "c.md", chunkIndex: 0),
        ]
        _ = try store.add(chunks: chunks, vectors: [[1, 0], [0.8, 0.2], [0.2, 0.8]])

        let andHits = try store.searchFTS(query: "apple banana", topK: 5)
        XCTAssertEqual(andHits.count, 1, "AND 只应命中同时包含两个词的切片")
        XCTAssertEqual(andHits.first?.source, "a.md")

        let orHits = try store.searchFTS(query: "apple quokka", topK: 5)
        XCTAssertEqual(Set(orHits.map(\.source)), ["a.md", "b.md"],
                       "AND 无结果时回退 OR，应命中只含 apple 的切片")
        store.close()
    }

    /// 停用词过滤 + 显著 token 上限：问句里的虚词不参与 AND，超长查询截断。
    func testFTSQueryFiltersStopwordsAndCapsTokens() {
        XCTAssertEqual(TextAnalyzer.ftsQuery("What is the apple", mode: .and), "\"apple\"")
        let chinese = TextAnalyzer.ftsQuery("如何 配置 这个 系统", mode: .and)
        XCTAssertTrue(chinese.contains("\"配置\""))
        XCTAssertTrue(chinese.contains("\"系统\""))
        XCTAssertFalse(chinese.contains("如何"))
        XCTAssertFalse(chinese.contains("这个"))

        // 全字母词避免 NLTokenizer 把数字拆成独立 token。
        let alphabet = Array("abcdefghijklmnopqrstuvwxyz")
        let words = (0..<40).map { index in
            String([alphabet[index / 26], alphabet[index % 26]]) + "q"
        }
        let tokens = TextAnalyzer.ftsSignificantTokens(words.joined(separator: " "))
        XCTAssertEqual(tokens.count, TextAnalyzer.maxSignificantTokens)
        XCTAssertEqual(Set(tokens).count, tokens.count, "显著 token 必须去重")
        XCTAssertTrue(tokens.contains(words[0]))
        XCTAssertFalse(tokens.contains(words[39]), "超出上限的短尾 token 应被截断")
    }

    /// 混合检索先融合 rowid/分数再 hydrate：RRF 选出的 Top1 才读正文，
    /// 并保留两路各自的原始分数。
    func testHybridSearchFusesBeforeHydratingTopK() throws {
        let store = try VectorStore(dbPath: directory.appendingPathComponent("hybrid.sqlite").path,
                                    dimension: 2)
        let chunks = [
            Chunk(text: "apple banana fruit", source: "vector.md", chunkIndex: 0),
            Chunk(text: "banana turbine", source: "fts.md", chunkIndex: 0),
        ]
        // 查询向量更接近 vector.md，但 fts.md 额外命中全文通道。
        _ = try store.add(chunks: chunks, vectors: [[1, 0], [0.2, 0.98]])

        let hits = try store.hybridSearch(vector: [1, 0], query: "turbine", topK: 1)
        XCTAssertEqual(hits.count, 1, "只 hydrate RRF 定出的前 topK 条")
        XCTAssertEqual(hits.first?.source, "fts.md")
        XCTAssertNotNil(hits.first?.rrfScore)
        XCTAssertNotNil(hits.first?.vectorScore)
        XCTAssertNotNil(hits.first?.ftsScore)
        store.close()
    }

    func testChunkCounterPersistsAcrossReopenAndLegacyFallback() throws {
        let path = directory.appendingPathComponent("counter.sqlite").path
        do {
            let store = try VectorStore(dbPath: path, dimension: 2)
            _ = try store.add(chunks: [
                Chunk(text: "a", source: "a.md", chunkIndex: 0),
                Chunk(text: "b", source: "a.md", chunkIndex: 1),
                Chunk(text: "c", source: "b.md", chunkIndex: 0),
            ], vectors: [[1, 0], [0, 1], [1, 1]])
            XCTAssertEqual(store.count, 3)
            XCTAssertEqual(try store.deleteBySource("a.md"), 2)
            XCTAssertEqual(store.count, 1)
            store.close()
        }
        do {
            let store = try VectorStore(dbPath: path, dimension: 2)
            XCTAssertEqual(store.count, 1, "计数器必须随库持久化，重开后无需 COUNT(*)")
            // 模拟升级前的旧库：没有 meta 计数器。
            try store.db.execute("DELETE FROM meta WHERE key = 'chunks_count'")
            store.close()
        }
        do {
            let legacy = try VectorStore(dbPath: path, dimension: 2)
            XCTAssertEqual(legacy.count, 3,
                           "旧库回退 MAX(rowid)（O(1)，允许略高于实际值）")
            XCTAssertTrue(legacy.hasChunks)
            legacy.close()
        }
    }

    /// 回填签名位宽与库不一致时不得降级停用二值通道：必须抛出专用错误，
    /// 由回填调度丢弃 store 按当前 provider 位宽重建（对应线上
    /// 「96 字节签名写进 bit[128] bin 表」的历史窗口状态）。
    func testFillPendingVectorThrowsOnSignatureWidthMismatch() throws {
        let store = try VectorStore(dbPath: directory.appendingPathComponent("selfheal.sqlite").path,
                                    dimension: 3, enableSQLiteVec: true,
                                    binaryDimension: 16, binarySource: "test")
        XCTAssertTrue(store.binaryStorageEnabled)
        let rowids = try store.addTextsOnly(chunks: [
            Chunk(text: "正文一", source: "a.md", chunkIndex: 0),
            Chunk(text: "正文二", source: "a.md", chunkIndex: 1),
        ])
        XCTAssertEqual(rowids.count, 2)
        XCTAssertEqual(store.pendingVectorCount(), 2)

        // 宽度错误的签名（3 字节 != 16/8=2 字节）：不降级停用，直接抛专用错误，
        // 由回填调度丢弃 store 按当前位宽重建；该条不出队、保持 pending。
        var thrown: Error?
        XCTAssertNoThrow(thrown = {
            do {
                try store.fillPendingVector(rowid: rowids[0], vector: [1, 0, 0],
                                            binarySignature: Data([1, 2, 3]))
                return nil
            } catch { return error }
        }())
        let mismatch = try XCTUnwrap(thrown as? BinarySignatureWidthMismatchError,
                                     "位宽不匹配必须抛出 BinarySignatureWidthMismatchError")
        XCTAssertEqual(mismatch.signatureBytes, 3)
        XCTAssertEqual(mismatch.declaredBytes, 2)
        XCTAssertTrue(store.binaryStorageEnabled, "不得降级停用二值通道")
        XCTAssertEqual(store.pendingVectorCount(), 2, "出错的条目必须保持 pending")

        // 位宽正确的签名照常回填（二值通道未被污染）。
        try store.fillPendingVector(rowid: rowids[0], vector: [1, 0, 0],
                                    binarySignature: Data([1, 2]))
        XCTAssertEqual(store.pendingVectorCount(), 1)
        let hits = try store.search(vector: [1, 0, 0], topK: 1)
        XCTAssertEqual(hits.first?.text, "正文一", "主向量已回填、可检索")

        // 无签名的主向量回填不受影响（签名 nil 时跳过位宽校验）。
        try store.fillPendingVector(rowid: rowids[1], vector: [0, 1, 0])
        XCTAssertEqual(store.pendingVectorCount(), 0)
        store.close()
    }

    /// 已回填齐全的库不得被 enqueueMissingVectors 误判为缺向量：vec0 主表的
    /// rowid 点查不可靠（返回空），必须走影子表 `chunk_vectors_rowids`
    ///（对应线上「手动补做把 1925 条已回填切片全部重新入队」）。
    func testEnqueueMissingVectorsDoesNotReenqueueFilledChunks() throws {
        let store = try VectorStore(dbPath: directory.appendingPathComponent("enqueue.sqlite").path,
                                    dimension: 3, enableSQLiteVec: true)
        XCTAssertTrue(store.vectorEnabled)
        let chunks = [
            Chunk(text: "alpha text", source: "a.md", chunkIndex: 0),
            Chunk(text: "beta text", source: "b.md", chunkIndex: 0),
        ]
        XCTAssertEqual(try store.add(chunks: chunks, vectors: [[1, 0, 0], [0, 1, 0]]), 2)
        XCTAssertEqual(store.pendingVectorCount(), 0)
        XCTAssertEqual(try store.enqueueMissingVectors(), 0,
                       "向量齐全时不应有任何条目被重新入队")
        store.close()
    }
}
