import XCTest
@testable import BaseAgentCore

final class VectorStoreBinaryANNTests: XCTestCase {
    private var directory: URL!
    private let dimension = 16
    private let binaryDimension = 16

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("bann-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func dbPath(_ name: String) -> String {
        directory.appendingPathComponent("\(name).sqlite").path
    }

    private func makeStore(_ name: String) throws -> VectorStore {
        try VectorStore(dbPath: dbPath(name), dimension: dimension, elementType: .int8,
                        binaryDimension: binaryDimension, binarySource: "sig",
                        indexFingerprint: "fp-\(name)")
    }

    private func signature(bit: Int) -> Data {
        var bytes = Data(repeating: 0, count: binaryDimension / 8)
        bytes[bit / 8] |= UInt8(1 << (bit % 8))
        return bytes
    }

    private func seed(_ store: VectorStore, count: Int) throws {
        var chunks: [Chunk] = []
        var vectors: [[Float]] = []
        var signatures: [Data] = []
        for index in 0..<count {
            var vector = [Float](repeating: 0, count: dimension)
            vector[index % dimension] = 1
            chunks.append(Chunk(text: "chunk \(index)", source: "s\(index).md", chunkIndex: 0))
            vectors.append(vector)
            signatures.append(signature(bit: index % binaryDimension))
        }
        _ = try store.add(chunks: chunks, vectors: vectors, binarySignatures: signatures)
    }

    func testBuildProducesReadyStatus() throws {
        let store = try makeStore("ready")
        try seed(store, count: 40)
        let result = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
        XCTAssertFalse(result.cancelled)
        let status = store.binaryANNIndexStatus()
        XCTAssertTrue(status.isReady, status.summary)
        XCTAssertEqual(status.blocks, 1)
        XCTAssertEqual(status.indexedRows, 40)
        XCTAssertEqual(status.dimension, binaryDimension)
        XCTAssertEqual(status.coverage, 1.0, accuracy: 0.0001)
        store.close()
    }

    func testBuildRejectsIncompleteSignatures() throws {
        let store = try makeStore("incomplete")
        _ = try store.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0)],
            vectors: [[1] + [Float](repeating: 0, count: dimension - 1)])
        XCTAssertThrowsError(
            try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4))
        XCTAssertFalse(store.binaryANNIndexStatus().isReady)
        store.close()
    }

    func testBuildIsCancellableAndResumable() throws {
        let store = try makeStore("resume")
        try seed(store, count: 64)
        var calls = 0
        let partial = try store.buildBinaryANNIndex(
            clusters: 4, clusterSize: 16, iterations: 4,
            progress: { _ in },
            isCancelled: { calls += 1; return calls > 1 })
        XCTAssertTrue(partial.cancelled)
        XCTAssertEqual(store.binaryANNIndexStatus().blocks, 1)
        _ = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
        let status = store.binaryANNIndexStatus()
        XCTAssertTrue(status.isReady, status.summary)
        XCTAssertEqual(status.indexedRows, 64)
        store.close()
    }

    func testClearAndDropRemoveIndex() throws {
        let store = try makeStore("clear")
        try seed(store, count: 40)
        _ = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
        XCTAssertTrue(store.binaryANNIndexStatus().isReady)
        try store.dropBinaryANNIndex()
        XCTAssertFalse(store.binaryANNIndexStatus().isReady)
        _ = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
        try store.clear()
        XCTAssertEqual(store.binaryANNIndexStatus().blocks, 0)
        store.close()
    }

    func testDeleteAccumulatesStaleRows() throws {
        let store = try makeStore("stale")
        try seed(store, count: 40)
        _ = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
        let ids = try store.rowIDs(source: "s0.md")
        _ = try store.deleteRowIDs(ids)
        let status = store.binaryANNIndexStatus()
        XCTAssertEqual(status.staleRows, 1)
        XCTAssertEqual(status.effectiveCoverage, 39.0 / 40.0, accuracy: 0.0001)
        store.close()
    }

    func testBannQueryMatchesBruteForceWithFullProbe() throws {
        let store = try makeStore("query")
        try seed(store, count: 40)
        _ = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
        let query = signature(bit: 7)
        let bann = try XCTUnwrap(
            try store.bannCandidatesLocked(signature: query, topK: 5, probeOverride: 4))
        XCTAssertEqual(bann.count, 5)
        // 暴力汉明基准。sqlite-vec 对同分候选的返回顺序未定义（实测为 rowid
        // 降序），因此这里用确定性的 (分数降序, rowid 升序) 与 bann 比对，
        // 仍保持「全簇探测 == 暴力检索」的语义。
        let rows = try store.db.query(
            "SELECT rowid, embedding FROM \(VectorStore.binaryVectorTable) ORDER BY rowid")
        var expected: [(Int64, Double)] = []
        for row in rows {
            guard let rowID = row["rowid"]?.intValue,
                  let data = row["embedding"]?.dataValue else { continue }
            let distance = VectorEncoder.hamming(data, query)
            expected.append((rowID, 1 - Double(distance) / Double(binaryDimension)))
        }
        expected.sort { left, right in
            if left.1 == right.1 { return left.0 < right.0 }
            return left.1 > right.1
        }
        XCTAssertEqual(bann.map(\.0), Array(expected.prefix(5).map(\.0)))
        XCTAssertEqual(bann.map(\.1), Array(expected.prefix(5).map(\.1)))
        store.close()
    }

    func testSearchUsesBannAndSkipsANN() throws {
        let store = try makeStore("use-bann")
        try seed(store, count: 40)
        _ = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
        let query = signature(bit: 3)
        let hits = try store.hybridSearch(
            vector: [1] + [Float](repeating: 0, count: dimension - 1),
            query: "zzz", topK: 3,
            binary: BinaryCoarseOptions(signature: query, candidateLimit: 20),
            ann: ANNQueryOptions(nprobe: 4, rescore: 0))
        XCTAssertFalse(hits.isEmpty)
        XCTAssertGreaterThan(store.bannQueryCount, 0)
        XCTAssertEqual(store.annQueryCount, 0)
        XCTAssertEqual(store.binaryVecQueryCount, 0)
        store.close()
    }

    func testSearchFallsBackToVec0WhenBannNotBuilt() throws {
        let store = try makeStore("vec-fallback")
        try seed(store, count: 40)
        let query = signature(bit: 1)
        _ = try store.hybridSearch(
            vector: [1] + [Float](repeating: 0, count: dimension - 1),
            query: "zzz", topK: 3,
            binary: BinaryCoarseOptions(signature: query, candidateLimit: 20),
            ann: ANNQueryOptions(nprobe: 4, rescore: 0))
        XCTAssertEqual(store.bannQueryCount, 0)
        XCTAssertGreaterThan(store.binaryVecQueryCount, 0)
        XCTAssertEqual(store.annQueryCount, 0)
        store.close()
    }

    func testTailIsSearchableAfterBuild() throws {
        let store = try makeStore("tail")
        try seed(store, count: 32)
        _ = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
        _ = try store.add(
            chunks: [Chunk(text: "tail", source: "tail.md", chunkIndex: 0)],
            vectors: [[Float](repeating: 0, count: dimension)],
            binarySignatures: [Data(repeating: 0xFF, count: binaryDimension / 8)])
        let hits = try store.hybridSearch(
            vector: [1] + [Float](repeating: 0, count: dimension - 1),
            query: "tail", topK: 5, binary: BinaryCoarseOptions(
                signature: Data(repeating: 0xFF, count: binaryDimension / 8),
                candidateLimit: 20))
        XCTAssertTrue(hits.contains { $0.text == "tail" })
        store.close()
    }

    func testTailCountRefreshesAfterWrites() throws {
        let store = try makeStore("tail-count")
        try seed(store, count: 40)
        _ = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
        XCTAssertEqual(store.binaryANNIndexStatus().tailRows, 0)

        _ = try store.add(
            chunks: [Chunk(text: "tail", source: "tail.md", chunkIndex: 0)],
            vectors: [[1] + [Float](repeating: 0, count: dimension - 1)],
            binarySignatures: [signature(bit: 0)])
        XCTAssertEqual(store.binaryANNIndexStatus().tailRows, 1)
        XCTAssertTrue(store.binaryANNIndexStatus().isUsable)

        let rows = try store.db.query("SELECT rowid FROM chunks WHERE text = 'tail'")
        let rowID = try XCTUnwrap(rows.first?["rowid"]?.intValue)
        _ = try store.deleteRowIDs([rowID])
        XCTAssertEqual(store.binaryANNIndexStatus().tailRows, 0)
        XCTAssertTrue(store.binaryANNIndexStatus().isReady)
        store.close()
    }

    func testDeletedRowsAreExcludedFromBannResults() throws {
        let store = try makeStore("deleted")
        try seed(store, count: 20)
        _ = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
        _ = try store.deleteBySource("s3.md")
        let hits = try store.hybridSearch(
            vector: [1] + [Float](repeating: 0, count: dimension - 1),
            query: "zzz", topK: 5,
            binary: BinaryCoarseOptions(signature: signature(bit: 3), candidateLimit: 20))
        XCTAssertFalse(hits.contains { $0.source == "s3.md" })
        store.close()
    }

    func testStageTwoUsesRerankMirror() throws {
        let store = try makeStore("rerank-stage2")
        try seed(store, count: 20)
        _ = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
        XCTAssertEqual(store.binaryIndexStatus().stage2Backend, "rerank")
        store.close()
    }
}
