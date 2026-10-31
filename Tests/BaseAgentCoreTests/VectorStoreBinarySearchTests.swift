import XCTest
@testable import BaseAgentCore

/// 两阶段二值检索：stage 1 汉明粗筛、stage 2 余弦精排、ANN 旁路与回退。
final class VectorStoreBinarySearchTests: XCTestCase {
    private var directory: URL!
    private let dimension = 16
    private let binaryDimension = 16

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-binary-search-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func dbPath(_ name: String) -> String {
        directory.appendingPathComponent("\(name).sqlite").path
    }

    private func vector(_ first: Float, _ second: Float) -> [Float] {
        Array(repeating: first, count: 8) + Array(repeating: second, count: 8)
    }

    private func makeStore(_ name: String) throws -> VectorStore {
        try VectorStore(dbPath: dbPath(name), dimension: dimension,
                        elementType: .int8, binaryDimension: binaryDimension,
                        binarySource: "sig", indexFingerprint: "fp-\(name)")
    }

    /// 查询 [1×8, -1×8]；A 与查询同签名但余弦低，B 翻转 1 bit 但余弦最高，
    /// C 与查询完全相反。用于验证 stage 2 不是只按汉明距离排序。
    private func fixture() -> (query: [Float], a: [Float], b: [Float], c: [Float],
                               querySignature: Data, signatures: [Data]) {
        let query = vector(1, -1)
        let a = Array(repeating: Float(0.01), count: 8)
            + Array(repeating: Float(-1), count: 8)
        let b = Array(repeating: Float(1), count: 8)
            + Array(repeating: Float(-1), count: 7) + [Float(1)]
        let c = vector(-1, 1)
        return (query, a, b, c,
                Data([0xFF, 0x00]),
                [Data([0xFF, 0x00]), Data([0xFF, 0x80]), Data([0x00, 0xFF])])
    }

    func testTwoStageSearchRanksByCosineAndSetsBinaryScores() throws {
        let store = try makeStore("two-stage")
        let fixture = fixture()
        _ = try store.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0),
                     Chunk(text: "beta", source: "b.md", chunkIndex: 0),
                     Chunk(text: "gamma", source: "c.md", chunkIndex: 0)],
            vectors: [fixture.a, fixture.b, fixture.c],
            binarySignatures: fixture.signatures)
        let hits = try store.hybridSearch(
            vector: fixture.query, query: "zzz", topK: 3,
            binary: BinaryCoarseOptions(signature: fixture.querySignature,
                                        candidateLimit: 200))
        XCTAssertEqual(hits.map(\.text), ["beta", "alpha", "gamma"])
        XCTAssertEqual(hits[0].vectorScore ?? 0, 0.875, accuracy: 0.001)
        XCTAssertEqual(hits[0].binaryScore ?? 0, 0.9375, accuracy: 0.001)
        XCTAssertEqual(hits[1].vectorScore ?? 0, 0.714, accuracy: 0.01)
        XCTAssertEqual(hits[1].binaryScore ?? 0, 1.0, accuracy: 0.0001)
        XCTAssertNil(hits[0].ftsScore)
        XCTAssertEqual(store.annQueryCount, 0, "二值可用时不应查询 ANN")
        store.close()
    }

    /// stage 1 只保留 20 个汉明最近的候选；FTS-only 命中不在候选池内，
    /// 仍必须通过 BM25/RRF 进入最终结果，且 `binaryScore` 为 nil。
    func testStageOneCandidatesLeaveFtsOnlyHitWithoutBinaryScore() throws {
        let store = try makeStore("fts-fusion")
        let query = vector(1, -1)
        let querySignature = Data([0xFF, 0x00])
        var chunks: [Chunk] = []
        var vectors: [[Float]] = []
        var signatures: [Data] = []
        for index in 0..<22 {
            chunks.append(Chunk(text: "filler \(index)", source: "f\(index).md",
                                chunkIndex: 0))
            vectors.append(query)
            signatures.append(querySignature)
        }
        chunks.append(Chunk(text: "needle", source: "needle.md", chunkIndex: 0))
        vectors.append(vector(-1, 1))
        signatures.append(Data([0x00, 0xFF]))
        for index in 22..<25 {
            chunks.append(Chunk(text: "tail \(index)", source: "t\(index).md",
                                chunkIndex: 0))
            vectors.append(vector(-1, 1))
            signatures.append(Data([0x00, 0xFF]))
        }
        _ = try store.add(chunks: chunks, vectors: vectors,
                          binarySignatures: signatures)

        let hits = try store.hybridSearch(
            vector: query, query: "needle", topK: 10,
            binary: BinaryCoarseOptions(signature: querySignature,
                                        candidateLimit: 20))
        let needle = try XCTUnwrap(hits.first { $0.text == "needle" })
        XCTAssertNil(needle.binaryScore)
        XCTAssertNil(needle.vectorScore)
        XCTAssertNotNil(needle.ftsScore)
        store.close()
    }

    func testBinaryFallsBackWhenCoverageIncomplete() throws {
        let store = try makeStore("coverage-fallback")
        let query = vector(1, -1)
        _ = try store.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0),
                     Chunk(text: "beta", source: "b.md", chunkIndex: 0)],
            vectors: [query, vector(-1, 1)])
        XCTAssertEqual(store.binaryIndexStatus().coverage, 0)

        let hits = try store.hybridSearch(
            vector: query, query: "zzz", topK: 2,
            binary: BinaryCoarseOptions(signature: Data([0xFF, 0x00]),
                                        candidateLimit: 200),
            ann: ANNQueryOptions(nprobe: 2, rescore: 0))
        XCTAssertEqual(hits.count, 2)
        XCTAssertTrue(hits.allSatisfy { $0.binaryScore == nil })
        XCTAssertGreaterThan(store.annQueryCount, 0, "覆盖不足时应回退 ANN/精确路径")
        store.close()
    }

    func testBinaryFallsBackOnQuerySignatureLengthMismatch() throws {
        let store = try makeStore("length-fallback")
        let fixture = fixture()
        _ = try store.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0),
                     Chunk(text: "beta", source: "b.md", chunkIndex: 0)],
            vectors: [fixture.a, fixture.b],
            binarySignatures: [fixture.signatures[0], fixture.signatures[1]])

        let hits = try store.hybridSearch(
            vector: fixture.query, query: "zzz", topK: 2,
            binary: BinaryCoarseOptions(signature: Data([0xFF]), candidateLimit: 200),
            ann: ANNQueryOptions(nprobe: 2, rescore: 0))
        XCTAssertEqual(hits.count, 2)
        XCTAssertTrue(hits.allSatisfy { $0.binaryScore == nil })
        XCTAssertGreaterThan(store.annQueryCount, 0)
        store.close()
    }

    func testTargetedSearchIgnoresBinaryOptions() throws {
        let store = try makeStore("targeted")
        let fixture = fixture()
        _ = try store.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0),
                     Chunk(text: "beta", source: "b.md", chunkIndex: 0)],
            vectors: [fixture.a, fixture.b],
            binarySignatures: [fixture.signatures[0], fixture.signatures[1]])
        let options = BinaryCoarseOptions(signature: fixture.querySignature,
                                          candidateLimit: 200)

        let filtered = try store.hybridSearch(
            vector: fixture.query, query: "zzz", topK: 2,
            sourceFilter: "a.md", binary: options)
        XCTAssertEqual(filtered.map(\.text), ["alpha"])
        XCTAssertNil(filtered.first?.binaryScore)

        let contains = try store.hybridSearch(
            vector: fixture.query, query: "zzz", topK: 2,
            sourceContains: "a", binary: options)
        XCTAssertEqual(contains.map(\.text), ["alpha"])
        XCTAssertNil(contains.first?.binaryScore)
        store.close()
    }

    func testBinaryAvailableDoesNotQueryANN() throws {
        let store = try makeStore("ann-bypass")
        let fixture = fixture()
        _ = try store.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0),
                     Chunk(text: "beta", source: "b.md", chunkIndex: 0)],
            vectors: [fixture.a, fixture.b],
            binarySignatures: [fixture.signatures[0], fixture.signatures[1]])
        let ann = ANNQueryOptions(nprobe: 2, rescore: 0)

        let binaryHits = try store.hybridSearch(
            vector: fixture.query, query: "zzz", topK: 2,
            binary: BinaryCoarseOptions(signature: fixture.querySignature,
                                        candidateLimit: 200),
            ann: ann)
        XCTAssertFalse(binaryHits.isEmpty)
        XCTAssertEqual(store.annQueryCount, 0)

        _ = try store.hybridSearch(
            vector: fixture.query, query: "zzz", topK: 2,
            ann: ann)
        XCTAssertEqual(store.annQueryCount, 1, "不带二值时仍应走原 ANN 通道")
        store.close()
    }
}
