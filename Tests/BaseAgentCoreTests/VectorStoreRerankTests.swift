import XCTest
@testable import BaseAgentCore

final class VectorStoreRerankTests: XCTestCase {
    private var directory: URL!
    private let dimension = 16

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rerank-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func makeStore(_ name: String, type: VectorElementType = .int8) throws -> VectorStore {
        try VectorStore(dbPath: directory.appendingPathComponent("\(name).sqlite").path,
                        dimension: dimension, elementType: type,
                        binaryDimension: dimension, binarySource: "sig",
                        indexFingerprint: "fp-\(name)")
    }

    private func vector(_ index: Int) -> [Float] {
        var value = [Float](repeating: 0, count: dimension)
        value[index % dimension] = 1
        return value
    }

    func testAddMirrorsEncodedBytesIntoRerankTable() throws {
        let store = try makeStore("mirror")
        _ = try store.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0),
                     Chunk(text: "beta", source: "b.md", chunkIndex: 0)],
            vectors: [vector(0), vector(1)],
            binarySignatures: [Data([0xFF, 0x00]), Data([0x00, 0xFF])])
        XCTAssertTrue(store.rerankStorageEnabled)
        XCTAssertEqual(store.rerankRows, 2)
        let count = try store.db.scalar("SELECT COUNT(*) FROM chunk_vectors_rerank").intValue
        XCTAssertEqual(count, 2)
        store.close()
    }

    func testDeleteAndClearResetRerankCounter() throws {
        let store = try makeStore("delete")
        _ = try store.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0)],
            vectors: [vector(0)],
            binarySignatures: [Data([0xFF, 0x00])])
        let rowID = try XCTUnwrap(try store.rowIDs(source: "a.md").first)
        _ = try store.deleteRowIDs([rowID])
        XCTAssertEqual(store.rerankRows, 0)
        try store.clear()
        XCTAssertEqual(try store.db.scalar("SELECT COUNT(*) FROM chunk_vectors_rerank").intValue, 0)
        store.close()
    }

    func testReopenKeepsRerankAndRebuildDropsIt() throws {
        let path = directory.appendingPathComponent("reopen.sqlite").path
        var store: VectorStore? = try VectorStore(
            dbPath: path, dimension: dimension, elementType: .int8,
            binaryDimension: dimension, binarySource: "sig", indexFingerprint: "fp")
        _ = try store!.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0)],
            vectors: [vector(0)], binarySignatures: [Data([0xFF, 0x00])])
        XCTAssertEqual(store!.rerankRows, 1)
        store!.close()

        store = try VectorStore(
            dbPath: path, dimension: dimension, elementType: .int8,
            binaryDimension: dimension, binarySource: "sig", indexFingerprint: "fp")
        XCTAssertEqual(store!.rerankRows, 1)
        _ = try store!.rebuildForCurrentConfiguration()
        XCTAssertEqual(store!.rerankRows, 0)
        store!.close()
    }

    func testBinMainTypeDisablesRerankMirror() throws {
        let store = try makeStore("bin-main", type: .bin)
        XCTAssertFalse(store.rerankStorageEnabled)
        XCTAssertEqual(store.rerankRows, 0)
        store.close()
    }

    func testExactVectorsPrefersRerankWhenMainRowIsDeleted() throws {
        let store = try makeStore("prefer-rerank")
        _ = try store.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0)],
            vectors: [vector(0)], binarySignatures: [Data([0xFF, 0x00])])
        let rowID = try XCTUnwrap(try store.rowIDs(source: "a.md").first)
        try store.db.execute("DELETE FROM \(VectorStore.vectorTable) WHERE rowid = ?",
                             [.int(rowID)])
        let decoded = try store.exactVectorsLocked(rowIDs: [rowID])
        XCTAssertEqual(decoded[rowID]?.count, dimension)
        XCTAssertEqual(Double(decoded[rowID]?[0] ?? 0), 1, accuracy: 0.02)
        store.close()
    }

    func testExactVectorsFallsBackForMissingRerankRows() throws {
        let store = try makeStore("fallback-rerank")
        _ = try store.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0),
                     Chunk(text: "beta", source: "b.md", chunkIndex: 0)],
            vectors: [vector(0), vector(1)],
            binarySignatures: [Data([0xFF, 0x00]), Data([0x00, 0xFF])])
        let ids = try store.rowIDs(source: "a.md") + store.rowIDs(source: "b.md")
        try store.db.execute("DELETE FROM \(VectorStore.rerankVectorTable) WHERE rowid = ?",
                             [.int(ids[0])])
        let decoded = try store.exactVectorsLocked(rowIDs: ids)
        XCTAssertEqual(decoded.count, 2)
        XCTAssertEqual(decoded[ids[0]]?.count, dimension)
        store.close()
    }

    func testExactVectorsReadsMoreThanOneBatch() throws {
        let store = try makeStore("batches")
        var chunks: [Chunk] = []
        var vectors: [[Float]] = []
        var signatures: [Data] = []
        for index in 0..<300 {
            chunks.append(Chunk(text: "chunk \(index)", source: "s\(index).md", chunkIndex: 0))
            vectors.append(vector(index))
            signatures.append(Data(repeating: UInt8(index % 255), count: dimension / 8))
        }
        _ = try store.add(chunks: chunks, vectors: vectors, binarySignatures: signatures)
        let ids = try store.db.query("SELECT rowid FROM chunks").compactMap { $0["rowid"]?.intValue }
        let decoded = try store.exactVectorsLocked(rowIDs: Array(ids.reversed()))
        XCTAssertEqual(decoded.count, 300)
        store.close()
    }
}
