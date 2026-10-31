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
}
