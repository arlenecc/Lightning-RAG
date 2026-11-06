import XCTest
@testable import BaseAgentCore

/// P1-b：数据安全性与精确扫描内存。
final class RAGDataSafetyTests: XCTestCase {

    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-p1b-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    /// 生成 >4MB 文件：头/中/尾采样区全是 'A'，只在 1MB 处（采样盲区）
    /// 写一个标记字节。两个文件采样指纹相同，内容确实不同。
    private func largeFile(_ name: String, marker: UInt8) throws -> String {
        let size = 4 * 1024 * 1024 + 4096
        var data = Data(repeating: 0x41, count: size)
        data[1_000_000] = marker
        let path = directory.appendingPathComponent(name).path
        try data.write(to: URL(fileURLWithPath: path))
        return path
    }

    func testSampledFingerprintCollisionDoesNotDropDistinctFiles() throws {
        let a = try largeFile("a.bin", marker: 0x42)
        let b = try largeFile("b.bin", marker: 0x43)
        XCTAssertEqual(RAGEngine.sourceFingerprint(a), RAGEngine.sourceFingerprint(b),
                       "前置条件：两个不同文件的采样指纹必须相同")
        XCTAssertNotEqual(RAGEngine.fullContentHash(a), RAGEngine.fullContentHash(b))

        let files = [
            ScannedFile(path: a, relativePath: "a.bin"),
            ScannedFile(path: b, relativePath: "b.bin"),
        ]
        let result = RAGEngine.groupDuplicateFiles(files, manifest: [:],
                                                   force: true, verifyFingerprint: true)
        XCTAssertEqual(result.unique.count, 2, "采样盲区不同的文件必须都参与解析")
        XCTAssertTrue(result.groups.isEmpty, "不能被当成副本清掉向量")
    }

    func testIdenticalLargeFilesAreStillDeduplicated() throws {
        let a = try largeFile("a.bin", marker: 0x42)
        let b = try largeFile("b.bin", marker: 0x42)
        let files = [
            ScannedFile(path: a, relativePath: "a.bin"),
            ScannedFile(path: b, relativePath: "b.bin"),
        ]
        let result = RAGEngine.groupDuplicateFiles(files, manifest: [:],
                                                   force: true, verifyFingerprint: true)
        XCTAssertEqual(result.unique.count, 1)
        XCTAssertEqual(result.groups.count, 1, "真正的副本仍应合并，不能修过头")
    }

    func testFullContentHashFailsClosedForUnreadableFile() {
        XCTAssertEqual(RAGEngine.fullContentHash(
            directory.appendingPathComponent("missing.bin").path), "")
    }

    func testExactScanBatchesAllRowsWithoutLoadingWholeTable() throws {
        // enableSQLiteVec=false → 普通表走全量精确扫描；插入数超过单批上限，
        // 旧实现会一次性 `SELECT rowid, embedding` 把整表读进内存。
        let store = try VectorStore(
            dbPath: directory.appendingPathComponent("index.sqlite").path,
            dimension: 2, enableSQLiteVec: false)
        let count = 460
        var chunks: [Chunk] = []
        var vectors: [[Float]] = []
        for i in 0..<count {
            chunks.append(Chunk(text: "doc-\(i)", source: "s\(i).md", chunkIndex: 0))
            vectors.append([Float(i) / 1000.0, 1.0])
        }
        XCTAssertEqual(try store.add(chunks: chunks, vectors: vectors), count)

        let hits = try store.search(vector: [0.5, 1.0], topK: 4)
        XCTAssertEqual(hits.count, 4, "所有批次都要参与评分，不能只读前一批")
        XCTAssertEqual(hits.first?.text, "doc-459",
                       "最后一批的最近邻必须能被找到")
        XCTAssertTrue(zip(hits, hits.dropFirst()).allSatisfy { $0.score >= $1.score })
    }

    func testHybridSearchAcceptsIndependentFTSOverfetch() throws {
        let store = try VectorStore(
            dbPath: directory.appendingPathComponent("hybrid.sqlite").path,
            dimension: 2, enableSQLiteVec: false)
        let chunks = [
            Chunk(text: "apple banana fruit", source: "a.md", chunkIndex: 0),
            Chunk(text: "orange apple juice", source: "b.md", chunkIndex: 0),
            Chunk(text: "apple pie recipe", source: "c.md", chunkIndex: 0),
        ]
        XCTAssertEqual(try store.add(chunks: chunks,
                                     vectors: [[1, 0], [0.8, 0.2], [0.6, 0.4]]), 3)
        let low = try store.hybridSearch(vector: [1, 0], query: "apple",
                                         topK: 2, ftsOverfetch: 1)
        let high = try store.hybridSearch(vector: [1, 0], query: "apple",
                                          topK: 2, ftsOverfetch: 4)
        XCTAssertEqual(low.count, 2)
        XCTAssertEqual(high.count, 2)
        XCTAssertEqual(low.first?.text, high.first?.text,
                       "小库上两种候选倍数不应改变 Top1，只影响候选池大小")
    }
}
