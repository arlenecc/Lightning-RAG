import XCTest
import Foundation
@testable import BaseAgentCore

/// 诊断（块回收）：整块被删空回收后重新写入，是否丢行。
/// 复现线上模式：按 source upsert（先删全部旧行→块清空回收→再插全部新行），
/// 对应今天反复重新同步 + 反复全量重填的破坏场景。
final class Vec0ChunkReclaimIntegrityTests: XCTestCase {
    func testDeleteAllThenReaddDoesNotLoseRows() throws {
        let dir = NSTemporaryDirectory() + "vec0-reclaim-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let store = try VectorStore(dbPath: dir + "/t.sqlite", dimension: 128,
                                    enableSQLiteVec: true, elementType: .int8,
                                    binaryDimension: 768, binarySource: "test")
        defer { store.close() }

        func makeChunks(_ round: Int) -> [Chunk] {
            (0..<2000).map { Chunk(text: "正文\(round)-\($0)",
                                   source: "src\(round).md", chunkIndex: $0) }
        }
        func vec(_ i: Int) -> [Float] {
            var v = [Float](repeating: 0, count: 128); v[i % 128] = 1; return v
        }

        for round in 0..<3 {
            let chunks = makeChunks(round)
            let vectors = (0..<chunks.count).map { vec($0 + round) }
            let signatures = (0..<chunks.count).map { i -> Data in
                Data((0..<96).map { UInt8((i + round + $0) & 0xFF) })
            }
            let written = try store.add(chunks: chunks, vectors: vectors,
                                        binarySignatures: signatures)
            XCTAssertEqual(written, 2000, "第 \(round) 轮写入数不符")
            XCTAssertEqual(try store.countVectorRows(), 2000,
                           "第 \(round) 轮主向量影子表行数不符")
            XCTAssertEqual(store.binaryRows, 2000, "第 \(round) 轮 bin 行数不符")

            // KNN 全量可检索
            var probe = [Float](repeating: 0, count: 128); probe[5] = 1
            let hits = try store.search(vector: probe, topK: 2000)
            XCTAssertEqual(hits.count, 2000, "第 \(round) 轮 KNN 行数不符")

            // 按 source 删除全部（触发块清空回收），下一轮重新写入
            let removed = try store.deleteBySource("src\(round).md")
            XCTAssertEqual(removed, 2000)
            XCTAssertEqual(try store.countVectorRows(), 0, "删除后影子表应为空")
            XCTAssertEqual(store.binaryRows, 0)
        }
    }
}
