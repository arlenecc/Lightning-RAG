import XCTest
import Foundation
@testable import BaseAgentCore

/// 诊断：vec0 表上反复 DELETE+INSERT（fillPendingVector 的幂等重填模式）
/// 是否会丢失行。对应线上事故：1925 条全量重填后主向量影子表只剩 720 行。
final class Vec0RefillIntegrityTests: XCTestCase {
    func testRepeatedDeleteInsertDoesNotLoseVec0Rows() throws {
        let dir = NSTemporaryDirectory() + "vec0-refill-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let store = try VectorStore(dbPath: dir + "/t.sqlite", dimension: 128,
                                    enableSQLiteVec: true, elementType: .int8)
        defer { store.close() }

        let chunks = (0..<100).map { Chunk(text: "正文\($0)", source: "a.md", chunkIndex: $0) }
        let rowids = try store.addTextsOnly(chunks: chunks)
        XCTAssertEqual(rowids.count, 100)

        // 首轮回填
        for (i, rid) in rowids.enumerated() {
            var v = [Float](repeating: 0, count: 128); v[i % 128] = 1
            try store.fillPendingVector(rowid: rid, vector: v)
        }
        XCTAssertEqual(try store.enqueueMissingVectors(), 0, "首轮回填后不得有缺失")

        // 幂等重填 3 轮（模拟反复 reload 误判后的全量重填）
        for round in 1...3 {
            for (i, rid) in rowids.enumerated() {
                var v = [Float](repeating: 0, count: 128); v[(i + round) % 128] = 1
                try store.fillPendingVector(rowid: rid, vector: v)
            }
            XCTAssertEqual(try store.enqueueMissingVectors(), 0,
                           "第 \(round) 轮重填后不得有行丢失")
        }

        // KNN 实际可检索性：单位向量精确命中
        var probe = [Float](repeating: 0, count: 128); probe[7] = 1
        let hits = try store.search(vector: probe, topK: 100)
        XCTAssertEqual(hits.count, 100, "KNN 必须能取回全部 100 行")
    }
}
