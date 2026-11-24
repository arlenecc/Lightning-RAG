import XCTest
import Foundation
@testable import BaseAgentCore

/// 复现「极速检索模式切换抹掉主向量」：
/// 元数据连接（openForMetadata，enableSQLiteVec=false）打开既有 vec0 库时，
/// ensureVectorTable(useVec0:false) 会把 USING vec0 判为「类型不匹配」而
/// DROP 整张主向量表——切一次极速模式就丢一次全部主向量。
final class MetadataOpenMustNotWipeVectorsTests: XCTestCase {
    func testMetadataOpenPreservesVec0Vectors() throws {
        let dir = NSTemporaryDirectory() + "meta-open-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let path = dir + "/index.sqlite"
        let fingerprint = "local:test|dimension=128|matryoshka=0|element=int8"

        // 1) 向量模式写入 100 条主向量 + 签名
        let store = try VectorStore(dbPath: path, dimension: 128,
                                    enableSQLiteVec: true, elementType: .int8,
                                    binaryDimension: 768, binarySource: "local:test",
                                    indexFingerprint: fingerprint)
        let chunks = (0..<100).map { Chunk(text: "正文\($0)", source: "a.md", chunkIndex: $0) }
        let vectors = (0..<100).map { i -> [Float] in
            var v = [Float](repeating: 0, count: 128); v[i % 128] = 1; return v
        }
        let signatures = (0..<100).map { i -> Data in
            Data((0..<96).map { UInt8((i + $0) & 0xFF) })
        }
        XCTAssertEqual(try store.add(chunks: chunks, vectors: vectors,
                                     binarySignatures: signatures), 100)
        XCTAssertEqual(try store.countVectorRows(), 100)
        XCTAssertEqual(store.binaryRows, 100)
        store.close()

        // 2) 元数据连接打开（极速模式的目录/摘要/KG 路径）——不得动向量数据
        let meta = try VectorStore.openForMetadata(dbPath: path)
        meta.close()

        // 3) 重新以向量模式打开：主向量与签名必须原样保留
        let reopened = try VectorStore(dbPath: path, dimension: 128,
                                       enableSQLiteVec: true, elementType: .int8,
                                       binaryDimension: 768, binarySource: "local:test",
                                       indexFingerprint: fingerprint)
        defer { reopened.close() }
        XCTAssertEqual(try reopened.countVectorRows(), 100,
                       "元数据打开不得删除主向量表")
        XCTAssertEqual(reopened.binaryRows, 100, "元数据打开不得删除二值签名")
        XCTAssertNil(reopened.rebuildReason)
        XCTAssertTrue(reopened.vectorEnabled)
        var probe = [Float](repeating: 0, count: 128); probe[3] = 1
        XCTAssertEqual(try reopened.search(vector: probe, topK: 100).count, 100,
                       "主向量必须仍然可检索")
    }
}
