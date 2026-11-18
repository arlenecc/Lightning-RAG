import XCTest
import Foundation
@testable import BaseAgentCore
@testable import ColBERTReader

/// 检索阅读模式审查修复项的行为回归（不依赖真实 ColBERT 模型）。
final class ColbertFixRegressionTests: XCTestCase {

    // MARK: - ⑤-a 稳定 rowID / 真实 chunkIndex

    func testStableRowIDIsDeterministicAndNonNegative() {
        let a = RAGEngine.stableRowID("abc123#0007")
        let b = RAGEngine.stableRowID("abc123#0007")
        XCTAssertEqual(a, b, "同一 docID 必须得到稳定 rowID（不像 String.hashValue 每进程随机）")
        XCTAssertGreaterThanOrEqual(a, 0, "rowID 必须非负，避免 abs(Int.min) 溢出陷阱")
        XCTAssertNotEqual(RAGEngine.stableRowID("abc123#0007"),
                          RAGEngine.stableRowID("abc123#0008"),
                          "不同 docID 应映射到不同 rowID")
    }

    func testColbertChunkIndexParsesRealOrdinal() {
        XCTAssertEqual(RAGEngine.colbertChunkIndex("a1b2c3d4#0042", fallback: 9), 42)
        XCTAssertEqual(RAGEngine.colbertChunkIndex("deadbeef#0000", fallback: 9), 0)
        XCTAssertEqual(RAGEngine.colbertChunkIndex("no-hash-suffix", fallback: 9), 9,
                       "无法解析时回退到名次")
        XCTAssertEqual(RAGEngine.colbertChunkIndex("x#notnum", fallback: 3), 3)
    }

    // MARK: - ⑤-c 残差损坏记录防御

    private let dim = 64
    private func queryTokens() -> [[Float]] {
        var q = [[Float]](repeating: [Float](repeating: 0, count: dim), count: 2)
        q[0][0] = 1; q[1][1] = 1
        return q
    }

    func testTruncatedResidualBlobThrowsInsteadOfCrashing() throws {
        let good = try ResidualCodec.quantize(
            docID: "doc", vectors: [[Float](repeating: 0.5, count: dim)], dim: dim, bits: 4)
        let truncated = ResidualRecord(
            docID: good.docID, tokenCount: good.tokenCount, dim: good.dim, bits: good.bits,
            scale: good.scale, offset: good.offset, centerFP16: good.centerFP16,
            blob: Data(good.blob.prefix(1)))
        XCTAssertThrowsError(
            try ResidualCodec.maxSim(query: queryTokens(), record: truncated),
            "截断的 blob 必须抛 quant 错误，而不是数组越界 trap")
    }

    func testWrongDimensionCenterThrows() throws {
        let good = try ResidualCodec.quantize(
            docID: "doc", vectors: [[Float](repeating: 0.5, count: dim)], dim: dim, bits: 4)
        let bad = ResidualRecord(
            docID: good.docID, tokenCount: good.tokenCount, dim: good.dim, bits: good.bits,
            scale: good.scale, offset: good.offset,
            centerFP16: Array(good.centerFP16.dropLast()), blob: good.blob)
        XCTAssertThrowsError(try ResidualCodec.maxSim(query: queryTokens(), record: bad))
    }

    // MARK: - 低优先项 ④ LIKE 通配符转义

    func testLikePatternEscapesWildcards() {
        XCTAssertEqual(SearchEngine.likePattern("report.pdf"), "%report.pdf%")
        XCTAssertEqual(SearchEngine.likePattern("a%b_c\\d"), "%a\\%b\\_c\\\\d%",
                       "% / _ / \\ 必须转义，避免放大 LIKE 候选集")
        XCTAssertEqual(SearchEngine.likePattern(""), "%%")
    }

    // MARK: - 低优先项 ⑤ contentHash 扩位

    func testContentHashIs128BitAndStable() {
        let h = RAGEngine.contentHash("同一段正文")
        XCTAssertEqual(h.count, 32, "128 位 = 32 个十六进制字符，远低于 32 位时的碰撞概率")
        XCTAssertEqual(h, RAGEngine.contentHash("同一段正文"))
        XCTAssertNotEqual(h, RAGEngine.contentHash("另一段正文"))
        XCTAssertTrue(h.allSatisfy { $0.isHexDigit })
    }
}
