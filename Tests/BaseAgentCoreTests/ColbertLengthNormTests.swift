import XCTest
import Foundation
@testable import BaseAgentCore
@testable import ColBERTReader

/// MaxSim 长度归一化：防止「包含少量查询词的长分片」碾压「主题精确命中的短分片」。
///
/// 回归：检索阅读分片把多个 300-token 段拼成一条 record（实测 166~5122 token），
/// sum-of-max 在长 record 上取 max 系统性偏高，相关性差的长片会挤掉相关短片。
final class ColbertLengthNormTests: XCTestCase {
    private let dim = 64

    /// query 向量：3 个 token，各是单位基向量 e0/e1/e2。
    private func queryTokens() -> [[Float]] {
        var q = [[Float]](repeating: [Float](repeating: 0, count: dim), count: 3)
        q[0][0] = 1; q[1][1] = 1; q[2][2] = 1
        return q
    }

    /// 构造分片向量：hits 个 token 精确匹配 query 基向量，其余为随机正交噪声（低分）。
    private func docVectors(hits: Int, noise: Int) -> [[Float]] {
        var out: [[Float]] = []
        for i in 0..<hits {
            var v = [Float](repeating: 0, count: dim)
            v[i % 3] = 1
            out.append(v)
        }
        for j in 0..<noise {
            var v = [Float](repeating: 0, count: dim)
            // 噪声集中在 e10..e20，与 query 的 e0/e1/e2 近正交
            v[10 + (j % 10)] = 1
            out.append(v)
        }
        return out
    }

    func testLongDilutedChunkDoesNotBeatShortRelevantChunk() throws {
        let q = queryTokens()
        // 短片：3 个 token 全部精确命中（主题高度相关）
        let short = try ResidualCodec.quantize(docID: "short", vectors: docVectors(hits: 3, noise: 0), dim: dim)
        // 长片：同样 3 个命中 token，但掺了 1500 个无关噪声 token（主题稀释）
        let long = try ResidualCodec.quantize(docID: "long", vectors: docVectors(hits: 3, noise: 1500), dim: dim)

        let rawShort = try ResidualCodec.maxSim(query: q, record: short, normalizeLength: false)
        let rawLong = try ResidualCodec.maxSim(query: q, record: long, normalizeLength: false)
        let normShort = try ResidualCodec.maxSim(query: q, record: short, normalizeLength: true)
        let normLong = try ResidualCodec.maxSim(query: q, record: long, normalizeLength: true)

        // 量化有微小误差，命中 3 个 query 的 sum 理论都≈3，两者原始分接近；
        // 长度归一化后长片被压低，相关短片应相对更高。
        XCTAssertGreaterThan(normShort / normLong, rawShort / rawLong,
                             "归一化后应提升短片相对长片的分数比")
        XCTAssertLessThan(normLong, normShort,
                          "掺大量噪声的长片归一化后应低于精确命中的短片")
    }

    /// 中等长度片（接近 300 token 基准）几乎不受归一化影响。
    func testReferenceLengthChunkNearlyUnchanged() throws {
        let q = queryTokens()
        let rec = try ResidualCodec.quantize(docID: "base", vectors: docVectors(hits: 3, noise: 297), dim: dim)
        let raw = try ResidualCodec.maxSim(query: q, record: rec, normalizeLength: false)
        let norm = try ResidualCodec.maxSim(query: q, record: rec, normalizeLength: true)
        XCTAssertEqual(raw, norm, accuracy: 0.01, "300 token 基准片归一化前后应基本一致")
    }
}
