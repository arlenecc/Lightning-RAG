import XCTest
import Foundation
@testable import BaseAgentCore

/// 12 条真实候选取样下的 ONNX reranker 批处理回归基准。
///
/// 目的：在生产候选池规模（12 条）上对比 `batchSize` 与长度分桶的效果，
/// 回答「分桶批处理后 batch8 是否仍比 batch1 慢」。
///
/// 真实候选来自本地真实索引的 `chunks` 表（只读），按 rowid 等距取样 12 条，
/// 跨书、长度分布保持真实；不依赖召回质量，只关心重排耗时与分数一致性。
///
/// 环境变量：
/// - `RERANK_BENCH=1`：启用（与其它 reranker 基准一致）
/// - `RERANK_BENCH_KB`：真实索引路径，
///   默认 `~/app-space/rag_data/test-d9db5851/index.sqlite`
/// - `RERANK_BENCH_QUERY`：查询文本，默认「绿松石如何辨别真假？」
/// - `RERANK_BENCH_RUNS`：每个方案的计时轮数，默认 7
final class RerankerBatchingBenchmarkTests: XCTestCase {

    private static let modelName = "jina-reranker-v2-base-multilingual-onnx"
    private static let candidateCount = 12
    private static let defaultIndex = ("~/app-space/rag_data/test-d9db5851/index.sqlite"
        as NSString).expandingTildeInPath

    func testTwelveRealCandidatesBatchComparison() async throws {
        guard ProcessInfo.processInfo.environment["RERANK_BENCH"] == "1" else {
            throw XCTSkip("未设置 RERANK_BENCH=1，跳过真实模型基准测试")
        }
        let env = ProcessInfo.processInfo.environment
        let directory = EmbeddingBenchmarkTests.resolveModelsRoot()
            .appendingPathComponent(Self.modelName, isDirectory: true)
        guard FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("model_quantized.onnx").path) else {
            throw XCTSkip("模型缺失：\(directory.path)")
        }

        let indexPath = env["RERANK_BENCH_KB"] ?? Self.defaultIndex
        guard FileManager.default.fileExists(atPath: indexPath) else {
            throw XCTSkip("真实索引不存在：\(indexPath)")
        }
        let query = env["RERANK_BENCH_QUERY"] ?? "绿松石如何辨别真假？"
        let candidates = try Self.realCandidates(indexPath: indexPath,
                                                 limit: Self.candidateCount)
        guard candidates.count == Self.candidateCount else {
            throw XCTSkip("候选不足 \(Self.candidateCount) 条：\(candidates.count)")
        }

        let batch1 = ONNXReranker(directory: directory, modelName: Self.modelName + "-b1",
                                  batchSize: 1)
        let batch8 = ONNXReranker(directory: directory, modelName: Self.modelName + "-b8",
                                  batchSize: 8)
        defer { Task { await batch1.unload(); await batch8.unload() } }

        let runs = max(3, Int(env["RERANK_BENCH_RUNS"] ?? "") ?? 7)
        // 预热：首次包含 session 建图/内存分配，只记录耗时、不计入统计。
        let warmStart1 = Date()
        _ = try await batch1.rerank(query: query, documents: candidates)
        let warm1 = Date().timeIntervalSince(warmStart1) * 1000
        let warmStart8 = Date()
        _ = try await batch8.rerank(query: query, documents: candidates)
        let warm8 = Date().timeIntervalSince(warmStart8) * 1000

        let times1 = try await Self.measure(runs: runs) {
            try await batch1.rerank(query: query, documents: candidates)
        }
        let times8 = try await Self.measure(runs: runs) {
            try await batch8.rerank(query: query, documents: candidates)
        }
        let scores1 = try await batch1.rerank(query: query, documents: candidates)
        let scores8 = try await batch8.rerank(query: query, documents: candidates)

        XCTAssertEqual(scores1.count, Self.candidateCount)
        XCTAssertEqual(scores8.count, Self.candidateCount)
        let top1 = Self.topIndex(scores1)
        let top8 = Self.topIndex(scores8)
        XCTAssertEqual(top1, top8, "批处理不应改变 top1：b1=\(scores1) b8=\(scores8)")
        let drift = zip(scores1, scores8).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(drift, 0.15, "批大小/分桶带来的分数漂移应可控：\(drift)")

        let report = Self.report(query: query, candidates: candidates,
                                 times1: times1, times8: times8,
                                 drift: drift, top1: top1,
                                 warm1: warm1, warm8: warm8)
        try report.write(toFile: "/tmp/rerank_bucket_bench.md", atomically: true,
                         encoding: .utf8)
        print(report)
    }

    // MARK: - 取样与计时

    /// 从真实索引读 12 条真实切片：按 rowid 排序后等距取样，跨书且长度分布真实。
    private static func realCandidates(indexPath: String, limit: Int) throws -> [String] {
        let db = try SQLiteDatabase(path: indexPath, readOnly: true)
        let rows = try db.query(
            "SELECT text FROM chunks WHERE length(text) BETWEEN 80 AND 1600 ORDER BY rowid")
        let texts = rows.compactMap { $0["text"]?.stringValue }
        guard texts.count >= limit else { return texts }
        return (0..<limit).map { index in
            let position = Int(Double(index) * Double(texts.count - 1) / Double(limit - 1))
            return texts[position]
        }
    }

    private static func measure(
        runs: Int, _ body: () async throws -> [Double]
    ) async throws -> [Double] {
        var times: [Double] = []
        times.reserveCapacity(runs)
        for _ in 0..<runs {
            let start = Date()
            _ = try await body()
            times.append(Date().timeIntervalSince(start) * 1000)
        }
        return times
    }

    private static func topIndex(_ scores: [Double]) -> Int? {
        scores.indices.max { scores[$0] < scores[$1] }
    }

    private static func report(
        query: String, candidates: [String], times1: [Double], times8: [Double],
        drift: Double, top1: Int?, warm1: Double, warm8: Double
    ) -> String {
        func median(_ values: [Double]) -> Double {
            let sorted = values.sorted()
            return sorted[sorted.count / 2]
        }
        func fmt(_ values: [Double]) -> String {
            String(format: "%.1f", median(values))
        }
        let lengths = candidates.map(\.count)
        var report = "# ONNX reranker 12 条真实候选批处理基准\n\n"
        report += "- 查询：\(query)\n"
        report += "- 候选：真实索引 `chunks` 等距取样 \(candidates.count) 条，"
        report += "长度 min=\(lengths.min() ?? 0) / 中位=\(lengths.sorted()[lengths.count / 2]) "
        report += "/ max=\(lengths.max() ?? 0) 字\n"
        report += "- 每方案计时 \(times1.count) 轮，取中位数（预热不计）\n\n"
        report += "| 方案 | 单次耗时中位 (ms) | 全部轮次 (ms) |\n|---|---|---|\n"
        report += "| batchSize=1（逐条） | \(fmt(times1)) "
        report += "| \(times1.map { String(format: "%.1f", $0) }.joined(separator: ", ")) |\n"
        report += "| batchSize=8（长度分桶） | \(fmt(times8)) "
        report += "| \(times8.map { String(format: "%.1f", $0) }.joined(separator: ", ")) |\n"
        report += "\n- 首次（含加载/建图）：b1=\(String(format: "%.0f", warm1))ms "
        report += "b8=\(String(format: "%.0f", warm8))ms\n"
        report += "- 最大分数漂移：\(String(format: "%.4f", drift))，top1=\(top1.map(String.init) ?? "-")\n"
        return report
    }
}
