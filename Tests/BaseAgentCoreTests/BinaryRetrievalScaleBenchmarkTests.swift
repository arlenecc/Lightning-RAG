import XCTest
import Foundation
import Accelerate
@testable import BaseAgentCore

/// 大数据量二值粗筛 vs ANN vs 精确扫描的临时基准（默认跳过，不提交）。
/// 运行：`BINARY_BENCH=1 swift test --disable-sandbox \
///        --filter BinaryRetrievalScaleBenchmarkTests`
final class BinaryRetrievalScaleBenchmarkTests: XCTestCase {
    private struct SplitMix64 {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        mutating func unit() -> Float {
            Float(next() >> 11) / Float(1 << 53)
        }
        mutating func signed() -> Float { unit() * 2 - 1 }
    }

    // 主向量维度（落盘 int8）与二值签名位宽；默认对齐产品默认配置：
    // jina-v5 截断到 128 维落盘、签名用截断前原生 768 维。
    private static let dimension = Int(
        ProcessInfo.processInfo.environment["BINARY_BENCH_MAIN_DIM"] ?? "") ?? 128
    private static let signatureBits = Int(
        ProcessInfo.processInfo.environment["BINARY_BENCH_SIG_BITS"] ?? "") ?? 768
    private static let clusters = Int(
        ProcessInfo.processInfo.environment["BINARY_BENCH_CLUSTERS"] ?? "") ?? 96
    private static let clusterNoise: Float = 0.35

    private func makeCenters(rng: inout SplitMix64) -> [[Float]] {
        (0..<Self.clusters).map { _ in
            (0..<Self.signatureBits).map { _ in rng.signed() }
        }
    }

    /// 原生向量截断到主向量维度再归一化（模拟 jina-v5 的 Matryoshka 截断）。
    private static func mainVector(_ raw: [Float]) -> [Float] {
        VectorCodec.normalize(Array(raw.prefix(dimension)))
    }

    private func vector(center: [Float], rng: inout SplitMix64,
                        noise: Float) -> [Float] {
        center.map { $0 + rng.signed() * noise }
    }

    private static func percentile(_ values: [Double], _ p: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let index = min(sorted.count - 1, max(0, Int((Double(sorted.count) - 1) * p)))
        return sorted[index]
    }

    private static func recall(_ hits: [SearchHit],
                               groundTruth: Set<Int64>, k: Int) -> Double {
        guard !groundTruth.isEmpty else { return 0 }
        let prefix = hits.prefix(k).map(\.rowID)
        return Double(prefix.filter { groundTruth.contains($0) }.count)
            / Double(min(k, groundTruth.count))
    }

    private func runCase(rows: Int) throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("binary-bench-\(rows)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let dbPath = root.appendingPathComponent("index.sqlite").path
        let store = try VectorStore(
            dbPath: dbPath, dimension: Self.dimension, elementType: .int8,
            binaryDimension: Self.signatureBits, binarySource: "bench")
        defer { store.close() }

        var rng = SplitMix64(state: 0x1234_5678_9ABC_DEF0)
        let centers = makeCenters(rng: &rng)

        let batchSize = 5_000
        let insertStarted = Date()
        var inserted = 0
        while inserted < rows {
            let count = min(batchSize, rows - inserted)
            let source = "bench-source-\(inserted / batchSize)"
            var chunks: [Chunk] = []
            var vectors: [[Float]] = []
            var signatures: [Data] = []
            chunks.reserveCapacity(count)
            vectors.reserveCapacity(count)
            signatures.reserveCapacity(count)
            for offset in 0..<count {
                let index = inserted + offset
                let center = centers[index % centers.count]
                let raw = vector(center: center, rng: &rng, noise: Self.clusterNoise)
                chunks.append(Chunk(text: "bench chunk \(index) cluster \(index % centers.count)",
                                    source: source, chunkIndex: index))
                vectors.append(Self.mainVector(raw))
                signatures.append(VectorEncoder.encodeSignature(raw,
                                                                bitDimension: Self.signatureBits))
            }
            try store.add(chunks: chunks, vectors: vectors, binarySignatures: signatures)
            inserted += count
        }
        let insertSeconds = Date().timeIntervalSince(insertStarted)
        let status = store.binaryIndexStatus()
        XCTAssertTrue(status.isComplete, "二值签名覆盖率 \(status.coverage)")

        let benchANN = ProcessInfo.processInfo.environment["BINARY_BENCH_ANN"] != "0"
        let benchBann = ProcessInfo.processInfo.environment["BINARY_BENCH_BANN"] != "0"
        let annStarted = Date()
        let annResult = benchANN
            ? try store.buildANNIndex(clusters: 64, clusterSize: 512, iterations: 8)
            : ANNBuildResult()
        let annSeconds = Date().timeIntervalSince(annStarted)
        let candidateLimit = Int(
            ProcessInfo.processInfo.environment["BINARY_BENCH_CANDIDATES"] ?? "") ?? 200
        let benchTrue = ProcessInfo.processInfo.environment["BINARY_BENCH_TRUE"] == "1"
        var trueVectors: [(rowID: Int64, values: [Float], norm: Float)] = []
        if benchTrue {
            let rows = try store.db.query(
                "SELECT rowid, embedding FROM \(VectorStore.vectorTable)")
            trueVectors.reserveCapacity(rows.count)
            for row in rows {
                guard let rowID = row["rowid"]?.intValue,
                      let data = row["embedding"]?.dataValue else { continue }
                let values = VectorEncoder.decode(data, as: .int8,
                                                  dimension: Self.dimension)
                var squares: Float = 0
                vDSP_svesq(values, 1, &squares, vDSP_Length(values.count))
                trueVectors.append((rowID, values, sqrt(squares)))
            }
            print("[binary-bench] true ground-truth vectors loaded \(trueVectors.count)")
        }

        func trueTopK(_ query: [Float], k: Int) -> Set<Int64> {
            var scored: [(Int64, Double)] = []
            scored.reserveCapacity(trueVectors.count)
            for item in trueVectors {
                var dot: Float = 0
                vDSP_dotpr(query, 1, item.values, 1, &dot, vDSP_Length(query.count))
                let denominator = item.norm
                scored.append((item.rowID, denominator > 0 ? Double(dot / denominator) : 0))
            }
            scored.sort { $0.1 > $1.1 }
            return Set(scored.prefix(k).map(\.0))
        }
        // 查询：取已知簇中心附近的新向量，衡量 top-50 召回是否遗漏。
        var queryRaws: [[Float]] = []
        var queries: [[Float]] = []
        queries.reserveCapacity(30)
        queryRaws.reserveCapacity(30)
        for _ in 0..<30 {
            let center = centers[Int(rng.next() % UInt64(centers.count))]
            let raw = vector(center: center, rng: &rng, noise: 0.2)
            queryRaws.append(raw)
            queries.append(Self.mainVector(raw))
        }

        var exactTimes: [Double] = []
        var annTimes: [Double] = []
        var binaryTimes: [Double] = []
        var exactRecall10: [Double] = []
        var exactRecall50: [Double] = []
        var annRecall10: [Double] = []
        var annRecall50: [Double] = []
        var binaryRecall10: [Double] = []
        var binaryRecall50: [Double] = []
        var stage1Recall50: [Double] = []
        var stage1Times: [Double] = []
        var trueExactRecall50: [Double] = []
        var trueBinaryRecall50: [Double] = []
        var trueANNRecall50: [Double] = []
        var trueStage1Recall50: [Double] = []
        var trueExactRecall10: [Double] = []
        var trueBinaryRecall10: [Double] = []
        var trueANNRecall10: [Double] = []
        var trueStage1Recall10: [Double] = []
        var stage2Times: [Double] = []
        var stage2Vec0Times: [Double] = []
        var binaryHitCount = 0
        var binaryMissingScore = 0
        // bann（自研二值 ANN）与 vec0 同轮对比。
        var bannTimes: [Double] = []
        var bannRecall10: [Double] = []
        var bannRecall50: [Double] = []
        var trueBannRecall10: [Double] = []
        var trueBannRecall50: [Double] = []
        var bannStage1Times: [Double] = []
        var bannStage1Coverage50: [Double] = []
        var bannBuildSeconds = 0.0
        var bannResult = ANNBuildResult()
        var bannHitCount = 0
        var exactHitsByQuery: [[SearchHit]] = []
        var exactTruth50ByQuery: [Set<Int64>] = []
        var exactTruth10ByQuery: [Set<Int64>] = []
        var trueTruth50ByQuery: [Set<Int64>?] = []
        var trueTruth10ByQuery: [Set<Int64>?] = []

        func stage1RowIDs(signature: Data, limit: Int) throws -> (Set<Int64>, Double) {
            let started = CFAbsoluteTimeGetCurrent()
            let rows = try store.db.query(
                "SELECT rowid FROM \(VectorStore.binaryVectorTable) "
                + "WHERE embedding MATCH vec_bit(?) AND k = ?",
                [.blob(signature), .int(Int64(limit))])
            let ms = (CFAbsoluteTimeGetCurrent() - started) * 1000
            return (Set(rows.compactMap { $0["rowid"]?.intValue }), ms)
        }

        func timed(_ body: () throws -> [SearchHit]) rethrows -> ([SearchHit], Double) {
            let started = CFAbsoluteTimeGetCurrent()
            let hits = try body()
            return (hits, (CFAbsoluteTimeGetCurrent() - started) * 1000)
        }

        // 预热。
        for query in queries.prefix(3) {
            _ = try store.hybridSearch(vector: query, query: "", topK: 50)
        }

        func record(exact: [SearchHit], exactMs: Double,
                    ann: [SearchHit], annMs: Double,
                    binary: [SearchHit], binaryMs: Double,
                    stage1: Set<Int64>, stage1Ms: Double,
                    trueTruth: Set<Int64>?, trueTruth10: Set<Int64>?) {
            exactHitsByQuery.append(exact)
            exactTruth50ByQuery.append(Set(exact.prefix(50).map(\.rowID)))
            exactTruth10ByQuery.append(Set(exact.prefix(10).map(\.rowID)))
            trueTruth50ByQuery.append(trueTruth)
            trueTruth10ByQuery.append(trueTruth10)
            exactTimes.append(exactMs)
            annTimes.append(annMs)
            binaryTimes.append(binaryMs)
            stage1Times.append(stage1Ms)
            let truth = Set(exact.prefix(50).map(\.rowID))
            let truth10 = Set(exact.prefix(10).map(\.rowID))
            exactRecall10.append(Self.recall(exact, groundTruth: truth10, k: 10))
            exactRecall50.append(Self.recall(exact, groundTruth: truth, k: 50))
            annRecall10.append(Self.recall(ann, groundTruth: truth10, k: 10))
            annRecall50.append(Self.recall(ann, groundTruth: truth, k: 50))
            binaryRecall10.append(Self.recall(binary, groundTruth: truth10, k: 10))
            binaryRecall50.append(Self.recall(binary, groundTruth: truth, k: 50))
            let stage1Prefix = exact.prefix(50).map(\.rowID)
            stage1Recall50.append(
                Double(stage1Prefix.filter { stage1.contains($0) }.count)
                    / Double(min(50, truth.count)))
            if let trueTruth {
                trueExactRecall50.append(Self.recall(exact, groundTruth: trueTruth, k: 50))
                trueBinaryRecall50.append(Self.recall(binary, groundTruth: trueTruth, k: 50))
                trueANNRecall50.append(Self.recall(ann, groundTruth: trueTruth, k: 50))
                let stage1Prefix = Array(stage1)
                trueStage1Recall50.append(
                    Double(stage1Prefix.filter { trueTruth.contains($0) }.count)
                        / Double(min(50, trueTruth.count)))
            }
            if let trueTruth10 {
                trueExactRecall10.append(Self.recall(exact, groundTruth: trueTruth10, k: 10))
                trueBinaryRecall10.append(Self.recall(binary, groundTruth: trueTruth10, k: 10))
                trueANNRecall10.append(Self.recall(ann, groundTruth: trueTruth10, k: 10))
                let stage1Prefix = Array(stage1)
                trueStage1Recall10.append(
                    Double(stage1Prefix.filter { trueTruth10.contains($0) }.count)
                        / Double(min(10, trueTruth10.count)))
            }
            binaryHitCount += binary.count
            binaryMissingScore += binary.filter { $0.binaryScore == nil }.count
        }

        for (queryIndex, query) in queries.enumerated() {
            let signature = VectorEncoder.encodeSignature(queryRaws[queryIndex],
                                                          bitDimension: Self.signatureBits)
            let binaryOptions = BinaryCoarseOptions(signature: signature,
                                                    candidateLimit: candidateLimit)
            let annOptions: ANNQueryOptions? = benchANN
                ? ANNQueryOptions(nprobe: 4, rescore: 64)
                : nil

            // stage 1 单独测一次：区分「汉明粗筛漏召回」与「精排/融合漏召回」。
            let (stage1, stage1Ms) = try stage1RowIDs(signature: signature,
                                                      limit: candidateLimit)
            let trueTruth: Set<Int64>? = benchTrue ? trueTopK(query, k: 50) : nil
            let trueTruth10: Set<Int64>? = benchTrue ? trueTopK(query, k: 10) : nil
            // 轮换顺序，避免后跑的方法总是吃到更热页缓存。
            switch queryIndex % 3 {
            case 0:
                let (exact, exactMs) = try timed {
                    try store.hybridSearch(vector: query, query: "", topK: 50)
                }
                let (ann, annMs) = try timed {
                    try store.hybridSearch(vector: query, query: "", topK: 50,
                                           ann: annOptions)
                }
                let (binaryHits, binaryMs) = try timed {
                    try store.hybridSearch(vector: query, query: "", topK: 50,
                                           binary: binaryOptions)
                }
                record(exact: exact, exactMs: exactMs, ann: ann, annMs: annMs,
                       binary: binaryHits, binaryMs: binaryMs,
                       stage1: stage1, stage1Ms: stage1Ms,
                       trueTruth: trueTruth, trueTruth10: trueTruth10)
            case 1:
                let (ann, annMs) = try timed {
                    try store.hybridSearch(vector: query, query: "", topK: 50,
                                           ann: annOptions)
                }
                let (binaryHits, binaryMs) = try timed {
                    try store.hybridSearch(vector: query, query: "", topK: 50,
                                           binary: binaryOptions)
                }
                let (exact, exactMs) = try timed {
                    try store.hybridSearch(vector: query, query: "", topK: 50)
                }
                record(exact: exact, exactMs: exactMs, ann: ann, annMs: annMs,
                       binary: binaryHits, binaryMs: binaryMs,
                       stage1: stage1, stage1Ms: stage1Ms,
                       trueTruth: trueTruth, trueTruth10: trueTruth10)
            default:
                let (binaryHits, binaryMs) = try timed {
                    try store.hybridSearch(vector: query, query: "", topK: 50,
                                           binary: binaryOptions)
                }
                let (exact, exactMs) = try timed {
                    try store.hybridSearch(vector: query, query: "", topK: 50)
                }
                let (ann, annMs) = try timed {
                    try store.hybridSearch(vector: query, query: "", topK: 50,
                                           ann: annOptions)
                }
                record(exact: exact, exactMs: exactMs, ann: ann, annMs: annMs,
                       binary: binaryHits, binaryMs: binaryMs,
                       stage1: stage1, stage1Ms: stage1Ms,
                       trueTruth: trueTruth, trueTruth10: trueTruth10)
            }
        }

        // 构建二值 ANN，再用同一批查询测 bann 路径（现有循环里的 binary 走 vec0）。
        if benchBann {
            let started = Date()
            bannResult = try store.buildBinaryANNIndex(
                clusters: Self.clusters, clusterSize: 512, iterations: 8)
            bannBuildSeconds = Date().timeIntervalSince(started)
            let bannStatus = store.binaryANNIndexStatus()
            print("[binary-bench] bann build=\(String(format: "%.1f", bannBuildSeconds))s "
                + "indexed=\(bannResult.indexedRows) blocks=\(bannResult.blocks) "
                + "ready=\(bannStatus.isReady) summary=\(bannStatus.summary)")
            for (queryIndex, query) in queries.enumerated() {
                let signature = VectorEncoder.encodeSignature(
                    queryRaws[queryIndex], bitDimension: Self.signatureBits)
                let (bannHits, bannMs) = try timed {
                    try store.hybridSearch(
                        vector: query, query: "", topK: 50,
                        binary: BinaryCoarseOptions(signature: signature,
                                                    candidateLimit: candidateLimit))
                }
                bannTimes.append(bannMs)
                bannHitCount += bannHits.count
                guard queryIndex < exactHitsByQuery.count else { continue }
                bannRecall10.append(Self.recall(
                    bannHits, groundTruth: exactTruth10ByQuery[queryIndex], k: 10))
                bannRecall50.append(Self.recall(
                    bannHits, groundTruth: exactTruth50ByQuery[queryIndex], k: 50))
                if let trueTruth = trueTruth50ByQuery[queryIndex] {
                    trueBannRecall50.append(Self.recall(bannHits, groundTruth: trueTruth, k: 50))
                }
                if let trueTruth = trueTruth10ByQuery[queryIndex] {
                    trueBannRecall10.append(Self.recall(bannHits, groundTruth: trueTruth, k: 10))
                }
                let stageStarted = CFAbsoluteTimeGetCurrent()
                if let bannStage = try store.bannCandidatesLocked(
                    signature: signature, topK: candidateLimit) {
                    bannStage1Times.append((CFAbsoluteTimeGetCurrent() - stageStarted) * 1000)
                    let ids = Set(bannStage.map(\.0))
                    let exactTop = exactHitsByQuery[queryIndex].prefix(50).map(\.rowID)
                    bannStage1Coverage50.append(
                        Double(exactTop.filter { ids.contains($0) }.count)
                            / Double(min(50, exactTop.count)))
                }
            }
        }

        // stage 2 单独测一次：从 bin KNN 候选点查主向量、解码、余弦重排。
        // 同时直接读主表模拟旧路径，对比 rerank 镜像的收益。
        for (queryIndex, query) in queries.enumerated() {
            let signature = VectorEncoder.encodeSignature(queryRaws[queryIndex],
                                                          bitDimension: Self.signatureBits)
            let (ids, _) = try stage1RowIDs(signature: signature, limit: candidateLimit)
            let storedStarted = CFAbsoluteTimeGetCurrent()
            let stored = try store.exactVectorsLocked(rowIDs: Array(ids))
            var scored: [(Int64, Double)] = []
            scored.reserveCapacity(stored.count)
            for (rowID, values) in stored {
                scored.append((rowID, VectorCodec.cosine(query, values)))
            }
            scored.sort { $0.1 > $1.1 }
            stage2Times.append((CFAbsoluteTimeGetCurrent() - storedStarted) * 1000)

            let vec0Started = CFAbsoluteTimeGetCurrent()
            let ordered = ids.sorted()
            if !ordered.isEmpty {
                let placeholders = Array(repeating: "?", count: ordered.count)
                    .joined(separator: ",")
                let rows = try store.db.query(
                    "SELECT rowid, embedding FROM \(VectorStore.vectorTable) "
                    + "WHERE rowid IN (\(placeholders))",
                    ordered.map { SQLValue.int($0) })
                var scoredVec0: [(Int64, Double)] = []
                scoredVec0.reserveCapacity(rows.count)
                for row in rows {
                    guard let rowID = row["rowid"]?.intValue,
                          let data = row["embedding"]?.dataValue else { continue }
                    let values = VectorEncoder.decode(data, as: .int8,
                                                      dimension: Self.dimension)
                    scoredVec0.append((rowID, VectorCodec.cosine(query, values)))
                }
                scoredVec0.sort { $0.1 > $1.1 }
            }
            stage2Vec0Times.append((CFAbsoluteTimeGetCurrent() - vec0Started) * 1000)
        }

        func average(_ values: [Double]) -> Double {
            values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
        }
        let annBuildText = benchANN
            ? String(format: "%.1f", annSeconds) + "s"
            : "skipped"
        let bannBuildText = benchBann
            ? String(format: "%.1f", bannBuildSeconds) + "s"
            : "skipped"
        let message = """
        [binary-bench] rows=\(rows) candidates=\(candidateLimit) \
        main_dim=\(Self.dimension) sig_bits=\(Self.signatureBits) clusters=\(Self.clusters) \
        insert=\(String(format: "%.1f", insertSeconds))s \
        ann_build=\(annBuildText) \
        ann_indexed=\(annResult.indexedRows) tail=\(annResult.tailRows) blocks=\(annResult.blocks) \
        bann_build=\(bannBuildText) \
        bann_indexed=\(bannResult.indexedRows) tail=\(bannResult.tailRows) blocks=\(bannResult.blocks) \
        exact_ms p50=\(String(format: "%.1f", Self.percentile(exactTimes, 0.5))) \
        p95=\(String(format: "%.1f", Self.percentile(exactTimes, 0.95))); \
        ann_ms p50=\(String(format: "%.1f", Self.percentile(annTimes, 0.5))) \
        p95=\(String(format: "%.1f", Self.percentile(annTimes, 0.95))); \
        binary_vec0_ms p50=\(String(format: "%.1f", Self.percentile(binaryTimes, 0.5))) \
        p95=\(String(format: "%.1f", Self.percentile(binaryTimes, 0.95))); \
        binary_bann_ms p50=\(String(format: "%.1f", Self.percentile(bannTimes, 0.5))) \
        p95=\(String(format: "%.1f", Self.percentile(bannTimes, 0.95))); \
        recall@10 vec0=\(String(format: "%.1f", average(binaryRecall10) * 100))% \
        bann=\(String(format: "%.1f", average(bannRecall10) * 100))% \
        ann=\(String(format: "%.1f", average(annRecall10) * 100))%; \
        recall@50 vec0=\(String(format: "%.1f", average(binaryRecall50) * 100))% \
        bann=\(String(format: "%.1f", average(bannRecall50) * 100))% \
        ann=\(String(format: "%.1f", average(annRecall50) * 100))%; \
        stage1_recall@50 vec0=\(String(format: "%.1f", average(stage1Recall50) * 100))% \
        bann=\(String(format: "%.1f", average(bannStage1Coverage50) * 100))% \
        stage1_ms p50=\(String(format: "%.1f", Self.percentile(stage1Times, 0.5))) \
        bann_stage1_ms p50=\(String(format: "%.1f", Self.percentile(bannStage1Times, 0.5))); \
        true_recall@10 exact=\(String(format: "%.1f", average(trueExactRecall10) * 100))% \
        vec0=\(String(format: "%.1f", average(trueBinaryRecall10) * 100))% \
        bann=\(String(format: "%.1f", average(trueBannRecall10) * 100))% \
        ann=\(String(format: "%.1f", average(trueANNRecall10) * 100))% \
        true_stage1_coverage@10=\(String(format: "%.1f", average(trueStage1Recall10) * 100))% \
        true_recall@50 exact=\(String(format: "%.1f", average(trueExactRecall50) * 100))% \
        vec0=\(String(format: "%.1f", average(trueBinaryRecall50) * 100))% \
        bann=\(String(format: "%.1f", average(trueBannRecall50) * 100))% \
        ann=\(String(format: "%.1f", average(trueANNRecall50) * 100))% \
        true_stage1_coverage@50=\(String(format: "%.1f", average(trueStage1Recall50) * 100))% \
        stage2_rerank_ms p50=\(String(format: "%.1f", Self.percentile(stage2Times, 0.5))); \
        stage2_vec0_ms p50=\(String(format: "%.1f", Self.percentile(stage2Vec0Times, 0.5))); \
        binary_hits=\(binaryHitCount) bann_hits=\(bannHitCount) \
        missing_binary_score=\(binaryMissingScore)
        """
        print(message)
        XCTAssertGreaterThan(binaryHitCount, 0)
    }

    func testScaleRetrieval() throws {
        guard ProcessInfo.processInfo.environment["BINARY_BENCH"] == "1" else {
            throw XCTSkip("未设置 BINARY_BENCH=1")
        }
        let sizes = ProcessInfo.processInfo.environment["BINARY_BENCH_SIZES"]?
            .split(separator: ",").compactMap { Int($0) } ?? [50_000, 200_000]
        for size in sizes {
            try runCase(rows: size)
        }
    }
}
