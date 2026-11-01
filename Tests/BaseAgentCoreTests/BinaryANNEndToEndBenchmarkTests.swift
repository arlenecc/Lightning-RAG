import XCTest
import Foundation
import Accelerate
@testable import BaseAgentCore

/// 二段式检索全流程基准（临时，不提交）。
///
/// 直接构造索引后走 `RAGEngine.search`：查询签名 → 二值 ANN/vec0 粗筛
/// → stage 2 重排 → RRF。对比 bann、vec0 暴力、精确 dense 三路。
///
/// 环境变量：
/// - `E2E_SIZES`：逗号分隔，默认 `250000,1000000`
/// - `E2E_QUERIES`：查询条数，默认 12
/// - `E2E_MAIN_DIM` / `E2E_SIG_BITS` / `E2E_CLUSTERS`：默认 128/768/96
final class BinaryANNEndToEndBenchmarkTests: XCTestCase {
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

    private static let mainDim = Int(
        ProcessInfo.processInfo.environment["E2E_MAIN_DIM"] ?? "") ?? 128
    private static let signatureBits = Int(
        ProcessInfo.processInfo.environment["E2E_SIG_BITS"] ?? "") ?? 768
    private static let clusters = Int(
        ProcessInfo.processInfo.environment["E2E_CLUSTERS"] ?? "") ?? 96
    private static let queryCount = Int(
        ProcessInfo.processInfo.environment["E2E_QUERIES"] ?? "") ?? 12
    private static let clusterNoise: Float = 0.35

    private final class ScaleEmbedding: EmbeddingProvider,
                                         EmbeddingBinarySigning, @unchecked Sendable {
        let identifier = "binary-ann-scale-fake"
        let dimension = mainDim
        let binaryDimension = signatureBits
        let binarySourceIdentifier = "binary-ann-scale"
        var queries: [String: (vector: [Float], signature: Data)] = [:]

        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map { _ in [Float](repeating: 0, count: dimension) }
        }

        func embedQuery(_ text: String) async throws -> [Float] {
            queries[text]?.vector ?? []
        }

        func embedDocumentsWithSignatures(_ texts: [String])
            async throws -> BinaryEmbeddingOutput {
            BinaryEmbeddingOutput(
                vectors: try await embedDocuments(texts),
                signatures: texts.map { _ in Data() })
        }

        func embedQueryWithSignature(_ text: String) async throws
            -> (vector: [Float], signature: Data) {
            queries[text] ?? ([], Data())
        }
    }

    private static func percentile(_ values: [Double], _ p: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let index = min(sorted.count - 1, max(0, Int((Double(sorted.count) - 1) * p)))
        return sorted[index]
    }

    private static func recall(_ hits: [SearchHit],
                               truth: Set<Int64>, k: Int) -> Double {
        guard !truth.isEmpty, k > 0 else { return 0 }
        let prefix = hits.prefix(k).map(\.rowID)
        return Double(prefix.filter { truth.contains($0) }.count)
            / Double(min(k, truth.count))
    }

    private static func overlap(_ a: [SearchHit], _ b: [SearchHit], k: Int) -> Double {
        guard k > 0 else { return 0 }
        let left = Set(a.prefix(k).map(\.rowID))
        let right = Set(b.prefix(k).map(\.rowID))
        return Double(left.intersection(right).count) / Double(k)
    }

    private func makeCenters(rng: inout SplitMix64) -> [[Float]] {
        (0..<Self.clusters).map { _ in
            (0..<Self.signatureBits).map { _ in rng.signed() }
        }
    }

    private func vector(center: [Float], rng: inout SplitMix64,
                        noise: Float) -> [Float] {
        center.map { $0 + rng.signed() * noise }
    }

    private static func mainVector(_ raw: [Float]) -> [Float] {
        VectorCodec.normalize(Array(raw.prefix(mainDim)))
    }

    private func runSize(_ rows: Int) async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("binary-e2e-\(rows)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)

        var rng = SplitMix64(state: 0xDEAD_BEEF_1234_5678)
        let centers = makeCenters(rng: &rng)
        let embedding = ScaleEmbedding()

        // 查询：已知簇心附近的新向量，交给假 embedding 按文本回放。
        var queryVectors: [String: (vector: [Float], signature: Data)] = [:]
        var queryNames: [String] = []
        for index in 0..<Self.queryCount {
            let center = centers[Int(rng.next() % UInt64(centers.count))]
            let raw = vector(center: center, rng: &rng, noise: 0.2)
            let name = "scale query \(index)"
            queryNames.append(name)
            queryVectors[name] = (
                Self.mainVector(raw),
                VectorEncoder.encodeSignature(raw, bitDimension: Self.signatureBits))
        }
        embedding.queries = queryVectors

        // 另取若干已入库切片，用「轻微扰动」查询验证真正需要的切片不会丢。
        let targetIndices = stride(
            from: rows / 2, to: rows,
            by: max(1, rows / max(1, Self.queryCount))).prefix(Self.queryCount)
        let targetIndexSet = Set(targetIndices)
        var targetRaw: [Int: [Float]] = [:]
        var targetRowIDs: [Int64] = []
        var targetNames: [String] = []

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragBinaryCoarseEnabled = true
        config.ragBinaryCoarseCandidates = 200
        config.ragANNEnabled = false
        config.ragANNAutoBuild = false
        config.ragRerankEnabled = false
        config.ragVectorElementType = .int8
        config.ragANNClusters = 96
        config.ragANNClusterSize = 512
        config.ragQueryTimeoutSeconds = 120
        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: embedding, reranker: nil)

        let store = try VectorStore(
            dbPath: engine.indexDBPath,
            dimension: Self.mainDim,
            elementType: .int8,
            binaryDimension: Self.signatureBits,
            binarySource: embedding.binarySourceIdentifier,
            indexFingerprint: engine.vectorIndexFingerprint)
        let batchSize = 5_000
        let insertStarted = Date()
        var inserted = 0
        while inserted < rows {
            let count = min(batchSize, rows - inserted)
            let source = "e2e-source-\(inserted / batchSize)"
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
                if targetIndexSet.contains(index) { targetRaw[index] = raw }
                chunks.append(Chunk(
                    text: "bench chunk \(index) cluster \(index % centers.count)",
                    source: source, chunkIndex: index))
                vectors.append(Self.mainVector(raw))
                signatures.append(VectorEncoder.encodeSignature(
                    raw, bitDimension: Self.signatureBits))
            }
            try store.add(chunks: chunks, vectors: vectors, binarySignatures: signatures)
            inserted += count
        }
        let insertSeconds = Date().timeIntervalSince(insertStarted)
        print("[e2e] rows=\(rows) insert=\(String(format: "%.1f", insertSeconds))s")

        for (offset, index) in targetIndices.enumerated() {
            guard let base = targetRaw[index] else { continue }
            let noisy = vector(center: base, rng: &rng, noise: 0.02)
            let name = "target query \(offset)"
            targetNames.append(name)
            embedding.queries[name] = (
                Self.mainVector(noisy),
                VectorEncoder.encodeSignature(noisy, bitDimension: Self.signatureBits))
            targetRowIDs.append(Int64(index + 1))
        }
        let allNames = queryNames + targetNames

        // 精确 dense 基准（关闭二值路径）。
        var exactTimes: [Double] = []
        var exactHits: [[SearchHit]] = []
        for name in allNames {
            let started = CFAbsoluteTimeGetCurrent()
            let hits = try store.hybridSearch(
                vector: embedding.queries[name]!.vector, query: "", topK: 10)
            exactTimes.append((CFAbsoluteTimeGetCurrent() - started) * 1000)
            exactHits.append(hits)
        }

        // 先测 vec0 二值暴力（bann 尚未构建）。
        var bruteTimes: [Double] = []
        var bruteHits: [[SearchHit]] = []
        for name in allNames {
            let signed = embedding.queries[name]!
            let started = CFAbsoluteTimeGetCurrent()
            let hits = try store.hybridSearch(
                vector: signed.vector, query: "", topK: 10,
                binary: BinaryCoarseOptions(
                    signature: signed.signature, candidateLimit: 200))
            bruteTimes.append((CFAbsoluteTimeGetCurrent() - started) * 1000)
            bruteHits.append(hits)
        }
        store.close()

        // 构建 bann 后走完整 RAGEngine.search。
        let buildStarted = Date()
        let build = try XCTUnwrap(engine.buildANNIndex())
        let buildSeconds = Date().timeIntervalSince(buildStarted)
        let bannStatus = try XCTUnwrap(engine.binaryANNIndexStatus())
        XCTAssertTrue(bannStatus.isReady, bannStatus.summary)
        print("[e2e] rows=\(rows) bann_build=\(String(format: "%.1f", buildSeconds))s "
            + "indexed=\(build.indexedRows) blocks=\(build.blocks)")

        var annTimes: [Double] = []
        var annHits: [[SearchHit]] = []
        for name in allNames {
            let started = CFAbsoluteTimeGetCurrent()
            let hits = try await engine.search(query: name, topK: 10)
            annTimes.append((CFAbsoluteTimeGetCurrent() - started) * 1000)
            annHits.append(hits)
        }
        let annState = engine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(annState["binary_backend"]?.stringValue, "bann")
        XCTAssertEqual(annState["stage2_backend"]?.stringValue, "rerank")

        // 删除 bann，同一引擎回退 vec0 二值暴力，测全流程差异。
        try engine.dropANNIndex()
        var engineBruteTimes: [Double] = []
        var engineBruteHits: [[SearchHit]] = []
        for name in allNames {
            let started = CFAbsoluteTimeGetCurrent()
            let hits = try await engine.search(query: name, topK: 10)
            engineBruteTimes.append((CFAbsoluteTimeGetCurrent() - started) * 1000)
            engineBruteHits.append(hits)
        }
        let bruteState = engine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(bruteState["binary_backend"]?.stringValue, "sqlite-vec")
        engine.close()

        func average(_ values: [Double]) -> Double {
            values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
        }
        var exactRecall10: [Double] = []
        var bruteRecall10: [Double] = []
        var annRecall10: [Double] = []
        var annVsBrute: [Double] = []
        for index in queryNames.indices {
            let truth = Set(exactHits[index].prefix(10).map(\.rowID))
            exactRecall10.append(1)
            bruteRecall10.append(Self.recall(bruteHits[index], truth: truth, k: 10))
            annRecall10.append(Self.recall(annHits[index], truth: truth, k: 10))
            annVsBrute.append(Self.overlap(annHits[index], engineBruteHits[index], k: 10))
        }
        var targetExactHit: [Double] = []
        var targetBruteHit: [Double] = []
        var targetBannHit: [Double] = []
        for offset in targetNames.indices {
            let index = queryNames.count + offset
            let rowID = targetRowIDs[offset]
            func contains(_ hits: [SearchHit]) -> Double {
                hits.prefix(10).contains { $0.rowID == rowID } ? 1 : 0
            }
            targetExactHit.append(contains(exactHits[index]))
            targetBruteHit.append(contains(engineBruteHits[index]))
            targetBannHit.append(contains(annHits[index]))
        }
        let line = """
        [e2e] rows=\(rows) exact_ms p50=\(String(format: "%.1f", Self.percentile(exactTimes, 0.5))) \
        store_vec0_ms p50=\(String(format: "%.1f", Self.percentile(bruteTimes, 0.5))) \
        engine_bann_ms p50=\(String(format: "%.1f", Self.percentile(annTimes, 0.5))) \
        engine_vec0_ms p50=\(String(format: "%.1f", Self.percentile(engineBruteTimes, 0.5))) \
        engine_bann_exact@10=\(String(format: "%.1f", average(annRecall10) * 100))% \
        engine_vec0_exact@10=\(String(format: "%.1f", average(bruteRecall10) * 100))% \
        bann_vs_vec0@10=\(String(format: "%.1f", average(annVsBrute) * 100))% \
        target_hit@10 exact=\(String(format: "%.1f", average(targetExactHit) * 100))% \
        bann=\(String(format: "%.1f", average(targetBannHit) * 100))% \
        vec0=\(String(format: "%.1f", average(targetBruteHit) * 100))%
        """
        print(line)
        let outPath = ProcessInfo.processInfo.environment["E2E_OUT"] ?? "/tmp/e2e-binary-ann.log"
        if let handle = FileHandle(forWritingAtPath: outPath) {
            try? handle.seekToEnd()
            try? handle.write(contentsOf: Data((line + "\n").utf8))
            try? handle.close()
        }
    }

    func testScaleFullFlow() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["E2E_BENCH"] == "1" else {
            throw XCTSkip("未设置 E2E_BENCH=1")
        }
        let sizes = (env["E2E_SIZES"] ?? "250000,1000000")
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        FileManager.default.createFile(
            atPath: env["E2E_OUT"] ?? "/tmp/e2e-binary-ann.log", contents: nil)
        for rows in sizes {
            try await runSize(rows)
        }
    }
}
