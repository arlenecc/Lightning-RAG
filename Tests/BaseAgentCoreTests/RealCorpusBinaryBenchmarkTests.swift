import XCTest
import Foundation
import Accelerate
import MLX
@testable import BaseAgentCore

/// 真实语料 + jina-v5 的二值粗筛验证（临时，不提交）。
///
/// 环境变量：
/// - `REAL_BENCH_LIST`：文件名列表（相对 `REAL_BENCH_DIR`），必填
/// - `REAL_BENCH_DIR`：语料目录，默认 `/Volumes/SSD/book-clean`
/// - `REAL_BENCH_CANDIDATES`：二值候选数，默认 200
/// - `REAL_BENCH_QUERIES`：查询条数，默认 30
/// - `REAL_BENCH_ANN=0`：跳过 ANN 对比
/// - `REAL_BENCH_DIM`：Matryoshka 截断维，默认 128
final class RealCorpusBinaryBenchmarkTests: XCTestCase {
    private struct SplitMix64 {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        mutating func index(_ upper: Int) -> Int {
            guard upper > 0 else { return 0 }
            return Int(next() % UInt64(upper))
        }
    }

    private static func percentile(_ values: [Double], _ p: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let index = min(sorted.count - 1, max(0, Int((Double(sorted.count) - 1) * p)))
        return sorted[index]
    }

    private static func overlap(_ a: [SearchHit], _ b: [SearchHit], k: Int) -> Double {
        let left = Set(a.prefix(k).map(\.rowID))
        let right = Set(b.prefix(k).map(\.rowID))
        guard k > 0 else { return 0 }
        return Double(left.intersection(right).count) / Double(k)
    }

    private static func recall(_ hits: [SearchHit],
                               truth: Set<Int64>, k: Int) -> Double {
        guard !truth.isEmpty, k > 0 else { return 0 }
        let prefix = hits.prefix(k).map(\.rowID)
        return Double(prefix.filter { truth.contains($0) }.count)
            / Double(min(k, truth.count))
    }

    private func configureMLX() {
        guard MLX.GPU.metallib == nil else { return }
        let base = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        for relative in [".build/release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib",
                         ".build/debug/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib",
                         ".build/out/Products/Release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"] {
            let candidate = base.appendingPathComponent(relative)
            if FileManager.default.fileExists(atPath: candidate.path) {
                MLX.GPU.metallib = candidate
                return
            }
        }
    }

    func testRealCorpusRetrieval() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let listPath = env["REAL_BENCH_LIST"] else {
            throw XCTSkip("未设置 REAL_BENCH_LIST")
        }
        let corpusDir = env["REAL_BENCH_DIR"] ?? "/Volumes/SSD/book-clean"
        let names = try String(contentsOfFile: listPath, encoding: .utf8)
            .split(separator: "\n")
            .map { String($0).trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard !names.isEmpty else { throw XCTSkip("语料列表为空") }

        LocalModelFactory.shared.clearCaches()
        let modelsRoot = EmbeddingBenchmarkTests.resolveModelsRoot()
        setenv("BASE_AGENT_MODELS_DIR", modelsRoot.path, 1)
        configureMLX()

        let keep = env["REAL_BENCH_KEEP"] == "1"
        let root = keep
            ? URL(fileURLWithPath: env["REAL_BENCH_ROOT"] ?? "/tmp/real_bench_root")
            : URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("real-binary-bench-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        if !keep { defer { try? FileManager.default.removeItem(at: root) } }
        let workspace = root
        let indexExists = FileManager.default.fileExists(
            atPath: root.appendingPathComponent("rag_data").path)

        // XCTest release 下 stdout 可能被缓冲，结果同时落文件。
        let outPath = env["REAL_BENCH_OUT"] ?? "/tmp/real_bench_result.log"
        FileManager.default.createFile(atPath: outPath, contents: nil)
        let outHandle = FileHandle(forWritingAtPath: outPath)
        defer { try? outHandle?.close() }
        func emit(_ line: String) {
            print(line)
            try? outHandle?.write(contentsOf: Data((line + "\n").utf8))
        }
        if !indexExists {
            for name in names {
                let src = (corpusDir as NSString).appendingPathComponent(name)
                guard FileManager.default.fileExists(atPath: src) else {
                    throw XCTSkip("语料缺失：\(src)")
                }
                let dst = (kb.path as NSString).appendingPathComponent(name)
                try? FileManager.default.copyItem(atPath: src, toPath: dst)
            }
        }

        var config = AgentConfig()
        config.workspace = workspace.path
        config.ragANNEnabled = false
        config.ragANNAutoBuild = false
        config.ragRerankEnabled = false
        config.ragIngestConcurrency = 4
        config.ragBinaryCoarseEnabled = true
        config.ragBinaryCoarseCandidates = Int(env["REAL_BENCH_CANDIDATES"] ?? "") ?? 200
        config.ragMatryoshkaDim = Int(env["REAL_BENCH_DIM"] ?? "") ?? 128
        let embedding = try XCTUnwrap(LocalModelFactory.shared.makeEmbedding(config: config))

        let warmStarted = Date()
        _ = try await embedding.embedDocuments(["预热文本：检索增强生成"])
        emit(String(format: "[real-bench] warmup %.1fs model=%@ dim=%d signature=%d",
                    Date().timeIntervalSince(warmStarted), embedding.identifier,
                    embedding.dimension,
                    (embedding as? EmbeddingBinarySigning)?.binaryDimension ?? 0))

        let engine = RAGEngine(config: config, knowledgeBase: kb.path, embedding: embedding)
        defer { engine.close() }
        let started = Date()
        if indexExists {
            emit("[real-bench] 复用已有索引：\(root.path)")
        } else {
            let stats = await engine.ingest(force: true, log: { line in
                emit(String(format: "[real-bench +%.1fs] ", Date().timeIntervalSince(started)) + line)
            })
            emit(String(format: "[real-bench] ingest %.1fs files=%d processed=%d chunks=%d errors=%d",
                        Date().timeIntervalSince(started), names.count, stats.processed,
                        stats.chunks, stats.errors.count))
        }

        let signing = embedding as? EmbeddingBinarySigning
        let store = try VectorStore(
            dbPath: engine.indexDBPath,
            dimension: embedding.dimension,
            elementType: engine.effectiveVectorElementType,
            binaryDimension: engine.binarySignatureDimension,
            binarySource: engine.binarySignatureSource,
            indexFingerprint: engine.vectorIndexFingerprint)
        defer { store.close() }
        let chunkRows = try store.db.query(
            "SELECT rowid, text, source FROM \(VectorStore.chunksTable) ORDER BY rowid")
        emit("[real-bench] chunks=\(chunkRows.count) status=\(store.binaryIndexStatus().rows)"
            + " coverage=\(store.binaryIndexStatus().coverage)")

        let queryCount = min(Int(env["REAL_BENCH_QUERIES"] ?? "") ?? 30, chunkRows.count)
        let stride = max(1, chunkRows.count / max(1, queryCount))
        var samples: [(rowID: Int64, text: String)] = []
        var cursor = 0
        while cursor < chunkRows.count && samples.count < queryCount {
            let row = chunkRows[cursor]
            if let rowID = row["rowid"]?.intValue,
               let text = row["text"]?.stringValue, !text.isEmpty {
                samples.append((rowID, String(text.prefix(400))))
            }
            cursor += stride
        }
        emit("[real-bench] queries=\(samples.count) candidateLimit=\(config.ragBinaryCoarseCandidates)")
        guard !samples.isEmpty else { throw XCTSkip("没有可用查询切片") }

        // 全流程：RAGEngine.search 内部走二值粗筛（含查询签名与状态汇报）。
        let engineStatus = engine.status()
        let binaryState = engineStatus["binary_coarse"]?.stringValue ?? "?"
        let annBypassed = engineStatus["ann_bypassed"]?.boolValue ?? false
        let engineHits = try await engine.search(query: samples[0].text, topK: 5)
        emit("[real-bench] engine.search hits=\(engineHits.count) "
            + "binary_coarse=\(binaryState) ann_bypassed=\(annBypassed) "
            + "coverage=\(engine.binaryIndexStatus()?.coverage ?? -1)")

        var signedQueries: [(vector: [Float], signature: Data)] = []
        signedQueries.reserveCapacity(samples.count)
        for sample in samples {
            guard let signing else { throw XCTSkip("embedding 不支持二值签名") }
            signedQueries.append(try await signing.embedQueryWithSignature(sample.text))
        }

        // 真实余弦 Ground Truth：Float 查询 vs int8 落盘向量（与 stage 2 完全一致）。
        let dim = embedding.dimension
        let trueRows = try store.db.query(
            "SELECT rowid, embedding FROM \(VectorStore.vectorTable)")
        var trueVectors: [(rowID: Int64, values: [Float], norm: Float)] = []
        trueVectors.reserveCapacity(trueRows.count)
        for row in trueRows {
            guard let rowID = row["rowid"]?.intValue,
                  let data = row["embedding"]?.dataValue else { continue }
            let values = VectorEncoder.decode(data, as: engine.effectiveVectorElementType,
                                              dimension: dim)
            var squares: Float = 0
            vDSP_svesq(values, 1, &squares, vDSP_Length(values.count))
            trueVectors.append((rowID, values, sqrt(squares)))
        }
        func trueTopK(_ query: [Float], k: Int) -> Set<Int64> {
            var scored: [(Int64, Double)] = []
            scored.reserveCapacity(trueVectors.count)
            let queryNorm = sqrt(Self.dot(query, query))
            for item in trueVectors {
                var dot: Float = 0
                vDSP_dotpr(query, 1, item.values, 1, &dot, vDSP_Length(query.count))
                let denom = Double(queryNorm) * Double(item.norm)
                scored.append((item.rowID, denom > 0 ? Double(dot) / denom : 0))
            }
            scored.sort { $0.1 > $1.1 }
            return Set(scored.prefix(k).map(\.0))
        }

        let benchANN = env["REAL_BENCH_ANN"] != "0"
        let benchBann = env["REAL_BENCH_BANN"] != "0"
        // 索引库会被复用：先删掉上一轮留下的 bann，确保主循环里的
        // binary 指标确实走 vec0，再在循环后重建 bann 做对比。
        if benchBann {
            try store.dropBinaryANNIndex()
        }
        let annResult = benchANN
            ? try store.buildANNIndex(clusters: 32, clusterSize: 256, iterations: 8)
            : ANNBuildResult()
        let annOptions: ANNQueryOptions? = benchANN
            ? ANNQueryOptions(nprobe: 4, rescore: 64)
            : nil
        emit("[real-bench] ann indexed=\(annResult.indexedRows) blocks=\(annResult.blocks)")

        var exactTimes: [Double] = []
        var binaryTimes: [Double] = []
        var annTimes: [Double] = []
        var bannTimes: [Double] = []
        var overlap10: [Double] = []
        var overlap50: [Double] = []
        var bannOverlap10: [Double] = []
        var bannOverlap50: [Double] = []
        var trueExact10: [Double] = []
        var trueBinary10: [Double] = []
        var trueANN10: [Double] = []
        var trueBann10: [Double] = []
        var trueExact50: [Double] = []
        var trueBinary50: [Double] = []
        var trueANN50: [Double] = []
        var trueBann50: [Double] = []
        var stage1Coverage: [Double] = []
        var bannStage1Coverage: [Double] = []
        var bannBuildSeconds = 0.0
        var bannResult = ANNBuildResult()
        var exactByQuery: [[SearchHit]] = []
        var true10ByQuery: [Set<Int64>] = []
        var true50ByQuery: [Set<Int64>] = []

        func stripSelf(_ hits: [SearchHit], rowID: Int64) -> [SearchHit] {
            hits.filter { $0.rowID != rowID }
        }

        // 预热。
        for query in signedQueries.prefix(2) {
            _ = try store.hybridSearch(vector: query.vector, query: "", topK: 50,
                                       binary: BinaryCoarseOptions(signature: query.signature,
                                                                   candidateLimit: config.ragBinaryCoarseCandidates))
            _ = try store.hybridSearch(vector: query.vector, query: "", topK: 50)
        }

        for (index, signed) in signedQueries.enumerated() {
            let sample = samples[index]
            let true10 = trueTopK(signed.vector, k: 10)
            let true50 = trueTopK(signed.vector, k: 50)

            let exactStarted = CFAbsoluteTimeGetCurrent()
            let exactRaw = try store.hybridSearch(vector: signed.vector, query: "", topK: 50)
            exactTimes.append((CFAbsoluteTimeGetCurrent() - exactStarted) * 1000)
            let exact = stripSelf(exactRaw, rowID: sample.rowID)
            exactByQuery.append(exact)
            true10ByQuery.append(true10)
            true50ByQuery.append(true50)

            let binaryStarted = CFAbsoluteTimeGetCurrent()
            let binaryRaw = try store.hybridSearch(
                vector: signed.vector, query: "", topK: 50,
                binary: BinaryCoarseOptions(signature: signed.signature,
                                            candidateLimit: config.ragBinaryCoarseCandidates))
            binaryTimes.append((CFAbsoluteTimeGetCurrent() - binaryStarted) * 1000)
            let binary = stripSelf(binaryRaw, rowID: sample.rowID)

            let ann: [SearchHit]
            if benchANN {
                let annStarted = CFAbsoluteTimeGetCurrent()
                let raw = try store.hybridSearch(vector: signed.vector, query: "", topK: 50,
                                                 ann: annOptions)
                annTimes.append((CFAbsoluteTimeGetCurrent() - annStarted) * 1000)
                ann = stripSelf(raw, rowID: sample.rowID)
            } else {
                annTimes.append(exactTimes.last ?? 0)
                ann = exact
            }

            overlap10.append(Self.overlap(binary, exact, k: 10))
            overlap50.append(Self.overlap(binary, exact, k: 50))
            trueExact10.append(Self.recall(exact, truth: true10, k: 10))
            trueBinary10.append(Self.recall(binary, truth: true10, k: 10))
            trueANN10.append(Self.recall(ann, truth: true10, k: 10))
            trueExact50.append(Self.recall(exact, truth: true50, k: 50))
            trueBinary50.append(Self.recall(binary, truth: true50, k: 50))
            trueANN50.append(Self.recall(ann, truth: true50, k: 50))

            let stageRows = try store.db.query(
                "SELECT rowid FROM \(VectorStore.binaryVectorTable) "
                + "WHERE embedding MATCH vec_bit(?) AND k = ?",
                [.blob(signed.signature), .int(Int64(config.ragBinaryCoarseCandidates))])
            let stageIDs = Set(stageRows.compactMap { $0["rowid"]?.intValue })
            stage1Coverage.append(
                Double(true50.filter { stageIDs.contains($0) }.count)
                    / Double(min(50, true50.count)))
        }

        // 构建二值 ANN，再用同一批查询测 bann 路径（上面的 binary 走 vec0）。
        if benchBann {
            let started = Date()
            bannResult = try store.buildBinaryANNIndex(
                clusters: 32, clusterSize: 256, iterations: 8)
            bannBuildSeconds = Date().timeIntervalSince(started)
            let bannStatus = store.binaryANNIndexStatus()
            emit("[real-bench] bann build=\(String(format: "%.1f", bannBuildSeconds))s "
                + "indexed=\(bannResult.indexedRows) blocks=\(bannResult.blocks) "
                + "ready=\(bannStatus.isReady) summary=\(bannStatus.summary)")
            for (index, signed) in signedQueries.enumerated() {
                let sample = samples[index]
                let bannStarted = CFAbsoluteTimeGetCurrent()
                let raw = try store.hybridSearch(
                    vector: signed.vector, query: "", topK: 50,
                    binary: BinaryCoarseOptions(signature: signed.signature,
                                                candidateLimit: config.ragBinaryCoarseCandidates))
                bannTimes.append((CFAbsoluteTimeGetCurrent() - bannStarted) * 1000)
                let bann = stripSelf(raw, rowID: sample.rowID)
                let exact = exactByQuery[index]
                let true10 = true10ByQuery[index]
                let true50 = true50ByQuery[index]
                bannOverlap10.append(Self.overlap(bann, exact, k: 10))
                bannOverlap50.append(Self.overlap(bann, exact, k: 50))
                trueBann10.append(Self.recall(bann, truth: true10, k: 10))
                trueBann50.append(Self.recall(bann, truth: true50, k: 50))
                if let stage = try store.bannCandidatesLocked(
                    signature: signed.signature,
                    topK: config.ragBinaryCoarseCandidates) {
                    let ids = Set(stage.map(\.0))
                    bannStage1Coverage.append(
                        Double(true50.filter { ids.contains($0) }.count)
                            / Double(min(50, true50.count)))
                }
            }
        }

        func average(_ values: [Double]) -> Double {
            values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
        }
        let bannBuildText = benchBann
            ? String(format: "%.1f", bannBuildSeconds) + "s"
            : "skipped"
        let bannText = benchBann
            ? """
            bann_ms p50=\(String(format: "%.1f", Self.percentile(bannTimes, 0.5))) \
            overlap@10 bann=\(String(format: "%.1f", average(bannOverlap10) * 100))% \
            overlap@50 bann=\(String(format: "%.1f", average(bannOverlap50) * 100))% \
            true_recall@10 bann=\(String(format: "%.1f", average(trueBann10) * 100))% \
            true_recall@50 bann=\(String(format: "%.1f", average(trueBann50) * 100))% \
            bann_stage1_true_coverage@50=\(String(format: "%.1f", average(bannStage1Coverage) * 100))%
            """
            : "bann=skipped"
        let message = """
        [real-bench] chunks=\(chunkRows.count) queries=\(samples.count) \
        candidates=\(config.ragBinaryCoarseCandidates) dim=\(dim) \
        sig=\(engine.binarySignatureDimension) bann_build=\(bannBuildText) \
        exact_ms p50=\(String(format: "%.1f", Self.percentile(exactTimes, 0.5))) \
        binary_ms p50=\(String(format: "%.1f", Self.percentile(binaryTimes, 0.5))) \
        ann_ms p50=\(String(format: "%.1f", Self.percentile(annTimes, 0.5))) \
        overlap@10=\(String(format: "%.1f", average(overlap10) * 100))% \
        overlap@50=\(String(format: "%.1f", average(overlap50) * 100))% \
        true_recall@10 exact=\(String(format: "%.1f", average(trueExact10) * 100))% \
        binary=\(String(format: "%.1f", average(trueBinary10) * 100))% \
        ann=\(String(format: "%.1f", average(trueANN10) * 100))% \
        true_recall@50 exact=\(String(format: "%.1f", average(trueExact50) * 100))% \
        binary=\(String(format: "%.1f", average(trueBinary50) * 100))% \
        ann=\(String(format: "%.1f", average(trueANN50) * 100))% \
        stage1_true_coverage@50=\(String(format: "%.1f", average(stage1Coverage) * 100))% \
        \(bannText)
        """
        emit(message)
        XCTAssertFalse(samples.isEmpty)
        await embedding.unload()
    }

    private static func dot(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        var dot: Float = 0
        for i in 0..<a.count { dot += a[i] * b[i] }
        return dot
    }
}
