import XCTest
import Foundation
@testable import BaseAgentCore

/// RAG 全链路回归：
/// 解析 → 切片 → 向量入库 → 混合检索 → RRF → 重排 → 阈值/TopN → ANN → 工具。
///
/// 默认使用仓库 `models/` 下的内置 V5 embedding 与 Jina reranker；
/// 也可以用 `EMBED_BENCH_MODELS_DIR` 指向 App 包的
/// `Contents/Resources/models` 做发布产物回归。
/// 任一模型缺失时自动跳过，因此可随普通 `swift test` 一起运行。
final class RAGPipelineVerificationTests: XCTestCase {
    private let modelName = "jina-embeddings-v5-text-nano-mlx-int4"
    private let rerankerName = "jina-reranker-v2-base-multilingual-onnx"
    private var root: URL!
    private var kb: URL!

    override func setUpWithError() throws {
        EmbeddingBenchmarkTests.configureMetallib()
        unsetenv("BASE_AGENT_MODELS_DIR")
        LocalModelFactory.shared.clearCaches()
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rag-pipeline-\(UUID().uuidString)")
        kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        unsetenv("BASE_AGENT_MODELS_DIR")
        LocalModelFactory.shared.clearCaches()
        if let root { try? FileManager.default.removeItem(at: root) }
    }

    func testBundledModelsFullPipeline() async throws {
        // 仓库 models/ 优先；发布回归时用 EMBED_BENCH_MODELS_DIR 指向
        // `dist/Lightning-RAG.app/Contents/Resources/models`。
        let modelsRoot = EmbeddingBenchmarkTests.resolveModelsRoot()
        let embeddingDir = modelsRoot.appendingPathComponent(modelName, isDirectory: true)
        let rerankerDir = modelsRoot.appendingPathComponent(rerankerName, isDirectory: true)
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: embeddingDir.appendingPathComponent("model.safetensors").path)
                || FileManager.default.fileExists(
                    atPath: embeddingDir.appendingPathComponent("weights.safetensors").path),
            "embedding 模型缺失：\(embeddingDir.path)")
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: rerankerDir.appendingPathComponent("model_quantized.onnx").path),
            "reranker 模型缺失：\(rerankerDir.path)")
        setenv("BASE_AGENT_MODELS_DIR", modelsRoot.path, 1)
        defer { unsetenv("BASE_AGENT_MODELS_DIR") }

        let targets = Array(EmbeddingBenchmarkTests.targets.prefix(6))
        let documentNames = targets.indices.map { String(format: "doc%02d.md", $0 + 1) }
        for (index, target) in targets.enumerated() {
            let text = "# 文档\(index + 1)\n\n\(target.passage)\n"
            try text.write(to: kb.appendingPathComponent(documentNames[index]),
                           atomically: true, encoding: .utf8)
        }

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragChunkSize = 800
        config.ragChunkOverlap = 80
        config.ragSemanticChunking = true
        config.ragVectorElementType = .int8
        config.ragRerankEnabled = true
        config.ragRerankTopN = 5
        config.ragRerankThreshold = 0.45
        config.ragANNEnabled = true
        config.ragANNAutoBuild = false
        config.ragANNMinChunks = 0
        config.ragANNClusters = 4
        config.ragANNClusterSize = 16
        config.ragANNProbe = 4
        config.ragANNRescore = 64
        config.ragQueryTimeoutSeconds = 30

        let embedding = try XCTUnwrap(
            LocalModelFactory.shared.makeEmbedding(config: config)
                as? JinaV5MLXEmbeddingProvider,
            "默认配置未解析到内置 Jina V5 embedding")
        let reranker = try XCTUnwrap(
            LocalModelFactory.shared.makeReranker(config: config) as? ONNXReranker,
            "默认配置未解析到内置 Jina reranker")
        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: embedding, reranker: reranker)
        defer { engine.close() }
        addTeardownBlock { await embedding.unload() }

        let started = Date()
        let stats = await engine.ingest(force: true)
        let elapsed = Date().timeIntervalSince(started)
        XCTAssertTrue(stats.errors.isEmpty, stats.errors.joined(separator: "; "))
        XCTAssertEqual(stats.processed, targets.count)
        XCTAssertGreaterThanOrEqual(stats.chunks, targets.count)
        print("[rag-verify] ingest processed=\(stats.processed) chunks=\(stats.chunks) "
            + "elapsed=\(String(format: "%.2f", elapsed))s")

        let status = engine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(status["vector_element_type"]?.stringValue, "int8")
        XCTAssertEqual(status["vector_rebuild_required"]?.boolValue, false)
        XCTAssertTrue(
            status["vector_index_fingerprint"]?.stringValue?.contains("dimension=128") == true,
            "索引维度应为 128：\(status["vector_index_fingerprint"]?.stringValue ?? "-")")
        XCTAssertEqual(status["source_count"]?.intValue, targets.count)
        print("[rag-verify] status chunks=\(status["chunks_stored"]?.intValue ?? -1) "
            + "backend=\(status["vector_backend"]?.stringValue ?? "-") "
            + "element=\(status["vector_element_type"]?.stringValue ?? "-")")

        let db = try SQLiteDatabase(path: engine.indexDBPath, readOnly: true)
        defer { db.close() }
        let meta = try db.query(
            "SELECT key, value FROM meta WHERE key IN ('embedding_dim', 'vector_element_type')")
        let values = Dictionary(uniqueKeysWithValues: meta.compactMap { row in
            row["key"]?.stringValue.flatMap { key in
                row["value"]?.stringValue.map { (key, $0) }
            }
        })
        XCTAssertEqual(values["embedding_dim"], "128")
        XCTAssertEqual(values["vector_element_type"], "int8")

        // 直接对全部切片重排，排除候选池/排序映射对重排能力的干扰。
        let chunkRows = try db.query("SELECT source, text FROM chunks")
        let chunkTexts = chunkRows.compactMap { $0["text"]?.stringValue }
        let chunkSources = chunkRows.compactMap { $0["source"]?.stringValue }
        for index in [3, 5] {
            let scores = try await reranker.rerank(
                query: targets[index].query, documents: chunkTexts)
            let ranked = zip(chunkSources, scores).sorted { $0.1 > $1.1 }
            let top = ranked.prefix(3).map {
                "(\(($0.0 as NSString).lastPathComponent), \(String(format: "%.4f", $0.1)))"
            }
            print("[rag-verify] all-chunks rerank q\(index + 1) top=\(top)")
            XCTAssertEqual((ranked.first?.0 as NSString?)?.lastPathComponent,
                           documentNames[index],
                           "全部切片重排应把目标文档排第一：\(top)")
            XCTAssertGreaterThanOrEqual(ranked.first?.1 ?? 0, config.ragRerankThreshold,
                                        "目标切片应达到默认重排阈值：\(top)")
        }

        // 关闭重排的基线。
        var plainConfig = config
        plainConfig.ragRerankEnabled = false
        let plainEngine = RAGEngine(config: plainConfig, knowledgeBase: kb.path,
                                    embedding: embedding, reranker: nil)
        defer { plainEngine.close() }

        var plainTop1 = 0
        var rerankedTop1 = 0
        for (index, target) in targets.enumerated() {
            let expected = documentNames[index]
            let plainHits = try await plainEngine.search(query: target.query, topK: 3)
            let plainSources = plainHits.map { ($0.source as NSString).lastPathComponent }
            let plainRank = plainSources.firstIndex(of: expected).map { $0 + 1 } ?? Int.max
            if plainRank == 1 { plainTop1 += 1 }

            let hits = try await engine.search(query: target.query, topK: 5)
            XCTAssertLessThanOrEqual(hits.count, config.ragRerankTopN)
            let sources = hits.map { ($0.source as NSString).lastPathComponent }
            let rank = sources.firstIndex(of: expected).map { $0 + 1 } ?? Int.max
            if rank == 1 { rerankedTop1 += 1 }
            for hit in hits {
                XCTAssertTrue(hit.score >= 0 && hit.score <= 1,
                              "重排分数应归一化到 [0,1]：\(hit.score)")
            }
            let labeled = hits.map {
                "(\(($0.source as NSString).lastPathComponent),"
                    + " \(String(format: "%.4f", $0.score)))"
            }
            print("[rag-verify] query=\(target.query) "
                + "plainRank=\(plainRank == Int.max ? "-" : "\(plainRank)") "
                + "rerankRank=\(rank == Int.max ? "-" : "\(rank)") "
                + "hits=\(labeled)")
            XCTAssertLessThanOrEqual(rank, 3, "相关文档必须进入重排后 top3")
        }
        XCTAssertEqual(plainTop1, targets.count, "无重排时相关文档应全部 top1")
        XCTAssertEqual(rerankedTop1, targets.count, "重排后相关文档应全部 top1")

        // 全部低于阈值时回退原排序的候选，不能返回空。
        var strictConfig = config
        strictConfig.ragRerankThreshold = 0.95
        let strictEngine = RAGEngine(config: strictConfig, knowledgeBase: kb.path,
                                     embedding: embedding, reranker: reranker)
        defer { strictEngine.close() }
        let strictHits = try await strictEngine.search(query: targets[0].query, topK: 10)
        XCTAssertFalse(strictHits.isEmpty, "全部低于阈值时应回退候选而不是返回空")
        XCTAssertLessThanOrEqual(strictHits.count, strictConfig.ragRerankTopN)
        print("[rag-verify] strict-threshold fallback count=\(strictHits.count)")

        // ANN 索引构建与检索。二值签名覆盖完整时构建目标为二值 ANN，
        // 否则回退主 ANN；这里按当前解析目标校验对应索引。
        let indexTarget = engine.resolvedANNIndexTarget()
        let build = try XCTUnwrap(engine.buildANNIndex(), "ANN 索引构建被拒绝")
        XCTAssertFalse(build.cancelled)
        XCTAssertGreaterThan(build.indexedRows, 0)
        let indexStatus = try XCTUnwrap(
            indexTarget == .binary
                ? engine.binaryANNIndexStatus()
                : engine.annIndexStatus(),
            "ANN 状态不可用")
        XCTAssertTrue(indexStatus.isReady, indexStatus.summary)
        print("[rag-verify] \(indexTarget.rawValue) ann \(indexStatus.summary)")

        let annHits = try await engine.search(query: targets[2].query, topK: 3)
        let annSources = annHits.map { ($0.source as NSString).lastPathComponent }
        XCTAssertEqual(annSources.first, documentNames[2],
                       "ANN 开启后相关文档应仍为 top1：\(annSources)")

        // 工具入口。
        let searchTool = RagSearchTool(config: config, engine: engine)
        let searchResult = try await searchTool.run([
            "query": .string(targets[1].query),
        ])
        XCTAssertTrue(searchResult.success, searchResult.output)
        XCTAssertTrue(searchResult.output.contains(documentNames[1]), searchResult.output)
        XCTAssertTrue(searchResult.output.contains("相关度") || searchResult.output.contains("RRF"))

        let outlineTool = RagOutlineTool(config: config, engine: engine)
        let outlineResult = try await outlineTool.run([:])
        XCTAssertTrue(outlineResult.success, outlineResult.output)
        for name in documentNames {
            XCTAssertTrue(outlineResult.output.contains(name), outlineResult.output)
        }

        let statusTool = RagStatusTool(config: config, engine: engine)
        let statusResult = try await statusTool.run([:])
        XCTAssertTrue(statusResult.success, statusResult.output)
        XCTAssertTrue(statusResult.output.contains("已存储切片数"), statusResult.output)
        XCTAssertTrue(statusResult.output.contains("int8"), statusResult.output)

        print("[rag-verify] tools search/outline/status OK")
    }
}
