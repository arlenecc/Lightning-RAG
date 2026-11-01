import XCTest
import Foundation
@testable import BaseAgentCore

/// 真实语料 RAG 全流程验证（临时，不提交）。
///
/// 复用 `REAL_BENCH_KEEP=1` 留下的索引，覆盖：
/// 状态/覆盖率 → bann 构建 → 全流程检索（bann + rerank + RRF）→ 工具。
///
/// 环境变量：
/// - `REAL_FLOW_ROOT`：索引根，默认 `/tmp/real_bench_root`
/// - `REAL_FLOW_LIST`：语料列表（用于跳过校验），可选
final class RealCorpusFullFlowTests: XCTestCase {
    private var root: URL!
    private var kb: URL!
    private var engine: RAGEngine!
    private var reranker: RerankProvider?

    override func setUpWithError() throws {
        let env = ProcessInfo.processInfo.environment
        root = URL(fileURLWithPath: env["REAL_FLOW_ROOT"] ?? "/tmp/real_bench_root")
        kb = root.appendingPathComponent("kb")
        guard FileManager.default.fileExists(atPath: kb.path) else {
            throw XCTSkip("真实语料索引不存在：\(root.path)")
        }
        EmbeddingBenchmarkTests.configureMetallib()
        LocalModelFactory.shared.clearCaches()
        let modelsRoot = EmbeddingBenchmarkTests.resolveModelsRoot()
        setenv("BASE_AGENT_MODELS_DIR", modelsRoot.path, 1)
    }

    override func tearDownWithError() throws {
        engine?.close()
        let reranker = self.reranker
        Task { await reranker?.unload() }
        unsetenv("BASE_AGENT_MODELS_DIR")
        LocalModelFactory.shared.clearCaches()
    }

    private func makeConfig(rerank: Bool, binary: Bool) -> AgentConfig {
        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragMatryoshkaDim = 128
        config.ragVectorElementType = .int8
        config.ragBinaryCoarseEnabled = binary
        config.ragBinaryCoarseCandidates = 200
        config.ragANNEnabled = false
        config.ragANNAutoBuild = false
        config.ragRerankEnabled = rerank
        config.ragRerankTopN = 5
        config.ragRerankThreshold = 0
        config.ragQueryTimeoutSeconds = 60
        return config
    }

    private func sourceName(_ source: String) -> String {
        (source as NSString).lastPathComponent
    }

    private func isExpected(_ hit: SearchHit, source: String) -> Bool {
        sourceName(hit.source) == sourceName(source)
    }

    private struct QueryCase {
        let text: String
        let source: String
        let rowID: Int64
        let kind: String
        /// 只有「原文片段」查询有明确的目标切片；标题查询只校验来源。
        let checksChunk: Bool
    }

    private func loadCases(_ db: SQLiteDatabase) throws -> [QueryCase] {
        let rows = try db.query(
            "SELECT rowid, source, text FROM \(VectorStore.chunksTable) ORDER BY rowid")
        var grouped: [String: [(rowID: Int64, text: String)]] = [:]
        for row in rows {
            guard let rowID = row["rowid"]?.intValue,
                  let source = row["source"]?.stringValue,
                  let text = row["text"]?.stringValue, !text.isEmpty else { continue }
            grouped[source, default: []].append((rowID, text))
        }
        let sources = grouped.keys.sorted()
        var cases: [QueryCase] = []
        for source in sources {
            guard let chunks = grouped[source], !chunks.isEmpty else { continue }
            let base = ((source as NSString).lastPathComponent as NSString)
                .deletingPathExtension
            let title = base.components(separatedBy: " - ").first ?? base
            cases.append(QueryCase(text: title, source: source,
                                   rowID: chunks[0].rowID, kind: "title",
                                   checksChunk: false))
            let middle = chunks[chunks.count / 2]
            let passage = String(middle.text.prefix(180))
                .replacingOccurrences(of: "\n", with: " ")
            cases.append(QueryCase(text: passage, source: source,
                                   rowID: middle.rowID, kind: "passage",
                                   checksChunk: true))
        }
        return cases
    }

    private func evaluate(_ label: String, engine: RAGEngine) async throws
        -> (sourceTop1: Double, sourceTop5: Double, chunkTop5: Double,
            millis: Double, cases: [QueryCase]) {
        let db = try SQLiteDatabase(path: engine.indexDBPath, readOnly: true)
        defer { db.close() }
        let cases = try loadCases(db)
        _ = try await engine.search(query: cases[0].text, topK: 5)
        var times: [Double] = []
        var sourceTop1 = 0
        var sourceTop5 = 0
        var chunkTop5 = 0
        var chunkCases = 0
        var misses: [String] = []
        for item in cases {
            let started = CFAbsoluteTimeGetCurrent()
            let hits = try await engine.search(query: item.text, topK: 5)
            times.append((CFAbsoluteTimeGetCurrent() - started) * 1000)
            let hasSource = hits.contains(where: { isExpected($0, source: item.source) })
            if hits.first.map({ isExpected($0, source: item.source) }) == true {
                sourceTop1 += 1
            }
            if hasSource {
                sourceTop5 += 1
            } else {
                misses.append("\(item.kind):\(sourceName(item.source))")
            }
            if item.checksChunk {
                chunkCases += 1
                if hits.contains(where: { $0.rowID == item.rowID }) {
                    chunkTop5 += 1
                }
            }
        }
        let sorted = times.sorted()
        let p50 = sorted[min(sorted.count - 1, sorted.count / 2)]
        func ratio(_ value: Int, of total: Int) -> Double {
            Double(value) / Double(max(1, total)) * 100
        }
        print(String(format: "[flow] %@ cases=%d source@1=%.1f%% source@5=%.1f%% "
                     + "self_chunk@5=%.1f%% (%d/%d) p50=%.1fms misses=%@",
                     label, cases.count, ratio(sourceTop1, of: cases.count),
                     ratio(sourceTop5, of: cases.count),
                     ratio(chunkTop5, of: chunkCases), chunkTop5, chunkCases,
                     p50, misses.isEmpty ? "none" : misses.joined(separator: ",")))
        return (ratio(sourceTop1, of: cases.count),
                ratio(sourceTop5, of: cases.count),
                ratio(chunkTop5, of: chunkCases), p50, cases)
    }

    func testRealCorpusFullFlow() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["REAL_FLOW_BENCH"] == "1" else {
            throw XCTSkip("未设置 REAL_FLOW_BENCH=1")
        }
        let embedding = try XCTUnwrap(LocalModelFactory.shared.makeEmbedding(
            config: makeConfig(rerank: false, binary: true)))
        guard let signing = embedding as? EmbeddingBinarySigning else {
            throw XCTSkip("embedding 不支持二值签名")
        }
        XCTAssertGreaterThan(signing.binaryDimension, 0)

        let baseEngine = RAGEngine(
            config: makeConfig(rerank: false, binary: true),
            knowledgeBase: kb.path, embedding: embedding, reranker: nil)
        engine = baseEngine

        // 先只读校验指纹，避免 VectorStore 误判后重置真实索引。
        let fingerprint = baseEngine.vectorIndexFingerprint
        let db = try SQLiteDatabase(path: baseEngine.indexDBPath, readOnly: true)
        defer { db.close() }
        let meta = try db.query(
            "SELECT key, value FROM meta WHERE key IN "
            + "('vector_index_fingerprint', 'embedding_dim', 'binary_dimension')")
        let values = Dictionary(uniqueKeysWithValues: meta.compactMap { row in
            row["key"]?.stringValue.flatMap { key in
                row["value"]?.stringValue.map { (key, $0) }
            }
        })
        XCTAssertEqual(values["vector_index_fingerprint"], fingerprint)
        XCTAssertEqual(values["embedding_dim"], "128")
        XCTAssertEqual(values["binary_dimension"], "768")

        let status = baseEngine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(status["chunks_stored"]?.intValue, 7999)
        XCTAssertEqual(status["source_count"]?.intValue, 15)
        XCTAssertEqual(baseEngine.resolvedANNIndexTarget().rawValue, "binary")
        // 主向量 / 二值签名 / FTS / documents 四套存储与切片数一致。
        for table in [VectorStore.vectorTable, VectorStore.binaryVectorTable,
                      VectorStore.ftsTable] {
            XCTAssertEqual(
                try db.scalar("SELECT COUNT(*) FROM \(table)").intValue, 7999,
                "\(table) 行数与切片不一致")
        }
        XCTAssertEqual(
            try db.scalar("SELECT COUNT(*) FROM \(VectorStore.documentsTable)").intValue, 15)
        // manifest：15 本全部解析/切片完成，且没有重复或过短丢弃。
        let manifest = ManifestStore(
            path: (baseEngine.ragDir as NSString).appendingPathComponent("manifest.json")).load()
        XCTAssertEqual(manifest.count, 15)
        XCTAssertEqual(manifest.values.compactMap(\.chunkCount).reduce(0, +), 7999)
        XCTAssertTrue(manifest.values.allSatisfy {
            ($0.complete ?? false) && ($0.duplicateOf ?? "").isEmpty
                && ($0.discarded ?? false) == false
        }, "存在未完成/重复/丢弃的来源：\(manifest.keys.sorted())")
        XCTAssertEqual(status["binary_coarse"]?.stringValue, "ready")
        XCTAssertEqual(status["binary_ann"]?.stringValue, "ready")
        XCTAssertEqual(status["binary_backend"]?.stringValue, "bann")
        // 旧索引可能尚未建立 stage 2 镜像：此时应明确回退 vec0，而不是谎报 rerank。
        let rerankRows = try XCTUnwrap(status["rerank_rows"]?.intValue)
        let mirrorCount = Int(
            try db.scalar("SELECT COUNT(*) FROM \(VectorStore.rerankVectorTable)").intValue ?? 0)
        XCTAssertEqual(rerankRows, mirrorCount)
        XCTAssertEqual(status["stage2_backend"]?.stringValue,
                       rerankRows == 7999 ? "rerank" : "vec0")
        print("[flow] status chunks=\(status["chunks_stored"]?.intValue ?? -1) "
            + "binary=\(status["binary_coarse"]?.stringValue ?? "-")/"
            + "\(status["binary_backend"]?.stringValue ?? "-") "
            + "bann=\(status["binary_ann"]?.stringValue ?? "-") "
            + "stage2=\(status["stage2_backend"]?.stringValue ?? "-") "
            + "mirror=\(rerankRows)")

        let fullConfig = makeConfig(rerank: true, binary: true)
        reranker = LocalModelFactory.shared.makeReranker(config: fullConfig)
        guard let reranker else { throw XCTSkip("reranker 模型缺失") }
        let fullEngine = RAGEngine(config: fullConfig, knowledgeBase: kb.path,
                                   embedding: embedding, reranker: reranker)
        defer { fullEngine.close() }

        let plainConfig = makeConfig(rerank: false, binary: true)
        let plainEngine = RAGEngine(config: plainConfig, knowledgeBase: kb.path,
                                    embedding: embedding, reranker: nil)
        defer { plainEngine.close() }

        let exactConfig = makeConfig(rerank: false, binary: false)
        let exactEngine = RAGEngine(config: exactConfig, knowledgeBase: kb.path,
                                    embedding: embedding, reranker: nil)
        defer { exactEngine.close() }

        let full = try await evaluate("bann+rerank", engine: fullEngine)
        let plain = try await evaluate("bann", engine: plainEngine)
        let exact = try await evaluate("exact-dense", engine: exactEngine)
        // 以精确 dense 为基线：二值粗筛只允许有限的近似损失，不能出现整本漏召。
        let sourceFloor = min(95, exact.sourceTop5)
        XCTAssertGreaterThanOrEqual(full.sourceTop5, sourceFloor,
                                    "全流程 source@5 低于精确基线：\(full.sourceTop5)%")
        XCTAssertGreaterThanOrEqual(plain.sourceTop5, sourceFloor,
                                    "bann source@5 低于精确基线：\(plain.sourceTop5)%")
        XCTAssertLessThanOrEqual(exact.sourceTop5 - full.sourceTop5, 5,
                                 "全流程相对精确路径 source@5 损失过大")
        XCTAssertGreaterThanOrEqual(full.chunkTop5, 90, "全流程 self chunk@5 召回不足")
        XCTAssertLessThanOrEqual(exact.chunkTop5 - full.chunkTop5, 10,
                                 "全流程相对精确路径 chunk@5 损失过大")
        print(String(format: "[flow] delta source@5 full=%.1fpp plain=%.1fpp "
                     + "chunk@5 full=%.1fpp",
                     full.sourceTop5 - exact.sourceTop5,
                     plain.sourceTop5 - exact.sourceTop5,
                     full.chunkTop5 - exact.chunkTop5))

        // 工具入口。
        let toolConfig = fullConfig
        let searchTool = RagSearchTool(config: toolConfig, engine: fullEngine)
        let searchCase = try XCTUnwrap(full.cases.first(where: \.checksChunk))
        let searchResult = try await searchTool.run([
            "query": .string(searchCase.text),
        ])
        XCTAssertTrue(searchResult.success, searchResult.output)
        XCTAssertTrue(searchResult.output.contains("相关度")
            || searchResult.output.contains("RRF"), searchResult.output)
        XCTAssertTrue(searchResult.output.contains(sourceName(searchCase.source)),
                      searchResult.output)

        let outlineTool = RagOutlineTool(config: toolConfig, engine: fullEngine)
        let outlineResult = try await outlineTool.run([:])
        XCTAssertTrue(outlineResult.success, outlineResult.output)
        for name in fullEngine.listDocuments(limit: 100).names {
            XCTAssertTrue(outlineResult.output.contains((name as NSString).lastPathComponent),
                          outlineResult.output)
        }
        // 单本文档大纲：目录/摘要内容也必须能取出，而不只是文件名列表。
        let firstName = try XCTUnwrap(fullEngine.listDocuments(limit: 1).names.first)
        let bookOutline = try await outlineTool.run(["book_name": .string(firstName)])
        XCTAssertTrue(bookOutline.success, bookOutline.output)
        XCTAssertTrue(bookOutline.output.contains("文档:"), bookOutline.output)
        XCTAssertTrue(bookOutline.output.contains("章节结构:")
            || bookOutline.output.contains("内容摘要:"), bookOutline.output)

        let statusTool = RagStatusTool(config: toolConfig, engine: fullEngine)
        let statusResult = try await statusTool.run([:])
        XCTAssertTrue(statusResult.success, statusResult.output)
        XCTAssertTrue(statusResult.output.contains("已存储切片数"), statusResult.output)
        XCTAssertTrue(statusResult.output.contains("二值 ANN"), statusResult.output)
        print("[flow] tools search/outline/status OK")

        // 来源定向检索。
        let sourceCase = full.cases[0]
        let filtered = try await fullEngine.search(
            query: sourceCase.text, topK: 5, source: sourceName(sourceCase.source))
        XCTAssertFalse(filtered.isEmpty)
        XCTAssertTrue(filtered.allSatisfy {
            isExpected($0, source: sourceCase.source)
        }, "来源过滤仍返回其它来源：\(filtered.map(\.source))")
        print("[flow] source filter OK hits=\(filtered.count)")
    }
}
