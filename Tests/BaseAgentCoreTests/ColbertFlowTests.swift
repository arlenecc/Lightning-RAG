import XCTest
import Foundation
@testable import BaseAgentCore
@testable import ColBERTReader

/// 检索阅读模式全流程的状态与分流回归（不依赖真实模型/知识库）。
final class ColbertFlowTests: XCTestCase {
    private func makeEngine(enabled: Bool) -> RAGEngine {
        var config = AgentConfig()
        config.colbertReaderEnabled = enabled
        return RAGEngine(config: config, knowledgeBase: "colbert-flow-test")
    }

    /// ColBERT 模式下 status 走独立快照：vector_backend=colbert、rag_mode=colbert，
    /// 不再尝试打开普通向量库而误报「向量库不可用」。
    func testStatusUsesColbertBackend() {
        let engine = makeEngine(enabled: true)
        let status = engine.status(maxAge: 0)
        XCTAssertEqual(status["vector_backend"]?.stringValue, "colbert")
        XCTAssertEqual(status["rag_mode"]?.stringValue, "colbert")
        XCTAssertNotNil(status["chunks_stored"]?.intValue)
    }

    /// 普通模式 status 不带 colbert 标识，避免两套分流互相污染。
    func testPlainModeStatusNotTaggedColbert() {
        let engine = makeEngine(enabled: false)
        let status = engine.status(maxAge: 0)
        XCTAssertNotEqual(status["vector_backend"]?.stringValue, "colbert")
        XCTAssertNil(status["rag_mode"]?.stringValue)
    }

    /// 关闭检索阅读模式时构建 ANN 应明确拒绝，而不是静默建普通库。
    func testBuildColbertANNDisabledThrows() async {
        let engine = makeEngine(enabled: false)
        do {
            _ = try await engine.buildColbertANN()
            XCTFail("关闭模式时 buildColbertANN 应抛错")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("未启用")
                          || error.localizedDescription.contains("colbert")
                          || error is AgentError)
        }
    }

    /// 关闭模式时 colbertANNStatus 返回「未启用」，供 UI 清空旧摘要。
    func testColbertANNStatusDisabled() {
        let engine = makeEngine(enabled: false)
        XCTAssertEqual(engine.colbertANNStatus(), "未启用")
    }

    /// top-1 分片原文进上下文供 LLM 总结，其余分片只提示在气泡里。
    func testPrimaryContextIncludesTop1TextOnly() {
        let citations = [
            RAGCitation(id: "1", rank: 1, source: "合同法.md", heading: "第6章 违约责任",
                        text: "TOP1 分片正文：当事人一方不履行合同义务应当承担违约责任。",
                        score: 0.83, isPrimary: true),
            RAGCitation(id: "2", rank: 2, source: "海商法.md", heading: "",
                        text: "TOP2 分片正文（只在气泡）：船舶优先权。", score: 0.51),
        ]
        let context = ColBERTReaderService.primaryContext(citations: citations)
        XCTAssertTrue(context.contains("TOP1 分片正文"))
        XCTAssertFalse(context.contains("TOP2 分片正文"))
        XCTAssertTrue(context.contains("合同法.md"))
        XCTAssertTrue(context.contains("第6章 违约责任"))
        XCTAssertTrue(context.contains("其余 1 条"))
        XCTAssertTrue(context.contains("0.8300"))
    }

    /// 只有 1 条召回时上下文用「全部进入」措辞，且仍带原文。
    func testPrimaryContextSingleHit() {
        let citations = [
            RAGCitation(id: "1", rank: 1, source: "doc.md", text: "唯一分片正文。",
                        score: 0.7, isPrimary: true),
        ]
        let context = ColBERTReaderService.primaryContext(citations: citations)
        XCTAssertTrue(context.contains("全部进入上下文"))
        XCTAssertTrue(context.contains("唯一分片正文"))
    }

    /// 空召回给出明确提示，不让 LLM 臆造。
    func testPrimaryContextEmpty() {
        XCTAssertTrue(ColBERTReaderService.primaryContext(citations: [])
            .contains("未找到相关内容"))
    }

    /// 长度归一化开关默认关（原始 sum-of-max），且能经 JSON 配置往返（开启态被正确读回）。
    func testLengthNormConfigRoundTrip() throws {
        XCTAssertFalse(AgentConfig().colbertLengthNorm, "默认应关闭长度归一化")
        var config = AgentConfig()
        config.colbertLengthNorm = true
        config.colbertReaderEnabled = true
        let data = try JSONSerialization.data(withJSONObject: config.export())
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("colbert-norm-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("config.json")
        try data.write(to: file)
        let loaded = AgentConfig.load(file.path)
        XCTAssertTrue(loaded.colbertLengthNorm, "colbert_length_norm=true 应被读回")
    }

    /// 粗筛候选数严格由面板档位派生：3 倍融合池、无隐藏下限，档位 10 不再被抬到 60。
    func testCoarseCandidatesStrictlyDriveFetchK() {
        XCTAssertEqual(SearchEngine.coarseFetchK(10), 30)
        XCTAssertEqual(SearchEngine.coarseFetchK(20), 60)
        XCTAssertEqual(SearchEngine.coarseFetchK(50), 150)
        XCTAssertEqual(SearchEngine.coarseFetchK(100), 300)
        // 非法值（<1）防御性归到最小 3 条，不出现 0/负候选。
        XCTAssertEqual(SearchEngine.coarseFetchK(0), 3)
        XCTAssertEqual(SearchEngine.coarseFetchK(-5), 3)
    }

    /// 指定来源的兜底上限随 coarse 放大，且不低于 200，避免写死上限漏召回。
    func testSourceCandidateLimitFollowsCoarse() {
        XCTAssertEqual(SearchEngine.sourceCandidateLimit(10), 200)
        XCTAssertEqual(SearchEngine.sourceCandidateLimit(50), 200)
        XCTAssertEqual(SearchEngine.sourceCandidateLimit(100), 400)
    }

    /// 查询侧剪枝默认开启（轻量），且能经 JSON 配置往返（关闭态被正确读回）。
    func testQueryPruneConfigRoundTrip() throws {
        XCTAssertTrue(AgentConfig().colbertQueryPrune, "默认应开启查询侧轻量剪枝")
        var config = AgentConfig()
        config.colbertQueryPrune = false
        config.colbertReaderEnabled = true
        let data = try JSONSerialization.data(withJSONObject: config.export())
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("colbert-qprune-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("config.json")
        try data.write(to: file)
        let loaded = AgentConfig.load(file.path)
        XCTAssertFalse(loaded.colbertQueryPrune, "colbert_query_prune=false 应被读回")
        // 引擎选项默认也应为开启；显式关闭应贯通。
        XCTAssertTrue(ColBERTReaderOptions().queryPrune)
        XCTAssertFalse(ColBERTReaderOptions(queryPrune: false).queryPrune)
    }

    /// 开关贯通到引擎选项：关闭时 maxSim 不做长度归一化（分数等于原始 sum-of-max）。
    func testLengthNormOptionReachesMaxSim() throws {
        let dim = 64
        let vectors: [[Float]] = (0..<900).map { i in
            var v = [Float](repeating: 0, count: dim)
            v[i % 4] = 1
            return v
        }
        let record = try ResidualCodec.quantize(docID: "long", vectors: vectors, dim: dim)
        var q = [Float](repeating: 0, count: dim)
        q[0] = 1
        let raw = try ResidualCodec.maxSim(query: [q], record: record, normalizeLength: false)
        let norm = try ResidualCodec.maxSim(query: [q], record: record, normalizeLength: true)
        XCTAssertLessThan(norm, raw, "开启长度归一化时长片分数应被压低")
        let opts = ColBERTReaderOptions(lengthNorm: true)
        XCTAssertTrue(opts.lengthNorm)
        XCTAssertFalse(ColBERTReaderOptions().lengthNorm)
    }
}
