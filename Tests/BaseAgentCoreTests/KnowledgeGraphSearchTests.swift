import Foundation
import XCTest
@testable import BaseAgentCore

/// RAGEngine 知识图谱检索（local/global/hybrid/mix）集成测试。
/// 用门面手动建一个小图，验证四种模式命中不同的子图。
final class KnowledgeGraphSearchTests: XCTestCase {

    private final class FakeEmbedding: EmbeddingProvider, @unchecked Sendable {
        let identifier = "fake-embedding"
        let dimension = 3
        func embedQuery(_ text: String) async throws -> [Float] { [1, 0, 0] }
        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map { _ in [1, 0, 0] }
        }
    }

    private var rootDir: URL?
    private var engine: RAGEngine!

    override func setUp() {
        super.setUp()
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("kg-search-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let kb = root.appendingPathComponent("kb")
        try? FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        var config = AgentConfig()
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        engine = RAGEngine(config: config, knowledgeBase: kb.path,
                           embedding: FakeEmbedding(), reranker: nil)
        rootDir = root
    }

    override func tearDown() {
        if let dir = rootDir { try? FileManager.default.removeItem(at: dir) }
        rootDir = nil
        engine = nil
        super.tearDown()
    }

    private func buildSmallGraph() async throws {
        _ = try await engine.graphAddEntity(name: "特朗普", type: "person")
        _ = try await engine.graphAddEntity(name: "总统职位", type: "concept")
        _ = try await engine.graphAddEntity(name: "拜登", type: "person")
        _ = try await engine.graphAddRelationship(source: "特朗普", target: "总统职位", relation: "担任")
        _ = try await engine.graphAddRelationship(source: "拜登", target: "总统职位", relation: "担任")
    }

    func testLocalModePullsNeighborSubgraph() async throws {
        try await buildSmallGraph()
        let r = try await engine.graphSearch(query: "特朗普", mode: .local, topK: 20)
        let names = r.entities.map { $0.name }
        XCTAssertTrue(names.contains("特朗普"), "命中实体应出现")
        XCTAssertTrue(names.contains("总统职位"), "邻居实体应被拉入（local 拉邻边）")
    }

    func testGlobalModeReachesBothEnds() async throws {
        try await buildSmallGraph()
        let r = try await engine.graphSearch(query: "担任", mode: .global, topK: 20)
        // 命中关系「担任」反查两端 + 中心实体：特朗普、拜登、总统职位
        let names = Set(r.entities.map { $0.name })
        XCTAssertTrue(names.contains("总统职位"))
        XCTAssertTrue(names.contains("特朗普"))
        XCTAssertTrue(names.contains("拜登"))
    }

    func testHybridModeCombinesEntityAndRelation() async throws {
        try await buildSmallGraph()
        let r = try await engine.graphSearch(query: "特朗普 担任", mode: .hybrid, topK: 20)
        XCTAssertFalse(r.entities.isEmpty)
        let names = Set(r.entities.map { $0.name })
        // 实体命中（特朗普）+ 关系命中（担任反查拜登/总统职位）
        XCTAssertTrue(names.contains("特朗普"))
        XCTAssertTrue(names.contains("总统职位"))
    }

    func testMixModeReturnsGraphPlusChunks() async throws {
        try await buildSmallGraph()
        let r = try await engine.graphSearch(query: "特朗普", mode: .mix, topK: 20)
        XCTAssertFalse(r.entities.isEmpty, "mix 至少应给图谱部分")
        // 原文 chunk 依赖向量库（本测试未 ingest），可能为空，不强制断言
    }

    func testUnknownQueryReturnsEmpty() async throws {
        try await buildSmallGraph()
        let r = try await engine.graphSearch(query: "不存在的实体xyz", mode: .local, topK: 20)
        XCTAssertTrue(r.entities.isEmpty)
    }

    func testGetEntityAndNeighbors() async throws {
        try await buildSmallGraph()
        let fetched = engine.graphEntity(named: "总统职位")
        XCTAssertNotNil(fetched)
        let neighbors = engine.graphNeighbors(name: "总统职位")
        XCTAssertEqual(Set(neighbors), ["特朗普", "拜登"], "总统职位应连接两任总统")
    }
}
