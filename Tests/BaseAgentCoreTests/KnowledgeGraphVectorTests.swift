import Foundation
import XCTest
@testable import BaseAgentCore

/// 知识图谱实体/关系向量化：文本格式、token 截断、关系双向 id、构建状态与排序。
final class KnowledgeGraphVectorTests: XCTestCase {

    private var files: [String] = []

    override func tearDown() {
        for f in files { try? FileManager.default.removeItem(atPath: f) }
        files.removeAll()
        super.tearDown()
    }

    /// 用空白分词模拟 tokenizer；向量按关键字区分。
    private final class CountingEmbedding: EmbeddingProvider, EmbeddingTokenCounting,
                                           EmbeddingQueryBatching, @unchecked Sendable {
        let identifier = "kg-vector-test"
        let dimension = 4
        var maxContentTokens: Int { 10_000 }

        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map(Self.vector)
        }

        func embedQuery(_ text: String) async throws -> [Float] { Self.vector(text) }

        func embedQueries(_ texts: [String]) async throws -> [[Float]] {
            texts.map(Self.vector)
        }

        func makeTokenCounter() async -> (@Sendable (String) -> Int)? {
            { text in text.split(whereSeparator: \.isWhitespace).count }
        }

        private static func vector(_ text: String) -> [Float] {
            let lower = text.lowercased()
            if lower.contains("alpha") { return [1, 0, 0, 0] }
            if lower.contains("beta") { return [0, 1, 0, 0] }
            return [0, 0, 1, 0]
        }
    }

    private func makeStore() throws -> (KnowledgeGraphStore, String) {
        let path = NSTemporaryDirectory()
            .appending("kg-vector-\(UUID().uuidString).sqlite")
        files.append(path)
        return (try KnowledgeGraphStore(dbPath: path, embedding: CountingEmbedding()), path)
    }

    private func entity(_ name: String, description: String,
                        sources: [String] = []) -> KGEntity {
        KGEntity(id: UUID().uuidString, name: name, type: "Concept",
                 description: description, sourceIDs: sources)
    }

    private func relation(_ src: String, _ dst: String, type: String = "likes",
                          description: String = "", weight: Double = 1,
                          sources: [String] = []) -> KGRelationship {
        KGRelationship(id: UUID().uuidString, srcEntity: src, dstEntity: dst,
                       relationType: type, description: description, weight: weight,
                       sourceIDs: sources)
    }

    // MARK: - 向量文本

    func testEntityEmbeddingTextFormat() throws {
        let (store, _) = try makeStore()
        let e = entity("Alpha", description: "第一行\n第二行")
        XCTAssertEqual(store.entityEmbeddingText(e), "Alpha\n第一行\n第二行")
    }

    func testRelationshipEmbeddingTextFormat() throws {
        let (store, _) = try makeStore()
        let r = relation("Alpha", "Beta", type: "likes", description: "关系描述")
        XCTAssertEqual(store.relationshipEmbeddingText(r),
                       "likes\tAlpha\nBeta\n关系描述")
        // 无向关系反序文本：关键字不变，源/目标互换。
        XCTAssertEqual(store.reverseRelationshipEmbeddingText(r),
                       "likes\tBeta\nAlpha\n关系描述")
    }

    func testEmbeddingTextTruncatesToThousandTokens() async throws {
        let (store, _) = try makeStore()
        await store.loadTokenCounter()
        let longDescription = Array(repeating: "word", count: 1_500).joined(separator: " ")
        let text = store.entityEmbeddingText(entity("Alpha", description: longDescription))
        let tokens = text.split(whereSeparator: \.isWhitespace).count
        XCTAssertLessThanOrEqual(tokens, 1_000, "实体向量文本应截断到 1000 token")
        XCTAssertTrue(text.hasPrefix("Alpha\n"), "截断必须保留实体名")
    }

    // MARK: - 关系双向 id

    func testRelationshipVectorTargetsUseForwardAndReverseIDs() throws {
        let (store, _) = try makeStore()
        let e = entity("Alpha", description: "描述")
        let r = relation(e.name, "Beta", description: "描述")
        let targets = store.vectorTargets(entities: [e], relationships: [r])
        XCTAssertEqual(targets.count, 3, "实体 1 条 + 关系正序/反序各 1 条")
        XCTAssertEqual(targets[1].id, r.id)
        XCTAssertEqual(targets[1].text, "likes\tAlpha\nBeta\n描述")
        XCTAssertEqual(targets[2].id, r.reverseVectorID)
        XCTAssertEqual(targets[2].text, "likes\tBeta\nAlpha\n描述")
    }

    func testRelationshipSearchDedupesForwardAndReverseVectors() async throws {
        let (store, path) = try makeStore()
        let r = relation("Alpha", "Beta", description: "描述")
        try store.upsertRelationship(r)
        try await store.storeVectors(targets: store.vectorTargets(entities: [], relationships: [r]))

        let db = try SQLiteDatabase(path: path, readOnly: true)
        defer { db.close() }
        let rows = try db.scalar("SELECT COUNT(*) FROM kg_vectors WHERE target_type = 'relationship'")
        XCTAssertEqual(rows.intValue, 2, "关系应落正序 + 反序两条向量")

        let hits = try store.searchRelationships(vector: [1, 0, 0, 0], topK: 10,
                                                 similarityFloor: 0.2)
        XCTAssertEqual(hits.count, 1, "正反两条向量应合并为同一条关系")
        XCTAssertEqual(hits.first?.relationship.id, r.id)
        XCTAssertGreaterThanOrEqual(hits.first?.score ?? 0, 0.99)
    }

    // MARK: - 构建状态

    func testBuildStateRoundTrip() throws {
        let (store, _) = try makeStore()
        XCTAssertNil(store.buildState())
        try store.markBuildState(.building)
        XCTAssertEqual(store.buildState(), .building)
        try store.markBuildState(.complete)
        XCTAssertEqual(store.buildState(), .complete)
        try store.markBuildState(.stale)
        XCTAssertEqual(store.buildState(), .stale)
    }

    // MARK: - 排序

    func testEntityRankingCombinesCosineAndDegree() {
        let lowCosineHighDegree = KGPassage(text: "a", kind: .entity, label: "热节点",
                                            score: 0.50, degree: 10)
        let highCosineLeaf = KGPassage(text: "b", kind: .entity, label: "叶子",
                                       score: 0.55, degree: 0)
        let sorted = KGCandidateRanking.entities([highCosineLeaf, lowCosineHighDegree])
        XCTAssertEqual(sorted.first?.label, "热节点",
                       "度数（中心性）应参与最终排序，而非只做同分兜底")
    }

    func testRelationRankingCombinesCosineRankAndWeight() {
        let strongEvidence = KGPassage(text: "a", kind: .relation, label: "甲 --x--> 乙",
                                       score: 0.50, degree: 10, weight: 5)
        let weakEvidence = KGPassage(text: "b", kind: .relation, label: "丙 --y--> 丁",
                                     score: 0.55, degree: 0, weight: 1)
        let sorted = KGCandidateRanking.relations([weakEvidence, strongEvidence])
        XCTAssertEqual(sorted.first?.label, "甲 --x--> 乙",
                       "rank + weight 应参与最终排序")
    }
}
