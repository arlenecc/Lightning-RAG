import Foundation
import XCTest
@testable import BaseAgentCore

/// RAG 知识图谱（KnowledgeGraphStore）的手动实体消歧：
/// 「构建知识图谱」阶段仅做字符全匹配，本测试覆盖「事后」用向量语义相似度
/// 合并同指实体的路径（resolveEntityDuplicates）。
final class KnowledgeGraphResolutionTests: XCTestCase {

    private var files: [String] = []

    override func tearDown() {
        for f in files { try? FileManager.default.removeItem(atPath: f) }
        files.removeAll()
        super.tearDown()
    }

    /// 用名字词元决定向量：川普/特朗普 → 同维（余弦 1.0）；总统 → 另一维。
    private final class KGSimulatedEmbedding: EmbeddingProvider, @unchecked Sendable {
        let identifier = "kg-resolution-embedding"
        let dimension = 3
        func embedQuery(_ text: String) async throws -> [Float] { Self.vector(text) }
        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map(Self.vector)
        }
        private static func vector(_ text: String) -> [Float] {
            let lower = text.lowercased()
            if lower.contains("特朗普") || lower.contains("川普") { return [1, 0, 0] }
            if lower.contains("总统") { return [0, 1, 0] }
            if lower.contains("苹果") { return [0, 0, 1] }
            return [0, 0, 0]
        }
    }

    private func makeStore() throws -> KnowledgeGraphStore {
        let path = NSTemporaryDirectory()
            .appending("kg-resolution-\(UUID().uuidString).sqlite")
        files.append(path)
        return try KnowledgeGraphStore(dbPath: path, embedding: KGSimulatedEmbedding())
    }

    private func entity(_ id: String, _ name: String, _ type: String,
                        _ description: String, sources: [String] = []) -> KGEntity {
        KGEntity(id: id, name: name, type: type, description: description,
                 properties: [:], sourceIDs: sources, createdAt: 0, updatedAt: 0)
    }

    private func relation(_ id: String, _ src: String, _ dst: String,
                          _ type: String, _ description: String = "",
                          weight: Double = 1, sources: [String] = []) -> KGRelationship {
        KGRelationship(id: id, srcEntity: src, dstEntity: dst, relationType: type,
                       description: description, weight: weight, properties: [:],
                       sourceIDs: sources, createdAt: 0, updatedAt: 0)
    }

    // MARK: - 手动语义合并

    /// 川普 / 特朗普 向量相同（≥阈值）→ 合并为同一实体；关系端点随之规范。
    func testResolveMergesAliasesAndRewiresRelations() async throws {
        let store = try makeStore()
        try store.upsertEntity(entity("e1", "特朗普", "person", "美国前总统", sources: ["c1"]))
        try store.upsertEntity(entity("e2", "川普", "person", "第45任总统", sources: ["c2"]))
        try store.upsertEntity(entity("e3", "总统职位", "concept", "最高行政职位"))
        try store.upsertRelationship(relation("r1", "川普", "总统职位", "担任",
                                              "川普担任总统", sources: ["c2"]))

        let result = try await store.resolveEntityDuplicates(threshold: 0.9)

        // 3 个实体 → 合并 1 组 → 2 个实体（川普并入特朗普）。
        XCTAssertEqual(result.entities.count, 2, "同指实体应被合并")
        XCTAssertEqual(result.mergedGroups, 1)
        XCTAssertEqual(result.removedEntities, 1)
        XCTAssertNil(result.entities.first { $0.name == "川普" },
                     "被合并实体不应残留")
        XCTAssertNotNil(result.entities.first { $0.name == "特朗普" })

        // 代表实体的描述应并入被合并实体的描述（来源也合并）。
        let trump = result.entities.first { $0.name == "特朗普" }!
        XCTAssertTrue(trump.description.contains("第45任总统"), "描述应合并")
        XCTAssertTrue(trump.sourceIDs.contains("c1") && trump.sourceIDs.contains("c2"),
                      "来源应合并")

        // 关系端点被规范：川普 → 特朗普，且保留了原关系。
        XCTAssertTrue(result.relationships.contains {
            $0.srcEntity == "特朗普" && $0.dstEntity == "总统职位" && $0.relationType == "担任"
        }, "关系端点必须指向消歧后的规范名")
    }

    /// 语义不同（余弦 < 阈值）的实体不得合并；type 冲突即便向量相近也不合并。
    func testResolveKeepsDistinctEntities() async throws {
        let store = try makeStore()
        try store.upsertEntity(entity("e1", "苹果公司", "organization", "科技公司"))
        try store.upsertEntity(entity("e2", "苹果水果", "other", "一种水果"))
        // 即便阈值调到 0，也需要向量相似；苹果公司 vs 苹果水果 余弦 = 0，不合并。
        let result = try await store.resolveEntityDuplicates(threshold: 0.0)

        XCTAssertEqual(result.entities.count, 2, "语义不同的实体不应被合并")
        XCTAssertEqual(result.mergedGroups, 0)
    }

    /// 两端被消歧到同一实体时，关系退化为自环，必须跳过（不写入）。
    func testResolveSkipsSelfLoopRelations() async throws {
        let store = try makeStore()
        try store.upsertEntity(entity("e1", "特朗普", "person", "美国前总统"))
        try store.upsertEntity(entity("e2", "川普", "person", "第45任总统"))
        try store.upsertRelationship(relation("r1", "特朗普", "川普", "别名"))

        let result = try await store.resolveEntityDuplicates(threshold: 0.9)

        XCTAssertEqual(result.entities.count, 1, "别名实体应合并为一个")
        XCTAssertTrue(result.relationships.isEmpty, "自环关系应被跳过")
    }

    /// 实体数不足 2 时直接返回，不做任何操作（幂等安全）。
    func testResolveNoopWhenTooFew() async throws {
        let store = try makeStore()
        try store.upsertEntity(entity("e1", "特朗普", "person", "美国前总统"))
        let result = try await store.resolveEntityDuplicates(threshold: 0.9)
        XCTAssertEqual(result.entities.count, 1)
        XCTAssertEqual(result.mergedGroups, 0)
    }

    /// 删除实体时，实体本身及其向量、所有端点指向它的关系一并删除。
    func testDeleteEntityRemovesEntityAndRelationships() throws {
        let store = try makeStore()
        try store.upsertEntity(entity("e1", "特朗普", "person", "美国前总统"))
        try store.upsertEntity(entity("e2", "总统职位", "concept", "最高行政职位"))
        try store.upsertRelationship(relation("r1", "特朗普", "总统职位", "担任"))

        try store.deleteEntity(name: "特朗普")

        // 实体被删
        XCTAssertNil(try store.entityByName("特朗普"))
        // 关联边被删
        let rels = try store.allRelationships()
        XCTAssertTrue(rels.isEmpty, "端点指向被删实体的关系应一并删除")
        // 另一端实体保留
        XCTAssertNotNil(try store.entityByName("总统职位"))
    }
}
