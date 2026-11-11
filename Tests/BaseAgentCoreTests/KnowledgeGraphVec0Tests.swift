import Foundation
import XCTest
@testable import BaseAgentCore

/// 固定向量 embedding，用于验证 vec0 与精确扫描结果一致。
private final class FixedEmbedding: EmbeddingProvider {
    let identifier = "kg-vec0-test"
    let dimension = 8
    private let vectors: [String: [Float]]

    init(_ vectors: [String: [Float]]) { self.vectors = vectors }

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map { vectors[$0] ?? [Float](repeating: 0, count: dimension) }
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        vectors[text] ?? [Float](repeating: 0, count: dimension)
    }
}

final class KnowledgeGraphVec0Tests: XCTestCase {

    private var path: String!
    private let embedding = FixedEmbedding([
        "A": [1, 0, 0, 0, 0, 0, 0, 0],
        "B": [0.6, 0.8, 0, 0, 0, 0, 0, 0],
        "C": [0, 1, 0, 0, 0, 0, 0, 0],
        "R": [1, 0, 0, 0, 0, 0, 0, 0],
    ])

    override func setUpWithError() throws {
        path = NSTemporaryDirectory().appending("kg-vec0-\(UUID().uuidString).sqlite")
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(atPath: path)
    }

    private func entity(_ id: String, _ name: String) -> KGEntity {
        KGEntity(id: id, name: name, type: "Concept", description: name, sourceIDs: ["c1"])
    }

    private func seed(_ store: KnowledgeGraphStore) async throws {
        for item in [entity("e1", "A"), entity("e2", "B"), entity("e3", "C")] {
            try store.upsertEntity(item)
            try await store.storeEntityVector(id: item.id, text: item.name)
        }
    }

    func testVec0MatchesExactScanAndDeletion() async throws {
        let query: [Float] = [1, 0, 0, 0, 0, 0, 0, 0]
        var vec0: [(String, Double)] = []

        do {
            let store = try KnowledgeGraphStore(dbPath: path, embedding: embedding,
                                                enableSQLiteVec: true)
            XCTAssertTrue(store.usesVec0Index, "sqlite-vec 应可用并建立 vec0 索引")
            try await seed(store)
            vec0 = try store.searchEntities(vector: query, topK: 2, similarityFloor: 0)
                .map { ($0.entity.id, $0.score) }
            XCTAssertEqual(vec0.map(\.0), ["e1", "e2"])
            XCTAssertGreaterThan(vec0[0].1, vec0[1].1)

            // 删除实体必须同步清理 vec0 行与映射。
            try store.deleteEntity(name: "C")
            let afterDelete = try store.searchEntities(vector: query, topK: 5,
                                                       similarityFloor: 0)
            XCTAssertFalse(afterDelete.contains { $0.entity.id == "e3" })
        }

        // 同一数据库，关闭 vec0 后走精确扫描，结果应与 vec0 一致。
        do {
            let store = try KnowledgeGraphStore(dbPath: path, embedding: embedding,
                                                enableSQLiteVec: false)
            XCTAssertFalse(store.usesVec0Index)
            let scan = try store.searchEntities(vector: query, topK: 2, similarityFloor: 0)
                .map { ($0.entity.id, $0.score) }
            XCTAssertEqual(scan.map(\.0), vec0.map(\.0))
            for (lhs, rhs) in zip(scan, vec0) {
                XCTAssertEqual(lhs.1, rhs.1, accuracy: 1e-6)
            }
        }
    }

    func testSingleVecTablePartitionsByTargetType() async throws {
        let store = try KnowledgeGraphStore(dbPath: path, embedding: embedding,
                                            enableSQLiteVec: true)
        XCTAssertTrue(store.usesVec0Index)
        try store.upsertEntity(entity("e1", "A"))
        try await store.storeEntityVector(id: "e1", text: "A")
        let rel = KGRelationship(id: "r1", srcEntity: "A", dstEntity: "B",
                                 relationType: "likes", description: "d",
                                 weight: 1, sourceIDs: ["c1"])
        try store.upsertRelationship(rel)
        try await store.storeRelationshipVector(id: "r1", text: "R")

        let query: [Float] = [1, 0, 0, 0, 0, 0, 0, 0]
        let entities = try store.searchEntities(vector: query, topK: 10, similarityFloor: 0)
        let relations = try store.searchRelationships(vector: query, topK: 10,
                                                      similarityFloor: 0)
        XCTAssertEqual(entities.map(\.entity.id), ["e1"])
        XCTAssertEqual(relations.map(\.relationship.id), ["r1"])

        // 实体与关系向量确实落在同一张 vec0 表，通过 target_type 分区。
        let db = try SQLiteDatabase(path: path)
        XCTAssertEqual(try db.scalar("SELECT COUNT(*) FROM kg_vectors_vec").intValue, 2)
        XCTAssertEqual(try db.scalar("""
        SELECT COUNT(*) FROM kg_vectors_vec WHERE target_type = 'entity'
        """).intValue, 1)
        XCTAssertEqual(try db.scalar("""
        SELECT COUNT(*) FROM kg_vectors_vec WHERE target_type = 'relationship'
        """).intValue, 1)
    }

    func testMigratesLegacyVectorsToVec0() async throws {        do {
            // 先在“无 vec0”模式下写入，模拟旧版只写 kg_vectors 的数据。
            let store = try KnowledgeGraphStore(dbPath: path, embedding: embedding,
                                                enableSQLiteVec: false)
            XCTAssertFalse(store.usesVec0Index)
            try store.upsertEntity(entity("e1", "A"))
            try await store.storeEntityVector(id: "e1", text: "A")
        }
        do {
            let store = try KnowledgeGraphStore(dbPath: path, embedding: embedding,
                                                enableSQLiteVec: true)
            XCTAssertTrue(store.usesVec0Index)
            let hits = try store.searchEntities(vector: [1, 0, 0, 0, 0, 0, 0, 0],
                                                topK: 1, similarityFloor: 0.5)
            XCTAssertEqual(hits.first?.entity.id, "e1")
            XCTAssertGreaterThan(hits.first?.score ?? 0, 0.9)
        }
    }
}
