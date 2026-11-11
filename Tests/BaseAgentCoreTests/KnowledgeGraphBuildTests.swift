import Foundation
import XCTest
@testable import BaseAgentCore

/// 图谱构建相关：关系身份归一、原子替换、快照版本校验、断点缓存。
final class KnowledgeGraphBuildTests: XCTestCase {

    private var files: [String] = []

    override func tearDown() {
        for f in files { try? FileManager.default.removeItem(atPath: f) }
        files.removeAll()
        super.tearDown()
    }

    private final class StubEmbedding: EmbeddingProvider, @unchecked Sendable {
        let identifier = "kg-build-test"
        let dimension = 4
        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map { _ in [1, 0, 0, 0] }
        }
        func embedQuery(_ text: String) async throws -> [Float] { [1, 0, 0, 0] }
    }

    private func makeStore() throws -> (KnowledgeGraphStore, String) {
        let path = NSTemporaryDirectory()
            .appending("kg-build-\(UUID().uuidString).sqlite")
        files.append(path)
        return (try KnowledgeGraphStore(dbPath: path, embedding: StubEmbedding()), path)
    }

    private func tempSnapshot() -> String {
        let path = NSTemporaryDirectory()
            .appending("kg-snapshot-\(UUID().uuidString).json")
        files.append(path)
        return path
    }

    private func entity(_ name: String, type: String = "Person") -> KGEntity {
        KGEntity(id: UUID().uuidString, name: name, type: type,
                 description: "描述 \(name)", sourceIDs: ["c1"])
    }

    private func relation(_ src: String, _ dst: String, type: String = "likes",
                          weight: Double = 1, sources: [String] = ["c1"]) -> KGRelationship {
        KGRelationship(id: UUID().uuidString, srcEntity: src, dstEntity: dst,
                       relationType: type, weight: weight, sourceIDs: sources)
    }

    // MARK: - 关系身份

    func testPairKeyIsDirectionInsensitive() {
        XCTAssertEqual(KGRelationship.pairKey("Alpha", "Beta"),
                       KGRelationship.pairKey(" Beta ", "alpha"))
    }

    func testRelationshipUpsertMergesSameEntityPair() {
        let graph = KnowledgeGraph(snapshotPath: tempSnapshot())
        graph.upsertEntity(KGEntity(id: "a", name: "Alpha", type: "Person"))
        graph.upsertEntity(KGEntity(id: "b", name: "Beta", type: "Org"))
        _ = graph.upsertRelationship(relation("Alpha", "Beta", type: "likes",
                                              weight: 1, sources: ["c1"]))
        let merged = graph.upsertRelationship(relation("Beta", "Alpha", type: "reports_to",
                                                       weight: 3, sources: ["c2"]))
        XCTAssertEqual(graph.relationCount, 1, "同一对实体只保留一条边")
        XCTAssertEqual(merged.sourceIDs.sorted(), ["c1", "c2"])
        XCTAssertEqual(merged.weight, 3, "weight 取较大值")
        XCTAssertTrue(merged.relationType.contains("likes"))
        XCTAssertTrue(merged.relationType.contains("reports_to"))
    }

    func testEntityUpsertMergesTypes() {
        let graph = KnowledgeGraph(snapshotPath: tempSnapshot())
        graph.upsertEntity(KGEntity(id: "a", name: "Alpha", type: "Person"))
        let merged = graph.upsertEntity(KGEntity(id: "b", name: "alpha", type: "Organization"))
        XCTAssertEqual(merged.type, "Organization/Person")
    }

    // MARK: - 原子替换与版本

    func testReplaceAllIsAtomicAndBumpsRevisionOnce() async throws {
        let (store, path) = try makeStore()
        try store.upsertEntity(entity("Old"))
        let before = store.revision()
        try store.upsertRelationship(relation("Old", "Gone"))
        // 先写一条向量，replaceAll 后应被清空。
        try await store.storeVectors(targets: store.vectorTargets(
            entities: [entity("Old")], relationships: []))
        let entities = [entity("Alpha"), entity("Beta")]
        let relationships = [relation("Alpha", "Beta")]
        try store.replaceAll(entities: entities, relationships: relationships)
        XCTAssertEqual(store.counts().entities, 2)
        XCTAssertEqual(store.counts().relationships, 1)
        XCTAssertEqual(store.revision(), before + 2, "一次 replaceAll 只自增一次版本")

        let db = try SQLiteDatabase(path: path, readOnly: true)
        defer { db.close() }
        let vectors = try db.scalar("SELECT COUNT(*) FROM kg_vectors")
        XCTAssertEqual(vectors.intValue, 0, "全量替换应同时清空旧向量")
    }

    func testSnapshotRevisionMismatchFallsBackToDatabase() throws {
        let (store, _) = try makeStore()
        let snapshot = tempSnapshot()
        try store.upsertEntity(entity("Alpha"))
        let graph1 = KnowledgeGraph(snapshotPath: snapshot,
                                    revisionProvider: { store.revision() })
        try graph1.loadOrRestore(from: store)
        XCTAssertEqual(graph1.entityCount, 1)

        // 数据库随后被外部写入，快照落后一个版本。
        try store.upsertEntity(entity("Beta"))
        let graph2 = KnowledgeGraph(snapshotPath: snapshot,
                                    revisionProvider: { store.revision() })
        try graph2.loadOrRestore(from: store)
        XCTAssertEqual(graph2.entityCount, 2, "快照落后时必须从数据库重建")
    }

    func testSnapshotRestoresWhenRevisionMatches() throws {
        let (store, _) = try makeStore()
        let snapshot = tempSnapshot()
        try store.upsertEntity(entity("Alpha"))
        let graph1 = KnowledgeGraph(snapshotPath: snapshot,
                                    revisionProvider: { store.revision() })
        try graph1.loadOrRestore(from: store)

        let graph2 = KnowledgeGraph(snapshotPath: snapshot,
                                    revisionProvider: { store.revision() })
        try graph2.loadOrRestore(from: store)
        XCTAssertEqual(graph2.entityCount, 1)
    }

    // MARK: - 断点缓存

    func testExtractionCacheAndProcessedChunks() throws {
        let (store, _) = try makeStore()
        try store.cacheExtraction(chunkID: "chunk-1", output: "entity<|#|>A<|#|>X<|#|>d")
        XCTAssertEqual(store.cachedExtraction(chunkID: "chunk-1"),
                       "entity<|#|>A<|#|>X<|#|>d")

        let hash = KnowledgeGraphBuilder.contentHash("切片正文")
        try store.markChunksProcessed(hashes: ["chunk-1": hash, "chunk-2": hash])
        XCTAssertEqual(store.processedChunkIDs(), ["chunk-1", "chunk-2"])
        XCTAssertEqual(store.processedChunkHashes()["chunk-1"], hash)

        try store.clearExtractionCache()
        XCTAssertNil(store.cachedExtraction(chunkID: "chunk-1"))
    }

    func testContentHashIsStableAndContentSensitive() {
        XCTAssertEqual(KnowledgeGraphBuilder.contentHash("同一段文本"),
                       KnowledgeGraphBuilder.contentHash("同一段文本"))
        XCTAssertNotEqual(KnowledgeGraphBuilder.contentHash("同一段文本"),
                          KnowledgeGraphBuilder.contentHash("同一段文本（改）"))
    }

    // MARK: - 实体消歧

    func testEntityResolutionMergesSamePairRelations() async throws {
        let (store, _) = try makeStore()
        let a1 = KGEntity(id: "a1", name: "Alpha", type: "Person",
                          description: "A1", sourceIDs: ["c1"])
        let a2 = KGEntity(id: "a2", name: "Alpha Chen", type: "Person",
                          description: "A2", sourceIDs: ["c2"])
        let b = KGEntity(id: "b", name: "Beta", type: "Org",
                         description: "B", sourceIDs: ["c3"])
        for e in [a1, a2, b] { try store.upsertEntity(e) }
        var r1 = relation("Alpha", "Beta", type: "likes", weight: 1, sources: ["c1"])
        r1.description = "d1"
        var r2 = relation("Alpha Chen", "Beta", type: "reports_to", weight: 2, sources: ["c2"])
        r2.description = "d2"
        try store.upsertRelationship(r1)
        try store.upsertRelationship(r2)

        let result = try await store.resolveEntityDuplicates(threshold: 0.9)
        XCTAssertEqual(result.entities.count, 2, "同类型实体应合并")
        XCTAssertEqual(result.relationships.count, 1)
        XCTAssertEqual(result.relationships[0].weight, 3, "同实体对关系 weight 累加")
        XCTAssertTrue(result.relationships[0].relationType.contains("likes"))
        XCTAssertTrue(result.relationships[0].relationType.contains("reports_to"))
    }

    // MARK: - 收流保护

    /// 抽取限额：每切片实体 20、关系 50、描述 100 字（可由配置调整）。
    func testExtractionLimitsCapRowsAndTruncateDescriptions() {
        let longDescription = String(repeating: "描", count: 150)
        var lines: [String] = []
        for i in 0..<25 {
            lines.append("entity<|#|>E\(i)<|#|>Person<|#|>\(longDescription)")
        }
        for _ in 0..<60 {
            lines.append("relation<|#|>E0<|#|>E1<|#|>knows<|#|>\(longDescription)")
        }
        let limits = KGExtractionLimits(maxEntitiesPerChunk: 20,
                                        maxRelationsPerChunk: 50,
                                        maxDescriptionCharacters: 100)
        let parsed = KnowledgeGraphBuilder.parseExtraction(
            lines.joined(separator: "\n"), limits: limits)

        XCTAssertEqual(parsed.entities.count, 20)
        XCTAssertEqual(parsed.relations.count, 50)
        XCTAssertTrue(parsed.entities.allSatisfy { $0.description.count == 100 })
        XCTAssertTrue(parsed.relations.allSatisfy { $0.description.count == 100 })
    }

    func testExtractionPromptUsesConfiguredLimits() {
        let limits = KGExtractionLimits(maxEntitiesPerChunk: 20,
                                        maxRelationsPerChunk: 50,
                                        maxDescriptionCharacters: 100)
        let system = KnowledgeGraphPrompts.entityExtractionSystem(limits: limits)
        let task = KnowledgeGraphPrompts.entityExtractionTask(
            chunk: "Alice 在 Acme 工作。", limits: limits)
        for prompt in [system, task] {
            XCTAssertTrue(prompt.contains("20"))
            XCTAssertTrue(prompt.contains("50"))
            XCTAssertTrue(prompt.contains("100"))
        }
    }

    /// 小模型收尾阶段会无限复读同一实体对；抽取应在重复达阈值后提前取消生成，
    /// 而不是把 asyncMaxTokens 全部耗光（真机 2B 曾复读到 6k+ 字符）。
    func testExtractionStopsAfterRepeatedRelationRows() async throws {
        let (store, _) = try makeStore()
        let graph = KnowledgeGraph(snapshotPath: tempSnapshot())
        let llm = RepeatingExtractionLLM()
        let builder = KnowledgeGraphBuilder(store: store, graph: graph, llm: llm)

        let raw = try await builder.extractRaw(chunk: "Alice 在 Acme 工作。")

        // 流终止回调是异步触发的，等一小段时间再断言。
        for _ in 0..<50 where !llm.wasCancelled {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertTrue(llm.wasCancelled, "重复行触发后应取消底层生成流")
        let parsed = KnowledgeGraphBuilder.parseExtraction(raw)
        XCTAssertEqual(parsed.entities.count, 2)
        XCTAssertGreaterThanOrEqual(parsed.relations.count, 1)
        XCTAssertLessThan(parsed.relations.count, 60, "重复关系不应无限增长")
    }
}

/// 持续输出重复关系行的假模型，直到流被调用方取消。
private final class RepeatingExtractionLLM: LLMService, @unchecked Sendable {
    let identifier = "kg-repeat-llm"
    private let lock = NSLock()
    private var cancelled = false

    var wasCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return cancelled
    }

    func chatStream(messages: [ChatMessage], tools: [JSONValue],
                    temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                continuation.yield(LLMStreamEvent(
                    kind: .content,
                    content: "entity<|#|>Alice<|#|>Person<|#|>工程师。\n"
                        + "entity<|#|>Acme<|#|>Organization<|#|>公司。\n"))
                var produced = 0
                while !Task.isCancelled && produced < 500 {
                    produced += 1
                    continuation.yield(LLMStreamEvent(
                        kind: .content,
                        content: "relation<|#|>Alice<|#|>Acme<|#|>employ, work at"
                            + "<|#|>Alice 在 Acme 工作。\n"))
                    try? await Task.sleep(nanoseconds: 1_000_000)
                }
                continuation.yield(LLMStreamEvent(kind: .done))
                continuation.finish()
            }
            continuation.onTermination = { _ in
                self.lock.lock()
                self.cancelled = true
                self.lock.unlock()
                task.cancel()
            }
        }
    }
}
