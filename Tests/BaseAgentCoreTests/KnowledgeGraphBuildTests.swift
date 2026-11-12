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
            chunk: "Alice 在 Acme 工作。", sectionContext: "H1 → 团队", limits: limits)
        for prompt in [system, task] {
            XCTAssertTrue(prompt.contains("20"))
            XCTAssertTrue(prompt.contains("50"))
            XCTAssertTrue(prompt.contains("100"))
        }
    }

    /// 输入围栏、不可信元数据标注与格式反例：防止文档标题/正文伪造 prompt 段落，
    /// 并降低小模型把 relation 行写成 entity 的概率。
    func testExtractionPromptWrapsInputAndLabelsContextUntrusted() {
        let limits = KGExtractionLimits()
        let task = KnowledgeGraphPrompts.entityExtractionTask(
            chunk: "Alice 在 Acme 工作。", sectionContext: "H1 → 团队", limits: limits)
        let system = KnowledgeGraphPrompts.entityExtractionSystem(limits: limits)

        XCTAssertTrue(task.contains("---Input Text---\n```"))
        XCTAssertTrue(task.hasSuffix("---Output---"))
        XCTAssertTrue(task.contains("untrusted metadata"))
        XCTAssertTrue(task.contains("H1 → 团队"))
        XCTAssertTrue(system.contains("Wrong:"))
        XCTAssertTrue(system.contains("Correct:"))
        XCTAssertTrue(system.contains("Artifact"))
        XCTAssertFalse(system.contains("Steve Jobs"), "模板示例不应残留真实实体")
    }

    func testSummarizePromptUsesDescriptionLimit() {
        let prompt = KnowledgeGraphPrompts.summarizeEntityDescriptionsPrompt(
            name: "Alice-Acme", descriptionType: "Relation",
            list: "1. Alice 在 Acme 工作。", maxCharacters: 100)
        XCTAssertTrue(prompt.contains("100 characters"))
        XCTAssertTrue(prompt.contains("Relation Name: Alice-Acme"))
        XCTAssertTrue(prompt.contains("one single paragraph"))
        XCTAssertFalse(prompt.contains("600 tokens"))
    }

    /// 小模型偶尔只输出到 keywords（4 段），缺 description；应保留该关系并用
    /// keywords 兜底，而不是整条丢弃。
    func testParseRelationWithoutDescriptionFallsBackToKeywords() {
        let raw = "relation<|#|>Alice<|#|>Acme<|#|>employed_by, works_at\n<|COMPLETE|>"
        let parsed = KnowledgeGraphBuilder.parseExtraction(raw)
        XCTAssertEqual(parsed.relations.count, 1)
        XCTAssertEqual(parsed.relations[0].keywords, "employed_by, works_at")
        XCTAssertEqual(parsed.relations[0].description, "employed_by, works_at")
    }

    func testParseSkipsNumericOnlyEntities() {
        let raw = "entity<|#|>2010<|#|>Other<|#|>年份\n"
            + "entity<|#|>Acme Corp<|#|>Organization<|#|>公司\n"
            + "relation<|#|>Acme Corp<|#|>2010<|#|>founded_in<|#|>成立于 2010 年"
        let parsed = KnowledgeGraphBuilder.parseExtraction(raw)
        XCTAssertEqual(parsed.entities.map { $0.name }, ["Acme Corp"])
        XCTAssertTrue(parsed.relations.isEmpty)
    }

    /// 重复行保护触发的收流通常是模型退化复读，末行仍完整时不再补抽，
    /// 避免把同样的内容再生成一遍（2B 实测两轮实体对完全一致且耗时翻倍）。
    func testNoGleaningAfterDuplicateStopWithCompleteLastRow() async throws {
        var lines = [
            "entity<|#|>Alice<|#|>Person<|#|>工程师",
            "entity<|#|>Acme<|#|>Organization<|#|>公司",
        ]
        for _ in 0..<13 {
            lines.append("relation<|#|>Alice<|#|>Acme<|#|>employed_by")
        }
        let llm = ScriptedExtractionLLM(outputs: [lines.joined(separator: "\n")])
        let (store, _) = try makeStore()
        let graph = KnowledgeGraph(snapshotPath: tempSnapshot())
        let builder = KnowledgeGraphBuilder(store: store, graph: graph, llm: llm)

        let raw = try await builder.extractRaw(chunk: "Alice 在 Acme 工作。")

        XCTAssertEqual(llm.callCount, 1, "重复行收流后不应再补抽")
        XCTAssertFalse(raw.contains("<|COMPLETE|>"))
    }

    /// 首轮残缺时追加一次补抽，并带上首轮输出作为「已抽内容」。
    func testGleaningAddsMissedRowsWhenFirstPassIncomplete() async throws {
        let first = "entity<|#|>Alice<|#|>Person<|#|>工程师\n"
            + "relation<|#|>Alice<|#|>Acme"
        let second = "relation<|#|>Alice<|#|>Acme<|#|>employed_by"
            + "<|#|>Alice 在 Acme 工作。\n<|COMPLETE|>"
        let llm = ScriptedExtractionLLM(outputs: [first, second])
        let (store, _) = try makeStore()
        let graph = KnowledgeGraph(snapshotPath: tempSnapshot())
        let builder = KnowledgeGraphBuilder(store: store, graph: graph, llm: llm)

        let raw = try await builder.extractRaw(chunk: "Alice 在 Acme 工作。")

        XCTAssertEqual(llm.callCount, 2)
        XCTAssertTrue(llm.lastUserPrompt?.contains("---Previous Extraction---") == true)
        XCTAssertTrue(raw.contains("<|COMPLETE|>"))
        let parsed = KnowledgeGraphBuilder.parseExtraction(raw)
        XCTAssertTrue(parsed.relations.contains {
            $0.source == "Alice" && $0.target == "Acme"
        })
    }

    /// 首轮已正常收尾时不应多跑一轮补抽。
    func testGleaningSkippedWhenFirstPassComplete() async throws {
        let llm = ScriptedExtractionLLM(outputs: [
            "entity<|#|>Alice<|#|>Person<|#|>工程师\n<|COMPLETE|>",
        ])
        let (store, _) = try makeStore()
        let graph = KnowledgeGraph(snapshotPath: tempSnapshot())
        let builder = KnowledgeGraphBuilder(store: store, graph: graph, llm: llm)

        let raw = try await builder.extractRaw(chunk: "Alice 在 Acme 工作。")

        XCTAssertEqual(llm.callCount, 1)
        XCTAssertTrue(raw.contains("<|COMPLETE|>"))
    }

    /// `rag_kg_extraction_format` 可持久化，未知值回落到分隔符模式。
    func testExtractionFormatConfigRoundTrip() throws {
        let path = NSTemporaryDirectory()
            + "kg-format-\(UUID().uuidString).json"
        defer { try? FileManager.default.removeItem(atPath: path) }
        var config = AgentConfig()
        XCTAssertEqual(config.ragKGExtractionFormat, .json)
        config.ragKGExtractionFormat = .json
        config.ragKGMaxConcurrency = 3
        try config.save(path)
        let loaded = AgentConfig.load(path)
        XCTAssertEqual(loaded.ragKGExtractionFormat, .json)
        XCTAssertEqual(loaded.ragKGMaxConcurrency, 3)
        XCTAssertEqual(KGExtractionFormat(stored: "JSON"), .json)
        XCTAssertEqual(KGExtractionFormat(stored: "yaml"), .delimited)
    }

    /// JSON 模式：解析代码围栏、数组/字符串 keywords、数字端点过滤。
    func testJSONExtractionParser() {
        let raw = """
        ```json
        {
          "entities": [
            {"name": "Alice Chen", "type": "Person", "description": "工程师"},
            {"name": "2010", "type": "Other", "description": "年份"}
          ],
          "relationships": [
            {"source": "Alice Chen", "target": "Acme Corp",
             "keywords": ["employed_by", "works_at"], "description": "Alice 在 Acme 工作。"},
            {"source": "Acme Corp", "target": "2010", "keywords": "founded_in"}
          ]
        }
        ```
        """
        let parsed = KnowledgeGraphBuilder.parseExtractionJSON(raw)
        XCTAssertEqual(parsed.entities.map { $0.name }, ["Alice Chen"])
        XCTAssertEqual(parsed.relations.count, 1)
        XCTAssertEqual(parsed.relations[0].keywords, "employed_by, works_at")
        XCTAssertEqual(parsed.relations[0].description, "Alice 在 Acme 工作。")
    }

    func testJSONExtractionParserAppliesLimitsAndDescriptionFallback() {
        let entities = (1...25).map {
            #"{"name":"E\#($0)","type":"Concept","description":"d"}"#
        }.joined(separator: ",")
        let raw = """
        {"entities":[\(entities)],
         "relationships":[{"source":"E1","target":"E2","keywords":"related_to"}]}
        """
        let limits = KGExtractionLimits(maxEntitiesPerChunk: 20,
                                        maxRelationsPerChunk: 50,
                                        maxDescriptionCharacters: 100)
        let parsed = KnowledgeGraphBuilder.parseExtractionJSON(raw, limits: limits)
        XCTAssertEqual(parsed.entities.count, 20)
        XCTAssertEqual(parsed.relations.first?.description, "related_to")
    }

    func testJSONExtractionPrompts() {
        let limits = KGExtractionLimits()
        let system = KnowledgeGraphPrompts.entityExtractionJSONSystem(limits: limits)
        let task = KnowledgeGraphPrompts.entityExtractionJSONTask(
            chunk: "Alice 在 Acme 工作。", sectionContext: "H1 → 团队", limits: limits)
        XCTAssertTrue(system.contains("\"entities\""))
        XCTAssertTrue(system.contains("\"relationships\""))
        XCTAssertTrue(system.contains("Do not use Markdown code fences"))
        XCTAssertTrue(task.contains("Return only the JSON object"))
        XCTAssertTrue(task.contains("---Input Text---\n```"))
        XCTAssertTrue(task.contains("untrusted metadata"))
    }

    // MARK: - JSON 收流保护

    /// 正常 JSON 输出不应触发收流停止。
    func testJSONGuardAllowsNormalOutput() {
        var state = KnowledgeGraphBuilder.JSONExtractionGuard(rescanInterval: 8)
        let entities = (1...10).map {
            #"{"name":"实体\#($0)","type":"Concept","description":"这是第 \#($0) 个实体的描述内容。"}"#
        }.joined(separator: ",")
        let output = #"{"entities":[\#(entities)],"relationships":[]}"#
        XCTAssertFalse(state.shouldStop(output: output),
                       "正常完整输出不应触发收流：\(state.stoppedReason)")
        XCTAssertEqual(state.stoppedReason, "")
    }

    /// 同一实体名重复出现（实体数组复读）→ 快停。
    func testJSONGuardStopsOnRepeatedEntityArray() {
        var state = KnowledgeGraphBuilder.JSONExtractionGuard(rescanInterval: 8)
        let entity = #"{"name":"Alice Chen","type":"Person","description":"工程师，团队负责人。"}"#
        let first = #"{"entities":[\#(entity)],"relationships":[]}"#
        XCTAssertFalse(state.shouldStop(output: first))
        // 复读：完整对象之后又重新输出实体数组。
        let repeated = first + #" {"entities":[\#(entity),\#(entity)]"#
        XCTAssertTrue(state.shouldStop(output: repeated),
                      "实体数组复读应触发收流")
        XCTAssertTrue(state.stoppedReason.contains("Alice Chen"),
                      "停止原因应指出重复的实体名：\(state.stoppedReason)")
    }

    /// 关系端点合法重复（枢纽实体）不快停；流式输出长时间无新增唯一值
    /// 触发停滞停（模拟真实收流：每次增长后调用一次判定）。
    func testJSONGuardStallsWithoutNewContent() {
        var state = KnowledgeGraphBuilder.JSONExtractionGuard(
            rescanInterval: 8, stallChars: 120)
        let relation = #"{"source":"枢纽实体","target":"已知目标","keywords":"related_to","description":"重复引用同一对端点的关系描述，内容较长。"},"#
        var output = #"{"entities":[],"relationships":["#
        var stopped = false
        for _ in 0..<6 {
            output += relation
            if state.shouldStop(output: output) {
                stopped = true
                break
            }
        }
        XCTAssertTrue(stopped, "无新增唯一值的流式输出应触发停滞收流")
        XCTAssertTrue(state.stoppedReason.contains("未产生新的实体或关系"),
                      "应为停滞原因：\(state.stoppedReason)")
        // 对照：每步都有新端点值时不应停滞。
        var healthy = KnowledgeGraphBuilder.JSONExtractionGuard(
            rescanInterval: 8, stallChars: 120)
        var growing = #"{"entities":[],"relationships":["#
        for i in 0..<6 {
            growing += #"{"source":"枢纽实体","target":"目标\#(i)","keywords":"related_to","description":"每条关系都引入新的端点值。"},"#
            XCTAssertFalse(healthy.shouldStop(output: growing),
                           "持续出现新端点的正常输出不应收流：\(healthy.stoppedReason)")
        }
    }

    /// 复读尾巴：第一个配平对象应被完整解析，复读内容被丢弃。
    func testJSONParseSalvagesFirstBalancedObject() {
        let good = """
        {"entities":[{"name":"Alice Chen","type":"Person","description":"工程师"}],
         "relationships":[{"source":"Alice Chen","target":"Acme Corp","keywords":"works_at"}]}
        """
        let repeatTail = #" {"entities":[{"name":"Alice Chen","type":"Person""#
        let parsed = KnowledgeGraphBuilder.parseExtractionJSON(good + repeatTail)
        XCTAssertEqual(parsed.entities.map { $0.name }, ["Alice Chen"])
        XCTAssertEqual(parsed.relations.count, 1)
    }

    /// 收流截断：半截 JSON 应补齐括号后恢复已输出的实体/关系。
    func testJSONParseRecoversTruncatedOutput() {
        // 截断在数组中间（元素完整、数组未闭合）。
        let truncatedArray = """
        {"entities":[{"name":"Alice Chen","type":"Person","description":"工程师"},
                     {"name":"Acme Corp","type":"Organization","description":"科技公司"},
        """
        let parsed = KnowledgeGraphBuilder.parseExtractionJSON(truncatedArray)
        XCTAssertEqual(parsed.entities.map { $0.name }, ["Alice Chen", "Acme Corp"])
        // 截断在字符串中间。
        let truncatedString = #"{"entities":[{"name":"Alice Chen","type":"Pers"#
        let parsed2 = KnowledgeGraphBuilder.parseExtractionJSON(truncatedString)
        XCTAssertEqual(parsed2.entities.map { $0.name }, ["Alice Chen"])
        // 截断在悬空键名处（丢弃不完整项后恢复）。
        let danglingKey = #"{"entities":[{"name":"Alice Chen","type":"Pers"}], "rel"#
        let parsed3 = KnowledgeGraphBuilder.parseExtractionJSON(danglingKey)
        XCTAssertEqual(parsed3.entities.map { $0.name }, ["Alice Chen"])
        // 完全无法配平的内容仍返回空。
        let garbage = "这不是 JSON 输出，没有任何结构化内容。"
        XCTAssertTrue(KnowledgeGraphBuilder.parseExtractionJSON(garbage).isEmpty)
    }

    /// JSON 模式下按配置选择 JSON prompt 与解析器，且不做分隔符格式的补抽。
    func testBuilderUsesJSONFormat() async throws {
        let raw = """
        {"entities":[{"name":"Alice Chen","type":"Person","description":"工程师"}],
         "relationships":[{"source":"Alice Chen","target":"Acme Corp",
                           "keywords":"employed_by","description":"Alice 在 Acme 工作。"}]}
        """
        let llm = ScriptedExtractionLLM(outputs: [raw])
        let (store, _) = try makeStore()
        let graph = KnowledgeGraph(snapshotPath: tempSnapshot())
        let builder = KnowledgeGraphBuilder(store: store, graph: graph, llm: llm,
                                            format: .json)

        let parsed = try await builder.extract(chunk: "Alice 在 Acme 工作。")

        XCTAssertEqual(llm.callCount, 1)
        XCTAssertTrue(llm.lastUserPrompt?.contains("JSON object") == true)
        XCTAssertEqual(parsed.entities.map { $0.name }, ["Alice Chen"])
        XCTAssertEqual(parsed.relations.count, 1)
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
        XCTAssertEqual(Set(parsed.entities.map { $0.name.lowercased() }), ["alice", "acme"])
        XCTAssertGreaterThanOrEqual(parsed.relations.count, 1)
        XCTAssertLessThan(parsed.relations.count, 60, "重复关系不应无限增长")
    }
}

/// 按调用次序返回固定输出的假模型，用于验证补抽逻辑。
private final class ScriptedExtractionLLM: LLMService, @unchecked Sendable {
    let identifier = "kg-scripted-llm"
    private let outputs: [String]
    private let lock = NSLock()
    private var calls = 0
    private var prompts: [String] = []

    init(outputs: [String]) {
        self.outputs = outputs
    }

    var callCount: Int {
        lock.lock(); defer { lock.unlock() }
        return calls
    }

    var lastUserPrompt: String? {
        lock.lock(); defer { lock.unlock() }
        return prompts.last
    }

    func chatStream(messages: [ChatMessage], tools: [JSONValue],
                    temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        let index: Int
        lock.lock()
        index = calls
        calls += 1
        prompts.append(messages.last?.content ?? "")
        lock.unlock()
        let output = index < outputs.count ? outputs[index] : "<|COMPLETE|>"
        return AsyncThrowingStream { continuation in
            continuation.yield(LLMStreamEvent(kind: .content, content: output))
            continuation.yield(LLMStreamEvent(kind: .done))
            continuation.finish()
        }
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
