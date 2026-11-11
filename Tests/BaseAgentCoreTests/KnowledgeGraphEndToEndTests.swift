import Foundation
import XCTest
@testable import BaseAgentCore

// MARK: - 全链路样例：抽取 → 入库 → 消歧 → 建图 → 快照 → 五种策略检索

/// 词袋哈希 embedding：同一个词在不同文本里落到同一维，便于验证实体消歧。
private final class BagOfWordsEmbedding: EmbeddingProvider {
    let identifier = "kg-e2e-bow"
    let dimension = 101

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map(Self.vector)
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        Self.vector(text)
    }

    static func vector(_ text: String) -> [Float] {
        var v = [Float](repeating: 0, count: 101)
        for token in text.lowercased().split(whereSeparator: {
            !$0.isLetter && !$0.isNumber
        }) {
            var h: UInt64 = 0xcbf29ce484222325
            for byte in token.utf8 {
                h ^= UInt64(byte)
                h = h &* 0x100000001b3
            }
            v[Int(h % 101)] += 1
        }
        return VectorCodec.normalize(v)
    }
}

/// 按 prompt 类型返回样例抽取结果，模拟辅助模型。
private final class SampleKGLLM: LLMService {
    let identifier = "kg-e2e-fake"

    func chatStream(messages: [ChatMessage], tools: [JSONValue],
                    temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        let prompt = messages.last?.content ?? ""
        let answer = reply(to: prompt)
        return AsyncThrowingStream { continuation in
            continuation.yield(LLMStreamEvent(kind: .content, content: answer))
            continuation.yield(LLMStreamEvent(kind: .done))
            continuation.finish()
        }
    }

    private func reply(to prompt: String) -> String {
        if prompt.contains("expert keyword extractor") {
            return keywords(for: prompt)
        }
        if prompt.contains("Extract entities and relationships") {
            return extraction(for: prompt)
        }
        if prompt.contains("synthesize a list of descriptions") {
            let name = prompt.components(separatedBy: "Entity Name: ")
                .last?
                .components(separatedBy: "\n")
                .first ?? "实体"
            return "\(name) 在样例数据中被多次提及，信息已综合整理。"
        }
        return ""
    }

    private func keywords(for prompt: String) -> String {
        let query = prompt.components(separatedBy: "User Query: ").last ?? prompt
        let names = ["Alice Chen", "Alice", "Acme Corp", "Project Aurora",
                     "Shanghai", "Swift", "Bob"]
        let found = names.filter { query.localizedCaseInsensitiveContains($0) }
        let high = found.isEmpty ? ["知识图谱"] : found
        func json(_ values: [String]) -> String {
            "[" + values.map { "\"\($0)\"" }.joined(separator: ",") + "]"
        }
        return "{\"high_level_keywords\": \(json(high)), "
            + "\"low_level_keywords\": \(json(found))}"
    }

    private func extraction(for prompt: String) -> String {
        if prompt.contains("Alice Chen") {
            return """
            entity<|#|>Alice Chen<|#|>Person<|#|>Alice Chen 是 Acme Corp 的软件工程师，负责领导 Project Aurora。
            entity<|#|>Alice<|#|>Person<|#|>Alice 是 Bob 在 Project Aurora 上的协作伙伴。
            entity<|#|>Acme Corp<|#|>Organization<|#|>Acme Corp 是一家开发工具公司。
            entity<|#|>Project Aurora<|#|>Project<|#|>Project Aurora 是内部知识图谱项目。
            entity<|#|>Bob<|#|>Person<|#|>Bob 是 Acme Corp 的研究员。
            relation<|#|>Alice Chen<|#|>Acme Corp<|#|>就职<|#|>Alice Chen 在 Acme Corp 担任软件工程师。
            relation<|#|>Alice<|#|>Project Aurora<|#|>领导<|#|>Alice 负责领导 Project Aurora。
            relation<|#|>Bob<|#|>Alice<|#|>协作<|#|>Bob 与 Alice 在 Project Aurora 上协作。
            relation<|#|>Bob<|#|>Project Aurora<|#|>参与<|#|>Bob 参与 Project Aurora 的开发。
            <|COMPLETE|>
            """
        }
        if prompt.contains("Shanghai") {
            return """
            entity<|#|>Acme Corp<|#|>Organization<|#|>Acme Corp 总部位于上海，成立于 2010 年。
            entity<|#|>Shanghai<|#|>Location<|#|>Shanghai 是 Acme Corp 总部所在地。
            relation<|#|>Acme Corp<|#|>Shanghai<|#|>总部位于<|#|>Acme Corp 总部位于 Shanghai。
            <|COMPLETE|>
            """
        }
        if prompt.contains("Dana") {
            return """
            entity<|#|>Dana<|#|>Person<|#|>Dana 是 Acme Corp 的数据科学家。
            entity<|#|>Acme Corp<|#|>Organization<|#|>Acme Corp 是一家开发工具公司。
            relation<|#|>Dana<|#|>Acme Corp<|#|>就职<|#|>Dana 在 Acme Corp 担任数据科学家。
            <|COMPLETE|>
            """
        }
        if prompt.contains("Swift") {
            return """
            entity<|#|>Project Aurora<|#|>Project<|#|>Project Aurora 使用 Swift 实现，并存储实体与关系。
            entity<|#|>Swift<|#|>Technology<|#|>Swift 是 Project Aurora 使用的编程语言。
            relation<|#|>Project Aurora<|#|>Swift<|#|>使用<|#|>Project Aurora 使用 Swift 开发。
            relation<|#|>Project Aurora<|#|>OrbitDB<|#|>依赖<|#|>Project Aurora 依赖 OrbitDB 存储图数据。
            <|COMPLETE|>
            """
        }
        return ""
    }
}

final class KnowledgeGraphEndToEndTests: XCTestCase {

    private var root: URL!
    private var kb: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("kg-e2e-\(UUID().uuidString)")
        kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func config(strategy: RAGRetrievalStrategy) -> AgentConfig {
        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragRetrievalStrategy = strategy
        // 这里的假模型输出固定为分隔符行；应用默认已切到 JSON，
        // 端到端测试显式使用分隔符格式，避免假模型与解析器错配。
        config.ragKGExtractionFormat = .delimited
        config.ragChunkSize = 1200
        config.ragChunkOverlap = 0
        return config
    }

    private func writeSamples() throws {
        try """
        # 团队

        Alice Chen is a software engineer at Acme Corp. Alice Chen leads Project Aurora.
        Bob is a researcher at Acme Corp. Bob collaborates with Alice on Project Aurora.
        Project Aurora is an internal knowledge project and the team meets every Monday.
        """.write(to: kb.appendingPathComponent("people.md"), atomically: true, encoding: .utf8)

        try """
        # 公司

        Acme Corp is headquartered in Shanghai. Acme Corp was founded in 2010.
        Acme Corp builds developer tools and employs more than one hundred engineers.
        """.write(to: kb.appendingPathComponent("company.md"), atomically: true, encoding: .utf8)

        try """
        # 项目

        Project Aurora is a knowledge graph project. Project Aurora uses Swift.
        Project Aurora stores entities and relationships in a local database.
        """.write(to: kb.appendingPathComponent("project.md"), atomically: true, encoding: .utf8)
    }

    /// 新增文档后走增量抽取 + 内存图局部 upsert + 防抖快照。
    func testIncrementalUpdateAfterNewDocument() async throws {
        try writeSamples()
        let strategy = RAGRetrievalStrategy.local
        let engine = RAGEngine(config: config(strategy: strategy), knowledgeBase: kb.path,
                               embedding: BagOfWordsEmbedding(), reranker: nil)
        _ = await engine.ingest()
        _ = await engine.buildKnowledgeGraph(auxLLM: SampleKGLLM())
        XCTAssertEqual(engine.knowledgeGraphStatus().entities, 8)

        try """
        # 新成员

        Dana is a data scientist at Acme Corp. Dana works with Bob on reporting.
        Dana leads the analytics guild and reviews the monthly metrics dashboard.
        Acme Corp hired Dana in 2024 to build the reporting platform, and Dana
        collaborates with the platform team on data quality and metric definitions.
        """.write(to: kb.appendingPathComponent("extra.md"), atomically: true,
                  encoding: .utf8)

        let ingest = await engine.ingest()
        XCTAssertGreaterThan(ingest.processed, 0,
                             "processed=\(ingest.processed) skipped=\(ingest.skipped) "
                             + "discarded=\(ingest.discarded) errors=\(ingest.errors)")
        XCTAssertNotNil(engine.graphEntity(named: "Dana"), "增量更新应把新实体并入内存图")
        XCTAssertGreaterThanOrEqual(engine.knowledgeGraphStatus().entities, 9)
        let danaRelations = engine.graphIncidentRelationships(name: "Dana")
        XCTAssertTrue(danaRelations.contains {
            ($0.srcEntity == "Dana" && $0.dstEntity == "Acme Corp")
                || ($0.srcEntity == "Acme Corp" && $0.dstEntity == "Dana")
        }, "relations=\(danaRelations.map { "\($0.srcEntity)->\($0.dstEntity)" })")

        // 等待防抖窗口后重开引擎，增量结果应仍可读。
        try await Task.sleep(nanoseconds: 1_300_000_000)
        engine.close()
        let reopened = RAGEngine(config: config(strategy: strategy), knowledgeBase: kb.path,
                                 embedding: BagOfWordsEmbedding(), reranker: nil)
        XCTAssertNotNil(reopened.graphEntity(named: "Dana"))
        reopened.close()
    }

    func testFullPipelineWithSampleDocuments() async throws {        try writeSamples()
        let fakeLLM = SampleKGLLM()

        // 1) 入库 + 建图
        let engine = RAGEngine(config: config(strategy: .mix), knowledgeBase: kb.path,
                               embedding: BagOfWordsEmbedding(), reranker: nil)
        let ingest = await engine.ingest()
        XCTAssertEqual(ingest.processed, 3)
        XCTAssertTrue(ingest.errors.isEmpty, ingest.errors.joined(separator: "; "))

        let build = await engine.buildKnowledgeGraph(auxLLM: fakeLLM)
        XCTAssertEqual(build.failedChunks, 0)
        XCTAssertGreaterThanOrEqual(build.chunksProcessed, 3)
        XCTAssertEqual(build.entities, 8, "5 个团队实体 + Shanghai + Swift + UNKNOWN 占位")
        XCTAssertEqual(build.relationships, 7)

        // 2) 入库后的实体/关系与归因切片
        let alice = try XCTUnwrap(engine.graphEntity(named: "Alice Chen"))
        XCTAssertEqual(alice.type, "Person")
        XCTAssertFalse(alice.sourceIDs.isEmpty, "实体必须带来源切片")
        let aurora = try XCTUnwrap(engine.graphEntity(named: "Project Aurora"))
        XCTAssertTrue(aurora.description.contains("综合"), "多段描述应被归纳")
        let auroraRelations = engine.graphIncidentRelationships(name: "Project Aurora")
        XCTAssertTrue(auroraRelations.contains { $0.relationType.contains("使用") })
        XCTAssertTrue(auroraRelations.contains { $0.relationType.contains("领导") })
        XCTAssertTrue(auroraRelations.contains { $0.relationType.contains("依赖") })
        // 关系引用了未抽取的实体：应落库 UNKNOWN 占位节点。
        let placeholder = try XCTUnwrap(engine.graphEntity(named: "OrbitDB"))
        XCTAssertEqual(placeholder.type, "UNKNOWN")

        let status = engine.knowledgeGraphStatus()
        XCTAssertTrue(status.built)
        XCTAssertEqual(status.entities, 8)
        XCTAssertEqual(status.relationships, 7)
        XCTAssertTrue(FileManager.default.fileExists(atPath: engine.knowledgeGraphSnapshotPath))

        // 3) 实体消歧：Alice 与 Alice Chen 合并，关系端点同步改写
        let resolution = try await engine.resolveKnowledgeGraphDuplicates(threshold: 0.6)
        XCTAssertEqual(resolution.mergedGroups, 1)
        XCTAssertEqual(resolution.removedEntities, 1)
        XCTAssertEqual(resolution.entities, 7)
        XCTAssertNil(engine.graphEntity(named: "Alice"))
        let merged = try XCTUnwrap(engine.graphEntity(named: "Alice Chen"))
        XCTAssertTrue(merged.description.contains("协作伙伴"))
        let aliceRelations = engine.graphIncidentRelationships(name: "Alice Chen")
        XCTAssertEqual(aliceRelations.count, 3, "就职、领导、协作")
        engine.close()

        // 4) 五种策略：local / global / hybrid / mix / naive
        let local = try await search(strategy: .local, query: "Alice Chen 负责哪个项目？")
        XCTAssertTrue(local.contains("【实体】"), local)
        XCTAssertTrue(local.contains("Alice Chen"), local)
        XCTAssertTrue(local.contains("Project Aurora"), local)

        let global = try await search(strategy: .global, query: "Acme Corp 总部位于哪里？")
        XCTAssertTrue(global.contains("【关系】"), global)
        XCTAssertTrue(global.contains("Shanghai"), global)

        let hybrid = try await search(strategy: .hybrid, query: "Alice Chen 和 Acme Corp 的关系")
        XCTAssertTrue(hybrid.contains("【实体】"), hybrid)
        XCTAssertTrue(hybrid.contains("【关系】"), hybrid)
        XCTAssertTrue(hybrid.contains("Acme Corp"), hybrid)

        let mix = try await search(strategy: .mix, query: "Project Aurora 使用什么技术？")
        XCTAssertTrue(mix.contains("【原文片段】"), mix)
        XCTAssertTrue(mix.contains("Swift"), mix)

        let naive = try await search(strategy: .naive, query: "Project Aurora 使用什么技术？")
        XCTAssertFalse(naive.contains("知识图谱检索到"), naive)
        XCTAssertTrue(naive.contains("Project Aurora") || naive.contains("Swift"), naive)
    }

    private func search(strategy: RAGRetrievalStrategy, query: String) async throws -> String {
        let engine = RAGEngine(config: config(strategy: strategy), knowledgeBase: kb.path,
                               embedding: BagOfWordsEmbedding(), reranker: nil)
        engine.auxLLM = SampleKGLLM()
        defer { engine.close() }
        XCTAssertTrue(engine.knowledgeGraphStatus().built, "\(strategy) 图谱状态应为 complete")
        XCTAssertNotNil(engine.graphEntity(named: "Alice Chen"), "\(strategy) 图谱应可加载")
        return try await engine.searchFormatted(query: query)
    }
}
