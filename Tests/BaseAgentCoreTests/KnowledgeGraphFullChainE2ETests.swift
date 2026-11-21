import Foundation
import XCTest
@testable import BaseAgentCore
@testable import ColBERTReader

/// 知识图谱全流程端到端实测：抽取→对齐消歧→构建→快照→四种策略检索→融合增强，
/// 覆盖向量模式、极速模式（colbert.db）与多分区聚合。
final class KnowledgeGraphFullChainE2ETests: XCTestCase {
    private final class Emb: EmbeddingProvider {
        let identifier = "kg-chain-emb"
        let dimension = 32
        func embedDocuments(_ texts: [String]) async throws -> [[Float]] { texts.map(Self.vec) }
        func embedQuery(_ text: String) async throws -> [Float] { Self.vec(text) }
        static func vec(_ text: String) -> [Float] {
            var v = [Float](repeating: 0, count: 32)
            for token in text.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }) {
                var h: UInt64 = 0xcbf29ce484222325
                for b in token.utf8 { h ^= UInt64(b); h = h &* 0x100000001b3 }
                v[Int(h % 32)] += 1
            }
            return VectorCodec.normalize(v)
        }
    }

    /// 假辅助模型：按 prompt 关键字返回 delimited 抽取/关键词/归纳结果；记录关键词调用次数。
    private final class FakeAux: LLMService, @unchecked Sendable {
        let identifier = "kg-chain-aux"
        private let lock = NSLock()
        private var _keywordCalls = 0
        var keywordCalls: Int { lock.lock(); defer { lock.unlock() }; return _keywordCalls }
        func chatStream(messages: [ChatMessage], tools: [JSONValue],
                        temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
            let prompt = messages.last?.content ?? ""
            let answer: String
            if prompt.contains("expert keyword extractor") || prompt.contains("关键词提取器") {
                lock.lock(); _keywordCalls += 1; lock.unlock()
                answer = "{\"high_level_keywords\":[\"Alice\"],\"low_level_keywords\":[\"Alice\"]}"
            } else if prompt.contains("Extract") || prompt.contains("entity<|#|>") {
                answer = """
                entity<|#|>Alice<|#|>Person<|#|>Alice 是工程师。
                entity<|#|>Acme<|#|>Org<|#|>Acme 是公司。
                relation<|#|>Alice<|#|>Acme<|#|>就职<|#|>Alice 在 Acme 工作。
                <|COMPLETE|>
                """
            } else if prompt.contains("synthesize") || prompt.contains("description") {
                answer = "综合描述：多次提及。"
            } else { answer = "" }
            return AsyncThrowingStream { c in
                c.yield(LLMStreamEvent(kind: .content, content: answer))
                c.yield(LLMStreamEvent(kind: .done))
                c.finish()
            }
        }
    }

    private func config(root: URL, kb: URL, strategy: RAGRetrievalStrategy = .hybrid) -> AgentConfig {
        var c = AgentConfig()
        c.workspace = root.path
        c.knowledgeBase = kb.path
        c.ragRerankEnabled = false
        c.ragRetrievalStrategy = strategy
        c.ragKGExtractionFormat = .delimited
        c.ragChunkSize = 400
        c.ragChunkOverlap = 0
        return c
    }

    /// 场景1：向量模式全链路——ingest→全量构建→四策略检索都拿到图谱 context。
    func testVectorModeFullChainAllStrategies() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("kg-chain-v-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try String(repeating: "Alice is an engineer at Acme. ", count: 30)
            .write(to: kb.appendingPathComponent("a.md"), atomically: true, encoding: .utf8)

        let engine = RAGEngine(config: config(root: root, kb: kb), knowledgeBase: kb.path,
                               embedding: Emb(), reranker: nil)
        let ingest = await engine.ingest()
        XCTAssertGreaterThan(ingest.chunks, 0)
        let stats = await engine.buildKnowledgeGraph(auxLLM: FakeAux())
        XCTAssertGreaterThan(stats.entities, 0, "实体抽取必须有结果")
        XCTAssertGreaterThan(stats.relationships, 0, "关系抽取必须有结果")
        XCTAssertTrue(engine.knowledgeGraphStatus().built)

        // 四种非 naive 策略：graphSearch 直接验证各分支都召回实体/关系
        for mode in [RAGEngine.GraphSearchMode.local, .global, .hybrid, .mix] {
            let result = try await engine.graphSearch(query: "Alice", mode: mode, topK: 10)
            XCTAssertFalse(result.entities.isEmpty,
                           "\(mode) 策略应召回实体")
        }
        // searchWithCitations 融合路径：hybrid 策略下 context 带图谱段
        let outcome = try await engine.searchWithCitations(query: "Alice", topK: 3, source: nil)
        XCTAssertTrue(outcome.context.contains("知识图谱"),
                      "融合结果应含图谱 context: \(outcome.context.prefix(80))")
    }

    /// 场景2：多分区聚合（含极速模式 colbert.db）→ 一张图谱，外部切片反查命中。
    func testMultiZoneAndColbertAggregatedIntoOneGraph() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("kg-chain-mz-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        // 分区A：向量模式 engine，ingest 造 chunks
        let kbA = root.appendingPathComponent("zoneA")
        try FileManager.default.createDirectory(at: kbA, withIntermediateDirectories: true)
        try String(repeating: "Alice is an engineer. ", count: 20)
            .write(to: kbA.appendingPathComponent("a.md"), atomically: true, encoding: .utf8)
        let engineA = RAGEngine(config: config(root: root, kb: kbA), knowledgeBase: kbA.path,
                                embedding: Emb(), reranker: nil)
        _ = await engineA.ingest()

        // 分区B：极速模式 engine，colbert.db document 表写正文（不加载模型）
        let kbB = root.appendingPathComponent("zoneB")
        try FileManager.default.createDirectory(at: kbB, withIntermediateDirectories: true)
        var cfgB = config(root: root, kb: kbB)
        cfgB.colbertReaderEnabled = true
        let engineB = RAGEngine(config: cfgB, knowledgeBase: kbB.path,
                                embedding: Emb(), reranker: nil)
        let colbert = try ColBERTReaderEngine(dbPath: engineB.colbertDBPath,
                                              modelDir: root.path)
        _ = colbert
        let db = try SQLiteDB(path: engineB.colbertDBPath)
        try db.execute("INSERT INTO document(rowid,doc_id,title,source,content) VALUES(?,?,?,?,?)",
                       [.int(1), .text("b#0000"), .text("第一章"), .text("自然/病毒星球.md"),
                        .text("Alice visits the virus planet.")])

        // 聚合两库切片（模拟 Agent.allZonesKnowledgeGraphSourceChunks 的 zone 前缀规则）
        let chunksA = engineA.knowledgeGraphSourceChunks().map {
            KnowledgeGraphStore.ChunkListItem(id: "zoneA|\($0.id)", text: $0.text,
                                              source: "社会/\($0.source)", heading: $0.heading)
        }
        let chunksB = engineB.knowledgeGraphSourceChunks().map {
            KnowledgeGraphStore.ChunkListItem(id: "zoneB|\($0.id)", text: $0.text,
                                              source: "自然/\($0.source)", heading: $0.heading)
        }
        XCTAssertFalse(chunksA.isEmpty)
        XCTAssertEqual(chunksB.count, 1)

        // 统一构建到 engineA 的图谱
        let stats = await engineA.buildKnowledgeGraph(auxLLM: FakeAux(),
                                                      extraChunkRefs: chunksA + chunksB)
        XCTAssertGreaterThan(stats.entities, 0)
        // 外部切片反查：两库原文都能拿回（P0-1 回归）
        let store = try XCTUnwrap(try? engineA.testKnowledgeGraphStore())
        let refs = try store.chunkRefs(ids: chunksA.map(\.id) + chunksB.map(\.id))
        XCTAssertEqual(refs.count, chunksA.count + chunksB.count,
                       "所有跨库切片都应能反查到原文")
        XCTAssertTrue(refs.contains { $0.source.hasPrefix("自然/") })
        // 增量比对：外部切片哈希可见，不被误判删除（P0-2 回归）
        XCTAssertEqual(store.listExternalChunkRefs().count, chunksA.count + chunksB.count)
    }

    /// 场景3：关键词缓存——第二次检索不再调用 LLM 抽取（性能实测）。
    func testKeywordCacheAvoidsRepeatLLM() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("kg-chain-cache-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try String(repeating: "Alice at Acme. ", count: 20)
            .write(to: kb.appendingPathComponent("a.md"), atomically: true, encoding: .utf8)
        let aux = FakeAux()
        let engine = RAGEngine(config: config(root: root, kb: kb), knowledgeBase: kb.path,
                               embedding: Emb(), reranker: nil)
        _ = await engine.ingest()
        _ = await engine.buildKnowledgeGraph(auxLLM: aux)
        let afterBuild = aux.keywordCalls
        _ = try await engine.searchWithCitations(query: "Alice", topK: 3, source: nil)
        let afterFirst = aux.keywordCalls
        _ = try await engine.searchWithCitations(query: "Alice", topK: 3, source: nil)
        let afterSecond = aux.keywordCalls
        XCTAssertEqual(afterFirst - afterBuild, 1, "首次检索抽一次关键词")
        XCTAssertEqual(afterSecond - afterFirst, 0, "同 query 第二次应命中缓存，不再调 LLM")
    }
}
