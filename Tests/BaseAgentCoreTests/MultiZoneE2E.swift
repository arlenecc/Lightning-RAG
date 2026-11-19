import XCTest
@testable import BaseAgentCore

/// 端到端实测：两个独立知识分区各自同步不同语料，联合检索跨越两库召回。
final class MultiZoneE2E: XCTestCase {
    private final class FakeEmbedding: EmbeddingProvider {
        let identifier = "mz-fake"
        let dimension = 8
        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map { text in
                var v = [Float](repeating: 0.01, count: dimension)
                for scalar in text.unicodeScalars {
                    v[Int(scalar.value) % dimension] += 1
                }
                return VectorCodec.normalize(v)
            }
        }
        func embedQuery(_ text: String) async throws -> [Float] {
            try await embedDocuments([text])[0]
        }
    }

    func testTwoZonesIngestThenFusedSearchRecallsBoth() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mz-e2e-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let kbA = root.appendingPathComponent("law"); try FileManager.default.createDirectory(at: kbA, withIntermediateDirectories: true)
        let kbB = root.appendingPathComponent("med"); try FileManager.default.createDirectory(at: kbB, withIntermediateDirectories: true)
        try String(repeating: "合同法违约责任条款与赔偿责任的详细规定。", count: 10)
            .write(to: kbA.appendingPathComponent("law.md"), atomically: true, encoding: .utf8)
        try String(repeating: "医学临床诊断治疗方案与药物副作用的说明。", count: 10)
            .write(to: kbB.appendingPathComponent("med.md"), atomically: true, encoding: .utf8)

        func makeEngine(kb: URL) -> RAGEngine {
            var config = AgentConfig()
            config.workspace = root.path
            config.knowledgeBase = kb.path
            config.ragRerankEnabled = false
            config.ragSemanticChunking = false
            config.ragChunkSize = 120
            config.ragChunkOverlap = 0
            return RAGEngine(config: config, embedding: FakeEmbedding(), reranker: nil)
        }
        let engA = makeEngine(kb: kbA)
        let engB = makeEngine(kb: kbB)
        let sA = await engA.ingest()
        let sB = await engB.ingest()
        XCTAssertEqual(sA.processed, 1); XCTAssertEqual(sB.processed, 1)
        XCTAssertGreaterThan(sA.chunks, 0); XCTAssertGreaterThan(sB.chunks, 0)

        // 各自检索自己的库
        let outA = try await engA.searchWithCitations(query: "合同违约责任", topK: 3)
        let outB = try await engB.searchWithCitations(query: "合同违约责任", topK: 3)
        // 联合检索：两库结果 RRF 融合
        let fused = MultiZoneRetrieval.fuse([
            .init(zoneID: "law", zoneName: "法律", citations: outA.citations),
            .init(zoneID: "med", zoneName: "医学", citations: outB.citations),
        ], topK: 6)
        XCTAssertGreaterThan(fused.count, 0, "联合检索应有结果")
        XCTAssertTrue(fused.allSatisfy { $0.id.contains("|") }, "每条结果带分区前缀")
        XCTAssertTrue(fused.contains { $0.id.hasPrefix("law|") }, "应含法律分区候选")
        XCTAssertTrue(fused.contains { $0.id.hasPrefix("med|") }, "应含医学分区候选")
        engA.close(); engB.close()
    }

    /// 配置往返 + Agent 重建：zoneEngines 仍与分区一一对应，且各分区存储目录互不重叠。
    func testConfigRoundtripRebuildsZoneEnginesWithIsolatedStores() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mz-rebuild-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let kbA = root.appendingPathComponent("a"); try FileManager.default.createDirectory(at: kbA, withIntermediateDirectories: true)
        let kbB = root.appendingPathComponent("b"); try FileManager.default.createDirectory(at: kbB, withIntermediateDirectories: true)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeZones = [
            KnowledgeZone(id: "za", name: "法律", source: kbA.path, enabled: true, isDefault: true),
            KnowledgeZone(id: "zb", name: "医学", source: kbB.path, enabled: true),
        ]
        config.migrate()
        XCTAssertEqual(config.knowledgeBase, kbA.path, "默认分区同步到 knowledgeBase")

        // 配置写盘 → 重新加载（模拟 App 重启/Agent 重建）
        let path = root.appendingPathComponent("config.json").path
        try JSONSerialization.data(withJSONObject: config.export()).write(to: URL(fileURLWithPath: path))
        let reloaded = AgentConfig.load(path)
        XCTAssertEqual(reloaded.knowledgeZones.count, 2)

        let agent = Agent(llm: UnloadedLLMService(), config: reloaded,
                          callbacks: NoopCallbacks())
        XCTAssertEqual(agent.zoneEngines.count, 2)
        XCTAssertTrue(agent.zoneEngines["za"] === agent.ragEngine)
        XCTAssertNotNil(agent.zoneEngines["zb"])
        XCTAssertNotEqual(agent.zoneEngines["za"]!.ragDir,
                          agent.zoneEngines["zb"]!.ragDir, "两分区存储目录必须隔离")
        agent.close()
    }

    /// 验证 Agent 层联合检索确实跨多库召回（而非只查默认库）。
    func testAgentMultiZoneSearchRecallsBothZones() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mz-agent-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let lawDir = root.appendingPathComponent("law")
        let medDir = root.appendingPathComponent("med")
        try FileManager.default.createDirectory(at: lawDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: medDir, withIntermediateDirectories: true)
        try String(repeating: "合同法违约责任条款与赔偿损失的详细规定，当事人应当履行义务。", count: 5)
            .write(to: lawDir.appendingPathComponent("law.md"), atomically: true, encoding: .utf8)
        try String(repeating: "医学临床诊断治疗方案与药物副作用的详细说明，患者需要遵医嘱。", count: 5)
            .write(to: medDir.appendingPathComponent("med.md"), atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = lawDir.path
        config.ragRerankEnabled = false
        config.ragSemanticChunking = false
        config.ragChunkSize = 200
        config.knowledgeZones = [
            KnowledgeZone(id: "law", name: "法律", source: lawDir.path, enabled: true, isDefault: true),
            KnowledgeZone(id: "med", name: "医学", source: medDir.path, enabled: true),
        ]
        let agent = Agent(llm: UnloadedLLMService(), config: config, callbacks: NoopCallbacks())
        XCTAssertEqual(agent.zoneEngines.count, 2)
        for zone in config.knowledgeZones {
            let stats = await agent.zoneEngines[zone.id]!.ingest()
            XCTAssertEqual(stats.processed, 1, "\(zone.name) 应同步 1 文档")
        }
        // 联合检索（无 source）：应从两个分区都召回
        let fused = try await agent.multiZoneSearch(query: "违约责任与临床治疗", topK: 6)
        XCTAssertTrue(fused.citations.contains { $0.id.hasPrefix("law|") }, "应含法律分区")
        XCTAssertTrue(fused.citations.contains { $0.id.hasPrefix("med|") }, "应含医学分区")
        // 定向到某分区：只召回该分区
        let scoped = try await agent.multiZoneSearch(query: "违约责任与临床治疗", topK: 6,
                                                     source: "医学")
        XCTAssertTrue(scoped.citations.allSatisfy { $0.id.hasPrefix("med|") },
                      "source=分区名时应只查该分区")
        agent.close()
    }

    private final class NoopCallbacks: AgentCallbacks {}
}
