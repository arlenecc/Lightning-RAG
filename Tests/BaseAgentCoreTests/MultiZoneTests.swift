import XCTest
@testable import BaseAgentCore

final class MultiZoneTests: XCTestCase {
    func testLegacyConfigMigratesToSingleZone() throws {
        var config = AgentConfig()
        config.knowledgeBase = "/tmp/kb-legal"
        config.ragMode = "vector"
        config.migrate()
        XCTAssertEqual(config.knowledgeZones.count, 1)
        XCTAssertEqual(config.knowledgeZones.first?.source, "/tmp/kb-legal")
        XCTAssertTrue(config.knowledgeZones.first?.isDefault == true)
        XCTAssertFalse(config.knowledgeZones.first?.id.isEmpty ?? true)
    }

    func testMultipleZonesDefaultUniqueAndMounted() {
        var config = AgentConfig()
        config.knowledgeZones = [
            KnowledgeZone(id: "a", name: "法律", source: "/kb/a", enabled: true, isDefault: true),
            KnowledgeZone(id: "b", name: "医学", source: "/kb/b", enabled: true, isDefault: true),
            KnowledgeZone(id: "c", name: "关闭", source: "/kb/c", enabled: false),
        ]
        config.migrate()
        XCTAssertEqual(config.knowledgeZones.filter { $0.isDefault }.count, 1, "默认分区唯一")
        XCTAssertEqual(config.mountedZones.count, 2, "仅启用的分区挂载")
        XCTAssertEqual(config.defaultZone?.name, "法律")
    }

    func testRoundtripZonesViaExport() throws {
        var config = AgentConfig()
        config.knowledgeZones = [
            KnowledgeZone(id: "z1", name: "法律", source: "/kb/law", enabled: true, isDefault: true),
            KnowledgeZone(id: "z2", name: "医学", source: "/kb/med", enabled: false),
        ]
        let data = try JSONSerialization.data(withJSONObject: config.export())
        let path = NSTemporaryDirectory() + "mz-\(UUID().uuidString).json"
        try data.write(to: URL(fileURLWithPath: path))
        defer { try? FileManager.default.removeItem(atPath: path) }
        let restored = AgentConfig.load(path)
        XCTAssertEqual(restored.knowledgeZones.count, 2)
        XCTAssertEqual(restored.knowledgeZones.first?.name, "法律")
        XCTAssertEqual(restored.knowledgeZones.last?.enabled, false)
    }

    /// 不同分区 rowID 从 1 开始会碰撞，融合必须用 分区:id 复合键避免错误合并。
    func testFuseSeparatesSameRowIDAcrossZones() {
        let c1 = RAGCitation(id: "1", rank: 1, source: "a.md", text: "法律切片", score: 0.9)
        let c2 = RAGCitation(id: "1", rank: 1, source: "b.md", text: "医学切片", score: 0.8)
        let resultA = MultiZoneRetrieval.ZoneResult(zoneName: "法律", citations: [c1])
        let resultB = MultiZoneRetrieval.ZoneResult(zoneName: "医学", citations: [c2])
        let fused = MultiZoneRetrieval.fuse([resultA, resultB], topK: 10)
        XCTAssertEqual(fused.count, 2, "同 rowID 不同分区必须各自保留")
        XCTAssertTrue(fused.allSatisfy { $0.id.contains("|") }, "id 需含分区前缀")
        XCTAssertEqual(fused.filter { $0.text == "法律切片" }.count, 1)
        XCTAssertEqual(fused.filter { $0.text == "医学切片" }.count, 1)
    }

    /// 领域名相同但 id 不同的分区不得被融合误合并。
    func testFuseSeparatesSameNameDifferentID() {
        let c1 = RAGCitation(id: "1", rank: 1, source: "a.md", text: "A库切片")
        let c2 = RAGCitation(id: "1", rank: 1, source: "b.md", text: "B库切片")
        let rA = MultiZoneRetrieval.ZoneResult(zoneID: "id-a", zoneName: "同名", citations: [c1])
        let rB = MultiZoneRetrieval.ZoneResult(zoneID: "id-b", zoneName: "同名", citations: [c2])
        XCTAssertEqual(MultiZoneRetrieval.fuse([rA, rB], topK: 10).count, 2)
    }

    func testEmptySourceZoneSurvivesMigration() {
        var config = AgentConfig()
        config.workspace = "/tmp/ws"
        config.knowledgeZones = [
            KnowledgeZone(id: "z", name: "新库", source: "", enabled: true, isDefault: false),
        ]
        config.migrate()
        XCTAssertEqual(config.knowledgeZones.count, 1, "空 source 分区不得被丢弃")
        XCTAssertEqual(config.knowledgeZones.first?.source,
                       "/tmp/ws/knowledge_base", "空 source 补成默认目录")
    }

    func testDuplicateSourceZonesDeduped() {
        var config = AgentConfig()
        config.knowledgeZones = [
            KnowledgeZone(id: "a", name: "一", source: "/kb/same", isDefault: true),
            KnowledgeZone(id: "b", name: "二", source: "/kb/same"),
            KnowledgeZone(id: "c", name: "三", source: "/kb/other"),
        ]
        config.migrate()
        XCTAssertEqual(config.knowledgeZones.count, 2, "同源分区去重")
    }

    func testFuseAppliesZoneWeight() {
        let c = RAGCitation(id: "1", rank: 1, source: "a.md", text: "x")
        let heavy = MultiZoneRetrieval.ZoneResult(zoneName: "A", citations: [c], weight: 1)
        let light = MultiZoneRetrieval.ZoneResult(zoneName: "B", citations: [c], weight: 0.1)
        let fused = MultiZoneRetrieval.fuse([heavy, light], topK: 10)
        XCTAssertEqual(fused.count, 2)
        let a = fused.first { $0.id.hasPrefix("A|") }
        let b = fused.first { $0.id.hasPrefix("B|") }
        XCTAssertGreaterThan(a?.score ?? 0, (b?.score ?? 0) * 2, "权重高的分区得分显著更高")
    }
}
