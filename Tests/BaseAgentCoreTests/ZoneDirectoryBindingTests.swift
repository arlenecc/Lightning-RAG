import XCTest
@testable import BaseAgentCore

/// 回归：配置的分区源目录必须正确绑定到各 zoneEngine，不能都落到默认库。
final class ZoneDirectoryBindingTests: XCTestCase {
    private final class NoopCallbacks: AgentCallbacks {}

    func testEachZoneEngineBindsItsOwnSource() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("zone-bind-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let dirs = ["law", "med", "history"].map { root.appendingPathComponent($0) }
        for d in dirs { try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true) }

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeZones = [
            KnowledgeZone(id: "law", name: "社会", source: dirs[0].path, enabled: true, isDefault: true),
            KnowledgeZone(id: "med", name: "自然", source: dirs[1].path, enabled: true),
            KnowledgeZone(id: "his", name: "历史", source: dirs[2].path, enabled: true),
        ]
        config.migrate()

        let agent = Agent(llm: UnloadedLLMService(), config: config, callbacks: NoopCallbacks())
        XCTAssertEqual(agent.zoneEngines.count, 3)
        XCTAssertTrue(agent.zoneEngines["law"] === agent.ragEngine)
        XCTAssertTrue(agent.zoneEngines["law"]?.knowledgeBase.hasSuffix("/law") ?? false)
        XCTAssertTrue(agent.zoneEngines["med"]?.knowledgeBase.hasSuffix("/med") ?? false)
        XCTAssertTrue(agent.zoneEngines["his"]?.knowledgeBase.hasSuffix("/history") ?? false)
        XCTAssertEqual(Set(agent.zoneEngines.values.map(\.knowledgeBase)).count, 3,
                       "三个分区的源目录必须互不相同")
        XCTAssertEqual(Set(agent.zoneEngines.values.map(\.ragDir)).count, 3,
                       "三个分区的 ragDir 必须互不相同")
        agent.close()
    }
}
