import XCTest
@testable import BaseAgentCore

/// 验证配置签名稳定（供 UI 判断"配置是否变化"以决定是否重建 Agent）。
final class ConfigSignatureTests: XCTestCase {
    private func sig(_ c: AgentConfig) -> String {
        guard let d = try? JSONSerialization.data(withJSONObject: c.export(),
                                                   options: [.sortedKeys]) else { return "" }
        return String(decoding: d, as: UTF8.self)
    }

    func testSignatureStableAcrossCalls() {
        var c = AgentConfig()
        c.knowledgeZones = [
            KnowledgeZone(id: "a", name: "法律", source: "/kb/law", isDefault: true),
            KnowledgeZone(id: "b", name: "医学", source: "/kb/med"),
        ]
        c.migrate()
        let original = sig(c)
        XCTAssertEqual(original, sig(c), "同一配置多次导出签名必须一致")
        c.ragMode = "colbert"
        XCTAssertNotEqual(sig(c), original, "模式变化应改变签名")
    }

    func testSignatureChangesWithZones() {
        var c = AgentConfig()
        let before = sig(c)
        c.knowledgeZones.append(KnowledgeZone(id: "x", name: "新库", source: "/kb/x"))
        XCTAssertNotEqual(sig(c), before, "分区变化应改变签名")
    }
}
