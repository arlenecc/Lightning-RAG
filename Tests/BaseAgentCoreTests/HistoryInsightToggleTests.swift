import XCTest
@testable import BaseAgentCore

/// 历史提炼生产端开关：默认关闭（产物无下游消费者），session 仍入队不丢。
final class HistoryInsightToggleTests: XCTestCase {
    /// 默认关闭；编解码往返保持。
    func testDefaultOffAndRoundTrip() throws {
        var config = AgentConfig()
        XCTAssertFalse(config.memoryHistoryInsightEnabled)
        let exported = try XCTUnwrap(config.export() as? [String: Any])
        XCTAssertEqual(exported["memory_history_insight_enabled"] as? Bool, false)
        config.memoryHistoryInsightEnabled = true
        let exportedOn = try XCTUnwrap(config.export() as? [String: Any])
        XCTAssertEqual(exportedOn["memory_history_insight_enabled"] as? Bool, true)
    }

    /// 关闭时 fireFinished 的 hasPending 必须为 false（不拉起辅助模型），
    /// 但 RawSessionStore 的入队方法不受开关影响（数据不丢）——源码级断言。
    func testFireFinishedGatedByConfig() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let agent = try String(contentsOf: root
            .appendingPathComponent("Sources/BaseAgentCore/Agent/Agent.swift"), encoding: .utf8)
        XCTAssertTrue(agent.contains("config.memoryHistoryInsightEnabled"),
            "fireFinished 必须受开关控制")
        let store = try String(contentsOf: root
            .appendingPathComponent("Sources/BaseAgentCore/Memory/RawSessionStore.swift"),
                               encoding: .utf8)
        XCTAssertTrue(store.contains("enqueuePendingInsights"),
            "归档仍须入队（数据保留）")
    }
}
