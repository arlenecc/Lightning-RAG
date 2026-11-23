import XCTest
@testable import BaseAgentCore

/// 记忆整理建议生产端开关：默认关闭，避免无下游产物在后台每轮重试。
final class MemoryTidyToggleTests: XCTestCase {
    /// 默认关闭；编解码往返保持。
    func testDefaultOffAndRoundTrip() throws {
        var config = AgentConfig()
        XCTAssertFalse(config.memoryTidyEnabled)
        let exported = try XCTUnwrap(config.export() as? [String: Any])
        XCTAssertEqual(exported["memory_tidy_enabled"] as? Bool, false)

        config.memoryTidyEnabled = true
        let exportedOn = try XCTUnwrap(config.export() as? [String: Any])
        XCTAssertEqual(exportedOn["memory_tidy_enabled"] as? Bool, true)
    }

    /// JSON 里的 memory_tidy_enabled 能正确解码。
    func testDecodeFromJSON() throws {
        let path = NSTemporaryDirectory() + "agent-config-\(UUID().uuidString).json"
        defer { try? FileManager.default.removeItem(atPath: path) }
        try #"{"memory_tidy_enabled": true}"#
            .write(toFile: path, atomically: true, encoding: .utf8)
        XCTAssertTrue(AgentConfig.load(path).memoryTidyEnabled)

        try #"{"memory_tidy_enabled": false}"#
            .write(toFile: path, atomically: true, encoding: .utf8)
        XCTAssertFalse(AgentConfig.load(path).memoryTidyEnabled)
    }

    /// 后台整理调度必须受开关控制（源码级断言，与 HistoryInsightToggleTests 一致）。
    func testFireFinishedGatedByConfig() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let agent = try String(contentsOf: root
            .appendingPathComponent("Sources/BaseAgentCore/Agent/Agent.swift"), encoding: .utf8)
        XCTAssertTrue(agent.contains("config.memoryTidyEnabled"),
            "fireFinished 的记忆整理必须受 memory_tidy_enabled 控制")
    }
}
