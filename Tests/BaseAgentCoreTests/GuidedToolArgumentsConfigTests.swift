import XCTest
import Foundation
@testable import BaseAgentCore

/// opt-in 开关的默认关闭与配置往返。
final class GuidedToolArgumentsConfigTests: XCTestCase {
    func testDefaultsOffAndRoundTrips() throws {
        let config = AgentConfig()
        XCTAssertFalse(config.guidedToolArguments, "默认必须关闭，普通对话零影响")
        XCTAssertTrue(AgentConfig.mainLLMOnlyKeys.contains("guided_tool_arguments"))

        var enabled = config
        enabled.guidedToolArguments = true
        let exported = enabled.export()
        XCTAssertEqual(exported["guided_tool_arguments"] as? Bool, true)

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("guided-tool-args-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("config.json")
        try JSONSerialization.data(withJSONObject: exported).write(to: path)
        XCTAssertTrue(AgentConfig.load(path.path).guidedToolArguments)
    }
}
