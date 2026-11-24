import XCTest
import Foundation
@testable import BaseAgentCore

/// 约束解码开关的默认开启与配置往返：未写配置键时按开启处理，
/// 显式 false 仍可关闭（普通对话与远端 API 后端本就不受影响）。
final class GuidedToolArgumentsConfigTests: XCTestCase {
    func testDefaultsOnAndRoundTrips() throws {
        let config = AgentConfig()
        XCTAssertTrue(config.guidedToolArguments, "默认应开启本地工具调用约束解码")
        XCTAssertTrue(AgentConfig.mainLLMOnlyKeys.contains("guided_tool_arguments"))

        var disabled = config
        disabled.guidedToolArguments = false
        let exported = disabled.export()
        XCTAssertEqual(exported["guided_tool_arguments"] as? Bool, false)

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("guided-tool-args-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("config.json")
        try JSONSerialization.data(withJSONObject: exported).write(to: path)
        XCTAssertFalse(AgentConfig.load(path.path).guidedToolArguments,
                       "显式 false 必须被尊重")
    }

    func testMissingKeyUsesDefaultOn() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("guided-tool-default-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("config.json")
        try JSONSerialization.data(withJSONObject: [String: Any]()).write(to: path)
        XCTAssertTrue(AgentConfig.load(path.path).guidedToolArguments,
                      "旧配置未写 guided_tool_arguments 时按默认开启")
    }
}
