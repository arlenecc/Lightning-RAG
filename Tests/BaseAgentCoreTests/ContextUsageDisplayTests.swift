import XCTest
@testable import BaseAgentCore

/// 主界面上下文占用百分比显示的计算边界（方法在 Config，Core 模块可直接测）。
final class ContextUsageDisplayTests: XCTestCase {
    func testZeroCases() {
        let config = AgentConfig()
        XCTAssertEqual(config.contextUsagePercent(totalTokens: 0), 0)
        XCTAssertEqual(config.contextUsagePercent(totalTokens: -1), 0)
    }

    func testTypicalRatios() {
        var config = AgentConfig()
        config.maxContextTokens = 8_192
        config.kvCacheMaxTokens = 0
        XCTAssertEqual(config.effectiveMainContextTokens, 8_192)
        XCTAssertEqual(config.contextUsagePercent(totalTokens: 4096), 50)
        XCTAssertEqual(config.contextUsagePercent(totalTokens: 7671), 94)
        XCTAssertEqual(config.contextUsagePercent(totalTokens: 40), 0)
    }

    func testClampedTo100() {
        let config = AgentConfig()
        XCTAssertEqual(config.contextUsagePercent(totalTokens: 1_000_000), 100)
    }

    /// 与配置实际生效窗口一致（KV 上限与 maxContextTokens 取小）。
    func testUsesEffectiveWindow() {
        var config = AgentConfig()
        config.maxContextTokens = 32_768
        config.kvCacheMaxTokens = 12_000
        XCTAssertEqual(config.effectiveMainContextTokens, 12_000)
        XCTAssertEqual(config.contextUsagePercent(totalTokens: 6_000), 50)
    }
}
