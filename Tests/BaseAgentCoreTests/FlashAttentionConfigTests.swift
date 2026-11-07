import Foundation
import XCTest
@testable import BaseAgentCore

/// Flash Attention 默认启用策略的配置层回归：
/// 默认开启 → MLX/GGUF 都走 fused attention 路径（KV 不量化）；
/// 关闭 → 恢复 int8 KV 量化省显存。
final class FlashAttentionConfigTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("flash-config-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testDefaultsToEnabledWithUnquantizedKV() {
        let config = AgentConfig()
        XCTAssertTrue(config.flashAttention, "Flash Attention 默认启用")
        // 用户配置 kv_cache_int8 默认 true，但 flash 优先时必须让 MLX
        // 走标准 KV 缓存（否则量化缓存绕过 MLXFast fused attention）。
        XCTAssertTrue(config.kvCacheInt8)
        XCTAssertFalse(config.effectiveMainKVInt8)
        XCTAssertFalse(config.effectiveAuxKVInt8)
    }

    func testDisablingRestoresKVQuantization() {
        var config = AgentConfig()
        config.flashAttention = false
        XCTAssertTrue(config.effectiveMainKVInt8)
        XCTAssertTrue(config.effectiveAuxKVInt8)

        config.kvCacheInt8 = false
        XCTAssertFalse(config.effectiveMainKVInt8, "关闭量化后仍应为 f16")

        config.auxKvCacheInt8 = false
        XCTAssertFalse(config.effectiveAuxKVInt8)
    }

    func testExportLoadRoundTrip() throws {
        var config = AgentConfig()
        config.flashAttention = false
        config.kvCacheInt8 = true
        let exported = config.export()
        XCTAssertEqual(exported["flash_attention"] as? Bool, false)

        let path = directory.appendingPathComponent("config.json")
        let data = try JSONSerialization.data(withJSONObject: exported)
        try data.write(to: path)

        let loaded = AgentConfig.load(path.path)
        XCTAssertFalse(loaded.flashAttention)
        XCTAssertTrue(loaded.effectiveMainKVInt8, "关闭 flash 后 KV 量化应生效")
    }

    /// 旧配置文件没有 flash_attention 键：按新默认（开启）处理。
    func testLegacyConfigDefaultsToEnabled() throws {
        var config = AgentConfig()
        config.flashAttention = false
        var exported = config.export()
        exported.removeValue(forKey: "flash_attention")
        let path = directory.appendingPathComponent("legacy.json")
        try JSONSerialization.data(withJSONObject: exported).write(to: path)

        let loaded = AgentConfig.load(path.path)
        XCTAssertTrue(loaded.flashAttention)
        XCTAssertFalse(loaded.effectiveMainKVInt8)
    }

    /// 主/辅模型切换该开关都应走轻量换模型路径，不整体重建 Agent。
    func testKeyClassifiedAsLLMOnly() {
        XCTAssertTrue(AgentConfig.mainLLMOnlyKeys.contains("flash_attention"))
        XCTAssertTrue(AgentConfig.auxLLMOnlyKeys.contains("flash_attention"))
    }
}
