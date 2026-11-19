import XCTest
@testable import BaseAgentCore

/// 多分区决策判定模式端到端实测（门控：本机有决策模型时运行）：
/// 验证 第一次判选库 → 各库 FTS 粗筛 → 加权 RRF → 第二次核心句判定 Top3 全链路。
final class MultiZoneDecisionE2E: XCTestCase {
    private final class NoopCallbacks: AgentCallbacks {}

    func testDecisionTwoStageAcrossZones() async throws {
        let modelDir = "/Users/arlenecc/codebase/agent-app/models/PKU-KCL-Meteor-0.8B-mlx-int4"
        guard FileManager.default.fileExists(atPath: modelDir + "/readout.safetensors") else {
            throw XCTSkip("本机无决策模型")
        }
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mz-decision-e2e-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let lawDir = root.appendingPathComponent("law")
        let weatherDir = root.appendingPathComponent("weather")
        try FileManager.default.createDirectory(at: lawDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: weatherDir, withIntermediateDirectories: true)

        try """
        # 合同违约责任
        当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。
        违约方应当赔偿守约方因此遭受的全部损失，包括合同履行后可以获得的利益。
        当事人可以约定一方违约时应当根据违约情况向对方支付一定数额的违约金。
        """.write(to: lawDir.appendingPathComponent("contract.md"), atomically: true, encoding: .utf8)
        try """
        # 天气记录
        今天的天气很好，阳光明媚，微风和煦，非常适合外出散步和旅游，公园里有很多人在享受这美好的时光。
        明天预计有小雨，出门记得带伞，注意保暖，气温会比今天下降几度，请适时增添衣物以防感冒着凉。
        后天天气转晴，适合进行户外运动，但早晚温差较大，建议穿着便于增减的衣物，注意及时补充水分。
        本周整体天气状况良好，只有周中有短暂的降雨过程，其余时间都以晴天和多云为主，适合安排各类户外活动。
        """.write(to: weatherDir.appendingPathComponent("weather.md"), atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = lawDir.path
        config.ragMode = "decision"
        config.ragRerankEnabled = false
        config.ragChunkSize = 100
        config.decisionCoarseCandidates = 20
        config.decisionTopN = 3
        config.decisionModelPath = modelDir
        config.decisionUseCoreSentences = true
        config.knowledgeZones = [
            KnowledgeZone(id: "law", name: "法律合同", source: lawDir.path, enabled: true, isDefault: true),
            KnowledgeZone(id: "weather", name: "天气", source: weatherDir.path, enabled: true),
        ]

        let agent = Agent(llm: UnloadedLLMService(), config: config, callbacks: NoopCallbacks())
        XCTAssertEqual(agent.zoneEngines.count, 2, "两个启用分区各一个 engine")
        XCTAssertTrue(agent.zoneEngines["law"] === agent.ragEngine, "默认分区即主 engine")

        // 两个分区各自独立同步（决策模式无需 embedding）。
        for zone in config.knowledgeZones {
            let engine = agent.zoneEngines[zone.id]!
            let stats = await engine.ingest()
            XCTAssertEqual(stats.processed, 1, "\(zone.name) 应同步 1 个文档")
            XCTAssertTrue(stats.errors.isEmpty, "\(zone.name): \(stats.errors.joined(separator: ";"))")
        }

        // 全链路：决策两次判定联合检索。法律问句应命中法律分区。
        let outcome = try await agent.multiZoneSearch(query: "合同违约责任如何赔偿", topK: 3)
        XCTAssertFalse(outcome.citations.isEmpty, "决策联合检索应有结果，context=\(outcome.context)")
        XCTAssertTrue(outcome.citations.allSatisfy { $0.id.hasPrefix("law|") || $0.id.hasPrefix("weather|") },
                      "结果应带分区前缀")
        let lawHits = outcome.citations.filter { $0.id.hasPrefix("law|") }.count
        XCTAssertGreaterThan(lawHits, 0, "合同问句应至少命中一个法律分区切片，实际=\(lawHits)/\(outcome.citations.count)")
        agent.close()
    }
}
