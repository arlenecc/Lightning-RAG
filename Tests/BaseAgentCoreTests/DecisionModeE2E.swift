import XCTest
@testable import BaseAgentCore

/// 决策判定模式端到端回归（门控：本机有决策模型时运行）：
/// 验证 解析→切片→FTS入库（无embedding）→粗筛→核心句→模型判定→citations 全链路。
final class DecisionModeE2E: XCTestCase {
    func testDecisionModeIngestAndSearch() async throws {
        let modelDir = "/Users/arlenecc/codebase/agent-app/models/PKU-KCL-Meteor-0.8B-mlx-int4"
        guard FileManager.default.fileExists(atPath: modelDir + "/readout.safetensors") else {
            throw XCTSkip("本机无决策模型")
        }
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("decision-e2e-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)

        let docs = [
            "contract.md": """
            # 合同违约责任
            当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。
            违约方应当赔偿守约方因此遭受的全部损失，包括合同履行后可以获得的利益，但不得超过订立合同时预见到或者应当预见到的因违约可能造成的损失。
            当事人可以约定一方违约时应当根据违约情况向对方支付一定数额的违约金，也可以约定因违约产生的损失赔偿额的计算方法。
            约定的违约金低于造成的损失的，当事人可以请求人民法院或者仲裁机构予以增加；约定的违约金过分高于造成的损失的，可以请求予以适当减少。
            """,
            "weather.md": """
            # 天气记录
            今天的天气很好，阳光明媚，微风和煦，非常适合外出散步和旅游，公园里有很多人在享受这美好的时光。
            明天预计有小雨，出门记得带伞，注意保暖，气温会比今天下降几度，请适时增添衣物以防感冒。
            后天天气转晴，适合进行户外运动，但早晚温差较大，建议穿着便于增减的衣物，注意及时补充水分。
            本周整体天气状况良好，只有周中有短暂的降雨过程，其余时间都以晴天和多云为主，适合安排各类户外活动。
            """,
        ]
        for (name, text) in docs {
            try text.write(to: kb.appendingPathComponent(name), atomically: true, encoding: .utf8)
        }

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragMode = "decision"
        config.ragRerankEnabled = false
        config.ragChunkSize = 100
        config.decisionCoarseCandidates = 20
        config.decisionTopN = 3
        config.decisionModelPath = modelDir
        config.decisionUseCoreSentences = true

        // 决策模式不需要 embedding：传 nil 也应能完整同步。
        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: nil, reranker: nil)
        let stats = await engine.ingest()
        XCTAssertEqual(stats.processed, 2, "决策模式应完成 2 个文档同步")
        XCTAssertTrue(stats.errors.isEmpty, stats.errors.joined(separator: "; "))

        // 检索：合同相关问句应把 contract 文档排在前面。
        let outcome = try await engine.searchWithCitations(query: "合同违约责任如何赔偿")
        guard let top = outcome.citations.first else {
            XCTFail("决策模式应返回 citations，context=\(outcome.context)")
            engine.close()
            return
        }
        XCTAssertEqual(top.source, "contract.md", "合同问句的 Top1 应为合同文档")
        engine.close()
    }
}
