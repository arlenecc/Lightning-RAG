import XCTest
@testable import BaseAgentCore

/// 决策模型打分器验证（门控：仅本机存在 PKU-KCL-Meteor 模型时运行）。
final class MeteorDecisionScorerTests: XCTestCase {
    func testRelevantPassageScoresHigher() async throws {
        let dir = "/Users/arlenecc/codebase/agent-app/models/PKU-KCL-Meteor-0.8B-mlx-int4"
        guard FileManager.default.fileExists(atPath: dir + "/readout.safetensors") else {
            throw XCTSkip("本机无决策模型，跳过")
        }
        let scorer = MeteorDecisionScorer(modelDir: dir)
        let query = "合同违约责任如何认定"
        let passages = [
            "这是一段关于天气和旅游的介绍，与法律无关。",
            "当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。",
        ]
        let scores = try await scorer.score(query: query, passages: passages,
                                            coreSentenceCount: 3, useCoreSentences: true)
        XCTAssertEqual(scores.count, 2)
        print("METEOR scores=\(scores.map { String(format: "%.4f", $0) }.joined(separator: ","))")
        XCTAssertGreaterThan(scores[1], scores[0], "含违约责任条款的切片判定分应显著更高")
    }

    /// 批量前向与逐条判定应给出一致的 Top3（padding 误差不影响排序）。
    func testBatchMatchesSingleTop3() async throws {
        let dir = "/Users/arlenecc/codebase/agent-app/models/PKU-KCL-Meteor-0.8B-mlx-int4"
        guard FileManager.default.fileExists(atPath: dir + "/readout.safetensors") else {
            throw XCTSkip("本机无决策模型，跳过")
        }
        let scorer = MeteorDecisionScorer(modelDir: dir)
        let query = "合同违约责任如何认定"
        let passages = [
            "今天的天气很好适合外出散步旅游。",
            "当事人一方不履行合同义务应当承担违约责任并赔偿损失。",
            "这是一段关于计算机软件开发的无关介绍。",
            "民法典规定违约方需承担继续履行等责任。",
            "本周股市行情波动较大投资者需谨慎。",
            "合同约定了双方的权利义务和违约后果。",
            "这道菜的烹饪方法非常简单易学。",
            "租赁合同中承租人应当按期支付租金。",
        ]
        let batch = try await scorer.score(query: query, passages: passages,
                                            coreSentenceCount: 3, useCoreSentences: true)
        var singles: [Double] = []
        for p in passages {
            singles.append(try await scorer.score(query: query, passages: [p],
                                                   coreSentenceCount: 3, useCoreSentences: true)[0])
        }
        let batchTop3 = Set(batch.enumerated().sorted { $0.element > $1.element }.prefix(3).map(\.offset))
        let singleTop3 = Set(singles.enumerated().sorted { $0.element > $1.element }.prefix(3).map(\.offset))
        XCTAssertEqual(batchTop3, singleTop3, "批量与逐条 Top3 应一致")
    }
}
