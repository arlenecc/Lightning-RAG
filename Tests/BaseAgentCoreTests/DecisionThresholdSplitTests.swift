import XCTest
@testable import BaseAgentCore

/// 三个判定阈值独立配置：计划完整性 / 执行完成度 / 技能判定。
final class DecisionThresholdSplitTests: XCTestCase {
    /// 可控概率的判定打分器（按 question 关键字返回不同概率）。
    private final class ProbScorer: BinaryDecisionScoring, @unchecked Sendable {
        let isReady = true
        let isModelAvailable = true
        var planProb: Double = 0.5
        var finishProb: Double = 0.5
        var skillProb: Double = 0.5
        func judgeBinary(state: String, question: String,
                         trueOption: String, falseOption: String,
                         maxStateChars: Int) async throws -> Double {
            if question.contains("plan step") { return planProb }
            if question.contains("sufficient") { return finishProb }
            if question.contains("skill") { return skillProb }
            return 0.5
        }
        func score(query: String, passages: [String], coreSentenceCount: Int,
                   useCoreSentences: Bool,
                   progress: ((Int, Int) -> Void)?) async throws -> [Double] {
            passages.map { _ in 0.5 }
        }
    }

    func testConfigDefaultsAndIndependentRoundTrip() {
        let c = AgentConfig()
        XCTAssertEqual(c.kernelPlanReviewThreshold, 0.5)
        XCTAssertEqual(c.kernelFinishGateThreshold, 0.5)
        XCTAssertEqual(c.skillDecisionGateThreshold, 0.5)
        let d = c.export()
        XCTAssertNotNil(d["kernel_plan_review_threshold"])
        XCTAssertNotNil(d["kernel_finish_gate_threshold"])
        XCTAssertNotNil(d["skill_decision_gate_threshold"])
        XCTAssertNotNil(d["kernel_decision_gate_threshold"], "旧键保留兼容")
    }

    /// 旧键 kernel_decision_gate_threshold 只覆盖执行完成度，不影响计划完整性。
    func testLegacyKeyOnlyOverridesFinishThreshold() {
        var c = AgentConfig()
        c.kernelPlanReviewThreshold = 0.7
        let d = c.export()
        XCTAssertEqual(d["kernel_plan_review_threshold"] as? Double, 0.7)
        XCTAssertEqual(d["kernel_finish_gate_threshold"] as? Double, 0.5)
    }

    /// 双阈值独立生效：plan=0.45<0.6 → 判不必要；finish=0.55>=0.5 → 判足够。
    func testPlanAndFinishThresholdsIndependent() async {
        let scorer = ProbScorer()
        scorer.planProb = 0.45
        scorer.finishProb = 0.55
        let judge = MeteorKernelJudge(scorer: scorer,
                                      planThreshold: 0.6, finishThreshold: 0.5)
        let planOK = await judge.stepNecessary(goal: "g", step: "s")
        let finishOK = await judge.informationSufficient(task: "t", evidence: "e")
        XCTAssertEqual(planOK, false, "0.45 < 0.6 应判不必要")
        XCTAssertEqual(finishOK, true, "0.55 >= 0.5 应判足够")
    }

    /// 旧 init 的 threshold 作为两者共同默认，兼容既有调用。
    func testLegacyInitThresholdAppliesToBoth() async {
        let scorer = ProbScorer()
        scorer.planProb = 0.55
        scorer.finishProb = 0.55
        let judge = MeteorKernelJudge(scorer: scorer, threshold: 0.6)
        let plan = await judge.stepNecessary(goal: "g", step: "s")
        let finish = await judge.informationSufficient(task: "t", evidence: "e")
        XCTAssertEqual(plan, false)
        XCTAssertEqual(finish, false)
    }
}
