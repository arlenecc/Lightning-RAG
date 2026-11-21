import XCTest
@testable import BaseAgentCore

/// 决策模型每次打分都输出可读日志（供用户对照调整阈值）。
final class DecisionScoreLogTests: XCTestCase {
    private final class FixedScorer: BinaryDecisionScoring, @unchecked Sendable {
        let isReady = true
        let isModelAvailable = true
        let prob: Double
        let scores: [Double]
        init(prob: Double = 0.5, scores: [Double] = []) {
            self.prob = prob; self.scores = scores
        }
        func judgeBinary(state: String, question: String,
                         trueOption: String, falseOption: String,
                         maxStateChars: Int) async throws -> Double { prob }
        func score(query: String, passages: [String], coreSentenceCount: Int,
                   useCoreSentences: Bool,
                   progress: ((Int, Int) -> Void)?) async throws -> [Double] {
            scores.isEmpty ? passages.map { _ in 0.5 } : scores
        }
    }

    /// 捕获某段代码里经 AgentLog 输出的全部日志行。
    private func captured(_ body: () async throws -> Void) async throws -> [String] {
        let box = LogBox()
        try await AgentLog.withTaskSink({ box.append($0) }, {
            try await body()
        })
        return box.lines
    }

    private final class LogBox: @unchecked Sendable {
        private let lock = NSLock()
        private var storage: [String] = []
        var lines: [String] { lock.lock(); defer { lock.unlock() }; return storage }
        func append(_ s: String) { lock.lock(); storage.append(s); lock.unlock() }
    }

    func testBinaryJudgmentsLogProbThresholdAndResult() async throws {
        let scorer = FixedScorer(prob: 0.42)
        let judge = MeteorKernelJudge(scorer: scorer,
                                      planThreshold: 0.6, finishThreshold: 0.5)
        let lines = try await captured {
            _ = await judge.stepNecessary(goal: "写个脚本", step: "查资料")
            _ = await judge.informationSufficient(task: "查知识库", evidence: "结果")
        }
        let decisionLines = lines.filter { $0.contains("[decision]") }
        XCTAssertEqual(decisionLines.count, 2, "两次打分各一条日志")
        XCTAssertTrue(decisionLines.contains { $0.contains("plan_step")
            && $0.contains("prob=0.420") && $0.contains("thr=0.60")
            && $0.contains("BLOCK") && $0.contains("写个脚本") },
            "计划评审需含分数/阈值/结果/问句: \(decisionLines)")
        XCTAssertTrue(decisionLines.contains { $0.contains("complete")
            && $0.contains("prob=0.420") && $0.contains("thr=0.50")
            && $0.contains("BLOCK") },
            "收尾拦截需含分数/阈值/结果: \(decisionLines)")
    }

    func testSkillFitLogsThresholdAndTarget() async throws {
        let judge = MeteorSkillJudge(scorer: FixedScorer(prob: 0.81), threshold: 0.5)
        let lines = try await captured {
            _ = await judge.isAppropriate(
                question: "python 脚本", conversationState: "",
                skill: (name: "python-scripting", description: "写脚本", tags: ["python"]))
        }
        let line = lines.first { $0.contains("[decision]") }
        XCTAssertNotNil(line)
        XCTAssertTrue(line?.contains("skill_fit") ?? false)
        XCTAssertTrue(line?.contains("prob=0.810") ?? false)
        XCTAssertTrue(line?.contains("thr=0.50") ?? false)
        XCTAssertTrue(line?.contains("PASS") ?? false)
        XCTAssertTrue(line?.contains("python-scripting") ?? false)
    }

    func testMultiSkillRankingLogsAllScoresAndCriterion() async throws {
        let judge = MeteorSkillJudge(scorer: FixedScorer(scores: [0.1, 0.9, 0.2]),
                                     threshold: 0.5)
        let lines = try await captured {
            _ = await judge.pickSkill(
                question: "查知识库", conversationState: "",
                candidates: [
                    (name: "a", description: "A", tags: []),
                    (name: "b", description: "B", tags: []),
                    (name: "c", description: "C", tags: []),
                ])
        }
        let line = lines.first { $0.contains("[decision]") }
        XCTAssertTrue(line?.contains("skill_pick") ?? false)
        XCTAssertTrue(line?.contains("a=0.100") ?? false, "应打印每个候选分数")
        XCTAssertTrue(line?.contains("b=0.900") ?? false)
        XCTAssertTrue(line?.contains("picked=b") ?? false)
        XCTAssertTrue(line?.contains("criterion=best>0&best>=mean") ?? false,
            "应说明多技能不受技能阈值控制")
    }
}
