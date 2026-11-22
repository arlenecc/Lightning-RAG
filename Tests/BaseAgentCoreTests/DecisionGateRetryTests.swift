import XCTest
@testable import BaseAgentCore

/// 决策门降级策略：超时可重试（连续 3 次才禁用），确定性错误立即禁用。
final class DecisionGateRetryTests: XCTestCase {

    /// 按脚本逐次响应的假打分器。
    private final class ScriptedScorer: BinaryDecisionScoring, @unchecked Sendable {
        enum Outcome {
            case value(Double)
            case slow          // 超过判定超时
            case failure
        }

        var isReady: Bool { true }
        var isModelAvailable: Bool { true }

        private let lock = NSLock()
        private var script: [Outcome]
        private(set) var calls = 0

        init(_ script: [Outcome]) {
            self.script = script
        }

        func judgeBinary(state: String, question: String,
                         trueOption: String, falseOption: String,
                         maxStateChars: Int) async throws -> Double {
            lock.lock()
            let outcome = script.isEmpty ? .failure : script.removeFirst()
            calls += 1
            lock.unlock()
            switch outcome {
            case .value(let v): return v
            case .slow:
                try await Task.sleep(nanoseconds: 300_000_000)
                return 0.5
            case .failure:
                throw AgentError.llm("模型缺失")
            }
        }

        func score(query: String, passages: [String], coreSentenceCount: Int,
                   useCoreSentences: Bool,
                   progress: ((Int, Int) -> Void)?) async throws -> [Double] {
            []
        }
    }

    func testTimeoutKeepsGateUntilThreeConsecutive() async {
        let scorer = ScriptedScorer([.slow, .slow, .slow, .value(0.9)])
        let judge = MeteorKernelJudge(scorer: scorer, timeoutSeconds: 0.05)
        // 前三次超时：放行（nil）但不永久禁用。
        for _ in 0..<3 {
            let r = await judge.informationSufficient(task: "t", evidence: "e")
            XCTAssertNil(r)
        }
        // 连续 3 次后禁用：第 4 次不再调用模型。
        let disabled = await judge.informationSufficient(task: "t", evidence: "e")
        XCTAssertNil(disabled)
        XCTAssertEqual(scorer.calls, 3)
    }

    func testSuccessResetsTimeoutStreak() async {
        let scorer = ScriptedScorer([.slow, .value(0.9), .slow, .slow])
        let judge = MeteorKernelJudge(scorer: scorer, timeoutSeconds: 0.05)
        let first = await judge.informationSufficient(task: "t", evidence: "e")
        XCTAssertNil(first)
        let second = await judge.informationSufficient(task: "t", evidence: "e")
        XCTAssertEqual(second, true)
        // 成功已清零计数，这两次仍只是超时放行，未达禁用。
        let third = await judge.informationSufficient(task: "t", evidence: "e")
        XCTAssertNil(third)
        let fourth = await judge.informationSufficient(task: "t", evidence: "e")
        XCTAssertNil(fourth)
        XCTAssertEqual(scorer.calls, 4)
    }

    func testDeterministicFailureDisablesImmediately() async {
        let scorer = ScriptedScorer([.failure, .value(0.9)])
        let judge = MeteorKernelJudge(scorer: scorer, timeoutSeconds: 0.05)
        let first = await judge.informationSufficient(task: "t", evidence: "e")
        XCTAssertNil(first)
        let second = await judge.informationSufficient(task: "t", evidence: "e")
        XCTAssertNil(second)
        XCTAssertEqual(scorer.calls, 1)
    }
}
