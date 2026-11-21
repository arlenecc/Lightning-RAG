import XCTest
@testable import BaseAgentCore

/// 自动注入摘要技能的决策判定（纯逻辑验证，不构造 Agent 避免后台任务干扰）。
final class SkillAutoInjectGateTests: XCTestCase {
    private final class StubJudge: SkillDecisionJudge, @unchecked Sendable {
        let appropriate: Bool?
        private let lock = NSLock()
        private(set) var calls = 0
        private(set) var lastQuestion = ""
        private(set) var lastState = ""
        init(_ appropriate: Bool?) { self.appropriate = appropriate }
        func pickSkill(question: String, conversationState: String,
                       candidates: [(name: String, description: String, tags: [String])]) async -> Int? { nil }
        func isAppropriate(question: String, conversationState: String,
                           skill: (name: String, description: String, tags: [String])) async -> Bool? {
            lock.lock(); calls += 1; lastQuestion = question; lastState = conversationState; lock.unlock()
            return appropriate
        }
    }

    private func skill() -> Skill {
        Skill(name: "python-scripting", description: "Use python scripts for automation",
              tags: ["python", "script"], keywords: ["python"],
              directory: "/tmp/skill", absolutePath: "/tmp/skill",
              entry: "SKILL.md", whenToUse: "", whenNotToUse: "", examples: "",
              fingerprint: "fp", body: "python instructions")
    }

    func testInappropriateSkillBlocked() async {
        let judge = StubJudge(false)
        let allowed = await Agent.evaluateAutoSkillInjection(
            judge: judge, skill: skill(), question: "python script",
            conversationState: "Plan: write script")
        XCTAssertFalse(allowed)
        XCTAssertEqual(judge.calls, 1)
        XCTAssertEqual(judge.lastQuestion, "python script")
        XCTAssertEqual(judge.lastState, "Plan: write script")
    }

    func testAppropriateSkillAllowed() async {
        let judge = StubJudge(true)
        let allowed = await Agent.evaluateAutoSkillInjection(
            judge: judge, skill: skill(), question: "python script", conversationState: "")
        XCTAssertTrue(allowed)
    }

    func testJudgeNilReturnsTrue() async {
        let allowed = await Agent.evaluateAutoSkillInjection(
            judge: nil, skill: skill(), question: "python script", conversationState: "")
        XCTAssertTrue(allowed, "无判定器应放行")
    }

    func testJudgeUndecidableReturnsTrue() async {
        let allowed = await Agent.evaluateAutoSkillInjection(
            judge: StubJudge(nil), skill: skill(), question: "python script", conversationState: "")
        XCTAssertTrue(allowed, "判定不可用应放行")
    }
}
