import XCTest
@testable import BaseAgentCore

/// 技能学习门控：至少 3 步计划 + 用户中途纠正/补充 + 全部完成才允许新建技能。
final class SkillLearningGateTests: XCTestCase {

    private func plan(goal: String, steps: Int, done: Bool) -> Plan {
        Plan(goal: goal, steps: (1...steps).map {
            PlanStep(title: "步骤\($0)", status: done ? "done" : "pending")
        })
    }

    func testEligibilityNeedsThreeStepsAndUserIntervention() {
        let session = Session(budget: ContextBudget(total: 32_768))
        XCTAssertFalse(session.isSkillLearningEligible())

        session.setPlan(plan(goal: "整理季度报告", steps: 3, done: true))
        XCTAssertFalse(session.isSkillLearningEligible(), "没有用户介入不应学习")

        session.noteSkillGuidance(planGoal: "整理季度报告", corrective: false,
                                  supplemental: false)
        XCTAssertFalse(session.isSkillLearningEligible(), "单纯「继续」不应学习")

        session.noteSkillGuidance(planGoal: "整理季度报告", corrective: false,
                                  supplemental: true)
        XCTAssertTrue(session.isSkillLearningEligible(), "用户补充要求后应允许学习")

        session.setPlan(plan(goal: "整理季度报告", steps: 2, done: true))
        XCTAssertFalse(session.isSkillLearningEligible(), "少于 3 步不应学习")

        session.setPlan(plan(goal: "整理季度报告", steps: 4, done: false))
        XCTAssertFalse(session.isSkillLearningEligible(), "未全部完成不应学习")

        session.setPlan(plan(goal: "另一个计划", steps: 3, done: true))
        session.noteSkillGuidance(planGoal: "另一个计划", corrective: true)
        XCTAssertTrue(session.isSkillLearningEligible())

        session.noteSkillGuidance(planGoal: "第三个计划", corrective: true)
        XCTAssertFalse(session.isSkillLearningEligible(), "目标不同的旧信号不应复用")
    }

    func testCorrectionAndSupplementDetection() {
        XCTAssertTrue(SkillManager.isCorrectiveFeedback("这里不对，应该先备份再删除"))
        XCTAssertTrue(SkillManager.isCorrectiveFeedback("That's wrong, use CSV instead"))
        XCTAssertFalse(SkillManager.isCorrectiveFeedback("继续执行下一步"))

        XCTAssertTrue(SkillManager.isSupplementalFeedback("补充一下，输出使用 Excel"))
        XCTAssertTrue(SkillManager.isSupplementalFeedback("记得把日期格式改成 ISO"))
        XCTAssertFalse(SkillManager.isSupplementalFeedback("继续"))
        XCTAssertFalse(SkillManager.isSupplementalFeedback("好的，下一步"))
        XCTAssertFalse(SkillManager.isSupplementalFeedback("ok"))
        XCTAssertFalse(SkillManager.isSupplementalFeedback("谢谢"))
    }

    func testSaveToolRejectsNewSkillWithoutEligibility() async throws {
        let root = NSTemporaryDirectory() + "/skill-gate-\(UUID().uuidString)"
        defer { try? FileManager.default.removeItem(atPath: root) }
        var config = AgentConfig()
        config.workspace = root

        let manager = SkillManager(workspace: root)
        let registry = ToolRegistry(config: config)
        let session = Session(budget: ContextBudget(total: 32_768))
        registry.session = session
        let tool = SkillSaveTool(
            config: config,
            registry: registry,
            manager: manager,
            retriever: SkillRetriever(manager: manager),
            index: SkillIndex(workspace: root))

        let args: [String: JSONValue] = [
            "name": .string("季度报告整理"),
            "description": .string("把季度数据整理成固定结构的报告"),
            "steps": .string("1. 收集数据\n2. 汇总\n3. 输出"),
        ]
        var result = try await tool.run(args)
        XCTAssertFalse(result.success)

        session.setPlan(plan(goal: "整理季度报告", steps: 3, done: true))
        session.noteSkillGuidance(planGoal: "整理季度报告", corrective: false,
                                  supplemental: true)
        result = try await tool.run(args)
        XCTAssertTrue(result.success, result.error)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: root + "/.agent/skills/季度报告整理/SKILL.md"))

        // 定时/无人值守任务不允许再新建技能。
        registry.setUnattended(true)
        let blocked = try await tool.run([
            "name": .string("表格清洗"),
            "description": .string("清洗导出的表格"),
            "steps": .string("1. 去重\n2. 补空值\n3. 校验"),
        ])
        XCTAssertFalse(blocked.success)
    }
}
