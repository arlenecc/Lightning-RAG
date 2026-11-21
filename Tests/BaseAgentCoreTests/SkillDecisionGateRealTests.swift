import XCTest
@testable import BaseAgentCore

/// 真机验证 MeteorSkillJudge 用真实决策模型做技能判定：相关技能放行、不相关拦截。
final class SkillDecisionGateRealTests: XCTestCase {
    private func makeJudge() throws -> MeteorSkillJudge {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let modelDir = repoRoot.appendingPathComponent("models/PKU-KCL-Meteor-0.8B-mlx-int4")
        guard FileManager.default.fileExists(
            atPath: modelDir.appendingPathComponent("readout.safetensors").path) else {
            throw XCTSkip("决策模型文件不存在，跳过真机技能判定测试")
        }
        let scorer = MeteorDecisionScorer(modelDir: modelDir.path)
        return MeteorSkillJudge(scorer: scorer, threshold: 0.5)
    }

    func testRelatedSkillApprovedUnrelatedRejected() async throws {
        let judge = try makeJudge()
        let question = "帮我用 Python 写个脚本批量重命名文件"
        let state = "用户需要一个文件批量重命名方案。"
        let related = (name: "python-scripting",
                       description: "使用 Python 编写自动化脚本处理文件与目录",
                       tags: ["python", "script", "file"])
        let unrelated = (name: "cooking-recipes",
                         description: "常见中式菜谱与烹饪技巧",
                         tags: ["cooking", "food"])
        let relatedOK = await judge.isAppropriate(question: question,
                                                  conversationState: state, skill: related)
        let unrelatedOK = await judge.isAppropriate(question: question,
                                                    conversationState: state, skill: unrelated)
        XCTAssertNotNil(relatedOK, "判定应可用")
        XCTAssertNotNil(unrelatedOK, "判定应可用")
        XCTAssertTrue(relatedOK == true, "Python 脚本技能应判为合适（实际 \(String(describing: relatedOK))）")
        XCTAssertTrue(unrelatedOK == false, "菜谱技能应判为不合适（实际 \(String(describing: unrelatedOK))）")
    }

    func testPickAmongRealCandidates() async throws {
        let judge = try makeJudge()
        let picked = await judge.pickSkill(
            question: "我想查询知识库里有哪些文档",
            conversationState: "",
            candidates: [
                (name: "rag-search", description: "检索本地知识库文档内容", tags: ["rag", "search"]),
                (name: "web-browse", description: "浏览互联网网页", tags: ["web"]),
                (name: "git-workflow", description: "Git 分支管理与提交流程", tags: ["git"]),
            ])
        XCTAssertNotNil(picked)
        XCTAssertEqual(picked, 0, "应选中知识库检索技能（实际 \(String(describing: picked))）")
    }
}
