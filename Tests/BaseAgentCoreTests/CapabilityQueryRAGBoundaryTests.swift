import XCTest
@testable import BaseAgentCore

/// 能力问询轻量直答 vs 知识库内容问句（必须走 LLM+RAG）的边界。
/// 截图 bug：「社会分区具体有什么」中的「会」曾误命中「能/会/可以…什么」。
final class CapabilityQueryRAGBoundaryTests: XCTestCase {
    func testKnowledgeContentQuestionsAreNotCapabilityQueries() {
        let questions = [
            "你知识库里有什么",
            "社会分区具体有什么",
            "知识库的社会分区具体有什么",
            "你有哪些文档",
            "历史分区有哪些书",
            "自然分区有什么资料",
        ]
        for q in questions {
            XCTAssertFalse(SkillManager.isCapabilityQuery(q),
                "知识库内容问句必须走 LLM+RAG，不能本地直答工具清单: \(q)")
        }
    }

    func testRealCapabilityQueriesStillUseDirectAnswer() {
        let queries = [
            "你能做什么",
            "你可以干什么",
            "你都能做什么",
            "你有什么工具",
            "有哪些技能",
            "介绍一下你自己",
            "你是谁",
            "what can you do",
        ]
        for q in queries {
            XCTAssertTrue(SkillManager.isCapabilityQuery(q),
                "真正的能力问询仍应轻量直答: \(q)")
        }
    }

    func testGreetingsUnaffected() {
        XCTAssertTrue(SkillManager.isGreeting("你好"))
        XCTAssertFalse(SkillManager.isCapabilityQuery("你好"))
    }
}
