import XCTest
@testable import BaseAgentCore

/// 知识库问句路由：清单问句（"有什么"）应触发 status 引导；
/// 文档内容问句（"X 主要讲什么"）即使字面上不含"知识库"，也不该被当成联网问题。
final class KnowledgeAnswerRoutingTests: XCTestCase {
    func testDocumentContentQuestionRecognizedWithoutKnowledgeLiteral() {
        for q in ["未来能源：我们能做些什么 主要讲些什么", "门捷列夫之梦 主要讲什么",
                  "寄生虫星球主要内容", "这本书讲什么"] {
            XCTAssertTrue(KernelLoop.looksLikeDocumentContentQuestion(q),
                "应识别为文档内容问句: \(q)")
            XCTAssertFalse(KernelLoop.looksLikeKnowledgeBaseQuestion(q),
                "字面量判定确实不含知识库（证明旧逻辑会误转联网）: \(q)")
        }
    }

    func testSystemPromptHasStatusVsSearchRule() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(contentsOf: root
            .appendingPathComponent("Sources/BaseAgentCore/Agent/SystemPrompt.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains("分区清单"), "应有清单问句规则")
        XCTAssertTrue(source.contains("拿分区与文档清单"), "应明确 status 拿清单")
        XCTAssertTrue(source.contains("不要对这种清单问题用 `op=\"search\"`"), "应禁止清单用 search")
        XCTAssertTrue(source.contains("问具体书/文档的内容"), "应区分内容问句")
    }

    /// 内核代发 web_search 的过滤必须同时排除文档内容问句（源码级接线断言）。
    func testWebFallbackExcludesDocumentContentQuestion() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(contentsOf: root
            .appendingPathComponent("Sources/BaseAgentCore/Kernel/Loop.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains("shouldNotWebSearch"), "应引入综合判定")
        XCTAssertTrue(source.contains("Self.looksLikeDocumentContentQuestion(userInput)\n                            && hasSuccessfulRagResult()"),
            "文档内容问句 + 历史 RAG 命中禁止代发 web_search")
        XCTAssertTrue(source.contains("let fallbackArgs"), "RAG 兜底应按问句选 op")
    }
}
