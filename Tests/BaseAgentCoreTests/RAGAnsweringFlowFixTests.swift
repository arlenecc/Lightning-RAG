import XCTest
@testable import BaseAgentCore

/// 修复「RAG 已命中却转 web_search / 意图空转」：四项改动的单元测试。
final class RAGAnsweringFlowFixTests: XCTestCase {
    /// 问句识别：某本书讲什么算内容问句；实时/网址问句不算。
    func testDocumentContentQuestionRecognition() {
        for q in ["寄生虫星球主要讲什么", "这本书的主要内容", "请概述一下这本书",
                  "这本书写了什么", "介绍一下文档内容"] {
            XCTAssertTrue(KernelLoop.looksLikeDocumentContentQuestion(q), "应识别: \(q)")
        }
        for q in [" parasite 最新消息", "这个网址是什么", "今天的实时新闻", "http://example.com 讲什么"] {
            XCTAssertFalse(KernelLoop.looksLikeDocumentContentQuestion(q), "不应识别为纯内容问句: \(q)")
        }
    }

    /// 决策门证据摘要：长工具结果不再被截成 180 字（通过反射不可行，改为行为断言：
    /// 内核里 preview 常量应为 600——用源码级一致性测试守护，见下方 fileCheck）。
    func testEvidencePreviewLengthInSource() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/BaseAgentCore/Kernel/Loop.swift")
        let source = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(source.contains("private func summarizeToolResults(limit: Int = 5)"))
        XCTAssertTrue(source.contains("String(trimmed.prefix(600))"))
        XCTAssertFalse(source.contains("String(trimmed.prefix(180))"))
    }

    /// outline 不再重置检索计数。
    func testOutlineDoesNotResetRagSearchCount() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/BaseAgentCore/Kernel/Loop.swift")
        let source = try String(contentsOf: url, encoding: .utf8)
        XCTAssertFalse(source.contains("case (\"rag_tools\", \"outline\"):\n                    liveCount = 0"))
        XCTAssertTrue(source.contains("拦截过早 web_search"))
    }

    /// 意图空转时补 done 的 helper 存在且被条件调用。
    func testStalledRagPlanCompletionWired() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/BaseAgentCore/Kernel/Loop.swift")
        let source = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(source.contains("completeStalledRagPlan()"))
        XCTAssertTrue(source.contains("didWork, hasToolSuccess, hasSuccessfulRagResult()"))
        XCTAssertTrue(source.contains("前置数据已就绪：现在直接基于上方 RAG 结果"))
    }
}
