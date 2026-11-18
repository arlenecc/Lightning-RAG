import XCTest
import Foundation
@testable import BaseAgentCore

/// 内核协议块（锚点提示）不得出现在给用户的答复里。
///
/// 回归：弱模型会把每轮注入 user 消息尾部的 `### [PLAN] … Now: …` 原样
/// 复述到 assistant 正文，用户看到「###[PLAN] Goal:…」而非回答。
final class ProtocolEchoTests: XCTestCase {
    /// 整段复述计划协议 → 剥成空串。
    func testStripPlanProtocolBlock() {
        let echo = """
        ###[PLAN]
        Goal:你知识库里有什么
        - [▶]1. 立即调用工具获取真实数据（如 web_scan / rag_tools(op="search")）
        -[]2. 基于工具结果整理并给出最终答复

        Now:1. 立即调用工具获取真实数据（如 web_scan / rag_tools(op="search")）
        """
        XCTAssertEqual(stripSummaryTags(echo), "")
    }

    /// 正文在前、协议块在后 → 只保留正文。
    func testStripPlanProtocolKeepsPrecedingAnswer() {
        let text = "已为你找到 3 篇文档。\n\n### [PLAN]\nGoal:查询\n- [ ]1. 检索\n\nNow:1. 检索"
        XCTAssertEqual(stripSummaryTags(text), "已为你找到 3 篇文档。")
    }

    /// 工作记忆协议块同样剥离。
    func testStripWorkingMemoryBlock() {
        let text = """
        ### [WORKING MEMORY]
        Current turn: 3
        <key_info>{"skill":"x"}</key_info>

        正常答复
        """
        XCTAssertEqual(stripSummaryTags(text), "正常答复")
    }

    /// 正常答复不受影响。
    func testNormalAnswerUntouched() {
        let text = "知识库里有 12 个文档，涵盖lys三主题。"
        XCTAssertEqual(stripSummaryTags(text), text)
    }

    /// 复述锚点协议被识别为「像系统提示回显」（轻量轮据此改用本地兜底）。
    func testAnchorEchoDetectedAsPromptEcho() {
        XCTAssertTrue(KernelLoop.looksLikeSystemPromptEcho(
            "###[PLAN]\nGoal:xxx\n- [▶]1. 调用工具\n\nNow:1. 调用工具"))
        XCTAssertTrue(KernelLoop.looksLikeSystemPromptEcho(
            "### [WORKING MEMORY]\nCurrent turn: 2"))
        XCTAssertFalse(KernelLoop.looksLikeSystemPromptEcho("你好，有什么可以帮你？"))
    }
}
