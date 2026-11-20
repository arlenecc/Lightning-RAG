import XCTest
@testable import BaseAgentCore

/// 定位：MiniCPM5 真机输出里 <function> 为何残留在正文。
/// 喂真机实测的 raw 分块，检查 ThinkSplitter 对 </think> 的识别与 ToolTagSplitter 的剥离。
final class ThinkSplitterMiniCPMTests: XCTestCase {
    private func run(rawChunks: [String], startInThink: Bool) -> (content: String, reasoning: String) {
        var think = ThinkSplitter(startInThink: startInThink)
        var tag = ToolTagSplitter()
        var content = "", reasoning = ""
        for chunk in rawChunks {
            for piece in think.feed(chunk) {
                if piece.reasoning { reasoning += piece.text }
                else { content += tag.feed(piece.text).joined() }
            }
        }
        for piece in think.flush() {
            if piece.reasoning { reasoning += piece.text }
            else { content += tag.feed(piece.text).joined() }
        }
        content += tag.flush().joined()
        return (content, reasoning)
    }

    /// 真机 raw 形态：think 内容 + </think>\n\n + function。
    func testRealShapeStartInThinkTrue() {
        let chunks = [
            "\n用户想知道知识库内容。\n",
            "</think>\n\n",
            "<function name=\"rag_tools\"><param name=\"op\">status</param></function>",
        ]
        let out = run(rawChunks: chunks, startInThink: true)
        XCTAssertFalse(out.content.contains("<function"), "正文不得残留 function: \(out.content)")
        XCTAssertTrue(out.reasoning.contains("用户想知道"), "思考应进 reasoning")
    }

    /// 如果模板把 startInThink 判成 false（没有 <think> 前缀），行为如何？
    func testRealShapeStartInThinkFalse() {
        let chunks = [
            "<think>\n用户想知道知识库内容。\n",
            "</think>\n\n",
            "<function name=\"rag_tools\"><param name=\"op\">status</param></function>",
        ]
        let out = run(rawChunks: chunks, startInThink: false)
        XCTAssertFalse(out.content.contains("<function"), "正文不得残留 function: \(out.content)")
    }
}
