import XCTest
@testable import BaseAgentCore

/// MiniCPM5 完整生成链路的端到端模拟：
/// 真实输出 = <think>思考</think>\n\n<function name=...><paramname=...>...
/// 走 ThinkSplitter → ToolTagSplitter（正文剥离） + extractTextToolCalls（工具解析）。
/// 之前只测了孤立正则，这里覆盖 splitter 与 parser 串联，确保工具能真正被触发、正文不残留。
final class MiniCPMEndToEndTests: XCTestCase {
    private let allowed: Set<String> = ["rag_tools", "web_scan"]

    /// 模拟模型按 chunk 吐出的完整生成（逐 chunk 喂给两个 splitter）。
    private func runPipeline(chunks: [String], startInThink: Bool) -> (visible: String, calls: [ToolCall]) {
        var think = ThinkSplitter(startInThink: startInThink)
        var tag = ToolTagSplitter()
        var raw = ""
        var visible = ""
        for chunk in chunks {
            raw += chunk
            for piece in think.feed(chunk) where !piece.reasoning {
                visible += tag.feed(piece.text).joined()
            }
        }
        for piece in think.flush() where !piece.reasoning {
            visible += tag.feed(piece.text).joined()
        }
        visible += tag.flush().joined()
        let calls = MLXLLMService.extractTextToolCalls(raw, allowedNames: allowed)
        return (visible, calls)
    }

    /// 你贴的真实格式：think + 裸 function + tokenizer 吞空格的 paramname。
    func testRealMiniCPMOutputTriggersTool() {
        let chunks = [
            "<think>\n用户问知识库，应该调用 rag_tools 查状态。\n</think>\n\n",
            "我来查一下：<function name=\"rag_tools\">",
            "<paramname=\"op\">status</param></function>",
        ]
        let result = runPipeline(chunks: chunks, startInThink: true)
        XCTAssertEqual(result.calls.count, 1, "必须解析出 1 个工具调用")
        XCTAssertEqual(result.calls.first?.name, "rag_tools")
        XCTAssertEqual(result.calls.first?.arguments["op"]?.stringValue, "status")
        XCTAssertTrue(result.visible.hasSuffix("我来查一下："), "正文不得残留 <function> XML")
        XCTAssertFalse(result.visible.contains("<function"), "不得残留 function 标签")
        XCTAssertFalse(result.visible.contains("<param"), "不得残留 param 标签")
    }

    /// 无 think（配置关闭思考）时同样能解析。
    func testWithoutThinkStillTriggersTool() {
        let chunks = [
            "<function name=\"rag_tools\">",
            "<param name=\"op\">status</param></function>",
        ]
        let result = runPipeline(chunks: chunks, startInThink: false)
        XCTAssertEqual(result.calls.count, 1)
        XCTAssertEqual(result.calls.first?.arguments["op"]?.stringValue, "status")
        XCTAssertTrue(result.visible.isEmpty)
    }

    /// 多参数 + CDATA（真实 URL 含 & 会被包 CDATA）。
    func testMultiParamWithCDATA() {
        let chunks = [
            "<function name=\"web_scan\"><paramname=\"url\">",
            "<![CDATA[https://x.com/?a=1&b=2]]></param>",
            "<paramname=\"op\">scan</param></function>",
        ]
        let result = runPipeline(chunks: chunks, startInThink: false)
        XCTAssertEqual(result.calls.count, 1)
        XCTAssertEqual(result.calls.first?.name, "web_scan")
        XCTAssertEqual(result.calls.first?.arguments["url"]?.stringValue,
                       "https://x.com/?a=1&b=2")
        XCTAssertEqual(result.calls.first?.arguments["op"]?.stringValue, "scan")
    }

    /// 既有 Qwen <tool_call> 格式必须在新 pipeline 下继续工作（不回归）。
    func testLegacyQwenToolCallUnaffected() {
        let chunks = [
            "正文<tool_call>{\"name\":\"rag_tools\",\"arguments\":{\"op\":\"status\"}}</tool_call>尾",
        ]
        let result = runPipeline(chunks: chunks, startInThink: false)
        XCTAssertEqual(result.calls.count, 1)
        XCTAssertEqual(result.calls.first?.name, "rag_tools")
        XCTAssertEqual(result.calls.first?.arguments["op"]?.stringValue, "status")
        XCTAssertEqual(result.visible, "正文尾")
    }
}
