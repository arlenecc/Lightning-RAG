import XCTest
@testable import BaseAgentCore

/// ToolTagSplitter 方言兼容回归：
/// - 既有 `<tool_call>...</tool_call>` 必须字节级不变；
/// - MiniCPM5 的裸 `<function name="...">...</function>` 必须从正文剥离；
/// - 正文里普通的 `<function>`（无 name 属性）不得被误剥。
final class ToolTagSplitterCompatTests: XCTestCase {
    private func splitAll(_ chunks: [String]) -> String {
        var splitter = ToolTagSplitter()
        var out = ""
        for chunk in chunks { out += splitter.feed(chunk).joined() }
        out += splitter.flush().joined()
        return out
    }

    func testLegacyToolCallStrippedExact() {
        let raw = "前置正文<tool_call>{\"name\":\"x\"}</tool_call>后置正文"
        XCTAssertEqual(splitAll([raw]), "前置正文后置正文")
    }

    func testLegacyToolCallAcrossChunks() {
        let out = splitAll(["前缀<tool_", "call>payload</tool_ca", "ll>后缀"])
        XCTAssertEqual(out, "前缀后缀")
    }

    func testMiniCPMFunctionStripped() {
        let raw = "说明文字<function name=\"rag_tools\">"
            + "<param name=\"op\">search</param></function>结束"
        XCTAssertEqual(splitAll([raw]), "说明文字结束")
    }

    /// MiniCPM5 真实输出：tokenizer 吞空格 → <paramname=..>，整块仍须从正文剥离。
    func testMiniCPMNoSpaceParamStripped() {
        let raw = "说明<function name=\"rag_tools\">"
            + "<paramname=\"op\">status</param></function>结束"
        XCTAssertEqual(splitAll([raw]), "说明结束")
    }

    func testMiniCPMFunctionAcrossChunks() {
        let out = splitAll([
            "开头<fun",
            "ction name=\"web_scan\"><param name=\"url\">x</param></fun",
            "ction>结尾",
        ])
        XCTAssertEqual(out, "开头结尾")
    }

    /// 普通 XML 标签（无 name 属性）是正文，不得被当作工具调用吞掉。
    func testPlainFunctionTagNotStripped() {
        let raw = "示例 <function>普通内容</function> 结束"
        XCTAssertEqual(splitAll([raw]), raw)
    }

    func testMixedDialectsBothStripped() {
        let raw = "A<tool_call>x</tool_call>B"
            + "<function name=\"y\"><param name=\"k\">v</param></function>C"
        XCTAssertEqual(splitAll([raw]), "ABC")
    }
}
