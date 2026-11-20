import XCTest
@testable import BaseAgentCore

/// ContentSanitizer 是全局替换（所有 MLX 模型正文都走它），必须证明：
/// 1. MiniCPM 裸 <function> 碎块被剥；
/// 2. Qwen/GLM4/Spark 的 <tool_call> 方言不被破坏；
/// 3. 普通正文（含 < > 数学符号、URL、代码）不丢失、不延迟到消失。
final class ContentSanitizerCompatTests: XCTestCase {
    private func run(_ chunks: [String]) -> String {
        var s = ContentSanitizer()
        var out = ""
        for c in chunks { out += s.feed(c).joined() }
        out += s.flush().joined()
        return out
    }

    // MARK: - MiniCPM 裸 function（已真机验证，锁定回归）

    func testMiniCPMWholeBlockStripped() {
        XCTAssertEqual(run(["前<function name=\"rag_tools\"><param name=\"op\">status</param></function>后"]), "前后")
    }

    func testMiniCPMFragmentedChunksStripped() {
        // 真机 tokenizer 碎块序列
        XCTAssertEqual(run([
            "前", "<function name", "=", "\"rag_tools\">",
            "<param name", "=", "\"op\">", "status", "</param>",
            "</function>", "后",
        ]), "前后")
    }

    func testMiniCPMNoSpaceParamStripped() {
        XCTAssertEqual(run([
            "<function name=\"x\"><paramname=\"k\">v</param></function>",
        ]), "")
    }

    // MARK: - Qwen / GLM4 / Spark <tool_call> 方言（关键回归）

    func testQwenToolCallJSONStripped() {
        let raw = "答:<tool_call>{\"name\":\"web_scan\",\"arguments\":{\"url\":\"https://x.com\"}}</tool_call>完"
        XCTAssertEqual(run([raw]), "答:完")
    }

    func testQwenToolCallFragmentedStripped() {
        XCTAssertEqual(run([
            "答:", "<tool_call>", "{\"name\":", "\"rag_tools\"}",
            "</tool_call>", "完",
        ]), "答:完")
    }

    func testGLM4ToolCallStripped() {
        let raw = "开始<tool_call>get_time<arg_key>zone</arg_key><arg_value>UTC</arg_value></tool_call>结束"
        XCTAssertEqual(run([raw]), "开始结束")
    }

    // MARK: - 普通正文（不得丢失）

    func testPlainTextPassthrough() {
        let text = "《寄生虫星球》指出寄生虫塑造宿主行为。病毒驱动基因转移。"
        XCTAssertEqual(run([text]), text)
    }

    func testPlainTextAcrossChunks() {
        XCTAssertEqual(run(["寄生", "虫星球", "主要讲", "宿主行为"]), "寄生虫星球主要讲宿主行为")
    }

    func testMathLessThanPreserved() {
        // 正文里的 '<' 数学符号不得被当成标签吞掉
        XCTAssertEqual(run(["温度 < 20 且 > 5 时生效"]), "温度 < 20 且 > 5 时生效")
        XCTAssertEqual(run(["a < b", " 且 3 > 2"]), "a < b 且 3 > 2")
    }

    func testAngleQuotedURLPreserved() {
        XCTAssertEqual(run(["见 <https://example.com/a?x=1> 说明"]), "见 <https://example.com/a?x=1> 说明")
    }

    func testCodeWithAngleBracketsPreserved() {
        XCTAssertEqual(run(["函数 if x < 10 { ... } 结束"]), "函数 if x < 10 { ... } 结束")
    }
}
