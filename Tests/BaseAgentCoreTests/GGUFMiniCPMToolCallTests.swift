import XCTest
@testable import BaseAgentCore

/// GGUF（LlamaCppLLMService）复用 LocalToolCallParser + ContentSanitizer：
/// 验证 MiniCPM5 裸 <function>/<param> 方言在 GGUF 路径能正确解析、正文不泄漏 XML。
final class GGUFMiniCPMToolCallTests: XCTestCase {
    private let allowed: Set<String> = ["rag_tools", "get_weather"]

    /// GGUF 的 done 事件用 LocalToolCallParser.extract 解析 raw——裸 function 必须能提取。
    func testExtractBareFunctionWithSpaces() {
        let raw = """
        前面正文
        <function name="rag_tools"><param name="op">status</param></function>
        """
        let calls = LocalToolCallParser.extract(raw: raw, allowedNames: allowed)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.name, "rag_tools")
        XCTAssertEqual(calls.first?.arguments["op"]?.stringValue, "status")
    }

    /// tokenizer 吞空格的 <paramname=...> 变体（MiniCPM5 真机出现过）。
    func testExtractBareFunctionWithoutParamSpace() {
        let raw = "<function name=\"rag_tools\"><paramname=\"op\">status</param></function>"
        let calls = LocalToolCallParser.extract(raw: raw, allowedNames: allowed)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.arguments["op"]?.stringValue, "status")
    }

    /// 多个 function 块（真机 status + update_plan 连续调用）。
    func testExtractMultipleFunctions() {
        let raw = """
        <function name="rag_tools"><param name="op">status</param></function>
        <function name="get_weather"><param name="city">Beijing</param></function>
        """
        let calls = LocalToolCallParser.extract(raw: raw, allowedNames: allowed)
        XCTAssertEqual(calls.count, 2)
        XCTAssertEqual(calls.map(\.name), ["rag_tools", "get_weather"])
    }

    /// 正文净化：GGUF 与 MLX 共用 ContentSanitizer，碎块 function 不泄漏到 .content。
    func testSanitizerStripsFragmentedFunctionForGGUF() {
        var sanitizer = ContentSanitizer()
        var visible = ""
        for chunk in ["前", "<function name", "=", "\"rag_tools\">",
                      "<param name", "=", "\"op\">", "status", "</param>",
                      "</function>", "后"] {
            visible += sanitizer.feed(chunk).joined()
        }
        visible += sanitizer.flush().joined()
        XCTAssertEqual(visible, "前后")
        XCTAssertFalse(visible.contains("<function"))
    }

    /// 其它模型方言不受影响（Qwen tool_call / GLM4 arg_key 仍可解析）。
    func testOtherDialectsStillWork() {
        let qwen = "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Beijing\"}}</tool_call>"
        let qcalls = LocalToolCallParser.extract(raw: qwen, allowedNames: allowed)
        XCTAssertEqual(qcalls.first?.name, "get_weather")
        XCTAssertEqual(qcalls.first?.arguments["city"]?.stringValue, "Beijing")

        let glm = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
        let gcalls = LocalToolCallParser.extract(raw: glm, allowedNames: allowed)
        XCTAssertEqual(gcalls.first?.name, "get_weather")
        XCTAssertEqual(gcalls.first?.arguments["city"]?.stringValue, "Beijing")
    }
}
