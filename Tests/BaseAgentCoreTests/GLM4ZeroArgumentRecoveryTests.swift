import XCTest
import Foundation
@testable import BaseAgentCore

/// `.glm4` 被拒事件的恢复逻辑回归。
///
/// 上游 `GLM4ToolCallParser` 强制要求至少一个 `<arg_key>`，Spark-X2.6 的
/// 零参数工具（`<tool_call>rag_status</tool_call>`）会被误判为
/// `malformedSyntax`。应用侧只在 `rejected.format == .glm4` 时用
/// `RejectedToolCall.rawTextPreview` 恢复；这里证明恢复范围严格受限，
/// 其它格式/普通文本不会被误恢复，既有解析结果不变。
final class GLM4ZeroArgumentRecoveryTests: XCTestCase {

    private let allowed: Set<String> = ["rag_status", "get_weather", "web_scan"]

    func testRecoversZeroArgumentCall() {
        let calls = LocalToolCallParser.recoverGLM4(
            from: "<tool_call>rag_status</tool_call>",
            allowedNames: allowed)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.name, "rag_status")
        XCTAssertTrue(calls.first?.arguments.isEmpty ?? false)
        XCTAssertFalse(calls.first?.id.isEmpty ?? true)
    }

    func testRecoversCallWithArguments() {
        let raw = "<tool_call>get_weather<arg_key>city</arg_key>"
            + "<arg_value>Beijing</arg_value></tool_call>"
        let calls = LocalToolCallParser.recoverGLM4(from: raw, allowedNames: allowed)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.name, "get_weather")
        XCTAssertEqual(calls.first?.arguments["city"]?.stringValue, "Beijing")
    }

    func testRecoversMultipleCallsAndPrefixedReasoning() {
        let raw = "思考过程</think>\n<tool_call>rag_status</tool_call>"
            + "<tool_call>web_scan<arg_key>url</arg_key>"
            + "<arg_value>https://cnn.com</arg_value></tool_call>"
        let calls = LocalToolCallParser.recoverGLM4(from: raw, allowedNames: allowed)
        XCTAssertEqual(calls.count, 2)
        XCTAssertEqual(calls.map(\.name), ["rag_status", "web_scan"])
        XCTAssertEqual(calls.last?.arguments["url"]?.stringValue, "https://cnn.com")
    }

    /// 恢复入口对非 GLM47 形态必须完全无输出 —— 这是「不影响其它模型」的边界。
    func testDoesNotRecoverNonGLM47Payloads() {
        let nonGLM47 = [
            #"<tool_call>{"name":"rag_status","arguments":{}}</tool_call>"#,
            "<tool_call><function=rag_status></function></tool_call>",
            "<tool_call>rag_status 这个工具可以查询状态</tool_call>",
            "这是一段普通回答，没有工具调用。",
            "<tool_call>evil</tool_call>",
            "",
        ]
        for raw in nonGLM47 {
            XCTAssertTrue(
                LocalToolCallParser.recoverGLM4(from: raw, allowedNames: allowed).isEmpty,
                "非 GLM47 载荷不应被恢复：\(raw)")
        }
    }

    /// 既有 `extract` 的 JSON / Qwen XML / 围栏结果不因新增恢复函数而改变。
    func testExistingExtractStaysUnchanged() {
        let json = #"<tool_call>{"name":"get_weather","arguments":{"city":"Beijing"}}</tool_call>"#
        let jsonCalls = LocalToolCallParser.extract(raw: json, allowedNames: allowed)
        XCTAssertEqual(jsonCalls.count, 1)
        XCTAssertEqual(jsonCalls.first?.name, "get_weather")
        XCTAssertEqual(jsonCalls.first?.arguments["city"]?.stringValue, "Beijing")

        let xml = "<tool_call><function=get_weather>"
            + "<parameter=city>Shanghai</parameter></function></tool_call>"
        let xmlCalls = LocalToolCallParser.extract(raw: xml, allowedNames: allowed)
        XCTAssertEqual(xmlCalls.count, 1)
        XCTAssertEqual(xmlCalls.first?.arguments["city"]?.stringValue, "Shanghai")

        let fenced = "```json\n{\"name\":\"rag_status\",\"arguments\":{}}\n```"
        XCTAssertEqual(LocalToolCallParser.extract(
            raw: fenced, allowedNames: allowed).count, 1)
    }
}
