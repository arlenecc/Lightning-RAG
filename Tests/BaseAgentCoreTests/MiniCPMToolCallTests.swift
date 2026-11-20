import XCTest
import Foundation
@testable import BaseAgentCore

/// MiniCPM5 工具调用方言回归：模板要求模型输出裸
/// `<function name="..."><param name="...">v</param></function>`（无 <tool_call> 外层），
/// 值可能包在 CDATA 里。此前 App 只认 <tool_call>/GLM4/JSON，导致 MiniCPM5 无法调用工具。
final class MiniCPMToolCallTests: XCTestCase {
    private let allowed: Set<String> = ["rag_tools", "web_scan", "get_weather"]

    func testBareFunctionWithParams() {
        let raw = "<function name=\"rag_tools\">"
            + "<param name=\"op\">search</param>"
            + "<param name=\"query\">气候变化的影响</param>"
            + "</function>"
        let calls = LocalToolCallParser.extract(raw: raw, allowedNames: allowed)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.name, "rag_tools")
        XCTAssertEqual(calls.first?.arguments["op"]?.stringValue, "search")
        XCTAssertEqual(calls.first?.arguments["query"]?.stringValue, "气候变化的影响")
    }

    func testCDATAValue() {
        let raw = "<function name=\"web_scan\">"
            + "<param name=\"url\"><![CDATA[https://example.com/a?x=1&y=2]]></param>"
            + "</function>"
        let calls = LocalToolCallParser.extract(raw: raw, allowedNames: allowed)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.arguments["url"]?.stringValue,
                       "https://example.com/a?x=1&y=2")
    }

    func testMultilineCDATAAndTypedValues() {
        let raw = "<function name=\"get_weather\">"
            + "<param name=\"city\">Beijing</param>"
            + "<param name=\"days\"><![CDATA[3]]></param>"
            + "</function>"
        let calls = LocalToolCallParser.extract(raw: raw, allowedNames: allowed)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.arguments["city"]?.stringValue, "Beijing")
        XCTAssertEqual(calls.first?.arguments["days"]?.intValue, 3)
    }

    func testUnknownToolIgnored() {
        let raw = "<function name=\"unknown_tool\"><param name=\"x\">1</param></function>"
        XCTAssertTrue(LocalToolCallParser.extract(raw: raw, allowedNames: allowed).isEmpty)
    }

    /// 既有 Qwen <parameter> 方言必须继续工作（不能被 param 兼容改动破坏）。
    func testQwenParameterStillWorks() {
        let raw = "<tool_call><function=rag_tools><parameter=op>search</parameter>"
            + "</function></tool_call>"
        let calls = LocalToolCallParser.extract(raw: raw, allowedNames: allowed)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.name, "rag_tools")
        XCTAssertEqual(calls.first?.arguments["op"]?.stringValue, "search")
    }
}
