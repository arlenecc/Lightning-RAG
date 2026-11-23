import XCTest
@testable import BaseAgentCore

/// web_search / web_scan 大返回（>2000 字）先提核心句压缩再进上下文。
/// web_search 的标题与 URL 必须原样保留（模型要靠 URL 调 web_scan）。
final class WebResultCoreCompressionTests: XCTestCase {
    private func searchOutput(count: Int, snippetChars: Int) -> String {
        var lines = ["[web_search] 测试查询", "并发 3 个引擎 · 去重后 \(count) 条", ""]
        for i in 1...count {
            lines.append("\(i). 网页标题 \(i) 关于元素周期表的发现历史")
            lines.append("   https://example.com/article/\(i)")
            lines.append("   " + String(repeating: "背景细节补充不影响结论。", count: snippetChars / 11))
        }
        return lines.joined(separator: "\n")
    }

    func testWebSearchCompressedOverThreshold() {
        let raw = searchOutput(count: 10, snippetChars: 600)
        XCTAssertGreaterThan(raw.count, 2_000)
        let out = KernelLoop.trimWebHistory("web_search", raw)
        XCTAssertLessThan(out.count, raw.count)
        // 全部 10 条标题与 URL 必须保留
        for i in 1...10 {
            XCTAssertTrue(out.contains("网页标题 \(i)"), "标题 \(i) 丢失")
            XCTAssertTrue(out.contains("https://example.com/article/\(i)"), "URL \(i) 丢失")
        }
        XCTAssertTrue(out.contains("核心句"))
    }

    func testWebSearchSmallUnchanged() {
        let small = "1. 标题\n   https://example.com\n   摘要。\n"
        XCTAssertEqual(KernelLoop.trimWebHistory("web_search", small), small)
    }

    func testWebScanBodyCompressedKeepsMeta() {
        var lines = ["# 页面标题", "来源: https://example.com/page", "状态: 200", "", ""]
        lines.append(String(repeating: "核心观点认为应该关注这个重要结论。", count: 200))
        let raw = lines.joined(separator: "\n")
        XCTAssertGreaterThan(raw.count, 2_000)
        let out = KernelLoop.trimWebHistory("web_scan", raw)
        XCTAssertLessThan(out.count, raw.count)
        XCTAssertTrue(out.hasPrefix("# 页面标题"), "标题元信息应保留")
        XCTAssertTrue(out.contains("https://example.com/page"), "URL 应保留")
        XCTAssertTrue(out.contains("核心句"))
    }

    func testOtherToolsUnchanged() {
        let big = String(repeating: "普通工具输出。", count: 1_000)
        XCTAssertEqual(KernelLoop.trimWebHistory("rag_tools", big), big)
        XCTAssertEqual(KernelLoop.trimWebHistory("shell_run", big), big)
    }
}
