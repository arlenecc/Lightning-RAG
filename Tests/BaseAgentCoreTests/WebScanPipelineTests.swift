import XCTest
@testable import BaseAgentCore

/// web_scan 全链路审查回归：非 HTML 数据源、数据接口摘要、输出预算、
/// 二进制拦截与无日期文章清单的垃圾链接过滤。
final class WebScanPipelineTests: XCTestCase {

    // MARK: - RSS / Atom

    func testFeedSummaryStripsCDATAAndTagPrefix() {
        let xml = """
        <?xml version="1.0"?><rss version="2.0"><channel><title>Feed</title>
        <item><title><![CDATA[Breaking news about the economy today]]></title>
        <link>https://example.com/a</link>
        <pubDate>Wed, 25 Nov 2026 10:00:00 GMT</pubDate></item>
        <item><title><![CDATA[Another story about technology markets]]></title>
        <link>https://example.com/b</link>
        <pubDate>Wed, 25 Nov 2026 09:00:00 GMT</pubDate></item>
        </channel></rss>
        """
        let summary = WebScanTool.summarizeFeed(xml) ?? ""
        XCTAssertTrue(summary.contains("Breaking news about the economy today"))
        XCTAssertTrue(summary.contains("https://example.com/a"))
        XCTAssertTrue(summary.contains("Wed, 25 Nov 2026"))
        XCTAssertFalse(summary.contains("CDATA"))
        XCTAssertFalse(summary.contains(">Breaking"))
        XCTAssertFalse(summary.contains("| >"), "字段不应带 '>' 垃圾前缀：\n\(summary)")
    }

    func testFeedSummaryHandlesAtomHref() {
        let xml = """
        <feed xmlns="http://www.w3.org/2005/Atom"><title>Atom</title>
        <entry><title>Atom entry about a new software release</title>
        <link href="https://example.com/entry/1"/>
        <updated>2026-11-25T10:00:00Z</updated></entry></feed>
        """
        let summary = WebScanTool.summarizeFeed(xml) ?? ""
        XCTAssertTrue(summary.contains("Atom entry about a new software release"))
        XCTAssertTrue(summary.contains("https://example.com/entry/1"))
    }

    // MARK: - 数据接口 / 数据源摘要

    func testHTMLDataSummaryParsesFullDocument() {
        // maxChars 是输出上限，不能被当作 HTML 解析窗口（旧实现会只解析前 16k）。
        let html = "<html><body>"
            + String(repeating: "<p>Row content line with details.</p>", count: 900)
            + "<p>TAILMARKER_SHOULD_APPEAR</p></body></html>"
        let full = ToolSupport.htmlToText(html)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let summary = WebDataCapture.summarize(
            data: Data(html.utf8), contentType: "text/html",
            label: "/api/list", maxChars: 16_000) ?? ""
        XCTAssertGreaterThan(full.count, 20_000)
        XCTAssertTrue(summary.contains("\(full.count) 字符"),
                      "摘要应基于完整解析结果：\n\(summary.prefix(200))")
        XCTAssertLessThanOrEqual(summary.count, 16_100)
    }

    func testJSONRecordsSummaryKeepsKeyFields() {
        let json = """
        {"Results":[{"title":"First item","date":"2026-11-25"},
        {"title":"Second item","date":"2026-11-25"}],"Count":2}
        """
        let summary = WebDataCapture.summarize(
            data: Data(json.utf8), contentType: "application/json",
            label: "/api/x") ?? ""
        XCTAssertTrue(summary.contains("返回结构化数据"))
        XCTAssertTrue(summary.contains("First item"))
        XCTAssertTrue(summary.contains("Second item"))
    }

    // MARK: - 非文本响应

    func testBinaryPayloadDetection() {
        XCTAssertTrue(WebScanTool.isBinaryPayload("%PDF-1.7 stream",
                                                 mimeType: "application/pdf"))
        XCTAssertTrue(WebScanTool.isBinaryPayload(
            String(repeating: "\u{FFFD}", count: 100) + "abc",
            mimeType: "application/octet-stream"))
        XCTAssertFalse(WebScanTool.isBinaryPayload(
            "<html><body><p>正文 hello world</p></body></html>", mimeType: ""))
        XCTAssertFalse(WebScanTool.isBinaryPayload(
            "{\"Results\":[{\"title\":\"ok\"}]}", mimeType: "application/json"))
    }

    // MARK: - 输出预算

    func testComposeOutputKeepsTailForSingleLongSection() {
        let section = String(repeating: "A", count: 16_000)
        let tail = "\n\n[静态提取正文]\n" + String(repeating: "S", count: 2_000)
        let out = WebScanTool.composeOutput(sections: [section], tail: tail)
        XCTAssertTrue(out.hasSuffix(tail))
        XCTAssertTrue(out.contains(section))
    }

    func testComposeOutputKeepsTailAndLaterSections() {
        let data = String(repeating: "A", count: 16_000)
        let rendered = String(repeating: "B", count: 12_000)
        let tail = "\n\n[静态提取正文]\n" + String(repeating: "S", count: 2_000)
        let out = WebScanTool.composeOutput(sections: [data, rendered], tail: tail)
        XCTAssertTrue(out.hasSuffix(tail), "静态正文与提示不能被大段挤掉")
        XCTAssertTrue(out.contains("A"))
        XCTAssertTrue(out.contains("B"), "后续渲染段应至少保留一部分")
        XCTAssertTrue(out.contains("该段已截断"))
    }

    func testComposeOutputNeverExceedsTailOnlyLimit() {
        let out = WebScanTool.composeOutput(
            sections: [String(repeating: "A", count: 30_000)],
            tail: String(repeating: "T", count: 25_000), limit: 20_000)
        XCTAssertLessThanOrEqual(out.count, 20_000)
    }

    // MARK: - 无日期文章清单的垃圾链接过滤

    func testArticleListSkipsNavigationLinks() {
        let times = String(repeating: "<time datetime=\"2026-11-25T10:00:00Z\"></time>",
                           count: 3)
        let nav = ["#content", "/psf/annual-report/2024/", "/community/sigs/",
                   "/events/python-user-group/", "/about/gettingstarted/"]
            .enumerated().map { index, path in
                "<a href=\"\(path)\"><span class=\"headline\">Navigation entry number \(index) for site menu</span></a>"
            }.joined()
        let articles = (1...5).map { index in
            "<a href=\"https://blog.example.com/2026/11/story-\(index)\">"
            + "<span class=\"headline\">A current article headline number \(index) about software releases</span></a>"
        }.joined()
        let html = "<html><body>\(times)\(nav)\(articles)</body></html>"
        let summary = ToolSupport.newsLinkSummary(
            html: html, baseURL: URL(string: "https://example.com/")!) ?? ""
        XCTAssertTrue(summary.contains("页面文章链接"))
        XCTAssertTrue(summary.contains("A current article headline number 1"))
        XCTAssertFalse(summary.contains("Navigation entry"),
                       "导航项不能混进文章清单：\n\(summary)")
    }
}
