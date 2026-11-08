import XCTest
@testable import BaseAgentCore

/// web_scan 真实站点旁路验证：默认跳过，设置 ENABLE_WEB_NETWORK_TESTS=1 后运行。
/// 只用静态抓取与纯逻辑（WKWebView 无法在 xctest 宿主创建）。
final class WebScanLiveTests: XCTestCase {

    private let pageURL = URL(
        string: "https://network.applytoeducation.com/Applicant/AttSearchex.aspx?DirectLink=1")!

    func testApplyToEducationScriptsAndEndpoints() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["ENABLE_WEB_NETWORK_TESTS"] == "1",
            "需要外网访问，设置 ENABLE_WEB_NETWORK_TESTS=1 启用")

        let (pageData, pageResponse) = try await URLSession.shared.data(from: pageURL)
        XCTAssertEqual((pageResponse as? HTTPURLResponse)?.statusCode, 200)
        let html = String(decoding: pageData, as: UTF8.self)

        let scripts = ToolSupport.selectScriptSourcesForInspection(
            html: html, baseURL: pageURL, limit: 2)
        XCTAssertEqual(scripts.map(\.path), [
            "/jscripts/azure-search-ui-settings.min.js",
            "/jscripts/azure-search-ui.min.js",
        ], "应优先选出含数据接口的 Azure Search 脚本")

        var scriptTexts: [String] = []
        for scriptURL in scripts {
            let (data, response) = try await URLSession.shared.data(from: scriptURL)
            XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
            scriptTexts.append(String(decoding: data, as: UTF8.self))
        }

        let hint = ToolSupport.dynamicLoadingHint(
            html: html,
            extracted: ToolSupport.htmlToText(html),
            externalScriptTexts: scriptTexts,
            inspectedScriptURLs: scripts.map(\.path))
        XCTAssertFalse(hint.isEmpty, "长正文 + 加载占位也应触发动态加载提示")
        XCTAssertTrue(hint.contains("主要是导航/页脚"))
        XCTAssertTrue(hint.contains("/api/SearchIndex/GetSearch"))
        XCTAssertTrue(hint.contains("/api/SearchIndex/Search"))
        XCTAssertTrue(hint.contains("已扫描 JS"))
        let endpointLine = hint.split(separator: "\n")
            .first { $0.hasPrefix("候选数据接口：") }
        XCTAssertNotNil(endpointLine)
        XCTAssertFalse(endpointLine?.lowercased().contains("jquery") ?? true,
                       "静态资源不应进入候选接口")
    }

    /// 端到端：真实列表页应自动渲染 + 直取页内数据接口，输出结构化职位列表，
    /// 而不是只有「Loading jobs please wait...」的静态外壳。
    func testApplyToEducationWebScanReturnsStructuredJobs() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["ENABLE_WEB_NETWORK_TESTS"] == "1",
            "需要外网访问，设置 ENABLE_WEB_NETWORK_TESTS=1 启用")

        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory()
            + "webscan-live-\(UUID().uuidString)"
        let tool = WebScanTool(config: config, callbacks: nil)
        defer { tool.shutdown() }
        let result = try await tool.run(["url": .string(pageURL.absoluteString)])
        XCTAssertTrue(result.success, "web_scan 失败：\(result.error)")
        XCTAssertTrue(result.output.contains("返回结构化数据"),
                      "应直取页面数据接口：\n\(result.output.prefix(800))")
        XCTAssertTrue(result.output.contains("position_Subject"))
        XCTAssertTrue(result.output.contains("posted_On"))
        XCTAssertTrue(result.output.contains("分页参数（响应自带）"))
        XCTAssertFalse(result.output.contains("Loading jobs please wait...")
                        && !result.output.contains("结构化数据"))
    }
}
