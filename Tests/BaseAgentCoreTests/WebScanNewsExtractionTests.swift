import XCTest
@testable import BaseAgentCore

/// web_scan 新闻门户强化 + 通用性回归：
/// 只对「新闻/文章首页」启用清单与骨架裁剪，其它站点保持原行为。
final class WebScanNewsExtractionTests: XCTestCase {

    private let base = URL(string: "https://edition.cnn.com/")!

    private var today: String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.string(from: Date())
    }

    private var todayPath: String {
        today.replacingOccurrences(of: "-", with: "/")
    }

    private func headline(_ href: String, _ title: String) -> String {
        "<a href=\"\(href)\"><span class=\"container__headline-text\">\(title)</span></a>"
    }

    // MARK: - 新闻首页（正向）

    func testNewsLinkSummaryReturnsDatedHeadlines() {
        let html = """
        <html><head><style>.x{color:red}</style></head><body>
        <header><nav>World Politics Business Sports Watch Listen Sign in</nav></header>
        \(headline("/\(todayPath)/business/ai-sell-off", "Tech stock selloff deepens as AI fears outweigh Nvidia return"))
        \(headline("/\(todayPath)/economy/gdp", "US economy grew faster last quarter than initially reported"))
        \(headline("/\(todayPath)/sport/nfl-week-12", "The Thanksgiving NFL feast gets started a day early"))
        \(headline("/\(todayPath)/science/voyager", "Voyager 1 stops sending data back to Earth after glitch"))
        \(headline("/2020/01/02/travel/museum", "An older feature story about museum architecture from last year"))
        <footer>Terms of Service Privacy Policy</footer>
        </body></html>
        """
        let summary = ToolSupport.newsLinkSummary(html: html, baseURL: base)
        XCTAssertNotNil(summary)
        let text = summary ?? ""
        XCTAssertTrue(text.contains("页面新闻链接"))
        XCTAssertTrue(text.contains("Tech stock selloff"))
        XCTAssertTrue(text.contains("The Thanksgiving NFL feast"))
        XCTAssertTrue(text.contains("[\(today)]"))
        XCTAssertTrue(text.contains("（当天）"))
        XCTAssertTrue(text.contains(
            "https://edition.cnn.com/\(todayPath)/business/ai-sell-off"))
        guard let fresh = text.range(of: "Tech stock selloff"),
              let old = text.range(of: "museum architecture") else {
            return XCTFail("头条与旧新闻都应出现在清单里：\n\(text)")
        }
        XCTAssertLessThan(fresh.lowerBound, old.lowerBound)
    }

    func testNewsLinkSummaryFiltersNavigationAndScriptNoise() {
        let html = """
        <body>
        <a href="/\(todayPath)/world/video/x">
          <span class="headline">• Video 1:08 ABELARDO DE LA ESPRIELLA VIA X/Reuters</span></a>
        <a href="/\(todayPath)/us/short"><span class="headline">US</span></a>
        <a href="/\(todayPath)/us/story-b">
          <script>function imageLoadError(img){}</script>
          <span class="headline">A real headline about the economy and the latest jobs report</span></a>
        \(headline("/\(todayPath)/business/retail", "Holiday retail sales rose again this November"))
        \(headline("/\(todayPath)/health/walking", "Several short walks may help blood sugar control"))
        \(headline("/\(todayPath)/tech/robots", "A small robot tour guide is operated remotely"))
        \(headline("/\(todayPath)/weather/storm", "A strong storm is expected before the holiday travel rush"))
        </body>
        """
        let text = ToolSupport.newsLinkSummary(html: html, baseURL: base) ?? ""
        XCTAssertFalse(text.isEmpty, "4 条有效日期型标题应产出清单")
        XCTAssertFalse(text.contains("Reuters"))
        XCTAssertFalse(text.contains("imageLoadError"))
        XCTAssertFalse(text.contains("function"))
        XCTAssertTrue(text.contains("A real headline"))
    }

    func testArticleSectionForUndatedNewsPage() {
        // BBC/Guardian/博客形态：URL 不带日期，但页面有真实 <time datetime>。
        let times = String(repeating: "<time datetime=\"\(today)T10:00:00Z\"></time>",
                           count: 3)
        let links = (1...6).map { index in
            headline("/news/article-\(index)",
                     "A current news headline number \(index) about technology and markets")
        }.joined()
        let html = "<html><body>\(times)\(links)</body></html>"
        let text = ToolSupport.newsLinkSummary(html: html, baseURL: base) ?? ""
        XCTAssertTrue(text.contains("页面文章链接"), "无日期但语义明确时应给文章清单：\n\(text)")
        XCTAssertTrue(text.contains("A current news headline"))
    }

    // MARK: - 非新闻站点（不得误命中）

    func testNoNewsSectionForMarketingHomepage() {
        // apple.com 形态：一堆营销卡片，无日期 URL、无 NewsArticle、无 time。
        let cards = (1...8).map { index in
            headline("/products/device-\(index)",
                     "Stream now the newest device with an amazing new feature \(index)")
        }.joined()
        XCTAssertNil(ToolSupport.newsLinkSummary(
            html: "<html><body>\(cards)</body></html>", baseURL: base))
    }

    func testNoNewsSectionForAjaxShell() {
        // 招聘站 SPA 壳：loader + AJAX + 少量链接，不能产出「新闻链接」段，
        // 否则会覆盖原本的数据接口直取结果。
        let html = """
        <html><body><nav>Navigation and footer content.</nav>
        <div class="jobs-loader">Loading jobs please wait...</div>
        <a href="/jobs/1"><span class="headline">Senior engineer position now available</span></a>
        <script>$.ajax({url: "/api/SearchIndex/GetSearch"});</script>
        </body></html>
        """
        XCTAssertNil(ToolSupport.newsLinkSummary(html: html, baseURL: base))
    }

    func testNoNewsSectionForStaleArchive() {
        // 旧归档页：日期完整但早已过时，不能当「今天的热点」。
        let links = (1...6).map { index in
            headline("/2020/01/\(index)/archive-\(index)",
                     "An archived document page from years ago number \(index)")
        }.joined()
        XCTAssertNil(ToolSupport.newsLinkSummary(
            html: "<html><body>\(links)</body></html>", baseURL: base))
    }

    func testNewsLinkSummaryNilForPlainPage() {
        let html = "<html><body><p>A plain document with no dated links at all.</p></body></html>"
        XCTAssertNil(ToolSupport.newsLinkSummary(html: html, baseURL: base))
    }

    // MARK: - 正文提取：默认零改变，仅新闻首页裁剪骨架

    func testHtmlToTextDefaultKeepsPageChrome() {
        let html = """
        <html><body><nav>World Politics Business Health Sports</nav>
        <main><p>Main article body stays here.</p></main>
        <footer>Terms of Service Privacy Policy</footer></body></html>
        """
        let text = ToolSupport.htmlToText(html)
        XCTAssertTrue(text.contains("World Politics"))
        XCTAssertTrue(text.contains("Main article body"))
    }

    func testHtmlToTextDropsPageChromeOnlyWhenRequested() {
        let head = """
        <html><body><header><nav>US World Politics Business Health Sports</nav></header>
        <main><h2>Developing</h2>
        <p>US Army detonates tons of unstable chemicals in Iowa.</p></main>
        <footer>Terms of Service Privacy Policy Advertise With Us</footer>
        <aside>Subscribe to our newsletter</aside>
        """
        let padding = String(repeating: "<p>Filler paragraph for a large homepage.</p>",
                             count: 8_000)
        let html = head + padding + "</body></html>"
        XCTAssertGreaterThan(html.count, 300_000)
        let text = ToolSupport.htmlToText(html, dropPageChrome: true)
        XCTAssertTrue(text.contains("US Army detonates"))
        XCTAssertFalse(text.contains("Advertise With Us"))
        XCTAssertFalse(text.contains("Subscribe to our newsletter"))
        XCTAssertFalse(text.contains("Health Sports"))
    }

    func testHtmlToTextKeepsHeaderWithHeadline() {
        // github/技术文档形态：header 里带文章标题，裁剪骨架时不能把它删掉。
        let head = """
        <html><body><header><h1>Array — JavaScript reference documentation</h1>
        <nav>Guides References Community</nav></header>
        <main><p>The Array object enables storing collections of values.</p></main>
        <footer>Licensed under CC-BY-SA</footer>
        """
        let padding = String(repeating: "<p>Filler paragraph for a large document.</p>",
                             count: 8_000)
        let html = head + padding + "</body></html>"
        let text = ToolSupport.htmlToText(html, dropPageChrome: true)
        XCTAssertTrue(text.contains("Array — JavaScript reference documentation"))
        XCTAssertFalse(text.contains("Licensed under CC-BY-SA"))
    }

    // MARK: - 动态加载判定

    func testNoDynamicHintForContentRichSSRPage() {
        // 门户级 SSR：几十万字符正文 + 大量 h2/time，页面里的 loader/fetch()
        // 只是广告与懒加载，不该触发 WebKit 重渲染。
        let paragraphs = (0..<900).map { _ in
            "<p>Global markets moved on Wednesday as investors weighed the latest economic data.</p>"
        }.joined()
        let html = "<html><body><div class=\"card-loader\">Loading...</div>"
            + paragraphs
            + String(repeating: "<h2>Section headline</h2><time datetime=\"2026-11-25\"></time>",
                     count: 6)
            + "<script>fetch('/api/v2/content');</script></body></html>"
        let content = ToolSupport.htmlToText(html)
        XCTAssertGreaterThan(content.count, 20_000)
        XCTAssertEqual(ToolSupport.dynamicLoadingHint(html: html, extracted: content), "")
    }

    func testDynamicHintStillTriggeredForLongDocumentWithFewTags() {
        // 普通长文/论坛：正文长但结构化标签少，仍按原规则判定动态内容。
        let paragraphs = (0..<400).map { _ in
            "<p>Long article text about software engineering practices and tooling.</p>"
        }.joined()
        let html = "<html><body><div class=\"comments-loader\">Loading...</div>"
            + paragraphs + "<h2>Comments</h2>"
            + "<script>fetch('/api/comments');</script></body></html>"
        let content = ToolSupport.htmlToText(html)
        XCTAssertGreaterThan(content.count, 20_000)
        XCTAssertTrue(ToolSupport.dynamicLoadingHint(html: html, extracted: content)
            .contains("web_scan 提示"))
    }

    func testDynamicHintStillTriggeredForAjaxShell() {
        let html = """
        <html><body><nav>Navigation and footer content.</nav>
        <div class="jobs-loader">Loading jobs please wait...</div>
        <script>$.ajax({url: "/api/SearchIndex/GetSearch"});</script>
        </body></html>
        """
        let content = ToolSupport.htmlToText(html)
        XCTAssertTrue(ToolSupport.dynamicLoadingHint(html: html, extracted: content)
            .contains("web_scan 提示"))
    }

    func testDiscoverFeedLinksOnlyAcceptsDeclaredFeeds() {
        let html = """
        <html><head>
        <link rel="alternate" type="application/rss+xml" title="RSS" href="/rss/topstories.xml">
        <link rel="alternate" hreflang="en-gb" href="https://edition.cnn.com/">
        </head><body><a href="/about/feedback">Feedback</a></body></html>
        """
        XCTAssertEqual(ToolSupport.discoverFeedLinks(html: html, baseURL: base)
            .map(\.absoluteString),
                       ["https://edition.cnn.com/rss/topstories.xml"])
    }
}
