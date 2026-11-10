import Foundation
import CoreFoundation

// MARK: - 网页 URL 安全策略

/// 所有「把 URL 交给网络层」的路径都必须走这里：
/// `web_scan` / `web_render` 首请求、WKWebView 主文档与 iframe 页内导航、
/// `webexec_js` 交给浏览器桥的 `url`。
///
/// 只校验入口远远不够：页面内的 JS 重定向、iframe、表单提交会绕开工具层。
/// 策略与 `WebRedirectGuard` 同源（协议白名单 + 解析后的私网/回环黑名单）。
enum WebURLPolicy {
    static let loopbackHosts: Set<String> = ["localhost", "127.0.0.1", "::1"]

    /// 返回拒绝原因；nil 表示允许。
    static func rejectionReason(for url: URL,
                                allowLoopbackForTesting: Bool = false) -> String? {
        guard let scheme = url.scheme?.lowercased(),
              scheme == "http" || scheme == "https" else {
            return "Only http/https URLs are supported, got: \(url.scheme ?? "none")"
        }
        guard let host = url.host, !host.isEmpty else {
            return "Invalid URL: missing host"
        }
        guard WebRedirectGuard.isBlockedHost(host) else { return nil }
        if allowLoopbackForTesting, loopbackHosts.contains(host.lowercased()) { return nil }
        return "Blocked: \(host) resolves to a private/loopback address (SSRF protection)."
    }

    /// 把用户/模型常写的裸域名规范化为可请求的 https URL。
    ///
    /// 事故背景（2026-11-09 cnn.com）：模型把 `url` 填成 `cnn.com`，工具因缺
    /// scheme 直接 `.fail`，内核又按「同参数失败」拦截后续重试，用户看到
    /// 连续 3 次失败、任务什么都没做就结束。补齐 scheme 是工具层该做的事，
    /// 不能指望小模型每次都记得写 `https://`。
    /// - 已有 http/https：原样返回（host 必须可解析）；
    /// - 明确的其它 scheme（file/javascript/ftp…）：返回 nil，不放宽 SSRF；
    /// - 裸域名（可带端口、路径、查询）：补 `https://`。
    static func normalizedURLString(_ raw: String) -> String? {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        if text.range(of: "^[a-z][a-z0-9+-]*:",
                      options: [.regularExpression, .caseInsensitive]) != nil {
            let scheme = String(text.split(separator: ":", maxSplits: 1).first ?? "")
                .lowercased()
            guard scheme == "http" || scheme == "https",
                  URL(string: text)?.host?.isEmpty == false else { return nil }
            return text
        }
        let domain = "^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\\.)+[a-z]{2,}"
            + "(?::[0-9]{1,5})?(?:[/?#][^\\s]*)?$"
        guard text.range(of: domain,
                         options: [.regularExpression, .caseInsensitive]) != nil else {
            return nil
        }
        return "https://" + text
    }

    static func normalizedURL(_ raw: String) -> URL? {
        guard let text = normalizedURLString(raw) else { return nil }
        return URL(string: text)
    }

    /// 字符串入口：先规范化裸域名，再做协议/SSRF 校验。
    static func validate(_ raw: String,
                         allowLoopbackForTesting: Bool = false) -> String? {
        guard let url = normalizedURL(raw) else { return "Invalid URL: \(raw)" }
        return rejectionReason(for: url,
                               allowLoopbackForTesting: allowLoopbackForTesting)
    }
}

// MARK: - web_scan

/// `web_scan`：抓取 http/https URL 并返回可见正文（HTML 去脚本/样式后提取）。
public final class WebScanTool: AgentTool, ShutdownableTool {
    public let name = "web_scan"
    public let description =
        "Fetch a URL and return its visible text content (HTML stripped, scripts removed). "
        + "For JS-driven list pages, also captures the page's own same-origin data API "
        + "requests, replays them, and summarizes JSON records into a readable list; "
        + "same-day records are merged across pages when a read-only paging API exists. "
        + "PREFER THIS TOOL over writing scraping scripts (code_run/shell_run): it handles "
        + "JS rendering, browser headers and data-API discovery automatically."

    private let config: AgentConfig
    private let session: URLSession
    private let redirectGuard: WebRedirectGuard
    private let requestTimeout: Double = 30

    public var parameters: JSONValue {
        ToolSupport.objectSchema(
            properties: [
                "url": ToolSupport.stringProperty(
                    "URL to fetch. A bare domain such as cnn.com is accepted "
                    + "and https:// is added automatically."),
                "raw": ToolSupport.boolProperty(
                    "Return raw HTML instead of extracted text (default false)."),
                "render": ToolSupport.stringProperty(
                    "'auto' (default): when dynamic loading is detected (SPA/AJAX shell), "
                    + "automatically re-render with the built-in WebKit view, return "
                    + "the visible text, and summarize the page's own data API responses "
                    + "as a structured list; 'force': always render; 'never': skip rendering "
                    + "(data API hints from the HTML/JS are still used)."),
            ],
            required: ["url"])
    }

    public init(config: AgentConfig, callbacks: AgentCallbacks?, registry: ToolRegistry? = nil) {
        self.config = config
        _ = callbacks
        _ = registry

        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 30
        configuration.timeoutIntervalForResource = 60
        configuration.httpAdditionalHeaders = ["User-Agent": "base-agent/0.1"]
        let guardDelegate = WebRedirectGuard()
        self.redirectGuard = guardDelegate
        self.session = URLSession(configuration: configuration,
                                  delegate: guardDelegate,
                                  delegateQueue: nil)
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        do {
            let urlText = try ToolSupport.requireString(args, "url")
            // 协议白名单：防止 file:// 读取本地文件或探测内网；裸域名补 https。
            guard let url = WebURLPolicy.normalizedURL(urlText) else {
                return .fail("Only http/https URLs are supported: \(String(urlText.prefix(80)))")
            }

            // 首请求也要做 SSRF 校验：只校验重定向会让
            // http://169.254.169.254/... 这样的直连绕过防护。
            let host = url.host ?? ""
            guard !host.isEmpty, !WebRedirectGuard.isBlockedHost(host) else {
                return .fail("Blocked URL: host resolves to a private, loopback or "
                    + "link-local address: \(String(host.prefix(80)))")
            }

            let raw = ToolSupport.optionalBool(args, "raw") ?? false
            var mainRequest = URLRequest(url: url)
            mainRequest.setValue(ToolSupport.browserUserAgent,
                                 forHTTPHeaderField: "User-Agent")
            var result = try await redirectGuard.fetch(session: session,
                                                       request: mainRequest,
                                                       limit: ToolSupport.maxFetchBytes)
            // 限流/瞬时不可用重试：429/503 是最常见的「等一等就好」，按
            // Retry-After（钳到 5s）等待后重试一次；重试失败按原始状态报告。
            if let http = result.response, [429, 503].contains(http.statusCode) {
                let retryAfter = http.value(forHTTPHeaderField: "Retry-After")
                    .flatMap { Double($0.trimmingCharacters(in: .whitespaces)) }
                let wait = min(max(retryAfter ?? 1.0, 0.5), 5.0)
                AgentLog.warning("[web_scan] HTTP \(http.statusCode)，\(Int(wait))s 后重试一次")
                try await Task.sleep(nanoseconds: UInt64(wait * 1_000_000_000))
                result = try await redirectGuard.fetch(session: session,
                                                       request: mainRequest,
                                                       limit: ToolSupport.maxFetchBytes)
            }
            guard let http = result.response else {
                return .fail("Request failed: invalid HTTP response")
            }
            guard http.statusCode < 400 else {
                // 403/412 是典型的反爬/权限拦截：给出降级路径，而不是让模型
                // 拿着裸状态码原地重试（真机反复「HTTP 403」空转的教训）。
                let hint: String
                switch http.statusCode {
                case 403, 412:
                    hint = "（疑似反爬或权限拦截：建议改用 web_scan(op=\"render\") 用内置 WebKit "
                        + "渲染重试；或改抓取该站的 RSS/feed / 移动版 URL）"
                case 429, 503:
                    hint = "（限流/暂不可用：已按 Retry-After 重试仍失败，"
                        + "请稍后再试或换 URL）"
                case 404:
                    hint = "（资源不存在：检查路径/域名，或尝试该站的镜像/备用入口）"
                default:
                    hint = ""
                }
                return .fail("HTTP \(http.statusCode)\(hint)")
            }
            if (300..<400).contains(http.statusCode) {
                return .fail(
                    "Request failed: redirect limit reached or non-http(s) redirect "
                    + "blocked (HTTP \(http.statusCode))")
            }

            let truncated = result.truncated
            let body = result.data
            let encoding = responseEncoding(http)
            let bodyText = String(data: body, encoding: encoding ?? .utf8)
                ?? String(decoding: body, as: UTF8.self)
            let truncationNote = truncated
                ? "\n… [响应超过 \(ToolSupport.maxFetchBytes / (1024 * 1024)) MB，已截断]"
                : ""

            // 非 HTML 数据源直通：API 返回的 JSON / XML / 纯文本不做 HTML 提取，
            // 避免 JSON 被 SwiftSoup 当标签剥成乱码或空串。JSON 额外 pretty-print。
            let mimeType = (http.value(forHTTPHeaderField: "Content-Type") ?? "")
                .lowercased()
            if !mimeType.contains("html") {
                if mimeType.contains("json") || Self.looksLikeJSONObject(body) {
                    var jsonText = bodyText
                    if let obj = try? JSONSerialization.jsonObject(with: body),
                       let pretty = try? JSONSerialization.data(
                        withJSONObject: obj,
                        options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]) {
                        jsonText = String(decoding: pretty, as: UTF8.self)
                    }
                    let label = mimeType.contains("json") ? mimeType : "application/json(嗅探)"
                    return .ok("[JSON 数据源，\(label)，\(body.count) 字节]\n"
                               + String(jsonText.prefix(20_000)) + truncationNote)
                }
                if mimeType.contains("xml") || mimeType.contains("csv")
                    || mimeType.contains("text/plain") {
                    // RSS/Atom feed 是一等公民：解析成结构化条目列表，
                    // 而不是把整份 XML 直通给模型（item 里的 CDATA/命名空间
                    // 会占满输出窗口，标题/链接才是模型真正需要的）。
                    if let feed = Self.summarizeFeed(bodyText) {
                        return .ok("[RSS/Atom 订阅源，\(mimeType)，\(body.count) 字节]\n"
                                   + feed + truncationNote)
                    }
                    return .ok("[数据源 \(mimeType)，\(body.count) 字节]\n"
                               + String(bodyText.prefix(20_000)) + truncationNote)
                }
            }

            if raw {
                // raw 也要跳过超长 head：cnn.com 首页 raw 直出会是几千字符的
                // 字体预加载清单，模型拿到的是完全无效的内容。
                var rawText = bodyText
                if bodyText.count > 20_000,
                   let bodyRange = bodyText.range(of: "<body", options: .caseInsensitive) {
                    rawText = String(bodyText[bodyRange.lowerBound...])
                }
                return .ok(String(rawText.prefix(20_000)) + truncationNote)
            }

            var content = ToolSupport.htmlToText(bodyText,
                                                 maxChars: ToolSupport.maxParseChars)

            // iframe 嵌套页：主文档正文过短时，自动抓取同源 iframe 的内容拼进
            // 结果（很多站点把数据放在 iframe 里）；跨域 iframe 只在提示中列出。
            if content.trimmingCharacters(in: .whitespacesAndNewlines).count < 300 {
                for src in ToolSupport.extractIframeSources(html: bodyText, limit: 2) {
                    guard let frameURL = URL(string: src, relativeTo: url)?.absoluteURL,
                          frameURL.host == url.host else { continue }
                    var frameRequest = URLRequest(url: frameURL)
                    frameRequest.timeoutInterval = requestTimeout
                    frameRequest.setValue(ToolSupport.browserUserAgent,
                                          forHTTPHeaderField: "User-Agent")
                    if let frameResult = try? await redirectGuard.fetch(
                        session: session,
                        request: frameRequest,
                        limit: ToolSupport.maxFetchBytes) {
                        let frameHTML = String(data: frameResult.data,
                                               encoding: encoding ?? .utf8)
                            ?? String(decoding: frameResult.data, as: UTF8.self)
                        let frameText = ToolSupport.htmlToText(frameHTML,
                                                               maxChars: ToolSupport.maxParseChars)
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                        if !frameText.isEmpty {
                            content += "\n\n[iframe \(src) 内容]\n"
                                + frameText.prefix(8_000)
                        }
                    }
                }
            }

            // 列表页/SPA 动态加载检测：正文短或主体只是加载占位时附降级建议，
            // 避免模型把加载占位误当有效内容。
            var hint = ToolSupport.dynamicLoadingHint(html: bodyText, extracted: content)
            // 同源 JS 源码同时供提示与接口直取复用，避免重复下载。
            var externalScriptTexts: [String] = []

            // 数据源线索增强：动态页的真实接口常写在同源 JS 里（HTML 内联只有少量线索），
            // 抓取评分最高的几个脚本，把其中的 /api/... 端点合并进提示。
            if !hint.isEmpty {
                var inspectedScriptURLs: [String] = []
                for scriptURL in ToolSupport.selectScriptSourcesForInspection(
                    html: bodyText, baseURL: url, limit: 2) {
                    var scriptRequest = URLRequest(url: scriptURL)
                    scriptRequest.timeoutInterval = 5
                    scriptRequest.setValue(ToolSupport.browserUserAgent,
                                           forHTTPHeaderField: "User-Agent")
                    let scriptLimit = min(ToolSupport.maxFetchBytes, 512 * 1024)
                    guard let scriptResult = try? await redirectGuard.fetch(
                        session: session, request: scriptRequest, limit: scriptLimit) else {
                        continue
                    }
                    let scriptText = ToolSupport.decodeUTF8Prefix(scriptResult.data)
                        ?? String(decoding: scriptResult.data, as: UTF8.self)
                    guard !scriptText.isEmpty else { continue }
                    externalScriptTexts.append(scriptText)
                    inspectedScriptURLs.append(scriptURL.path)
                }
                if !externalScriptTexts.isEmpty {
                    hint = ToolSupport.dynamicLoadingHint(
                        html: bodyText,
                        extracted: content,
                        externalScriptTexts: externalScriptTexts,
                        inspectedScriptURLs: inspectedScriptURLs)
                }
            }

            let renderMode = ToolSupport.string(args, "render")
            let shouldRender: Bool
            switch renderMode {
            case "force": shouldRender = true
            case "never": shouldRender = false
            default: shouldRender = !hint.isEmpty
            }

            let staticText = content.trimmingCharacters(in: .whitespacesAndNewlines)
            var sections: [String] = []
            var capturedRequests: [WebRenderer.CapturedRequest] = []

            if shouldRender {
                do {
                    // 自动降级：用内置 WebKit 渲染一次拿真实正文，并捕获页面
                    // 自己发出的数据请求（完整查询参数由页面 JS 运行时拼好）。
                    let outcome = try await WebRenderer.shared.render(
                        url: url,
                        waitMS: 500,
                        waitSelector: nil,
                        script: nil,
                        wantHTML: false,
                        wantScreenshot: false,
                        timeout: 45,
                        waitForTextGrowthFrom: hint.isEmpty ? nil : staticText.count,
                        maxContentWait: 8)
                    capturedRequests = outcome.capturedRequests
                    let rendered = outcome.text
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    if !rendered.isEmpty {
                        // 不能只按总长度决定取舍：CSS 隐藏的导航会让渲染正文比静态
                        // 短，但动态数据其实已经落地。一律附上，交模型判断。
                        let title = rendered.count > staticText.count
                            ? "[渲染后正文（检测到动态加载，已自动用内置 WebKit 渲染提取）"
                            : "[渲染后正文（WebKit，短于静态正文，供对照）"
                        sections.append(title + "，\(rendered.count) 字符]\n"
                            + String(rendered.prefix(12_000)))
                    }
                } catch is CancellationError {
                    throw AgentError.cancelled
                } catch AgentError.cancelled {
                    throw AgentError.cancelled
                } catch {
                    // 渲染失败不拖垮主流程：仍可按线索直取数据接口。
                }
            }

            // 数据接口直取：优先页面实发 URL，其次源码里的端点线索。只做同源 GET，
            // 且只采纳「含非空记录数组」的响应，避免错误体/登录页被当成数据。
            if !hint.isEmpty || shouldRender {
                var endpointHints: [String] = []
                for source in [bodyText] + externalScriptTexts {
                    for endpoint in ToolSupport.extractAPIEndpointHints(html: source, limit: 8) {
                        if !endpointHints.contains(endpoint) { endpointHints.append(endpoint) }
                    }
                }
                let candidates = WebDataCapture.candidateURLs(
                    captured: capturedRequests,
                    endpointHints: Array(endpointHints.prefix(12)),
                    baseURL: url, limit: 8)
                var dataSections: [String] = []
                for endpoint in candidates {
                    let label = Self.endpointLabel(endpoint)
                    var sourceData: Data?
                    var text: String?
                    // WKWebView 已拿到响应体时直接复用，省一次请求。
                    if let hit = capturedRequests.first(where: {
                        URL(string: $0.url, relativeTo: url)?.absoluteURL == endpoint
                    }), !hit.body.isEmpty {
                        let data = Data(hit.body.utf8)
                        sourceData = data
                        text = WebDataCapture.summarize(
                            data: data, contentType: "application/json", label: label)
                    } else if let fetched = await fetchEndpoint(endpoint, referer: url) {
                        sourceData = fetched.data
                        text = WebDataCapture.summarize(
                            data: fetched.data, contentType: fetched.contentType,
                            label: label)
                    }
                    guard let text, let sourceData else { continue }
                    // 第一页日期一致且总数更大时，用响应自带的搜索条件在只读接口
                    // 上继续翻页，把当天记录取全；任何异常都静默回退到第一页。
                    let merged = await fetchSameDayPages(
                        firstData: sourceData, label: label,
                        candidates: candidates, firstEndpoint: endpoint, referer: url)
                    dataSections.append(merged ?? text)
                    break
                }
                sections = dataSections + sections
            }

            if !sections.isEmpty {
                // 数据段最长 16k，静态正文只留短摘要，整体仍受 20k 输出上限约束。
                let tail = "\n\n[静态提取正文]\n" + String(content.prefix(2_000)) + hint
                return .ok(String((sections.joined(separator: "\n\n") + tail).prefix(20_000)))
            }
            return .ok(String(content.prefix(20_000)) + hint)
        } catch let error as URLError {
            if error.code == .timedOut {
                return .fail(
                    "Request failed (network unreachable or timed out after "
                    + "\(Int(requestTimeout))s): URLError: \(error.localizedDescription). "
                    + "Check the URL/host and retry, or use shell_run(curl) to diagnose connectivity.")
            }
            return .fail("Request failed: URLError: \(error.localizedDescription)")
        } catch {
            return ToolSupport.failure(error)
        }
    }

    /// 直取同源数据接口：带 XHR / Referer 头，沿用同一 URLSession（保留 cookie）。
    /// 只用于 GET，不对未知端点发 POST，避免误触发写操作。
    private func fetchEndpoint(_ endpoint: URL,
                               referer: URL) async -> (data: Data, contentType: String)? {
        var request = URLRequest(url: endpoint)
        request.timeoutInterval = 15
        request.setValue(ToolSupport.browserUserAgent, forHTTPHeaderField: "User-Agent")
        request.setValue("XMLHttpRequest", forHTTPHeaderField: "X-Requested-With")
        request.setValue(referer.absoluteString, forHTTPHeaderField: "Referer")
        request.setValue("application/json, text/plain, */*", forHTTPHeaderField: "Accept")
        guard let result = try? await redirectGuard.fetch(
            session: session, request: request,
            limit: min(ToolSupport.maxFetchBytes, 2 * 1024 * 1024)),
              (result.response?.statusCode ?? 500) < 400 else { return nil }
        return (result.data,
                result.response?.value(forHTTPHeaderField: "Content-Type") ?? "")
    }

    private static func endpointLabel(_ url: URL) -> String {
        let query = url.query.map { "?" + String($0.prefix(120)) } ?? ""
        return url.path + query
    }

    /// 按「响应自带的搜索条件 + 只读搜索端点」翻页，把同一天的记录合并成一份列表。
    /// 只改页码字段，最多 5 页 / 200 条；遇到其它日期或请求失败即停止。
    private func fetchSameDayPages(firstData: Data, label: String,
                                   candidates: [URL], firstEndpoint: URL,
                                   referer: URL) async -> String? {
        guard let root = try? JSONSerialization.jsonObject(with: firstData),
              let plan = WebDataCapture.pageSpec(
                  root: root, candidates: candidates, exclude: firstEndpoint),
              let dateKey = plan.dateKey, let dateValue = plan.dateValue else { return nil }
        let firstRecords = WebDataCapture.records(in: root)
        guard let total = WebDataCapture.totalCount(in: root),
              total > firstRecords.count,
              firstRecords.allSatisfy({ Self.datePrefix($0, key: dateKey) == dateValue })
        else { return nil }

        var all = firstRecords
        var seen = Set(firstRecords.map { Self.fingerprint($0) })
        var sawOtherDate = false

        // 分页只读、互不依赖：5 页并发取回，总耗时约等于单页而不是 5×15s。
        // 仍按页码顺序合并，保留「空页 / 遇到其它日期 / 条数达上限即停」语义。
        let session = self.session
        let redirectGuard = self.redirectGuard
        let endpoint = plan.endpoint
        let pageKey = plan.pageKey
        let startPage = plan.startPage
        let baseForm = plan.form
        let pages: [Int: Data] = await withTaskGroup(of: (Int, Data?).self) { group in
            for step in 1...5 {
                var form = baseForm
                form[pageKey] = String(startPage + step)
                group.addTask {
                    let data = await Self.postForm(
                        endpoint, form: form, referer: referer,
                        session: session, redirectGuard: redirectGuard)
                    return (step, data)
                }
            }
            var out: [Int: Data] = [:]
            for await (step, data) in group {
                if let data { out[step] = data }
            }
            return out
        }
        for step in 1...5 {
            guard let data = pages[step],
                  let pageRoot = try? JSONSerialization.jsonObject(with: data) else { break }
            let rows = WebDataCapture.records(in: pageRoot)
            guard !rows.isEmpty else { break }
            if rows.contains(where: { Self.datePrefix($0, key: dateKey) != dateValue }) {
                sawOtherDate = true
            }
            let fresh = rows.filter { Self.datePrefix($0, key: dateKey) == dateValue }
                .filter { seen.insert(Self.fingerprint($0)).inserted }
            all.append(contentsOf: fresh)
            if sawOtherDate || all.count >= 200 || all.count >= total { break }
        }
        guard all.count > firstRecords.count else { return nil }
        return WebDataCapture.summary(
            records: all, total: total,
            label: label + "（已自动合并当天分页）",
            pageInfo: plan.form, allDatesCollected: sawOtherDate, maxChars: 16_000)
    }

    /// 表单 POST：仅用于被识别为「只读搜索」的同源端点。
    /// 静态化是为了让多页并发请求只捕获 Sendable 的 session / guard，
    /// 不必把非 Sendable 的 tool 实例带进 `@Sendable` 闭包。
    private static func postForm(_ endpoint: URL, form: [String: String],
                                 referer: URL, session: URLSession,
                                 redirectGuard: WebRedirectGuard) async -> Data? {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = 15
        request.setValue(ToolSupport.browserUserAgent, forHTTPHeaderField: "User-Agent")
        request.setValue("application/x-www-form-urlencoded; charset=UTF-8",
                         forHTTPHeaderField: "Content-Type")
        request.setValue("XMLHttpRequest", forHTTPHeaderField: "X-Requested-With")
        request.setValue(referer.absoluteString, forHTTPHeaderField: "Referer")
        request.httpBody = Self.formBody(form)
        guard let result = try? await redirectGuard.fetch(
            session: session, request: request,
            limit: min(ToolSupport.maxFetchBytes, 2 * 1024 * 1024)),
              (result.response?.statusCode ?? 500) < 400 else { return nil }
        return result.data
    }

    private static func formBody(_ form: [String: String]) -> Data {
        let allowed = CharacterSet.alphanumerics
            .union(CharacterSet(charactersIn: "-._*"))
        let body = form.keys.sorted().map { key in
            let encodedKey = key.addingPercentEncoding(withAllowedCharacters: allowed) ?? key
            let encodedValue = (form[key] ?? "")
                .addingPercentEncoding(withAllowedCharacters: allowed) ?? ""
            return "\(encodedKey)=\(encodedValue)"
        }.joined(separator: "&")
        return Data(body.utf8)
    }

    private static func datePrefix(_ record: [String: Any], key: String) -> String? {
        let source = (record["Document"] as? [String: Any])
            ?? (record["document"] as? [String: Any]) ?? record
        guard let value = source[key] as? String else { return nil }
        return String(value.prefix(10))
    }

    private static func fingerprint(_ record: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: record,
                                                     options: [.sortedKeys]) else {
            return String(String(describing: record).prefix(300))
        }
        return String(decoding: data.prefix(400), as: UTF8.self)
    }

    /// 读取响应头里的 charset；缺失/非法时回退 UTF-8。
    private func responseEncoding(_ response: HTTPURLResponse) -> String.Encoding? {
        guard let name = response.textEncodingName else { return nil }
        let cfEncoding = CFStringConvertIANACharSetNameToEncoding(name as CFString)
        guard cfEncoding != kCFStringEncodingInvalidId else { return nil }
        return String.Encoding(rawValue: CFStringConvertEncodingToNSStringEncoding(cfEncoding))
    }

    /// RSS 2.0 / Atom feed 解析：提取条目的标题/链接/日期为紧凑列表。
    ///
    /// feed 的价值在「条目元数据」而非原始 XML（CDATA/命名空间会占满输出），
    /// 与 JSON 分支的 summarize 管道对齐。识别 `<item>`（RSS）与 `<entry>`
    /// （Atom）两种容器；链接取 `<link>` 文本或 `href` 属性；日期取
    /// `<pubDate>`/`<updated>`/`<published>`。
    static func summarizeFeed(_ xml: String) -> String? {
        let lowered = xml.prefix(2000).lowercased()
        guard lowered.contains("<rss") || lowered.contains("<feed")
            || lowered.contains("<rdf:rdf") else { return nil }

        func extract(_ tag: String, in text: String) -> String? {
            guard let start = text.range(of: "<\(tag)", options: [.caseInsensitive]) else {
                return nil
            }
            guard let gt = text[start.lowerBound...].range(of: ">",
                                                           options: [.caseInsensitive])
            else { return nil }
            let closeTag = "</\(tag)"
            guard let close = text[gt.lowerBound...].range(of: closeTag,
                                                           options: [.caseInsensitive])
            else { return nil }
            var inner = String(text[gt.lowerBound..<close.lowerBound])
            // CDATA 包裹的标题/链接：剥掉包裹层。
            if inner.hasPrefix("<![CDATA[") {
                inner = String(inner.dropFirst(9))
                if let end = inner.range(of: "]]>") {
                    inner = String(inner[..<end.lowerBound])
                }
            }
            return inner.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // 切分条目：RSS 用 <item>，Atom 用 <entry>。
        let openTag = xml.lowercased().contains("<entry") ? "entry" : "item"
        var entries: [String] = []
        var searchRange = xml.startIndex..<xml.endIndex
        while entries.count < 40 {
            guard let open = xml.range(of: "<\(openTag)", options: [.caseInsensitive],
                                       range: searchRange) else { break }
            guard let close = xml.range(of: "</\(openTag)>", options: [.caseInsensitive],
                                        range: open.lowerBound..<xml.endIndex)
            else { break }
            let body = String(xml[open.lowerBound..<close.lowerBound])
            var line = ""
            if let title = extract("title", in: body) {
                line += title.prefix(120).replacingOccurrences(of: "\n", with: " ")
            }
            if let link = extract("link", in: body), !link.isEmpty {
                line += " | \(String(link.prefix(200)))"
            } else {
                // Atom 的 <link href="..."/> 是自闭合标签，extract 抓不到内文，
                // 回退到 href 属性正则。
                if let m = body.range(of: "href=\"([^\"]+)\"",
                                      options: [.regularExpression, .caseInsensitive]) {
                    let href = String(body[m])
                        .replacingOccurrences(of: "href=\"", with: "")
                        .replacingOccurrences(of: "\"", with: "")
                    if !href.isEmpty { line += " | \(String(href.prefix(200)))" }
                }
            }
            let date = extract("pubDate", in: body) ?? extract("updated", in: body)
                ?? extract("published", in: body)
            if let date { line += " | \(date.prefix(40))" }
            if !line.isEmpty { entries.append("\(entries.count + 1). \(line)") }
            searchRange = close.upperBound..<xml.endIndex
        }

        guard !entries.isEmpty else { return nil }
        let total = xml.components(separatedBy: "<\(openTag)").count - 1
        var output = "共 \(total) 条，列出前 \(entries.count) 条：\n" + entries.joined(separator: "\n")
        if total > entries.count {
            output += "\n… [其余 \(total - entries.count) 条未列出；用 raw=true 获取原始 XML]"
        }
        return output
    }

    /// Content-Type 缺失时按内容嗅探 JSON（顶层对象或数组）。
    private static func looksLikeJSONObject(_ data: Data) -> Bool {
        let trimmed = data.prefix(64).reduce(into: Data()) { acc, byte in
            if !(byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D) {
                acc.append(byte)
            }
        }
        guard let first = trimmed.first else { return false }
        guard first == UInt8(ascii: "{") || first == UInt8(ascii: "[") else { return false }
        return (try? JSONSerialization.jsonObject(with: data)) != nil
    }

    public func shutdown() {
        session.invalidateAndCancel()
    }
}

/// 限制重定向次数并拒绝跳到非 http(s)（防止重定向 SSRF / file:// 读取）。
private struct WebFetchResult {
    var data: Data
    var response: HTTPURLResponse?
    var truncated: Bool
}

/// 重定向守卫 + 有界下载：限制重定向次数、阻断私网跳转、限制响应体字节数。
private final class WebRedirectGuard: NSObject, URLSessionTaskDelegate,
    URLSessionDataDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var redirectCounts: [Int: Int] = [:]
    private let maxRedirects = 5

    private final class State {
        var continuation: CheckedContinuation<WebFetchResult, Error>?
        var limit: Int
        var data = Data()
        var response: HTTPURLResponse?
        var truncated = false
        init(continuation: CheckedContinuation<WebFetchResult, Error>, limit: Int) {
            self.continuation = continuation
            self.limit = limit
        }
    }

    private var states: [Int: State] = [:]

    func fetch(session: URLSession, request: URLRequest, limit: Int) async throws -> WebFetchResult {
        let task = session.dataTask(with: request)
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                lock.lock()
                states[task.taskIdentifier] = State(continuation: continuation, limit: limit)
                let alreadyCancelled = Task.isCancelled
                lock.unlock()
                if alreadyCancelled {
                    finishCancelled(task)
                    return
                }
                task.resume()
            }
        } onCancel: {
            // 用户取消时立刻中止下载，避免 continuation 一直悬到超时。
            task.cancel()
            self.finishCancelled(task)
        }
    }

    /// 取消时主动摘除状态并 resume（didComplete 之后看到状态已被移除会直接返回）。
    private func finishCancelled(_ task: URLSessionTask) {
        lock.lock()
        redirectCounts.removeValue(forKey: task.taskIdentifier)
        let state = states.removeValue(forKey: task.taskIdentifier)
        let continuation = state?.continuation
        state?.continuation = nil
        lock.unlock()
        continuation?.resume(throwing: CancellationError())
    }

    func urlSession(_ session: URLSession,
                    task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        lock.lock()
        let count = (redirectCounts[task.taskIdentifier] ?? 0) + 1
        redirectCounts[task.taskIdentifier] = count
        lock.unlock()

        guard count <= maxRedirects,
              let target = request.url,
              let scheme = target.scheme?.lowercased(),
              scheme == "http" || scheme == "https" else {
            completionHandler(nil)
            return
        }
        // 重定向目标禁止落到 loopback / 私网 / link-local（云元数据 SSRF）。
        if WebRedirectGuard.isBlockedHost(target.host ?? "") {
            completionHandler(nil)
            return
        }
        completionHandler(request)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        lock.lock()
        if let state = states[dataTask.taskIdentifier] {
            state.response = response as? HTTPURLResponse
        }
        lock.unlock()
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        lock.lock()
        guard let state = states[dataTask.taskIdentifier] else {
            lock.unlock()
            return
        }
        if state.data.count + data.count <= state.limit {
            state.data.append(data)
        } else {
            let remaining = max(0, state.limit - state.data.count)
            if remaining > 0 { state.data.append(data.prefix(remaining)) }
            state.truncated = true
        }
        let shouldCancel = state.truncated
        lock.unlock()
        if shouldCancel { dataTask.cancel() }
    }

    /// 主机解析后任一地址属于私网/回环/链路本地/保留段 → 阻断。
    static func isBlockedHost(_ host: String) -> Bool {
        let lowered = host.lowercased()
        if lowered.isEmpty || lowered == "localhost" || lowered.hasSuffix(".local")
            || lowered == "metadata.google.internal" {
            return true
        }
        // IP 字面量直接判定：不依赖 getaddrinfo（受限/离线环境也可拦截）。
        var v4 = in_addr()
        var v6 = in6_addr()
        if inet_pton(AF_INET, host, &v4) == 1 || inet_pton(AF_INET6, host, &v6) == 1 {
            return isPrivateAddress(lowered)
        }
        var hints = addrinfo(ai_flags: 0, ai_family: AF_UNSPEC, ai_socktype: SOCK_STREAM,
                             ai_protocol: 0, ai_addrlen: 0, ai_canonname: nil,
                             ai_addr: nil, ai_next: nil)
        var result: UnsafeMutablePointer<addrinfo>?
        // 解析失败交给 URLSession 报网络错误：不能把它误报成「私网地址」，
        // 但显式的 localhost / .local 已在上面被拦截。
        guard getaddrinfo(host, nil, &hints, &result) == 0, let head = result else {
            return false
        }
        defer { freeaddrinfo(head) }
        var pointer: UnsafeMutablePointer<addrinfo>? = head
        while let info = pointer {
            defer { pointer = info.pointee.ai_next }
            guard let address = info.pointee.ai_addr else { continue }
            var buffer = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            if getnameinfo(address, info.pointee.ai_addrlen, &buffer, socklen_t(buffer.count),
                           nil, 0, NI_NUMERICHOST) == 0 {
                let ip = String(cString: buffer)
                if isPrivateAddress(ip) { return true }
            }
        }
        return false
    }

    private static func isPrivateAddress(_ ip: String) -> Bool {
        if ip == "::1" || ip == "0.0.0.0" || ip == "::" { return true }
        if ip.hasPrefix("fe80:") || ip.hasPrefix("fc") || ip.hasPrefix("fd") { return true }
        if ip.hasPrefix("169.254.") || ip.hasPrefix("127.") { return true }
        if ip.hasPrefix("10.") { return true }
        if ip.hasPrefix("192.168.") { return true }
        if ip.hasPrefix("100.64.") || ip.hasPrefix("100.65.")
            || ip.hasPrefix("100.66.") || ip.hasPrefix("100.67.") { return true }
        let parts = ip.split(separator: ".")
        if parts.count == 4, let first = Int(parts[0]), let second = Int(parts[1]) {
            if first == 172, (16...31).contains(second) { return true }
            if first >= 224 { return true }
        }
        // IPv4-mapped IPv6：::ffff:127.0.0.1
        if ip.lowercased().hasPrefix("::ffff:") {
            return isPrivateAddress(String(ip.dropFirst("::ffff:".count)))
        }
        return false
    }

    func urlSession(_ session: URLSession,
                    task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        lock.lock()
        redirectCounts.removeValue(forKey: task.taskIdentifier)
        let state = states.removeValue(forKey: task.taskIdentifier)
        lock.unlock()
        guard let state, let continuation = state.continuation else { return }
        state.continuation = nil

        if state.truncated {
            continuation.resume(returning: WebFetchResult(data: state.data,
                                                         response: state.response,
                                                         truncated: true))
            return
        }
        if let error {
            continuation.resume(throwing: error)
            return
        }
        continuation.resume(returning: WebFetchResult(data: state.data,
                                                     response: state.response,
                                                     truncated: false))
    }
}

// MARK: - webexec_js

/// `webexec_js`：通过 `config.browserEndpoint` 的 HTTP 桥执行 JavaScript。
/// 未配置桥时返回明确指引，而不是一个含糊的失败。
public final class WebExecJsTool: AgentTool, ShutdownableTool {
    public let name = "webexec_js"
    public let description =
        "Control a browser via a Playwright-style bridge: navigate to a URL and "
        + "evaluate JavaScript. Requires config.browser_endpoint; otherwise returns guidance."

    private let config: AgentConfig
    private let session: URLSession
    private let requestTimeout: Double = 60

    public var parameters: JSONValue {
        ToolSupport.objectSchema(
            properties: [
                "url": ToolSupport.stringProperty(
                    "URL to navigate to (optional if only evaluating)."),
                "script": ToolSupport.stringProperty("JavaScript to evaluate in the page."),
                "action": ToolSupport.stringProperty(
                    "Optional action: 'click', 'type', 'screenshot'."),
                "selector": ToolSupport.stringProperty("CSS selector for the action."),
                "value": ToolSupport.stringProperty("Value for the 'type' action."),
            ],
            required: ["script"])
    }

    /// 外部（非回环）浏览器桥会把脚本/页面内容发到远端主机，必须人工确认；
    /// headless 下由 fail-closed 默认拒绝。本机桥（localhost:9222 等）无需打扰用户。
    public func shouldConfirm(_ args: [String: JSONValue]) -> Bool {
        let endpoint = config.browserEndpoint
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: endpoint), let host = url.host?.lowercased() else {
            return false
        }
        return !WebURLPolicy.loopbackHosts.contains(host)
    }

    public init(config: AgentConfig, callbacks: AgentCallbacks?, registry: ToolRegistry? = nil) {
        self.config = config
        _ = callbacks
        _ = registry

        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 60
        configuration.timeoutIntervalForResource = 120
        self.session = URLSession(configuration: configuration)
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        do {
            let script = try ToolSupport.requireString(args, "script")
            let endpoint = config.browserEndpoint
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !endpoint.isEmpty else {
                return .fail(
                    "Browser bridge not configured. Set config.browser_endpoint to a "
                    + "Playwright HTTP bridge (e.g. http://localhost:9222) to use webexec_js.")
            }

            // 端点必须是绝对 http(s) URL：配错成 file:// / 相对路径时在发请求前失败。
            let base = endpoint.hasSuffix("/") ? String(endpoint.dropLast()) : endpoint
            guard let bridgeURL = URL(string: base + "/exec"),
                  let scheme = bridgeURL.scheme?.lowercased(),
                  scheme == "http" || scheme == "https",
                  bridgeURL.host?.isEmpty == false else {
                return .fail("Invalid browser_endpoint: expected an absolute http(s) URL "
                    + "with a host: \(String(endpoint.prefix(120)))")
            }

            // 模型给的 url 会由浏览器桥真实导航，必须与 web_scan / web_render 同策略：
            // 否则 file:// 或云元数据地址可借桥读取本机/内网内容。
            let navigationURL = ToolSupport.string(args, "url")
            if !navigationURL.isEmpty,
               let reason = WebURLPolicy.validate(navigationURL) {
                return .fail("Blocked navigation URL: \(reason)")
            }

            let payload = JSONValue.object([
                "url": .string(navigationURL),
                "script": .string(script),
                "action": .string(ToolSupport.string(args, "action")),
                "selector": .string(ToolSupport.string(args, "selector")),
                "value": .string(ToolSupport.string(args, "value")),
            ])

            var request = URLRequest(url: bridgeURL)
            request.httpMethod = "POST"
            request.timeoutInterval = requestTimeout
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = payload.jsonString().data(using: .utf8)

            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                return .fail("Browser bridge error: invalid HTTP response")
            }
            let bodyText = String(decoding: data, as: UTF8.self)
            guard http.statusCode < 400 else {
                return .fail("HTTP \(http.statusCode): \(String(bodyText.prefix(500)))")
            }

            let contentType = http.value(forHTTPHeaderField: "Content-Type") ?? ""
            let output: String
            if contentType.lowercased().hasPrefix("application/json"),
               let parsed = JSONValue.parse(bodyText) {
                output = parsed.jsonString()
            } else if contentType.lowercased().hasPrefix("application/json") {
                output = bodyText
            } else {
                output = JSONValue.object(["result": .string(bodyText)]).jsonString()
            }
            return .ok(String(output.prefix(8_000)))
        } catch {
            return .fail("Browser bridge error: \(ToolSupport.errorText(error))")
        }
    }

    public func shutdown() {
        session.invalidateAndCancel()
    }
}

/// 内置网页渲染：WKWebView 离屏加载并执行 JavaScript，支持等待 selector、
/// 交互脚本、渲染后 HTML 与截图。webexec_js（外部浏览器桥）的零依赖补充——
/// 当 web_scan 只能拿到加载占位（SPA/AJAX 页面）时用它拿到真实渲染结果。
public final class WebRenderTool: AgentTool, ShutdownableTool {
    public let name = "web_render"
    public let description =
        "Render a URL in an embedded WebKit view (JavaScript executes like a real browser), "
        + "optionally wait for a CSS selector, run a script, and return the visible text / "
        + "rendered HTML / script result / screenshot path. Use when web_scan returns "
        + "shell-only content (SPA/AJAX pages). No browser bridge configuration needed."

    private static let totalTimeout: Double = 90

    public var parameters: JSONValue {
        ToolSupport.objectSchema(
            properties: [
                "url": ToolSupport.stringProperty("URL to render (http/https only)."),
                "wait_ms": ToolSupport.stringProperty(
                    "Extra wait in milliseconds after the page finishes loading (default 1500, max 30000)."),
                "wait_selector": ToolSupport.stringProperty(
                    "CSS selector to poll for before extracting (up to 10s). Useful for SPAs."),
                "script": ToolSupport.stringProperty(
                    "JavaScript to evaluate after render. Return a string/number "
                    + "(e.g. document.querySelector('.x').innerText)."),
                "raw_html": ToolSupport.stringProperty(
                    "'true' to return the rendered document HTML instead of visible text."),
                "screenshot": ToolSupport.stringProperty(
                    "'true' to save a PNG snapshot and return its file path."),
            ],
            required: ["url"])
    }

    public init(config: AgentConfig, callbacks: AgentCallbacks?, registry: ToolRegistry? = nil) {
        _ = config
        _ = callbacks
        _ = registry
    }

    /// URL 校验：复用共享策略（与 WKWebView 页内导航、webexec_js 同源）。
    /// 返回错误文案；nil 表示通过。internal 便于测试。
    static func validateURL(_ raw: String) -> String? {
        WebURLPolicy.validate(
            raw, allowLoopbackForTesting: WebRenderer.localhostAllowedForTesting)
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let rawURL = ToolSupport.string(args, "url")
        if let error = Self.validateURL(rawURL) {
            return .fail(error)
        }
        guard let url = WebURLPolicy.normalizedURL(rawURL) else {
            return .fail("Invalid URL")
        }

        let waitMS = max(0, min(Int(Double(ToolSupport.string(args, "wait_ms")) ?? 1500), 30_000))
        let waitSelector = ToolSupport.string(args, "wait_selector")
        let script = ToolSupport.string(args, "script")
        let wantHTML = (ToolSupport.string(args, "raw_html") == "true")
        let wantScreenshot = (ToolSupport.string(args, "screenshot") == "true")

        do {
            let outcome = try await WebRenderer.shared.render(
                url: url,
                waitMS: waitMS,
                waitSelector: waitSelector,
                script: script,
                wantHTML: wantHTML,
                wantScreenshot: wantScreenshot,
                timeout: Self.totalTimeout)

            var sections: [String] = []
            let text = outcome.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if wantHTML {
                sections.append("[渲染后 HTML]\n"
                                + String(outcome.html?.prefix(40_000)
                                         ?? "(HTML 提取失败)"))
            } else if !text.isEmpty {
                sections.append("[渲染后正文，\(text.count) 字符]\n"
                                + String(text.prefix(20_000)))
            } else {
                sections.append("(页面渲染完成但没有可见文本——可能整页由 canvas/iframe 组成，"
                                + "可尝试 raw_html 或 script 提取)")
            }
            if let scriptResult = outcome.scriptResult {
                sections.append("[脚本返回]\n" + String(scriptResult.prefix(8_000)))
            } else if !script.isEmpty {
                // 只有确实传了脚本才提示返回值为空，否则没传脚本的调用会被误导。
                sections.append("[脚本返回] (no output——脚本需 return 字符串/数字/JSON 字符串)")
            }
            if let screenshotPath = outcome.screenshotPath {
                sections.append("[截图] \(screenshotPath)")
            }
            return .ok(sections.joined(separator: "\n\n"))
        } catch is CancellationError {
            throw AgentError.cancelled
        } catch AgentError.cancelled {
            throw AgentError.cancelled
        } catch {
            return .fail("web_render failed: \(ToolSupport.errorText(error))")
        }
    }

    public func shutdown() {
        WebRenderer.resetAll()
    }
}
