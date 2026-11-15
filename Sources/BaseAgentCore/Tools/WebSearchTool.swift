import Foundation
import SwiftSoup

// MARK: - 搜索引擎聚合（SearXNG 精简复刻）
//
// 对应 SearXNG 的 search/processors + engines/* + results.ResultContainer：
// - 每个引擎只负责「构造请求 / 解析结果」两件事（EngineAdapter）；
// - 聚合层并发调度、随机浏览器 UA、失败隔离与重试；
// - 统一做 URL 归一化、去重合并、垃圾结果过滤、位置加权评分与排序；
// - 最后输出一份紧凑列表给模型。
//
// 没有复刻的部分（对工具场景无价值）：查询语法（!bang / :lang / 时间前缀）、
// 引擎 traits 自动抓取、持久化缓存、验证码求解（Anubis/Altcha）、metrics。

// MARK: - 基础模型

/// 引擎标识。主引擎是通用网页搜索；备用引擎是实测可用、按延迟排序的
/// 垂直/补充检索源（Wikipedia、StackExchange、包仓库、学术库等）。
///
/// 使用结构体而不是 enum：备用池有 50 个动态配置，enum 会让请求构造、
/// 状态排序和 `engines` 参数解析都变成巨大的 switch。
struct WebSearchEngine: Hashable, Sendable {

    enum Tier: String, Sendable {
        case primary
        case backup
    }

    /// 工具参数与展示用的稳定短名（例如 `360`、`wikipedia_zh`）。
    let key: String
    let displayName: String
    /// 与 SearXNG 的 weight 同义，参与位置加权评分。
    let weight: Double
    let category: String
    let tier: Tier
    /// 本机实测响应延迟（毫秒），仅用于备用池排序与自动选择。
    let latencyMs: Int

    var isPrimary: Bool { tier == .primary }
}

extension WebSearchEngine {

    static let google = Self(key: "google", displayName: "Google", weight: 1.25,
                             category: "general", tier: .primary, latencyMs: 0)
    static let bing = Self(key: "bing", displayName: "Bing", weight: 1.10,
                           category: "general", tier: .primary, latencyMs: 0)
    static let duckduckgo = Self(key: "duckduckgo", displayName: "DuckDuckGo", weight: 1.00,
                                 category: "general", tier: .primary, latencyMs: 0)
    static let brave = Self(key: "brave", displayName: "Brave", weight: 1.05,
                            category: "general", tier: .primary, latencyMs: 0)
    static let baidu = Self(key: "baidu", displayName: "百度", weight: 0.90,
                            category: "general", tier: .primary, latencyMs: 0)
    static let sogou = Self(key: "sogou", displayName: "搜狗", weight: 0.70,
                            category: "general", tier: .primary, latencyMs: 0)
    static let so360 = Self(key: "360", displayName: "360", weight: 0.70,
                            category: "general", tier: .primary, latencyMs: 0)

    /// 默认启用的 7 个通用网页搜索引擎。
    static let primary: [Self] = [
        .google, .bing, .duckduckgo, .brave, .baidu, .sogou, .so360,
    ]

    /// 50 个实测可用的备用检索源（按延迟升序）。
    static var backup: [Self] { WebSearchBackupCatalog.engines }

    /// 主引擎 + 备用引擎。
    static var all: [Self] { primary + backup }

    /// 解析 `engines` 参数里的单个 token；支持短名、展示名与常见别名。
    static func lookup(_ token: String) -> Self? {
        let value = token.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !value.isEmpty else { return nil }
        let alias: [String: String] = [
            "ddg": "duckduckgo",
            "so": "360",
            "so.com": "360",
            "wiki": "wikipedia",
            "hf": "hf_models",
            "huggingface": "hf_models",
            "google news": "google_news",
            "bing news": "bing_news",
        ]
        let key = alias[value] ?? value
        return all.first {
            $0.key == key || $0.displayName.lowercased() == value
        }
    }
}

enum WebSearchTimeRange: String, CaseIterable, Sendable {
    case day, week, month, year

    var ddgValue: String {
        switch self {
        case .day: return "d"
        case .week: return "w"
        case .month: return "m"
        case .year: return "y"
        }
    }

    var googleValue: String {
        switch self {
        case .day: return "d"
        case .week: return "w"
        case .month: return "m"
        case .year: return "y"
        }
    }

    var braveValue: String {
        switch self {
        case .day: return "pd"
        case .week: return "pw"
        case .month: return "pm"
        case .year: return "py"
        }
    }

    var bingValue: String {
        switch self {
        case .day: return "ez1"
        case .week: return "ez2"
        case .month: return "ez3"
        case .year: return "ez4"
        }
    }
}

enum WebSearchSafeSearch: String, CaseIterable, Sendable {
    case off, moderate, strict

    var googleValue: String {
        switch self {
        case .off: return "off"
        case .moderate: return "medium"
        case .strict: return "high"
        }
    }

    var ddgValue: String { self == .strict ? "1" : "-1" }
    var bingValue: String { rawValue }
    var braveValue: String { rawValue }
}

struct WebSearchOptions: Sendable {
    var query: String
    var page: Int
    var language: String
    var timeRange: WebSearchTimeRange?
    var safeSearch: WebSearchSafeSearch
}

/// 单个引擎解析出的原始结果（位置为该引擎内的 1-based 排名）。
struct WebSearchHit: Sendable {
    var title: String
    var url: String
    var snippet: String
    var position: Int
}

/// 单个引擎的执行结果（含失败原因，失败不影响其它引擎）。
struct WebEngineOutput: Sendable {
    var engine: WebSearchEngine
    var hits: [WebSearchHit]
    var error: String?
    var elapsed: Double
}

/// 去重合并后的结果：同一 URL 可被多个引擎命中，位置列表参与评分。
struct WebMergedHit: Sendable {
    var title: String
    var url: String
    var snippet: String
    var positions: [WebSearchEngine: Int]
    var score: Double = 0

    var bestPosition: Int { positions.values.min() ?? Int.max }
}

enum WebSearchError: Error, LocalizedError {
    case blocked(String)
    case http(Int)
    case parse(String)
    case network(String)

    var errorDescription: String? {
        switch self {
        case .blocked(let reason): return reason
        case .http(let code): return "HTTP \(code)"
        case .parse(let reason): return "解析失败: \(reason)"
        case .network(let reason): return "网络错误: \(reason)"
        }
    }

    static func describe(_ error: Error) -> String {
        if let searchError = error as? WebSearchError {
            return searchError.errorDescription ?? "未知错误"
        }
        if let urlError = error as? URLError {
            return "网络错误: \(urlError.localizedDescription)"
        }
        return error.localizedDescription
    }
}

// MARK: - 随机浏览器身份

enum WebSearchHeaders {

    /// 与 SearXNG 的 gen_useragent 思路一致：每次请求随机挑一个真实浏览器 UA，
    /// 降低单一指纹被反爬策略命中的概率。
    private static let userAgents: [String] = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            + "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            + "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            + "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            + "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            + "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
            + "(KHTML, like Gecko) Version/18.6 Safari/605.1.15",
    ]
    // 只使用桌面 UA：移动端 UA 会让 Bing/百度/搜狗返回不同的移动版 DOM，
    // 解析器会间歇性拿到 0 条结果，随机身份不能以牺牲结构稳定性为代价。

    static func randomUserAgent() -> String {
        userAgents.randomElement() ?? userAgents[0]
    }

    /// `auto` 时按查询是否包含 CJK 推断界面语言。
    static func normalizedLanguage(_ raw: String, query: String) -> String {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty || text.lowercased() == "auto" {
            return containsCJK(query) ? "zh-CN" : "en-US"
        }
        let value = text.replacingOccurrences(of: "_", with: "-")
        let alias: [String: String] = [
            "zh": "zh-CN", "zh-hans": "zh-CN", "zh-hant": "zh-TW",
            "en": "en-US", "ja": "ja-JP", "ko": "ko-KR", "de": "de-DE",
            "fr": "fr-FR", "es": "es-ES", "ru": "ru-RU", "pt": "pt-BR",
        ]
        if let mapped = alias[value.lowercased()] { return mapped }
        let parts = value.split(separator: "-")
        if parts.count == 2 {
            return "\(parts[0].lowercased())-\(parts[1].uppercased())"
        }
        return value
    }

    static func acceptLanguage(_ language: String) -> String {
        let primary = language.split(separator: "-").first.map(String.init) ?? language
        if primary.lowercased() == "en" { return "\(language),en;q=0.9" }
        return "\(language),\(primary);q=0.9,en;q=0.8"
    }

    static func base(userAgent: String, language: String,
                     referer: String? = nil) -> [String: String] {
        var headers: [String: String] = [
            "User-Agent": userAgent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
                + "image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": acceptLanguage(language),
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        ]
        if let referer {
            headers["Referer"] = referer
            headers["Sec-Fetch-Site"] = "same-origin"
        }
        return headers
    }

    static func apply(_ headers: [String: String], to request: inout URLRequest) {
        for (key, value) in headers {
            request.setValue(value, forHTTPHeaderField: key)
        }
    }

    private static func containsCJK(_ text: String) -> Bool {
        text.unicodeScalars.contains { scalar in
            let value = Int(scalar.value)
            return (0x4E00...0x9FFF).contains(value)
                || (0x3400...0x4DBF).contains(value)
                || (0xF900...0xFAFF).contains(value)
        }
    }
}

// MARK: - HTTP 会话

/// 有界 HTTP 会话：禁用磁盘缓存、限制重定向目标（不跟随到私网 / 非 http）。
final class WebSearchSession: @unchecked Sendable {

    private let session: URLSession

    init() {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 7
        configuration.timeoutIntervalForResource = 10
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.httpCookieAcceptPolicy = .onlyFromMainDocumentDomain
        // URLSession 会强引用 delegate，这里不需要额外持有。
        self.session = URLSession(configuration: configuration,
                                  delegate: WebSearchRedirectGuard(),
                                  delegateQueue: nil)
    }

    func send(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw WebSearchError.network("没有收到有效 HTTP 响应")
        }
        return (data, http)
    }

    func shutdown() {
        session.invalidateAndCancel()
    }
}

private final class WebSearchRedirectGuard: NSObject, URLSessionTaskDelegate {

    func urlSession(_ session: URLSession,
                    task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        guard let url = request.url,
              let scheme = url.scheme?.lowercased(),
              scheme == "http" || scheme == "https",
              WebURLPolicy.rejectionReason(for: url) == nil else {
            completionHandler(nil)
            return
        }
        completionHandler(request)
    }
}

// MARK: - URL 归一化 / 去重

enum WebSearchURL {

    /// 常见跟踪参数：仅用于去重与展示清洗，不影响原始结果判断。
    private static let trackingParams: Set<String> = [
        "gclid", "gbraid", "wbraid", "fbclid", "msclkid", "yclid",
        "mc_cid", "mc_eid", "igshid", "ref_src", "ref_url", "spm", "scm",
        "_hsenc", "_hsmi", "ref", "referrer", "trk",
    ]

    private static let adHosts: [String] = [
        "doubleclick.net", "googleadservices.com", "googlesyndication.com",
        "adservice.google.com", "adnxs.com", "taboola.com", "outbrain.com",
    ]

    static func absolute(_ raw: String, base: String) -> String {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return "" }
        if text.hasPrefix("//") { return "https:" + text }
        if let baseURL = URL(string: base),
           let url = URL(string: text, relativeTo: baseURL)?.absoluteURL {
            return url.absoluteString
        }
        return text
    }

    /// 拆掉搜索引擎自己的跳转包装，拿到真实目标 URL。
    static func unwrap(_ raw: String) -> String {
        guard let components = URLComponents(string: raw) else { return raw }
        let host = components.host?.lowercased() ?? ""
        let path = components.path

        // Google: /url?q=<target>&sa=U
        if path == "/url", host.contains("google."),
           let target = components.queryItems?.first(where: { $0.name == "q" })?.value,
           !target.isEmpty {
            return target
        }

        // Bing: /ck/a?...&u=a1<base64url>
        if path == "/ck/a", host.contains("bing.com"),
           var value = components.queryItems?.first(where: { $0.name == "u" })?.value {
            if value.hasPrefix("a1") { value = String(value.dropFirst(2)) }
            if let decoded = decodeBase64URL(value), decoded.hasPrefix("http") {
                return decoded
            }
        }

        // DuckDuckGo: /l/?uddg=<target>
        if path.hasPrefix("/l/"), host.contains("duckduckgo.com"),
           let target = components.queryItems?.first(where: { $0.name == "uddg" })?.value,
           !target.isEmpty {
            return target
        }

        return raw
    }

    /// 去掉 fragment 与跟踪参数。保留参数原顺序：部分站点的查询参数有语义顺序，
    /// 只用于展示的 URL 不应被重排。
    static func clean(_ raw: String) -> String {
        guard var components = URLComponents(string: raw) else { return raw }
        components.fragment = nil
        if let items = components.queryItems, !items.isEmpty {
            let filtered = items.filter { item in
                let name = item.name.lowercased()
                return !name.hasPrefix("utm_") && !trackingParams.contains(name)
            }
            components.queryItems = filtered.isEmpty ? nil : filtered
        }
        if components.path.count > 1 && components.path.hasSuffix("/") {
            components.path.removeLast()
        }
        return components.string ?? raw
    }

    /// 去重键：忽略 http/https 差异、www. 前缀、默认端口、末尾斜杠与参数顺序。
    static func canonicalKey(_ raw: String) -> String? {
        guard var components = URLComponents(string: clean(raw)),
              let rawHost = components.host?.lowercased(), !rawHost.isEmpty else {
            return nil
        }
        var host = rawHost
        if host.hasPrefix("www.") { host.removeFirst(4) }
        var port = ""
        if let value = components.port, value != 80, value != 443 {
            port = ":\(value)"
        }
        if let items = components.queryItems, !items.isEmpty {
            components.queryItems = items.sorted {
                if $0.name != $1.name { return $0.name < $1.name }
                return ($0.value ?? "") < ($1.value ?? "")
            }
        }
        var path = components.path
        if path.count > 1 && path.hasSuffix("/") { path.removeLast() }
        let query = components.percentEncodedQuery.map { "?" + $0 } ?? ""
        return "\(host)\(port)\(path)\(query)"
    }

    /// 搜索引擎自身的检索页 / 跳转页（不可能作为结果返回给模型）。
    static func isNavigationPage(_ url: URL) -> Bool {
        guard let host = url.host?.lowercased() else { return true }
        let path = url.path.lowercased()
        if host == "google.com" || host.hasSuffix(".google.com") {
            return path == "/search" || path == "/url"
        }
        if host == "bing.com" || host.hasSuffix(".bing.com") {
            return path == "/search" || path == "/ck/a"
        }
        if host == "duckduckgo.com" || host.hasSuffix(".duckduckgo.com") {
            return path == "/html" || path == "/html/" || path.hasPrefix("/l/")
        }
        if host == "brave.com" || host.hasSuffix(".brave.com") {
            return path == "/search"
        }
        if host == "baidu.com" || host.hasSuffix(".baidu.com") {
            // /baidu.php 是百度的广告跳转，不是自然结果。
            return path == "/s" || path == "/baidu.php"
        }
        if host == "sogou.com" || host.hasSuffix(".sogou.com") {
            return path == "/web"
        }
        if host == "so.com" || host.hasSuffix(".so.com") {
            return path == "/s"
        }
        return false
    }

    static func isAdHost(_ host: String) -> Bool {
        let lowered = host.lowercased()
        return adHosts.contains { lowered == $0 || lowered.hasSuffix("." + $0) }
    }

    /// 百度的 /link?url=... 与搜狗的 /link?url=... 是服务端/JS 跳转，
    /// 需要额外解析一次才能拿到真实 URL（否则无法和其它引擎去重）。
    static func needsRedirectResolution(_ raw: String) -> Bool {
        guard let components = URLComponents(string: raw),
              let host = components.host?.lowercased() else { return false }
        guard components.path == "/link" else { return false }
        return host.hasSuffix("baidu.com") || host.hasSuffix("sogou.com")
    }

    static func decodeBase64URL(_ text: String) -> String? {
        var value = text.replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        let padding = (4 - value.count % 4) % 4
        if padding > 0 { value += String(repeating: "=", count: padding) }
        guard let data = Data(base64Encoded: value) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}

// MARK: - HTML / 表单 / 请求辅助

enum WebSearchHTML {

    static func document(_ data: Data) throws -> Document {
        guard let html = String(data: data, encoding: .utf8) else {
            throw WebSearchError.parse("响应不是 UTF-8 文本")
        }
        return try SwiftSoup.parse(html)
    }

    static func text(_ element: Element?) -> String {
        guard let element, let value = try? element.text() else { return "" }
        return clean(value)
    }

    static func text(_ elements: Elements) -> String {
        let joined = elements.compactMap { try? $0.text() }.joined(separator: " ")
        return clean(joined)
    }

    static func firstElement(_ element: Element, _ selectors: [String]) -> Element? {
        for selector in selectors {
            if let found = try? element.select(selector).first() {
                return found
            }
        }
        return nil
    }

    static func firstText(_ element: Element, _ selectors: [String]) -> String {
        for selector in selectors {
            if let found = try? element.select(selector).first() {
                let value = text(found)
                if !value.isEmpty { return value }
            }
        }
        return ""
    }

    static func attr(_ element: Element?, _ name: String) -> String {
        guard let element, let value = try? element.attr(name) else { return "" }
        return value.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func clean(_ raw: String) -> String {
        raw.replacingOccurrences(of: "\u{00A0}", with: " ")
            .replacingOccurrences(of: "\u{200B}", with: "")
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
    }

    /// 摘要里的站点级样板文案（百度译站提示等）对模型没有价值，统一剥掉。
    static func cleanSnippet(_ raw: String) -> String {
        var text = clean(raw)
        let boilerplates = [
            "查看此网页的中文翻译，请点击 翻译此页",
            "查看此网页的中文翻译,请点击 翻译此页",
        ]
        for boilerplate in boilerplates where text.hasPrefix(boilerplate) {
            text.removeFirst(boilerplate.count)
            text = clean(text)
        }
        return text
    }
}

enum WebSearchForm {
    static func body(_ pairs: [(String, String)]) -> Data {
        let allowed = CharacterSet.alphanumerics
            .union(CharacterSet(charactersIn: "-._~"))
        let encoded = pairs.map { pair -> String in
            let key = pair.0.addingPercentEncoding(withAllowedCharacters: allowed) ?? pair.0
            let value = pair.1.addingPercentEncoding(withAllowedCharacters: allowed) ?? pair.1
            return "\(key)=\(value)"
        }.joined(separator: "&")
        return Data(encoded.utf8)
    }
}

enum WebSearchRequest {
    static func make(_ url: String,
                     items: [URLQueryItem] = [],
                     headers: [String: String],
                     method: String = "GET",
                     body: Data? = nil,
                     timeout: Double = 7) -> URLRequest {
        var components = URLComponents(string: url)!
        if !items.isEmpty { components.queryItems = items }
        var request = URLRequest(url: components.url!)
        request.httpMethod = method
        request.httpBody = body
        request.timeoutInterval = timeout
        WebSearchHeaders.apply(headers, to: &request)
        return request
    }
}

// MARK: - 引擎适配器

protocol WebSearchEngineAdapter: Sendable {
    var engine: WebSearchEngine { get }
    var resolvesRedirectLinks: Bool { get }
    func request(options: WebSearchOptions, userAgent: String) -> URLRequest
    func parse(_ data: Data, options: WebSearchOptions) throws -> [WebSearchHit]
}

extension WebSearchEngineAdapter {
    var resolvesRedirectLinks: Bool { false }
}

enum WebSearchEngineRegistry {
    private static let primaryAdapters: [String: any WebSearchEngineAdapter] = [
        WebSearchEngine.google.key: GoogleSearchAdapter(),
        WebSearchEngine.bing.key: BingSearchAdapter(),
        WebSearchEngine.duckduckgo.key: DuckDuckGoSearchAdapter(),
        WebSearchEngine.brave.key: BraveSearchAdapter(),
        WebSearchEngine.baidu.key: BaiduSearchAdapter(),
        WebSearchEngine.sogou.key: SogouSearchAdapter(),
        WebSearchEngine.so360.key: So360SearchAdapter(),
    ]

    static func adapter(for engine: WebSearchEngine) -> (any WebSearchEngineAdapter)? {
        primaryAdapters[engine.key] ?? WebSearchBackupCatalog.adapter(for: engine)
    }
}

// MARK: Google（尽力而为：无 JS 时通常会被拦住，失败不阻塞其它引擎）

private struct GoogleSearchAdapter: WebSearchEngineAdapter {
    let engine: WebSearchEngine = .google

    func request(options: WebSearchOptions, userAgent: String) -> URLRequest {
        var headers = WebSearchHeaders.base(userAgent: userAgent, language: options.language)
        headers["Cookie"] = "CONSENT=YES+"
        let language = options.language.split(separator: "-").first.map(String.init) ?? "en"
        var items: [URLQueryItem] = [
            .init(name: "q", value: options.query),
            .init(name: "num", value: "20"),
            .init(name: "hl", value: language.lowercased()),
            .init(name: "safe", value: options.safeSearch.googleValue),
            .init(name: "pws", value: "0"),
            .init(name: "gbv", value: "1"),
        ]
        if options.page > 1 {
            items.append(.init(name: "start", value: String((options.page - 1) * 10)))
        }
        if let range = options.timeRange {
            items.append(.init(name: "tbs", value: "qdr:" + range.googleValue))
        }
        return WebSearchRequest.make("https://www.google.com/search",
                                     items: items, headers: headers)
    }

    func parse(_ data: Data, options: WebSearchOptions) throws -> [WebSearchHit] {
        guard let html = String(data: data, encoding: .utf8) else {
            throw WebSearchError.parse("Google 响应不是 UTF-8")
        }
        let lowered = html.lowercased()
        if lowered.contains("enablejs") || lowered.contains("/sorry/")
            || lowered.contains("unusual traffic") {
            throw WebSearchError.blocked("Google 返回 JS/人机验证页")
        }

        let document = try WebSearchHTML.document(data)
        var hits: [WebSearchHit] = []
        var seen = Set<String>()

        for block in try document.select("div.MjjYud, div.g") {
            let classes = (try? block.className()) ?? ""
            if classes.contains("ad") { continue }
            if !WebSearchHTML.attr(block, "data-text-ad").isEmpty { continue }

            let anchors = try block.select("a[href]")
            guard let anchor = anchors.first(where: {
                let href = (try? $0.attr("href")) ?? ""
                return !href.isEmpty && !href.hasPrefix("#") && !href.hasPrefix("javascript:")
            }) else { continue }

            var href = WebSearchURL.absolute(WebSearchHTML.attr(anchor, "href"),
                                             base: "https://www.google.com")
            href = WebSearchURL.unwrap(href)
            guard href.hasPrefix("http") else { continue }
            let key = WebSearchURL.canonicalKey(href) ?? href
            guard seen.insert(key).inserted else { continue }

            var title = WebSearchHTML.firstText(anchor, ["h3", "div[role=heading]"])
            if title.isEmpty { title = WebSearchHTML.text(anchor) }
            guard title.count >= 2 else { continue }

            let snippet = WebSearchHTML.firstText(block,
                                                 [".VwiC3b", ".IsZvec", "[data-sncf]", ".aCOpRe"])
            hits.append(WebSearchHit(title: title, url: href,
                                     snippet: snippet, position: hits.count + 1))
        }
        return hits
    }
}

// MARK: Bing

private struct BingSearchAdapter: WebSearchEngineAdapter {
    let engine: WebSearchEngine = .bing

    func request(options: WebSearchOptions, userAgent: String) -> URLRequest {
        let headers = WebSearchHeaders.base(userAgent: userAgent, language: options.language)
        let parts = options.language.split(separator: "-").map(String.init)
        var items: [URLQueryItem] = [
            .init(name: "q", value: options.query),
            .init(name: "count", value: "20"),
            .init(name: "first", value: String((options.page - 1) * 20 + 1)),
            .init(name: "setlang", value: (parts.first ?? "en").lowercased()),
            .init(name: "adlt", value: options.safeSearch.bingValue),
        ]
        if parts.count > 1 {
            items.append(.init(name: "mkt", value: "\(parts[0].lowercased())-\(parts[1].uppercased())"))
        }
        if let range = options.timeRange {
            items.append(.init(name: "filters", value: "ex1:\"\(range.bingValue)\""))
        }
        return WebSearchRequest.make("https://www.bing.com/search", items: items, headers: headers)
    }

    func parse(_ data: Data, options: WebSearchOptions) throws -> [WebSearchHit] {
        guard let html = String(data: data, encoding: .utf8) else {
            throw WebSearchError.parse("Bing 响应不是 UTF-8")
        }
        if html.lowercased().contains("captcha") && !html.contains("b_algo") {
            throw WebSearchError.blocked("Bing 要求人机验证")
        }

        let document = try WebSearchHTML.document(data)
        var hits: [WebSearchHit] = []

        for item in try document.select("ol#b_results > li.b_algo") {
            let classes = (try? item.className()) ?? ""
            if classes.contains("b_ad") { continue }
            guard let anchor = try item.select("h2 a").first() else { continue }

            var href = WebSearchHTML.attr(anchor, "href")
            guard !href.isEmpty else { continue }
            href = WebSearchURL.absolute(href, base: "https://www.bing.com")
            href = WebSearchURL.unwrap(href)

            let title = WebSearchHTML.text(anchor)
            guard title.count >= 2, href.hasPrefix("http") else { continue }

            let snippet = WebSearchHTML.firstText(item, [".b_caption p", ".b_snippet", "p"])
            hits.append(WebSearchHit(title: title, url: href,
                                     snippet: snippet, position: hits.count + 1))
        }
        if hits.isEmpty, html.lowercased().contains("验证") {
            throw WebSearchError.blocked("Bing 要求人机验证")
        }
        return hits
    }
}

// MARK: DuckDuckGo（no-JS html 端点，必须用表单 POST）

private struct DuckDuckGoSearchAdapter: WebSearchEngineAdapter {
    let engine: WebSearchEngine = .duckduckgo

    func request(options: WebSearchOptions, userAgent: String) -> URLRequest {
        var headers = WebSearchHeaders.base(userAgent: userAgent,
                                           language: options.language,
                                           referer: "https://html.duckduckgo.com/")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        headers["Origin"] = "https://html.duckduckgo.com"
        headers["Sec-Fetch-Site"] = "same-origin"

        var pairs: [(String, String)] = [
            ("q", options.query),
            ("b", ""),
            ("kl", Self.region(for: options.language)),
            ("kp", options.safeSearch.ddgValue),
        ]
        if let range = options.timeRange {
            pairs.append(("df", range.ddgValue))
        }
        if options.page > 1 {
            pairs.append(("s", String((options.page - 1) * 10)))
            pairs.append(("dc", String(options.page * 10 + 1)))
        }

        return WebSearchRequest.make("https://html.duckduckgo.com/html/",
                                     headers: headers,
                                     method: "POST",
                                     body: WebSearchForm.body(pairs))
    }

    func parse(_ data: Data, options: WebSearchOptions) throws -> [WebSearchHit] {
        guard let html = String(data: data, encoding: .utf8) else {
            throw WebSearchError.parse("DuckDuckGo 响应不是 UTF-8")
        }
        let lowered = html.lowercased()
        if lowered.contains("anomaly-modal") || lowered.contains("challenge-form") {
            throw WebSearchError.blocked("DuckDuckGo 返回人机挑战页")
        }

        let document = try WebSearchHTML.document(data)
        var hits: [WebSearchHit] = []

        for item in try document.select("div.result") {
            let classes = (try? item.className()) ?? ""
            if classes.contains("result--ad") { continue }
            guard let anchor = try item.select("a.result__a").first() else { continue }

            var href = WebSearchHTML.attr(anchor, "href")
            href = WebSearchURL.absolute(href, base: "https://duckduckgo.com")
            href = WebSearchURL.unwrap(href)

            let title = WebSearchHTML.text(anchor)
            guard title.count >= 2, href.hasPrefix("http") else { continue }

            let snippet = WebSearchHTML.firstText(item, [".result__snippet"])
            hits.append(WebSearchHit(title: title, url: href,
                                     snippet: snippet, position: hits.count + 1))
        }
        return hits
    }

    private static func region(for language: String) -> String {
        let lowered = language.lowercased()
        if lowered.hasPrefix("zh-tw") || lowered.hasPrefix("zh-hk") || lowered.hasPrefix("zh-hant") {
            return "tw-tzh"
        }
        if lowered.hasPrefix("zh") { return "cn-zh" }
        if lowered.hasPrefix("ja") { return "jp-jp" }
        if lowered.hasPrefix("ko") { return "kr-kr" }
        if lowered.hasPrefix("de") { return "de-de" }
        if lowered.hasPrefix("fr") { return "fr-fr" }
        if lowered.hasPrefix("es") { return "es-es" }
        if lowered.hasPrefix("ru") { return "ru-ru" }
        if lowered.hasPrefix("en") { return "us-en" }
        return "wt-wt"
    }
}

// MARK: Brave

private struct BraveSearchAdapter: WebSearchEngineAdapter {
    let engine: WebSearchEngine = .brave

    func request(options: WebSearchOptions, userAgent: String) -> URLRequest {
        var headers = WebSearchHeaders.base(userAgent: userAgent, language: options.language)
        let region = options.language.split(separator: "-").last.map { $0.lowercased() } ?? "us"
        headers["Cookie"] = "country=\(region); ui_lang=\(options.language.lowercased()); "
            + "useLocation=0; summarizer=0; safesearch=\(options.safeSearch.braveValue)"

        var items: [URLQueryItem] = [
            .init(name: "q", value: options.query),
            .init(name: "source", value: "web"),
            .init(name: "spellcheck", value: "0"),
        ]
        if options.page > 1 {
            items.append(.init(name: "offset", value: String(min(options.page - 1, 9))))
        }
        if let range = options.timeRange {
            items.append(.init(name: "tf", value: range.braveValue))
        }
        return WebSearchRequest.make("https://search.brave.com/search",
                                     items: items, headers: headers)
    }

    func parse(_ data: Data, options: WebSearchOptions) throws -> [WebSearchHit] {
        guard let html = String(data: data, encoding: .utf8) else {
            throw WebSearchError.parse("Brave 响应不是 UTF-8")
        }
        let lowered = html.lowercased()
        if lowered.contains("captcha") && !lowered.contains("data-type=\"web\"") {
            throw WebSearchError.blocked("Brave 要求人机验证")
        }

        let document = try WebSearchHTML.document(data)
        var hits: [WebSearchHit] = []

        for snippet in try document.select("div.snippet[data-type=\"web\"]") {
            let anchors = try snippet.select("a[href]")
            guard let anchor = anchors.first(where: {
                let href = (try? $0.attr("href")) ?? ""
                return href.hasPrefix("http") && !href.contains("search.brave.com")
            }) else { continue }

            var href = WebSearchHTML.attr(anchor, "href")
            href = WebSearchURL.absolute(href, base: "https://search.brave.com")
            guard href.hasPrefix("http") else { continue }

            var title = WebSearchHTML.firstText(snippet, ["div.title", ".title"])
            if title.isEmpty { title = WebSearchHTML.text(anchor) }
            // 首个 <a> 里同时包含站点名与 URL，取标题失败时至少保证非空。
            guard title.count >= 2 else { continue }

            let body = WebSearchHTML.firstText(snippet,
                                              [".generic-snippet .content", ".snippet-description",
                                               ".snippet-content", ".content"])
            hits.append(WebSearchHit(title: title, url: href,
                                     snippet: body, position: hits.count + 1))
        }
        if hits.isEmpty, lowered.contains("captcha") || html.contains("验证") {
            throw WebSearchError.blocked("Brave 要求人机验证")
        }
        return hits
    }
}

// MARK: 百度

private struct BaiduSearchAdapter: WebSearchEngineAdapter {
    let engine: WebSearchEngine = .baidu
    var resolvesRedirectLinks: Bool { true }

    func request(options: WebSearchOptions, userAgent: String) -> URLRequest {
        var headers = WebSearchHeaders.base(userAgent: userAgent, language: options.language)
        if options.language.lowercased().hasPrefix("zh") {
            headers["Accept-Language"] = "zh-CN,zh;q=0.9,en;q=0.8"
        }
        var items: [URLQueryItem] = [
            .init(name: "wd", value: options.query),
            .init(name: "rn", value: "20"),
            .init(name: "pn", value: String((options.page - 1) * 20)),
            .init(name: "ie", value: "utf-8"),
        ]
        if let range = options.timeRange {
            let seconds: Int
            switch range {
            case .day: seconds = 86_400
            case .week: seconds = 604_800
            case .month: seconds = 2_592_000
            case .year: seconds = 31_536_000
            }
            let now = Int(Date().timeIntervalSince1970)
            items.append(.init(name: "gpc", value: "stf=\(now - seconds),\(now)|stftype=1"))
        }
        return WebSearchRequest.make("https://www.baidu.com/s", items: items, headers: headers)
    }

    func parse(_ data: Data, options: WebSearchOptions) throws -> [WebSearchHit] {
        guard let html = String(data: data, encoding: .utf8) else {
            throw WebSearchError.parse("百度响应不是 UTF-8")
        }

        let document = try WebSearchHTML.document(data)
        var hits: [WebSearchHit] = []

        for container in try document.select("div.result.c-container, div.result-op.c-container") {
            let classes = (try? container.className()) ?? ""
            // EC_result 是百度的广告块；data-tuiguang 是推广标记。
            if classes.contains("ec_ppim") || classes.contains("EC_result") { continue }
            if !WebSearchHTML.attr(container, "data-tuiguang").isEmpty { continue }
            guard let anchor = try container.select("h3 a").first() else { continue }

            var href = WebSearchHTML.attr(anchor, "href")
            href = WebSearchURL.absolute(href, base: "https://www.baidu.com")
            let title = WebSearchHTML.text(anchor)
            guard title.count >= 2, href.hasPrefix("http") else { continue }
            if title.hasPrefix("广告") || href.contains("/baidu.php?") { continue }

            var snippet = WebSearchHTML.firstText(container,
                                                  [".c-abstract", ".c-span-last",
                                                   ".content-right_8Zs40", ".c-color-text"])
            if snippet.isEmpty {
                let whole = WebSearchHTML.text(container)
                snippet = whole.replacingOccurrences(of: title, with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            hits.append(WebSearchHit(title: title, url: href,
                                     snippet: snippet, position: hits.count + 1))
        }
        if hits.isEmpty, html.contains("百度安全验证") || html.contains("网络不给力") {
            throw WebSearchError.blocked("百度要求安全验证")
        }
        return hits
    }
}

// MARK: 搜狗

private struct SogouSearchAdapter: WebSearchEngineAdapter {
    let engine: WebSearchEngine = .sogou
    var resolvesRedirectLinks: Bool { true }

    func request(options: WebSearchOptions, userAgent: String) -> URLRequest {
        let headers = WebSearchHeaders.base(userAgent: userAgent, language: options.language)
        var items: [URLQueryItem] = [
            .init(name: "query", value: options.query),
            .init(name: "ie", value: "utf8"),
            .init(name: "page", value: String(options.page)),
        ]
        if let range = options.timeRange {
            items.append(.init(name: "s_from", value: "inttime_\(range.rawValue)"))
            items.append(.init(name: "tsn", value: "1"))
        }
        return WebSearchRequest.make("https://www.sogou.com/web", items: items, headers: headers)
    }

    func parse(_ data: Data, options: WebSearchOptions) throws -> [WebSearchHit] {
        guard let html = String(data: data, encoding: .utf8) else {
            throw WebSearchError.parse("搜狗响应不是 UTF-8")
        }

        let document = try WebSearchHTML.document(data)
        var hits: [WebSearchHit] = []

        for item in try document.select("div.vrwrap, div.rb") {
            let classes = (try? item.className()) ?? ""
            if classes.contains("hint") { continue }
            guard let anchor = try item.select("h3 a").first() else { continue }

            // 搜狗在结果容器上直接给出 data-url（真实地址）；只有 href 是
            // /link?url= 跳转时才采用，避免把隐藏的“推荐搜索”地址当成结果。
            var href = WebSearchHTML.attr(anchor, "href")
            if href.isEmpty || href.hasPrefix("/link?url=") {
                var direct = WebSearchHTML.attr(anchor, "data-url")
                if direct.isEmpty { direct = WebSearchHTML.attr(item, "data-url") }
                if direct.isEmpty, let holder = try? item.select("[data-url]").first() {
                    direct = WebSearchHTML.attr(holder, "data-url")
                }
                if !direct.isEmpty { href = direct }
            }
            href = WebSearchURL.absolute(href, base: "https://www.sogou.com")
            let title = WebSearchHTML.text(anchor)
            guard title.count >= 2, href.hasPrefix("http") else { continue }
            if title.hasPrefix("广告") { continue }

            let snippet = WebSearchHTML.firstText(item,
                                                  [".space-txt", ".text-layout", ".fz-mid"])
            hits.append(WebSearchHit(title: title, url: href,
                                     snippet: snippet, position: hits.count + 1))
        }
        if hits.isEmpty, html.contains("antispider") || html.contains("请输入验证码") {
            throw WebSearchError.blocked("搜狗要求验证码")
        }
        return hits
    }
}

// MARK: 360 搜索

private struct So360SearchAdapter: WebSearchEngineAdapter {
    let engine: WebSearchEngine = .so360

    func request(options: WebSearchOptions, userAgent: String) -> URLRequest {
        let headers = WebSearchHeaders.base(userAgent: userAgent, language: options.language)
        let items: [URLQueryItem] = [
            .init(name: "q", value: options.query),
            .init(name: "pn", value: String(options.page)),
        ]
        return WebSearchRequest.make("https://www.so.com/s", items: items, headers: headers)
    }

    func parse(_ data: Data, options: WebSearchOptions) throws -> [WebSearchHit] {
        guard let html = String(data: data, encoding: .utf8) else {
            throw WebSearchError.parse("360 响应不是 UTF-8")
        }

        let document = try WebSearchHTML.document(data)
        var hits: [WebSearchHit] = []

        for item in try document.select("li.res-list") {
            let classes = (try? item.className()) ?? ""
            if classes.contains("summary") || classes.contains("ad") { continue }
            if WebSearchHTML.attr(item, "data-result-type").lowercased().contains("ad") {
                continue
            }
            guard let anchor = try item.select("h3 a").first() else { continue }

            // 360 把真实地址放在 data-mdurl，避免再多一次跳转解析。
            var href = WebSearchHTML.attr(anchor, "data-mdurl")
            if href.isEmpty { href = WebSearchHTML.attr(anchor, "href") }
            href = WebSearchURL.absolute(href, base: "https://www.so.com")
            let title = WebSearchHTML.text(anchor)
            guard title.count >= 2, href.hasPrefix("http") else { continue }
            if title.hasPrefix("广告") { continue }

            let snippet = WebSearchHTML.firstText(item, [".res-desc", ".res-comm-con"])
            hits.append(WebSearchHit(title: title, url: href,
                                     snippet: snippet, position: hits.count + 1))
        }
        if hits.isEmpty, html.contains("验证") || html.lowercased().contains("captcha") {
            throw WebSearchError.blocked("360 要求安全验证")
        }
        return hits
    }
}

// MARK: - 并发聚合 / 去重 / 排序

enum WebSearchAggregator {

    static func search(options: WebSearchOptions,
                       engines: [WebSearchEngine],
                       session: WebSearchSession) async -> ([WebMergedHit], [WebEngineOutput]) {
        let reports = await runAll(options: options, engines: engines, session: session)
        return (merge(reports: reports, query: options.query), reports)
    }

    /// 并发执行一批引擎，返回结果稳定的报告列表。
    static func runAll(options: WebSearchOptions,
                       engines: [WebSearchEngine],
                       session: WebSearchSession) async -> [WebEngineOutput] {
        let outputs = await withTaskGroup(of: WebEngineOutput.self) { group in
            for engine in engines {
                guard let adapter = WebSearchEngineRegistry.adapter(for: engine) else { continue }
                group.addTask {
                    await run(adapter: adapter, options: options, session: session)
                }
            }
            var results: [WebEngineOutput] = []
            for await output in group { results.append(output) }
            return results
        }

        // 按目录顺序输出状态，保证每次结果稳定。
        let order = Dictionary(uniqueKeysWithValues:
            WebSearchEngine.all.enumerated().map { ($1, $0) })
        return outputs.sorted { (order[$0.engine] ?? 0) < (order[$1.engine] ?? 0) }
    }

    /// 把多批引擎报告（主引擎 + 备用引擎）统一去重、过滤、排序。
    static func merge(reports: [WebEngineOutput], query: String) -> [WebMergedHit] {
        var flat: [(WebSearchEngine, WebSearchHit)] = []
        for report in reports {
            for hit in report.hits { flat.append((report.engine, hit)) }
        }
        return merge(flat, query: query)
    }

    private static func run(adapter: any WebSearchEngineAdapter,
                            options: WebSearchOptions,
                            session: WebSearchSession) async -> WebEngineOutput {
        let start = Date()
        var lastError = "未知错误"
        // 引擎偶尔会对某个 UA/出口返回空包或临时限流；两次重试能显著降低
        // “整轮无结果”的概率，同时不重试真正的超时/解析错误。
        let maxAttempts = 3

        for attempt in 0..<maxAttempts {
            if Task.isCancelled { lastError = "已取消"; break }
            do {
                let request = adapter.request(options: options,
                                              userAgent: WebSearchHeaders.randomUserAgent())
                let (data, http) = try await session.send(request)

                // 429/503/403 常与临时限流或 UA 命中有关：换一个身份重试。
                if [429, 503, 403].contains(http.statusCode) && attempt < maxAttempts - 1 {
                    lastError = "HTTP \(http.statusCode)"
                    try? await Task.sleep(nanoseconds: UInt64.random(in: 300...900) * 1_000_000)
                    continue
                }
                // 某些引擎会按 UA/出口指纹直接回空包（Bing 就会这样）；换一个
                // 浏览器身份重试，别把空响应当成 0 结果。
                if data.isEmpty {
                    if attempt < maxAttempts - 1 {
                        lastError = "空响应"
                        try? await Task.sleep(nanoseconds: UInt64.random(in: 250...700) * 1_000_000)
                        continue
                    }
                    throw WebSearchError.network("空响应")
                }
                if http.statusCode == 202 && adapter.engine == .duckduckgo {
                    throw WebSearchError.blocked("DuckDuckGo 返回人机挑战页")
                }
                guard (200..<300).contains(http.statusCode) else {
                    throw WebSearchError.http(http.statusCode)
                }

                var hits = try adapter.parse(data, options: options)
                if adapter.resolvesRedirectLinks {
                    hits = await resolveRedirects(hits, session: session)
                }
                return WebEngineOutput(engine: adapter.engine, hits: hits, error: nil,
                                       elapsed: Date().timeIntervalSince(start))
            } catch is CancellationError {
                lastError = "已取消"
                break
            } catch {
                lastError = WebSearchError.describe(error)
                break
            }
        }

        return WebEngineOutput(engine: adapter.engine, hits: [], error: lastError,
                               elapsed: Date().timeIntervalSince(start))
    }

    // MARK: 跳转链接解析

    private static func resolveRedirects(_ hits: [WebSearchHit],
                                         session: WebSearchSession) async -> [WebSearchHit] {
        let targets = hits.filter { WebSearchURL.needsRedirectResolution($0.url) }.prefix(10)
        guard !targets.isEmpty else { return hits }

        let resolved: [String: String] = await withTaskGroup(of: (String, String?).self) { group in
            for hit in targets {
                group.addTask {
                    (hit.url, await resolve(hit.url, session: session))
                }
            }
            var map: [String: String] = [:]
            for await (original, target) in group {
                if let target { map[original] = target }
            }
            return map
        }
        guard !resolved.isEmpty else { return hits }

        return hits.map { hit in
            guard let target = resolved[hit.url] else { return hit }
            return WebSearchHit(title: hit.title, url: target,
                                snippet: hit.snippet, position: hit.position)
        }
    }

    private static func resolve(_ raw: String,
                                session: WebSearchSession) async -> String? {
        guard let url = URL(string: raw), let host = url.host?.lowercased() else { return nil }
        var request = URLRequest(url: url)
        // 百度是服务端 302，HEAD 即可拿到最终地址；搜狗是 200 + JS/meta 跳转。
        request.httpMethod = host.hasSuffix("baidu.com") ? "HEAD" : "GET"
        request.timeoutInterval = 4
        request.setValue(WebSearchHeaders.randomUserAgent(), forHTTPHeaderField: "User-Agent")

        guard let (data, response) = try? await session.send(request) else { return nil }

        if let final = response.url,
           final.host?.lowercased() != host,
           let scheme = final.scheme?.lowercased(), scheme == "http" || scheme == "https" {
            return final.absoluteString
        }
        // HEAD 不跟随（或目标拒绝 HEAD）时，直接读 3xx 的 Location。
        if let location = response.value(forHTTPHeaderField: "Location"),
           let target = URL(string: location, relativeTo: url)?.absoluteURL,
           target.host?.lowercased() != host,
           let scheme = target.scheme?.lowercased(), scheme == "http" || scheme == "https" {
            return target.absoluteString
        }
        guard request.httpMethod == "GET" else { return nil }

        let text = String(decoding: data.prefix(64 * 1024), as: UTF8.self)
        if let target = firstCapture(#"URL='([^']+)'"#, in: text),
           target.hasPrefix("http") {
            return target
        }
        if let target = firstCapture(#"location\.replace\(\s*["']([^"']+)["']\s*\)"#, in: text),
           target.hasPrefix("http") {
            return target
        }
        return nil
    }

    private static func firstCapture(_ pattern: String, in text: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern,
                                                   options: [.caseInsensitive]) else {
            return nil
        }
        let full = text as NSString
        guard let match = regex.firstMatch(in: text,
                                          range: NSRange(location: 0, length: full.length)),
              match.numberOfRanges > 1 else { return nil }
        return full.substring(with: match.range(at: 1))
    }

    // MARK: 去重 / 过滤 / 评分

    private static func merge(_ items: [(WebSearchEngine, WebSearchHit)],
                              query: String) -> [WebMergedHit] {
        var order: [String] = []
        var map: [String: WebMergedHit] = [:]

        for (engine, hit) in items {
            let rawURL = WebSearchURL.unwrap(WebSearchURL.clean(hit.url))
            let title = WebSearchHTML.clean(hit.title)
            guard isUsable(url: rawURL, title: title) else { continue }
            let key = WebSearchURL.canonicalKey(rawURL) ?? rawURL

            if var existing = map[key] {
                if let old = existing.positions[engine] {
                    existing.positions[engine] = min(old, hit.position)
                } else {
                    existing.positions[engine] = hit.position
                }
                if title.count > existing.title.count { existing.title = title }
                if hit.snippet.count > existing.snippet.count {
                    existing.snippet = WebSearchHTML.cleanSnippet(hit.snippet)
                }
                // 同一内容同时出现 http/https 时保留 https。
                if rawURL.lowercased().hasPrefix("https://")
                    && existing.url.lowercased().hasPrefix("http://") {
                    existing.url = rawURL
                }
                map[key] = existing
            } else {
                order.append(key)
                map[key] = WebMergedHit(title: title,
                                        url: rawURL,
                                        snippet: WebSearchHTML.cleanSnippet(hit.snippet),
                                        positions: [engine: hit.position])
            }
        }

        var merged = order.compactMap { map[$0] }
        for index in merged.indices {
            merged[index].score = score(merged[index], query: query)
        }
        merged.sort {
            if abs($0.score - $1.score) > 0.0001 { return $0.score > $1.score }
            if $0.bestPosition != $1.bestPosition { return $0.bestPosition < $1.bestPosition }
            if $0.positions.count != $1.positions.count {
                return $0.positions.count > $1.positions.count
            }
            return $0.title.count > $1.title.count
        }
        return merged
    }

    private static func isUsable(url: String, title: String) -> Bool {
        guard title.count >= 2, url.count <= 2048 else { return false }
        guard let parsed = URL(string: url),
              let scheme = parsed.scheme?.lowercased(),
              scheme == "http" || scheme == "https",
              let host = parsed.host, !host.isEmpty else { return false }
        if WebSearchURL.isNavigationPage(parsed) { return false }
        if WebSearchURL.isAdHost(host) { return false }
        // 360 的翻译 onebox 只有词典壳，没有正文，不值得占用结果位。
        if host == "fanyi.so.com" { return false }
        if title.lowercased() == "ad" || title.lowercased() == "sponsored" { return false }
        // 360 的站点链接卡片会把图标 alt + URL 当成标题；URL 本身已经单独输出。
        if title.lowercased().hasPrefix("http") { return false }
        return true
    }

    /// SearXNG 的 position 加权 + 多引擎一致性加成 + 查询词命中加成。
    private static func score(_ hit: WebMergedHit, query: String) -> Double {
        var base = 0.0
        for (engine, position) in hit.positions {
            base += engine.weight / Double(max(1, position))
        }
        let agreement = 1.0 + 0.35 * Double(max(0, hit.positions.count - 1))

        var bonus = 0.0
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if !needle.isEmpty {
            let haystack = (hit.title + " " + hit.snippet).lowercased()
            if haystack.contains(needle) { bonus += 0.20 }
            let tokens = needle.split(whereSeparator: {
                $0.isWhitespace || $0.isPunctuation
            }).map(String.init).filter { $0.count >= 2 }
            if !tokens.isEmpty {
                let matched = tokens.filter { haystack.contains($0) }.count
                bonus += 0.15 * Double(matched) / Double(tokens.count)
            }
        }
        return base * agreement + bonus
    }

    // MARK: 输出

    static func render(query: String,
                       results: [WebMergedHit],
                       total: Int,
                       reports: [WebEngineOutput],
                       elapsed: Double,
                       profile: String? = nil) -> String {
        // 引擎状态放在头部：结果再多也不会因为内核截断而丢掉失败信息。
        // 备用模式可能同时跑几十个源，逐条列出会挤占结果空间，改为汇总。
        let status: String
        if reports.count <= 10 {
            status = reports.map { report -> String in
                if let error = report.error {
                    return "\(report.engine.displayName) 0✗(\(String(error.prefix(40))))"
                }
                return "\(report.engine.displayName) \(report.hits.count)✓"
            }.joined(separator: " · ")
        } else {
            let success = reports.filter { $0.error == nil && !$0.hits.isEmpty }.count
            let failed = reports.filter { $0.error != nil }
            let empty = reports.count - success - failed.count
            var summary = "\(reports.count) 个 · 成功 \(success) · 失败 \(failed.count)"
            if empty > 0 { summary += " · 0 结果 \(empty)" }
            if !failed.isEmpty {
                let names = failed.prefix(8).map {
                    "\($0.engine.displayName):\(String(($0.error ?? "").prefix(24)))"
                }.joined(separator: ", ")
                summary += "（\(names)\(failed.count > 8 ? ", …" : "")）"
            }
            status = summary
        }

        var lines: [String] = []
        let profileTag = profile.map { " · \($0)" } ?? ""
        lines.append("[web_search] \"\(query)\"\(profileTag)")
        lines.append("并发 \(reports.count) 个引擎 · 去重后 \(total) 条 · 返回 \(results.count) 条 "
            + "· \(Int(elapsed * 1000))ms")
        lines.append("引擎状态: \(status)")
        lines.append("提示: 这里只有标题和摘要；需要正文时用 web_scan 打开下面的 URL。")
        lines.append("")

        for (index, hit) in results.enumerated() {
            var title = hit.title
            if title.count > 160 { title = String(title.prefix(160)) + "…" }
            lines.append("\(index + 1). \(title)")
            lines.append("   \(hit.url)")
            if !hit.snippet.isEmpty {
                let snippet = hit.snippet.count > 240
                    ? String(hit.snippet.prefix(240)) + "…" : hit.snippet
                lines.append("   \(snippet)")
            }
            let sources = hit.positions.sorted { $0.value < $1.value }
                .map { "\($0.key.displayName)#\($0.value)" }
                .joined(separator: " · ")
            lines.append("   来源: \(sources)")
            lines.append("")
        }

        var output = lines.joined(separator: "\n")
        // 内核默认按 8000 字符截断工具输出，这里留出余量并显式标注。
        if output.count > 7600 {
            output = String(output.prefix(7600)) + "\n… [结果已截断]"
        }
        return output
    }
}

// MARK: - web_search 工具

/// `web_search`：多引擎并发聚合搜索。
public final class WebSearchTool: AgentTool, ShutdownableTool {

    public let name = "web_search"
    public let description =
        "Search the web across multiple engines in parallel (Bing, DuckDuckGo, Brave, "
        + "Baidu, Sogou, 360, and best-effort Google) with randomized browser identities. "
        + "When those primary engines return too few results, it automatically falls back "
        + "to the fastest matching sources from a 50-source backup pool (Wikipedia, "
        + "StackExchange, package registries, academic APIs, news, media, etc.). Returns "
        + "ONE deduplicated, filtered, relevance-ranked list (title, URL, snippet, source "
        + "engines). Use this for any query needing current or external information, then "
        + "call web_scan on a result URL to read the full page. Prefer this over fetching "
        + "a search-engine results page with web_scan."

    private let session = WebSearchSession()

    public var parameters: JSONValue {
        ToolSupport.objectSchema(
            properties: [
                "query": ToolSupport.stringProperty("Search query."),
                "count": ToolSupport.numberProperty(
                    "Number of merged results to return (1-20, default 10)."),
                "page": ToolSupport.numberProperty(
                    "Result page (1-10, default 1). Paging is forwarded to each engine; "
                    + "some engines (e.g. Bing) do not support it."),
                "engines": ToolSupport.stringProperty(
                    "Comma-separated engine IDs, or 'primary' (default: 7 general engines), "
                    + "'backup' (50 fast specialized sources), 'all'. Examples: "
                    + "'bing,brave', 'wikipedia,stackoverflow,npm', 'backup'."),
                "time_range": ToolSupport.stringProperty(
                    "Optional recency filter.", enumValues: ["day", "week", "month", "year"]),
                "language": ToolSupport.stringProperty(
                    "BCP-47 language tag (e.g. zh-CN, en-US), or 'auto' (default; "
                    + "inferred from the query)."),
                "safe": ToolSupport.stringProperty(
                    "Safe-search level (applied by engines that support it).",
                    enumValues: ["off", "moderate", "strict"]),
            ],
            required: ["query"])
    }

    public init(config: AgentConfig, callbacks: AgentCallbacks?,
                registry: ToolRegistry? = nil) {
        _ = config
        _ = callbacks
        _ = registry
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let started = Date()
        do {
            let query = try ToolSupport.requireString(args, "query")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !query.isEmpty else { return .fail("query 不能为空") }
            guard query.count <= 500 else { return .fail("query 过长（最多 500 字符）") }

            let count = min(max(ToolSupport.optionalInt(args, "count") ?? 10, 1), 20)
            let page = min(max(ToolSupport.optionalInt(args, "page") ?? 1, 1), 10)
            let rawEngines = ToolSupport.string(args, "engines")
            let engines = Self.parseEngines(rawEngines)
            guard !engines.isEmpty else {
                let available = WebSearchEngine.primary.map(\.key).joined(separator: ", ")
                return .fail("未识别到有效引擎；可选 primary: \(available)；"
                    + "或 backup / all，或具体备用 ID（如 wikipedia、stackoverflow、npm）。")
            }

            let options = WebSearchOptions(
                query: query,
                page: page,
                language: WebSearchHeaders.normalizedLanguage(
                    ToolSupport.string(args, "language"), query: query),
                timeRange: WebSearchTimeRange(
                    rawValue: ToolSupport.string(args, "time_range").lowercased()),
                safeSearch: WebSearchSafeSearch(
                    rawValue: ToolSupport.string(args, "safe").lowercased()) ?? .off)

            let mode = Self.requestMode(rawEngines)
            var (merged, reports) = await WebSearchAggregator.search(
                options: options, engines: engines, session: session)
            if Task.isCancelled { throw AgentError.cancelled }
            var profile = mode.label

            // 主引擎结果过少时自动补充备用池里最匹配、最快的 8 个源。
            // 显式指定 backup/all/具体引擎时不自动追加，避免放大请求量。
            if mode == .primary && Self.needsBackup(merged: merged, reports: reports) {
                let backups = WebSearchBackupSelector.select(query: query, limit: 8)
                if !backups.isEmpty {
                    let (_, backupReports) = await WebSearchAggregator.search(
                        options: options, engines: backups, session: session)
                    if Task.isCancelled { throw AgentError.cancelled }
                    reports += backupReports
                    merged = WebSearchAggregator.merge(reports: reports, query: query)
                    profile = "primary+backup(\(backups.count))"
                }
            }

            let shown = Array(merged.prefix(count))
            guard !shown.isEmpty else {
                let detail = reports.map { report -> String in
                    if let error = report.error {
                        return "\(report.engine.displayName): \(error)"
                    }
                    return "\(report.engine.displayName): 0 条（结果被过滤或该站无结果）"
                }.joined(separator: "; ")
                return .fail("没有获取到可用搜索结果。引擎: \(detail)。"
                    + "可以改用 web_scan 直接抓取目标站点。")
            }

            return .ok(WebSearchAggregator.render(
                query: query,
                results: shown,
                total: merged.count,
                reports: reports,
                elapsed: Date().timeIntervalSince(started),
                profile: profile))
        } catch is CancellationError {
            throw AgentError.cancelled
        } catch AgentError.cancelled {
            // 上面的 `if Task.isCancelled` 主动抛出，必须原样透传，
            // 不能被下面的兜底 catch 折叠成普通失败结果。
            throw AgentError.cancelled
        } catch {
            return .fail("web_search 失败: \(WebSearchError.describe(error))")
        }
    }

    public func shutdown() {
        session.shutdown()
    }

    enum RequestMode {
        case primary
        case backup
        case all
        case explicit

        var label: String {
            switch self {
            case .primary: return "primary"
            case .backup: return "backup"
            case .all: return "all"
            case .explicit: return "custom"
            }
        }
    }

    static func requestMode(_ raw: String) -> RequestMode {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if text.isEmpty || text == "primary" || text == "default" { return .primary }
        if text == "backup" || text == "backups" { return .backup }
        if text == "all" || text == "*" { return .all }
        return .explicit
    }

    /// 主引擎结果过少时触发备用池：没有任何可用结果、结果不足 4 条，
    /// 或只有 1 个主引擎真正返回结果。
    static func needsBackup(merged: [WebMergedHit],
                            reports: [WebEngineOutput]) -> Bool {
        let successful = reports.filter { $0.error == nil && !$0.hits.isEmpty }.count
        return merged.count < 4 || successful < 2
    }

    /// 空 / `primary` → 默认 7 个主引擎；`backup` → 50 个备用源；
    /// `all` → 主引擎 + 备用源；其余按逗号/空格分隔的具体引擎 ID 解析。
    static func parseEngines(_ raw: String) -> [WebSearchEngine] {
        switch requestMode(raw) {
        case .primary: return WebSearchEngine.primary
        case .backup: return WebSearchEngine.backup
        case .all: return WebSearchEngine.all
        case .explicit: break
        }

        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        var seen = Set<WebSearchEngine>()
        var result: [WebSearchEngine] = []
        for token in text.split(whereSeparator: {
            $0 == "," || $0 == " " || $0 == "|" || $0 == ";" || $0 == "/"
        }) {
            if let engine = WebSearchEngine.lookup(String(token)),
               seen.insert(engine).inserted {
                result.append(engine)
            }
        }
        return result
    }
}
