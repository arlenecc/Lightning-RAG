import Foundation

/// 动态列表页的数据接口直取与结果精简。
///
/// `WebRenderer` 捕获页面自己发出的 XHR URL（参数由运行时拼好），这里把同源
/// 数据接口挑出来交给工具层重放，再把动辄数百 KB 的 JSON 压成可读列表。
/// 只输出「含非空记录数组」的响应：错误体、全 null 壳、登录页都不会被误当数据。
enum WebDataCapture {

    /// 挑选值得直取的同源数据 URL：页面实发请求优先，其次静态源码里的端点线索。
    /// - 只放行与页面同源的 http(s) URL（SSRF 面）；
    /// - 排除静态资源后缀与明显无关的资源请求。
    static func candidateURLs(captured: [WebRenderer.CapturedRequest],
                              endpointHints: [String],
                              baseURL: URL,
                              limit: Int = 6) -> [URL] {
        var result: [URL] = []
        var seen = Set<String>()
        let host = baseURL.host?.lowercased() ?? ""
        func add(_ raw: String) {
            guard result.count < limit,
                  let url = URL(string: raw, relativeTo: baseURL)?.absoluteURL,
                  let scheme = url.scheme?.lowercased(),
                  scheme == "http" || scheme == "https",
                  url.host?.lowercased() == host,
                  isDataURL(url),
                  seen.insert(url.absoluteString).inserted else { return }
            result.append(url)
        }
        for request in captured where request.status < 400 {
            add(request.url)
        }
        for hint in endpointHints {
            add(hint)
        }
        return result
    }

    private static func isDataURL(_ url: URL) -> Bool {
        let path = url.path.lowercased()
        let assets = [".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg",
                      ".ico", ".woff", ".woff2", ".ttf", ".map"]
        if assets.contains(where: { path.hasSuffix($0) }) { return false }
        // 素材/构建产物目录下的 .json 不是内容数据（如 /imgs/.../price_tracker_x.json
        // 的动画素材清单）：直取只会把无关记录顶到结果头部，挤掉真正的新闻数据。
        let assetDirs = ["/imgs/", "/images/", "/assets/", "/static/",
                         "/media/", "/fonts/", "/img/", "/js/"]
        if assetDirs.contains(where: { path.contains($0) }) { return false }
        let markers = ["/api/", "/ajax/", ".ashx", ".asmx", "graphql",
                       "search", "query", "json", "list", "getdata"]
        return markers.contains { (path + "?" + (url.query ?? "")).lowercased().contains($0) }
    }

    /// 把接口响应压成可读文本；无有效数据返回 nil。
    /// - JSON：找记录数组（顶层数组 / 对象里最大的字典数组），每条挑关键字段成一行；
    ///   带上总数、本页条数与分页线索，方便模型继续翻页。
    /// - HTML：转可见文本，过短或错误页视为无效。
    static func summarize(data: Data, contentType: String, label: String,
                          maxChars: Int = 16_000) -> String? {
        if let json = try? JSONSerialization.jsonObject(with: data) {
            return summarizeJSON(json, label: label, maxChars: maxChars)
        }
        let lowered = contentType.lowercased()
        guard lowered.contains("html") || lowered.contains("text/plain")
            || looksLikeHTML(data) else { return nil }
        let raw = String(decoding: data.prefix(400_000), as: UTF8.self)
        let text = ToolSupport.htmlToText(raw, maxChars: maxChars)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard text.count >= 200, !text.contains("No HTTP resource was found"),
              !text.contains("does not support http method") else { return nil }
        return "[\(label) 返回 HTML 内容，\(text.count) 字符]\n"
            + String(text.prefix(maxChars))
    }

    private static func looksLikeHTML(_ data: Data) -> Bool {
        let head = String(decoding: data.prefix(512), as: UTF8.self).lowercased()
        return head.contains("<html") || head.contains("<!doctype html")
    }

    // MARK: - JSON 记录数组

    private struct RecordSet {
        var records: [[String: Any]]
        var total: Int?
        var pageSize: Int?
        var pageInfo: [String: Any]?
    }

    private static func extractRecords(_ root: Any) -> RecordSet {
        if let rows = root as? [[String: Any]], !rows.isEmpty {
            return RecordSet(records: rows, total: rows.count, pageSize: nil, pageInfo: nil)
        }
        guard let dict = root as? [String: Any] else {
            return RecordSet(records: [], total: nil, pageSize: nil, pageInfo: nil)
        }
        // 键名优先于数组大小：Results/Items/Data 这类才是记录集，否则
        // Facets/FacetControls 等元数据数组可能更大，会把精简结果带偏。
        var best: [[String: Any]] = []
        var bestScore = -1
        for (key, value) in dict {
            guard let rows = value as? [[String: Any]], !rows.isEmpty else { continue }
            let nameScore: Int = ["results": 100, "items": 90, "records": 85,
                                  "documents": 80, "data": 75, "hits": 70,
                                  "list": 65]
                .first(where: { key.lowercased().contains($0.key) })?.value ?? 0
            let score = nameScore * 1_000 + rows.count
            if score > bestScore {
                bestScore = score
                best = rows
            }
        }
        let total = firstInt(dict, keys: ["Count", "Total", "TotalCount", "total", "count"])
        let pageSize = firstInt(dict, keys: ["PageSize", "pageSize", "Limit"])
        return RecordSet(records: best, total: total, pageSize: pageSize,
                         pageInfo: dict["SearchFilters"] as? [String: Any]
                            ?? dict["pagination"] as? [String: Any])
    }

    private static func firstInt(_ dict: [String: Any], keys: [String]) -> Int? {
        for key in keys {
            if let n = dict[key] as? Int { return n }
            if let n = dict[key] as? NSNumber { return n.intValue }
            if let s = dict[key] as? String, let n = Int(s) { return n }
        }
        return nil
    }

    private static func summarizeJSON(_ root: Any, label: String,
                                      maxChars: Int,
                                      allDatesCollected: Bool = false) -> String? {
        let set = extractRecords(root)
        guard !set.records.isEmpty else { return nil }

        var lines: [String] = []
        var head = "[\(label) 返回结构化数据"
        if let total = set.total { head += "：共 \(total) 条" }
        head += "，本页 \(set.records.count) 条"
        if let pageSize = set.pageSize, pageSize > 0 { head += "（每页 \(pageSize) 条）" }
        head += "；下面按记录列出关键字段]"
        lines.append(head)

        let limit = min(set.records.count, 60)
        // 列表页常按发布日期倒序：本页日期一致且总数更大时，明确提示还需翻页，
        // 否则模型会把第一页当成「今天全部」。
        if let dateKey = set.records.first.flatMap({ recordDateKey($0) }) {
            let dates = Set(set.records.prefix(limit).compactMap { rec -> String? in
                let source = (rec["Document"] as? [String: Any])
                    ?? (rec["document"] as? [String: Any]) ?? rec
                guard let value = source[dateKey] else { return nil }
                return String(shortValue(value, cap: 24).prefix(10))
            })
            if dates.count == 1, let date = dates.first, !date.isEmpty {
                lines.append("本页发布日期：\(date)（\(set.records.count) 条一致）")
                if !allDatesCollected, let total = set.total,
                   total > set.records.count {
                    lines.append("总数大于本页且日期一致：当天记录可能还有后续页，"
                        + "按下面的分页参数继续翻页才能取全。")
                }
            }
        }
        for (index, record) in set.records.prefix(limit).enumerated() {
            let fields = keyFields(record)
            guard !fields.isEmpty else { continue }
            lines.append("\(index + 1). " + fields.joined(separator: " | "))
        }
        if set.records.count > limit {
            lines.append("… 另有 \(set.records.count - limit) 条未展开")
        }
        if let pageInfo = set.pageInfo, !pageInfo.isEmpty {
            let parts = pageInfo
                .sorted { $0.key < $1.key }
                .prefix(8)
                .map { "\($0.key)=\(shortValue($0.value, cap: 80))" }
            lines.append("分页参数（响应自带）：" + parts.joined(separator: ", "))
            lines.append("如需更多页：用 code_run/shell_run 按上列参数显式翻页，"
                + "不要对未知 POST 接口盲发写请求。")
        }

        var text = lines.joined(separator: "\n")
        if text.count > maxChars {
            text = String(text.prefix(maxChars)) + "\n… [数据接口输出已截断]"
        }
        return text
    }

    /// 找发布/创建日期字段：优先 posted，其次 publish/created/date。
    private static func recordDateKey(_ record: [String: Any]) -> String? {
        let source = (record["Document"] as? [String: Any])
            ?? (record["document"] as? [String: Any]) ?? record
        for marker in ["posted", "publish", "created", "date"] {
            if let key = source.keys.filter({ $0.lowercased().contains(marker) })
                .sorted().first { return key }
        }
        return nil
    }

    /// 记录可能包一层 `Document`（Azure Search 形态）；挑出信息量最高的若干标量字段。
    private static func keyFields(_ record: [String: Any]) -> [String] {
        let source: [String: Any]
        if let doc = record["Document"] as? [String: Any] {
            source = doc
        } else if let doc = record["document"] as? [String: Any] {
            source = doc
        } else {
            source = record
        }
        let scored: [(key: String, score: Int, value: String)] = source.compactMap { key, value in
            guard let text = scalarText(value), !text.isEmpty else { return nil }
            return (key, fieldScore(key), shortValue(text, cap: fieldCap(key)))
        }
        return scored
            .sorted { $0.score == $1.score ? $0.key < $1.key : $0.score > $1.score }
            .prefix(8)
            .map { "\($0.key): \($0.value)" }
    }

    private static func scalarText(_ value: Any) -> String? {
        switch value {
        case let text as String:
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        case let number as NSNumber:
            return number.stringValue
        default:
            return nil
        }
    }

    private static func shortValue(_ value: Any, cap: Int) -> String {
        let raw: String
        if let text = value as? String { raw = text }
        else if let number = value as? NSNumber { raw = number.stringValue }
        else if value is NSNull { return "null" }
        else { return "…" }
        let flat = raw.replacingOccurrences(of: "\\s+", with: " ",
                                           options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return flat.count > cap ? String(flat.prefix(cap)) + "…" : flat
    }

    private static func fieldCap(_ key: String) -> Int {
        let lowered = key.lowercased()
        if lowered.contains("description") || lowered.contains("summary")
            || lowered.contains("content") { return 100 }
        return 140
    }

    /// 字段打分：标题/发布日期/雇主/地点优先，长描述与内部 ID 靠后。
    private static func fieldScore(_ key: String) -> Int {
        let lowered = key.lowercased()
        // 内部 UUID 主键（employer_Id / ...Ids）信息量低，别挤掉城市/描述等字段。
        if lowered == "id" || lowered.hasSuffix("_id") || lowered.hasSuffix("ids") {
            return 5
        }
        let ranked: [(String, Int)] = [
            ("position", 100), ("title", 98), ("subject", 96), ("job_code", 88),
            ("posted", 92), ("publish", 82), ("closing", 78), ("deadline", 78),
            ("employer", 86), ("company", 84), ("school", 70), ("name", 72),
            ("city", 68), ("location", 66), ("province", 62), ("country", 45),
            ("date", 60), ("salary", 52), ("category", 48), ("type", 46),
            ("fte", 40), ("status", 38), ("url", 34), ("link", 34),
            ("id", 28), ("description", 20), ("summary", 18),
        ]
        for (marker, score) in ranked where lowered.contains(marker) { return score }
        return 10
    }

    // MARK: - 只读分页合并

    struct PageSpec: Sendable {
        var endpoint: URL
        /// 表单字段：完全来自响应自带的搜索条件，翻页时只改页码。
        var form: [String: String]
        var pageKey: String
        var startPage: Int
        var dateKey: String?
        var dateValue: String?
    }

    /// 只认「只读搜索」端点：路径含 search/query/list/get/find 且不含写动词，
    /// 绝不对 Add/Create/Update/Delete/Apply/Cart 之类接口盲发 POST。
    static func isReadOnlySearchEndpoint(_ url: URL) -> Bool {
        let path = url.path.lowercased()
        let writeMarkers = ["add", "create", "update", "delete", "remove", "save",
                            "apply", "submit", "cart", "filter", "set", "edit"]
        guard !writeMarkers.contains(where: { path.contains($0) }) else { return false }
        return ["search", "query", "list", "get", "find"]
            .contains { path.contains($0) }
    }

    /// 翻页端点打分：纯 search/list 最高，PostSuggest 之类写/建议端点直接排除。
    private static func pagingEndpointScore(_ url: URL) -> Int {
        let last = url.lastPathComponent.lowercased()
        if ["add", "create", "update", "delete", "save", "apply", "submit", "post"]
            .contains(where: { last.contains($0) }) { return 0 }
        if ["search", "query", "list", "find"].contains(last) { return 100 }
        if ["search", "query", "list", "find"].contains(where: { last.contains($0) }) {
            return 70
        }
        return 20
    }

    /// 从响应自带的 `SearchFilters` 生成翻页模板；只适用于「第一页日期一致、
    /// 总数大于本页」的列表页，其它结构直接返回 nil。
    static func pageSpec(root: Any, candidates: [URL], exclude: URL?) -> PageSpec? {
        guard let dict = root as? [String: Any],
              let filters = dict["SearchFilters"] as? [String: Any],
              let pageKey = filters.keys.filter({
                  ["currentpage", "page", "pagenumber", "pageindex"]
                    .contains($0.lowercased())
              }).sorted().first else { return nil }
        let readOnly = candidates.filter {
            $0 != exclude && isReadOnlySearchEndpoint($0)
        }
        // 常见的 GetSearch / Search 成对形态里，GET 端点不支持 POST 翻页。
        // 优先找同模块的只读端点（/api/SearchIndex/*），再按末段语义打分，
        // 避免误挑 /api/searchip/getcountry 或 PostSuggest 这类无关端点。
        let parent = exclude?.deletingLastPathComponent().path.lowercased()
        let sameModule = readOnly.filter {
            $0.path != exclude?.path
                && $0.deletingLastPathComponent().path.lowercased() == parent
        }
        let pool = sameModule.isEmpty
            ? readOnly.filter { $0.path != exclude?.path } : sameModule
        let ranked = pool.sorted { pagingEndpointScore($0) > pagingEndpointScore($1) }
        guard let endpoint = ranked.first(where: { pagingEndpointScore($0) > 0 })
            ?? readOnly.first else { return nil }
        var form: [String: String] = [:]
        for (key, value) in filters {
            if value is NSNull { form[key] = "" }
            else if let text = value as? String { form[key] = text }
            else if let number = value as? NSNumber { form[key] = number.stringValue }
            else if JSONSerialization.isValidJSONObject(value),
                    let data = try? JSONSerialization.data(withJSONObject: value),
                    let text = String(data: data, encoding: .utf8) { form[key] = text }
        }
        guard form[pageKey] != nil else { return nil }
        let rows = extractRecords(root).records
        guard let first = rows.first else { return nil }
        let dateKey = recordDateKey(first)
        let source = (first["Document"] as? [String: Any])
            ?? (first["document"] as? [String: Any]) ?? first
        let dateValue = dateKey.flatMap { key -> String? in
            guard let value = source[key] as? String, value.count >= 10 else { return nil }
            return String(value.prefix(10))
        }
        return PageSpec(endpoint: endpoint, form: form, pageKey: pageKey,
                        startPage: firstInt(filters, keys: [pageKey]) ?? 0,
                        dateKey: dateKey, dateValue: dateValue)
    }

    static func records(in root: Any) -> [[String: Any]] {
        extractRecords(root).records
    }

    static func totalCount(in root: Any) -> Int? {
        guard let dict = root as? [String: Any] else { return nil }
        return firstInt(dict, keys: ["Count", "Total", "TotalCount", "total", "count"])
    }

    /// 合并多页同一天的记录后重新生成精简列表。
    static func summary(records: [[String: Any]], total: Int?, label: String,
                        pageInfo: [String: String]?, allDatesCollected: Bool,
                        maxChars: Int) -> String? {
        var dict: [String: Any] = ["Results": records]
        if let total { dict["Count"] = total }
        if let pageInfo { dict["SearchFilters"] = pageInfo }
        return summarizeJSON(dict, label: label, maxChars: maxChars,
                             allDatesCollected: allDatesCollected)
    }
}
