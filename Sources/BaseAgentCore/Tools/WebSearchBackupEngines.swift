import Foundation
import SwiftSoup

// MARK: - 备用检索源
//
// 这些源不是通用网页搜索，而是在本机实测可直连、结果可解析的垂直/补充
// 检索 API。它们作为主引擎结果不足时的补位：延迟排序 + 查询意图匹配，
// 每次自动备用只取最快、最相关的 8 个，避免 50 个源全量请求拖慢主流程。
//
// 延迟数值来自本机真实请求（2026-11，单次/二次探测的中位近似值），
// 仅用于排序；网络环境变化后可通过重新探测更新。

// MARK: 适配器

struct WebSearchBackupAdapter: WebSearchEngineAdapter, Sendable {
    let engine: WebSearchEngine
    let builder: @Sendable (WebSearchOptions, String) -> URLRequest?
    let parser: @Sendable (Data) throws -> [WebSearchHit]

    func request(options: WebSearchOptions, userAgent: String) -> URLRequest {
        builder(options, userAgent)
            ?? WebSearchRequest.make("https://backup.invalid/",
                                     headers: ["User-Agent": userAgent])
    }

    func parse(_ data: Data, options: WebSearchOptions) throws -> [WebSearchHit] {
        try parser(data)
    }
}

// MARK: 目录

enum WebSearchBackupCatalog {

    static let definitions: [WebSearchBackupAdapter] = {
        var list: [WebSearchBackupAdapter] = []

        // --- 百科 / 词典 / 新闻（MediaWiki，4）---
        list += [
            mediaWiki(e("wikipedia", "Wikipedia", "reference", 1000, weight: 0.72),
                      host: "en.wikipedia.org"),
            mediaWiki(e("wikipedia_zh", "维基百科(中文)", "reference", 1600, weight: 0.72),
                      host: "zh.wikipedia.org"),
            mediaWiki(e("wiktionary", "Wiktionary", "reference", 1000, weight: 0.65),
                      host: "en.wiktionary.org"),
            mediaWiki(e("wikinews", "Wikinews", "news", 1050, weight: 0.65),
                      host: "en.wikinews.org"),
        ]

        // --- 开发者问答（StackExchange，6）---
        list += [
            stackExchange(e("stackoverflow", "Stack Overflow", "code", 1400), site: "stackoverflow"),
            stackExchange(e("superuser", "Super User", "code", 1500), site: "superuser"),
            stackExchange(e("askubuntu", "Ask Ubuntu", "code", 1100), site: "askubuntu"),
            stackExchange(e("security_se", "Information Security", "code", 1200), site: "security"),
            stackExchange(e("math", "Mathematics", "science", 1400), site: "math"),
            stackExchange(e("physics", "Physics", "science", 1230), site: "physics"),
        ]

        // --- 代码 / 文档（4）---
        list += [
            json(e("github", "GitHub", "code", 2940, weight: 0.70),
                 "https://api.github.com/search/repositories?q={q}&per_page=10",
                 headers: ["Accept": "application/vnd.github+json"],
                 BackupJSONMapping(items: "items",
                                   title: ["full_name"],
                                   url: ["html_url"],
                                   snippet: ["description"])),
            json(e("gitlab", "GitLab", "code", 2000, weight: 0.65),
                 "https://gitlab.com/api/v4/projects?search={q}&per_page=10",
                 BackupJSONMapping(items: "",
                                   title: ["path_with_namespace"],
                                   url: ["web_url"],
                                   snippet: ["description"])),
            WebSearchBackupAdapter(
                engine: e("hn", "Hacker News", "general", 1700, weight: 0.70),
                builder: { options, ua in
                    let url = "https://hn.algolia.com/api/v1/search?query="
                        + BackupHTTP.query(options.query) + "&hitsPerPage=10"
                    return BackupHTTP.get(url, userAgent: ua, language: options.language)
                },
                parser: { data in try BackupParse.hackerNews(data) }),
            json(e("mdn", "MDN", "code", 1500, weight: 0.70),
                 "https://developer.mozilla.org/api/v1/search?q={q}&locale=en-US",
                 BackupJSONMapping(items: "documents",
                                   title: ["title"],
                                   url: ["mdn_url"],
                                   urlPrefix: "https://developer.mozilla.org",
                                   snippet: ["summary"])),
        ]

        // --- 包仓库（8）---
        list += [
            json(e("npm", "npm", "package", 1400, weight: 0.65),
                 "https://registry.npmjs.org/-/v1/search?text={q}&size=10",
                 BackupJSONMapping(items: "objects",
                                   title: ["package.name"],
                                   url: ["package.links.npm"],
                                   snippet: ["package.description"])),
            json(e("crates", "crates.io", "package", 2350, weight: 0.60),
                 "https://crates.io/api/v1/crates?q={q}&per_page=10",
                 BackupJSONMapping(items: "crates",
                                   title: ["name"],
                                   urlTemplate: "https://crates.io/crates/{title}",
                                   snippet: ["description"])),
            json(e("rubygems", "RubyGems", "package", 2100, weight: 0.60),
                 "https://rubygems.org/api/v1/search.json?query={q}",
                 BackupJSONMapping(items: "",
                                   title: ["name"],
                                   url: ["project_uri"],
                                   snippet: ["info"])),
            json(e("packagist", "Packagist", "package", 1290, weight: 0.60),
                 "https://packagist.org/search.json?q={q}&per_page=10",
                 BackupJSONMapping(items: "results",
                                   title: ["name"],
                                   urlTemplate: "https://packagist.org/packages/{title}",
                                   snippet: ["description"])),
            json(e("nuget", "NuGet", "package", 2500, weight: 0.60),
                 "https://azuresearch-usnc.nuget.org/query?q={q}&take=10",
                 BackupJSONMapping(items: "data",
                                   title: ["id"],
                                   urlTemplate: "https://www.nuget.org/packages/{title}",
                                   snippet: ["description"])),
            json(e("dockerhub", "Docker Hub", "package", 1100, weight: 0.60),
                 "https://hub.docker.com/v2/search/repositories/?query={q}&page_size=10",
                 BackupJSONMapping(items: "results",
                                   title: ["repo_name"],
                                   urlTemplate: "https://hub.docker.com/r/{title}",
                                   snippet: ["short_description", "description"])),
            json(e("hex", "Hex.pm", "package", 1170, weight: 0.60),
                 "https://hex.pm/api/packages?search={q}&sort=downloads",
                 BackupJSONMapping(items: "",
                                   title: ["name"],
                                   url: ["html_url"],
                                   snippet: ["meta.description"])),
            json(e("pubdev", "pub.dev", "package", 1060, weight: 0.60),
                 "https://pub.dev/api/search?q={q}",
                 BackupJSONMapping(items: "packages",
                                   title: ["package"],
                                   urlTemplate: "https://pub.dev/packages/{title}")),
        ]

        // --- 学术 / 科学（12）---
        list += [
            json(e("crossref", "Crossref", "academic", 2400),
                 "https://api.crossref.org/works?query={q}&rows=10",
                 BackupJSONMapping(items: "message.items",
                                   title: ["title"],
                                   url: ["URL", "DOI"],
                                   snippet: ["container-title", "abstract"])),
            json(e("openalex", "OpenAlex", "academic", 1700),
                 "https://api.openalex.org/works?search={q}&per-page=10",
                 BackupJSONMapping(items: "results",
                                   title: ["display_name"],
                                   url: ["primary_location.landing_page_url", "doi", "id"],
                                   snippet: ["primary_location.source.display_name", "type"])),
            xml(e("arxiv", "arXiv", "academic", 1460),
                "https://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results=10",
                item: "entry",
                title: ["title"], url: ["id", "link"], snippet: ["summary"]),
            json(e("europepmc", "Europe PMC", "academic", 1540),
                 "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}&format=json&pageSize=10",
                 BackupJSONMapping(items: "resultList.result",
                                   title: ["title"],
                                   urlTemplate: "https://europepmc.org/article/{source}/{id}",
                                   urlKeys: ["source": ["source"], "id": ["id"]],
                                   snippet: ["abstractText", "journalTitle"])),
            json(e("doaj", "DOAJ", "academic", 2200),
                 "https://doaj.org/api/search/articles/{q}?pageSize=10",
                 BackupJSONMapping(items: "results",
                                   title: ["bibjson.title"],
                                   url: ["bibjson.link[0].url"],
                                   snippet: ["bibjson.abstract"])),
            json(e("datacite", "DataCite", "academic", 3500),
                 "https://api.datacite.org/dois?query={q}&page[size]=10",
                 BackupJSONMapping(items: "data",
                                   title: ["attributes.titles[0].title"],
                                   url: ["attributes.url"],
                                   snippet: ["attributes.descriptions[0].description"])),
            json(e("ror", "ROR", "academic", 1500),
                 "https://api.ror.org/organizations?query={q}",
                 BackupJSONMapping(items: "items",
                                   title: ["names[0].value", "name"],
                                   url: ["links[0].value", "id"],
                                   snippet: ["locations[0].geonames_details.country_name"])),
            json(e("zenodo", "Zenodo", "academic", 2500),
                 "https://zenodo.org/api/records?q={q}&size=10",
                 BackupJSONMapping(items: "hits.hits",
                                   title: ["metadata.title"],
                                   url: ["links.self_html", "doi_url"],
                                   snippet: ["metadata.description"])),
            json(e("gbif", "GBIF", "science", 1450),
                 "https://api.gbif.org/v1/species/search?q={q}&limit=10",
                 BackupJSONMapping(items: "results",
                                   title: ["scientificName"],
                                   urlTemplate: "https://www.gbif.org/species/{key}",
                                   urlKeys: ["key": ["key"]],
                                   snippet: ["kingdom", "family", "genus"])),
            json(e("inaturalist", "iNaturalist", "science", 1600),
                 "https://api.inaturalist.org/v1/search?q={q}&per_page=10",
                 BackupJSONMapping(items: "results",
                                   title: ["record.name", "record.preferred_common_name"],
                                   urlTemplate: "https://www.inaturalist.org/taxa/{id}",
                                   urlKeys: ["id": ["record.id"]],
                                   snippet: ["record.rank", "record.preferred_common_name"])),
            json(e("uniprot", "UniProt", "science", 2700),
                 "https://rest.uniprot.org/uniprotkb/search?query={q}&format=json&size=10",
                 BackupJSONMapping(items: "results",
                                   title: ["proteinDescription.recommendedName.fullName.value"],
                                   urlTemplate: "https://www.uniprot.org/uniprotkb/{accession}",
                                   urlKeys: ["accession": ["primaryAccession"]],
                                   snippet: ["organism.scientificName", "proteinExistence"])),
            json(e("clinicaltrials", "ClinicalTrials.gov", "science", 1850),
                 "https://clinicaltrials.gov/api/v2/studies?query.term={q}&pageSize=10",
                 BackupJSONMapping(items: "studies",
                                   title: ["protocolSection.identificationModule.briefTitle"],
                                   urlTemplate: "https://clinicaltrials.gov/study/{nctId}",
                                   urlKeys: ["nctId": ["protocolSection.identificationModule.nctId"]],
                                   snippet: ["protocolSection.descriptionModule.briefSummary"])),
        ]

        // --- 图书 / 媒体（8）---
        list += [
            json(e("openverse", "Openverse", "media", 1900),
                 "https://api.openverse.org/v1/images/?q={q}&page_size=10",
                 BackupJSONMapping(items: "results",
                                   title: ["title"],
                                   url: ["foreign_landing_url"],
                                   snippet: ["creator", "provider"])),
            json(e("openverse_audio", "Openverse Audio", "media", 1260),
                 "https://api.openverse.org/v1/audio/?q={q}&page_size=10",
                 BackupJSONMapping(items: "results",
                                   title: ["title"],
                                   url: ["foreign_landing_url"],
                                   snippet: ["creator", "provider"])),
            json(e("openlibrary", "Open Library", "book", 2000),
                 "https://openlibrary.org/search.json?q={q}&limit=10",
                 BackupJSONMapping(items: "docs",
                                   title: ["title"],
                                   urlTemplate: "https://openlibrary.org{key}",
                                   urlKeys: ["key": ["key"]],
                                   snippet: ["author_name", "first_publish_year"])),
            json(e("internet_archive", "Internet Archive", "book", 1440),
                 "https://archive.org/advancedsearch.php?q={q}&fl[]=identifier&fl[]=title&rows=10&output=json",
                 BackupJSONMapping(items: "response.docs",
                                   title: ["title"],
                                   urlTemplate: "https://archive.org/details/{identifier}",
                                   urlKeys: ["identifier": ["identifier"]],
                                   snippet: ["identifier"])),
            json(e("itunes", "iTunes", "media", 1580),
                 "https://itunes.apple.com/search?term={q}&media=music&limit=10",
                 BackupJSONMapping(items: "results",
                                   title: ["trackName", "collectionName", "artistName"],
                                   url: ["trackViewUrl", "collectionViewUrl", "artistViewUrl"],
                                   snippet: ["artistName", "collectionName"])),
            json(e("deezer", "Deezer", "media", 1800),
                 "https://api.deezer.com/search?q={q}&limit=10",
                 BackupJSONMapping(items: "data",
                                   title: ["title"],
                                   url: ["link"],
                                   snippet: ["artist.name"])),
            json(e("tvmaze", "TVmaze", "media", 1370),
                 "https://api.tvmaze.com/search/shows?q={q}",
                 BackupJSONMapping(items: "",
                                   title: ["show.name"],
                                   url: ["show.url"],
                                   snippet: ["show.summary"])),
            json(e("musicbrainz", "MusicBrainz", "media", 1400),
                 "https://musicbrainz.org/ws/2/artist/?query={q}&fmt=json&limit=10",
                 BackupJSONMapping(items: "artists",
                                   title: ["name"],
                                   urlTemplate: "https://musicbrainz.org/artist/{id}",
                                   urlKeys: ["id": ["id"]],
                                   snippet: ["disambiguation", "country"])),
        ]

        // --- 数据 / 新闻 / 地点 / 金融（6）---
        list += [
            json(e("hf_models", "Hugging Face Models", "data", 1050, weight: 0.65),
                 "https://huggingface.co/api/models?search={q}&limit=10",
                 BackupJSONMapping(items: "",
                                   title: ["id"],
                                   urlTemplate: "https://huggingface.co/{title}",
                                   snippet: ["pipeline_tag", "tags"])),
            json(e("hf_datasets", "Hugging Face Datasets", "data", 1000, weight: 0.65),
                 "https://huggingface.co/api/datasets?search={q}&limit=10",
                 BackupJSONMapping(items: "",
                                   title: ["id"],
                                   urlTemplate: "https://huggingface.co/datasets/{title}",
                                   snippet: ["description", "tags"])),
            xml(e("google_news", "Google News", "news", 1900),
                "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en",
                item: "item", title: ["title"], url: ["link"], snippet: ["description"]),
            xml(e("bing_news", "Bing News", "news", 1200),
                "https://www.bing.com/news/search?q={q}&format=RSS",
                item: "item", title: ["title"], url: ["link"], snippet: ["description"]),
            json(e("openmeteo", "Open-Meteo Geocoding", "local", 2200),
                 "https://geocoding-api.open-meteo.com/v1/search?name={q}&count=10",
                 BackupJSONMapping(items: "results",
                                   title: ["name"],
                                   urlTemplate: "https://www.openstreetmap.org/?mlat={latitude}&mlon={longitude}",
                                   urlKeys: ["latitude": ["latitude"], "longitude": ["longitude"]],
                                   snippet: ["country", "admin1"])),
            json(e("coingecko", "CoinGecko", "finance", 1250),
                 "https://api.coingecko.com/api/v3/search?query={q}",
                 BackupJSONMapping(items: "coins",
                                   title: ["name"],
                                   urlTemplate: "https://www.coingecko.com/en/coins/{id}",
                                   urlKeys: ["id": ["id"]],
                                   snippet: ["symbol"])),
        ]

        // --- 补充通用网页引擎（2）---
        list += [
            html(e("yahoojp", "Yahoo! JAPAN", "general", 2800, weight: 0.72),
                 "https://search.yahoo.co.jp/search?p={q}",
                 BackupHTMLMapping(item: "div.sw-Card.Algo",
                                   title: "h3.sw-Card__titleMain",
                                   url: "a.sw-Card__titleInner",
                                   snippet: [".sw-Card__summary"],
                                   base: "https://search.yahoo.co.jp")),
            html(e("wiby", "Wiby", "general", 2100, weight: 0.60),
                 "https://wiby.me/?q={q}",
                 BackupHTMLMapping(item: "blockquote",
                                   title: "a.tlink",
                                   url: "a.tlink",
                                   base: "https://wiby.me")),
        ]

        return list
    }()

    /// 按实测延迟升序，作为“最快 50 个”的顺序。
    static let engines: [WebSearchEngine] = definitions
        .map(\.engine)
        .sorted { $0.latencyMs < $1.latencyMs }

    private static let map: [String: WebSearchBackupAdapter] =
        Dictionary(uniqueKeysWithValues: definitions.map { ($0.engine.key, $0) })

    static func adapter(for engine: WebSearchEngine) -> (any WebSearchEngineAdapter)? {
        map[engine.key]
    }
}

// MARK: - 配置构造辅助

private func e(_ key: String, _ name: String, _ category: String,
               _ latencyMs: Int, weight: Double = 0.62) -> WebSearchEngine {
    WebSearchEngine(key: key, displayName: name, weight: weight,
                    category: category, tier: .backup, latencyMs: latencyMs)
}

private func mediaWiki(_ engine: WebSearchEngine, host: String) -> WebSearchBackupAdapter {
    json(engine,
         "https://\(host)/w/api.php?action=query&list=search&srsearch={q}&format=json&srlimit=10",
         BackupJSONMapping(items: "query.search",
                           title: ["title"],
                           urlTemplate: "https://\(host)/wiki/{title}",
                           snippet: ["snippet"]))
}

private func stackExchange(_ engine: WebSearchEngine, site: String) -> WebSearchBackupAdapter {
    WebSearchBackupAdapter(
        engine: engine,
        builder: { options, ua in
            let url = "https://api.stackexchange.com/2.3/search/advanced"
                + "?order=desc&sort=relevance&q=\(BackupHTTP.query(options.query))"
                + "&site=\(site)&pagesize=10"
            return BackupHTTP.get(url, userAgent: ua, language: options.language)
        },
        parser: { data in try BackupParse.stackExchange(data) })
}

private func json(_ engine: WebSearchEngine,
                  _ urlTemplate: String,
                  headers: [String: String] = [:],
                  _ mapping: BackupJSONMapping) -> WebSearchBackupAdapter {
    WebSearchBackupAdapter(
        engine: engine,
        builder: { options, ua in
            let url = urlTemplate.replacingOccurrences(
                of: "{q}", with: BackupHTTP.query(options.query))
            return BackupHTTP.get(url, userAgent: ua, language: options.language,
                                  extraHeaders: headers)
        },
        parser: { data in try BackupParse.jsonHits(data, mapping) })
}

private func xml(_ engine: WebSearchEngine,
                 _ urlTemplate: String,
                 item: String,
                 title: [String],
                 url: [String],
                 snippet: [String]) -> WebSearchBackupAdapter {
    WebSearchBackupAdapter(
        engine: engine,
        builder: { options, ua in
            let value = urlTemplate.replacingOccurrences(
                of: "{q}", with: BackupHTTP.query(options.query))
            return BackupHTTP.get(value, userAgent: ua, language: options.language,
                                  accept: "application/rss+xml, application/atom+xml, "
                                    + "application/xml, text/xml, */*")
        },
        parser: { data in
            try BackupParse.xmlHits(data, item: item, title: title,
                                    url: url, snippet: snippet)
        })
}

private func html(_ engine: WebSearchEngine,
                  _ urlTemplate: String,
                  _ mapping: BackupHTMLMapping) -> WebSearchBackupAdapter {
    WebSearchBackupAdapter(
        engine: engine,
        builder: { options, ua in
            let url = urlTemplate.replacingOccurrences(
                of: "{q}", with: BackupHTTP.query(options.query))
            return BackupHTTP.get(url, userAgent: ua, language: options.language,
                                  accept: "text/html,application/xhtml+xml,*/*;q=0.8")
        },
        parser: { data in try BackupParse.htmlHits(data, mapping) })
}

// MARK: - 请求辅助

enum BackupHTTP {
    /// 严格百分号编码：不能直接用 `.urlQueryAllowed`，它允许 `&`/`=`
    /// 留在查询值里，会把用户查询注入成额外参数。
    static func query(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics
            .union(CharacterSet(charactersIn: "-._~"))
        return value.addingPercentEncoding(withAllowedCharacters: allowed) ?? value
    }

    static func get(_ url: String,
                    userAgent: String,
                    language: String = "en-US",
                    accept: String = "application/json, text/plain, */*",
                    extraHeaders: [String: String] = [:]) -> URLRequest? {
        guard let components = URLComponents(string: url),
              let finalURL = components.url else { return nil }
        var request = URLRequest(url: finalURL)
        request.timeoutInterval = 7
        var headers = WebSearchHeaders.base(userAgent: userAgent, language: language)
        headers["Accept"] = accept
        for (key, value) in extraHeaders { headers[key] = value }
        WebSearchHeaders.apply(headers, to: &request)
        return request
    }
}

// MARK: - 响应映射

struct BackupJSONMapping: Sendable {
    var items: String
    var title: [String]
    var url: [String] = []
    var urlTemplate: String?
    var urlKeys: [String: [String]] = [:]
    var urlPrefix: String?
    var snippet: [String] = []
}

struct BackupHTMLMapping: Sendable {
    var item: String
    var title: String
    var url: String
    var snippet: [String] = []
    var base: String
}

// MARK: - 解析

enum BackupParse {

    static func jsonHits(_ data: Data, _ mapping: BackupJSONMapping) throws -> [WebSearchHit] {
        guard let root = try? JSONSerialization.jsonObject(with: data) else {
            throw WebSearchError.parse("JSON 响应无法解析")
        }
        var hits: [WebSearchHit] = []
        for item in items(root, mapping.items) {
            let title = WebSearchHTML.clean(string(item, mapping.title))
            guard title.count >= 2 else { continue }

            var url = string(item, mapping.url)
            if url.isEmpty, let template = mapping.urlTemplate {
                url = render(template, item: item, title: title, keys: mapping.urlKeys)
            }
            if !url.lowercased().hasPrefix("http"), let prefix = mapping.urlPrefix {
                url = prefix + url
            }
            guard url.lowercased().hasPrefix("http") else { continue }

            let snippet = plain(string(item, mapping.snippet))
            hits.append(WebSearchHit(title: title, url: url, snippet: snippet,
                                     position: hits.count + 1))
            if hits.count >= 20 { break }
        }
        return hits
    }

    /// StackExchange 搜索 API 不返回正文，用标签/评分/回答数构成摘要。
    static func stackExchange(_ data: Data) throws -> [WebSearchHit] {
        guard let root = try? JSONSerialization.jsonObject(with: data) else {
            throw WebSearchError.parse("StackExchange JSON 无法解析")
        }
        var hits: [WebSearchHit] = []
        for item in items(root, "items") {
            let title = WebSearchHTML.clean(string(item, ["title"]))
            let url = string(item, ["link"])
            guard title.count >= 2, url.hasPrefix("http") else { continue }

            var parts: [String] = []
            let tags = string(item, ["tags"])
            if !tags.isEmpty { parts.append("tags: \(tags)") }
            let score = string(item, ["score"])
            if !score.isEmpty { parts.append("评分 \(score)") }
            let answers = string(item, ["answer_count"])
            if !answers.isEmpty { parts.append("\(answers) 个回答") }
            let owner = string(item, ["owner.display_name"])
            if !owner.isEmpty { parts.append(owner) }

            hits.append(WebSearchHit(title: title, url: url,
                                     snippet: parts.joined(separator: " · "),
                                     position: hits.count + 1))
            if hits.count >= 20 { break }
        }
        return hits
    }

    /// Hacker News：链接帖用 url，Ask HN 回退到 item 页。
    static func hackerNews(_ data: Data) throws -> [WebSearchHit] {
        guard let root = try? JSONSerialization.jsonObject(with: data) else {
            throw WebSearchError.parse("Hacker News JSON 无法解析")
        }
        var hits: [WebSearchHit] = []
        for item in items(root, "hits") {
            let title = WebSearchHTML.clean(string(item, ["title", "story_title"]))
            guard title.count >= 2 else { continue }
            let objectID = string(item, ["objectID"])
            var url = string(item, ["url", "story_url"])
            if url.isEmpty, !objectID.isEmpty {
                url = "https://news.ycombinator.com/item?id=\(objectID)"
            }
            guard url.lowercased().hasPrefix("http") else { continue }

            var parts: [String] = []
            let points = string(item, ["points"])
            if !points.isEmpty { parts.append("\(points) points") }
            let comments = string(item, ["num_comments"])
            if !comments.isEmpty { parts.append("\(comments) comments") }
            let author = string(item, ["author"])
            if !author.isEmpty { parts.append("by \(author)") }
            let created = string(item, ["created_at"])
            if !created.isEmpty { parts.append(String(created.prefix(10))) }

            hits.append(WebSearchHit(title: title, url: url,
                                     snippet: parts.joined(separator: " · "),
                                     position: hits.count + 1))
            if hits.count >= 20 { break }
        }
        return hits
    }

    /// RSS 2.0 / Atom：item/entry 里的 title/link/description/summary。
    static func xmlHits(_ data: Data,
                        item: String,
                        title: [String],
                        url: [String],
                        snippet: [String]) throws -> [WebSearchHit] {
        guard let xml = String(data: data, encoding: .utf8) else {
            throw WebSearchError.parse("XML 响应不是 UTF-8")
        }
        let document = try SwiftSoup.parse(xml, "")
        var hits: [WebSearchHit] = []
        for node in try document.select(item) {
            let value = WebSearchHTML.firstText(node, title)
            guard value.count >= 2 else { continue }

            var link = ""
            for tag in url {
                if let element = try? node.select(tag).first() {
                    let text = WebSearchHTML.text(element)
                    if text.hasPrefix("http") {
                        link = text
                    } else {
                        let href = WebSearchHTML.attr(element, "href")
                        if href.hasPrefix("http") { link = href }
                    }
                }
                if !link.isEmpty { break }
            }
            guard link.lowercased().hasPrefix("http") else { continue }

            let body = plain(WebSearchHTML.firstText(node, snippet))
            hits.append(WebSearchHit(title: value, url: link, snippet: body,
                                     position: hits.count + 1))
            if hits.count >= 20 { break }
        }
        return hits
    }

    static func htmlHits(_ data: Data, _ mapping: BackupHTMLMapping) throws -> [WebSearchHit] {
        let document = try WebSearchHTML.document(data)
        var hits: [WebSearchHit] = []
        for item in try document.select(mapping.item) {
            guard let anchor = try item.select(mapping.url).first() else { continue }
            var title = WebSearchHTML.firstText(item, [mapping.title])
            if title.isEmpty { title = WebSearchHTML.text(anchor) }
            guard title.count >= 2 else { continue }

            var url = WebSearchHTML.attr(anchor, "href")
            url = WebSearchURL.absolute(url, base: mapping.base)
            guard url.lowercased().hasPrefix("http") else { continue }

            var snippet = WebSearchHTML.firstText(item, mapping.snippet)
            if snippet.isEmpty {
                snippet = WebSearchHTML.text(item)
                    .replacingOccurrences(of: title, with: "")
                    .replacingOccurrences(of: WebSearchHTML.attr(anchor, "href"), with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            hits.append(WebSearchHit(title: title, url: url, snippet: snippet,
                                     position: hits.count + 1))
            if hits.count >= 20 { break }
        }
        return hits
    }

    // MARK: JSON 路径

    static func items(_ root: Any, _ path: String) -> [[String: Any]] {
        guard !path.isEmpty else {
            if let array = root as? [[String: Any]] { return array }
            if let dict = root as? [String: Any] { return [dict] }
            return []
        }
        guard let value = dig(root, path) else { return [] }
        if let array = value as? [Any] {
            return array.compactMap { $0 as? [String: Any] }
        }
        if let dict = value as? [String: Any] { return [dict] }
        return []
    }

    /// 支持 `a.b`、`title[0]`、`links[0].value` 这类简单路径。
    static func dig(_ object: Any, _ path: String) -> Any? {
        var current: Any? = object
        for raw in path.split(separator: ".") {
            var key = String(raw)
            var index: Int?
            if let open = key.firstIndex(of: "["), key.hasSuffix("]") {
                index = Int(key[key.index(after: open)..<key.index(before: key.endIndex)])
                key = String(key[..<open])
            }
            if !key.isEmpty {
                if let dict = current as? [String: Any] {
                    current = dict[key]
                } else if let array = current as? [Any], let position = Int(key) {
                    current = position < array.count ? array[position] : nil
                } else {
                    return nil
                }
            }
            if let index {
                guard let array = current as? [Any], index < array.count else { return nil }
                current = array[index]
            }
        }
        return current
    }

    static func string(_ object: [String: Any], _ paths: [String]) -> String {
        for path in paths {
            guard let value = dig(object, path) else { continue }
            let text = scalar(value)
            if !text.isEmpty { return text }
        }
        return ""
    }

    private static func scalar(_ value: Any) -> String {
        switch value {
        case let text as String:
            return WebSearchHTML.clean(text)
        case let number as NSNumber:
            return number.stringValue
        case let array as [Any]:
            let joined = array.map { scalar($0) }.filter { !$0.isEmpty }
                .joined(separator: ", ")
            return joined
        case let dict as [String: Any]:
            for key in ["value", "name", "title"] {
                if let nested = dict[key] { return scalar(nested) }
            }
            return ""
        default:
            return ""
        }
    }

    private static func render(_ template: String,
                               item: [String: Any],
                               title: String,
                               keys: [String: [String]]) -> String {
        var result = template
        if result.contains("{title}") {
            result = result.replacingOccurrences(
                of: "{title}", with: pathEncode(title.replacingOccurrences(of: " ", with: "_")))
        }
        for (placeholder, paths) in keys {
            let value = string(item, paths)
            result = result.replacingOccurrences(
                of: "{\(placeholder)}", with: pathEncode(value))
        }
        return result
    }

    private static func pathEncode(_ value: String) -> String {
        var allowed = CharacterSet.urlPathAllowed
        allowed.remove(charactersIn: "?#")
        return value.addingPercentEncoding(withAllowedCharacters: allowed) ?? value
    }

    static func plain(_ raw: String) -> String {
        guard !raw.isEmpty else { return "" }
        if let document = try? SwiftSoup.parse(raw), let text = try? document.text() {
            return WebSearchHTML.clean(text)
        }
        return WebSearchHTML.clean(raw)
    }
}

// MARK: - 备用选择

enum WebSearchBackupSelector {

    /// 按查询意图给备用源打分，再按实测延迟升序取前 `limit` 个。
    static func select(query: String, limit: Int) -> [WebSearchEngine] {
        let intents = categories(for: query)
        let scored = WebSearchEngine.backup.map { engine -> (WebSearchEngine, Int) in
            var score = 0
            if intents.contains(engine.category) { score += 4 }
            if intents.contains("general"),
               ["general", "reference", "news", "code"].contains(engine.category) {
                score += 1
            }
            if engine.category == "reference" { score += 1 }
            return (engine, score)
        }
        return scored.sorted {
            if $0.1 != $1.1 { return $0.1 > $1.1 }
            return $0.0.latencyMs < $1.0.latencyMs
        }.prefix(limit).map(\.0)
    }

    static func categories(for query: String) -> Set<String> {
        let text = query.lowercased()
        var result: Set<String> = []

        /// 拉丁词按词边界匹配：否则 `api` 会命中 `capital`、`tv` 会命中
        /// `tvshow` 之类，导致备用源选择完全跑偏；中文仍按子串匹配。
        func hit(_ words: [String]) -> Bool {
            for word in words {
                if isCJK(word) {
                    if text.contains(word) { return true }
                    continue
                }
                let pattern = "(?<![a-z0-9])"
                    + NSRegularExpression.escapedPattern(for: word)
                    + "(?![a-z0-9])"
                if text.range(of: pattern,
                              options: [.regularExpression, .caseInsensitive]) != nil {
                    return true
                }
            }
            return false
        }

        if hit(["code", "api", "error", "bug", "github", "gitlab", "npm", "编译",
                "install", "docker", "dockerhub", "kubernetes", "linux", "unix",
                "sql", "react", "vue", "node", "php", "ruby", "python", "swift",
                "rust", "java", "javascript", "typescript", "c++", "golang",
                "server", "cve", "vulnerability", "漏洞", "代码", "开发"]) {
            result.insert("code")
            result.insert("package")
        }
        if hit(["paper", "research", "study", "arxiv", "doi", "journal", "citation",
                "thesis", "preprint", "scholar", "论文", "研究", "学术"]) {
            result.insert("academic")
            result.insert("science")
        }
        if hit(["gene", "protein", "species", "biology", "chemical", "drug",
                "clinical", "trial", "astronomy", "physics", "math", "taxonomy",
                "基因", "蛋白", "物种", "临床", "药物"]) {
            result.insert("science")
        }
        if hit(["news", "latest", "today", "yesterday", "breaking", "election",
                "war", "announcement", "新闻", "最新", "今天", "突发"]) {
            result.insert("news")
        }
        if hit(["book", "novel", "author", "isbn", "literature", "ebook",
                "poetry", "图书", "小说", "作者", "书"]) {
            result.insert("book")
        }
        if hit(["movie", "film", "tv", "show", "anime", "manga", "music", "song",
                "artist", "album", "video", "trailer", "podcast", "电影", "剧",
                "动漫", "音乐", "歌手", "专辑", "视频"]) {
            result.insert("media")
        }
        if hit(["weather", "city", "country", "map", "address", "location",
                "near", "travel", "hotel", "restaurant", "天气", "城市", "地图",
                "地址", "附近", "旅游"]) {
            result.insert("local")
        }
        if hit(["stock", "crypto", "bitcoin", "ethereum", "finance", "coin",
                "ticker", "股票", "加密", "币", "金融"]) {
            result.insert("finance")
        }
        if hit(["dataset", "model", "llm", "benchmark", "corpus", "数据集",
                "模型", "语料"]) {
            result.insert("data")
        }
        if result.isEmpty {
            // 没有明确意图时优先通用/百科/新闻；不要默认塞开发者与包仓库，
            // 否则“法国首都”这类查询会被 npm/StackOverflow 占据备用名额。
            result.formUnion(["general", "reference", "news"])
        }
        return result
    }

    private static func isCJK(_ text: String) -> Bool {
        text.unicodeScalars.contains {
            let value = Int($0.value)
            return (0x4E00...0x9FFF).contains(value)
                || (0x3400...0x4DBF).contains(value)
                || (0xF900...0xFAFF).contains(value)
        }
    }
}
