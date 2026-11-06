import Foundation
import SwiftSoup

/// HTML / XHTML → Markdown 风格正文。EPUBParser 复用同一套转换逻辑。
enum HTMLMarkdown {

    /// 转换整篇文档，并返回标题大纲。
    ///
    /// - Parameter includeSVGText: 提取 `<svg><text>` 里的文字。固定版式 EPUB
    ///   常把整页正文画在 SVG 文本节点里，默认跳过 svg 会只剩页码/目录。
    static func convert(_ document: Document,
                        removeBoilerplate: Bool,
                        includeSVGText: Bool = false) -> (markdown: String, headings: [OutlineEntry]) {
        guard let root = document.body() ?? document.children().first() else {
            return ("", [])
        }

        // 网页解析去掉导航/页脚；EPUB 只去掉脚本样式，避免误删章节页眉内容。
        let selectors = removeBoilerplate
            ? "script,style,noscript,nav,footer"
            : "script,style,noscript"
        _ = try? root.select(selectors).remove()
        if removeBoilerplate {
            // `<header>` 常包着文章主标题（h1）与作者信息，整块删除会丢标题；
            // 只删不含标题的页头（logo / 导航条）。
            for header in ((try? root.select("header").array()) ?? []) {
                let hasHeading = !((try? header.select("h1,h2,h3").array()) ?? []).isEmpty
                if !hasHeading { _ = try? header.remove() }
            }
        }

        var blocks: [String] = []
        var headings: [OutlineEntry] = []
        renderChildren(root, blocks: &blocks, headings: &headings,
                       listDepth: 0, includeSVGText: includeSVGText)

        var markdown = blocks
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .joined(separator: "\n\n")
        markdown = markdown.replacingOccurrences(of: "\n{3,}", with: "\n\n",
                                                options: .regularExpression)
        return (markdown.trimmingCharacters(in: .whitespacesAndNewlines), headings)
    }

    // MARK: - 块级元素

    private static func renderChildren(_ element: Element,
                                       blocks: inout [String],
                                       headings: inout [OutlineEntry],
                                       listDepth: Int,
                                       includeSVGText: Bool) {
        // 只遍历元素会丢掉裸文本（手写 HTML、导出页常把正文直接写在
        // `<body>`/`<div>` 下而没有 `<p>` 包裹），按节点顺序逐个处理。
        for node in element.getChildNodes() {
            if let textNode = node as? TextNode {
                let text = normalizeInlineWhitespace(textNode.text())
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty { blocks.append(text) }
            } else if let child = node as? Element {
                renderBlock(child, blocks: &blocks, headings: &headings,
                            listDepth: listDepth, includeSVGText: includeSVGText)
            }
        }
    }

    private static func renderBlock(_ element: Element,
                                    blocks: inout [String],
                                    headings: inout [OutlineEntry],
                                    listDepth: Int,
                                    includeSVGText: Bool) {
        let tag = element.tagName().lowercased()

        switch tag {
        case "h1", "h2", "h3", "h4", "h5", "h6":
            let level = Int(tag.dropFirst()) ?? 1
            let text = inlineText(of: element).trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty {
                blocks.append(String(repeating: "#", count: level) + " " + text)
                headings.append(OutlineEntry(level: level, title: text))
            }

        case "p", "address", "figcaption":
            if let heading = boldOnlyHeading(of: element) {
                blocks.append(heading.markdown)
                headings.append(OutlineEntry(level: heading.level, title: heading.title))
            } else {
                let text = inlineText(of: element).trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty { blocks.append(text) }
            }

        case "ul", "ol":
            let list = renderList(element, ordered: tag == "ol", depth: listDepth)
            if !list.isEmpty { blocks.append(list) }

        case "table":
            let table = renderTable(element)
            if !table.isEmpty { blocks.append(table) }

        case "pre":
            let code = renderPre(element)
            if !code.isEmpty { blocks.append(code) }

        case "blockquote":
            var inner: [String] = []
            var ignoredHeadings: [OutlineEntry] = []
            renderChildren(element, blocks: &inner,
                           headings: &ignoredHeadings, listDepth: listDepth,
                           includeSVGText: includeSVGText)
            let text = inner.joined(separator: "\n\n")
            if !text.isEmpty {
                blocks.append(prefixLines(text, prefix: "> "))
            }

        case "svg":
            // 固定版式 EPUB 的整页正文常是 `<svg><text>…</text></svg>`；
            // 仅 EPUB 开启，网页/普通 HTML 保持原行为。
            if includeSVGText {
                let lines = svgTextLines(element)
                if !lines.isEmpty { blocks.append(lines.joined(separator: "\n")) }
            }

        case "br", "hr", "script", "style", "noscript", "nav",
             // 文档元信息与嵌入式容器：`<title>` 不是正文（frameset 页面没有
             // body，若不跳过会把标题文本当成正文），框架/多媒体的回退内容
             // 也不是正文——框架正文由本地框架接力单独读取。
             "head", "title", "meta", "link", "base", "template",
             "iframe", "frame", "frameset", "embed", "object", "canvas":
            break

        default:
            if hasBlockChildren(element) {
                renderChildren(element, blocks: &blocks,
                               headings: &headings, listDepth: listDepth,
                               includeSVGText: includeSVGText)
            } else if let heading = boldOnlyHeading(of: element) {
                // Calibre 等工具常用 <div><b>…</b></div> 排版小节标题。
                blocks.append(heading.markdown)
                headings.append(OutlineEntry(level: heading.level, title: heading.title))
            } else {
                let text = inlineText(of: element).trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty { blocks.append(text) }
            }
        }
    }

    private static func hasBlockChildren(_ element: Element) -> Bool {
        element.children().contains { blockTags.contains($0.tagName().lowercased()) }
    }

    /// 识别「整个块只有一个 <b>/<strong> 子元素」的伪标题。
    ///
    /// Calibre 转制的 EPUB 普遍用 `<p><b>…</b></p>` 排版小节标题（正文没有 h 标签），
    /// 不识别这类结构会导致大纲几乎为空。仅当块内无其他可见文本、
    /// 文本短且不含句末标点时才视为标题，避免误伤段内强调。
    private static func boldOnlyHeading(of element: Element) -> (markdown: String,
                                                                  level: Int,
                                                                  title: String)? {
        let children = element.children()
        guard children.count == 1 else { return nil }
        let only = children[0]
        let tag = only.tagName().lowercased()
        guard tag == "b" || tag == "strong" else { return nil }

        // 块内除 <b> 外不允许再有可见文本，否则是「段首强调」而不是标题。
        let leadingText = element.getChildNodes()
            .compactMap { $0 as? TextNode }
            .map { $0.text() }
            .joined()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard leadingText.isEmpty else { return nil }

        let title = rawText(only).trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "[ \\t\\u00A0\\u3000]+", with: " ",
                                  options: .regularExpression)
        guard title.count >= 2, title.count <= 60 else { return nil }
        guard !title.contains("。"), !title.contains("！"), !title.contains("？"),
              !title.contains("；"), !title.contains("，") else { return nil }
        return (markdown: "### \(title)", level: 3, title: title)
    }

    // MARK: - 列表 / 表格 / 代码

    private static func renderList(_ element: Element, ordered: Bool, depth: Int) -> String {
        var lines: [String] = []
        var index = 1
        let indent = String(repeating: "  ", count: max(0, depth))

        for item in element.children() where item.tagName().lowercased() == "li" {
            // 列表项内部允许出现块级子元素（`<li><p>…</p></li>`），但列表项
            // 必须是单行，否则 Markdown 列表会被拆断，这里按行内方式取文本。
            let text = liText(item)
            let prefix = ordered ? "\(index). " : "- "
            lines.append(indent + prefix + text)

            for child in item.children() {
                let tag = child.tagName().lowercased()
                if tag == "ul" || tag == "ol" {
                    let nested = renderList(child, ordered: tag == "ol", depth: depth + 1)
                    if !nested.isEmpty { lines.append(nested) }
                }
            }
            index += 1
        }
        return lines.joined(separator: "\n")
    }

    /// 列表项文本：拼成单行（块级子元素之间用空格分隔、忽略换行），
    /// 跳过嵌套列表。
    private static func liText(_ item: Element) -> String {
        func collect(_ element: Element) -> String {
            var result = ""
            for node in element.getChildNodes() {
                if let textNode = node as? TextNode {
                    result += textNode.text()
                    continue
                }
                guard let child = node as? Element else { continue }
                let tag = child.tagName().lowercased()
                if tag == "ul" || tag == "ol" { continue }
                if tag == "br" {
                    result += " "
                    continue
                }
                // 块级子元素之间补空格，避免「Item one」+「detail」粘成一个词。
                if blockTags.contains(tag) { result += " " }
                result += collect(child)
            }
            return result
        }
        return normalizeInlineWhitespace(
            collect(item).replacingOccurrences(of: "\n", with: " "))
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func renderTable(_ element: Element) -> String {
        var rowElements: [Element] = []
        for child in element.children() {
            let tag = child.tagName().lowercased()
            if tag == "tr" {
                rowElements.append(child)
            } else if tag == "thead" || tag == "tbody" || tag == "tfoot" {
                rowElements.append(contentsOf: child.children().filter {
                    $0.tagName().lowercased() == "tr"
                })
            }
        }

        var rows: [[String]] = []
        for row in rowElements {
            let cells = row.children().filter {
                let tag = $0.tagName().lowercased()
                return tag == "th" || tag == "td"
            }
            rows.append(cells.map { escapeTableCell(inlineText(of: $0)) })
        }
        return renderMarkdownTable(rows)
    }

    private static func renderPre(_ element: Element) -> String {
        let code = rawText(element)
            .replacingOccurrences(of: "\r\n", with: "\n")
            .trimmingCharacters(in: .newlines)
        guard !code.isEmpty else { return "" }

        var language = ""
        if let codeElement = try? element.select("code").first(),
           let classValue = try? codeElement.attr("class"),
           let match = languageRegex.firstMatch(
                in: classValue,
                range: NSRange(classValue.startIndex..<classValue.endIndex, in: classValue)),
           let range = Range(match.range(at: 1), in: classValue) {
            language = String(classValue[range])
        }
        return "```\(language)\n\(code)\n```"
    }

    // MARK: - 行内文本

    private static func inlineText(of element: Element, skipNestedLists: Bool = false) -> String {
        var result = ""
        for node in element.getChildNodes() {
            result += inlineText(of: node, skipNestedLists: skipNestedLists)
        }
        return normalizeInlineWhitespace(result)
    }

    private static func inlineText(of node: Node, skipNestedLists: Bool = false) -> String {
        if let textNode = node as? TextNode {
            return textNode.text()
        }
        guard let element = node as? Element else { return "" }

        let tag = element.tagName().lowercased()
        if skipNestedLists && (tag == "ul" || tag == "ol") { return "" }

        switch tag {
        case "br":
            return "\n"
        case "script", "style", "noscript", "head", "title", "meta", "link",
             "template", "iframe", "frame", "embed", "object", "svg", "canvas":
            return ""
        case "code":
            let code = rawText(element).trimmingCharacters(in: .newlines)
            return code.isEmpty ? "" : "`\(code)`"
        case "strong", "b":
            let inner = inlineText(of: element).trimmingCharacters(in: .whitespacesAndNewlines)
            return inner.isEmpty ? "" : "**\(inner)**"
        case "em", "i":
            let inner = inlineText(of: element).trimmingCharacters(in: .whitespacesAndNewlines)
            return inner.isEmpty ? "" : "*\(inner)*"
        case "img":
            return (try? element.attr("alt")) ?? ""
        case "p", "div", "section", "article", "li", "tr",
             "h1", "h2", "h3", "h4", "h5", "h6":
            let inner = inlineText(of: element)
            return "\n" + inner + "\n"
        default:
            return inlineText(of: element, skipNestedLists: skipNestedLists)
        }
    }

    private static func rawText(_ node: Node) -> String {
        if let textNode = node as? TextNode {
            return textNode.text()
        }
        guard let element = node as? Element else { return "" }
        if element.tagName().lowercased() == "br" { return "\n" }
        return element.getChildNodes().map { rawText($0) }.joined()
    }

    /// 收集 SVG 里 `<text>`（含 `<tspan>`）的文字，按文档顺序逐行返回。
    private static func svgTextLines(_ svg: Element) -> [String] {
        var lines: [String] = []
        func walk(_ element: Element) {
            if element.tagName().lowercased() == "text" {
                let value = inlineText(of: element)
                    .replacingOccurrences(of: "\n", with: " ")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !value.isEmpty { lines.append(value) }
                return
            }
            for child in element.children() { walk(child) }
        }
        walk(svg)
        return lines
    }

    private static func normalizeInlineWhitespace(_ text: String) -> String {
        var result = text.replacingOccurrences(of: "[ \t]+", with: " ",
                                               options: .regularExpression)
        result = result.replacingOccurrences(of: " *\n *", with: "\n",
                                             options: .regularExpression)
        return result.trimmingCharacters(in: .whitespaces)
    }

    // MARK: - Markdown 表格

    static func renderMarkdownTable(_ rows: [[String]]) -> String {
        guard let columnCount = rows.map(\.count).max(), columnCount > 0 else { return "" }
        func line(_ values: [String]) -> String {
            var padded = values
            if padded.count < columnCount {
                padded += Array(repeating: "", count: columnCount - padded.count)
            }
            return "| " + padded.joined(separator: " | ") + " |"
        }
        let header = line(rows.first ?? [])
        let separator = "| " + Array(repeating: "---", count: columnCount).joined(separator: " | ") + " |"
        let body = rows.dropFirst().map(line)
        return ([header, separator] + body).joined(separator: "\n")
    }

    static func escapeTableCell(_ value: String) -> String {
        value.replacingOccurrences(of: "|", with: "\\|")
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func prefixLines(_ text: String, prefix: String) -> String {
        text.components(separatedBy: "\n").map { prefix + $0 }.joined(separator: "\n")
    }

    private static let blockTags: Set<String> = [
        "address", "article", "aside", "blockquote", "details", "div", "dl", "dd", "dt",
        "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4",
        "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section",
        "table", "ul", "tr", "td", "th",
        // 固定版式 EPUB 常用 `<div><svg>…</svg></div>` 包裹整页；
        // 列入块级后 `renderBlock` 才会走进 svg（是否取文本由 includeSVGText 决定）。
        "svg",
    ]

    private static let languageRegex = try! NSRegularExpression(
        pattern: #"(?:language-|lang-)([A-Za-z0-9_+-]+)"#)
}

// MARK: - HTML 解析器

public final class HTMLParser: DocumentParser {

    public let name = "HTML"
    public let supportedExtensions = ["html", "htm", "xhtml"]

    public init() {}

    /// 单次解析的中间结果。
    private struct RenderedHTML {
        var markdown = ""
        var headings: [OutlineEntry] = []
        var title = ""
        /// 正文来自哪个本地框架文件（非空表示发生了框架接力）。
        var embeddedSource: String?
        /// 正文为空时的原因说明。
        var emptyReason = ""
    }

    public func parse(url: URL,
                      options: ParserOptions,
                      progress: ((ParserProgress) -> Void)?) async throws -> ParsedDocument {
        // 框架接力的边界固定为「入口文档所在目录」：浏览器另存为会把正文放在
        // <页面>_files/ 子目录里，但绝不该让外部 HTML 通过 file:// 或 ../
        // 把工作区外（甚至任意本地文件）的内容带进知识库。
        let root = url.standardizedFileURL.deletingLastPathComponent()
        let rendered = try render(url.standardizedFileURL, depth: 0, visited: [], root: root)
        let markdown = rendered.markdown.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !markdown.isEmpty else {
            throw AgentError.parse("HTML 未解析出正文：\(url.lastPathComponent)——\(rendered.emptyReason)")
        }

        var metadata: [String: String] = ["engine": "html"]
        if !rendered.title.isEmpty { metadata["title"] = rendered.title }
        if let embedded = rendered.embeddedSource {
            metadata["embedded_source"] = embedded
        }

        var parsed = ParsedDocument(text: markdown,
                                    source: url.lastPathComponent,
                                    outline: MarkdownOutline.build(rendered.headings),
                                    metadata: metadata)
        parsed = finish(parsed, options: options, profile: .structured)
        return parsed
    }

    // MARK: - 渲染（含本地 frame 接力）

    /// 解析单个 HTML 文件。页面正文过短时，自动接力到被保存网页同目录下的
    /// 本地 `iframe` / `frame` / `object` 内容（浏览器「网页另存为」会把正文
    /// 框架单独存成 `<页面>_files/N.html`，只读主文件会得到空正文）。
    private func render(_ url: URL, depth: Int, visited: Set<String>,
                        root: URL) throws -> RenderedHTML {
        guard let data = try? Data(contentsOf: url) else {
            throw AgentError.parse("无法读取文件：\(url.lastPathComponent)")
        }
        let html = DocumentTextDecoder.decodeHTML(data)

        let document: Document
        do {
            document = try SwiftSoup.parse(html, url.absoluteString)
        } catch {
            throw AgentError.parse("HTML 解析失败（\(url.lastPathComponent)）：\(error.localizedDescription)")
        }

        var result = RenderedHTML()
        result.title = (try? document.title())?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let (markdown, headings) = HTMLMarkdown.convert(document, removeBoilerplate: true)
        result.markdown = markdown
        result.headings = headings

        // 正文过短：优先从本地框架文件里取真正的正文。
        if depth < Self.maxFrameDepth, markdown.count < Self.frameFallbackThreshold,
           let embedded = renderEmbedded(document, base: url, currentLength: markdown.count,
                                         depth: depth, visited: visited, root: root),
           !embedded.markdown.isEmpty {
            var merged = embedded
            if merged.title.isEmpty { merged.title = result.title }
            return merged
        }

        if markdown.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            result.emptyReason = emptyReason(document)
        }
        return result
    }

    /// 在候选的本地框架文档里挑正文最多的一份。
    private func renderEmbedded(_ document: Document, base: URL, currentLength: Int,
                                depth: Int, visited: Set<String>,
                                root: URL) -> RenderedHTML? {
        var visited = visited
        visited.insert(base.standardizedFileURL.path)

        var references: [String] = []
        for selector in Self.frameSelectors {
            for element in ((try? document.select(selector).array()) ?? []) {
                let value = selector.contains("object")
                    ? (try? element.attr("data")) ?? ""
                    : (try? element.attr("src")) ?? ""
                if !value.isEmpty { references.append(value) }
                if references.count >= Self.maxFrameCandidates { break }
            }
        }

        var best: RenderedHTML?
        var bestSource: String?
        for reference in references.prefix(Self.maxFrameCandidates) {
            guard let target = localFrameURL(reference, base: base, root: root),
                  !visited.contains(target.path),
                  FileManager.default.fileExists(atPath: target.path) else { continue }
            do {
                var candidate = try render(target, depth: depth + 1, visited: visited,
                                           root: root)
                candidate.markdown = candidate.markdown
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !candidate.markdown.isEmpty else { continue }
                if best == nil || candidate.markdown.count > (best?.markdown.count ?? 0) {
                    best = candidate
                    bestSource = target.lastPathComponent
                }
            } catch {
                // 单个框架文件解析失败不影响其他候选。
                continue
            }
        }

        guard let best, best.markdown.count > currentLength else { return nil }
        var result = best
        result.embeddedSource = bestSource
        return result
    }

    /// 框架引用 → 本地文件 URL：只接受本地路径（不联网抓取），
    /// 忽略 data:/javascript:/about: 与 http(s) 引用。
    private func localFrameURL(_ reference: String, base: URL, root: URL) -> URL? {
        let cleaned = RAGArchivePath.stripFragmentAndQuery(reference)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { return nil }
        let lowered = cleaned.lowercased()
        for scheme in ["http://", "https://", "data:", "javascript:", "about:", "blob:",
                       "mailto:", "ftp://", "//"] where lowered.hasPrefix(scheme) {
            return nil
        }
        let decoded = cleaned.removingPercentEncoding ?? cleaned
        let extensionName = (decoded as NSString).pathExtension.lowercased()
        guard Self.frameExtensions.contains(extensionName) else { return nil }

        let candidate: URL
        if decoded.lowercased().hasPrefix("file://") {
            guard let url = URL(string: decoded), url.isFileURL else { return nil }
            candidate = url.standardizedFileURL
        } else {
            let baseDirectory = base.deletingLastPathComponent()
            candidate = URL(fileURLWithPath: decoded,
                            relativeTo: baseDirectory).standardizedFileURL
        }
        // 绝对路径、../ 上跳、符号链接都不得越出入口文档所在目录；
        // 用 realpath 比较，避免用 symlink 指向工作区外的文件。
        let resolved = candidate.resolvingSymlinksInPath().standardizedFileURL.path
        let rootPath = root.resolvingSymlinksInPath().standardizedFileURL.path
        let prefix = rootPath.hasSuffix("/") ? rootPath : rootPath + "/"
        guard resolved.hasPrefix(prefix) else { return nil }
        return candidate
    }

    /// 空正文的原因说明：区分「正文在框架里」「依赖 JavaScript 渲染」与真空白页。
    private func emptyReason(_ document: Document) -> String {
        let frames = (try? document.select("iframe,frame,object,embed").array().count) ?? 0
        if frames > 0 {
            return "正文在被保存网页的本地 iframe/frame 中，但未找到可读的本地文件（共 \(frames) 个框架引用）"
        }
        let scripts = (try? document.select("script").array().count) ?? 0
        // 只看 body：<title> 与 <meta> 的文本不算正文。
        let bodyText = ((try? document.body()?.text()) ?? "")?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if bodyText.isEmpty && scripts > 0 {
            return "页面依赖 JavaScript 渲染，静态解析无正文（含 \(scripts) 个脚本）"
        }
        if bodyText.isEmpty {
            return "页面没有可提取的文本内容"
        }
        return "页面文本全部被判定为导航/样式噪声"
    }

    // MARK: - 常量

    private static let frameSelectors = ["iframe", "frame", "object"]
    /// 只接力文档类型框架文件，避免把图片/脚本当成正文解析。
    private static let frameExtensions: Set<String> = [
        "html", "htm", "xhtml", "shtml", "php", "asp", "aspx", "jsp", "txt",
    ]
    /// 正文短于该长度时尝试框架接力。
    private static let frameFallbackThreshold = 400
    private static let maxFrameDepth = 2
    private static let maxFrameCandidates = 8
}
