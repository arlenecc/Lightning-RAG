import Foundation

// MARK: - 大纲抽取

/// 从正文识别标题层级，生成目录大纲。
/// 识别规则对齐原项目 document_outline.py：Markdown 标题、中文章节、数字编号、中文序号。
public enum DocumentOutline {

    /// 标题模式表（multiline）。还原器通过 `classify(_:)` 复用同一组规则。
    static let headingPatterns: [(String, NSRegularExpression.Options,
                                  (NSTextCheckingResult, NSString) -> (Int, String)?)] = [
        ("^[ \\t]{0,3}(#{1,6})[ \\t]+(.+?)[ \\t]*$", [.anchorsMatchLines], { match, s in
            let hashes = s.substring(with: match.range(at: 1))
            let title = s.substring(with: match.range(at: 2))
                .trimmingCharacters(in: .whitespaces)
            guard !title.isEmpty else { return nil }
            // VLM/OCR 常把正文句子写成 `##` 标题；句末标点是最可靠的排除信号。
            guard !Self.hasSentencePunctuation(title) else { return nil }
            // `# 2 / 164`、`# 164`、`# 1898 年，…` 是 OCR 页码/年份正文，不是标题。
            guard !Self.isPageHeaderLine(title),
                  !Self.isBareNumberTitle(title),
                  !Self.looksLikeDateProse(title) else { return nil }
            return (hashes.count, title)
        }),
        ("^[ \\t]*(第[一二三四五六七八九十百千零〇0-9]+[章节篇部卷])(?:[ \\t：:、.．]+(.*))?[ \\t]*$",
         [.anchorsMatchLines], { match, s in
            let marker = s.substring(with: match.range(at: 1))
            let title = match.range(at: 2).location == NSNotFound
                ? "" : s.substring(with: match.range(at: 2))
                    .trimmingCharacters(in: .whitespaces)
            // 正文句子也常以「第二章介绍方法。」开头，必须排除句末标点。
            guard Self.isPlausibleTitle(title) else { return nil }
            let level = marker.contains("节") ? 2 : 1
            return (level, title.isEmpty ? marker : "\(marker) \(title)")
        }),
        // 英文书 Chapter / Section / Part（含罗马数字与多级编号）。
        ("^[ \\t]{0,3}(chapter|section|part)[ \\t]+([0-9]{1,3}(?:\\.[0-9]{1,3}){0,3}|[ivxlcdm]{1,8})\\b"
            + "(?:[.:：)）\\]]?[ \\t]+(.*))?[ \\t]*$",
         [.anchorsMatchLines, .caseInsensitive], { match, s in
            let word = s.substring(with: match.range(at: 1)).lowercased()
            let number = s.substring(with: match.range(at: 2))
            let title = match.range(at: 3).location == NSNotFound
                ? "" : s.substring(with: match.range(at: 3))
                    .trimmingCharacters(in: .whitespaces)
            guard Self.isPlausibleTitle(title) else { return nil }
            let marker = word.prefix(1).uppercased() + word.dropFirst() + " " + number
            let level: Int
            if number.contains(".") {
                level = min(4, number.split(separator: ".").count)
            } else {
                level = word == "section" ? 2 : 1
            }
            return (level, title.isEmpty ? marker : "\(marker) \(title)")
        }),
        ("^[ \\t]*(\\d+(?:\\.\\d+)*)[、.． \\t]+(\\S.+)$", [.anchorsMatchLines], { match, s in
            let number = s.substring(with: match.range(at: 1))
            let title = s.substring(with: match.range(at: 2))
                .trimmingCharacters(in: .whitespaces)
            guard title.count <= 40, !title.contains("|"),
                  !title.hasPrefix("/"), !title.hasPrefix("／"),
                  !title.contains("，"), !title.contains(","),
                  !title.hasPrefix("年"), !title.hasPrefix("月"),
                  !title.hasPrefix("日"), !title.hasPrefix("时"),
                  !Self.looksLikeDateProse("\(number) \(title)"),
                  !Self.hasSentencePunctuation(title) else { return nil }
            // 纯 3 位以上数字（年份/页码/正文编号）不是章节号；
            // 带点号的层级编号（1.2.3）仍保留。
            if !number.contains("."), number.count >= 3 { return nil }
            // 表格/统计数据行（「23 大土地所有者 385 000 19 250 000」）
            // 带多个独立数字组，不是标题。
            let digitGroups = title.components(
                separatedBy: CharacterSet.decimalDigits.inverted
            ).filter { !$0.isEmpty }
            guard digitGroups.count <= 1 else { return nil }
            return (min(6, number.split(separator: ".").count), "\(number) \(title)")
        }),
        ("^[ \\t]*([一二三四五六七八九十]+)[、.．][ \\t]*(\\S.+)$", [.anchorsMatchLines], { match, s in
            let marker = s.substring(with: match.range(at: 1))
            let title = s.substring(with: match.range(at: 2))
                .trimmingCharacters(in: .whitespaces)
            guard title.count <= 60, !title.contains("|"),
                  !Self.hasSentencePunctuation(title) else { return nil }
            return (1, "\(marker)、\(title)")
        }),
        // 中文常用括号序号：（一）/（1）。孤立序号与页码脚注（「（3）上」）
        // 无信息量，要求标题文本至少 2 字。
        ("^[ \\t]*[（(]([一二三四五六七八九十]{1,3}|\\d{1,3})[)）][ \\t]*(\\S.*)?[ \\t]*$",
         [.anchorsMatchLines], { match, s in
            let marker = s.substring(with: match.range(at: 1))
            let title = match.range(at: 2).location == NSNotFound
                ? "" : s.substring(with: match.range(at: 2))
                    .trimmingCharacters(in: .whitespaces)
            guard Self.isPlausibleTitle(title), title.count >= 2 else { return nil }
            return (1, "(\(marker)) \(title)")
        }),
    ]

    /// 结构化文档只信任显式 Markdown 标题与语义章节标记（第X章 / Chapter）。
    ///
    /// 数字编号、中文序号、括号序号在 Word/PPT/HTML 正文里通常是列表或条款
    /// （「1. 背景」「一、组织分工」「（1）步骤」），不是章节标题。
    static let explicitHeadingPatterns = Array(headingPatterns.prefix(3))

    /// 这些格式的解析器会把真实标题输出成 Markdown `#`，正文编号不应再当标题。
    private static let explicitHeadingExtensions: Set<String> = [
        "doc", "docx", "xls", "xlsx", "ppt", "pptx",
        "html", "htm", "xhtml", "epub",
    ]

    static func usesExplicitHeadings(_ source: String) -> Bool {
        let ext = (source as NSString).pathExtension.lowercased()
        return explicitHeadingExtensions.contains(ext)
    }

    /// 单行标题分类：命中返回（层级, 标题）。供纯文本结构还原器复用。
    static func classify(_ line: String) -> (level: Int, title: String)? {
        // 页码/页眉行（含 `# 2 / 164` 这类伪标题）不是文档结构。
        guard !isPageHeaderLine(line) else { return nil }
        let s = line as NSString
        for (pattern, options, handler) in headingPatterns {
            guard let regex = RegexCache.shared.regex(pattern, options: options),
                  let match = regex.firstMatch(in: line, options: [],
                                               range: NSRange(location: 0, length: s.length)),
                  let parsed = handler(match, s) else { continue }
            return parsed
        }
        return nil
    }

    public static func extract(_ text: String, source: String = "") -> [OutlineNode] {
        guard !text.isEmpty else { return [] }
        let ns = text as NSString
        let length = ns.length
        var candidates: [(location: Int, level: Int, title: String)] = []
        // 代码围栏内的 `#` / 编号不是标题，必须跳过。
        let fencedRanges = codeFenceRanges(text)

        let patterns = usesExplicitHeadings(source) ? explicitHeadingPatterns : headingPatterns
        for (pattern, options, handler) in patterns {
            guard let regex = RegexCache.shared.regex(pattern, options: options) else { continue }
            regex.enumerateMatches(in: text, options: [], range: NSRange(location: 0, length: length)) { match, _, _ in
                guard let match, let parsed = handler(match, ns) else { return }
                if fencedRanges.contains(where: { NSLocationInRange(match.range.location, $0) }) {
                    return
                }
                // 行内去重：同一位置只保留第一个命中的规则。
                if candidates.contains(where: { $0.location == match.range.location }) { return }
                candidates.append((match.range.location, parsed.0, parsed.1))
            }
        }

        guard !candidates.isEmpty else { return [] }
        candidates.sort { $0.location < $1.location }

        var nodes: [OutlineNode] = []
        for (i, item) in candidates.enumerated() {
            let end = i + 1 < candidates.count ? candidates[i + 1].location : length
            nodes.append(OutlineNode(level: max(1, item.level),
                                     title: item.title,
                                     start: item.location,
                                     end: end))
        }
        return nodes
    }

    /// 供 rag_outline 与系统提示使用的大纲文本。
    public static func render(_ nodes: [OutlineNode], maxChars: Int = 4000) -> String {
        guard !nodes.isEmpty else { return "" }
        var lines: [String] = []
        var used = 0
        for node in nodes {
            let line = String(repeating: "  ", count: max(0, node.level - 1)) + node.title
            if used + line.count + 1 > maxChars {
                lines.append("…（大纲已截断）")
                break
            }
            lines.append(line)
            used += line.count + 1
        }
        return lines.joined(separator: "\n")
    }

    /// 页码/页眉行：`2 / 164`、`第 2 页`、`Page 3 of 10`，以及 OCR 把页码
    /// 写成 `# 2 / 164` 的伪标题。这类行不是文档结构，不能生成章节。
    static func isPageHeaderLine(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, trimmed.count <= 40 else { return false }
        let range = NSRange(trimmed.startIndex..<trimmed.endIndex, in: trimmed)
        return pageHeaderRegexes.contains { $0.firstMatch(in: trimmed, range: range) != nil }
    }

    /// 纯数字标题（`# 164`）：OCR 常把页码单独写成一行 Markdown 标题。
    static func isBareNumberTitle(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, trimmed.count <= 5 else { return false }
        return trimmed.allSatisfy(\.isNumber)
    }

    /// 年份/日期正文行（`1898 年，拜耶林克…`）：4 位年份 + 年/月/日开头，
    /// 是 OCR 正文而不是章节标题。
    static func looksLikeDateProse(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, trimmed.count <= 60 else { return false }
        return dateProseRegex.firstMatch(
            in: trimmed,
            range: NSRange(trimmed.startIndex..<trimmed.endIndex, in: trimmed)) != nil
    }

    /// 标题不应包含句末标点（正文句子常以「第X章…。」开头，会被误判为标题）。
    static func hasSentencePunctuation(_ title: String) -> Bool {
        title.contains("。") || title.contains("！") || title.contains("？")
            || title.contains("；")
    }

    /// 「第X章/Chapter/（一）」行的整体可读性校验。
    /// 含逗号的标题（「第一章 引言，背景」）保留，但只允许短行，
    /// 避免把「第二章开始，我们讨论了……」这类正文句子误判为标题。
    private static func isPlausibleTitle(_ title: String) -> Bool {
        guard !title.isEmpty else { return true }
        guard title.count <= 80, !hasSentencePunctuation(title) else { return false }
        if title.contains("，") || title.contains(",") {
            return title.count <= 30
        }
        return true
    }

    /// 页码/页眉行的完整匹配（允许 `#` 标题前缀，OCR 常把页码写成 Markdown 标题）。
    private static let pageHeaderRegexes: [NSRegularExpression] = [
        try! NSRegularExpression(
            pattern: #"^[ \t]{0,3}(?:#{1,6}[ \t]+)?(?:\d{1,5}[ \t]*[/／][ \t]*\d{1,5}(?:[ \t]*页)?|第[ \t]*\d{1,5}[ \t]*页(?:[ \t]*(?:[/／][ \t]*)?共?[ \t]*\d{1,5}[ \t]*页)?|page[ \t]*#?[ \t]*\d{1,5}(?:[ \t]*(?:of|/|／)[ \t]*\d{1,5})?|p\.[ \t]*\d{1,5}|[-–—][ \t]*\d{1,5}[ \t]*[-–—])[ \t]*$"#,
            options: [.caseInsensitive]),
    ]

    private static let dateProseRegex = try! NSRegularExpression(
        pattern: #"^(?:1[0-9]{3}|20[0-9]{2})[ \t]*[年月日]"#)

    /// Markdown 代码围栏范围（``` 与 ~~~），用于排除围栏内的伪标题。
    private static func codeFenceRanges(_ text: String) -> [NSRange] {
        let pattern = "(?m)^[ \\t]*(?:```|~~~)[^\\n]*\\n[\\s\\S]*?^[ \\t]*(?:```|~~~)[ \\t]*$"
        guard let regex = RegexCache.shared.regex(pattern) else { return [] }
        let full = NSRange(location: 0, length: (text as NSString).length)
        return regex.matches(in: text, options: [], range: full).map(\.range)
    }
}

// MARK: - 章节结构

struct DocumentSection {
    var heading: String
    var text: String
}

enum SectionSplitter {
    static func sections(_ text: String, nodes: [OutlineNode]) -> [DocumentSection] {
        guard !text.isEmpty else { return [] }
        let ns = text as NSString
        guard !nodes.isEmpty else {
            return [DocumentSection(heading: "", text: text)]
        }

        var sections: [DocumentSection] = []
        var stack: [(level: Int, title: String)] = []

        // 第一个标题之前的内容作为前言。
        if nodes[0].start > 0 {
            let prefix = ns.substring(with: NSRange(location: 0, length: nodes[0].start))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !prefix.isEmpty {
                sections.append(DocumentSection(heading: "", text: prefix))
            }
        }

        for node in nodes {
            while let last = stack.last, last.level >= node.level {
                stack.removeLast()
            }
            stack.append((node.level, node.title))
            let path = stack.map(\.title).joined(separator: " > ")
            let length = max(0, min(node.end, ns.length) - node.start)
            guard length > 0 else { continue }
            let body = ns.substring(with: NSRange(location: node.start, length: length))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !body.isEmpty else { continue }
            sections.append(DocumentSection(heading: path, text: body))
        }
        return sections
    }
}
