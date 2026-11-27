import Foundation

/// 清洗档位：决定用「OCR 噪声清理」还是「结构化文档保真」的口径。
///
/// PDF/图片 OCR 的文本层天然含大量噪声（断行、页码、装饰符号、URL 串），
/// 需要激进清洗；而现代 Office / HTML / EPUB / CSV 等结构化格式的文本
/// 是程序生成的，属于真实内容（URL、IP、编号、统计数字），不能按噪声删除。
public enum DocumentProfile: Sendable {
    /// OCR / 扫描件文本层：激进去噪。
    case ocr
    /// 结构化文档（现代 Office、HTML、EPUB、文本资料）：保真，只清理排版噪声。
    case structured

    /// 按扩展名推断默认档位（解析器未显式指定时的兜底）。
    public static func defaultProfile(for url: URL) -> DocumentProfile {
        switch url.pathExtension.lowercased() {
        case "html", "htm", "xhtml", "epub",
             "docx", "xlsx", "pptx", "doc", "xls", "ppt",
             "csv", "tsv", "json", "jsonl", "md", "markdown", "rst", "tex", "latex":
            return .structured
        default:
            return .ocr
        }
    }
}

/// 文本清洗：对应 Python `agent/rag/cleaner.py`。
///
/// 处理顺序（与原项目保持一致）：
/// 1. 暂存围栏代码块，避免破坏 Markdown 代码；
/// 2. 按页标记切分页面，跨页识别重复页眉/页脚；
/// 3. 去除控制字符、HTML 标签/实体、URL/邮箱、页码与噪声行；
/// 4. 合并硬换行、归一化空白、压缩连续空行。
///
/// 页面标记统一使用 `<!-- page N -->`（N 从 1 开始），由解析器写入，
/// 仅用于清洗阶段识别跨页页眉/页脚，最终文本与切片中不保留页码。
public enum DocumentCleaner {

    // MARK: - 公开 API

    /// 对解析器输出的正文做统一清洗。
    /// - Parameters:
    ///   - keepHeadersFooters: 为 true 时保留页眉页脚（仍会清理页码与空白）。
    ///   - markdown: 是否按 Markdown 处理；nil 时按内容自动识别。Markdown 模式会
    ///     保留标题、列表、链接、表格、引用、分隔线与代码，只清理 HTML/噪声。
    ///   - profile: 清洗档位。`.structured` 不做「URL/邮箱删除、散落数据行删除、
    ///     短行删除、硬换行合并」等面向 OCR 的破坏性处理，并保留段落空行。
    public static func clean(_ text: String, keepHeadersFooters: Bool = false,
                             markdown: Bool? = nil,
                             profile: DocumentProfile = .ocr) -> String {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return "" }

        let normalized = normalizeLineEndings(text)
        let detectedMarkdown = markdown ?? looksLikeMarkdown(normalized)
        let (stashedFences, fencedBlocks) = stashFencedBlocks(normalized)
        // 行内代码也要先保护，避免其中的 HTML/符号被当成正文清理。
        let (stashedInline, inlineBlocks) = stashInlineCode(
            stashedFences, startingIndex: fencedBlocks.count)
        // HTML 表格必须在通用标签清理之前处理：简单表格转 Markdown，
        // 复杂表格保留原始 HTML，用占位符避开后续标签清理。
        let (converted, hadTables, retainedTables) = convertHTMLTables(
            stashedInline, startingIndex: fencedBlocks.count + inlineBlocks.count)
        let codeBlocks = fencedBlocks + inlineBlocks + retainedTables
        let useMarkdown = detectedMarkdown || hadTables
        let segments = splitPages(converted)

        // 每页先做字符/标签级清理，再做跨页页眉页脚分析。
        var pageLines: [[String]] = []
        pageLines.reserveCapacity(segments.count)
        for segment in segments {
            pageLines.append(sanitize(segment.body, markdown: useMarkdown, profile: profile)
                .components(separatedBy: "\n"))
        }

        var headers: Set<String> = []
        var footers: Set<String> = []
        if !keepHeadersFooters {
            (headers, footers) = detectRepeatedMargins(pageLines)
        }

        var output: [String] = []
        output.reserveCapacity(segments.count * 2)
        for pageIndex in segments.indices {
            var lines = pageLines[pageIndex]
            if !keepHeadersFooters {
                lines = removeMargins(lines, headers: headers, footers: footers)
            }
            lines = filterGarbageLines(lines, markdown: useMarkdown, profile: profile)
            // 符号残渣在噪声行判定之后清理：先保留原始密度，避免把本该整行
            // 丢弃的 OCR 噪声洗成「看似正常」的短行。
            if profile == .ocr {
                lines = lines.map {
                    replace(noiseSymbolRunRegex, in: $0, with: " ")
                }
            }
            // 结构化档不合并硬换行：Office/HTML 的段内换行（w:br / <br>）是
            // 作者排版意图，合并后标题与正文会粘成一行。
            let joined = profile == .ocr
                ? joinHardWrappedLines(lines.joined(separator: "\n"), preserveMarkdown: useMarkdown)
                : lines.joined(separator: "\n").trimmingCharacters(in: .newlines)
            let pageText = joined.trimmingCharacters(in: .whitespacesAndNewlines)

            // 页标记只供本函数内部做跨页页眉页脚分析，不写入最终文本：
            // 溯源只需到文件（source），不再记录页码。
            if !pageText.isEmpty {
                output.append(pageText)
            }
        }

        var result = output.joined(separator: "\n\n")
        // 密度压缩放在还原代码/HTML 表格之前：占位符是单行，不会被误伤；
        // 代码块与保留表格内部的空行、缩进保持原样。
        result = normalizeDensity(result, profile: profile)
        result = restoreFencedBlocks(result, blocks: codeBlocks)
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// 压缩不必要的空行/行尾空格，同时保留 Markdown 结构所需的分隔。
    ///
    /// - 页标记、列表项、表格行、引用行之间不保留空行；
    /// - 短行（通常来自 OCR 逐行排版）之间不保留空行，长段落之间保留一个；
    /// - 标题、代码、保留的 HTML 表格等结构块前后保留一个空行。
    private static func normalizeDensity(_ text: String, profile: DocumentProfile) -> String {
        let rawLines = text.components(separatedBy: "\n")
        var lines: [String] = []
        lines.reserveCapacity(rawLines.count)
        for raw in rawLines {
            let noTrailing = trimTrailingHorizontalWhitespace(raw)
            lines.append(noTrailing.trimmingCharacters(in: .whitespaces).isEmpty
                ? "" : noTrailing)
        }

        var output: [String] = []
        output.reserveCapacity(lines.count)
        var blankRun = 0
        for line in lines {
            if line.isEmpty {
                blankRun += 1
                continue
            }
            if blankRun > 0 {
                let previous = output.last ?? ""
                if keepBlankLine(previous: previous, next: line, profile: profile),
                   output.last?.isEmpty == false {
                    output.append("")
                }
                blankRun = 0
            }
            output.append(line)
        }
        while output.last?.isEmpty == true { output.removeLast() }
        return output.joined(separator: "\n")
    }

    private static func keepBlankLine(previous: String, next: String,
                                      profile: DocumentProfile) -> Bool {
        guard !previous.isEmpty else { return false }
        let p = previous.trimmingCharacters(in: .whitespaces)
        let n = next.trimmingCharacters(in: .whitespaces)
        if isPageMarker(p) || isPageMarker(n) { return false }

        let pStructural = isDensityStructural(previous)
        let nStructural = isDensityStructural(next)
        if pStructural && nStructural {
            // 同一列表/表格/引用的连续行不需要空行。
            return !sameDensityBlockKind(p, n)
        }
        if !pStructural && !nStructural {
            // 结构化文档的相邻行是独立段落（短标题、条目、字段），
            // 空行是段落边界，必须保留；OCR 短行则是硬换行排版，需要合并。
            if profile == .structured { return true }
            // 短行多为 OCR 逐行排版，合并以提高密度；长段落之间保留一个空行。
            return p.count > 60 && n.count > 60
        }
        return true
    }

    private static func isDensityStructural(_ line: String) -> Bool {
        if line.hasPrefix(codePlaceholderPrefix) { return true }
        if isPageMarker(line) { return true }
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if isMarkdownHorizontalRule(trimmed) { return true }
        return isStructuralLine(line)
    }

    private static func sameDensityBlockKind(_ lhs: String, _ rhs: String) -> Bool {
        if lhs.hasPrefix("|"), rhs.hasPrefix("|") { return true }
        if lhs.hasPrefix(">"), rhs.hasPrefix(">") { return true }
        if let left = densityListKind(lhs), let right = densityListKind(rhs) {
            return left == right
        }
        return false
    }

    /// 无正则地裁剪行尾空格/制表符（批量处理数百万行时避免正则构造）。
    private static func trimTrailingHorizontalWhitespace(_ line: String) -> String {
        var end = line.endIndex
        while end > line.startIndex {
            let before = line.index(before: end)
            let c = line[before]
            if c == " " || c == "\t" { end = before } else { break }
        }
        return String(line[line.startIndex..<end])
    }

    private static func densityListKind(_ line: String) -> String? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        // 无正则判定列表前缀：[-*+] 后跟空白，或 1~3 位数字后跟 . / ) 与空白。
        if let first = trimmed.first, ["-", "*", "+"].contains(first),
           trimmed.count > 1, trimmed[trimmed.index(after: trimmed.startIndex)].isWhitespace {
            return "ul"
        }
        var idx = trimmed.startIndex
        var digits = 0
        while idx < trimmed.endIndex, digits < 3, trimmed[idx].isNumber {
            digits += 1
            idx = trimmed.index(after: idx)
        }
        if digits > 0, idx < trimmed.endIndex,
           (trimmed[idx] == "." || trimmed[idx] == ")") {
            let after = trimmed.index(after: idx)
            if after < trimmed.endIndex, trimmed[after].isWhitespace { return "ol" }
        }
        return nil
    }

    /// 页面标记（PDF 解析器与清洗器共用）。
    public static func pageMarker(_ oneBasedIndex: Int) -> String {
        "<!-- page \(oneBasedIndex) -->"
    }

    public static func isPageMarker(_ line: String) -> Bool {
        pageMarkerRegex.firstMatch(in: line, range: NSRange(line.startIndex..<line.endIndex, in: line)) != nil
    }

    /// 统计有效字符数（忽略所有空白），用于判断 PDF 文本层是否可用。
    public static func effectiveCharacterCount(_ text: String) -> Int {
        text.unicodeScalars.reduce(into: 0) { count, scalar in
            if !CharacterSet.whitespacesAndNewlines.contains(scalar) {
                count += 1
            }
        }
    }

    /// 跨页重复的页眉/页脚行键（与 `clean` 内部判定规则一致）。
    ///
    /// PDF 文本层可用性判断需要它：扫描书常把「每页相同的水印 + 网址」
    /// 写进文本层，原始字数够，但清洗时整行会被当作页眉页脚删除。
    static func repeatedMarginKeys(in pages: [String]) -> Set<String> {
        guard pages.count >= 2 else { return [] }
        let pageLines = pages.map { $0.components(separatedBy: "\n") }
        let (headers, footers) = detectRepeatedMargins(pageLines)
        return headers.union(footers)
    }

    /// 文本层用于 OCR 路由的有效字符数。
    ///
    /// 与 `effectiveCharacterCount` 的区别：排除跨页重复页眉/页脚、
    /// 同页完全重复的行，以及 URL/邮箱。这样「每页只有水印」的文字层
    /// 会被判为不可用，交给 OCR；正常正文页（即使带页眉页脚）不受影响。
    static func effectiveOCRTextCharacterCount(_ text: String,
                                               excludingMarginKeys marginKeys: Set<String>) -> Int {
        guard !text.isEmpty else { return 0 }
        let lines = text.components(separatedBy: "\n")
        let nonEmptyIndices = lines.indices.filter {
            !lines[$0].trimmingCharacters(in: .whitespaces).isEmpty
        }
        let headerZone = Set(nonEmptyIndices.prefix(3))
        let footerZone = Set(nonEmptyIndices.suffix(3))

        var seen: Set<String> = []
        var count = 0
        for (index, rawLine) in lines.enumerated() {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty else { continue }
            if headerZone.contains(index) || footerZone.contains(index),
               marginKeys.contains(marginKey(line)) { continue }
            // 同一句水印常同时出现在页首和页尾，重复行只计一次。
            guard seen.insert(line).inserted else { continue }

            var content = replace(urlRegex, in: line, with: "")
            content = replace(emailRegex, in: content, with: "")
            content = replace(bareWebAddressRegex, in: content, with: "")
            count += effectiveCharacterCount(content)
        }
        return count
    }

    /// 合并 PDF/OCR 产生的硬换行。以句末标点或 Markdown 结构行作为断点。
    public static func joinHardWrappedLines(_ text: String, preserveMarkdown: Bool = true) -> String {
        let lines = text.components(separatedBy: "\n")
        var output: [String] = []
        var buffer = ""

        func flush() {
            let trimmed = buffer.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty {
                output.append(preserveMarkdown ? trimmed : trimmed)
            }
            buffer = ""
        }

        for rawLine in lines {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty {
                flush()
                if output.last?.isEmpty == false {
                    output.append("")
                }
                continue
            }

            if isStructuralLine(rawLine) || (preserveMarkdown && isMarkdownHorizontalRule(line)) {
                flush()
                // 列表缩进、引用层级属于 Markdown 结构，不能用 trim 后的文本覆盖。
                let indent = rawLine.prefix { $0 == " " || $0 == "\t" }
                output.append(String(indent) + line)
                continue
            }

            if buffer.isEmpty {
                buffer = line
            } else if let last = buffer.last, let first = line.first,
                      isCJK(last), isCJK(first) {
                // 中文硬换行不能补空格，否则会得到「这是 一个」。
                buffer += line
            } else {
                buffer += " " + line
            }

            if let last = buffer.last, sentenceEndings.contains(last) {
                flush()
            }
        }
        flush()

        // 去掉尾部多余空行，保留段间空行。
        while output.last == "" { output.removeLast() }
        return output.joined(separator: "\n")
    }

    /// 归一化 Markdown：合并 3+ 空行、去行尾空白（围栏代码块内保持原样）。
    public static func normalizeMarkdown(_ text: String) -> String {
        let (stashed, blocks) = stashFencedBlocks(normalizeLineEndings(text))
        var lines = stashed.components(separatedBy: "\n")
        lines = lines.map { $0.replacingOccurrences(of: "[ \\t]+$", with: "", options: .regularExpression) }
        var result = collapseBlankLines(lines.joined(separator: "\n"))
        result = restoreFencedBlocks(result, blocks: blocks)
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// 解码 HTML 实体。命名实体按含义还原（`&amp;` → `&`、`&hellip;` → `…`），
    /// 数值实体按 Unicode 码点还原；未知实体替换为空格，避免把前后单词粘连。
    public static func stripHTMLEntities(_ text: String) -> String {
        var result = replace(htmlEntityRegex, in: text, with: " ") { match in
            guard let raw = capture(1, match, in: text) else { return " " }
            return decodeHTMLEntity(raw) ?? " "
        }
        // OCR 偶尔漏掉分号；只处理常见实体，避免误伤 URL 查询串里的 &。
        result = replace(htmlBareEntityRegex, in: result, with: " ") { match in
            guard let raw = capture(1, match, in: result) else { return " " }
            return decodeHTMLEntity(raw) ?? " "
        }
        return result
    }

    private static func decodeHTMLEntity(_ raw: String) -> String? {
        if raw.hasPrefix("#") {
            return decodeNumericEntity(raw)
        }
        if let value = htmlEntityMap[raw] { return value }
        return htmlEntityMap[raw.lowercased()]
    }

    private static func decodeNumericEntity(_ raw: String) -> String? {
        let body = raw.dropFirst()
        let value: UInt32?
        if body.first == "x" || body.first == "X" {
            value = UInt32(String(body.dropFirst()), radix: 16)
        } else {
            value = UInt32(String(body), radix: 10)
        }
        guard let value, value != 0, value <= 0x10FFFF,
              !(0xD800...0xDFFF).contains(value),
              let scalar = Unicode.Scalar(value) else { return nil }
        return String(Character(scalar))
    }

    // MARK: - 页面切分

    private struct PageSegment {
        var marker: String?
        var body: String
    }

    private static func splitPages(_ text: String) -> [PageSegment] {
        var segments: [PageSegment] = []
        var currentMarker: String?
        var currentLines: [String] = []

        func commit() {
            if currentMarker != nil || !currentLines.isEmpty {
                segments.append(PageSegment(marker: currentMarker,
                                            body: currentLines.joined(separator: "\n")))
            }
            currentLines = []
            currentMarker = nil
        }

        for line in text.components(separatedBy: "\n") {
            if isPageMarker(line) {
                commit()
                currentMarker = line.trimmingCharacters(in: .whitespaces)
            } else {
                currentLines.append(line)
            }
        }
        commit()
        return segments
    }

    // MARK: - 单页清理

    private static func sanitize(_ text: String, markdown: Bool,
                                 profile: DocumentProfile) -> String {
        var result = text

        // 脚本/样式/注释/不可见容器（含跨行块）。
        result = stripHTMLContainers(result)

        // OCR 常把标签写成 &lt;td&gt;：先解码实体，后面的标签清理才能生效。
        result = stripHTMLEntities(result)
        result = stripHTMLContainers(result)

        // 块级标签转成换行，保留段落边界；<hr> 在 Markdown 中转成水平线。
        result = replace(htmlHRRegex, in: result,
                         with: markdown ? "\n\n---\n\n" : "\n")
        result = replace(htmlBrRegex, in: result, with: "\n")
        if markdown {
            result = replace(htmlListItemOpenRegex, in: result, with: "\n- ")
        }
        result = replace(htmlBlockTagRegex, in: result, with: "\n")

        // 内联标签直接删除，不额外插空格，避免把单词拆开。
        result = replace(htmlTagRegex, in: result, with: "")
        // 补齐缺失闭合符的残留标签与属性片段。
        result = replace(htmlMalformedTagRegex, in: result, with: "")
        result = replace(htmlAttributeRegex, in: result, with: " ")
        result = replace(cssDeclarationRegex, in: result, with: " ")
        result = replace(emptyCSSBlockRegex, in: result, with: "")

        // Markdown 图片没有可用正文，始终删除；链接在 Markdown 模式保留语法。
        result = replace(markdownImageRegex, in: result, with: "")
        if profile == .ocr {
            // URL/邮箱只在 OCR 场景视为噪声；结构化文档里它们是正文
            // （部署地址、接口链接、联系人），删掉会造成内容缺失。
            if !markdown {
                result = replace(markdownLinkRegex, in: result, with: "$1")
                result = replace(urlRegex, in: result, with: "")
                result = replace(emailRegex, in: result, with: "")
            }
        }

        result = normalizeUnicodeScalars(result)
        result = normalizeHorizontalWhitespace(result, preserveIndent: markdown)
        // OCR 常在汉字之间插入空格（「新 中 新」），先合并再判断噪声行。
        // 结构化文档里的汉字空格是作者排版（「第一章 概述」），必须保留。
        if profile == .ocr {
            result = replace(cjkSpaceRegex, in: result, with: "")
        }
        result = normalizeRepeatedPunctuation(result)
        return result
    }

    private static func stripHTMLContainers(_ text: String) -> String {
        var result = text
        for regex in htmlContainerBlockRegexes {
            result = replace(regex, in: result, with: " ")
        }
        result = replace(htmlDoctypeRegex, in: result, with: " ")
        result = replace(htmlProcessingInstructionRegex, in: result, with: " ")
        result = replace(htmlCDATAMarkerRegex, in: result, with: "")
        return replace(commentRegex, in: result, with: " ")
    }

    /// 只折叠行内空白；Markdown 模式下保留行首缩进（嵌套列表/缩进代码）。
    private static func normalizeHorizontalWhitespace(_ text: String,
                                                     preserveIndent: Bool) -> String {
        let lines = text.components(separatedBy: "\n")
        let normalized = lines.map { line -> String in
            let leading = line.prefix { $0 == " " || $0 == "\t" }
            let indent = String(leading).replacingOccurrences(of: "\t", with: "    ")
            let body = String(line.dropFirst(leading.count))
            let collapsed = replace(horizontalSpaceRegex, in: body, with: " ")
            return preserveIndent ? indent + collapsed : collapsed
        }
        return normalized.joined(separator: "\n")
    }

    /// 合并重复标点：`......` → `……`，连续 `。。` → `。`。
    private static func normalizeRepeatedPunctuation(_ text: String) -> String {
        var result = replace(asciiEllipsisRegex, in: text, with: "……")
        result = replace(wideEllipsisRegex, in: result, with: "……")
        result = replace(repeatedPunctuationRegex, in: result, with: "$1")
        return result
    }

    private static func normalizeUnicodeScalars(_ text: String) -> String {
        var output = String()
        output.reserveCapacity(text.count)
        for scalar in text.unicodeScalars {
            let value = scalar.value
            // Unicode 非字符（每个平面末尾的 FFFE/FFFF）与 Tags 区块。
            if value & 0xFFFF == 0xFFFE || value & 0xFFFF == 0xFFFF {
                continue
            }
            if (0xFDD0...0xFDEF).contains(value) || (0xE0000...0xE007F).contains(value) {
                continue
            }
            switch value {
            case 0x00...0x08, 0x0B, 0x0C, 0x0E...0x1F, 0x7F...0x9F:
                // 控制字符 → 空格
                output.append(" ")
            case 0x00A0, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000:
                // 各种 Unicode 空白（含 NBSP）→ 空格
                output.append(" ")
            case 0x00AD, 0xFEFF, 0xFFFD, 0xFFF9...0xFFFC, 0x180E,
                 0xFE00...0xFE0F, 0xE0100...0xE01EF:
                // 软连字符 / BOM / 替换字符 / 注音控制 / 变体选择符直接删除
                continue
            case 0x061C, 0x200B...0x200F, 0x202A...0x202E,
                 0x2060...0x2064, 0x2066...0x2069, 0x206A...0x206F:
                // 零宽/方向控制/已废弃格式字符直接删除
                continue
            case 0xE002...0xF8FF, 0xF0000...0xFFFFD, 0x100000...0x10FFFD:
                // 私用区字符（保留清洗器自己的 E000/E001 占位符）
                continue
            case 0x0D:
                continue
            default:
                // OCR 偶尔把组合附加符号单独输出；前面没有基字符时删除。
                let category = scalar.properties.generalCategory
                if category == .nonspacingMark || category == .spacingMark {
                    guard let previous = output.unicodeScalars.last,
                          !CharacterSet.whitespacesAndNewlines.contains(previous),
                          !CharacterSet.punctuationCharacters.contains(previous) else {
                        continue
                    }
                }
                output.unicodeScalars.append(scalar)
            }
        }
        return output
    }

    // MARK: - 页眉页脚

    /// 只在每页首/尾各 3 个非空行中检测重复文本。
    /// 出现页数 ≥ max(2, 总页数 × 50%) 才判定为页眉/页脚。
    private static func detectRepeatedMargins(_ pages: [[String]]) -> (Set<String>, Set<String>) {
        guard pages.count >= 2 else { return ([], []) }

        var headerOccurrences: [String: Set<Int>] = [:]
        var footerOccurrences: [String: Set<Int>] = [:]

        for (pageIndex, lines) in pages.enumerated() {
            let nonEmpty = lines.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            for line in nonEmpty.prefix(3) {
                let key = marginKey(line)
                if !key.isEmpty { headerOccurrences[key, default: []].insert(pageIndex) }
            }
            for line in nonEmpty.suffix(3) {
                let key = marginKey(line)
                if !key.isEmpty { footerOccurrences[key, default: []].insert(pageIndex) }
            }
        }

        let threshold = max(2, Int(ceil(Double(pages.count) * 0.5)))
        let headers = Set(headerOccurrences.filter { $0.value.count >= threshold }.keys)
        let footers = Set(footerOccurrences.filter { $0.value.count >= threshold }.keys)
        return (headers, footers)
    }

    private static func removeMargins(_ lines: [String],
                                      headers: Set<String>,
                                      footers: Set<String>) -> [String] {
        guard !headers.isEmpty || !footers.isEmpty else { return lines }

        let nonEmptyIndices = lines.indices.filter {
            !lines[$0].trimmingCharacters(in: .whitespaces).isEmpty
        }
        let headerZone = Set(nonEmptyIndices.prefix(3))
        let footerZone = Set(nonEmptyIndices.suffix(3))

        var result: [String] = []
        result.reserveCapacity(lines.count)
        for (index, line) in lines.enumerated() {
            if line.trimmingCharacters(in: .whitespaces).isEmpty {
                result.append(line)
                continue
            }
            let key = marginKey(line)
            if headerZone.contains(index), headers.contains(key) { continue }
            if footerZone.contains(index), footers.contains(key) { continue }
            result.append(line)
        }
        return result
    }

    private static func marginKey(_ line: String) -> String {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, trimmed.count <= 200 else { return "" }
        return trimmed.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
    }

    // MARK: - 噪声行

    private static func filterGarbageLines(_ lines: [String], markdown: Bool,
                                           profile: DocumentProfile) -> [String] {
        var result: [String] = []
        result.reserveCapacity(lines.count)
        // 纯数字行只有在页首/页尾才视为页码：正文中的数字（年份、编号、
        // 价格）不能当成页码删掉。
        let nonEmptyIndices = lines.indices.filter {
            !lines[$0].trimmingCharacters(in: .whitespaces).isEmpty
        }
        let marginZone = Set(nonEmptyIndices.prefix(3))
            .union(nonEmptyIndices.suffix(3))

        for (index, rawLine) in lines.enumerated() {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty {
                result.append("")
                continue
            }
            // 页标记不进入最终文本（这里兜底处理直接传入的标记行）。
            if isPageMarker(line) {
                continue
            }
            // 明确的页码/页眉行（`2 / 164`、`第 2 页`、`Page 3 of 10`，
            // 以及 OCR 生成成 `# 2 / 164` 的伪标题）在任何位置都可删除。
            if isExplicitPageHeader(line) {
                continue
            }
            // 纯数字行只在页首/页尾才当页码。
            if marginZone.contains(index), isPageNumber(line) {
                continue
            }
            if isHorizontalRule(line) {
                // Markdown 的 --- / *** / ___ 是合法分隔线，不能按噪声删除。
                if markdown, isMarkdownHorizontalRule(line) {
                    result.append(line)
                }
                continue
            }
            // Markdown 结构行（标题/列表/引用/表格/围栏/缩进代码）一律保留。
            if isStructuralLine(rawLine) {
                let rendered = trimTrailingHorizontalWhitespace(rawLine)
                if result.last?.trimmingCharacters(in: .whitespaces) != line {
                    result.append(rendered)
                }
                continue
            }
            if isMarkdownInlineSyntax(line) {
                if result.last?.trimmingCharacters(in: .whitespaces) != line {
                    result.append(line)
                }
                continue
            }
            if isNoiseLine(line) {
                continue
            }
            // 册页脚注（「（3）上」「（12）下」）：页码标记的常见 OCR 形态，
            // 括号数字 + 可选上/中/下，全行无其他内容。
            if isBareVolumeMarker(line) {
                continue
            }
            // 散落数据行（表格提取残留：「23 大土地所有者 385 000' 19 250 000」）：
            // 多个独立数字组且中文占比低，对检索无信息量。结构化档跳过该判定：
            // Office/HTML 里的 IP、端口、容量、金额列表都是真实内容。
            if profile == .ocr {
                if isScatteredDataRow(line) {
                    continue
                }
                // 与 Python 一致：长度 < 20 且字母数字占比 < 30% 视为噪声。
                if line.count < 20, alphanumericRatio(line) < 0.3 {
                    continue
                }
            }
            if result.last?.trimmingCharacters(in: .whitespaces) == line {
                continue
            }
            result.append(line)
        }
        return result
    }

    private static func isMarkdownHorizontalRule(_ line: String) -> Bool {
        markdownHorizontalRuleRegex.firstMatch(
            in: line, range: NSRange(line.startIndex..<line.endIndex, in: line)) != nil
    }

    private static func isMarkdownInlineSyntax(_ line: String) -> Bool {
        markdownInlineSyntaxRegex.firstMatch(
            in: line, range: NSRange(line.startIndex..<line.endIndex, in: line)) != nil
    }

    /// OCR 噪声行：符号密度高、有效字符少，或在正文行里堆了多个装饰符号。
    private static func isNoiseLine(_ line: String) -> Bool {
        let scalars = line.unicodeScalars
        let total = scalars.count
        guard total >= 5 else { return false }

        var meaningful = 0
        var symbols = 0
        var noisySymbols = 0
        for scalar in scalars {
            if isCJKLetterScalar(scalar) || CharacterSet.alphanumerics.contains(scalar) {
                meaningful += 1
                continue
            }
            if CharacterSet.symbols.contains(scalar)
                || CharacterSet.punctuationCharacters.contains(scalar) {
                symbols += 1
            }
            if noisySymbolsSet.contains(scalar) {
                noisySymbols += 1
            }
        }

        if noisySymbols >= 2, meaningful < 12 { return true }
        if meaningful < 4, total >= 6 { return true }
        let meaningfulRatio = Double(meaningful) / Double(total)
        let symbolRatio = Double(symbols) / Double(total)
        if meaningfulRatio < 0.40, symbolRatio > 0.25 { return true }
        return false
    }

    private static func isPageNumber(_ line: String) -> Bool {
        for regex in pageNumberRegexes {
            if regex.firstMatch(in: line, range: NSRange(line.startIndex..<line.endIndex, in: line)) != nil {
                return true
            }
        }
        return false
    }

    /// 明确的页码/页眉行：形态特异（比值 / 第 N 页 / Page N，含 `# 2 / 164`
    /// 这类 OCR 伪标题），可安全地在任意位置删除。判定口径与大纲识别一致。
    private static func isExplicitPageHeader(_ line: String) -> Bool {
        DocumentOutline.isPageHeaderLine(line)
    }

    /// 册页脚注独立行：「（3）上」「（12）下」「(5)」。形态特异（括号数字 +
    /// 可选上/中/下、无其他文本），任意位置出现都可安全删除；
    /// 「（一）问题的提出」这类真标题带标题文本，不会命中。
    private static func isBareVolumeMarker(_ line: String) -> Bool {
        bareVolumeMarkerRegex.firstMatch(
            in: line, range: NSRange(line.startIndex..<line.endIndex, in: line)) != nil
    }

    /// 散落数据行：表格/统计提取残留。≥4 个独立数字组且中文占比 < 35%
    /// （日期「1949 年 10 月 1 日」只有 3 组、中文占比高，不会误删）。
    private static func isScatteredDataRow(_ line: String) -> Bool {
        guard !line.contains("|") else { return false }
        let digitGroups = line.components(
            separatedBy: CharacterSet.decimalDigits.inverted
        ).filter { !$0.isEmpty }
        guard digitGroups.count >= 4 else { return false }
        let cjkCount = line.filter { isCJK($0) }.count
        return Double(cjkCount) / Double(max(1, line.count)) < 0.35
    }

    private static func isHorizontalRule(_ line: String) -> Bool {
        horizontalRuleRegex.firstMatch(in: line,
                                        range: NSRange(line.startIndex..<line.endIndex, in: line)) != nil
    }

    private static func isStructuralLine(_ line: String) -> Bool {
        if line.hasPrefix(codePlaceholderPrefix) { return true }
        return structuralLineRegex.firstMatch(in: line,
                                              range: NSRange(line.startIndex..<line.endIndex, in: line)) != nil
    }

    /// 是否属于 CJK 字符块（用于硬换行合并时决定是否补空格）。
    private static func isCJK(_ character: Character) -> Bool {
        guard let scalar = character.unicodeScalars.first else { return false }
        return isCJKScalar(scalar)
    }

    private static func isCJKScalar(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x2E80...0x303F, 0x3040...0x30FF, 0x3400...0x4DBF,
             0x4E00...0x9FFF, 0xAC00...0xD7AF, 0xF900...0xFAFF,
             0xFF00...0xFFEF, 0x20000...0x2FA1F:
            return true
        default:
            return false
        }
    }

    /// 仅统计真正的 CJK 文字（不含 CJK 标点），用于噪声行判断。
    private static func isCJKLetterScalar(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x3040...0x30FF, 0x3400...0x4DBF, 0x4E00...0x9FFF,
             0xAC00...0xD7AF, 0xF900...0xFAFF, 0x20000...0x2FA1F:
            return true
        default:
            return false
        }
    }

    private static func alphanumericRatio(_ line: String) -> Double {
        guard !line.isEmpty else { return 0 }
        var alphanumeric = 0
        var total = 0
        for scalar in line.unicodeScalars {
            total += 1
            if CharacterSet.alphanumerics.contains(scalar) || scalar.value == 95
                || CharacterSet.whitespacesAndNewlines.contains(scalar) {
                alphanumeric += 1
            }
        }
        return total == 0 ? 0 : Double(alphanumeric) / Double(total)
    }

    // MARK: - 代码围栏暂存

    private static func stashFencedBlocks(_ text: String) -> (String, [String]) {
        var blocks: [String] = []
        var result: [String] = []
        var current: [String] = []
        var inFence = false

        for line in text.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            let isFence = trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~")
            if isFence {
                if inFence {
                    current.append(line)
                    blocks.append(current.joined(separator: "\n"))
                    result.append(codePlaceholder(blocks.count - 1))
                    current = []
                    inFence = false
                } else {
                    inFence = true
                    current = [line]
                }
                continue
            }
            if inFence {
                current.append(line)
            } else {
                result.append(line)
            }
        }

        // 未闭合的围栏按原样保留，避免吞掉后续正文。
        if inFence, !current.isEmpty {
            blocks.append(current.joined(separator: "\n"))
            result.append(codePlaceholder(blocks.count - 1))
        }
        return (result.joined(separator: "\n"), blocks)
    }

    /// 保护行内代码（反引号）不被 HTML/符号清理误伤。
    private static func stashInlineCode(_ text: String,
                                        startingIndex: Int) -> (String, [String]) {
        var blocks: [String] = []
        let result = replace(inlineCodeRegex, in: text, with: "") { match in
            guard let full = capture(0, match, in: text), !full.isEmpty else { return nil }
            blocks.append(full)
            return codePlaceholder(startingIndex + blocks.count - 1)
        }
        return (result, blocks)
    }

    // MARK: - HTML 表格 → Markdown / 保留 HTML

    /// 表格转换结果。`markdown == nil` 表示无法无损转换，应保留原 HTML。
    struct HTMLTableConversion {
        var markdown: String?
        var reason: String?
    }

    /// 提取平衡的 `<table>...</table>` 范围；支持嵌套表格，避免非贪婪正则截断。
    static func tableRanges(in text: String) -> [NSRange] {
        let matches = htmlTableTagRegex.matches(
            in: text, options: [],
            range: NSRange(text.startIndex..<text.endIndex, in: text))
        var ranges: [NSRange] = []
        var depth = 0
        var start: Int?
        for match in matches {
            guard let token = capture(0, match, in: text) else { continue }
            if token.hasPrefix("</") {
                guard depth > 0 else { continue }
                depth -= 1
                if depth == 0, let begin = start {
                    ranges.append(NSRange(location: begin,
                                          length: NSMaxRange(match.range) - begin))
                    start = nil
                }
            } else if !token.hasSuffix("/>") {
                if depth == 0 { start = match.range.location }
                depth += 1
            }
        }
        return ranges
    }

    /// 在通用标签清理前处理 HTML 表格：简单表格转 GFM；
    /// 合并单元格/嵌套/多行表头/块级内容等复杂表格保留原始 HTML。
    private static func convertHTMLTables(_ text: String, startingIndex: Int)
        -> (text: String, hadTables: Bool, retained: [String]) {
        let ranges = tableRanges(in: text)
        guard !ranges.isEmpty else { return (text, false, []) }

        var result = ""
        result.reserveCapacity(text.count)
        var cursor = 0
        var retained: [String] = []

        for range in ranges {
            guard range.location >= cursor,
                  let fullRange = Range(range, in: text),
                  let prefixRange = Range(NSRange(location: cursor,
                                                  length: range.location - cursor),
                                          in: text) else { continue }
            let html = String(text[fullRange])
            let conversion = convertHTMLTable(fromHTML: html)
            result += text[prefixRange]
            if let markdown = conversion.markdown {
                result += "\n\n" + markdown + "\n\n"
            } else {
                // 用占位符保护 HTML 表格，后续 sanitize 不会拆掉 table/td 标签。
                retained.append(cleanRetainedHTMLTable(html))
                result += "\n\n" + codePlaceholder(startingIndex + retained.count - 1) + "\n\n"
                AgentLog.info("[clean] 复杂表格保留 HTML：\(conversion.reason ?? "无法无损转换")")
            }
            cursor = NSMaxRange(range)
        }
        if let tail = Range(NSRange(location: cursor, length: text.utf16.count - cursor),
                            in: text) {
            result += text[tail]
        }
        return (result, true, retained)
    }

    /// 将单个 HTML 表格转成 GFM 表格；Ovis 输出清理也复用同一实现。
    static func convertHTMLTable(fromHTML html: String) -> HTMLTableConversion {
        if countOccurrences(of: "<table", in: html) > 1 {
            return HTMLTableConversion(markdown: nil, reason: "嵌套表格")
        }
        if hasMergedCells(in: html) {
            return HTMLTableConversion(markdown: nil, reason: "colspan/rowspan 合并单元格")
        }
        if caseInsensitiveContains(html, "<caption") {
            return HTMLTableConversion(markdown: nil, reason: "表格标题 caption")
        }
        if htmlCellBlockContentRegex.firstMatch(
            in: html, options: [], range: NSRange(html.startIndex..<html.endIndex, in: html)) != nil {
            return HTMLTableConversion(markdown: nil, reason: "单元格含列表/图片/公式等块级内容")
        }

        var inner = replace(htmlTableOpenRegex, in: html, with: "")
        inner = replace(htmlTableCloseRegex, in: inner, with: "")

        let rowMatches = htmlRowOpenRegex.matches(
            in: inner, options: [], range: NSRange(inner.startIndex..<inner.endIndex, in: inner))
        var rowHTMLs: [String] = []
        if rowMatches.isEmpty {
            rowHTMLs = [inner]
        } else {
            for (index, match) in rowMatches.enumerated() {
                let start = NSMaxRange(match.range)
                let end = index + 1 < rowMatches.count
                    ? rowMatches[index + 1].range.location : inner.utf16.count
                guard end >= start,
                      let range = Range(NSRange(location: start, length: end - start),
                                        in: inner) else { continue }
                rowHTMLs.append(String(inner[range]))
            }
        }

        // 只允许首行作为表头；多行表头或后置表头在 GFM 中无法表达。
        let headerRows = rowHTMLs.enumerated().filter {
            caseInsensitiveContains($0.element, "<th")
        }.map(\.offset)
        if headerRows.count > 1 {
            return HTMLTableConversion(markdown: nil, reason: "多行表头")
        }
        if let first = headerRows.first, first > 0 {
            return HTMLTableConversion(markdown: nil, reason: "非首行表头")
        }

        var rows: [[String]] = []
        for rowHTML in rowHTMLs {
            let cells = parseTableCells(rowHTML)
            if !cells.isEmpty { rows.append(cells) }
        }
        guard !rows.isEmpty else {
            return HTMLTableConversion(markdown: nil, reason: "无法解析表格行")
        }

        let columnCount = rows.map(\.count).max() ?? 0
        guard columnCount > 0 else {
            return HTMLTableConversion(markdown: nil, reason: "表格没有单元格")
        }

        func renderLine(_ values: [String]) -> String {
            var padded = values
            if padded.count < columnCount {
                padded += Array(repeating: "", count: columnCount - padded.count)
            } else if padded.count > columnCount {
                padded = Array(padded.prefix(columnCount))
            }
            return "| " + padded.joined(separator: " | ") + " |"
        }

        let header = renderLine(rows[0])
        let separator = "| " + Array(repeating: "---", count: columnCount)
            .joined(separator: " | ") + " |"
        let body = rows.dropFirst().map(renderLine)
        return HTMLTableConversion(markdown: ([header, separator] + body)
            .joined(separator: "\n"), reason: nil)
    }

    private static func hasMergedCells(in html: String) -> Bool {
        let range = NSRange(html.startIndex..<html.endIndex, in: html)
        for match in htmlSpanAttributeRegex.matches(in: html, options: [], range: range) {
            guard let raw = capture(1, match, in: html), let value = Int(raw) else { continue }
            if value > 1 { return true }
        }
        return false
    }

    private static func caseInsensitiveContains(_ text: String, _ needle: String) -> Bool {
        text.range(of: needle, options: .caseInsensitive) != nil
    }

    private static func countOccurrences(of needle: String, in text: String) -> Int {
        var count = 0
        var searchRange = text.startIndex..<text.endIndex
        while let found = text.range(of: needle, options: .caseInsensitive,
                                     range: searchRange) {
            count += 1
            searchRange = found.upperBound..<text.endIndex
        }
        return count
    }

    /// 保留 HTML 表格时只清脚本/样式/注释与 Unicode 噪声，保留 table 结构与属性。
    private static func cleanRetainedHTMLTable(_ html: String) -> String {
        var result = html
        result = replace(scriptRegex, in: result, with: " ")
        result = replace(styleRegex, in: result, with: " ")
        result = replace(noscriptRegex, in: result, with: " ")
        result = replace(commentRegex, in: result, with: " ")
        result = stripHTMLEntities(result)
        result = replace(noiseSymbolRunRegex, in: result, with: " ")
        result = normalizeUnicodeScalars(result)
        result = normalizeRepeatedPunctuation(result)
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func parseTableCells(_ rowHTML: String) -> [String] {
        let matches = htmlCellOpenRegex.matches(
            in: rowHTML, options: [], range: NSRange(rowHTML.startIndex..<rowHTML.endIndex,
                                                       in: rowHTML))
        guard !matches.isEmpty else { return [] }

        var cells: [String] = []
        for (index, match) in matches.enumerated() {
            let start = NSMaxRange(match.range)
            let end = index + 1 < matches.count
                ? matches[index + 1].range.location : rowHTML.utf16.count
            guard end >= start,
                  let range = Range(NSRange(location: start, length: end - start),
                                    in: rowHTML) else { continue }
            var content = String(rowHTML[range])
            // 缺失闭合标签时，下一个单元格前的残留闭合标签一并去掉。
            content = replace(htmlCellCloseRegex, in: content, with: " ")
            cells.append(cleanTableCell(content))

            let attributes = capture(2, match, in: rowHTML) ?? ""
            if let span = colspanValue(attributes), span > 1 {
                cells.append(contentsOf: Array(repeating: "", count: min(span - 1, 32)))
            }
        }
        return cells
    }

    private static func cleanTableCell(_ content: String) -> String {
        var value = content
        value = replace(htmlHRRegex, in: value, with: " ")
        value = replace(htmlBrRegex, in: value, with: " ")
        value = replace(htmlBlockTagRegex, in: value, with: " ")
        value = replace(htmlTagRegex, in: value, with: "")
        value = replace(htmlMalformedTagRegex, in: value, with: "")
        value = stripHTMLEntities(value)
        value = replace(horizontalSpaceRegex, in: value, with: " ")
        value = replace(noiseSymbolRunRegex, in: value, with: " ")
        value = value.replacingOccurrences(of: "\n", with: " ")
        value = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return value.replacingOccurrences(of: "|", with: "\\|")
    }

    private static func colspanValue(_ attributes: String) -> Int? {
        guard let match = htmlColspanRegex.firstMatch(
            in: attributes,
            range: NSRange(attributes.startIndex..<attributes.endIndex, in: attributes)),
            let raw = capture(1, match, in: attributes),
            let value = Int(raw) else { return nil }
        return max(1, value)
    }

    /// 轻量 Markdown 识别：标题、表格、围栏、链接权重更高；
    /// 列表/引用/强调/分隔线需要组合出现。
    private static func looksLikeMarkdown(_ text: String) -> Bool {
        var score = 0
        for line in text.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") { return true }
            guard !trimmed.isEmpty else { continue }
            if headingDetectRegex.firstMatch(
                in: trimmed, range: NSRange(trimmed.startIndex..<trimmed.endIndex,
                                            in: trimmed)) != nil {
                score += 3
            } else if trimmed.hasPrefix("|") && trimmed.hasSuffix("|") {
                score += 3
            } else if markdownTableSeparatorRegex.firstMatch(
                in: trimmed, range: NSRange(trimmed.startIndex..<trimmed.endIndex,
                                            in: trimmed)) != nil {
                score += 3
            } else if trimmed.hasPrefix(">") {
                score += 1
            } else if markdownListRegex.firstMatch(
                in: trimmed, range: NSRange(trimmed.startIndex..<trimmed.endIndex,
                                            in: trimmed)) != nil {
                score += 1
            }
            if markdownLinkTestRegex.firstMatch(
                in: trimmed, range: NSRange(trimmed.startIndex..<trimmed.endIndex,
                                            in: trimmed)) != nil {
                score += 3
            }
            if markdownEmphasisTestRegex.firstMatch(
                in: trimmed, range: NSRange(trimmed.startIndex..<trimmed.endIndex,
                                            in: trimmed)) != nil {
                score += 1
            }
            if isMarkdownHorizontalRule(trimmed) { score += 1 }
            if score >= 3 { return true }
        }
        return false
    }

    /// 用私用区字符做占位符：`\u{0}` 会在控制字符归一化时变成空格，
    /// 导致 restore 永远匹配不到。
    private static let codePlaceholderPrefix = "\u{E000}CODE"

    private static func codePlaceholder(_ index: Int) -> String {
        "\(codePlaceholderPrefix)\(index)\u{E001}"
    }

    private static func restoreFencedBlocks(_ text: String, blocks: [String]) -> String {
        guard !blocks.isEmpty else { return text }
        return replace(codePlaceholderRegex, in: text, with: "") { match in
            guard let raw = capture(1, match, in: text), let index = Int(raw),
                  blocks.indices.contains(index) else { return "" }
            return blocks[index]
        }
    }

    // MARK: - 正则替换工具

    /// 用闭包逐处替换；闭包返回 nil 表示保持原文。
    private static func replace(_ regex: NSRegularExpression,
                                in text: String,
                                with template: String,
                                transform: ((NSTextCheckingResult) -> String?)? = nil) -> String {
        let fullRange = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = regex.matches(in: text, options: [], range: fullRange)
        guard !matches.isEmpty else { return text }

        var result = ""
        result.reserveCapacity(text.count)
        var cursor = 0
        for match in matches {
            let range = match.range
            guard range.location != NSNotFound, range.location >= cursor else { continue }
            if let start = Range(NSRange(location: cursor, length: range.location - cursor), in: text) {
                result += text[start]
            }
            if let transform {
                // 闭包返回 nil 表示保留匹配原文。
                if let replacement = transform(match) {
                    result += replacement
                } else if let matchRange = Range(range, in: text) {
                    result += text[matchRange]
                }
            } else if let matchRange = Range(range, in: text) {
                let matched = String(text[matchRange])
                result += template.isEmpty ? "" : expandTemplate(template, matched: matched, match: match, in: text)
            }
            cursor = range.location + range.length
        }
        if let tail = Range(NSRange(location: cursor, length: text.utf16.count - cursor), in: text) {
            result += text[tail]
        }
        return result
    }

    /// 支持 `$1` 占位符的最小模板展开（避免对 `$` 转义的额外依赖）。
    private static func expandTemplate(_ template: String,
                                       matched: String,
                                       match: NSTextCheckingResult,
                                       in text: String) -> String {
        guard template == "$1" else { return template }
        guard match.numberOfRanges > 1, let group = capture(1, match, in: text) else { return matched }
        return group
    }

    private static func capture(_ index: Int, _ match: NSTextCheckingResult, in text: String) -> String? {
        guard index < match.numberOfRanges else { return nil }
        let range = match.range(at: index)
        guard range.location != NSNotFound, let swiftRange = Range(range, in: text) else { return nil }
        return String(text[swiftRange])
    }

    private static func normalizeLineEndings(_ text: String) -> String {
        text.replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
    }

    private static func collapseBlankLines(_ text: String) -> String {
        text.replacingOccurrences(of: "\n{3,}", with: "\n\n", options: .regularExpression)
    }

    // MARK: - 常量

    private static let sentenceEndings: Set<Character> = [
        "。", "！", "？", "!", "?", ".", "…", "”", "’", "」", "』", "）", "】", ">", "；", ";",
    ]

    /// 常见命名实体。大小写敏感项（Dagger/Prime）优先精确匹配，其余回退小写。
    private static let htmlEntityMap: [String: String] = [
        "amp": "&", "lt": "<", "gt": ">", "quot": "\"", "apos": "'",
        "nbsp": "\u{00A0}", "ensp": " ", "emsp": " ", "thinsp": " ",
        "zwnj": "", "zwj": "", "shy": "",
        "copy": "©", "reg": "®", "trade": "™", "hellip": "…",
        "mdash": "—", "ndash": "–", "lsquo": "‘", "rsquo": "’",
        "ldquo": "“", "rdquo": "”", "laquo": "«", "raquo": "»",
        "middot": "·", "bull": "•", "deg": "°", "plusmn": "±",
        "times": "×", "divide": "÷", "frac12": "½", "frac14": "¼", "frac34": "¾",
        "sup1": "¹", "sup2": "²", "sup3": "³", "micro": "µ",
        "para": "¶", "sect": "§", "dagger": "†", "Dagger": "‡",
        "permil": "‰", "prime": "′", "Prime": "″", "oline": "‾", "frasl": "⁄",
        "euro": "€", "pound": "£", "yen": "¥", "cent": "¢", "curren": "¤",
        "not": "¬", "macr": "¯", "acute": "´", "cedil": "¸", "uml": "¨",
        "ordf": "ª", "ordm": "º", "iexcl": "¡", "iquest": "¿",
        "larr": "←", "rarr": "→", "uarr": "↑", "darr": "↓", "harr": "↔",
        "crarr": "↵", "lArr": "⇐", "rArr": "⇒", "uArr": "⇑", "dArr": "⇓", "hArr": "⇔",
        "forall": "∀", "part": "∂", "exist": "∃", "empty": "∅", "nabla": "∇",
        "isin": "∈", "notin": "∉", "ni": "∋", "prod": "∏", "sum": "∑",
        "minus": "−", "lowast": "∗", "radic": "√", "prop": "∝", "infin": "∞",
        "ang": "∠", "and": "∧", "or": "∨", "cap": "∩", "cup": "∪",
        "int": "∫", "there4": "∴", "sim": "∼", "cong": "≅", "asymp": "≈",
        "ne": "≠", "equiv": "≡", "le": "≤", "ge": "≥", "sub": "⊂", "sup": "⊃",
        "nsub": "⊄", "sube": "⊆", "supe": "⊇", "oplus": "⊕", "otimes": "⊗",
        "perp": "⊥", "sdot": "⋅", "lceil": "⌈", "rceil": "⌉",
        "lfloor": "⌊", "rfloor": "⌋", "lang": "⟨", "rang": "⟩", "loz": "◊",
        "spades": "♠", "clubs": "♣", "hearts": "♥", "diams": "♦",
    ]

    private static let pageMarkerRegex = try! NSRegularExpression(
        pattern: #"^[ \t]*<!--[ \t]*page[ \t]+(\d+)[ \t]*-->[ \t]*$"#)

    /// 剥离文本中的分页标记行。标记只供清洗阶段使用，切片/embedding 输入
    /// 不得包含页码；溯源只保留文件名（source）。
    public static func stripPageMarkers(_ text: String) -> String {
        var cleaned: [String] = []
        for line in text.components(separatedBy: "\n") {
            if isPageMarker(line.trimmingCharacters(in: .whitespaces)) { continue }
            if line.trimmingCharacters(in: .whitespaces).isEmpty,
               cleaned.last?.trimmingCharacters(in: .whitespaces).isEmpty == true {
                continue
            }
            cleaned.append(line)
        }
        return cleaned.joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static let codePlaceholderRegex = try! NSRegularExpression(
        pattern: "\u{E000}CODE(\\d+)\u{E001}")

    private static let scriptRegex = try! NSRegularExpression(
        pattern: #"<script\b[^>]*>[\s\S]*?</script>"#, options: [.caseInsensitive])
    private static let styleRegex = try! NSRegularExpression(
        pattern: #"<style\b[^>]*>[\s\S]*?</style>"#, options: [.caseInsensitive])
    private static let noscriptRegex = try! NSRegularExpression(
        pattern: #"<noscript\b[^>]*>[\s\S]*?</noscript>"#, options: [.caseInsensitive])
    private static let commentRegex = try! NSRegularExpression(
        pattern: #"<!--[\s\S]*?-->"#)
    private static let htmlTagRegex = try! NSRegularExpression(
        pattern: #"</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*)?/?>"#)
    private static let markdownImageRegex = try! NSRegularExpression(
        pattern: #"!\[[^\]]*\]\([^)]*\)"#)
    private static let markdownLinkRegex = try! NSRegularExpression(
        pattern: #"\[([^\]]*)\]\([^)]*\)"#)
    private static let htmlEntityRegex = try! NSRegularExpression(
        pattern: #"&(#x[0-9A-Fa-f]{1,6}|#\d{1,7}|[A-Za-z][A-Za-z0-9]{1,10});"#)
    private static let htmlBareEntityRegex = try! NSRegularExpression(
        pattern: #"&(#x[0-9A-Fa-f]{1,6}|#\d{1,7}|nbsp|amp|lt|gt|quot|apos|hellip|mdash|ndash|ldquo|rdquo|middot|bull|copy|reg|trade)\b;?"#,
        options: [.caseInsensitive])
    private static let urlRegex = try! NSRegularExpression(
        pattern: #"https?://[^\s]+"#, options: [.caseInsensitive])
    /// 没有协议头的网址（水印里常见 `www.guminxuexiao.com`）。
    private static let bareWebAddressRegex = try! NSRegularExpression(
        pattern: #"\bwww\.[^\s]+"#, options: [.caseInsensitive])
    private static let emailRegex = try! NSRegularExpression(
        pattern: #"\S+@\S+\.\S+"#)
    private static let horizontalSpaceRegex = try! NSRegularExpression(
        pattern: #"[ \t]+"#)
    private static let cjkSpaceRegex = try! NSRegularExpression(
        pattern: #"(?<=[\u2E80-\u303F\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF\uFF00-\uFFEF])[ \t]+(?=[\u2E80-\u303F\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF\uFF00-\uFFEF])"#)
    private static let asciiEllipsisRegex = try! NSRegularExpression(
        pattern: #"\.{3,}"#)
    private static let wideEllipsisRegex = try! NSRegularExpression(
        pattern: #"…{2,}"#)
    private static let repeatedPunctuationRegex = try! NSRegularExpression(
        pattern: #"([。，、；：！？])\1+"#)
    private static let noisySymbolsSet: Set<Unicode.Scalar> = Set(
        "•·※▶◀■□◆◇●○◎★☆◯▷◁™®©〒".unicodeScalars)
    private static let horizontalRuleRegex = try! NSRegularExpression(
        pattern: #"^([\-=_*#~]{3,})\s*$"#)
    private static let markdownHorizontalRuleRegex = try! NSRegularExpression(
        pattern: #"^ {0,3}(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,}|=+\s*)$"#)
    private static let markdownInlineSyntaxRegex = try! NSRegularExpression(
        pattern: #"^\s*(?:\*{1,3}[^*\n]+\*{1,3}|_{1,3}[^_\n]+_{1,3}|\[[^\]]+\]\([^)]+\)|!\[[^\]]*\]\([^)]+\)|<https?://[^>]+>)"#)
    private static let structuralLineRegex = try! NSRegularExpression(
        pattern: #"^(?:\s{0,3}(?:#{1,6}\s|[-*+]\s|\d{1,3}[.)]\s|>|\||```|~~~)|\s{4,})"#)
    private static let headingDetectRegex = try! NSRegularExpression(
        pattern: #"^#{1,6}\s+\S"#)
    private static let markdownTableSeparatorRegex = try! NSRegularExpression(
        pattern: #"^\|?\s*:?-{3,}"#)
    private static let markdownListRegex = try! NSRegularExpression(
        pattern: #"^\s{0,3}(?:[-*+]\s|\d{1,3}[.)]\s)"#)
    private static let markdownLinkTestRegex = try! NSRegularExpression(
        pattern: #"\[[^\]]+\]\([^)]+\)"#)
    private static let markdownEmphasisTestRegex = try! NSRegularExpression(
        pattern: #"(?:\*\*[^*\n]+\*\*|__[^_\n]+__)"#)

    // HTML 标签/属性/样式残片
    private static let htmlContainerBlockRegexes: [NSRegularExpression] = [
        scriptRegex, styleRegex, noscriptRegex,
        try! NSRegularExpression(
            pattern: #"<svg\b[^>]*>[\s\S]*?</svg>"#, options: [.caseInsensitive]),
        try! NSRegularExpression(
            pattern: #"<canvas\b[^>]*>[\s\S]*?</canvas>"#, options: [.caseInsensitive]),
        try! NSRegularExpression(
            pattern: #"<iframe\b[^>]*>[\s\S]*?</iframe>"#, options: [.caseInsensitive]),
        try! NSRegularExpression(
            pattern: #"<object\b[^>]*>[\s\S]*?</object>"#, options: [.caseInsensitive]),
        try! NSRegularExpression(
            pattern: #"<template\b[^>]*>[\s\S]*?</template>"#, options: [.caseInsensitive]),
        try! NSRegularExpression(
            pattern: #"<head\b[^>]*>[\s\S]*?</head>"#, options: [.caseInsensitive]),
    ]
    private static let htmlDoctypeRegex = try! NSRegularExpression(
        pattern: #"<!doctype\b[^>]*>"#, options: [.caseInsensitive])
    private static let htmlProcessingInstructionRegex = try! NSRegularExpression(
        pattern: #"<\?xml\b[^>]*\?>"#, options: [.caseInsensitive])
    private static let htmlCDATAMarkerRegex = try! NSRegularExpression(
        pattern: #"<!\[CDATA\[|\]\]>"#)
    private static let htmlHRRegex = try! NSRegularExpression(
        pattern: #"<hr\b[^>]*?/?>"#, options: [.caseInsensitive])
    private static let htmlBrRegex = try! NSRegularExpression(
        pattern: #"<br\b[^>]*?/?>"#, options: [.caseInsensitive])
    private static let htmlListItemOpenRegex = try! NSRegularExpression(
        pattern: #"<li\b[^>]*>"#, options: [.caseInsensitive])
    private static let htmlBlockTagRegex = try! NSRegularExpression(
        pattern: #"</?(?:address|article|aside|blockquote|details|dialog|div|dl|dt|dd|fieldset|figcaption|figure|footer|form|h[1-6]|header|li|main|nav|ol|p|pre|section|summary|table|tbody|td|tfoot|th|thead|tr|ul)\b[^<>]*>"#,
        options: [.caseInsensitive])
    private static let htmlMalformedTagRegex = try! NSRegularExpression(
        pattern: #"</?(?:table|thead|tbody|tfoot|tr|td|th|div|span|p|br|hr|img|a|b|i|u|strong|em|sup|sub|font|ul|ol|li|h[1-6]|section|article|blockquote|pre|code|figure|figcaption)\b(?:\s+[\w:-]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s<>"']+))?)*\s*/?>?"#,
        options: [.caseInsensitive])
    private static let htmlAttributeRegex = try! NSRegularExpression(
        pattern: #"\b(?:class|style|id|width|height|align|valign|colspan|rowspan|border|cellpadding|cellspacing|href|src|srcset|alt|title|role|lang|xmlns|data-[\w-]+|aria-[\w-]+|on[a-z]+)\s*=\s*(?:"[^"]*"|'[^']*'|[^\s<>"']+)"#,
        options: [.caseInsensitive])
    private static let cssDeclarationRegex = try! NSRegularExpression(
        pattern: #"\b(?:color|background(?:-color)?|font(?:-size|-family|-weight)?|margin(?:-[\w-]+)?|padding(?:-[\w-]+)?|border(?:-[\w-]+)?|width|height|display|position|top|left|right|bottom|line-height|text-align|z-index)\s*:\s*[^;{}\n]{1,120};"#,
        options: [.caseInsensitive])
    private static let emptyCSSBlockRegex = try! NSRegularExpression(
        pattern: #"\{\s*\}"#)
    private static let noiseSymbolRunRegex = try! NSRegularExpression(
        pattern: #"[•·※▶◀■□◆◇●○◎★☆◯▷◁▪▫◦‣⁃─━│┃┄┅┆┇┈┉┊┋╌╍═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬]{2,}"#)

    // HTML 表格
    private static let htmlTableTagRegex = try! NSRegularExpression(
        pattern: #"<table\b[^>]*>|</table\b[^>]*>"#, options: [.caseInsensitive])
    private static let htmlTableOpenRegex = try! NSRegularExpression(
        pattern: #"<table\b[^>]*>"#, options: [.caseInsensitive])
    private static let htmlTableCloseRegex = try! NSRegularExpression(
        pattern: #"</table\b[^>]*>"#, options: [.caseInsensitive])
    private static let htmlRowOpenRegex = try! NSRegularExpression(
        pattern: #"<tr\b[^>]*>"#, options: [.caseInsensitive])
    private static let htmlCellOpenRegex = try! NSRegularExpression(
        pattern: #"<t([hd])\b([^>]*)>"#, options: [.caseInsensitive])
    private static let htmlCellCloseRegex = try! NSRegularExpression(
        pattern: #"</(?:td|th|tr|table)\b[^>]*>"#, options: [.caseInsensitive])
    private static let htmlColspanRegex = try! NSRegularExpression(
        pattern: #"colspan\s*=\s*["']?(\d{1,2})"#, options: [.caseInsensitive])
    private static let htmlSpanAttributeRegex = try! NSRegularExpression(
        pattern: #"<(?:td|th)\b[^>]*(?:colspan|rowspan)\s*=\s*["']?([0-9]+)"#,
        options: [.caseInsensitive])
    private static let htmlCellBlockContentRegex = try! NSRegularExpression(
        pattern: #"<(?:ul|ol|dl|pre|blockquote|figure|figcaption|img|video|audio|svg|math|form|h[1-6])\b"#,
        options: [.caseInsensitive])
    private static let inlineCodeRegex = try! NSRegularExpression(
        pattern: #"(?<!`)`([^`\n]+)`(?!`)"#)

    private static let pageNumberRegexes: [NSRegularExpression] = [
        try! NSRegularExpression(pattern: #"^\d{1,4}$"#),
        try! NSRegularExpression(pattern: #"^第\s*\d{1,4}\s*页$"#),
        try! NSRegularExpression(pattern: #"^page\s+\d{1,4}$"#, options: [.caseInsensitive]),
        try! NSRegularExpression(pattern: #"^[-–—]?\s*\d{1,4}\s*[-–—]?$"#),
        try! NSRegularExpression(pattern: #"^\d{1,4}\s*/\s*\d{1,4}$"#),
    ]

    /// 册页脚注独立行：「（3）上」「（12）下」「(5)」。
    private static let bareVolumeMarkerRegex = try! NSRegularExpression(
        pattern: #"^[（(]\s*\d{1,4}\s*[)）]\s*[上中下]?$"#)
}

// MARK: - 大纲

/// 章节/标题条目。`level` 从 1 开始。
public struct OutlineEntry: Sendable, Equatable {
    public var level: Int
    public var title: String

    public init(level: Int, title: String) {
        self.level = max(1, level)
        self.title = title
    }
}

/// 从 Markdown 标题或 EPUB 目录生成 `ParsedDocument.outline`。
public enum MarkdownOutline {
    /// 按层级缩进（每级 2 空格），格式与原项目目录摘要一致。
    public static func build(_ entries: [OutlineEntry]) -> String {
        let cleaned = entries.filter { !$0.title.trimmingCharacters(in: .whitespaces).isEmpty }
        guard !cleaned.isEmpty else { return "" }
        let baseLevel = cleaned.map(\.level).min() ?? 1
        return cleaned.map { entry in
            let indent = String(repeating: "  ", count: max(0, entry.level - baseLevel))
            return indent + entry.title.trimmingCharacters(in: .whitespaces)
        }.joined(separator: "\n")
    }

    /// 扫描 Markdown 标题，忽略围栏代码块内的 `#`。
    public static func scan(_ markdown: String) -> [OutlineEntry] {
        var entries: [OutlineEntry] = []
        var inFence = false
        for line in markdown.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
                inFence.toggle()
                continue
            }
            if inFence { continue }
            guard let match = headingRegex.firstMatch(
                in: line, range: NSRange(line.startIndex..<line.endIndex, in: line)) else { continue }
            guard let markerRange = Range(match.range(at: 1), in: line),
                  let titleRange = Range(match.range(at: 2), in: line) else { continue }
            let level = line[markerRange].count
            var title = String(line[titleRange]).trimmingCharacters(in: .whitespaces)
            while title.hasSuffix("#") {
                title.removeLast()
                title = title.trimmingCharacters(in: .whitespaces)
            }
            if title.isEmpty { continue }
            // 与 DocumentOutline 的 md 标题规则对齐：句末标点的 `#` 行
            // 多为 VLM/OCR 把正文句子写成的伪标题，不进大纲。
            if DocumentOutline.hasSentencePunctuation(title) { continue }
            entries.append(OutlineEntry(level: level, title: title))
        }
        return entries
    }

    private static let headingRegex = try! NSRegularExpression(
        pattern: #"^(#{1,6})\s+(.+?)\s*$"#)
}
