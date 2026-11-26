import Foundation
import ZIPFoundation
import AEXML
import CryptoKit

/// Office Open XML 解析：.docx / .xlsx / .pptx。
///
/// 仅使用 ZipFoundation + AEXML，不引入新依赖：
/// - docx：word/document.xml 的段落、列表、表格、脚注/尾注；
/// - xlsx：sharedStrings + styles（日期/百分比格式）+ 各 sheet 的行列；
/// - pptx：ppt/slides 的文本框、表格、SmartArt 与演讲者备注。
public final class OfficeParser: DocumentParser, VLMBackedParser {

    public let name = "Office"
    public let supportedExtensions = ["docx", "xlsx", "pptx"]

    private let config: AgentConfig?
    private let providedVLM: VLMService?
    /// 与 EPUB/MOBI 一致：只在真正需要 OCR 嵌入图片时懒加载模型。
    private lazy var resolvedVLM: VLMService? = providedVLM ?? config.flatMap {
        LocalModelFactory.shared.makeRAGOCRVLM(config: $0)
    }
    public var vlm: VLMService? { resolvedVLM }

    public init(config: AgentConfig? = nil, vlm: VLMService? = nil) {
        self.config = config
        self.providedVLM = vlm
    }

    public func parse(url: URL,
                      options: ParserOptions,
                      progress: ((ParserProgress) -> Void)?) async throws -> ParsedDocument {
        let archive = try RAGArchiveReader.open(url)

        do {
            let extensionName = url.pathExtension.lowercased()
            // 先按 rels 找出所有嵌入位图（排除背景/矢量图），用 Ovis 转成 Markdown；
            // 再交给各格式在图片锚点处回填，避免图片内信息在文本化时丢失。
            let candidates = embeddedImages(archive: archive, extensionName: extensionName)
            let imageMarkdown = try await recognizeEmbeddedImages(
                candidates: candidates, options: options, progress: progress)
            let result: (text: String, outline: [OutlineEntry], metadata: [String: String])
            switch extensionName {
            case "docx":
                result = try parseDocx(archive: archive, imageMarkdown: imageMarkdown)
            case "xlsx":
                result = try parseXlsx(archive: archive, imageMarkdown: imageMarkdown)
            case "pptx":
                result = try parsePptx(archive: archive, imageMarkdown: imageMarkdown)
            default:
                throw AgentError.unsupported("不支持的 Office 格式：\(extensionName)")
            }

            guard !result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw AgentError.parse("Office 文档未解析出正文：\(url.lastPathComponent)")
            }

            var metadata = result.metadata
            metadata["engine"] = extensionName

            var parsed = ParsedDocument(text: result.text,
                                        source: url.lastPathComponent,
                                        outline: MarkdownOutline.build(result.outline),
                                        metadata: metadata)
            parsed = finish(parsed, options: options, profile: .structured)
            return parsed
        } catch let error as AgentError {
            throw error
        } catch {
            throw AgentError.parse("Office 解析失败（\(url.lastPathComponent)）：\(error.localizedDescription)")
        }
    }

    // MARK: - 嵌入图片 OCR

    /// 归档内一张待 OCR 的嵌入图片。key 是各格式在锚点处查表的标识：
    /// DOCX/PPTX 用关系 id，XLSX 用 `sheet路径#行,列`。
    private struct EmbeddedImage {
        var key: String
        var data: Data
    }

    /// 扫描 OOXML 关系与 blip 锚点，收集正文中嵌图（不含背景图）。
    private func embeddedImages(archive: Archive, extensionName: String) -> [EmbeddedImage] {
        switch extensionName {
        case "docx":
            return docxEmbeddedImages(archive: archive)
        case "xlsx":
            return xlsxEmbeddedImages(archive: archive)
        case "pptx":
            return pptxEmbeddedImages(archive: archive)
        default:
            return []
        }
    }

    private func recognizeEmbeddedImages(
        candidates: [EmbeddedImage], options: ParserOptions,
        progress: ((ParserProgress) -> Void)?
    ) async throws -> [String: String] {
        guard !candidates.isEmpty else { return [:] }
        let ovis = OfficeEmbeddedImageOcr.resolveVLM(vlm, options: options)
        guard let ovis else {
            // 没有视觉模型时仍写占位，明确标记图片信息未能提取，避免静默遗漏。
            AgentLog.warning("Office 含嵌入图片但未配置 OCR 视觉模型，图片信息未提取")
            return Dictionary(uniqueKeysWithValues: candidates.map {
                ($0.key, "[图片未识别：OCR 模型不可用]")
            })
        }

        // 去重：同一 media 被多个锚点引用时只识别一次。用稳定指纹（内容哈希）
        // 建索引，避免同图多锚点回填时做 O(n²) 的 Data 全量比较。
        var unique: [EmbeddedImage] = []
        var keysByFingerprint: [String: [String]] = [:]
        var fingerprints: [String: String] = [:]
        for candidate in candidates {
            let fingerprint = SHA256.hash(data: candidate.data)
                .map { String(format: "%02x", $0) }.joined()
            if let group = keysByFingerprint[fingerprint] {
                keysByFingerprint[fingerprint] = group + [candidate.key]
                continue
            }
            keysByFingerprint[fingerprint] = [candidate.key]
            fingerprints[candidate.key] = fingerprint
            unique.append(candidate)
        }

        // 与 EPUB/MOBI 一致按批并发；VLM 实例池负责串行化单实例，单页失败跳过。
        var markdownByFingerprint: [String: String] = [:]
        let batchSize = PDFPageRenderer.maxConcurrentPages
        var cursor = 0
        while cursor < unique.count {
            if Task.isCancelled { throw AgentError.cancelled }
            let end = min(cursor + batchSize, unique.count)
            let batch = Array(unique[cursor..<end])
            let outcomes = await withTaskGroup(of: (String, String?, String).self) { group in
                for candidate in batch {
                    group.addTask {
                        do {
                            let markdown = try await OfficeEmbeddedImageOcr.recognize(
                                key: candidate.key, data: candidate.data,
                                vlm: ovis, options: options)
                            return (candidate.key, markdown, "")
                        } catch {
                            return (candidate.key, nil, error.localizedDescription)
                        }
                    }
                }
                var collected: [(String, String?, String)] = []
                for await item in group { collected.append(item) }
                return collected
            }
            for (key, markdown, errorDescription) in outcomes {
                if markdown == nil, !errorDescription.isEmpty {
                    AgentLog.warning("Office 嵌入图片 OCR 失败（\(key)）：\(errorDescription)")
                }
                guard let fingerprint = fingerprints[key] else { continue }
                // 失败也写占位：图片位置不会静默空白，上层可见且可重试。
                let value = markdown ?? "[图片 OCR 失败：\(errorDescription)]"
                markdownByFingerprint[fingerprint] = value
            }
            progress?(.page(index: end, total: unique.count, stage: "图片OCR"))
            cursor = end
        }
        var result: [String: String] = [:]
        for (fingerprint, keys) in keysByFingerprint {
            guard let markdown = markdownByFingerprint[fingerprint] else { continue }
            for key in keys { result[key] = markdown }
        }
        return result
    }

    /// DOCX：正文与页眉/页脚中的 a:blip/@r:embed → word/media/*。
    private func docxEmbeddedImages(archive: Archive) -> [EmbeddedImage] {
        let paths = ["word/document.xml", "word/footnotes.xml", "word/endnotes.xml"]
            + RAGArchiveReader.filePaths(archive)
                .filter {
                    $0.hasPrefix("word/")
                        && ($0.contains("/header") || $0.contains("/footer"))
                        && $0.hasSuffix(".xml")
                }
                .sorted()
        var result: [EmbeddedImage] = []
        for partPath in paths {
            guard let data = RAGArchiveReader.data(archive, path: partPath),
                  let document = try? RAGXMLHelper.parsePreservingWhitespace(
                    data, source: partPath) else { continue }
            let rels = relationships(archive, path: relatedPartPath(partPath))
            // 现代 DrawingML (a:blip r:embed) 与老式 VML (v:imagedata r:id) 都覆盖。
            let references = RAGXMLHelper.descendants(document.root, "blip")
                + RAGXMLHelper.descendants(document.root, "imagedata")
            for node in references {
                guard !OfficeEmbeddedImageOcr.isBackground(node),
                      let rid = RAGXMLHelper.attribute(node, ["r:embed", "embed", "r:link", "link", "r:id", "id"]),
                      let relationship = rels.first(where: { $0.id == rid }),
                      relationship.type.lowercased().hasSuffix("image") else { continue }
                let mediaPath = resolvePartTarget(relationship.target, baseDirectory: "word")
                guard let image = RAGArchiveReader.data(archive, path: mediaPath),
                      OfficeEmbeddedImageOcr.isRaster(image) else { continue }
                // 正文用裸 rId（与 docxText 一致），页眉页脚加 part 路径隔离。
                let key = partPath == "word/document.xml" ? rid : partPath + "#" + rid
                result.append(EmbeddedImage(key: key, data: image))
            }
        }
        return result
    }

    /// PPTX：slide 与 notesSlide 中的 a:blip r:embed → ppt/media/*。
    private func pptxEmbeddedImages(archive: Archive) -> [EmbeddedImage] {
        let paths = RAGArchiveReader.filePaths(archive)
            .filter {
                $0.range(of: #"^ppt/slides/slide\d+\.xml$"#, options: .regularExpression) != nil
                    || $0.range(of: #"^ppt/notesSlides/notesSlide\d+\.xml$"#,
                                options: .regularExpression) != nil
            }
        var result: [EmbeddedImage] = []
        for path in paths {
            guard let data = RAGArchiveReader.data(archive, path: path),
                  let slide = try? RAGXMLHelper.parsePreservingWhitespace(data, source: path) else {
                continue
            }
            let rels = relationships(archive, path: relatedPartPath(path))
            // 关系 id 在不同 slide 可能重复，key 加 slide 路径前缀隔离。
            let references = RAGXMLHelper.descendants(slide.root, "blip")
                + RAGXMLHelper.descendants(slide.root, "imagedata")
            for node in references {
                guard !OfficeEmbeddedImageOcr.isBackground(node),
                      let rid = RAGXMLHelper.attribute(node, ["r:embed", "embed", "r:link", "link", "r:id", "id"]),
                      let relationship = rels.first(where: { $0.id == rid }),
                      relationship.type.lowercased().hasSuffix("image") else { continue }
                let imagePath = resolvePartTarget(relationship.target, baseDirectory: "ppt/slides")
                guard let image = RAGArchiveReader.data(archive, path: imagePath),
                      OfficeEmbeddedImageOcr.isRaster(image) else { continue }
                result.append(EmbeddedImage(key: path + "#" + rid, data: image))
            }
        }
        return result
    }

    /// XLSX：worksheet 的 drawing 关系 → drawingN.xml 的锚点 → xl/media/*。
    /// 按 `xdr:from` 的行列定位，key 形如 `sheet路径#3,2`。
    private func xlsxEmbeddedImages(archive: Archive) -> [EmbeddedImage] {
        let relPaths = RAGArchiveReader.filePaths(archive)
            .filter { $0.hasPrefix("xl/worksheets/_rels/") && $0.hasSuffix(".rels") }
        var result: [EmbeddedImage] = []
        for relPath in relPaths {
            let sheetPath = relPath
                .replacingOccurrences(of: "xl/worksheets/_rels/", with: "xl/worksheets/")
                .replacingOccurrences(of: ".rels", with: "")
            for relationship in relationships(archive, path: relPath)
            where relationship.type.lowercased().hasSuffix("drawing") {
                let drawingPath = resolvePartTarget(relationship.target,
                                                    baseDirectory: "xl/worksheets")
                guard let drawingData = RAGArchiveReader.data(archive, path: drawingPath),
                      let drawing = try? RAGXMLHelper.parsePreservingWhitespace(
                        drawingData, source: drawingPath) else { continue }
                let drawingRels = relationships(archive, path: relatedPartPath(drawingPath))
                let positioned = RAGXMLHelper.descendants(drawing.root, "twoCellAnchor")
                    + RAGXMLHelper.descendants(drawing.root, "oneCellAnchor")
                // absoluteAnchor 没有 from 行列，按 A1 处理，图片信息仍不丢失。
                let absolute = RAGXMLHelper.descendants(drawing.root, "absoluteAnchor")
                for anchor in positioned + absolute {
                    let from = RAGXMLHelper.descendants(anchor, "from").first
                    let column = from.flatMap {
                        RAGXMLHelper.descendants($0, "col").first?
                            .value.flatMap({ Int($0.trimmingCharacters(in: .whitespaces)) })
                    } ?? 0
                    let row = from.flatMap {
                        RAGXMLHelper.descendants($0, "row").first?
                            .value.flatMap({ Int($0.trimmingCharacters(in: .whitespaces)) })
                    } ?? 0
                    guard let blip = RAGXMLHelper.descendants(anchor, "blip").first,
                          let rid = RAGXMLHelper.attribute(blip, ["r:embed", "embed"]),
                          let imageRel = drawingRels.first(where: { $0.id == rid }) else { continue }
                    let imagePath = resolvePartTarget(imageRel.target,
                                                      baseDirectory: "xl/drawings")
                    guard let image = RAGArchiveReader.data(archive, path: imagePath),
                          OfficeEmbeddedImageOcr.isRaster(image) else { continue }
                    // drawing 锚点是 0-based，而 worksheet 的 row@r 是 1-based。
                    result.append(EmbeddedImage(
                        key: "\(sheetPath)#\(row + 1),\(column)", data: image))
                }
            }
        }
        return result
    }

    // MARK: - DOCX

    private func parseDocx(archive: Archive,
                           imageMarkdown: [String: String]) throws -> (String, [OutlineEntry], [String: String]) {
        guard let data = RAGArchiveReader.data(archive, path: "word/document.xml") else {
            throw AgentError.parse("DOCX 缺少 word/document.xml")
        }
        let document = try RAGXMLHelper.parsePreservingWhitespace(data, source: "word/document.xml")
        guard let body = RAGXMLHelper.descendants(document.root, "body").first else {
            throw AgentError.parse("DOCX 缺少文档正文（w:body）")
        }

        let numbering = docxNumbering(archive)
        let styles = docxStyles(archive)
        let footnotes = DocxNotes(texts: docxNotes(archive, path: "word/footnotes.xml",
                                                  imageMarkdown: imageMarkdown))
        let endnotes = DocxNotes(texts: docxNotes(archive, path: "word/endnotes.xml",
                                                 imageMarkdown: imageMarkdown))
        let counters = DocxListCounters()

        var blocks: [String] = []
        var headings: [OutlineEntry] = []
        renderDocxChildren(body, blocks: &blocks, headings: &headings,
                           numbering: numbering, styles: styles, counters: counters,
                           footnotes: footnotes, endnotes: endnotes,
                           imageMarkdown: imageMarkdown)
        blocks.append(contentsOf: footnotes.sections(title: "脚注"))
        blocks.append(contentsOf: endnotes.sections(title: "尾注"))

        let headersFooters = docxHeadersFooters(archive, imageMarkdown: imageMarkdown)
        if !headersFooters.isEmpty {
            blocks.append("## 页眉页脚")
            blocks.append(contentsOf: headersFooters)
        }

        var metadata: [String: String] = [:]
        if footnotes.count > 0 { metadata["footnotes"] = String(footnotes.count) }
        if endnotes.count > 0 { metadata["endnotes"] = String(endnotes.count) }
        if !headersFooters.isEmpty { metadata["header_footer_blocks"] = String(headersFooters.count) }
        if !imageMarkdown.isEmpty { metadata["embedded_images_ocr"] = String(imageMarkdown.count) }

        return (blocks.joined(separator: "\n\n"), headings, metadata)
    }

    private func renderDocxChildren(_ element: AEXMLElement,
                                    blocks: inout [String],
                                    headings: inout [OutlineEntry],
                                    numbering: DocxNumbering,
                                    styles: [String: Int],
                                    counters: DocxListCounters,
                                    footnotes: DocxNotes,
                                    endnotes: DocxNotes,
                                    imageMarkdown: [String: String]) {
        for child in element.children {
            switch RAGXMLHelper.localName(child.name).lowercased() {
            case "p":
                let text = docxParagraphText(child, footnotes: footnotes, endnotes: endnotes,
                                             imageMarkdown: imageMarkdown)
                guard !text.isEmpty else { continue }
                if let level = docxHeadingLevel(child, styles: styles) {
                    blocks.append(String(repeating: "#", count: level) + " " + text)
                    headings.append(OutlineEntry(level: level, title: text))
                } else if let marker = docxListMarker(child, numbering: numbering,
                                                      counters: counters) {
                    blocks.append(marker + text)
                } else {
                    blocks.append(text)
                }

            case "tbl":
                let table = docxTable(child, footnotes: footnotes, endnotes: endnotes,
                                      imageMarkdown: imageMarkdown)
                if !table.isEmpty { blocks.append(table) }

            case "sectpr", "tblpr", "trpr", "tcpr", "ppr", "rpr", "props":
                continue

            case "sdt", "sdtcontent", "body":
                renderDocxChildren(child, blocks: &blocks, headings: &headings,
                                   numbering: numbering, styles: styles, counters: counters,
                                   footnotes: footnotes, endnotes: endnotes,
                                   imageMarkdown: imageMarkdown)

            default:
                renderDocxChildren(child, blocks: &blocks, headings: &headings,
                                   numbering: numbering, styles: styles, counters: counters,
                                   footnotes: footnotes, endnotes: endnotes,
                                   imageMarkdown: imageMarkdown)
            }
        }
    }

    /// 段落文本：按运行顺序拼接 `w:t`，并把 `w:br`/`w:cr` 转成换行、
    /// `w:tab` 转成制表符；同时跳过批注/删除文本/域代码/注音等非正文内容。
    private func docxParagraphText(_ paragraph: AEXMLElement,
                                   footnotes: DocxNotes?,
                                   endnotes: DocxNotes?,
                                   imageMarkdown: [String: String]) -> String {
        docxText(paragraph, footnotes: footnotes, endnotes: endnotes,
                 imageMarkdown: imageMarkdown)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func docxText(_ element: AEXMLElement,
                          footnotes: DocxNotes?,
                          endnotes: DocxNotes?,
                          imageMarkdown: [String: String]) -> String {
        var result = ""
        for child in element.children {
            switch RAGXMLHelper.localName(child.name).lowercased() {
            // pPr/rPr 是属性容器；rPh/rt 是注音；delText 是修订删除的文本；
            // instrText/fldChar 是域代码，都不属于正文。
            case "ppr", "rpr", "sectpr", "rph", "rt", "deltext", "instrtext",
                 "fldchar", "del":
                continue
            case "t":
                result += preservedText(child)
            case "br", "cr":
                result += "\n"
            case "tab":
                result += "\t"
            case "footnotereference":
                if let id = RAGXMLHelper.attribute(child, ["id"]) {
                    result += footnotes?.marker(for: id) ?? ""
                }
            case "endnotereference":
                if let id = RAGXMLHelper.attribute(child, ["id"]) {
                    result += endnotes?.marker(for: id) ?? ""
                }
            // 图片/OLE/公式没有文本节点，取 docPr 的 descr/name 作替代文本，
            // 否则整段图像内容在纯文本 RAG 里彻底消失。
            case "drawing", "object", "pict":
                // 优先放回 Ovis 识别出的图片正文；没有 OCR 结果时退回替代文本。
                let refs = RAGXMLHelper.descendants(child, "blip")
                    + RAGXMLHelper.descendants(child, "imagedata")
                let markdown = refs
                    .compactMap { RAGXMLHelper.attribute($0, ["r:embed", "embed", "r:link", "link", "r:id", "id"]) }
                    .compactMap { imageMarkdown[$0] }
                    .joined(separator: "\n")
                if !markdown.isEmpty {
                    result += markdown
                } else if let alt = docxAlternativeText(child), !alt.isEmpty {
                    result += "[\(alt)]"
                }
            case "sym":
                if let char = RAGXMLHelper.attribute(child, ["char"]),
                   let scalarValue = UInt32(char, radix: 16),
                   let scalar = UnicodeScalar(scalarValue) {
                    result += String(scalar)
                }
            default:
                result += docxText(child, footnotes: footnotes, endnotes: endnotes,
                                   imageMarkdown: imageMarkdown)
            }
        }
        return result
    }

    /// `wp:docPr` / `v:shape` 的替代文本；优先 descr，其次 title/name/alt。
    private func docxAlternativeText(_ element: AEXMLElement) -> String? {
        for name in ["docpr", "shape"] {
            for node in RAGXMLHelper.descendants(element, name) {
                for key in ["descr", "title", "name", "alt"] {
                    if let value = RAGXMLHelper.attribute(node, [key])?
                        .trimmingCharacters(in: .whitespacesAndNewlines),
                       !value.isEmpty {
                        return value
                    }
                }
            }
        }
        return nil
    }

    /// 文本节点取值：声明了 `xml:space="preserve"` 的保留原始空白，
    /// 其余按 Word 语义忽略前后空白。
    private func preservedText(_ element: AEXMLElement) -> String {
        let raw = element.value ?? ""
        if let space = RAGXMLHelper.attribute(element, ["xml:space", "space"]),
           space == "preserve" {
            return raw
        }
        return raw.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// 优先读取 outlineLvl，其次从 pStyle 的 Heading N / 标题 N 读取。
    ///
    /// 很多文档的样式 ID 是 `1`/`2`/`Heading1` 这类无语义的值（例如 WPS 导出），
    /// 只有 `styles.xml` 里的 `<w:name>` 才写着「heading 1」；因此
    /// `styles`（styleId → 标题级别）在正则匹配失败时兜底。
    private func docxHeadingLevel(_ paragraph: AEXMLElement,
                                  styles: [String: Int] = [:]) -> Int? {
        guard let properties = RAGXMLHelper.firstChild(paragraph, "pPr") else { return nil }

        if let outline = RAGXMLHelper.firstChild(properties, "outlineLvl"),
           let raw = RAGXMLHelper.attribute(outline, ["val"]),
           let level = Int(raw.trimmingCharacters(in: .whitespaces)) {
            // outlineLvl 是零基；val=9 表示正文（无大纲级别）。
            // 非法值不能早退，否则丢失 pStyle / styles.xml 的兜底机会。
            if level >= 0, level <= 8 { return min(6, level + 1) }
        }

        if let style = RAGXMLHelper.firstChild(properties, "pStyle"),
           let value = RAGXMLHelper.attribute(style, ["val"])?
            .trimmingCharacters(in: .whitespaces) {
            if value.lowercased() == "title" { return 1 }
            let range = NSRange(value.startIndex..<value.endIndex, in: value)
            if let match = Self.docxHeadingStyleRegex.firstMatch(in: value, range: range),
               let levelRange = Range(match.range(at: 1), in: value),
               let level = Int(value[levelRange]) {
                return level
            }
            if let level = styles[value] { return level }
        }
        return nil
    }

    /// 从 `styles.xml` 读取「样式 ID → 标题级别」。
    ///
    /// Word/WPS 常把样式 ID 与显示名解耦（ID 为 `1`，名称为「heading 1」），
    /// 只看 `pStyle` 的值会漏掉全部标题；这里同时认 `<w:name>` 与
    /// `<w:outlineLvl>`，并沿 basedOn 链继承。
    private func docxStyles(_ archive: Archive) -> [String: Int] {
        guard let data = RAGArchiveReader.data(archive, path: "word/styles.xml"),
              let document = try? RAGXMLHelper.parsePreservingWhitespace(
                data, source: "word/styles.xml") else {
            return [:]
        }

        struct StyleInfo {
            var name = ""
            var outlineLevel: Int?
            var basedOn = ""
        }

        var infos: [String: StyleInfo] = [:]
        var order: [String] = []
        for style in RAGXMLHelper.descendants(document.root, "style") {
            guard let id = RAGXMLHelper.attribute(style, ["styleId"])?
                .trimmingCharacters(in: .whitespaces), !id.isEmpty else { continue }
            var info = StyleInfo()
            info.name = RAGXMLHelper.firstChild(style, "name")
                .flatMap { RAGXMLHelper.attribute($0, ["val"]) } ?? ""
            if let outline = RAGXMLHelper.firstChild(style, "pPr")
                .flatMap({ RAGXMLHelper.firstChild($0, "outlineLvl") }),
               let raw = RAGXMLHelper.attribute(outline, ["val"]),
               let level = Int(raw.trimmingCharacters(in: .whitespaces)) {
                info.outlineLevel = min(6, max(1, level + 1))
            }
            info.basedOn = RAGXMLHelper.firstChild(style, "basedOn")
                .flatMap { RAGXMLHelper.attribute($0, ["val"]) } ?? ""
            infos[id] = info
            order.append(id)
        }

        // basedOn 链可能指向尚未处理的样式，先按文档顺序展开再递归补齐。
        var levels: [String: Int] = [:]
        func resolve(_ id: String, depth: Int = 0) -> Int? {
            if let cached = levels[id] { return cached }
            guard depth < 8, let info = infos[id] else { return nil }
            var level = info.outlineLevel
            if level == nil {
                let range = NSRange(info.name.startIndex..<info.name.endIndex, in: info.name)
                if info.name.lowercased() == "title" {
                    level = 1
                } else if let match = Self.docxHeadingStyleRegex.firstMatch(
                    in: info.name, range: range),
                          let levelRange = Range(match.range(at: 1), in: info.name),
                          let parsed = Int(info.name[levelRange]) {
                    level = parsed
                }
            }
            if level == nil, !info.basedOn.isEmpty {
                level = resolve(info.basedOn, depth: depth + 1)
            }
            if let level { levels[id] = level }
            return level
        }
        for id in order { _ = resolve(id) }
        return levels
    }

    /// 列表标记：`w:numPr` + numbering.xml 的编号格式决定「- 」还是「1. 」。
    private func docxListMarker(_ paragraph: AEXMLElement, numbering: DocxNumbering,
                                counters: DocxListCounters) -> String? {
        guard let properties = RAGXMLHelper.firstChild(paragraph, "pPr"),
              let numberingProperties = RAGXMLHelper.firstChild(properties, "numPr"),
              let numIdElement = RAGXMLHelper.firstChild(numberingProperties, "numId"),
              let numId = RAGXMLHelper.attribute(numIdElement, ["val"])?
                .trimmingCharacters(in: .whitespaces),
              !numId.isEmpty, numId != "0" else { return nil }

        let rawLevel = RAGXMLHelper.firstChild(numberingProperties, "ilvl")
            .flatMap { RAGXMLHelper.attribute($0, ["val"]) }
            .flatMap { Int($0.trimmingCharacters(in: .whitespaces)) } ?? 0
        let level = min(8, max(0, rawLevel))
        // 超过 3 级缩进的排版在 Markdown 里没有意义，直接压平。
        let indent = String(repeating: "  ", count: min(2, level))
        let info = numbering.level(numId: numId, index: level)
        return indent + counters.marker(numId: numId, level: level, ordered: info.ordered)
    }

    private func docxTable(_ table: AEXMLElement, footnotes: DocxNotes?,
                           endnotes: DocxNotes?,
                           imageMarkdown: [String: String]) -> String {
        var rows: [[String]] = []
        // Markdown 表无纵向合并语法：把 vMerge 区域全部文字并入 restart 起点，
        // continue 行保留空占位以维持列对齐。
        struct VerticalAnchor { var column: Int; var row: Int; var text: String }
        var anchors: [VerticalAnchor] = []
        for row in RAGXMLHelper.children(table, "tr") {
            var cells: [String] = []
            var column = 0
            for cell in RAGXMLHelper.children(row, "tc") {
                let properties = RAGXMLHelper.firstChild(cell, "tcPr")
                let span = properties
                    .flatMap { RAGXMLHelper.firstChild($0, "gridSpan") }
                    .flatMap { RAGXMLHelper.attribute($0, ["val"]) }
                    .flatMap { Int($0.trimmingCharacters(in: .whitespaces)) } ?? 1
                let gridSpan = min(max(span, 1), 33)
                let merge = properties
                    .flatMap { RAGXMLHelper.firstChild($0, "vMerge") }
                let mergeValue = merge
                    .flatMap { RAGXMLHelper.attribute($0, ["val"]) }?
                    .trimmingCharacters(in: .whitespaces).lowercased()
                let isRestart = merge != nil && (mergeValue == "restart" || mergeValue == "1")
                let isContinue = merge != nil && !isRestart
                let text = docxCellText(cell, footnotes: footnotes, endnotes: endnotes,
                                        imageMarkdown: imageMarkdown)

                while cells.count < column { cells.append("") }
                if isRestart {
                    anchors.append(VerticalAnchor(column: column, row: rows.count, text: text))
                }
                if isContinue {
                    if let index = anchors.lastIndex(where: {
                        $0.column == column && $0.row < rows.count
                    }), !text.isEmpty {
                        let previous = anchors[index].text
                        anchors[index].text = previous.isEmpty ? text : previous + " " + text
                    }
                }
                cells.append(isContinue ? "" : text)
                if gridSpan > 1 {
                    cells.append(contentsOf: Array(repeating: "", count: gridSpan - 1))
                }
                column += gridSpan
                guard column <= 256 else { break }
            }
            if !cells.isEmpty { rows.append(cells) }
        }
        for anchor in anchors where rows.indices.contains(anchor.row)
            && rows[anchor.row].indices.contains(anchor.column) {
            rows[anchor.row][anchor.column] = anchor.text
        }
        return HTMLMarkdown.renderMarkdownTable(rows)
    }

    private func docxCellText(_ cell: AEXMLElement, footnotes: DocxNotes?,
                              endnotes: DocxNotes?,
                              imageMarkdown: [String: String]) -> String {
        var paragraphs: [String] = []
        for child in cell.children {
            switch RAGXMLHelper.localName(child.name).lowercased() {
            case "p":
                let text = docxParagraphText(child, footnotes: footnotes, endnotes: endnotes,
                                             imageMarkdown: imageMarkdown)
                if !text.isEmpty { paragraphs.append(text) }
            case "tbl", "sdt", "sdtcontent":
                // 嵌套表格/内容控件：扁平化为单元格文本，避免内容丢失。
                for nested in RAGXMLHelper.descendants(child, "p") {
                    let text = docxParagraphText(nested, footnotes: footnotes,
                                                 endnotes: endnotes,
                                                 imageMarkdown: imageMarkdown)
                    if !text.isEmpty { paragraphs.append(text) }
                }
            default:
                break
            }
        }
        return HTMLMarkdown.escapeTableCell(paragraphs.joined(separator: " "))
    }

    /// 编号定义：numId → 各级是否有序。
    private func docxNumbering(_ archive: Archive) -> DocxNumbering {
        guard let data = RAGArchiveReader.data(archive, path: "word/numbering.xml"),
              let document = try? RAGXMLHelper.parsePreservingWhitespace(
                data, source: "word/numbering.xml") else {
            return DocxNumbering()
        }

        var abstractLevels: [String: [Int: Bool]] = [:]
        for abstract in RAGXMLHelper.descendants(document.root, "abstractNum") {
            guard let id = RAGXMLHelper.attribute(abstract, ["abstractNumId"]) else { continue }
            var levels: [Int: Bool] = [:]
            for level in RAGXMLHelper.children(abstract, "lvl") {
                levels[docxLevelIndex(level)] = docxLevelIsOrdered(level)
            }
            abstractLevels[id] = levels
        }

        var numbering = DocxNumbering()
        for num in RAGXMLHelper.descendants(document.root, "num") {
            guard let numId = RAGXMLHelper.attribute(num, ["numId"]),
                  let abstractElement = RAGXMLHelper.firstChild(num, "abstractNumId"),
                  let abstractId = RAGXMLHelper.attribute(abstractElement, ["val"]) else { continue }
            var levels = abstractLevels[abstractId] ?? [:]
            // lvlOverride 可以覆盖某一级的编号格式。
            for override in RAGXMLHelper.children(num, "lvlOverride") {
                guard let level = RAGXMLHelper.firstChild(override, "lvl") else { continue }
                levels[docxLevelIndex(override)] = docxLevelIsOrdered(level)
            }
            numbering.byNumId[numId] = levels
        }
        return numbering
    }

    private func docxLevelIndex(_ element: AEXMLElement) -> Int {
        RAGXMLHelper.attribute(element, ["ilvl"])
            .flatMap { Int($0.trimmingCharacters(in: .whitespaces)) } ?? 0
    }

    /// `numFmt` 为 bullet/none 视为无序列表，其余（decimal/chineseCounting…）为有序。
    private func docxLevelIsOrdered(_ level: AEXMLElement) -> Bool {
        guard let format = RAGXMLHelper.firstChild(level, "numFmt")
            .flatMap({ RAGXMLHelper.attribute($0, ["val"]) })?.lowercased() else { return true }
        return format != "bullet" && format != "none"
    }

    /// 页眉/页脚：按归档路径排序逐个解析 `word/headerN.xml` / `footerN.xml`，
    /// 去重后作为独立块附在正文末尾。同一内容常被多个 section 引用，逐引用
    /// 输出会重复，因此按文件内容去重。
    private func docxHeadersFooters(_ archive: Archive,
                                    imageMarkdown: [String: String]) -> [String] {
        let paths = RAGArchiveReader.filePaths(archive)
            .filter {
                $0.hasPrefix("word/") && ($0.contains("/header") || $0.contains("/footer"))
                    && $0.hasSuffix(".xml")
            }
            .sorted()
        var result: [String] = []
        var seen = Set<String>()
        for path in paths {
            guard let data = RAGArchiveReader.data(archive, path: path),
                  let document = try? RAGXMLHelper.parsePreservingWhitespace(
                    data, source: path) else { continue }
            let label = path.contains("header") ? "页眉" : "页脚"
            // 页眉页脚段落需按 part 查表，避免与正文同 rId 串用。
            var headerMarkdown: [String: String] = [:]
            for (key, value) in imageMarkdown where key.hasPrefix(path + "#") {
                headerMarkdown[String(key.dropFirst(path.count + 1))] = value
            }
            for paragraph in RAGXMLHelper.descendants(document.root, "p") {
                let text = docxParagraphText(paragraph, footnotes: nil, endnotes: nil,
                                             imageMarkdown: headerMarkdown)
                guard !text.isEmpty, seen.insert(text).inserted else { continue }
                result.append("\(label)：\(text)")
            }
        }
        return result
    }

    /// 脚注/尾注：id → 文本。分隔线伪注释用 `w:type` 标识（separator /
    /// continuationSeparator），而不是固定 id 0/1——真实文档里 id=1 常是正文脚注。
    private func docxNotes(_ archive: Archive, path: String,
                           imageMarkdown: [String: String]) -> [String: String] {
        guard let data = RAGArchiveReader.data(archive, path: path),
              let document = try? RAGXMLHelper.parsePreservingWhitespace(data, source: path) else {
            return [:]
        }
        var noteMarkdown: [String: String] = [:]
        for (key, value) in imageMarkdown where key.hasPrefix(path + "#") {
            noteMarkdown[String(key.dropFirst(path.count + 1))] = value
        }
        var notes: [String: String] = [:]
        for note in RAGXMLHelper.descendants(document.root, "footnote")
            + RAGXMLHelper.descendants(document.root, "endnote") {
            guard let id = RAGXMLHelper.attribute(note, ["id"])?
                .trimmingCharacters(in: .whitespaces), !id.isEmpty else { continue }
            if let type = RAGXMLHelper.attribute(note, ["type"])?.lowercased(),
               type != "normal" {
                continue
            }
            let text = RAGXMLHelper.children(note, "p")
                .map { docxParagraphText($0, footnotes: nil, endnotes: nil,
                                         imageMarkdown: noteMarkdown) }
                .filter { !$0.isEmpty }
                .joined(separator: "\n")
            if !text.isEmpty { notes[id] = text }
        }
        return notes
    }

    // MARK: - XLSX

    private func parseXlsx(archive: Archive,
                           imageMarkdown: [String: String]) throws -> (String, [OutlineEntry], [String: String]) {
        guard let workbookData = RAGArchiveReader.data(archive, path: "xl/workbook.xml") else {
            throw AgentError.parse("XLSX 缺少 xl/workbook.xml")
        }
        let workbook = try RAGXMLHelper.parsePreservingWhitespace(workbookData,
                                                                 source: "xl/workbook.xml")
        let sheets = RAGXMLHelper.descendants(workbook.root, "sheet")
        guard !sheets.isEmpty else {
            throw AgentError.parse("XLSX 未声明任何工作表")
        }
        let date1904 = RAGXMLHelper.descendants(workbook.root, "workbookPr")
            .first
            .flatMap { RAGXMLHelper.attribute($0, ["date1904"]) }
            .map { $0 == "1" || $0.lowercased() == "true" } ?? false
        let styles = xlsxStyles(archive)

        var relationships: [String: String] = [:]
        if let relsData = RAGArchiveReader.data(archive, path: "xl/_rels/workbook.xml.rels"),
           let rels = try? AEXMLDocument(xml: relsData) {
            for relationship in RAGXMLHelper.descendants(rels.root, "Relationship") {
                if let id = RAGXMLHelper.attribute(relationship, ["Id"]),
                   let target = RAGXMLHelper.attribute(relationship, ["Target"]) {
                    relationships[id] = target
                }
            }
        }

        var sharedStrings: [String] = []
        if let sharedData = RAGArchiveReader.data(archive, path: "xl/sharedStrings.xml"),
           let shared = try? RAGXMLHelper.parsePreservingWhitespace(
            sharedData, source: "xl/sharedStrings.xml") {
            sharedStrings = RAGXMLHelper.descendants(shared.root, "si")
                .map { xlsxRichTextPreserved($0) }
        }

        var sections: [String] = []
        var outline: [OutlineEntry] = []
        var truncatedSheets: [String] = []
        var truncatedSheetList = false
        for (index, sheet) in sheets.enumerated() {
            if index >= Self.maxXlsxSheets { truncatedSheetList = true; break }
            let sheetName = RAGXMLHelper.attribute(sheet, ["name"]) ?? "Sheet\(index + 1)"
            var sheetPath: String?
            if let relationshipID = RAGXMLHelper.attribute(sheet, ["r:id"]),
               let target = relationships[relationshipID] {
                sheetPath = resolvePartTarget(target, baseDirectory: "xl")
            }
            if let candidate = sheetPath,
               RAGArchiveReader.data(archive, path: candidate) != nil {
                sheetPath = candidate
            } else {
                sheetPath = "xl/worksheets/sheet\(index + 1).xml"
            }
            guard let path = sheetPath,
                  let sheetData = RAGArchiveReader.data(archive, path: path),
                  let sheetDocument = try? RAGXMLHelper.parsePreservingWhitespace(
                    sheetData, source: path) else {
                continue
            }

            var rows: [[String]] = []
            var truncated = false
            let mergeRanges = xlsxMergeRanges(sheetDocument.root)
            var consumedImageKeys = Set<String>()
            for row in RAGXMLHelper.descendants(sheetDocument.root, "row") {
                if rows.count >= Self.maxXlsxRows { truncated = true; break }
                var values: [Int: String] = [:]
                for (cellIndex, cell) in RAGXMLHelper.children(row, "c").enumerated() {
                    let column = RAGXMLHelper.attribute(cell, ["r"])
                        .map { columnIndex(from: $0) } ?? cellIndex
                    // 远端列（异常引用 / 格式残留）会把行宽撑到上万列，
                    // 限制到常见工作表范围，避免生成巨型空表。
                    guard column >= 0, column < Self.maxXlsxColumns else { continue }
                    var value = xlsxCellText(cell, sharedStrings: sharedStrings,
                                             styles: styles, date1904: date1904)
                    // 图片锚点处的 Ovis 文本回填到所在单元格，避免图片信息缺失。
                    if let rowIndex = Int(RAGXMLHelper.attribute(row, ["r"]) ?? "") {
                        let key = "\(path)#\(rowIndex),\(column)"
                        if let markdown = imageMarkdown[key] {
                            value = value.isEmpty ? markdown : value + " " + markdown
                            consumedImageKeys.insert(key)
                        }
                    }
                    // 单元格里的换行/竖线会截断 Markdown 表格，必须先转义。
                    if !value.isEmpty { values[column] = HTMLMarkdown.escapeTableCell(value) }
                }
                guard !values.isEmpty else { continue }
                let width = (values.keys.max() ?? 0) + 1
                rows.append((0..<width).map { values[$0] ?? "" })
            }
            applyXlsxMerges(rows: &rows, ranges: mergeRanges)
            if truncated { truncatedSheets.append(sheetName) }

            let table = HTMLMarkdown.renderMarkdownTable(rows)
            // 锚定在空单元格/空行/absoluteAnchor 的图片没有 <c> 可回填，
            // 追加到该 sheet 末尾，不让 OCR 结果被静默丢弃。
            let orphans = imageMarkdown.keys
                .filter { $0.hasPrefix(path + "#") && !consumedImageKeys.contains($0) }
                .sorted()
            // 空表但含图片时必须保留该 sheet，否则图片信息随 continue 一起丢失。
            guard !table.isEmpty || !orphans.isEmpty else { continue }
            var section = "## Sheet: \(sheetName)\n\n"
            if !table.isEmpty { section += table }
            for key in orphans {
                if let markdown = imageMarkdown[key] {
                    section += "\n\n[嵌入图片 \(key)]\n" + markdown
                }
            }
            sections.append(section)
            outline.append(OutlineEntry(level: 2, title: "Sheet: \(sheetName)"))
        }

        guard !sections.isEmpty else {
            throw AgentError.parse("XLSX 未读取到任何工作表数据")
        }
        var metadata: [String: String] = ["sheets": String(sections.count)]
        if !imageMarkdown.isEmpty { metadata["embedded_images_ocr"] = String(imageMarkdown.count) }
        if truncatedSheetList {
            metadata["truncated_sheet_list"] = String(sheets.count)
            metadata["max_sheets"] = String(Self.maxXlsxSheets)
        }
        if !truncatedSheets.isEmpty {
            // 行数超过上限时明确记录，避免大表被静默截断后无法察觉。
            metadata["truncated_sheets"] = truncatedSheets.joined(separator: ",")
            metadata["max_rows_per_sheet"] = String(Self.maxXlsxRows)
        }
        return (sections.joined(separator: "\n\n"), outline, metadata)
    }

    /// `mergeCells/mergeCell ref="A1:C2"`：用左上角的值填充整个区域，
    /// Markdown 表格没有合并语法，填充后检索仍能命中合并单元格文本。
    private func xlsxMergeRanges(_ root: AEXMLElement) -> [(row: Int, column: Int, rowEnd: Int, columnEnd: Int)] {
        var result: [(row: Int, column: Int, rowEnd: Int, columnEnd: Int)] = []
        for container in RAGXMLHelper.descendants(root, "mergeCells") {
            for cell in RAGXMLHelper.children(container, "mergeCell") {
                guard let ref = RAGXMLHelper.attribute(cell, ["ref"]) else { continue }
                let parts = ref.split(separator: ":", maxSplits: 1).map(String.init)
                guard parts.count == 2,
                      let first = xlsxCellPosition(parts[0]),
                      let last = xlsxCellPosition(parts[1]) else { continue }
                result.append((first.row, first.column,
                               min(last.row, Self.maxXlsxRows - 1),
                               min(last.column, Self.maxXlsxColumns - 1)))
            }
        }
        return result
    }

    private func xlsxCellPosition(_ reference: String) -> (row: Int, column: Int)? {
        let letters = reference.prefix { $0.isLetter }
        guard !letters.isEmpty,
              let row = Int(reference.dropFirst(letters.count)) else { return nil }
        return (row - 1, columnIndex(from: String(letters)))
    }

    private func applyXlsxMerges(rows: inout [[String]],
                                 ranges: [(row: Int, column: Int, rowEnd: Int, columnEnd: Int)]) {
        for range in ranges {
            guard rows.indices.contains(range.row),
                  rows[range.row].indices.contains(range.column) else { continue }
            let value = rows[range.row][range.column]
            guard !value.isEmpty else { continue }
            for row in max(range.row, 0)...min(range.rowEnd, rows.count - 1) {
                guard rows[row].indices.contains(range.column) else { continue }
                for column in range.column...min(range.columnEnd, rows[row].count - 1) {
                    if rows[row][column].isEmpty { rows[row][column] = value }
                }
            }
        }
    }

    /// 单元格取值：按 `t` 类型分支，并对数值套用日期/百分比格式。
    private func xlsxCellText(_ cell: AEXMLElement, sharedStrings: [String],
                              styles: XlsxStyles, date1904: Bool) -> String {
        let type = RAGXMLHelper.attribute(cell, ["t"]) ?? ""
        switch type {
        case "s":
            guard let raw = RAGXMLHelper.firstChild(cell, "v")?.value?
                .trimmingCharacters(in: .whitespacesAndNewlines),
                  let index = Int(raw), sharedStrings.indices.contains(index) else { return "" }
            return sharedStrings[index]
        case "inlineStr":
            return xlsxRichText(cell).trimmingCharacters(in: .whitespacesAndNewlines)
        case "str":
            return RAGXMLHelper.firstChild(cell, "v")?.value?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        case "b":
            guard let raw = RAGXMLHelper.firstChild(cell, "v")?.value?
                .trimmingCharacters(in: .whitespacesAndNewlines) else { return "" }
            return raw == "1" ? "TRUE" : "FALSE"
        case "e":
            return RAGXMLHelper.firstChild(cell, "v")?.value?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        default:
            let styleIndex = RAGXMLHelper.attribute(cell, ["s"])
                .flatMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            guard let raw = RAGXMLHelper.firstChild(cell, "v")?.value?
                .trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
                // 无缓存值的公式也要保留语义：Excel/WPS 常在保存后清空 <v>。
                if let formula = RAGXMLHelper.firstChild(cell, "f")?.value?
                    .trimmingCharacters(in: .whitespacesAndNewlines), !formula.isEmpty {
                    return "=\(formula)"
                }
                return ""
            }
            if styles.isDateStyle(styleIndex), let serial = Double(raw),
               let text = xlsxDateText(serial: serial, date1904: date1904) {
                return text
            }
            if styles.isPercentStyle(styleIndex), let number = Double(raw) {
                return xlsxPercentText(number)
            }
            // 有公式且带缓存值：同时给出公式与结果，避免只留值丢失推导过程。
            if let formula = RAGXMLHelper.firstChild(cell, "f")?.value?
                .trimmingCharacters(in: .whitespacesAndNewlines), !formula.isEmpty {
                return "=\(formula) → \(raw)"
            }
            return raw
        }
    }

    /// 富文本取值：跳过 `rPh`（注音），避免注音字符混进正文。
    private func xlsxRichText(_ element: AEXMLElement) -> String {
        xlsxRichTextPreserved(element).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// 保留 `xml:space="preserve"` 的 `<t>` 原始空白；共享字符串里前后空格
    /// 是数据的一部分，不能无条件 trim。
    private func xlsxRichTextPreserved(_ element: AEXMLElement) -> String {
        var result = ""
        func walk(_ node: AEXMLElement) {
            for child in node.children {
                switch RAGXMLHelper.localName(child.name).lowercased() {
                case "rph", "phoneticpr": continue
                case "t": result += preservedText(child)
                default: walk(child)
                }
            }
        }
        walk(element)
        return result
    }

    /// 数字格式表：cellXfs 顺序即单元格 `s` 属性，映射到 numFmtId。
    private func xlsxStyles(_ archive: Archive) -> XlsxStyles {
        guard let data = RAGArchiveReader.data(archive, path: "xl/styles.xml"),
              let document = try? RAGXMLHelper.parsePreservingWhitespace(
                data, source: "xl/styles.xml") else {
            return XlsxStyles()
        }
        var styles = XlsxStyles()
        for format in RAGXMLHelper.descendants(document.root, "numFmt") {
            guard let rawID = RAGXMLHelper.attribute(format, ["numFmtId"])
                .flatMap({ Int($0.trimmingCharacters(in: .whitespaces)) }),
                  let code = RAGXMLHelper.attribute(format, ["formatCode"]) else { continue }
            styles.customFormats[rawID] = code
        }
        if let cellXfs = RAGXMLHelper.descendants(document.root, "cellXfs").first {
            styles.cellFormatIDs = RAGXMLHelper.children(cellXfs, "xf").map {
                RAGXMLHelper.attribute($0, ["numFmtId"])
                    .flatMap { Int($0.trimmingCharacters(in: .whitespaces)) } ?? 0
            }
        }
        return styles
    }

    /// Excel 序列号 → 日期文本（1900 系统含 1900-02-29 虚构闰日）。
    private func xlsxDateText(serial: Double, date1904: Bool) -> String? {
        guard serial.isFinite, serial >= 0, serial < 2_958_466 else { return nil }
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0) ?? .current

        var components = DateComponents()
        components.year = date1904 ? 1904 : 1899
        components.month = date1904 ? 1 : 12
        components.day = date1904 ? 1 : 30
        guard let base = calendar.date(from: components) else { return nil }

        // Excel 1900 系统虚构了 1900-02-29（序列号 60），Foundation 无此日期，
        // 直接输出Excel 的显示值；其余日期按 1899-12-30 基准换算。
        if !date1904, serial == 60 { return "1900-02-29" }
        // 1900 系统在 1900-03-01 之前的日子比基准少一天（序列号 1 = 1900-01-01）。
        let offset = date1904 || serial >= 60 ? serial : serial + 1
        guard let date = calendar.date(byAdding: .second,
                                       value: Int((offset * 86_400).rounded()), to: base) else {
            return nil
        }
        let parts = calendar.dateComponents([.year, .month, .day, .hour, .minute, .second],
                                            from: date)
        let day = String(format: "%04d-%02d-%02d", parts.year ?? 0, parts.month ?? 0,
                         parts.day ?? 0)
        let hour = parts.hour ?? 0, minute = parts.minute ?? 0, second = parts.second ?? 0
        if hour == 0 && minute == 0 && second == 0 { return day }
        if second != 0 { return day + String(format: " %02d:%02d:%02d", hour, minute, second) }
        return day + String(format: " %02d:%02d", hour, minute)
    }

    /// 百分比：0.5 → 「50%」，去掉无意义的小数尾零。
    private func xlsxPercentText(_ number: Double) -> String {
        var body = String(format: "%.4f", number * 100)
        if body.contains(".") {
            while body.hasSuffix("0") { body.removeLast() }
            if body.hasSuffix(".") { body.removeLast() }
        }
        return body + "%"
    }

    /// "C12" → 2（零基列号）。
    private func columnIndex(from reference: String) -> Int {
        var value = 0
        for scalar in reference.uppercased().unicodeScalars {
            guard scalar.value >= 65, scalar.value <= 90 else { break }
            value = value * 26 + Int(scalar.value - 64)
        }
        return max(0, value - 1)
    }

    // MARK: - PPTX

    private func parsePptx(archive: Archive,
                           imageMarkdown: [String: String]) throws -> (String, [OutlineEntry], [String: String]) {
        let presentationRels = relationships(archive, path: "ppt/_rels/presentation.xml.rels")
        var relationshipsByID: [String: String] = [:]
        for relationship in presentationRels {
            relationshipsByID[relationship.id] = relationship.target
        }

        var slidePaths: [String] = []
        if let presentationData = RAGArchiveReader.data(archive, path: "ppt/presentation.xml"),
           let presentation = try? RAGXMLHelper.parsePreservingWhitespace(
            presentationData, source: "ppt/presentation.xml") {
            for slideID in RAGXMLHelper.descendants(presentation.root, "sldId") {
                guard let relationshipID = RAGXMLHelper.attribute(slideID, ["r:id"]),
                      let target = relationshipsByID[relationshipID] else { continue }
                slidePaths.append(resolvePartTarget(target, baseDirectory: "ppt"))
            }
        }

        // 缺失 presentation 关系时回退到归档内全部 slideN.xml。
        if slidePaths.isEmpty {
            slidePaths = RAGArchiveReader.filePaths(archive)
                .filter { $0.range(of: #"^ppt/slides/slide\d+\.xml$"#,
                                   options: .regularExpression) != nil }
                .sorted { slideNumber($0) < slideNumber($1) }
        }
        guard !slidePaths.isEmpty else {
            throw AgentError.parse("PPTX 未找到任何幻灯片")
        }

        var sections: [String] = []
        var outline: [OutlineEntry] = []
        var notesSlides = 0
        for (index, path) in slidePaths.enumerated() {
            guard let data = RAGArchiveReader.data(archive, path: path),
                  let slide = try? RAGXMLHelper.parsePreservingWhitespace(data, source: path) else {
                continue
            }
            let slideRels = relationships(archive, path: relatedPartPath(path))
            let (content, title) = renderSlide(slide.root, useRelationships: slideRels,
                                               archive: archive, slidePath: path,
                                               imageMarkdown: imageMarkdown)
            // 标题页用幻灯片标题做小节标题，便于在大纲与正文里定位。
            let sectionTitle = title.map { "Slide \(index + 1)：\($0)" } ?? "Slide \(index + 1)"
            var section = "## \(sectionTitle)"
            let body = content.trimmingCharacters(in: .whitespacesAndNewlines)
            if !body.isEmpty {
                section += "\n\n" + body
            }
            if let notesTarget = slideRels.first(where: {
                $0.type.lowercased().hasSuffix("notesslide")
            })?.target {
                let notesPath = resolvePartTarget(notesTarget, baseDirectory: "ppt/slides")
                if let notesData = RAGArchiveReader.data(archive, path: notesPath),
                   let notesDocument = try? RAGXMLHelper.parsePreservingWhitespace(
                    notesData, source: notesPath) {
                    let notes = renderNotesSlide(notesDocument.root,
                                                 notesPath: notesPath,
                                                 imageMarkdown: imageMarkdown)
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    if !notes.isEmpty {
                        section += "\n\n**演讲者备注**\n\n" + notes
                        notesSlides += 1
                    }
                }
            }
            sections.append(section)
            outline.append(OutlineEntry(level: 2, title: sectionTitle))
        }

        guard !sections.isEmpty else {
            throw AgentError.parse("PPTX 未读取到任何幻灯片内容")
        }
        var metadata: [String: String] = ["slides": String(sections.count)]
        if notesSlides > 0 { metadata["notes_slides"] = String(notesSlides) }
        if !imageMarkdown.isEmpty { metadata["embedded_images_ocr"] = String(imageMarkdown.count) }
        return (sections.joined(separator: "\n\n"), outline, metadata)
    }

    /// 关系文件：`ppt/slides/slide1.xml` → `ppt/slides/_rels/slide1.xml.rels`。
    private func relatedPartPath(_ path: String) -> String {
        let directory = (path as NSString).deletingLastPathComponent
        let name = (path as NSString).lastPathComponent
        return directory.isEmpty ? "_rels/\(name).rels" : "\(directory)/_rels/\(name).rels"
    }

    private func relationships(_ archive: Archive, path: String) -> [OfficeRelationship] {
        guard let data = RAGArchiveReader.data(archive, path: path),
              let document = try? RAGXMLHelper.parsePreservingWhitespace(data, source: path) else {
            return []
        }
        return RAGXMLHelper.descendants(document.root, "Relationship").compactMap { element in
            guard let id = RAGXMLHelper.attribute(element, ["Id"]),
                  let target = RAGXMLHelper.attribute(element, ["Target"]) else { return nil }
            return OfficeRelationship(id: id,
                                      type: RAGXMLHelper.attribute(element, ["Type"]) ?? "",
                                      target: target)
        }
    }

    /// 按文档顺序收集 sp / graphicFrame（不深入其内部，避免重复取文本）。
    private func collectShapes(_ element: AEXMLElement) -> [AEXMLElement] {
        var result: [AEXMLElement] = []
        func walk(_ node: AEXMLElement) {
            for child in node.children {
                let name = RAGXMLHelper.localName(child.name).lowercased()
                if name == "sp" || name == "graphicframe" || name == "pic" {
                    result.append(child)
                } else {
                    walk(child)
                }
            }
        }
        walk(element)
        return result
    }

    private func renderSlide(_ root: AEXMLElement, useRelationships: [OfficeRelationship],
                             archive: Archive, slidePath: String,
                             imageMarkdown: [String: String]) -> (String, String?) {
        var blocks: [String] = []
        var title: String?

        for shape in collectShapes(root) {
            let name = RAGXMLHelper.localName(shape.name).lowercased()
            if name == "sp" {
                let isTitle = shapeIsTitle(shape)
                for paragraph in RAGXMLHelper.descendants(shape, "p") {
                    let text = pptxParagraphText(paragraph)
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !text.isEmpty else { continue }
                    if isTitle, title == nil { title = text }
                    blocks.append(text)
                }
            } else if name == "graphicframe" {
                var rendered = false
                for table in RAGXMLHelper.descendants(shape, "tbl") {
                    let markdown = pptxTable(table)
                    if !markdown.isEmpty {
                        blocks.append(markdown)
                        rendered = true
                    }
                }
                // SmartArt：图形内容在 ppt/diagrams/dataN.xml 里。
                if !rendered, let diagram = diagramTexts(shape, useRelationships: useRelationships,
                                                        archive: archive), !diagram.isEmpty {
                    blocks.append(diagram.joined(separator: "\n"))
                    rendered = true
                }
                // 图表：图表标题/分类/系列标签是唯一可读文本，由 chartN.xml 提供。
                if !rendered, let chart = chartTexts(shape, useRelationships: useRelationships,
                                                    archive: archive), !chart.isEmpty {
                    blocks.append(chart.joined(separator: "\n"))
                }
            } else if name == "pic" {
                // 优先放回 Ovis 识别的图片正文，没有结果时使用替代文本。
                let refs = RAGXMLHelper.descendants(shape, "blip")
                    + RAGXMLHelper.descendants(shape, "imagedata")
                let markdown = refs
                    .compactMap { RAGXMLHelper.attribute($0, ["r:embed", "embed", "r:link", "link", "r:id", "id"]) }
                    .compactMap { imageMarkdown[slidePath + "#" + $0] }
                    .joined(separator: "\n")
                if !markdown.isEmpty {
                    blocks.append(markdown)
                } else if let alt = pptxAlternativeText(shape), !alt.isEmpty {
                    blocks.append("[\(alt)]")
                }
            }
        }
        return (blocks.joined(separator: "\n\n"), title)
    }

    /// 演讲者备注：跳过节标题/页码/日期等占位符，只取备注正文。
    private func renderNotesSlide(_ root: AEXMLElement, notesPath: String,
                                  imageMarkdown: [String: String]) -> String {
        var paragraphs: [String] = []
        for shape in collectShapes(root) {
            let name = RAGXMLHelper.localName(shape.name).lowercased()
            if name == "pic" {
                let refs = RAGXMLHelper.descendants(shape, "blip")
                    + RAGXMLHelper.descendants(shape, "imagedata")
                let markdown = refs
                    .compactMap { RAGXMLHelper.attribute($0, ["r:embed", "embed", "r:link", "link", "r:id", "id"]) }
                    .compactMap { imageMarkdown[notesPath + "#" + $0] }
                    .joined(separator: "\n")
                if !markdown.isEmpty { paragraphs.append(markdown) }
                continue
            }
            guard name == "sp" else { continue }
            if let type = RAGXMLHelper.descendants(shape, "ph").first
                .flatMap({ RAGXMLHelper.attribute($0, ["type"]) })?.lowercased(),
               Self.pptxNonContentPlaceholders.contains(type) {
                continue
            }
            for paragraph in RAGXMLHelper.descendants(shape, "p") {
                let text = pptxParagraphText(paragraph)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty { paragraphs.append(text) }
            }
        }
        return paragraphs.joined(separator: "\n\n")
    }

    /// 图片替代文本：PPTX 用 `p:cNvPr descr=`，与 DOCX 的 `wp:docPr` 不同。
    private func pptxAlternativeText(_ shape: AEXMLElement) -> String? {
        for node in RAGXMLHelper.descendants(shape, "cNvPr") {
            for key in ["descr", "title", "name"] {
                if let value = RAGXMLHelper.attribute(node, [key])?
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                   !value.isEmpty { return value }
            }
        }
        return nil
    }

    /// 图表文本：从 chartN.xml 取标题（c:tx/c:v）、分类轴标签（c:pt/c:v）
    /// 与系列名（c:ser/c:tx/c:v），保证图表语义不因无正文而丢失。
    private func chartTexts(_ shape: AEXMLElement, useRelationships: [OfficeRelationship],
                            archive: Archive) -> [String]? {
        guard let chartID = RAGXMLHelper.descendants(shape, "chart").first
            .flatMap({ RAGXMLHelper.attribute($0, ["r:id"]) }),
              let relationship = useRelationships.first(where: { $0.id == chartID }) else {
            return nil
        }
        let path = resolvePartTarget(relationship.target, baseDirectory: "ppt/slides")
        guard let data = RAGArchiveReader.data(archive, path: path),
              let document = try? RAGXMLHelper.parsePreservingWhitespace(data, source: path),
              document.root.name.lowercased().contains("chart") else { return nil }
        var lines: [String] = []
        for node in RAGXMLHelper.descendants(document.root, "v") {
            guard node.children.isEmpty,
                  let text = node.value?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !text.isEmpty else { continue }
            if lines.last != text { lines.append(text) }
        }
        return lines.isEmpty ? nil : Array(lines.prefix(64))
    }

    /// SmartArt 文本：`dgm:relIds` 指向 ppt/diagrams/dataN.xml 的数据点文本。
    private func diagramTexts(_ shape: AEXMLElement, useRelationships: [OfficeRelationship],
                              archive: Archive) -> [String]? {
        guard let relIds = RAGXMLHelper.descendants(shape, "relIds").first,
              let dataID = RAGXMLHelper.attribute(relIds, ["r:dm", "dm"]),
              let relationship = useRelationships.first(where: { $0.id == dataID }) else {
            return nil
        }
        let path = resolvePartTarget(relationship.target, baseDirectory: "ppt/slides")
        guard let data = RAGArchiveReader.data(archive, path: path),
              let document = try? RAGXMLHelper.parsePreservingWhitespace(data, source: path) else {
            return nil
        }
        var lines: [String] = []
        for point in RAGXMLHelper.descendants(document.root, "pt") {
            let type = RAGXMLHelper.attribute(point, ["type"])?.lowercased() ?? ""
            if Self.diagramNonTextPointTypes.contains(type) { continue }
            let text = leafTexts(point).trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty { lines.append(text) }
        }
        return lines
    }

    /// 段落文本：`a:br` 转换行，跳过幻灯片编号等域。
    private func pptxParagraphText(_ paragraph: AEXMLElement) -> String {
        var text = ""
        func walk(_ node: AEXMLElement) {
            for child in node.children {
                switch RAGXMLHelper.localName(child.name).lowercased() {
                case "br":
                    text += "\n"
                case "t":
                    text += child.value ?? ""
                case "fld":
                    let type = RAGXMLHelper.attribute(child, ["type"])?.lowercased() ?? ""
                    if type == "slidenum" { continue }
                    walk(child)
                default:
                    walk(child)
                }
            }
        }
        walk(paragraph)
        return text
    }

    /// 取元素下所有叶子 `t` 文本（如 `a:t`），跳过容器节点自身的空白。
    private func leafTexts(_ element: AEXMLElement) -> String {
        RAGXMLHelper.descendants(element, "t")
            .filter { $0.children.isEmpty }
            .compactMap { $0.value }
            .joined()
    }

    private func shapeIsTitle(_ shape: AEXMLElement) -> Bool {
        for placeholder in RAGXMLHelper.descendants(shape, "ph") {
            guard let type = RAGXMLHelper.attribute(placeholder, ["type"]) else { continue }
            if type == "title" || type == "ctrTitle" { return true }
        }
        return false
    }

    private func pptxTable(_ table: AEXMLElement) -> String {
        var rows: [[String]] = []
        for row in RAGXMLHelper.children(table, "tr") {
            let cells = RAGXMLHelper.children(row, "tc")
            rows.append(cells.map { cell in
                // 单元格内换行是内容结构，不能压平成空格后丢进 Markdown 表格。
                let text = RAGXMLHelper.descendants(cell, "p")
                    .map { pptxParagraphText($0).trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                    .joined(separator: "\n")
                return HTMLMarkdown.escapeTableCell(text)
            })
        }
        return HTMLMarkdown.renderMarkdownTable(rows)
    }

    private func slideNumber(_ path: String) -> Int {
        let name = (path as NSString).deletingPathExtension
        // 只取 slide 后的连续数字，避免 slide1-2.xml 被拼成 12。
        guard let range = name.range(of: #"slide(\d+)"#, options: .regularExpression),
              let digits = name[range].range(of: #"\d+"#, options: .regularExpression) else {
            return 0
        }
        return Int(name[digits]) ?? 0
    }

    // MARK: - 路径

    /// 将关系文件里的相对 Target 解析为归档绝对路径。
    private func resolvePartTarget(_ target: String, baseDirectory: String) -> String {
        let decoded = target.removingPercentEncoding ?? target
        if decoded.hasPrefix("/") {
            return RAGArchivePath.normalize(String(decoded.dropFirst()))
        }
        if decoded.hasPrefix(baseDirectory + "/") {
            return RAGArchivePath.normalize(decoded)
        }
        return RAGArchivePath.normalize(baseDirectory + "/" + decoded)
    }

    // MARK: - 类型

    /// 归档内的 OOXML 关系项。
    private struct OfficeRelationship {
        var id = ""
        var type = ""
        var target = ""
    }

    /// DOCX 编号定义：numId → 各级是否有序。
    private struct DocxNumbering {
        struct Level { var ordered = false }
        var byNumId: [String: [Int: Bool]] = [:]

        func level(numId: String, index: Int) -> Level {
            Level(ordered: byNumId[numId]?[index] ?? false)
        }
    }

    /// DOCX 列表计数器：按 (numId, 层级) 递增，回到上层时重置深层计数。
    private final class DocxListCounters {
        private var values: [String: [Int: Int]] = [:]

        func marker(numId: String, level: Int, ordered: Bool) -> String {
            var levels = values[numId] ?? [:]
            for deeper in levels.keys.filter({ $0 > level }) {
                levels[deeper] = nil
            }
            defer { values[numId] = levels }
            guard ordered else { return "- " }
            let next = (levels[level] ?? 0) + 1
            levels[level] = next
            return "\(next). "
        }
    }

    /// 脚注/尾注：按首次引用顺序编号，输出 `[^n]` 标记与末尾列表。
    private final class DocxNotes {
        private let texts: [String: String]
        private var order: [String] = []

        init(texts: [String: String]) { self.texts = texts }

        var count: Int { order.count }

        func marker(for id: String) -> String {
            guard let text = texts[id], !text.isEmpty else { return "" }
            if let index = order.firstIndex(of: id) { return "[^\(index + 1)]" }
            order.append(id)
            return "[^\(order.count)]"
        }

        /// 末尾的注释列表；无引用时不输出。
        func sections(title: String) -> [String] {
            let entries = order.compactMap { texts[$0] }.filter { !$0.isEmpty }
            guard !entries.isEmpty else { return [] }
            var lines = ["## \(title)"]
            for (index, text) in entries.enumerated() {
                lines.append("\(index + 1). \(text)")
            }
            return [lines.joined(separator: "\n")]
        }
    }

    private struct XlsxStyles {
        /// cellXfs 顺序 = 单元格 `s` 属性；值为 numFmtId。
        var cellFormatIDs: [Int] = []
        var customFormats: [Int: String] = [:]

        func isDateStyle(_ styleIndex: Int?) -> Bool {
            guard let styleIndex, cellFormatIDs.indices.contains(styleIndex) else { return false }
            let id = cellFormatIDs[styleIndex]
            if Self.builtinDateFormats.contains(id) { return true }
            if let code = customFormats[id] { return Self.formatIsDate(code) }
            return false
        }

        func isPercentStyle(_ styleIndex: Int?) -> Bool {
            guard let styleIndex, cellFormatIDs.indices.contains(styleIndex) else { return false }
            let id = cellFormatIDs[styleIndex]
            if Self.builtinPercentFormats.contains(id) { return true }
            if let code = customFormats[id] { return Self.formatIsPercent(code) }
            return false
            }

            /// 百分比判定需忽略引号字面量与 `%%` 转义，形如 `yyyy"%"mm` 是日期而非百分比。
            private static func formatIsPercent(_ code: String) -> Bool {
            var cleaned = ""
            var inQuotes = false
            var escaped = false
            var index = code.startIndex
            while index < code.endIndex {
                let character = code[index]
                if escaped { escaped = false; index = code.index(after: index); continue }
                switch character {
                case "\\":
                    escaped = true
                case "\"":
                    inQuotes.toggle()
                case "%":
                    let next = code.index(after: index)
                    if !inQuotes && !(next < code.endIndex && code[next] == "%") {
                        cleaned.append(character)
                    }
                    if next < code.endIndex && code[next] == "%" {
                        index = next
                    }
                default:
                    if !inQuotes { cleaned.append(character) }
                }
                index = code.index(after: index)
            }
            return cleaned.contains("%")
            }

            /// 日期判定：剥掉引号文本、方括号段与转义字符后，含 y/m/d/h/s 即为日期时间。
            private static func formatIsDate(_ code: String) -> Bool {
            var cleaned = ""
            var inQuotes = false
            var inBrackets = false
            var escaped = false
            for character in code {
                if escaped {
                    escaped = false
                    continue
                }
                switch character {
                case "\\": escaped = true
                case "\"": inQuotes.toggle()
                case "[": if !inQuotes { inBrackets = true }
                case "]": if !inQuotes { inBrackets = false }
                default:
                    if !inQuotes && !inBrackets { cleaned.append(character) }
                }
            }
            return cleaned.lowercased().contains { "ymdhs".contains($0) }
        }

        /// ECMA-376 内置日期/时间格式编号（含东亚与时长格式）。
        private static let builtinDateFormats: Set<Int> = Set(
            Array(14...22) + Array(27...36) + Array(45...47) + Array(50...58))
        private static let builtinPercentFormats: Set<Int> = [9, 10]
    }

    // MARK: - 常量

    private static let docxHeadingStyleRegex = try! NSRegularExpression(
        pattern: #"(?i)(?:heading|标题)[\s_-]*([1-6])"#)

    /// PPTX 占位符里不属于正文的类型（页码/页脚/日期/页眉）。
    private static let pptxNonContentPlaceholders: Set<String> = ["sldnum", "ftr", "dt", "hdr"]

    /// SmartArt 数据点里不含可见文本的类型。
    private static let diagramNonTextPointTypes: Set<String> = [
        "doc", "pres", "conn", "partrans", "sibtrans", "sib",
    ]

    /// XLSX 安全上限：防止异常列引用/超长表把内存与 Markdown 撑爆。
    private static let maxXlsxColumns = 256
    private static let maxXlsxRows = 20_000
    private static let maxXlsxSheets = 100
}
