import Foundation
import ImageIO
import ZIPFoundation
import AEXML
import SwiftSoup

// MARK: - 共享 ZIP / XML / 路径工具（OfficeParser 复用）

/// 归档读取：统一把 ZIPFoundation 的失败转换为 AgentError。
enum RAGArchiveReader {
    /// 单个 ZIP 条目的解压上限：防御压缩比极高的 zip bomb，同时容纳扫描版
    /// EPUB 的整页图 / 整章 XHTML（常规数 MB，极端转制本可达数十 MB）。
    /// 放宽到 128 MiB 是为了避免超限条目被静默跳过、整章正文丢失。
    /// 不能只信目录里的 uncompressedSize（data descriptor 可为 0 或被伪造），
    /// 解压时还要逐块累加校验。OfficeParser 复用同一上限。
    static let maxEntryBytes = 128 * 1024 * 1024

    static func open(_ url: URL) throws -> Archive {
        do {
            return try Archive(url: url, accessMode: .read)
        } catch {
            throw AgentError.parse("无法解包文件 \(url.lastPathComponent)：\(error.localizedDescription)")
        }
    }

    static func data(_ archive: Archive, path: String,
                     maxBytes: Int = RAGArchiveReader.maxEntryBytes) -> Data? {
        guard let entry = entry(archive, path: path) else { return nil }
        guard entry.uncompressedSize <= UInt64(maxBytes) else { return nil }
        var data = Data()
        do {
            // Consumer 是 throws 的：超限立即中断解压，不会把整条炸弹读进内存。
            _ = try archive.extract(entry) { chunk in
                guard data.count + chunk.count <= maxBytes else {
                    throw AgentError.parse("归档条目超过解压上限：\(path)")
                }
                data.append(chunk)
            }
            return data
        } catch {
            return nil
        }
    }

    static func contains(_ archive: Archive, path: String) -> Bool {
        entry(archive, path: path) != nil
    }

    /// 精确查找失败时按「大小写不敏感 + Unicode 预组合」回退。
    /// Windows 转制工具常写出与 ZIP 条目大小写/组合形式不一致的 href，
    /// 旧实现直接返回 nil，整章被静默跳过。
    private static func entry(_ archive: Archive, path: String) -> Entry? {
        let normalized = RAGArchivePath.normalize(path)
        guard !normalized.isEmpty else { return nil }
        if let exact = archive[normalized], exact.type == .file { return exact }
        let target = normalized.precomposedStringWithCanonicalMapping.lowercased()
        for candidate in archive where candidate.type == .file {
            if candidate.path.precomposedStringWithCanonicalMapping.lowercased() == target {
                return candidate
            }
        }
        return nil
    }

    static func filePaths(_ archive: Archive) -> [String] {
        var paths: [String] = []
        for entry in archive where entry.type == .file {
            paths.append(entry.path)
        }
        return paths
    }
}

/// ZIPFoundation 的 `Archive` 句柄不能跨线程并发读；多章 OCR 并发时用它
/// 把归档读取串行化。图片解码与归一化在锁外执行，仍可并行。
final class SerializedArchiveReader: @unchecked Sendable {
    private let archive: Archive
    private let lock = NSLock()

    init(_ archive: Archive) { self.archive = archive }

    func data(path: String) -> Data? {
        lock.lock(); defer { lock.unlock() }
        return RAGArchiveReader.data(archive, path: path)
    }
}

/// AEXML 辅助：按本地名（忽略命名空间前缀）查找元素/属性。
enum RAGXMLHelper {
    static func localName(_ name: String) -> String {
        name.split(separator: ":").last.map(String.init) ?? name
    }

    static func children(_ element: AEXMLElement, _ local: String) -> [AEXMLElement] {
        let target = local.lowercased()
        return element.children.filter { localName($0.name).lowercased() == target }
    }

    static func firstChild(_ element: AEXMLElement, _ local: String) -> AEXMLElement? {
        children(element, local).first
    }

    /// 深度优先，保持文档顺序。
    static func descendants(_ element: AEXMLElement, _ local: String) -> [AEXMLElement] {
        let target = local.lowercased()
        var result: [AEXMLElement] = []
        func walk(_ node: AEXMLElement) {
            for child in node.children {
                if localName(child.name).lowercased() == target {
                    result.append(child)
                }
                walk(child)
            }
        }
        walk(element)
        return result
    }

    static func attribute(_ element: AEXMLElement, _ names: [String]) -> String? {
        // 先精确匹配（如 r:id），再按本地名匹配。
        for name in names {
            if let value = element.attributes[name] { return value }
        }
        for name in names {
            let target = localName(name).lowercased()
            for (key, value) in element.attributes
            where localName(key).lowercased() == target {
                return value
            }
        }
        return nil
    }

    static func parse(_ data: Data, source: String) throws -> AEXMLDocument {
        do {
            return try AEXMLDocument(xml: data)
        } catch {
            throw AgentError.parse("XML 解析失败（\(source)）：\(error.localizedDescription)")
        }
    }

    /// 保留文本节点原始空白地解析：AEXML 默认会 trim 文本值，Office 文档里
    /// `<w:t xml:space="preserve">` 的前导/尾随空格被吃掉会让中英文粘连
    /// （「第 1 章」变成「第1章」）。调用方需自行按 `xml:space` 决定是否 trim。
    static func parsePreservingWhitespace(_ data: Data, source: String) throws -> AEXMLDocument {
        var options = AEXMLOptions()
        options.parserSettings.shouldTrimWhitespace = false
        do {
            return try AEXMLDocument(xml: data, options: options)
        } catch {
            throw AgentError.parse("XML 解析失败（\(source)）：\(error.localizedDescription)")
        }
    }
}

/// EPUB/OPF 内部路径规范化与相对路径解析。
enum RAGArchivePath {
    static func normalize(_ path: String) -> String {
        var components: [String] = []
        // 部分转制工具写 Windows 反斜杠路径，ZIP 内实际是 `/`。
        let unified = path.replacingOccurrences(of: "\\", with: "/")
        for rawComponent in unified.split(separator: "/", omittingEmptySubsequences: false) {
            let component = String(rawComponent)
            if component.isEmpty || component == "." { continue }
            if component == ".." {
                if let last = components.last, last != ".." {
                    components.removeLast()
                } else {
                    components.append("..")
                }
            } else {
                components.append(component)
            }
        }
        return components.joined(separator: "/")
    }

    static func stripFragmentAndQuery(_ href: String) -> String {
        var value = href
        if let hash = value.firstIndex(of: "#") {
            value = String(value[..<hash])
        }
        if let query = value.firstIndex(of: "?") {
            value = String(value[..<query])
        }
        return value
    }

    /// 以 `baseFile` 的目录为基准解析 href，并做 URL 解码与 ../ 归一化。
    static func resolve(_ href: String, relativeTo baseFile: String) -> String {
        let cleaned = stripFragmentAndQuery(href).trimmingCharacters(in: .whitespacesAndNewlines)
        let decoded = cleaned.removingPercentEncoding ?? cleaned
        if decoded.hasPrefix("/") {
            return normalize(String(decoded.dropFirst()))
        }
        let baseDirectory = (baseFile as NSString).deletingLastPathComponent
        if baseDirectory.isEmpty || baseDirectory == "." {
            return normalize(decoded)
        }
        return normalize(baseDirectory + "/" + decoded)
    }
}

// MARK: - EPUB 解析器

/// EPUB 解析：ZipFoundation 解包 + AEXML 解析 container/OPF/NCX + SwiftSoup 清理 XHTML。
public final class EPUBParser: DocumentParser {

    public let name = "EPUB"
    public let supportedExtensions = ["epub"]

    /// 扫描页判据（整书预采样与逐章兜底共用）：「平均每张图分摊的有效文字」
    /// 低于该值 → 图是正文扫描页而非插图。扫描页的文字层每图只有几十字符，
    /// 正常插图周围有成千上万字，远超此线。
    static let scannedCharsPerImage = 150.0

    /// 跨章 OCR 的最大并发章节数。章节再多也只同时压这么多，
    /// 实际识别并发仍由 VLM 实例池与 Vision 闸门限制。
    static let maxConcurrentOCRChapters = 3

    private let providedVLM: VLMService?
    private let config: AgentConfig?
    private lazy var resolvedVLM: VLMService? = {
        guard let config else { return nil }
        // rag_ovis_instances=2 时返回双实例池：实例各自串行、池级并行。
        return LocalModelFactory.shared.makeRAGOCRVLM(config: config)
    }()
    /// OvisOCR3 仅在扫描页且选择 ovis 后端时才解析/加载。
    public var vlm: VLMService? { providedVLM ?? resolvedVLM }

    public init() {
        self.providedVLM = nil
        self.config = nil
    }

    public init(vlm: VLMService?) {
        self.providedVLM = vlm
        self.config = nil
    }

    public init(config: AgentConfig) {
        self.providedVLM = nil
        self.config = config
    }

    private struct ManifestItem {
        let id: String
        let href: String
        let mediaType: String
        let properties: String
        let path: String

        var isXHTML: Bool {
            mediaType.contains("xhtml") || mediaType.contains("html")
                || path.lowercased().hasSuffix(".xhtml")
                || path.lowercased().hasSuffix(".html")
                || path.lowercased().hasSuffix(".htm")
        }
    }

    private struct NavigationEntry {
        let title: String
        let path: String
        let level: Int
    }

    /// 单个 spine 条目的解析结果。OCR 延后到第二阶段执行，
    /// 这里保留文字层、图片列表与预分配的起始页号。
    private struct ChapterPlan {
        let index: Int
        let path: String
        let markdown: String
        let documentHeadings: [OutlineEntry]
        let imagePaths: [String]
        let isScanned: Bool
        let startPage: Int
    }

    private typealias OCROutcome = (text: String, pages: Int, recognized: Int)

    public func parse(url: URL,
                      options: ParserOptions,
                      progress: ((ParserProgress) -> Void)?) async throws -> ParsedDocument {
        let archive = try RAGArchiveReader.open(url)

        guard let containerData = RAGArchiveReader.data(archive, path: "META-INF/container.xml") else {
            throw AgentError.parse("EPUB 缺少 META-INF/container.xml：\(url.lastPathComponent)")
        }
        let container = try RAGXMLHelper.parse(containerData, source: "META-INF/container.xml")
        guard let rootfile = RAGXMLHelper.descendants(container.root, "rootfile").first,
              let declaredOPF = RAGXMLHelper.attribute(rootfile, ["full-path"]),
              !declaredOPF.isEmpty else {
            throw AgentError.parse("EPUB container.xml 未声明 OPF 文件：\(url.lastPathComponent)")
        }

        let opfPath = RAGArchivePath.normalize(declaredOPF.removingPercentEncoding ?? declaredOPF)
        guard let opfData = RAGArchiveReader.data(archive, path: opfPath) else {
            throw AgentError.parse("EPUB 缺少 OPF 文件：\(opfPath)")
        }
        let opf = try RAGXMLHelper.parse(opfData, source: opfPath)

        let title = metadataTitle(opf.root)
        let manifest = parseManifest(opf.root, opfPath: opfPath)
        let spineElement = RAGXMLHelper.firstChild(opf.root, "spine")
        let spine = parseSpinePaths(spineElement, manifest: manifest, archive: archive)
        let spinePaths = spine.paths

        guard !spinePaths.isEmpty else {
            throw AgentError.parse("EPUB 未找到可读的 spine 章节：\(url.lastPathComponent)")
        }
        if spine.recovered > 0 {
            AgentLog.warning(
                "EPUB: spine 只含封面/目录，已从 manifest/归档补齐 \(spine.recovered) 个正文章节")
        }

        let tocID = spineElement.flatMap { RAGXMLHelper.attribute($0, ["toc"]) }
        let (navigation, tocSource) = parseNavigation(archive: archive,
                                                      manifest: manifest,
                                                      tocID: tocID,
                                                      opfPath: opfPath)

        var plans: [ChapterPlan] = []
        var nextPageNumber = 1
        var scannedChapters = 0
        var ocrPages = 0

        // 整书预采样：转制/Kindle 扫描书的文字层常只有目录与点缀（总量可能接近一万），
        // 绝对阈值判不住。采样前 20 个 spine 条目，图片 ≥ 10 张且平均每图
        // 不足 300 字符 → 整书按扫描版处理，含图条目一律走 OCR。
        let scannedBook = Self.looksLikeScannedBook(
            spinePaths: Array(spinePaths.prefix(20)), archive: archive)
        if scannedBook {
            AgentLog.info("EPUB: 预采样判定为扫描版（图片多、平均每图文字不足 300），含图条目将走 OCR")
        }

        // 第一阶段（串行）：读取并解析全部 spine 条目，判定哪些章要走 OCR，
        // 同时按图片数预分配页号。页号只依赖图片数，可以在 OCR 前确定，
        // 因此并发执行 OCR 不会打乱页标记顺序。
        for (index, path) in spinePaths.enumerated() {
            guard let data = RAGArchiveReader.data(archive, path: path), !data.isEmpty else {
                AgentLog.warning("EPUB: spine 条目无法读取，已跳过：\(path)")
                continue
            }
            let xhtml = DocumentTextDecoder.decodeHTML(data)
            guard let document = try? SwiftSoup.parse(xhtml, path) else {
                AgentLog.warning("EPUB: spine 条目无法解析为 HTML，已跳过：\(path)")
                continue
            }

            let (markdown, documentHeadings) = HTMLMarkdown.convert(
                document, removeBoilerplate: false, includeSVGText: true)
            let imagePaths = Self.imagePaths(in: document, chapterPath: path)

            // 纯图片页（扫描版 EPUB）：文字层不足时按图片顺序 OCR，
            // 结果用与 PDF 相同的页标记拼接。
            // 扫描页三层判定（阈值统一为「每图分摊 150 有效文字」）：
            // 层 1 整书：预采样前 20 个 spine 条目，图 ≥10 且平均每图 <150 字
            //   → 转制/Kindle 扫描书（文字层只有目录点缀），所有含图条目一律 OCR；
            // 层 2 逐章：整书未命中时逐章判「文字 < 图数 × 150」——独立扫描章、
            //   常规扫描页（每条目一页一图、文字近乎 0）都在此命中。
            //   单图条目文字 <150 是层 2 的特例（1 × 150），整页插图条目可能被
            //   误 OCR（白识别一次，OCR 空结果时保留原文字层），但保证不漏。
            let effectiveChars = DocumentCleaner.effectiveCharacterCount(markdown)
            let isScannedChapter = !imagePaths.isEmpty
                && (scannedBook
                    || Double(effectiveChars) / Double(imagePaths.count)
                        < Self.scannedCharsPerImage)
            let startPage = nextPageNumber
            if isScannedChapter {
                scannedChapters += 1
                ocrPages += imagePaths.count
                nextPageNumber += imagePaths.count
            }

            plans.append(ChapterPlan(index: index,
                                     path: path,
                                     markdown: markdown,
                                     documentHeadings: documentHeadings,
                                     imagePaths: imagePaths,
                                     isScanned: isScannedChapter,
                                     startPage: startPage))
        }

        // 第二阶段（有界并发）：跨章同时执行 OCR。单章内部仍按批并发，
        // 章节级并发让「每章一页」的扫描书不再逐页串行等待。归档读取
        // 由 SerializedArchiveReader 串行化。
        var ocrResults: [Int: OCROutcome] = [:]
        let scannedPlans = plans.filter(\.isScanned)
        if !scannedPlans.isEmpty {
            let ocrService = try ocrVLMService(options: options)
            let reader = SerializedArchiveReader(archive)
            let limit = min(Self.maxConcurrentOCRChapters, scannedPlans.count)
            try await withThrowingTaskGroup(of: (Int, OCROutcome).self) { group in
                var pending = scannedPlans.makeIterator()
                var running = 0
                while running < limit, let plan = pending.next() {
                    running += 1
                    let imagePaths = plan.imagePaths
                    let index = plan.index
                    let startPage = plan.startPage
                    group.addTask {
                        let result = try await Self.recognizeImages(
                            imagePaths, reader: reader, vlm: ocrService,
                            options: options, startingPage: startPage)
                        return (index, result)
                    }
                }
                while let (index, result) = try await group.next() {
                    ocrResults[index] = result
                    if let plan = pending.next() {
                        let imagePaths = plan.imagePaths
                        let planIndex = plan.index
                        let startPage = plan.startPage
                        group.addTask {
                            let result = try await Self.recognizeImages(
                                imagePaths, reader: reader, vlm: ocrService,
                                options: options, startingPage: startPage)
                            return (planIndex, result)
                        }
                    }
                }
            }
        }

        // 第三阶段：按 spine 原始顺序组装章节与大纲。
        var chapters: [String] = []
        var headings: [OutlineEntry] = []
        for plan in plans {
            var chapter = plan.markdown
            if plan.isScanned, let ocr = ocrResults[plan.index] {
                if !ocr.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    chapter = ocr.text
                } else if !plan.markdown.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    AgentLog.warning("EPUB: 第 \(plan.index + 1) 章 OCR 无结果，保留原文字层")
                }
            }

            guard !chapter.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }

            let tocTitle = navigation.first(where: { $0.path == plan.path })?.title
                ?? navigation.first(where: {
                    ($0.path as NSString).lastPathComponent
                        == (plan.path as NSString).lastPathComponent
                })?.title
            // 扫描页常常是「每个 spine 项一页」；没有真实 TOC/标题时
            // 不要为每页合成「第 N 节」标题，否则大纲会被几百个条目撑爆。
            let explicitTitle = tocTitle ?? plan.documentHeadings.first?.title
            let chapterTitle = explicitTitle ?? "第 \(plan.index + 1) 节"

            // 章节标题转 Markdown 标题；已有标题时保持原结构。
            if MarkdownOutline.scan(chapter).isEmpty,
               !plan.isScanned || explicitTitle != nil {
                chapter = "## \(chapterTitle)\n\n" + chapter
            }
            chapters.append(chapter)

            let scanned = MarkdownOutline.scan(chapter)
            if scanned.isEmpty {
                if !plan.isScanned || explicitTitle != nil {
                    headings.append(OutlineEntry(level: 2, title: chapterTitle))
                }
            } else {
                headings.append(contentsOf: scanned)
            }
        }

        guard !chapters.isEmpty else {
            throw AgentError.parse("EPUB 未解析出正文：\(url.lastPathComponent)")
        }

        // 大纲取条目多的一方：正文 headings 反映实际排版结构（含伪标题识别），
        // 但很多书只有 NCX/nav 目录才完整；反之纯排版书 headings 更细。
        var outline = MarkdownOutline.build(headings)
        if headings.count < navigation.count {
            outline = MarkdownOutline.build(navigation.map {
                OutlineEntry(level: $0.level, title: $0.title)
            })
        }

        var metadata: [String: String] = [
            "engine": "epub",
            "chapters": String(chapters.count),
            "toc": tocSource,
            "opf": opfPath,
        ]
        if scannedChapters > 0 {
            metadata["ocr_engine"] = options.epubOCREngine.rawValue
            metadata["ocr_pages"] = String(ocrPages)
            metadata["scanned_chapters"] = String(scannedChapters)
        }
        if spine.recovered > 0 {
            metadata["recovered_chapters"] = String(spine.recovered)
        }
        if !title.isEmpty { metadata["title"] = title }

        var parsed = ParsedDocument(text: chapters.joined(separator: "\n\n"),
                                    source: url.lastPathComponent,
                                    outline: outline,
                                    metadata: metadata)
        parsed = finish(parsed, options: options, profile: .structured)
        return parsed
    }

    // MARK: - OPF

    private func metadataTitle(_ root: AEXMLElement) -> String {
        let metadata = RAGXMLHelper.firstChild(root, "metadata") ?? root
        for element in RAGXMLHelper.descendants(metadata, "title") {
            let value = element.value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if !value.isEmpty { return value }
        }
        return ""
    }

    private func parseManifest(_ root: AEXMLElement, opfPath: String) -> [ManifestItem] {
        let manifestElement = RAGXMLHelper.firstChild(root, "manifest") ?? root
        return RAGXMLHelper.children(manifestElement, "item").compactMap { item in
            guard let rawID = RAGXMLHelper.attribute(item, ["id"]),
                  let rawHref = RAGXMLHelper.attribute(item, ["href"]) else { return nil }
            let id = rawID.trimmingCharacters(in: .whitespacesAndNewlines)
            let href = rawHref.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !id.isEmpty, !href.isEmpty else { return nil }
            return ManifestItem(
                id: id,
                href: href,
                mediaType: (RAGXMLHelper.attribute(item, ["media-type"]) ?? "").lowercased(),
                properties: (RAGXMLHelper.attribute(item, ["properties"]) ?? "").lowercased(),
                path: RAGArchivePath.resolve(href, relativeTo: opfPath))
        }
    }

    private struct SpineResolution {
        let paths: [String]
        /// 从 manifest / 归档补齐的正文条目数（不含封面与目录页）。
        let recovered: Int
    }

    /// spine 缺失或不完整时回退到 manifest 中的 XHTML 项，再回退到归档内全部
    /// HTML/XHTML。
    ///
    /// 只看 `paths.isEmpty` 不够：相当一部分转制书 spine 里只有封面和目录页，
    /// 正文 XHTML 都在 manifest 里却没被 spine 引用，旧实现会静默丢掉正文，
    /// 最终只剩书名/目录。
    private func parseSpinePaths(_ spine: AEXMLElement?,
                                 manifest: [ManifestItem],
                                 archive: Archive) -> SpineResolution {
        var paths: [String] = []
        var seen = Set<String>()

        func append(_ path: String) {
            guard !path.isEmpty, seen.insert(path).inserted else { return }
            paths.append(path)
        }

        var spineItems: [ManifestItem] = []
        if let spine {
            for itemref in RAGXMLHelper.children(spine, "itemref") {
                guard let rawIDRef = RAGXMLHelper.attribute(itemref, ["idref"]) else { continue }
                let idref = rawIDRef.trimmingCharacters(in: .whitespacesAndNewlines)
                guard let item = manifestItem(matching: idref, in: manifest) else { continue }
                spineItems.append(item)
                append(item.path)
            }
        }

        // spine 是否至少包含一个实际存在的正文章节（排除封面/目录）。
        let hasBody = spineItems.contains {
            !isFrontMatter($0) && RAGArchiveReader.contains(archive, path: $0.path)
        }

        var recovered = 0
        if paths.isEmpty || !hasBody {
            for item in manifest where item.isXHTML && !isNavigation(item) {
                guard !seen.contains(item.path) else { continue }
                append(item.path)
                if !isFrontMatter(item) { recovered += 1 }
            }
        }

        if paths.isEmpty || !hasBody {
            // manifest 也没给出正文 → 扫描归档里的 HTML 文件作最后兜底。
            let htmlExtensions = ["xhtml", "html", "htm"]
            let candidates = RAGArchiveReader.filePaths(archive)
                .filter {
                    htmlExtensions.contains(($0 as NSString).pathExtension.lowercased())
                        && !Self.isNavigationName($0)
                }
                .sorted()
            for path in candidates {
                guard !seen.contains(path) else { continue }
                append(path)
                if !Self.isFrontMatterName(path) { recovered += 1 }
            }
        }

        return SpineResolution(paths: paths, recovered: recovered)
    }

    private func manifestItem(matching idref: String, in manifest: [ManifestItem]) -> ManifestItem? {
        manifest.first { $0.id == idref }
            ?? manifest.first { $0.id.caseInsensitiveCompare(idref) == .orderedSame }
    }

    /// 目录页：`properties="nav"` 的 EPUB3 nav，或 nav/toc/contents 命名的 XHTML。
    private func isNavigation(_ item: ManifestItem) -> Bool {
        if item.properties.split(separator: " ").contains("nav") { return true }
        return Self.isNavigationName(item.path)
    }

    private static func isNavigationName(_ path: String) -> Bool {
        let stem = ((path as NSString).lastPathComponent as NSString)
            .deletingPathExtension.lowercased()
        return ["nav", "toc", "contents"].contains(stem)
    }

    /// 封面页/封面图：id 或文件名以 cover 开头（封面不算补齐的正文）。
    private func isCover(_ item: ManifestItem) -> Bool {
        item.id.lowercased().hasPrefix("cover") || Self.isCoverName(item.path)
    }

    private static func isCoverName(_ path: String) -> Bool {
        ((path as NSString).lastPathComponent as NSString)
            .deletingPathExtension.lowercased().hasPrefix("cover")
    }

    /// 前置页（不算正文）：封面、目录、扉页、版权页等。
    /// 这些页面常被工具写进 spine，若当成正文就会误判 spine 完整而不再补齐后续章节。
    private static func isFrontMatterName(_ path: String) -> Bool {
        if isNavigationName(path) || isCoverName(path) { return true }
        let stem = ((path as NSString).lastPathComponent as NSString)
            .deletingPathExtension.lowercased()
        return ["titlepage", "title-page", "halftitle", "half-title",
                "copyright", "imprint", "colophon", "dedication", "epigraph"].contains(stem)
    }

    private func isFrontMatter(_ item: ManifestItem) -> Bool {
        isCover(item) || Self.isFrontMatterName(item.path)
    }

    // MARK: - 目录（NCX / nav.xhtml）

    private func parseNavigation(archive: Archive,
                                 manifest: [ManifestItem],
                                 tocID: String?,
                                 opfPath: String) -> ([NavigationEntry], String) {
        let ncxItem = manifest.first(where: { $0.id == tocID })
            ?? manifest.first(where: { $0.mediaType == "application/x-dtbncx+xml" })
            ?? manifest.first(where: { $0.path.lowercased().hasSuffix(".ncx") })

        if let ncxItem,
           let data = RAGArchiveReader.data(archive, path: ncxItem.path),
           let document = try? AEXMLDocument(xml: data) {
            let entries = parseNCX(document, ncxPath: ncxItem.path)
            if !entries.isEmpty { return (entries, "ncx") }
        }

        let navItem = manifest.first(where: { $0.properties.split(separator: " ").contains("nav") })
            ?? manifest.first(where: {
                let name = ($0.path as NSString).lastPathComponent.lowercased()
                return name == "nav.xhtml" || name == "toc.xhtml"
            })

        if let navItem,
           let data = RAGArchiveReader.data(archive, path: navItem.path) {
            let entries = parseNav(data, navPath: navItem.path)
            if !entries.isEmpty { return (entries, "nav") }
        }

        return ([], "none")
    }

    private func parseNCX(_ document: AEXMLDocument, ncxPath: String) -> [NavigationEntry] {
        guard let navMap = RAGXMLHelper.descendants(document.root, "navMap").first else { return [] }
        var entries: [NavigationEntry] = []

        func walk(_ parent: AEXMLElement, level: Int) {
            for point in RAGXMLHelper.children(parent, "navPoint") {
                let labelElement = RAGXMLHelper.firstChild(point, "navLabel")
                    .flatMap { RAGXMLHelper.firstChild($0, "text") }
                let title = (labelElement?.value ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let src = RAGXMLHelper.firstChild(point, "content")
                    .flatMap { RAGXMLHelper.attribute($0, ["src"]) }
                if !title.isEmpty, let src {
                    entries.append(NavigationEntry(
                        title: title,
                        path: RAGArchivePath.resolve(src, relativeTo: ncxPath),
                        level: level))
                }
                walk(point, level: level + 1)
            }
        }
        walk(navMap, level: 1)
        return entries
    }

    private func parseNav(_ data: Data, navPath: String) -> [NavigationEntry] {
        let html = DocumentTextDecoder.decodeHTML(data)
        guard let document = try? SwiftSoup.parse(html, navPath) else { return [] }

        let navElements = (try? document.select("nav").array()) ?? []
        let container: Element?
        if let tocNav = navElements.first(where: {
            (try? $0.attr("epub:type")) == "toc" || (try? $0.attr("type")) == "toc"
        }) {
            container = tocNav
        } else if let firstNav = navElements.first {
            container = firstNav
        } else {
            container = document.body()
        }
        guard let container, let list = try? container.select("ol").first() else { return [] }

        var entries: [NavigationEntry] = []
        func walk(_ listElement: Element, level: Int) {
            for item in listElement.children() where item.tagName().lowercased() == "li" {
                if let anchor = item.children().first(where: {
                    $0.tagName().lowercased() == "a"
                }) {
                    let title = ((try? anchor.text()) ?? "")
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    let href = (try? anchor.attr("href")) ?? ""
                    if !title.isEmpty, !href.isEmpty {
                        entries.append(NavigationEntry(
                            title: title,
                            path: RAGArchivePath.resolve(href, relativeTo: navPath),
                            level: level))
                    }
                }
                for child in item.children() where child.tagName().lowercased() == "ol" {
                    walk(child, level: level + 1)
                }
            }
        }
        walk(list, level: 1)
        return entries
    }

    // MARK: - 扫描页 OCR

    /// 预采样前 N 个 spine 条目判定整书是否扫描版：
    /// 图片 ≥ 10 张且「总有效文字 / 图片数」不足 150 字符 → 扫描版
    /// （与逐章判定共用 scannedCharsPerImage 判据）。
    /// 扫描书的文字层（目录/页眉/内置 OCR 层）每图只有几十字符，
    /// 总量可能接近一万；正常图文书每张插图周围有成千上万字，远超此线。
    static func looksLikeScannedBook(spinePaths: [String], archive: Archive) -> Bool {
        guard !spinePaths.isEmpty else { return false }
        var totalImages = 0
        var totalChars = 0
        for path in spinePaths {
            guard let data = RAGArchiveReader.data(archive, path: path), !data.isEmpty,
                  let document = try? SwiftSoup.parse(
                      DocumentTextDecoder.decodeHTML(data), path) else { continue }
            let images = imagePaths(in: document, chapterPath: path).count
            guard images > 0 else { continue }
            totalImages += images
            totalChars += DocumentCleaner.effectiveCharacterCount(
                HTMLMarkdown.convert(document, removeBoilerplate: false,
                                     includeSVGText: true).markdown)
        }
        guard totalImages >= 10 else { return false }
        return Double(totalChars) / Double(totalImages) < 150
    }

    /// 按文档顺序收集 XHTML 中的图片路径。
    ///
    /// 除 `<img src>` / SVG `<image>` 外，还识别转制工具常用的其他引用方式：
    /// `srcset`、`data-src`、`<object data>`、`<video poster>`、内联
    /// `style="background-image:url(...)"`。漏掉这些会让纯图片页被当成空文本
    /// 直接跳过，扫描页 OCR 也就不会触发。
    private static func imagePaths(in document: Document, chapterPath: String) -> [String] {
        var result: [String] = []
        var seen = Set<String>()

        func add(_ raw: String) {
            var value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            // CSS 里可能带引号：url('a.png')
            value = value.trimmingCharacters(in: CharacterSet(charactersIn: "'\""))
            guard !value.isEmpty, !value.lowercased().hasPrefix("data:") else { return }
            let path = RAGArchivePath.resolve(value, relativeTo: chapterPath)
            if !path.isEmpty, seen.insert(path).inserted {
                result.append(path)
            }
        }

        func addSrcset(_ value: String) {
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.lowercased().hasPrefix("data:") else { return }
            // "a.png 1x, b.png 2x" → 逐个 URL（逗号只作分隔符）。
            for candidate in trimmed.split(separator: ",") {
                let url = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
                    .split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true)
                    .first.map(String.init) ?? ""
                add(url)
            }
        }

        func addStyleBackgrounds(_ value: String) {
            var rest = Substring(value)
            while let range = rest.range(of: "url(", options: .caseInsensitive) {
                rest = rest[range.upperBound...]
                guard let end = rest.firstIndex(of: ")") else { break }
                add(String(rest[..<end]))
                rest = rest[rest.index(after: end)...]
            }
        }

        func walk(_ node: Node) {
            if let element = node as? Element {
                let tag = element.tagName().lowercased()
                switch tag {
                case "img":
                    add((try? element.attr("src")) ?? "")
                    add((try? element.attr("data-src")) ?? "")
                    add((try? element.attr("data-original")) ?? "")
                    addSrcset((try? element.attr("srcset")) ?? "")
                case "image":
                    add((try? element.attr("xlink:href")) ?? "")
                    add((try? element.attr("href")) ?? "")
                    add((try? element.attr("src")) ?? "")
                case "object":
                    add((try? element.attr("data")) ?? "")
                case "video":
                    add((try? element.attr("poster")) ?? "")
                default:
                    break
                }
                addStyleBackgrounds((try? element.attr("style")) ?? "")
            }
            for child in node.getChildNodes() { walk(child) }
        }
        walk(document)
        return result
    }

    /// 解析扫描章所需的 OCR 后端；不可用时立即失败。
    /// 在多章并发 OCR 前统一解析一次，避免每个任务重复判断。
    private func ocrVLMService(options: ParserOptions) throws -> VLMService? {
        guard options.epubOCREngine == .ovis || options.epubOCREngine == .custom else {
            return nil
        }
        guard let service = vlm, service.isAvailable else {
            throw AgentError.parse(
                "扫描版 EPUB 需要 OCR 视觉模型，但视觉模型不可用（未配置本地 VLM）")
        }
        return service
    }

    /// 对一章内的图片逐页 OCR；返回带页标记的 Markdown 与页数。
    ///
    /// 两条后端路径都先做统一归一化（缩放/灰度/对比度/锐化/可选二值化）：
    /// - `.vision`：每批最多 8 页并发调用 Apple Vision，全局受闸门限制；
    /// - `.ovis` / `.custom`：借 OCR VLM 实例池并发识别（多实例 2 路，单实例池内仍串行）。
    ///
    /// 多章并发时归档句柄由 `SerializedArchiveReader` 串行化，
    /// 图片解码与归一化在锁外并行。
    private static func recognizeImages(_ paths: [String],
                                        reader: SerializedArchiveReader,
                                        vlm: VLMService?,
                                        options: ParserOptions,
                                        startingPage: Int) async throws -> OCROutcome {
        guard !paths.isEmpty else { return ("", 0, 0) }

        let ovis: VLMService?
        if options.epubOCREngine == .ovis || options.epubOCREngine == .custom {
            ovis = vlm
        } else {
            ovis = nil
        }

        let batchSize = PDFPageRenderer.maxConcurrentPages
        var sections: [String] = []
        var recognized = 0
        var page = startingPage
        var cursor = 0

        while cursor < paths.count {
            if Task.isCancelled { throw AgentError.cancelled }
            let end = min(cursor + batchSize, paths.count)
            let batch = Array(paths[cursor..<end])

            // 图片读取串行（同一个 ZIP 句柄），解码在锁外执行。
            var images: [PDFPageImage] = []
            images.reserveCapacity(batch.count)
            for (offset, relativePath) in batch.enumerated() {
                guard let data = reader.data(path: relativePath),
                      let image = Self.decodeImage(data) else {
                    AgentLog.warning("EPUB: 无法解码图片 \(relativePath)，已跳过")
                    continue
                }
                images.append(PDFPageImage(pageIndex: page + offset, image: image))
            }

            if !images.isEmpty {
                let texts: [(Int, String)]
                if let ovis {
                    // 归一化在批内串行做（CIImage 处理占 CPU），
                    // 识别借实例池并发；结果收集后汇总，避免竞态。
                    let normalized: [(Int, CGImage)] = images.map {
                        ($0.pageIndex, Self.normalizedImage($0.image, options: options))
                    }
                    texts = try await withThrowingTaskGroup(
                        of: (Int, String).self) { group in
                        for (pageIndex, source) in normalized {
                            group.addTask {
                                do {
                                    try Task.checkCancellation()
                                    let raw = try await ovis.recognize(
                                        image: source,
                                        prompt: OvisPrompt.pageToMarkdown,
                                        maxTokens: OvisPDFParser.maxTokens)
                                    return (pageIndex, OvisPDFParser.cleanVLMOutput(raw))
                                } catch is CancellationError {
                                    throw CancellationError()
                                } catch {
                                    // 单页失败跳过，不拖垮整批。
                                    AgentLog.warning(
                                        "EPUB: 第 \(pageIndex) 页 Ovis OCR 失败，已跳过: "
                                        + error.localizedDescription)
                                    return (pageIndex, "")
                                }
                            }
                        }
                        var results: [(Int, String)] = []
                        results.reserveCapacity(normalized.count)
                        for try await item in group {
                            results.append(item)
                        }
                        return results
                    }
                } else {
                    texts = try await Self.visionBatch(images, options: options)
                }

                for (pageNumber, text) in texts.sorted(by: { $0.0 < $1.0 }) {
                    sections.append(DocumentCleaner.pageMarker(pageNumber))
                    let body = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !body.isEmpty {
                        sections.append(body)
                        recognized += 1
                    }
                }
            }

            page += batch.count
            cursor = end
        }

        return (sections.joined(separator: "\n\n"),
                page - startingPage, recognized)
    }

    private static func visionBatch(_ images: [PDFPageImage],
                                    options: ParserOptions) async throws -> [(Int, String)] {
        // 复用 PDF 的 Vision 批处理：统一归一化、全局并发闸门与取消传播。
        try await VisionPDFParser.recognizeBatch(
            images,
            preprocess: options.ocrPreprocess,
            binarize: options.ocrBinarize,
            languages: options.ocrLanguages)
    }

    private static func normalizedImage(_ image: CGImage,
                                        options: ParserOptions) -> CGImage {
        guard options.ocrPreprocess else { return image }
        return OCRImagePreprocessor.normalize(image, binarize: options.ocrBinarize)
            ?? image
    }

    private static func decodeImage(_ data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            return nil
        }
        return CGImageSourceCreateImageAtIndex(
            source, 0,
            [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
    }
}
