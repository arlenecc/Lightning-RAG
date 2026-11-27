import Foundation
import PDFKit
import CoreGraphics

/// 文本层并行提取的单 worker 结果，按页序合并回主流程。
private struct TextLayerOutcome: Sendable {
    /// 文本页最终内容（可能已做表格替换）。
    var texts: [(index: Int, text: String)] = []
    /// 所有成功提取页的原始文本层，用于跨页重复水印判定。
    var rawTexts: [(index: Int, text: String)] = []
    var scanPages: [Int] = []
    var failedIndices: [Int] = []
    var textPages = 0
    var tableRegions = 0
    var scannerPages = 0
    var failedPages = 0
    var textSeconds = 0.0
    var ocrSeconds = 0.0
}

/// PDF 引擎：OvisOCR3 逐页视觉识别。
///
/// - `pdf_engine=ovis`：所有扫描页都交给视觉模型；
/// - `pdf_engine=hybrid`：扫描页先用 Apple Vision OCR，只有检测到表格线框的
///   页面才交给 OvisOCR3，速度接近纯 Vision、同时保留表格结构。
///
/// 原项目行为：
/// - 所有页面（即使有文本层）都栅格化为图片后交给视觉模型；
/// - 渲染最长边限制 1408，页级并发 4，输出按页序拼接并保留页码标记；
/// - 近空白页直接跳过，单页失败不静默降级到其他引擎。
public final class OvisPDFParser: DocumentParser, VLMBackedParser {

    private let providedVLM: VLMService?
    private let config: AgentConfig?
    private lazy var resolvedVLM: VLMService? = {
        guard let config else { return nil }
        // rag_ovis_instances=2 时返回双实例池：实例各自串行、池级并行。
        return LocalModelFactory.shared.makeRAGOCRVLM(config: config)
    }()

    /// 只有真正解析 PDF 时才解析/加载 Ovis 模型，避免纯文本知识库启动即告警。
    public var vlm: VLMService? { providedVLM ?? resolvedVLM }
    public let name = "OvisOCR3"
    public let supportedExtensions = ["pdf"]

    /// 与 Python Ovis 路径一致的渲染上限。
    private static let maxImageSide = 1408
    /// 扫描版 EPUB 复用同一单页输出上限。
    static let maxTokens = 1024
    /// 纯文本 PDF 中裁剪出的表格区域允许输出更多 token。
    static let tableMaxTokens = 2048
    /// 混合引擎下 Vision 结果的有效字符下限；过少视为失败并回退 OvisOCR3。
    static let visionMinCharacters = 12

    public init(vlm: VLMService?) {
        self.providedVLM = vlm
        self.config = nil
    }

    public init(config: AgentConfig) {
        self.providedVLM = nil
        self.config = config
    }

    public func parse(url: URL,
                      options: ParserOptions,
                      progress: ((ParserProgress) -> Void)?) async throws -> ParsedDocument {
        // 与 Vision 路径一致：只取页数/书签，PDFDocument 在进入渲染与
        // VLM 识别前释放，避免大文档的解析缓存与页面位图叠加。
        let (totalPages, bookmarkEntries): (Int, [OutlineEntry]) = try autoreleasepool {
            guard let document = PDFDocument(url: url) else {
                throw AgentError.parse("无法打开 PDF：\(url.lastPathComponent)")
            }
            if document.isLocked, !document.unlock(withPassword: "") {
                throw AgentError.parse("PDF 已加密且无法用空密码解锁：\(url.lastPathComponent)")
            }
            return (document.pageCount, PDFBookmarks.entries(of: document))
        }
        guard totalPages > 0 else {
            throw AgentError.parse("PDF 不包含任何页面：\(url.lastPathComponent)")
        }
        let pageLimit = options.maxPages > 0 ? min(totalPages, options.maxPages) : totalPages

        var pageTexts = [String?](repeating: nil, count: pageLimit)
        var ocrPages = 0
        var visionPages = 0
        var visionAttempts = 0
        var failedPages = 0
        var textPages = 0
        var tableRegions = 0
        var scannerPages = 0
        var missingOCRPages = 0
        var textSeconds = 0.0
        var renderSeconds = 0.0
        var ocrSeconds = 0.0
        var visionSeconds = 0.0
        let hybrid = options.pdfEngine == .hybrid
        var scanPages: [Int] = []

        // 统一流程：先用 PDFKit 文本层。文字版页面直接提取并转 Markdown；
        // 文本层不足的扫描页先收集，再按引擎批量处理。
        // 视觉模型延迟到真正需要 OCR 时才解析/加载，纯文字 PDF 不依赖它。
        //
        // 文本层按页并行：每个 worker 独占一个 PDFDocument（PDFKit 不能跨线程
        // 共享同一文档），文本提取与表格区域渲染都是 CPU 单线程工作，串行时
        // 大 PDF 只有一个核在跑。表格 OCR 仍由 VLM 实例池/MLX 全局串行。
        let tableVLM = self.vlm
        let plan = PDFProcessingPlan.plan(fileURL: url)
        let textWorkerCount = max(1, min(plan.workerCount, pageLimit))
        let textOutcomes = try await withThrowingTaskGroup(of: TextLayerOutcome.self) { group in
            for worker in 0..<textWorkerCount {
                group.addTask {
                    guard let workerDocument = PDFDocument(url: url) else {
                        throw AgentError.parse("无法打开 PDF：\(url.lastPathComponent)")
                    }
                    if workerDocument.isLocked,
                       !workerDocument.unlock(withPassword: "") {
                        throw AgentError.parse(
                            "PDF 已加密且无法用空密码解锁：\(url.lastPathComponent)")
                    }
                    var outcome = TextLayerOutcome()
                    for index in stride(from: worker, to: pageLimit, by: textWorkerCount) {
                        if Task.isCancelled { throw AgentError.cancelled }
                        let pageNumber = index + 1
                        progress?(.page(index: pageNumber, total: pageLimit, stage: "text"))
                        // 每页在全局闸门内创建/释放 PDFPage：多文件并发时
                        // 不会同时有过多页面进入 PDFKit 文本行布局。
                        // 图片多/像素大的页面由提取器改用 CGPDFScanner。
                        let textStarted = Date()
                        let extraction = try await PDFTextExtractor.extract(
                            for: workerDocument, pageIndex: index)
                        guard let extraction else {
                            outcome.texts.append((index, ""))
                            outcome.rawTexts.append((index, ""))
                            outcome.failedIndices.append(index)
                            outcome.failedPages += 1
                            continue
                        }
                        let rawText = extraction.text
                        if extraction.usedScanner { outcome.scannerPages += 1 }
                        let text = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
                        outcome.textSeconds += Date().timeIntervalSince(textStarted)
                        // 原始文本留给合并阶段做跨页判定：只有水印/页眉页脚
                        // 的文本层原始字数可能过阈值，清洗后却为空。
                        outcome.rawTexts.append((index, text))
                        if DocumentCleaner.effectiveCharacterCount(text)
                            >= PDFPageRenderer.textLayerMinCharacters {
                            // 文字版页面仍先直接提取；如果检测到表格区域，只把表格裁剪
                            // 出来交给 OvisOCR3 转 Markdown，再拼回原位置，其余文本不动。
                            var pageText = text
                            let tableStarted = Date()
                            // 扫描器文本的行号/偏移与 PDFKit 的 page.string 不同，
                            // `page.selection(for:)` 无法映射，跳过表格替换。
                            if !extraction.usedScanner,
                               let replaced = await Self.replacingTables(
                                document: workerDocument, pageIndex: index,
                                text: rawText, pageNumber: pageNumber,
                                options: options, vlm: tableVLM) {
                                pageText = replaced.text
                                    .trimmingCharacters(in: .whitespacesAndNewlines)
                                outcome.tableRegions += replaced.count
                            }
                            outcome.ocrSeconds += Date().timeIntervalSince(tableStarted)
                            outcome.texts.append((index, pageText))
                            outcome.textPages += 1
                            continue
                        }
                        outcome.scanPages.append(index)
                    }
                    return outcome
                }
            }
            var all: [TextLayerOutcome] = []
            for try await outcome in group {
                all.append(outcome)
            }
            return all
        }

        var failedPageIndices: Set<Int> = []
        var rawTexts = [String?](repeating: nil, count: pageLimit)
        var provisionalTexts: [(index: Int, text: String)] = []
        for outcome in textOutcomes {
            textSeconds += outcome.textSeconds
            ocrSeconds += outcome.ocrSeconds
            textPages += outcome.textPages
            tableRegions += outcome.tableRegions
            scannerPages += outcome.scannerPages
            failedPages += outcome.failedPages
            failedPageIndices.formUnion(outcome.failedIndices)
            for (index, text) in outcome.rawTexts { rawTexts[index] = text }
            provisionalTexts.append(contentsOf: outcome.texts)
            scanPages.append(contentsOf: outcome.scanPages)
        }

        // 文本页若只有跨页重复的水印/页眉页脚，清洗后会被整段删除。
        // 这类页面必须改走 OCR，否则整份文档会被误判为文字版并清洗为空。
        let marginKeys = DocumentCleaner.repeatedMarginKeys(
            in: rawTexts.map { $0 ?? "" })
        for (index, pageText) in provisionalTexts {
            guard !failedPageIndices.contains(index) else {
                pageTexts[index] = ""
                continue
            }
            let raw = rawTexts[index] ?? ""
            if DocumentCleaner.effectiveOCRTextCharacterCount(
                raw, excludingMarginKeys: marginKeys)
                >= PDFPageRenderer.textLayerMinCharacters {
                pageTexts[index] = pageText
            } else {
                textPages -= 1
                scanPages.append(index)
            }
        }
        scanPages.sort()
        if scannerPages > 0 {
            AgentLog.info("PDF 文本层：\(scannerPages) 页图片过多，改用有界扫描器提取")
        }

        var firstOCRMessage = true
        if hybrid {
            // 混合引擎：多 worker 并行产图 + Vision OCR 流水线重叠；
            // 表格页或 Vision 文本过少的页面再借实例池并发交给 OvisOCR3。
            // 提前访问一次触发 VLM lazy 解析，避免并发任务竞争 lazy 初始化。
            let vlmService = self.vlm
            let vlmAvailable = vlmService?.isAvailable ?? false
            var firstVisionMessage = true
            let outcomes = try await PDFPageScanRunner.run(
                url: url,
                pages: scanPages,
                pageTotal: pageLimit,
                targetLongSide: Self.maxImageSide,
                dpi: 200,
                progress: progress) { pageIndex, image in
                    if vlmAvailable, PDFPageRenderer.looksLikeTable(image) {
                        return (text: nil, needsVLM: true, busySeconds: 0)
                    }
                    if firstVisionMessage {
                        firstVisionMessage = false
                        progress?(.message("Vision OCR 首次调用需初始化系统模型，之后并发处理"))
                    }
                    progress?(.page(index: pageIndex + 1, total: pageLimit, stage: "ocr"))
                    let result = try await VisionPDFParser.recognizeSingle(
                        image: image,
                        preprocess: options.ocrPreprocess,
                        binarize: options.ocrBinarize,
                        languages: options.ocrLanguages)
                    let cleaned = (result.text ?? "")
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    if DocumentCleaner.effectiveCharacterCount(cleaned)
                        >= Self.visionMinCharacters {
                        return (text: cleaned, needsVLM: false,
                                busySeconds: result.busySeconds)
                    }
                    // Vision 文本过少：仍需 Ovis 兜底，但保留已提取的少量文本。
                    // 若 VLM 不可用，至少不会把这一页变成空白。
                    return (text: cleaned, needsVLM: true, busySeconds: result.busySeconds)
                }

            var ovisFallback: [Int] = []
            var needsVLMCount = 0
            for outcome in outcomes {
                renderSeconds += outcome.renderSeconds
                visionSeconds += outcome.ocrSeconds
                if outcome.isBlank {
                    if outcome.renderFailed {
                        failedPages += 1
                        AgentLog.warning(
                            "PDF 第 \(outcome.index + 1) 页渲染失败，已跳过")
                    }
                    pageTexts[outcome.index] = ""
                } else if let text = outcome.text {
                    pageTexts[outcome.index] = text
                    visionPages += 1
                } else if outcome.needsVLM {
                    needsVLMCount += 1
                    // 先写入 Vision 抓到的少量文本；Ovis 成功时会覆盖。
                    pageTexts[outcome.index] = outcome.text ?? ""
                    ovisFallback.append(outcome.index)
                } else {
                    pageTexts[outcome.index] = ""
                }
            }
            visionAttempts += visionPages + needsVLMCount

            // 表格页 + Vision 文本过少的页面按页序回退 OvisOCR3：重新产图
            //（直提取/渲染都很便宜）+ 借实例池并发识别。不缓存整批图像——
            // 需要兜底的页面可能占全书大半，全部驻留会撑爆内存。
            let ovisPages = ovisFallback.sorted()
            if !ovisPages.isEmpty, !(vlmService?.isAvailable ?? false) {
                // VLM 不可用：保留 Vision 兜底文本并显式计数/告警，
                // 让 manifest 知道这些页识别质量不足、后续可重试。
                failedPages += ovisPages.count
                AgentLog.warning("PDF hybrid: \(ovisPages.count) 页需要 OvisOCR3 但视觉模型不可用，"
                    + "已保留 Vision 部分识别结果")
            }
            if let vlm = vlmService, vlm.isAvailable, !ovisPages.isEmpty {
                ocrPages += ovisPages.count
                if firstOCRMessage {
                    firstOCRMessage = false
                    progress?(.message("OvisOCR3 首次识别需加载模型，之后按实例数并发处理"))
                }
                let outcomes2 = try await PDFPageScanRunner.run(
                    url: url,
                    pages: ovisPages,
                    pageTotal: pageLimit,
                    targetLongSide: Self.maxImageSide,
                    dpi: 200,
                    progress: progress) { pageIndex, image in
                        progress?(.page(index: pageIndex + 1, total: pageLimit, stage: "ocr"))
                        let ocrStarted = Date()
                        do {
                            try Task.checkCancellation()
                            let raw = try await vlm.recognize(
                                image: image,
                                prompt: OvisPrompt.pageToMarkdown,
                                maxTokens: Self.maxTokens)
                            return (text: raw, needsVLM: false,
                                    busySeconds: Date().timeIntervalSince(ocrStarted))
                        } catch is CancellationError {
                            throw CancellationError()
                        } catch {
                            // 单页失败跳过，不拖垮整批。
                            return (text: nil, needsVLM: false,
                                    busySeconds: Date().timeIntervalSince(ocrStarted))
                        }
                    }

                for outcome in outcomes2 {
                    renderSeconds += outcome.renderSeconds
                    ocrSeconds += outcome.ocrSeconds
                    if outcome.isBlank {
                        pageTexts[outcome.index] = ""
                    } else if let raw = outcome.text {
                        pageTexts[outcome.index] = Self.cleanVLMOutput(raw)
                    } else {
                        failedPages += 1
                        pageTexts[outcome.index] = ""
                        AgentLog.warning(
                            "OvisOCR3 第 \(outcome.index + 1) 页失败，已跳过")
                    }
                }
            }
        } else {
            // 纯 Ovis：多 worker 并行产图 + OvisOCR3 识别流水线重叠，
            // 借实例池并发（多实例时 2 路并行，单实例时池内仍串行）。
            let vlmService = self.vlm
            let outcomes = try await PDFPageScanRunner.run(
                url: url,
                pages: scanPages,
                pageTotal: pageLimit,
                targetLongSide: Self.maxImageSide,
                dpi: 200,
                progress: progress) { pageIndex, image in
                    guard let vlm = vlmService, vlm.isAvailable else {
                        return (text: "", needsVLM: true, busySeconds: 0)
                    }
                    if firstOCRMessage {
                        firstOCRMessage = false
                        progress?(.message("OvisOCR3 首次识别需加载模型，之后按实例数并发处理"))
                    }
                    progress?(.page(index: pageIndex + 1, total: pageLimit, stage: "ocr"))
                    let ocrStarted = Date()
                    do {
                        try Task.checkCancellation()
                        let raw = try await vlm.recognize(
                            image: image,
                            prompt: OvisPrompt.pageToMarkdown,
                            maxTokens: Self.maxTokens)
                        return (text: raw, needsVLM: false,
                                busySeconds: Date().timeIntervalSince(ocrStarted))
                    } catch is CancellationError {
                        throw CancellationError()
                    } catch {
                        // 单页失败跳过，不拖垮整本书。
                        return (text: nil, needsVLM: false,
                                busySeconds: Date().timeIntervalSince(ocrStarted))
                    }
                }

            for outcome in outcomes {
                renderSeconds += outcome.renderSeconds
                ocrSeconds += outcome.ocrSeconds
                if outcome.isBlank {
                    // 空白页不算失败；产图失败必须计入 failed_pages，
                    // 否则整份文件会被标记 complete，失败页不再重试。
                    if outcome.renderFailed {
                        failedPages += 1
                        AgentLog.warning(
                            "PDF 第 \(outcome.index + 1) 页渲染失败，已跳过")
                    }
                    pageTexts[outcome.index] = ""
                    continue
                }
                if let text = outcome.text {
                    if outcome.needsVLM {
                        // OvisOCR3 不可用。
                        missingOCRPages += 1
                        pageTexts[outcome.index] = ""
                    } else {
                        ocrPages += 1
                        pageTexts[outcome.index] = Self.cleanVLMOutput(text)
                    }
                } else {
                    failedPages += 1
                    pageTexts[outcome.index] = ""
                    AgentLog.warning("OvisOCR3 第 \(outcome.index + 1) 页失败，已跳过")
                }
            }
        }

        let sections = pageTexts.enumerated().flatMap { (index, text) -> [String] in
            let body = (text ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            if body.isEmpty {
                return [DocumentCleaner.pageMarker(index + 1)]
            }
            return [DocumentCleaner.pageMarker(index + 1), body]
        }
        let markdown = sections.joined(separator: "\n\n")
        // 因视觉模型不可用而未能识别的扫描页同样是缺失内容：
        // 计入 failed_pages，配置修复后的下次同步会重试该文件。
        failedPages += missingOCRPages

        guard pageTexts.contains(where: { !($0 ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
            if missingOCRPages > 0 {
                throw AgentError.parse(
                    "PDF 需要 OCR，但 OvisOCR3 不可用或未识别出文本：\(url.lastPathComponent)")
            }
            throw AgentError.parse("未能从 PDF 提取任何文本：\(url.lastPathComponent)")
        }

        var metadata: [String: String] = [
            "engine": hybrid ? "hybrid" : "ovis",
            "text_pages": String(textPages),
            "ocr_pages": String(ocrPages),
            "vision_pages": String(visionPages),
            "failed_pages": String(failedPages),
            "table_regions": String(tableRegions),
            "scanner_pages": String(scannerPages),
            "timing_text": String(format: "%.2f", textSeconds),
            "timing_render": String(format: "%.2f", renderSeconds),
            "timing_ocr": String(format: "%.2f", ocrSeconds),
            "timing_vision": String(format: "%.2f", visionSeconds),
        ]
        // 只有真正做过 OCR（扫描页或表格区域）才解析视觉模型，
        // 纯文本且无表格的 PDF 不加载 Ovis。
        if (ocrPages > 0 || tableRegions > 0), let vlm = self.vlm, vlm.isAvailable {
            metadata["model"] = vlm.identifier
        }
        // 拼接后的页面文本（文字层提取 + OCR）是无标记纯文本：
        // 用启发式还原标题结构转成 Markdown，标题才能进入大纲与切片 heading path。
        // hybrid 表格页的 Markdown 表格、Ovis 引擎输出的标题不受影响。
        var structureEntries: [OutlineEntry] = []
        var finalText = markdown
        if !PlainTextStructure.isMarkdownLike(markdown),
           let restored = PlainTextStructure.restore(markdown) {
            finalText = restored.markdown
            structureEntries = restored.entries
            metadata["structure_restored"] = "true"
        }
        var outlineEntries = bookmarkEntries
        outlineEntries.append(contentsOf: structureEntries)
        var parsed = ParsedDocument(text: finalText,
                                    source: url.lastPathComponent,
                                    outline: MarkdownOutline.build(outlineEntries),
                                    metadata: metadata,
                                    pageCount: pageLimit)
        parsed = finish(parsed, options: options, profile: .ocr)
        return parsed
    }

    // MARK: - 纯文本页中的表格

    /// 页内表格裁剪结果：裁剪图与文本区间。CGImage 在池外由本结构强引用，
    /// 可以安全暴露给跨 await 的 VLM 调用。
    private struct TableCrop {
        let range: NSRange
        let image: CGImage
    }

    /// 在文本页中识别表格行块，只裁剪表格区域交给 OvisOCR3 转 Markdown，
    /// 再把返回内容替换回原位置；页面其它文本保持不变。
    /// 静态方法：文本层并行时每个 worker 都能安全调用，VLM 由调用方统一解析。
    ///
    /// `PDFPage` 只在同步的 autoreleasepool 内使用：选区、整页渲染与裁剪
    /// 全部完成后再 await VLM。否则页面对象会跨异步调用驻留，多文件并发时
    /// 解析缓存无法释放。
    private static func replacingTables(document: PDFDocument, pageIndex: Int,
                                        text: String, pageNumber: Int,
                                        options: ParserOptions,
                                        vlm: VLMService?)
        async -> (text: String, count: Int)? {
        let blocks = Self.tableBlocks(in: text)
        guard !blocks.isEmpty else { return nil }
        guard let vlm, vlm.isAvailable else {
            AgentLog.warning("PDF 第 \(pageNumber) 页检测到表格，但 OvisOCR3 不可用，保留文本层")
            return nil
        }

        let crops: [TableCrop] = autoreleasepool {
            guard let page = document.page(at: pageIndex) else { return [] }
            let rotation = ((page.rotation % 360) + 360) % 360
            guard rotation == 0 else {
                AgentLog.warning(
                    "PDF 第 \(pageNumber) 页旋转 \(page.rotation)°，表格裁剪暂不支持，保留文本层")
                return []
            }
            guard let fullImage = PDFPageRenderer.render(
                page: page, dpi: 300, maxLongSide: 4_096) else {
                AgentLog.warning("PDF 第 \(pageNumber) 页表格区域渲染失败，保留文本层")
                return []
            }

            var crops: [TableCrop] = []
            for range in blocks {
                guard let selection = page.selection(for: range) else { continue }
                var bounds = selection.bounds(for: page)
                guard !bounds.isNull, !bounds.isInfinite,
                      bounds.width > 8, bounds.height > 8 else { continue }
                bounds = bounds.insetBy(dx: -6, dy: -6)
                guard let cropped = PDFPageRenderer.crop(
                    image: fullImage, page: page, rect: bounds) else { continue }
                crops.append(TableCrop(range: range, image: cropped))
            }
            return crops
        }
        guard !crops.isEmpty else { return nil }

        var replacements: [(range: NSRange, markdown: String)] = []
        for crop in crops {
            if Task.isCancelled { return nil }
            let source = options.ocrPreprocess
                ? (OCRImagePreprocessor.normalize(crop.image,
                                                  binarize: options.ocrBinarize) ?? crop.image)
                : crop.image
            do {
                let raw = try await vlm.recognize(image: source,
                                                  prompt: OvisPrompt.tableToMarkdown,
                                                  maxTokens: Self.tableMaxTokens)
                let markdown = Self.cleanVLMOutput(raw)
                // 裁剪区域异常时模型可能输出整片空单元格表格；
                // 没有任何字母/数字的结果视为无效，保留文本层。
                guard markdown.rangeOfCharacter(from: .alphanumerics) != nil else {
                    AgentLog.warning(
                        "PDF 第 \(pageNumber) 页表格 OCR 结果为空，保留文本层")
                    continue
                }
                replacements.append((crop.range, markdown))
            } catch AgentError.cancelled {
                return nil
            } catch {
                AgentLog.warning(
                    "PDF 第 \(pageNumber) 页表格 OCR 失败，保留文本层: "
                    + error.localizedDescription)
            }
        }
        guard !replacements.isEmpty else { return nil }

        let mutable = NSMutableString(string: text)
        for item in replacements.sorted(by: { $0.range.location > $1.range.location }) {
            mutable.replaceCharacters(in: item.range, with: "\n\n\(item.markdown)\n\n")
        }
        return (mutable as String, replacements.count)
    }

    /// 用行内多列特征识别表格块：连续 ≥2 行，或单行含 Markdown 管道符。
    ///
    /// PDFKit 文本层常把表格输出成单空格分隔的列（如 `1 3 5 1`），
    /// 既没有管道符也没有制表符/双空格；这里额外识别「连续 ≥3 行、
    /// 列数一致、所有字段都较短，且至少一半行的纯数字字段占该行字段的一半以上」的块。
    private static func tableBlocks(in text: String) -> [NSRange] {
        var lines: [(range: NSRange, text: String)] = []
        text.enumerateSubstrings(in: text.startIndex..., options: [.byLines]) {
            substring, range, _, _ in
            guard let substring else { return }
            lines.append((NSRange(range, in: text), substring))
        }
        guard !lines.isEmpty else { return [] }

        // 先按列数为单空格行分组；连续 ≥3 行、至少一半行的纯数字字段数 ≥2
        // 才认定为表格。版权页 ISBN、目录页码等虽有数字但数字行占比低，
        // 用「数字行占比」而不是「任意一行含数字」可避免误报。
        var singleSpaceTable = [Bool](repeating: false, count: lines.count)
        var cursor = 0
        while cursor < lines.count {
            guard let fieldCount = singleSpaceFieldCount(lines[cursor].text) else {
                cursor += 1
                continue
            }
            var end = cursor + 1
            while end < lines.count,
                  singleSpaceFieldCount(lines[end].text) == fieldCount {
                end += 1
            }
            let run = cursor..<end
            // 数字行要求纯数字字段至少占一半，避免把 ISBN 行（2 个数字 +
            // 3 个文字字段）当成表格行。
            let numericRows = run.filter { line in
                let fields = lines[line].text.trimmingCharacters(in: .whitespaces)
                    .split(separator: " ", omittingEmptySubsequences: true)
                let numeric = numericFieldCount(lines[line].text)
                return numeric >= 2 && numeric * 2 >= fields.count
            }.count
            if run.count >= 3,
               numericRows >= 2,
               numericRows * 2 >= run.count {
                for index in run { singleSpaceTable[index] = true }
            }
            cursor = end
        }

        var blocks: [NSRange] = []
        var start: Int?
        var hasPipe = false

        func flush(endIndex: Int) {
            guard let first = start else { return }
            let count = endIndex - first
            if count >= 2 || (count >= 1 && hasPipe) {
                let end = NSMaxRange(lines[endIndex - 1].range)
                blocks.append(NSRange(location: lines[first].range.location,
                                      length: end - lines[first].range.location))
            }
            start = nil
            hasPipe = false
        }

        for (index, line) in lines.enumerated() {
            if isTableLine(line.text) || singleSpaceTable[index] {
                if start == nil { start = index }
                if line.text.contains("|") { hasPipe = true }
            } else if start != nil {
                flush(endIndex: index)
            }
        }
        if start != nil { flush(endIndex: lines.count) }
        return blocks
    }

    private static func isTableLine(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.count >= 6 else { return false }
        if trimmed.filter({ $0 == "|" }).count >= 2 { return true }

        let tabFields = trimmed.components(separatedBy: "\t")
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        if tabFields.count >= 3 { return true }

        let spaceRuns = trimmed.components(separatedBy: "  ")
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        return spaceRuns.count >= 3
    }

    /// 单空格分隔的多列行（PDFKit 文本层常见）：返回列数，普通正文返回 nil。
    private static func singleSpaceFieldCount(_ line: String) -> Int? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.count >= 6 else { return nil }
        // 带显式表格特征的行交给 isTableLine 判定。
        if trimmed.contains("|") || trimmed.contains("\t")
            || trimmed.contains("  ") { return nil }
        // 以句末标点结束的行是正文，不是表格行。
        if let last = trimmed.last,
           "。．.!?！？；;，,、…".contains(last) { return nil }
        let fields = trimmed.split(separator: " ", omittingEmptySubsequences: true)
        guard (3...16).contains(fields.count) else { return nil }
        guard fields.allSatisfy({ $0.count <= 24 }) else { return nil }
        return fields.count
    }

    /// 行内纯数字字段数量（允许小数点、千分位、百分号、区间符号）。
    private static func numericFieldCount(_ line: String) -> Int {
        let fields = line.trimmingCharacters(in: .whitespaces)
            .split(separator: " ", omittingEmptySubsequences: true)
        let pattern = #"^[0-9][0-9.,%％/()+\-－~～∼]*$"#
        return fields.filter {
            String($0).range(of: pattern, options: .regularExpression) != nil
        }.count
    }

    // MARK: - 输出清理（对应 Python `_clean_vlm_output`）

    /// 供 PDF 与扫描版 EPUB 共用的 VLM 输出清理。
    static func cleanVLMOutput(_ text: String) -> String {
        var result = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !result.isEmpty else { return "" }

        result = stripOuterCodeFence(result)
        result = convertHTMLTables(result)
        result = replace(htmlContainerTagRegex, in: result, with: " ")
        result = result.replacingOccurrences(of: "\n{3,}", with: "\n\n", options: .regularExpression)
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// 模型常把整段结果再包一层 ```markdown / ```html 代码围栏。
    /// 仅当首个与最后一个非空行都是围栏时剥离，避免误伤正文中的独立代码块。
    private static func stripOuterCodeFence(_ text: String) -> String {
        let lines = text.components(separatedBy: "\n")
        guard let first = lines.firstIndex(where: {
                  !$0.trimmingCharacters(in: .whitespaces).isEmpty
              }),
              let last = lines.lastIndex(where: {
                  !$0.trimmingCharacters(in: .whitespaces).isEmpty
              }),
              first < last else { return text }

        func isFence(_ line: String) -> Bool {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            return trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~")
        }
        guard isFence(lines[first]), isFence(lines[last]) else { return text }
        return lines[(first + 1)..<last].joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Ovis 偶尔输出 HTML 表格；简单表格转 Markdown，复杂表格保留 HTML。
    private static func convertHTMLTables(_ text: String) -> String {
        let ranges = DocumentCleaner.tableRanges(in: text)
        guard !ranges.isEmpty else { return text }
        let mutable = NSMutableString(string: text)
        for range in ranges.reversed() {
            guard let swiftRange = Range(range, in: text) else { continue }
            let html = String(text[swiftRange])
            let conversion = DocumentCleaner.convertHTMLTable(fromHTML: html)
            mutable.replaceCharacters(in: range,
                                      with: conversion.markdown ?? html)
        }
        return mutable as String
    }

    private static func replace(_ regex: NSRegularExpression, in text: String, with template: String) -> String {
        regex.stringByReplacingMatches(in: text,
                                       options: [],
                                       range: NSRange(text.startIndex..<text.endIndex, in: text),
                                       withTemplate: template)
    }

    private static let htmlContainerTagRegex = try! NSRegularExpression(
        pattern: #"</?(?:html|body)(?:\s[^>]*)?>"#, options: [.caseInsensitive])
}
