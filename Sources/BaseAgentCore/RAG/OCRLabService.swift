import Foundation
import CoreGraphics
import ImageIO

/// OCR 识别测试使用的引擎。
public enum OCRLabEngine: String, CaseIterable, Hashable, Sendable {
    case apple
    case ovis
    case hybrid
    case custom

    public var title: String {
        switch self {
        case .apple: return "Apple OCR"
        case .ovis: return "OvisOCR3"
        case .hybrid: return "混合（Vision+Ovis表格，仅 PDF）"
        case .custom: return "自定义 OCR"
        }
    }

    /// 混合引擎只对 PDF 有意义。
    public var isPDFOnly: Bool { self == .hybrid }
}

/// 单页识别结果（PDF 按页拆分，图片固定一页）。
public struct OCRLabPage: Sendable {
    public let index: Int
    public let markdown: String

    public init(index: Int, markdown: String) {
        self.index = index
        self.markdown = markdown
    }
}

/// 一次识别测试的完整结果。
public struct OCRLabResult: Sendable {
    public let pages: [OCRLabPage]
    public let pageCount: Int
    public let engineLabel: String
    public let metadata: [String: String]

    public init(pages: [OCRLabPage], pageCount: Int,
                engineLabel: String, metadata: [String: String]) {
        self.pages = pages
        self.pageCount = pageCount
        self.engineLabel = engineLabel
        self.metadata = metadata
    }

    /// 保存文件用的完整 Markdown（多页时保留页码标记）。
    public var markdown: String {
        guard pages.count > 1 else { return pages.first?.markdown ?? "" }
        return pages.map { page in
            let body = page.markdown.trimmingCharacters(in: .whitespacesAndNewlines)
            if body.isEmpty {
                return DocumentCleaner.pageMarker(page.index + 1)
            }
            return DocumentCleaner.pageMarker(page.index + 1) + "\n\n" + body
        }.joined(separator: "\n\n")
    }
}

/// 独立的 OCR 识别测试入口：复用正式解析流程，但按页返回结果供左右对照。
public enum OCRLabService {
    public static let imageExtensions: Set<String> = [
        "png", "jpg", "jpeg", "tiff", "tif", "bmp", "heic", "heif", "webp",
    ]

    public static func isImage(_ url: URL) -> Bool {
        imageExtensions.contains(url.pathExtension.lowercased())
    }

    public static func isPDF(_ url: URL) -> Bool {
        url.pathExtension.lowercased() == "pdf"
    }

    public static func run(url: URL,
                           engine: OCRLabEngine,
                           config: AgentConfig,
                           progress: ((ParserProgress) -> Void)? = nil) async throws -> OCRLabResult {
        // 单次 OCR 测试用完即释放 Ovis 权重，下一次测试再按需加载。
        await LocalModelFactory.shared.beginOCRJob()
        defer { Task { await LocalModelFactory.shared.endOCRJob() } }
        if isImage(url) {
            return try await runImage(url: url, engine: engine, config: config, progress: progress)
        }
        if isPDF(url) {
            return try await runPDF(url: url, engine: engine, config: config, progress: progress)
        }
        throw AgentError.parse("仅支持图片或 PDF：\(url.lastPathComponent)")
    }

    // MARK: - 图片

    private static func runImage(url: URL,
                                 engine: OCRLabEngine,
                                 config: AgentConfig,
                                 progress: ((ParserProgress) -> Void)?) async throws -> OCRLabResult {
        guard !engine.isPDFOnly else {
            throw AgentError.parse("混合引擎仅适用于 PDF 文件")
        }
        progress?(.message("读取图片：\(url.lastPathComponent)"))
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw AgentError.parse("无法读取图片：\(url.lastPathComponent)")
        }

        // 与正式入库一致：先做统一归一化（灰度/对比度拉伸/锐化，可选 Otsu）。
        let normalized = OCRImagePreprocessor.normalize(
            image, binarize: config.ragOCRBinarize) ?? image

        let text: String
        switch engine {
        case .apple:
            progress?(.message("Apple Vision OCR…"))
            text = try VisionPDFParser.recognizeText(
                image: normalized, languages: ["zh-Hans", "en-US"])
        case .ovis:
            let vlm = makeOvisOCR3VLM(config: config)
            try ensureAvailable(vlm, label: "OvisOCR3")
            progress?(.message("OvisOCR3 识别…"))
            text = try await vlm.recognize(
                image: normalized, prompt: OvisPrompt.pageToMarkdown, maxTokens: 2048)
        case .custom:
            let path = config.ragOCRModelPath
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !path.isEmpty else {
                throw AgentError.parse("未配置自定义 OCR 模型路径")
            }
            let vlm = LocalModelFactory.shared.makeOCRModel(config: config)
            try ensureAvailable(vlm, label: "自定义 OCR 模型")
            progress?(.message("自定义 OCR 模型识别…"))
            text = try await vlm.recognize(
                image: normalized, prompt: OvisPrompt.pageToMarkdown, maxTokens: 2048)
        case .hybrid:
            throw AgentError.parse("混合引擎仅适用于 PDF 文件")
        }

        let cleaned = (engine == .apple)
            ? text.trimmingCharacters(in: .whitespacesAndNewlines)
            : OvisPDFParser.cleanVLMOutput(text)
        let page = OCRLabPage(index: 0, markdown: cleaned)
        return OCRLabResult(
            pages: [page],
            pageCount: 1,
            engineLabel: engine.title,
            metadata: ["engine": engine.rawValue,
                       "width": String(image.width),
                       "height": String(image.height)])
    }

    // MARK: - PDF

    private static func runPDF(url: URL,
                               engine: OCRLabEngine,
                               config: AgentConfig,
                               progress: ((ParserProgress) -> Void)?) async throws -> OCRLabResult {
        var options = ParserOptions(config: config)
        let parser: DocumentParser

        switch engine {
        case .apple:
            options.pdfEngine = .vision
            parser = VisionPDFParser()
        case .ovis:
            options.pdfEngine = .ovis
            let vlm = makeOvisOCR3VLM(config: config)
            try ensureAvailable(vlm, label: "OvisOCR3")
            parser = OvisPDFParser(vlm: vlm)
        case .hybrid:
            options.pdfEngine = .hybrid
            let vlm = makeOvisOCR3VLM(config: config)
            try ensureAvailable(vlm, label: "OvisOCR3")
            parser = OvisPDFParser(vlm: vlm)
        case .custom:
            let path = config.ragOCRModelPath
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !path.isEmpty else {
                throw AgentError.parse("未配置自定义 OCR 模型路径")
            }
            options.pdfEngine = .ovis
            let vlm = LocalModelFactory.shared.makeOCRModel(config: config)
            try ensureAvailable(vlm, label: "自定义 OCR 模型")
            parser = OvisPDFParser(vlm: vlm)
        }

        progress?(.message("开始解析 PDF：\(url.lastPathComponent)"))
        let document = try await parser.parse(url: url, options: options, progress: progress)
        let pages = splitPages(document.text, pageCount: document.pageCount)
        return OCRLabResult(
            pages: pages,
            pageCount: max(document.pageCount, pages.count),
            engineLabel: engine.title,
            metadata: document.metadata)
    }

    // MARK: - 辅助

    /// OvisOCR3 选项固定使用内置 MLX 模型，不受当前自定义 OCR 配置影响。
    private static func makeOvisOCR3VLM(config: AgentConfig) -> VLMService {
        var cfg = config
        cfg.ragOCRModelPath = ""
        cfg.ragOCRMMProjPath = ""
        cfg.localVLMModel = ""
        cfg.localVLMBackend = "mlx"
        cfg.ragOCRModelName = AgentConfig().ragOCRModelName
        return LocalModelFactory.shared.makeOCRModel(config: cfg)
    }

    private static func ensureAvailable(_ vlm: VLMService, label: String) throws {
        guard vlm.isAvailable else {
            throw AgentError.parse("\(label) 不可用，请检查模型路径与后端配置")
        }
    }

    /// 按 `<!-- page N -->` 标记拆分 PDF 结果；无标记时整份作为一页。
    private static func splitPages(_ text: String, pageCount: Int) -> [OCRLabPage] {
        let pattern = #"<!--\s*page\s+(\d+)\s*-->"#
        guard let regex = try? NSRegularExpression(
            pattern: pattern, options: [.caseInsensitive]) else {
            return [OCRLabPage(index: 0, markdown: text)]
        }
        let ns = text as NSString
        let matches = regex.matches(in: text,
                                    range: NSRange(location: 0, length: ns.length))
        guard !matches.isEmpty else {
            return [OCRLabPage(index: 0, markdown: text)]
        }

        var pages = [String](repeating: "", count: max(pageCount, matches.count))
        for (offset, match) in matches.enumerated() {
            let number = Int(ns.substring(with: match.range(at: 1))) ?? (offset + 1)
            let start = match.range.location + match.range.length
            let end = offset + 1 < matches.count
                ? matches[offset + 1].range.location
                : ns.length
            let body = ns.substring(with: NSRange(location: start,
                                                  length: max(0, end - start)))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard number >= 1, number <= pages.count else { continue }
            pages[number - 1] = body
        }
        return pages.enumerated().map {
            OCRLabPage(index: $0.offset, markdown: $0.element)
        }
    }
}
