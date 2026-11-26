import Foundation
import CoreGraphics

/// 文档解析选项。PDF 引擎由 `PDFEngine` 决定：
/// - `.vision`：PDFKit 文本层 + 统一归一化 + Apple Vision OCR（默认）
/// - `.ovis`：OvisOCR3 视觉模型逐页转 Markdown（显式指定才启用）
/// - `.hybrid`：Vision 为主，仅表格页交给 OvisOCR3（扫描书籍最快）
public struct ParserOptions: Sendable {
    public var pdfEngine: PDFEngine
    /// 扫描版 EPUB 的 OCR 后端（两条路径都会先做统一归一化）。
    public var epubOCREngine: PDFEngine
    public var ocrPreprocess: Bool
    public var ocrBinarize: Bool
    public var ocrLanguages: [String]
    /// 扫描件的渲染 DPI（Vision 方案）。
    public var renderDPI: Double
    /// 单文件最多解析页数，0 = 不限制。
    public var maxPages: Int
    /// 是否保留页眉页脚（默认清洗掉）。
    public var keepHeadersFooters: Bool

    public init(pdfEngine: PDFEngine = .vision,
                epubOCREngine: PDFEngine = .vision,
                ocrPreprocess: Bool = true,
                ocrBinarize: Bool = true,
                ocrLanguages: [String] = ["zh-Hans", "en-US"],
                renderDPI: Double = 200,
                maxPages: Int = 0,
                keepHeadersFooters: Bool = false) {
        self.pdfEngine = pdfEngine
        self.epubOCREngine = epubOCREngine
        self.ocrPreprocess = ocrPreprocess
        self.ocrBinarize = ocrBinarize
        self.ocrLanguages = ocrLanguages
        self.renderDPI = renderDPI
        self.maxPages = maxPages
        self.keepHeadersFooters = keepHeadersFooters
    }

    public init(config: AgentConfig) {
        self.init(
            pdfEngine: config.ragPDFEngine,
            epubOCREngine: config.ragEPUBOCREngine,
            ocrPreprocess: config.ragOCRPreprocess,
            ocrBinarize: config.ragOCRBinarize
        )
    }
}

public enum ParserProgress: Sendable {
    case page(index: Int, total: Int, stage: String)
    case message(String)
}

/// 所有格式解析器的统一接口。
public protocol DocumentParser: AnyObject {
    var name: String { get }
    /// 小写扩展名，不含点；`["*"]` 表示兜底。
    var supportedExtensions: [String] { get }
    func parse(url: URL, options: ParserOptions,
               progress: ((ParserProgress) -> Void)?) async throws -> ParsedDocument
}

public extension DocumentParser {
    var supportedExtensions: [String] { [] }

    func supports(_ url: URL) -> Bool {
        let ext = url.pathExtension.lowercased()
        return supportedExtensions.contains(ext) || supportedExtensions.contains("*")
    }

    /// 解析后统一清洗（页眉页脚/空白/断行），并标记避免重复清洗。
    /// - Parameter profile: 清洗档位；OCR 文本层用 `.ocr`，现代 Office /
    ///   HTML / EPUB / 文本资料等结构化内容用 `.structured`（保真，不删 URL/编号）。
    func finish(_ document: ParsedDocument, options: ParserOptions,
                profile: DocumentProfile) -> ParsedDocument {
        var doc = document
        doc.text = DocumentCleaner.clean(doc.text,
                                        keepHeadersFooters: options.keepHeadersFooters,
                                        profile: profile)
        doc.metadata["cleaned"] = "1"
        doc.metadata["clean_profile"] = profile == .structured ? "structured" : "ocr"
        return doc
    }
}

/// 视觉语言模型（OvisOCR3）接口。由本地 MLX-VLM 服务实现。
public protocol VLMService: AnyObject, Sendable {
    var identifier: String { get }
    var isAvailable: Bool { get }
    /// 对单页图像执行 OCR/版面转 Markdown。
    func recognize(image: CGImage, prompt: String, maxTokens: Int) async throws -> String
    /// 释放模型权重，provider 本身保留；下次调用会按需重新加载。
    func unload() async
}

public extension VLMService {
    func unload() async {}
}

/// 默认 Ovis 提示词（与原项目一致：整页转 Markdown，保留表格/公式）。
public enum OvisPrompt {
    public static let pageToMarkdown = """
    请将这张文档图片转换为 Markdown。要求：
    1. 保留原文语言与阅读顺序，不要总结或改写；
    2. 标题用 #/##/### 表示，列表保留项目符号；
    3. 表格转换为 Markdown 表格，尽量保持行列结构；
    4. 公式用 LaTeX 行内/块级表示；
    5. 只输出 Markdown 正文，不要添加解释或代码围栏。
    """

    public static let tableToMarkdown = """
    提取图片中的表格并转换为 Markdown 表格，只输出表格本身，不要添加代码围栏或解释。
    """
}

/// 解析器注册表：按扩展名路由，PDF 由配置选择 ovis / vision / hybrid。
public final class DocumentParserRegistry: @unchecked Sendable {
    private var parsers: [DocumentParser]
    private let config: AgentConfig?
    private let vlm: VLMService?

    public init(parsers: [DocumentParser], config: AgentConfig? = nil) {
        self.parsers = parsers
        self.config = config
        self.vlm = parsers.compactMap { ($0 as? VLMBackedParser)?.vlm }.first
    }

    public func register(_ parser: DocumentParser) {
        parsers.append(parser)
    }

    /// 每个文件使用独立解析器实例，供多文件并发 ingest；
    /// PDF/EPUB 的 VLM 服务仍由 LocalModelFactory 共享并串行调用。
    public func freshParser(for url: URL, options: ParserOptions) -> DocumentParser? {
        let ext = url.pathExtension.lowercased()
        if ext == "pdf" {
            if options.pdfEngine == .vision {
                return VisionPDFParser()
            }
            if let config {
                return OvisPDFParser(config: config)
            }
        }
        if ext == "epub", let config {
            return EPUBParser(config: config)
        }
        if ["docx", "xlsx", "pptx"].contains(ext), let config {
            return OfficeParser(config: config)
        }
        if ["doc", "xls", "ppt"].contains(ext), let config {
            return LegacyOfficeParser(config: config)
        }
        return parser(for: url, options: options)
    }

    public func parser(for url: URL, options: ParserOptions) -> DocumentParser? {
        let ext = url.pathExtension.lowercased()
        if ext == "pdf" {
            if options.pdfEngine == .vision,
               let vision = parsers.first(where: { $0 is VisionPDFParser }) {
                return vision
            }
            if let ovis = parsers.first(where: { $0 is OvisPDFParser }) {
                return ovis
            }
        }
        if let exact = parsers.first(where: { $0.supportedExtensions.contains(ext) }) {
            return exact
        }
        return parsers.first(where: { $0.supportedExtensions.contains("*") })
    }

    /// 默认工厂：PDF（ovis + vision）、EPUB、MOBI/AZW3、Office、HTML/文本。
    public static func makeDefault(config: AgentConfig) -> DocumentParserRegistry {
        let registry = DocumentParserRegistry(
            parsers: [
                OvisPDFParser(config: config),
                VisionPDFParser(),
                EPUBParser(config: config),
                MobiParser(config: config),
                OfficeParser(config: config),
                LegacyOfficeParser(config: config),
                HTMLParser(),
                PlainTextParser(),
                ImageParser(),
            ],
            config: config
        )
        return registry
    }
}

/// 需要 VLM 的解析器标记（用于注册表取共享实例）。
public protocol VLMBackedParser: AnyObject {
    var vlm: VLMService? { get }
}
