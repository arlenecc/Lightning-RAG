import Foundation

/// 旧版 Office（Word/Excel/PowerPoint 97–2003）统一解析入口。
///
/// 按扩展名分派到三个内置提取器，统一做大小校验、空正文检查和
/// `.structured` 清洗；不依赖 LibreOffice、Python 或任何子进程。
public final class LegacyOfficeParser: DocumentParser, VLMBackedParser {

    public let name = "LegacyOffice"
    public let supportedExtensions = ["doc", "xls", "ppt"]

    private let config: AgentConfig?
    private let providedVLM: VLMService?
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
        let ext = url.pathExtension.lowercased()
        guard supportedExtensions.contains(ext) else {
            throw AgentError.unsupported("不支持的旧版 Office 格式：\(ext)")
        }
        if Task.isCancelled { throw AgentError.cancelled }

        let values = try? url.resourceValues(forKeys: [.fileSizeKey])
        if let size = values?.fileSize, size > LegacyOfficeLimits.maxFileSize {
            throw AgentError.parse(
                "旧版 Office 文档超过 \(LegacyOfficeLimits.maxFileSize / 1024 / 1024)MB 上限"
                + "（\(url.lastPathComponent)）")
        }

        progress?(.message("解析旧版 \(formatName(ext)) 文档…"))

        let result: (text: String, outline: [OutlineEntry], metadata: [String: String])
        do {
            switch ext {
            case "doc":
                let output = try LegacyWordExtractor().extract(url: url)
                result = (output.markdown, output.outline, output.metadata)
            case "xls":
                let output = try LegacyExcelExtractor().extract(url: url)
                result = (output.markdown, output.outline, output.metadata)
            case "ppt":
                let output = try LegacyPowerPointExtractor().extract(url: url)
                result = (output.markdown, output.outline, output.metadata)
            default:
                throw AgentError.unsupported("不支持的旧版 Office 格式：\(ext)")
            }
        } catch is CancellationError {
            throw AgentError.cancelled
        } catch let error as AgentError {
            throw error
        } catch {
            throw AgentError.parse(
                "旧版 Office 解析失败（\(url.lastPathComponent)）：\(error.localizedDescription)")
        }

        if Task.isCancelled { throw AgentError.cancelled }
        guard !result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AgentError.parse("旧版 Office 文档未解析出正文（\(url.lastPathComponent)）")
        }

        var metadata = result.metadata
        metadata["engine"] = ext

        // 旧格式图片信息：从 OLE2 流提取位图并交给 Ovis OCR。
        // 旧版 DOC/XLS/PPT 的图片不以 OOXML rels 组织，无法精确锚定段落，
        // 因此 OCR 正文按页/流顺序附在文档末尾的独立小节，保证信息不丢失。
        let embedded = LegacyOfficeImageExtractor.images(in: url, ext: ext)
        // 没有图片时不访问 lazy vlm，避免无谓的视觉服务解析。
        let imageText = embedded.isEmpty ? [] : try await LegacyOfficeImageExtractor.recognize(
            embedded, vlm: vlm, options: options, progress: progress)
        if !imageText.isEmpty {
            metadata["embedded_images_ocr"] = String(imageText.count)
            metadata["embedded_images_placement"] = "appendix"
        }

        var text = result.text
        if !imageText.isEmpty {
            text += "\n\n## 嵌入图片内容\n\n" + imageText.joined(separator: "\n\n")
        }
        var parsed = ParsedDocument(text: text,
                                    source: url.lastPathComponent,
                                    outline: MarkdownOutline.build(result.outline),
                                    metadata: metadata)
        parsed = finish(parsed, options: options, profile: .structured)
        if Task.isCancelled { throw AgentError.cancelled }
        progress?(.message("旧版 \(formatName(ext)) 文档解析完成"))
        return parsed
    }

    private func formatName(_ ext: String) -> String {
        switch ext {
        case "doc": return "Word"
        case "xls": return "Excel"
        case "ppt": return "PowerPoint"
        default: return "Office"
        }
    }
}
