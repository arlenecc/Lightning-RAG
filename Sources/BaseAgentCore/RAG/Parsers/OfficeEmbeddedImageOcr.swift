import Foundation
import CoreGraphics
import ImageIO
import AEXML

/// Office 嵌入图片 OCR：把 OOXML media 里的图片交给 OvisOCR3 转成 Markdown。
/// 与扫描书不同，Office 插图是混排正文中的局部内容，因此只产出
/// 「rId / media 路径 → Markdown」映射，由各格式在图片锚点处回填。
enum OfficeEmbeddedImageOcr {

    /// 图片解码 + 统一归一化 + Ovis 识别 + Markdown 清洗。
    /// 单张失败只告警并跳过，不拖垮整篇文档；同一图片在调用方按 key 缓存。
    static func recognize(
        key: String,
        data: Data,
        vlm: VLMService,
        options: ParserOptions
    ) async throws -> String? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            AgentLog.warning("Office 嵌入图片无法解码：\(key)")
            return "[图片无法解码]"
        }
        // 超大图先降采样解码：OCR 归一化本来就会把长边压到 1500，
        // 全分辨率解码会让批并发时的内存峰值失控（128MB 压缩图可解出数百 MB 位图）。
        let decodeOptions: [CFString: Any]
        if let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
           let width = props[kCGImagePropertyPixelWidth] as? Int,
           let height = props[kCGImagePropertyPixelHeight] as? Int,
           width > 0, height > 0, width * height > 16_000_000 {
            let scale = sqrt(Double(16_000_000) / Double(width * height))
            decodeOptions = [
                kCGImageSourceShouldCacheImmediately: true,
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceThumbnailMaxPixelSize: max(1, Int(Double(max(width, height)) * scale)),
            ]
        } else {
            decodeOptions = [kCGImageSourceShouldCacheImmediately: true]
        }
        guard let image = CGImageSourceCreateImageAtIndex(
                source, 0, decodeOptions as CFDictionary) else {
            AgentLog.warning("Office 嵌入图片无法解码：\(key)")
            return "[图片无法解码]"
        }
        let prepared = options.ocrPreprocess
            ? (OCRImagePreprocessor.normalize(image, binarize: options.ocrBinarize) ?? image)
            : image
        let raw = try await vlm.recognize(
            image: prepared, prompt: OvisPrompt.pageToMarkdown,
            maxTokens: OvisPDFParser.maxTokens)
        let markdown = OvisPDFParser.cleanVLMOutput(raw)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return markdown.isEmpty ? nil : markdown
    }

    /// 是否应为该文档调用 Ovis：模型不可用时返回 nil 降级，不阻塞文本解析。
    /// 是否算失败由调用方按 metadata 标记（OCR 是图片信息增强，不能拖垮正文）。
    static func resolveVLM(_ vlm: VLMService?, options _: ParserOptions) -> VLMService? {
        vlm
    }

    /// AEXML 节点是否位于背景图容器内（w:background / p:bg / sheet picture 背景）。
    static func isBackground(_ element: AEXMLElement) -> Bool {
        var cursor: AEXMLElement? = element
        while let node = cursor {
            let name = RAGXMLHelper.localName(node.name).lowercased()
            if name == "background" || name == "bg" { return true }
            cursor = node.parent
        }
        return false
    }

    /// 图片 magic 判定：只处理位图，排除 EMF/WMF 等矢量图（Ovis 无法识别）。
    static func isRaster(_ data: Data) -> Bool {
        let bytes = Array(data.prefix(12))
        func has(_ prefix: [UInt8]) -> Bool {
            bytes.count >= prefix.count && Array(bytes[0..<prefix.count]) == prefix
        }
        if has([0xFF, 0xD8, 0xFF]) { return true }                         // JPEG
        if has([0x89, 0x50, 0x4E, 0x47]) { return true }                   // PNG
        if has([0x47, 0x49, 0x46]) { return true }                         // GIF
        if has([0x42, 0x4D]) { return true }                               // BMP
        if has([0x49, 0x49, 0x2A, 0x00])
            || has([0x4D, 0x4D, 0x00, 0x2A]) { return true }               // TIFF
        if bytes.count >= 12 {
            let brand = String(decoding: bytes[4..<12], as: UTF8.self)
            if brand.hasPrefix("ftyphe") { return true }                   // HEIC/HEIF
        }
        if bytes.count >= 12, has([0x52, 0x49, 0x46, 0x46]),
           Array(bytes[8..<12]) == Array("WEBP".utf8) { return true }
        return false
    }
}
