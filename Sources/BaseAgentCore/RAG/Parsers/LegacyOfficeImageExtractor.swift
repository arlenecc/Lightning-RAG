import Foundation
import CryptoKit

/// 旧版 Office（DOC/XLS/PPT）OLE2 流内嵌图片提取。
///
/// 与 OOXML 的关系锚点不同，Binary Office 的图片字节散落在 OLE2 的
/// `Data` / `Pictures` / `Workbook` / `PowerPoint Document` 流里，且常带有
/// 非图片头部（Word 的 fc/lcb 前缀、Escher Blip 头）。这里用 magic 扫描 +
/// 边界收尾的方式提取真实位图，再交给 Ovis OCR。
enum LegacyOfficeImageExtractor {

    private struct Candidate {
        let label: String
        let data: Data
    }

    /// 各格式的候选流：按与图片内容相关性排序。
    private static func streamNames(for ext: String) -> [String] {
        switch ext {
        case "doc":
            return ["Data", "WordDocument", "1Table", "0Table", "ObjectPool"]
        case "xls":
            return ["Workbook", "Book"]
        case "ppt":
            return ["Pictures", "PowerPoint Document", "Current User"]
        default:
            return []
        }
    }

    /// 从 OLE2 流中扫描位图 magic。限制单张大小防止异常文件耗尽内存。
    static func images(in url: URL, ext: String) -> [Data] {
        guard let compound = try? OLE2CompoundFile.open(url: url) else { return [] }
        var result: [Data] = []
        for name in streamNames(for: ext) {
            guard compound.containsStream(named: name),
                  let stream = try? compound.streamData(named: name) else { continue }
            result.append(contentsOf: scan(stream, label: name))
        }
        return result
    }

    /// 直接对已读取的 Data 扫描（供测试与特殊调用）。
    static func images(in data: Data, label: String = "stream") -> [Data] {
        scan(data, label: label)
    }

    private static let maxImageBytes = 64 * 1024 * 1024

    private static func scan(_ data: Data, label: String) -> [Data] {
        let bytes = [UInt8](data)
        guard bytes.count >= 8 else { return [] }
        var result: [Data] = []
        var cursor = 0
        while cursor < bytes.count - 4 {
            guard let kind = kind(at: bytes, cursor) else { cursor += 1; continue }
            let start = cursor
            if let length = imageLength(kind, bytes, start),
               length > 8, length <= maxImageBytes, start + length <= bytes.count {
                result.append(Data(bytes[start..<(start + length)]))
                cursor = start + length
                continue
            }
            // 没有声明长度（或长度异常）时按 magic 起点向后找结束标记。
            if let end = findEnd(kind, bytes, start), end > start,
               end - start <= maxImageBytes {
                result.append(Data(bytes[start..<end]))
                cursor = end
            } else {
                cursor += 1
            }
        }
        return result
    }

    private enum ImageKind { case jpeg, png, gif, bmp, tiff }

    private static func kind(at bytes: [UInt8], _ i: Int) -> ImageKind? {
        guard i + 4 <= bytes.count else { return nil }
        if bytes[i] == 0xFF, bytes[i+1] == 0xD8, bytes[i+2] == 0xFF { return .jpeg }
        if bytes[i] == 0x89, bytes[i+1] == 0x50, bytes[i+2] == 0x4E, bytes[i+3] == 0x47 {
            return .png
        }
        if bytes[i] == 0x47, bytes[i+1] == 0x49, bytes[i+2] == 0x46 { return .gif }
        if bytes[i] == 0x42, bytes[i+1] == 0x4D { return .bmp }
        if (bytes[i] == 0x49 && bytes[i+1] == 0x49 && bytes[i+2] == 0x2A && bytes[i+3] == 0)
            || (bytes[i] == 0x4D && bytes[i+1] == 0x4D && bytes[i+2] == 0 && bytes[i+3] == 0x2A) {
            return .tiff
        }
        return nil
    }

    private static func imageLength(_ kind: ImageKind, _ bytes: [UInt8], _ start: Int) -> Int? {
        switch kind {
        case .png:
            // IHDR 长度后 4 字节是大端宽度/高度，不能直接得总长；用 IEND 搜索。
            return nil
        case .gif:
            return nil
        case .bmp:
            guard start + 10 <= bytes.count else { return nil }
            let size = Int(bytes[start+2]) | (Int(bytes[start+3]) << 8)
                | (Int(bytes[start+4]) << 16) | (Int(bytes[start+5]) << 24)
            return size > 0 ? size : nil
        case .jpeg, .tiff:
            return nil
        }
    }

    private static func findEnd(_ kind: ImageKind, _ bytes: [UInt8], _ start: Int) -> Int? {
        let limit = min(bytes.count, start + maxImageBytes)
        switch kind {
        case .jpeg:
            var i = start + 2
            while i + 1 < limit {
                if bytes[i] == 0xFF, bytes[i+1] == 0xD9 { return i + 2 }
                i += 1
            }
        case .png:
            let marker: [UInt8] = [0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82]
            return find(bytes, marker, from: start, to: limit)
        case .gif:
            // GIF 的 LZW 数据里可能出现 0x3B，必须按子块结构走：
            // Image Descriptor 后的 LZW minimum code size + 一系列 (len, data) 子块，
            // 子块长度为 0 后再看下一块；0x3B 只在块边界才是 trailer。
            return gifEnd(bytes, start: start, limit: limit)
        case .bmp:
            return nil
        case .tiff:
            // TIFF 无尾标记：到下一个 magic 或流尾为止。
            var i = start + 4
            while i < limit {
                if Self.kind(at: bytes, i) != nil { return i }
                i += 1
            }
            return limit
        }
        return nil
    }

    /// GIF 块结构解析：跳过 header / LSD / 可选扩展块 / 图像描述符，
    /// 沿 LZW 子块链走到 trailer(0x3B)。解析不完整时回退到流尾，宁多勿切。
    private static func gifEnd(_ bytes: [UInt8], start: Int, limit: Int) -> Int? {
        guard start + 13 <= limit else { return nil }
        var i = start + 13                                 // header(6) + LSD(7)
        let packed = bytes[start + 10]                     // LSD packed 位
        if packed & 0x80 != 0 { i += 3 * (1 << ((packed & 0x07) + 1)) } // GCT
        while i < limit {
            let block = bytes[i]
            if block == 0x3B { return i + 1 }               // trailer
            if block == 0x21 {                              // extension
                i += 2
                while i < limit, bytes[i] != 0 {
                    let size = Int(bytes[i]); i += 1 + size
                }
                i += 1
                continue
            }
            if block == 0x2C {                              // image descriptor
                i += 10
                let localPacked = bytes[i - 1]
                if localPacked & 0x80 != 0 { i += 3 * (1 << ((localPacked & 0x07) + 1)) }
                guard i < limit else { return nil }
                i += 1                                      // LZW min code size
                while i < limit, bytes[i] != 0 {
                    let size = Int(bytes[i]); i += 1 + size
                }
                i += 1
                continue
            }
            return nil
        }
        return nil
    }

    private static func find(_ bytes: [UInt8], _ needle: [UInt8],
                             from: Int, to: Int) -> Int? {
        guard let first = needle.first, needle.count > 0, from + needle.count <= to else {
            return nil
        }
        var i = from
        while i + needle.count <= to {
            if bytes[i] == first {
                var matched = true
                for j in 1..<needle.count where bytes[i + j] != needle[j] {
                    matched = false
                    break
                }
                if matched { return i + needle.count }
            }
            i += 1
        }
        return nil
    }

    /// 与 OOXML 一致的 Ovis 调度：按 SHA256 去重、按批并发、单图失败跳过。
    static func recognize(_ images: [Data], vlm: VLMService?,
                          options: ParserOptions,
                          progress: ((ParserProgress) -> Void)?) async throws -> [String] {
        guard !images.isEmpty else { return [] }
        let ovis = OfficeEmbeddedImageOcr.resolveVLM(vlm, options: options)
        guard let ovis else {
            AgentLog.warning("旧版 Office 含嵌入图片但未配置 OCR 视觉模型，图片信息未提取")
            return Array(repeating: "[图片未识别：OCR 模型不可用]", count: images.count)
        }

        var unique: [Data] = []
        var fingerprints = Set<String>()
        for image in images {
            let fingerprint = SHA256.hash(data: image)
                .map { String(format: "%02x", $0) }.joined()
            guard fingerprints.insert(fingerprint).inserted else { continue }
            unique.append(image)
        }

        var markdowns: [String] = []
        let batchSize = PDFPageRenderer.maxConcurrentPages
        var cursor = 0
        while cursor < unique.count {
            if Task.isCancelled { throw AgentError.cancelled }
            let end = min(cursor + batchSize, unique.count)
            let batch = Array(unique[cursor..<end])
            let texts = await withTaskGroup(of: String?.self) { group in
                for image in batch {
                    group.addTask {
                        do {
                            return try await OfficeEmbeddedImageOcr.recognize(
                                key: "legacy-image", data: image, vlm: ovis, options: options)
                        } catch {
                            AgentLog.warning("旧版 Office 图片 OCR 失败："
                                + error.localizedDescription)
                            return nil
                        }
                    }
                }
                var collected: [String?] = []
                for await text in group { collected.append(text) }
                return collected
            }
            markdowns.append(contentsOf: texts.compactMap { $0 })
            progress?(.page(index: end, total: unique.count, stage: "图片OCR"))
            cursor = end
        }
        return markdowns
    }
}
