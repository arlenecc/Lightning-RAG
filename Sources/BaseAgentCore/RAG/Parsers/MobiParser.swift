import Foundation
import CoreGraphics
import ImageIO
import CryptoKit
import SwiftSoup

/// 扫描页：只持有原始图片字节，UI 按需解码，翻页即可释放 CGImage。
public struct MobiScannedPage: Sendable {
    public let index: Int
    public let imageData: Data
    public let source: String

    public func cgImage() -> CGImage? {
        guard let source = CGImageSourceCreateWithData(imageData as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(
            source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
    }
}

/// 供 UI 懒加载的扫描书页数据（不持有解码后的 CGImage）。
public struct MobiScanDocument: Sendable {
    public let pages: [MobiScannedPage]
    public let stitchedRanges: [Range<Int>]
}

/// 扫描版判定诊断，写入 ParsedDocument.metadata 以便复核。
public struct MobiScanDiagnostics: Sendable {
    public let isScanned: Bool
    public let reasons: [String]
    public let textLength: Int
    public let imageCount: Int
    public let trailingRecords: Int
    public let readablePerImage: Double
    public let hasTextTag: Bool
    public let pageBreakCount: Int

    /// 依据用户给定的扫描书特征判定：
    /// textLength 只有几 KB 且几乎全是 HTML 标签；图片数量巨大；
    /// firstNonBookIndex 之后还有成百上千条记录；无 <text> 标签；
    /// 源码几乎找不到可读连续句子，只有 <img>/<mbp:pagebreak/>。
    /// 正则只编译一次：批量转换 5000+ 本时，逐本新建 NSRegularExpression
    /// 与全量 lowercased() 复制是持续 CPU 占用的主要来源。
    nonisolated(unsafe) private static let textTagRegex: NSRegularExpression? = {
        try? NSRegularExpression(pattern: #"<text(?=[\s>])"#, options: .caseInsensitive)
    }()

    /// 大小写不敏感地统计子串出现次数，只返回计数不构造子串。
    private static func countOccurrences(of needle: String, in text: String) -> Int {
        guard !needle.isEmpty else { return 0 }
        let ns = text as NSString
        var cursor = 0
        var count = 0
        let total = ns.length
        while cursor < total {
            let found = ns.range(of: needle, options: .caseInsensitive,
                                 range: NSRange(location: cursor, length: total - cursor))
            if found.location == NSNotFound { break }
            count += 1
            cursor = found.location + max(1, found.length)
        }
        return count
    }

    public static func detect(textLength: Int, imageCount: Int, trailingRecords: Int,
                              html: String) -> MobiScanDiagnostics {
        let ns = html as NSString
        let range = NSRange(location: 0, length: ns.length)
        // 精确匹配 <text ...>/<text>，避免 <textarea>/<textpath> 误命中；
        // 直接用 caseInsensitive 在原文上匹配，不再全量 lowercased() 复制。
        let hasTextTag = Self.textTagRegex?
            .firstMatch(in: html, options: [], range: range) != nil
        // 单次线性扫描同时统计分页符、<img> 与可读字符，避免多次全量切分/正则替换。
        // 用缓存的 caseInsensitive 正则在原文上统计，避免 html.lowercased() 的
        // 全量字符串复制（这是批量场景下最昂贵的分配之一）。
        let pageBreakCount = Self.countOccurrences(of: "mbp:pagebreak", in: html)
        let imgCount = Self.countOccurrences(of: "<img", in: html)
        // 去标签与实体：一次正则替换，不再全量 lowercased 后切分。
        let stripped = html
            .replacingOccurrences(of: #"<[^>]+>"#, with: " ",
                                  options: [.regularExpression, .caseInsensitive])
            .replacingOccurrences(of: "&nbsp;", with: " ")
        let readable = stripped.unicodeScalars.filter {
            !CharacterSet.whitespacesAndNewlines.contains($0)
        }.count
        let perImage = imageCount > 0 ? Double(readable) / Double(imageCount) : 0

        var reasons: [String] = []
        let shortText = textLength > 0 && textLength <= 32 * 1024
        let manyImages = imageCount >= 16
        let manyTrailing = trailingRecords >= 50
        if shortText { reasons.append("textLength=\(textLength)B") }
        if manyImages { reasons.append("图片 \(imageCount) 张") }
        if manyTrailing { reasons.append("尾记录 \(trailingRecords) 条") }
        if !hasTextTag { reasons.append("无 <text> 标签") }
        let sparseText = perImage < 150
        let manyBreaks = pageBreakCount >= 10
        let imageOnly = imgCount >= 10 || manyBreaks
        if sparseText { reasons.append(String(format: "每图可读字符 %.1f", perImage)) }
        if imageOnly { reasons.append("img=\(imgCount)/pagebreak=\(pageBreakCount)") }

        // 扫描书的核心特征是「图片 + 强制分页」而非单纯图片多：
        // 图文混排书插图再多也有正文，不能因 perImage 低就丢弃正文。
        let core = shortText && manyImages && manyTrailing && !hasTextTag && sparseText
            && readable <= 512
        let epubStyleFallback = imageCount >= 10 && perImage < 150 && !hasTextTag
            && manyBreaks && readable <= 512
        if epubStyleFallback { reasons.append("EPUB 式每图文本密度兜底") }
        let scanned = core || epubStyleFallback
        return MobiScanDiagnostics(isScanned: scanned, reasons: reasons,
                                   textLength: textLength, imageCount: imageCount,
                                   trailingRecords: trailingRecords,
                                   readablePerImage: perImage, hasTextTag: hasTextTag,
                                   pageBreakCount: pageBreakCount)
    }
}

/// MOBI / AZW3（KF8）解析器：PDB 容器 → Header/EXTH → 文本解压 →
/// 普通书走 HTML 文本化；扫描书走图片提取 + 拼图 + OCR（与 EPUB 相同后端）。
public final class MobiParser: DocumentParser, VLMBackedParser {
    public let name = "MOBI/AZW3"
    public let supportedExtensions = ["mobi", "azw3", "prc"]
    public let vlm: VLMService?

    private let config: AgentConfig?

    private let drmPID: String?

    public init(config: AgentConfig? = nil, drmPID: String? = nil) {
        self.config = config
        // 显式 PID 优先；否则从配置读取，使 RAG 同步/转换工具无需特殊构造即可解 DRM V2。
        self.drmPID = drmPID ?? config.flatMap {
            $0.ragMobiDRMPID.isEmpty ? nil : $0.ragMobiDRMPID
        }
        self.vlm = config.map { LocalModelFactory.shared.makeRAGOCRVLM(config: $0) }
    }

    // MARK: - DocumentParser

    public func parse(url: URL, options: ParserOptions,
                      progress: ((ParserProgress) -> Void)?) async throws -> ParsedDocument {
        progress?(.message("MOBI: 读取 PDB 容器"))
        let data = try Data(contentsOf: url)
        let pdb = try MobiPDB(data: data)
        guard let zero = pdb.record(at: 0) else {
            throw AgentError.parse("MOBI: 缺少 Record 0")
        }
        let palm = try MobiPalmHeader(zero)
        // 旧式纯 PalmDOC 文件没有 MOBI 头，用 libmobi 同样的兜底处理。
        let header = (try? MobiHeader(recordZero: zero))
            ?? MobiHeader.fallback(palm: palm, recordCount: pdb.recordCount)
        var metadata: [String: String] = [
            "format": url.pathExtension.lowercased(),
            "mobi_type": String(header.mobiType),
            "mobi_version": String(header.version),
            "text_length": String(palm.textLength),
            "first_non_book_index": String(header.firstNonBookIndex),
        ]
        if let exthOffset = header.exthOffset,
           let exth = try? MobiEXTH(zero, start: exthOffset) {
            for (key, value) in exth.metadata() { metadata["exth_\(key)"] = value }
        }
        if !header.fullName.isEmpty { metadata["title"] = header.fullName }

        // 文本 Record 排除 HuffDic 字典 Record 与尾部资源 Record。
        let textSet = Self.textRecordSet(pdb: pdb, header: header, palm: palm)
        let compression = MobiCompressionKind(rawValue: palm.compressionRaw)
            ?? .none
        let effectiveRecords: [Data]
        if MobiDRM.isProtected(palm) {
            let drm = try MobiDRM.decrypt(pdb: pdb, header: header, palm: palm,
                                          recordIndexes: textSet.indexes, pid: drmPID)
            effectiveRecords = drm.records
            metadata["drm"] = drm.method
        } else {
            effectiveRecords = textSet.records
        }
        let rawText = try MobiCompression.decompress(
            records: effectiveRecords, kind: compression,
            expectedLength: palm.textLength, huffmanRecords: textSet.dictionary)
        let encoding: String.Encoding = header.textEncoding == 65001 ? .utf8 : .windowsCP1252
        let kf8Offset = header.isKF8 ? Int(header.kf8Boundary ?? 0) : 0
        let flows = MobiFlowParser.flows(pdb: pdb, header: header,
                                         rawml: rawText, kf8Offset: kf8Offset)
        let htmlData = flows.first { $0.mime.contains("html") }?.data ?? rawText
        let html = String(data: htmlData, encoding: encoding)
            ?? String(decoding: htmlData, as: UTF8.self)
        let toc = MobiIndexParser.toc(pdb: pdb, header: header, kf8Offset: kf8Offset)
        let dictMarks = MobiIndexParser.dictionaryMarks(pdb: pdb, header: header,
                                                        kf8Offset: kf8Offset)
        metadata["flow_count"] = String(flows.count)
        metadata["toc_count"] = String(toc.count)
        if !dictMarks.isEmpty {
            metadata["dictionary_mark_count"] = String(dictMarks.count)
            metadata["dictionary_marks"] = dictMarks.prefix(20).joined(separator: " | ")
        }

        let imageData = Self.extractImages(pdb: pdb, header: header, html: html)
        let trailing = max(0, pdb.recordCount - Int(header.firstNonBookIndex))
        let diagnostics = MobiScanDiagnostics.detect(
            textLength: palm.textLength, imageCount: imageData.count,
            trailingRecords: trailing, html: html)
        metadata["scanned"] = diagnostics.isScanned ? "1" : "0"
        metadata["diagnostic_reasons"] = diagnostics.reasons.joined(separator: "; ")

        if diagnostics.isScanned {
            // 与 EPUB ocrVLMService 一致：选择 VLM 但模型不可用时显式报错，
            // 不静默降级到 Vision 也不产出空文档。
            let useVLM = options.epubOCREngine == .ovis || options.epubOCREngine == .custom
            if useVLM && !(vlm?.isAvailable ?? false) {
                throw AgentError.parse(
                    "扫描版 MOBI/AZW3 需要 OCR 视觉模型，但视觉模型不可用（未配置本地 VLM）")
            }
            progress?(.message("MOBI: 判定为扫描版，开始图片 OCR"))
            let pages = try await Self.recognize(
                images: imageData, html: html, options: options,
                vlm: vlm, progress: progress)
            metadata["ocr_pages"] = String(pages.imageCount)
            metadata["ocr_engine"] = options.epubOCREngine.rawValue
            metadata["stitched_groups"] = String(pages.stitchedRanges.count)
            // 与 EPUB 一致：OCR 没有产出任何文本时保留 HTML 文本化结果，
            // 不让扫描版变成空文档。
            let text = pages.recognizedPages == 0 ? Self.htmlToText(html) : pages.text
            metadata["ocr_fallback"] = pages.recognizedPages == 0 ? "1" : "0"
            metadata["ocr_pages_recognized"] = String(pages.recognizedPages)
            let doc = ParsedDocument(text: text, source: url.lastPathComponent,
                                     outline: Self.outline(from: text, toc: toc),
                                     metadata: metadata, pageCount: pages.imageCount)
            return finish(doc, options: options, profile: .ocr)
        }

        progress?(.message("MOBI: 文本化 HTML"))
        let text = Self.htmlToText(html)
        let doc = ParsedDocument(text: text, source: url.lastPathComponent,
                                 outline: Self.outline(from: text, toc: toc), metadata: metadata)
        return finish(doc, options: options, profile: .structured)
    }

    /// 供 UI 按需加载：只返回图片字节，不解码、不常驻 CGImage。
    public func scanDocument(at url: URL) throws -> MobiScanDocument? {
        let data = try Data(contentsOf: url)
        let pdb = try MobiPDB(data: data)
        guard let zero = pdb.record(at: 0) else { return nil }
        let palm = try MobiPalmHeader(zero)
        // 旧式纯 PalmDOC 文件没有 MOBI 头，用 libmobi 同样的兜底处理。
        let header = (try? MobiHeader(recordZero: zero))
            ?? MobiHeader.fallback(palm: palm, recordCount: pdb.recordCount)
        let textSet = Self.textRecordSet(pdb: pdb, header: header, palm: palm)
        let kind = MobiCompressionKind(rawValue: palm.compressionRaw) ?? .none
        let effectiveRecords: [Data]
        if MobiDRM.isProtected(palm) {
            effectiveRecords = try MobiDRM.decrypt(
                pdb: pdb, header: header, palm: palm,
                recordIndexes: textSet.indexes, pid: drmPID).records
        } else {
            effectiveRecords = textSet.records
        }
        let raw = try MobiCompression.decompress(
            records: effectiveRecords, kind: kind,
            expectedLength: palm.textLength, huffmanRecords: textSet.dictionary)
        let encoding: String.Encoding = header.textEncoding == 65001 ? .utf8 : .windowsCP1252
        let html = String(data: raw, encoding: encoding)
            ?? String(decoding: raw, as: UTF8.self)
        let images = Self.extractImages(pdb: pdb, header: header, html: html)
        let groups = Self.stitchRanges(html: html, imageCount: images.count)
        let pages = images.enumerated().map {
            MobiScannedPage(index: $0.offset, imageData: $0.element,
                            source: Self.imageMagic($0.element))
        }
        return MobiScanDocument(pages: pages, stitchedRanges: groups)
    }

    /// 按 libmobi util.c:1813-1852 选择文本记录：起始 = 1 + KF8 偏移，
    /// 数量取 PalmDOC header 的 text_record_count，并剥离记录尾块。
    static func textRecordSet(pdb: MobiPDB, header: MobiHeader,
                                      palm: MobiPalmHeader)
        -> (records: [Data], dictionary: [Data]?, indexes: [Int]) {
        // 只有 KF8 文件才用 EXTH 121 作为文本起始偏移；
        // 非 KF8 文件里该 EXTH 字段另有含义（ncx 样本里是 12），不能当偏移。
        // kf8Boundary / recordCount 来自文件头，畸形值可使 start+count 溢出或
        // 下界大于上界；Swift 的 Range(lower..<upper) 在 lower>upper 时会 brk 1。
        let rawOffset = header.isKF8 ? Int(header.kf8Boundary ?? 0) : 0
        // KF8 boundary 指向 KF8 资源/文本切换点；某些 calibre 转换的 HuffDic 书里
        // 它指向文件末尾的资源区（如 1715/1716），不能当文本起点。
        // 文本记录必须在 firstNonBookIndex 之前，因此把 start 钳制到该范围内。
        let textEnd = max(1, min(pdb.recordCount, Int(header.firstNonBookIndex)))
        var start = (rawOffset >= 0 && rawOffset < pdb.recordCount) ? 1 + rawOffset : 1
        if start >= textEnd { start = 1 }
        let count = palm.recordCount > 0 ? min(palm.recordCount, pdb.recordCount) : 0
        guard start > 0, start <= pdb.recordCount, count > 0 else { return ([], nil, []) }
        let upper = min(textEnd, start + count)
        guard start <= upper else { return ([], nil, []) }
        let huffStart = Int(header.huffmanRecordOffset)
        let huffEnd = huffStart >= 0 && header.huffmanRecordCount <= UInt32.max - UInt32(huffStart)
            ? huffStart + Int(header.huffmanRecordCount) : pdb.recordCount
        let flags = header.extraDataFlags ?? 0
        var text: [Data] = []
        var indexes: [Int] = []
        for index in start..<upper {
            if huffStart > 0, index >= huffStart, index < huffEnd { continue }
            guard let record = pdb.record(at: index) else { continue }
            let extra = extraDataSize([UInt8](record), flags: flags)
            text.append(extra > 0 && extra < record.count
                        ? Data(record.prefix(record.count - extra)) : record)
            indexes.append(index)
        }
        let dictUpper = min(pdb.recordCount, huffEnd)
        let dictionary = (huffStart > 0 && huffStart < dictUpper)
            ? (huffStart..<dictUpper).compactMap { pdb.record(at: $0) }
            : nil
        return (text, dictionary, indexes)
    }

    /// libmobi read.c:481-509：从记录尾部倒序解析 multibyte / extra 数据块。
    private static func extraDataSize(_ bytes: [UInt8], flags: UInt16) -> Int {
        guard flags != 0, !bytes.isEmpty else { return 0 }
        var cursor = bytes.count - 1
        var extra = 0
        for bit in stride(from: 15, through: 1, by: -1)
        where flags & (UInt16(1) << UInt16(bit)) != 0 {
            var value: UInt32 = 0
            var shift: UInt32 = 0
            var byteCount = 0
            var stopped = false
            while cursor >= 0, byteCount < 4 {
                let byte = bytes[cursor]
                cursor -= 1
                value |= UInt32(byte & 0x7F) << shift
                shift += 7
                byteCount += 1
                if byte & 0x80 != 0 { stopped = true; break }
            }
            guard stopped else { return 0 }
            let size = Int(value)
            extra += size
            cursor -= max(0, size - byteCount)
        }
        if flags & 1 != 0, cursor >= 0 {
            extra += Int(bytes[cursor] & 0x3) + 1
        }
        return extra
    }

    // MARK: - 图片提取

    private static func extractImages(pdb: MobiPDB, header: MobiHeader,
                                      html: String) -> [Data] {
        // 先按资源 UID 建池，再按 HTML 中 <img> 的引用顺序取图：
        // 与 EPUB imagePaths(in:) 的文档序遍历语义一致，避免 Record 顺序
        // 与页面顺序不一致时 OCR 文本错页。
        var pool: [Int: Data] = [:]
        func store(_ data: Data, uid: Int) {
            guard isImage(data), pool[uid] == nil else { return }
            pool[uid] = data
        }

        let first = Int(header.firstImageIndex)
        let upper = min(pdb.recordCount, max(1, Int(header.firstNonBookIndex)))
        if first > 0, first < upper {
            for index in first..<upper {
                if let data = pdb.record(at: index) { store(data, uid: index - first) }
            }
        }
        // 封面是独立 record，不占 HTML 资源 UID；用 -1 让它排在没有引用时最后输出。
        if let cover = header.coverRecordIndex, let data = pdb.record(at: Int(cover)) {
            store(data, uid: -1)
        }
        // KF8 资源遍历（含跨 Record 合并的大图）是资源 UID 的权威来源。
        for resource in MobiFlowParser.resources(pdb: pdb, header: header)
        where ["jpeg", "png", "gif", "bmp", "tiff", "webp", "heic"].contains(resource.kind) {
            store(resource.data, uid: resource.index)
        }
        // KF8 ZIP 资源：EOCD 在最后一条 Record，本地文件头在前面的 Record，
        // 必须把整个尾部 Record 区间拼接后再交给 ZIP 阅读器，逐条解析会全丢。
        let boundary = Int(header.kf8Boundary ?? 0)
        let zipStart = boundary > 0 && boundary < pdb.recordCount ? boundary : upper
        if zipStart < pdb.recordCount {
            var zipBuffer = Data()
            var firstPKIndex = zipStart
            for index in zipStart..<pdb.recordCount {
                guard let record = pdb.record(at: index) else { continue }
                if zipBuffer.isEmpty && !record.containsData([0x50, 0x4B, 0x03, 0x04]) {
                    continue
                }
                if zipBuffer.isEmpty { firstPKIndex = index }
                // 上限防止异常文件把整个数据库读进内存。
                if zipBuffer.count + record.count > 512 * 1024 * 1024 { break }
                zipBuffer.append(record)
            }
            if !zipBuffer.isEmpty {
                // 无 resourceNNNN 命名的条目按序号递增 UID，避免全部折叠成同一个 key。
                for (entryIndex, entry) in MobiZIPReader.images(in: zipBuffer).enumerated() {
                    let uid = zipUID(entry.name) ?? (10_000 + firstPKIndex * 1_000 + entryIndex)
                    store(entry.data, uid: uid)
                }
            }
        }

        let refs = imageReferences(in: html)
        var result: [Data] = []
        var fingerprints = Set<String>()
        func emit(_ uid: Int) {
            guard let data = pool[uid] else { return }
            // hashValue 按进程随机且前 64 字节对同源 JPEG 常相同，会静默丢页；
            // 用整图 SHA256 + 长度做稳定指纹。
            let fingerprint = Self.imageFingerprint(data)
            guard fingerprints.insert(fingerprint).inserted else { return }
            result.append(data)
        }
        for uid in refs { emit(uid) }
        for uid in pool.keys.sorted() where !refs.contains(uid) { emit(uid) }
        return result
    }

    /// 稳定图片指纹：SHA256 全文 + 长度，避免进程随机哈希与前缀碰撞。
    private static func imageFingerprint(_ data: Data) -> String {
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        return "\(data.count):\(digest)"
    }

    /// 从 `<img recindex="00001">` / `src="resource00000.jpeg"` 按出现顺序取资源 UID。
    /// libmobi parse_rawml.c:1728-1735：recindex 从 1 起，转成 0 起的 resource UID。
    private static func imageReferences(in html: String) -> [Int] {
        let ns = html as NSString
        guard let tagRegex = try? NSRegularExpression(pattern: #"<img\b[^>]*>"#,
                                                       options: [.caseInsensitive]),
              let recRegex = try? NSRegularExpression(pattern: #"recindex\s*=\s*["']?(\d+)"#,
                                                       options: [.caseInsensitive]),
              let resRegex = try? NSRegularExpression(
                pattern: #"resource0*(\d+)\.[a-z0-9]+"#, options: [.caseInsensitive]),
              let embedRegex = try? NSRegularExpression(
                pattern: #"kindle:embed:([0-9a-z]+)"#, options: [.caseInsensitive]) else {
            return []
        }
        let tags = tagRegex.matches(in: html, options: [],
                                    range: NSRange(location: 0, length: ns.length))
        var result: [Int] = []
        for tag in tags {
            let text = ns.substring(with: tag.range)
            let length = NSRange(location: 0, length: (text as NSString).length)
            if let match = recRegex.firstMatch(in: text, options: [], range: length),
               match.numberOfRanges > 1,
               let range = Range(match.range(at: 1), in: text),
               let value = Int(text[range]), value > 0 {
                result.append(value - 1)
                continue
            }
            if let match = resRegex.firstMatch(in: text, options: [], range: length),
               match.numberOfRanges > 1,
               let range = Range(match.range(at: 1), in: text),
               let value = Int(text[range]) {
                result.append(value)
                continue
            }
            // KF8 常见 kindle:embed:XXXX；libmobi 用 base32 字符集解码成资源 UID。
            if let match = embedRegex.firstMatch(in: text, options: [], range: length),
               match.numberOfRanges > 1,
               let range = Range(match.range(at: 1), in: text),
               let value = decodeKindleEmbed(String(text[range])) {
                result.append(value)
            }
        }
        return result
    }

    /// libmobi util.c: `kindle:embed:` 后是 base32（字母 a-v + 0-9）编码的资源 UID。
    private static func decodeKindleEmbed(_ token: String) -> Int? {
        let alphabet = Array("abcdefghijklmnopqrstuv0123456789")
        var value = 0
        for scalar in token.lowercased().unicodeScalars {
            guard let digit = alphabet.firstIndex(of: Character(scalar)) else { return nil }
            value = value &* 32 + digit
        }
        return value > 0 ? value - 1 : nil
    }

    private static func zipUID(_ name: String) -> Int? {
        guard let match = try? NSRegularExpression(pattern: #"resource0*(\d+)\."#,
                                                    options: [.caseInsensitive])
                .firstMatch(in: name, options: [],
                            range: NSRange(location: 0, length: (name as NSString).length)),
              match.numberOfRanges > 1,
              let range = Range(match.range(at: 1), in: name) else { return nil }
        return Int(name[range])
    }

    private static func isImage(_ data: Data) -> Bool {
        let prefix = data.prefix(12)
        if prefix.hasPrefix([0xFF, 0xD8, 0xFF]) { return true }                 // JPEG
        if prefix.hasPrefix([0x89, 0x50, 0x4E, 0x47]) { return true }           // PNG
        if prefix.hasPrefix([0x47, 0x49, 0x46]) { return true }                 // GIF
        if prefix.hasPrefix([0x42, 0x4D]) { return true }                       // BMP
        if prefix.hasPrefix([0x49, 0x49, 0x2A, 0x00])
            || prefix.hasPrefix([0x4D, 0x4D, 0x00, 0x2A]) { return true }       // TIFF
        if prefix.count >= 12 {
            let brand = String(decoding: prefix[4..<12], as: UTF8.self)
            if brand == "ftypheic" || brand == "ftypheif" || brand == "ftypmif1" {
                return true                                                     // HEIC/HEIF
            }
        }
        // RIFF????WEBP
        if prefix.count >= 12, prefix.hasPrefix([0x52, 0x49, 0x46, 0x46]),
           Array(prefix[8..<12]) == Array("WEBP".utf8) { return true }
        return false
    }

    private static func imageMagic(_ data: Data) -> String {
        let prefix = data.prefix(4)
        if prefix.hasPrefix([0xFF, 0xD8]) { return "jpeg" }
        if prefix.hasPrefix([0x89, 0x50]) { return "png" }
        if prefix.hasPrefix([0x47, 0x49]) { return "gif" }
        if prefix.hasPrefix([0x42, 0x4D]) { return "bmp" }
        return "other"
    }

    /// 从 HTML 推断哪些连续图片需要拼接；组大小优先取真实结构（表格内 img 数），
    /// 没有结构信息时才退回启发值 3。
    private static func stitchRanges(html: String, imageCount: Int) -> [Range<Int>] {
        guard imageCount > 1, MobiStitchEngine.shouldStitch(html: html) else { return [] }
        let step = stitchGroupSize(html: html) ?? 3
        var ranges: [Range<Int>] = []
        var cursor = 0
        while cursor < imageCount {
            let upper = min(imageCount, cursor + step)
            if upper - cursor > 1 { ranges.append(cursor..<upper) }
            cursor = upper
        }
        return ranges
    }

    /// 统计 <table> 内 <img> 数量的众数，作为一组切片大小。
    private static func stitchGroupSize(html: String) -> Int? {
        let pattern = #"<table\b[^>]*>(.*?)</table\s*>"#
        guard let regex = try? NSRegularExpression(
            pattern: pattern, options: [.caseInsensitive, .dotMatchesLineSeparators]) else {
            return nil
        }
        let ns = html as NSString
        let matches = regex.matches(in: html, options: [],
                                    range: NSRange(location: 0, length: ns.length))
        var counts: [Int: Int] = [:]
        for match in matches where match.numberOfRanges > 1 {
            let body = ns.substring(with: match.range(at: 1))
            let count = body.lowercased().components(separatedBy: "<img").count - 1
            if count >= 2 { counts[count, default: 0] += 1 }
        }
        return counts.max { $0.value < $1.value }?.key
    }

    // MARK: - OCR

    private struct OCRResult {
        let text: String
        let imageCount: Int
        let stitchedRanges: [Range<Int>]
        /// 实际识别出正文的页数；0 表示 OCR 全失败，调用方须回退 HTML 文本。
        let recognizedPages: Int
    }

    private static func recognize(images: [Data], html: String,
                                  options: ParserOptions,
                                  vlm: VLMService?,
                                  progress: ((ParserProgress) -> Void)?) async throws -> OCRResult {
        let groups = stitchRanges(html: html, imageCount: images.count)
        let pageGroups = MobiStitchEngine.groups(imageCount: images.count, stitchRanges: groups)
        guard !pageGroups.isEmpty else {
            return OCRResult(text: "", imageCount: images.count, stitchedRanges: groups,
                             recognizedPages: 0)
        }

        let useVLM = options.epubOCREngine == .ovis || options.epubOCREngine == .custom
        var sections: [String] = []
        var recognizedPages = 0
        let batchSize = PDFPageRenderer.maxConcurrentPages
        var cursor = 0
        while cursor < pageGroups.count {
            if Task.isCancelled { throw AgentError.cancelled }
            let end = min(cursor + batchSize, pageGroups.count)
            // 与 EPUB recognizeImages 一致：每批才解码图片，批结束即释放 CGImage。
            var batch: [(page: Int, image: CGImage)] = []
            batch.reserveCapacity(end - cursor)
            var nextPage = cursor + 1
            for group in pageGroups[cursor..<end] {
                let decoded = group.compactMap { index -> CGImage? in
                    guard let source = CGImageSourceCreateWithData(images[index] as CFData, nil)
                    else { return nil }
                    return CGImageSourceCreateImageAtIndex(
                        source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
                }
                guard !decoded.isEmpty else { continue }
                if decoded.count > 1, let stitched = MobiStitchEngine.stitch(decoded) {
                    batch.append((page: nextPage, image: stitched))
                    nextPage += 1
                } else {
                    // 拼图失败不能只取第一张（其余切片内容会静默丢失）：
                    // 每张切片独立成页，按原顺序追加。
                    AgentLog.warning("MOBI: 第 \(nextPage) 组拼图失败，"
                        + "按 \(decoded.count) 张切片独立 OCR")
                    for slice in decoded {
                        batch.append((page: nextPage, image: slice))
                        nextPage += 1
                    }
                }
            }
            guard !batch.isEmpty else {
                cursor = end
                continue
            }
            let texts: [(Int, String)]
            if useVLM, let vlm {
                let normalized = batch.map {
                    ($0.page, options.ocrPreprocess
                        ? (OCRImagePreprocessor.normalize($0.image, binarize: options.ocrBinarize)
                            ?? $0.image)
                        : $0.image)
                }
                texts = try await withThrowingTaskGroup(of: (Int, String).self) { group in
                    for (pageNumber, image) in normalized {
                        group.addTask {
                            do {
                                try Task.checkCancellation()
                                let raw = try await vlm.recognize(
                                    image: image, prompt: OvisPrompt.pageToMarkdown,
                                    maxTokens: OvisPDFParser.maxTokens)
                                return (pageNumber, OvisPDFParser.cleanVLMOutput(raw))
                            } catch is CancellationError {
                                throw CancellationError()
                            } catch {
                                AgentLog.warning("MOBI: 第 \(pageNumber) 页 OCR 失败: "
                                    + error.localizedDescription)
                                return (pageNumber, "")
                            }
                        }
                    }
                    var collected: [(Int, String)] = []
                    for try await item in group { collected.append(item) }
                    return collected
                }
            } else {
                let pages = batch.map {
                    PDFPageImage(pageIndex: $0.page, image: $0.image)
                }
                texts = try await VisionPDFParser.recognizeBatch(
                    pages, preprocess: options.ocrPreprocess,
                    binarize: options.ocrBinarize, languages: options.ocrLanguages)
            }
            for (pageNumber, text) in texts.sorted(by: { $0.0 < $1.0 }) {
                let body = text.trimmingCharacters(in: .whitespacesAndNewlines)
                // 只对有正文的页写页标记：空页标记会被 DocumentCleaner 全部剥掉，
                // 造成「pages.text 非空但最终正文为空」的假成功。
                guard !body.isEmpty else { continue }
                recognizedPages += 1
                sections.append(DocumentCleaner.pageMarker(pageNumber))
                sections.append(body)
            }
            progress?(.page(index: end, total: pageGroups.count, stage: "OCR"))
            cursor = end
        }
        return OCRResult(text: sections.joined(separator: "\n\n"),
                         imageCount: images.count, stitchedRanges: groups,
                         recognizedPages: recognizedPages)
    }

    // MARK: - HTML 文本化

    /// SwiftSoup 的 Node.ownerDocument() 是沿 parent 链递归，DOM 深度达数千层时
    /// 会在解析阶段（reconstructFormattingElements → notifyMutationOwners）栈溢出。
    /// 畸形/恶意 MOBI 可含上千层嵌套 <b>/<i>/<span>，这里在解析前把超深嵌套的
    /// 开标签剥离为普通文本，限制 DOM 深度在安全范围。
    private static func limitHTMLDepth(_ html: String, maxDepth: Int = 96) -> String {
        var depth = 0
        var result = ""
        result.reserveCapacity(html.count)
        var index = html.startIndex
        let end = html.endIndex
        while index < end {
            guard html[index] == "<", index < end else {
                result.append(html[index])
                index = html.index(after: index)
                continue
            }
            // 找标签结束（忽略属性内的 > 简化处理）
            guard let close = html[index...].firstIndex(of: ">") else {
                result.append(contentsOf: html[index...]); break
            }
            let rawTag = String(html[index...close])
            let isClosing = rawTag.hasPrefix("</")
            let isOpen = rawTag.hasPrefix("<") && !isClosing && !rawTag.hasPrefix("<!")
            let name = rawTag
                .trimmingCharacters(in: CharacterSet(charactersIn: "<>/ "))
                .split(whereSeparator: { $0 == " " || $0 == "/" }).first.map(String.init)?
                .lowercased() ?? ""
            if isOpen && !["br", "img", "hr", "meta", "link", "input"].contains(name) {
                if depth >= maxDepth {
                    // 超深的格式化/容器标签本身无文本语义，直接丢弃其标签字节，
                    // 既不让 DOM 变深，也不在正文里留下 &lt;b&gt; 之类的噪音。
                    index = html.index(after: close)
                    continue
                }
                depth += 1
            } else if isClosing, depth > 0 {
                depth -= 1
            }
            result.append(contentsOf: rawTag)
            index = html.index(after: close)
        }
        return result
    }

    private static func htmlToText(_ html: String) -> String {
        let normalized = limitHTMLDepth(html).replacingOccurrences(
            of: "<mbp:pagebreak", with: "<div data-pagebreak=\"1\"")
        var lines: [String] = []
        if let document = try? SwiftSoup.parse(normalized) {
            let _ = try? document.select("script, style, head, title").remove()
            if let body = document.body() {
                collect(body, into: &lines)
            }
            // calibre 转换的 AZW3 常把整本书的多个 XHTML 文档直接拼在一个 flow 里：
            // 每个章节自带 <?xml?><html><body>，正文 <p> 落在“空 body”之外，
            // SwiftSoup 只保留第一个空 body。body 收集到的常常只有分页标记/空白，
            // 因此以“非空白文本字符数”判断，不足时回退整个 document 遍历。
            let substantive = lines.reduce(0) { $0 + $1.trimmingCharacters(in: .whitespacesAndNewlines).count }
            if substantive < 64 {
                lines.removeAll()
                collect(document, into: &lines)
            }
        } else {
            let raw = html.replacingOccurrences(of: #"<[^>]+>"#, with: "\n",
                                                options: .regularExpression)
            lines = raw.components(separatedBy: .newlines).map {
                $0.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        return lines.filter { !$0.isEmpty }.joined(separator: "\n")
    }

    private static func collect(_ node: SwiftSoup.Node, into lines: inout [String]) {
        // 迭代式 DFS：calibre 把多章 XHTML 拼进一个 flow，SwiftSoup 对畸形嵌套
        // 会生成极深节点树，递归既可能栈溢出，也会因深度上限截断导致正文丢失。
        struct Frame { let node: SwiftSoup.Node }
        var stack: [Frame] = [Frame(node: node)]
        while let frame = stack.popLast() {
            let current = frame.node
            if let text = current as? SwiftSoup.TextNode {
                let value = text.text().trimmingCharacters(in: .whitespacesAndNewlines)
                if !value.isEmpty { lines.append(value) }
                continue
            }
            guard let element = current as? SwiftSoup.Element else { continue }
            if let pageBreak = try? element.attr("data-pagebreak"), pageBreak == "1" {
                lines.append("\n")
            }
            let tag = element.tagName().lowercased()
            if tag == "img" {
                let alt = (try? element.attr("alt")) ?? ""
                let title = (try? element.attr("title")) ?? ""
                let label = alt.isEmpty ? title : alt
                lines.append(label.isEmpty ? "[图片]" : "[\(label)]")
            }
            if ["p", "div", "h1", "h2", "h3", "h4", "li", "br", "tr"].contains(tag) {
                lines.append("\n")
            }
            for child in element.getChildNodes().reversed() {
                stack.append(Frame(node: child))
            }
        }
    }

    private static func outline(from text: String, toc: [MobiTOCItem] = []) -> String {
        if !toc.isEmpty {
            return toc.map { item in
                String(repeating: "#", count: max(1, min(6, item.level))) + " " + item.title
            }.joined(separator: "\n")
        }
        let candidates = text.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { $0.count >= 2 && $0.count <= 80 }
        return Array(candidates.prefix(60)).joined(separator: "\n")
    }
}

private extension Data {
    func hasPrefix(_ bytes: [UInt8]) -> Bool {
        guard count >= bytes.count else { return false }
        return zip(prefix(bytes.count), bytes).allSatisfy { $0 == $1 }
    }

    func containsData(_ bytes: [UInt8]) -> Bool {
        guard !bytes.isEmpty, count >= bytes.count else { return false }
        // 不再复制整条 Data、也不为每个位置建临时数组：
        // 用UnsafeRawBufferPointer 滑动窗口逐字节比较。
        return withUnsafeBytes { raw in
            let source = raw.bindMemory(to: UInt8.self)
            guard source.count >= bytes.count else { return false }
            outer: for index in 0...(source.count - bytes.count) {
                for j in 0..<bytes.count where source[index + j] != bytes[j] {
                    continue outer
                }
                return true
            }
            return false
        }
    }
}
