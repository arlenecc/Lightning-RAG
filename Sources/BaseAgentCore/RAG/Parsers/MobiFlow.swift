import Foundation

/// KF8 flow 分片（HTML / CSS / SVG）。
public struct MobiFlowPart: Sendable {
    public let index: Int
    public let mime: String
    public let data: Data
}

/// 资源记录（图片/字体/音视频）。
public struct MobiResource: Sendable {
    public let index: Int
    public let kind: String
    public let data: Data
}

/// KF8 FDST flow 切分与图片资源重组。
/// 参考 libmobi read.c:769-837（FDST）、parse_rawml.c:572-640（资源遍历）。
public enum MobiFlowParser {

    // MARK: - FDST

    public static func flows(pdb: MobiPDB, header: MobiHeader, rawml: Data,
                             kf8Offset: Int = 0) -> [MobiFlowPart] {
        let fallback = [MobiFlowPart(index: 0,
                                     mime: "application/xhtml+xml", data: rawml)]
        guard header.isKF8, header.fdstSectionCount > 1,
              header.fdstIndex != 0xFFFFFFFF else { return fallback }
        let recordIndex = Int(header.fdstIndex) + kf8Offset
        guard let record = pdb.record(at: recordIndex) else { return fallback }
        let bytes = [UInt8](record)
        guard bytes.count >= 12,
              String(decoding: bytes[0..<4], as: UTF8.self) == "FDST",
              be32(bytes, 4) == 12 else { return fallback }
        let count = Int(be32(bytes, 8))
        guard count > 1, count == Int(header.fdstSectionCount),
              bytes.count >= 12 + count * 8 else { return fallback }

        // rawml 只解码一次，避免 flow 数 × 全文大小的重复解码分配。
        let rawmlText = String(decoding: rawml, as: UTF8.self)
        var parts: [MobiFlowPart] = []
        var invalidFlows = 0
        for index in 0..<count {
            let start = Int(be32(bytes, 12 + index * 8))
            let end = Int(be32(bytes, 16 + index * 8))
            guard start <= end, end <= rawml.count else {
                invalidFlows += 1
                continue
            }
            parts.append(MobiFlowPart(index: index,
                                      mime: mimeType(for: index, rawmlText: rawmlText),
                                      data: Data(rawml[start..<end])))
        }
        // 越界 flow 过多时整表不可信，回退整个 rawml，避免选错正文 flow。
        if invalidFlows > 0 {
            AgentLog.warning("KF8 FDST: \(invalidFlows)/\(count) 个 flow 越界，回退完整 rawml")
            return fallback
        }
        return parts.isEmpty ? fallback : parts
    }

    /// libmobi util.c:2195-2218：从 `kindle:flow:%04zu?mime=` 属性判定 MIME。
    private static func mimeType(for index: Int, rawmlText: String) -> String {
        if index == 0 { return "application/xhtml+xml" }
        let text = rawmlText
        let marker = "kindle:flow:" + String(format: "%04d", index) + "?mime="
        guard let found = text.range(of: marker) else { return "application/octet-stream" }
        let tail = String(text[found.upperBound...].prefix(80))
        if tail.contains("text/css") { return "text/css" }
        if tail.contains("image/svg+xml") { return "image/svg+xml" }
        return "application/octet-stream"
    }

    // MARK: - 资源（含跨 Record 大图）

    public static func resources(pdb: MobiPDB, header: MobiHeader) -> [MobiResource] {
        let first = Int(header.firstImageIndex)
        guard first > 0, first < pdb.recordCount else { return [] }
        var result: [MobiResource] = []
        var index = first
        while index < pdb.recordCount {
            guard let record = pdb.record(at: index) else { index += 1; continue }
            var bytes = [UInt8](record)
            if isBoundary(bytes) { break }

            if let kind = imageKind(bytes) {
                // 一张扫描页可能跨多条 Record：累积后续 Record 直到出现结束标记。
                if !hasImageEnd(kind, bytes) {
                    var next = index + 1
                    let cap = 256 * 1024 * 1024
                    while next < pdb.recordCount, !hasImageEnd(kind, bytes), bytes.count < cap {
                        guard let more = pdb.record(at: next) else { next += 1; continue }
                        let moreBytes = [UInt8](more)
                        if isBoundary(moreBytes) { break }
                        // 下一条 Record 是另一张图/媒体头时，当前无尾标记图片已结束。
                        if imageKind(moreBytes) != nil || media(moreBytes) != nil { break }
                        bytes.append(contentsOf: moreBytes)
                        next += 1
                    }
                    index = next - 1
                }
                result.append(MobiResource(index: result.count, kind: kind,
                                           data: Data(bytes)))
            } else if let (kind, payload) = media(bytes) {
                result.append(MobiResource(index: result.count, kind: kind,
                                           data: Data(payload)))
            }
            index += 1
        }
        return result
    }

    /// 图片类型识别；libmobi util.c:2599-2648 的 magic 规则。
    private static func imageKind(_ bytes: [UInt8]) -> String? {
        guard bytes.count >= 8 else { return nil }
        if bytes[0] == 0xFF, bytes[1] == 0xD8, bytes[2] == 0xFF { return "jpeg" }
        if bytes[0] == 0x89, bytes[1] == 0x50, bytes[2] == 0x4E, bytes[3] == 0x47,
           bytes[4] == 0x0D, bytes[5] == 0x0A, bytes[6] == 0x1A, bytes[7] == 0x0A {
            return "png"
        }
        if bytes[0] == 0x47, bytes[1] == 0x49, bytes[2] == 0x46, bytes[3] == 0x38 {
            return "gif"
        }
        if bytes[0] == 0x42, bytes[1] == 0x4D { return "bmp" }
        if (bytes[0] == 0x49 && bytes[1] == 0x49 && bytes[2] == 0x2A && bytes[3] == 0)
            || (bytes[0] == 0x4D && bytes[1] == 0x4D && bytes[2] == 0 && bytes[3] == 0x2A) {
            return "tiff"
        }
        if bytes[0] == 0x52, bytes[1] == 0x49, bytes[2] == 0x46, bytes[3] == 0x46,
           bytes.count >= 12, Array(bytes[8..<12]) == Array("WEBP".utf8) {
            return "webp"
        }
        if bytes.count >= 12,
           String(decoding: bytes[4..<12], as: UTF8.self).hasPrefix("ftyphe") {
            return "heic"
        }
        return nil
    }

    private static func hasImageEnd(_ kind: String, _ bytes: [UInt8]) -> Bool {
        switch kind {
        case "jpeg":
            return bytes.count >= 2
                && bytes[bytes.count - 2] == 0xFF && bytes[bytes.count - 1] == 0xD9
        case "png":
            let iend: [UInt8] = [0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82]
            return bytes.count >= iend.count
                && Array(bytes.suffix(iend.count)) == iend
        case "gif":
            return bytes.last == 0x3B
        case "bmp":
            guard bytes.count >= 6 else { return false }
            let declared = Int(bytes[2]) | (Int(bytes[3]) << 8)
                | (Int(bytes[4]) << 16) | (Int(bytes[5]) << 24)
            return declared > 0 && bytes.count >= declared
        case "webp":
            // RIFF 长度字段 = 文件总长 - 8，小端序
            guard bytes.count >= 8 else { return false }
            let declared = Int(bytes[4]) | (Int(bytes[5]) << 8)
                | (Int(bytes[6]) << 16) | (Int(bytes[7]) << 24)
            return declared > 0 && bytes.count >= declared + 8
        case "tiff", "heic":
            // TIFF 无尾标记、HEIC 是 ISO BMFF box 结构；跨 Record 累积用
            // 下一 Record 是否为已知 magic / BOUNDARY 作为停止条件。
            return false
        default:
            return false
        }
    }

    /// FONT / AUDI / VIDE：头部 4 字节 magic，后 4 字节大端载荷偏移。
    private static func media(_ bytes: [UInt8]) -> (String, [UInt8])? {
        guard bytes.count >= 8,
              let magic = String(bytes: bytes[0..<4], encoding: .ascii) else { return nil }
        switch magic {
        case "AUDI", "VIDE":
            let offset = Int(be32(bytes, 4))
            guard offset > 0, offset < bytes.count else { return nil }
            return (magic == "AUDI" ? "mp3" : "mpg", Array(bytes[offset...]))
        case "FONT":
            return ("font", Array(bytes[8...]))
        default:
            return nil
        }
    }

    private static func isBoundary(_ bytes: [UInt8]) -> Bool {
        guard bytes.count >= 4 else { return false }
        if bytes.count >= 8, String(decoding: bytes[0..<8], as: UTF8.self) == "BOUNDARY" {
            return true
        }
        return bytes[0] == 0xE9 && bytes[1] == 0x8E && bytes[2] == 0x0D && bytes[3] == 0x0A
    }

    private static func be32(_ bytes: [UInt8], _ offset: Int) -> UInt32 {
        guard offset >= 0, offset + 4 <= bytes.count else { return 0 }
        return (UInt32(bytes[offset]) << 24) | (UInt32(bytes[offset + 1]) << 16)
            | (UInt32(bytes[offset + 2]) << 8) | UInt32(bytes[offset + 3])
    }
}
