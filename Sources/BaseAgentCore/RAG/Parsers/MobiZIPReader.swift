import Foundation
import Compression

/// MOBI/AZW3（KF8）把图片资源打包在尾部 ZIP 块里。这里实现一个只读的
/// 最小 ZIP 读取器：解析 EOCD + Central Directory + Local Header，
/// 支持 store(0) / deflate(8)，不依赖第三方解压库。
public enum MobiZIPReader {

    public struct Entry: Sendable {
        public let name: String
        public let data: Data
    }

    private static let imageExtensions: Set<String> = [
        "jpg", "jpeg", "png", "gif", "bmp", "tif", "tiff", "webp", "heic", "heif",
    ]

    /// 从 ZIP 字节流里提取图片条目。上限防止 zip bomb 与异常文件耗尽内存。
    public static func images(in data: Data,
                              maxEntries: Int = 2_000,
                              maxTotalBytes: Int = 256 * 1024 * 1024) -> [Entry] {
        let bytes = [UInt8](data)
        guard let eocd = findEOCD(bytes) else { return [] }
        guard eocd + 22 <= bytes.count else { return [] }
        let entryCount = Int(u16(bytes, eocd + 10))
        let centralOffset = Int(u32(bytes, eocd + 16))
        guard entryCount > 0, entryCount <= maxEntries,
              centralOffset >= 0, centralOffset < bytes.count else { return [] }

        var cursor = centralOffset
        var result: [Entry] = []
        var total = 0
        for _ in 0..<entryCount {
            guard cursor + 46 <= bytes.count,
                  u32(bytes, cursor) == 0x02014b50 else { break }
            let method = Int(u16(bytes, cursor + 10))
            let compressedSize = Int(u32(bytes, cursor + 20))
            let uncompressedSize = Int(u32(bytes, cursor + 24))
            let nameLength = Int(u16(bytes, cursor + 28))
            let extraLength = Int(u16(bytes, cursor + 30))
            let commentLength = Int(u16(bytes, cursor + 32))
            let localOffset = Int(u32(bytes, cursor + 42))
            guard nameLength > 0, nameLength <= 4096,
                  cursor + 46 + nameLength + extraLength + commentLength <= bytes.count,
                  localOffset >= 0, localOffset + 30 <= bytes.count,
                  compressedSize > 0, compressedSize <= bytes.count,
                  uncompressedSize > 0, uncompressedSize <= maxTotalBytes,
                  total + uncompressedSize <= maxTotalBytes else {
                cursor += 46 + nameLength + extraLength + commentLength
                continue
            }
            let nameData = Data(bytes[(cursor + 46)..<(cursor + 46 + nameLength)])
            let name = String(decoding: nameData, as: UTF8.self)
            let ext = (name as NSString).pathExtension.lowercased()
            if imageExtensions.contains(ext),
               let local = readLocal(bytes, offset: localOffset,
                                     method: method, compressedSize: compressedSize,
                                     expectedSize: uncompressedSize) {
                result.append(Entry(name: name, data: local))
                total += local.count
            }
            cursor += 46 + nameLength + extraLength + commentLength
        }
        return result
    }

    /// 在尾部 64KB 内倒序查找 End of Central Directory 记录。
    private static func findEOCD(_ bytes: [UInt8]) -> Int? {
        let lower = max(0, bytes.count - 65_557)
        var index = bytes.count - 22
        while index >= lower {
            if u32(bytes, index) == 0x06054b50 { return index }
            index -= 1
        }
        return nil
    }

    private static func readLocal(_ bytes: [UInt8], offset: Int, method: Int,
                                  compressedSize: Int, expectedSize: Int) -> Data? {
        guard u32(bytes, offset) == 0x04034b50 else { return nil }
        let nameLength = Int(u16(bytes, offset + 26))
        let extraLength = Int(u16(bytes, offset + 28))
        let start = offset + 30 + nameLength + extraLength
        guard start >= 0, start + compressedSize <= bytes.count else { return nil }
        switch method {
        case 0:
            return Data(bytes[start..<(start + compressedSize)])
        case 8:
            // 本机 Compression 框架的 COMPRESSION_ZLIB 实测收发裸 deflate
            // （与 ZIP 条目一致），直接解压，不能再补 zlib 头。
            let payload = Data(bytes[start..<(start + compressedSize)])
            var output = [UInt8](repeating: 0, count: expectedSize)
            let count = decode(payload, into: &output)
            guard count == expectedSize else { return nil }
            return Data(output.prefix(count))
        default:
            return nil
        }
    }

    private static func decode(_ payload: Data, into output: inout [UInt8]) -> Int {
        payload.withUnsafeBytes { source in
            output.withUnsafeMutableBytes { destination in
                compression_decode_buffer(
                    destination.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    destination.count,
                    source.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    payload.count, nil, COMPRESSION_ZLIB)
            }
        }
    }

    /// ZIP 结构（libmobi/miniz）全部是小端，与 PDB/MOBI 头的大端相反。
    private static func u16(_ bytes: [UInt8], _ offset: Int) -> UInt16 {
        guard offset + 2 <= bytes.count else { return 0 }
        return UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
    }

    private static func u32(_ bytes: [UInt8], _ offset: Int) -> UInt32 {
        guard offset + 4 <= bytes.count else { return 0 }
        return UInt32(bytes[offset]) | (UInt32(bytes[offset + 1]) << 8)
            | (UInt32(bytes[offset + 2]) << 16) | (UInt32(bytes[offset + 3]) << 24)
    }
}
