import Foundation

// MARK: - MOBI / AZW3 容器解析（Palm Database）
//
// 所有多字节整数均为大端序（Big-Endian）。文件格式参考 PalmDOC / Mobipocket
// 公开规范：PDB 容器 → Record 0（PalmDOC header + MOBI header + EXTH）→
// 文本 Record（压缩或明文）→ 图片 Record / KF8 资源块。

/// 边界安全的大端序读取器；越界返回 nil，绝不让畸形文件触发越界崩溃。
struct MobiByteReader {
    let bytes: [UInt8]
    private(set) var offset: Int

    init(_ data: Data, offset: Int = 0) {
        self.bytes = [UInt8](data)
        self.offset = offset
    }

    var remaining: Int { max(0, bytes.count - offset) }

    mutating func skip(_ count: Int) {
        offset = min(bytes.count, max(0, offset + count))
    }

    mutating func uint8() -> UInt8? {
        guard offset < bytes.count else { return nil }
        defer { offset += 1 }
        return bytes[offset]
    }

    mutating func uint16() -> UInt16? {
        guard offset + 2 <= bytes.count else { return nil }
        defer { offset += 2 }
        return (UInt16(bytes[offset]) << 8) | UInt16(bytes[offset + 1])
    }

    mutating func uint32() -> UInt32? {
        guard offset + 4 <= bytes.count else { return nil }
        defer { offset += 4 }
        return (UInt32(bytes[offset]) << 24) | (UInt32(bytes[offset + 1]) << 16)
            | (UInt32(bytes[offset + 2]) << 8) | UInt32(bytes[offset + 3])
    }

    mutating func data(_ count: Int) -> Data? {
        guard count >= 0, offset + count <= bytes.count else { return nil }
        defer { offset += count }
        return Data(bytes[offset..<(offset + count)])
    }

    mutating func ascii(_ count: Int) -> String? {
        guard let raw = data(count) else { return nil }
        return String(decoding: raw, as: UTF8.self)
            .trimmingCharacters(in: CharacterSet(charactersIn: "\0 "))
    }
}

/// PDB（Palm Database）记录条目：4 字节偏移 + 4 字节属性/唯一 ID。
public struct MobiPDBRecordInfo: Sendable {
    public let offset: Int
    public let attributes: UInt8
}

/// PDB 容器：解析头、记录偏移表与每条记录的字节范围。
public struct MobiPDB: Sendable {
    public let name: String
    public let type: UInt32
    public let creator: UInt32
    public let records: [Data]
    public let recordInfo: [MobiPDBRecordInfo]

    public var recordCount: Int { records.count }

    public init(data: Data) throws {
        var reader = MobiByteReader(data)
        guard let name = reader.ascii(32),
              let attributes = reader.uint16(),
              let version = reader.uint16(),
              reader.data(12) != nil,             // 三个日期字段
              reader.data(12) != nil,             // modNumber / appInfo / sortInfo
              let type = reader.uint32(),
              let creator = reader.uint32(),
              reader.data(8) != nil,              // uniqueIDseed / nextRecordList
              let count = reader.uint16() else {
            throw AgentError.parse("MOBI: PDB 头过短或字段缺失")
        }
        _ = attributes
        _ = version
        // PDB 记录数是 UInt16；大型合集书（全集/套装）可有 2~4 万条记录，
        // 原 16384 上限会误拒合法文件。仅排除 0 与超出 UInt16 范围，
        // 并记录表长度必须能放进文件（下方逐条读取自然保证）。
        guard count > 0 else {
            throw AgentError.parse("MOBI: 记录数为 0")
        }
        var info: [MobiPDBRecordInfo] = []
        info.reserveCapacity(Int(count))
        for _ in 0..<Int(count) {
            guard let offset = reader.uint32(), let attr = reader.uint8(),
                  reader.data(3) != nil else {
                throw AgentError.parse("MOBI: PDB 记录表截断")
            }
            info.append(MobiPDBRecordInfo(offset: Int(offset), attributes: attr))
        }
        // 记录表与第一条记录之间有 2 字节填充。
        reader.skip(2)
        var decoded: [Data] = []
        decoded.reserveCapacity(info.count)
        var validInfo: [MobiPDBRecordInfo] = []
        validInfo.reserveCapacity(info.count)
        for (index, entry) in info.enumerated() {
            let next = index + 1 < info.count ? info[index + 1].offset : data.count
            // 部分文件记录表声明完整但实际被截断（offset 超出文件大小），
            // 或记录表被混淆为随机大数。libmobi 会直接判损坏；这里做容错：
            // 跳过无效记录、保留前面可用的记录，尽量提取部分正文。
            guard entry.offset >= 0, entry.offset < data.count,
                  next >= entry.offset, next <= data.count else { continue }
            decoded.append(Data(data[entry.offset..<next]))
            validInfo.append(entry)
        }
        guard !decoded.isEmpty else {
            throw AgentError.parse("MOBI: 全部记录偏移非法（记录表损坏或文件截断）")
        }
        self.name = name
        self.type = type
        self.creator = creator
        self.recordInfo = validInfo
        self.records = decoded
    }

    public func record(at index: Int) -> Data? {
        guard index >= 0, index < records.count else { return nil }
        return records[index]
    }
}

/// PalmDOC 头（Record 0 前 16 字节）。
public struct MobiPalmHeader: Sendable {
    public let compressionRaw: UInt16
    public let textLength: Int
    public let recordCount: Int
    public let recordSize: Int
    public let encryptionRaw: UInt16

    public enum Compression: UInt16, Sendable {
        case none = 1
        case palmDOC = 2
        case huffDic = 17480
        case huffDicLegacy = 3
    }

    public var compression: Compression? { Compression(rawValue: compressionRaw) }
    public var isEncrypted: Bool { encryptionRaw != 0 }

    public init(_ data: Data) throws {
        var reader = MobiByteReader(data)
        guard let compression = reader.uint16(),
              reader.uint16() != nil,
              let length = reader.uint32(),
              let records = reader.uint16(),
              let size = reader.uint16(),
              let encryption = reader.uint16() else {
            throw AgentError.parse("MOBI: PalmDOC 头截断")
        }
        self.compressionRaw = compression
        self.textLength = Int(length)
        self.recordCount = Int(records)
        self.recordSize = Int(size)
        self.encryptionRaw = encryption
    }
}

/// MOBI 头关键字段（相对 MOBI 头起始位置的固定偏移）。
public struct MobiHeader: Sendable {
    public let version: UInt32
    public let mobiType: UInt32
    public let textEncoding: UInt32
    public let firstNonBookIndex: UInt32
    public let fullName: String
    public let firstImageIndex: UInt32
    public let huffmanRecordOffset: UInt32
    public let huffmanRecordCount: UInt32
    /// 实际是 datp_rec_index / datp_rec_count（libmobi mobi.h:326），
    /// HuffDic 字典从 HUFF/CDIC 记录自身解析，不使用这两个字段。
    public let huffmanTableOffset: UInt32
    public let huffmanTableLength: UInt32
    public let exthOffset: Int?
    public let coverRecordIndex: UInt32?
    public let extraDataFlags: UInt16?
    public let kf8Boundary: UInt32?        // EXTH 121 原始值
    /// MOBI 头相对偏移 0xB0/0xB4（Record0 绝对 192/196）。
    public let fdstIndex: UInt32
    public let fdstSectionCount: UInt32
    /// Record0 绝对 244 / 40 / 44。
    public let ncxIndex: UInt32
    public let orthIndex: UInt32
    public let inflIndex: UInt32

    /// libmobi util.c: 格式版本 >= 8 才是 KF8；mobiType 232 亦为 KF8。
    public var isKF8: Bool { version >= 8 || mobiType == 232 }

    public init(version: UInt32, mobiType: UInt32, textEncoding: UInt32,
                firstNonBookIndex: UInt32, fullName: String,
                firstImageIndex: UInt32, huffmanRecordOffset: UInt32,
                huffmanRecordCount: UInt32, huffmanTableOffset: UInt32,
                huffmanTableLength: UInt32, exthOffset: Int?,
                coverRecordIndex: UInt32?, extraDataFlags: UInt16?,
                kf8Boundary: UInt32?, fdstIndex: UInt32 = 0xFFFFFFFF,
                fdstSectionCount: UInt32 = 0, ncxIndex: UInt32 = 0xFFFFFFFF,
                orthIndex: UInt32 = 0xFFFFFFFF, inflIndex: UInt32 = 0xFFFFFFFF) {
        self.version = version
        self.mobiType = mobiType
        self.textEncoding = textEncoding
        self.firstNonBookIndex = firstNonBookIndex
        self.fullName = fullName
        self.firstImageIndex = firstImageIndex
        self.huffmanRecordOffset = huffmanRecordOffset
        self.huffmanRecordCount = huffmanRecordCount
        self.huffmanTableOffset = huffmanTableOffset
        self.huffmanTableLength = huffmanTableLength
        self.exthOffset = exthOffset
        self.coverRecordIndex = coverRecordIndex
        self.extraDataFlags = extraDataFlags
        self.kf8Boundary = kf8Boundary
        self.fdstIndex = fdstIndex
        self.fdstSectionCount = fdstSectionCount
        self.ncxIndex = ncxIndex
        self.orthIndex = orthIndex
        self.inflIndex = inflIndex
    }

    /// 纯 PalmDOC（Record 0 无 MOBI 头）的兜底：libmobi 同样允许此类旧文件。
    public static func fallback(palm: MobiPalmHeader, recordCount: Int) -> MobiHeader {
        MobiHeader(version: 0, mobiType: 2, textEncoding: 1252,
                   firstNonBookIndex: UInt32(recordCount), fullName: "",
                   firstImageIndex: 0, huffmanRecordOffset: 0, huffmanRecordCount: 0,
                   huffmanTableOffset: 0, huffmanTableLength: 0,
                   exthOffset: nil, coverRecordIndex: nil,
                   extraDataFlags: nil, kf8Boundary: nil)
    }

    /// 从 Record 0 解析 PalmDOC + MOBI + EXTH。MOBI 头从偏移 16 开始。
    public init(recordZero: Data) throws {
        let palm = try MobiPalmHeader(recordZero)
        var mobi = MobiByteReader(recordZero, offset: 16)
        guard mobi.ascii(4) == "MOBI",
              let declaredLength = mobi.uint32(),
              let mobiType = mobi.uint32(),
              let textEncoding = mobi.uint32() else {
            throw AgentError.parse("MOBI: 缺少 MOBI 标识")
        }
        // libmobi read.c: 旧文件 header_length 可能为 0，回退为 24。
        let headerLength = declaredLength == 0 ? 24 : Int(declaredLength)
        guard headerLength >= 24 else {
            throw AgentError.parse("MOBI: MOBI 头长度异常 \(declaredLength)")
        }
        let start = 16
        func field(_ relative: Int) -> UInt32? {
            guard relative >= 0, relative + 4 <= headerLength,
                  start + relative + 4 <= recordZero.count else { return nil }
            var reader = MobiByteReader(recordZero, offset: start + relative)
            return reader.uint32()
        }
        let version = field(0x14) ?? 0
        let firstNonBook = field(0x40) ?? 0
        let nameOffset = field(0x44) ?? 0
        let nameLength = field(0x48) ?? 0
        let firstImage = field(0x5C) ?? 0
        let huffOffset = field(0x60) ?? 0
        let huffCount = field(0x64) ?? 0
        let huffTableOffset = field(0x68) ?? 0
        let huffTableLength = field(0x6C) ?? 0
        let exthFlags = field(0x70) ?? 0
        let fdstIndex = field(0xB0) ?? 0xFFFFFFFF
        let fdstSectionCount = field(0xB4) ?? 0
        let ncxIndex = field(0xE4) ?? 0xFFFFFFFF
        let orthIndex = field(0x18) ?? 0xFFFFFFFF
        let inflIndex = field(0x1C) ?? 0xFFFFFFFF
        let encoding: String.Encoding = textEncoding == 65001 ? .utf8 : .windowsCP1252

        var fullName = ""
        if nameOffset > 0, nameLength > 0, nameLength <= 1024,
           Int(nameOffset) + Int(nameLength) <= recordZero.count,
           let decoded = String(data: recordZero[Int(nameOffset)..<(Int(nameOffset) + Int(nameLength))],
                                encoding: encoding) {
            fullName = decoded
        }

        // libmobi mobi.h: extra_flags 在绝对偏移 242（= start + 0xE2）。
        var extraFlags: UInt16?
        if headerLength >= 0xE4, start + 0xE2 + 2 <= recordZero.count {
            var reader = MobiByteReader(recordZero, offset: start + 0xE2)
            extraFlags = reader.uint16()
        }

        // EXTH 位于 MOBI 头之后；flag 置位时必存在，未置位也尝试兜底解析。
        var exth: MobiEXTH?
        let candidate = start + headerLength
        if candidate + 12 <= recordZero.count,
           let parsed = try? MobiEXTH(recordZero, start: candidate, encoding: encoding) {
            exth = parsed
        }
        let exthCover = exth?.uint32(for: 201)
        let headerCover = field(0xF0)
        let cover: UInt32?
        if let exthCover, exthCover != 0, exthCover != 0xFFFFFFFF {
            cover = exthCover
        } else if let headerCover, headerCover != 0 {
            cover = headerCover
        } else {
            cover = nil
        }

        self.version = version
        self.mobiType = mobiType
        self.textEncoding = textEncoding
        self.firstNonBookIndex = firstNonBook
        self.fullName = fullName
        self.firstImageIndex = firstImage
        self.huffmanRecordOffset = huffOffset
        self.huffmanRecordCount = huffCount
        self.huffmanTableOffset = huffTableOffset
        self.huffmanTableLength = huffTableLength
        self.exthOffset = exth != nil ? candidate : nil
        self.coverRecordIndex = cover
        self.extraDataFlags = extraFlags
        self.kf8Boundary = exth?.uint32(for: 121)
        self.fdstIndex = fdstIndex
        self.fdstSectionCount = fdstSectionCount
        self.ncxIndex = ncxIndex
        self.orthIndex = orthIndex
        self.inflIndex = inflIndex
        _ = palm
        _ = exthFlags
    }
}

/// EXTH 元数据记录：type + length（含 8 字节头）+ payload。
public struct MobiEXTHRecord: Sendable {
    public let type: UInt32
    public let payload: Data
}

public struct MobiEXTH: Sendable {
    public let records: [MobiEXTHRecord]
    private let encoding: String.Encoding

    public init(_ data: Data, start: Int,
                encoding: String.Encoding = .utf8) throws {
        guard start >= 0, start + 12 <= data.count else {
            throw AgentError.parse("MOBI: EXTH 偏移越界")
        }
        var header = MobiByteReader(data, offset: start)
        guard header.ascii(4) == "EXTH",
              let length = header.uint32(),
              let count = header.uint32() else {
            throw AgentError.parse("MOBI: EXTH 标识缺失")
        }
        let end = min(data.count, start + Int(length))
        var cursor = start + 12
        var result: [MobiEXTHRecord] = []
        result.reserveCapacity(min(Int(count), 4096))
        for _ in 0..<Int(count) {
            guard cursor + 8 <= end else { break }
            var reader = MobiByteReader(data, offset: cursor)
            guard let type = reader.uint32(), let recordLength = reader.uint32(),
                  recordLength >= 8, cursor + Int(recordLength) <= end else { break }
            let payload = Data(data[(cursor + 8)..<(cursor + Int(recordLength))])
            result.append(MobiEXTHRecord(type: type, payload: payload))
            cursor += Int(recordLength)
        }
        self.records = result
        self.encoding = encoding
    }

    public func uint32(for type: UInt32) -> UInt32? {
        guard let record = records.first(where: { $0.type == type }) else { return nil }
        var reader = MobiByteReader(record.payload)
        return reader.uint32()
    }

    public func string(for type: UInt32) -> String? {
        guard let record = records.first(where: { $0.type == type }),
              var value = String(data: record.payload, encoding: encoding) else { return nil }
        value = value.trimmingCharacters(in: .whitespacesAndNewlines)
        // libmobi util.c: 元数据里的 HTML 实体需要还原（&amp;、&lt; 等）。
        for (entity, replacement) in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                                      ("&quot;", "\""), ("&#39;", "'"), ("&nbsp;", " ")] {
            value = value.replacingOccurrences(of: entity, with: replacement)
        }
        return value
    }

    /// 常用元数据。type 常量按 libmobi mobi.h：
    /// title=99（503 是 UpdatedTitle）、language=524（109 是 rights）。
    public func metadata() -> [String: String] {
        var out: [String: String] = [:]
        let mapping: [(UInt32, String)] = [
            (99, "title"), (100, "author"), (101, "publisher"),
            (102, "imprint"), (103, "description"), (104, "isbn"),
            (105, "subject"), (106, "pubdate"), (108, "contributor"),
            (109, "rights"), (113, "asin"), (503, "updated_title"),
            (524, "language"),
        ]
        for (type, key) in mapping {
            if let value = string(for: type), !value.isEmpty { out[key] = value }
        }
        return out
    }
}

// MARK: - 压缩解压

/// 压缩类型（PalmDOC header 的 compression 字段）。
public enum MobiCompressionKind: UInt16, Sendable {
    case none = 1
    case palmDOC = 2
    /// libmobi: MOBI_COMPRESSION_HUFFCDIC = 17480（不是 3）。
    case huffDic = 17480
    /// 少数非标准工具写成 3，兼容处理。
    case huffDicLegacy = 3
}

public enum MobiCompressionError: Error {
    case unsupported(UInt16)
    case corrupted(String)
}

public enum MobiCompression {
    /// 单本电子书的正文解压上限。畸形文件的 textLength 可达 UInt32.max，
    /// 不设上限会在并发同步时分配数 GB 内存并触发 SIGTRAP。
    public static let maxTextOutputBytes = 256 * 1024 * 1024

    /// 按压缩类型解压文本 Record；HuffDic 需要 HUFF/CDIC 字典记录。
    public static func decompress(
        records: [Data],
        kind: MobiCompressionKind,
        expectedLength: Int,
        huffmanRecords: [Data]? = nil
    ) throws -> Data {
        switch kind {
        case .none:
            let joined = records.joined()
            let limit = expectedLength > 0
                ? min(expectedLength, Self.maxTextOutputBytes) : joined.count
            return Data(joined.prefix(limit))
        case .palmDOC:
            // textLength 来自文件头（可达 UInt32.max），直接预分配会让畸形文件
            // 在并发解析时触发内存分配 fatalError(SIGTRAP)。钳制到合理上限。
            let safeLength = min(max(0, expectedLength), Self.maxTextOutputBytes)
            var out = Data()
            out.reserveCapacity(min(safeLength, records.count * 16 * 1024))
            for record in records {
                try decompressPalmDOC(record, into: &out, limit: safeLength)
            }
            return out
        case .huffDic, .huffDicLegacy:
            guard let huffmanRecords else {
                throw MobiCompressionError.corrupted("缺少 HuffDic 字典记录")
            }
            return try decompressHuffDic(records, dictionary: huffmanRecords,
                                         limit: expectedLength)
        }
    }

    /// PalmDOC / LZ77：控制字节驱动的字面量、重复、滑动窗口与空格编码。
    public static func decompressPalmDOC(_ data: Data, into output: inout Data,
                                         limit: Int) throws {
        let bytes = [UInt8](data)
        var i = 0
        while i < bytes.count {
            if limit > 0, output.count >= limit { return }
            let control = bytes[i]
            i += 1
            switch control {
            case 0:
                guard i < bytes.count else {
                    throw MobiCompressionError.corrupted("字面量截断")
                }
                output.append(bytes[i]); i += 1
            case 1...8:
                let count = Int(control)
                guard i + count <= bytes.count else {
                    throw MobiCompressionError.corrupted("短字面量串截断")
                }
                output.append(contentsOf: bytes[i..<(i + count)])
                i += count
            case 9...0x7F:
                // libmobi compression.c: 该区间是「单字节原样输出」，不是重复串。
                output.append(control)
            case 0x80...0xBF:
                // 双字节 LZ77：下一字节低 3 位是长度，距离直接取 11 位值。
                guard i < bytes.count else {
                    throw MobiCompressionError.corrupted("滑窗引用截断")
                }
                let next = bytes[i]; i += 1
                let pair = (UInt32(control) << 8) | UInt32(next)
                let length = Int(next & 0x07) + 3
                let distance = Int((pair >> 3) & 0x7FF)
                guard distance > 0, distance <= output.count else {
                    throw MobiCompressionError.corrupted("滑窗距离越界")
                }
                let start = output.count - distance
                for index in 0..<length {
                    if limit > 0, output.count >= limit { return }
                    output.append(output[start + index])
                }
            case 0xC0...0xFF:
                // 空格 + (byte ^ 0x80)，例如 0xC1 → 空格 + 'A'。
                output.append(0x20)
                output.append(control ^ 0x80)
            default:
                throw MobiCompressionError.corrupted("未知 PalmDOC 控制字节")
            }
        }
    }

    private static func repeatedBytes(_ row: UInt8, _ count: Int) -> [UInt8] {
        Array(repeating: row, count: count)
    }
}

// MARK: - HuffDic（HUFF/CDIC）
//
// 按 libmobi src/read.c（mobi_parse_huff / mobi_parse_cdic）与
// src/compression.c（mobi_decompress_huffman_internal）实现：
// HUFF 记录给 256 项一级索引表与 32 组 min/max 码表；CDIC 记录给
// 符号偏移表与字典数据；解压用 32 位 MSB 查表 + 递归展开压缩符号。

private struct MobiHuffCdic {
    var table1 = [UInt32](repeating: 0, count: 256)
    var mincode = [UInt32](repeating: 0, count: 33)
    var maxcode = [UInt32](repeating: 0xFFFFFFFF, count: 33)
    var codeLength = 0
    var indexCount = 0
    var indexRead = 0
    var symbolOffsets: [UInt16] = []
    /// 每条 CDIC 记录去掉 16 字节头后的字典数据。
    var symbols: [[UInt8]] = []
}

private struct MobiHuffReader {
    let bytes: [UInt8]
    var offset = 0
    var buffer: UInt64 = 0

    mutating func refill() {
        var value: UInt64 = 0
        for index in 0..<8 {
            let position = offset + index
            let byte = position < bytes.count ? bytes[position] : 0
            value |= UInt64(byte) << UInt64((7 - index) * 8)
        }
        buffer = value
        // 8 字节填充，但只前进 4 字节（4 字节重叠）。
        offset += 4
    }
}

private enum MobiHuffError: Error, CustomStringConvertible {
    case corrupt(String)

    var description: String {
        switch self {
        case .corrupt(let message): return "HuffDic: \(message)"
        }
    }
}

private func huffBE32(_ bytes: [UInt8], _ at: Int) throws -> UInt32 {
    guard at >= 0, at + 4 <= bytes.count else {
        throw MobiHuffError.corrupt("HUFF/CDIC 字段越界")
    }
    return (UInt32(bytes[at]) << 24) | (UInt32(bytes[at + 1]) << 16)
        | (UInt32(bytes[at + 2]) << 8) | UInt32(bytes[at + 3])
}

private func parseHuffCdic(huffRecord: Data, cdicRecords: [Data]) throws -> MobiHuffCdic {
    var cdic = MobiHuffCdic()
    let huff = [UInt8](huffRecord)
    guard huff.count >= 24,
          String(decoding: huff[0..<4], as: UTF8.self) == "HUFF" else {
        throw MobiHuffError.corrupt("HUFF 标识无效")
    }
    guard try huffBE32(huff, 4) >= 24 else {
        throw MobiHuffError.corrupt("HUFF 头长度无效")
    }
    let data1 = Int(try huffBE32(huff, 8))
    let data2 = Int(try huffBE32(huff, 12))
    guard data1 >= 0, data1 + 256 * 4 <= huff.count,
          data2 >= 0, data2 + 31 * 8 <= huff.count else {
        throw MobiHuffError.corrupt("HUFF 码表越界")
    }
    for index in 0..<256 {
        cdic.table1[index] = try huffBE32(huff, data1 + index * 4)
    }
    cdic.mincode[0] = 0
    cdic.maxcode[0] = 0xFFFFFFFF
    for length in 1..<33 {
        let minimum = try huffBE32(huff, data2 + (length - 1) * 8)
        let maximum = try huffBE32(huff, data2 + (length - 1) * 8 + 4)
        let shift = UInt32(32 - length)
        cdic.mincode[length] = minimum << shift
        cdic.maxcode[length] = ((maximum &+ 1) << shift) &- 1
    }

    for record in cdicRecords {
        let bytes = [UInt8](record)
        guard bytes.count >= 16,
              String(decoding: bytes[0..<4], as: UTF8.self) == "CDIC",
              try huffBE32(bytes, 4) >= 16 else {
            throw MobiHuffError.corrupt("CDIC 标识或头长度无效")
        }
        let count = Int(try huffBE32(bytes, 8))
        let codeLength = Int(try huffBE32(bytes, 12))
        guard count > 0, count <= 1 << 20, codeLength > 0, codeLength <= 16 else {
            throw MobiHuffError.corrupt("CDIC 索引数/码长无效")
        }
        if cdic.indexCount == 0 {
            cdic.indexCount = count
            cdic.codeLength = codeLength
        } else {
            guard count == cdic.indexCount, codeLength == cdic.codeLength else {
                throw MobiHuffError.corrupt("多条 CDIC 参数不一致")
            }
        }
        let remaining = count - cdic.indexRead
        let toRead = min(remaining, 1 << codeLength)
        guard toRead > 0, 16 + toRead * 2 <= bytes.count else {
            throw MobiHuffError.corrupt("CDIC 偏移表越界")
        }
        for index in 0..<toRead {
            let position = 16 + index * 2
            let offset = UInt16(bytes[position]) << 8 | UInt16(bytes[position + 1])
            let symbolStart = 16 + Int(offset)
            guard symbolStart + 2 <= bytes.count else {
                throw MobiHuffError.corrupt("CDIC 符号偏移越界")
            }
            cdic.symbolOffsets.append(offset)
        }
        cdic.symbols.append(Array(bytes[16...]))
        cdic.indexRead += toRead
    }
    guard cdic.indexRead == cdic.indexCount, !cdic.symbols.isEmpty else {
        throw MobiHuffError.corrupt("CDIC 符号数量不完整")
    }
    return cdic
}

private func huffmanDecode(_ bytes: [UInt8], cdic: MobiHuffCdic,
                           depth: Int = 0, maxOutput: Int = 64 * 1024 * 1024) throws -> [UInt8] {
    guard depth <= 20 else { throw MobiHuffError.corrupt("Huffman 递归过深") }
    guard maxOutput >= 0 else { throw MobiHuffError.corrupt("Huffman 输出超出上限") }
    guard !bytes.isEmpty else { return [] }
    var reader = MobiHuffReader(bytes: bytes)
    reader.refill()
    var bitcount = 32
    var bitsleft = bytes.count * 8
    var output: [UInt8] = []

    while bitsleft > 0 {
        if bitcount <= 0 {
            bitcount += 32
            reader.refill()
        }
        let code = UInt32((reader.buffer >> UInt64(bitcount)) & 0xFFFFFFFF)
        let first = cdic.table1[Int(code >> 24)]
        var codeLength = Int(first & 0x1F)
        guard codeLength > 0, codeLength <= 32 else {
            throw MobiHuffError.corrupt("Huffman 码长无效")
        }
        var maxcode: UInt32
        if codeLength == 32 {
            maxcode = 0xFFFFFFFF
        } else {
            let shift = UInt32(32 - codeLength)
            maxcode = ((first >> 8) &+ 1) << shift
            maxcode &-= 1
        }
        if first & 0x80 == 0 {
            // libmobi: 非终止码字先无条件取 maxcode_table[code_length]，
            // 再在 code 小于 mincode 时逐档加长，不能保留 t1 算出的 maxcode。
            maxcode = cdic.maxcode[codeLength]
            while code < cdic.mincode[codeLength] {
                codeLength += 1
                guard codeLength < 33 else {
                    throw MobiHuffError.corrupt("mincode 表越界")
                }
                maxcode = cdic.maxcode[codeLength]
            }
        }
        bitcount -= codeLength
        bitsleft -= codeLength
        if bitsleft < 0 { break }

        let shift = UInt32(32 - codeLength)
        // C 里 unsigned 减法会回绕，Swift 会 trap；非法码字必须显式拒绝。
        guard maxcode >= code else {
            throw MobiHuffError.corrupt("Huffman maxcode 小于码字: index=\(code >> 24) "
                + "t1=\(first) len=\(codeLength) max=\(maxcode) code=\(code) "
                + "bitcount=\(bitcount)")
        }
        let index = Int((maxcode - code) >> shift)
        guard index >= 0, index < cdic.symbolOffsets.count else {
            throw MobiHuffError.corrupt("符号索引越界")
        }
        let cdicIndex = index >> cdic.codeLength
        guard cdicIndex < cdic.symbols.count else {
            throw MobiHuffError.corrupt("CDIC 记录索引越界")
        }
        let symbolData = cdic.symbols[cdicIndex]
        let offset = Int(cdic.symbolOffsets[index])
        guard offset + 2 <= symbolData.count else {
            throw MobiHuffError.corrupt("符号长度字段越界")
        }
        let rawLength = (UInt32(symbolData[offset]) << 8) | UInt32(symbolData[offset + 1])
        let isDecompressed = rawLength & 0x8000 != 0
        let length = Int(rawLength & 0x7FFF)
        let start = offset + 2
        guard start + length <= symbolData.count else {
            throw MobiHuffError.corrupt("符号数据越界")
        }
        let symbol = Array(symbolData[start..<(start + length)])
        let decoded = isDecompressed ? symbol : try huffmanDecode(
            symbol, cdic: cdic, depth: depth + 1, maxOutput: maxOutput - output.count)
        guard decoded.count <= maxOutput - output.count else {
            throw MobiHuffError.corrupt("Huffman 输出超出上限")
        }
        output.append(contentsOf: decoded)
    }
    return output
}

extension MobiCompression {
    /// HuffDic：HUFF + 若干 CDIC 记录构成字典；每条文本 Record 从首字节
    /// 开始按 32 位码流解码（首字节不是字典索引）。
    public static func decompressHuffDic(_ records: [Data], dictionary: [Data],
                                         limit: Int) throws -> Data {
        guard let huff = dictionary.first else {
            throw MobiHuffError.corrupt("缺少 HUFF 记录")
        }
        let cdic = try parseHuffCdic(huffRecord: huff,
                                     cdicRecords: Array(dictionary.dropFirst()))
        var output = Data()
        for record in records {
            let decoded = try huffmanDecode([UInt8](record), cdic: cdic)
            output.append(contentsOf: decoded)
            if limit > 0, output.count >= limit { break }
        }
        return limit > 0 ? Data(output.prefix(limit)) : output
    }
}
