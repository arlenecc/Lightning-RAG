import Foundation

/// INDX 条目：label 是 hex id 或词典词，真正标题常存在 CNCX 字符串里。
public struct MobiIndexEntry: Sendable {
    public let label: String
    /// 解码前的原始 label 字节；infl 规则是字节级小程序，必须用它而非 UTF-8 字符串。
    public let rawLabel: [UInt8]
    public let tags: [UInt32: [UInt32]]
}

public struct MobiIndex: Sendable {
    public let entries: [MobiIndexEntry]
    public let cncx: Data?
    public let encoding: UInt32
    public let name: String
}

public struct MobiTOCItem: Sendable {
    public let title: String
    public let level: Int
    public let position: Int
}

/// libmobi src/index.c 的 Swift 复刻：INDX → TAGX / IDXT / CNCX，
/// 用于 NCX 目录、词典 orth 词条与 infl 变形规则。
public enum MobiIndexParser {

    private static let notSet: UInt32 = 0xFFFFFFFF

    // MARK: - INDX 主解析

    public static func parseIndex(pdb: MobiPDB, recordIndex: Int,
                                  kf8Offset: Int = 0) throws -> MobiIndex {
        let start = recordIndex + kf8Offset
        guard start > 0, start < pdb.recordCount,
              let first = pdb.record(at: start) else {
            throw AgentError.parse("INDX: 记录号越界")
        }
        let meta = [UInt8](first)
        guard meta.count >= 56,
              String(decoding: meta[0..<4], as: UTF8.self) == "INDX",
              let headerLength = optionalBE32(meta, 4),
              headerLength > 0, Int(headerLength) <= meta.count else {
            throw AgentError.parse("INDX: 元数据记录无效")
        }
        let encoding = optionalBE32(meta, 28) ?? notSet
        let totalEntries = Int(optionalBE32(meta, 36) ?? 0)
        let idxtOffset = Int(optionalBE32(meta, 20) ?? 0)
        let entriesInRecord = Int(optionalBE32(meta, 24) ?? 0)
        let cncxCount = Int(optionalBE32(meta, 52) ?? 0)
        let ordtType = Int(optionalBE32(meta, 164) ?? 0)
        let ordtCount = Int(optionalBE32(meta, 168) ?? 0)
        let ordt2Offset = Int(optionalBE32(meta, 176) ?? 0)
        let nameOffset = Int(optionalBE32(meta, 180) ?? 0)
        let nameLength = Int(optionalBE32(meta, 184) ?? 0)
        guard totalEntries >= 0, totalEntries <= 10_000 * 65_535,
              entriesInRecord > 0, entriesInRecord <= 10_000,
              cncxCount >= 0, cncxCount <= 0xF else {
            throw AgentError.parse("INDX: 头部字段异常")
        }

        let tagxStart = Int(headerLength)
        let descriptors = parseTAGX(meta, at: tagxStart)
        // ORDT2 section 以 "ORDT" magic 开头（libmobi index.c:mobi_parse_ordt），
        // 必须跳过 4 字节再按 big-endian 读取 UTF-16 码点表。
        let ordt2: [UInt16]? = {
            guard ordt2Offset > 0, ordtCount > 0,
                  ordt2Offset + 4 + ordtCount * 2 <= meta.count,
                  String(decoding: meta[ordt2Offset..<(ordt2Offset + 4)], as: UTF8.self) == "ORDT"
            else { return nil }
            return (0..<ordtCount).map { index in
                let position = ordt2Offset + 4 + index * 2
                return UInt16(meta[position]) << 8 | UInt16(meta[position + 1])
            }
        }()
        let controlByteCount = descriptors.filter { $0.control }.count
        let indexName = readLabel(meta, length: Int(meta[0]), offset: nameOffset,
                                  available: nameLength, ordtType: ordtType,
                                  ordt2: ordt2, ligt: nil as [UInt16]?, encoding: encoding)

        var entries: [MobiIndexEntry] = []
        var recordCursor = start
        while recordCursor < pdb.recordCount, entries.count < totalEntries {
            guard let record = pdb.record(at: recordCursor) else { break }
            let bytes = [UInt8](record)
            if recordCursor > start {
                let localIdxt = Int(optionalBE32(bytes, 20) ?? UInt32(idxtOffset))
                let localCount = Int(optionalBE32(bytes, 24) ?? UInt32(entriesInRecord))
                entries.append(contentsOf: parseEntries(
                    bytes, idxtOffset: localIdxt, count: localCount,
                    descriptors: descriptors, controlByteCount: controlByteCount,
                    ordtType: ordtType, ordt2: ordt2, ligt: nil as [UInt16]?, encoding: encoding))
            }
            recordCursor += 1
        }

        // libmobi index.c:746-752 —— 条目数与声明不符时索引不可信。
        guard entries.count == totalEntries else {
            throw AgentError.parse(
                "INDX 条目数不符：声明 \(totalEntries)，实际 \(entries.count)（记录 \(start)）")
        }

        var cncx: Data?
        if cncxCount > 0 {
            // libmobi index.c:mobi_parse_index —— 首个 INDX record 是 meta，
            // 其后正好有 entriesInRecord 个数据 record，CNCX 紧跟其后。
            // 注意不是 lastRecord + 1，也不是 "CMET" 资源记录。
            let cncxIndex = start + 1 + entriesInRecord
            if cncxIndex < pdb.recordCount {
                cncx = pdb.record(at: cncxIndex)
            }
        }
        return MobiIndex(entries: entries, cncx: cncx, encoding: encoding,
                         name: indexName)
    }

    // MARK: - TAGX

    private struct TagDescriptor {
        let tag: UInt32
        let valuesCount: Int
        let bitmask: UInt8
        let control: Bool
    }

    private static func parseTAGX(_ bytes: [UInt8], at start: Int) -> [TagDescriptor] {
        guard start >= 0, start + 12 <= bytes.count,
              String(decoding: bytes[start..<(start + 4)], as: UTF8.self) == "TAGX" else {
            return []
        }
        let length = Int(be32(bytes, start + 4))
        guard length >= 12, start + length <= bytes.count else { return [] }
        var result: [TagDescriptor] = []
        let count = (length - 12) / 4
        for index in 0..<count {
            let position = start + 12 + index * 4
            let control = bytes[position + 3]
            result.append(TagDescriptor(tag: UInt32(bytes[position]),
                                        valuesCount: Int(bytes[position + 1]),
                                        bitmask: bytes[position + 2],
                                        control: control != 0))
        }
        return result
    }

    // MARK: - IDXT / entries

    private static func parseEntries(_ bytes: [UInt8], idxtOffset: Int, count: Int,
                                     descriptors: [TagDescriptor],
                                     controlByteCount: Int,
                                     ordtType: Int, ordt2: [UInt16]?, ligt: [UInt16]?,
                                     encoding: UInt32) -> [MobiIndexEntry] {
        guard idxtOffset > 0, idxtOffset + count * 2 + 4 <= bytes.count,
              String(decoding: bytes[idxtOffset..<(idxtOffset + 4)], as: UTF8.self) == "IDXT" else {
            return []
        }
        var offsets: [Int] = []
        for index in 0..<count {
            let position = idxtOffset + 4 + index * 2
            offsets.append(Int(bytes[position]) << 8 | Int(bytes[position + 1]))
        }
        offsets.append(idxtOffset)   // 哨兵

        var result: [MobiIndexEntry] = []
        for index in 0..<count {
            let from = offsets[index]
            let to = offsets[index + 1]
            guard from >= 0, to <= bytes.count, from < to else { continue }
            let entryBytes = Array(bytes[from..<to])
            result.append(parseEntry(entryBytes, descriptors: descriptors,
                                     controlByteCount: controlByteCount,
                                     ordtType: ordtType, ordt2: ordt2,
                                     ligt: ligt, encoding: encoding))
        }
        return result
    }

    private static func parseEntry(_ bytes: [UInt8], descriptors: [TagDescriptor],
                                   controlByteCount: Int, ordtType: Int,
                                   ordt2: [UInt16]?, ligt: [UInt16]?,
                                   encoding: UInt32) -> MobiIndexEntry {
        guard !bytes.isEmpty else {
            return MobiIndexEntry(label: "", rawLabel: [], tags: [:])
        }
        let labelLength = Int(bytes[0])
        let labelEnd = min(bytes.count, 1 + labelLength)
        let rawLabel = Array(bytes[1..<labelEnd])
        let label = readLabel(bytes, length: labelLength, offset: 1,
                              available: labelEnd, ordtType: ordtType,
                              ordt2: ordt2, ligt: ligt, encoding: encoding)
        var cursor = labelEnd
        let controlStart = cursor
        cursor = min(bytes.count, cursor + controlByteCount)
        let controls = Array(bytes[controlStart..<cursor])

        // libmobi index.c 分两轮：第一轮收集每个 tag 的 value_count / value_bytes
        // （多位 bitmask 全置位时会从 entry 读一个 varlen 字节预算），第二轮才按
        // 收集顺序读 tag 值。合并成一轮会让游标错位。
        struct PendingTag {
            let tag: UInt32
            let valuesCount: Int
            let valueCount: Int?
            let byteBudget: Int?
        }
        var pending: [PendingTag] = []
        var controlIndex = 0
        for descriptor in descriptors {
            if descriptor.control {
                controlIndex += 1
                continue
            }
            guard controlIndex < controls.count else { continue }
            let masked = controls[controlIndex] & descriptor.bitmask
            guard masked != 0 else { continue }

            var valueCount: Int?
            var byteBudget: Int?
            if masked == descriptor.bitmask {
                if descriptor.bitmask.nonzeroBitCount > 1 {
                    let (budget, used) = readVarlen(bytes, at: cursor)
                    guard used > 0 else { continue }
                    cursor += used
                    byteBudget = Int(budget)
                } else {
                    valueCount = 1
                }
            } else {
                var m = Int(masked)
                var bitmask = Int(descriptor.bitmask)
                while bitmask > 0, bitmask & 1 == 0 {
                    m >>= 1
                    bitmask >>= 1
                }
                valueCount = m
            }
            pending.append(PendingTag(tag: descriptor.tag,
                                      valuesCount: descriptor.valuesCount,
                                      valueCount: valueCount, byteBudget: byteBudget))
        }

        var tags: [UInt32: [UInt32]] = [:]
        for item in pending {
            var values: [UInt32] = []
            if let budget = item.byteBudget {
                guard budget > 0, budget <= 4096 else { continue }
                var consumedTotal = 0
                while consumedTotal < budget, values.count < 1024 {
                    let (value, used) = readVarlen(bytes, at: cursor)
                    guard used > 0 else { break }
                    cursor += used
                    consumedTotal += used
                    values.append(value)
                }
            } else if let count = item.valueCount {
                guard count > 0, count <= 1024 else { continue }
                for _ in 0..<(count * item.valuesCount) {
                    let (value, used) = readVarlen(bytes, at: cursor)
                    guard used > 0 else { break }
                    cursor += used
                    values.append(value)
                }
            }
            if !values.isEmpty { tags[item.tag] = values }
        }
        return MobiIndexEntry(label: label, rawLabel: rawLabel, tags: tags)
    }

    // MARK: - 标签 / 字符串

    private static func readLabel(_ bytes: [UInt8], length: Int, offset: Int,
                                  available: Int, ordtType: Int,
                                  ordt2: [UInt16]?, ligt: [UInt16]?,
                                  encoding: UInt32) -> String {
        guard length > 0, offset >= 0, offset + length <= bytes.count,
              offset + length <= available else { return "" }
        if let ordt2, !ordt2.isEmpty {
            return decodeORDT(bytes, length: length, offset: offset,
                              ordtType: ordtType, ordt2: ordt2)
        }
        return decode(Array(bytes[offset..<(offset + length)]), encoding: encoding)
    }

    /// libmobi index.c:mobi_getstring_ordt —— ORDT2 把编码索引映射成 UTF-16，
    /// 再处理连字与代理对，最终转成 UTF-8 字符串。
    private static func decodeORDT(_ bytes: [UInt8], length: Int, offset: Int,
                                   ordtType: Int, ordt2: [UInt16]) -> String {
        let step = ordtType == 1 ? 1 : 2
        let end = min(offset + length, bytes.count)
        var cursor = offset
        var scalars = String.UnicodeScalarView()
        while cursor + step <= end {
            let index = readORDTIndex(bytes, at: cursor, step: step)
            cursor += step
            var code = index < ordt2.count ? Int(ordt2[index]) : index
            if code <= 5, cursor + step <= end {
                let next = readORDTIndex(bytes, at: cursor, step: step)
                let lig = ligature(code, next)
                if lig == 0xFFFD {
                    // 非法连字：回退，不消耗第二个编码单元
                } else {
                    code = lig
                    cursor += step
                }
            }
            if code >= 0xD800 && code <= 0xDBFF, cursor + step <= end {
                let lowIndex = readORDTIndex(bytes, at: cursor, step: step)
                let low = lowIndex < ordt2.count ? Int(ordt2[lowIndex]) : lowIndex
                if low >= 0xDC00 && low <= 0xDFFF {
                    code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                    cursor += step
                } else {
                    code = 0xFFFD
                }
            }
            if code == 0 || (code >= 0xDC00 && code <= 0xDFFF)
                || (code >= 0xFDD0 && code <= 0xFDEF)
                || (code & 0xFFFE) == 0xFFFE || code > 0x10FFFF {
                code = 0xFFFD
            }
            scalars.append(Unicode.Scalar(UInt32(code)) ?? Unicode.Scalar(0xFFFD)!)
        }
        return String(scalars)
    }

    private static func readORDTIndex(_ bytes: [UInt8], at position: Int, step: Int) -> Int {
        if step == 1 { return Int(bytes[position]) }
        return Int(bytes[position]) << 8 | Int(bytes[position + 1])
    }

    /// libmobi util.c:mobi_ligature_to_utf16 —— ORDT 连字（Œ/œ/Æ/æ/ß）
    private static func ligature(_ first: Int, _ second: Int) -> Int {
        switch first {
        case 1 where second == 0x45: return 0x152
        case 2 where second == 0x65: return 0x153
        case 3 where second == 0x45: return 0xC6
        case 4 where second == 0x65: return 0xE6
        case 5 where second == 0x73: return 0xDF
        default: return 0xFFFD
        }
    }

    private static func decode(_ bytes: [UInt8], encoding: UInt32) -> String {
        let used: String.Encoding = encoding == 65001 ? .utf8 : .windowsCP1252
        return String(data: Data(bytes), encoding: used)
            ?? String(decoding: bytes, as: UTF8.self)
    }

    private static func cncxString(_ cncx: Data?, offset: UInt32,
                                   encoding: UInt32) -> String? {
        guard let cncx else { return nil }
        let bytes = [UInt8](cncx)
        var cursor = Int(offset)
        let (length, consumed) = readVarlen(bytes, at: cursor)
        cursor += consumed
        guard consumed > 0, length > 0, cursor + Int(length) <= bytes.count else { return nil }
        return decode(Array(bytes[cursor..<(cursor + Int(length))]), encoding: encoding)
    }

    private static func readVarlen(_ bytes: [UInt8], at start: Int) -> (UInt32, Int) {
        guard start >= 0, start < bytes.count else { return (0, 0) }
        var value: UInt32 = 0
        var consumed = 0
        for index in start..<min(start + 4, bytes.count) {
            let byte = bytes[index]
            value = (value << 7) | UInt32(byte & 0x7F)
            consumed += 1
            if byte & 0x80 != 0 { break }
        }
        return (value, consumed)
    }

    // MARK: - TOC / 词典

    public static func toc(pdb: MobiPDB, header: MobiHeader,
                           kf8Offset: Int = 0) -> [MobiTOCItem] {
        guard header.ncxIndex != notSet else { return [] }
        guard let index = try? parseIndex(pdb: pdb,
                                          recordIndex: Int(header.ncxIndex),
                                          kf8Offset: kf8Offset) else { return [] }
        var items: [MobiTOCItem] = []
        for entry in index.entries {
            guard let titleOffset = entry.tags[3]?.first,
                  let title = cncxString(index.cncx, offset: titleOffset,
                                         encoding: index.encoding),
                  !title.isEmpty else { continue }
            let position = Int(entry.tags[1]?.first ?? 0)
            let level = Int(entry.tags[4]?.first ?? 1)
            items.append(MobiTOCItem(title: title, level: level, position: position))
        }
        return items
    }

    /// orth 词条 + infl 变形规则（V2）；返回「词: 形式1,形式2」字符串。
    public static func dictionaryMarks(pdb: MobiPDB, header: MobiHeader,
                                       kf8Offset: Int = 0) -> [String] {
        guard header.orthIndex != notSet, header.inflIndex != notSet else { return [] }
        guard let orth = try? parseIndex(pdb: pdb, recordIndex: Int(header.orthIndex),
                                         kf8Offset: kf8Offset),
              let infl = try? parseIndex(pdb: pdb, recordIndex: Int(header.inflIndex),
                                         kf8Offset: kf8Offset) else { return [] }
        var result: [String] = []
        for entry in orth.entries where !entry.label.isEmpty {
            guard let groupIndexes = entry.tags[42], !groupIndexes.isEmpty else { continue }
            var forms: [String] = []
            for groupIndex in groupIndexes {
                guard let ruleEntry = infl.entries.indices.contains(Int(groupIndex))
                    ? infl.entries[Int(groupIndex)] : nil,
                      let groups = ruleEntry.tags[5],
                      let parts = ruleEntry.tags[26],
                      groups.count == parts.count else { continue }
                for (position, partIndex) in parts.enumerated() {
                    guard let rule = infl.entries.indices.contains(Int(partIndex))
                        ? infl.entries[Int(partIndex)] : nil else { continue }
                    // libmobi 在 CNCX 组名为空时输出不带 name 属性的 iform，
                    // 这里同样接受缺失组名，只产出变形词本身。
                    let groupName = cncxString(infl.cncx, offset: groups[position],
                                               encoding: infl.encoding)
                    let transformed = applyInflationRule(
                        Array(entry.label.utf8), rule: rule.rawLabel)
                    if let form = String(bytes: transformed, encoding: .utf8),
                       !form.isEmpty, form != entry.label {
                        if let groupName = groupName, !groupName.isEmpty {
                            forms.append("\(groupName)=\(form)")
                        } else {
                            forms.append(form)
                        }
                    }
                }
            }
            if !forms.isEmpty {
                result.append("\(entry.label): \(forms.joined(separator: ", "))")
            }
        }
        return result
    }

    /// libmobi index.c:960-1014 的变形规则虚拟机。
    private static func applyInflationRule(_ input: [UInt8], rule: [UInt8]) -> [UInt8] {
        var decoded = input
        let originalLength = input.count
        var position = originalLength
        var insert = true
        var left = false

        for byte in rule {
            let c = Int(byte)
            if c <= 4 {
                insert = c <= 2
                let newLeft = (c & 2) != 0
                if newLeft != left {
                    position = newLeft ? originalLength : 0
                }
                left = newLeft
            } else if c > 10 && c < 20 {
                if !left { position = originalLength }
                position -= c - 10
            } else if insert {
                if position < 0 || position > decoded.count { continue }
                decoded.insert(byte, at: position)
                if !left { position += 1 }
            } else {
                if position < 0 || position >= decoded.count { continue }
                decoded.remove(at: position)
                if left { position -= 1 }
            }
        }
        return decoded
    }

    // MARK: - 工具

    private static func be32(_ bytes: [UInt8], _ offset: Int) -> UInt32 {
        guard offset >= 0, offset + 4 <= bytes.count else { return 0 }
        return (UInt32(bytes[offset]) << 24) | (UInt32(bytes[offset + 1]) << 16)
            | (UInt32(bytes[offset + 2]) << 8) | UInt32(bytes[offset + 3])
    }

    private static func optionalBE32(_ bytes: [UInt8], _ offset: Int) -> UInt32? {
        guard offset >= 0, offset + 4 <= bytes.count else { return nil }
        return be32(bytes, offset)
    }
}
