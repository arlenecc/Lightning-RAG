import Foundation

// MARK: - Mobipocket DRM（libmobi src/encryption.c 的 Swift 复刻）
//
// PC1/PK1 是逐字节流密码：每字节先用 8 组 UInt16 状态生成密钥字节，
// 再用解出的明文字节反馈更新 16 字节 keyCopy。所有 UInt16 运算按回绕处理。
// V1 用内置 reader key 解出主密钥；V2 用 10 字符 PID 派生设备密钥，
// 再解开 Record 0 里的 voucher cookie 得到主密钥。

private struct MobiPK1 {
    private var x1a2: UInt16 = 0
    private var si: UInt16 = 0
    private var x1a0 = [UInt16](repeating: 0, count: 8)
    private var keyCopy: [UInt8]

    init(key: [UInt8]) { keyCopy = key }

    private mutating func code(_ index: Int) -> UInt16 {
        var dx = x1a2 &+ UInt16(index)
        var ax = x1a0[index]
        var cx: UInt16 = 0x015A
        let bx: UInt16 = 0x4E35
        swap(&ax, &si)
        swap(&ax, &dx)
        if ax != 0 { ax = ax &* bx }
        swap(&ax, &cx)
        if ax != 0 {
            ax = ax &* si
            cx = cx &+ ax
        }
        swap(&ax, &si)
        ax = ax &* bx
        dx = dx &+ cx
        ax = ax &+ 1
        x1a2 = dx
        x1a0[index] = ax
        return ax ^ dx
    }

    private mutating func assemble() -> UInt16 {
        x1a0[0] = (UInt16(keyCopy[0]) << 8) | UInt16(keyCopy[1])
        var inter = code(0)
        for index in 1..<8 {
            x1a0[index] = x1a0[index - 1]
                ^ ((UInt16(keyCopy[index * 2]) << 8) | UInt16(keyCopy[index * 2 + 1]))
            inter ^= code(index)
        }
        return inter
    }

    /// PC1 解密：用解出的明文字节反馈更新 keyCopy。
    mutating func process(_ input: [UInt8]) -> [UInt8] {
        transform(input, feedbackIsInput: false)
    }

    /// PC1 加密：用输入明文字节反馈（libmobi encryption.c:171-174）。
    mutating func encrypt(_ input: [UInt8]) -> [UInt8] {
        transform(input, feedbackIsInput: true)
    }

    private mutating func transform(_ input: [UInt8], feedbackIsInput: Bool) -> [UInt8] {
        var result = [UInt8]()
        result.reserveCapacity(input.count)
        for byte in input {
            let inter = assemble()
            let stream = UInt8(truncatingIfNeeded: inter >> 8)
                ^ UInt8(truncatingIfNeeded: inter)
            let output = byte ^ stream
            for index in 0..<16 { keyCopy[index] ^= feedbackIsInput ? byte : output }
            result.append(output)
        }
        return result
    }
}

private enum MobiDRMError: Error, CustomStringConvertible {
    case invalid(String)
    var description: String {
        switch self { case .invalid(let message): return "MOBI DRM: \(message)" }
    }
}

/// DRM 解密后的文本记录 + 解密方式（写入 metadata 便于诊断）。
public struct MobiDRMResult: Sendable {
    public let records: [Data]
    public let method: String

    public init(records: [Data], method: String) {
        self.records = records
        self.method = method
    }
}

public enum MobiDRM {

    private static let readerKeyV1: [UInt8] = Array("QDCVEPMU675RUBSZ".utf8)
    private static let readerKeyV2: [UInt8] = [
        0x72, 0x38, 0x33, 0xB0, 0xB4, 0xF2, 0xE3, 0xCA,
        0xDF, 0x09, 0x01, 0xD6, 0xE2, 0xE0, 0x3F, 0x96,
    ]

    // MARK: - 公开入口

    /// - parameters:
    ///   - recordIndexes: 文本 Record 序号（解密前按 libmobi 从 1 开始，混合 KF8 时 1+kf8offset）
    ///   - pid: V2 所需的 10 字符 PID，如 "LTKULBB*5V"；V1 忽略。
    public static func decrypt(pdb: MobiPDB, header: MobiHeader,
                               palm: MobiPalmHeader,
                               recordIndexes: [Int],
                               pid: String?) throws -> MobiDRMResult {
        guard let zero = pdb.record(at: 0) else {
            throw MobiDRMError.invalid("缺少 Record 0")
        }
        let key: [UInt8]
        let method: String
        switch palm.encryptionRaw {
        case 1:
            key = try keyV1(recordZero: zero, header: header)
            method = "v1"
        case 2:
            guard let pid else {
                throw MobiDRMError.invalid("V2 需要 PID（例如 LTKULBB*5V）")
            }
            key = try keyV2(recordZero: zero, header: header, pid: pid)
            method = "v2"
        default:
            throw MobiDRMError.invalid("不支持的加密类型 \(palm.encryptionRaw)")
        }

        var flags = header.extraDataFlags ?? 0
        if palm.compressionRaw != 17480, palm.compressionRaw != 3 {
            // 非 HuffDic 时 multibyte extra 也在加密范围内（libmobi encryption.c:1223）。
            flags &= 0xFFFE
        }
        var decrypted: [Data] = []
        decrypted.reserveCapacity(recordIndexes.count)
        for index in recordIndexes {
            guard let record = pdb.record(at: index) else { continue }
            let bytes = [UInt8](record)
            let extra = try extraSize(bytes, flags: flags, record: index)
            guard extra < bytes.count else {
                throw MobiDRMError.invalid("第 \(index) 条记录尾块越界")
            }
            var cipher = MobiPK1(key: key)
            let clear = cipher.process(Array(bytes[0..<(bytes.count - extra)]))
            // 与未加密路径一致：交给解压器的记录不含尾部 extra 块。
            decrypted.append(Data(clear))
        }
        return MobiDRMResult(records: decrypted, method: method)
    }

    /// 是否识别为受 DRM 保护（未解密前无法读取正文）。
    public static func isProtected(_ palm: MobiPalmHeader) -> Bool {
        palm.encryptionRaw == 1 || palm.encryptionRaw == 2
    }

    // MARK: - V1

    private static func keyV1(recordZero: Data, header: MobiHeader) throws -> [UInt8] {
        let bytes = [UInt8](recordZero)
        guard bytes.count >= 24 else {
            throw MobiDRMError.invalid("Record 0 过短")
        }
        let offset: Int
        if header.version <= 1 {
            offset = 14
        } else {
            // MOBI header 从 Record0+16 开始，header_length 在 Record0+20。
            let length = readBE32(bytes, 20)
            offset = Int(length) + 16 + (header.version > 2 ? 12 : 0)
        }
        guard offset >= 0, offset + 16 <= bytes.count else {
            throw MobiDRMError.invalid("V1 密钥偏移越界")
        }
        var cipher = MobiPK1(key: readerKeyV1)
        return cipher.process(Array(bytes[offset..<(offset + 16)]))
    }

    // MARK: - V2

    private static func keyV2(recordZero: Data, header: MobiHeader,
                              pid: String) throws -> [UInt8] {
        guard pid.utf8.count == 10 else {
            throw MobiDRMError.invalid("PID 必须是 10 个字符")
        }
        let pidBytes = Array(pid.utf8)
        // PID 校验只作诊断：voucher cookie 能否解开才是权威，不因 checksum 差异拦截。
        if !validPID(pidBytes) {
            AgentLog.warning("MOBI DRM V2: PID checksum 不匹配，仍尝试 voucher 解密")
        }
        let bytes = [UInt8](recordZero)
        guard let drmOffset = readOptionalBE32(bytes, 16 + 152),
              let drmCount = readOptionalBE32(bytes, 16 + 156),
              drmCount > 0, drmCount <= 32,
              Int(drmOffset) + 48 <= bytes.count else {
            throw MobiDRMError.invalid("V2 DRM voucher 字段无效")
        }

        var pidKey = [UInt8](repeating: 0, count: 16)
        for index in 0..<8 { pidKey[index] = pidBytes[index] }
        var deviceCipher = MobiPK1(key: readerKeyV2)
        let deviceKey = deviceCipher.encrypt(pidKey)

        for voucherIndex in 0..<Int(drmCount) {
            let start = Int(drmOffset) + voucherIndex * 48
            guard start + 48 <= bytes.count else { break }
            let verification = readBE32(bytes, start)
            let checksum = bytes[start + 12]
            let cookie = Array(bytes[(start + 16)..<(start + 48)])

            for (tempKey, expectedType) in [(deviceKey, UInt32(1)), (readerKeyV2, UInt32(3))] {
                guard tempKey.reduce(0, { Int($0) + Int($1) }) & 0xFF == Int(checksum) else {
                    continue
                }
                var cookieCipher = MobiPK1(key: tempKey)
                let plain = cookieCipher.process(cookie)
                let flags = readBE32(plain, 4)
                guard flags & 0x1F == expectedType,
                      readBE32(plain, 0) == verification,
                      timeValid(plain) else { continue }
                return Array(plain[8..<24])
            }
        }
        throw MobiDRMError.invalid("没有匹配的 DRM voucher，请检查 PID")
    }

    private static func timeValid(_ cookie: [UInt8]) -> Bool {
        let from = readBE32(cookie, 28)
        let to = readBE32(cookie, 24)
        let minutes = UInt32(Date().timeIntervalSince1970 / 60)
        if from == 0, to == 0xFFFFFFFF { return true }
        if from != 0, minutes < from { return false }
        if to != 0xFFFFFFFF, minutes > to { return false }
        return true
    }

    /// libmobi mobi_drmpid_checksum：PID 末两位是 CRC32 取反后的映射字符。
    private static func validPID(_ pid: [UInt8]) -> Bool {
        let alphabet = Array("ABCDEFGHIJKLMNPQRSTUVWXYZ123456789".utf8)
        var crc: UInt32 = 0xFFFFFFFF
        for byte in pid[0..<8] { crc = crc32Byte(crc, byte) }
        // libmobi 写的是 ~m_crc32(...)：标准 crc32 已含最终取反，
        // 再取反恰好回到这里未取反的内部值，不能重复取反。
        crc ^= crc >> 16
        for index in 0..<2 {
            let value = UInt8(truncatingIfNeeded: crc)
            let length = alphabet.count   // libmobi map_length = 34
            let position = (Int(value / UInt8(length)) ^ Int(value % UInt8(length))) % length
            guard pid[8 + index] == alphabet[position] else { return false }
            crc >>= 8
        }
        return true
    }

    private static func crc32Byte(_ crc: UInt32, _ byte: UInt8) -> UInt32 {
        var value = crc ^ UInt32(byte)
        for _ in 0..<8 {
            value = (value >> 1) ^ (value & 1 != 0 ? 0xEDB88320 : 0)
        }
        return value
    }

    // MARK: - 工具

    /// 记录尾附加块：libmobi read.c:481-509 的倒序 varlen 解析。
    private static func extraSize(_ bytes: [UInt8], flags: UInt16,
                                  record: Int) throws -> Int {
        guard flags != 0, !bytes.isEmpty else { return 0 }
        var cursor = bytes.count - 1
        var extra = 0
        for bit in stride(from: 15, through: 1, by: -1)
        where flags & (UInt16(1) << UInt16(bit)) != 0 {
            var value: UInt32 = 0
            var shift: UInt32 = 0
            var count = 0
            var stopped = false
            while cursor >= 0, count < 4 {
                let byte = bytes[cursor]
                cursor -= 1
                value |= UInt32(byte & 0x7F) << shift
                shift += 7
                count += 1
                if byte & 0x80 != 0 { stopped = true; break }
            }
            guard stopped else {
                throw MobiDRMError.invalid(
                    "第 \(record) 条记录尾块 varlen 未终止，无法可靠剥离 extra")
            }
            extra += Int(value)
            cursor -= max(0, Int(value) - count)
        }
        if flags & 1 != 0, cursor >= 0 {
            extra += Int(bytes[cursor] & 0x3) + 1
        }
        return extra
    }

    private static func readBE32(_ bytes: [UInt8], _ offset: Int) -> UInt32 {
        guard offset >= 0, offset + 4 <= bytes.count else { return 0 }
        return (UInt32(bytes[offset]) << 24) | (UInt32(bytes[offset + 1]) << 16)
            | (UInt32(bytes[offset + 2]) << 8) | UInt32(bytes[offset + 3])
    }

    private static func readOptionalBE32(_ bytes: [UInt8], _ offset: Int) -> UInt32? {
        guard offset >= 0, offset + 4 <= bytes.count else { return nil }
        return readBE32(bytes, offset)
    }
}
