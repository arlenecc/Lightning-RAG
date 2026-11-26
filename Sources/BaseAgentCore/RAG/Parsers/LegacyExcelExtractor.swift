import Foundation

/// 同步解析循环无法被系统中断，只能在阶段边界和循环内主动观察取消信号。
private func checkExcelCancellation() throws {
    if Task.isCancelled { throw AgentError.cancelled }
}

/// Excel 97–2003 `.xls`（BIFF8）提取。
///
/// 只读取 `Workbook` 流的 BIFF8 记录：全局子流负责工作表位置、共享字符串、
/// 数字格式与 XF；每个工作表子流负责单元格。公式只反编译常见 token 子集，
/// 无法安全还原时保留缓存值并计数，避免把错误表达式写进 Markdown。
struct LegacyExcelExtractor {

    struct Output {
        let markdown: String
        let outline: [OutlineEntry]
        let metadata: [String: String]
    }

    func extract(url: URL) throws -> Output {
        do {
            return try extractInternal(url: url)
        } catch let error as AgentError {
            throw error
        } catch {
            throw AgentError.parse(
                "旧版 Excel 解析失败（\(url.lastPathComponent)）：\(error.localizedDescription)")
        }
    }

    // MARK: - 入口

    private func extractInternal(url: URL) throws -> Output {
        try checkExcelCancellation()
        let values = try url.resourceValues(forKeys: [.fileSizeKey])
        if let size = values.fileSize, size > LegacyOfficeLimits.maxFileSize {
            throw AgentError.parse(
                "Excel 文档超过 \(LegacyOfficeLimits.maxFileSize / 1024 / 1024)MB 上限"
                + "（\(url.lastPathComponent)）")
        }

        try checkExcelCancellation()
        let compound: OLE2CompoundFile
        do {
            compound = try OLE2CompoundFile.open(url: url)
        } catch {
            throw AgentError.parse(
                "旧版 Excel 解析失败（\(url.lastPathComponent)）：\(error.localizedDescription)")
        }

        guard compound.containsStream(named: "Workbook") else {
            if compound.containsStream(named: "Book") {
                throw AgentError.unsupported(
                    "仅支持 Excel 97–2003（BIFF8）工作簿，BIFF5 / Excel 95 格式暂不支持"
                    + "（\(url.lastPathComponent)）")
            }
            throw AgentError.parse("Excel 工作簿缺少 Workbook 流（\(url.lastPathComponent)）")
        }

        try checkExcelCancellation()
        let stream: Data
        do {
            stream = try compound.streamData(named: "Workbook")
        } catch {
            throw AgentError.parse(
                "Excel 工作簿流读取失败（\(url.lastPathComponent)）：\(error.localizedDescription)")
        }

        try checkExcelCancellation()
        let globals = try parseGlobals(stream, fileName: url.lastPathComponent)
        let formatter = ExcelNumberFormatter(formats: globals.formats,
                                             xfFormatIDs: globals.xfFormatIDs,
                                             date1904: globals.date1904)
        var context = ParseContext()
        var sections: [String] = []
        var outline: [OutlineEntry] = []
        var skippedSheets = 0

        for sheet in globals.sheets.prefix(LegacyOfficeLimits.maxExcelSheets) {
            try checkExcelCancellation()
            do {
                guard let buffer = try parseSheet(stream: stream, info: sheet,
                                                  sharedStrings: globals.sharedStrings,
                                                  formatter: formatter,
                                                  context: &context) else {
                    skippedSheets += 1
                    continue
                }
                guard let table = try buffer.markdownTable() else { continue }
                sections.append("## Sheet: \(sheet.name)\n\n\(table)")
                outline.append(OutlineEntry(level: 2, title: "Sheet: \(sheet.name)"))
            } catch AgentError.cancelled {
                throw AgentError.cancelled
            } catch {
                skippedSheets += 1
                AgentLog.warning(
                    "跳过无法解析的 Excel 工作表 \(sheet.name)：\(error.localizedDescription)")
            }
        }

        guard !sections.isEmpty else {
            throw AgentError.parse("Excel 工作簿未解析出正文（\(url.lastPathComponent)）")
        }

        var metadata: [String: String] = [
            "engine": "xls",
            "legacy_format": "biff8",
            "sheets": String(sections.count),
            "formula_unparsed": String(context.formulaUnparsed),
            "skipped_cells": String(context.skippedCells),
            "row_limit_hit": context.rowLimitHit ? "true" : "false",
            "column_limit_hit": context.columnLimitHit ? "true" : "false",
        ]
        let sheetLimitHit = globals.sheets.count > LegacyOfficeLimits.maxExcelSheets
        if sheetLimitHit { metadata["sheet_limit_hit"] = "true" }
        if skippedSheets > 0 {
            metadata["skipped_sheets"] = String(skippedSheets)
            // 入库层按 `failed_pages` 判断是否完整；损坏的工作表与失败页等价，
            // 必须让增量同步下次重试，而不是被 manifest 的 complete=true 永久跳过。
            metadata["failed_pages"] = String(skippedSheets)
        }

        var markdown = sections.joined(separator: "\n\n")
        let limited = LegacyOfficeLimits.limitText(markdown)
        if limited.truncated {
            markdown = limited.text
            metadata["truncated"] = "true"
            metadata["truncated_reason"] = "text_limit"
            AgentLog.warning(
                "Excel 正文超过 \(LegacyOfficeLimits.maxTextCharacters) 字符，已截断："
                + url.lastPathComponent)
        }
        if context.rowLimitHit || context.columnLimitHit {
            metadata["truncated"] = "true"
            if metadata["truncated_reason"] == nil {
                metadata["truncated_reason"] = "row_column_limit"
            }
        }
        if sheetLimitHit || skippedSheets > 0 {
            metadata["truncated"] = "true"
            if metadata["truncated_reason"] == nil {
                metadata["truncated_reason"] = "sheet_limit"
            }
        }

        return Output(markdown: markdown, outline: outline, metadata: metadata)
    }

    // MARK: - 全局子流

    private struct WorkbookGlobals {
        var date1904 = false
        var formats: [UInt16: String] = [:]
        var xfFormatIDs: [UInt16] = []
        var sheets: [SheetInfo] = []
        var sharedStrings: [String] = []
    }

    private struct SheetInfo {
        var name: String
        let offset: Int
    }

    private func parseGlobals(_ stream: Data, fileName: String) throws -> WorkbookGlobals {
        var reader = BIFFRecordReader(data: stream)
        guard let bof = try reader.next(), bof.type == 0x0809 else {
            throw AgentError.parse("Excel 工作簿缺少 BOF 记录（\(fileName)）")
        }
        guard bof.payload.count >= 4,
              let version = try? BIFFBytes.u16(bof.payload, 0),
              let substreamType = try? BIFFBytes.u16(bof.payload, 2) else {
            throw AgentError.parse("Excel 工作簿 BOF 记录不完整（\(fileName)）")
        }
        guard substreamType == 0x0005 else {
            throw AgentError.parse("Excel 工作簿 BOF 不是全局子流（\(fileName)）")
        }
        guard version == 0x0600 else {
            throw AgentError.unsupported(
                "仅支持 Excel 97–2003（BIFF8）工作簿，BIFF5 / Excel 95 格式暂不支持"
                + "（\(fileName)）")
        }

        var globals = WorkbookGlobals()
        var sstSegments: [Data] = []
        var collectingSST = false
        var recordIndex = 0

        while let record = try reader.next() {
            recordIndex += 1
            if recordIndex % 4096 == 0 { try checkExcelCancellation() }
            if collectingSST {
                if record.type == 0x003C {
                    sstSegments.append(record.payload)
                    continue
                }
                collectingSST = false
            }

            switch record.type {
            case 0x000A:
                if !sstSegments.isEmpty {
                    globals.sharedStrings = try Self.decodeSST(sstSegments)
                }
                return globals

            case 0x002F:
                throw AgentError.unsupported(
                    "Excel 工作簿已加密或受密码保护（\(fileName)）")

            case 0x0022:
                if let value = try? BIFFBytes.u16(record.payload, 0) {
                    globals.date1904 = value == 1
                }

            case 0x041E:
                guard record.payload.count >= 5,
                      let identifier = try? BIFFBytes.u16(record.payload, 0),
                      let decoded = try? Self.decodeUnicodeString(record.payload,
                                                                  at: 2,
                                                                  cchSize: 2) else {
                    break
                }
                globals.formats[identifier] = decoded.value

            case 0x00E0:
                // XF 记录 offset 2 是数字格式 id；索引顺序即单元格里的 xf。
                globals.xfFormatIDs.append(
                    (try? BIFFBytes.u16(record.payload, 2)) ?? 0)

            case 0x0085:
                if var sheet = try? Self.decodeBoundsheet(record.payload) {
                    if sheet.name.isEmpty {
                        sheet.name = "Sheet\(globals.sheets.count + 1)"
                    }
                    globals.sheets.append(sheet)
                }

            case 0x00FC:
                sstSegments = [record.payload]
                collectingSST = true

            default:
                break
            }
        }

        throw AgentError.parse("Excel 工作簿缺少 EOF 记录（\(fileName)）")
    }

    private static func decodeBoundsheet(_ payload: Data) throws -> SheetInfo? {
        guard payload.count >= 8 else { return nil }
        let offset = Int(try BIFFBytes.u32(payload, 0))
        // grbit 低字节是表类型：0 = 工作表，1 = 宏，2 = 图表，6 = VBA 模块。
        let sheetType = try BIFFBytes.u8(payload, 4)
        guard sheetType == 0x00 else { return nil }
        let cch = Int(try BIFFBytes.u8(payload, 6))
        let flags = try BIFFBytes.u8(payload, 7)
        let name = try decodeChars(payload, start: 8, count: cch,
                                   highByte: flags & 0x01 != 0)
        return SheetInfo(name: name, offset: offset)
    }

    private static func decodeSST(_ segments: [Data]) throws -> [String] {
        var reader = SSTReader(segments: segments)
        _ = try reader.readUInt32() // 引用总数，仅用于校验记录是否可读
        let uniqueCount = Int(try reader.readUInt32())
        guard uniqueCount >= 0, uniqueCount <= LegacyOfficeLimits.maxRecordCount else {
            throw BIFFError("共享字符串数量异常")
        }
        var strings: [String] = []
        strings.reserveCapacity(uniqueCount)
        for index in 0..<uniqueCount {
            if index % 4096 == 0 { try checkExcelCancellation() }
            strings.append(try reader.readString())
        }
        return strings
    }

    private static func decodeChars(_ data: Data, start: Int, count: Int,
                                    highByte: Bool) throws -> String {
        guard start >= 0, start <= data.count, count >= 0,
              count <= LegacyOfficeLimits.maxTextCharacters else {
            throw BIFFError("字符串长度异常")
        }
        var units: [UInt16] = []
        units.reserveCapacity(count)
        var offset = start
        if highByte {
            guard count <= (data.count - offset) / 2 else {
                throw BIFFError("字符串被截断")
            }
            for _ in 0..<count {
                units.append(try BIFFBytes.u16(data, offset))
                offset += 2
            }
        } else {
            guard count <= data.count - offset else {
                throw BIFFError("字符串被截断")
            }
            for _ in 0..<count {
                units.append(UInt16(try BIFFBytes.u8(data, offset)))
                offset += 1
            }
        }
        return String(decoding: units, as: UTF16.self)
    }

    /// 读取 BIFF8 的 XLUnicodeString / ShortXLUnicodeString（不支持跨记录）。
    fileprivate static func decodeUnicodeString(_ data: Data, at start: Int,
                                                cchSize: Int) throws -> (value: String,
                                                                         end: Int) {
        var offset = start
        let cch: Int
        if cchSize == 1 {
            cch = Int(try BIFFBytes.u8(data, offset))
            offset += 1
        } else {
            cch = Int(try BIFFBytes.u16(data, offset))
            offset += 2
        }
        let flags = try BIFFBytes.u8(data, offset)
        offset += 1
        let highByte = flags & 0x01 != 0
        var richRuns = 0
        var extendedBytes = 0
        if flags & 0x08 != 0 {
            richRuns = Int(try BIFFBytes.u16(data, offset))
            offset += 2
        }
        if flags & 0x04 != 0 {
            extendedBytes = Int(try BIFFBytes.u32(data, offset))
            offset += 4
        }

        let value = try decodeChars(data, start: offset, count: cch,
                                    highByte: highByte)
        offset += highByte ? cch * 2 : cch

        let metadataBytes = richRuns * 4 + extendedBytes
        guard metadataBytes >= 0, metadataBytes <= data.count - offset else {
            throw BIFFError("字符串附加数据被截断")
        }
        offset += metadataBytes
        return (value, offset)
    }

    // MARK: - 工作表

    private struct ParseContext {
        var formulaUnparsed = 0
        var formulaTokensUsed = 0
        var rowLimitHit = false
        var columnLimitHit = false
        /// 记录格式错误/索引越界而未能提取的单元格数，写入元数据避免静默丢值。
        var skippedCells = 0
    }

    private struct SheetBuffer {
        var rows: [Int: [Int: String]] = [:]
        var maxRow = -1
        var maxColumn = -1

        mutating func set(row: Int, column: Int, value: String,
                          context: inout ParseContext) {
            guard row >= 0, column >= 0 else { return }
            guard row < LegacyOfficeLimits.maxExcelRows else {
                context.rowLimitHit = true
                return
            }
            guard column < LegacyOfficeLimits.maxExcelColumns else {
                context.columnLimitHit = true
                return
            }
            guard !value.isEmpty else { return }
            rows[row, default: [:]][column] = value
            maxRow = max(maxRow, row)
            maxColumn = max(maxColumn, column)
        }

        func markdownTable() throws -> String? {
            guard maxRow >= 0, maxColumn >= 0 else { return nil }
            var rows: [[String]] = []
            rows.reserveCapacity(maxRow + 1)
            for row in 0...maxRow {
                if row % 1024 == 0 { try checkExcelCancellation() }
                let values = self.rows[row] ?? [:]
                var cells: [String] = []
                cells.reserveCapacity(maxColumn + 1)
                for column in 0...maxColumn {
                    cells.append(HTMLMarkdown.escapeTableCell(values[column] ?? ""))
                }
                rows.append(cells)
            }
            return HTMLMarkdown.renderMarkdownTable(rows)
        }
    }

    private func parseSheet(stream: Data, info: SheetInfo,
                            sharedStrings: [String],
                            formatter: ExcelNumberFormatter,
                            context: inout ParseContext) throws -> SheetBuffer? {
        guard info.offset >= 0, info.offset <= stream.count - 4 else {
            return nil
        }
        var reader = BIFFRecordReader(data: stream, offset: info.offset)
        guard let bof = try reader.next(),
              bof.type == 0x0809,
              let version = try? BIFFBytes.u16(bof.payload, 0), version == 0x0600,
              let substreamType = try? BIFFBytes.u16(bof.payload, 2),
              substreamType == 0x0010 else {
            return nil
        }

        var buffer = SheetBuffer()
        var pendingString: (row: Int, column: Int)?
        var recordIndex = 0

        while let record = try reader.next() {
            recordIndex += 1
            if recordIndex % 4096 == 0 { try checkExcelCancellation() }
            // STRING 记录必须紧跟在缓存结果为字符串的 FORMULA 之后。
            if record.type != 0x0207 { pendingString = nil }

            switch record.type {
            case 0x000A:
                return buffer

            case 0x00FD: // LABELSST
                guard record.payload.count >= 10,
                      let row = try? BIFFBytes.u16(record.payload, 0),
                      let column = try? BIFFBytes.u16(record.payload, 2),
                      let index = try? BIFFBytes.u32(record.payload, 6) else {
                    context.skippedCells += 1
                    break
                }
                let stringIndex = Int(index)
                guard stringIndex >= 0, stringIndex < sharedStrings.count else {
                    context.skippedCells += 1
                    break
                }
                buffer.set(row: Int(row), column: Int(column),
                           value: sharedStrings[stringIndex], context: &context)

            case 0x0204: // LABEL
                guard record.payload.count >= 8,
                      let row = try? BIFFBytes.u16(record.payload, 0),
                      let column = try? BIFFBytes.u16(record.payload, 2),
                      let decoded = try? Self.decodeUnicodeString(record.payload,
                                                                  at: 6,
                                                                  cchSize: 2) else {
                    context.skippedCells += 1
                    break
                }
                buffer.set(row: Int(row), column: Int(column),
                           value: decoded.value, context: &context)

            case 0x0203: // NUMBER
                guard record.payload.count >= 14,
                      let row = try? BIFFBytes.u16(record.payload, 0),
                      let column = try? BIFFBytes.u16(record.payload, 2),
                      let xf = try? BIFFBytes.u16(record.payload, 4),
                      let value = try? BIFFBytes.f64(record.payload, 6) else {
                    context.skippedCells += 1
                    break
                }
                buffer.set(row: Int(row), column: Int(column),
                           value: formatter.text(value, xf: xf), context: &context)

            case 0x027E: // RK
                guard record.payload.count >= 10,
                      let row = try? BIFFBytes.u16(record.payload, 0),
                      let column = try? BIFFBytes.u16(record.payload, 2),
                      let xf = try? BIFFBytes.u16(record.payload, 4),
                      let rk = try? BIFFBytes.u32(record.payload, 6) else {
                    context.skippedCells += 1
                    break
                }
                buffer.set(row: Int(row), column: Int(column),
                           value: formatter.text(Self.rkValue(rk), xf: xf),
                           context: &context)

            case 0x00BD: // MULRK
                guard record.payload.count >= 12,
                      let row = try? BIFFBytes.u16(record.payload, 0),
                      let firstColumn = try? BIFFBytes.u16(record.payload, 2) else {
                    break
                }
                let count = (record.payload.count - 6) / 6
                guard count > 0 else { break }
                for index in 0..<count {
                    let base = 4 + index * 6
                    guard let xf = try? BIFFBytes.u16(record.payload, base),
                          let rk = try? BIFFBytes.u32(record.payload, base + 2) else {
                        break
                    }
                    buffer.set(row: Int(row),
                               column: Int(firstColumn) + index,
                               value: formatter.text(Self.rkValue(rk), xf: xf),
                               context: &context)
                }

            case 0x0205: // BOOLERR
                guard record.payload.count >= 8,
                      let row = try? BIFFBytes.u16(record.payload, 0),
                      let column = try? BIFFBytes.u16(record.payload, 2),
                      let value = try? BIFFBytes.u8(record.payload, 6),
                      let isError = try? BIFFBytes.u8(record.payload, 7) else {
                    break
                }
                let text: String
                if isError != 0 {
                    text = Self.errorText(value)
                } else {
                    text = value != 0 ? "TRUE" : "FALSE"
                }
                buffer.set(row: Int(row), column: Int(column),
                           value: text, context: &context)

            case 0x0006: // FORMULA
                guard record.payload.count >= 22,
                      let row = try? BIFFBytes.u16(record.payload, 0),
                      let column = try? BIFFBytes.u16(record.payload, 2),
                      let xf = try? BIFFBytes.u16(record.payload, 4),
                      let cachedCount = try? BIFFBytes.u16(record.payload, 20) else {
                    break
                }
                let cce = Int(cachedCount)
                let result = Data(record.payload[6..<14])
                var display: String?
                var decoded = false

                if cce > 0, cce <= LegacyOfficeLimits.maxFormulaTokensPerCell,
                   cce <= record.payload.count - 22,
                   context.formulaTokensUsed <= LegacyOfficeLimits.maxFormulaTokensTotal - cce {
                    context.formulaTokensUsed += cce
                    let tokens = Data(record.payload[22..<(22 + cce)])
                    var decompiler = FormulaDecompiler(tokens: tokens)
                    if let formula = decompiler.decode() {
                        display = "=" + formula
                        decoded = true
                    }
                }

                if decoded {
                    buffer.set(row: Int(row), column: Int(column),
                               value: display ?? "", context: &context)
                } else {
                    context.formulaUnparsed += 1
                    switch formatter.cachedFormulaValue(result, xf: xf) {
                    case .text(let text):
                        buffer.set(row: Int(row), column: Int(column),
                                   value: text, context: &context)
                    case .pendingString:
                        pendingString = (Int(row), Int(column))
                    case .empty, .none:
                        break
                    }
                }

            case 0x0207: // STRING（FORMULA 的字符串缓存结果）
                guard let pending = pendingString else { break }
                pendingString = nil
                guard let decoded = try? Self.decodeUnicodeString(record.payload,
                                                                  at: 0,
                                                                  cchSize: 2) else {
                    break
                }
                buffer.set(row: pending.row, column: pending.column,
                           value: decoded.value, context: &context)

            default:
                break
            }
        }

        return buffer
    }

    private static func rkValue(_ rk: UInt32) -> Double {
        var value: Double
        if rk & 0x02 != 0 {
            value = Double(Int32(bitPattern: rk) >> 2)
        } else {
            let bits = UInt64(rk & 0xFFFF_FFFC) << 32
            value = Double(bitPattern: bits)
        }
        if rk & 0x01 != 0 { value /= 100 }
        return value
    }

    fileprivate static func errorText(_ code: UInt8) -> String {
        switch code {
        case 0x00: return "#NULL!"
        case 0x07: return "#DIV/0!"
        case 0x0F: return "#VALUE!"
        case 0x17: return "#REF!"
        case 0x1D: return "#NAME?"
        case 0x24: return "#NUM!"
        case 0x2A: return "#N/A"
        default: return "#ERROR"
        }
    }
}

// MARK: - BIFF8 记录读取

private struct BIFFRecordReader {
    let data: Data
    var offset: Int
    var recordCount = 0

    init(data: Data, offset: Int = 0) {
        self.data = data
        self.offset = offset
    }

    mutating func next() throws -> (type: UInt16, payload: Data)? {
        guard offset < data.count else { return nil }
        guard offset <= data.count - 4 else {
            throw BIFFError("BIFF 记录头被截断")
        }
        let type = try BIFFBytes.u16(data, offset)
        let length = Int(try BIFFBytes.u16(data, offset + 2))
        offset += 4
        guard length <= data.count - offset else {
            throw BIFFError("BIFF 记录数据被截断")
        }
        let start = data.index(data.startIndex, offsetBy: offset)
        let end = data.index(start, offsetBy: length)
        let payload = Data(data[start..<end])
        offset += length

        recordCount += 1
        guard recordCount <= LegacyOfficeLimits.maxRecordCount else {
            throw BIFFError("BIFF 记录数超过上限")
        }
        return (type, payload)
    }
}

private struct BIFFError: LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? { message }
}

private enum BIFFBytes {
    static func u8(_ data: Data, _ offset: Int) throws -> UInt8 {
        guard offset >= 0, offset < data.count else {
            throw BIFFError("记录被截断")
        }
        return data[data.startIndex + offset]
    }

    static func u16(_ data: Data, _ offset: Int) throws -> UInt16 {
        let low = try u8(data, offset)
        let high = try u8(data, offset + 1)
        return UInt16(low) | (UInt16(high) << 8)
    }

    static func u32(_ data: Data, _ offset: Int) throws -> UInt32 {
        var value: UInt32 = 0
        for index in 0..<4 {
            value |= UInt32(try u8(data, offset + index)) << UInt32(index * 8)
        }
        return value
    }

    static func f64(_ data: Data, _ offset: Int) throws -> Double {
        var bits: UInt64 = 0
        for index in 0..<8 {
            bits |= UInt64(try u8(data, offset + index)) << UInt64(index * 8)
        }
        return Double(bitPattern: bits)
    }
}

// MARK: - 共享字符串

/// 跨 SST / CONTINUE 读取字符串。字符数据跨记录时，续段首字节是
/// 新的编码标志；字符串边界或富文本/扩展数据跨记录时没有该标志字节。
private struct SSTReader {
    let segments: [Data]
    var segmentIndex = 0
    var offset = 0

    mutating func readUInt16() throws -> UInt16 {
        let low = try readRawByte()
        let high = try readRawByte()
        return UInt16(low) | (UInt16(high) << 8)
    }

    mutating func readUInt32() throws -> UInt32 {
        var value: UInt32 = 0
        for index in 0..<4 {
            value |= UInt32(try readRawByte()) << UInt32(index * 8)
        }
        return value
    }

    mutating func readRawByte() throws -> UInt8 {
        while segmentIndex < segments.count {
            let segment = segments[segmentIndex]
            if offset < segment.count {
                let byte = segment[segment.startIndex + offset]
                offset += 1
                return byte
            }
            segmentIndex += 1
            offset = 0
        }
        throw BIFFError("共享字符串表被截断")
    }

    mutating func skipRaw(_ count: Int) throws {
        guard count >= 0 else { throw BIFFError("共享字符串附加数据长度异常") }
        var remaining = count
        while remaining > 0 {
            while segmentIndex < segments.count,
                  offset >= segments[segmentIndex].count {
                segmentIndex += 1
                offset = 0
            }
            guard segmentIndex < segments.count else {
                throw BIFFError("共享字符串附加数据被截断")
            }
            let available = segments[segmentIndex].count - offset
            let take = min(available, remaining)
            offset += take
            remaining -= take
        }
    }

    mutating func readString() throws -> String {
        let cch = Int(try readUInt16())
        let flags = try readRawByte()
        var highByte = flags & 0x01 != 0
        let hasRichRuns = flags & 0x08 != 0
        let hasExtended = flags & 0x04 != 0
        var richRuns = 0
        var extendedBytes = 0
        if hasRichRuns { richRuns = Int(try readUInt16()) }
        if hasExtended { extendedBytes = Int(try readUInt32()) }

        guard cch <= LegacyOfficeLimits.maxTextCharacters else {
            throw BIFFError("共享字符串过长")
        }
        var units: [UInt16] = []
        units.reserveCapacity(cch)
        var remaining = cch

        while remaining > 0 {
            if segmentIndex >= segments.count {
                throw BIFFError("共享字符串字符被截断")
            }
            if offset >= segments[segmentIndex].count {
                segmentIndex += 1
                guard segmentIndex < segments.count else {
                    throw BIFFError("共享字符串字符被截断")
                }
                let continuation = segments[segmentIndex]
                guard continuation.count > 0 else {
                    throw BIFFError("共享字符串续段为空")
                }
                highByte = continuation[continuation.startIndex] & 0x01 != 0
                offset = 1
                continue
            }

            let segment = segments[segmentIndex]
            if highByte {
                guard offset + 1 < segment.count else {
                    throw BIFFError("共享字符串字符被截断")
                }
                let low = segment[segment.startIndex + offset]
                let high = segment[segment.startIndex + offset + 1]
                units.append(UInt16(low) | (UInt16(high) << 8))
                offset += 2
            } else {
                units.append(UInt16(segment[segment.startIndex + offset]))
                offset += 1
            }
            remaining -= 1
        }

        if richRuns > 0 || extendedBytes > 0 {
            try skipRaw(richRuns * 4 + extendedBytes)
        }
        return String(decoding: units, as: UTF16.self)
    }
}

// MARK: - 数字格式

private struct ExcelNumberFormatter {
    enum CachedFormulaValue {
        case text(String)
        case pendingString
        case empty
    }

    private enum NumberKind {
        case general
        case number(decimals: Int, thousands: Bool, scientific: Bool, currency: String?)
        case percent(decimals: Int)
        case dateTime(hasDate: Bool, hasTime: Bool)
    }

    let formats: [UInt16: String]
    let xfFormatIDs: [UInt16]
    let date1904: Bool

    func text(_ value: Double, xf: UInt16) -> String {
        guard value.isFinite else { return "#NUM!" }
        let formatID = Int(xf) < xfFormatIDs.count ? xfFormatIDs[Int(xf)] : 0
        let code = formats[formatID]
            ?? Self.builtinFormats[Int(formatID)]
            ?? "General"
        switch Self.kind(of: code) {
        case .general:
            return Self.generalText(value)
        case .number(let decimals, let thousands, let scientific, let currency):
            let body: String
            if scientific {
                body = String(format: ".%\(min(max(decimals, 0), 15))e",
                              locale: Locale(identifier: "en_US_POSIX"), value)
            } else {
                body = Self.fixedText(value, decimals: decimals, thousands: thousands)
            }
            return (currency.map { $0 + " " } ?? "") + body
        case .percent(let decimals):
            return Self.fixedText(value * 100, decimals: decimals) + "%"
        case .dateTime(let hasDate, let hasTime):
            return dateText(value, hasDate: hasDate, hasTime: hasTime)
        }
    }

    func cachedFormulaValue(_ result: Data, xf: UInt16) -> CachedFormulaValue? {
        guard result.count >= 8 else { return nil }
        if let marker = try? BIFFBytes.u16(result, 0), marker == 0xFFFF {
            guard let type = try? BIFFBytes.u8(result, 2) else { return nil }
            switch type {
            case 0x00:
                return .pendingString
            case 0x01:
                let value = (try? BIFFBytes.u8(result, 4)) ?? 0
                return .text(value != 0 ? "TRUE" : "FALSE")
            case 0x02:
                let code = (try? BIFFBytes.u8(result, 4)) ?? 0
                return .text(LegacyExcelExtractor.errorText(code))
            case 0x03:
                return .empty
            default:
                return nil
            }
        }
        guard let value = try? BIFFBytes.f64(result, 0) else { return nil }
        return .text(text(value, xf: xf))
    }

    static func generalText(_ value: Double) -> String {
        guard value.isFinite else { return "#NUM!" }
        if value == 0 { return "0" }
        if abs(value) <= 9.007_199_254_740_992e15, value.rounded() == value {
            return String(Int64(value))
        }
        return String(format: "%.10g",
                      locale: Locale(identifier: "en_US_POSIX"), value)
    }

    private static func fixedText(_ value: Double, decimals: Int,
                                  thousands: Bool = false) -> String {
        let digits = min(max(decimals, 0), 15)
        guard thousands else {
            return String(format: "%.\(digits)f",
                          locale: Locale(identifier: "en_US_POSIX"), value)
        }
        let formatter = NumberFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.numberStyle = .decimal
        formatter.minimumFractionDigits = digits
        formatter.maximumFractionDigits = digits
        formatter.usesGroupingSeparator = true
        return formatter.string(from: NSNumber(value: value))
            ?? String(format: "%.\(digits)f", locale: Locale(identifier: "en_US_POSIX"), value)
    }

    private static func kind(of code: String) -> NumberKind {
        let sanitized = sanitize(code)
        let lower = sanitized.lowercased()
        let elapsedTime = code.lowercased().range(
            of: "\\[(h|m|s)\\]", options: .regularExpression) != nil
        let hasTime = elapsedTime || lower.contains("h") || lower.contains("s")
        let hasDate = lower.contains("y") || lower.contains("d")

        if sanitized.contains("%") {
            return .percent(decimals: decimalPlaces(in: sanitized) ?? 0)
        }
        if hasDate || hasTime {
            return .dateTime(hasDate: hasDate || !hasTime, hasTime: hasTime)
        }
        if sanitized == "General" {
            return .general
        }
        if sanitized.contains("0") || sanitized.contains("#") || sanitized.contains(".") {
            let decimals = decimalPlaces(in: sanitized) ?? 0
            let thousands = sanitized.contains(",")
            let scientific = sanitized.range(of: #"e[+-]?\d"#,
                                             options: [.regularExpression, .caseInsensitive]) != nil
            return .number(decimals: decimals, thousands: thousands,
                           scientific: scientific, currency: currencySymbol(in: code))
        }
        return .general
    }

    /// 从 `[$¥-804]` / `"$"#,##0` 取货币符号；方括号 locale 后缀要剥掉。
    private static func currencySymbol(in code: String) -> String? {
        if let match = code.range(of: #"\[\$([^-\]]+)"#,
                                  options: .regularExpression) {
            let start = code.index(match.lowerBound, offsetBy: 2)
            let symbol = String(code[start..<match.upperBound])
                .trimmingCharacters(in: .whitespaces)
            if !symbol.isEmpty { return symbol }
        }
        if let match = code.range(of: #"^"([^"]+)""#, options: .regularExpression) {
            let start = code.index(match.lowerBound, offsetBy: 1)
            let symbol = String(code[start..<code.index(match.upperBound, offsetBy: -1)])
            if symbol.contains(where: { !$0.isLetter && !$0.isNumber }) { return symbol }
        }
        return nil
    }

    /// 去掉引号字面量、反斜杠转义与颜色/条件/货币等方括号。
    private static func sanitize(_ code: String) -> String {
        var result = ""
        var inQuotes = false
        var escaped = false
        var inBracket = false
        for character in code {
            if escaped {
                result.append(character)
                escaped = false
                continue
            }
            if character == "\\" {
                escaped = true
                continue
            }
            if character == "\"" {
                inQuotes.toggle()
                continue
            }
            if inQuotes { continue }
            if character == "[" {
                inBracket = true
                continue
            }
            if character == "]" {
                inBracket = false
                continue
            }
            if inBracket { continue }
            result.append(character)
        }
        return result
    }

    private static func decimalPlaces(in code: String) -> Int? {
        guard let dot = code.firstIndex(of: ".") else { return nil }
        var count = 0
        for character in code[code.index(after: dot)...] {
            if character == "0" || character == "#" { count += 1 } else { break }
        }
        return count
    }

    private func dateText(_ serial: Double, hasDate: Bool, hasTime: Bool) -> String {
        guard serial >= 0, serial <= 3_000_000 else {
            return Self.generalText(serial)
        }
        var days = Int(floor(serial))
        let fraction = serial - floor(serial)
        // 1900 日期系统把 1900 年当作闰年，序列号 60 对应虚构的 02-29，
        // Foundation 无此日期，直接返回 Excel 的显示文本。
        if !date1904, days == 60 {
            let seconds = Int((fraction * 86_400).rounded())
            let hours = seconds / 3600, minutes = (seconds % 3600) / 60, secs = seconds % 60
            if hasTime {
                return hasDate
                    ? String(format: "1900-02-29 %02d:%02d:%02d", hours, minutes, secs)
                    : String(format: "%02d:%02d:%02d", hours, minutes, secs)
            }
            return "1900-02-29"
        }
        // 61 及之后比 1899-12-31 基准少一天（60 号虚构日偏移抵消）。
        if !date1904, days >= 61 { days -= 1 }

        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        let components = date1904
            ? DateComponents(year: 1904, month: 1, day: 1)
            : DateComponents(year: 1899, month: 12, day: 31)
        guard let base = calendar.date(from: components) else {
            return Self.generalText(serial)
        }
        let seconds = Int((fraction * 86_400).rounded())
        guard let date = calendar.date(byAdding: .second,
                                       value: days * 86_400 + seconds,
                                       to: base) else {
            return Self.generalText(serial)
        }
        let parts = calendar.dateComponents([.year, .month, .day, .hour, .minute, .second],
                                           from: date)
        let year = parts.year ?? 0
        let month = parts.month ?? 0
        let day = parts.day ?? 0
        let hour = parts.hour ?? 0
        let minute = parts.minute ?? 0
        let second = parts.second ?? 0

        if hasDate && hasTime {
            return String(format: "%04d-%02d-%02d %02d:%02d:%02d",
                          year, month, day, hour, minute, second)
        }
        if hasTime {
            return String(format: "%02d:%02d:%02d", hour, minute, second)
        }
        return String(format: "%04d-%02d-%02d", year, month, day)
    }

    private static let builtinFormats: [Int: String] = [
        0: "General",
        1: "0",
        2: "0.00",
        3: "#,##0",
        4: "#,##0.00",
        9: "0%",
        10: "0.00%",
        11: "0.00E+00",
        12: "# ?/?",
        13: "# ??/??",
        14: "m/d/yy",
        15: "d-mmm-yy",
        16: "d-mmm",
        17: "mmm-yy",
        18: "h:mm AM/PM",
        19: "h:mm:ss AM/PM",
        20: "h:mm",
        21: "h:mm:ss",
        22: "m/d/yy h:mm",
        37: "#,##0 ;(#,##0)",
        38: "#,##0 ;[Red](#,##0)",
        39: "#,##0.00;(#,##0.00)",
        40: "#,##0.00;[Red](#,##0.00)",
        45: "mm:ss",
        46: "[h]:mm:ss",
        47: "mmss.0",
        48: "##0.0E+0",
        49: "@",
    ]
}

// MARK: - 公式反编译

private struct FormulaDecompiler {
    let tokens: Data
    var offset = 0
    var stack: [String] = []

    init(tokens: Data) {
        self.tokens = tokens
    }

    mutating func decode() -> String? {
        while offset < tokens.count {
            guard let token = try? BIFFBytes.u8(tokens, offset) else {
                return nil
            }
            offset += 1
            guard step(token) else { return nil }
            guard stack.count <= 64 else { return nil }
        }
        guard stack.count == 1 else { return nil }
        return stack[0]
    }

    private mutating func step(_ token: UInt8) -> Bool {
        switch token {
        case 0x1F: // ptgNum
            guard let value = try? BIFFBytes.f64(tokens, offset) else { return false }
            offset += 8
            stack.append(ExcelNumberFormatter.generalText(value))
            return true

        case 0x1E: // ptgInt
            guard let value = try? BIFFBytes.u16(tokens, offset) else { return false }
            offset += 2
            stack.append(String(value))
            return true

        case 0x17: // ptgStr
            guard let decoded = try? LegacyExcelExtractor.decodeUnicodeString(
                tokens, at: offset, cchSize: 1) else { return false }
            offset = decoded.end
            let escaped = decoded.value.replacingOccurrences(of: "\"", with: "\"\"")
            stack.append("\"\(escaped)\"")
            return true

        case 0x1D: // ptgBool
            guard let value = try? BIFFBytes.u8(tokens, offset) else { return false }
            offset += 1
            stack.append(value != 0 ? "TRUE" : "FALSE")
            return true

        case 0x1C: // ptgErr
            guard let code = try? BIFFBytes.u8(tokens, offset) else { return false }
            offset += 1
            stack.append(LegacyExcelExtractor.errorText(code))
            return true

        case 0x16: // ptgMissArg
            stack.append("")
            return true

        case 0x24, 0x44, 0x64: // ptgRef
            guard let row = try? BIFFBytes.u16(tokens, offset),
                  let column = try? BIFFBytes.u16(tokens, offset + 2) else {
                return false
            }
            offset += 4
            stack.append(Self.cellReference(row: Int(row),
                                            column: Int(column & 0x3FFF)))
            return true

        case 0x25, 0x45, 0x65: // ptgArea
            guard let firstRow = try? BIFFBytes.u16(tokens, offset),
                  let lastRow = try? BIFFBytes.u16(tokens, offset + 2),
                  let firstColumn = try? BIFFBytes.u16(tokens, offset + 4),
                  let lastColumn = try? BIFFBytes.u16(tokens, offset + 6) else {
                return false
            }
            offset += 8
            stack.append(Self.cellReference(row: Int(firstRow),
                                            column: Int(firstColumn & 0x3FFF))
                + ":" + Self.cellReference(row: Int(lastRow),
                                           column: Int(lastColumn & 0x3FFF)))
            return true

        case 0x21, 0x41, 0x61: // ptgFunc
            guard let index = try? BIFFBytes.u16(tokens, offset) else { return false }
            offset += 2
            guard let name = Self.functionNames[index],
                  let arity = Self.fixedArities[index],
                  stack.count >= arity else {
                return false
            }
            let arguments = stack.suffix(arity)
            stack.removeLast(arity)
            stack.append(name + "(" + arguments.joined(separator: ",") + ")")
            return true

        case 0x22, 0x42, 0x62: // ptgFuncVar
            guard let argumentCount = try? BIFFBytes.u8(tokens, offset),
                  let index = try? BIFFBytes.u16(tokens, offset + 1) else {
                return false
            }
            offset += 3
            guard let name = Self.functionNames[index],
                  Int(argumentCount) <= stack.count,
                  Int(argumentCount) <= 64 else {
                return false
            }
            let count = Int(argumentCount)
            let arguments = count > 0 ? Array(stack.suffix(count)) : []
            if count > 0 { stack.removeLast(count) }
            stack.append(name + "(" + arguments.joined(separator: ",") + ")")
            return true

        case 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
             0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E:
            guard stack.count >= 2 else { return false }
            let rhs = stack.removeLast()
            let lhs = stack.removeLast()
            let symbol: String
            switch token {
            case 0x03: symbol = "+"
            case 0x04: symbol = "-"
            case 0x05: symbol = "*"
            case 0x06: symbol = "/"
            case 0x07: symbol = "^"
            case 0x08: symbol = "&"
            case 0x09: symbol = "<"
            case 0x0A: symbol = "<="
            case 0x0B: symbol = "="
            case 0x0C: symbol = ">="
            case 0x0D: symbol = ">"
            default: symbol = "<>"
            }
            stack.append(lhs + symbol + rhs)
            return true

        case 0x12: // ptgUplus
            return stack.count >= 1

        case 0x13: // ptgUminus
            guard let value = stack.popLast() else { return false }
            stack.append("-" + value)
            return true

        case 0x14: // ptgPercent
            guard let value = stack.popLast() else { return false }
            stack.append(value + "%")
            return true

        case 0x15: // ptgParen
            guard let value = stack.popLast() else { return false }
            stack.append("(" + value + ")")
            return true

        default:
            return false
        }
    }

    private static func cellReference(row: Int, column: Int) -> String {
        var value = max(0, column) + 1
        var letters = ""
        while value > 0 {
            let remainder = (value - 1) % 26
            letters = String(UnicodeScalar(65 + remainder)!) + letters
            value = (value - 1) / 26
        }
        return letters + String(max(0, row) + 1)
    }

    private static let functionNames: [UInt16: String] = [
        0: "COUNT", 1: "IF", 2: "ISNA", 3: "ISERROR", 4: "SUM", 5: "AVERAGE",
        6: "MIN", 7: "MAX", 8: "ROW", 9: "COLUMN", 10: "NA", 12: "STDEV",
        15: "SIN", 16: "COS", 17: "TAN", 18: "ATAN", 19: "PI", 20: "SQRT",
        21: "EXP", 22: "LN", 23: "LOG10", 24: "ABS", 25: "INT", 26: "SIGN",
        27: "ROUND", 28: "LOOKUP", 29: "INDEX", 30: "REPT", 31: "MID",
        32: "LEN", 33: "VALUE", 34: "TRUE", 35: "FALSE", 36: "AND", 37: "OR",
        38: "NOT", 39: "MOD", 65: "DATE", 66: "TIME", 67: "DAY", 68: "MONTH",
        69: "YEAR", 70: "WEEKDAY", 71: "HOUR", 72: "MINUTE", 73: "SECOND",
        74: "NOW", 75: "AREAS", 76: "ROWS", 77: "COLUMNS", 97: "ATAN2",
        98: "ASIN", 99: "ACOS", 100: "CHOOSE", 101: "HLOOKUP", 102: "VLOOKUP",
        109: "LOG", 112: "UPPER", 113: "LOWER", 114: "PROPER", 115: "LEFT",
        116: "RIGHT", 117: "EXACT", 118: "TRIM", 119: "REPLACE",
        120: "SUBSTITUTE", 121: "CODE", 124: "FIND", 125: "CELL",
        126: "ISERR", 127: "ISTEXT", 128: "ISNUMBER", 129: "ISBLANK",
        130: "T", 131: "N", 132: "DATEVALUE", 133: "TIMEVALUE",
        169: "COUNTA", 336: "CONCATENATE",
    ]

    private static let fixedArities: [UInt16: Int] = [
        24: 1, 25: 1, 26: 1, 27: 2, 30: 2, 31: 3, 32: 1, 33: 1,
        65: 3, 66: 3, 67: 1, 68: 1, 69: 1, 70: 2, 71: 1, 72: 1, 73: 1,
        97: 2, 98: 1, 99: 1, 112: 1, 113: 1, 114: 1, 115: 2, 116: 2,
        118: 1, 119: 4, 120: 3, 121: 1, 124: 3, 132: 1, 133: 1,
    ]
}
