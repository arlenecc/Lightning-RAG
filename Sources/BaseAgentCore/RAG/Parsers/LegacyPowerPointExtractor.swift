import Foundation

/// PowerPoint 97–2003 `.ppt` 提取。
///
/// 只读取 `PowerPoint Document` 流的记录树：`SlideListWithText` 里的
/// `SlidePersistAtom` 决定幻灯片顺序，`TextHeaderAtom` 区分标题/正文/备注，
/// `TextCharsAtom` / `TextBytesAtom` 提供文本。列表缺失时回退到文件中的
/// `Slide` 容器顺序。所有偏移、递归深度和记录数量都有明确上界。
struct LegacyPowerPointExtractor {

    struct Output {
        let markdown: String
        let outline: [OutlineEntry]
        let metadata: [String: String]
    }

    // MARK: - 记录类型

    private enum RecordType {
        static let document: UInt16 = 0x03E8
        static let slide: UInt16 = 0x03EE
        static let slideAtom: UInt16 = 0x03EF
        static let notes: UInt16 = 0x03F0
        static let notesAtom: UInt16 = 0x03F1
        static let slidePersistAtom: UInt16 = 0x03F3
        static let textHeaderAtom: UInt16 = 0x0F9F
        static let textCharsAtom: UInt16 = 0x0FA0
        static let textBytesAtom: UInt16 = 0x0FA8
        static let slideListWithText: UInt16 = 0x0FF0
        static let containerVersion: UInt8 = 0x0F
    }

    /// TextHeaderAtom 的标题类型（0 = 标题，4 = 其他标题占位符）。
    private static let titleTextTypes: Set<UInt32> = [0, 4]
    /// TextHeaderAtom 的备注类型。
    private static let notesTextType: UInt32 = 2

    private struct TextBlock {
        let headerType: UInt32?
        let paragraphs: [String]
        let atomCount: Int
    }

    private struct SlideContainer {
        let slideId: UInt32
        let blocks: [TextBlock]
    }

    private struct NotesContainer {
        let slideIdRef: UInt32
        let blocks: [TextBlock]
    }

    private struct SlideTextGroup {
        let slideId: UInt32
        let blocks: [TextBlock]
    }

    private struct ScanState {
        var recordCount = 0
        var slides: [SlideContainer] = []
        var notes: [NotesContainer] = []
        var slideGroups: [SlideTextGroup] = []
    }

    private struct RecordHeader {
        let version: UInt8
        let instance: UInt16
        let type: UInt16
        let payload: Range<Int>
    }

    // MARK: - 入口

    func extract(url: URL) throws -> Output {
        do {
            return try extractInternal(url: url)
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as AgentError {
            throw error
        } catch {
            throw AgentError.parse(
                "旧版 PowerPoint 解析失败（\(url.lastPathComponent)）：\(error.localizedDescription)")
        }
    }

    private func extractInternal(url: URL) throws -> Output {
        let fileName = url.lastPathComponent
        let values = try url.resourceValues(forKeys: [.fileSizeKey])
        if let size = values.fileSize, size > LegacyOfficeLimits.maxFileSize {
            throw AgentError.parse(
                "PowerPoint 文档超过 \(LegacyOfficeLimits.maxFileSize / 1024 / 1024)MB 上限"
                + "（\(fileName)）")
        }

        let compound: OLE2CompoundFile
        do {
            compound = try OLE2CompoundFile.open(url: url)
        } catch {
            throw AgentError.parse(
                "旧版 PowerPoint 解析失败（\(fileName)）：\(error.localizedDescription)")
        }

        if compound.containsStream(named: "EncryptionInfo")
            || compound.containsStream(named: "EncryptedSummary") {
            throw AgentError.unsupported(
                "PowerPoint 文档已加密或受密码保护（\(fileName)）")
        }
        guard compound.containsStream(named: "PowerPoint Document") else {
            throw AgentError.parse(
                "不是有效的 PowerPoint 97–2003 文件（\(fileName)）："
                + "缺少 PowerPoint Document 流")
        }

        let stream: Data
        do {
            stream = try compound.streamData(named: "PowerPoint Document")
        } catch {
            throw AgentError.parse(
                "PowerPoint 文档流读取失败（\(fileName)）：\(error.localizedDescription)")
        }
        guard stream.count >= 8 else {
            throw AgentError.parse("PowerPoint 记录头不完整（\(fileName)）")
        }
        guard let firstHeader = Self.uint16(stream, at: 0),
              let firstType = Self.uint16(stream, at: 2),
              firstType == RecordType.document,
              UInt8(firstHeader & 0x000F) == RecordType.containerVersion else {
            throw AgentError.unsupported(
                "仅支持 PowerPoint 97–2003（PPT）格式，"
                + "PowerPoint 95 及更早版本暂不支持（\(fileName)）")
        }

        var state = ScanState()
        try scan(stream, range: 0..<stream.count, depth: 0,
                 state: &state, fileName: fileName)

        let allSlides = selectSlides(from: state)
        let slideLimitHit = allSlides.count > LegacyOfficeLimits.maxSlides
        let slides = Array(allSlides.prefix(LegacyOfficeLimits.maxSlides))
        let notes = notesForSlides(slides, notes: state.notes)

        var sections: [String] = []
        var outline: [OutlineEntry] = []
        var notesSlides = 0
        var textAtomCount = 0
        var hasContent = false

        for (index, slide) in slides.enumerated() {
            let rendered = renderSlide(slide, number: index + 1, notes: notes[index])
            sections.append(rendered.section)
            outline.append(OutlineEntry(level: 2, title: rendered.outlineTitle))
            if rendered.hasNotes { notesSlides += 1 }
            if rendered.hasContent { hasContent = true }
            textAtomCount += slide.blocks.reduce(0) { $0 + $1.atomCount }
            textAtomCount += notes[index].reduce(0) { $0 + $1.atomCount }
        }

        guard hasContent else {
            throw AgentError.parse("PowerPoint 文档未解析出正文（\(fileName)）")
        }

        var metadata: [String: String] = [
            "engine": "ppt",
            "legacy_format": "ppt-97",
            "slides": String(sections.count),
            "text_atoms": String(textAtomCount),
        ]
        if notesSlides > 0 { metadata["notes_slides"] = String(notesSlides) }
        if slideLimitHit {
            metadata["slide_limit_hit"] = "true"
            metadata["truncated"] = "true"
            metadata["truncated_reason"] = "slide_limit"
            AgentLog.warning(
                "PowerPoint 幻灯片超过 \(LegacyOfficeLimits.maxSlides) 页，已截断：\(fileName)")
        }

        var markdown = sections.joined(separator: "\n\n")
        let limited = LegacyOfficeLimits.limitText(markdown)
        if limited.truncated {
            markdown = limited.text
            metadata["truncated"] = "true"
            if metadata["truncated_reason"] == nil {
                metadata["truncated_reason"] = "text_limit"
            }
            AgentLog.warning(
                "PowerPoint 正文超过 \(LegacyOfficeLimits.maxTextCharacters) 字符，已截断："
                + fileName)
        }

        return Output(markdown: markdown, outline: outline, metadata: metadata)
    }

    // MARK: - 记录树遍历

    private func scan(_ data: Data, range: Range<Int>, depth: Int,
                      state: inout ScanState, fileName: String) throws {
        guard depth <= LegacyOfficeLimits.maxRecordDepth else {
            throw depthLimitError(fileName)
        }
        var offset = range.lowerBound
        while offset < range.upperBound {
            let header = try readHeader(data, at: offset, end: range.upperBound,
                                         fileName: fileName)
            try countRecord(&state, fileName: fileName)

            if header.version == RecordType.containerVersion {
                switch header.type {
                case RecordType.document:
                    try scan(data, range: header.payload, depth: depth + 1,
                             state: &state, fileName: fileName)
                case RecordType.slide:
                    try processSlide(data, payload: header.payload, depth: depth + 1,
                                     state: &state, fileName: fileName)
                case RecordType.notes:
                    try processNotes(data, payload: header.payload, depth: depth + 1,
                                     state: &state, fileName: fileName)
                case RecordType.slideListWithText where header.instance == 0:
                    try processSlideList(data, payload: header.payload, depth: depth + 1,
                                         state: &state, fileName: fileName)
                default:
                    try scan(data, range: header.payload, depth: depth + 1,
                             state: &state, fileName: fileName)
                }
            }
            offset = header.payload.upperBound
        }
    }

    /// 遍历一个容器内的文本记录；`onAtom` 用于收集 SlideAtom / NotesAtom。
    private func walkText(_ data: Data, range: Range<Int>, depth: Int,
                          state: inout ScanState, fileName: String,
                          collector: inout TextCollector,
                          onAtom: (UInt16, Data) -> Void = { _, _ in }) throws {
        guard depth <= LegacyOfficeLimits.maxRecordDepth else {
            throw depthLimitError(fileName)
        }
        var offset = range.lowerBound
        while offset < range.upperBound {
            let header = try readHeader(data, at: offset, end: range.upperBound,
                                         fileName: fileName)
            try countRecord(&state, fileName: fileName)

            if header.version == RecordType.containerVersion {
                try walkText(data, range: header.payload, depth: depth + 1,
                             state: &state, fileName: fileName,
                             collector: &collector, onAtom: onAtom)
            } else {
                let payload = data[header.payload]
                switch header.type {
                case RecordType.textHeaderAtom:
                    if let type = Self.uint32(payload, at: 0) {
                        collector.begin(type)
                    }
                case RecordType.textCharsAtom:
                    collector.append(Self.decodeChars(payload))
                case RecordType.textBytesAtom:
                    collector.append(Self.decodeBytes(payload))
                default:
                    onAtom(header.type, payload)
                }
            }
            offset = header.payload.upperBound
        }
    }

    private func processSlide(_ data: Data, payload: Range<Int>, depth: Int,
                              state: inout ScanState, fileName: String) throws {
        var slideId: UInt32 = 0
        var collector = TextCollector()
        try walkText(data, range: payload, depth: depth, state: &state,
                     fileName: fileName, collector: &collector) { type, atom in
            if type == RecordType.slideAtom, let value = Self.uint32(atom, at: 8) {
                slideId = value
            }
        }
        state.slides.append(SlideContainer(slideId: slideId,
                                           blocks: collector.finish()))
    }

    private func processNotes(_ data: Data, payload: Range<Int>, depth: Int,
                              state: inout ScanState, fileName: String) throws {
        var slideIdRef: UInt32 = 0
        var collector = TextCollector()
        try walkText(data, range: payload, depth: depth, state: &state,
                     fileName: fileName, collector: &collector) { type, atom in
            if type == RecordType.notesAtom, let value = Self.uint32(atom, at: 0) {
                slideIdRef = value
            }
        }
        state.notes.append(NotesContainer(slideIdRef: slideIdRef,
                                          blocks: collector.finish()))
    }

    private func processSlideList(_ data: Data, payload: Range<Int>, depth: Int,
                                  state: inout ScanState, fileName: String) throws {
        guard depth <= LegacyOfficeLimits.maxRecordDepth else {
            throw depthLimitError(fileName)
        }

        var groups: [SlideTextGroup] = []
        var currentId: UInt32?
        var collector = TextCollector()

        func flush() {
            guard let id = currentId else { return }
            groups.append(SlideTextGroup(slideId: id, blocks: collector.finish()))
            collector = TextCollector()
        }

        var offset = payload.lowerBound
        while offset < payload.upperBound {
            let header = try readHeader(data, at: offset, end: payload.upperBound,
                                         fileName: fileName)
            try countRecord(&state, fileName: fileName)

            if header.version == RecordType.containerVersion {
                try walkText(data, range: header.payload, depth: depth + 1,
                             state: &state, fileName: fileName,
                             collector: &collector)
            } else if header.type == RecordType.slidePersistAtom {
                flush()
                currentId = Self.uint32(data[header.payload], at: 12) ?? 0
            } else {
                let atom = data[header.payload]
                switch header.type {
                case RecordType.textHeaderAtom:
                    if let type = Self.uint32(atom, at: 0) { collector.begin(type) }
                case RecordType.textCharsAtom:
                    collector.append(Self.decodeChars(atom))
                case RecordType.textBytesAtom:
                    collector.append(Self.decodeBytes(atom))
                default:
                    break
                }
            }
            offset = header.payload.upperBound
        }
        flush()
        state.slideGroups.append(contentsOf: groups)
    }

    private func countRecord(_ state: inout ScanState, fileName: String) throws {
        state.recordCount += 1
        guard state.recordCount <= LegacyOfficeLimits.maxRecordCount else {
            throw AgentError.parse(
                "PowerPoint 记录数量超过 \(LegacyOfficeLimits.maxRecordCount) 条上限"
                + "（\(fileName)）")
        }
        if state.recordCount % 4096 == 0, Task.isCancelled {
            throw CancellationError()
        }
    }

    /// 读取记录头并校验负载边界。
    private func readHeader(_ data: Data, at offset: Int, end: Int,
                            fileName: String) throws -> RecordHeader {
        guard offset >= 0, end <= data.count, offset <= end else {
            throw AgentError.parse("PowerPoint 记录偏移越界（\(fileName)）")
        }
        if end - offset < 8 {
            if Self.isZeroPadding(data, from: offset, to: end) {
                return RecordHeader(version: 0, instance: 0, type: 0,
                                    payload: end..<end)
            }
            throw AgentError.parse("PowerPoint 记录头不完整（\(fileName)）")
        }
        guard let raw = Self.uint16(data, at: offset),
              let type = Self.uint16(data, at: offset + 2),
              let length = Self.uint32(data, at: offset + 4) else {
            throw AgentError.parse("PowerPoint 记录头不完整（\(fileName)）")
        }
        let payloadStart = offset + 8
        let payloadLength = Int(length)
        guard payloadLength <= end - payloadStart else {
            throw AgentError.parse("PowerPoint 记录长度越界（\(fileName)）")
        }
        return RecordHeader(version: UInt8(raw & 0x000F),
                            instance: raw >> 4,
                            type: type,
                            payload: payloadStart..<(payloadStart + payloadLength))
    }

    private func depthLimitError(_ fileName: String) -> AgentError {
        AgentError.parse(
            "PowerPoint 记录嵌套深度超过 \(LegacyOfficeLimits.maxRecordDepth) 层上限"
            + "（\(fileName)）")
    }

    // MARK: - 幻灯片与备注组装

    /// 优先使用 SlideListWithText 的顺序；该列表缺失或只覆盖部分幻灯片时，
    /// 用 Slide 容器补齐。
    private func selectSlides(from state: ScanState) -> [SlideContainer] {
        guard !state.slideGroups.isEmpty else {
            return state.slides
        }

        var containerIndexByID: [UInt32: Int] = [:]
        for (index, slide) in state.slides.enumerated() where slide.slideId != 0 {
            if containerIndexByID[slide.slideId] == nil {
                containerIndexByID[slide.slideId] = index
            }
        }

        var usedContainerIndices = Set<Int>()
        var selected: [SlideContainer] = []
        var selectedContainerIndex: [Int?] = []

        for (index, group) in state.slideGroups.enumerated() {
            var containerIndex: Int?
            if group.slideId != 0, let matched = containerIndexByID[group.slideId],
               !usedContainerIndices.contains(matched) {
                containerIndex = matched
            } else if index < state.slides.count,
                      !usedContainerIndices.contains(index) {
                // 真实文件里 SlideListWithText 的 ID 与 SlideAtom 字段不保证
                // 一致，列表也可能只覆盖部分页；按位置回退保留容器文本。
                containerIndex = index
            }

            if let containerIndex {
                usedContainerIndices.insert(containerIndex)
            }
            let container = containerIndex.map { state.slides[$0] }
            selected.append(SlideContainer(
                slideId: group.slideId != 0 ? group.slideId : (container?.slideId ?? 0),
                blocks: preferredBlocks(list: group.blocks, container: container?.blocks)))
            selectedContainerIndex.append(containerIndex)
        }

        let unmatchedIndices = state.slides.indices.filter {
            !usedContainerIndices.contains($0)
        }
        guard !unmatchedIndices.isEmpty else { return selected }

        // 仅在列表顺序与容器顺序一致时，才能按容器位置把缺失页插回原处；
        // 否则按文件顺序追加，至少不丢页。
        let mappedIndices = selectedContainerIndex.compactMap { $0 }
        guard mappedIndices == mappedIndices.sorted() else {
            return selected + unmatchedIndices.map { state.slides[$0] }
        }

        var insertions: [Int: [SlideContainer]] = [:]
        var trailing: [SlideContainer] = []
        for containerIndex in unmatchedIndices {
            let target = selectedContainerIndex.indices.first { outputIndex in
                guard let mapped = selectedContainerIndex[outputIndex] else {
                    return false
                }
                return mapped > containerIndex
            }
            if let target {
                insertions[target, default: []].append(state.slides[containerIndex])
            } else {
                trailing.append(state.slides[containerIndex])
            }
        }

        var merged: [SlideContainer] = []
        for index in selected.indices {
            merged.append(contentsOf: insertions[index] ?? [])
            merged.append(selected[index])
        }
        merged.append(contentsOf: trailing)
        return merged
    }

    /// 列表与容器都有文本时取更完整的一份：列表可能只保存标题，容器则
    /// 保留全部形状文本；两者相同时优先列表，避免改变既有顺序。
    private func preferredBlocks(list: [TextBlock],
                                 container: [TextBlock]?) -> [TextBlock] {
        guard let container, !container.isEmpty else { return list }
        guard !list.isEmpty else { return container }
        let listLength = list.reduce(0) { $0 + textLength($1) }
        let containerLength = container.reduce(0) { $0 + textLength($1) }
        return containerLength > listLength ? container : list
    }

    private func textLength(_ block: TextBlock) -> Int {
        block.paragraphs.reduce(0) { $0 + $1.count }
    }

    /// 备注按 NotesAtom 的 slideIdRef 挂到对应幻灯片；全部缺失关联时按顺序兜底。
    private func notesForSlides(_ slides: [SlideContainer],
                                notes: [NotesContainer]) -> [[TextBlock]] {
        var result = Array(repeating: [TextBlock](), count: slides.count)
        var used = Set<Int>()

        for (slideIndex, slide) in slides.enumerated() where slide.slideId != 0 {
            for (noteIndex, note) in notes.enumerated()
            where !used.contains(noteIndex) && note.slideIdRef == slide.slideId {
                result[slideIndex].append(contentsOf: note.blocks)
                used.insert(noteIndex)
            }
        }

        // 只有部分备注匹配到 slideId 时，未匹配备注仍需按顺序补到还没有备注的
        // 幻灯片上；此前仅在全部未匹配时才兜底，会静默丢失这部分内容。
        if used.count < notes.count {
            let unmatched = notes.enumerated().filter { !used.contains($0.offset) }
                .map(\.element)
            var targets = slides.indices.filter { result[$0].isEmpty }
            if targets.isEmpty { targets = Array(slides.indices) }
            for (index, note) in zip(targets, unmatched) {
                result[index].append(contentsOf: note.blocks)
            }
        }
        return result
    }

    private func renderSlide(_ slide: SlideContainer, number: Int,
                             notes: [TextBlock])
        -> (section: String, outlineTitle: String,
            hasContent: Bool, hasNotes: Bool) {
        var title: String?
        var body: [String] = []

        for block in slide.blocks {
            if block.headerType == Self.notesTextType { continue }
            if let header = block.headerType, Self.titleTextTypes.contains(header),
               title == nil {
                let candidate = Self.titleText(block.paragraphs)
                if !candidate.isEmpty {
                    title = candidate
                    continue
                }
            }
            body.append(contentsOf: block.paragraphs)
        }

        var section = "## " + Self.outlineTitle(slideNumber: number, title: title)
        if !body.isEmpty {
            section += "\n\n" + body.joined(separator: "\n\n")
        }

        let noteParagraphs = notes.flatMap(\.paragraphs)
        if !noteParagraphs.isEmpty {
            section += "\n\n**演讲者备注**\n\n"
                + noteParagraphs.joined(separator: "\n\n")
        }

        return (section,
                Self.outlineTitle(slideNumber: number, title: title),
                title != nil || !body.isEmpty || !noteParagraphs.isEmpty,
                !noteParagraphs.isEmpty)
    }

    private static func outlineTitle(slideNumber: Int, title: String?) -> String {
        guard let title, !title.isEmpty else { return "Slide \(slideNumber)" }
        return "Slide \(slideNumber)：\(title)"
    }

    private static func titleText(_ paragraphs: [String]) -> String {
        paragraphs.joined(separator: " ")
            .replacingOccurrences(of: "\\s+", with: " ",
                                  options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - 文本 atom

    /// 文本块：一个 TextHeaderAtom 后紧跟的所有文本 atom 按顺序拼接。
    private struct TextCollector {
        private var blocks: [TextBlock] = []
        private var headerType: UInt32?
        private var pending = ""
        private var atomCount = 0

        mutating func begin(_ type: UInt32) {
            flush()
            headerType = type
        }

        mutating func append(_ text: String) {
            pending += text
            atomCount += 1
        }

        mutating func finish() -> [TextBlock] {
            flush()
            return blocks
        }

        private mutating func flush() {
            defer {
                headerType = nil
                pending = ""
                atomCount = 0
            }
            let paragraphs = Self.paragraphs(from: pending)
            guard !paragraphs.isEmpty else { return }
            blocks.append(TextBlock(headerType: headerType,
                                    paragraphs: paragraphs,
                                    atomCount: atomCount))
        }

        private static func paragraphs(from text: String) -> [String] {
            let normalized = text
                .replacingOccurrences(of: "\r\n", with: "\n")
                .replacingOccurrences(of: "\r", with: "\n")
                .replacingOccurrences(of: "\u{000B}", with: "\n")
            return normalized
                .components(separatedBy: "\n")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        }
    }

    private static func decodeChars(_ payload: Data) -> String {
        let count = payload.count / 2
        guard count > 0 else { return "" }
        var units: [UInt16] = []
        units.reserveCapacity(count)
        for index in 0..<count {
            if let unit = uint16(payload, at: index * 2) {
                units.append(unit)
            }
        }
        return String(decoding: units, as: UTF16.self)
    }

    /// PowerPoint 97 的 `TextBytesAtom` 按「ANSI」写入，实际文件可能是
    /// Windows-1252、UTF-8 或 GBK/GB18030。CP1252 几乎能解码任意字节，
    /// 因此必须先看解码结果的内容：GBK 字节可能恰好是合法 UTF-8（例如
    /// 「中文」的 D6 D0 CE C4 会解成希伯来/希腊字母），不能只按编码顺序试。
    private static func decodeBytes(_ payload: Data) -> String {
        let utf8 = String(data: payload, encoding: .utf8)
        let gb = String(data: payload, encoding: gb18030Encoding)

        if let utf8 {
            let utf8Stats = textStats(utf8)
            if utf8Stats.hasCJK { return utf8 }
            // 合法 UTF-8 的希腊/西里尔文也能被 GB18030 解成汉字，只有出现
            // 多种文字混杂或 C1 控制字符时才判定为 GBK 乱码。
            if utf8Stats.hasMojibakeSignals, let gb,
               textStats(gb).looksLikeChinese {
                return gb
            }
            return utf8
        }

        if let gb, textStats(gb).looksLikeChinese {
            return gb
        }
        return String(data: payload, encoding: .windowsCP1252)
            ?? String(data: payload, encoding: .isoLatin1) ?? ""
    }

    private static let gb18030Encoding = String.Encoding(
        rawValue: CFStringConvertEncodingToNSStringEncoding(
            CFStringEncoding(CFStringEncodings.GB_18030_2000.rawValue)))

    private struct TextStats {
        let characterCount: Int
        let cjkCount: Int
        let longestCJKRun: Int
        let suspiciousScriptGroups: UInt32
        let hasControlCharacters: Bool

        var hasCJK: Bool { cjkCount > 0 }

        /// GBK 字节被误读成 UTF-8 的典型特征：希伯来/希腊/西里尔等多种
        /// 文字混杂，或出现正常文本不会有的 C1 控制字符。单一文字的合法
        /// UTF-8（例如整段西里尔文）不据此改判。
        var hasMojibakeSignals: Bool {
            suspiciousScriptGroups.nonzeroBitCount >= 2 || hasControlCharacters
        }

        /// GB18030 解出的文本要像中文而不是偶发的乱码：至少两个汉字、
        /// 存在连续汉字，且汉字占非空白字符的四分之一以上。
        var looksLikeChinese: Bool {
            cjkCount >= 2 && longestCJKRun >= 2
                && Double(cjkCount) >= Double(characterCount) * 0.25
        }
    }

    private static func textStats(_ text: String) -> TextStats {
        var characterCount = 0
        var cjkCount = 0
        var currentRun = 0
        var longestRun = 0
        var suspiciousScriptGroups: UInt32 = 0
        var hasControlCharacters = false

        for scalar in text.unicodeScalars {
            let value = scalar.value
            guard value > 0x20 else { continue }
            characterCount += 1
            if isCJK(value) {
                cjkCount += 1
                currentRun += 1
                longestRun = max(longestRun, currentRun)
            } else {
                currentRun = 0
                if value >= 0x7F && value <= 0x9F {
                    hasControlCharacters = true
                }
                if let group = suspiciousScriptGroup(value) {
                    suspiciousScriptGroups |= 1 << group
                }
            }
        }

        return TextStats(characterCount: characterCount,
                         cjkCount: cjkCount,
                         longestCJKRun: longestRun,
                         suspiciousScriptGroups: suspiciousScriptGroups,
                         hasControlCharacters: hasControlCharacters)
    }

    private static func isCJK(_ value: UInt32) -> Bool {
        switch value {
        case 0x3400...0x4DBF,   // CJK Extension A
             0x4E00...0x9FFF,   // CJK Unified Ideographs
             0xF900...0xFAFF,   // CJK Compatibility Ideographs
             0x3000...0x303F,   // CJK 标点
             0xFF00...0xFFEF:   // 全角字符
            return true
        default:
            return false
        }
    }

    /// GBK 双字节被误读成 UTF-8 时通常落在这些非拉丁字母区；真正的 UTF-8
    /// 西文（含 Latin-1 / Latin Extended 重音字母）不会触发。返回文字分组，
    /// 用于区分「多种文字混杂的乱码」和「单一文字的合法 UTF-8」。
    private static func suspiciousScriptGroup(_ value: UInt32) -> UInt32? {
        switch value {
        case 0x0370...0x03FF: return 0   // Greek
        case 0x0400...0x04FF: return 1   // Cyrillic
        case 0x0530...0x058F: return 2   // Armenian
        case 0x0590...0x05FF: return 3   // Hebrew
        case 0x0600...0x06FF: return 4   // Arabic
        case 0x0700...0x077F: return 5   // Syriac / Arabic Supplement
        case 0x0780...0x07FF: return 6   // Thaana / NKo
        case 0x0900...0x0DFF: return 7   // 印度系文字
        case 0x0E00...0x0EFF: return 8   // Thai / Lao
        case 0x10A0...0x10FF: return 9   // Georgian
        case 0x1100...0x11FF: return 10  // Hangul Jamo
        case 0x1200...0x137F: return 11  // Ethiopic
        default: return nil
        }
    }

    // MARK: - 无符号读取（相对 Data.startIndex）

    private static func uint16(_ data: Data, at offset: Int) -> UInt16? {
        let start = data.startIndex + offset
        guard offset >= 0, start >= data.startIndex, start + 2 <= data.endIndex else {
            return nil
        }
        return UInt16(data[start]) | (UInt16(data[start + 1]) << 8)
    }

    private static func uint32(_ data: Data, at offset: Int) -> UInt32? {
        let start = data.startIndex + offset
        guard offset >= 0, start >= data.startIndex, start + 4 <= data.endIndex else {
            return nil
        }
        return UInt32(data[start])
            | (UInt32(data[start + 1]) << 8)
            | (UInt32(data[start + 2]) << 16)
            | (UInt32(data[start + 3]) << 24)
    }

    private static func isZeroPadding(_ data: Data, from offset: Int, to end: Int) -> Bool {
        var index = data.startIndex + offset
        let endIndex = data.startIndex + end
        while index < endIndex {
            if data[index] != 0 { return false }
            index += 1
        }
        return true
    }
}
