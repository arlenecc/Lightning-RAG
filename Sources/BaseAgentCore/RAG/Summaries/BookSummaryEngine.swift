import Foundation
import CryptoKit

/// 逐章/全书摘要引擎。
///
/// - 章节结构由 `OutlineResolver` 从 Markdown 与已存目录择优生成；
/// - 章节正文按辅助模型上下文预算分段，段结果持久化后增量复用；
/// - map 阶段按 `batchConcurrency` 并行处理分片（后端支持的独立会话数），
///   reduce 生成 50–100 字章节摘要，全部章节完成后生成 300–500 字全书摘要；
/// - 摘要请求一律关闭 thinking 并使用独立会话，避免分片共享 KV；
/// - 任何模型调用前后都检查取消标记，中断时只保留已完成分段。
public final class BookSummaryEngine: @unchecked Sendable {

    private let store: VectorStore
    /// 同时执行的摘要分片数（默认 5，上限 5）。MLX/ONNX 后端能为每个
    /// `isolated` 请求提供独立 KV/past，可真正并行；GGUF/VLM 单 context
    /// 后端会按能力标记退化为逐条执行。
    private let batchConcurrency: Int
    /// 预留 system/指令与输出 token 后的分段预算上限。
    private static let maxSegmentBudget = 8_000
    /// 章节正文超过该字符数时强制 map-reduce，避免一次 map 吞掉整章细节。
    private static let mapReduceThreshold = 4_000
    /// 长章节每个 map 分片的最大字符数。
    private static let mapSegmentCharacters = 3_000

    public init(store: VectorStore, batchConcurrency: Int = 5) {
        self.store = store
        self.batchConcurrency = max(1, min(5, batchConcurrency))
    }

    // MARK: - Token 估算与分段

    /// CJK 按 1 token，其它字符按 0.25 token；向上取整以偏保守。
    public static func estimateTokens(_ text: String) -> Int {
        guard !text.isEmpty else { return 0 }
        let tokens = text.reduce(0.0) { partial, character in
            partial + (isCJK(character) ? 1.0 : 0.25)
        }
        return Int(ceil(tokens))
    }

    /// 按空行分段；累计超出 token/字符预算时切段，超长段落再按句末标点/字符硬切。
    public static func splitSegments(_ body: String, tokenBudget: Int,
                                     characterBudget: Int? = nil) -> [String] {
        let text = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return [] }
        let budget = max(64, tokenBudget)
        let charBudget = characterBudget.map { max(64, $0) }
        let paragraphs = text.components(separatedBy: "\n\n").compactMap { raw -> String? in
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
        func fits(_ value: String) -> Bool {
            guard estimateTokens(value) <= budget else { return false }
            if let charBudget, value.count > charBudget { return false }
            return true
        }

        var segments: [String] = []
        var current = ""
        for paragraph in paragraphs {
            if !fits(paragraph) {
                if !current.isEmpty {
                    segments.append(current)
                    current = ""
                }
                segments.append(contentsOf: splitOversizedParagraph(
                    paragraph, tokenBudget: budget, characterBudget: charBudget))
                continue
            }
            let candidate = current.isEmpty ? paragraph : current + "\n\n" + paragraph
            if fits(candidate) {
                current = candidate
            } else {
                if !current.isEmpty { segments.append(current) }
                current = paragraph
            }
        }
        if !current.isEmpty { segments.append(current) }
        return segments
    }

    /// 把一个超预算段落切成多个分片。
    ///
    /// 使用 `String.Index` 单调前进，每个字符只扫描一次；旧实现每轮把剩余文本
    /// 转成 `[Character]` 并复制，遇到 Excel 这种整张表只有一个段落的长文本会
    /// 退化成 O(n²)。
    private static func splitOversizedParagraph(_ text: String, tokenBudget: Int,
                                                characterBudget: Int?) -> [String] {
        let end = text.endIndex
        var start = text.startIndex
        var pieces: [String] = []

        while start < end {
            while start < end, text[start].isWhitespace {
                start = text.index(after: start)
            }
            guard start < end else { break }

            var index = start
            var tokens = 0.0
            var count = 0
            var lastBreakOffset = -1
            var cut: String.Index?

            while index < end {
                let character = text[index]
                let cost = tokenCost(character)
                let exceedsTokens = tokens + cost > Double(tokenBudget)
                let exceedsCharacters = characterBudget.map { count + 1 > $0 } ?? false
                if (exceedsTokens || exceedsCharacters), count > 0 {
                    // 与旧逻辑一致：只有句末标点落在当前片段后半段时才回退断句，
                    // 避免每遇到一个逗号/句号就切得过碎。
                    if lastBreakOffset >= 0, lastBreakOffset >= count / 2 {
                        cut = text.index(after: text.index(start, offsetBy: lastBreakOffset))
                    } else {
                        cut = index
                    }
                    break
                }
                tokens += cost
                count += 1
                if isSentenceEnd(character) { lastBreakOffset = count - 1 }
                index = text.index(after: index)
            }

            let cutIndex = cut ?? end
            let piece = String(text[start..<cutIndex])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !piece.isEmpty { pieces.append(piece) }
            start = cutIndex
        }
        return pieces
    }

    // MARK: - 摘要规范化

    public static func normalizeChapterSummary(_ raw: String) -> String {
        normalizeSummary(raw, maxLength: 100, preferredCut: 30)
    }

    public static func normalizeBookSummary(_ raw: String) -> String {
        normalizeSummary(raw, maxLength: 500, preferredCut: 250)
    }

    private static func normalizeSummary(_ raw: String, maxLength: Int,
                                         preferredCut: Int) -> String {
        let lines = raw.replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "```", with: "")
            .components(separatedBy: "\n")
            .compactMap(stripSummaryLine)
        var text = joinSummaryLines(lines)
        text = stripSummaryLabels(text)
        text = text.replacingOccurrences(of: "\\s+", with: " ",
                                         options: .regularExpression)
        text = text.replacingOccurrences(of: " ([。，！？；：、])", with: "$1",
                                         options: .regularExpression)
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        while let first = text.first, "\"'“”‘’「」『』".contains(first) {
            text.removeFirst()
        }
        while let last = text.last, "\"'“”‘’「」『』".contains(last) {
            text.removeLast()
        }
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard text.count > maxLength else { return text }

        let prefix = String(text.prefix(maxLength))
        if let punctuation = prefix.lastIndex(where: { isSentenceEnd($0) }) {
            let length = prefix.distance(from: prefix.startIndex, to: punctuation) + 1
            if length >= preferredCut {
                return String(prefix[..<prefix.index(after: punctuation)])
            }
        }
        return prefix.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func stripSummaryLine(_ raw: String) -> String? {
        var line = raw.trimmingCharacters(in: .whitespaces)
        line = line
            .replacingOccurrences(of: "**", with: "")
            .replacingOccurrences(of: "__", with: "")
            .replacingOccurrences(of: "`", with: "")
        while line.hasPrefix("#") || line.hasPrefix(">") {
            line.removeFirst()
            line = line.trimmingCharacters(in: .whitespaces)
        }
        if let range = line.range(of: #"^[-*+]\s+"#, options: .regularExpression) {
            line.removeSubrange(range)
        }
        if let range = line.range(of: #"^[0-9]+[.、)]\s*"#, options: .regularExpression) {
            line.removeSubrange(range)
        }
        line = line.trimmingCharacters(in: .whitespaces)
        guard !line.isEmpty else { return nil }
        let labels = ["本章摘要", "章节摘要", "全书摘要", "内容梗概", "内容概要", "摘要", "总结"]
        if labels.contains(where: { line == $0 || line == "\($0)：" || line == "\($0):" }) {
            return nil
        }
        return line
    }

    private static func joinSummaryLines(_ lines: [String]) -> String {
        var result = ""
        for line in lines {
            guard !result.isEmpty else {
                result = line
                continue
            }
            if let last = result.last, let first = line.first,
               last.isASCII && first.isASCII,
               last.isLetter || last.isNumber,
               first.isLetter || first.isNumber {
                result += " " + line
            } else {
                result += line
            }
        }
        return result
    }

    private static func stripSummaryLabels(_ raw: String) -> String {
        var text = raw
        let pattern = #"^(?:本章摘要|章节摘要|全书摘要|内容梗概|内容概要|摘要|总结)\s*[：:]\s*"#
        for _ in 0..<3 {
            guard let range = text.range(of: pattern, options: .regularExpression) else {
                break
            }
            text.removeSubrange(range)
            text = text.trimmingCharacters(in: .whitespaces)
        }
        return text
    }

    // MARK: - 章节结构

    public func resolve(_ document: DocumentRecord, markdown: String) -> ResolvedBookOutline {
        let parsedOutline = OutlineResolver.outlineText(fromChaptersJSON: document.chapters)
            ?? Self.outlineFromDigest(document.digest)
        let resolved = OutlineResolver.resolve(
            markdown: markdown,
            parsedOutline: parsedOutline,
            docName: document.docName,
            preferParser: OutlineResolver.prefersParserOutline(for: document.docName))
        // 无标题纯正文（旧版 Word 未识别到任何标题样式）本来没有章节可用，
        // 但正文仍是完整可摘要的内容：合成一个「全文」章节，让逐章与全书
        // 摘要流程都能继续，而不是直接标记完成并跳过全书摘要。
        // 目录保持原样，避免把「全文」写回入库时的大纲。
        let body = markdown.trimmingCharacters(in: .whitespacesAndNewlines)
        guard resolved.sections.isEmpty, !body.isEmpty else { return resolved }
        let section = BookSection(
            index: 0, level: 1, title: "全文", heading: "全文", body: body,
            start: 0, end: (markdown as NSString).length,
            contentHash: Self.shortHash("全文\n" + body))
        return ResolvedBookOutline(
            nodes: resolved.nodes, sections: [section], source: resolved.source,
            digest: resolved.digest, chaptersJSON: resolved.chaptersJSON,
            hadRepairs: resolved.hadRepairs)
    }

    /// `documents.digest` 首行固定为「# 目录」，其余是入库时择优的目录。
    private static func outlineFromDigest(_ digest: String) -> String {
        let lines = digest.components(separatedBy: "\n")
        guard let first = lines.first else { return "" }
        let body = first.trimmingCharacters(in: .whitespaces).hasPrefix("#")
            ? lines.dropFirst().joined(separator: "\n")
            : digest
        return body.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - 逐章处理

    /// 处理 `next_section` 指向的章节；已摘要且内容未变的章节自动跳过。
    public func processNextChapter(document: DocumentRecord,
                                   markdown: String,
                                   llm: LLMService,
                                   now: Date,
                                   isCancelled: @escaping () -> Bool) async throws -> ChapterProcessResult {
        let cancellation = SummaryCancellation(isCancelled)
        let resolved = resolve(document, markdown: markdown)
        let total = resolved.sections.count
        if resolved.digest != document.digest || resolved.chaptersJSON != document.chapters {
            try? store.updateDocumentOutline(docID: document.docID,
                                             digest: resolved.digest,
                                             chapters: resolved.chaptersJSON)
        }
        guard total > 0 else {
            try store.upsertSummaryJob(SummaryJobRecord(
                docID: document.docID, status: .completed, nextSection: 0,
                totalSections: 0, lastError: "", updatedAt: now.timeIntervalSince1970))
            return ChapterProcessResult(
                sectionIndex: 0, totalSections: 0, chapterCompleted: true,
                bookCompleted: true, titleRepaired: resolved.hadRepairs, summary: "")
        }

        let job = try store.summaryJob(docID: document.docID)
        if job?.status == .completed, job?.totalSections == total {
            return ChapterProcessResult(
                sectionIndex: max(0, total - 1), totalSections: total,
                chapterCompleted: true, bookCompleted: true,
                titleRepaired: resolved.hadRepairs, summary: "")
        }

        var stored = try store.chapterSummaries(docID: document.docID)
        // 目录修复可能收缩章节数：越界的旧摘要不会再被覆盖，先清掉。
        if stored.contains(where: { $0.sectionIndex >= total }) {
            try? store.deleteChapterSummaries(docID: document.docID,
                                             fromSectionIndex: total)
            stored.removeAll { $0.sectionIndex >= total }
        }
        let storedByIndex = Dictionary(
            stored.map { ($0.sectionIndex, $0) },
            uniquingKeysWith: { first, _ in first })
        let missing = Self.firstMissingSectionIndex(
            sections: resolved.sections, storedByIndex: storedByIndex)
        // next_section 只在正常推进时可信；目录/正文变化或任务缺失时，
        // 以第一个真正缺少有效摘要的章节为准，避免跳过需要重做的章节。
        var index = max(0, job?.nextSection ?? 0)
        if missing < index { index = missing }
        if job == nil || job?.totalSections != total || job?.nextSection != index {
            try store.upsertSummaryJob(SummaryJobRecord(
                docID: document.docID, status: .pending,
                nextSection: index, totalSections: total,
                lastError: job?.lastError ?? "", updatedAt: now.timeIntervalSince1970))
        }

        while index < total {
            let section = resolved.sections[index]
            if let existing = try store.chapterSummary(docID: document.docID, sectionIndex: index),
               existing.contentHash == section.contentHash,
               !existing.summary.isEmpty {
                index += 1
                continue
            }
            try cancellation.check()
            do {
                let summary = try await summarizeSection(
                    document: document, section: section, llm: llm,
                    now: now, cancellation: cancellation)
                try store.upsertChapterSummary(ChapterSummaryRecord(
                    docID: document.docID, sectionIndex: index,
                    heading: section.heading, title: section.title,
                    level: section.level, summary: summary,
                    contentHash: section.contentHash,
                    updatedAt: now.timeIntervalSince1970))
                let next = index + 1
                let chapterCompleted = next >= total
                try store.upsertSummaryJob(SummaryJobRecord(
                    docID: document.docID,
                    status: chapterCompleted ? .summarizingBook : .pending,
                    nextSection: next, totalSections: total,
                    lastError: "", updatedAt: now.timeIntervalSince1970))
                return ChapterProcessResult(
                    sectionIndex: index, totalSections: total,
                    chapterCompleted: chapterCompleted, bookCompleted: false,
                    titleRepaired: resolved.hadRepairs, summary: summary)
            } catch {
                if !cancellation.isCancellation(error) {
                    try? store.upsertSummaryJob(SummaryJobRecord(
                        docID: document.docID, status: .failed, nextSection: index,
                        totalSections: total,
                        lastError: error.localizedDescription,
                        updatedAt: now.timeIntervalSince1970))
                }
                throw error
            }
        }

        // 所有章节都已有可用摘要，进入全书摘要阶段。
        try store.upsertSummaryJob(SummaryJobRecord(
            docID: document.docID, status: .summarizingBook, nextSection: total,
            totalSections: total, lastError: "", updatedAt: now.timeIntervalSince1970))
        return ChapterProcessResult(
            sectionIndex: max(0, total - 1), totalSections: total,
            chapterCompleted: true, bookCompleted: false,
            titleRepaired: resolved.hadRepairs, summary: "")
    }

    private func summarizeSection(document: DocumentRecord,
                                  section: BookSection,
                                  llm: LLMService,
                                  now: Date,
                                  cancellation: SummaryCancellation) async throws -> String {
        let body = section.body.trimmingCharacters(in: .whitespacesAndNewlines)
        if body.count < 40 {
            return Self.fallbackSummary(for: section)
        }
        let budget = Self.segmentBudget(contextWindow: llm.contextWindow)
        // 超过 4000 字的章节即使整体塞得进上下文，也强制按 3000 字一段做
        // map-reduce，避免一次 map 吞掉整章、丢掉中后段细节。
        let characterBudget = body.count > Self.mapReduceThreshold
            ? Self.mapSegmentCharacters : nil
        let segments = Self.splitSegments(body, tokenBudget: budget,
                                          characterBudget: characterBudget)
        guard !segments.isEmpty else { return Self.fallbackSummary(for: section) }

        let persisted = try store.chapterSegments(
            docID: document.docID, sectionIndex: section.index)
        let persistedByIndex = Dictionary(
            persisted.map { ($0.segmentIndex, $0) },
            uniquingKeysWith: { first, _ in first })
        var segmentSummaries = [String?](repeating: nil, count: segments.count)
        var requests: [SummaryRequest] = []
        for (segmentIndex, segment) in segments.enumerated() {
            try cancellation.check()
            let hash = Self.shortHash(segment)
            if let reused = persistedByIndex[segmentIndex],
               reused.segmentHash == hash,
               !reused.summary.isEmpty {
                segmentSummaries[segmentIndex] = reused.summary
                continue
            }
            let prompt = """
            请用 30-60 字概括下面章节片段的核心内容，只输出摘要正文，不要标题或解释。

            章节：\(section.heading)
            片段：
            \(segment)
            """
            requests.append(SummaryRequest(
                index: segmentIndex,
                system: "你是专业的书籍摘要助手，只输出摘要正文。",
                prompt: prompt,
                maxLength: 80, preferredCut: 20,
                emptyMessage: "辅助模型未返回有效的分段摘要"))
        }

        let outcome = await runSummaryBatch(requests, llm: llm, cancellation: cancellation)
        for (segmentIndex, summary) in outcome.summaries.sorted(by: { $0.key < $1.key }) {
            try store.upsertChapterSegment(ChapterSegmentRecord(
                docID: document.docID, sectionIndex: section.index,
                segmentIndex: segmentIndex,
                segmentHash: Self.shortHash(segments[segmentIndex]),
                summary: summary, updatedAt: now.timeIntervalSince1970))
            segmentSummaries[segmentIndex] = summary
        }
        if let failure = outcome.failure {
            throw failure.asError()
        }
        guard let summaries = Self.collectedSummaries(segmentSummaries) else {
            throw AgentError.llm("辅助模型未返回有效的分段摘要")
        }

        var reduced = try await reduceChapter(
            section: section, segmentSummaries: summaries,
            llm: llm, cancellation: cancellation)
        if reduced.count < 30 {
            let retryPrompt = """
            上一次摘要过短。请严格输出 50-100 字的中文摘要，只输出摘要正文。

            章节：\(section.heading)
            各片段摘要：
            \(summaries.joined(separator: "\n"))

            上一次结果：\(reduced)
            """
            let retryRaw = try await complete(
                system: "你是专业的书籍摘要助手，只输出摘要正文。",
                prompt: retryPrompt, llm: llm, temperature: 0.2,
                cancellation: cancellation)
            let retry = Self.normalizeChapterSummary(retryRaw)
            if retry.count > reduced.count { reduced = retry }
        }
        return reduced.isEmpty ? Self.fallbackSummary(for: section) : reduced
    }

    private func reduceChapter(section: BookSection, segmentSummaries: [String],
                               llm: LLMService,
                               cancellation: SummaryCancellation) async throws -> String {
        let prompt = """
        以下是章节《\(section.heading)》各片段的内容概括。请综合成 50-100 字的中文摘要，只输出摘要正文，不要标题或解释。

        \(segmentSummaries.enumerated().map { "\($0.offset + 1). \($0.element)" }
            .joined(separator: "\n"))
        """
        let raw = try await complete(
            system: "你是专业的书籍摘要助手，只输出摘要正文。",
            prompt: prompt, llm: llm, temperature: 0.2,
            cancellation: cancellation)
        return Self.normalizeChapterSummary(raw)
    }

    private static func fallbackSummary(for section: BookSection) -> String {
        let body = section.body
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !body.isEmpty else { return "\(section.title)（正文较短）" }
        let excerpt = String(body.prefix(60))
        return normalizeChapterSummary("\(section.title)：\(excerpt)")
    }

    // MARK: - 全书摘要

    /// 全部章节完成后的 map-reduce 全书摘要；组结果持久化后增量复用。
    public func processBookSummary(document: DocumentRecord,
                                   markdown: String,
                                   llm: LLMService,
                                   now: Date,
                                   isCancelled: @escaping () -> Bool) async throws -> BookSummaryRecord {
        let cancellation = SummaryCancellation(isCancelled)
        let resolved = resolve(document, markdown: markdown)
        if resolved.digest != document.digest || resolved.chaptersJSON != document.chapters {
            try? store.updateDocumentOutline(docID: document.docID,
                                             digest: resolved.digest,
                                             chapters: resolved.chaptersJSON)
        }
        let sections = resolved.sections
        guard !sections.isEmpty else {
            try store.upsertSummaryJob(SummaryJobRecord(
                docID: document.docID, status: .completed, nextSection: 0,
                totalSections: 0, lastError: "", updatedAt: now.timeIntervalSince1970))
            throw AgentError.storage("本书没有可识别章节，跳过全书摘要")
        }
        let stored = try store.chapterSummaries(docID: document.docID)
        if stored.contains(where: { $0.sectionIndex >= sections.count }) {
            try? store.deleteChapterSummaries(docID: document.docID,
                                             fromSectionIndex: sections.count)
        }
        let storedByIndex = Dictionary(
            stored.map { ($0.sectionIndex, $0) },
            uniquingKeysWith: { first, _ in first })
        var chapters: [ChapterSummaryRecord] = []
        var firstMissing: Int?
        for (offset, section) in sections.enumerated() {
            if let row = storedByIndex[section.index],
               row.contentHash == section.contentHash,
               !row.summary.isEmpty {
                chapters.append(row)
            } else if firstMissing == nil {
                firstMissing = offset
            }
        }
        if let firstMissing {
            // 目录修复或正文变化导致摘要与当前章节对不上：退回逐章阶段，
            // 不能拿旧摘要生成全书摘要。
            try store.upsertSummaryJob(SummaryJobRecord(
                docID: document.docID, status: .pending, nextSection: firstMissing,
                totalSections: sections.count,
                lastError: "章节摘要与当前目录不一致，需要重新摘要",
                updatedAt: now.timeIntervalSince1970))
            throw AgentError.storage("章节摘要与当前目录不一致，已回到逐章处理")
        }
        guard chapters.count == sections.count else {
            throw AgentError.storage("本书尚未生成完整的章节摘要")
        }

        let lines = chapters.map { "- [\($0.heading)] \($0.summary)" }
        let budget = Self.segmentBudget(contextWindow: llm.contextWindow)
        let groups = Self.splitSegments(lines.joined(separator: "\n\n"),
                                        tokenBudget: budget)
        guard !groups.isEmpty else {
            throw AgentError.storage("本书没有可汇总的章节摘要")
        }

        let persisted = try store.bookSummarySegments(docID: document.docID)
        let persistedByIndex = Dictionary(
            persisted.map { ($0.groupIndex, $0) },
            uniquingKeysWith: { first, _ in first })
        var groupSummaries = [String?](repeating: nil, count: groups.count)
        var requests: [SummaryRequest] = []
        for (groupIndex, group) in groups.enumerated() {
            try cancellation.check()
            let hash = Self.shortHash(group)
            if let reused = persistedByIndex[groupIndex],
               reused.groupHash == hash,
               !reused.summary.isEmpty {
                groupSummaries[groupIndex] = reused.summary
                continue
            }
            let prompt = """
            以下是本书部分章节的摘要。请合并为 150-250 字的内容梗概，保留主线、结构与关键结论，只输出正文。

            \(group)
            """
            requests.append(SummaryRequest(
                index: groupIndex,
                system: "你是专业的书籍摘要助手，只输出正文。",
                prompt: prompt,
                maxLength: 300, preferredCut: 100,
                emptyMessage: "辅助模型未返回有效的分段摘要"))
        }

        let outcome = await runSummaryBatch(requests, llm: llm, cancellation: cancellation)
        for (groupIndex, summary) in outcome.summaries.sorted(by: { $0.key < $1.key }) {
            try store.upsertBookSummarySegment(BookSummarySegmentRecord(
                docID: document.docID, groupIndex: groupIndex,
                groupHash: Self.shortHash(groups[groupIndex]),
                summary: summary, updatedAt: now.timeIntervalSince1970))
            groupSummaries[groupIndex] = summary
        }
        if let failure = outcome.failure {
            throw failure.asError()
        }
        guard let summaries = Self.collectedSummaries(groupSummaries) else {
            throw AgentError.llm("辅助模型未返回有效的分段摘要")
        }

        try cancellation.check()
        let reducePrompt = """
        以下是全书各部分的梗概。请生成 300-500 字的中文全书摘要，覆盖主题、结构、核心观点与结论，只输出正文。

        \(summaries.enumerated().map { "\($0.offset + 1). \($0.element)" }
            .joined(separator: "\n\n"))
        """
        let raw = try await complete(
            system: "你是专业的书籍摘要助手，只输出正文。",
            prompt: reducePrompt, llm: llm, temperature: 0.2,
            cancellation: cancellation)
        var summary = Self.normalizeBookSummary(raw)
        if summary.count < 100 {
            // 模型偶尔只回一行标题或极短概览；再给一次明确的长摘要机会，
            // 但只接受更长的结果，避免重试反而更差。
            try cancellation.check()
            let retryPrompt = """
            上一次全书摘要过短。请输出 300-500 字的中文全书摘要，覆盖主题、结构、核心观点与结论，只输出正文。

            上一次结果：\(summary)
            """
            let retryRaw = try await complete(
                system: "你是专业的书籍摘要助手，只输出正文。",
                prompt: retryPrompt, llm: llm, temperature: 0.2,
                cancellation: cancellation)
            let retry = Self.normalizeBookSummary(retryRaw)
            if retry.count > summary.count { summary = retry }
        }
        // 只返回标签（如「摘要：」）或空内容时不能落库，否则任务会被当成完成，
        // 候选队列永远不再重试这本书。
        guard !summary.isEmpty else {
            throw AgentError.llm("辅助模型未返回有效的全书摘要")
        }
        let sourceHash = Self.shortHash(chapters.map {
            "\($0.sectionIndex):\($0.contentHash):\($0.summary)"
        }.joined(separator: "\n"))
        let record = BookSummaryRecord(
            docID: document.docID, summary: summary, outline: resolved.digest,
            chapterCount: sections.count, sourceHash: sourceHash,
            updatedAt: now.timeIntervalSince1970)
        try store.upsertBookSummary(record)
        try store.upsertSummaryJob(SummaryJobRecord(
            docID: document.docID, status: .completed,
            nextSection: max(sections.count, chapters.count),
            totalSections: sections.count,
            lastError: "", updatedAt: now.timeIntervalSince1970))
        return record
    }

    // MARK: - 辅助模型调用

    /// 并发执行一批 map 请求。每个请求都是关闭 thinking 的独立会话，
    /// 因此不会互相复用或污染 KV；`batchConcurrency` 限制同时占用的槽位。
    ///
    /// 后端支持批量生成时优先把整批请求合成一次模型调用（单线程 batched
    /// prefill/decode，见 `BatchedMLXGenerator`），避免多路并发 generation
    /// 撞上共享权重被 tracer 临时替换导致的崩溃；不支持或失败时回退到
    /// 逐条 TaskGroup 路径。
    private func runSummaryBatch(_ requests: [SummaryRequest],
                                 llm: LLMService,
                                 cancellation: SummaryCancellation) async -> SummaryBatchResult {
        guard !requests.isEmpty else { return SummaryBatchResult() }
        if let batchResult = await runBatchGeneration(
            requests, llm: llm, cancellation: cancellation) {
            return batchResult
        }
        let llmBox = SummaryLLMBox(llm)
        // 串行后端（GGUF/VLM）不支持多请求并行，退化为逐条执行，
        // 避免并发任务堆在内部互斥锁前空等。
        let limit = min(
            llm.supportsConcurrentIsolatedRequests ? batchConcurrency : 1,
            requests.count)
        var result = SummaryBatchResult()

        await withTaskGroup(of: SummaryBatchOutcome.self) { group in
            var next = 0
            for _ in 0..<limit {
                let request = requests[next]
                next += 1
                group.addTask { [self] in
                    await executeSummaryRequest(
                        request, llm: llmBox.value, cancellation: cancellation)
                }
            }
            while let outcome = await group.next() {
                switch outcome {
                case .success(let index, let summary):
                    result.summaries[index] = summary
                case .failure(let failure):
                    // 只保留首个失败原因，并取消仍在跑的其它分片。
                    if result.failure == nil { result.failure = failure }
                    group.cancelAll()
                }
                if result.failure == nil, next < requests.count, !cancellation.cancelled {
                    let request = requests[next]
                    next += 1
                    group.addTask { [self] in
                        await executeSummaryRequest(
                            request, llm: llmBox.value, cancellation: cancellation)
                    }
                }
            }
        }
        return result
    }

    /// 批量生成路径：整批一次模型调用，任一行失败或为空时返回 nil 回退逐条。
    private func runBatchGeneration(_ requests: [SummaryRequest],
                                    llm: LLMService,
                                    cancellation: SummaryCancellation) async -> SummaryBatchResult? {
        guard requests.count > 1 else { return nil }
        do {
            try cancellation.check()
            // 摘要上限按字符数；中文约 1 token/字，取 4 倍余量并封顶。
            let maxTokensPerRow = max(
                256, min(2048, (requests.map(\.maxLength).max() ?? 100) * 4))
            let batchRequests = requests.map { (system: $0.system, prompt: $0.prompt) }
            guard
                let raw = try await llm.generateSummariesBatch(
                    requests: batchRequests,
                    temperature: 0.2,
                    maxTokensPerRow: maxTokensPerRow)
            else { return nil }

            var result = SummaryBatchResult()
            for (request, text) in zip(requests, raw) {
                let summary = Self.normalizeSummary(
                    text, maxLength: request.maxLength,
                    preferredCut: request.preferredCut)
                if summary.isEmpty { return nil }
                result.summaries[request.index] = summary
            }
            return result
        } catch {
            // 取消（Task.cancel / 协调器闭包）不重跑逐条路径，直接按取消返回；
            // 其它错误（不支持批量、生成异常）回退到逐条路径复跑。
            if error is CancellationError || cancellation.isCancellation(error) {
                var result = SummaryBatchResult()
                result.failure = SummaryBatchFailure(
                    isCancelled: true, message: "摘要生成已取消")
                return result
            }
            return nil
        }
    }

    private func executeSummaryRequest(_ request: SummaryRequest,
                                       llm: LLMService,
                                       cancellation: SummaryCancellation) async -> SummaryBatchOutcome {
        do {
            try cancellation.check()
            let raw = try await complete(
                system: request.system, prompt: request.prompt,
                llm: llm, temperature: 0.2, cancellation: cancellation)
            let summary = Self.normalizeSummary(
                raw, maxLength: request.maxLength,
                preferredCut: request.preferredCut)
            guard !summary.isEmpty else {
                throw AgentError.llm(request.emptyMessage)
            }
            return .success(index: request.index, summary: summary)
        } catch {
            return .failure(SummaryBatchFailure(
                isCancelled: cancellation.isCancellation(error),
                message: error.localizedDescription))
        }
    }

    /// 摘要专用调用：显式关闭 thinking，并使用不共享 KV 的独立会话。
    private func complete(system: String, prompt: String, llm: LLMService,
                          temperature: Double,
                          cancellation: SummaryCancellation) async throws -> String {
        try cancellation.check()
        let messages = [
            ChatMessage(role: .system, content: system),
            ChatMessage(role: .user, content: prompt),
        ]
        // 不依赖 `auxLocalLLMEnableThinking` 的全局默认值：用户可能为前台对话
        // 打开思考模式，但摘要走 thinking 只会更慢且不一定更好。
        let options = LLMRequestOptions(enableThinking: false, isolated: true)
        var output = ""
        for try await event in llm.chatStream(messages: messages, tools: [],
                                              temperature: temperature,
                                              options: options) {
            try cancellation.check()
            if event.kind == .content { output += event.content }
        }
        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw AgentError.llm("辅助模型未返回摘要内容")
        }
        return trimmed
    }

    // MARK: - 工具

    /// 第一个没有“内容哈希匹配且非空”摘要的章节下标；全部有效时返回总数。
    static func firstMissingSectionIndex(sections: [BookSection],
                                         storedByIndex: [Int: ChapterSummaryRecord]) -> Int {
        for (offset, section) in sections.enumerated() {
            guard let row = storedByIndex[section.index],
                  row.contentHash == section.contentHash,
                  !row.summary.isEmpty else { return offset }
        }
        return sections.count
    }

    private static func segmentBudget(contextWindow: Int) -> Int {
        min(maxSegmentBudget, max(1024, contextWindow - 1400))
    }

    /// 按顺序收集分片摘要；任何缺口返回 nil，避免用缺段的结果生成摘要。
    private static func collectedSummaries(_ values: [String?]) -> [String]? {
        var result: [String] = []
        result.reserveCapacity(values.count)
        for value in values {
            guard let value else { return nil }
            result.append(value)
        }
        return result
    }

    private static func shortHash(_ text: String) -> String {
        let data = Data(text.utf8)
        return SHA256.hash(data: data).prefix(8)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private static func tokenCost(_ character: Character) -> Double {
        isCJK(character) ? 1.0 : 0.25
    }

    private static func isSentenceEnd(_ character: Character) -> Bool {
        "。！？；.!?;\n".contains(character)
    }

    private static func isCJK(_ character: Character) -> Bool {
        guard character.unicodeScalars.count == 1,
              let scalar = character.unicodeScalars.first else { return false }
        switch scalar.value {
        case 0x2E80...0x9FFF, 0xF900...0xFAFF, 0xFF00...0xFFEF,
             0x3040...0x30FF, 0xAC00...0xD7AF, 0x20000...0x2FA1F:
            return true
        default:
            return false
        }
    }
}

// MARK: - 摘要批次

/// 统一 `Task.isCancelled` 与协调器传入的取消闭包。
private final class SummaryCancellation: @unchecked Sendable {
    private let isCancelled: () -> Bool

    init(_ isCancelled: @escaping () -> Bool) {
        self.isCancelled = isCancelled
    }

    var cancelled: Bool {
        Task.isCancelled || isCancelled()
    }

    func check() throws {
        if cancelled { throw AgentError.cancelled }
    }

    func isCancellation(_ error: Error) -> Bool {
        if cancelled { return true }
        if case AgentError.cancelled = error { return true }
        return false
    }
}

/// 一次 map 分片的全部输入；由父任务预构建，避免并发任务读取可变状态。
private struct SummaryRequest: Sendable {
    let index: Int
    let system: String
    let prompt: String
    let maxLength: Int
    let preferredCut: Int
    let emptyMessage: String
}

private struct SummaryBatchFailure: Sendable {
    let isCancelled: Bool
    let message: String

    func asError() -> Error {
        isCancelled ? AgentError.cancelled : AgentError.llm(message)
    }
}

private enum SummaryBatchOutcome: Sendable {
    case success(index: Int, summary: String)
    case failure(SummaryBatchFailure)
}

private struct SummaryBatchResult {
    var summaries: [Int: String] = [:]
    var failure: SummaryBatchFailure?
}

/// 把 LLMService 引用交给并发任务；服务实例自身仍需线程安全。
private final class SummaryLLMBox: @unchecked Sendable {
    let value: LLMService

    init(_ value: LLMService) {
        self.value = value
    }
}
