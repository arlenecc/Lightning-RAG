import XCTest
@testable import BaseAgentCore

/// 记录每次调用、请求选项与并发峰值的辅助模型桩。
private final class SummaryStubLLM: LLMService, @unchecked Sendable {
    let identifier = "summary-stub"
    let contextWindow: Int
    private let concurrent: Bool
    private let lock = NSLock()
    private var prompts: [String] = []
    private var optionsStorage: [LLMRequestOptions] = []
    private var callCountStorage = 0
    private var activeStorage = 0
    private var peakStorage = 0
    private let failOnCall: Int?
    private let response: String
    private let responseForCall: ((Int) -> String)?
    private let delay: TimeInterval

    init(contextWindow: Int = 32768, failOnCall: Int? = nil,
         response: String = String(repeating: "章节内容概括。", count: 8),
         responseForCall: ((Int) -> String)? = nil,
         delay: TimeInterval = 0, concurrent: Bool = true) {
        self.contextWindow = contextWindow
        self.failOnCall = failOnCall
        self.response = response
        self.responseForCall = responseForCall
        self.delay = delay
        self.concurrent = concurrent
    }

    var supportsConcurrentIsolatedRequests: Bool { concurrent }

    var calls: Int {
        lock.lock()
        defer { lock.unlock() }
        return callCountStorage
    }

    var recordedPrompts: [String] {
        lock.lock()
        defer { lock.unlock() }
        return prompts
    }

    var recordedOptions: [LLMRequestOptions] {
        lock.lock()
        defer { lock.unlock() }
        return optionsStorage
    }

    var peakConcurrency: Int {
        lock.lock()
        defer { lock.unlock() }
        return peakStorage
    }

    private func enterActive() {
        lock.lock()
        activeStorage += 1
        peakStorage = max(peakStorage, activeStorage)
        lock.unlock()
    }

    private func leaveActive() {
        lock.lock()
        activeStorage -= 1
        lock.unlock()
    }

    func chatStream(messages: [ChatMessage], tools: [JSONValue],
                    temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        chatStream(messages: messages, tools: tools,
                   temperature: temperature, options: .default)
    }

    func chatStream(messages: [ChatMessage], tools: [JSONValue],
                    temperature: Double,
                    options: LLMRequestOptions) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        lock.lock()
        callCountStorage += 1
        let current = callCountStorage
        prompts.append(messages.last?.content ?? "")
        optionsStorage.append(options)
        let shouldFail = failOnCall == current
        let text = responseForCall?(current) ?? response
        lock.unlock()
        enterActive()

        return AsyncThrowingStream { continuation in
            if shouldFail {
                leaveActive()
                continuation.finish(throwing: AgentError.cancelled)
                return
            }
            if delay <= 0 {
                leaveActive()
                continuation.yield(LLMStreamEvent(kind: .content, content: text))
                continuation.yield(LLMStreamEvent(kind: .done))
                continuation.finish()
                return
            }
            let task = Task {
                try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                leaveActive()
                continuation.yield(LLMStreamEvent(kind: .content, content: text))
                continuation.yield(LLMStreamEvent(kind: .done))
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

final class BookSummaryEngineTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-summary-engine-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func makeStore() throws -> VectorStore {
        try VectorStore(dbPath: directory.appendingPathComponent("index.sqlite").path,
                        dimension: 2, enableSQLiteVec: false)
    }

    @discardableResult
    private func insertDocument(_ store: VectorStore, name: String = "book.md",
                                contentHash: String = "content-hash") throws -> DocumentRecord {
        let mdPath = directory.appendingPathComponent(name).path
        try store.upsertDocument(docName: name, digest: "# 目录\n\n" + name,
                                 chapters: "[]", mdPath: mdPath,
                                 contentHash: contentHash)
        return try XCTUnwrap(store.document(docID: VectorStore.hashSource(mdPath)))
    }

    func testTokenEstimateTreatsCJKAndLatinDifferently() throws {
        XCTAssertEqual(BookSummaryEngine.estimateTokens(""), 0)
        XCTAssertEqual(BookSummaryEngine.estimateTokens("中文摘要"), 4)
        XCTAssertEqual(BookSummaryEngine.estimateTokens("abcd"), 1)
        XCTAssertEqual(BookSummaryEngine.estimateTokens("ab中文"), 3)
    }

    func testSegmentSplitterRespectsBudgetAndKeepsOrder() throws {
        let first = String(repeating: "甲", count: 40)
        let second = String(repeating: "乙", count: 40)
        let third = String(repeating: "丙", count: 40)
        let body = [first, second, third].joined(separator: "\n\n")

        let segments = BookSummaryEngine.splitSegments(body, tokenBudget: 100)
        XCTAssertEqual(segments, [first + "\n\n" + second, third])
        XCTAssertEqual(segments.joined(separator: "\n\n"), body)
        for segment in segments {
            XCTAssertLessThanOrEqual(BookSummaryEngine.estimateTokens(segment), 100)
        }

        let longParagraph = String(repeating: "甲", count: 250)
        let pieces = BookSummaryEngine.splitSegments(longParagraph, tokenBudget: 100)
        XCTAssertEqual(pieces.count, 3)
        XCTAssertEqual(pieces.joined(), longParagraph)
    }

    func testLongChapterSegmentsRespectCharacterBudget() throws {
        let body = String(repeating: "甲", count: 3_500)
        let segments = BookSummaryEngine.splitSegments(
            body, tokenBudget: 8_000, characterBudget: 3_000)
        XCTAssertEqual(segments.map(\.count), [3_000, 500])
        XCTAssertEqual(segments.joined(), body)
    }

    /// 大表格是一整段无空行文本，分段必须线性完成；旧实现每轮复制剩余文本，
    /// 500k 字符在本地会超过 3 秒，并在 20M 字符上限下退化为小时级。
    func testLongSingleParagraphSplitsInLinearTime() throws {
        let body = String(repeating: "甲", count: 500_000)
        let started = Date()
        let segments = BookSummaryEngine.splitSegments(
            body, tokenBudget: 8_000, characterBudget: 3_000)
        let elapsed = Date().timeIntervalSince(started)

        XCTAssertEqual(segments.count, 167)
        XCTAssertEqual(segments.joined(), body)
        for segment in segments {
            XCTAssertLessThanOrEqual(segment.count, 3_000)
        }
        XCTAssertLessThan(elapsed, 1.5, "500k 单段落分段耗时 \(elapsed)s，疑似重复扫描剩余文本")
    }

    /// `digest` 为控制提示长度会截断；章节树仍保存完整大纲，摘要解析应优先使用它，
    /// 否则带标点的旧版 Office 标题在第一次摘要时会被丢掉。
    func testResolveUsesFullChaptersBeyondTruncatedDigest() throws {
        let store = try makeStore()
        let markdown = """
        ## Sheet: 汇总？

        汇总内容。

        ## Sheet: 明细？

        明细内容。

        ## Sheet: 附录？

        附录内容。
        """
        let chapters = """
        [{"title":"Sheet: 汇总？","level":2},
         {"title":"Sheet: 明细？","level":2},
         {"title":"Sheet: 附录？","level":2}]
        """
        let record = DocumentRecord(
            docID: VectorStore.hashSource("workbook.xls"),
            docName: "workbook.xls",
            digest: "# 目录\n\nSheet: 汇总？",
            chapters: chapters)
        let resolved = BookSummaryEngine(store: store).resolve(record, markdown: markdown)

        XCTAssertEqual(resolved.sections.count, 3)
        XCTAssertEqual(resolved.nodes.map(\.title),
                       ["Sheet: 汇总？", "Sheet: 明细？", "Sheet: 附录？"])
    }

    /// 无标题纯正文（旧版 Word 没有识别到任何标题样式）不能因为章节为空
    /// 就跳过全书摘要：整份 Markdown 作为唯一的「全文」章节参与汇总。
    func testResolveWithoutHeadingsUsesWholeTextAsSingleSection() throws {
        let store = try makeStore()
        let document = try insertDocument(store, name: "untitled.doc")
        let markdown = "这是一份没有标题的文档。\n\n"
            + String(repeating: "正文内容。", count: 30)
        let resolved = BookSummaryEngine(store: store).resolve(document, markdown: markdown)

        XCTAssertTrue(resolved.nodes.isEmpty)
        XCTAssertEqual(resolved.sections.count, 1)
        let section = try XCTUnwrap(resolved.sections.first)
        XCTAssertEqual(section.index, 0)
        XCTAssertEqual(section.level, 1)
        XCTAssertEqual(section.title, "全文")
        XCTAssertEqual(section.heading, "全文")
        XCTAssertEqual(section.body, markdown)
        XCTAssertFalse(section.contentHash.isEmpty)
        // 合成章节只服务于摘要，不能把「全文」写回入库目录。
        XCTAssertEqual(resolved.digest, document.digest)
        XCTAssertEqual(resolved.chaptersJSON, document.chapters)

        // 真正没有正文时仍保持无章节，避免产生无意义的摘要。
        XCTAssertTrue(BookSummaryEngine(store: store)
            .resolve(document, markdown: " \n\t").sections.isEmpty)
    }

    func testChapterUnderFourThousandCharactersStaysSingleMap() async throws {
        let store = try makeStore()
        let document = try insertDocument(store)
        let engine = BookSummaryEngine(store: store)
        let markdown = "# 第一章 概述\n\n" + String(repeating: "正文内容。", count: 780)

        let stub = SummaryStubLLM()
        let result = try await engine.processNextChapter(
            document: document, markdown: markdown, llm: stub,
            now: Date(), isCancelled: { false })

        XCTAssertTrue(result.chapterCompleted)
        // 单段 map + 一次 reduce，不额外切分。
        XCTAssertEqual(stub.calls, 2)
        XCTAssertEqual(try store.chapterSegments(
            docID: document.docID, sectionIndex: 0).count, 1)
    }

    func testChapterOverFourThousandCharactersMapsAndCapsAtFive() async throws {
        let store = try makeStore()
        let document = try insertDocument(store)
        let engine = BookSummaryEngine(store: store)
        let paragraphs = ["甲", "乙", "丙", "丁", "戊", "己", "庚"].map {
            String(repeating: $0, count: 3_500)
        }
        let markdown = "# 第一章 概述\n\n" + paragraphs.joined(separator: "\n\n")

        let stub = SummaryStubLLM(delay: 0.05)
        let result = try await engine.processNextChapter(
            document: document, markdown: markdown, llm: stub,
            now: Date(), isCancelled: { false })

        XCTAssertTrue(result.chapterCompleted)
        // 7 个 3500 字段落，每个按 3000 + 500 切分。
        XCTAssertEqual(try store.chapterSegments(
            docID: document.docID, sectionIndex: 0).count, 14)
        XCTAssertGreaterThanOrEqual(stub.peakConcurrency, 4)
        XCTAssertLessThanOrEqual(stub.peakConcurrency, 5,
                                 "并发 map 不应超过 5 个")
    }

    func testChapterSummaryReusesPersistedSegmentsAfterInterruption() async throws {
        let store = try makeStore()
        let document = try insertDocument(store)
        let engine = BookSummaryEngine(store: store)
        let paragraphs = ["甲", "乙", "丙"].map {
            String(repeating: $0, count: 600)
        }
        let markdown = "# 第一章 概述\n\n" + paragraphs.joined(separator: "\n\n")

        let specs = BookSummaryEngine.splitSegments(
            paragraphs.joined(separator: "\n\n"), tokenBudget: 1024)
        XCTAssertEqual(specs.count, 3)

        let interrupted = SummaryStubLLM(contextWindow: 1600, failOnCall: 2)
        do {
            _ = try await engine.processNextChapter(
                document: document, markdown: markdown, llm: interrupted,
                now: Date(), isCancelled: { false })
            XCTFail("第二个分段调用应被中断")
        } catch {
            // 并发 map 下失败发生在第 2 次调用后，仍在跑的其它分片可能已成功。
            XCTAssertGreaterThanOrEqual(interrupted.calls, 2)
            XCTAssertLessThanOrEqual(interrupted.calls, 3)
        }
        let persistedCount = try store.chapterSegments(
            docID: document.docID, sectionIndex: 0).count
        XCTAssertGreaterThanOrEqual(persistedCount, 1)
        XCTAssertLessThanOrEqual(persistedCount, 2)

        let resumed = SummaryStubLLM(contextWindow: 1600)
        let result = try await engine.processNextChapter(
            document: document, markdown: markdown, llm: resumed,
            now: Date(), isCancelled: { false })
        // 已落库的分段不再调用；剩余 map 加一次 reduce。
        XCTAssertEqual(resumed.calls, 4 - persistedCount)
        XCTAssertTrue(result.chapterCompleted)
        XCTAssertEqual(try store.chapterSegments(docID: document.docID, sectionIndex: 0).count, 3)
        XCTAssertFalse(result.summary.isEmpty)
        for options in interrupted.recordedOptions + resumed.recordedOptions {
            XCTAssertEqual(options.enableThinking, false)
            XCTAssertTrue(options.isolated)
        }
    }

    func testChapterSegmentMapRunsConcurrently() async throws {
        let store = try makeStore()
        let document = try insertDocument(store)
        let engine = BookSummaryEngine(store: store)
        let paragraphs = ["甲", "乙", "丙", "丁", "戊"].map {
            String(repeating: $0, count: 600)
        }
        let markdown = "# 第一章 概述\n\n" + paragraphs.joined(separator: "\n\n")

        let stub = SummaryStubLLM(contextWindow: 1600, delay: 0.1)
        let result = try await engine.processNextChapter(
            document: document, markdown: markdown, llm: stub,
            now: Date(), isCancelled: { false })

        XCTAssertTrue(result.chapterCompleted)
        XCTAssertGreaterThanOrEqual(stub.peakConcurrency, 4,
                                    "默认 5 个分片应并行执行而不是逐段串行")
        XCTAssertLessThanOrEqual(stub.peakConcurrency, 5,
                                 "并发分片数应受 batchConcurrency 限制")
        XCTAssertEqual(try store.chapterSegments(docID: document.docID, sectionIndex: 0).count, 5)
    }

    func testSerialBackendFallsBackToSequentialMap() async throws {
        let store = try makeStore()
        let document = try insertDocument(store)
        let engine = BookSummaryEngine(store: store)
        let paragraphs = ["甲", "乙", "丙"].map {
            String(repeating: $0, count: 600)
        }
        let markdown = "# 第一章 概述\n\n" + paragraphs.joined(separator: "\n\n")

        let stub = SummaryStubLLM(contextWindow: 1600, delay: 0.05,
                                   concurrent: false)
        let result = try await engine.processNextChapter(
            document: document, markdown: markdown, llm: stub,
            now: Date(), isCancelled: { false })

        XCTAssertTrue(result.chapterCompleted)
        XCTAssertEqual(stub.peakConcurrency, 1,
                       "单 context 后端不应并发提交分片")
        XCTAssertEqual(try store.chapterSegments(docID: document.docID, sectionIndex: 0).count, 3)
    }

    func testChapterSummaryNormalizesToFiftyToOneHundredCharacters() throws {
        let raw = "**摘要：**\n\n" + String(repeating: "本章介绍研究方法与实验设计。", count: 20)
        let normalized = BookSummaryEngine.normalizeChapterSummary(raw)
        XCTAssertFalse(normalized.contains("**"))
        XCTAssertFalse(normalized.hasPrefix("摘要"))
        XCTAssertGreaterThanOrEqual(normalized.count, 50)
        XCTAssertLessThanOrEqual(normalized.count, 100)

        XCTAssertEqual(BookSummaryEngine.normalizeChapterSummary("  \"简短内容。\"  "),
                       "简短内容。")
    }

    func testTinySectionSkipsModelCall() async throws {
        let store = try makeStore()
        let document = try insertDocument(store)
        let engine = BookSummaryEngine(store: store)
        let markdown = "# 第一章 概述\n\n很短的正文。"
        let stub = SummaryStubLLM(failOnCall: 1, response: "不应被使用")

        let result = try await engine.processNextChapter(
            document: document, markdown: markdown, llm: stub,
            now: Date(), isCancelled: { false })
        XCTAssertEqual(stub.calls, 0)
        XCTAssertFalse(result.summary.isEmpty)
        XCTAssertTrue(result.chapterCompleted)

        let saved = try XCTUnwrap(store.chapterSummary(docID: document.docID, sectionIndex: 0))
        XCTAssertEqual(saved.summary, result.summary)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.status, .summarizingBook)
    }

    func testBookSummaryMapReducePersistsGroupsAndProducesLongSummary() async throws {
        let store = try makeStore()
        let document = try insertDocument(store)
        let engine = BookSummaryEngine(store: store)
        let sections = 30
        let markdown = (0..<sections).map { index in
            "# 第\(index + 1)节\n\n" + String(repeating: "正文内容。", count: 20)
        }.joined(separator: "\n\n")

        let resolved = engine.resolve(document, markdown: markdown)
        XCTAssertEqual(resolved.sections.count, sections)
        for section in resolved.sections {
            try store.upsertChapterSummary(ChapterSummaryRecord(
                docID: document.docID, sectionIndex: section.index,
                heading: section.heading, title: section.title,
                level: section.level,
                summary: String(repeating: "章节摘要内容。", count: 10),
                contentHash: section.contentHash))
        }

        let stub = SummaryStubLLM(
            contextWindow: 1600,
            response: String(repeating: "全书内容概括。", count: 50))
        let record = try await engine.processBookSummary(
            document: document, markdown: markdown, llm: stub,
            now: Date(), isCancelled: { false })

        let groups = try store.bookSummarySegments(docID: document.docID)
        XCTAssertGreaterThanOrEqual(groups.count, 2)
        XCTAssertGreaterThanOrEqual(record.summary.count, 300)
        XCTAssertLessThanOrEqual(record.summary.count, 500)
        XCTAssertFalse(record.outline.isEmpty)
        XCTAssertEqual(record.chapterCount, sections)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.status, .completed)
        XCTAssertGreaterThanOrEqual(stub.calls, groups.count + 1)
        XCTAssertGreaterThan(groups.count, 1)
        for options in stub.recordedOptions {
            XCTAssertEqual(options.enableThinking, false,
                           "全书 map-reduce 也必须强制关闭 thinking")
            XCTAssertTrue(options.isolated)
        }
    }

    func testBookSummaryReturnsToChapterPhaseWhenChapterContentChanged() async throws {
        let store = try makeStore()
        let document = try insertDocument(store)
        let engine = BookSummaryEngine(store: store)
        let markdown = (0..<2).map { index in
            "# 第\(index + 1)章\n\n" + String(repeating: "正文内容。", count: 20)
        }.joined(separator: "\n\n")
        let resolved = engine.resolve(document, markdown: markdown)
        XCTAssertEqual(resolved.sections.count, 2)
        for section in resolved.sections {
            try store.upsertChapterSummary(ChapterSummaryRecord(
                docID: document.docID, sectionIndex: section.index,
                heading: section.heading, title: section.title,
                level: section.level, summary: "过期章节摘要",
                contentHash: "stale-\(section.index)"))
        }
        try store.upsertSummaryJob(SummaryJobRecord(
            docID: document.docID, status: .summarizingBook,
            nextSection: 2, totalSections: 2))
        let stub = SummaryStubLLM(
            response: String(repeating: "全书内容概括。", count: 50))

        do {
            _ = try await engine.processBookSummary(
                document: document, markdown: markdown, llm: stub,
                now: Date(), isCancelled: { false })
            XCTFail("章节正文与摘要不一致时不应生成全书摘要")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("逐章"))
        }
        XCTAssertEqual(stub.calls, 0)
        XCTAssertNil(try store.bookSummary(docID: document.docID))
        let job = try store.summaryJob(docID: document.docID)
        XCTAssertEqual(job?.status, .pending)
        XCTAssertEqual(job?.nextSection, 0)
    }

    func testBookSummaryRejectsLabelOnlyOutputWithoutPersistingEmptySummary() async throws {
        let store = try makeStore()
        let document = try insertDocument(store)
        let engine = BookSummaryEngine(store: store)
        let markdown = (0..<2).map { index in
            "# 第\(index + 1)章\n\n" + String(repeating: "正文内容。", count: 20)
        }.joined(separator: "\n\n")
        let resolved = engine.resolve(document, markdown: markdown)
        for section in resolved.sections {
            try store.upsertChapterSummary(ChapterSummaryRecord(
                docID: document.docID, sectionIndex: section.index,
                heading: section.heading, title: section.title,
                level: section.level,
                summary: String(repeating: "章节摘要内容。", count: 10),
                contentHash: section.contentHash))
        }
        try store.upsertSummaryJob(SummaryJobRecord(
            docID: document.docID, status: .summarizingBook,
            nextSection: resolved.sections.count,
            totalSections: resolved.sections.count))
        let stub = SummaryStubLLM(responseForCall: { call in
            // 第一组（唯一一组）正常，最终 reduce 与重试只返回标签。
            call == 1 ? String(repeating: "全书内容概括。", count: 50) : "摘要："
        })

        do {
            _ = try await engine.processBookSummary(
                document: document, markdown: markdown, llm: stub,
                now: Date(), isCancelled: { false })
            XCTFail("规范化后为空的全书摘要不应写入数据库")
        } catch {}
        XCTAssertNil(try store.bookSummary(docID: document.docID))
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.status,
                       .summarizingBook)
        XCTAssertGreaterThanOrEqual(stub.calls, 2)
    }

    func testShrunkOutlineRechecksChapterSummariesBeforeBookStage() async throws {
        let store = try makeStore()
        let document = try insertDocument(store)
        let engine = BookSummaryEngine(store: store)
        let oldMarkdown = [
            "# 第一章 概述\n\n" + String(repeating: "旧版正文内容。", count: 20),
            "# 第二章 细节\n\n" + String(repeating: "旧版细节内容。", count: 20),
            "# 第三章 结论\n\n" + String(repeating: "旧版结论内容。", count: 20),
        ].joined(separator: "\n\n")
        let old = engine.resolve(document, markdown: oldMarkdown)
        XCTAssertEqual(old.sections.count, 3)
        for section in old.sections {
            try store.upsertChapterSummary(ChapterSummaryRecord(
                docID: document.docID, sectionIndex: section.index,
                heading: section.heading, title: section.title,
                level: section.level, summary: "旧版章节摘要",
                contentHash: section.contentHash))
        }
        try store.upsertSummaryJob(SummaryJobRecord(
            docID: document.docID, status: .summarizingBook,
            nextSection: 3, totalSections: 3))

        // 目录收缩且第一章正文变化：旧摘要全部失效，必须重新逐章摘要，
        // 不能因为 next_section 已越过新总数就直接进入全书摘要。
        let newMarkdown = "# 第一章 概述\n\n"
            + String(repeating: "全新正文内容。", count: 20)
        let stub = SummaryStubLLM()
        let result = try await engine.processNextChapter(
            document: document, markdown: newMarkdown, llm: stub,
            now: Date(), isCancelled: { false })

        XCTAssertEqual(result.totalSections, 1)
        XCTAssertGreaterThan(stub.calls, 0)
        XCTAssertFalse(result.summary.isEmpty)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.status,
                       .summarizingBook)
        XCTAssertEqual(try store.chapterSummary(docID: document.docID, sectionIndex: 0)?
            .summary, result.summary)
        XCTAssertTrue(try store.chapterSummaries(docID: document.docID)
            .allSatisfy { $0.sectionIndex < 1 },
            "目录收缩后越界的旧摘要应被清理")
    }

    func testUntitledDocumentProducesChapterAndBookSummary() async throws {
        let store = try makeStore()
        let document = try insertDocument(store, name: "untitled.doc")
        let engine = BookSummaryEngine(store: store)
        let markdown = String(repeating: "这是一段没有标题的正文内容。", count: 40)

        let chapterStub = SummaryStubLLM()
        let chapter = try await engine.processNextChapter(
            document: document, markdown: markdown, llm: chapterStub,
            now: Date(), isCancelled: { false })
        XCTAssertEqual(chapter.sectionIndex, 0)
        XCTAssertEqual(chapter.totalSections, 1)
        XCTAssertTrue(chapter.chapterCompleted)
        XCTAssertFalse(chapter.bookCompleted)
        XCTAssertFalse(chapter.summary.isEmpty)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.status, .summarizingBook)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.nextSection, 1)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.totalSections, 1)

        let bookStub = SummaryStubLLM(
            response: String(repeating: "全书内容概括。", count: 50))
        let record = try await engine.processBookSummary(
            document: document, markdown: markdown, llm: bookStub,
            now: Date(), isCancelled: { false })
        XCTAssertEqual(record.chapterCount, 1)
        XCTAssertFalse(record.summary.isEmpty)
        XCTAssertTrue(record.outline.contains("untitled.doc"))
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.status, .completed)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.totalSections, 1)
        XCTAssertEqual(try store.bookSummary(docID: document.docID)?.summary, record.summary)
    }

    /// 入库批处理：一次调用处理完所有缺失章节，随后可生成全书摘要。
    func testBatchChapterProcessingCompletesAllSections() async throws {
        let store = try makeStore()
        let document = try insertDocument(store, name: "batch.md")
        let engine = BookSummaryEngine(store: store, summaryMode: .coreSentences)
        let markdown = """
        # 第一章 起点

        核心观点是问题必须被清晰定义。因此，我们先梳理背景与目标。
        这一段补充细节，说明范围与限制条件。

        # 第二章 推进

        实际上，方法的选择取决于约束条件。因此需要依次验证关键假设。
        这一章继续展开论据与实现路径。

        # 第三章 结论

        综上所述，结论应该由证据支撑。最后总结可复用的经验。
        这一段用于补充边界情况与后续工作。
        """

        let result = try await engine.processNextChapter(
            document: document, markdown: markdown, llm: nil,
            now: Date(), isCancelled: { false }, maxChapters: Int.max)

        XCTAssertTrue(result.chapterCompleted)
        XCTAssertEqual(result.totalSections, 3)
        let chapters = try store.chapterSummaries(docID: document.docID)
        XCTAssertEqual(chapters.count, 3)
        XCTAssertTrue(chapters.allSatisfy { !$0.summary.isEmpty })
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.status, .summarizingBook)

        let book = try await engine.processBookSummary(
            document: document, markdown: markdown, llm: nil,
            now: Date(), isCancelled: { false })
        XCTAssertFalse(book.summary.isEmpty)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.status, .completed)
    }

    /// 核心句模式：不依赖辅助模型，章节与全书摘要都由本地核心句组成。
    func testCoreSentenceModeWorksWithoutAuxiliaryModel() async throws {
        let store = try makeStore()
        let document = try insertDocument(store, name: "core-mode.md")
        let engine = BookSummaryEngine(store: store, summaryMode: .coreSentences)
        let paragraphs = [
            "核心观点句必须放在段首。",
            "这是一段用于补充细节的普通叙述，不承担主要观点。",
            "因此，全书的主线可以概括为从问题到方法的推进。",
            "另一个细节例子只是用来解释背景。",
            "最后总结一句：结论清晰且可验证。",
        ]
        let markdown = "# 第一章 概述\n\n"
            + String(repeating: paragraphs.joined(separator: "\n\n"), count: 4)

        let chapter = try await engine.processNextChapter(
            document: document, markdown: markdown, llm: nil,
            now: Date(), isCancelled: { false })
        XCTAssertTrue(chapter.chapterCompleted)
        XCTAssertFalse(chapter.summary.isEmpty)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.status, .summarizingBook)

        let record = try await engine.processBookSummary(
            document: document, markdown: markdown, llm: nil,
            now: Date(), isCancelled: { false })
        XCTAssertFalse(record.summary.isEmpty)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.status, .completed)
    }
}
