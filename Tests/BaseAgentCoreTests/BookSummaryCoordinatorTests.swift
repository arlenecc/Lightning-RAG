import XCTest
@testable import BaseAgentCore

private final class SummaryClock: @unchecked Sendable {
    var now: Date
    init(_ now: Date) { self.now = now }
}

/// 协调器测试桩：可延迟、可失败，并记录 preload/unload/close 生命周期。
private final class CoordinatorStubLLM: LLMService, PreloadableLLMService,
    AsyncUnloadableLLMService, ClosableLLMService, KVCacheInvalidating,
    @unchecked Sendable {
    let identifier = "coordinator-stub"
    let contextWindow = 32768
    let supportsConcurrentIsolatedRequests = true
    private let lock = NSLock()
    private let delay: TimeInterval
    private let failFirstCalls: Int
    private let failPreload: Bool
    private let response: String
    private var callsStorage = 0
    private var preloadedStorage = false
    private var unloadedStorage = false
    private var closedStorage = false
    private var invalidationsStorage = 0

    init(delay: TimeInterval = 0, failFirstCalls: Int = 0,
         failPreload: Bool = false,
         response: String = String(repeating: "章节摘要内容。", count: 8)) {
        self.delay = delay
        self.failFirstCalls = failFirstCalls
        self.failPreload = failPreload
        self.response = response
    }

    var calls: Int {
        lock.lock()
        defer { lock.unlock() }
        return callsStorage
    }

    var preloaded: Bool {
        lock.lock()
        defer { lock.unlock() }
        return preloadedStorage
    }

    var unloaded: Bool {
        lock.lock()
        defer { lock.unlock() }
        return unloadedStorage
    }

    var closed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return closedStorage
    }

    var invalidations: Int {
        lock.lock()
        defer { lock.unlock() }
        return invalidationsStorage
    }

    func invalidateKVCache() {
        lock.lock()
        invalidationsStorage += 1
        lock.unlock()
    }

    func preload() async throws {
        lock.lock()
        let shouldFail = failPreload
        if !shouldFail { preloadedStorage = true }
        lock.unlock()
        if shouldFail { throw AgentError.llm("辅助模型预加载失败") }
    }

    func unloadAndWait() async {
        lock.lock()
        unloadedStorage = true
        lock.unlock()
    }

    func close() {
        lock.lock()
        closedStorage = true
        lock.unlock()
    }

    func chatStream(messages: [ChatMessage], tools: [JSONValue],
                    temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        lock.lock()
        callsStorage += 1
        let call = callsStorage
        let shouldFail = call <= failFirstCalls
        let delay = self.delay
        let response = self.response
        lock.unlock()

        return AsyncThrowingStream { continuation in
            if shouldFail {
                continuation.finish(throwing: AgentError.llm("辅助模型测试失败"))
                return
            }
            if delay <= 0 {
                continuation.yield(LLMStreamEvent(kind: .content, content: response))
                continuation.yield(LLMStreamEvent(kind: .done))
                continuation.finish()
                return
            }
            Task {
                try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                continuation.yield(LLMStreamEvent(kind: .content, content: response))
                continuation.yield(LLMStreamEvent(kind: .done))
                continuation.finish()
            }
        }
    }
}

final class BookSummaryCoordinatorTests: XCTestCase {
    private var directory: URL!
    private var clock: SummaryClock!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-summary-coordinator-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        clock = SummaryClock(Date(timeIntervalSince1970: 1_000))
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func makeStore() throws -> VectorStore {
        try VectorStore(dbPath: directory.appendingPathComponent("index.sqlite").path,
                        dimension: 2, enableSQLiteVec: false)
    }

    private func insertDocument(_ store: VectorStore, markdown: String,
                               name: String = "book.md") throws -> DocumentRecord {
        let mdPath = directory.appendingPathComponent(name).path
        try markdown.write(toFile: mdPath, atomically: true, encoding: .utf8)
        try store.upsertDocument(docName: name, digest: "# 目录\n\n\(name)",
                                 chapters: "[]", mdPath: mdPath,
                                 contentHash: "content-\(name)")
        return try XCTUnwrap(store.document(docID: VectorStore.hashSource(mdPath)))
    }

    private func makeCoordinator(store: VectorStore, resident: LLMService?,
                                 transient: (@Sendable (AgentConfig) -> LLMService)? = nil)
        -> BookSummaryCoordinator {
        let clock = self.clock!
        return BookSummaryCoordinator(
            engine: BookSummaryEngine(store: store),
            store: store,
            configProvider: {
                // 这些用例验证辅助模型会话生命周期，显式走 aux_model 分支。
                var config = AgentConfig()
                config.bookSummaryMode = .auxModel
                return config
            },
            residentAuxProvider: { resident },
            transientAuxFactory: transient,
            now: { clock.now },
            log: { _ in })
    }

    private func chapterMarkdown(_ count: Int, bodyLength: Int = 100) -> String {
        (0..<count).map { index in
            "# 第\(index + 1)章 主题\(index + 1)\n\n"
                + String(repeating: "正文内容。", count: max(1, bodyLength / 6))
        }.joined(separator: "\n\n")
    }

    func testResidentModeProcessesOneChapterPerFifteenMinutes() async throws {
        let store = try makeStore()
        let markdown = chapterMarkdown(1)
        let document = try insertDocument(store, markdown: markdown)
        let resident = CoordinatorStubLLM()
        let coordinator = makeCoordinator(store: store, resident: resident)

        let began = await coordinator.beginSession(mode: .resident)
        XCTAssertTrue(began)
        let first = await coordinator.runTick(mode: .resident)
        guard case .processedChapter(_, _, let sectionIndex, _, let completed) = first else {
            return XCTFail("首个 tick 应处理章节，实际 \(first)")
        }
        XCTAssertEqual(sectionIndex, 0)
        XCTAssertTrue(completed)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.status, .summarizingBook)
        XCTAssertFalse(resident.preloaded)
        XCTAssertFalse(resident.closed)

        let second = await coordinator.runTick(mode: .resident)
        guard case .throttled(let wait) = second else {
            return XCTFail("常驻模式 15 分钟内应节流，实际 \(second)")
        }
        XCTAssertEqual(wait, 900, accuracy: 0.01)
    }

    /// 核心句模式：不创建/预加载辅助模型，直接本地处理章节。
    func testCoreSentenceModeNeedsNoAuxiliaryModel() async throws {
        let store = try makeStore()
        let markdown = chapterMarkdown(1, bodyLength: 200)
        _ = try insertDocument(store, markdown: markdown)
        let transient = CoordinatorStubLLM()
        var config = AgentConfig()
        config.bookSummaryMode = .coreSentences
        let clock = self.clock!
        let coordinator = BookSummaryCoordinator(
            engine: BookSummaryEngine(store: store, summaryMode: .coreSentences),
            store: store,
            configProvider: { config },
            residentAuxProvider: { nil },
            transientAuxFactory: { _ in transient },
            now: { clock.now },
            log: { _ in })

        let began = await coordinator.beginSession(mode: .temporary)
        XCTAssertTrue(began)
        XCTAssertFalse(transient.preloaded, "核心句模式不应预加载辅助模型")
        let first = await coordinator.runTick(mode: .temporary)
        guard case .processedChapter(_, _, _, _, let completed) = first else {
            return XCTFail("核心句模式应直接处理章节，实际 \(first)")
        }
        XCTAssertTrue(completed)
        XCTAssertEqual(transient.calls, 0, "核心句模式不应调用辅助模型")
        await coordinator.endSession()
    }

    func testTemporaryModeProcessesOneChapterPerFiveMinutes() async throws {
        let store = try makeStore()
        let markdown = chapterMarkdown(1)
        _ = try insertDocument(store, markdown: markdown)
        let transient = CoordinatorStubLLM()
        let coordinator = makeCoordinator(
            store: store, resident: nil, transient: { _ in transient })

        let began = await coordinator.beginSession(mode: .temporary)
        XCTAssertTrue(began)
        XCTAssertTrue(transient.preloaded)
        let first = await coordinator.runTick(mode: .temporary)
        guard case .processedChapter = first else {
            return XCTFail("临时模式首个 tick 应处理章节，实际 \(first)")
        }

        let second = await coordinator.runTick(mode: .temporary)
        guard case .throttled(let wait) = second else {
            return XCTFail("临时模式 5 分钟内应节流，实际 \(second)")
        }
        XCTAssertEqual(wait, 300, accuracy: 0.01)

        await coordinator.endSession()
        XCTAssertTrue(transient.unloaded)
        XCTAssertTrue(transient.closed)
    }

    func testTransientPreloadFailureWaitsForUnload() async throws {
        let store = try makeStore()
        _ = try insertDocument(store, markdown: chapterMarkdown(1))
        let transient = CoordinatorStubLLM(failPreload: true)
        let coordinator = makeCoordinator(
            store: store, resident: nil, transient: { _ in transient })

        let began = await coordinator.beginSession(mode: .temporary)
        XCTAssertFalse(began)
        // 只 close() 只是发起异步卸载；预加载失败路径也必须等待权重真正释放，
        // 否则退出流程可能与仍在执行的 GPU 工作并发。
        XCTAssertTrue(transient.unloaded)
        XCTAssertTrue(transient.closed)
        XCTAssertFalse(transient.preloaded)
    }

    func testTickNeverProcessesMoreThanOneChapter() async throws {
        let store = try makeStore()
        let markdown = chapterMarkdown(3)
        let document = try insertDocument(store, markdown: markdown)
        let resident = CoordinatorStubLLM()
        let coordinator = makeCoordinator(store: store, resident: resident)

        let outcome = await coordinator.runTick(mode: .resident)
        guard case .processedChapter(_, _, let sectionIndex, let total, _) = outcome else {
            return XCTFail("应处理一个章节，实际 \(outcome)")
        }
        XCTAssertEqual(sectionIndex, 0)
        XCTAssertEqual(total, 3)
        XCTAssertEqual(try store.chapterSummaries(docID: document.docID).count, 1)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.nextSection, 1)
        // 单章节正文一段：map + reduce 两次调用。
        XCTAssertEqual(resident.calls, 2)
    }

    func testCompletedBookMovesToBookSummarySlot() async throws {
        let store = try makeStore()
        let markdown = chapterMarkdown(3)
        let document = try insertDocument(store, markdown: markdown)
        let engine = BookSummaryEngine(store: store)
        let resolved = engine.resolve(document, markdown: markdown)
        XCTAssertEqual(resolved.sections.count, 3)
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
            nextSection: 3, totalSections: 3, updatedAt: 1_000))

        let resident = CoordinatorStubLLM(
            response: String(repeating: "全书内容概括。", count: 50))
        let coordinator = makeCoordinator(store: store, resident: resident)
        let outcome = await coordinator.runTick(mode: .resident)

        guard case .processedBook = outcome else {
            return XCTFail("章节完成后应生成全书摘要，实际 \(outcome)")
        }
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.status, .completed)
        XCTAssertEqual(try store.bookSummary(docID: document.docID)?.summary.count, 350)
    }

    /// 旧版无标题纯正文曾被直接写成 `completed / total_sections=0`。
    /// 候选重新纳入这类文档后，不能再把它当全书阶段调用注定失败的汇总，
    /// 应先补出「全文」章节摘要，再进入全书摘要。
    func testCompletedJobWithoutSectionsRetriesChapterPhase() async throws {
        let store = try makeStore()
        let markdown = String(repeating: "这是一段没有标题的正文内容。", count: 40)
        let document = try insertDocument(store, markdown: markdown,
                                          name: "untitled.doc")
        try store.upsertSummaryJob(SummaryJobRecord(
            docID: document.docID, status: .completed, nextSection: 0,
            totalSections: 0, updatedAt: 0))
        let resident = CoordinatorStubLLM()
        let coordinator = makeCoordinator(store: store, resident: resident)

        let outcome = await coordinator.runTick(mode: .resident)
        guard case .processedChapter(_, _, let sectionIndex, let total, _) = outcome else {
            return XCTFail("无标题文档应先补出章节摘要，实际 \(outcome)")
        }
        XCTAssertEqual(sectionIndex, 0)
        XCTAssertEqual(total, 1)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.status, .summarizingBook)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.totalSections, 1)
    }

    func testFailureDoesNotAdvanceNextSection() async throws {
        let store = try makeStore()
        let markdown = chapterMarkdown(1)
        let document = try insertDocument(store, markdown: markdown)
        let resident = CoordinatorStubLLM(failFirstCalls: 2)
        let coordinator = makeCoordinator(store: store, resident: resident)

        let first = await coordinator.runTick(mode: .resident)
        guard case .failed(_, let message) = first else {
            return XCTFail("首次失败应返回 failed，实际 \(first)")
        }
        XCTAssertFalse(message.isEmpty)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.nextSection, 0)
        XCTAssertEqual(try store.chapterSummaries(docID: document.docID).count, 0)
        XCTAssertNil(try store.summaryLastTickAt())

        let second = await coordinator.runTick(mode: .resident)
        guard case .failed = second else {
            return XCTFail("第二次失败应返回 failed，实际 \(second)")
        }
        let third = await coordinator.runTick(mode: .resident)
        XCTAssertEqual(third, .noWork)
        XCTAssertEqual(resident.calls, 2)
        XCTAssertEqual(try store.chapterSummaries(docID: document.docID).count, 0)
    }

    func testMissingMarkdownFailureRotatesCandidates() async throws {
        let store = try makeStore()
        // 先入库、后删除文件：模拟 Markdown 被移动/清理但索引记录仍在。
        let missingPath = directory.appendingPathComponent("missing.md").path
        try store.upsertDocument(docName: "missing.md", digest: "# 目录\n\nmissing.md",
                                 chapters: "[]", mdPath: missingPath,
                                 contentHash: "content-missing")
        let broken = try XCTUnwrap(
            store.document(docID: VectorStore.hashSource(missingPath)))
        let good = try insertDocument(store, markdown: chapterMarkdown(1),
                                      name: "good.md")
        let resident = CoordinatorStubLLM()
        let coordinator = makeCoordinator(store: store, resident: resident)

        let first = await coordinator.runTick(mode: .resident)
        guard case .failed(let failedName, let message) = first else {
            return XCTFail("缺失 Markdown 应失败，实际 \(first)")
        }
        XCTAssertEqual(failedName, "missing.md")
        XCTAssertFalse(message.isEmpty)
        let job = try XCTUnwrap(try store.summaryJob(docID: broken.docID))
        XCTAssertEqual(job.status, .failed)
        XCTAssertEqual(job.nextSection, 0)
        XCTAssertFalse(job.lastError.isEmpty)
        XCTAssertEqual(job.updatedAt, clock.now.timeIntervalSince1970)
        // 失败必须推进 updated_at 把文档沉到队尾，否则前 10 本持续失败的
        // 文档会在每个新会话里反复占满候选窗口，后面的书永远轮不到。
        XCTAssertEqual(try store.summaryCandidates(limit: 10).first?.docID, good.docID)

        let second = await coordinator.runTick(mode: .resident)
        guard case .processedChapter(_, let docName, _, _, _) = second else {
            return XCTFail("失败后应轮转到其他文档，实际 \(second)")
        }
        XCTAssertEqual(docName, "good.md")
        XCTAssertEqual(resident.calls, 2)
    }

    func testBookSummaryFailureKeepsBookStageAndRotates() async throws {
        let store = try makeStore()
        let markdown = chapterMarkdown(1)
        let document = try insertDocument(store, markdown: markdown)
        let engine = BookSummaryEngine(store: store)
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
            nextSection: 1, totalSections: 1, updatedAt: 0))
        let good = try insertDocument(store, markdown: chapterMarkdown(1),
                                      name: "good.md")
        let resident = CoordinatorStubLLM(failFirstCalls: 1)
        let coordinator = makeCoordinator(store: store, resident: resident)

        let first = await coordinator.runTick(mode: .resident)
        guard case .failed = first else {
            return XCTFail("全书摘要调用失败应返回 failed，实际 \(first)")
        }
        let job = try XCTUnwrap(try store.summaryJob(docID: document.docID))
        // 保留全书阶段：否则下一次 tick 要先跑一轮逐章检查才回到全书摘要。
        XCTAssertEqual(job.status, .summarizingBook)
        XCTAssertEqual(job.nextSection, 1)
        XCTAssertEqual(job.totalSections, 1)
        XCTAssertEqual(job.updatedAt, clock.now.timeIntervalSince1970)
        XCTAssertFalse(job.lastError.isEmpty)
        XCTAssertEqual(try store.summaryCandidates(limit: 10).first?.docID, good.docID)

        let second = await coordinator.runTick(mode: .resident)
        guard case .processedChapter(_, let docName, _, _, _) = second else {
            return XCTFail("失败后应轮转到其他文档，实际 \(second)")
        }
        XCTAssertEqual(docName, "good.md")
    }

    func testCancelStopsCurrentChapterAndKeepsSegments() async throws {
        let store = try makeStore()
        let paragraphs = ["甲", "乙", "丙"].map {
            String(repeating: $0, count: 600)
        }
        let markdown = "# 第一章 概述\n\n" + paragraphs.joined(separator: "\n\n")
        let document = try insertDocument(store, markdown: markdown)
        let transient = CoordinatorStubLLM(delay: 0.2)
        let coordinator = makeCoordinator(
            store: store, resident: nil, transient: { _ in transient })
        let began = await coordinator.beginSession(mode: .temporary)
        XCTAssertTrue(began)

        let tick = Task { await coordinator.runTick(mode: .temporary) }
        let deadline = Date().addingTimeInterval(2)
        while Date() < deadline {
            if (try? store.chapterSegments(docID: document.docID, sectionIndex: 0).count) ?? 0 >= 1 {
                break
            }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        await coordinator.cancelCurrent()
        let outcome = await tick.value

        guard case .failed(_, let message) = outcome else {
            return XCTFail("取消应结束当前 tick，实际 \(outcome)")
        }
        XCTAssertTrue(message.contains("取消"))
        XCTAssertEqual(try store.chapterSummaries(docID: document.docID).count, 0)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.nextSection, 0)
        let kept = try store.chapterSegments(docID: document.docID, sectionIndex: 0)
        XCTAssertGreaterThanOrEqual(kept.count, 1)

        await coordinator.endSession()
        XCTAssertTrue(transient.unloaded)
    }

    func testThrottleWaitIsClampedWhenLastTickIsInTheFuture() async throws {
        let store = try makeStore()
        _ = try insertDocument(store, markdown: chapterMarkdown(1))
        let resident = CoordinatorStubLLM()
        let coordinator = makeCoordinator(store: store, resident: resident)
        // 系统时钟回拨后 last_tick_at 可能落在未来；等待时间不能超过一个完整间隔。
        try store.setSummaryLastTickAt(clock.now.timeIntervalSince1970 + 86_400)

        let outcome = await coordinator.runTick(mode: .resident)
        guard case .throttled(let wait) = outcome else {
            return XCTFail("未来时间戳应触发节流，实际 \(outcome)")
        }
        XCTAssertEqual(wait, BookSummaryMode.resident.interval, accuracy: 0.01)
        XCTAssertEqual(resident.calls, 0)
    }

    func testTickInvalidatesAuxiliaryKVCacheAfterUse() async throws {
        let store = try makeStore()
        _ = try insertDocument(store, markdown: chapterMarkdown(1))
        let resident = CoordinatorStubLLM()
        let coordinator = makeCoordinator(store: store, resident: resident)

        let outcome = await coordinator.runTick(mode: .resident)
        guard case .processedChapter = outcome else {
            return XCTFail("应处理一个章节，实际 \(outcome)")
        }
        XCTAssertGreaterThanOrEqual(resident.invalidations, 1,
                                    "一次 tick 结束后应丢弃辅助模型的跨轮 KV cache")
    }

    func testCheckIdleWorkReflectsCandidateQueue() throws {
        let store = try makeStore()
        let coordinator = makeCoordinator(store: store, resident: nil)
        XCTAssertEqual(coordinator.checkIdleWork(), .noWork)

        _ = try insertDocument(store, markdown: chapterMarkdown(1))
        XCTAssertEqual(coordinator.checkIdleWork(), .ready)
    }

    func testCheckIdleWorkRotatesMissingMarkdown() throws {
        let store = try makeStore()
        let mdPath = directory.appendingPathComponent("gone.md").path
        try store.upsertDocument(docName: "gone.md", digest: "# 目录\n\ngone.md",
                                 chapters: "[]", mdPath: mdPath,
                                 contentHash: "content-gone")
        let docID = VectorStore.hashSource(mdPath)
        let coordinator = makeCoordinator(store: store, resident: nil)

        // 只有不可读候选时不应为必然失败的 tick 预加载临时模型。
        XCTAssertEqual(coordinator.checkIdleWork(), .noWork)
        // 但必须落库轮转，否则这些文档会永远占住候选窗口的队首。
        let job = try store.summaryJob(docID: docID)
        XCTAssertEqual(job?.status, .failed)
        XCTAssertEqual(job?.updatedAt ?? 0, clock.now.timeIntervalSince1970,
                       accuracy: 0.001)

        try chapterMarkdown(1).write(toFile: mdPath, atomically: true, encoding: .utf8)
        XCTAssertEqual(coordinator.checkIdleWork(), .ready)
    }

    func testCheckIdleWorkReachesReadableDocumentBehindMissingOnes() throws {
        let store = try makeStore()
        var missingIDs: [String] = []
        for index in 0..<10 {
            let name = "gone-\(index).md"
            let mdPath = directory.appendingPathComponent(name).path
            missingIDs.append(VectorStore.hashSource(mdPath))
            try store.upsertDocument(docName: name, digest: "# 目录\n\n\(name)",
                                     chapters: "[]", mdPath: mdPath,
                                     contentHash: "content-\(name)")
        }
        _ = try insertDocument(store, markdown: chapterMarkdown(1), name: "alive.md")
        let coordinator = makeCoordinator(store: store, resident: nil)

        // 前 10 个候选全部失联时也要能发现后面的可读文档，否则后面的书会饿死。
        XCTAssertEqual(coordinator.checkIdleWork(), .ready)
        for docID in missingIDs {
            XCTAssertEqual(try store.summaryJob(docID: docID)?.status, .failed,
                           "失联候选必须轮转，否则 tick 仍会反复选中它们")
        }
    }

    func testCheckIdleWorkStopsAfterScanLimit() throws {
        let store = try makeStore()
        for index in 0..<101 {
            let name = "gone-\(index).md"
            let mdPath = directory.appendingPathComponent(name).path
            try store.upsertDocument(docName: name, digest: "# 目录\n\n\(name)",
                                     chapters: "[]", mdPath: mdPath,
                                     contentHash: "content-\(name)")
        }
        let coordinator = makeCoordinator(store: store, resident: nil)

        // 大批失联文档不能把空闲会话的启动拖住；应提示稍后继续检查。
        XCTAssertEqual(coordinator.checkIdleWork(), .rotated)
    }

    func testCancelCurrentBlocksFurtherTicksUntilNextSession() async throws {
        let store = try makeStore()
        _ = try insertDocument(store, markdown: chapterMarkdown(1))
        let resident = CoordinatorStubLLM()
        let coordinator = makeCoordinator(store: store, resident: resident)

        let began = await coordinator.beginSession(mode: .resident)
        XCTAssertTrue(began)
        await coordinator.cancelCurrent()
        let stopped = await coordinator.runTick(mode: .resident)
        XCTAssertEqual(stopped, .noWork)
        XCTAssertEqual(resident.calls, 0)

        await coordinator.endSession()
        let resumed = await coordinator.beginSession(mode: .resident)
        XCTAssertTrue(resumed)
        let outcome = await coordinator.runTick(mode: .resident)
        guard case .processedChapter = outcome else {
            return XCTFail("新会话应恢复处理，实际 \(outcome)")
        }
    }
}
