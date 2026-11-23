import Foundation

/// 闲时摘要协调器：
/// - 常驻辅助模型：15 分钟一个章节；临时辅助模型：5 分钟一个章节；
/// - 一个 tick 最多处理一个章节或一次全书摘要；
/// - 成功后才写 `last_tick_at`，失败不推进 `next_section`；
/// - 临时模型由会话创建/卸载，常驻模型只借用、不关闭。
public final class BookSummaryCoordinator: @unchecked Sendable {

    private let engine: BookSummaryEngine
    private let store: VectorStore
    private let configProvider: @Sendable () -> AgentConfig
    private let residentAuxProvider: @Sendable () -> LLMService?
    private let transientAuxFactory: (@Sendable (AgentConfig) -> LLMService)?
    private let now: @Sendable () -> Date
    private let log: @Sendable (String) -> Void

    private let lock = NSLock()
    private var sessionMode: BookSummaryMode?
    private var transientLLM: LLMService?
    private var tickTask: Task<BookSummaryTickOutcome, Never>?
    private var cancelRequested = false
    /// 会话级停止：`cancelCurrent` 之后不再开始新的 tick，直到下个会话。
    private var stopRequested = false
    private var retryCounts: [String: Int] = [:]
    private var currentDocName: String?
    private var currentSection = 0
    private var currentTotal = 0
    /// 同一文档每个空闲会话最多重试次数；超过后让位给其他书。
    private let maxRetriesPerDocument = 2

    public init(engine: BookSummaryEngine,
                store: VectorStore,
                configProvider: @escaping @Sendable () -> AgentConfig,
                residentAuxProvider: @escaping @Sendable () -> LLMService?,
                transientAuxFactory: (@Sendable (AgentConfig) -> LLMService)? = nil,
                now: @escaping @Sendable () -> Date = { Date() },
                log: @escaping @Sendable (String) -> Void = { _ in }) {
        self.engine = engine
        self.store = store
        self.configProvider = configProvider
        self.residentAuxProvider = residentAuxProvider
        self.transientAuxFactory = transientAuxFactory
        self.now = now
        self.log = log
    }

    // MARK: - 会话

    /// 开始空闲会话；临时模式会创建并预加载辅助模型。
    public func beginSession(mode: BookSummaryMode) async -> Bool {
        lock.lock()
        if let existing = sessionMode {
            lock.unlock()
            return existing == mode
        }
        lock.unlock()

        // 核心句模式完全本地、无需辅助模型：直接开会话，不预加载也不借用。
        if configProvider().bookSummaryMode == .coreSentences {
            guard !Task.isCancelled else { return false }
            lock.lock()
            sessionMode = mode
            retryCounts.removeAll()
            cancelRequested = false
            stopRequested = false
            lock.unlock()
            log("章节摘要：核心句模式空闲会话（不加载辅助模型）")
            return true
        }

        if mode == .resident {
            guard !Task.isCancelled else { return false }
            guard residentAuxProvider() != nil else {
                log("章节摘要：常驻辅助模型不可用，跳过本次空闲会话")
                return false
            }
            lock.lock()
            sessionMode = .resident
            retryCounts.removeAll()
            cancelRequested = false
            stopRequested = false
            lock.unlock()
            log("章节摘要：开始常驻辅助模型空闲会话（900 秒/章）")
            return true
        }

        guard !Task.isCancelled else { return false }
        guard let factory = transientAuxFactory else {
            log("章节摘要：没有临时辅助模型工厂，跳过本次空闲会话")
            return false
        }
        let llm = factory(configProvider())
        if let preloadable = llm as? PreloadableLLMService {
            do {
                try await preloadable.preload()
            } catch {
                await releaseTransient(llm)
                log("章节摘要：临时辅助模型加载失败：\(error.localizedDescription)")
                return false
            }
        }
        // 预加载不可中断；如果等待期间会话已被取消，必须释放刚加载的模型。
        if Task.isCancelled {
            await releaseTransient(llm)
            log("章节摘要：会话已取消，放弃刚加载的临时辅助模型")
            return false
        }

        lock.lock()
        sessionMode = .temporary
        transientLLM = llm
        retryCounts.removeAll()
        cancelRequested = false
        stopRequested = false
        lock.unlock()
        log("章节摘要：开始临时辅助模型空闲会话（300 秒/章）")
        return true
    }

    /// 空闲会话开始前检查候选；用于避免为必然失败的 tick 预加载临时模型。
    ///
    /// Markdown 被移动/删除的候选会在这里记为失败并推进 `updated_at`：
    /// 候选队列按 `updated_at` 升序取前 N 本，若总让不可读文档占住队首，
    /// 排在后面的可读文档就永远轮不到（`runTick` 只取回同一页候选）。
    public func checkIdleWork() -> BookSummaryIdleAvailability {
        var scanned = 0
        var rotated = Set<String>()
        while scanned < Self.maxReadabilityScan {
            let candidates: [DocumentRecord]
            do {
                candidates = try store.summaryCandidates(limit: 10)
            } catch {
                // 读库失败时不要误报“没有待处理章节”，交给 tick 给出具体错误。
                return .ready
            }
            guard !candidates.isEmpty else { return .noWork }

            // 已经轮转过的候选重新出现在页首，说明前面没有其它待处理文档
            // （不足一页时它们还会留在同一页里），此时没必要继续扫描。
            let fresh = candidates.filter { !rotated.contains($0.docID) }
            guard !fresh.isEmpty else { return .noWork }

            let unreadable = fresh.filter {
                !FileManager.default.isReadableFile(atPath: $0.mdPath)
            }
            guard !unreadable.isEmpty else { return .ready }

            for document in unreadable {
                rotated.insert(document.docID)
                recordFailure(docID: document.docID,
                              message: "找不到可读的 Markdown：\(document.mdPath)")
            }
            log("章节摘要：轮转 \(unreadable.count) 个 Markdown 不可读的候选")
            scanned += unreadable.count
        }
        return .rotated
    }

    /// 单次检查最多轮转的不可读候选数量；避免大批量失联文档把
    /// 空闲会话的启动拖得过久。
    private static let maxReadabilityScan = 100

    /// 执行一次节流 tick；同一时间只允许一个 tick 在跑。
    public func runTick(mode: BookSummaryMode) async -> BookSummaryTickOutcome {
        lock.lock()
        let stopping = stopRequested
        lock.unlock()
        guard !stopping else { return .noWork }

        let coreSentences = configProvider().bookSummaryMode == .coreSentences
        let llm: LLMService?
        if coreSentences {
            llm = nil
        } else {
            switch mode {
            case .resident:
                llm = residentAuxProvider()
            case .temporary:
                lock.lock()
                llm = transientLLM
                lock.unlock()
            }
        }
        if !coreSentences, llm == nil {
            log("章节摘要：\(mode == .resident ? "常驻" : "临时")辅助模型不可用")
            return .noWork
        }

        let lastTick: Double? = (try? store.summaryLastTickAt()) ?? nil
        if let lastTick {
            let elapsed = now().timeIntervalSince1970 - lastTick
            if elapsed < mode.interval {
                // 时钟回拨时 last_tick_at 可能在未来；等待时间最多一个完整间隔，
                // 避免空闲会话睡眠数天。
                let wait = min(mode.interval, max(0, mode.interval - elapsed))
                return .throttled(waitSeconds: wait)
            }
        }

        lock.lock()
        let running = tickTask != nil
        let stoppingNow = stopRequested
        if running || stoppingNow {
            lock.unlock()
            return running ? .throttled(waitSeconds: 0) : .noWork
        }
        cancelRequested = false
        lock.unlock()

        // `checkIdleWork` 读库失败时会放行，让 tick 给出具体错误；这里不能再
        // 用 `try?` 吞掉异常，否则会话会误报“没有待处理章节”并静默结束。
        let candidates: [DocumentRecord]
        do {
            candidates = try store.summaryCandidates(limit: 10)
        } catch {
            let message = "读取候选文档失败：\(error.localizedDescription)"
            log("章节摘要：\(message)")
            return .failed(docName: "候选列表", message: message)
        }
        guard let document = candidates.first(where: {
            retryCount(docID: $0.docID) < maxRetriesPerDocument
        }) else {
            return .noWork
        }

        guard FileManager.default.fileExists(atPath: document.mdPath),
              let markdown = try? String(contentsOfFile: document.mdPath, encoding: .utf8) else {
            let message = "找不到可读的 Markdown：\(document.mdPath)"
            let attempts = incrementRetry(docID: document.docID)
            recordFailure(docID: document.docID, message: message)
            log("章节摘要：\(document.docName) \(message)（第 \(attempts)/\(maxRetriesPerDocument) 次）")
            return .failed(docName: document.docName, message: message)
        }

        let job = try? store.summaryJob(docID: document.docID)
        // 旧版无标题纯正文可能停在 completed / total_sections=0：先回逐章阶段
        // 合成「全文」章节摘要，否则会调用注定失败的全书汇总。
        let isBookSlot = job?.status == .summarizingBook
            || (job?.status == .completed && (job?.totalSections ?? 0) > 0)
        lock.lock()
        currentDocName = document.docName
        currentSection = max(0, job?.nextSection ?? 0)
        currentTotal = job?.totalSections ?? 0
        lock.unlock()

        let task = Task { [weak self] () -> BookSummaryTickOutcome in
            guard let self else { return .noWork }
            return await self.performTick(document: document, markdown: markdown,
                                          llm: llm, isBookSlot: isBookSlot)
        }
        lock.lock()
        // 候选/读文件期间可能已被 cancelCurrent 停止；此时不能再发布新 tick。
        if stopRequested {
            lock.unlock()
            task.cancel()
            _ = await task.value
            return .noWork
        }
        tickTask = task
        lock.unlock()

        let outcome = await task.value
        lock.lock()
        if tickTask == task { tickTask = nil }
        lock.unlock()
        return outcome
    }

    /// 停止当前章节调用；已完成分段保留，下个空闲会话继续。
    public func cancelCurrent() async {
        lock.lock()
        cancelRequested = true
        stopRequested = true
        let task = tickTask
        lock.unlock()
        task?.cancel()
        if let task {
            _ = await task.value
            lock.lock()
            if tickTask == task { tickTask = nil }
            lock.unlock()
        }
    }

    /// 结束会话：等待当前任务停止，卸载临时模型；不触碰常驻辅助模型。
    public func endSession() async {
        await cancelCurrent()
        lock.lock()
        let llm = transientLLM
        transientLLM = nil
        sessionMode = nil
        retryCounts.removeAll()
        currentDocName = nil
        currentSection = 0
        currentTotal = 0
        cancelRequested = false
        stopRequested = false
        lock.unlock()

        if let llm {
            await releaseTransient(llm)
            log("章节摘要：临时辅助模型已卸载")
        }
    }

    /// 释放临时辅助模型：先等待权重真正卸载，再关闭连接。
    ///
    /// `close()` 对本地模型只是发起异步释放；加载失败/取消时也必须走这里，
    /// 否则退出流程可能和仍在执行的 GPU 工作并发。
    private func releaseTransient(_ llm: LLMService) async {
        if let unloadable = llm as? AsyncUnloadableLLMService {
            await unloadable.unloadAndWait()
        }
        (llm as? ClosableLLMService)?.close()
    }

    @MainActor
    public func statusSnapshot() -> String {
        lock.lock()
        defer { lock.unlock() }
        guard sessionMode != nil else { return "章节摘要：空闲" }
        if let name = currentDocName {
            if currentTotal > 0 {
                return "章节摘要：\(name) \(min(currentSection + 1, currentTotal))/\(currentTotal)"
            }
            return "章节摘要：\(name)"
        }
        return "章节摘要：等待处理"
    }

    // MARK: - Tick 执行

    private func performTick(document: DocumentRecord, markdown: String,
                             llm: LLMService?, isBookSlot: Bool) async -> BookSummaryTickOutcome {
        // 摘要的每次 tick 都是一组独立提示；保留跨 tick 的 KV 只会把上一章内容
        // 带进下一次提示并长期占用内存，因此在结束（成功/失败/取消）时统一丢弃。
        defer { (llm as? KVCacheInvalidating)?.invalidateKVCache() }
        do {
            if isBookSlot {
                let record = try await engine.processBookSummary(
                    document: document, markdown: markdown, llm: llm,
                    now: now(), isCancelled: { [weak self] in
                        guard let self else { return true }
                        return self.isCancelled
                    })
                try? store.setSummaryLastTickAt(now().timeIntervalSince1970)
                resetRetry(docID: document.docID)
                lock.lock()
                currentSection = max(record.chapterCount - 1, 0)
                currentTotal = record.chapterCount
                lock.unlock()
                log("章节摘要：✓ \(document.docName) 全书摘要完成（\(record.summary.count) 字）")
                return .processedBook(docID: document.docID, docName: document.docName)
            }

            let result = try await engine.processNextChapter(
                document: document, markdown: markdown, llm: llm,
                now: now(), isCancelled: { [weak self] in
                    guard let self else { return true }
                    return self.isCancelled
                })
            try? store.setSummaryLastTickAt(now().timeIntervalSince1970)
            resetRetry(docID: document.docID)
            lock.lock()
            currentSection = result.sectionIndex
            currentTotal = result.totalSections
            lock.unlock()
            log("章节摘要：✓ \(document.docName) 第 \(result.sectionIndex + 1)/"
                + "\(result.totalSections) 节（\(result.summary.count) 字）")
            return .processedChapter(
                docID: document.docID, docName: document.docName,
                sectionIndex: result.sectionIndex,
                totalSections: result.totalSections,
                chapterCompleted: result.chapterCompleted)
        } catch {
            if isCancellation(error) {
                log("章节摘要：已取消 \(document.docName) 的当前处理，已完成分段保留")
                return .failed(docName: document.docName, message: "摘要任务已取消")
            }
            let attempts = incrementRetry(docID: document.docID)
            // 引擎可能已经写过 failed/pending 状态；这里只更新 updated_at 让
            // 候选队列轮转，避免个别持续失败的文档长期占住队首。
            recordFailure(docID: document.docID,
                          message: error.localizedDescription)
            log("章节摘要：✗ \(document.docName) \(error.localizedDescription)"
                + "（第 \(attempts)/\(maxRetriesPerDocument) 次）")
            return .failed(docName: document.docName,
                           message: error.localizedDescription)
        }
    }

    // MARK: - 状态

    /// 记录一次失败尝试：保留引擎写入的进度，只把 updated_at 推进到当前时间。
    /// 候选队列按 `updated_at` 升序取前 N 本；若失败不落库，前 N 本持续失败
    /// 会在每个新会话里反复占满窗口，后面的文档永远轮不到。
    private func recordFailure(docID: String, message: String) {
        let job = try? store.summaryJob(docID: docID)
        // 已进入全书阶段的文档要保留原状态（completed / summarizingBook），
        // 否则下次会先跑一轮逐章检查才回到全书摘要槽位。
        let status: BookSummaryJobStatus
        switch job?.status {
        case .completed: status = .completed
        case .summarizingBook: status = .summarizingBook
        default: status = .failed
        }
        try? store.upsertSummaryJob(SummaryJobRecord(
            docID: docID,
            status: status,
            nextSection: max(0, job?.nextSection ?? 0),
            totalSections: max(0, job?.totalSections ?? 0),
            lastError: message,
            updatedAt: now().timeIntervalSince1970))
    }

    private var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelRequested || stopRequested
    }

    private func isCancellation(_ error: Error) -> Bool {
        if isCancelled || Task.isCancelled { return true }
        if case AgentError.cancelled = error { return true }
        return false
    }

    private func retryCount(docID: String) -> Int {
        lock.lock()
        defer { lock.unlock() }
        return retryCounts[docID] ?? 0
    }

    @discardableResult
    private func incrementRetry(docID: String) -> Int {
        lock.lock()
        defer { lock.unlock() }
        let next = (retryCounts[docID] ?? 0) + 1
        retryCounts[docID] = next
        return next
    }

    private func resetRetry(docID: String) {
        lock.lock()
        retryCounts[docID] = nil
        lock.unlock()
    }
}
