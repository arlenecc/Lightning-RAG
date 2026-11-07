import Foundation

/// 后台事实抽取的**文件游标**：按 4 小时节流，增量读取当天
/// `memory/raw_session/<date>.txt` 里完整闭合的 SESSION 块。
///
/// 数据源是落盘文件而不是会话内存，因此折叠 / 清空对话都不会丢内容；
/// 游标记录「日期 + 已消费的字节偏移 + 上次调度时间」，跨天自动归零。
/// 4h 节流按调度时间计：被前台取消的那次也算一次，避免连续对话退化成
/// 每轮重试；游标只在成功入库后推进，积压内容会在下个窗口一次性补上。
/// 只处理以 `@@SESSION-END` 闭合的块，写了一半的尾部留到下次，避免把半个
/// session 当成事实喂给模型。只有 `commit` 才推进游标，取消 / 模型无响应
/// 会原样重跑（写入侧 `addObservations` / `createRelation` 本身幂等）。
public final class FactExtractionWorker: @unchecked Sendable {

    public struct Batch: Sendable {
        public let day: String
        /// 骨架化后的对话文本（已剥离 SYSTEM / TOOL / TURN-PROMPT 段）。
        public let transcript: String
        /// 本次消费到的文件字节偏移。
        public let newOffset: UInt64
        public let sessionCount: Int
    }

    /// 两次成功抽取之间的最小间隔。
    public static let interval: TimeInterval = 4 * 3_600
    /// 单次后台任务最多喂给模型的字符数（保尾，优先最近的对话）。
    public static let maxTranscriptChars = 16_000
    /// 单次增量读取的字节上限，防止极端情况下把整个文件读进内存。
    private static let maxReadBytes = 8_000_000

    private struct Cursor: Codable {
        var day: String = ""
        var offset: UInt64 = 0
        var lastRunAt: TimeInterval = 0
    }

    private let dir: MemoryDirectory
    private let cursorFile: String
    private let lock = NSLock()

    public init(directory: MemoryDirectory) {
        self.dir = directory
        self.cursorFile = directory.path("raw_session/fact_extraction_cursor.json")
    }

    // MARK: - 调度

    /// 距上次调度 ≥4h，且当天文件存在未消费的完整 session。
    public func isDue(now: Date = Date()) -> Bool {
        lock.lock(); defer { lock.unlock() }
        let cursor = normalizedCursor(day: MemoryDirectory.dayString(now))
        guard now.timeIntervalSince1970 - cursor.lastRunAt >= Self.interval else { return false }
        return pending(cursor) != nil
    }

    /// 取出本次待抽取的批次；**不推进游标**。
    public func takePending(now: Date = Date()) -> Batch? {
        lock.lock(); defer { lock.unlock() }
        let cursor = normalizedCursor(day: MemoryDirectory.dayString(now))
        return pending(cursor)
    }

    /// 记录一次调度（无论最终成功与否）。被前台请求取消时不能立刻重试，
    /// 否则连续对话会退化成每轮都拉起后台模型；此时文件游标未推进，
    /// 下一个 4h 窗口会把积压的增量一次性补上，不丢数据。
    public func markAttempt(now: Date = Date()) {
        lock.lock(); defer { lock.unlock() }
        var cursor = loadCursor()
        let day = MemoryDirectory.dayString(now)
        if cursor.day != day {
            cursor.day = day
            cursor.offset = 0
        }
        cursor.lastRunAt = now.timeIntervalSince1970
        saveCursor(cursor)
    }

    /// 抽取成功（已写入 memory graph）后推进游标与 4h 时间戳。
    public func commit(_ batch: Batch, now: Date = Date()) {
        lock.lock(); defer { lock.unlock() }
        var cursor = loadCursor()
        if cursor.day == batch.day {
            cursor.offset = max(cursor.offset, batch.newOffset)
        } else {
            cursor.day = batch.day
            cursor.offset = batch.newOffset
        }
        cursor.lastRunAt = now.timeIntervalSince1970
        saveCursor(cursor)
    }

    // MARK: - 增量读取

    private func pending(_ cursor: Cursor) -> Batch? {
        let file = dir.rawFile(day: cursor.day)
        guard let handle = try? FileHandle(forReadingFrom: URL(fileURLWithPath: file)) else {
            return nil
        }
        defer { try? handle.close() }
        guard let size = try? handle.seekToEnd(), size > cursor.offset else { return nil }
        do { try handle.seek(toOffset: cursor.offset) } catch { return nil }
        let length = Int(min(size - cursor.offset, UInt64(Self.maxReadBytes)))
        guard let data = try? handle.read(upToCount: length),
              let text = String(data: data, encoding: .utf8) else { return nil }

        // 只取到最后一个完整 `@@SESSION-END` 行；尾部半截留到下次。
        // 标记行是纯 ASCII，因此切点一定落在 UTF-8 字符边界上。
        var consumed = 0
        var lastEnd = 0
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            consumed += line.count + 1
            if line.hasPrefix("@@SESSION-END") { lastEnd = consumed }
        }
        guard lastEnd > 0 else { return nil }
        let complete = String(text.prefix(lastEnd))
        let blocks = RawSessionStore.splitSessions(complete)
        guard !blocks.isEmpty else { return nil }

        var skeleton = ""
        for block in blocks {
            let body = RawSessionStore.stripToSkeleton(block.body)
            if !body.isEmpty { skeleton += body + "\n\n" }
        }
        if skeleton.count > Self.maxTranscriptChars {
            skeleton = String(skeleton.suffix(Self.maxTranscriptChars))
        }
        return Batch(day: cursor.day, transcript: skeleton,
                     newOffset: cursor.offset + UInt64(complete.utf8.count),
                     sessionCount: blocks.count)
    }

    // MARK: - 游标持久化

    private func normalizedCursor(day: String) -> Cursor {
        var cursor = loadCursor()
        if cursor.day != day {
            cursor.day = day
            cursor.offset = 0
            saveCursor(cursor)
        }
        return cursor
    }

    private func loadCursor() -> Cursor {
        guard let raw = try? String(contentsOfFile: cursorFile, encoding: .utf8),
              let data = raw.data(using: .utf8),
              let cursor = try? JSONDecoder().decode(Cursor.self, from: data) else {
            return Cursor()
        }
        return cursor
    }

    private func saveCursor(_ cursor: Cursor) {
        guard let data = try? JSONEncoder().encode(cursor),
              let text = String(data: data, encoding: .utf8) else { return }
        try? FileManager.default.createDirectory(
            atPath: (cursorFile as NSString).deletingLastPathComponent,
            withIntermediateDirectories: true)
        _ = dir.write(text, to: cursorFile)
    }
}
