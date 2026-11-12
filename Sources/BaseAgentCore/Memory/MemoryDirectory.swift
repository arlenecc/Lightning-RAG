import Foundation

/// 工作目录下 `memory/` 多级记忆的统一路径与文件设施。
///
/// 与现有 `.agent/` 下的 `GraphMemoryStore` **并存、互不影响**：
/// 后者是知识图谱，这里是基于纯文本文件的 L0–L4 分层。
///
/// ```
/// <workspace>/memory/
///   memory_management_sop.md      L0  META-SOP（首次落盘，之后以文件为准）
///   global_mem_insight.txt        L1  极简索引（≤30 行，常驻 system prompt）
///   global_mem.txt                L2  全局事实库
///   *_sop.md / *.py               L3  任务级精简记录
///   history_insight.md            历史语义提炼产物（改进7）
///   raw_session/                  L4  历史会话层
///     YYYY-MM-DD.txt                 当天对话全文（追加）
///     YYYY-MM-DD-digest.txt          当天各轮摘要（追加）
///     all_history.txt                历史 USER/Agent 骨架（追加）
///     YYYY-MM-DD.zip                 已归档的过期日期记录
/// ```
///
/// 线程安全：同一 workspace 的**所有实例共享一把进程级锁**（Agent、
/// StartLongTermUpdateTool、MemoryMaintenance 各建实例，各自持锁会让
/// read→校验→write 变成非原子）。追加用 `FileHandle.seekToEnd()`，
/// 不做「读全文+整文件覆盖」，避免并发互相覆盖。
public final class MemoryDirectory: @unchecked Sendable {
    public let root: String

    private static let registryLock = NSLock()
    private static var sharedLocks: [String: NSRecursiveLock] = [:]
    private let lock: NSRecursiveLock
    private let fm = FileManager.default

    public init(workspace: String) {
        let root = (workspace as NSString).appendingPathComponent("memory")
        self.root = root
        Self.registryLock.lock()
        if let existing = Self.sharedLocks[root] {
            self.lock = existing
        } else {
            let created = NSRecursiveLock()
            Self.sharedLocks[root] = created
            self.lock = created
        }
        Self.registryLock.unlock()
    }

    // MARK: 路径

    public var rawSessionDir: String { path("raw_session") }
    public var sopFile: String { path("memory_management_sop.md") }
    public var l1File: String { path("global_mem_insight.txt") }
    public var l2File: String { path("global_mem.txt") }
    public var historyInsightFile: String { path("history_insight.md") }
    public var allHistoryFile: String { path("raw_session/all_history.txt") }
    public var tidyProposalFile: String { path("memory_tidy_proposal.md") }
    /// 待语义提炼的 session 队列（持久化）。归档完成即入队，提炼成功后才出队，
    /// 避免寒暄轮/后台取消/辅助模型不可用时"原文件已删、提炼永久丢失"。
    public var pendingInsightFile: String { path("raw_session/pending_insight.json") }

    public func path(_ relative: String) -> String {
        (root as NSString).appendingPathComponent(relative)
    }

    /// 当天日期文件名（本地时区）。
    public static func dayString(_ date: Date = Date()) -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd"
        return f.string(from: date)
    }

    public static func timestamp(_ date: Date = Date()) -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return f.string(from: date)
    }

    /// session 标识：`MMdd_HHmmss_NNN`（秒级 + 进程内递增序号）。
    ///
    /// 必须全局唯一：它同时是 zip 内文件名、`all_history.txt` 的 session 标记、
    /// `history_insight.md` 的增量游标。分钟精度会让同一分钟的两个请求生成
    /// 同名 zip 条目（覆盖/打包失败），且字符串游标无法区分跨年同月同日。
    private static let idLock = NSLock()
    private static var idSequence: UInt64 = 0
    public static func sessionId(_ date: Date = Date()) -> String {
        let f = DateFormatter()
        f.dateFormat = "MMdd_HHmmss"
        let stamp = f.string(from: date)
        idLock.lock()
        idSequence &+= 1
        let seq = idSequence % 1000
        idLock.unlock()
        return String(format: "%@_%03d", stamp, seq)
    }

    public func rawFile(day: String) -> String {
        path("raw_session/\(day).txt")
    }

    public func digestFile(day: String) -> String {
        path("raw_session/\(day)-digest.txt")
    }

    public func archiveFile(day: String) -> String {
        path("raw_session/\(day).zip")
    }

    // MARK: 初始化布局

    /// 建目录并补齐缺失的默认文件。**不覆盖**任何已存在文件
    /// （L0/L1/L2 允许人工编辑，覆盖会造成"记忆被删改"的持久性伤害）。
    public func ensureLayout() {
        lock.lock(); defer { lock.unlock() }
        try? fm.createDirectory(atPath: root, withIntermediateDirectories: true)
        try? fm.createDirectory(atPath: rawSessionDir, withIntermediateDirectories: true)

        seedIfMissing(sopFile, MEMORY_MANAGEMENT_SOP)
        seedIfMissing(l1File, DEFAULT_GLOBAL_MEM_INSIGHT)
        seedIfMissing(l2File, "# [Global Facts]\n\n> 按 ## [SECTION] 组织，仅存行动验证过的环境特异性事实。\n")
        seedIfMissing(allHistoryFile, "")
    }

    private func seedIfMissing(_ file: String, _ content: String) {
        guard !fm.fileExists(atPath: file) else { return }
        try? AtomicFile.write(content, to: file)
    }

    // MARK: 读

    /// 读文本，带**字符**上限；文件不存在返回 nil。
    ///
    /// 按字节 `read(upToCount:)` 而非 `String(contentsOfFile:)` 全量入内存：
    /// `all_history.txt` 可能上 MB，后台整理与前台 THINK 轮都调它，全量读会
    /// 撑内存峰值并让主线程卡在这把共享锁上。读取本身不持锁——`AtomicFile.write`
    /// 是原子 rename、`append` 是单次尾部写，读到中间态最多少一行，下一轮会重读。
    public func read(_ file: String, limit: Int = 200_000) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: URL(fileURLWithPath: file)) else {
            return nil
        }
        defer { try? handle.close() }
        let byteCap = min(limit * 4 + 4, 20_000_000)
        guard let data = try? handle.read(upToCount: byteCap) else { return "" }
        guard let text = ToolSupport.decodeUTF8Prefix(data) else { return nil }
        return String(text.prefix(limit))
    }

    public func exists(_ file: String) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return fm.fileExists(atPath: file)
    }

    /// 列出目录下条目名（非递归）。
    public func list(_ dir: String) -> [String] {
        lock.lock(); defer { lock.unlock() }
        return (try? fm.contentsOfDirectory(atPath: dir)) ?? []
    }

    /// L1 索引文本（≤ `maxL1Chars`；超限时截断并附提示）。
    public func loadL1(maxChars: Int) -> String {
        let text = read(l1File, limit: maxChars) ?? ""
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        if lines.count > 30 {
            return lines.prefix(30).joined(separator: "\n")
                + "\n(L1 超过 30 行，请按 META-SOP 整理后再写入)"
        }
        return text
    }

    /// L0 META-SOP 文本。文件存在则以文件为准（允许人工修订），否则回退内置常量。
    public func loadSOP(maxChars: Int) -> String {
        if let text = read(sopFile, limit: maxChars), !text.isEmpty { return text }
        return String(MEMORY_MANAGEMENT_SOP.prefix(maxChars))
    }

    /// 文件内容签名：用**内容哈希**而非 mtime+size。
    ///
    /// META-SOP 只允许「词级别等长 patch」；等长修改在某些文件系统上 mtime
    /// 粒度不足以区分，会让 L1 已改但 system 前缀仍用旧缓存（记忆实时更新失效）。
    /// L0/L1 都有硬上限（12k/4k 字符），每个 THINK 轮读两个小文件的成本可忽略；
    /// 与实际注入内容同源，签名相同 ⇒ 前缀逐字节相同 ⇒ KV 可复用。
    public func signature() -> String {
        let sop = read(sopFile, limit: SystemPromptLimits.maxSOPChars) ?? ""
        let l1 = read(l1File, limit: SystemPromptLimits.maxL1Chars) ?? ""
        return "\(Self.stableHash(sop))|\(Self.stableHash(l1))|\(sop.utf8.count)|\(l1.utf8.count)"
    }

    /// FNV-1a 64 位；不能用 Swift 的 `Hasher`（跨进程 seed 随机，结果不稳定）。
    private static func stableHash(_ text: String) -> UInt64 {
        var h: UInt64 = 14_695_981_039_346_656_037
        for byte in text.utf8 {
            h ^= UInt64(byte)
            h &*= 1_099_511_628_211
        }
        return h
    }

    // MARK: 写

    /// **追加**写（改进1/2 的核心）。用 `FileHandle` 尾部追加，不做整文件覆盖。
    /// 返回是否成功；失败只记录、不抛出，落盘失败绝不影响主对话。
    @discardableResult
    public func append(_ text: String, to file: String) -> Bool {
        guard !text.isEmpty else { return true }
        lock.lock(); defer { lock.unlock() }
        guard let data = text.data(using: .utf8) else { return false }
        do {
            try fm.createDirectory(atPath: (file as NSString).deletingLastPathComponent,
                                   withIntermediateDirectories: true)
            if !fm.fileExists(atPath: file) {
                try data.write(to: URL(fileURLWithPath: file))
                return true
            }
            let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: file))
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
            return true
        } catch {
            return false
        }
    }

    /// 原子整文件写（仅用于首次创建 / 备份 / 受控的 L1 更新）。
    @discardableResult
    public func write(_ text: String, to file: String) -> Bool {
        lock.lock(); defer { lock.unlock() }
        do { try AtomicFile.write(text, to: file); return true } catch { return false }
    }

    /// 写前留一份 `.bak`，用于 L1/L2/L3 被辅助引擎整理前可回滚。
    public func backup(_ file: String) {
        lock.lock(); defer { lock.unlock() }
        guard let text = try? String(contentsOfFile: file, encoding: .utf8) else { return }
        try? AtomicFile.write(text, to: file + ".bak")
    }

    /// 词级别 patch：`old` 必须**唯一**命中，否则拒绝（禁止盲改记忆）。
    @discardableResult
    public func patch(file: String, old: String, new: String) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard !old.isEmpty,
              let text = try? String(contentsOfFile: file, encoding: .utf8),
              text.components(separatedBy: old).count == 2 else { return false }
        let replaced = text.replacingOccurrences(of: old, with: new)
        do { try AtomicFile.write(replaced, to: file); return true } catch { return false }
    }

    public func remove(_ file: String) {
        lock.lock(); defer { lock.unlock() }
        try? fm.removeItem(atPath: file)
    }

    // MARK: 待提炼队列（改进7 的可靠性保障）

    /// 归档成功后入队。持久化，避免寒暄轮 / 后台被取消 / 辅助模型不可用时
    /// 「原文件已删、语义提炼永久丢失」。
    public func enqueuePendingInsights(_ ids: [String]) {
        guard !ids.isEmpty else { return }
        lock.lock(); defer { lock.unlock() }
        var all = Self.decodePending(readUnlocked(pendingInsightFile))
        for id in ids where !all.contains(id) { all.append(id) }
        _ = writeUnlocked(Self.encodePending(all), to: pendingInsightFile)
    }

    /// 取出全部待处理 session（**不删除**，提炼成功后由 `clearPendingInsights` 出队）。
    public func pendingInsightIds() -> [String] {
        lock.lock(); defer { lock.unlock() }
        return Self.decodePending(readUnlocked(pendingInsightFile))
    }

    public func clearPendingInsights(_ done: [String]) {
        guard !done.isEmpty else { return }
        lock.lock(); defer { lock.unlock() }
        let removed = Set(done)
        let rest = Self.decodePending(readUnlocked(pendingInsightFile))
            .filter { !removed.contains($0) }
        _ = writeUnlocked(Self.encodePending(rest), to: pendingInsightFile)
    }

    private static func decodePending(_ text: String?) -> [String] {
        guard let text, let data = text.data(using: .utf8),
              let value = try? JSONDecoder().decode(JSONValue.self, from: data),
              let ids = value["ids"]?.arrayValue else { return [] }
        return ids.compactMap(\.stringValue)
    }

    private static func encodePending(_ ids: [String]) -> String {
        JSONValue.object(["ids": .array(ids.map { .string($0) })])
            .jsonString(pretty: true)
    }

    /// 锁内专用的无锁读写（调用方已持 `lock`）。
    private func readUnlocked(_ file: String) -> String? {
        guard let text = try? String(contentsOfFile: file, encoding: .utf8) else { return nil }
        return text
    }

    private func writeUnlocked(_ text: String, to file: String) -> Bool {
        do { try AtomicFile.write(text, to: file); return true } catch { return false }
    }
}
