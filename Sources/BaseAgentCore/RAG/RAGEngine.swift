import Foundation
import CryptoKit
import Darwin

// MARK: - 统计 / Manifest

public struct IngestStats: Sendable {
    public var processed: Int = 0
    public var skipped: Int = 0
    public var removed: Int = 0
    /// 正文过短、整份丢弃的文件数量。
    public var discarded: Int = 0
    /// 按内容哈希清理掉的重复副本数量。
    public var deduplicated: Int = 0
    public var chunks: Int = 0
    public var errors: [String] = []
    public var cancelled: Bool = false

    public init() {}

    public var summary: String {
        var parts = ["已处理 \(processed) 个文件", "跳过 \(skipped)", "删除 \(removed)",
                     "去重 \(deduplicated)", "写入 \(chunks) 个切片"]
        if discarded > 0 { parts.append("丢弃过短 \(discarded)") }
        if !errors.isEmpty { parts.append("失败 \(errors.count)") }
        if cancelled { parts.append("已取消") }
        return parts.joined(separator: "，")
    }
}

/// 手动向量去重结果。
public struct VectorDedupStats: Sendable {
    public var removedChunks: Int = 0
    public var affectedSources: Int = 0
    public var emptiedSources: Int = 0
    public var remainingChunks: Int = 0

    public init() {}

    public var summary: String {
        "移除 \(removedChunks) 个重复切片，涉及 \(affectedSources) 个来源"
            + (emptiedSources > 0 ? "，清空 \(emptiedSources) 个副本来源" : "")
            + "，剩余 \(remainingChunks) 条切片"
    }
}

/// 手动向量去重前的扫描结果（不修改数据）。
public struct VectorDedupPreview: Sendable {
    public var totalChunks: Int = 0
    public var duplicateChunks: Int = 0
    public var affectedSources: Int = 0

    public init(totalChunks: Int = 0,
                duplicateChunks: Int = 0,
                affectedSources: Int = 0) {
        self.totalChunks = totalChunks
        self.duplicateChunks = duplicateChunks
        self.affectedSources = affectedSources
    }
}

struct RAGManifestEntry: Codable, Sendable {
    var mtime: Double
    var size: Int64
    var fingerprint: String
    var digest: String
    var docName: String
    var mdPath: String
    /// 生成该条目时使用的解析/切片签名。缺失表示旧版 manifest，需要重建。
    /// 每个文件单独记录，配置变化或中途取消后可以只重建未完成的文件。
    var ingestSignature: String?
    /// 完成入库的切片数。缺失表示旧版 manifest；实际库内数量不一致时强制重建。
    var chunkCount: Int?
    /// 是否完整处理。存在失败页/旧版条目时为 false，增量同步会重试。
    var complete: Bool?
    /// 内容重复副本指向的保留来源。该条目没有自己的向量，增量同步会跳过。
    var duplicateOf: String?
    /// 正文过短、已整体丢弃的来源。内容未变化时增量同步不再解析。
    var discarded: Bool? = nil
}

struct RAGManifest: Codable, Sendable {
    var files: [String: RAGManifestEntry] = [:]
}

/// journal 中的一条增量记录：只包含本次 merge 触达的条目。
private struct RAGManifestJournalRecord: Codable {
    var updated: [String: RAGManifestEntry] = [:]
    var removed: [String] = []
}

/// manifest 的增量写入。
///
/// 每次 merge 只向 `.journal` 追加一行 JSON（updated + removed），避免
/// N 个文件每次都重写完整 manifest 形成 O(N²) 的编解码与写盘。加载时先读
/// 压实后的基线 `manifest.json`，再按行回放 journal；尾部半行（崩溃中断）
/// 解析失败直接忽略。
///
/// 压实（compact）在 flock 内重写基线后删除 journal。写基线先于删 journal，
/// 中途读取最多重复回放一次增量记录（set/remove 幂等），不会读到缺数据的
/// 中间态。跨进程仍由 `.lock` 文件串行化。
final class ManifestStore: @unchecked Sendable {
    let path: String
    private var journalPath: String { path + ".journal" }
    /// journal 超过该体积时在 merge 内自动压实，避免无限增长。
    private static let compactThresholdBytes = 512 * 1024
    private let lock = NSRecursiveLock()

    init(path: String) { self.path = path }

    func load() -> [String: RAGManifestEntry] {
        lock.lock(); defer { lock.unlock() }
        return loadUnlocked()
    }

    private func loadUnlocked() -> [String: RAGManifestEntry] {
        var files = loadBaseUnlocked()
        for record in journalRecordsUnlocked() {
            apply(record, to: &files)
        }
        return files
    }

    private func loadBaseUnlocked() -> [String: RAGManifestEntry] {
        guard let data = FileManager.default.contents(atPath: path),
              let manifest = try? JSONDecoder().decode(RAGManifest.self, from: data) else {
            return [:]
        }
        return manifest.files
    }

    private func journalRecordsUnlocked() -> [RAGManifestJournalRecord] {
        guard let data = FileManager.default.contents(atPath: journalPath) else { return [] }
        let decoder = JSONDecoder()
        return data.split(separator: 0x0A).compactMap { line -> RAGManifestJournalRecord? in
            try? decoder.decode(RAGManifestJournalRecord.self, from: Data(line))
        }
    }

    private func apply(_ record: RAGManifestJournalRecord,
                       to files: inout [String: RAGManifestEntry]) {
        for key in record.removed { files.removeValue(forKey: key) }
        for (key, value) in record.updated { files[key] = value }
    }

    /// 合并写入：只覆盖/删除本次触达的条目，保留其它实例写入的记录。
    /// 增量记录追加到 journal；journal 过大时在同一次 flock 内压实。
    func merge(updated: [String: RAGManifestEntry], removed: Set<String>) {
        guard !updated.isEmpty || !removed.isEmpty else { return }
        lock.lock(); defer { lock.unlock() }
        withFileLock {
            appendJournalLocked(updated: updated, removed: removed)
            if journalSizeUnlocked() > Self.compactThresholdBytes {
                compactLocked()
            }
        }
    }

    private func appendJournalLocked(updated: [String: RAGManifestEntry],
                                     removed: Set<String>) {
        let record = RAGManifestJournalRecord(
            updated: updated,
            removed: Array(removed).sorted())
        guard let data = try? JSONEncoder().encode(record) else { return }

        // 记录前置换行：上次崩溃留下的半行会成为独立坏行被跳过，
        // 不会和后续追加的记录拼接成一条无法解析的长行。
        var payload = Data("\n".utf8)
        payload.append(data)

        let fd = open(journalPath, O_CREAT | O_WRONLY | O_APPEND, 0o644)
        guard fd >= 0 else {
            // 无法打开 journal（磁盘/权限异常）时退回完整重写，保证不丢更新。
            var files = loadUnlocked()
            apply(record, to: &files)
            writeBaseLocked(files)
            return
        }
        payload.withUnsafeBytes { raw in
            _ = Darwin.write(fd, raw.baseAddress, raw.count)
        }
        close(fd)
    }

    private func journalSizeUnlocked() -> Int {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: journalPath),
              let size = attributes[.size] as? NSNumber else { return 0 }
        return size.intValue
    }

    /// 把 journal 回放结果写成新的基线并删除 journal。
    /// 同步结束时调用一次，让下一次加载保持 O(N) 的单文件编解码。
    func compact() {
        lock.lock(); defer { lock.unlock() }
        withFileLock { compactLocked() }
    }

    private func compactLocked() {
        guard FileManager.default.fileExists(atPath: journalPath) else { return }
        writeBaseLocked(loadUnlocked())
        try? FileManager.default.removeItem(atPath: journalPath)
    }

    private func writeBaseLocked(_ files: [String: RAGManifestEntry]) {
        let manifest = RAGManifest(files: files)
        if let data = try? JSONEncoder().encode(manifest) {
            try? AtomicFile.write(data, to: path)
        }
    }

    private func withFileLock<T>(_ body: () -> T) -> T {
        let lockPath = path + ".lock"
        let fd = open(lockPath, O_CREAT | O_RDWR, 0o644)
        guard fd >= 0 else { return body() }
        flock(fd, LOCK_EX)
        defer {
            flock(fd, LOCK_UN)
            close(fd)
        }
        return body()
    }

    func clear() {
        lock.lock(); defer { lock.unlock() }
        try? FileManager.default.removeItem(atPath: path)
        try? FileManager.default.removeItem(atPath: journalPath)
    }
}

/// 本轮同步内的「正文内容哈希」去重登记表。
///
/// 解析/清洗后、切片/embedding 前先算正文哈希：
/// - 命中 manifest 中仍完整的历史文档 → 直接写 tombstone，跳过切片与向量化；
/// - 命中本轮正在处理且正文相同的文档 → 等它结束：成功则写 tombstone 跳过，
///   失败/取消则由等待者接手，避免把去重结果押在一个最终失败的文件上。
final class ContentDedupRegistry: @unchecked Sendable {
    enum Claim: Sendable {
        /// 由本次调用负责切片/embedding。
        case owner
        /// 正文已由 `canonical` 负责，当前来源只需写 tombstone。
        case duplicate(canonical: String)
    }

    private struct Waiter {
        let source: String
        let continuation: CheckedContinuation<Claim, Never>
    }

    private let lock = NSLock()
    /// digest → manifest 中可能可用的保留来源（第一次遇到时再验证库内切片）。
    private let manifestCandidates: [String: [String]]
    private var resolvedDigests = Set<String>()
    /// digest → 已确认可用的保留来源（历史文档或本轮完成文档）。
    private var canonicals: [String: String] = [:]
    /// digest → 当前负责切片/embedding 的来源。
    private var owners: [String: String] = [:]
    private var waiters: [String: [Waiter]] = [:]
    private let isUsable: @Sendable (String) -> Bool

    /// - Parameters:
    ///   - manifest: 同步开始时的 manifest 快照（已排除本轮删除的来源）。
    ///   - sources: 本轮仍存在的来源集合。
    ///   - isUsable: 历史条目是否仍有切片在库中；用于避免 tombstone 指向空来源。
    init(manifest: [String: RAGManifestEntry],
         sources: Set<String>,
         isUsable: @escaping @Sendable (String) -> Bool) {
        self.isUsable = isUsable
        var candidates: [String: [String]] = [:]
        for (source, entry) in manifest where sources.contains(source) {
            guard entry.complete == true,
                  (entry.chunkCount ?? 0) > 0,
                  entry.duplicateOf == nil,
                  !entry.digest.isEmpty else { continue }
            candidates[entry.digest, default: []].append(source)
        }
        manifestCandidates = candidates.mapValues {
            $0.sorted { $0.localizedStandardCompare($1) == .orderedAscending }
        }
    }

    /// 认领正文哈希。返回 `.duplicate` 时调用方不得再切片/embedding。
    func claim(digest: String, source: String) async -> Claim {
        // 历史 manifest 候选只在该 digest 第一次出现时验证一次（COUNT 查询），
        // 避免同步开头对每条 manifest 都扫一遍库。
        var candidatesToVerify: [String]?
        lock.lock()
        if !resolvedDigests.contains(digest) {
            resolvedDigests.insert(digest)
            candidatesToVerify = manifestCandidates[digest]
        }
        lock.unlock()

        if let candidatesToVerify {
            // 关闭查询可能在处理中删掉旧向量，因此以「当前库内确实有切片」为准。
            let usable = candidatesToVerify.first {
                $0 != source && isUsable($0)
            }
            if let usable {
                lock.lock()
                if canonicals[digest] == nil { canonicals[digest] = usable }
                lock.unlock()
            }
        }

        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                lock.lock()
                if let canonical = canonicals[digest], canonical != source {
                    lock.unlock()
                    continuation.resume(returning: .duplicate(canonical: canonical))
                    return
                }
                if let owner = owners[digest] {
                    if owner == source {
                        lock.unlock()
                        continuation.resume(returning: .owner)
                        return
                    }
                    waiters[digest, default: []].append(
                        Waiter(source: source, continuation: continuation))
                    lock.unlock()
                    return
                }
                owners[digest] = source
                lock.unlock()
                continuation.resume(returning: .owner)
            }
        } onCancel: {
            // 等待中的任务取消后必须唤醒自己，否则 owner 永远等不到结果。
            self.removeWaiter(digest: digest, source: source)
        }
    }

    /// owner 成功完成入库后调用，唤醒所有等待者写 tombstone。
    func complete(digest: String, source: String) {
        lock.lock()
        canonicals[digest] = source
        if owners[digest] == source { owners.removeValue(forKey: digest) }
        let waiting = waiters.removeValue(forKey: digest) ?? []
        lock.unlock()
        for waiter in waiting {
            waiter.continuation.resume(returning: .duplicate(canonical: source))
        }
    }

    /// owner 失败或取消后调用：让下一个等待者接手，其余继续排队。
    func fail(digest: String, source: String) {
        lock.lock()
        guard owners[digest] == source else {
            lock.unlock()
            return
        }
        guard var list = waiters[digest], !list.isEmpty else {
            owners.removeValue(forKey: digest)
            lock.unlock()
            return
        }
        let next = list.removeFirst()
        if list.isEmpty {
            waiters.removeValue(forKey: digest)
        } else {
            waiters[digest] = list
        }
        owners[digest] = next.source
        lock.unlock()
        next.continuation.resume(returning: .owner)
    }

    /// 同步结束时兜底：唤醒所有等待者，避免异常路径留下永久挂起的任务。
    func cancelAll() {
        lock.lock()
        let waiting = waiters
        waiters.removeAll()
        owners.removeAll()
        lock.unlock()
        for list in waiting.values {
            for waiter in list {
                waiter.continuation.resume(returning: .owner)
            }
        }
    }

    private func removeWaiter(digest: String, source: String) {
        lock.lock()
        guard var list = waiters[digest],
              let index = list.firstIndex(where: { $0.source == source }) else {
            lock.unlock()
            return
        }
        let waiter = list.remove(at: index)
        if list.isEmpty {
            waiters.removeValue(forKey: digest)
        } else {
            waiters[digest] = list
        }
        lock.unlock()
        waiter.continuation.resume(returning: .owner)
    }
}

// MARK: - RAG Engine

// MARK: - 并发 Ingest 任务

struct ScannedFile: Sendable {
    let path: String
    let relativePath: String
}

/// 原始文件采样指纹相同的文件组。代表文件负责解析/入库，其余副本清理向量。
/// 全量哈希二次校验后，只有内容确实一致的文件才会进入同一组。
struct DuplicateFileGroup: Sendable {
    let representative: ScannedFile
    let duplicates: [ScannedFile]
}

private struct IngestJob: Sendable {
    let index: Int
    let total: Int
    let file: ScannedFile
    let label: String
}

private struct IngestOutcome: Sendable {
    var source: String
    var skipped = false
    var processed = false
    /// 正文过短、整份丢弃（不切片/向量化）。
    var discarded = false
    var cancelled = false
    /// 命中正文哈希、只写 tombstone 的副本（没有切片/向量化）。
    var deduplicated = false
    var chunks = 0
    var error: String?
}

/// 工作队列：多个 worker 依次领取文件，避免慢文件拖住整批。
private actor IngestJobQueue {
    private let jobs: [IngestJob]
    private var nextIndex = 0

    init(_ jobs: [IngestJob]) {
        self.jobs = jobs
    }

    func next() -> IngestJob? {
        guard nextIndex < jobs.count else { return nil }
        let job = jobs[nextIndex]
        nextIndex += 1
        return job
    }
}

private actor IngestAccumulator {
    private var outcomes: [IngestOutcome] = []

    func record(_ outcome: IngestOutcome) {
        outcomes.append(outcome)
    }

    func snapshot() -> [IngestOutcome] {
        outcomes
    }
}

/// 文档查找结果：歧义时返回全部候选，避免静默选错同名书。
public enum DocumentLookupResult: Sendable, Equatable {
    case found(DocumentRecord)
    case ambiguous([DocumentRecord])
    case notFound
}

/// 已入库文档的清单条目（manifest 视角）：MCP `list_document` 等外部接口使用。
public struct RAGDocumentSummary: Sendable, Equatable {
    public var docName: String
    public var source: String
    public var mdPath: String
    public var chunkCount: Int?
    public var sizeBytes: Int64?
    public var modifiedAt: Double?
    public var duplicateOf: String?
    public var discarded: Bool?

    public init(docName: String, source: String, mdPath: String,
                chunkCount: Int? = nil, sizeBytes: Int64? = nil,
                modifiedAt: Double? = nil, duplicateOf: String? = nil,
                discarded: Bool? = nil) {
        self.docName = docName
        self.source = source
        self.mdPath = mdPath
        self.chunkCount = chunkCount
        self.sizeBytes = sizeBytes
        self.modifiedAt = modifiedAt
        self.duplicateOf = duplicateOf
        self.discarded = discarded
    }
}

/// 知识遗忘/删除结果。
public enum RAGDeleteOutcome: Sendable, Equatable {
    /// 已删除：`chunksRemoved` = 清掉的切片数，`docName` = 实际删除的文档。
    case deleted(chunksRemoved: Int, docName: String)
    /// 同名/同路径多候选，拒绝猜测性删除。
    case ambiguous([String])
    case notFound
}

/// `rag_outline` 使用的增强目录：修复后的目录 + 已生成的章节/全书摘要。
public struct EnrichedDocumentOutline: Sendable, Equatable {
    public var record: DocumentRecord
    public var chapterSummaries: [ChapterSummaryRecord]
    public var bookSummary: BookSummaryRecord?
    public var jobProgress: DocumentSummaryProgress
    /// 目录修复后的 TOC；无摘要时回退 `record.digest`。
    public var repairedTOC: String

    public init(record: DocumentRecord, chapterSummaries: [ChapterSummaryRecord],
                bookSummary: BookSummaryRecord?,
                jobProgress: DocumentSummaryProgress,
                repairedTOC: String) {
        self.record = record
        self.chapterSummaries = chapterSummaries
        self.bookSummary = bookSummary
        self.jobProgress = jobProgress
        self.repairedTOC = repairedTOC
    }
}

/// 知识库引擎：解析 → 清洗 → 切片 → embedding → SQLite 写入；
/// 检索走 FTS5 + 向量 RRF，再按配置做 rerank。
public final class RAGEngine {
    public let config: AgentConfig
    public let ragDir: String
    public let markdownDir: String
    /// 图谱抽取限额（来自配置，构建/增量都使用同一套参数）。
    private var kgExtractionLimits: KGExtractionLimits {
        KGExtractionLimits(
            maxEntitiesPerChunk: config.ragKGMaxEntitiesPerChunk,
            maxRelationsPerChunk: config.ragKGMaxRelationsPerChunk,
            maxDescriptionCharacters: config.ragKGMaxDescriptionChars)
    }
    /// 按知识库隔离的向量库路径（ragDir/index.sqlite）。
    public var indexDBPath: String {
        (ragDir as NSString).appendingPathComponent("index.sqlite")
    }
    private let manifestStore: ManifestStore
    public private(set) var knowledgeBase: String

    private let parserRegistry: DocumentParserRegistry
    private var embedding: EmbeddingProvider?
    private var reranker: RerankProvider?
    private var store: VectorStore?
    private let storeLock = NSLock()
    /// 知识图谱子系统的惰性实例。
    private var kgStore: KnowledgeGraphStore?
    private var kgGraph: KnowledgeGraph?
    private let kgLock = NSLock()
    /// 知识图谱抽取/关键词提取用的辅助模型（由 AppModel 注入，与 Agent.auxLLM 同实例）。
    public var auxLLM: LLMService?
    /// 保护「使用 store」的整段读操作；close/reload 必须等读操作结束。
    private let searchLock = NSLock()
    /// ingest/close/reload 的互斥标志。不能用 NSRecursiveLock：
    /// async 函数在 await 后可能换线程，跨线程 unlock 会让锁永久保持占用。
    private let ingestFlagLock = NSLock()
    private var ingestActive = false
    private var closePending = false
    private var cancelRequested = false
    private let cancelLock = NSLock()
    /// `status()` 的快照缓存。轮次提示词与状态栏都只读这里，
    /// 避免在千万级切片的库上重复执行全表 COUNT。
    private let statusLock = NSLock()
    private var statusSnapshot: [String: JSONValue]?
    private var statusSnapshotTime: TimeInterval = 0
    /// 写操作使快照失效时自增；并发中的 `status()` 若跨了一次失效，
    /// 不再把旧统计写回缓存（否则会继续展示最长 `maxAge` 秒）。
    private var statusGeneration: UInt64 = 0
    /// 超时后返回的「跳过」快照：短时间内不再重复等待同一个超时，
    /// 否则状态栏每 30 秒刷新一次、每次都要白等满整个超时。
    private var statusSkipSnapshot: [String: JSONValue]?
    private var statusSkipUntil: TimeInterval = 0
    private var lastTimeoutLog: TimeInterval = 0
    private static let statusSkipCacheSeconds: TimeInterval = 30
    /// 后台打开向量库：上次同步被中断/应用被强杀后，SQLite 需要先恢复遗留
    /// WAL，可能要数分钟。打开放到后台队列，读路径只等 `timeout` 秒。
    private let openCondition = NSCondition()
    private var openInFlight = false
    private var openFailure: Error?
    private let openQueue = DispatchQueue(label: "base-agent.rag.open", qos: .utility)
    /// ANN 后台自动构建状态。构建不占 `searchLock`，检索与状态查询随时可用。
    private let annBuildLock = NSLock()
    private var annBuildTask: Task<Void, Never>?
    private var annBuildScheduled = false
    private var annBuildActive = false
    private var annBuildProgress: ANNBuildProgress?
    private var annBuildGeneration: UInt64 = 0
    /// 最近一次构建的目标；仅用于状态展示（运行结束后不再有含义）。
    private var annBuildTarget: ANNIndexTarget = .dense

    public static let supportedExtensions: Set<String> = [
        "pdf", "txt", "md", "markdown", "html", "htm", "xhtml",
        "epub", "docx", "xlsx", "pptx", "doc", "xls", "ppt",
        "csv", "json", "rst", "tex", "log",
        // 独立图片文件：Apple Vision OCR。
        "png", "jpg", "jpeg", "tiff", "tif", "bmp", "heic", "heif", "webp",
    ]

    public init(config: AgentConfig,
                knowledgeBase: String? = nil,
                embedding: EmbeddingProvider? = nil,
                reranker: RerankProvider? = nil,
                parserRegistry: DocumentParserRegistry? = nil) {
        self.config = config
        // Vision OCR 全局闸门：rag_vision_concurrency > 0 时覆盖默认（CPU 核数）。
        // 须在首次 OCR 前设置——闸门信号量在首次访问时才惰性初始化。
        if config.ragVisionConcurrency > 0 {
            VisionPDFParser.requestedConcurrency = config.ragVisionConcurrency
        }
        let kb = knowledgeBase ?? config.resolvedKnowledgeBase
        // 统一成真实路径：macOS 的 /var 与 /private/var 是 firmlink，
        // URL.resolvingSymlinksInPath() 不会转换，必须用 realpath。
        self.knowledgeBase = Self.canonicalPath(kb)
        // 知识库目录只读（原始文件）；索引、markdown 产物、manifest 全部
        // 放在工作目录下按知识库隔离的 rag_data 子目录，不对原文件目录做任何改动。
        // 传入的 knowledgeBase 参数优先于 config（测试/多库场景）。
        self.ragDir = AgentConfig.ragIndexDirectory(forKB: self.knowledgeBase, workspace: config.workspace)
        Self.migrateLegacyIndex(base: (self.ragDir as NSString).deletingLastPathComponent,
                                target: self.ragDir,
                                workspace: config.workspace,
                                currentKB: self.knowledgeBase)
        self.markdownDir = (self.ragDir as NSString).appendingPathComponent("documents")
        self.manifestStore = ManifestStore(
            path: (self.ragDir as NSString).appendingPathComponent("manifest.json")
        )
        // Matryoshka：nomic / jina-v3 等支持俄罗斯套娃的模型先截断、再重新归一化。
        // 入库与查询必须共用同一实例，保证两侧变换完全一致。
        let baseEmbedding = embedding ?? LocalModelFactory.shared.makeEmbedding(config: config)
        self.embedding = baseEmbedding.map {
            MatryoshkaEmbedding.wrap($0,
                                     mode: config.ragMatryoshka,
                                     dimension: config.ragMatryoshkaDim)
        }
        self.reranker = reranker ?? LocalModelFactory.shared.makeReranker(config: config)
        self.parserRegistry = parserRegistry ?? DocumentParserRegistry.makeDefault(config: config)
        try? FileManager.default.createDirectory(atPath: self.markdownDir,
                                                 withIntermediateDirectories: true)
    }

    /// `realpath` 规范化：解析 `/var` → `/private/var` 与其它符号链接，
    /// 保证扫描路径、相对路径和 manifest 键完全一致。
    private static func canonicalPath(_ path: String) -> String {
        let expanded = (path as NSString).expandingTildeInPath
        guard let resolved = realpath(expanded, nil) else {
            return URL(fileURLWithPath: expanded).standardizedFileURL.path
        }
        defer { free(resolved) }
        return String(cString: resolved)
    }

    // MARK: 旧版共享索引迁移

    /// 旧版布局把所有知识库目录的索引共用 `rag_data` 根目录，切换知识库目录
    /// 后同步会把对方条目当「已删除文件」清空。首次按新布局访问时把遗留文件
    /// 搬进归属知识库的专属子目录：优先读取 `index_owner.json` 记录的归属，
    /// 无记录时视为当前配置的知识库（旧版本最后同步的就是它）。
    private static func migrateLegacyIndex(base: String, target: String,
                                           workspace: String, currentKB: String) {
        let fm = FileManager.default
        let legacyNames = ["manifest.json", "manifest.json.lock", "index.sqlite",
                           "index.sqlite-wal", "index.sqlite-shm",
                           "ingest_signature.txt", "documents"]
        let root = { (name: String) in (base as NSString).appendingPathComponent(name) }
        guard legacyNames.contains(where: { fm.fileExists(atPath: root($0)) }) else { return }

        let lockPath = root(".migrate.lock")
        let fd = open(lockPath, O_CREAT | O_RDWR, 0o644)
        if fd >= 0 { flock(fd, LOCK_EX) }
        defer {
            if fd >= 0 {
                flock(fd, LOCK_UN)
                Darwin.close(fd)
                try? fm.removeItem(atPath: lockPath)
            }
        }

        // 归属判定：marker 记录的 KB 优先（覆盖上次迁移被打断的场景），
        // 无记录时视为当前配置的知识库（旧版本最后同步的就是它）。
        var ownerKB = currentKB
        if let data = try? Data(contentsOf: URL(fileURLWithPath: root("index_owner.json"))),
           let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: String],
           let recorded = obj["kb"], !recorded.isEmpty {
            ownerKB = recorded
        }
        let ownerDir = ownerKB == currentKB
            ? target
            : AgentConfig.ragIndexDirectory(forKB: ownerKB, workspace: workspace)
        try? fm.createDirectory(atPath: ownerDir, withIntermediateDirectories: true)
        // 逐文件搬移；目标已存在同名文件时不覆盖（新布局数据优先）。
        var moved = 0
        for name in legacyNames {
            let src = root(name)
            let dst = (ownerDir as NSString).appendingPathComponent(name)
            guard fm.fileExists(atPath: src), !fm.fileExists(atPath: dst) else { continue }
            if (try? fm.moveItem(atPath: src, toPath: dst)) != nil {
                moved += 1
            }
        }
        if !fm.fileExists(atPath: root("index_owner.json")),
           let data = try? JSONSerialization.data(withJSONObject: ["kb": ownerKB],
                                                  options: [.sortedKeys]) {
            try? data.write(to: URL(fileURLWithPath: root("index_owner.json")), options: .atomic)
        }
        if moved > 0 {
            AgentLog.warning("RAG: 旧版共享索引已迁移至 \(ownerDir)（归属 \(ownerKB)），共 \(moved) 项")
        }
    }

    deinit {
        // 兜底：调用方忘记 close 时也不要把 SQLite 连接留到进程结束。
        store?.close()
    }

    // MARK: Store

    /// 当前生效的 Matryoshka 截断维（未启用时为 0）。
    public var effectiveMatryoshkaDimension: Int {
        (embedding as? MatryoshkaEmbedding)?.truncateDimension ?? 0
    }

    /// 当前生效的落盘元素类型：embedding provider 的硬性约束优先
    /// （当前内置模型均无约束），否则使用用户配置。
    public var effectiveVectorElementType: VectorElementType {
        (embedding as? EmbeddingVectorElementConstraining)?
            .requiredVectorElementType ?? config.ragVectorElementType
    }

    /// 二值签名位宽；主存储为 bin 时不启用（bin 只有单阶段检索）。
    public var binarySignatureDimension: Int {
        guard effectiveVectorElementType != .bin else { return 0 }
        return (embedding as? EmbeddingBinarySigning)?.binaryDimension ?? 0
    }

    /// 二值签名来源标识。Matryoshka 不改变签名来源，避免截断维变化误清空 bin 表。
    public var binarySignatureSource: String {
        guard binarySignatureDimension > 0, let embedding else { return "" }
        return (embedding as? EmbeddingBinarySigning)?.binarySourceIdentifier
            ?? embedding.identifier
    }

    /// 索引指纹：向量语义 + 落盘布局（`embedding.identifier | dimension |
    /// matryoshkaDim | elementType`）。任一字段变化都必须显式重建。
    ///
    /// Matryoshka 未实际截断时记 0：只切换 `auto`/`on` 而实际行为不变时，
    /// 不应被误判成需要重建。
    public var vectorIndexFingerprint: String {
        guard let embedding else { return "" }
        return [
            embedding.identifier,
            "dimension=\(embedding.dimension)",
            "matryoshka=\(effectiveMatryoshkaDimension)",
            "element=\(effectiveVectorElementType.rawValue)",
        ].joined(separator: "|")
    }

    private func ensureStore() throws -> VectorStore {
        storeLock.lock(); defer { storeLock.unlock() }
        if let store { return store }
        guard let embedding else {
            throw AgentError.storage("embedding 模型不可用，无法打开向量库；请检查 local_embedding_model 配置")
        }
        let newStore = try VectorStore(dbPath: self.indexDBPath,
                                       dimension: embedding.dimension,
                                       enableSQLiteVec: config.ragSQLiteVecEnabled,
                                       elementType: effectiveVectorElementType,
                                       binaryDimension: binarySignatureDimension,
                                       binarySource: binarySignatureSource,
                                       indexFingerprint: vectorIndexFingerprint)
        if newStore.didResetForDimensionChange {
            // 空库按新维度/类型采纳配置后，缓存的 manifest 若不同步失效，
            // 下一次同步会把所有文件当成「未变化」而全部跳过。
            manifestStore.clear()
            AgentLog.warning("RAG: 向量库为空且配置已变化，旧索引已重置，"
                + "已清空 manifest，下次同步将全量重建")
        }
        if let reason = newStore.rebuildReason {
            // 不静默删数据：旧切片与文档保留，FTS 仍可检索，向量通道暂时关闭。
            AgentLog.warning("RAG: 向量库与当前配置不匹配，需显式重建（\(reason)）；"
                + "向量检索暂停，全文检索仍可用")
        }
        store = newStore
        return newStore
    }

    /// 在读锁内使用 store：close/reload 会等待整个 body 结束。
    private func withStore<T>(_ body: (VectorStore) throws -> T) throws -> T {
        searchLock.lock(); defer { searchLock.unlock() }
        return try body(try ensureStore())
    }

    /// 只读查询的安全封装：限时取锁/打开库/执行，失败（含超时）时打日志并返回 nil，
    /// 避免 `withStore` 无超时导致的永久卡死，以及 `try?` 直接吞错导致的「静默失败」。
    /// 用于 rag_outline / 文档列举等读路径。
    private func withStoreSafe<T>(timeout: Double = RAGEngine.defaultReadTimeout,
                                  _ body: (VectorStore) throws -> T) -> T? {
        let budget = max(2, timeout)
        do {
            return try withStoreBounded(timeout: budget, body)
        } catch {
            AgentLog.warning("RAG: 只读查询失败（\(String(format: "%.0f", budget))s）："
                + error.localizedDescription)
            return nil
        }
    }

    private static let defaultReadTimeout: Double = 5

    /// 供 UI/工具读取当前生效的查询超时秒数（0 = 关闭超时）。
    public var queryTimeoutSeconds: Double { config.ragQueryTimeoutSeconds }

    /// 只读查询的实际超时：显式参数优先，否则用配置（0 = 不设超时）。
    private func resolvedTimeout(_ explicit: Double?) -> Double {
        explicit ?? config.ragQueryTimeoutSeconds
    }

    /// 带超时的只读查询入口：限时取 `searchLock`、限时等「打开库」、限时执行语句。
    /// 任何一步超时都抛 `AgentError.timeout`，由调用方跳过本次查询
    /// （状态查询转成提示快照，检索转成工具失败信息），不阻塞整个过程。
    private func withStoreBounded<T>(timeout: Double,
                                     _ body: (VectorStore) throws -> T) throws -> T {
        guard timeout > 0 else { return try withStore(body) }
        let deadline = Date().addingTimeInterval(timeout)
        guard searchLock.lock(before: deadline) else {
            throw logTimeout(AgentError.timeout(
                "知识库正忙（同步/维护进行中），本次查询已跳过"))
        }
        defer { searchLock.unlock() }
        do {
            let store = try storeBounded(by: deadline)
            let remaining = deadline.timeIntervalSinceNow
            guard remaining > 0 else { throw AgentError.timeout("查询超出时限被跳过") }
            return try store.withQueryDeadline(seconds: remaining) {
                try body(store)
            }
        } catch {
            throw logTimeout(error)
        }
    }

    /// 带超时的检索入口：限时取 `searchLock`、限时等「打开库」，然后把剩余
    /// 期限交给 `hybridSearch`（向量候选在写入锁内取，FTS/hydrate 走只读
    /// WAL 快照，不再整段持有写入锁）。任何一步超时都抛 `AgentError.timeout`，
    /// 由调用方跳过本次查询。
    private func withSnapshotStoreBounded<T>(
        timeout: Double,
        _ body: (VectorStore, Date?) throws -> T
    ) throws -> T {
        guard timeout > 0 else { return try withStore { try body($0, nil) } }
        let deadline = Date().addingTimeInterval(timeout)
        guard searchLock.lock(before: deadline) else {
            throw logTimeout(AgentError.timeout(
                "知识库正忙（同步/维护进行中），本次查询已跳过"))
        }
        defer { searchLock.unlock() }
        do {
            let store = try storeBounded(by: deadline)
            let remaining = deadline.timeIntervalSinceNow
            guard remaining > 0 else { throw AgentError.timeout("查询超出时限被跳过") }
            return try body(store, deadline)
        } catch {
            throw logTimeout(error)
        }
    }

    /// 限时拿到已打开的 store。打开动作放后台队列执行（同一时刻只有一个），
    /// 超时即抛出；打开完成后后续查询自动可用，不会重复建连接。
    private func storeBounded(by deadline: Date) throws -> VectorStore {
        storeLock.lock()
        if let store { storeLock.unlock(); return store }
        storeLock.unlock()

        openCondition.lock()
        if !openInFlight {
            openInFlight = true
            openFailure = nil
            openCondition.unlock()
            openQueue.async { [weak self] in self?.runBackgroundOpen() }
            openCondition.lock()
        }
        while openInFlight {
            guard openCondition.wait(until: deadline) else {
                openCondition.unlock()
                throw AgentError.timeout(
                    "知识库仍在打开（可能是上次同步中断后的恢复），本次查询已跳过")
            }
        }
        let failure = openFailure
        openCondition.unlock()
        if let failure { throw failure }
        storeLock.lock(); defer { storeLock.unlock() }
        guard let store else { throw AgentError.storage("向量库不可用") }
        return store
    }

    private func runBackgroundOpen() {
        var failure: Error?
        do { _ = try ensureStore() } catch { failure = error }
        openCondition.lock()
        openInFlight = false
        openFailure = failure
        openCondition.broadcast()
        openCondition.unlock()
        if let failure {
            AgentLog.warning("RAG: 后台打开向量库失败: \(failure.localizedDescription)")
        }
    }

    /// 超时日志限流：同一次卡顿只提示一次，避免刷屏。
    private func logTimeout(_ error: Error) -> Error {
        guard case AgentError.timeout = error else { return error }
        statusLock.lock()
        let now = Date().timeIntervalSince1970
        let shouldLog = now - lastTimeoutLog > 60
        if shouldLog { lastTimeoutLog = now }
        statusLock.unlock()
        guard shouldLog else { return error }
        AgentLog.warning("RAG: 查询超时已跳过（\(error.localizedDescription)）；"
            + "若持续出现，可能是上次同步中断后索引仍在恢复，或向量库过大")
        return error
    }

    public func close() {
        // 有同步在途时不要关闭连接：事务被打断会留下未完成状态。
        // 记下 pending，等同步收尾后自动关闭，避免旧连接长期悬挂。
        guard tryBeginIngest() else {
            storeLock.lock()
            closePending = true
            storeLock.unlock()
            cancel()
            AgentLog.warning("RAG: 同步进行中，close 推迟到同步结束")
            return
        }
        defer { releaseIngestFlag() }
        searchLock.lock(); defer { searchLock.unlock() }
        storeLock.lock(); defer { storeLock.unlock() }
        store?.close()
        store = nil
        kgLock.lock()
        kgGraph = nil
        kgStore = nil
        kgLock.unlock()
    }

    public func reload() {
        guard tryBeginIngest() else {
            AgentLog.warning("RAG: 同步进行中，跳过 reload")
            return
        }
        defer {
            invalidateStatusCache()
            releaseIngestFlag()
        }
        searchLock.lock(); defer { searchLock.unlock() }
        storeLock.lock()
        store?.close()
        store = nil
        storeLock.unlock()
    }

    // MARK: 知识图谱

    /// 知识图谱快照文件路径。
    public var knowledgeGraphSnapshotPath: String {
        (ragDir as NSString).appendingPathComponent("knowledge_graph.snapshot.json")
    }

    /// 懒加载知识图谱存储层（独立 SQLite 连接 + embedding）。
    private func ensureKnowledgeGraphStore() throws -> KnowledgeGraphStore {
        kgLock.lock(); defer { kgLock.unlock() }
        if let kgStore { return kgStore }
        guard let embedding else {
            throw AgentError.storage("embedding 模型不可用，无法使用知识图谱")
        }
        let s = try KnowledgeGraphStore(dbPath: indexDBPath, embedding: embedding,
                                        enableSQLiteVec: config.ragSQLiteVecEnabled)
        kgStore = s
        return s
    }

    /// 懒加载内存图谱（优先快照恢复，否则从数据库加载）。
    private func ensureKnowledgeGraph() throws -> (KnowledgeGraphStore, KnowledgeGraph) {
        let s = try ensureKnowledgeGraphStore()
        if kgGraph == nil {
            let g = KnowledgeGraph(snapshotPath: knowledgeGraphSnapshotPath,
                                   revisionProvider: { [weak s] in try? s?.revision() })
            try? g.loadOrRestore(from: s)
            kgGraph = g
        }
        guard let g = kgGraph else {
            throw AgentError.storage("知识图谱初始化失败")
        }
        return (s, g)
    }

    /// 图谱状态（不加载 embedding 模型，直接查 SQLite）。
    public func knowledgeGraphStatus() -> (built: Bool, entities: Int, relationships: Int) {
        guard let db = try? SQLiteDatabase(path: indexDBPath, readOnly: true) else {
            return (false, 0, 0)
        }
        defer { db.close() }
        let e = (try? db.scalar("SELECT COUNT(*) FROM \(KnowledgeGraphStore.entitiesTable)").intValue) ?? 0
        let r = (try? db.scalar("SELECT COUNT(*) FROM \(KnowledgeGraphStore.relationshipsTable)").intValue) ?? 0
        let entities = Int(e), relationships = Int(r)
        let state = (try? db.scalar("SELECT value FROM meta WHERE key = ?",
                                    [.text(KnowledgeGraphStore.buildStateKey)]))?.stringValue
        let built: Bool
        if let state {
            // 新库以构建状态为准：只有全量抽取并建图完成（complete）才算建成。
            built = state == KnowledgeGraphStore.BuildState.complete.rawValue
        } else {
            // 旧库没有状态标记：沿用「有实体或关系」的判定。
            built = entities > 0 || relationships > 0
        }
        return (built: built, entities: entities, relationships: relationships)
    }

    /// 构建知识图谱：后台用辅助模型在已完成切片上抽取实体/关系，聚合入库，
    /// 随后建 SwiftGraph 并落快照。weight 随证据重算（不保留旧标量）。
    @discardableResult
    public func buildKnowledgeGraph(
        auxLLM: LLMService? = nil,
        progress: ((Int, Int, String) -> Void)? = nil,
        log: ((String) -> Void)? = nil
    ) async -> KnowledgeGraphBuilder.BuildStats {
        var stats = KnowledgeGraphBuilder.BuildStats()
        var priorState: KnowledgeGraphStore.BuildState?
        var priorRevision = 0
        do {
            let llm = auxLLM ?? self.auxLLM
            guard let llm else {
                log?("[kg] 辅助模型不可用，无法构建知识图谱（请先在配置中启用辅助模型）")
                return stats
            }
            self.auxLLM = llm
            let store = try ensureKnowledgeGraphStore()
            priorState = try store.buildState()
            priorRevision = store.revision()
            // 先在旁路图上构建，成功后才替换内存图；失败时旧图谱仍可用。
            let graph = KnowledgeGraph(snapshotPath: knowledgeGraphSnapshotPath,
                                       revisionProvider: { [weak store] in
                                           try? store?.revision()
                                       })
            let builder = KnowledgeGraphBuilder(store: store, graph: graph, llm: llm,
                                                limits: kgExtractionLimits,
                                                format: config.ragKGExtractionFormat,
                                                maxConcurrency: config.ragKGMaxConcurrency)
            stats = try await builder.build(progress: progress, log: log)
            kgLock.lock()
            kgGraph = graph
            kgLock.unlock()
        } catch {
            if let store = try? ensureKnowledgeGraphStore() {
                // 数据库已被原子替换过（revision 变化）则只能等重建；
                // 否则恢复构建前状态，旧图谱继续可用。
                if store.revision() != priorRevision {
                    try? store.markBuildState(.stale)
                } else {
                    try? store.markBuildState(priorState ?? .stale)
                }
                // 抽取缓存按内容哈希保留：重试时可直接复用已完成的切片。
            }
            log?("[kg] 知识图谱构建失败: \(error.localizedDescription)")
        }
        return stats
    }

    /// 知识库内容变化后图谱失效：标记 stale 并丢弃内存图，检索自动回退 naive，
    /// 直到用户重新构建知识图谱。
    private func invalidateKnowledgeGraph(reason: String) {
        guard let store = try? ensureKnowledgeGraphStore() else { return }
        do {
            try store.markBuildState(.stale)
            kgLock.lock()
            kgGraph = nil
            kgLock.unlock()
            AgentLog.info("[kg] 图谱已失效（\(reason)），重新构建后才能使用图谱检索")
        } catch {
            AgentLog.warning("[kg] 图谱失效标记写入失败：\(error.localizedDescription)")
        }
    }

    /// 新切片入库后，仅对未进入图谱的切片做增量抽取，并合并进现有图谱与快照。
    private func updateKnowledgeGraphIncrementally() async {
        guard let llm = auxLLM else { return }
        guard let store = try? ensureKnowledgeGraphStore(),
              store.buildState() == .complete else { return }
        // 已有切片被改写/删除时，增量无法撤回旧证据，只能整体重建。
        guard let refs = try? store.listChunkRefs() else { return }
        let current = Dictionary(uniqueKeysWithValues: refs.map {
            ($0.id, KnowledgeGraphBuilder.contentHash($0.text))
        })
        let processed = store.processedChunkHashes()
        if processed.contains(where: { current[$0.key] != $0.value }) {
            invalidateKnowledgeGraph(reason: "已有切片被修改或删除")
            return
        }
        guard let (_, graph) = try? ensureKnowledgeGraph() else { return }
        let builder = KnowledgeGraphBuilder(store: store, graph: graph, llm: llm,
                                            limits: kgExtractionLimits,
                                            format: config.ragKGExtractionFormat,
                                            maxConcurrency: config.ragKGMaxConcurrency)
        do {
            let result = try await builder.buildIncremental()
            // 只对变更项做内存图 upsert，避免整图重建；快照交给 1 秒防抖保存。
            for entity in result.changedEntities {
                graph.replaceEntity(entity)
            }
            for id in result.removedRelationshipIDs {
                graph.removeRelationship(id: id)
            }
            for relationship in result.changedRelationships {
                graph.replaceRelationship(relationship)
            }
            if result.stats.chunksProcessed > 0 {
                AgentLog.info("[kg] 增量更新：\(result.stats.chunksProcessed) 个切片，"
                    + "变更实体 \(result.stats.entities)、关系 \(result.stats.relationships)")
            }
        } catch {
            try? store.markBuildState(.stale)
            AgentLog.warning("[kg] 增量更新失败，图谱标记为 stale：\(error.localizedDescription)")
        }
    }

    /// 手动触发图谱实体消歧：用向量语义相似度对同指实体做进一步合并。
    ///
    /// 「构建知识图谱」阶段仅做字符全匹配（快速、不依赖向量），本方法用于事后
    /// 「进一步」收敛重复实体（如「川普」与「特朗普」）。需要 embedding 可用
    /// （懒加载 store 会校验，缺失时抛错）。合并后重建内存图并落快照。
    @discardableResult
    public func resolveKnowledgeGraphDuplicates(threshold: Double = 0.90) async throws
        -> (mergedGroups: Int, removedEntities: Int, entities: Int, relationships: Int) {
        let store = try ensureKnowledgeGraphStore()
        let (_, graph) = try ensureKnowledgeGraph()
        let result = try await store.resolveEntityDuplicates(threshold: threshold)
        graph.rebuild(entities: result.entities, relationships: result.relationships)
        return (result.mergedGroups, result.removedEntities,
                result.entities.count, result.relationships.count)
    }

    // MARK: - 知识图谱 MCP 门面

    /// 图谱检索策略（GraphRAG 风格）。
    public enum GraphSearchMode: String, Sendable {
        /// 从命中实体拉邻边与关联实体，适合问具体细节。
        case local
        /// 从命中关系反查两边实体及度数，适合跨文档的宏观主题/总结。
        case global
        /// 实体和关系两路都做，子图轮询交织合并，适合既要细节又要全貌。
        case hybrid
        /// 实体 + 关系 + 原文 chunk 全部都做，适合在图谱语境之上还想要原始文本片段兜底。
        case mix
    }

    /// 图谱检索结果。
    public struct GraphSearchResult: Sendable {
        public let mode: String
        public let entities: [KGEntity]
        public let relationships: [KGRelationship]
        public let chunks: [String]
        public init(mode: String, entities: [KGEntity],
                    relationships: [KGRelationship], chunks: [String]) {
            self.mode = mode
            self.entities = entities
            self.relationships = relationships
            self.chunks = chunks
        }
    }

    public func graphEntity(named name: String) -> KGEntity? {
        (try? ensureKnowledgeGraph().1.entity(named: name)) ?? nil
    }

    public func graphNeighbors(name: String) -> [String] {
        (try? ensureKnowledgeGraph().1.neighbors(of: name)) ?? []
    }

    public func graphDegree(name: String) -> Int {
        (try? ensureKnowledgeGraph().1.degree(of: name)) ?? 0
    }

    public func graphIncidentRelationships(name: String) -> [KGRelationship] {
        (try? ensureKnowledgeGraph().1.incidentRelationships(of: name)) ?? []
    }

    public func graphSubgraph(root: String, depth: Int,
                              edgeTypes: [String]?) -> KGSubgraph {
        (try? ensureKnowledgeGraph().1.subgraph(root: root, depth: depth, edgeTypes: edgeTypes))
            ?? KGSubgraph(entities: [], relationships: [])
    }

    public func graphPath(from a: String, to b: String,
                          algorithm: KGPathAlgorithm) -> KGPath {
        (try? ensureKnowledgeGraph().1.shortestPath(from: a, to: b, algorithm: algorithm))
            ?? KGPath(nodes: [], relationships: [])
    }

    /// 新增实体节点（写入存储层 + 内存图，并落快照）。
    public func graphAddEntity(name: String, type: String = "") async throws -> KGEntity {
        let store = try ensureKnowledgeGraphStore()
        let (_, graph) = try ensureKnowledgeGraph()
        let now = Date().timeIntervalSince1970
        // 同名实体（含 UNKNOWN 占位）就地升级，避免同一实体出现两行。
        if var existing = try store.entityByName(name) {
            if !type.isEmpty { existing.type = type }
            existing.updatedAt = now
            try store.upsertEntity(existing)
            graph.upsertEntity(existing)
            graph.flush()
            return existing
        }
        let entity = KGEntity(id: UUID().uuidString, name: name, type: type,
                              description: "", properties: [:], sourceIDs: [],
                              createdAt: now, updatedAt: now)
        try store.upsertEntity(entity)
        graph.upsertEntity(entity)
        graph.flush()
        return entity
    }

    /// 新增关系边（同一对实体合并 keywords/来源；端点不存在时落库 `UNKNOWN` 占位）。
    public func graphAddRelationship(source: String, target: String,
                                     relation: String) async throws -> KGRelationship {
        let store = try ensureKnowledgeGraphStore()
        let (_, graph) = try ensureKnowledgeGraph()
        let now = Date().timeIntervalSince1970
        if var existing = try store.relationship(byPair: source, dst: target) {
            var keywords = Set(existing.relationType.split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty })
            let trimmed = relation.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { keywords.insert(trimmed) }
            existing.relationType = keywords.sorted().joined(separator: ",")
            existing.updatedAt = now
            try store.upsertRelationship(existing)
            graph.upsertRelationship(existing)
            graph.flush()
            return existing
        }
        let rel = KGRelationship(id: UUID().uuidString, srcEntity: source, dstEntity: target,
                                 relationType: relation, description: "", weight: 1.0,
                                 properties: [:], sourceIDs: [], createdAt: now, updatedAt: now)
        try store.upsertRelationship(rel)
        for endpoint in [source, target] {
            if (try store.entityByName(endpoint)) == nil {
                let placeholder = KGEntity(id: "unknown-\(endpoint.lowercased())",
                                           name: endpoint, type: "UNKNOWN",
                                           description: "", properties: [:], sourceIDs: [],
                                           createdAt: now, updatedAt: now)
                try store.upsertEntity(placeholder)
            }
        }
        graph.upsertRelationship(rel)
        graph.flush()
        return rel
    }

    /// 删除实体节点及其关联边（存储层 + 内存图同步）。
    public func graphDeleteEntity(name: String) async throws {
        let store = try ensureKnowledgeGraphStore()
        let (_, graph) = try ensureKnowledgeGraph()
        try store.deleteEntity(name: name)
        graph.deleteEntity(name)
        graph.flush()
    }

    /// 图谱检索（GraphRAG 风格）：local/global/hybrid/mix 四种策略。
    /// 命中实体/关系先用字符包含匹配（不依赖 embedding，始终可用）；
    /// mix 模式额外叠加原文 chunk 兜底。
    public func graphSearch(query: String, mode: GraphSearchMode,
                            topK: Int = 20) async throws -> GraphSearchResult {
        let (_, graph) = try ensureKnowledgeGraph()
        let tokens = query.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count >= 2 }
        let allE = graph.allEntities()
        let allR = graph.allRelationships()
        let matchedE = allE.filter { e in
            let hay = (e.name + " " + e.type + " " + e.description).lowercased()
            return tokens.contains { hay.contains($0) }
        }
        let matchedR = allR.filter { r in
            let hay = (r.relationType + " " + r.srcEntity + " " + r.dstEntity + " " + r.description)
                .lowercased()
            return tokens.contains { hay.contains($0) }
        }

        var entities: [KGEntity] = []
        var relationships: [KGRelationship] = []

        func collectEntity(_ e: KGEntity) {
            if !entities.contains(where: { $0.name.lowercased() == e.name.lowercased() }) {
                entities.append(e)
            }
        }
        func collectRelationship(_ r: KGRelationship) {
            if !relationships.contains(where: { $0.id == r.id }) {
                relationships.append(r)
            }
        }

        switch mode {
        case .local:
            // 命中实体的 1 跳邻居子图（适合具体细节）。
            for e in matchedE {
                collectEntity(e)
                for nb in graph.neighbors(of: e.name) {
                    if let ne = graph.entity(named: nb) { collectEntity(ne) }
                }
                for rel in graph.incidentRelationships(of: e.name) { collectRelationship(rel) }
            }
        case .global:
            // 命中关系反查两端实体 + 度数（适合跨文档宏观主题/总结）。
            for r in matchedR {
                collectRelationship(r)
                for end in [r.srcEntity, r.dstEntity] {
                    if let e = graph.entity(named: end) { collectEntity(e) }
                }
            }
            entities.sort { graph.degree(of: $0.name) > graph.degree(of: $1.name) }
            relationships.sort { graph.relationRank($0) > graph.relationRank($1) }
        case .hybrid:
            // 实体 + 关系两路都做，子图轮询交织合并。
            for e in matchedE {
                collectEntity(e)
                for nb in graph.neighbors(of: e.name) {
                    if let ne = graph.entity(named: nb) { collectEntity(ne) }
                }
                for rel in graph.incidentRelationships(of: e.name) { collectRelationship(rel) }
            }
            for r in matchedR {
                collectRelationship(r)
                for end in [r.srcEntity, r.dstEntity] {
                    if let e = graph.entity(named: end) { collectEntity(e) }
                }
            }
        case .mix:
            // hybrid + 原文 chunk 兜底。
            let hybrid = try await graphSearch(query: query, mode: .hybrid, topK: topK)
            entities = hybrid.entities
            relationships = hybrid.relationships
        }

        if entities.count > topK { entities = Array(entities.prefix(topK)) }
        if relationships.count > topK { relationships = Array(relationships.prefix(topK)) }

        var chunks: [String] = []
        if mode == .mix {
            if let hits = try? await search(query: query, topK: min(topK, 10), source: nil) {
                chunks = hits.map { $0.text }
            }
        }
        return GraphSearchResult(mode: mode.rawValue, entities: entities,
                                 relationships: relationships, chunks: chunks)
    }

    /// 启动时把本地 Embedding / Reranker 权重加载好并保持常驻。
    ///
    /// 首次检索现场加载 ONNX session 实测要 10~15s；提前加载后首问只付检索时间。
    /// 单个模型失败只记录到 `failures`，不影响另一个模型继续预加载。
    @discardableResult
    public func preloadAuxiliaryModels() async -> RAGPreloadOutcome {
        var loaded: [String] = []
        var failures: [String] = []
        if let embedding {
            do {
                try await embedding.preload()
                loaded.append(embedding.identifier)
            } catch {
                failures.append("Embedding(\(embedding.identifier)): "
                    + error.localizedDescription)
            }
        }
        // 关闭重排序时不必白占内存加载 reranker。
        if config.ragRerankEnabled, let reranker {
            do {
                try await reranker.preload()
                loaded.append(reranker.identifier)
            } catch {
                failures.append("Reranker(\(reranker.identifier)): "
                    + error.localizedDescription)
            }
        }
        if !loaded.isEmpty {
            AgentLog.info("[rag] 预加载完成并常驻：\(loaded.joined(separator: "、"))")
        }
        for failure in failures {
            AgentLog.warning("[rag] 预加载失败：\(failure)")
        }
        return RAGPreloadOutcome(loaded: loaded, failures: failures)
    }

    /// 尝试占用 ingest 独占标志；已有 ingest/close/reload 时返回 false。
    private func tryBeginIngest() -> Bool {
        ingestFlagLock.lock(); defer { ingestFlagLock.unlock() }
        guard !ingestActive else { return false }
        ingestActive = true
        return true
    }

    private func releaseIngestFlag() {
        ingestFlagLock.lock()
        ingestActive = false
        ingestFlagLock.unlock()
    }

    /// 同步结束后的延迟关闭（由 ingest 的 defer 调用）。
    ///
    /// 不能直接调用 `close()`：它会把 `cancelRequested` 置为 true，
    /// 若此时已有新的 ingest 抢到标志，会把新同步误取消。
    private func closeIfPending() {
        storeLock.lock()
        let pending = closePending
        storeLock.unlock()
        guard pending else { return }
        // 另一个 ingest 已开始：保留 pending，由它的 defer 再次尝试。
        guard tryBeginIngest() else { return }
        defer { releaseIngestFlag() }
        storeLock.lock()
        closePending = false
        storeLock.unlock()
        searchLock.lock(); defer { searchLock.unlock() }
        storeLock.lock()
        store?.close()
        store = nil
        storeLock.unlock()
    }

    public func cancel() {
        cancelLock.lock(); defer { cancelLock.unlock() }
        cancelRequested = true
    }

    private var isCancelled: Bool {
        cancelLock.lock(); defer { cancelLock.unlock() }
        return cancelRequested
    }

    // MARK: Ingest

    /// 增量同步知识库。返回统计；同一实例并发调用会被拒绝。
    ///
    /// ANN 索引的自动构建在同步收尾、释放独占标志后转到后台任务执行，
    /// 因此同步一返回，检索与状态查询即可继续（构建不持有 `searchLock`）。
    public func ingest(force: Bool = false,
                       progress: ((Int, Int, String) -> Void)? = nil,
                       log: ((String) -> Void)? = nil) async -> IngestStats {
        guard tryBeginIngest() else {
            var stats = IngestStats()
            stats.errors.append("另一次同步正在进行，已忽略本次请求")
            return stats
        }
        let logFn: (String) -> Void = { message in
            if let log { log(message) } else { AgentLog.info(message) }
        }
        // 同步期间允许两个 Ovis 实例并行识别；最后一个 OCR 任务结束时
        // 由 OCRModelLifecycle 立即释放权重，不等 3 分钟空闲回收。
        await LocalModelFactory.shared.beginOCRJob()
        var stats = IngestStats()
        var autoBuildRequested = false
        defer {
            invalidateStatusCache()
            releaseIngestFlag()
            closeIfPending()
            if autoBuildRequested {
                scheduleANNAutoBuild(log: logFn)
            }
            Task { await LocalModelFactory.shared.endOCRJob() }
        }
        cancelLock.lock(); cancelRequested = false; cancelLock.unlock()

        guard FileManager.default.fileExists(atPath: knowledgeBase) else {
            stats.errors.append("知识库目录不存在: \(knowledgeBase)")
            return stats
        }
        guard embedding != nil else {
            stats.errors.append("embedding 模型不可用")
            return stats
        }

        // 解析引擎 / 切片参数 / embedding 模型变化会让旧切片失效。
        // 签名按文件记录在 manifest 中：配置变化时逐文件重建，中途取消后
        // 已完成的文件保持新签名，下次同步只继续处理剩余文件。
        let configSignature = makeIngestSignature()
        let signatureHash = Self.contentHash(configSignature)
        let effectiveForce = force
        // 每次同步都打印一次生效的切片参数：用户可据此核对「面板设置 →
        // 实际切片 → 检索呈现」是否一致，而不是只看面板数字。
        let effectiveChunking = Self.resolvedChunkingOptions(config: config,
                                                             embedding: embedding)
        logFn("RAG: 切片参数 \(effectiveChunking.options.chunkSizeTokens) token"
            + "（含 overlap \(effectiveChunking.options.overlapTokens)），"
            + "语义切片 \(config.ragSemanticChunking ? "开" : "关")"
            + (effectiveChunking.clampedFrom.map {
                "，面板设置的 \($0) 已按 embedding 模型输入上限收口"
            } ?? ""))

        let files = scanFiles()
        logFn("RAG: 开始同步，共 \(files.count) 个文件（force=\(effectiveForce)）")
        logFn("RAG: 初始化索引库…")
        let store: VectorStore
        do {
            store = try ensureStore()
        } catch {
            stats.errors.append(error.localizedDescription)
            logFn("RAG: 索引库初始化失败：\(error.localizedDescription)")
            return stats
        }
        let existingChunkCount = store.count
        // B 方案闸门：配置与索引指纹不符时不静默删数据，但也不能继续写入
        // （新向量与旧向量语义/类型不同，混在一个库里检索结果不可信）。
        if let reason = store.rebuildReason {
            stats.errors.append("向量库需要重建（\(reason)）；"
                + "请先在设置中执行「重建向量库」再同步")
            logFn("RAG: 向量库待重建（\(reason)），已拒绝本次同步；全文检索仍可用")
            return stats
        }
        logFn("RAG: 索引库就绪（现有 \(existingChunkCount) 切片）")
        if effectiveForce {
            // force 会先删后插全部来源；旧索引的 rowid 全部失效，续建没有意义。
            // 同步开始前直接重置，结束后由后台任务按需全量重建。
            do {
                try store.dropANNIndex()
                logFn("RAG: 强制同步前已重置 ANN 索引，结束后将按需重建")
            } catch {
                logFn("RAG: 重置 ANN 索引失败（不影响本次同步）：\(error.localizedDescription)")
            }
        }

        var manifest = manifestStore.load()
        logFn("RAG: manifest 载入 \(manifest.count) 条")
        // 数据库被删除/重建但 manifest 还在时，增量逻辑会把所有文件当成
        // 「未变化」而跳过，知识库永久为空；此时必须全量重建。
        if !store.hasChunks, !manifest.isEmpty {
            logFn("RAG: 向量库为空但 manifest 仍存在，执行全量重建")
            manifest.removeAll()
            manifestStore.clear()
        }
        // 显式强制全量重建时先把所有条目标记为待重建：如果用户中途停止，
        // 未处理文件不会带着旧签名被下次同步误判为「已完成」。
        if effectiveForce, !manifest.isEmpty {
            let stale = manifest.mapValues { entry -> RAGManifestEntry in
                var copy = entry
                copy.ingestSignature = nil
                return copy
            }
            manifestStore.merge(updated: stale, removed: [])
            manifest = stale
        }
        let staleCount = manifest.values.filter { $0.ingestSignature != signatureHash }.count
        if !effectiveForce, staleCount > 0 {
            logFn("RAG: \(staleCount) 个文件使用旧签名（配置变化或上次未完成），将逐文件重建")
        }

        // 原始文件采样指纹去重：同一本书放在不同目录时只解析一个代表文件。
        // 默认信任 manifest 里的 mtime + size + 指纹，未变化文件不再读盘采样；
        // force 或 verify_fingerprint 开启时才重新校验内容。
        let (uniqueFiles, duplicateGroups) = Self.groupDuplicateFiles(
            files, manifest: manifest, force: effectiveForce,
            verifyFingerprint: config.ragVerifyFingerprint)
        let total = uniqueFiles.count
        let duplicateCount = duplicateGroups.reduce(0) { $0 + $1.duplicates.count }
        if duplicateCount > 0 {
            logFn("RAG: 文件指纹去重：跳过 \(duplicateCount) 个重复副本，"
                + "实际处理 \(total)/\(files.count) 个文件")
        }

        let currentSources = Set(files.map { $0.path })
        var removedKeys = Set<String>()

        // 1) 清理已删除文件的向量与缓存。
        for (source, entry) in manifest where !currentSources.contains(source) {
            do {
                _ = try store.deleteBySource(source)
                // 文档摘要/大纲记录也要删除，否则 rag_outline 会一直列出已删文件。
                try? store.deleteDocumentByMdPath(entry.mdPath)
                try? store.deleteDocument(docName: entry.docName)
                try? FileManager.default.removeItem(atPath: entry.mdPath)
                removedKeys.insert(source)
                stats.removed += 1
                logFn("RAG: 删除已不存在文档 \(source)")
            } catch {
                stats.errors.append("删除失败 \(source): \(error.localizedDescription)")
            }
        }

        // 2) 多文件并发：最多 6 个文件同时解析（ragIngestConcurrency，
        //    上限 6）；每个文件内部
        //    「解析 → 切片 → embedding → 入库」形成流水线。
        //    embedding 由共享调度器串行进入模型，并按吞吐/内存自动调整子批。
        guard let baseEmbedding = embedding else {
            stats.errors.append("embedding 模型不可用")
            return stats
        }
        let scheduler = AdaptiveEmbeddingScheduler(
            base: baseEmbedding,
            options: .init(minBatchSize: config.ragEmbeddingMinBatch,
                           maxBatchSize: config.ragEmbeddingMaxBatch,
                           initialBatchSize: config.ragEmbeddingInitialBatch,
                           step: config.ragEmbeddingBatchStep,
                           memoryLimitMB: Double(config.ragEmbeddingMemoryLimitMB),
                           throughputTolerance: config.ragEmbeddingThroughputTolerance,
                           probeBatches: config.ragEmbeddingProbeBatches,
                           probeSeconds: config.ragEmbeddingProbeSeconds,
                           tokenBudgetMin: config.ragEmbeddingTokenBudgetMin,
                           tokenBudgetMax: config.ragEmbeddingTokenBudgetMax,
                           initialTokenBudget: config.ragEmbeddingTokenBudgetInitial,
                           tokenBudgetStep: config.ragEmbeddingTokenBudgetStep),
            log: logFn)
        let binarySignaturesEnabled = effectiveVectorElementType != .bin
            && scheduler.binaryDimension != nil
        let jobs = uniqueFiles.enumerated().map { index, file in
            IngestJob(index: index, total: total, file: file,
                      label: "[\(index + 1)/\(total)] \(file.relativePath)")
        }
        let accumulator = IngestAccumulator()
        let concurrency = min(max(1, config.ragIngestConcurrency), 6,
                              max(1, uniqueFiles.count))
        // 与转换工具同一套分批策略：≥16MB 的 PDF 单独成批、批间串行，
        // 避免「一个文件夹里大批大文件」同时解析导致内存叠加。
        // 文件内部仍按 PDFProcessingPlan 收紧页级窗口。
        let fileSizes = jobs.map {
            PDFProcessingPlan.fileSize(of: URL(fileURLWithPath: $0.file.path))
        }
        let batches = PDFProcessingPlan.batchIndices(
            fileSizes: fileSizes,
            baseConcurrency: concurrency,
            heavyFileBytes: PDFProcessingPlan.largeFileBytes)
        let heavyCount = fileSizes.filter { $0 >= PDFProcessingPlan.largeFileBytes }.count
        if heavyCount > 0 {
            logFn("RAG: \(heavyCount)/\(total) 个大文件将逐个解析，避免并发叠加内存")
        }
        let registry = parserRegistry
        let storeRef = store
        let manifestStoreRef = manifestStore
        // 已删除文件不再作为副本的保留来源，避免 tombstone 误判为仍完整。
        var snapshot = manifest
        for key in removedKeys { snapshot.removeValue(forKey: key) }
        let manifestSnapshot = snapshot
        // 正文哈希去重登记表：解析/清洗后就能判断的内容副本不再切片/embedding。
        // 历史 manifest 条目在首次命中时验证库内切片，避免 tombstone 指向空来源。
        let contentRegistry = ContentDedupRegistry(
            manifest: snapshot,
            sources: currentSources,
            isUsable: { source in storeRef.chunkCount(source: source) > 0 })
        let markdownDirRef = markdownDir
        let knowledgeBaseRef = knowledgeBase
        let signature = signatureHash
        let force = effectiveForce
        let configRef = config
        let cancelledCheck: () -> Bool = { [weak self] in
            guard let self else { return true }
            return self.isCancelled
        }

        // 批间串行、批内并发：轻文件最多 concurrency 个同时处理，
        // 大文件所在批次只有它自己。
        for batch in batches {
            if Task.isCancelled || cancelledCheck() { break }
            let batchJobs = batch.map { jobs[$0] }
            let queue = IngestJobQueue(batchJobs)
            await withTaskGroup(of: Void.self) { group in
                for _ in 0..<batchJobs.count {
                    group.addTask {
                        while let job = await queue.next() {
                            if Task.isCancelled || cancelledCheck() { break }
                            let outcome = await Self.process(
                                job: job,
                                force: force,
                                config: configRef,
                                binarySignaturesEnabled: binarySignaturesEnabled,
                                registry: registry,
                                contentRegistry: contentRegistry,
                                store: storeRef,
                                embedding: scheduler,
                                manifestStore: manifestStoreRef,
                                manifestSnapshot: manifestSnapshot,
                                signatureHash: signature,
                                markdownDir: markdownDirRef,
                                knowledgeBase: knowledgeBaseRef,
                                isCancelled: cancelledCheck,
                                log: logFn,
                                progress: progress
                            )
                            await accumulator.record(outcome)
                            if outcome.cancelled { break }
                        }
                    }
                }
                await group.waitForAll()
            }
        }
        // 兜底唤醒：worker 在异常路径退出时也不能让等待同一正文的任务永久挂起。
        contentRegistry.cancelAll()

        if isCancelled { stats.cancelled = true }
        for outcome in await accumulator.snapshot() {
            if outcome.discarded { stats.discarded += 1 }
            if outcome.skipped { stats.skipped += 1 }
            if outcome.processed {
                stats.processed += 1
                stats.chunks += outcome.chunks
            }
            if outcome.deduplicated { stats.deduplicated += 1 }
            if outcome.cancelled { stats.cancelled = true }
            if let error = outcome.error { stats.errors.append(error) }
        }
        // 已删除文件的 manifest 条目统一收尾；成功/失败条目已由 worker 逐文件落盘。
        manifestStore.merge(updated: [:], removed: removedKeys)

        if stats.cancelled {
            logFn("RAG: 同步已取消，跳过副本向量清理由下次完整同步执行")
        } else {
            // 同步完成后按 content hash 再做一次全局去重，兜住本轮未能提前
            // 判定（例如并发时序/旧版本 manifest）的正文副本。
            stats.deduplicated += Self.deduplicateByContentHash(
                manifestStore: manifestStore,
                duplicateGroups: duplicateGroups,
                existingSources: Set(manifest.keys),
                signatureHash: signatureHash,
                store: store,
                markdownDir: markdownDir,
                knowledgeBase: knowledgeBase,
                log: logFn)
            if stats.deduplicated > 0 {
                logFn("RAG: 副本去重完成，移除 \(stats.deduplicated) 个重复来源的向量")
            }
        }
        // 全局签名文件仅作「最近一次完整完成」的诊断记录；增量判断以每个
        // manifest 条目的 ingestSignature 为准，因此取消后仍能续跑。
        if !stats.cancelled {
            try? AtomicFile.write(Data((configSignature + "\n").utf8),
                                  to: ingestSignaturePath)
        }
        if stats.cancelled {
            logFn("RAG: 同步已停止 — \(stats.summary)")
            logFn("RAG: 已完成文件已保留，下次同步将从剩余未完成文件继续")
        } else {
            progress?(total, total, "")
            let metrics = scheduler.metricsSnapshot()
            if metrics.texts > 0 {
                logFn(String(format: "RAG: embedding 实测 %ld token，%.0f token/秒，"
                             + "峰值内存 %.0f MB，当前预算 %ld token / 最多 %ld 条",
                             metrics.tokens, metrics.tokensPerSecond,
                             metrics.peakMemoryMB, metrics.tokenBudget,
                             metrics.maxBatchTexts))
            }
            logFn("RAG: 同步完成 — \(stats.summary)")
            // 同步成功后按需构建/续建 ANN 索引。放到 defer 里、独占标志释放后
            // 由后台任务执行：同步返回即可继续检索，构建不再冻结读路径。
            autoBuildRequested = true
        }
        // journal 已在 worker 内增量追加；同步结束时压实成单文件基线，
        // 下一次加载不必回放整轮同步的全部记录。放在所有 merge（含收尾去重）
        // 之后，保证 manifest.json 对诊断/外部读取仍是最新完整快照。
        manifestStore.compact()
        // 新切片/去重完成后：去重改变了 chunk 标识，只能重建；纯新增则增量更新图谱。
        if stats.deduplicated > 0 {
            invalidateKnowledgeGraph(reason: "切片去重改变了 chunk 标识")
        } else if stats.processed > 0 {
            await updateKnowledgeGraphIncrementally()
        }
        return stats
    }

    /// 影响切片内容的配置签名。
    private func makeIngestSignature() -> String {
        let embeddingModel = config.localEmbeddingModel.isEmpty
            ? config.ragEmbeddingModel : config.localEmbeddingModel
        let fields: [(String, String)] = [
            // 渲染/清洗逻辑变化时递增，强制旧缓存文件重建。
            // 8：结构化清洗档（HTML/Office/EPUB 保真）+ HTML 本地框架接力 +
            //    Office 备注/列表/脚注/日期格式完善。
            // 9：语义切片改为 CJK 无空格拼接 + 弱边界短片合并。
            // 10：语义切片相似度改用内置 ONNX 模型（CPU）。
            // 11：语义切片模型更换为 potion-multilingual-128M int4（无 512 token 上限）。
            ("pipeline_version", "11"),
            ("pdf_engine", config.ragPDFEngine.rawValue),
            ("epub_ocr_engine", config.ragEPUBOCREngine.rawValue),
            ("ocr_engine", config.ragOCREngine),
            ("ocr_model", config.ragOCRModelName),
            ("ocr_path", config.ragOCRModelPath),
            ("ocr_preprocess", String(config.ragOCRPreprocess)),
            ("ocr_binarize", String(config.ragOCRBinarize)),
            ("chunk_size", String(config.ragChunkSize)),
            ("chunk_overlap", String(config.ragChunkOverlap)),
            // 语义切片开关会改变切片边界：不进签名的话，切换后旧切片会被
            // 当作「文件未变化」继续沿用，设置等于没生效。
            ("semantic_chunking", String(config.ragSemanticChunking)),
            // embedding 后端 / 权重目录决定向量语义与可用窗口（切片收口上限），
            // 换后端后旧向量必须重算。
            ("embedding_backend", AgentConfig.normalizeEmbeddingBackend(
                config.localEmbeddingBackend)),
            ("embedding_path", config.localEmbeddingModelPath),
            ("embedding_model", embeddingModel),
            // 落盘元素类型与 Matryoshka 截断维都会改变向量本身（量化/截断），
            // 旧切片必须按新配置重算；未实际截断时记 0，避免无谓重建。
            ("vector_element_type", effectiveVectorElementType.rawValue),
            ("matryoshka_dim", String(effectiveMatryoshkaDimension)),
        ]
        return fields.map { "\($0.0)=\($0.1)" }.joined(separator: "\n")
    }

    private var ingestSignaturePath: String {
        (ragDir as NSString).appendingPathComponent("ingest_signature.txt")
    }

    private func scanFiles() -> [ScannedFile] {
        let fm = FileManager.default
        let base = URL(fileURLWithPath: knowledgeBase)
        guard let enumerator = fm.enumerator(at: base,
                                             includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey],
                                             options: [.skipsHiddenFiles, .skipsPackageDescendants]) else {
            return []
        }
        var output: [ScannedFile] = []
        for case let url as URL in enumerator {
            let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey])
            if values?.isDirectory == true {
                if url.lastPathComponent == ".rag" || url.lastPathComponent == "rag" {
                    enumerator.skipDescendants()
                }
                continue
            }
            guard values?.isRegularFile == true else { continue }
            let ext = url.pathExtension.lowercased()
            guard Self.supportedExtensions.contains(ext) else { continue }
            let path = url.path
            let relative = path.hasPrefix(knowledgeBase)
                ? String(path.dropFirst(knowledgeBase.count)).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
                : url.lastPathComponent
            // 跳过 sidecar 与隐藏缓存。
            if relative.hasSuffix(".meta.json") { continue }
            output.append(ScannedFile(path: path, relativePath: relative))
        }
        output.sort { $0.relativePath.localizedStandardCompare($1.relativePath) == .orderedAscending }
        return output
    }

    // MARK: - 单文件流水线

    /// 一个文件的完整处理：解析 → 清洗 → 切片 → embedding → 入库 → manifest。
    /// 多文件由 `ingest` 的 worker 池并发调用；共享的 store/manifest/scheduler
    /// 内部都有锁或串行队列。
    private static func process(
        job: IngestJob,
        force: Bool,
        config: AgentConfig,
        binarySignaturesEnabled: Bool,
        registry: DocumentParserRegistry,
        contentRegistry: ContentDedupRegistry,
        store: VectorStore,
        embedding: AdaptiveEmbeddingScheduler,
        manifestStore: ManifestStore,
        manifestSnapshot: [String: RAGManifestEntry],
        signatureHash: String,
        markdownDir: String,
        knowledgeBase: String,
        isCancelled: () -> Bool,
        log: @escaping (String) -> Void,
        progress: ((Int, Int, String) -> Void)?
    ) async -> IngestOutcome {
        let file = job.file
        let source = file.path
        var outcome = IngestOutcome(source: source)
        // 认领正文哈希后，无论成功、失败还是取消都要通知等待者，避免挂死。
        var contentDigest: String?
        defer {
            if let digest = contentDigest {
                if outcome.processed {
                    contentRegistry.complete(digest: digest, source: source)
                } else {
                    contentRegistry.fail(digest: digest, source: source)
                }
            }
        }

        if isCancelled() || Task.isCancelled {
            outcome.cancelled = true
            return outcome
        }

        let fileStarted = Date()
        log("RAG: \(job.label) 开始处理（\(Self.fileSizeText(source))）")
        progress?(job.index + 1, job.total, file.relativePath)
        let signature = Self.fileSignature(source)

        // 内容重复副本：文件未变化且保留来源仍完整时直接跳过，不重复解析。
        if !force, let signature, let entry = manifestSnapshot[source],
           let canonical = entry.duplicateOf, !canonical.isEmpty,
           entry.ingestSignature == signatureHash,
           abs(entry.mtime - signature.mtime) < 0.001,
           entry.size == signature.size,
           !entry.fingerprint.isEmpty,
           (!config.ragVerifyFingerprint
                || Self.sourceFingerprint(source, size: signature.size) == entry.fingerprint),
           let canonicalEntry = manifestSnapshot[canonical],
           canonicalEntry.complete == true,
           (canonicalEntry.chunkCount ?? 0) > 0,
           entry.digest.isEmpty || canonicalEntry.digest == entry.digest {
            outcome.skipped = true
            log("RAG: \(job.label) 跳过（重复副本，保留 \(canonicalEntry.docName)）")
            return outcome
        }

        var canSkip = false
        var redoReason: String?
        if !force, let signature, let entry = manifestSnapshot[source] {
            if entry.ingestSignature != signatureHash {
                // 配置签名变化：顶部已汇总，逐文件不再刷屏。
            } else if entry.complete != true {
                redoReason = "上次未完整完成"
            } else if entry.discarded == true {
                // 正文过短、已在之前同步中整份丢弃：文件未变化就不再解析。
                if abs(entry.mtime - signature.mtime) >= 0.001 {
                    redoReason = "文件已变更"
                } else if entry.size != signature.size {
                    redoReason = "文件大小已变更"
                } else if config.ragVerifyFingerprint,
                          Self.sourceFingerprint(source, size: signature.size) != entry.fingerprint {
                    redoReason = "文件内容已变更"
                } else {
                    outcome.skipped = true
                    log("RAG: \(job.label) 跳过（内容过短，已丢弃）")
                    return outcome
                }
            } else if entry.chunkCount == nil || (entry.chunkCount ?? 0) <= 0 {
                redoReason = "缺少切片完成标记"
            } else if abs(entry.mtime - signature.mtime) >= 0.001 {
                redoReason = "文件已变更"
            } else if entry.size != signature.size {
                redoReason = "文件大小已变更"
            } else if entry.digest.isEmpty {
                redoReason = "摘要缺失"
            } else if config.ragVerifyFingerprint,
                      Self.sourceFingerprint(source, size: signature.size) != entry.fingerprint {
                // 默认信任 mtime + size；只有显式开启 verify_fingerprint 才为
                // 「保留时间戳的同尺寸改动」再读一遍采样指纹。
                redoReason = "文件内容已变更"
            } else if entry.mdPath.isEmpty
                        || !FileManager.default.fileExists(atPath: entry.mdPath) {
                redoReason = "Markdown 缓存缺失"
            } else {
                let actualChunks = store.chunkCount(source: source)
                if actualChunks != entry.chunkCount {
                    // manifest 说完成，但库里没有对应切片（被删除/上次写入中断/外部改动）。
                    redoReason = "索引切片不完整（\(actualChunks)/\(entry.chunkCount ?? 0)）"
                } else {
                    canSkip = true
                }
            }
        }
        if canSkip {
            outcome.skipped = true
            log("RAG: \(job.label) 跳过（文件未变化）")
            return outcome
        }
        if let redoReason {
            log("RAG: \(job.label) 重新处理：\(redoReason)")
        }

        do {
            let options = ParserOptions(config: config)
            guard let parser = registry.freshParser(for: URL(fileURLWithPath: source),
                                                    options: options) else {
                throw AgentError.unsupported("没有可用的解析器: \(file.relativePath)")
            }
            let parseStarted = Date()
            let parsed = try await Self.parse(file: file, parser: parser,
                                              options: options, label: job.label, log: log)
            // 解析/OCR 可能耗时很久；完成后立刻响应停止/取消。
            if isCancelled() || Task.isCancelled { throw AgentError.cancelled }

            // 标准解析器已在 finish() 中清洗；仅对自定义解析器兜底清洗一次。
            let text = parsed.metadata["cleaned"] == "1"
                ? parsed.text
                : DocumentCleaner.clean(
                    parsed.text, keepHeadersFooters: false,
                    profile: DocumentProfile.defaultProfile(
                        for: URL(fileURLWithPath: file.path)))
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw AgentError.parse("解析结果为空")
            }
            // 正文过短：多半是抽取失败遗留的书名/目录名或图片元数据，
            // 整份丢弃，不切片、不向量化，避免碎片污染检索结果。
            // 标题/目录行（DocumentOutline 能识别的行）整行不计入正文。
            let outlineNodes = DocumentOutline.extract(text, source: file.relativePath)
            let bodyCharacters = Self.bodyCharacterCount(
                text, headingStarts: Set(outlineNodes.map(\.start)))
            if bodyCharacters < Self.minBodyCharacters {
                let previous = manifestSnapshot[source]
                let marker = RAGManifestEntry(
                    mtime: signature?.mtime ?? 0,
                    size: signature?.size ?? 0,
                    fingerprint: Self.sourceFingerprint(source, size: signature?.size),
                    digest: "",
                    docName: file.relativePath,
                    mdPath: "",
                    ingestSignature: signatureHash,
                    chunkCount: 0,
                    complete: true,
                    discarded: true)
                // 没有切片/向量，下次同步内容不变时按 manifest 标记直接跳过。
                manifestStore.merge(updated: [source: marker], removed: [])
                // 该来源此前可能已入库（例如文件被改短）：旧切片、缓存 Markdown
                // 与文档记录/摘要必须一起清掉，避免继续命中旧内容。
                removeIndexedSource(source: source, entry: previous, store: store,
                                    markdownDir: markdownDir, knowledgeBase: knowledgeBase)
                outcome.discarded = true
                log("RAG: \(job.label) 丢弃（正文仅 \(bodyCharacters) 字，视为无效碎片）")
                return outcome
            }
            let parseBreakdown = Self.parseTimingText(parsed.metadata)
            let parseSeconds = Date().timeIntervalSince(parseStarted)

            // 正文去重前移到这里：同一内容无论来自不同目录还是不同格式
            // （md/html/pdf…），都只保留一份切片与向量。命中时直接写 tombstone，
            // 省掉一次切片和整篇 embedding；命中本轮在处理的文件则等它结束，
            // 避免把 tombstone 指向最终失败/取消的来源。
            let textDigest = Self.contentHash(text)
            let claim = await contentRegistry.claim(digest: textDigest, source: source)
            switch claim {
            case .duplicate(let canonical):
                let previous = manifestSnapshot[source]
                let marker = RAGManifestEntry(
                    mtime: signature?.mtime ?? 0,
                    size: signature?.size ?? 0,
                    fingerprint: Self.sourceFingerprint(source, size: signature?.size),
                    digest: textDigest,
                    docName: file.relativePath,
                    mdPath: "",
                    ingestSignature: signatureHash,
                    chunkCount: 0,
                    complete: false,
                    duplicateOf: canonical)
                manifestStore.merge(updated: [source: marker], removed: [])
                // 该来源之前可能以独立文档入过库：清掉旧向量与 Markdown/摘要，
                // 避免检索同时命中两份正文。
                removeIndexedSource(source: source, entry: previous, store: store,
                                    markdownDir: markdownDir, knowledgeBase: knowledgeBase)
                outcome.skipped = true
                outcome.deduplicated = true
                let keepName = manifestSnapshot[canonical]?.docName
                    ?? relativeSourcePath(canonical, knowledgeBase: knowledgeBase)
                log("RAG: \(job.label) 跳过（正文与 \(keepName) 相同，已写 tombstone）")
                return outcome
            case .owner:
                contentDigest = textDigest
            }
            if isCancelled() || Task.isCancelled { throw AgentError.cancelled }

            let chunkStarted = Date()
            var chunkDocument = parsed
            chunkDocument.text = text
            var lastChunkLog = Date()
            // 切片参数只有一处来源：设置面板（再按 embedding 模型窗口收口）。
            // 检索结果呈现等下游环节读同一个生效值，避免「设置了长切片、
            // 某个环节仍按短切片处理」。收口情况在同步开头统一打印一次。
            var chunkOptions = Self.resolvedChunkingOptions(config: config,
                                                            embedding: embedding).options
            // 模型分词器可用时改用精确 token 计数：整块（含 overlap）不超过
            // 上下文窗口，避免尾部被 embedding 的 prefix(maxLength) 静默截断。
            if let provider = embedding as? EmbeddingTokenCounting,
               let counter = await provider.makeTokenCounter() {
                chunkOptions.tokenCounter = counter
            }
            let chunks = await TextChunker(options: chunkOptions)
                .chunk(document: chunkDocument, variant: .hybrid) { done, total in
                    // 语义切片用本地词向量算相似度，大文档仍可能耗时较久——
                    // 与向量化进度同节奏（60 秒一条）输出切片进度。
                    let now = Date()
                    guard now.timeIntervalSince(lastChunkLog) >= 60 else { return }
                    lastChunkLog = now
                    log("RAG: \(job.label) 切片中 \(done)/\(total) 段"
                        + "（已用 \(Self.elapsedText(chunkStarted))）")
                }
            guard !chunks.isEmpty else {
                throw AgentError.parse("切片为空")
            }
            let chunkSeconds = Date().timeIntervalSince(chunkStarted)
            if isCancelled() || Task.isCancelled { throw AgentError.cancelled }

            // markdown 已就绪后立即进入 embedding；共享调度器会串行调用模型，
            // 并按当前吞吐与进程内存自动调整子批大小。
            let embedStarted = Date()
            var lastEmbedLog = Date()
            let onEmbedProgress: (Int, Int) -> Void = { done, count in
                let now = Date()
                guard now.timeIntervalSince(lastEmbedLog) >= 60 else { return }
                lastEmbedLog = now
                log("RAG: \(job.label) 向量化中 \(done)/\(count)"
                    + "（已用 \(Self.elapsedText(embedStarted))）")
            }
            let texts = chunks.map(\.text)
            let signed: BinaryEmbeddingOutput?
            if binarySignaturesEnabled {
                signed = try await embedding.embedDocumentsWithSignatures(
                    texts, progress: onEmbedProgress)
            } else {
                signed = nil
            }
            let vectors: [[Float]]
            if let signed {
                vectors = signed.vectors
            } else {
                vectors = try await embedding.embedDocuments(
                    texts, progress: onEmbedProgress)
            }
            guard vectors.count == chunks.count else {
                throw AgentError.storage("embedding 数量与切片不一致")
            }
            if let signed, signed.signatures.count != chunks.count {
                throw AgentError.storage("embedding 签名数量与切片不一致")
            }
            let embedSeconds = Date().timeIntervalSince(embedStarted)
            if isCancelled() || Task.isCancelled { throw AgentError.cancelled }

            let writeStarted = Date()
            _ = try store.add(chunks: chunks, vectors: vectors,
                              binarySignatures: signed?.signatures)
            let writeSeconds = Date().timeIntervalSince(writeStarted)
            if isCancelled() || Task.isCancelled { throw AgentError.cancelled }

            // 缓存 Markdown/规范化正文，供 rag_outline 与调试使用。
            let cacheStarted = Date()
            let mdPath = Self.cacheMarkdownPath(for: source,
                                                markdownDir: markdownDir,
                                                knowledgeBase: knowledgeBase)
            try AtomicFile.write(text, to: mdPath)

            let digestInfo = Self.buildDigest(parsed: parsed, docName: file.relativePath,
                                              markdown: text)
            try store.upsertDocument(docName: file.relativePath,
                                     digest: digestInfo.digest,
                                     chapters: digestInfo.chapters,
                                     mdPath: mdPath,
                                     contentHash: Self.contentHash(text))
            let cacheSeconds = Date().timeIntervalSince(cacheStarted)
            // 有页面解析/OCR 失败时仍保留已成功的内容，但标记为未完整完成：
            // 下次增量同步会重试，而不是因为 manifest 命中而永久跳过。
            let failedPages = parsed.metadata["failed_pages"].flatMap { Int($0) } ?? 0

            let entry = RAGManifestEntry(
                mtime: signature?.mtime ?? 0,
                size: signature?.size ?? 0,
                fingerprint: Self.sourceFingerprint(source, size: signature?.size),
                digest: Self.contentHash(text),
                docName: file.relativePath,
                mdPath: mdPath,
                ingestSignature: signatureHash,
                chunkCount: chunks.count,
                complete: failedPages == 0
            )
            // 逐文件落盘：即使同步中途停止或进程退出，下次只重做未完成的文件。
            manifestStore.merge(updated: [source: entry], removed: [])

            outcome.processed = true
            outcome.chunks = chunks.count

            // 每个文件只输出启动和完成两条日志；完成行汇总各阶段耗时。
            let parsePart = parseBreakdown.isEmpty
                ? String(format: "%.1fs", parseSeconds)
                : String(format: "%.1fs", parseSeconds) + "（\(parseBreakdown)）"
            let pages = parsed.pageCount > 0 ? " / \(parsed.pageCount) 页" : ""
            let failedPart = failedPages > 0
                ? "，失败 \(failedPages) 页（下次同步将重试）" : ""
            log("RAG: ✓ \(job.label) 完成：\(text.count) 字符\(pages)，"
                + "\(chunks.count) 片\(failedPart)；"
                + "解析 \(parsePart)，切片 \(String(format: "%.1fs", chunkSeconds))"
                + "，embedding \(String(format: "%.1fs", embedSeconds))"
                + "（子批 \(embedding.currentBatchSize)）"
                + "，入库 \(String(format: "%.1fs", writeSeconds))"
                + "，缓存/摘要 \(String(format: "%.1fs", cacheSeconds))"
                + "，总 \(Self.elapsedText(fileStarted))"
                + "，内存 \(Int(ProcessMemory.footprintMB()))MB")
            return outcome
        } catch AgentError.cancelled {
            // 用户主动停止：不是解析失败，不能清掉该文件已有的索引。
            outcome.cancelled = true
            return outcome
        } catch {
            outcome.error = "\(file.relativePath): \(error.localizedDescription)"
            log("RAG: ✗ \(file.relativePath): \(error.localizedDescription)"
                + "（内存 \(Int(ProcessMemory.footprintMB()))MB）")
            // 文件已变更但处理失败：清掉旧索引，避免继续命中过期内容；
            // 同时移除 manifest 条目，让下次同步重试。
            if let old = manifestSnapshot[source] {
                _ = try? store.deleteBySource(source)
                try? store.deleteDocumentByMdPath(old.mdPath)
                try? store.deleteDocument(docName: old.docName)
                try? FileManager.default.removeItem(atPath: old.mdPath)
            }
            manifestStore.merge(updated: [:], removed: [source])
            return outcome
        }
    }

    private static func parse(
        file: ScannedFile,
        parser: DocumentParser,
        options: ParserOptions,
        label: String,
        log: @escaping (String) -> Void
    ) async throws -> ParsedDocument {
        let url = URL(fileURLWithPath: file.path)
        // 解析器按页/章节回调进度；日志只在开始时、每分钟和结束时输出，
        // 避免逐页刷屏，又能判断长文档没有卡死。
        let parseStarted = Date()
        var lastLogAt = Date()
        var document = try await parser.parse(url: url, options: options) { event in
            switch event {
            case .page(let index, let total, let stage):
                let now = Date()
                guard now.timeIntervalSince(lastLogAt) >= 60 else { return }
                lastLogAt = now
                log("RAG: \(label) 解析中：第 \(index)/\(total) 页"
                    + " · \(Self.stageText(stage))"
                    + "（已用 \(Self.elapsedText(parseStarted))）")
            case .message(let message):
                // 解析器只保留关键消息（如首次加载 OCR 模型），直接输出。
                log("RAG: \(label) \(message)")
            }
        }
        document.source = file.path
        return document
    }

    // MARK: Digest

    /// 只统计正文文字：标题/目录行（`DocumentOutline.extract` 识别的行）整行不计，
    /// 其余行忽略空白后累加，含标点与数字。
    static func bodyCharacterCount(_ text: String, headingStarts: Set<Int>) -> Int {
        var count = 0
        var location = 0
        for raw in text.components(separatedBy: "\n") {
            let lineStart = location
            location += (raw as NSString).length + 1
            guard !headingStarts.contains(lineStart) else { continue }
            count += DocumentCleaner.effectiveCharacterCount(raw)
        }
        return count
    }

    /// 正文文字少于该值时整份丢弃（抽取失败遗留的书名/目录名/图片元数据）。
    static let minBodyCharacters = 100

    /// 入库与摘要共用 `OutlineResolver` 的择优结果，避免 `digest` 用解析器大纲、
    /// `chapters` 却用正则大纲的两套章节树。
    private static func buildDigest(parsed: ParsedDocument, docName: String,
                                    markdown: String) -> (digest: String, chapters: String) {
        let resolved = OutlineResolver.resolve(
            markdown: markdown,
            parsedOutline: parsed.outline,
            docName: docName,
            preferParser: OutlineResolver.prefersParserOutline(for: docName),
            includeSections: false)
        return (resolved.digest, resolved.chaptersJSON)
    }

    public func buildAndStoreDigest(docName: String, markdown: String,
                                    mdPath: String = "") throws -> String {
        let resolved = OutlineResolver.resolve(
            markdown: markdown,
            parsedOutline: "",
            docName: docName,
            preferParser: OutlineResolver.prefersParserOutline(for: docName),
            includeSections: false)
        _ = try withStore { store in
            try store.upsertDocument(docName: docName, digest: resolved.digest,
                                     chapters: resolved.chaptersJSON,
                                     mdPath: mdPath,
                                     contentHash: Self.contentHash(markdown))
        }
        return resolved.digest
    }

    // MARK: 向量去重

    /// 扫描向量库并返回重复切片统计，不删除任何数据。
    public func previewVectorDedupe() throws -> VectorDedupPreview {
        guard tryBeginIngest() else {
            throw AgentError.storage("另一个知识库任务正在进行，请稍后重试")
        }
        defer {
            releaseIngestFlag()
            closeIfPending()
        }
        let store = try ensureStore()
        let scan = try store.previewDuplicateTextChunks()
        return VectorDedupPreview(totalChunks: store.count,
                                  duplicateChunks: scan.duplicateRows,
                                  affectedSources: scan.removedBySource.count)
    }

    /// 手动清理向量库内重复的切片/向量，保留入库时间最早的一份。
    /// 同时更新 manifest：仍有剩余切片的来源刷新 chunkCount；被完全清空的
    /// 副本写 `duplicateOf` tombstone，避免下次同步重新解析。
    public func deduplicateVectors() throws -> VectorDedupStats {
        guard tryBeginIngest() else {
            throw AgentError.storage("另一个知识库任务正在进行，请稍后重试")
        }
        defer {
            invalidateStatusCache()
            releaseIngestFlag()
            closeIfPending()
        }

        let store = try ensureStore()
        let dedupe = try store.deduplicateTextChunks()
        var stats = VectorDedupStats()
        stats.removedChunks = dedupe.removedRows
        stats.affectedSources = dedupe.removedBySource.count
        stats.remainingChunks = store.count
        guard !dedupe.removedBySource.isEmpty else { return stats }

        let manifest = manifestStore.load()
        guard !manifest.isEmpty else {
            AgentLog.info("[vector-dedupe] 移除 \(stats.removedChunks) 个重复切片（无 manifest 可更新）")
            return stats
        }

        let signatureHash = Self.contentHash(makeIngestSignature())
        var updates: [String: RAGManifestEntry] = [:]
        var removedEntries = Set<String>()

        for source in dedupe.removedBySource.keys.sorted() {
            guard var entry = manifest[source] else { continue }
            let remaining = store.chunkCount(source: source)
            if remaining == 0 {
                Self.removeIndexedSource(source: source, entry: entry, store: store,
                                         markdownDir: markdownDir,
                                         knowledgeBase: knowledgeBase)
                stats.emptiedSources += 1
                if let canonical = dedupe.canonicalBySource[source],
                   canonical != source,
                   manifest[canonical] != nil {
                    entry.duplicateOf = canonical
                    entry.chunkCount = 0
                    entry.complete = false
                    entry.ingestSignature = signatureHash
                    entry.mdPath = ""
                    // 按切片去重时整篇摘要可能不同，清空 digest 让副本跳过校验
                    // 依赖 canonical 的完整性，而不是整篇内容完全一致。
                    entry.digest = ""
                    updates[source] = entry
                } else {
                    removedEntries.insert(source)
                }
            } else {
                entry.chunkCount = remaining
                entry.ingestSignature = signatureHash
                updates[source] = entry
            }
        }

        manifestStore.merge(updated: updates, removed: removedEntries)
        // 手动去重是独立操作，结束后立即压实，外部读取 manifest.json 时
        // 能看到 tombstone（不必知道 journal 的存在）。
        manifestStore.compact()
        AgentLog.info("[vector-dedupe] 移除 \(stats.removedChunks) 个重复切片，"
            + "涉及 \(stats.affectedSources) 个来源，清空 \(stats.emptiedSources) 个副本来源")
        return stats
    }

    // MARK: 切片参数

    /// 实际生效的切片参数：唯一来源是设置面板的 `rag_chunk_size` /
    /// `rag_chunk_overlap`，再按 embedding 模型的输入窗口收口——超过窗口的
    /// 尾部会被模型自己的 `prefix(maxLength)` 静默截断，那才是真正的
    /// 「设了长切片、实际按短切片入库」，所以必须显式压到上限并记录。
    /// `clampedFrom` 非 nil 表示收口前用户设置的值。
    static func resolvedChunkingOptions(
        config: AgentConfig,
        embedding: EmbeddingProvider?
    ) -> (options: ChunkingOptions, clampedFrom: Int?) {
        var options = ChunkingOptions(config: config)
        let configured = options.chunkSizeTokens
        guard let provider = embedding as? EmbeddingTokenCounting else {
            return (options, nil)
        }
        let limit = max(16, provider.maxContentTokens)
        guard configured > limit else { return (options, nil) }
        options.chunkSizeTokens = limit
        // overlap 也计入模型输入预算，同步压到 10%。
        options.overlapTokens = min(options.overlapTokens, limit / 10)
        return (options, configured)
    }

    /// 当前生效的切片 token 预算（含 overlap）。检索呈现等下游环节必须用它，
    /// 不能再用与切片大小无关的固定字符数。
    var effectiveChunkSizeTokens: Int {
        Self.resolvedChunkingOptions(config: config, embedding: embedding)
            .options.chunkSizeTokens
    }

    /// 单条检索结果的字符上限：不得小于一个切片的实际长度，否则用户把切片
    /// 调大后，检索结果仍按固定的 1500 字符截短（切了长块却只看到前半段）。
    /// 按英文安全换算（1 token ≈ 4 字符）估算并留 64 字符余量：英文最坏
    /// 情况约 4 字符/token，中文按字切分时上限更宽松，不会截断。
    static func resultCharLimit(configured: Int, chunkTokens: Int) -> Int {
        let estimated = max(0, chunkTokens) * 4 + 64
        return max(200, max(configured, estimated))
    }

    /// 格式化输出的总字符上限：不得超过内核单次工具结果上限
    /// （`kernelMaxToolChars`）。否则 RAG 认为自己输出了 12000 字符，
    /// 内核拼接对话时又会静默截到 6000，用户看到的截断位置与提示不一致。
    /// `maxTotalChars` 为调用方显式指定的上限（测试/内部用途）。
    static func outputCharLimit(config: AgentConfig, maxTotalChars: Int? = nil) -> Int {
        if let maxTotalChars { return max(1_000, maxTotalChars) }
        let kernelCap = config.kernelMaxToolChars > 0 ? config.kernelMaxToolChars : 8_000
        return max(1_000, min(config.ragSearchMaxTotalChars, kernelCap))
    }

    // MARK: ANN 索引

    /// 手动/自动构建的目标：二值签名覆盖完整时建 bann，否则建主 ANN。
    public enum ANNIndexTarget: String, Sendable {
        case dense
        case binary
    }

    /// 当前配置下应构建哪种 ANN：二值粗筛开启且签名覆盖完整 → binary。
    public func resolvedANNIndexTarget(timeout: Double? = nil) -> ANNIndexTarget {
        guard config.ragBinaryCoarseEnabled,
              effectiveVectorElementType != .bin,
              binarySignatureDimension > 0 else { return .dense }
        return (try? withStoreBounded(timeout: resolvedTimeout(timeout)) {
            annIndexTarget(for: $0)
        }) ?? .dense
    }

    private func annIndexTarget(for store: VectorStore) -> ANNIndexTarget {
        guard config.ragBinaryCoarseEnabled, store.elementType != .bin else { return .dense }
        let binary = store.binaryIndexStatus()
        return binary.isComplete && binary.dimension > 0 ? .binary : .dense
    }

    /// 当前生效的 ANN 检索参数；开关关闭时返回 nil（检索走精确路径）。
    var annQueryOptions: ANNQueryOptions? {
        guard config.ragANNEnabled else { return nil }
        return ANNQueryOptions(nprobe: max(1, config.ragANNProbe),
                               rescore: max(0, config.ragANNRescore))
    }

    /// ANN 索引状态。走带超时的只读入口：sqlite-vec 全表扫描在这类大库上
    /// 可能数分钟，状态查询超时就返回 nil，不阻塞界面。
    public func annIndexStatus(timeout: Double? = nil) -> ANNIndexStatus? {
        try? withStoreBounded(timeout: resolvedTimeout(timeout)) { $0.annIndexStatus() }
    }

    /// 二值 ANN 索引状态。与主 ANN 一样走带超时的只读入口。
    public func binaryANNIndexStatus(timeout: Double? = nil) -> ANNIndexStatus? {
        try? withStoreBounded(timeout: resolvedTimeout(timeout)) {
            $0.binaryANNIndexStatus()
        }
    }

    /// 二值粗筛状态。与 ANN 一样走带超时的只读入口。
    public func binaryIndexStatus(timeout: Double? = nil) -> BinaryIndexStatus? {
        try? withStoreBounded(timeout: resolvedTimeout(timeout)) { $0.binaryIndexStatus() }
    }

    /// 离线评测 ANN 召回质量：同一批查询分别走 ANN 与精确检索，
    /// 计算 recall@K / nDCG@K。不 hydrate 正文、不修改索引，
    /// 用于在真实知识库上验证 `ragANNProbe` / `ragANNRescore` 是否够用。
    public func evaluateANNQuality(queries: [[Float]], topK: Int = 20,
                                   nprobe: Int = 2,
                                   rescore: Int = 64) -> ANNQualityReport? {
        try? withStore { store in
            try store.evaluateANNQuality(queries: queries, topK: topK,
                                         nprobe: nprobe, rescore: rescore)
        }
    }

    /// 构建（或从断点续建）ANN 索引。与同步/清空串行，可被 `cancel()` 中断
    /// （已写完的块保留，下次接着建）。返回 nil 表示被拒绝或失败。
    ///
    /// 构建期间不持有 `searchLock`：只短暂持有 store 的写锁与 ingest 标志，
    /// 检索/状态查询可以并行执行。
    @discardableResult
    public func buildANNIndex(progress: ((ANNBuildProgress) -> Void)? = nil,
                              log: ((String) -> Void)? = nil) -> ANNBuildResult? {
        let logFn: (String) -> Void = { message in
            if let log { log(message) } else { AgentLog.info(message) }
        }
        guard tryBeginIngest() else {
            logFn("RAG: 另一次同步/维护正在进行，已忽略本次 ANN 索引构建")
            return nil
        }
        defer {
            invalidateStatusCache()
            releaseIngestFlag()
            closeIfPending()
            annBuildLock.lock()
            annBuildActive = false
            annBuildProgress = nil
            annBuildLock.unlock()
        }
        cancelLock.lock(); cancelRequested = false; cancelLock.unlock()
        return performANNBuild(progress: progress, log: logFn)
    }

    /// 实际的构建流程。调用方必须已持有 ingest 标志。
    /// 不通过 `withStore` 取 `searchLock`，只依赖 store 自身的分块锁。
    @discardableResult
    private func performANNBuild(progress: ((ANNBuildProgress) -> Void)?,
                                 log logFn: @escaping (String) -> Void) -> ANNBuildResult? {
        var lastLog = Date.distantPast
        let clusters = max(2, config.ragANNClusters)
        let clusterSize = max(1, config.ragANNClusterSize)
        do {
            let store = try ensureStore()
            let target = annIndexTarget(for: store)
            annBuildLock.lock()
            annBuildTarget = target
            annBuildLock.unlock()
            let progressHandler: (ANNBuildProgress) -> Void = { [weak self] update in
                self?.recordANNBuildProgress(update)
                progress?(update)
                // 每块日志限流：块粒度是 6 位数的切片量，逐块打印会刷屏。
                guard Date().timeIntervalSince(lastLog) >= 10,
                      update.stage == "blocks" else { return }
                lastLog = Date()
                logFn("RAG: \(update.message)")
            }
            let cancelled: () -> Bool = { [weak self] in
                Task.isCancelled || (self?.isCancelled ?? true)
            }
            let result: ANNBuildResult
            switch target {
            case .binary:
                result = try store.buildBinaryANNIndex(
                    clusters: clusters, clusterSize: clusterSize,
                    progress: progressHandler, isCancelled: cancelled)
            case .dense:
                result = try store.buildANNIndex(
                    clusters: clusters, clusterSize: clusterSize,
                    progress: progressHandler, isCancelled: cancelled)
            }
            logFn("RAG: \(result.summary)")
            invalidateStatusCache()
            return result
        } catch {
            logFn("RAG: ANN 索引构建失败：\(error.localizedDescription)")
            AgentLog.warning("RAG: ANN 索引构建失败: \(error.localizedDescription)")
            return nil
        }
    }

    /// 删除 ANN 索引（回到精确检索）。与同步/清空串行，失败原样抛出。
    public func dropANNIndex() throws {
        guard tryBeginIngest() else {
            throw AgentError.storage("另一个知识库任务正在进行，请稍后重试")
        }
        defer {
            invalidateStatusCache()
            releaseIngestFlag()
            closeIfPending()
        }
        let store = try ensureStore()
        try store.dropANNIndex()
        try store.dropBinaryANNIndex()
        AgentLog.info("RAG: ANN 与二值 ANN 索引已删除，检索回退精确扫描")
    }

    // MARK: ANN 后台自动构建

    /// ANN 自动构建的实时状态（进度 / 是否在跑）。
    public struct ANNAutoBuildState: Sendable {
        public var isRunning: Bool = false
        public var progress: ANNBuildProgress?
        public init() {}
    }

    public var annAutoBuildState: ANNAutoBuildState {
        annBuildLock.lock(); defer { annBuildLock.unlock() }
        var state = ANNAutoBuildState()
        state.isRunning = annBuildScheduled || annBuildActive
        state.progress = annBuildProgress
        return state
    }

    /// 取消正在进行的（后台/手动）ANN 构建。已写完的块保留，下次可续建。
    public func cancelANNAutoBuild() {
        cancel()
        annBuildLock.lock()
        let task = annBuildTask
        annBuildLock.unlock()
        task?.cancel()
    }

    /// 等待当前后台 ANN 构建结束（测试 / 设置面板收尾用）。
    public func awaitANNAutoBuild() async {
        annBuildLock.lock()
        let task = annBuildTask
        annBuildLock.unlock()
        await task?.value
    }

    private func recordANNBuildProgress(_ update: ANNBuildProgress) {
        annBuildLock.lock()
        annBuildActive = true
        annBuildProgress = update
        annBuildLock.unlock()
    }

    /// 同步收尾后调度后台构建。真正的「是否值得建」判断放在后台任务里，
    /// 避免把状态查询重新塞回同步返回路径。
    private func scheduleANNAutoBuild(log logFn: @escaping (String) -> Void) {
        guard config.ragANNAutoBuild,
              config.ragANNEnabled || config.ragBinaryCoarseEnabled else { return }
        annBuildLock.lock()
        guard annBuildTask == nil else {
            annBuildLock.unlock()
            return
        }
        annBuildScheduled = true
        annBuildGeneration &+= 1
        let generation = annBuildGeneration
        annBuildLock.unlock()

        let task = Task.detached(priority: .utility) { [weak self] in
            guard let self else { return }
            self.runBackgroundANNAutoBuild(log: logFn)
            self.annBuildLock.lock()
            if self.annBuildGeneration == generation {
                self.annBuildScheduled = false
                self.annBuildTask = nil
                self.annBuildActive = false
                self.annBuildProgress = nil
            }
            self.annBuildLock.unlock()
        }
        annBuildLock.lock()
        // 任务可能已经完成并清掉了 scheduled：此时不再回填已结束的句柄。
        if annBuildScheduled, annBuildGeneration == generation {
            annBuildTask = task
        }
        annBuildLock.unlock()
    }

    /// 后台自动构建：先判断是否值得建，再短暂占用 ingest 标志执行。
    private func runBackgroundANNAutoBuild(log logFn: @escaping (String) -> Void) {
        guard config.ragANNAutoBuild,
              config.ragANNEnabled || config.ragBinaryCoarseEnabled else { return }
        let overview: (chunks: Int, target: ANNIndexTarget, status: ANNIndexStatus)?
        do {
            let store = try ensureStore()
            let target = annIndexTarget(for: store)
            let status = target == .binary
                ? store.binaryANNIndexStatus()
                : store.annIndexStatus()
            overview = (store.count, target, status)
        } catch {
            logFn("RAG: 自动构建 ANN 索引跳过：\(error.localizedDescription)")
            return
        }
        guard let overview, overview.chunks >= config.ragANNMinChunks else { return }

        let chunks = overview.chunks
        let target = overview.target
        let status = overview.status
        let clusters = max(2, config.ragANNClusters)
        let clusterSize = max(1, config.ragANNClusterSize)
        let blockRows = clusters * clusterSize
        let threshold = max(config.ragANNMinChunks / 2, blockRows / 4)
        let staleNeedsRebuild = status.indexedRows > 0
            && Double(status.staleRows)
                >= Double(status.indexedRows) * VectorStore.annStaleRebuildRatio
        let label = target == .binary ? "二值 ANN 索引" : "ANN 索引"
        if status.blocks == 0 {
            logFn("RAG: 自动构建\(label)（\(chunks) 切片，\(clusters) 簇 × \(clusterSize) 行/块）")
        } else if staleNeedsRebuild {
            logFn("RAG: 自动全量重建\(label)（陈旧 \(status.staleRows)/\(status.indexedRows) 切片）")
        } else if status.unindexedRows > 0 || status.tailRows >= threshold {
            logFn("RAG: 自动续建\(label)（尾块 \(status.tailRows) 切片）")
        } else {
            return
        }

        guard tryBeginIngest() else {
            logFn("RAG: 另一次同步/维护正在进行，跳过本次自动构建\(label)")
            return
        }
        defer {
            invalidateStatusCache()
            releaseIngestFlag()
            closeIfPending()
        }
        cancelLock.lock(); cancelRequested = false; cancelLock.unlock()
        _ = performANNBuild(progress: nil, log: logFn)
    }

    // MARK: Search

    /// 多来源子串一次最多定向检索的来源数；超过时回退全局召回 + 过滤。
    private static let maxSourceGroups = 8
    /// reranker 失败/超时降级时最多返回的条数：TopN 可能被调到 5+，
    /// 但降级路径没有重排分数支撑，不能把整个 RRF 候选池塞给会话。
    private static let rerankFallbackLimit = 3

    /// rerank 墙钟超时：触发后立刻回退 RRF，不让整个检索被模型拖住。
    private struct RerankTimeout: Error {}

    /// reranker 返回分数数量与候选数不符：不能静默跳过重排（旧行为会让
    /// TopN 直接沿用未过滤的 RRF 候选），必须与失败/超时走同一条降级路径。
    private struct RerankCountMismatch: Error {
        let got: Int
        let expected: Int
    }

    /// 给可取消的异步操作加墙钟超时；0 表示不限制。
    static func withTimeout<T: Sendable>(
        _ seconds: Double,
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        guard seconds > 0 else { return try await operation() }
        return try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask { try await operation() }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                throw RerankTimeout()
            }
            defer { group.cancelAll() }
            guard let result = try await group.next() else { throw RerankTimeout() }
            return result
        }
    }

    /// 跨来源融合：每个来源内部已经做过 RRF，这里按组内名次再做一次 RRF，
    /// 避免按来源顺序拼接导致后面的来源永远排最后。
    static func fuseSourceGroups(_ groups: [[SearchHit]], topK: Int,
                                 rrfK: Int = 60) -> [SearchHit] {
        guard topK > 0, !groups.isEmpty else { return [] }
        var scores: [Int64: Double] = [:]
        var hits: [Int64: SearchHit] = [:]
        let k = Double(max(1, rrfK))
        for group in groups {
            for (rank, hit) in group.enumerated() {
                scores[hit.rowID, default: 0] += 1.0 / (k + Double(rank + 1))
                if hits[hit.rowID] == nil { hits[hit.rowID] = hit }
            }
        }
        var output = hits.values.map { hit -> SearchHit in
            var copy = hit
            let score = scores[hit.rowID] ?? 0
            copy.rrfScore = score
            copy.score = score
            return copy
        }
        output.sort { ($0.rrfScore ?? 0) > ($1.rrfScore ?? 0) }
        return Array(output.prefix(topK))
    }

    public func search(query: String, topK: Int = 3,
                       source: String? = nil,
                       timeout: Double? = nil) async throws -> [SearchHit] {
        let limit = resolvedTimeout(timeout)
        // 端到端总预算：单段各有超时（embedding 20s、查询 5s、rerank 5s），
        // 但串起来最坏 ~150s，rag_query_timeout 根本约束不到。给整段一个
        // 墙钟上限；各段超时仍生效，总预算只防「多段依次慢慢超时」叠成长卡顿。
        let totalBudget = max(30, limit * 3)
        do {
            return try await Self.withTimeout(Double(totalBudget)) {
                try await self.searchBody(query: query, topK: topK,
                                          source: source, timeout: timeout)
            }
        } catch is RerankTimeout {
            throw AgentError.timeout("rag_tools(op=\"search\") 端到端超时（\(Int(totalBudget))s），"
                + "已放弃本次检索；可稍后重试或缩小 topK")
        }
    }

    private func searchBody(query: String, topK: Int, source: String?,
                            timeout: Double?) async throws -> [SearchHit] {
        guard let embedding else { throw AgentError.storage("embedding 模型不可用") }
        let limit = resolvedTimeout(timeout)
        // topK 来自模型参数：硬性封顶，避免一次检索把 reranker 批量推到很高。
        let requested = max(1, min(topK, 20))
        // 先做耗时的 embedQuery，再取 store：避免 await 期间 close/reload
        // 把已经取到的连接关掉，导致在途检索报「数据库已关闭」。
        let queryVector: [Float]
        let binaryOptions: BinaryCoarseOptions?
        // embedding 单独给宽裕预算：正常 query 向量化远小于 1s，但主模型生成、
        // 后台摘要同抢 GPU 时可能被拖慢数秒；给足余量避免「慢」被误判成「超时」，
        // 同时保底防止 MLX 卡死导致静默挂起。
        let embedBudget = max(20, limit)
        if config.ragBinaryCoarseEnabled,
           effectiveVectorElementType != .bin,
           let signing = embedding as? EmbeddingBinarySigning,
           signing.binaryDimension > 0 {
            let signed = try await Self.withTimeout(embedBudget) {
                try await signing.embedQueryWithSignature(query)
            }
            queryVector = signed.vector
            binaryOptions = BinaryCoarseOptions(
                signature: signed.signature,
                candidateLimit: config.ragBinaryCoarseCandidates)
        } else {
            do {
                queryVector = try await Self.withTimeout(embedBudget) {
                    try await embedding.embedQuery(query)
                }
            } catch is RerankTimeout {
                throw AgentError.timeout("查询向量化超时（\(embedBudget)s），"
                    + "请稍后重试，或检查 embedding 模型是否因内存压力卡住")
            }
            binaryOptions = nil
        }
        let rerankEnabled = config.ragRerankEnabled && reranker != nil
        // 开启重排序时，最终条数由设置里的 TopN 决定（候选池按它放大）；
        // 未开启时仍按调用方 top_k。两者都封顶 20。
        let resultCount = rerankEnabled
            ? max(1, min(config.ragRerankTopN, 20))
            : requested
        // 候选池 = max(调用方 top_k, TopN) × 面板里的倍数（默认 3）；
        // 倍数越大召回越稳，但送进 ONNX 的切片也越多，需要按机器延迟权衡。
        let candidateMultiplier = max(1, min(config.ragRerankCandidateMultiplier, 5))
        let candidateCount = max(requested, resultCount)
            * (rerankEnabled ? candidateMultiplier : 1)

        let fetched: [SearchHit]? = try withSnapshotStoreBounded(timeout: limit) {
            store, deadline in
            func run(sourceFilter: String? = nil,
                     sourceContains: String? = nil) throws -> [SearchHit] {
                try store.hybridSearch(
                    vector: queryVector, query: query, topK: candidateCount,
                    sourceFilter: sourceFilter, sourceContains: sourceContains,
                    rrfK: config.ragRRFK,
                    overfetch: config.ragVectorOverfetch,
                    ftsOverfetch: config.ragFTSOverfetch,
                    binary: binaryOptions,
                    // ANN 索引就绪时向量通道走分块 IVF；不可用（未构建/
                    // 尾块超预算/维度不符）时内部自动回退精确路径。
                    ann: annQueryOptions,
                    deadline: deadline
                )
            }
            // source 是「来源子串」而不是 SQL 过滤条件：先解析成真实来源，
            // 再在该来源内部定向检索。直接全局召回后过滤，在大库上会因为
            // 候选池被其他来源占满而返回空。
            guard let needle = source?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !needle.isEmpty else {
                return try run()
            }
            // 来源名单优先取 manifest：千万级切片的库上
            // `SELECT DISTINCT source ... LIMIT` 实测 13.7 秒，会占住
            // searchLock；manifest 每次同步/去重后都会更新。
            // 只有旧库没有 manifest（或 manifest 为空）时才回退到索引扫描。
            let manifestSources = manifestStore.load()
                .filter { ($0.value.duplicateOf ?? "").isEmpty }
                .map(\.key)
            let knownSources = manifestSources.isEmpty ? store.listSources() : manifestSources
            let matches = knownSources.filter {
                $0.localizedCaseInsensitiveContains(needle)
            }
            switch matches.count {
            case 0:
                // 明确指定了来源却没有任何匹配：不能退化成全局检索，
                // 否则其他文件的内容会被误认为来自该来源。
                return []
            case 1:
                return try run(sourceFilter: matches[0])
            case 2...Self.maxSourceGroups:
                return Self.fuseSourceGroups(
                    try matches.map { try run(sourceFilter: $0) },
                    topK: candidateCount, rrfK: config.ragRRFK)
            default:
                // 子串过宽（命中太多来源）：保留全局召回 + 过滤，
                // 避免一次检索打开几十个来源。
                return try run(sourceContains: needle)
            }
        }
        guard var hits = fetched else {
            AgentLog.info("RAG: 向量库为空，未检索")
            return []
        }

        // 正常返回 TopN；reranker 失败/超时降级时收紧到 3 条。
        var resultLimit = resultCount
        if rerankEnabled, let reranker, !hits.isEmpty {
            // 先抽出纯文本：闭包要跨任务边界，避免捕获整个 SearchHit 数组。
            let documents = hits.map(\.text)
            do {
                let scores = try await Self.withTimeout(config.ragRerankTimeoutSeconds) {
                    try await reranker.rerank(query: query, documents: documents)
                }
                if scores.count == hits.count {
                    for index in hits.indices {
                        hits[index].score = scores[index]
                    }
                    // 重排序后丢弃阈值以下结果（默认 0.45，适配 jina ONNX 经
                    // sigmoid 后的分数分布；bge-mlx 方案可上调至 0.7）；全部
                    // 低于阈值时回退到原排序的候选，避免模型拿不到上下文。
                    let threshold = config.ragRerankThreshold
                    let filtered = hits.filter { $0.score >= threshold }
                        .sorted { $0.score > $1.score }
                    hits = filtered.isEmpty
                        ? Array(hits.sorted { $0.score > $1.score }.prefix(resultCount))
                        : filtered
                } else {
                    throw RerankCountMismatch(got: scores.count, expected: hits.count)
                }
            } catch {
                // 失败或超时：明确按 RRF 分排序（hybridSearch 融合时已写入 rrfScore），
                // 保证降级结果与未开启重排序时的顺序一致。
                hits.sort { ($0.rrfScore ?? $0.score) > ($1.rrfScore ?? $1.score) }
                // 降级只用 RRF 分数，缺少 reranker 的相关性过滤；
                // 只给会话前 3 条，避免 TopN 调大后把上下文塞爆。
                resultLimit = min(resultLimit, Self.rerankFallbackLimit)
                if error is RerankTimeout {
                    AgentLog.warning("RAG rerank 超时（\(config.ragRerankTimeoutSeconds)s），"
                        + "已回退 RRF 排序（最多 3 条）")
                } else if let mismatch = error as? RerankCountMismatch {
                    AgentLog.warning("RAG rerank 返回分数数量不符"
                        + "（\(mismatch.got)/\(mismatch.expected)），"
                        + "使用 RRF 排序（最多 3 条）")
                } else {
                    AgentLog.warning("RAG rerank 失败，使用 RRF 排序（最多 3 条）: "
                        + error.localizedDescription)
                }
            }
        }
        return Array(hits.prefix(resultLimit))
    }

    // MARK: 知识图谱检索

    /// 知识图谱检索 + 统一 reranker 精排 + 格式化。返回 nil 表示回退原始实现。
    private func knowledgeGraphSearchFormatted(query: String,
                                               source: String?) async throws -> String? {
        guard config.ragRetrievalStrategy.usesKnowledgeGraph else { return nil }
        guard let embedding else { return nil }
        let (kgStore, kgGraph) = try ensureKnowledgeGraph()
        // 构建中/知识库已变化（stale）的图谱不可用于检索，避免拿旧图谱回答。
        if let state = kgStore.buildState(), state != .complete {
            AgentLog.info("[kg] 图谱状态为 \(state.rawValue)，检索回退 naive")
            return nil
        }
        guard !kgGraph.isEmpty else {
            AgentLog.info("[kg] 知识图谱未构建，检索回退 naive")
            return nil
        }
        guard let auxLLM = self.auxLLM else {
            AgentLog.info("[kg] 辅助模型不可用，检索回退 naive")
            return nil
        }
        // 新进程/未同步时切片库可能尚未打开，图谱检索需要按来源反查原文。
        guard let store = try? ensureStore() else { return nil }
        let builder = KnowledgeGraphBuilder(store: kgStore, graph: kgGraph, llm: auxLLM,
                                            limits: kgExtractionLimits,
                                            format: config.ragKGExtractionFormat,
                                            maxConcurrency: config.ragKGMaxConcurrency)
        let retriever = KnowledgeGraphRetriever(kgStore: kgStore, graph: kgGraph,
                                                builder: builder, embedding: embedding,
                                                config: config, chunkStore: store)
        let candidates = try await retriever.retrieve(query: query, source: source)
        guard !candidates.isEmpty else { return nil }
        // 总预算 = system prompt + query + 图谱 + 原文。
        let reserved = TokenEstimator.estimate(query)
            + TokenEstimator.estimate(BASE_SYSTEM_PROMPT)
        return await formatKnowledgeGraph(query: query, candidates: candidates,
                                          reservedTokens: reserved)
    }

    /// 三路候选统一走同一个 reranker 精排，按实体/关系/原文分组并做 token 预算截断。
    private func formatKnowledgeGraph(query: String,
                                      candidates: KGRetrievalCandidates,
                                      reservedTokens: Int) async -> String {
        var passages = candidates.allPassages
        // 1) 统一 reranker 精排（失败/超时则保留图谱自身排序）。
        if config.ragRerankEnabled, let reranker, !passages.isEmpty {
            let texts = passages.map(\.text)
            let scores = (try? await Self.withTimeout(config.ragRerankTimeoutSeconds) {
                try await reranker.rerank(query: query, documents: texts)
            }) ?? []
            if scores.count == texts.count {
                for i in passages.indices { passages[i].score = scores[i] }
            }
        }
        // 2) 分组排序：实体/关系按余弦 + 度数/权重合成，原文按相似度稳定排序。
        let entities = KGCandidateRanking.entities(passages.filter { $0.kind == .entity })
        let relations = KGCandidateRanking.relations(passages.filter { $0.kind == .relation })
        let chunks = passages.filter { $0.kind == .chunk }
            .enumerated()
            .sorted { lhs, rhs in
                if lhs.element.score != rhs.element.score {
                    return lhs.element.score > rhs.element.score
                }
                return lhs.offset < rhs.offset
            }
            .map(\.element)

        // 3) 预算截断：实体 6000、关系 8000；总预算 30000 含 system + query。
        let entitySection = assembleSection("实体", passages: entities,
                                            budgetTokens: config.ragKGEntityTokenBudget)
        let relationSection = assembleSection("关系", passages: relations,
                                              budgetTokens: config.ragKGRelationTokenBudget)
        let entityTokens = TokenEstimator.estimate(entitySection)
        let relationTokens = TokenEstimator.estimate(relationSection)
        let chunkBudget = max(0, config.ragKGTotalTokenBudget
            - entityTokens - relationTokens - reservedTokens)
        let chunkSection = assembleSection("原文片段", passages: chunks,
                                           budgetTokens: chunkBudget)

        var lines: [String] = []
        let header = "从知识图谱检索到 \(entities.count) 个实体、\(relations.count) 条关系"
            + (chunks.isEmpty ? "" : "、\(chunks.count) 个原文片段")
            + "（\(config.ragRetrievalStrategy.rawValue) 策略"
            + (config.ragRerankEnabled ? " + 重排序" : "") + "）："
        lines.append(header)
        if !entitySection.isEmpty { lines.append(""); lines.append(entitySection) }
        if !relationSection.isEmpty { lines.append(""); lines.append(relationSection) }
        if !chunkSection.isEmpty { lines.append(""); lines.append(chunkSection) }
        if lines.count <= 1 { return "（知识库中未找到相关内容）" }
        return lines.joined(separator: "\n")
    }

    /// 组装一个分组，按 token 预算截断。
    private func assembleSection(_ title: String, passages: [KGPassage],
                                 budgetTokens: Int) -> String {
        guard !passages.isEmpty else { return "" }
        var lines = ["【\(title)】"]
        var used = TokenEstimator.estimate(title)
        for p in passages {
            let item = p.text
            let tokens = TokenEstimator.estimate(item)
            if used + tokens > budgetTokens { break }
            lines.append(item)
            lines.append("")
            used += tokens + 1
        }
        guard lines.count > 1 else { return "" }
        return lines.joined(separator: "\n")
    }

    public func searchFormatted(query: String, topK: Int = 3, source: String? = nil,
                                maxCharsPerResult: Int? = nil,
                                maxTotalChars: Int? = nil) async throws -> String {
        // 图谱策略：非 naive 且图谱已就绪时走知识图谱检索，否则回退原始实现。
        if config.ragRetrievalStrategy.usesKnowledgeGraph {
            if let kgOutput = try await knowledgeGraphSearchFormatted(query: query, source: source) {
                return kgOutput
            }
        }
        let hits = try await search(query: query, topK: topK, source: source)
        guard !hits.isEmpty else { return "（知识库中未找到相关内容）" }
        let perResultLimit = max(200, maxCharsPerResult ?? Self.resultCharLimit(
            configured: config.ragSearchMaxCharsPerResult,
            chunkTokens: effectiveChunkSizeTokens))
        // 总上限必须与内核的单次工具结果上限对齐：超出部分会被内核静默截掉，
        // RAG 自己的截断提示反而失真。
        let totalLimit = Self.outputCharLimit(config: config,
                                              maxTotalChars: maxTotalChars)
        var lines = ["从知识库检索到 \(hits.count) 条相关内容（混合检索 + RRF\(config.ragRerankEnabled ? " + 重排序" : "")）：", ""]
        var total = lines.joined().count
        for (index, hit) in hits.enumerated() {
            let name = (hit.source as NSString).lastPathComponent
            let heading = hit.heading.isEmpty ? "" : "\n章节: \(hit.heading)"
            // RRF 分数（≈1/61）与 rerank/余弦不可比，展示时标明口径。
            let isRRF = hit.rrfScore != nil && abs((hit.rrfScore ?? 0) - hit.score) < 1e-9
            let scoreLabel = isRRF ? "RRF" : "相关度"
            let header = "### [\(index + 1)] 来源: \(name) (\(scoreLabel): "
                + String(format: "%.4f", hit.score) + ")\(heading)"
            var text = hit.text
            if text.count > perResultLimit {
                text = String(text.prefix(perResultLimit)) + "…"
            }
            lines.append(header)
            lines.append(text)
            lines.append("")
            total += header.count + text.count + 1
            if total >= totalLimit {
                lines.append("…（结果过长，已截断）")
                break
            }
        }
        let output = lines.joined(separator: "\n")
        return await condenseIfNeeded(output, hits: hits)
    }

    // MARK: 辅助模型精简

    /// 可选的辅助模型精简器（由 Agent 注入，动态读取当前 auxLLM）。
    /// 返回 nil 表示不精简（未配置辅助模型、精简失败或结果为空）。
    public var auxCondenser: ((String) async -> String?)?

    /// 精简触发条件：
    /// - 总字数落在 [min, max] 区间；
    /// - 且（多文档 或 接近上限的超长结果）。
    ///
    /// 单文档的中等长度结果直接原样返回：调用辅助模型只会增加延迟，
    /// 摘要反而丢掉主模型需要的原文细节。
    static func shouldCondense(outputChars: Int, documentCount: Int,
                               config: AgentConfig) -> Bool {
        guard outputChars >= config.ragAuxSummarizeMinChars,
              outputChars <= config.ragAuxSummarizeMaxChars else { return false }
        return documentCount >= 2
            || outputChars >= config.ragAuxSummarizeMinChars * 3 / 2
    }

    /// 精简失败或输出为空时回退原文，绝不让检索结果变空。
    private func condenseIfNeeded(_ output: String, hits: [SearchHit]) async -> String {
        let documentCount = Set(hits.map(\.source)).count
        guard Self.shouldCondense(outputChars: output.count,
                                  documentCount: documentCount,
                                  config: config),
              let condenser = auxCondenser else { return output }
        if let condensed = await condenser(output), !condensed.isEmpty {
            return condensed
        }
        return output
    }

    // MARK: Meta

    public func getDocumentOutline(docName: String) -> DocumentRecord? {
        try? withStore { $0.getDocumentDigest(docName: docName) }
    }

    /// SQL 限量列表：返回前 `limit` 个文档名与总条数。
    public func listDocuments(limit: Int) -> (names: [String], total: Int) {
        let result = withStoreSafe { store -> ([String], Int) in
            (try store.documentNames(limit: limit), try store.documentCount())
        }
        guard let result else { return ([], 0) }
        return (names: result.0, total: result.1)
    }

    /// 文档查找：精确完整路径/名称 > basename 精确 > 唯一模糊；多候选返回歧义。
    public func lookupDocument(_ query: String) -> DocumentLookupResult {
        let target = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !target.isEmpty else { return .notFound }
        let result = withStoreSafe { store -> DocumentLookupResult in
            let candidates = try store.lookupDocuments(query: target, limit: 50)
            let lower = target.lowercased()
            let standardized = (target as NSString).standardizingPath
            let exact = candidates.filter { record in
                record.docName.lowercased() == lower
                    || record.mdPath.lowercased() == lower
                    || (!standardized.isEmpty
                        && (record.mdPath as NSString).standardizingPath == standardized)
            }
            if exact.count == 1 { return .found(exact[0]) }
            if exact.count > 1 { return .ambiguous(exact) }

            let basename = (target as NSString).lastPathComponent.lowercased()
            let sameBasename = candidates.filter { record in
                (record.docName as NSString).lastPathComponent.lowercased() == basename
                    || (record.mdPath as NSString).lastPathComponent.lowercased() == basename
            }
            if sameBasename.count == 1 { return .found(sameBasename[0]) }
            if sameBasename.count > 1 { return .ambiguous(sameBasename) }

            if candidates.count == 1 { return .found(candidates[0]) }
            if candidates.count > 1 { return .ambiguous(candidates) }
            return .notFound
        }
        return result ?? .notFound
    }

    /// 增强目录：修复后的 TOC + 已完成的章节/全书摘要。
    public func getEnrichedOutline(docName: String) -> EnrichedDocumentOutline? {
        guard case .found(let record) = lookupDocument(docName) else { return nil }
        return getEnrichedOutline(for: record)
    }

    /// 按已解析的记录读取增强目录：用 `doc_id` 直接命中，避免同名文档二次查找
    /// 产生歧义，也不会在两次查询之间被路径变化影响。
    public func getEnrichedOutline(for record: DocumentRecord) -> EnrichedDocumentOutline? {
        guard !record.docID.isEmpty else { return nil }
        return withStoreSafe { store in
            // 以库内当前行为准；记录可能来自较早的查询。
            let current = try store.document(docID: record.docID) ?? record
            let chapters = try store.chapterSummaries(docID: current.docID)
            let book = try store.bookSummary(docID: current.docID)
            let progress = try store.summaryProgress(docID: current.docID)
            let outline = book?.outline.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return EnrichedDocumentOutline(
                record: current,
                chapterSummaries: chapters,
                bookSummary: book,
                jobProgress: progress,
                repairedTOC: outline.isEmpty ? current.digest : outline)
        }
    }

    public func listDocuments() -> [String] {
        (withStoreSafe { $0.listDocuments() }) ?? []
    }

    // MARK: - MCP 服务薄层 API（RAGMCPServer 使用，不影响既有入库/检索流程）

    /// 已入库文档清单（manifest 视角）：含切片数、文件大小、重复副本标记。
    /// 与 `listDocuments(limit:)`（documents 表视角）互补，供外部 MCP 客户端浏览。
    public func listIngestedDocuments(limit: Int = 500) -> [RAGDocumentSummary] {
        let manifest = manifestStore.load()
        let entries = manifest
            .sorted { $0.key < $1.key }
            .prefix(max(1, limit))
        return entries.map { source, entry in
            RAGDocumentSummary(
                docName: entry.docName,
                source: source,
                mdPath: entry.mdPath,
                chunkCount: entry.chunkCount,
                sizeBytes: entry.size,
                modifiedAt: entry.mtime,
                duplicateOf: entry.duplicateOf,
                discarded: entry.discarded)
        }
    }

    /// 知识遗忘/删除：清掉文档对应切片向量、documents 表记录与 markdown 缓存。
    /// 多候选同名时拒绝猜测性删除（返回 `ambiguous`），绝不静默删错文件。
    /// 存储层错误向上抛出，由调用方（MCP/工具层）转成可读提示。
    public func deleteDocument(_ name: String) throws -> RAGDeleteOutcome {
        let target = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !target.isEmpty else { return .notFound }
        switch lookupDocument(target) {
        case .notFound:
            return .notFound
        case .ambiguous(let records):
            return .ambiguous(records.map(\.docName))
        case .found(let record):
            // manifest key = 完整来源路径；删除逻辑与 ingest 清理共用同一条
            // `removeIndexedSource` 路径，保证向量/记录/缓存清理行为一致。
            let manifest = manifestStore.load()
            let entry = manifest.first { $0.value.docName == record.docName }
                ?? manifest.first { $0.value.mdPath == record.mdPath && !record.mdPath.isEmpty }
            let source = entry?.key
                ?? record.mdPath
                ?? ((knowledgeBase as NSString)
                    .appendingPathComponent((record.docName as NSString).lastPathComponent))
            // 先清掉切片向量（拿行数做反馈），再走统一的清理路径删除
            // documents 表记录与 markdown 缓存；removeIndexedSource 内部会再
            // deleteBySource 一次（幂等，返回 0）。
            let chunksRemoved = try withStore { store -> Int in
                let chunks = try store.deleteBySource(source)
                Self.removeIndexedSource(
                    source: source, entry: entry?.value, store: store,
                    markdownDir: markdownDir, knowledgeBase: knowledgeBase)
                return chunks
            }
            invalidateStatusCache()
            invalidateKnowledgeGraph(reason: "文档已删除")
            return .deleted(chunksRemoved: chunksRemoved, docName: record.docName)
        }
    }

    /// 读取文档解析后的 markdown 缓存原文（`get_document` 透传用）。
    /// 未命中或缓存文件缺失返回 nil。
    public func readMarkdown(for docName: String) -> String? {
        guard case .found(let record) = lookupDocument(docName) else { return nil }
        let path = record.mdPath
        guard !path.isEmpty, FileManager.default.fileExists(atPath: path) else { return nil }
        return try? String(contentsOfFile: path, encoding: .utf8)
    }

    /// 创建闲时章节摘要协调器：与检索复用同一个 `VectorStore` 实例。
    ///
    /// 调用方必须在 `close()` / `reload()` 之前结束协调器会话，否则连接会在
    /// 摘要任务使用期间被关闭。打开大库可能耗时（遗留 WAL 恢复），调用方应
    /// 在后台线程创建，避免阻塞启动/UI。
    public func makeBookSummaryCoordinator(
        residentAuxProvider: @escaping @Sendable () -> LLMService?,
        transientAuxFactory: (@Sendable (AgentConfig) -> LLMService)? = nil,
        log: @escaping @Sendable (String) -> Void = { _ in }
    ) throws -> BookSummaryCoordinator {
        let store = try ensureStore()
        let config = self.config
        return BookSummaryCoordinator(
            engine: BookSummaryEngine(store: store),
            store: store,
            configProvider: { config },
            residentAuxProvider: residentAuxProvider,
            transientAuxFactory: transientAuxFactory,
            log: log)
    }

    /// 知识库状态。默认复用最近一次快照，避免在轮次/检索热路径上做全表扫描。
    ///
    /// `sourceLimit` 只返回前 N 个来源名，避免 7 万本书时把全部路径读进内存。
    /// 大库上 `COUNT(*)` / `COUNT(DISTINCT source)` / `DISTINCT ... LIMIT`
    /// 都是秒级到分钟级（56GB、2100 万切片实测约 6 分钟），因此：
    /// - 切片数直接用 `VectorStore` 维护的计数器（O(1)）；
    /// - 来源数用 manifest 条目数（同步/去重后都会更新），不再 COUNT DISTINCT；
    /// - 结果缓存 `maxAge` 秒，写操作会主动失效；
    /// - `timeout` 秒内没拿到结果就返回带 `error` 的提示快照（默认取配置
    ///   `rag_query_timeout_seconds`，0 = 不设超时），调用方照常展示，
    ///   不会因为这个查询把启动/对话卡住。
    public func status(sourceLimit: Int = 256,
                       maxAge: TimeInterval = 600,
                       timeout: Double? = nil) -> [String: JSONValue] {
        // provider/存储层明确不支持的元素类型：打开库会直接失败，优先给出配置
        // 原因，否则调用方只能看到笼统的「向量库不可用」。当前四种类型均可用。
        if let unsupported = effectiveVectorElementType.unsupportedReason {
            return unsupportedVectorTypeSnapshot(unsupported)
        }
        if let cached = cachedStatus(maxAge: maxAge) {
            return mergingANNBuildState(cached)
        }
        // `maxAge <= 0` 表示调用方要绕过缓存（诊断/刚同步完），此时也不复用
        // 「上次超时」的跳过快照，直接重试。
        if maxAge > 0, let skipped = cachedSkipSnapshot() {
            return mergingANNBuildState(skipped)
        }
        let limit = resolvedTimeout(timeout)
        statusLock.lock()
        let generation = statusGeneration
        statusLock.unlock()
        let manifest = manifestStore.load()
        let indexedEntries = manifest.filter { ($0.value.duplicateOf ?? "").isEmpty }
        let indexedSources = indexedEntries.count
        // 来源名单直接来自 manifest（同步/去重后都会更新）：千万级切片的库上
        // `SELECT DISTINCT source ... LIMIT` 实测 13.7 秒，会长期占住 searchLock，
        // 把并发的 rag_search 卡住。只有旧库没有 manifest 时才回退到索引扫描。
        let sortedSources = indexedEntries.keys.sorted()
        let manifestSources = sourceLimit > 0
            ? Array(sortedSources.prefix(sourceLimit))
            : sortedSources
        let snapshot = try? withStoreBounded(timeout: limit) { store -> [String: JSONValue] in
            let sources = manifestSources.isEmpty
                ? store.listSources(limit: sourceLimit)
                : manifestSources
            let ann = store.annIndexStatus()
            let binary = store.binaryIndexStatus()
            let binaryANN = store.binaryANNIndexStatus()
            let binaryState: String
            let binaryNote: String
            if !config.ragBinaryCoarseEnabled {
                binaryState = "off"
                binaryNote = "已关闭"
            } else if store.elementType == .bin {
                binaryState = "bypassed_bin"
                binaryNote = "bin 主存储仅支持单阶段检索"
            } else if binarySignatureDimension == 0 || !binary.storageEnabled {
                binaryState = "unsupported"
                binaryNote = binary.reason ?? "embedding 不支持二值签名"
            } else if !binary.isComplete {
                binaryState = "incomplete"
                let percent = Int((binary.coverage * 100).rounded())
                binaryNote = "覆盖 \(percent)%，请重新同步"
            } else {
                binaryState = "ready"
                binaryNote = ""
            }
            // 二值 ANN 只覆盖「签名完整」这一条路径；其余状态交给回退链。
            let binaryANNState: String
            let binaryANNNote: String
            if !config.ragBinaryCoarseEnabled {
                binaryANNState = "off"
                binaryANNNote = "已关闭"
            } else if store.elementType == .bin {
                binaryANNState = "bypassed_bin"
                binaryANNNote = "bin 主存储仅支持单阶段检索"
            } else if binaryState == "unsupported" || binaryState == "incomplete" {
                binaryANNState = "incomplete"
                binaryANNNote = binaryNote
            } else if binaryANN.blocks == 0 {
                binaryANNState = "not_built"
                binaryANNNote = "本次回退 bin 表暴力汉明"
            } else if !binaryANN.isUsable {
                binaryANNState = "unusable"
                binaryANNNote = binaryANN.summary
            } else if binaryANN.needsRebuild {
                binaryANNState = "needs_rebuild"
                binaryANNNote = binaryANN.summary
            } else if binaryANN.tailRows > 0 {
                binaryANNState = "tail"
                binaryANNNote = binaryANN.summary
            } else {
                binaryANNState = "ready"
                binaryANNNote = ""
            }
            let binaryBackend: String
            if binaryANNState == "ready" || binaryANNState == "tail" {
                binaryBackend = "bann"
            } else if binary.storageEnabled {
                binaryBackend = "sqlite-vec"
            } else {
                binaryBackend = "none"
            }
            return [
                "knowledge_base": .string(knowledgeBase),
                "rag_dir": .string(ragDir),
                // 旧库的 MAX(rowid) 基线在“删光但自增序列残留”时会虚高；
                // 先用 O(1) 的 LIMIT 1 判断是否真的有切片。
                "chunks_stored": .int(store.hasChunks ? store.count : 0),
                "source_count": .int(indexedSources > 0 ? indexedSources : sources.count),
                "sources": .array(sources.map { .string($0) }),
                "vector_backend": .string(store.vectorEnabled ? "sqlite-vec" : "sqlite-exact"),
                // ANN 索引：检索是否走索引、能否用、是否该重建。
                "ann_index": .string(ann.summary),
                "ann_usable": .bool(ann.isUsable),
                "ann_needs_rebuild": .bool(ann.needsRebuild || ann.unindexedRows > 0),
                // 陈旧行与实际有效覆盖率（删除/更新留下的旧 rowid）。
                "ann_stale_rows": .int(ann.staleRows),
                "ann_effective_coverage": .double(ann.effectiveCoverage),
                // 二值粗筛：ready 时检索走「汉明粗筛 → 余弦精排」，
                // ANN 查询被整体旁路（ann_bypassed）。
                "binary_coarse": .string(binaryState),
                "binary_dimension": .int(binary.dimension),
                "binary_coverage": .double(binary.coverage),
                "binary_rows": .int(binary.rows),
                "binary_backend": .string(binaryBackend),
                "binary_note": .string(binaryNote),
                "ann_bypassed": .bool(binaryState == "ready"),
                // 二值 ANN（自研分块 IVF）：未构建时回退 bin 表暴力汉明。
                "binary_ann": .string(binaryANNState),
                "binary_ann_index": .string(binaryANN.summary),
                "binary_ann_note": .string(binaryANNNote),
                // stage 2 主向量读取路径：rerank 镜像 / 部分镜像 / vec0 点查。
                "rerank_rows": .int(binary.rerankRows),
                "stage2_backend": .string(binary.stage2Backend),
                // 向量落盘类型 / Matryoshka 截断维 / 待重建闸门。
                "vector_element_type": .string(store.elementType.rawValue),
                "matryoshka_dim": .int(effectiveMatryoshkaDimension),
                "vector_index_fingerprint": .string(store.indexFingerprint),
                "vector_rebuild_required": .bool(store.requiresRebuild),
                "rebuild_reason": store.rebuildReason.map { .string($0) } ?? .null,
            ]
        }
        guard let snapshot else {
            // 超时/库不可用：返回「跳过」快照，保留上一次已知统计。
            return skipSnapshot(limit: limit)
        }
        let result = snapshot
        statusLock.lock()
        // 统计期间若有同步/去重/清空完成，本次结果已过期，不写回缓存。
        if statusGeneration == generation {
            statusSnapshot = result
            statusSnapshotTime = Date().timeIntervalSince1970
        }
        statusLock.unlock()
        return mergingANNBuildState(result)
    }

    /// 把实时 ANN 构建状态合并进状态快照。构建进度变化频繁，不进缓存，
    /// 每次读取时叠加，保证 `rag_status` 能看到进度。
    private func mergingANNBuildState(_ snapshot: [String: JSONValue]) -> [String: JSONValue] {
        annBuildLock.lock()
        let running = annBuildScheduled || annBuildActive
        let progress = annBuildProgress
        let target = annBuildTarget
        annBuildLock.unlock()
        var output = snapshot
        output["ann_building"] = .bool(running)
        output["ann_build_progress"] = .double(progress?.fraction ?? 0)
        output["ann_build_message"] = .string(progress?.message ?? "")
        output["ann_build_target"] = .string(running ? target.rawValue : "")
        output["binary_ann_building"] = .bool(running && target == .binary)
        if running && target == .binary {
            output["binary_ann"] = .string("building")
        }
        return output
    }

    /// 只读最近一次 `status()` 快照，不打开数据库、不扫描任何表。
    /// 首轮 `dynamicContextPrompt` 用它，避免把数分钟的 COUNT 放在模型调用之前。
    public func cachedStatus(maxAge: TimeInterval = 600) -> [String: JSONValue]? {
        statusLock.lock(); defer { statusLock.unlock() }
        guard let statusSnapshot else { return nil }
        if maxAge > 0,
           Date().timeIntervalSince1970 - statusSnapshotTime > maxAge {
            return nil
        }
        return statusSnapshot
    }

    /// 短时间内已超时过就不再重试：状态栏每 30 秒刷新一次，
    /// 每次都白等满超时是没意义的。
    private func cachedSkipSnapshot() -> [String: JSONValue]? {
        statusLock.lock(); defer { statusLock.unlock() }
        guard let statusSkipSnapshot,
              Date().timeIntervalSince1970 < statusSkipUntil else { return nil }
        return statusSkipSnapshot
    }

    /// sqlite-vec 不支持当前元素类型时的状态快照：保留上次统计，
    /// 但明确标出配置错误与待重建原因，供设置面板直接展示。
    private func unsupportedVectorTypeSnapshot(_ reason: String) -> [String: JSONValue] {
        statusLock.lock(); defer { statusLock.unlock() }
        var snapshot = statusSnapshot ?? [
            "knowledge_base": .string(knowledgeBase),
            "rag_dir": .string(ragDir),
            "chunks_stored": .int(0),
            "source_count": .int(0),
            "sources": .array([]),
        ]
        snapshot["vector_element_type"] = .string(effectiveVectorElementType.rawValue)
        snapshot["matryoshka_dim"] = .int(effectiveMatryoshkaDimension)
        snapshot["vector_rebuild_required"] = .bool(true)
        snapshot["rebuild_reason"] = .string(reason)
        snapshot["error"] = .string(reason)
        return snapshot
    }

    /// 超时/不可用时给出的提示快照：保留上一次已知统计，附加可读原因。
    private func skipSnapshot(limit: Double) -> [String: JSONValue] {
        let reason = limit > 0
            ? "查询超时（\(Self.timeoutLabel(limit))），已跳过"
            : "向量库暂时不可用，已跳过"
        logTimeout(AgentError.timeout(reason))
        statusLock.lock()
        var snapshot = statusSnapshot ?? [
            "knowledge_base": .string(knowledgeBase),
            "rag_dir": .string(ragDir),
            "chunks_stored": .int(0),
            "source_count": .int(0),
            "sources": .array([]),
            "vector_element_type": .string(effectiveVectorElementType.rawValue),
            "matryoshka_dim": .int(effectiveMatryoshkaDimension),
            "vector_rebuild_required": .bool(false),
        ]
        snapshot["error"] = .string(reason)
        statusSkipSnapshot = snapshot
        statusSkipUntil = Date().timeIntervalSince1970 + Self.statusSkipCacheSeconds
        statusLock.unlock()
        return snapshot
    }

    private static func timeoutLabel(_ seconds: Double) -> String {
        seconds >= 1
            ? "\(Int(seconds.rounded()))s 未响应"
            : "\(Int((seconds * 1000).rounded()))ms 未响应"
    }

    /// 数据变化（同步/删除/去重/清空/重载）后丢弃快照，下次展示重新统计。
    public func invalidateStatusCache() {
        statusLock.lock()
        statusSnapshot = nil
        statusSnapshotTime = 0
        statusSkipSnapshot = nil
        statusSkipUntil = 0
        statusGeneration &+= 1
        statusLock.unlock()
    }

    // MARK: 显式重建

    /// 显式重建向量库的结果。
    public struct VectorRebuildResult: Sendable {
        /// 被清空的旧切片数。
        public var removedChunks: Int = 0
        /// 重建后的全量同步统计；未要求重同步时为 nil。
        public var ingest: IngestStats?

        public init(removedChunks: Int = 0, ingest: IngestStats? = nil) {
            self.removedChunks = removedChunks
            self.ingest = ingest
        }

        public var summary: String {
            var text = "已清空 \(removedChunks) 个旧切片并按当前配置重建向量表"
            if let ingest { text += "；\(ingest.summary)" }
            return text
        }
    }

    /// 按当前配置显式重建向量库：重建向量表（新维度/元素类型/指纹），
    /// 清空旧切片、文档缓存、ANN 索引、manifest 与 markdown 缓存；
    /// 默认随后立即执行一次全量同步。
    ///
    /// 返回 nil 表示被拒绝（另一个同步/维护正在进行）或重建失败，原因已写入日志。
    @discardableResult
    public func rebuildVectorIndex(resync: Bool = true,
                                   progress: ((Int, Int, String) -> Void)? = nil,
                                   log: ((String) -> Void)? = nil) async -> VectorRebuildResult? {
        let logFn: (String) -> Void = { message in
            if let log { log(message) } else { AgentLog.info(message) }
        }
        guard tryBeginIngest() else {
            logFn("RAG: 另一次同步/维护正在进行，已忽略本次向量库重建")
            return nil
        }
        var removed = 0
        var failure: Error?
        do {
            try withStore { store in
                removed = try store.rebuildForCurrentConfiguration()
            }
        } catch {
            failure = error
        }
        invalidateStatusCache()
        releaseIngestFlag()
        closeIfPending()
        if let failure {
            logFn("RAG: 重建向量库失败：\(failure.localizedDescription)")
            AgentLog.warning("RAG: 重建向量库失败: \(failure.localizedDescription)")
            return nil
        }
        // 旧向量已全部失效：manifest 与 markdown 缓存一并清掉，强制全量重算。
        // 不删知识库原始文件；之后仍可中断续跑。
        manifestStore.clear()
        try? FileManager.default.removeItem(atPath: markdownDir)
        try? FileManager.default.createDirectory(atPath: markdownDir,
                                                 withIntermediateDirectories: true)
        logFn("RAG: 向量库已按当前配置重建（清空 \(removed) 个切片）"
            + (resync ? "，开始全量重新同步…" : "，请手动执行同步"))

        // 用户可能在清空旧表期间点了「停止同步」：Task 取消或引擎取消标志
        // 任一成立都不再启动全量同步，避免停止请求被 ingest 的初始化重置。
        let shouldResync = resync && !Task.isCancelled && !isCancelled
        let ingestStats = shouldResync
            ? await ingest(force: true, progress: progress, log: log)
            : nil
        return VectorRebuildResult(removedChunks: removed, ingest: ingestStats)
    }

    public func clear() throws {
        // 与同步串行：否则同步结束时会把 manifest 重新写回来。
        guard tryBeginIngest() else {
            throw AgentError.storage("同步进行中，无法清空知识库")
        }
        defer { releaseIngestFlag() }
        try withStore { store in
            try store.clear()
        }
        try? FileManager.default.removeItem(atPath: markdownDir)
        try? FileManager.default.createDirectory(atPath: markdownDir, withIntermediateDirectories: true)
        manifestStore.clear()
        invalidateStatusCache()
    }

    // MARK: Cache paths

    private static func cacheMarkdownPath(for source: String,
                                          markdownDir: String,
                                          knowledgeBase: String) -> String {
        var relative = source
        if source.hasPrefix(knowledgeBase) {
            relative = String(source.dropFirst(knowledgeBase.count))
                .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        }
        return (markdownDir as NSString).appendingPathComponent(Self.safeFilename(relative) + ".md")
    }

    static func safeFilename(_ path: String) -> String {
        let h = Insecure.MD5.hash(data: Data(path.utf8)).prefix(6)
            .map { String(format: "%02x", $0) }.joined()
        let base = (path as NSString).lastPathComponent
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        let prefix = String(base.unicodeScalars.filter { allowed.contains($0) }.prefix(32))
        let safe = prefix.isEmpty ? "document" : prefix
        return "\(safe)_\(h)"
    }

    // MARK: Fingerprints

    // MARK: 重复文件 / 副本清理

    /// 用「文件大小 + 采样指纹」把同一本书的不同目录副本分组。
    /// 每组只保留一个代表文件参与解析；优先已有 manifest 的完整条目。
    static func groupDuplicateFiles(
        _ files: [ScannedFile],
        manifest: [String: RAGManifestEntry],
        force: Bool,
        verifyFingerprint: Bool
    ) -> (unique: [ScannedFile], groups: [DuplicateFileGroup]) {
        struct FingerprintKey: Hashable {
            let size: Int64
            let fingerprint: String
        }

        var grouped: [FingerprintKey: [ScannedFile]] = [:]
        var unique: [ScannedFile] = []
        // 先按文件大小分组：只有大小相同的文件才可能是副本，绝大多数文件
        // 不必读取头部/尾部采样，增量同步时能省掉大量磁盘 IO。
        var sameSizeGroups: [Int64: [ScannedFile]] = [:]

        for file in files {
            guard let signature = fileSignature(file.path) else {
                unique.append(file)
                continue
            }
            sameSizeGroups[signature.size, default: []].append(file)
        }

        for (size, sameSize) in sameSizeGroups {
            if sameSize.count == 1 {
                unique.append(sameSize[0])
                continue
            }
            for file in sameSize {
                // 默认复用 manifest 指纹（mtime + size 已匹配，说明文件未变化）；
                // force / verify_fingerprint 或文件变化时才重新采样，避免每次
                // 增量同步都对同尺寸文件做头部/尾部读取。
                let fingerprint: String
                if !force, !verifyFingerprint,
                   let entry = manifest[file.path],
                   !entry.fingerprint.isEmpty,
                   abs(entry.mtime - (fileSignature(file.path)?.mtime ?? -1)) < 0.001,
                   entry.size == size {
                    fingerprint = entry.fingerprint
                } else {
                    fingerprint = sourceFingerprint(file.path, size: size)
                }
                guard !fingerprint.isEmpty else {
                    unique.append(file)
                    continue
                }
                grouped[FingerprintKey(size: size, fingerprint: fingerprint),
                        default: []].append(file)
            }
        }

        var groups: [DuplicateFileGroup] = []
        for group in grouped.values {
            guard group.count > 1 else {
                if let only = group.first { unique.append(only) }
                continue
            }
            // 采样指纹对 >4MB 文件只读头/中/尾各 64KB：两个不同文件完全可能
            // 在这三段里一致。直接据此判重复会把整本书的向量清掉、内容永久丢失。
            // 只对候选组做全量哈希二次校验（比解析整本 PDF 便宜得多）。
            var verified: [String: [ScannedFile]] = [:]
            var unreadable: [ScannedFile] = []
            for file in group {
                let digest = fullContentHash(file.path)
                if digest.isEmpty {
                    unreadable.append(file)
                } else {
                    verified[digest, default: []].append(file)
                }
            }
            for bucket in verified.values {
                guard bucket.count > 1,
                      let representative = bucket.min(by: {
                          duplicateRank($0, manifest) < duplicateRank($1, manifest)
                      }) else {
                    unique.append(contentsOf: bucket)
                    continue
                }
                unique.append(representative)
                groups.append(DuplicateFileGroup(
                    representative: representative,
                    duplicates: bucket.filter { $0.path != representative.path }))
            }
            // 无法读取的一律当作独立文件：不能因「无法证明相同」而删内容。
            unique.append(contentsOf: unreadable)
        }

        unique.sort {
            $0.relativePath.localizedStandardCompare($1.relativePath) == .orderedAscending
        }
        groups.sort {
            $0.representative.relativePath.localizedStandardCompare(
                $1.representative.relativePath) == .orderedAscending
        }
        return (unique, groups)
    }

    /// 代表文件优先：完整完成 > 已有 manifest > 路径字典序。
    private static func duplicateRank(
        _ file: ScannedFile,
        _ manifest: [String: RAGManifestEntry]
    ) -> (Int, Int, String) {
        let entry = manifest[file.path]
        return (entry?.complete == true ? 0 : 1,
                entry == nil ? 1 : 0,
                file.relativePath)
    }

    /// 同步完成后清理重复来源：
    /// - 原始指纹相同且代表文件完整成功时，清掉其它副本；
    /// - 正文 content hash 相同的来源只保留一个，删除其余向量/摘要缓存。
    private static func deduplicateByContentHash(
        manifestStore: ManifestStore,
        duplicateGroups: [DuplicateFileGroup],
        existingSources: Set<String>,
        signatureHash: String,
        store: VectorStore,
        markdownDir: String,
        knowledgeBase: String,
        log: @escaping (String) -> Void
    ) -> Int {
        let finalManifest = manifestStore.load()
        var removeSet = Set<String>()
        var tombstoneCanonical: [String: String] = [:]

        // 原始指纹相同的副本：代表文件完整成功时，清掉其它副本已入库的旧向量。
        for group in duplicateGroups {
            guard let representative = finalManifest[group.representative.path],
                  representative.complete == true,
                  (representative.chunkCount ?? 0) > 0 else { continue }
            for file in group.duplicates
            where finalManifest[file.path] != nil || store.chunkCount(source: file.path) > 0 {
                removeSet.insert(file.path)
            }
        }

        // 正文 content hash 相同的来源只保留一个；其余来源写 tombstone，
        // 下次增量同步直接跳过，不再重复解析。
        var byDigest: [String: [(source: String, entry: RAGManifestEntry)]] = [:]
        for (source, entry) in finalManifest
        where !entry.digest.isEmpty && entry.duplicateOf == nil && !removeSet.contains(source) {
            byDigest[entry.digest, default: []].append((source, entry))
        }
        for group in byDigest.values where group.count > 1 {
            let ordered = group.sorted { lhs, rhs in
                let leftComplete = lhs.entry.complete == true
                let rightComplete = rhs.entry.complete == true
                if leftComplete != rightComplete { return leftComplete }
                let leftExisting = existingSources.contains(lhs.source)
                let rightExisting = existingSources.contains(rhs.source)
                if leftExisting != rightExisting { return leftExisting }
                return lhs.source.localizedStandardCompare(rhs.source) == .orderedAscending
            }
            let canonical = ordered[0].source
            for item in ordered.dropFirst() {
                tombstoneCanonical[item.source] = canonical
            }
        }

        guard !removeSet.isEmpty || !tombstoneCanonical.isEmpty else { return 0 }

        for source in removeSet.sorted() {
            let entry = finalManifest[source]
            removeIndexedSource(source: source, entry: entry, store: store,
                                markdownDir: markdownDir, knowledgeBase: knowledgeBase)
            let name = entry?.docName
                ?? relativeSourcePath(source, knowledgeBase: knowledgeBase)
            log("RAG: 去重移除副本 \(name)")
        }

        var tombstones: [String: RAGManifestEntry] = [:]
        for (source, canonical) in tombstoneCanonical {
            guard let entry = finalManifest[source] else { continue }
            removeIndexedSource(source: source, entry: entry, store: store,
                                markdownDir: markdownDir, knowledgeBase: knowledgeBase)
            var marker = entry
            marker.duplicateOf = canonical
            marker.chunkCount = 0
            marker.complete = false
            marker.ingestSignature = signatureHash
            marker.mdPath = ""
            tombstones[source] = marker
            let keepName = finalManifest[canonical]?.docName ?? canonical
            log("RAG: 去重移除副本 \(entry.docName)，保留 \(keepName)")
        }
        manifestStore.merge(updated: tombstones, removed: removeSet)
        return removeSet.count + tombstones.count
    }

    private static func removeIndexedSource(
        source: String,
        entry: RAGManifestEntry?,
        store: VectorStore,
        markdownDir: String,
        knowledgeBase: String
    ) {
        _ = try? store.deleteBySource(source)
        let mdPath = entry?.mdPath
            ?? cacheMarkdownPath(for: source, markdownDir: markdownDir,
                                knowledgeBase: knowledgeBase)
        let docName = entry?.docName
            ?? relativeSourcePath(source, knowledgeBase: knowledgeBase)
        try? store.deleteDocumentByMdPath(mdPath)
        try? store.deleteDocument(docName: docName)
        if !mdPath.isEmpty {
            try? FileManager.default.removeItem(atPath: mdPath)
        }
    }

    private static func relativeSourcePath(_ source: String,
                                           knowledgeBase: String) -> String {
        source.hasPrefix(knowledgeBase)
            ? String(source.dropFirst(knowledgeBase.count))
                .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            : (source as NSString).lastPathComponent
    }

    struct FileSignature {
        var mtime: Double
        var size: Int64
    }

    static func fileSignature(_ path: String) -> FileSignature? {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let size = attrs[.size] as? NSNumber else { return nil }
        let mtime = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        return FileSignature(mtime: mtime, size: size.int64Value)
    }

    static func contentHash(_ text: String) -> String {
        let digest = SHA256.hash(data: Data(text.utf8))
        return digest.prefix(8).map { String(format: "%02x", $0) }.joined()
    }

    static func fileSizeText(_ path: String) -> String {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let size = attrs[.size] as? NSNumber else { return "大小未知" }
        return ByteCountFormatter.string(fromByteCount: size.int64Value, countStyle: .file)
    }

    static func elapsedText(_ start: Date) -> String {
        String(format: "%.1fs", Date().timeIntervalSince(start))
    }

    static func stageText(_ stage: String) -> String {
        switch stage {
        case "text": return "文本层"
        case "render": return "页面渲染"
        case "ocr": return "OCR"
        default: return stage
        }
    }

    /// 解析器写入的分阶段耗时：文本层 / 渲染 / Vision OCR / Ovis OCR。
    static func parseTimingText(_ metadata: [String: String]) -> String {
        let fields: [(key: String, name: String)] = [
            ("timing_text", "文本层"),
            ("timing_render", "渲染"),
            ("timing_vision", "Vision"),
            ("timing_ocr", "OCR"),
        ]
        var parts: [String] = []
        for field in fields {
            guard let raw = metadata[field.key],
                  let value = Double(raw), value >= 0.05 else { continue }
            parts.append("\(field.name) \(String(format: "%.1fs", value))")
        }
        return parts.joined(separator: " / ")
    }

    /// 原始文件指纹：≤4MB 全量；大文件取头/中/尾各 64KB。
    static func sourceFingerprint(_ path: String, size: Int64? = nil) -> String {
        let fm = FileManager.default
        let total: Int64
        if let size {
            total = size
        } else if let attrs = try? fm.attributesOfItem(atPath: path),
                  let number = attrs[.size] as? NSNumber {
            total = number.int64Value
        } else {
            return ""
        }
        guard let handle = FileHandle(forReadingAtPath: path) else { return "" }
        defer { try? handle.close() }
        var hasher = SHA256()
        hasher.update(data: Data(String(total).utf8))
        let sample = 64 * 1024
        do {
            if total <= 4 * 1024 * 1024 {
                while let data = try handle.read(upToCount: 1 << 20), !data.isEmpty {
                    hasher.update(data: data)
                }
            } else {
                hasher.update(data: try handle.read(upToCount: sample) ?? Data())
                try handle.seek(toOffset: UInt64(max(0, total / 2 - Int64(sample / 2))))
                hasher.update(data: try handle.read(upToCount: sample) ?? Data())
                try handle.seek(toOffset: UInt64(max(0, total - Int64(sample))))
                hasher.update(data: try handle.read(upToCount: sample) ?? Data())
            }
        } catch {
            return ""
        }
        return hasher.finalize().prefix(8).map { String(format: "%02x", $0) }.joined()
    }

    /// 全量内容哈希：只用于重复文件候选组的二次校验（见 `groupDuplicateFiles`）。
    /// 流式读取，失败返回空串，调用方必须按「无法证明相同」保守处理。
    static func fullContentHash(_ path: String) -> String {
        guard let handle = FileHandle(forReadingAtPath: path) else { return "" }
        defer { try? handle.close() }
        var hasher = SHA256()
        do {
            while let data = try handle.read(upToCount: 1 << 20), !data.isEmpty {
                hasher.update(data: data)
            }
        } catch {
            return ""
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
