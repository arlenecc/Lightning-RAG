import Foundation
import CryptoKit

/// SQLite + sqlite-vec 向量库。
///
/// 设计要点（对应原 LanceDB 实现的替换方案）：
/// - **主表与向量表分离**：`chunks` 存正文/元数据，`chunk_vectors` 只存向量。
/// - **单事务写入**：正文 + FTS5 + 向量在同一个 `BEGIN IMMEDIATE` 事务内完成，
///   失败整体回滚。
/// - **单事务删除**：先取 rowid，再按批删除三张表，仍在一个事务内。
/// - **两段式 KNN**：先查向量表拿 rowid，再回主表按 rowid 取正文；绝不 JOIN
///   虚拟表。
/// - **过滤**：SQLite 虚拟表过滤不可靠，统一用 over-fetch 后在第二阶段过滤；
///   定向检索（source=...）时自动放大召回数。
/// - **混合检索**：FTS5 + 向量，RRF(rank Reciprocal Fusion) 融合。
public final class VectorStore: @unchecked Sendable {

    public static let chunksTable = "chunks"
    public static let vectorTable = "chunk_vectors"
    public static let ftsTable = "chunks_fts"
    public static let documentsTable = "documents"
    public static let chapterSummariesTable = "chapter_summaries"
    public static let chapterSegmentsTable = "chapter_summary_segments"
    public static let bookSummariesTable = "book_summaries"
    public static let bookSegmentsTable = "book_summary_segments"
    public static let summaryJobTable = "summary_job_state"
    public static let summaryRuntimeTable = "summary_runtime"
    /// 二值签名表。rowid 与 `chunks` 对齐，只存派生签名。
    public static let binaryVectorTable = "chunk_vectors_bin"
    /// stage 2 精排的主向量镜像（普通表，按 rowid 点查）。
    public static let rerankVectorTable = "chunk_vectors_rerank"

    public let db: SQLiteDatabase
    /// 数据库路径；检索用独立的只读连接打开 WAL 快照，不阻塞写入。
    private let dbPath: String
    /// 只读连接与其锁。惰性创建，首次检索时打开。
    private let readLock = NSLock()
    private var readDB: SQLiteDatabase?
    /// 当前向量维度。极少数模型（缺少 config.json）会在首次推理后才发现
    /// 真实维度；若库还是空的，允许自动按真实维度重建。
    public private(set) var dimension: Int
    /// 落盘元素类型（fp32 / int8 / fp16 / bin）。
    /// fp16 没有 vec0 原生列类型，使用普通表 + 精确余弦扫描。
    public private(set) var elementType: VectorElementType
    /// 索引指纹：embedding 标识 + 维度 + Matryoshka 截断维 + 元素类型。
    /// 非空时与 meta 中的值比对，不一致即要求显式重建。
    public let indexFingerprint: String
    /// 非 nil 表示当前库与配置不匹配，需要显式重建后才能写入/向量检索。
    /// 不清空旧数据：FTS 仍可检索，用户可以在重建前导出或确认。
    public private(set) var rebuildReason: String?
    public private(set) var vectorEnabled: Bool = false
    /// 维度变化导致旧索引被清空；RAGEngine 据此同步失效 manifest。
    public private(set) var didResetForDimensionChange = false
    /// sqlite-vec 扩展是否可用（与 `vectorEnabled` 区分：待重建时后者为 false，
    /// 但清空/重建仍应优先建 vec0 表）。
    let sqliteVecAvailable: Bool
    /// 写入/删除/检索共用。ANN 实现（`VectorStoreANN.swift`）也在同一把锁内，
    /// 因此改为 internal。
    let lock = NSRecursiveLock()
    let deleteBatchSize = 400
    private let insertBatchSize = 200
    /// sqlite-vec 的 KNN k 上限为 4096；分批/过滤不能越过该值。
    let maxVectorK = 4096
    /// 精确来源定向检索一次最多读取的向量行数。超过后回退全局路径，
    /// 避免单个超大来源把全部向量读进内存。
    private static let scopedExactRowLimit = 50_000
    /// 维护的切片总数，持久化在 meta 表。写操作增量更新，避免在状态/检索
    /// 热路径上执行 `COUNT(*)`：千万级切片的库上实测需要数分钟。
    /// 旧库没有该键时用 `MAX(rowid)` 作为基线（O(1)，可能略高但不阻塞）。
    private var cachedChunkCount: Int?
    private static let chunkCountKey = "chunks_count"
    private static let dimensionKey = "embedding_dim"
    private static let elementTypeKey = "vector_element_type"
    private static let fingerprintKey = "vector_index_fingerprint"
    static let binaryRowsKey = "binary_rows"
    static let binaryDimensionKey = "binary_dimension"
    static let binarySourceKey = "binary_source"
    static let binaryQuantizerVersionKey = "binary_quantizer_version"
    static let binaryQuantizerVersion = 1
    /// ANN 簇心缓存（懒加载）。扩展无法新增存储属性，因此声明在这里，
    /// 由 `VectorStoreANN.swift` 读写。
    var annCentroidCache: ANNCentroidCache?
    /// 二值签名位宽；0 表示 provider 不支持或主类型为 bin。
    let binaryDimension: Int
    /// 签名来源标识，用于 bin 表 meta 一致性校验。
    let binarySource: String
    /// bin 表是否可读写（sqlite-vec + 表存在 + 维度/来源有效）。
    var binaryStorageEnabled: Bool = false
    /// bin 表当前行数，O(1) 覆盖率计数器。
    var binaryRows: Int = 0
    /// 不可用原因（供 status 展示）。
    var binaryStorageReason: String?
    /// rerank 镜像表是否可用（bin 主存储不需要镜像）。
    var rerankStorageEnabled: Bool = false
    /// 镜像表当前行数，O(1) 覆盖率计数器。
    var rerankRows: Int = 0
    /// 镜像不可用原因（供状态展示）。
    var rerankStorageReason: String?
    /// 二值查询实际走 stage 1 的次数；供测试验证 ANN 旁路。
    var annQueryCount: Int = 0
    /// 最近一次二值回退原因与日志时间（60 秒节流）。
    var lastBinaryFallbackReason: String?
    var lastBinaryFallbackLog: Date = .distantPast

    public init(dbPath: String, dimension: Int, enableSQLiteVec: Bool = true,
                elementType: VectorElementType = .fp32,
                binaryDimension: Int = 0, binarySource: String = "",
                indexFingerprint: String = "") throws {
        let resolvedDimension = max(1, dimension)
        // 非法维度（如 bin 维度不是 8 的倍数）在打开库之前就报错，
        // 避免先建出一张错误的表再回滚。
        try elementType.validate(dimension: resolvedDimension)
        let resolvedBinaryDimension = max(0, binaryDimension)
        guard resolvedBinaryDimension == 0 || resolvedBinaryDimension % 8 == 0 else {
            throw AgentError.storage(
                "二值签名维度 \(resolvedBinaryDimension) 必须是 8 的倍数")
        }
        self.dimension = resolvedDimension
        self.elementType = elementType
        self.indexFingerprint = indexFingerprint
        self.dbPath = dbPath
        self.binaryDimension = resolvedBinaryDimension
        self.binarySource = binarySource
        try FileManager.default.createDirectory(
            atPath: (dbPath as NSString).deletingLastPathComponent,
            withIntermediateDirectories: true
        )
        self.db = try SQLiteDatabase(path: dbPath)
        self.sqliteVecAvailable = enableSQLiteVec && db.vectorExtensionLoaded
        try createBaseSchema()
        self.cachedChunkCount = Self.readStoredCount(db, key: Self.chunkCountKey)
        if rebuildReason != nil {
            // 待重建：旧表保持原样（不删数据），但向量通道暂时关闭。
            self.vectorEnabled = false
        } else if sqliteVecAvailable, elementType.supportsVec0Index {
            var enabled = true
            do {
                try ensureVectorTable(dimension: self.dimension, useVec0: true)
            } catch {
                AgentLog.warning("vec0 建表失败，退化为精确向量扫描: \(error.localizedDescription)")
                enabled = false
                _ = try? db.execute("DROP TABLE IF EXISTS \(Self.vectorTable)")
                try ensureVectorTable(dimension: self.dimension, useVec0: false)
            }
            self.vectorEnabled = enabled
        } else {
            try ensureVectorTable(dimension: self.dimension, useVec0: false)
        }
        do {
            try setupRerankStorageLocked()
        } catch {
            AgentLog.warning("rerank 镜像初始化失败，stage 2 回退 vec0: \(error.localizedDescription)")
            rerankStorageEnabled = false
            rerankStorageReason = "初始化失败：\(error.localizedDescription)"
        }
        do {
            try setupBinaryIndexLocked()
        } catch {
            AgentLog.warning("二值索引初始化失败，已停用: \(error.localizedDescription)")
            binaryStorageEnabled = false
            binaryStorageReason = "初始化失败：\(error.localizedDescription)"
        }
    }

    public func close() {
        lock.lock()
        readLock.lock()
        readDB?.close()
        readDB = nil
        readLock.unlock()
        lock.unlock()
        db.close()
    }

    /// 当前库是否需要显式重建（配置与索引指纹/维度/元素类型不符）。
    public var requiresRebuild: Bool { rebuildReason != nil }

    /// 惰性打开只读连接。读连接与写连接共享同一个 WAL，读事务看到的是
    /// 建立快照时的版本，写入/删除可以继续提交而不被阻塞。
    private func readDatabase() throws -> SQLiteDatabase {
        readLock.lock(); defer { readLock.unlock() }
        if let readDB { return readDB }
        let reader = try SQLiteDatabase(path: dbPath, readOnly: true)
        readDB = reader
        return reader
    }

    /// 在只读快照内执行查询（FTS 候选 + hydrate 必须处于同一版本）。
    /// 快照期间不持有 `lock`，写入/删除可并发完成。
    func withReadSnapshot<T>(deadline: Date? = nil,
                             _ body: (SQLiteDatabase) throws -> T) throws -> T {
        let reader = try readDatabase()
        func run() throws -> T {
            try reader.readTransaction {
                // 先执行一次读，确保 WAL 快照在进入闭包前已经建立：
                // 之后的写入在本次快照内不可见，rowid 不会消失。
                _ = try reader.query("SELECT 1 FROM \(Self.chunksTable) LIMIT 1")
                return try body(reader)
            }
        }
        guard let deadline else { return try run() }
        let seconds = deadline.timeIntervalSinceNow
        guard seconds > 0 else { throw AgentError.timeout("查询超出时限被跳过") }
        return try reader.withDeadline(seconds: seconds, run)
    }

    /// 在期限内执行只读查询：先限时获取切片锁（避免大批量写入把读路径无限期
    /// 挡住），再把剩余时间交给底层语句级期限。超时抛 `AgentError.timeout`，
    /// 调用方跳过本次查询即可。
    public func withQueryDeadline<T>(seconds: Double, _ body: () throws -> T) throws -> T {
        guard seconds > 0 else { throw AgentError.timeout("查询超出时限被跳过") }
        let deadline = Date().addingTimeInterval(seconds)
        guard lock.lock(before: deadline) else {
            throw AgentError.timeout("知识库正在写入，本次查询已跳过")
        }
        defer { lock.unlock() }
        let remaining = deadline.timeIntervalSinceNow
        guard remaining > 0 else { throw AgentError.timeout("查询超出时限被跳过") }
        return try db.withDeadline(seconds: remaining, body)
    }

    // MARK: - Schema

    private func createBaseSchema() throws {
        try db.execScript("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            id TEXT NOT NULL UNIQUE,
            source TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            heading TEXT NOT NULL DEFAULT '',
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_source_hash ON chunks(source_hash, chunk_index);
        CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            tokens,
            source_hash UNINDEXED,
            chunk_index UNINDEXED,
            tokenize = 'unicode61 remove_diacritics 2'
        );

        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            doc_name TEXT NOT NULL,
            digest TEXT NOT NULL DEFAULT '',
            chapters TEXT NOT NULL DEFAULT '',
            md_path TEXT NOT NULL DEFAULT '',
            content_hash TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_documents_name ON documents(doc_name);
        CREATE INDEX IF NOT EXISTS idx_documents_md_path ON documents(md_path);

        CREATE TABLE IF NOT EXISTS chapter_summaries (
            doc_id TEXT NOT NULL,
            section_index INTEGER NOT NULL,
            heading TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            level INTEGER NOT NULL DEFAULT 1,
            summary TEXT NOT NULL DEFAULT '',
            content_hash TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL,
            PRIMARY KEY (doc_id, section_index)
        );

        CREATE TABLE IF NOT EXISTS chapter_summary_segments (
            doc_id TEXT NOT NULL,
            section_index INTEGER NOT NULL,
            segment_index INTEGER NOT NULL,
            segment_hash TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL,
            PRIMARY KEY (doc_id, section_index, segment_index)
        );

        CREATE TABLE IF NOT EXISTS book_summaries (
            doc_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL DEFAULT '',
            outline TEXT NOT NULL DEFAULT '',
            chapter_count INTEGER NOT NULL DEFAULT 0,
            source_hash TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS book_summary_segments (
            doc_id TEXT NOT NULL,
            group_index INTEGER NOT NULL,
            group_hash TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL,
            PRIMARY KEY (doc_id, group_index)
        );

        CREATE TABLE IF NOT EXISTS summary_job_state (
            doc_id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'pending',
            next_section INTEGER NOT NULL DEFAULT 0,
            total_sections INTEGER NOT NULL DEFAULT 0,
            last_error TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS summary_runtime (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """)
        // 旧库没有 content_hash：补列后才能判断正文是否变化。
        try ensureColumn(table: Self.documentsTable, column: "content_hash",
                         definition: "TEXT NOT NULL DEFAULT ''")

        let stored = try db.scalar(
            "SELECT value FROM meta WHERE key = ?", [.text(Self.dimensionKey)]).stringValue
        let storedTypeRaw = try db.scalar(
            "SELECT value FROM meta WHERE key = ?", [.text(Self.elementTypeKey)]).stringValue
        let storedFingerprint = try db.scalar(
            "SELECT value FROM meta WHERE key = ?", [.text(Self.fingerprintKey)]).stringValue
        let existingSQL = try db.scalar(
            "SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
            [.text(Self.vectorTable)]).stringValue
        // 旧库没有元素类型 meta：从建表语句推断；普通 BLOB 表按 fp32 处理。
        let storedType = storedTypeRaw.flatMap { VectorElementType(rawValue: $0) }
            ?? existingSQL.flatMap { VectorElementType.infer(fromSchema: $0) }
            ?? (existingSQL != nil ? .fp32 : nil)
        let hasRows = (try? db.scalar("SELECT 1 FROM \(Self.chunksTable) LIMIT 1").intValue) != nil

        var reason: String?
        if let stored {
            if Int(stored) != dimension {
                reason = "向量维度 \(stored) → \(dimension)"
            }
        } else if hasRows {
            reason = "向量库缺少维度元数据"
        }
        if reason == nil, let storedType, storedType != elementType {
            reason = "向量元素类型 \(storedType.rawValue) → \(elementType.rawValue)"
        }
        if reason == nil, !indexFingerprint.isEmpty {
            if let storedFingerprint, !storedFingerprint.isEmpty {
                if storedFingerprint != indexFingerprint {
                    reason = "embedding / 索引参数已变化"
                }
            } else if hasRows {
                reason = "旧库缺少索引指纹"
            }
        }

        if let reason {
            if hasRows {
                rebuildReason = reason
                AgentLog.warning("向量库需要重建：\(reason)（旧数据保留，重建前向量检索关闭）")
            } else {
                AgentLog.info("向量库配置变化（\(reason)），空库直接采用新配置")
                try adoptCurrentConfigurationLocked()
            }
        }
        // 仅在配置一致（或空库已采纳新配置）时补齐/刷新 meta 键。
        // 待重建时必须保留旧 meta：否则下次打开会把新配置误当成已生效，
        // 进而按新类型重建向量表，静默丢掉旧向量。
        if rebuildReason == nil {
            try writeVectorMetaLocked()
        }
    }

    /// 空库直接采纳当前维度/类型/指纹：丢弃旧向量表与 ANN 索引（没有数据损失）。
    private func adoptCurrentConfigurationLocked() throws {
        didResetForDimensionChange = true
        try db.transaction {
            try db.execute("DROP TABLE IF EXISTS \(Self.vectorTable)")
            if binaryStorageEnabled {
                try db.execute("DROP TABLE IF EXISTS \(Self.binaryVectorTable)")
                try deleteBinaryMetaLocked()
                binaryRows = 0
            }
            binaryStorageEnabled = false
            try db.execute("DROP TABLE IF EXISTS \(Self.annCellsTable)")
            try db.execute("DROP TABLE IF EXISTS \(Self.annBlocksTable)")
            try db.execute("DELETE FROM meta WHERE key GLOB 'ann_*'")
            try dropRerankStorageLocked()
            try db.execute("DELETE FROM meta WHERE key IN (?, ?, ?)",
                           [.text(Self.dimensionKey), .text(Self.elementTypeKey),
                            .text(Self.fingerprintKey)])
            // 空库也可能残留 documents/计数器（上次同步中断），一并归零。
            try db.execute("DELETE FROM \(Self.chunksTable)")
            try db.execute("DELETE FROM \(Self.ftsTable)")
            try db.execute("DELETE FROM \(Self.documentsTable)")
            try db.execute("DELETE FROM \(Self.chapterSummariesTable)")
            try db.execute("DELETE FROM \(Self.chapterSegmentsTable)")
            try db.execute("DELETE FROM \(Self.bookSummariesTable)")
            try db.execute("DELETE FROM \(Self.bookSegmentsTable)")
            try db.execute("DELETE FROM \(Self.summaryJobTable)")
            try db.execute("DELETE FROM \(Self.summaryRuntimeTable)")
            try db.execute(
                "INSERT INTO meta(key, value) VALUES('\(Self.chunkCountKey)', '0') "
                + "ON CONFLICT(key) DO UPDATE SET value = excluded.value")
        }
        annCentroidCache = nil
    }

    /// 写入当前配置的索引 meta。
    private func writeVectorMetaLocked() throws {
        var values: [(String, String)] = [
            (Self.dimensionKey, String(dimension)),
            (Self.elementTypeKey, elementType.rawValue),
        ]
        if !indexFingerprint.isEmpty {
            values.append((Self.fingerprintKey, indexFingerprint))
        }
        for (key, value) in values {
            try db.execute("""
            INSERT INTO meta(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, [.text(key), .text(value)])
        }
    }

    private func ensureVectorTable(dimension: Int, useVec0: Bool) throws {
        let existingSQL = try db.scalar(
            "SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
            [.text(Self.vectorTable)]
        ).stringValue

        if let existingSQL, existingSQL.contains("USING vec0") != useVec0 {
            try db.execute("DROP TABLE IF EXISTS \(Self.vectorTable)")
        } else if let existingSQL, useVec0 {
            // 维度与元素类型已由 meta 校验；列类型不符时说明是旧表，
            // 调用方只有在空库/显式重建时才会走到这里，直接替换。
            if VectorElementType.infer(fromSchema: existingSQL) == elementType {
                return
            }
            try db.execute("DROP TABLE IF EXISTS \(Self.vectorTable)")
        } else if existingSQL != nil, !useVec0 {
            // 普通表已存在，确认列结构即可。
            return
        }

        if useVec0 {
            try db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS \(Self.vectorTable)
            USING vec0(\(elementType.columnDefinition(dimension: dimension)))
            """)
        } else {
            try db.execScript("""
            CREATE TABLE IF NOT EXISTS \(Self.vectorTable) (
                rowid INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL
            );
            """)
        }
    }

    // MARK: - Write

    /// 批量写入（按 source upsert）：同一 source 的旧切片先删后插。
    /// 所有 source 的全部改动都在**一个事务**内，任意失败整体回滚。
    @discardableResult
    public func add(chunks: [Chunk], vectors: [[Float]]) throws -> Int {
        try add(chunks: chunks, vectors: vectors, binarySignatures: nil)
    }

    /// 批量写入（按 source upsert）。签名非 nil 时，主向量与 bin 行在同一
    /// 事务内写入，rowid 严格一致；签名缺失时只写主向量，覆盖率随之下降。
    @discardableResult
    public func add(chunks: [Chunk], vectors: [[Float]],
                    binarySignatures: [Data]?) throws -> Int {
        guard !chunks.isEmpty else { return 0 }
        guard chunks.count == vectors.count else {
            throw AgentError.storage(
                "chunks 与 vectors 数量不一致: \(chunks.count) vs \(vectors.count)")
        }
        if let binarySignatures, binarySignatures.count != chunks.count {
            throw AgentError.storage(
                "二值签名数量与切片数量不一致: "
                + "\(binarySignatures.count) vs \(chunks.count)")
        }
        // FTS 预分词是纯 CPU 工作（实测约占单切片写入九成），放在写锁外批量并行完成：
        // 写事务只做插入，检索取向量候选时不会被整批分词阻塞。
        let ftsTexts = TextAnalyzer.ftsIndexTexts(chunks.map(\.text))
        lock.lock(); defer { lock.unlock() }
        if let rebuildReason {
            throw AgentError.storage(
                "向量库需要重建（\(rebuildReason)）；请先在设置中执行「重建向量库」")
        }
        // 先确定计数基线：旧库没有 meta 计数器时用 MAX(rowid) 兜底。
        ensureChunkCounterLocked()

        // 模型未声明维度时，首次推理才发现真实值：空库允许直接重建；
        // 已有数据的库必须报错，避免静默清空用户的知识库。
        if let first = vectors.first, first.count != dimension {
            guard !hasChunks else {
                throw AgentError.storage(
                    "向量维度 \(first.count) != \(dimension)；请重建向量库后再同步")
            }
            try elementType.validate(dimension: first.count)
            AgentLog.warning("VectorStore: 按实际输出维度 \(first.count) 重建空库（原 \(dimension)）")
            try rebuildDimensionLocked(first.count)
        }

        if let binarySignatures, binaryStorageEnabled {
            let expected = binaryDimension / 8
            for signature in binarySignatures where signature.count != expected {
                throw AgentError.storage(
                    "二值签名字节数 \(signature.count) != \(expected)")
            }
        }

        let uniqueSources = Set(chunks.map(\.source))
        var inserted = 0
        var insertedBinary = 0
        try withChunkCounterRollback {
            try db.transaction {
                // 1) 删除这些 source 的旧数据（分批，事务内）。
                for source in uniqueSources {
                    _ = try deleteBySourceLocked(source)
                }
                // 2) 分批插入正文 + FTS + 向量（以及可选 bin 行）。
                var offset = 0
                while offset < chunks.count {
                    let end = min(offset + insertBatchSize, chunks.count)
                    for i in offset..<end {
                        let chunk = chunks[i]
                        let vector = VectorCodec.normalize(vectors[i])
                        guard vector.count == dimension else {
                            throw AgentError.storage("向量维度 \(vector.count) != \(dimension)")
                        }
                        let metadataJSON = JSONValue.object(chunk.metadata.mapValues { .string($0) }).jsonString()
                        let returned = try db.query("""
                        INSERT INTO chunks(id, source, source_hash, chunk_index, text, heading, metadata, created_at)
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?) RETURNING rowid
                        """, [
                            .text(chunk.id), .text(chunk.source),
                            .text(Self.hashSource(chunk.source)),
                            .int(Int64(chunk.chunkIndex)), .text(chunk.text),
                            .text(chunk.heading), .text(metadataJSON),
                            .double(Date().timeIntervalSince1970),
                        ])
                        guard let rowID = returned.first?["rowid"]?.intValue else {
                            throw AgentError.storage("写入 chunks 后未返回 rowid: \(chunk.id)")
                        }
                        try db.execute(
                            "INSERT INTO chunks_fts(rowid, tokens, source_hash, chunk_index) VALUES(?, ?, ?, ?)",
                            [.int(rowID), .text(ftsTexts[i]),
                             .text(Self.hashSource(chunk.source)), .int(Int64(chunk.chunkIndex))]
                        )
                        try insertVector(rowID: rowID, vector: vector)
                        if binaryStorageEnabled, let binarySignatures {
                            try insertBinarySignature(rowID: rowID,
                                                      signature: binarySignatures[i])
                            insertedBinary += 1
                        }
                        inserted += 1
                    }
                    offset = end
                }
                // 删除阶段已扣减，这里只补新增，随事务一起提交。
                if let cachedChunkCount {
                    self.cachedChunkCount = cachedChunkCount + inserted
                    try persistChunkCounterLocked()
                }
                if binaryStorageEnabled {
                    binaryRows += insertedBinary
                    try persistBinaryCounterLocked()
                }
                try persistRerankCounterLocked()
            }
        }
        AgentLog.info("VectorStore: 写入 \(inserted) 个切片"
            + (insertedBinary > 0 ? "、\(insertedBinary) 条二值签名" : "")
            + "（\(uniqueSources.count) 个来源，单事务）")
        return inserted
    }

    /// 仅在库为空时重建向量表（调用方已持锁）。
    private func rebuildDimensionLocked(_ newDimension: Int) throws {
        let resolved = max(1, newDimension)
        try elementType.validate(dimension: resolved)
        try db.transaction {
            try db.execute("DROP TABLE IF EXISTS \(Self.vectorTable)")
            // 只在空库上重建，但旧维度的 ANN 索引同样必须丢弃。
            try? resetANNIndexLocked()
            try ensureVectorTable(dimension: resolved, useVec0: vectorEnabled)
            if binaryStorageEnabled {
                try db.execute("DELETE FROM \(Self.binaryVectorTable)")
                binaryRows = 0
                try persistBinaryCounterLocked()
            }
            try resetRerankStorageLocked()
            // 只在空库上重建；计数器同步归零，避免自增 rowid 残留造成虚高。
            try db.execute(
                "INSERT INTO meta(key, value) VALUES('\(Self.chunkCountKey)', '0') "
                + "ON CONFLICT(key) DO UPDATE SET value = excluded.value")
        }
        // 事务成功后再更新内存状态：失败时仍与磁盘上的旧维度一致。
        dimension = resolved
        didResetForDimensionChange = true
        cachedChunkCount = 0
        try writeVectorMetaLocked()
        // 镜像表已在事务内重置，这里用新维度刷新其 meta。
        try writeRerankMetaLocked()
    }

    private func insertVector(rowID: Int64, vector: [Float]) throws {
        let blob = VectorEncoder.encode(vector, as: elementType)
        // int8/bit 必须包 vec_int8()/vec_bit()，否则 sqlite-vec 会把裸 blob
        // 一律按 float32 解析；普通表不需要 subtype，直接存原始字节。
        let expression = vectorEnabled ? elementType.bindExpression("?") : "?"
        if vectorEnabled {
            try db.execute(
                "INSERT INTO \(Self.vectorTable)(rowid, embedding) VALUES(?, \(expression))",
                [.int(rowID), .blob(blob)])
        } else {
            try db.execute(
                "INSERT OR REPLACE INTO \(Self.vectorTable)(rowid, embedding) VALUES(?, \(expression))",
                [.int(rowID), .blob(blob)])
        }
        try insertRerankVector(rowID: rowID, blob: blob)
    }

    /// 删除一个 source 的全部切片。单事务、分批。
    @discardableResult
    public func deleteBySource(_ source: String) throws -> Int {
        lock.lock(); defer { lock.unlock() }
        return try withChunkCounterRollback {
            try db.transaction {
                try deleteBySourceLocked(source)
            }
        }
    }

    private func deleteBySourceLocked(_ source: String) throws -> Int {
        let hash = Self.hashSource(source)
        return try deleteRowsLocked(
            selectSQL: "SELECT rowid FROM chunks WHERE source_hash = ?",
            bindings: [.text(hash)]
        )
    }

    private func deleteRowsLocked(selectSQL: String, bindings: [SQLValue]) throws -> Int {
        let rows = try db.query(selectSQL, bindings)
        let rowIDs = rows.compactMap { $0["rowid"]?.intValue }
        return try deleteRowIDsLocked(rowIDs)
    }

    private func deleteRowIDsLocked(_ rowIDs: [Int64]) throws -> Int {
        guard !rowIDs.isEmpty else { return 0 }
        // 删除前先确定基线；旧库没有计数器时用 MAX(rowid) 兜底。
        ensureChunkCounterLocked()
        var index = 0
        while index < rowIDs.count {
            let batch = Array(rowIDs[index..<min(index + deleteBatchSize, rowIDs.count)])
            let placeholders = Array(repeating: "?", count: batch.count).joined(separator: ",")
            let args = batch.map { SQLValue.int($0) }
            try db.execute("DELETE FROM \(Self.chunksTable) WHERE rowid IN (\(placeholders))", args)
            try db.execute("DELETE FROM \(Self.ftsTable) WHERE rowid IN (\(placeholders))", args)
            try db.execute("DELETE FROM \(Self.vectorTable) WHERE rowid IN (\(placeholders))", args)
            try deleteRerankRowsLocked(batch)
            if binaryStorageEnabled {
                let removedBinary = try db.execute(
                    "DELETE FROM \(Self.binaryVectorTable) WHERE rowid IN (\(placeholders))",
                    args)
                binaryRows = max(0, binaryRows - removedBinary)
            }
            index += batch.count
        }
        // ANN 只追加：块内旧 rowid 不随删除回收，这里累计陈旧行。
        // 与删除同事务提交；超过阈值的整体重建由 buildANNIndex 处理。
        try recordANNStaleRowsLocked(rowIDs)
        if let cachedChunkCount {
            self.cachedChunkCount = max(0, cachedChunkCount - rowIDs.count)
            try persistChunkCounterLocked()
        }
        try persistBinaryCounterLocked()
        try persistRerankCounterLocked()
        return rowIDs.count
    }

    /// 按 rowid 删除切片/全文索引/向量（单事务）。
    @discardableResult
    public func deleteRowIDs(_ rowIDs: [Int64]) throws -> Int {
        guard !rowIDs.isEmpty else { return 0 }
        lock.lock(); defer { lock.unlock() }
        return try withChunkCounterRollback {
            try db.transaction {
                try deleteRowIDsLocked(rowIDs)
            }
        }
    }

    /// 向量去重结果。
    public struct ChunkDedupeResult: Sendable {
        /// 扫描过的切片数。
        public var scannedRows: Int
        /// 发现的重复切片数（预览时为检测数，清理时为删除数）。
        public var duplicateRows: Int
        /// 实际删除的切片数；预览模式为 0。
        public var removedRows: Int
        public var removedBySource: [String: Int]
        /// 被清理来源 → 保留切片所属来源（用于写 tombstone）。
        public var canonicalBySource: [String: String]

        public init(scannedRows: Int = 0,
                    duplicateRows: Int = 0,
                    removedRows: Int = 0,
                    removedBySource: [String: Int] = [:],
                    canonicalBySource: [String: String] = [:]) {
            self.scannedRows = scannedRows
            self.duplicateRows = duplicateRows
            self.removedRows = removedRows
            self.removedBySource = removedBySource
            self.canonicalBySource = canonicalBySource
        }
    }

    /// 按切片正文哈希清理重复向量，保留 `created_at` 最早的一份。
    ///
    /// 相同正文在同一 Embedding 模型下会产生相同向量；用正文 SHA256 前 128 bit
    /// 作为键，避免把整个向量读入内存。分页扫描 + 单事务删除，保持 rowid 顺序稳定。
    @discardableResult
    public func deduplicateTextChunks(pageSize: Int = 512) throws -> ChunkDedupeResult {
        try scanDuplicateTextChunks(pageSize: pageSize, deleteDuplicates: true)
    }

    /// 只统计重复切片，不删除任何数据。用于清理前的确认弹窗。
    public func previewDuplicateTextChunks(pageSize: Int = 512) throws -> ChunkDedupeResult {
        try scanDuplicateTextChunks(pageSize: pageSize, deleteDuplicates: false)
    }

    private func scanDuplicateTextChunks(pageSize: Int,
                                         deleteDuplicates: Bool) throws -> ChunkDedupeResult {
        lock.lock(); defer { lock.unlock() }
        var result = ChunkDedupeResult()
        var firstSourceByHash: [Data: String] = [:]
        var cursorCreatedAt = -Double.greatestFiniteMagnitude
        var cursorRowID = Int64.min

        try withChunkCounterRollback {
            try db.transaction {
                while true {
                    let rows = try db.query("""
                        SELECT rowid, source, created_at, text FROM \(Self.chunksTable)
                        WHERE (created_at > ? OR (created_at = ? AND rowid > ?))
                        ORDER BY created_at ASC, rowid ASC
                        LIMIT ?
                        """, [
                            .double(cursorCreatedAt),
                            .double(cursorCreatedAt),
                            .int(cursorRowID),
                            .int(Int64(max(32, pageSize))),
                        ])
                    if rows.isEmpty { break }
                    result.scannedRows += rows.count

                    var duplicateRowIDs: [Int64] = []
                    for row in rows {
                        guard let rowID = row["rowid"]?.intValue,
                              let createdAt = row["created_at"]?.doubleValue else {
                            // 游标无法前进时直接结束，避免死循环。
                            return
                        }
                        cursorCreatedAt = createdAt
                        cursorRowID = rowID

                        guard let source = row["source"]?.stringValue,
                              let text = row["text"]?.stringValue,
                              !text.isEmpty else { continue }

                        let digest = SHA256.hash(data: Data(text.utf8))
                        let key = Data(digest.prefix(16))
                        if let keptSource = firstSourceByHash[key] {
                            duplicateRowIDs.append(rowID)
                            result.duplicateRows += 1
                            result.removedBySource[source, default: 0] += 1
                            if result.canonicalBySource[source] == nil {
                                result.canonicalBySource[source] = keptSource
                            }
                        } else {
                            firstSourceByHash[key] = source
                        }
                    }
                    if deleteDuplicates, !duplicateRowIDs.isEmpty {
                        result.removedRows += try deleteRowIDsLocked(duplicateRowIDs)
                    }
                }
            }
        }
        return result
    }

    /// 清空全部切片/向量/文档元数据（单事务）。ANN 索引同步清空：
    /// 索引里的 rowid 全部失效，留着只会误导状态展示。
    /// 待重建的库在清空后会按当前配置重建向量表，清空即可视为已完成重建。
    public func clear() throws {
        lock.lock(); defer { lock.unlock() }
        let needsRecreate = requiresRebuild
        var enabled = vectorEnabled
        try withChunkCounterRollback {
            try db.transaction {
                try db.execute("DELETE FROM \(Self.chunksTable)")
                try db.execute("DELETE FROM \(Self.ftsTable)")
                if !needsRecreate {
                    try db.execute("DELETE FROM \(Self.vectorTable)")
                }
                try db.execute("DELETE FROM \(Self.documentsTable)")
                try db.execute("DELETE FROM \(Self.chapterSummariesTable)")
                try db.execute("DELETE FROM \(Self.chapterSegmentsTable)")
                try db.execute("DELETE FROM \(Self.bookSummariesTable)")
                try db.execute("DELETE FROM \(Self.bookSegmentsTable)")
                try db.execute("DELETE FROM \(Self.summaryJobTable)")
                try db.execute("DELETE FROM \(Self.summaryRuntimeTable)")
                if binaryStorageEnabled {
                    try db.execute("DELETE FROM \(Self.binaryVectorTable)")
                    binaryRows = 0
                    try persistBinaryCounterLocked()
                }
                try resetRerankStorageLocked()
                // ANN 索引里的 rowid 与维度都失效，直接丢弃（表可能还没建）。
                try? resetANNIndexLocked()
                self.cachedChunkCount = 0
                try persistChunkCounterLocked()
                if needsRecreate {
                    enabled = try recreateVectorTableLocked()
                }
                try writeVectorMetaLocked()
            }
        }
        if needsRecreate { vectorEnabled = enabled }
        rebuildReason = nil
    }

    /// 显式重建：按当前维度/元素类型/指纹重建向量表，并清空所有切片与文档记录。
    /// 调用方负责随后重新 ingest（本方法只负责让存储层回到可写入状态）。
    /// 返回被清空的切片数。
    @discardableResult
    public func rebuildForCurrentConfiguration() throws -> Int {
        lock.lock(); defer { lock.unlock() }
        try elementType.validate(dimension: dimension)
        var removed = 0
        var enabled = vectorEnabled
        try withChunkCounterRollback {
            try db.transaction {
                removed = (try? db.scalar("SELECT COUNT(*) FROM \(Self.chunksTable)").intValue)
                    .flatMap { Int($0) } ?? maxRowIDLocked
                try db.execute("DELETE FROM \(Self.chunksTable)")
                try db.execute("DELETE FROM \(Self.ftsTable)")
                try db.execute("DELETE FROM \(Self.documentsTable)")
                try db.execute("DELETE FROM \(Self.chapterSummariesTable)")
                try db.execute("DELETE FROM \(Self.chapterSegmentsTable)")
                try db.execute("DELETE FROM \(Self.bookSummariesTable)")
                try db.execute("DELETE FROM \(Self.bookSegmentsTable)")
                try db.execute("DELETE FROM \(Self.summaryJobTable)")
                try db.execute("DELETE FROM \(Self.summaryRuntimeTable)")
                if binaryStorageEnabled {
                    try db.execute("DELETE FROM \(Self.binaryVectorTable)")
                    binaryRows = 0
                    try persistBinaryCounterLocked()
                }
                try resetRerankStorageLocked()
                try? resetANNIndexLocked()
                enabled = try recreateVectorTableLocked()
                self.cachedChunkCount = 0
                try persistChunkCounterLocked()
                try writeVectorMetaLocked()
            }
        }
        vectorEnabled = enabled
        rebuildReason = nil
        didResetForDimensionChange = true
        AgentLog.info("VectorStore: 已按当前配置重建向量表（清空 \(removed) 个切片，待重新同步）")
        return removed
    }

    /// 丢弃旧向量表并按当前配置重建。返回 sqlite-vec 是否可用。
    private func recreateVectorTableLocked() throws -> Bool {
        try db.execute("DROP TABLE IF EXISTS \(Self.vectorTable)")
        // 主表重建后镜像行全部失效，按当前配置重建空镜像。
        try resetRerankStorageLocked()
        if sqliteVecAvailable, elementType.supportsVec0Index {
            do {
                try ensureVectorTable(dimension: dimension, useVec0: true)
                return true
            } catch {
                AgentLog.warning("vec0 建表失败，退化为精确向量扫描: \(error.localizedDescription)")
                _ = try? db.execute("DROP TABLE IF EXISTS \(Self.vectorTable)")
            }
        }
        try ensureVectorTable(dimension: dimension, useVec0: false)
        return false
    }

    // MARK: - Read / Search

    /// 切片总数。优先返回持久化计数器，旧库回退到 `MAX(rowid)`。
    /// 两者都是 O(1)，绝不在热路径上执行 `COUNT(*)` 全表扫描。
    public var count: Int {
        lock.lock(); defer { lock.unlock() }
        if let cachedChunkCount { return cachedChunkCount }
        return maxRowIDLocked
    }

    /// 是否存在至少一条切片。用 `LIMIT 1` 探测（O(1)，实测 2100 万切片约 0.01s），
    /// 不信任可能来自旧库 `MAX(rowid)` 的自增基线，避免空库被误判为非空。
    public var hasChunks: Bool {
        lock.lock(); defer { lock.unlock() }
        return (try? db.scalar("SELECT 1 FROM \(Self.chunksTable) LIMIT 1").intValue) != nil
    }

    private var maxRowIDLocked: Int {
        (try? db.scalar("SELECT COALESCE(MAX(rowid), 0) FROM \(Self.chunksTable)").intValue)
            .flatMap { Int($0) } ?? 0
    }

    private func ensureChunkCounterLocked() {
        if cachedChunkCount == nil {
            cachedChunkCount = maxRowIDLocked
        }
    }

    /// 事务失败时恢复内存计数器：SQLite 已回滚磁盘写入，内存值也必须回滚，
    /// 否则后续 count/status 会与库内实际切片数不一致。
    private func withChunkCounterRollback<T>(_ body: () throws -> T) rethrows -> T {
        let chunksBefore = cachedChunkCount
        let binaryBefore = binaryRows
        let rerankBefore = rerankRows
        do {
            return try body()
        } catch {
            cachedChunkCount = chunksBefore
            binaryRows = binaryBefore
            rerankRows = rerankBefore
            throw error
        }
    }

    /// 计数器写入失败必须让所在事务一起回滚，避免磁盘与内存分叉。
    private func persistChunkCounterLocked() throws {
        guard let cachedChunkCount else { return }
        try db.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            + "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [.text(Self.chunkCountKey), .text(String(cachedChunkCount))])
    }

    static func readStoredCount(_ db: SQLiteDatabase, key: String) -> Int? {
        guard let value = try? db.scalar(
            "SELECT value FROM meta WHERE key = ?", [.text(key)]).stringValue,
            let count = Int(value) else { return nil }
        return count
    }

    /// 已索引来源总数。**全表扫描**，只在非热路径/诊断时使用；
    /// 状态展示优先使用 manifest 中的来源数（见 `RAGEngine.status`）。
    public func sourceCount() -> Int {
        (try? db.scalar("SELECT COUNT(DISTINCT source) FROM \(Self.chunksTable)").intValue)
            .flatMap { Int($0) } ?? 0
    }

    public func listSources(limit: Int = 0) -> [String] {
        let rows: [[String: SQLValue]]?
        if limit > 0 {
            rows = try? db.query(
                "SELECT DISTINCT source FROM \(Self.chunksTable) ORDER BY source LIMIT ?",
                [.int(Int64(limit))])
        } else {
            rows = try? db.query(
                "SELECT DISTINCT source FROM \(Self.chunksTable) ORDER BY source")
        }
        return rows?.compactMap { $0["source"]?.stringValue } ?? []
    }

    /// 单个来源实际入库的切片数，用于校验 manifest 的完成状态。
    public func chunkCount(source: String) -> Int {
        let value = try? db.scalar(
            "SELECT COUNT(*) FROM \(Self.chunksTable) WHERE source = ?",
            [.text(source)]).intValue
        return value.flatMap { Int($0) } ?? 0
    }

    /// 两段式向量 KNN：
    /// 1) 向量表 KNN 取 rowid（over-fetch）；
    /// 2) 主表按 rowid 批量取正文（无 JOIN）。
    ///
    /// `ann` 非空时优先走分块 IVF 向量索引；索引不可用（未构建/维度不符/
    /// 尾块超出可扫描范围）时自动回退到精确路径，调用方无需判断。
    public func search(vector: [Float], topK: Int, sourceFilter: String? = nil,
                       sourceContains: String? = nil,
                       ann: ANNQueryOptions? = nil) throws -> [SearchHit] {
        guard topK > 0 else { return [] }
        guard vector.count == dimension else {
            throw AgentError.storage("查询向量维度 \(vector.count) != \(dimension)")
        }
        // 与写入/删除共用一把锁：两段式查询之间不能有事务提交，
        // 否则 KNN 拿到的 rowid 可能在 hydrate 时已被删掉。
        lock.lock(); defer { lock.unlock() }
        if let rebuildReason {
            AgentLog.info("向量库待重建（\(rebuildReason)），本次跳过向量检索")
            return []
        }
        guard hasChunks else { return [] }
        // 精确来源：直接在该来源内部检索。全局 KNN 的候选池有上限
        // （sqlite-vec k 最大 4096），大库里目标来源的切片可能永远排不进
        // 前 k 名，过滤后就会返回空。
        if let sourceFilter {
            return try scopedSourceSearchLocked(vector: vector, topK: topK,
                                                source: sourceFilter,
                                                sourceContains: sourceContains)
        }
        if let ann,
           let hits = try annSearchLocked(vector: vector, topK: topK, options: ann,
                                          sourceFilter: sourceFilter,
                                          sourceContains: sourceContains) {
            return hits
        }
        return try bruteSearchLocked(vector: vector, topK: topK,
                                     sourceFilter: sourceFilter,
                                     sourceContains: sourceContains)
    }

    /// 精确来源检索：只读取该来源的向量并排序，绕开全局 KNN 的候选上限。
    /// 调用方需已持有 `lock` 且确认库非空。
    func scopedSourceSearchLocked(vector: [Float], topK: Int, source: String,
                                  sourceContains: String?) throws -> [SearchHit] {
        let rows = chunkCount(source: source)
        guard rows > 0 else { return [] }
        // 超大来源仍走精确扫描会一次性占用过多内存：退回原有路径，
        // 由调用方的超时预算兜底。
        guard rows <= Self.scopedExactRowLimit else {
            return try bruteSearchLocked(vector: vector, topK: topK,
                                         sourceFilter: source,
                                         sourceContains: sourceContains)
        }
        let query = VectorCodec.normalize(vector)
        let candidates = try knnExact(query: query, sourceFilter: source)
        return try hydrate(candidates: candidates, sourceFilter: source,
                           sourceContains: sourceContains, limit: topK)
    }

    /// 精确检索（vec0 KNN 或全表扫描）。调用方需已持有 `lock` 且确认库非空。
    func bruteSearchLocked(vector: [Float], topK: Int, sourceFilter: String?,
                           sourceContains: String?) throws -> [SearchHit] {
        let candidates = try bruteCandidatesLocked(vector: vector, topK: topK,
                                                   sourceFilter: sourceFilter,
                                                   sourceContains: sourceContains)
        return try hydrate(candidates: candidates, sourceFilter: sourceFilter,
                           sourceContains: sourceContains, limit: topK)
    }

    /// 向量通道候选（只返回 rowid + 相似度，不读正文）。
    /// 调用方需已持有 `lock` 且确认库非空。
    func vectorCandidatesLocked(vector: [Float], topK: Int,
                                sourceFilter: String?,
                                ann: ANNQueryOptions?,
                                targeted: Bool = false) throws -> [(Int64, Double)] {
        guard topK > 0 else { return [] }
        if let sourceFilter {
            return try scopedSourceCandidatesLocked(vector: vector, topK: topK,
                                                    source: sourceFilter)
        }
        if let ann,
           let candidates = try annCandidatesLocked(vector: vector, topK: topK,
                                                    options: ann,
                                                    targeted: targeted) {
            return candidates
        }
        return try bruteCandidatesLocked(vector: vector, topK: topK,
                                         sourceFilter: nil)
    }

    /// 精确来源候选：在该来源内部召回，绕开全局 KNN 的候选上限。
    func scopedSourceCandidatesLocked(vector: [Float], topK: Int,
                                      source: String) throws -> [(Int64, Double)] {
        let rows = chunkCount(source: source)
        guard rows > 0 else { return [] }
        guard rows <= Self.scopedExactRowLimit else {
            // 超大来源：退回全局多取并过滤，由调用方超时预算兜底。
            return try bruteCandidatesLocked(vector: vector, topK: max(topK, 20),
                                             sourceFilter: source)
        }
        let query = VectorCodec.normalize(vector)
        return Array(try knnExact(query: query, sourceFilter: source).prefix(topK))
    }

    /// 精确候选（vec0 KNN 或全表扫描），不读正文。
    /// `sourceContains` 只用于决定 vec0 的 k 要放大多少：子串匹配在 SQL 里
    /// 无法走索引，必须先多取候选再按来源列过滤，否则目标来源会被全局
    /// Top-k 挤空（与 sourceFilter 的处理一致）。
    func bruteCandidatesLocked(vector: [Float], topK: Int,
                               sourceFilter: String?,
                               sourceContains: String? = nil) throws -> [(Int64, Double)] {
        let query = VectorCodec.normalize(vector)
        let needle = sourceContains?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if vectorEnabled {
            let widened = sourceFilter != nil || !needle.isEmpty
            var k = min(max(topK, widened ? topK * 20 : topK), maxVectorK)
            while true {
                var candidates = try knnVec0(query: query, k: k)
                if let sourceFilter {
                    candidates = try filterCandidatesLocked(candidates, source: sourceFilter)
                }
                if !needle.isEmpty {
                    candidates = try filterCandidatesLocked(candidates, contains: needle)
                }
                if candidates.count >= topK || !widened || k >= maxVectorK {
                    return Array(candidates.prefix(topK))
                }
                k = min(k * 2, maxVectorK)
            }
        }
        let candidates = try knnExact(query: query, sourceFilter: sourceFilter)
        guard !needle.isEmpty else { return Array(candidates.prefix(topK)) }
        let filtered = try filterCandidatesLocked(candidates, contains: needle)
        return Array(filtered.prefix(topK))
    }

    /// 候选行按来源过滤（只读 source 列，比 hydrate 全文便宜得多）。
    private func filterCandidatesLocked(_ candidates: [(Int64, Double)],
                                        source: String? = nil,
                                        contains: String? = nil) throws -> [(Int64, Double)] {
        try filterCandidates(db, candidates, source: source, contains: contains)
    }

    /// 候选行按来源过滤。`db` 可以是写连接，也可以是检索用的只读快照连接。
    private func filterCandidates(_ db: SQLiteDatabase,
                                  _ candidates: [(Int64, Double)],
                                  source: String? = nil,
                                  contains: String? = nil) throws -> [(Int64, Double)] {
        guard !candidates.isEmpty, source != nil || contains != nil else { return candidates }
        var allowed = Set<Int64>()
        allowed.reserveCapacity(candidates.count)
        let ids = candidates.map(\.0)
        var index = 0
        while index < ids.count {
            let batch = Array(ids[index..<min(index + deleteBatchSize, ids.count)])
            let placeholders = Array(repeating: "?", count: batch.count).joined(separator: ",")
            let rows = try db.query(
                "SELECT rowid, source FROM \(Self.chunksTable) WHERE rowid IN (\(placeholders))",
                batch.map { SQLValue.int($0) })
            for row in rows {
                guard let rowID = row["rowid"]?.intValue,
                      let candidateSource = row["source"]?.stringValue else { continue }
                if let source, candidateSource != source { continue }
                if let contains,
                   !candidateSource.localizedCaseInsensitiveContains(contains) { continue }
                allowed.insert(rowID)
            }
            index += batch.count
        }
        return candidates.filter { allowed.contains($0.0) }
    }

    /// 供 stage 2 与 ANN 重打分使用：优先读 rerank 镜像，缺失行回退主向量表。
    /// rowid 升序 + 256 条一批，降低 B-tree 随机跳转与 prepare/step 次数。
    func exactVectorsLocked(rowIDs: [Int64]) throws -> [Int64: [Float]] {
        guard !rowIDs.isEmpty else { return [:] }
        let ordered = rowIDs.sorted()
        let batchSize = 256
        var output: [Int64: [Float]] = [:]
        output.reserveCapacity(ordered.count)
        var index = 0
        while index < ordered.count {
            let batch = Array(ordered[index..<min(index + batchSize, ordered.count)])
            var missing = batch
            if rerankStorageEnabled, rerankRows > 0 {
                let placeholders = Array(repeating: "?", count: batch.count)
                    .joined(separator: ",")
                let rows = try db.query(
                    "SELECT rowid, embedding FROM \(Self.rerankVectorTable) "
                    + "WHERE rowid IN (\(placeholders))",
                    batch.map { SQLValue.int($0) })
                for row in rows {
                    guard let rowID = row["rowid"]?.intValue,
                          let data = row["embedding"]?.dataValue else { continue }
                    output[rowID] = VectorEncoder.decode(data, as: elementType,
                                                         dimension: dimension)
                }
                let found = Set(output.keys)
                missing = batch.filter { !found.contains($0) }
            }
            if !missing.isEmpty {
                let placeholders = Array(repeating: "?", count: missing.count)
                    .joined(separator: ",")
                let rows = try db.query(
                    "SELECT rowid, embedding FROM \(Self.vectorTable) "
                    + "WHERE rowid IN (\(placeholders))",
                    missing.map { SQLValue.int($0) })
                for row in rows {
                    guard let rowID = row["rowid"]?.intValue,
                          let data = row["embedding"]?.dataValue else { continue }
                    output[rowID] = VectorEncoder.decode(data, as: elementType,
                                                         dimension: dimension)
                }
            }
            index += batch.count
        }
        return output
    }

    /// 某个来源的全部切片 rowid（按 chunk_index 升序）。走 (source_hash,
    /// chunk_index) 索引，不做全表扫描。
    public func rowIDs(source: String) throws -> [Int64] {
        let rows = try db.query("""
        SELECT rowid FROM \(Self.chunksTable) WHERE source_hash = ? ORDER BY chunk_index ASC
        """, [.text(Self.hashSource(source))])
        return rows.compactMap { $0["rowid"]?.intValue }
    }

    private func knnVec0(query: [Float], k: Int) throws -> [(Int64, Double)] {
        // 查询侧与入库侧必须使用同一种编码（int8 / bit 都要包 subtype）。
        let encoded = VectorEncoder.encode(query, as: elementType)
        let rows = try db.query(
            "SELECT rowid, distance FROM \(Self.vectorTable) "
            + "WHERE embedding MATCH \(elementType.bindExpression("?")) AND k = ?",
            [.blob(encoded), .int(Int64(k))]
        )
        return rows.compactMap { row in
            guard let rowID = row["rowid"]?.intValue else { return nil }
            let distance = row["distance"]?.doubleValue ?? 2.0
            return (rowID, elementType.similarity(distance: distance, dimension: dimension))
        }
    }

    private func knnExact(query: [Float], sourceFilter: String?) throws -> [(Int64, Double)] {
        let queryBits = elementType.isBinary ? VectorEncoder.encodeBits(query) : Data()
        var scored: [(Int64, Double)] = []

        func appendScores(rows: [[String: SQLValue]]) {
            for row in rows {
                guard let rowID = row["rowid"]?.intValue,
                      let data = row["embedding"]?.dataValue else { continue }
                if elementType.isBinary {
                    let distance = Double(VectorEncoder.hamming(data, queryBits))
                    scored.append((rowID, elementType.similarity(distance: distance,
                                                                 dimension: dimension)))
                } else {
                    let candidate = VectorEncoder.decode(data, as: elementType,
                                                         dimension: dimension)
                    scored.append((rowID, VectorCodec.cosine(query, candidate)))
                }
            }
        }

        if let sourceFilter {
            // 仍然是两段式：先取该 source 的 rowid，再按批读向量（不做 JOIN）。
            // 单条 IN 语句的变量数有上限，大来源必须分批，否则会直接报错。
            let ids = try rowIDs(source: sourceFilter)
            guard !ids.isEmpty else { return [] }
            scored.reserveCapacity(ids.count)
            var index = 0
            while index < ids.count {
                let batch = Array(ids[index..<min(index + deleteBatchSize, ids.count)])
                let placeholders = Array(repeating: "?", count: batch.count).joined(separator: ",")
                let rows = try db.query(
                    "SELECT rowid, embedding FROM \(Self.vectorTable) "
                    + "WHERE rowid IN (\(placeholders))",
                    batch.map { SQLValue.int($0) })
                appendScores(rows: rows)
                index += batch.count
            }
        } else {
            let rows = try db.query("SELECT rowid, embedding FROM \(Self.vectorTable)")
            scored.reserveCapacity(rows.count)
            appendScores(rows: rows)
        }
        return scored.sorted { $0.1 > $1.1 }
    }

    /// 按 rowid 批量读取正文。用列序号 + `stream` 直接取值，避免为每个候选
    /// 构造 `[String: SQLValue]` 字典（ANN/FTS 路径一次可能取数百行）。
    private struct HydratedChunk {
        let text: String
        let source: String
        let chunkIndex: Int
        let heading: String
    }

    private func loadChunksLocked(rowIDs: [Int64]) throws -> [Int64: HydratedChunk] {
        try loadChunks(db, rowIDs: rowIDs)
    }

    /// 按 rowid 批量读取正文；`db` 可以是写连接或只读快照连接。
    private func loadChunks(_ db: SQLiteDatabase,
                            rowIDs: [Int64]) throws -> [Int64: HydratedChunk] {
        guard !rowIDs.isEmpty else { return [:] }
        var output: [Int64: HydratedChunk] = [:]
        output.reserveCapacity(rowIDs.count)
        var index = 0
        while index < rowIDs.count {
            let batch = Array(rowIDs[index..<min(index + deleteBatchSize, rowIDs.count)])
            let placeholders = Array(repeating: "?", count: batch.count).joined(separator: ",")
            let sql = """
            SELECT rowid, text, source, chunk_index, heading
            FROM \(Self.chunksTable) WHERE rowid IN (\(placeholders))
            """
            try db.stream(sql, batch.map { SQLValue.int($0) }) { row in
                let rowID = row.int(0)
                output[rowID] = HydratedChunk(
                    text: row.text(1) ?? "",
                    source: row.text(2) ?? "",
                    chunkIndex: Int(row.int(3)),
                    heading: row.text(4) ?? "")
                return true
            }
            index += batch.count
        }
        return output
    }

    /// 二段式收尾：按 rowid 批量取正文并过滤（精确路径与 ANN 路径共用）。
    /// 调用方需已持有 `lock`。
    func hydrate(candidates: [(Int64, Double)], sourceFilter: String?,
                 sourceContains: String? = nil, limit: Int) throws -> [SearchHit] {
        try hydrate(db, candidates: candidates, sourceFilter: sourceFilter,
                    sourceContains: sourceContains, limit: limit)
    }

    /// 二段式收尾；`db` 可以是写连接或只读快照连接。
    func hydrate(_ db: SQLiteDatabase, candidates: [(Int64, Double)], sourceFilter: String?,
                 sourceContains: String? = nil, limit: Int) throws -> [SearchHit] {
        guard !candidates.isEmpty else { return [] }
        let starts = stride(from: 0, to: candidates.count, by: deleteBatchSize).map { $0 }
        var result: [SearchHit] = []
        result.reserveCapacity(min(limit, candidates.count))
        for start in starts {
            guard result.count < limit else { break }
            let batch = Array(candidates[start..<min(start + deleteBatchSize, candidates.count)])
            let byID = try loadChunks(db, rowIDs: batch.map(\.0))
            for (rowID, score) in batch {
                guard let row = byID[rowID] else { continue }
                if let sourceFilter, row.source != sourceFilter { continue }
                if let sourceContains,
                   !row.source.localizedCaseInsensitiveContains(sourceContains) { continue }
                result.append(SearchHit(
                    rowID: rowID,
                    text: row.text,
                    source: row.source,
                    chunkIndex: row.chunkIndex,
                    heading: row.heading,
                    score: score,
                    vectorScore: score
                ))
                if result.count >= limit { break }
            }
        }
        return result
    }

    /// FTS5 关键词检索，BM25 分数归一化为 [0,1)。
    ///
    /// 查询 token 先按「停用词过滤 + 最多 32 个显著 token」构造 AND 表达式；
    /// AND 一条都命中不了时再退化为 OR，避免多个关键词被 OR 稀释成噪声。
    ///
    /// 候选查询与 hydrate 在同一个只读事务内完成，不持有写入锁：删除/写入
    /// 可以并发提交，命中集合不会出现「FTS 有、主表已被删」的半截结果。
    public func searchFTS(query: String, topK: Int, sourceFilter: String? = nil,
                          sourceContains: String? = nil,
                          deadline: Date? = nil) throws -> [SearchHit] {
        guard topK > 0 else { return [] }
        let fetchLimit = sourceContains == nil ? topK : topK * 20
        return try withReadSnapshot(deadline: deadline) { reader in
            let ranked = try ftsCandidates(reader, query: query, limit: fetchLimit,
                                           sourceFilter: sourceFilter)
            guard !ranked.isEmpty else { return [] }
            let hydrated = try hydrateRaw(reader, ranked: ranked, sourceFilter: sourceFilter,
                                          sourceContains: sourceContains, limit: topK)
            return hydrated.map { hit in
                var copy = hit
                let raw = hit.score
                copy.score = raw > 0 ? raw / (1 + raw) : 0
                copy.ftsScore = raw
                return copy
            }
        }
    }

    /// FTS5 候选（rowid + 原始 BM25 分，不读正文）。调用方需已持有 `lock`。
    func ftsCandidatesLocked(query: String, limit: Int,
                             sourceFilter: String?) throws -> [(Int64, Double)] {
        try ftsCandidates(db, query: query, limit: limit, sourceFilter: sourceFilter)
    }

    /// FTS5 候选；`db` 可以是写连接或只读快照连接。
    func ftsCandidates(_ db: SQLiteDatabase, query: String, limit: Int,
                       sourceFilter: String?) throws -> [(Int64, Double)] {
        guard limit > 0 else { return [] }
        let andQuery = TextAnalyzer.ftsQuery(query, mode: .and)
        let orQuery = TextAnalyzer.ftsQuery(query, mode: .or)
        guard !orQuery.isEmpty else { return [] }

        func run(_ match: String) throws -> [[String: SQLValue]] {
            if let sourceFilter {
                return try db.query("""
                SELECT rowid, bm25(\(Self.ftsTable)) AS rank
                FROM \(Self.ftsTable)
                WHERE tokens MATCH ? AND source_hash = ?
                ORDER BY rank LIMIT ?
                """, [.text(match), .text(Self.hashSource(sourceFilter)), .int(Int64(limit))])
            }
            return try db.query("""
            SELECT rowid, bm25(\(Self.ftsTable)) AS rank
            FROM \(Self.ftsTable)
            WHERE tokens MATCH ?
            ORDER BY rank LIMIT ?
            """, [.text(match), .int(Int64(limit))])
        }

        var rows: [[String: SQLValue]] = []
        do {
            rows = try run(andQuery)
            if rows.isEmpty, andQuery != orQuery {
                rows = try run(orQuery)
            }
        } catch {
            AgentLog.warning("FTS5 检索失败，退化为纯向量检索: \(error.localizedDescription)")
            return []
        }

        return rows.compactMap { row -> (Int64, Double)? in
            guard let rowID = row["rowid"]?.intValue else { return nil }
            // bm25() 越小越相关，通常是负数。
            let raw = -(row["rank"]?.doubleValue ?? 0)
            return (rowID, raw)
        }
    }

    private func hydrateRaw(_ db: SQLiteDatabase, ranked: [(Int64, Double)],
                            sourceFilter: String?,
                            sourceContains: String? = nil, limit: Int) throws -> [SearchHit] {
        guard !ranked.isEmpty else { return [] }
        let byID = try loadChunks(db, rowIDs: ranked.map(\.0))
        var result: [SearchHit] = []
        for (rowID, score) in ranked {
            guard let row = byID[rowID] else { continue }
            if let sourceFilter, row.source != sourceFilter { continue }
            if let sourceContains,
               !row.source.localizedCaseInsensitiveContains(sourceContains) { continue }
            result.append(SearchHit(
                rowID: rowID,
                text: row.text,
                source: row.source,
                chunkIndex: row.chunkIndex,
                heading: row.heading,
                score: score,
                ftsScore: score
            ))
            if result.count >= limit { break }
        }
        return result
    }

    /// 混合检索：FTS5 + 向量 → RRF 融合。
    ///
    /// 两路只取 rowid + 分数，先做 RRF 定出前 `topK`，最后一次性 hydrate 正文，
    /// 避免为几百个候选读取全文。`overfetch` 只按 1-2 倍内层放大。
    ///
    /// 向量候选在写入锁内取（vec0/ANN 用主连接，不能与删除并发），随后在
    /// 锁内开启只读快照并释放锁；FTS 候选、RRF 与 hydrate 都在同一快照内
    /// 完成。写入/删除在快照外可继续提交，这次检索看到的仍是同一版本。
    public func hybridSearch(vector: [Float], query: String, topK: Int,
                             sourceFilter: String? = nil,
                             sourceContains: String? = nil,
                             rrfK: Int = 60, overfetch: Int = 2,
                             binary: BinaryCoarseOptions? = nil,
                             ann: ANNQueryOptions? = nil,
                             deadline: Date? = nil) throws -> [SearchHit] {
        guard topK > 0 else { return [] }
        guard vector.count == dimension else {
            throw AgentError.storage("查询向量维度 \(vector.count) != \(dimension)")
        }
        // 向量候选必须与写入/删除串行；有期限时不能无限期等待正在提交的
        // 大批量写入，超时由调用方跳过本次检索。
        if let deadline {
            guard lock.lock(before: deadline) else {
                throw AgentError.timeout("知识库正在写入，本次查询已跳过")
            }
        } else {
            lock.lock()
        }
        var unlocked = false
        defer { if !unlocked { lock.unlock() } }
        guard hasChunks else { return [] }

        // 待重建时向量通道暂停，但全文通道仍然可用（与 status 的提示一致）。
        let vectorUsable = rebuildReason == nil
        if !vectorUsable {
            AgentLog.info("向量库待重建（\(rebuildReason ?? "")），本次只用全文检索")
        }

        // 内层候选量：每路仅比最终条数多 1-2 倍，不再乘 4。
        let inner = max(topK * max(1, overfetch), 20)
        let targeted = sourceContains != nil
        let candidateCount = targeted ? max(inner * 4, 20) : inner
        var binaryScores: [Int64: Double] = [:]
        var vectorCandidates: [(Int64, Double)] = []
        if vectorUsable {
            if let binary, sourceFilter == nil, !targeted {
                switch try binaryCoarseCandidatesLocked(
                    vector: vector, signature: binary.signature,
                    candidateLimit: binary.candidateLimit,
                    candidateCount: candidateCount, targeted: false) {
                case .used(let candidates, let scores):
                    vectorCandidates = candidates
                    binaryScores = scores
                case .skipped(let reason):
                    noteBinaryFallbackLocked(reason)
                    vectorCandidates = try vectorCandidatesLocked(
                        vector: vector, topK: candidateCount,
                        sourceFilter: sourceFilter, ann: ann, targeted: targeted)
                }
            } else {
                vectorCandidates = try vectorCandidatesLocked(
                    vector: vector, topK: candidateCount,
                    sourceFilter: sourceFilter, ann: ann, targeted: targeted)
            }
        }

        // 仍持有写入锁时开启读快照：确保 FTS 候选与 hydrate 落在同一个
        // 数据库版本；建立快照后即可解锁，让写入/删除并发提交。
        let reader = try readDatabase()
        let seconds: Double?
        if let deadline {
            let remaining = deadline.timeIntervalSinceNow
            guard remaining > 0 else { throw AgentError.timeout("查询超出时限被跳过") }
            seconds = remaining
        } else {
            seconds = nil
        }

        func run() throws -> [SearchHit] {
            try reader.readTransaction {
                _ = try reader.query("SELECT 1 FROM \(Self.chunksTable) LIMIT 1")
                lock.unlock()
                unlocked = true

                var ftsCandidates = try self.ftsCandidates(
                    reader, query: query, limit: candidateCount,
                    sourceFilter: sourceFilter)
                var vectorPool = vectorCandidates
                if let needle = sourceContains?.trimmingCharacters(in: .whitespacesAndNewlines),
                   !needle.isEmpty {
                    vectorPool = try self.filterCandidates(reader, vectorPool, contains: needle)
                    ftsCandidates = try self.filterCandidates(reader, ftsCandidates,
                                                              contains: needle)
                }
                guard !vectorPool.isEmpty || !ftsCandidates.isEmpty else { return [] }

                var rrfScores: [Int64: Double] = [:]
                var vectorScores: [Int64: Double] = [:]
                var ftsScores: [Int64: Double] = [:]
                let k = Double(max(1, rrfK))
                for (rank, candidate) in vectorPool.enumerated() {
                    rrfScores[candidate.0, default: 0] += 1.0 / (k + Double(rank + 1))
                    vectorScores[candidate.0] = candidate.1
                }
                for (rank, candidate) in ftsCandidates.enumerated() {
                    rrfScores[candidate.0, default: 0] += 1.0 / (k + Double(rank + 1))
                    ftsScores[candidate.0] = candidate.1
                }

                let fused = rrfScores.sorted { $0.value > $1.value }
                guard !fused.isEmpty else { return [] }
                // hydrate 会跳过快照建立前已被删除的 rowid，多取少量兜底。
                let pool = fused.prefix(min(fused.count, topK * 2))
                let hydrated = try self.hydrate(
                    reader, candidates: pool.map { ($0.key, $0.value) },
                    sourceFilter: sourceFilter,
                    sourceContains: sourceContains,
                    limit: topK)
                return hydrated.map { hit in
                    var copy = hit
                    copy.rrfScore = hit.score
                    copy.vectorScore = vectorScores[hit.rowID]
                    copy.ftsScore = ftsScores[hit.rowID]
                    copy.binaryScore = binaryScores[hit.rowID]
                    return copy
                }
            }
        }
        guard let seconds else { return try run() }
        return try reader.withDeadline(seconds: seconds, run)
    }

    // MARK: - Documents

    public func upsertDocument(docName: String, digest: String, chapters: String = "",
                               mdPath: String = "", contentHash: String = "") throws {
        let docID = Self.hashSource(mdPath.isEmpty ? docName : mdPath)
        let now = Date().timeIntervalSince1970
        try db.transaction {
            // 先找出同名/同 id 的旧记录：正文变化时摘要必须一起失效。
            let existing = try db.query(
                "SELECT doc_id, content_hash FROM \(Self.documentsTable) "
                + "WHERE doc_id = ? OR doc_name = ?",
                [.text(docID), .text(docName)])
            let staleIDs = Set(existing.compactMap { $0["doc_id"]?.stringValue })
            let contentChanged = !contentHash.isEmpty && existing.contains {
                ($0["content_hash"]?.stringValue ?? "") != contentHash
            }
            if contentChanged {
                for id in staleIDs where !id.isEmpty {
                    try deleteSummaryRowsLocked(docID: id)
                }
                // 新 doc_id 也可能是首次出现；清一次无副作用。
                try deleteSummaryRowsLocked(docID: docID)
            } else {
                // 正文未变但缓存路径变化：下面会删掉旧 doc_id 的文档行，
                // 其摘要必须同步清理，避免成为永远无法访问的孤儿数据。
                for id in staleIDs where !id.isEmpty && id != docID {
                    try deleteSummaryRowsLocked(docID: id)
                }
            }

            try db.execute("DELETE FROM \(Self.documentsTable) WHERE doc_id = ?", [.text(docID)])
            if !mdPath.isEmpty {
                try db.execute("DELETE FROM \(Self.documentsTable) WHERE doc_name = ? AND md_path != ?",
                               [.text(docName), .text(mdPath)])
            }
            try db.execute("""
            INSERT INTO \(Self.documentsTable)(doc_id, doc_name, digest, chapters, md_path, content_hash, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """, [.text(docID), .text(docName), .text(digest), .text(chapters),
                  .text(mdPath), .text(contentHash), .double(now)])
        }
    }

    /// 目录修复专用：只更新 digest/chapters，不改变 content_hash，也不失效摘要。
    public func updateDocumentOutline(docID: String, digest: String, chapters: String) throws {
        guard !docID.isEmpty else { return }
        try db.execute("""
        UPDATE \(Self.documentsTable) SET digest = ?, chapters = ? WHERE doc_id = ?
        """, [.text(digest), .text(chapters), .text(docID)])
    }

    public func getDocumentDigest(docName: String) -> DocumentRecord? {
        let target = docName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !target.isEmpty else { return nil }
        guard let candidates = try? lookupDocuments(query: target, limit: 50),
              !candidates.isEmpty else { return nil }
        let lower = target.lowercased()
        let standardized = (target as NSString).standardizingPath
        let exact = candidates.filter {
            $0.docName.lowercased() == lower
                || ($0.docName as NSString).lastPathComponent.lowercased() == lower
                || (!standardized.isEmpty
                    && ($0.mdPath as NSString).standardizingPath == standardized)
        }
        if exact.count == 1 { return exact[0] }
        if exact.count > 1 { return nil }
        let fuzzy = candidates.filter {
            let name = $0.docName.lowercased()
            return name.contains(lower) || lower.contains(name)
        }
        // 旧 API 无歧义返回值：唯一模糊命中才返回，多候选交给 lookupDocument。
        return fuzzy.count == 1 ? fuzzy[0] : nil
    }

    public func allDocumentRows() -> [DocumentRecord] {
        guard let rows = try? db.query("""
        SELECT doc_id, doc_name, digest, chapters, md_path, content_hash
        FROM \(Self.documentsTable)
        WHERE doc_name != '' ORDER BY doc_name
        """) else { return [] }
        return rows.map { Self.documentRecord(from: $0) }
    }

    public func listDocuments() -> [String] {
        (try? documentNames(limit: 1_000_000)) ?? []
    }

    /// SQL 限量列出文档名。`rag_outline` 列表不要再把整表读进内存。
    public func documentNames(limit: Int, offset: Int = 0) throws -> [String] {
        guard limit > 0 else { return [] }
        let rows = try db.query("""
        SELECT doc_name FROM \(Self.documentsTable)
        WHERE doc_name != '' ORDER BY doc_name LIMIT ? OFFSET ?
        """, [.int(Int64(max(0, limit))), .int(Int64(max(0, offset)))])
        return rows.compactMap { $0["doc_name"]?.stringValue }
    }

    public func documentCount() throws -> Int {
        Int(try db.scalar(
            "SELECT COUNT(*) FROM \(Self.documentsTable) WHERE doc_name != ''").intValue ?? 0)
    }

    public func getDocumentByMdPath(_ mdPath: String) -> DocumentRecord? {
        guard !mdPath.isEmpty else { return nil }
        let target = (mdPath as NSString).standardizingPath
        guard let rows = try? db.query("""
        SELECT doc_id, doc_name, digest, chapters, md_path, content_hash
        FROM \(Self.documentsTable) WHERE md_path = ? LIMIT 1
        """, [.text(target)]) else { return nil }
        return rows.first.map(Self.documentRecord(from:))
    }

    public func document(docID: String) throws -> DocumentRecord? {
        guard !docID.isEmpty else { return nil }
        let rows = try db.query("""
        SELECT doc_id, doc_name, digest, chapters, md_path, content_hash
        FROM \(Self.documentsTable) WHERE doc_id = ? LIMIT 1
        """, [.text(docID)])
        return rows.first.map(Self.documentRecord(from:))
    }

    /// SQL 侧候选过滤：精确 doc_name / md_path，或 `instr` 模糊包含。
    /// 不在 SQL 里做 basename / 唯一性判断，调用方拿到候选后统一解析。
    public func lookupDocuments(query: String, limit: Int = 50) throws -> [DocumentRecord] {
        let raw = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty, limit > 0 else { return [] }
        let standardized = (raw as NSString).standardizingPath
        let rows = try db.query("""
        SELECT doc_id, doc_name, digest, chapters, md_path, content_hash
        FROM \(Self.documentsTable)
        WHERE doc_name != '' AND (
            doc_name = ? COLLATE NOCASE
            OR md_path = ? COLLATE NOCASE
            OR instr(lower(doc_name), lower(?)) > 0
        )
        ORDER BY doc_name LIMIT ?
        """, [.text(raw), .text(standardized), .text(raw), .int(Int64(limit))])
        return rows.map { Self.documentRecord(from: $0) }
    }

    public func deleteDocument(docName: String) throws {
        let docID = Self.hashSource(docName)
        try db.transaction {
            let rows = try db.query(
                "SELECT doc_id FROM \(Self.documentsTable) WHERE doc_id = ? OR doc_name = ?",
                [.text(docID), .text(docName)])
            for id in Set(rows.compactMap { $0["doc_id"]?.stringValue }) {
                try deleteSummaryRowsLocked(docID: id)
            }
            try db.execute("DELETE FROM \(Self.documentsTable) WHERE doc_id = ? OR doc_name = ?",
                           [.text(docID), .text(docName)])
        }
    }

    public func deleteDocumentByMdPath(_ mdPath: String) throws -> Bool {
        guard !mdPath.isEmpty else { return false }
        let target = (mdPath as NSString).standardizingPath
        var removed = false
        try db.transaction {
            let rows = try db.query(
                "SELECT doc_id FROM \(Self.documentsTable) WHERE md_path = ?",
                [.text(target)])
            for id in Set(rows.compactMap { $0["doc_id"]?.stringValue }) {
                try deleteSummaryRowsLocked(docID: id)
            }
            removed = rows.count > 0
            try db.execute("DELETE FROM \(Self.documentsTable) WHERE md_path = ?", [.text(target)])
        }
        return removed
    }

    // MARK: - Chapter summaries

    public func upsertChapterSummary(_ record: ChapterSummaryRecord) throws {
        try db.execute("""
        INSERT INTO \(Self.chapterSummariesTable)
            (doc_id, section_index, heading, title, level, summary, content_hash, updated_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id, section_index) DO UPDATE SET
            heading = excluded.heading,
            title = excluded.title,
            level = excluded.level,
            summary = excluded.summary,
            content_hash = excluded.content_hash,
            updated_at = excluded.updated_at
        """, [.text(record.docID), .int(Int64(record.sectionIndex)), .text(record.heading),
              .text(record.title), .int(Int64(record.level)), .text(record.summary),
              .text(record.contentHash), .double(record.updatedAt)])
    }

    public func chapterSummaries(docID: String) throws -> [ChapterSummaryRecord] {
        let rows = try db.query("""
        SELECT doc_id, section_index, heading, title, level, summary, content_hash, updated_at
        FROM \(Self.chapterSummariesTable) WHERE doc_id = ? ORDER BY section_index
        """, [.text(docID)])
        return rows.map(Self.chapterSummaryRecord(from:))
    }

    public func chapterSummary(docID: String, sectionIndex: Int) throws -> ChapterSummaryRecord? {
        let rows = try db.query("""
        SELECT doc_id, section_index, heading, title, level, summary, content_hash, updated_at
        FROM \(Self.chapterSummariesTable) WHERE doc_id = ? AND section_index = ? LIMIT 1
        """, [.text(docID), .int(Int64(sectionIndex))])
        return rows.first.map(Self.chapterSummaryRecord(from:))
    }

    public func deleteChapterSummary(docID: String, sectionIndex: Int) throws {
        try db.execute(
            "DELETE FROM \(Self.chapterSummariesTable) WHERE doc_id = ? AND section_index = ?",
            [.text(docID), .int(Int64(sectionIndex))])
    }

    /// 目录收缩后清理越界章节：摘要与分段一起删除，避免进度被旧数据抬高。
    public func deleteChapterSummaries(docID: String, fromSectionIndex: Int) throws {
        guard !docID.isEmpty else { return }
        try db.transaction {
            try db.execute("""
            DELETE FROM \(Self.chapterSummariesTable)
            WHERE doc_id = ? AND section_index >= ?
            """, [.text(docID), .int(Int64(max(0, fromSectionIndex)))])
            try db.execute("""
            DELETE FROM \(Self.chapterSegmentsTable)
            WHERE doc_id = ? AND section_index >= ?
            """, [.text(docID), .int(Int64(max(0, fromSectionIndex)))])
        }
    }

    // MARK: - Chapter segments

    public func upsertChapterSegment(_ record: ChapterSegmentRecord) throws {
        try db.execute("""
        INSERT INTO \(Self.chapterSegmentsTable)
            (doc_id, section_index, segment_index, segment_hash, summary, updated_at)
        VALUES(?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id, section_index, segment_index) DO UPDATE SET
            segment_hash = excluded.segment_hash,
            summary = excluded.summary,
            updated_at = excluded.updated_at
        """, [.text(record.docID), .int(Int64(record.sectionIndex)),
              .int(Int64(record.segmentIndex)),
              .text(record.segmentHash), .text(record.summary), .double(record.updatedAt)])
    }

    public func chapterSegments(docID: String, sectionIndex: Int) throws -> [ChapterSegmentRecord] {
        let rows = try db.query("""
        SELECT doc_id, section_index, segment_index, segment_hash, summary, updated_at
        FROM \(Self.chapterSegmentsTable)
        WHERE doc_id = ? AND section_index = ? ORDER BY segment_index
        """, [.text(docID), .int(Int64(sectionIndex))])
        return rows.map {
            ChapterSegmentRecord(
                docID: $0["doc_id"]?.stringValue ?? "",
                sectionIndex: Int($0["section_index"]?.intValue ?? 0),
                segmentIndex: Int($0["segment_index"]?.intValue ?? 0),
                segmentHash: $0["segment_hash"]?.stringValue ?? "",
                summary: $0["summary"]?.stringValue ?? "",
                updatedAt: $0["updated_at"]?.doubleValue ?? 0)
        }
    }

    public func deleteChapterSegments(docID: String, sectionIndex: Int) throws {
        try db.execute("""
        DELETE FROM \(Self.chapterSegmentsTable) WHERE doc_id = ? AND section_index = ?
        """, [.text(docID), .int(Int64(sectionIndex))])
    }

    // MARK: - Book summaries

    public func upsertBookSummary(_ record: BookSummaryRecord) throws {
        try db.execute("""
        INSERT INTO \(Self.bookSummariesTable)
            (doc_id, summary, outline, chapter_count, source_hash, updated_at)
        VALUES(?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            summary = excluded.summary,
            outline = excluded.outline,
            chapter_count = excluded.chapter_count,
            source_hash = excluded.source_hash,
            updated_at = excluded.updated_at
        """, [.text(record.docID), .text(record.summary), .text(record.outline),
              .int(Int64(record.chapterCount)), .text(record.sourceHash),
              .double(record.updatedAt)])
    }

    public func bookSummary(docID: String) throws -> BookSummaryRecord? {
        let rows = try db.query("""
        SELECT doc_id, summary, outline, chapter_count, source_hash, updated_at
        FROM \(Self.bookSummariesTable) WHERE doc_id = ? LIMIT 1
        """, [.text(docID)])
        return rows.first.map {
            BookSummaryRecord(
                docID: $0["doc_id"]?.stringValue ?? "",
                summary: $0["summary"]?.stringValue ?? "",
                outline: $0["outline"]?.stringValue ?? "",
                chapterCount: Int($0["chapter_count"]?.intValue ?? 0),
                sourceHash: $0["source_hash"]?.stringValue ?? "",
                updatedAt: $0["updated_at"]?.doubleValue ?? 0)
        }
    }

    public func upsertBookSummarySegment(_ record: BookSummarySegmentRecord) throws {
        try db.execute("""
        INSERT INTO \(Self.bookSegmentsTable)
            (doc_id, group_index, group_hash, summary, updated_at)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(doc_id, group_index) DO UPDATE SET
            group_hash = excluded.group_hash,
            summary = excluded.summary,
            updated_at = excluded.updated_at
        """, [.text(record.docID), .int(Int64(record.groupIndex)), .text(record.groupHash),
              .text(record.summary), .double(record.updatedAt)])
    }

    public func bookSummarySegments(docID: String) throws -> [BookSummarySegmentRecord] {
        let rows = try db.query("""
        SELECT doc_id, group_index, group_hash, summary, updated_at
        FROM \(Self.bookSegmentsTable) WHERE doc_id = ? ORDER BY group_index
        """, [.text(docID)])
        return rows.map {
            BookSummarySegmentRecord(
                docID: $0["doc_id"]?.stringValue ?? "",
                groupIndex: Int($0["group_index"]?.intValue ?? 0),
                groupHash: $0["group_hash"]?.stringValue ?? "",
                summary: $0["summary"]?.stringValue ?? "",
                updatedAt: $0["updated_at"]?.doubleValue ?? 0)
        }
    }

    public func deleteBookSummarySegments(docID: String) throws {
        try db.execute("DELETE FROM \(Self.bookSegmentsTable) WHERE doc_id = ?", [.text(docID)])
    }

    // MARK: - Summary jobs

    public func upsertSummaryJob(_ record: SummaryJobRecord) throws {
        try db.execute("""
        INSERT INTO \(Self.summaryJobTable)
            (doc_id, status, next_section, total_sections, last_error, updated_at)
        VALUES(?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            status = excluded.status,
            next_section = excluded.next_section,
            total_sections = excluded.total_sections,
            last_error = excluded.last_error,
            updated_at = excluded.updated_at
        """, [.text(record.docID), .text(record.status.rawValue),
              .int(Int64(record.nextSection)), .int(Int64(record.totalSections)),
              .text(record.lastError), .double(record.updatedAt)])
    }

    public func summaryJob(docID: String) throws -> SummaryJobRecord? {
        let rows = try db.query("""
        SELECT doc_id, status, next_section, total_sections, last_error, updated_at
        FROM \(Self.summaryJobTable) WHERE doc_id = ? LIMIT 1
        """, [.text(docID)])
        return rows.first.map {
            SummaryJobRecord(
                docID: $0["doc_id"]?.stringValue ?? "",
                status: BookSummaryJobStatus(rawValue: $0["status"]?.stringValue ?? "") ?? .pending,
                nextSection: Int($0["next_section"]?.intValue ?? 0),
                totalSections: Int($0["total_sections"]?.intValue ?? 0),
                lastError: $0["last_error"]?.stringValue ?? "",
                updatedAt: $0["updated_at"]?.doubleValue ?? 0)
        }
    }

    /// 候选书：未开始、章节未完成、全书摘要待生成或上次失败。
    /// 已完成且有全书摘要的书不再进入队列。
    public func summaryCandidates(limit: Int) throws -> [DocumentRecord] {
        guard limit > 0 else { return [] }
        let rows = try db.query("""
        SELECT d.doc_id, d.doc_name, d.digest, d.chapters, d.md_path, d.content_hash
        FROM \(Self.documentsTable) d
        LEFT JOIN \(Self.summaryJobTable) j ON j.doc_id = d.doc_id
        LEFT JOIN \(Self.bookSummariesTable) b ON b.doc_id = d.doc_id
        WHERE d.md_path != '' AND (
            j.doc_id IS NULL
            OR j.status IN ('pending', 'summarizing_book', 'failed')
            -- 完成但缺少全书摘要：包括旧版无标题纯正文直接写成的
            -- completed / total_sections=0，现在由引擎合成「全文」章节补出摘要。
            OR (j.status = 'completed' AND b.doc_id IS NULL)
        )
        ORDER BY COALESCE(j.updated_at, 0), d.updated_at
        LIMIT ?
        """, [.int(Int64(limit))])
        return rows.map { Self.documentRecord(from: $0) }
    }

    public func summaryProgress(docID: String) throws -> DocumentSummaryProgress {
        let job = try summaryJob(docID: docID)
        let total = Int(try db.scalar("""
        SELECT COUNT(*) FROM \(Self.chapterSummariesTable)
        WHERE doc_id = ? AND summary != ''
        """, [.text(docID)]).intValue ?? 0)
        // 任务记录了当前目录的实际章节总数；不能用摘要行数反推，
        // 否则目录收缩后遗留的旧行会把进度抬高到 total 之外。
        let chapterCount = job?.totalSections ?? total
        let completed = Int(try db.scalar("""
        SELECT COUNT(*) FROM \(Self.chapterSummariesTable)
        WHERE doc_id = ? AND summary != '' AND section_index < ?
        """, [.text(docID), .int(Int64(max(0, chapterCount)))]).intValue ?? 0)
        let book = try bookSummary(docID: docID)
        return DocumentSummaryProgress(
            docID: docID,
            chapterCount: max(chapterCount, 0),
            completedChapters: max(completed, 0),
            hasBookSummary: !(book?.summary.isEmpty ?? true),
            jobStatus: job?.status)
    }

    /// 上次实际处理时间：跨重启仍遵守 15/5 分钟节流。
    public func summaryLastTickAt() throws -> Double? {
        let value = try db.scalar(
            "SELECT value FROM \(Self.summaryRuntimeTable) WHERE key = 'last_tick_at'").stringValue
        return value.flatMap(Double.init)
    }

    public func setSummaryLastTickAt(_ value: Double) throws {
        try db.execute("""
        INSERT INTO \(Self.summaryRuntimeTable)(key, value) VALUES('last_tick_at', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, [.text(String(value))])
    }

    public func deleteSummaries(docID: String) throws {
        guard !docID.isEmpty else { return }
        try db.transaction {
            try deleteSummaryRowsLocked(docID: docID)
        }
    }

    private func deleteSummaryRowsLocked(docID: String) throws {
        guard !docID.isEmpty else { return }
        try db.execute("DELETE FROM \(Self.chapterSummariesTable) WHERE doc_id = ?", [.text(docID)])
        try db.execute("DELETE FROM \(Self.chapterSegmentsTable) WHERE doc_id = ?", [.text(docID)])
        try db.execute("DELETE FROM \(Self.bookSummariesTable) WHERE doc_id = ?", [.text(docID)])
        try db.execute("DELETE FROM \(Self.bookSegmentsTable) WHERE doc_id = ?", [.text(docID)])
        try db.execute("DELETE FROM \(Self.summaryJobTable) WHERE doc_id = ?", [.text(docID)])
    }

    // MARK: - Helpers

    private static func documentRecord(from row: [String: SQLValue]) -> DocumentRecord {
        DocumentRecord(
            docID: row["doc_id"]?.stringValue ?? "",
            docName: row["doc_name"]?.stringValue ?? "",
            digest: row["digest"]?.stringValue ?? "",
            chapters: row["chapters"]?.stringValue ?? "",
            mdPath: row["md_path"]?.stringValue ?? "",
            contentHash: row["content_hash"]?.stringValue ?? "")
    }

    private static func chapterSummaryRecord(from row: [String: SQLValue]) -> ChapterSummaryRecord {
        ChapterSummaryRecord(
            docID: row["doc_id"]?.stringValue ?? "",
            sectionIndex: Int(row["section_index"]?.intValue ?? 0),
            heading: row["heading"]?.stringValue ?? "",
            title: row["title"]?.stringValue ?? "",
            level: Int(row["level"]?.intValue ?? 1),
            summary: row["summary"]?.stringValue ?? "",
            contentHash: row["content_hash"]?.stringValue ?? "",
            updatedAt: row["updated_at"]?.doubleValue ?? 0)
    }

    /// 旧库增量补列。SQLite 没有 `ADD COLUMN IF NOT EXISTS`，先查 table_info。
    private func ensureColumn(table: String, column: String, definition: String) throws {
        let rows = try db.query("PRAGMA table_info(\(table))")
        guard !rows.contains(where: { $0["name"]?.stringValue == column }) else { return }
        try db.execute("ALTER TABLE \(table) ADD COLUMN \(column) \(definition)")
    }

    /// 原项目使用 md5(source)[:12] 作为稳定短哈希，这里保持一致。
    public static func hashSource(_ source: String) -> String {
        let digest = Insecure.MD5.hash(data: Data(source.utf8))
        return digest.prefix(6).map { String(format: "%02x", $0) }.joined()
    }
}
