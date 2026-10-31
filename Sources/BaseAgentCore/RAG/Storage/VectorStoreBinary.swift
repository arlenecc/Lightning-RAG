import Foundation

/// bin 表对外的只读状态快照。不触发任何查询，全部来自 O(1) 计数器。
public struct BinaryIndexStatus: Sendable {
    /// sqlite-vec + bin 表存在 + N > 0。
    public var storageEnabled: Bool
    public var dimension: Int
    public var rows: Int
    public var chunkRows: Int
    public var coverage: Double
    public var isComplete: Bool
    public var reason: String?

    public init(storageEnabled: Bool = false,
                dimension: Int = 0,
                rows: Int = 0,
                chunkRows: Int = 0,
                coverage: Double = 0,
                isComplete: Bool = false,
                reason: String? = nil) {
        self.storageEnabled = storageEnabled
        self.dimension = dimension
        self.rows = rows
        self.chunkRows = chunkRows
        self.coverage = coverage
        self.isComplete = isComplete
        self.reason = reason
    }
}

/// 二值粗筛的查询参数；存储可用性与覆盖率由 `VectorStore` 判定。
public struct BinaryCoarseOptions: Sendable {
    public var signature: Data
    public var candidateLimit: Int

    public init(signature: Data, candidateLimit: Int) {
        self.signature = signature
        self.candidateLimit = max(1, candidateLimit)
    }
}

extension VectorStore {
    enum BinaryCoarseQueryOutcome {
        case used(candidates: [(Int64, Double)], scores: [Int64: Double])
        case skipped(String)
    }

    /// stage 1 二值 KNN + stage 2 主向量余弦精排。不可用时返回跳过原因，
    /// 由 `hybridSearch` 回退原向量通道。调用方需已持有 `lock`。
    func binaryCoarseCandidatesLocked(
        vector: [Float], signature: Data, candidateLimit: Int,
        candidateCount: Int, targeted: Bool
    ) throws -> BinaryCoarseQueryOutcome {
        guard !targeted else { return .skipped("定向检索不参与二值粗筛") }
        guard rebuildReason == nil else { return .skipped("向量库待重建") }
        guard elementType != .bin else {
            return .skipped("bin 主存储仅支持单阶段检索")
        }
        guard binaryStorageEnabled else {
            return .skipped(binaryStorageReason ?? "bin 表不可用")
        }
        let expectedBytes = binaryDimension / 8
        guard signature.count == expectedBytes else {
            return .skipped("查询签名长度 \(signature.count) != \(expectedBytes)")
        }
        let chunkRows = count
        guard chunkRows > 0 else { return .skipped("知识库为空") }
        guard binaryRows == chunkRows else {
            return .skipped("二值签名覆盖 \(binaryRows)/\(chunkRows)，请重新同步")
        }

        let k = min(maxVectorK, max(candidateLimit, candidateCount))
        let rows = try db.query(
            "SELECT rowid, distance FROM \(Self.binaryVectorTable) "
            + "WHERE embedding MATCH vec_bit(?) AND k = ?",
            [.blob(signature), .int(Int64(k))])
        let hits = rows.compactMap { row -> (Int64, Double)? in
            guard let rowID = row["rowid"]?.intValue else { return nil }
            return (rowID, row["distance"]?.doubleValue ?? Double(binaryDimension))
        }
        guard !hits.isEmpty else { return .used(candidates: [], scores: [:]) }

        let storedVectors = try exactVectorsLocked(rowIDs: hits.map(\.0))
        var reranked: [(rowID: Int64, cosine: Double, binary: Double)] = []
        reranked.reserveCapacity(hits.count)
        for (rowID, distance) in hits {
            guard let stored = storedVectors[rowID], stored.count == vector.count else {
                continue
            }
            let cosine = VectorCodec.cosine(vector, stored)
            let binaryScore = max(0, 1 - distance / Double(binaryDimension))
            reranked.append((rowID, cosine, binaryScore))
        }
        reranked.sort { $0.cosine > $1.cosine }
        let selected = Array(reranked.prefix(candidateCount))
        return .used(
            candidates: selected.map { ($0.rowID, $0.cosine) },
            scores: Dictionary(uniqueKeysWithValues: selected.map { ($0.rowID, $0.binary) }))
    }

    // MARK: - 打开库时对齐

    /// 按当前参数对齐 bin 表与 meta。派生数据不一致时直接丢弃重建。
    /// 调用方：`VectorStore.init`（未持锁，单线程）。
    func setupBinaryIndexLocked() throws {
        binaryRows = 0
        binaryStorageEnabled = false
        binaryStorageReason = nil

        guard binaryDimension > 0 else {
            binaryStorageReason = "embedding 不支持二值签名"
            return
        }
        guard elementType != .bin else {
            binaryStorageReason = "bin 主存储仅支持单阶段检索"
            return
        }
        guard sqliteVecAvailable else {
            binaryStorageReason = "sqlite-vec 不可用"
            return
        }
        guard binaryDimension % 8 == 0 else {
            binaryStorageReason = "二值签名维度必须是 8 的倍数"
            return
        }
        guard !binarySource.isEmpty else {
            binaryStorageReason = "缺少二值签名来源"
            return
        }

        let existingSQL = try db.scalar(
            "SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
            [.text(Self.binaryVectorTable)]).stringValue
        let storedDimension = try db.scalar(
            "SELECT value FROM meta WHERE key = ?",
            [.text(Self.binaryDimensionKey)]).stringValue
        let storedSource = try db.scalar(
            "SELECT value FROM meta WHERE key = ?",
            [.text(Self.binarySourceKey)]).stringValue
        let storedVersion = try db.scalar(
            "SELECT value FROM meta WHERE key = ?",
            [.text(Self.binaryQuantizerVersionKey)]).stringValue
        let storedRows = Self.readStoredCount(db, key: Self.binaryRowsKey)

        let schemaMatches = existingSQL?.contains("USING vec0") == true
            && existingSQL?.contains("bit[\(binaryDimension)]") == true
        if existingSQL != nil, schemaMatches,
           storedDimension == String(binaryDimension),
           storedSource == binarySource,
           storedVersion == String(Self.binaryQuantizerVersion),
           let storedRows {
            binaryRows = max(0, storedRows)
            binaryStorageEnabled = true
            return
        }

        // 表缺失、表结构不符或 meta 任一不符：直接丢弃重建空表。
        try db.transaction {
            try db.execute("DROP TABLE IF EXISTS \(Self.binaryVectorTable)")
            try deleteBinaryMetaLocked()
            // bin 表重建后旧签名 rowid 全部失效，二值 ANN 一起丢弃。
            try dropBinaryANNIndexLocked()
            try db.execute("""
            CREATE VIRTUAL TABLE \(Self.binaryVectorTable)
            USING vec0(embedding bit[\(binaryDimension)])
            """)
            binaryRows = 0
            try writeBinaryMetaLocked()
        }
        binaryStorageEnabled = true
    }

    // MARK: - Meta

    /// 为指定 rowid 写入一条已校验长度的二值签名。
    func insertBinarySignature(rowID: Int64, signature: Data) throws {
        try db.execute(
            "INSERT INTO \(Self.binaryVectorTable)(rowid, embedding) "
            + "VALUES(?, vec_bit(?))",
            [.int(rowID), .blob(signature)])
    }

    /// 增量持久化 bin 行数；必须与所在事务一起提交/回滚。
    func persistBinaryCounterLocked() throws {
        guard binaryStorageEnabled else { return }
        try db.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            + "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [.text(Self.binaryRowsKey), .text(String(binaryRows))])
    }

    func writeBinaryMetaLocked() throws {
        let values: [(String, String)] = [
            (Self.binaryDimensionKey, String(binaryDimension)),
            (Self.binarySourceKey, binarySource),
            (Self.binaryQuantizerVersionKey, String(Self.binaryQuantizerVersion)),
            (Self.binaryRowsKey, String(binaryRows)),
        ]
        for (key, value) in values {
            try db.execute("""
            INSERT INTO meta(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, [.text(key), .text(value)])
        }
    }

    func deleteBinaryMetaLocked() throws {
        try db.execute("DELETE FROM meta WHERE key IN (?, ?, ?, ?)", [
            .text(Self.binaryRowsKey),
            .text(Self.binaryDimensionKey),
            .text(Self.binarySourceKey),
            .text(Self.binaryQuantizerVersionKey),
        ])
    }

    // MARK: - 状态

    /// 只读快照；`rows` / `chunkRows` 均来自增量计数器。
    public func binaryIndexStatus() -> BinaryIndexStatus {
        lock.lock(); defer { lock.unlock() }
        return binaryIndexStatusLocked()
    }

    func binaryIndexStatusLocked() -> BinaryIndexStatus {
        let chunkRows = count
        let coverage = chunkRows > 0
            ? min(1.0, Double(binaryRows) / Double(chunkRows))
            : 0
        return BinaryIndexStatus(
            storageEnabled: binaryStorageEnabled,
            dimension: binaryDimension,
            rows: binaryRows,
            chunkRows: chunkRows,
            coverage: coverage,
            isComplete: binaryStorageEnabled && chunkRows > 0 && binaryRows == chunkRows,
            reason: binaryStorageReason)
    }

    /// 同一原因 60 秒内只记一条，避免每次查询刷日志。
    func noteBinaryFallbackLocked(_ reason: String) {
        let now = Date()
        if reason == lastBinaryFallbackReason,
           now.timeIntervalSince(lastBinaryFallbackLog) < 60 {
            return
        }
        lastBinaryFallbackReason = reason
        lastBinaryFallbackLog = now
        AgentLog.info("二值粗筛回退：\(reason)")
    }
}
