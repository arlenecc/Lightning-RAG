import Foundation

/// 二值 ANN 索引元数据（存 meta 表，键前缀 `bann_`）。
struct BANNStoredMeta {
    var version = 0
    var clusters = 0
    var blockRows = 0
    var dim = 0
    var source = ""
    var blocks = 0
    var rows = 0
    var staleRows = 0
    var lastRowID: Int64 = 0
    var builtAt: Double = 0
}

/// 基于二值签名的分块 IVF 索引。
///
/// 复用主 ANN 的 bits 块布局、k-means、打包与汉明打分，但使用独立的
/// `bann_*` 表与冷却水位：bann 只服务于二值粗筛，构建参数（签名位宽、
/// 来源、块大小）与主 ANN 无关，任何不一致都直接丢弃重建。
extension VectorStore {
    static let bannFormatVersion = 2
    static let bannCellsTable = "bann_cells"
    static let bannBlocksTable = "bann_blocks"
    private var bannMetaPrefix: String { "bann_" }

    // MARK: - Schema / meta

    func ensureBANNSchemaLocked() throws {
        try db.execScript("""
        CREATE TABLE IF NOT EXISTS \(Self.bannBlocksTable) (
            block_id INTEGER PRIMARY KEY,
            clusters INTEGER NOT NULL,
            dim INTEGER NOT NULL,
            payload BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS \(Self.bannCellsTable) (
            block_id INTEGER NOT NULL,
            cluster_id INTEGER NOT NULL,
            seq INTEGER NOT NULL,
            count INTEGER NOT NULL,
            payload BLOB NOT NULL,
            PRIMARY KEY (block_id, cluster_id, seq)
        ) WITHOUT ROWID;
        """)
    }

    func bannStoredMetaLocked() throws -> BANNStoredMeta {
        var meta = BANNStoredMeta()
        let rows = try db.query("SELECT key, value FROM meta WHERE key GLOB '\(bannMetaPrefix)*'")
        var values: [String: String] = [:]
        for row in rows {
            if let key = row["key"]?.stringValue, let value = row["value"]?.stringValue {
                values[key] = value
            }
        }
        meta.version = Int(values["bann_ver"] ?? "") ?? 0
        meta.clusters = Int(values["bann_clusters"] ?? "") ?? 0
        meta.blockRows = Int(values["bann_block_rows"] ?? "") ?? 0
        meta.dim = Int(values["bann_dim"] ?? "") ?? 0
        meta.source = values["bann_source"] ?? ""
        meta.blocks = Int(values["bann_blocks"] ?? "") ?? 0
        meta.rows = Int(values["bann_rows"] ?? "") ?? 0
        meta.staleRows = Int(values["bann_stale_rows"] ?? "") ?? 0
        meta.lastRowID = Int64(values["bann_last_rowid"] ?? "") ?? 0
        meta.builtAt = Double(values["bann_built_at"] ?? "") ?? 0
        return meta
    }

    /// 写索引 meta。`meta.rows` 是**本块新增行数**：`bann_rows` 在已有值上
    /// 累加（块数、水位、簇数等其余键直接覆盖），陈旧计数沿用已有值。
    func writeBANNMetaLocked(_ meta: BANNStoredMeta) throws {
        let stored = try bannStoredMetaLocked()
        let values: [(String, String)] = [
            ("bann_ver", String(meta.version)),
            ("bann_clusters", String(meta.clusters)),
            ("bann_block_rows", String(meta.blockRows)),
            ("bann_dim", String(meta.dim)),
            ("bann_source", meta.source),
            ("bann_blocks", String(meta.blocks)),
            ("bann_rows", String(stored.rows + meta.rows)),
            ("bann_stale_rows", String(stored.staleRows)),
            ("bann_last_rowid", String(meta.lastRowID)),
            ("bann_built_at", String(meta.builtAt)),
        ]
        for (key, value) in values {
            try db.execute("""
            INSERT INTO meta(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, [.text(key), .text(value)])
        }
    }

    /// 清空索引内容、保留表结构（清空知识库 / 维度变化时调用）。
    func resetBANNIndexLocked() throws {
        try db.execute("DELETE FROM \(Self.bannCellsTable)")
        try db.execute("DELETE FROM \(Self.bannBlocksTable)")
        try db.execute("DELETE FROM meta WHERE key GLOB '\(bannMetaPrefix)*'")
        bannCentroidCache = nil
    }

    /// 丢弃索引表与元数据（释放占用，随时可重建）。
    func dropBinaryANNIndexLocked() throws {
        try db.execute("DROP TABLE IF EXISTS \(Self.bannCellsTable)")
        try db.execute("DROP TABLE IF EXISTS \(Self.bannBlocksTable)")
        try db.execute("DELETE FROM meta WHERE key GLOB '\(bannMetaPrefix)*'")
        bannCentroidCache = nil
    }

    public func dropBinaryANNIndex() throws {
        lock.lock(); defer { lock.unlock() }
        try db.transaction { try dropBinaryANNIndexLocked() }
    }

    /// 删除/更新切片后累计陈旧行数：只有落在索引水位之前的 rowid 才在块内。
    /// 调用方需已在同一个写事务里（删除与计数一起提交）。
    func recordBANNStaleRowsLocked(_ rowIDs: [Int64]) throws {
        guard !rowIDs.isEmpty,
              let meta = try? bannStoredMetaLocked(), meta.blocks > 0,
              meta.lastRowID > 0 else { return }
        let stale = rowIDs.reduce(0) { $0 + ($1 <= meta.lastRowID ? 1 : 0) }
        guard stale > 0 else { return }
        let updated = min(meta.rows, max(0, meta.staleRows) + stale)
        guard updated != meta.staleRows else { return }
        try db.execute("""
        INSERT INTO meta(key, value) VALUES('bann_stale_rows', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, [.text(String(updated))])
    }

    // MARK: - 状态

    /// 尾块精确补齐的行数上限（按 128MB / 签名字节数折算）。
    func bannTailScanRowCap() -> Int {
        guard binaryDimension > 0 else { return 0 }
        let rowBytes = VectorElementType.bin.blobSize(dimension: binaryDimension)
        guard rowBytes > 0 else { return 0 }
        return max(1024, Self.annTailScanByteBudget / rowBytes)
    }

    /// 尾块大小：签名水位之后的 bin 行数。
    /// 计数器在「按来源重同步」时会低估，因此再用 `rowid > watermark` 的
    /// 实际行数兜底；统计封顶在 cap + 1，避免大库上扫全表。
    func bannTailRowCountLocked(after rowID: Int64, indexedRows: Int) -> Int {
        let byCount = max(0, binaryRows - indexedRows)
        let cap = bannTailScanRowCap()
        let counted = (try? db.scalar("""
        SELECT COUNT(*) FROM (
            SELECT 1 FROM \(Self.binaryVectorTable) WHERE rowid > ? LIMIT ?
        )
        """, [.int(rowID), .int(Int64(cap + 1))]).intValue).flatMap { Int($0) } ?? 0
        return max(byCount, counted)
    }

    /// 建议重建的尾块规模：超过四分之一预算时提示（读取已明显变慢）。
    func bannRebuildTailThreshold() -> Int {
        max(8_192, bannTailScanRowCap() / 4)
    }

    public func binaryANNIndexStatus() -> ANNIndexStatus {
        lock.lock(); defer { lock.unlock() }
        return binaryANNIndexStatusLocked()
    }

    func binaryANNIndexStatusLocked() -> ANNIndexStatus {
        var status = ANNIndexStatus()
        guard let meta = try? bannStoredMetaLocked(), meta.blocks > 0 else {
            status.summary = "二值 ANN 索引未构建"
            return status
        }
        let total = count
        let tail = bannTailRowCountLocked(after: meta.lastRowID, indexedRows: meta.rows)
        let cap = bannTailScanRowCap()
        let unindexed = max(0, tail - cap)
        let dimOK = meta.dim > 0 && meta.dim == binaryDimension
        let sourceOK = !binarySource.isEmpty && meta.source == binarySource
        let versionOK = meta.version == Self.bannFormatVersion
        let stale = min(meta.rows, max(0, meta.staleRows))
        status.clusters = meta.clusters
        status.blockRows = meta.blockRows
        status.blocks = meta.blocks
        status.indexedRows = meta.rows
        status.tailRows = tail
        status.unindexedRows = unindexed
        status.staleRows = stale
        status.dimension = meta.dim
        status.builtAt = meta.builtAt
        status.isReady = dimOK && sourceOK && versionOK && tail == 0
        status.isUsable = dimOK && sourceOK && versionOK && unindexed == 0
        status.coverage = total > 0 ? min(1, Double(meta.rows) / Double(total)) : 0
        // 删除后 total 可能小于已索引行数；分母取两者较大值，避免把
        // 「索引里还有陈旧行」误算成 100% 有效覆盖。
        let denominator = max(total, meta.rows)
        status.effectiveCoverage = denominator > 0
            ? min(1, Double(max(0, meta.rows - stale)) / Double(denominator))
            : 0
        status.needsRebuild = status.isUsable
            && (tail > bannRebuildTailThreshold()
                || Double(stale) / Double(max(1, meta.rows)) >= Self.annStaleRebuildRatio)
        if !versionOK {
            status.summary = "二值 ANN 索引格式版本 \(meta.version) 与当前 "
                + "\(Self.bannFormatVersion) 不符，需重建"
            return status
        }
        if !dimOK {
            status.summary = "二值 ANN 索引位宽 \(meta.dim) 与当前 "
                + "\(binaryDimension) 不符，需重建"
            return status
        }
        if !sourceOK {
            status.summary = "二值 ANN 索引签名来源与当前配置不符，需重建"
            return status
        }
        var parts = [
            "二值 ANN 索引：\(status.clusters) 簇/块 × "
                + "\(status.blockRows / max(1, status.clusters)) 行/簇",
            "\(status.blocks) 块",
            "已索引 \(status.indexedRows) 切片",
            "\(status.dimension) bit",
        ]
        if status.tailRows > 0 { parts.append("尾块 \(status.tailRows) 切片（精确补齐）") }
        if status.unindexedRows > 0 { parts.append("尾块超出补齐上限，需重建索引") }
        if status.staleRows > 0 { parts.append("陈旧 \(status.staleRows) 切片") }
        parts.append(String(format: "覆盖 %.1f%%", status.coverage * 100))
        if status.effectiveCoverage < status.coverage - 0.0001, status.staleRows > 0 {
            parts.append(String(format: "实际覆盖 %.1f%%", status.effectiveCoverage * 100))
        }
        if status.needsRebuild { parts.append("建议重建") }
        status.summary = parts.joined(separator: "，")
        return status
    }

    // MARK: - 构建

    func streamBinarySignatures(from rowID: Int64, dim: Int,
                                _ body: (Int64, UnsafeRawPointer) throws -> Bool) throws {
        let rowBytes = VectorElementType.bin.blobSize(dimension: dim)
        guard rowBytes > 0 else { return }
        try db.stream(
            "SELECT rowid, embedding FROM \(Self.binaryVectorTable) WHERE rowid > ?",
            [.int(rowID)]) { row in
                let hitRowID = row.int(0)
                guard let blob = row.blob(1), blob.count >= rowBytes else { return true }
                return try body(hitRowID, blob.pointer)
            }
    }

    func gatherBANNBlockLocked(after rowID: Int64, maxRows: Int,
                               into buffer: inout ANNBlockBuffer) throws {
        buffer.reset(maxRows: maxRows, dim: binaryDimension, elementType: .bin)
        var scratch = [Int8]()
        try streamBinarySignatures(from: rowID, dim: binaryDimension) { hitRowID, pointer in
            buffer.append(rowID: hitRowID, pointer: pointer, dim: binaryDimension,
                          elementType: .bin, scratch: &scratch)
            return buffer.count < maxRows
        }
        if buffer.needsSort { buffer.sortByRowID() }
    }

    func writeBANNBlockLocked(blockID: Int, clusters: Int, blockRows: Int,
                              iterations: Int, buffer: ANNBlockBuffer) throws {
        let rows = buffer.count
        guard rows > 0 else { return }
        let activeClusters = min(clusters, rows)
        var centroids = [Float](repeating: 0, count: clusters * binaryDimension)
        var counts = [Int32](repeating: 0, count: clusters)
        if let seeds = annSeedCentroids(buffer: buffer, dim: binaryDimension,
                                        clusters: activeClusters),
           let trained = ANNClustering.train(
            buffer: buffer, rows: rows, dim: binaryDimension,
            clusters: activeClusters, iterations: iterations, seeds: seeds) {
            for i in 0..<(activeClusters * binaryDimension) {
                centroids[i] = trained.centroids[i]
            }
            for cluster in 0..<activeClusters { counts[cluster] = trained.counts[cluster] }
        }
        let assignments = ANNClustering.assign(
            buffer: buffer, rows: rows, dim: binaryDimension,
            clusters: activeClusters, centroids: centroids)
        var offsets = [Int](repeating: 0, count: activeClusters + 1)
        for assignment in assignments { offsets[Int(assignment) + 1] += 1 }
        for cluster in 0..<activeClusters { offsets[cluster + 1] += offsets[cluster] }
        var order = [Int32](repeating: 0, count: rows)
        var cursor = offsets
        for (index, assignment) in assignments.enumerated() {
            let cluster = Int(assignment)
            order[cursor[cluster]] = Int32(index)
            cursor[cluster] += 1
        }

        try db.transaction {
            for cluster in 0..<activeClusters {
                let start = offsets[cluster]
                let end = offsets[cluster + 1]
                guard end > start else { continue }
                let payload = ANNPacking.packCell(buffer: buffer, order: order,
                                                  range: start..<end,
                                                  dim: binaryDimension)
                try db.execute("""
                INSERT OR REPLACE INTO \(Self.bannCellsTable)
                    (block_id, cluster_id, seq, count, payload)
                VALUES(?, ?, 0, ?, ?)
                """, [.int(Int64(blockID)), .int(Int64(cluster)),
                      .int(Int64(end - start)), .blob(payload)])
            }
            let payload = ANNPacking.packCentroids(
                centroids: centroids, counts: counts,
                clusters: activeClusters, dim: binaryDimension)
            try db.execute("""
            INSERT OR REPLACE INTO \(Self.bannBlocksTable)(block_id, clusters, dim, payload)
            VALUES(?, ?, ?, ?)
            """, [.int(Int64(blockID)), .int(Int64(activeClusters)),
                  .int(Int64(binaryDimension)), .blob(payload)])
            try writeBANNMetaLocked(BANNStoredMeta(
                version: Self.bannFormatVersion, clusters: clusters, blockRows: blockRows,
                dim: binaryDimension, source: binarySource, blocks: blockID + 1,
                rows: rows, lastRowID: buffer.lastRowID,
                builtAt: Date().timeIntervalSince1970))
        }
        bannCentroidCache = nil
    }

    /// 构建（或增量续跑）二值 ANN 索引。
    ///
    /// 必须等 `binaryRows == count`（签名覆盖完整）才能构建；否则缺失的切片
    /// 永远进不了索引，状态会静默失真。已建好的块按 meta 水位复用，取消/
    /// 中断后再次调用会从断点继续。块间释放锁，检索/状态查询不被整体冻结。
    @discardableResult
    public func buildBinaryANNIndex(
        clusters: Int = 256, clusterSize: Int = 512, iterations: Int = 8,
        progress: ((ANNBuildProgress) -> Void)? = nil,
        isCancelled: (() -> Bool)? = nil
    ) throws -> ANNBuildResult {
        let started = Date()
        guard binaryStorageEnabled, binaryDimension > 0 else {
            throw AgentError.storage("bin 表不可用，无法构建二值 ANN")
        }
        guard binaryRows == count, binaryRows > 0 else {
            throw AgentError.storage("二值签名覆盖不完整，请重新同步后再构建二值 ANN")
        }
        let clusterCount = min(max(2, clusters), 4096)
        let rowsPerCluster = min(max(1, clusterSize), 8192)
        let blockRows = clusterCount * rowsPerCluster
        func locked<T>(_ body: () throws -> T) rethrows -> T {
            lock.lock(); defer { lock.unlock() }
            return try body()
        }
        var meta = BANNStoredMeta()
        var stopRowID: Int64 = 0
        try locked {
            try ensureBANNSchemaLocked()
            meta = try bannStoredMetaLocked()
            let paramsChanged = meta.version != Self.bannFormatVersion
                || meta.dim != binaryDimension || meta.source != binarySource
                || meta.clusters != clusterCount || meta.blockRows != blockRows
            let staleTooHigh = meta.blocks > 0 && meta.rows > 0
                && Double(meta.staleRows) / Double(meta.rows) >= Self.annStaleRebuildRatio
            if paramsChanged || staleTooHigh {
                if meta.blocks > 0 {
                    AgentLog.info("二值 ANN: 参数/陈旧比例触发整体重建")
                }
                try resetBANNIndexLocked()
                meta = BANNStoredMeta()
            }
            stopRowID = (try? db.scalar(
                "SELECT COALESCE(MAX(rowid), 0) FROM \(Self.binaryVectorTable)").intValue)
                .flatMap { Int64($0) } ?? 0
        }
        var result = ANNBuildResult()
        result.clusters = clusterCount
        result.blockRows = blockRows
        var blockID = meta.blocks
        var indexedRows = meta.rows
        var watermark = meta.lastRowID
        let total = binaryRows
        let remaining = max(0, total - indexedRows)
        let estimatedBlocks = max(1, (remaining + blockRows - 1) / blockRows)
        if indexedRows > 0 {
            AgentLog.info("二值 ANN: 从断点续建（已有 \(blockID) 块 / \(indexedRows) 切片）")
        }
        progress?(ANNBuildProgress(
            stage: "start", blocks: blockID, totalBlocks: estimatedBlocks,
            rows: indexedRows, totalRows: max(total, indexedRows),
            message: "开始构建二值 ANN 索引"
                + "（\(clusterCount) 簇/块 × \(rowsPerCluster) 行/簇）"))

        while true {
            let step = try locked { () -> (finished: Bool, cancelled: Bool,
                                           update: ANNBuildProgress?) in
                if let isCancelled, isCancelled() { return (true, true, nil) }
                if watermark >= stopRowID { return (true, false, nil) }
                var buffer = ANNBlockBuffer()
                try gatherBANNBlockLocked(after: watermark, maxRows: blockRows, into: &buffer)
                guard buffer.count > 0 else { return (true, false, nil) }
                try writeBANNBlockLocked(blockID: blockID, clusters: clusterCount,
                                         blockRows: blockRows, iterations: iterations,
                                         buffer: buffer)
                blockID += 1
                indexedRows += buffer.count
                watermark = buffer.lastRowID
                return (false, false, ANNBuildProgress(
                    stage: "blocks", blocks: blockID,
                    totalBlocks: max(estimatedBlocks, blockID),
                    rows: indexedRows, totalRows: max(total, indexedRows),
                    message: "已写入 \(blockID) 块二值 ANN（\(indexedRows) 切片）"))
            }
            if let update = step.update { progress?(update) }
            if step.cancelled { result.cancelled = true; break }
            if step.finished { break }
        }

        locked {
            result.blocks = blockID
            result.indexedRows = indexedRows
            result.tailRows = max(0, count - indexedRows)
        }
        result.elapsed = Date().timeIntervalSince(started)
        var summary = "二值 ANN 索引：\(blockID) 块 / \(indexedRows) 切片，耗时 "
            + String(format: "%.1fs", result.elapsed)
        if result.tailRows > 0 { summary += "，尾块 \(result.tailRows) 切片" }
        if result.cancelled { summary += "（已取消，下次可续建）" }
        result.summary = summary
        AgentLog.info(summary)
        return result
    }
}
