import Foundation
import Accelerate

/// 分块 IVF 向量索引：移植自主工程 VectorStoreANN（块内 k-means + int8 量化 +
/// 聚簇落盘 + 水位尾扫 + nprobe/rescore）。仅索引 doc_summary_vec 的 64 维中心向量。
final class ANNIndex {
    static let version = 1
    static let clusters = 512
    static let clusterSize = 512
    static let iterations = 8
    static let defaultNprobe = 32
    static let defaultRescore = 64
    static let minRowsForANN = 20_000
    static let tailByteBudget = 128 * 1024 * 1024
    static let staleRebuildRatio = 0.15

    private let db: SQLiteDB
    private let dim = ColBERT.dimension
    /// 簇心缓存可被并发的检索线程读、后台构建线程清空。用独立锁保护属性访问，
    /// 且绝不持锁做 db.query / db.transaction，避免与 DB 锁形成 ABBA 死锁。
    private let cacheLock = NSLock()
    private var centroidCache: ANNCentroidCache?

    init(db: SQLiteDB) { self.db = db }

    /// 锁内读取缓存；仅当簇心布局与当前 meta 一致才复用。
    private func cachedCentroids(matching m: Meta) -> ANNCentroidCache? {
        cacheLock.lock(); defer { cacheLock.unlock() }
        guard let cache = centroidCache,
              cache.blocks == m.blocks, cache.clusters == m.clusters,
              cache.dim == m.dim else { return nil }
        return cache
    }

    /// 线程安全地写入或清空缓存（不持锁做任何 db 操作）。
    private func storeCentroids(_ cache: ANNCentroidCache?) {
        cacheLock.lock()
        centroidCache = cache
        cacheLock.unlock()
    }

    struct Status {
        var blocks = 0
        var indexedRows = 0
        var tailRows = 0
        var staleRows = 0
        var totalRows = 0
        var clusters = 0
        var usable = false
        var needsRebuild = false
        var summary = "ANN 索引未构建"
    }

    struct Candidate {
        let rowID: Int64
        let score: Double
    }

    // MARK: - schema / meta

    func createSchema() throws {
        try db.exec("""
        CREATE TABLE IF NOT EXISTS doc_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ann_blocks (
            block_id INTEGER PRIMARY KEY,
            clusters INTEGER NOT NULL,
            dim INTEGER NOT NULL,
            payload BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ann_cells (
            block_id INTEGER NOT NULL,
            cluster_id INTEGER NOT NULL,
            seq INTEGER NOT NULL,
            count INTEGER NOT NULL,
            payload BLOB NOT NULL,
            PRIMARY KEY (block_id, cluster_id, seq)
        ) WITHOUT ROWID;
        """)
    }

    private struct Meta {
        var version = 0
        var clusters = 0
        var blockRows = 0
        var dim = 0
        var blocks = 0
        var rows = 0
        var staleRows = 0
        var lastRowID: Int64 = 0
        var builtAt = 0.0
    }

    private func meta() -> Meta {
        guard let rows = try? db.query("SELECT key, value FROM doc_meta WHERE key GLOB 'ann_*'") else {
            return Meta()
        }
        var values: [String: String] = [:]
        for row in rows {
            if let k = row["key"]?.stringValue, let v = row["value"]?.stringValue { values[k] = v }
        }
        var m = Meta()
        m.version = Int(values["ann_ver"] ?? "") ?? 0
        m.clusters = Int(values["ann_clusters"] ?? "") ?? 0
        m.blockRows = Int(values["ann_block_rows"] ?? "") ?? 0
        m.dim = Int(values["ann_dim"] ?? "") ?? 0
        m.blocks = Int(values["ann_blocks"] ?? "") ?? 0
        m.rows = Int(values["ann_rows"] ?? "") ?? 0
        m.staleRows = Int(values["ann_stale_rows"] ?? "") ?? 0
        m.lastRowID = Int64(values["ann_last_rowid"] ?? "") ?? 0
        m.builtAt = Double(values["ann_built_at"] ?? "") ?? 0
        return m
    }

    private func writeMeta(_ m: Meta, accumulateRows: Int) throws {
        let stored = meta()
        let values: [(String, String)] = [
            ("ann_ver", String(Self.version)),
            ("ann_clusters", String(m.clusters)),
            ("ann_block_rows", String(m.blockRows)),
            ("ann_dim", String(m.dim)),
            ("ann_blocks", String(m.blocks)),
            ("ann_rows", String(stored.rows + accumulateRows)),
            ("ann_stale_rows", String(m.staleRows)),
            ("ann_last_rowid", String(m.lastRowID)),
            ("ann_built_at", String(m.builtAt)),
        ]
        for (key, value) in values {
            try db.execute("""
            INSERT INTO doc_meta(key,value) VALUES(?,?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, [.text(key), .text(value)])
        }
    }

    private func reset() throws {
        try db.execute("DELETE FROM ann_cells")
        try db.execute("DELETE FROM ann_blocks")
        try db.execute("DELETE FROM doc_meta WHERE key GLOB 'ann_*'")
        storeCentroids(nil)
    }

    func drop() throws {
        try db.transaction {
            try db.execute("DROP TABLE IF EXISTS ann_cells")
            try db.execute("DROP TABLE IF EXISTS ann_blocks")
            try db.execute("DELETE FROM doc_meta WHERE key GLOB 'ann_*'")
        }
        storeCentroids(nil)
    }

    /// 删除/替换已索引行时累计陈旧计数；超过阈值触发整体重建。
    func recordStale(rowIDs: [Int64]) throws {
        let m = meta()
        guard m.blocks > 0, m.lastRowID > 0 else { return }
        let stale = rowIDs.reduce(0) { $0 + ($1 <= m.lastRowID ? 1 : 0) }
        guard stale > 0 else { return }
        let updated = min(m.rows, m.staleRows + stale)
        guard updated != m.staleRows else { return }
        try db.execute("""
        INSERT INTO doc_meta(key,value) VALUES('ann_stale_rows',?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, [.text(String(updated))])
    }

    private func tailCap(dim: Int) -> Int {
        guard dim > 0 else { return 0 }
        return max(1024, Self.tailByteBudget / (dim * MemoryLayout<Float>.stride))
    }

    func status() -> Status {
        var s = Status()
        let m = meta()
        guard m.blocks > 0 else { return s }
        let total = Int((try? db.query("SELECT value FROM doc_meta WHERE key = 'doc_count'")
            .first?["value"]?.stringValue) ?? "") ?? 0
        let cap = tailCap(dim: m.dim)
        let counted = Int((try? db.query("""
            SELECT COUNT(*) AS n FROM (SELECT 1 FROM doc_vectors WHERE rowid > ? LIMIT ?)
            """, [.int(m.lastRowID), .int(Int64(cap + 1))]).first?["n"]?.intValue) ?? 0) ?? 0
        let tail = max(max(0, total - m.rows), counted)
        let stale = min(m.rows, max(0, m.staleRows))
        s.blocks = m.blocks
        s.indexedRows = m.rows
        s.tailRows = tail
        s.staleRows = stale
        s.totalRows = total
        s.clusters = m.clusters
        s.usable = m.version == Self.version && m.dim == dim && tail <= cap
        let rebuildTail = max(8192, cap / 4)
        s.needsRebuild = tail > rebuildTail || (m.rows > 0 && Double(stale) / Double(m.rows) >= Self.staleRebuildRatio)
        if s.usable {
            s.summary = "\(m.blocks) 块 × \(m.clusters) 簇，已索引 \(m.rows)/\(total) 行，尾块 \(tail)"
        } else {
            s.summary = "索引不可用（版本/维度不符或尾块 \(tail) 超过 \(cap)），需重建"
        }
        return s
    }

    // MARK: - build

    struct BuildResult {
        var blocks = 0
        var rows = 0
        var tailRows = 0
        var elapsed: TimeInterval = 0
        var summary = ""
    }

    @discardableResult
    func build(clusters: Int = ANNIndex.clusters, clusterSize: Int = ANNIndex.clusterSize,
               iterations: Int = ANNIndex.iterations, force: Bool = false,
               progress: ((Int, Int) -> Void)? = nil,
               isCancelled: (() -> Bool)? = nil) throws -> BuildResult {
        try createSchema()
        let started = Date()
        let clusterCount = min(max(2, clusters), 4096)
        let rowPerCluster = min(max(1, clusterSize), 8192)
        let blockRows = clusterCount * rowPerCluster

        var m = meta()
        if force || m.version != Self.version || m.dim != dim
            || m.clusters != clusterCount || m.blockRows != blockRows
            || (m.rows > 0 && Double(m.staleRows) / Double(m.rows) >= Self.staleRebuildRatio) {
            try reset()
            m = Meta()
        }
        let stopRowID = Int64((try? db.query("SELECT COALESCE(MAX(rowid),0) AS n FROM doc_vectors")
            .first?["n"]?.intValue) ?? 0) ?? 0
        let total = Int((try? db.query("SELECT value FROM doc_meta WHERE key='doc_count'")
            .first?["value"]?.stringValue) ?? "") ?? 0
        let estimated = max(1, (max(0, total - m.rows) + blockRows - 1) / blockRows)
        var blockID = m.blocks
        var indexedRows = m.rows
        var watermark = m.lastRowID

        while watermark < stopRowID {
            // 块间检查取消：已写块与水位随每块事务持久化（writeBlock 内提交），
            // 中断后下次构建从 rowid 水位续建，不丢已索引数据。
            if isCancelled?() == true { throw CancellationError() }
            var buffer = ANNBlockBuffer()
            try gather(after: watermark, maxRows: blockRows, into: &buffer)
            guard !buffer.rowids.isEmpty else { break }
            try writeBlock(blockID: blockID, buffer: buffer, clusters: clusterCount,
                           blockRows: blockRows, iterations: iterations)
            blockID += 1
            indexedRows += buffer.count
            watermark = buffer.lastRowID
            progress?(blockID, estimated)
        }
        var result = BuildResult()
        result.blocks = blockID
        result.rows = indexedRows
        result.tailRows = max(0, total - indexedRows)
        result.elapsed = -started.timeIntervalSinceNow
        result.summary = "ANN 索引：\(blockID) 块 / \(indexedRows) 行，耗时 "
            + String(format: "%.1fs", result.elapsed)
        return result
    }

    private func gather(after rowID: Int64, maxRows: Int, into buffer: inout ANNBlockBuffer) throws {
        buffer.reset(maxRows: maxRows, dim: dim)
        var cursor = rowID
        while buffer.count < maxRows {
            let rows = try db.query("""
                SELECT rowid, embedding FROM doc_vectors
                WHERE rowid > ? ORDER BY rowid LIMIT ?
                """, [.int(cursor), .int(Int64(maxRows - buffer.count))])
            guard let lastRowID = rows.last?["rowid"]?.intValue else { break }
            for row in rows {
                guard let id = row["rowid"]?.intValue,
                      let data = row["embedding"]?.dataValue,
                      data.count == dim * MemoryLayout<Float>.stride else { continue }
                let floats = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                buffer.append(rowID: id, floats: floats, dim: dim)
                cursor = id
            }
            // 防御：cursor 只在成功 append 时前进；窗口内存在损坏 blob（尤其
            // 全部损坏）时会卡死外层循环。每窗口强制推进到窗口最大 rowid。
            cursor = max(cursor, lastRowID)
        }
    }

    private func writeBlock(blockID: Int, buffer: ANNBlockBuffer, clusters: Int,
                            blockRows: Int, iterations: Int) throws {
        let rows = buffer.count
        guard rows > 0 else { return }
        let active = min(clusters, rows)
        var centroids = [Float](repeating: 0, count: active * dim)
        var counts = [Int32](repeating: 0, count: active)
        if let seeds = seedCentroids(buffer: buffer, clusters: active),
           let trained = ANNClustering.train(buffer: buffer, rows: rows, dim: dim,
                                             clusters: active, iterations: iterations, seeds: seeds) {
            centroids = trained.centroids
            counts = trained.counts
        }
        let assignments = ANNClustering.assign(buffer: buffer, rows: rows, dim: dim,
                                               clusters: active, centroids: centroids)
        var offsets = [Int](repeating: 0, count: active + 1)
        for a in assignments { offsets[Int(a) + 1] += 1 }
        for c in 0..<active { offsets[c + 1] += offsets[c] }
        var order = [Int32](repeating: 0, count: rows)
        var cursor = offsets
        for (index, a) in assignments.enumerated() {
            let c = Int(a)
            order[cursor[c]] = Int32(index)
            cursor[c] += 1
        }
        let blockCentroids = centroids + [Float](repeating: 0, count: (clusters - active) * dim)
        let blockCounts = counts + [Int32](repeating: 0, count: clusters - active)

        try db.transaction {
            for c in 0..<active {
                let payload = ANNPacking.packCell(buffer: buffer, order: order,
                                                  range: offsets[c]..<offsets[c + 1], dim: dim)
                try db.execute("""
                INSERT OR REPLACE INTO ann_cells(block_id,cluster_id,seq,count,payload)
                VALUES(?,?,0,?,?)
                """, [.int(Int64(blockID)), .int(Int64(c)), .int(Int64(offsets[c + 1] - offsets[c])),
                      .blob(payload)])
            }
            try db.execute("""
            INSERT OR REPLACE INTO ann_blocks(block_id,clusters,dim,payload) VALUES(?,?,?,?)
            """, [.int(Int64(blockID)), .int(Int64(clusters)), .int(Int64(dim)),
                  .blob(ANNPacking.packCentroids(centroids: blockCentroids,
                                                 counts: blockCounts, clusters: clusters, dim: dim))])
            var m = meta()
            m.version = Self.version
            m.clusters = clusters
            m.blockRows = blockRows
            m.dim = dim
            m.blocks = blockID + 1
            m.lastRowID = buffer.lastRowID
            m.builtAt = Date().timeIntervalSince1970
            try writeMeta(m, accumulateRows: rows)
        }
        storeCentroids(nil)
    }

    private func seedCentroids(buffer: ANNBlockBuffer, clusters: Int) -> [Float]? {
        let rows = buffer.count
        guard rows > 0, clusters > 0 else { return nil }
        let sampleCount = min(rows, max(clusters * 4, 1024))
        var seeds = [Float](repeating: 0, count: clusters * dim)
        var row = [Float](repeating: 0, count: dim)
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        var used = Set<Int>()
        for cluster in 0..<clusters {
            var pick = 0, attempts = 0
            repeat {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                pick = Int((state >> 33) % UInt64(sampleCount))
                attempts += 1
            } while used.contains(pick) && attempts < 32
            used.insert(pick)
            let source = sampleCount >= rows ? min(pick, rows - 1)
                : min(Int(Double(pick) * Double(rows) / Double(sampleCount)), rows - 1)
            buffer.dequantize(row: source, dim: dim, into: &row)
            ANNClustering.normalize(&row, count: dim)
            for i in 0..<dim { seeds[cluster * dim + i] = row[i] }
        }
        return seeds
    }

    // MARK: - search

    /// 返回 nil 表示 ANN 不可用，调用方应回退精确检索。
    /// `knownStatus` 由调用方复用已算好的 status（避免每次检索重复查 doc_meta / COUNT）。
    func candidates(query: [Float], topK: Int, nprobe: Int = ANNIndex.defaultNprobe,
                    rescore: Int = ANNIndex.defaultRescore,
                    knownStatus: Status? = nil) -> [Candidate]? {
        let m = meta()
        guard m.blocks > 0, m.version == Self.version, m.dim == dim,
              query.count == dim else { return nil }
        let cap = tailCap(dim: dim)
        let currentStatus = knownStatus ?? status()
        guard currentStatus.tailRows <= cap else { return nil }
        guard let cache = centroidCache(for: m) else { return nil }
        do {
            var normalized = query
            ANNClustering.normalize(&normalized, count: dim)
            var pool = ANNTopKCollector(capacity: max(topK * 8, 256))
            var scratch: [Float] = []
            try normalized.withUnsafeBufferPointer { pointer in
                guard let base = pointer.baseAddress else { return }
                for block in 0..<m.blocks {
                    let probes = cache.nearest(query: base, block: block,
                                               nprobe: nprobe, scratch: &scratch)
                    try scoreCells(block: block, clusters: probes, query: base, pool: &pool)
                }
            }
            let tail = currentStatus.tailRows
            if tail > 0 { try scanTail(query: normalized, after: m.lastRowID, limit: tail, pool: &pool) }
            var found = pool.sortedItems()
            if rescore > 0, !found.isEmpty {
                found = exactRescore(found, query: normalized, limit: max(topK, rescore))
            }
            guard !found.isEmpty else { return nil }
            return Array(found.prefix(topK)).map { Candidate(rowID: $0.0, score: $0.1) }
        } catch {
            return nil
        }
    }

    private func centroidCache(for m: Meta) -> ANNCentroidCache? {
        // 只在锁内读；db.query / 解码放在锁外，避免与构建线程的 DB 事务形成 ABBA 死锁。
        if let cache = cachedCentroids(matching: m) { return cache }
        guard let rows = try? db.query("SELECT block_id,payload FROM ann_blocks ORDER BY block_id"),
              rows.count == m.blocks else { return nil }
        var codes: [Int8] = []
        var scales: [Float] = []
        codes.reserveCapacity(m.blocks * m.clusters * m.dim)
        scales.reserveCapacity(m.blocks * m.clusters)
        for row in rows {
            guard let data = row["payload"]?.dataValue else { return nil }
            let decoded = ANNPacking.decodeCentroids(data, clusters: m.clusters, dim: m.dim)
            guard decoded != nil else { return nil }
            codes.append(contentsOf: decoded!.codes)
            scales.append(contentsOf: decoded!.scales)
        }
        let cache = ANNCentroidCache(blocks: m.blocks, clusters: m.clusters,
                                     dim: m.dim, codes: codes, scales: scales)
        storeCentroids(cache)
        return cache
    }

    private func scoreCells(block: Int, clusters: [Int], query: UnsafePointer<Float>,
                            pool: inout ANNTopKCollector) throws {
        guard !clusters.isEmpty else { return }
        let placeholders = Array(repeating: "?", count: clusters.count).joined(separator: ",")
        var bindings: [SQLValue] = [.int(Int64(block))]
        bindings.append(contentsOf: clusters.map { .int(Int64($0)) })
        let rows = try db.query("""
        SELECT payload FROM ann_cells WHERE block_id = ? AND cluster_id IN (\(placeholders))
        ORDER BY cluster_id, seq
        """, bindings)
        let rowBytes = 12 + dim
        for row in rows {
            guard let payload = row["payload"]?.dataValue else { continue }
            let count = payload.count / rowBytes
            guard count > 0 else { continue }
            var rowids = [Int64](repeating: 0, count: count)
            var decoded = [Float](repeating: 0, count: count * dim)
            var scores = [Float](repeating: 0, count: count)
            payload.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                var scales = [Float](repeating: 0, count: count)
                for i in 0..<count {
                    let offset = i * rowBytes
                    rowids[i] = raw.loadUnaligned(fromByteOffset: offset, as: Int64.self)
                    scales[i] = raw.loadUnaligned(fromByteOffset: offset + 8, as: Float.self) / 127
                }
                decoded.withUnsafeMutableBufferPointer { buffer in
                    guard let out = buffer.baseAddress else { return }
                    for i in 0..<count {
                        let codes = base.advanced(by: i * rowBytes + 12).assumingMemoryBound(to: Int8.self)
                        vDSP_vflt8(codes, 1, out + i * dim, 1, vDSP_Length(dim))
                        var factor = scales[i]
                        vDSP_vsmul(out + i * dim, 1, &factor, out + i * dim, 1, vDSP_Length(dim))
                    }
                    vDSP_mmul(out, 1, query, 1, &scores, 1,
                              vDSP_Length(count), 1, vDSP_Length(dim))
                }
            }
            for i in 0..<count { pool.offer(rowid: rowids[i], score: Double(scores[i])) }
        }
    }

    private func scanTail(query: [Float], after rowID: Int64, limit: Int,
                          pool: inout ANNTopKCollector) throws {
        var cursor = rowID
        var scanned = 0
        while scanned < limit {
            let rows = try db.query("""
            SELECT rowid, embedding FROM doc_vectors WHERE rowid > ? ORDER BY rowid LIMIT ?
            """, [.int(cursor), .int(Int64(min(1024, limit - scanned)))])
            guard let lastRowID = rows.last?["rowid"]?.intValue else { break }
            for row in rows {
                guard let id = row["rowid"]?.intValue, let data = row["embedding"]?.dataValue,
                      data.count == dim * MemoryLayout<Float>.stride else { continue }
                let floats = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                var dot: Float = 0
                floats.withUnsafeBufferPointer { p in
                    query.withUnsafeBufferPointer { q in
                        guard let a = p.baseAddress, let b = q.baseAddress else { return }
                        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(dim))
                    }
                }
                pool.offer(rowid: id, score: Double(dot))
                scanned += 1
            }
            // 防御：坏 BLOB 行会被上面的 guard 跳过，若整批都损坏，cursor 与
            // scanned 都不前进，会用同一 cursor 反复查同一批行造成死循环。
            // 每批无条件推进到窗口最大 rowid（与 gather 同策略）。
            cursor = max(cursor, lastRowID)
        }
    }

    private func exactRescore(_ candidates: [(Int64, Double)], query: [Float],
                              limit: Int) -> [(Int64, Double)] {
        let head = Array(candidates.prefix(limit).map(\.0))
        guard !head.isEmpty else { return candidates }
        var exact: [Int64: [Float]] = [:]
        var offset = 0
        while offset < head.count {
            let end = min(offset + 200, head.count)
            let placeholders = Array(repeating: "?", count: end - offset).joined(separator: ",")
            let bindings: [SQLValue] = head[offset..<end].map { .int($0) }
            if let rows = try? db.query("""
                SELECT rowid, embedding FROM doc_vectors WHERE rowid IN (\(placeholders))
                """, bindings) {
                for row in rows {
                    guard let id = row["rowid"]?.intValue, let data = row["embedding"]?.dataValue,
                          data.count == dim * MemoryLayout<Float>.stride else { continue }
                    exact[id] = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                }
            }
            offset = end
        }
        guard !exact.isEmpty else { return candidates }
        var out: [(Int64, Double)] = []
        for (id, _) in candidates.prefix(limit) {
            guard let vector = exact[id] else { continue }
            out.append((id, VectorMath.cosine(query, vector)))
        }
        guard !out.isEmpty else { return candidates }
        out.sort { $0.1 > $1.1 }
        return out
    }
}
