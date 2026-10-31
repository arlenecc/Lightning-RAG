import Foundation

/// stage 2 精排用的主向量镜像。
///
/// sqlite-vec 的 `vec0` 表在按 rowid 点查时仍要扫过虚拟表；大库上 200 个候选
/// 的随机点查接近线性扫描。这里把主向量编码字节按 rowid 镜像到普通表，
/// `exactVectorsLocked` 升序分批读取，把随机 B-tree 跳转变成顺序读取。
/// 镜像只存派生数据，主表仍是唯一事实来源；缺失行逐行回退主表。
extension VectorStore {
    static let rerankRowsKey = "rerank_rows"
    static let rerankDimensionKey = "rerank_dimension"
    static let rerankElementTypeKey = "rerank_element_type"

    /// 按当前主向量配置对齐镜像表。派生数据不一致时直接丢弃重建。
    func setupRerankStorageLocked() throws {
        rerankStorageEnabled = false
        rerankRows = 0
        rerankStorageReason = nil
        guard elementType != .bin else {
            rerankStorageReason = "bin 主存储不需要 rerank 镜像"
            try? db.transaction {
                try db.execute("DROP TABLE IF EXISTS \(Self.rerankVectorTable)")
                try deleteRerankMetaLocked()
            }
            return
        }

        let existingSQL = try db.scalar(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            [.text(Self.rerankVectorTable)]).stringValue
        let storedDimension = try db.scalar(
            "SELECT value FROM meta WHERE key = ?", [.text(Self.rerankDimensionKey)]).stringValue
        let storedType = try db.scalar(
            "SELECT value FROM meta WHERE key = ?", [.text(Self.rerankElementTypeKey)]).stringValue
        let storedRows = Self.readStoredCount(db, key: Self.rerankRowsKey)
        let schemaOK = existingSQL?.contains("embedding BLOB") == true
        if schemaOK, storedDimension == String(dimension),
           storedType == elementType.rawValue, let storedRows {
            rerankRows = max(0, storedRows)
            rerankStorageEnabled = true
            return
        }

        try db.transaction {
            try db.execute("DROP TABLE IF EXISTS \(Self.rerankVectorTable)")
            try deleteRerankMetaLocked()
            try db.execute("""
            CREATE TABLE \(Self.rerankVectorTable) (
                rowid INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL
            )
            """)
            rerankRows = 0
            try writeRerankMetaLocked()
        }
        rerankStorageEnabled = true
    }

    func insertRerankVector(rowID: Int64, blob: Data) throws {
        guard rerankStorageEnabled else { return }
        try db.execute(
            "INSERT OR REPLACE INTO \(Self.rerankVectorTable)(rowid, embedding) VALUES(?, ?)",
            [.int(rowID), .blob(blob)])
        rerankRows += 1
    }

    @discardableResult
    func deleteRerankRowsLocked(_ rowIDs: [Int64]) throws -> Int {
        guard rerankStorageEnabled, !rowIDs.isEmpty else { return 0 }
        var removed = 0
        var index = 0
        while index < rowIDs.count {
            let batch = Array(rowIDs[index..<min(index + deleteBatchSize, rowIDs.count)])
            let placeholders = Array(repeating: "?", count: batch.count).joined(separator: ",")
            removed += try db.execute(
                "DELETE FROM \(Self.rerankVectorTable) WHERE rowid IN (\(placeholders))",
                batch.map { SQLValue.int($0) })
            index += batch.count
        }
        rerankRows = max(0, rerankRows - removed)
        return removed
    }

    /// 清空并重建镜像表（clear / 重建配置时使用），同时把计数归零。
    func resetRerankStorageLocked(recreate: Bool = true) throws {
        guard elementType != .bin else {
            rerankRows = 0
            return
        }
        rerankRows = 0
        try db.execute("DROP TABLE IF EXISTS \(Self.rerankVectorTable)")
        guard recreate else { return }
        try db.execute("""
        CREATE TABLE IF NOT EXISTS \(Self.rerankVectorTable) (
            rowid INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL
        )
        """)
        try writeRerankMetaLocked()
    }

    /// 丢弃镜像表与元数据（空库采纳新配置时使用）。
    func dropRerankStorageLocked() throws {
        rerankRows = 0
        try db.execute("DROP TABLE IF EXISTS \(Self.rerankVectorTable)")
        try deleteRerankMetaLocked()
    }

    func persistRerankCounterLocked() throws {
        guard rerankStorageEnabled else { return }
        try db.execute("""
        INSERT INTO meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, [.text(Self.rerankRowsKey), .text(String(rerankRows))])
    }

    func writeRerankMetaLocked() throws {
        guard elementType != .bin else { return }
        let values: [(String, String)] = [
            (Self.rerankRowsKey, String(rerankRows)),
            (Self.rerankDimensionKey, String(dimension)),
            (Self.rerankElementTypeKey, elementType.rawValue),
        ]
        for (key, value) in values {
            try db.execute("""
            INSERT INTO meta(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, [.text(key), .text(value)])
        }
    }

    func deleteRerankMetaLocked() throws {
        try db.execute("DELETE FROM meta WHERE key IN (?, ?, ?)", [
            .text(Self.rerankRowsKey), .text(Self.rerankDimensionKey),
            .text(Self.rerankElementTypeKey),
        ])
    }
}
