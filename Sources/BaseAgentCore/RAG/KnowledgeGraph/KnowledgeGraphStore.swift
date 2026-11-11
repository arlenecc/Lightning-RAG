import Foundation

// MARK: - 知识图谱存储层

/// 知识图谱的 SQLite 持久化层：`entities` / `relationships` / `kg_vectors` 三张表。
///
/// 与 `VectorStore`（切片主存储）共享同一个 SQLite 文件，但使用**独立的连接**
/// 打开（WAL 模式下多连接安全），互不阻塞。向量默认 int8 落盘，优先写单张
/// vec0 分区表 `kg_vectors_vec` 做 KNN；vec0 不可用时回退精确余弦扫描。
public final class KnowledgeGraphStore: @unchecked Sendable {
    public static let entitiesTable = "entities"
    public static let relationshipsTable = "relationships"
    public static let kgVectorsTable = "kg_vectors"

    private static let dimensionKey = "kg_vector_dim"
    private static let entitiesCountKey = "kg_entities_count"
    private static let relationshipsCountKey = "kg_relationships_count"
    /// 图谱数据版本：任何实体/关系写入都会自增，快照靠它判断是否落后于数据库。
    static let revisionKey = "kg_revision"
    /// 图谱构建状态：building / complete / stale。
    static let buildStateKey = "kg_build_state"
    static let extractionCacheTable = "kg_extraction_cache"
    static let summaryCacheTable = "kg_summary_cache"
    static let processedChunksTable = "kg_processed_chunks"

    /// 图谱构建状态。`stale` 表示知识库内容已变化，旧图谱不可再用于检索。
    public enum BuildState: String, Sendable {
        case building
        case complete
        case stale
    }

    /// 实体/关系向量文本的 token 截断上限（嵌入前）。
    private let maxEmbeddingTokens = 1_000

    private let db: SQLiteDatabase
    public let dbPath: String
    private let embedding: EmbeddingProvider
    /// 向量维度（跟随 embedding provider 输出，jina-v5 默认 128）。
    public private(set) var dimension: Int
    private let lock = NSRecursiveLock()

    private var tokenCounter: (@Sendable (String) -> Int)?
    /// 是否使用 sqlite-vec 的 vec0 虚拟表做 KNN（失败时回退精确扫描）。
    private let useVec0: Bool
    private var vec0Ready = false
    private static let vecTableName = "kg_vectors_vec"
    private static let vectorRowidsTable = "kg_vector_rowids"
    private static let vec0DimKey = "kg_vec0_dim"
    private static let vec0MigratedKey = "kg_vec0_migrated"
    /// 单表 vec0 布局版本；旧版双表布局会被清理并按新布局重建。
    private static let vec0LayoutVersion = "2"

    public init(dbPath: String, embedding: EmbeddingProvider,
                enableSQLiteVec: Bool = true) throws {
        self.dbPath = dbPath
        self.embedding = embedding
        self.dimension = max(1, embedding.dimension)
        self.useVec0 = enableSQLiteVec
        self.db = try SQLiteDatabase(path: dbPath)
        try createSchema()
        setupVec0Locked()
    }

    // MARK: - Schema

    private func createSchema() throws {
        lock.lock(); defer { lock.unlock() }
        try db.execScript("""
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS \(Self.entitiesTable) (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT '',
            description TEXT NOT NULL DEFAULT '',
            properties TEXT NOT NULL DEFAULT '{}',
            source_ids TEXT NOT NULL DEFAULT '[]',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_kg_entities_name ON \(Self.entitiesTable)(name);

        CREATE TABLE IF NOT EXISTS \(Self.relationshipsTable) (
            id TEXT PRIMARY KEY,
            src_entity TEXT NOT NULL,
            dst_entity TEXT NOT NULL,
            relation_type TEXT NOT NULL DEFAULT '',
            description TEXT NOT NULL DEFAULT '',
            weight REAL NOT NULL DEFAULT 1,
            properties TEXT NOT NULL DEFAULT '{}',
            source_ids TEXT NOT NULL DEFAULT '[]',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_kg_rel_src ON \(Self.relationshipsTable)(src_entity);
        CREATE INDEX IF NOT EXISTS idx_kg_rel_dst ON \(Self.relationshipsTable)(dst_entity);

        CREATE TABLE IF NOT EXISTS \(Self.kgVectorsTable) (
            vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            embedding BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_kg_vectors_target ON \(Self.kgVectorsTable)(target_type, target_id);

        CREATE TABLE IF NOT EXISTS \(Self.vectorRowidsTable) (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            UNIQUE(target_type, target_id)
        );

        CREATE TABLE IF NOT EXISTS \(Self.extractionCacheTable) (
            chunk_id TEXT PRIMARY KEY,
            content_hash TEXT,
            output TEXT NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_kg_extraction_hash
            ON \(Self.extractionCacheTable)(content_hash);
        CREATE TABLE IF NOT EXISTS \(Self.processedChunksTable) (
            chunk_id TEXT PRIMARY KEY,
            processed_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS \(Self.summaryCacheTable) (
            cache_key TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            updated_at REAL NOT NULL
        );
        """)
        // 旧库迁移：记录切片内容哈希，用于判断「同一 chunk id 的内容是否已变化」。
        try? db.execute("ALTER TABLE \(Self.processedChunksTable) ADD COLUMN content_hash TEXT")
        // 抽取缓存按内容哈希复用：重建时切片 id 变化但正文相同仍可命中。
        try? db.execute("ALTER TABLE \(Self.extractionCacheTable) ADD COLUMN content_hash TEXT")
        try? db.execute("""
        CREATE INDEX IF NOT EXISTS idx_kg_extraction_hash
            ON \(Self.extractionCacheTable)(content_hash)
        """)

        let storedDim = try db.scalar(
            "SELECT value FROM meta WHERE key = ?", [.text(Self.dimensionKey)]).stringValue
        if let stored = storedDim, Int(stored) != dimension {
            // 维度变化：实体/关系正文仍在，向量需重建；直接清空向量表即可。
            AgentLog.warning("知识图谱向量维度 \(stored) → \(dimension)，已清空 kg_vectors")
            try db.execute("DELETE FROM \(Self.kgVectorsTable)")
        }
        try db.execute("""
        INSERT INTO meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, [.text(Self.dimensionKey), .text(String(dimension))])
    }

    /// 测试/诊断：vec0 索引是否可用；不可用时检索自动走精确扫描。
    var usesVec0Index: Bool {
        lock.lock(); defer { lock.unlock() }
        return vec0Ready
    }

    // MARK: - vec0 索引

    /// 单张 vec0 表同时存实体与关系向量，用 `target_type` 分区键区分。
    /// 建表/迁移失败时保持 `vec0Ready = false`，检索自动回退精确扫描。
    private func setupVec0Locked() {
        guard useVec0, db.vectorExtensionLoaded else {
            vec0Ready = false
            return
        }
        lock.lock(); defer { lock.unlock() }
        do {
            // 旧版（实体/关系各一张 vec0 表）直接清理，kg_vectors 是完整数据源。
            try db.execScript("""
            DROP TABLE IF EXISTS kg_entity_vectors;
            DROP TABLE IF EXISTS kg_relation_vectors;
            DROP TABLE IF EXISTS kg_entity_vector_map;
            DROP TABLE IF EXISTS kg_relation_vector_map;
            """)
            let storedDim = try db.scalar("SELECT value FROM meta WHERE key = ?",
                                          [.text(Self.vec0DimKey)]).stringValue
            let storedLayout = try db.scalar("SELECT value FROM meta WHERE key = ?",
                                             [.text("kg_vec0_layout")]).stringValue
            let layout = "\(dimension)|\(Self.vec0LayoutVersion)"
            let dimChanged = storedDim.map { Int($0) != dimension } ?? false
            if dimChanged || storedLayout != layout {
                // 维度或布局变化：vec0 结构无法原地变更，重建后再从 kg_vectors 迁移。
                try db.execScript("""
                DROP TABLE IF EXISTS \(Self.vecTableName);
                DELETE FROM \(Self.vectorRowidsTable);
                """)
            }
            try db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS \(Self.vecTableName)
            USING vec0(
                target_type TEXT PARTITION KEY,
                +target_id TEXT,
                embedding int8[\(dimension)] distance_metric=cosine
            )
            """)
            try setMetaLocked(Self.vec0DimKey, String(dimension))
            try setMetaLocked("kg_vec0_layout", layout)

            let migrated = try db.scalar("SELECT value FROM meta WHERE key = ?",
                                         [.text(Self.vec0MigratedKey)]).stringValue
            if migrated != layout {
                try migrateToVec0Locked()
                try setMetaLocked(Self.vec0MigratedKey, layout)
            }
            vec0Ready = true
        } catch {
            vec0Ready = false
            AgentLog.warning("知识图谱 vec0 建表失败，退化为精确向量扫描: "
                + error.localizedDescription)
        }
    }

    private func setMetaLocked(_ key: String, _ value: String) throws {
        try db.execute("""
        INSERT INTO meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, [.text(key), .text(value)])
    }

    private func migrateToVec0Locked() throws {
        let rows = try db.query("""
        SELECT target_type, target_id, embedding FROM \(Self.kgVectorsTable)
        """)
        try db.transaction {
            for row in rows {
                guard let rawType = row["target_type"]?.stringValue,
                      let targetType = KGVectorTargetType(rawValue: rawType),
                      let targetID = row["target_id"]?.stringValue,
                      let blob = row["embedding"]?.dataValue,
                      blob.count == dimension else { continue }
                let exists = try db.scalar("""
                SELECT COUNT(*) FROM \(Self.vectorRowidsTable)
                WHERE target_type = ? AND target_id = ?
                """, [.text(targetType.rawValue), .text(targetID)]).intValue
                guard exists == 0 else { continue }
                try insertVecRowLocked(type: targetType, targetID: targetID, blob: blob)
            }
        }
    }

    private func insertVecRowLocked(type: KGVectorTargetType, targetID: String,
                                    blob: Data) throws {
        try db.execute("""
        INSERT INTO \(Self.vectorRowidsTable)(target_type, target_id) VALUES(?, ?)
        """, [.text(type.rawValue), .text(targetID)])
        guard let rowid = try db.scalar("SELECT last_insert_rowid()").intValue else {
            throw AgentError.storage("vec0 映射写入失败")
        }
        try db.execute("""
        INSERT INTO \(Self.vecTableName)(rowid, target_type, target_id, embedding)
        VALUES(?, ?, ?, vec_int8(?))
        """, [.int(rowid), .text(type.rawValue), .text(targetID), .blob(blob)])
    }

    /// 删除目标对应的向量（普通表 + vec0 + rowid 映射）。
    private func deleteVectorLocked(type: KGVectorTargetType, targetID: String) throws {
        try db.execute("""
        DELETE FROM \(Self.kgVectorsTable) WHERE target_type = ? AND target_id = ?
        """, [.text(type.rawValue), .text(targetID)])
        guard vec0Ready else { return }
        guard let rowid = try db.scalar("""
        SELECT rowid FROM \(Self.vectorRowidsTable)
        WHERE target_type = ? AND target_id = ?
        """, [.text(type.rawValue), .text(targetID)]).intValue, rowid > 0 else { return }
        try db.execute("DELETE FROM \(Self.vecTableName) WHERE rowid = ?", [.int(rowid)])
        try db.execute("DELETE FROM \(Self.vectorRowidsTable) WHERE rowid = ?", [.int(rowid)])
    }

    private func deleteAllVectorsLocked() throws {
        try db.execute("DELETE FROM \(Self.kgVectorsTable)")
        guard vec0Ready else { return }
        // vec0 整表删除走重建，避免逐行 xUpdate。
        try resetVec0Locked()
    }

    private func resetVec0Locked() throws {
        try db.execScript("""
        DROP TABLE IF EXISTS \(Self.vecTableName);
        DELETE FROM \(Self.vectorRowidsTable);
        """)
        try db.execute("""
        CREATE VIRTUAL TABLE \(Self.vecTableName)
        USING vec0(
            target_type TEXT PARTITION KEY,
            +target_id TEXT,
            embedding int8[\(dimension)] distance_metric=cosine
        )
        """)
    }

    // MARK: - 计数与状态

    public func counts() -> (entities: Int, relationships: Int) {
        lock.lock(); defer { lock.unlock() }
        let e = (try? db.scalar("SELECT COUNT(*) FROM \(Self.entitiesTable)"))?.intValue ?? 0
        let r = (try? db.scalar("SELECT COUNT(*) FROM \(Self.relationshipsTable)"))?.intValue ?? 0
        return (entities: Int(e), relationships: Int(r))
    }

    public func hasGraphData() -> Bool {
        let c = counts()
        return c.entities > 0 || c.relationships > 0
    }

    /// 记录构建状态（全量构建开始写 building，建图完成后写 complete，
    /// 知识库内容变化时写 stale）。
    public func markBuildState(_ state: BuildState) throws {
        lock.lock(); defer { lock.unlock() }
        try db.execute("""
        INSERT INTO meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, [.text(Self.buildStateKey), .text(state.rawValue)])
    }

    /// 读取构建状态；旧库没有该标记时返回 nil（由调用方决定如何判定）。
    public func buildState() -> BuildState? {
        lock.lock(); defer { lock.unlock() }
        let raw = (try? db.scalar("SELECT value FROM meta WHERE key = ?",
                                  [.text(Self.buildStateKey)]))?.stringValue
        return raw.flatMap(BuildState.init(rawValue:))
    }

    /// 当前数据版本（每次实体/关系写入后自增）。
    public func revision() -> Int {
        lock.lock(); defer { lock.unlock() }
        let raw = (try? db.scalar("SELECT value FROM meta WHERE key = ?",
                                  [.text(Self.revisionKey)]))?.stringValue
        return Int(raw ?? "0") ?? 0
    }

    private func bumpRevisionLocked() throws {
        let current = (try? db.scalar("SELECT value FROM meta WHERE key = ?",
                                      [.text(Self.revisionKey)]))?.stringValue
        let next = (Int(current ?? "0") ?? 0) + 1
        try db.execute("""
        INSERT INTO meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, [.text(Self.revisionKey), .text(String(next))])
    }

    // MARK: - 抽取断点缓存

    /// 读取切片抽取的原始输出（构建中断后重跑可跳过 LLM 抽取）。
    public func cachedExtraction(chunkID: String) -> String? {
        lock.lock(); defer { lock.unlock() }
        return (try? db.scalar("SELECT output FROM \(Self.extractionCacheTable) WHERE chunk_id = ?",
                               [.text(chunkID)]))?.stringValue
    }

    /// 优先按 chunk id（断点续跑），再按内容哈希（跨重建复用同一正文的抽取结果）。
    public func cachedExtraction(chunkID: String, contentHash: String) -> String? {
        lock.lock(); defer { lock.unlock() }
        if let hit = try? db.scalar("""
        SELECT output FROM \(Self.extractionCacheTable) WHERE chunk_id = ?
        """, [.text(chunkID)]).stringValue {
            return hit
        }
        guard !contentHash.isEmpty else { return nil }
        return try? db.scalar("""
        SELECT output FROM \(Self.extractionCacheTable)
        WHERE content_hash = ? ORDER BY updated_at DESC LIMIT 1
        """, [.text(contentHash)]).stringValue
    }

    public func cacheExtraction(chunkID: String, output: String) throws {
        try cacheExtraction(chunkID: chunkID, contentHash: "", output: output)
    }

    public func cacheExtraction(chunkID: String, contentHash: String,
                                output: String) throws {
        lock.lock(); defer { lock.unlock() }
        try db.execute("""
        INSERT INTO \(Self.extractionCacheTable)(chunk_id, content_hash, output, updated_at)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET content_hash = excluded.content_hash,
            output = excluded.output, updated_at = excluded.updated_at
        """, [.text(chunkID), .text(contentHash), .text(output),
              .double(Date().timeIntervalSince1970)])
    }

    /// 抽取提示词版本变化时清空缓存；否则只清理过期条目，保留可复用结果。
    public func prepareExtractionCache(version: String,
                                       maxAge: TimeInterval = 30 * 86_400) throws {
        lock.lock(); defer { lock.unlock() }
        let key = "kg_extraction_cache_version"
        let stored = try db.scalar("SELECT value FROM meta WHERE key = ?",
                                   [.text(key)]).stringValue
        if stored != version {
            try db.execute("DELETE FROM \(Self.extractionCacheTable)")
            try db.execute("""
            INSERT INTO meta(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, [.text(key), .text(version)])
        }
        let cutoff = Date().timeIntervalSince1970 - maxAge
        try db.execute("DELETE FROM \(Self.extractionCacheTable) WHERE updated_at < ?",
                       [.double(cutoff)])
    }

    public func extractionCacheCount() -> Int {
        lock.lock(); defer { lock.unlock() }
        return Int((try? db.scalar("SELECT COUNT(*) FROM \(Self.extractionCacheTable)"))?.intValue ?? 0)
    }

    /// 描述归纳结果缓存：相同名称 + 相同描述集合直接复用，避免重建时重复调模型。
    public func cachedSummary(key: String) -> String? {
        lock.lock(); defer { lock.unlock() }
        return try? db.scalar("SELECT summary FROM \(Self.summaryCacheTable) WHERE cache_key = ?",
                              [.text(key)]).stringValue
    }

    public func cacheSummary(key: String, summary: String) {
        guard !summary.isEmpty else { return }
        lock.lock(); defer { lock.unlock() }
        try? db.execute("""
        INSERT INTO \(Self.summaryCacheTable)(cache_key, summary, updated_at)
        VALUES(?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET summary = excluded.summary,
            updated_at = excluded.updated_at
        """, [.text(key), .text(summary), .double(Date().timeIntervalSince1970)])
    }

    public func pruneSummaryCache(maxAge: TimeInterval = 90 * 86_400) {
        lock.lock(); defer { lock.unlock() }
        try? db.execute("DELETE FROM \(Self.summaryCacheTable) WHERE updated_at < ?",
                        [.double(Date().timeIntervalSince1970 - maxAge)])
    }

    public func clearExtractionCache() throws {
        lock.lock(); defer { lock.unlock() }
        try db.execute("DELETE FROM \(Self.extractionCacheTable)")
    }

    /// 已进入图谱的切片 id（增量构建只处理不在集合里的新切片）。
    public func processedChunkIDs() -> Set<String> {
        lock.lock(); defer { lock.unlock() }
        let rows = (try? db.query("SELECT chunk_id FROM \(Self.processedChunksTable)")) ?? []
        return Set(rows.compactMap { $0["chunk_id"]?.stringValue })
    }

    public func markChunksProcessed(_ chunkIDs: [String]) throws {
        guard !chunkIDs.isEmpty else { return }
        let now = Date().timeIntervalSince1970
        try markChunksProcessed(hashes: Dictionary(
            uniqueKeysWithValues: chunkIDs.map { ($0, "") }), now: now)
    }

    /// 记录切片已处理及其内容哈希；哈希不一致说明切片内容被改写，图谱需要重建。
    public func markChunksProcessed(hashes: [String: String]) throws {
        try markChunksProcessed(hashes: hashes, now: Date().timeIntervalSince1970)
    }

    private func markChunksProcessed(hashes: [String: String], now: Double) throws {
        guard !hashes.isEmpty else { return }
        lock.lock(); defer { lock.unlock() }
        try db.transaction {
            for (id, hash) in hashes {
                try db.execute("""
                INSERT INTO \(Self.processedChunksTable)(chunk_id, processed_at, content_hash)
                VALUES(?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET processed_at = excluded.processed_at,
                    content_hash = excluded.content_hash
                """, [.text(id), .double(now), .text(hash)])
            }
        }
    }

    /// 已处理切片 → 内容哈希（旧记录无哈希时返回空串，会被判为内容不一致）。
    public func processedChunkHashes() -> [String: String] {
        lock.lock(); defer { lock.unlock() }
        let rows = (try? db.query("""
            SELECT chunk_id, COALESCE(content_hash, '') AS content_hash
            FROM \(Self.processedChunksTable)
            """)) ?? []
        var result: [String: String] = [:]
        for row in rows {
            guard let id = row["chunk_id"]?.stringValue else { continue }
            result[id] = row["content_hash"]?.stringValue ?? ""
        }
        return result
    }

    // MARK: - 实体 CRUD

    @discardableResult
    public func upsertEntity(_ entity: KGEntity) throws -> KGEntity {
        lock.lock(); defer { lock.unlock() }
        try writeEntityLocked(entity)
        try bumpRevisionLocked()
        return entity
    }

    private func writeEntityLocked(_ entity: KGEntity) throws {
        let now = Date().timeIntervalSince1970
        try db.execute("""
        INSERT INTO \(Self.entitiesTable)(id, name, type, description, properties, source_ids, created_at, updated_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            type = excluded.type,
            description = excluded.description,
            properties = excluded.properties,
            source_ids = excluded.source_ids,
            updated_at = excluded.updated_at
        """, [
            .text(entity.id), .text(entity.name), .text(entity.type),
            .text(entity.description), .text(Self.encodeProperties(entity.properties)),
            .text(Self.encodeStringArray(entity.sourceIDs)),
            .double(entity.createdAt > 0 ? entity.createdAt : now), .double(now),
        ])
    }

    public func entity(id: String) throws -> KGEntity? {
        lock.lock(); defer { lock.unlock() }
        let rows = try db.query("SELECT * FROM \(Self.entitiesTable) WHERE id = ?", [.text(id)])
        return rows.first.flatMap(Self.decodeEntity)
    }

    public func entityByName(_ name: String) throws -> KGEntity? {
        lock.lock(); defer { lock.unlock() }
        let rows = try db.query("""
        SELECT * FROM \(Self.entitiesTable) WHERE name = ? COLLATE NOCASE LIMIT 1
        """, [.text(name)])
        return rows.first.flatMap(Self.decodeEntity)
    }

    public func entities(names: [String]) throws -> [KGEntity] {
        guard !names.isEmpty else { return [] }
        lock.lock(); defer { lock.unlock() }
        var out: [KGEntity] = []
        for name in names {
            if let e = try entityByName(name) { out.append(e) }
        }
        return out
    }

    public func allEntities() throws -> [KGEntity] {
        lock.lock(); defer { lock.unlock() }
        let rows = try db.query("SELECT * FROM \(Self.entitiesTable)")
        return rows.compactMap(Self.decodeEntity)
    }

    // MARK: - 关系 CRUD

    @discardableResult
    public func upsertRelationship(_ rel: KGRelationship) throws -> KGRelationship {
        lock.lock(); defer { lock.unlock() }
        try writeRelationshipLocked(rel)
        try bumpRevisionLocked()
        return rel
    }

    private func writeRelationshipLocked(_ rel: KGRelationship) throws {
        let now = Date().timeIntervalSince1970
        try db.execute("""
        INSERT INTO \(Self.relationshipsTable)(id, src_entity, dst_entity, relation_type, description, weight, properties, source_ids, created_at, updated_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            src_entity = excluded.src_entity,
            dst_entity = excluded.dst_entity,
            relation_type = excluded.relation_type,
            description = excluded.description,
            weight = excluded.weight,
            properties = excluded.properties,
            source_ids = excluded.source_ids,
            updated_at = excluded.updated_at
        """, [
            .text(rel.id), .text(rel.srcEntity), .text(rel.dstEntity),
            .text(rel.relationType), .text(rel.description), .double(rel.weight),
            .text(Self.encodeProperties(rel.properties)),
            .text(Self.encodeStringArray(rel.sourceIDs)),
            .double(rel.createdAt > 0 ? rel.createdAt : now), .double(now),
        ])
    }

    public func relationship(id: String) throws -> KGRelationship? {
        lock.lock(); defer { lock.unlock() }
        let rows = try db.query("SELECT * FROM \(Self.relationshipsTable) WHERE id = ?", [.text(id)])
        return rows.first.flatMap(Self.decodeRelationship)
    }

    /// 按无向三元组键（src/dst 字典序归一 + 谓词）查找既有关系。
    public func relationship(byCanonicalKey src: String, dst: String,
                             type: String) throws -> KGRelationship? {
        lock.lock(); defer { lock.unlock() }
        let rows = try db.query("SELECT * FROM \(Self.relationshipsTable)")
        for row in rows {
            guard let rel = Self.decodeRelationship(row) else { continue }
            if rel.canonicalKey == KGRelationship.canonicalKey(src: src, dst: dst, type: type) {
                return rel
            }
        }
        return nil
    }

    public func allRelationships() throws -> [KGRelationship] {
        lock.lock(); defer { lock.unlock() }
        let rows = try db.query("SELECT * FROM \(Self.relationshipsTable)")
        return rows.compactMap(Self.decodeRelationship)
    }

    /// 按无向实体对查找关系（不区分谓词措辞），重复时取 weight 最大的一条。
    public func relationship(byPair src: String, dst: String) throws -> KGRelationship? {
        lock.lock(); defer { lock.unlock() }
        let key = KGRelationship.pairKey(src, dst)
        let rows = try db.query("SELECT * FROM \(Self.relationshipsTable)")
        var best: KGRelationship?
        for row in rows {
            guard let rel = Self.decodeRelationship(row), rel.pairKey == key else { continue }
            if best == nil || rel.weight > (best?.weight ?? 0) { best = rel }
        }
        return best
    }

    public func deleteRelationship(id: String) throws {
        lock.lock(); defer { lock.unlock() }
        try db.transaction {
            try db.execute("DELETE FROM \(Self.relationshipsTable) WHERE id = ?", [.text(id)])
            for vectorID in [id, id + KGRelationship.reverseVectorSuffix] {
                try deleteVectorLocked(type: .relationship, targetID: vectorID)
            }
        }
        try bumpRevisionLocked()
    }

    /// 原子替换全部实体/关系（全量构建成功后才落地，失败不影响旧图谱）。
    public func replaceAll(entities: [KGEntity], relationships: [KGRelationship]) throws {
        lock.lock(); defer { lock.unlock() }
        try db.transaction {
            try deleteAllVectorsLocked()
            try db.execute("DELETE FROM \(Self.entitiesTable)")
            try db.execute("DELETE FROM \(Self.relationshipsTable)")
            // 全量重建后由调用方重新标记成功的切片，旧标记先清空。
            try db.execute("DELETE FROM \(Self.processedChunksTable)")
            for entity in entities { try writeEntityLocked(entity) }
            for rel in relationships { try writeRelationshipLocked(rel) }
        }
        try bumpRevisionLocked()
    }

    // MARK: - 向量化

    /// 实体向量文本：`name + "\n" + description`（截断到 1000 token）。
    func entityEmbeddingText(_ entity: KGEntity) -> String {
        truncate(entity.name + "\n" + entity.description)
    }

    /// 关系向量文本：`keywords + "\t" + src + "\n" + dst + "\n" + description`。
    func relationshipEmbeddingText(_ rel: KGRelationship) -> String {
        truncate(rel.relationType + "\t" + rel.srcEntity + "\n"
            + rel.dstEntity + "\n" + rel.description)
    }

    /// 关系反序向量文本（源/目标互换）：关系无向，正反两个方向各存一条。
    func reverseRelationshipEmbeddingText(_ rel: KGRelationship) -> String {
        truncate(rel.relationType + "\t" + rel.dstEntity + "\n"
            + rel.srcEntity + "\n" + rel.description)
    }

    /// 待向量化目标：实体一条，关系正序 + 反序各一条。
    func vectorTargets(entities: [KGEntity],
                       relationships: [KGRelationship])
        -> [(type: KGVectorTargetType, id: String, text: String)] {
        var targets: [(type: KGVectorTargetType, id: String, text: String)] = []
        targets.reserveCapacity(entities.count + relationships.count * 2)
        for entity in entities {
            targets.append((type: .entity, id: entity.id,
                            text: entityEmbeddingText(entity)))
        }
        for rel in relationships {
            targets.append((type: .relationship, id: rel.id,
                            text: relationshipEmbeddingText(rel)))
            targets.append((type: .relationship, id: rel.reverseVectorID,
                            text: reverseRelationshipEmbeddingText(rel)))
        }
        return targets
    }

    /// 计算并落盘实体向量（int8）。
    public func storeEntityVector(id: String, text: String) async throws {
        let vectors = try await embedding.embedDocuments([truncate(text)])
        guard let vector = vectors.first else { return }
        try storeVector(targetType: .entity, targetID: id, vector: vector)
    }

    /// 计算并落盘关系向量（int8）。
    public func storeRelationshipVector(id: String, text: String) async throws {
        let vectors = try await embedding.embedDocuments([truncate(text)])
        guard let vector = vectors.first else { return }
        try storeVector(targetType: .relationship, targetID: id, vector: vector)
    }

    /// 批量计算并落盘向量（builder 侧可一次嵌入多条，减少往返）。
    public func storeVectors(targets: [(type: KGVectorTargetType, id: String, text: String)],
                             progress: ((Int, Int) -> Void)? = nil) async throws {
        guard !targets.isEmpty else { return }
        var offset = 0
        while offset < targets.count {
            try Task.checkCancellation()
            let batch = Array(targets[offset..<min(offset + 32, targets.count)])
            let texts = batch.map(\.text)
            let vectors = try await embedding.embedDocuments(texts)
            guard vectors.count == batch.count else {
                throw AgentError.storage("知识图谱向量数量不一致")
            }
            for (item, vector) in zip(batch, vectors) {
                try storeVector(targetType: item.type, targetID: item.id, vector: vector)
            }
            offset += batch.count
            progress?(offset, targets.count)
        }
    }

    private func storeVector(targetType: KGVectorTargetType, targetID: String,
                             vector: [Float]) throws {
        lock.lock(); defer { lock.unlock() }
        let normalized = VectorCodec.normalize(vector)
        let blob = VectorEncoder.encodeInt8(normalized)
        try db.transaction {
            // 同目标只保留最新一条向量（先删后插，避免 upsert 残留旧维度）。
            try deleteVectorLocked(type: targetType, targetID: targetID)
            try db.execute("""
            INSERT INTO \(Self.kgVectorsTable)(target_type, target_id, embedding)
            VALUES(?, ?, ?)
            """, [.text(targetType.rawValue), .text(targetID), .blob(blob)])
            guard vec0Ready else { return }
            do {
                try insertVecRowLocked(type: targetType, targetID: targetID, blob: blob)
            } catch {
                // vec0 单次写入失败不影响普通表检索，但索引会缺这一行。
                AgentLog.warning("知识图谱 vec0 写入失败（\(targetID)）: "
                    + error.localizedDescription)
            }
        }
    }

    // MARK: - 检索

    public struct EntityHit: Sendable {
        public let entity: KGEntity
        public let score: Double
    }

    public struct RelationHit: Sendable {
        public let relationship: KGRelationship
        public let score: Double
    }

    /// 实体向量检索：精确余弦扫描，过滤低于相似度下限的候选，返回 topK。
    public func searchEntities(vector: [Float], topK: Int,
                               similarityFloor: Double) throws -> [EntityHit] {
        try search(targetType: .entity, vector: vector, topK: topK,
                   similarityFloor: similarityFloor).compactMap { rowID, score in
            guard let entity = try? entity(id: rowID) else { return nil }
            return EntityHit(entity: entity, score: score)
        }
    }

    /// 关系向量检索。
    ///
    /// 每条关系存正序 + 反序两条向量，命中后按关系 id 合并、保留较高分，
    /// 因此先多取一倍候选再截断，避免同一关系的两条向量挤占 topK。
    public func searchRelationships(vector: [Float], topK: Int,
                                    similarityFloor: Double) throws -> [RelationHit] {
        let candidates = try search(targetType: .relationship, vector: vector,
                                    topK: max(1, topK) * 2,
                                    similarityFloor: similarityFloor)
        var best: [String: (relationship: KGRelationship, score: Double)] = [:]
        for (rowID, score) in candidates {
            let baseID = rowID.hasSuffix(KGRelationship.reverseVectorSuffix)
                ? String(rowID.dropLast(KGRelationship.reverseVectorSuffix.count))
                : rowID
            guard let rel = try? relationship(id: baseID) else { continue }
            if let existing = best[rel.id], existing.score >= score { continue }
            best[rel.id] = (rel, score)
        }
        return best.values
            .sorted {
                if $0.score != $1.score { return $0.score > $1.score }
                return $0.relationship.id < $1.relationship.id
            }
            .prefix(max(0, topK))
            .map { RelationHit(relationship: $0.relationship, score: $0.score) }
    }

    /// 向量检索：优先走 vec0 KNN（sqlite-vec），不可用时回退精确余弦扫描。
    private func search(targetType: KGVectorTargetType, vector: [Float], topK: Int,
                        similarityFloor: Double) throws -> [(String, Double)] {
        lock.lock(); defer { lock.unlock() }
        let limit = max(0, topK)
        guard limit > 0 else { return [] }
        // 查询向量与库内向量口径一致：先对称量化为 int8 再反量化，避免
        // 查询侧浮点精度与入库侧量化误差造成分数不可比。
        let normalized = VectorCodec.normalize(vector)
        let quantized = VectorEncoder.encodeInt8(normalized)
        if vec0Ready {
            do {
                return try vecSearchLocked(targetType: targetType, quantized: quantized,
                                           limit: limit, similarityFloor: similarityFloor)
            } catch {
                // 索引损坏/扩展异常时不在每次查询重试，直接降级到扫描。
                vec0Ready = false
                AgentLog.warning("知识图谱 vec0 检索失败，退化为精确扫描: "
                    + error.localizedDescription)
            }
        }
        let query = VectorCodec.normalize(
            VectorEncoder.dequantizeInt8(quantized, dimension: dimension))
        var scored: [(String, Double)] = []
        let rows = try db.query("""
        SELECT target_id, embedding FROM \(Self.kgVectorsTable) WHERE target_type = ?
        """, [.text(targetType.rawValue)])
        for row in rows {
            guard let id = row["target_id"]?.stringValue,
                  let data = row["embedding"]?.dataValue else { continue }
            let candidate = VectorEncoder.dequantizeInt8(data, dimension: dimension)
            let score = VectorCodec.cosine(query, candidate)
            if score >= similarityFloor {
                scored.append((id, score))
            }
        }
        return Array(scored.sorted { $0.1 > $1.1 }.prefix(limit))
    }

    /// vec0 KNN：同一张表按 `target_type` 分区，距离由 sqlite-vec 计算，
    /// 再按 int8 + cosine 口径换算成相似度。
    private func vecSearchLocked(targetType: KGVectorTargetType, quantized: Data,
                                 limit: Int, similarityFloor: Double)
        throws -> [(String, Double)] {
        let rows = try db.query("""
        SELECT target_id, distance FROM \(Self.vecTableName)
        WHERE target_type = ? AND embedding MATCH vec_int8(?) AND k = ?
        """, [.text(targetType.rawValue), .blob(quantized), .init(limit)])
        var scored: [(String, Double)] = []
        for row in rows {
            guard let targetID = row["target_id"]?.stringValue,
                  let distance = row["distance"]?.doubleValue else { continue }
            let score = VectorElementType.int8.similarity(distance: distance,
                                                          dimension: dimension)
            if score >= similarityFloor {
                scored.append((targetID, score))
            }
        }
        return scored
    }

    // MARK: - 原文反查

    /// 按切片 id 反查正文（用于实体/关系 → 原文的归因反查）。
    public func chunkTexts(ids: [String]) throws -> [String: String] {
        guard !ids.isEmpty else { return [:] }
        lock.lock(); defer { lock.unlock() }
        var out: [String: String] = [:]
        for id in ids {
            let row = try db.scalar("SELECT text FROM \(VectorStore.chunksTable) WHERE id = ?",
                                    [.text(id)])
            if let text = row.stringValue { out[id] = text }
        }
        return out
    }

    /// 按切片 id 反查完整切片引用（正文 + 来源 + 标题）。
    public struct ChunkRef: Sendable {
        public let id: String
        public let text: String
        public let source: String
        public let heading: String
    }

    /// 全量切片列表（供图谱构建逐切片抽取）。按 chunk_index 排序。
    public struct ChunkListItem: Sendable {
        public let id: String
        public let text: String
        public let source: String
        public let heading: String
    }

    public func listChunkRefs() throws -> [ChunkListItem] {
        lock.lock(); defer { lock.unlock() }
        var out: [ChunkListItem] = []
        try db.stream("SELECT id, text, source, heading FROM \(VectorStore.chunksTable) "
                      + "ORDER BY source, chunk_index") { row in
            guard let id = row.text(0) else { return true }
            out.append(ChunkListItem(id: id,
                                     text: row.text(1) ?? "",
                                     source: row.text(2) ?? "",
                                     heading: row.text(3) ?? ""))
            return true
        }
        return out
    }

    public func chunkRefs(ids: [String]) throws -> [ChunkRef] {
        guard !ids.isEmpty else { return [] }
        lock.lock(); defer { lock.unlock() }
        var out: [ChunkRef] = []
        // 批量 IN 查询：召回阶段可能有上千个归因切片，逐条 SELECT 太慢。
        var offset = 0
        while offset < ids.count {
            let batch = Array(ids[offset..<min(offset + 400, ids.count)])
            let placeholders = Array(repeating: "?", count: batch.count).joined(separator: ",")
            let rows = try db.query("""
            SELECT id, text, source, heading FROM \(VectorStore.chunksTable)
            WHERE id IN (\(placeholders))
            """, batch.map { .text($0) })
            for row in rows {
                guard let cid = row["id"]?.stringValue else { continue }
                out.append(ChunkRef(id: cid,
                                    text: row["text"]?.stringValue ?? "",
                                    source: row["source"]?.stringValue ?? "",
                                    heading: row["heading"]?.stringValue ?? ""))
            }
            offset += batch.count
        }
        return out
    }

    // MARK: - 清空 / 重建

    /// 删除指定实体及其向量，并删除所有端点指向它的关系（无向边依赖两端存在）。
    public func deleteEntity(name: String) throws {
        lock.lock(); defer { lock.unlock() }
        guard let entity = try entityByName(name) else { return }
        let key = entity.name.lowercased()
        try db.transaction {
            try db.execute("DELETE FROM \(Self.entitiesTable) WHERE lower(name) = ?", [.text(key)])
            try deleteVectorLocked(type: .entity, targetID: entity.id)
            // 关联关系的正序 + 反序向量一并清理，避免留下孤儿向量。
            let relIDs = try db.query("""
            SELECT id FROM \(Self.relationshipsTable)
            WHERE lower(src_entity) = ? OR lower(dst_entity) = ?
            """, [.text(key), .text(key)]).compactMap { $0["id"]?.stringValue }
            for relID in relIDs {
                for vectorID in [relID, relID + KGRelationship.reverseVectorSuffix] {
                    try deleteVectorLocked(type: .relationship, targetID: vectorID)
                }
            }
            try db.execute("""
            DELETE FROM \(Self.relationshipsTable)
            WHERE lower(src_entity) = ? OR lower(dst_entity) = ?
            """, [.text(key), .text(key)])
        }
        try bumpRevisionLocked()
    }

    /// 清空图谱（purge）：实体/关系/向量全部删除。重建路径会据此重算 weight。
    public func clear() throws {
        lock.lock(); defer { lock.unlock() }
        try db.transaction {
            try deleteAllVectorsLocked()
            try db.execute("DELETE FROM \(Self.entitiesTable)")
            try db.execute("DELETE FROM \(Self.relationshipsTable)")
            try db.execute("DELETE FROM \(Self.processedChunksTable)")
        }
        try bumpRevisionLocked()
    }

    // MARK: - 实体消歧（手动向量语义合并）

    /// 手动触发：用向量语义相似度对图谱内同指实体做一次合并。
    ///
    /// 与「构建知识图谱」的全量重建不同，本方法**不删除既有抽取证据**，只在
    /// 现有实体/关系之上做同指合并，用于抽取/构建阶段（仅字符全匹配）之后
    /// 「进一步」收敛重复实体：
    /// - 全量 embed 实体名 → 两两余弦 + type 冲突保护 → 并查集归组；
    /// - 每组选代表（证据最多者），其余实体描述/来源并入代表后删除；
    /// - 关系端点按合并映射规范化，自环跳过，同 `canonicalKey` 的关系聚合（权重累加）；
    /// - 重算受影响实体与关系的向量；返回合并后的实体/关系供调用方重建内存图。
    ///
    /// 调用方应保证图谱已构建且当前无并发写入（UI 侧通过维护态标志拦截）。
    public func resolveEntityDuplicates(threshold: Double) async throws
        -> EntityResolutionResult {
        let entities = try allEntities()
        guard entities.count >= 2 else {
            return EntityResolutionResult(entities: entities,
                                          relationships: try allRelationships(),
                                          mergedGroups: 0, removedEntities: 0)
        }
        let relations = try allRelationships()

        // 1) 全量 embed 实体名（分批，复用内部 embedding）。
        var vecByID: [String: [Float]] = [:]
        var offset = 0
        while offset < entities.count {
            try Task.checkCancellation()
            let batch = Array(entities[offset..<min(offset + 32, entities.count)])
            let vectors = try await embedding.embedDocuments(batch.map(\.name))
            guard vectors.count == batch.count else {
                throw AgentError.storage("知识图谱实体名 embedding 数量不一致")
            }
            for (e, v) in zip(batch, vectors) {
                vecByID[e.id] = VectorCodec.normalize(v)
            }
            offset += batch.count
        }

        // 2) 两两相似度 → 合并边（type 冲突保护）。
        // 先按 type 分桶，只比较同 type 或 type 为空的组合，避免全量 O(N²)。
        var parent: [String: String] = [:]
        for e in entities { parent[e.id] = e.id }
        func find(_ x: String) -> String {
            var c = x
            while let p = parent[c], p != c {
                parent[c] = parent[p]!
                c = p
            }
            return c
        }
        var buckets: [String: [Int]] = [:]
        func typeComponents(_ raw: String) -> Set<String> {
            Set(raw.split(separator: "/")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
                .filter { !$0.isEmpty && $0 != "unknown" })
        }
        for (index, e) in entities.enumerated() {
            let components = typeComponents(e.type)
            if components.isEmpty {
                buckets["", default: []].append(index)
            } else {
                for component in components {
                    buckets[component, default: []].append(index)
                }
            }
        }
        for i in 0..<entities.count {
            let a = entities[i]
            guard let va = vecByID[a.id] else { continue }
            let components = typeComponents(a.type)
            var candidates: [Int]
            if components.isEmpty {
                // 类型未知/为空：与所有实体都兼容。
                candidates = Array((i + 1)..<entities.count)
            } else {
                // 同类型、多类型中任一相同、或对方类型为空才比较。
                candidates = buckets[""] ?? []
                for component in components {
                    candidates.append(contentsOf: buckets[component] ?? [])
                }
            }
            for j in Set(candidates) where j > i {
                let b = entities[j]
                guard let vb = vecByID[b.id] else { continue }
                if VectorCodec.cosine(va, vb) >= threshold {
                    parent[find(a.id)] = find(b.id)
                }
            }
        }

        // 3) 分组为连通分量，选代表（证据最多，其次名字较长）。
        var groups: [String: [KGEntity]] = [:]
        for e in entities {
            let root = find(e.id)
            groups[root] = (groups[root] ?? []) + [e]
        }
        var canonicalByOldName: [String: String] = [:]
        var newEntities: [KGEntity] = []
        var removedIDs: [String] = []
        var mergedGroups = 0
        for (_, group) in groups {
            if group.count == 1 {
                newEntities.append(group[0]); continue
            }
            mergedGroups += 1
            let rep = group.max { lhs, rhs in
                let diff = lhs.sourceIDs.count - rhs.sourceIDs.count
                return diff != 0 ? diff < 0 : lhs.name.count < rhs.name.count
            }!
            var descParts: [String] = []
            if !rep.description.isEmpty { descParts.append(rep.description) }
            var sourceIDs = Set(rep.sourceIDs)
            var filePaths = Set((rep.properties["file_path"] ?? "")
                .components(separatedBy: ",").filter { !$0.isEmpty })
            for e in group where e.id != rep.id {
                if !e.description.isEmpty, !descParts.contains(e.description) {
                    descParts.append(e.description)
                }
                sourceIDs.formUnion(e.sourceIDs)
                if let fp = e.properties["file_path"] {
                    filePaths.formUnion(fp.components(separatedBy: ",").filter { !$0.isEmpty })
                }
                canonicalByOldName[e.name.lowercased()] = rep.name
                removedIDs.append(e.id)
            }
            var props = rep.properties
            if !filePaths.isEmpty {
                props["file_path"] = filePaths.sorted().joined(separator: ",")
            }
            props["source_chunk_ids"] = sourceIDs.sorted().joined(separator: ",")
            var merged = rep
            merged.description = descParts.joined(separator: "\n")
            merged.sourceIDs = sourceIDs.sorted()
            merged.properties = props
            // 代表实体类型为空/UNKNOWN 时，继承组内更具体的类型。
            let concreteType = group.compactMap { $0.type.isEmpty || $0.type == "UNKNOWN"
                ? nil : $0.type }.first
            if let concreteType, merged.type.isEmpty || merged.type == "UNKNOWN" {
                merged.type = concreteType
            }
            merged.updatedAt = Date().timeIntervalSince1970
            newEntities.append(merged)
        }

        // 4) 关系规范化 + 聚合（按 canonicalKey 合并重复，权重累加）。
        var relAcc: [String: KGRelationship] = [:]
        for r in relations {
            let ns = canonicalByOldName[r.srcEntity.lowercased()] ?? r.srcEntity
            let nd = canonicalByOldName[r.dstEntity.lowercased()] ?? r.dstEntity
            guard ns.lowercased() != nd.lowercased() else { continue }  // 自环跳过
            // 关系身份按实体对归一，谓词措辞合并到同一条边。
            let canonical = KGRelationship.pairKey(ns, nd)
            if let existing = relAcc[canonical] {
                relAcc[canonical] = aggregateRelationship(existing, with: r)
            } else {
                var newR = r
                newR.srcEntity = ns
                newR.dstEntity = nd
                relAcc[canonical] = newR
            }
        }
        let newRelations = Array(relAcc.values)

        // 5) 原子写库：删除被合并实体 + 重建关系，upsert 代表实体。
        try writeResolution(removedIDs: removedIDs, entities: newEntities,
                            relations: newRelations)

        // 6) 重算向量（实体一条，关系正序 + 反序各一条）。
        let targets = vectorTargets(entities: newEntities, relationships: newRelations)
        try await storeVectors(targets: targets)

        return EntityResolutionResult(entities: newEntities, relationships: newRelations,
                                      mergedGroups: mergedGroups,
                                      removedEntities: removedIDs.count)
    }

    /// 内存聚合两条同 `canonicalKey` 关系（关键词取并集、描述拼接、来源合并、权重累加）。
    private func aggregateRelationship(_ base: KGRelationship,
                                       with other: KGRelationship) -> KGRelationship {
        var result = base
        result.weight = base.weight + other.weight
        var kws = Set(base.relationType
            .components(separatedBy: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty })
        for kw in other.relationType.components(separatedBy: ",") {
            let t = kw.trimmingCharacters(in: .whitespacesAndNewlines)
            if !t.isEmpty { kws.insert(t) }
        }
        result.relationType = kws.sorted().joined(separator: ",")
        if !other.description.isEmpty {
            result.description = base.description.isEmpty
                ? other.description
                : base.description + "\n" + other.description
        }
        var sources = Set(base.sourceIDs)
        sources.formUnion(other.sourceIDs)
        result.sourceIDs = sources.sorted()
        result.properties["description"] = result.description
        result.properties["keywords"] = result.relationType
        result.properties["weight"] = String(result.weight)
        result.properties["source_chunk_ids"] = result.sourceIDs.joined(separator: ",")
        result.updatedAt = Date().timeIntervalSince1970
        return result
    }

    /// 事务内：删除被合并实体及其向量、清空并重建关系表、upsert 代表实体。
    private func writeResolution(removedIDs: [String], entities: [KGEntity],
                                 relations: [KGRelationship]) throws {
        lock.lock(); defer { lock.unlock() }
        try db.transaction {
            for id in removedIDs {
                try db.execute("DELETE FROM \(Self.entitiesTable) WHERE id = ?", [.text(id)])
                try db.execute("""
                DELETE FROM \(Self.kgVectorsTable)
                WHERE target_type = ? AND target_id = ?
                """, [.text(KGVectorTargetType.entity.rawValue), .text(id)])
            }
            try db.execute("DELETE FROM \(Self.relationshipsTable)")
            // 关系表整体重建，旧关系（含反序）向量一并清空后再按新 id 写回。
            try db.execute("DELETE FROM \(Self.kgVectorsTable) WHERE target_type = ?",
                           [.text(KGVectorTargetType.relationship.rawValue)])
            for e in entities { try upsertEntity(e) }
            for r in relations { try upsertRelationship(r) }
        }
    }

    /// 实体消歧结果（供调用方重建内存图）。
    public struct EntityResolutionResult: Sendable {
        public let entities: [KGEntity]
        public let relationships: [KGRelationship]
        public let mergedGroups: Int
        public let removedEntities: Int
    }

    // MARK: - 编码辅助

    private static func encodeProperties(_ props: [String: String]) -> String {
        JSONValue.object(props.mapValues { .string($0) }).jsonString()
    }

    private static func decodeProperties(_ raw: String?) -> [String: String] {
        guard let raw, let obj = JSONValue.parse(raw)?.objectValue else { return [:] }
        return obj.compactMapValues { $0.stringValue }
    }

    private static func encodeStringArray(_ values: [String]) -> String {
        JSONValue.array(values.map { .string($0) }).jsonString()
    }

    private static func decodeStringArray(_ raw: String?) -> [String] {
        guard let raw, let arr = JSONValue.parse(raw)?.arrayValue else { return [] }
        return arr.compactMap(\.stringValue)
    }

    private static func decodeEntity(_ row: [String: SQLValue]) -> KGEntity? {
        guard let id = row["id"]?.stringValue, let name = row["name"]?.stringValue else {
            return nil
        }
        return KGEntity(id: id, name: name,
                        type: row["type"]?.stringValue ?? "",
                        description: row["description"]?.stringValue ?? "",
                        properties: decodeProperties(row["properties"]?.stringValue),
                        sourceIDs: decodeStringArray(row["source_ids"]?.stringValue),
                        createdAt: row["created_at"]?.doubleValue ?? 0,
                        updatedAt: row["updated_at"]?.doubleValue ?? 0)
    }

    private static func decodeRelationship(_ row: [String: SQLValue]) -> KGRelationship? {
        guard let id = row["id"]?.stringValue else { return nil }
        return KGRelationship(id: id,
                              srcEntity: row["src_entity"]?.stringValue ?? "",
                              dstEntity: row["dst_entity"]?.stringValue ?? "",
                              relationType: row["relation_type"]?.stringValue ?? "",
                              description: row["description"]?.stringValue ?? "",
                              weight: row["weight"]?.doubleValue ?? 1,
                              properties: decodeProperties(row["properties"]?.stringValue),
                              sourceIDs: decodeStringArray(row["source_ids"]?.stringValue),
                              createdAt: row["created_at"]?.doubleValue ?? 0,
                              updatedAt: row["updated_at"]?.doubleValue ?? 0)
    }

    /// 载入精确 token 计数器（若底层 embedding 支持）。
    public func loadTokenCounter() async {
        if let counter = await (embedding as? EmbeddingTokenCounting)?.makeTokenCounter() {
            lock.lock(); tokenCounter = counter; lock.unlock()
        }
    }

    /// 按 token 上限截断文本（有精确计数器用精确计数器，否则用粗略估算）。
    private func truncate(_ text: String) -> String {
        guard text.count > 200 else { return text }
        let counter = { lock.lock(); defer { lock.unlock() }; return tokenCounter }()
        if let counter {
            let budget = maxEmbeddingTokens
            if counter(text) <= budget { return text }
            // 二分找最大前缀。
            var lo = 0, hi = text.count
            while lo < hi {
                let mid = (lo + hi + 1) / 2
                let prefix = String(text.prefix(mid))
                if counter(prefix) <= budget { lo = mid } else { hi = mid - 1 }
            }
            return String(text.prefix(lo))
        }
        // 粗略兜底：CJK 约 1.3 字符/token，ASCII 约 3.5；保守取 1.6 字符/token。
        let charBudget = maxEmbeddingTokens * 16 / 10
        return text.count > charBudget ? String(text.prefix(charBudget)) : text
    }
}
