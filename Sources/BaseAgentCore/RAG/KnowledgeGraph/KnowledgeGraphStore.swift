import Foundation

// MARK: - 知识图谱存储层

/// 知识图谱的 SQLite 持久化层：`entities` / `relationships` / `kg_vectors` 三张表。
///
/// 与 `VectorStore`（切片主存储）共享同一个 SQLite 文件，但使用**独立的连接**
/// 打开（WAL 模式下多连接安全），互不阻塞。向量默认 int8 落盘，检索走精确
/// 余弦扫描（图谱规模远小于切片全库，无需 vec0 ANN）。
public final class KnowledgeGraphStore: @unchecked Sendable {
    public static let entitiesTable = "entities"
    public static let relationshipsTable = "relationships"
    public static let kgVectorsTable = "kg_vectors"

    private static let dimensionKey = "kg_vector_dim"
    private static let entitiesCountKey = "kg_entities_count"
    private static let relationshipsCountKey = "kg_relationships_count"

    /// 实体/关系向量文本的 token 截断上限（嵌入前）。
    private let maxEmbeddingTokens = 1_000

    private let db: SQLiteDatabase
    public let dbPath: String
    private let embedding: EmbeddingProvider
    /// 向量维度（跟随 embedding provider 输出，jina-v5 默认 128）。
    public private(set) var dimension: Int
    private let lock = NSRecursiveLock()

    private var tokenCounter: (@Sendable (String) -> Int)?

    public init(dbPath: String, embedding: EmbeddingProvider) throws {
        self.dbPath = dbPath
        self.embedding = embedding
        self.dimension = max(1, embedding.dimension)
        self.db = try SQLiteDatabase(path: dbPath)
        try createSchema()
    }

    // MARK: - Schema

    private func createSchema() throws {
        lock.lock(); defer { lock.unlock() }
        try db.execScript("""
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

    // MARK: - 实体 CRUD

    @discardableResult
    public func upsertEntity(_ entity: KGEntity) throws -> KGEntity {
        lock.lock(); defer { lock.unlock() }
        let now = Date().timeIntervalSince1970
        let props = Self.encodeProperties(entity.properties)
        let sources = Self.encodeStringArray(entity.sourceIDs)
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
            .text(entity.description), .text(props), .text(sources),
            .double(entity.createdAt > 0 ? entity.createdAt : now), .double(now),
        ])
        return entity
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
        let now = Date().timeIntervalSince1970
        let props = Self.encodeProperties(rel.properties)
        let sources = Self.encodeStringArray(rel.sourceIDs)
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
            .text(props), .text(sources),
            .double(rel.createdAt > 0 ? rel.createdAt : now), .double(now),
        ])
        return rel
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

    /// 计算并落盘实体向量（int8）。
    public func storeEntityVector(id: String, text: String) async throws {
        let vectors = try await embedding.embedDocuments([text])
        guard let vector = vectors.first else { return }
        try storeVector(targetType: .entity, targetID: id, vector: vector)
    }

    /// 计算并落盘关系向量（int8）。
    public func storeRelationshipVector(id: String, text: String) async throws {
        let vectors = try await embedding.embedDocuments([text])
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
        // 同目标只保留最新一条向量（先删后插，避免 upsert 残留旧维度）。
        try db.execute("DELETE FROM \(Self.kgVectorsTable) WHERE target_type = ? AND target_id = ?",
                       [.text(targetType.rawValue), .text(targetID)])
        try db.execute("""
        INSERT INTO \(Self.kgVectorsTable)(target_type, target_id, embedding) VALUES(?, ?, ?)
        """, [.text(targetType.rawValue), .text(targetID), .blob(blob)])
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
    public func searchRelationships(vector: [Float], topK: Int,
                                    similarityFloor: Double) throws -> [RelationHit] {
        try search(targetType: .relationship, vector: vector, topK: topK,
                   similarityFloor: similarityFloor).compactMap { rowID, score in
            guard let rel = try? relationship(id: rowID) else { return nil }
            return RelationHit(relationship: rel, score: score)
        }
    }

    /// 精确余弦扫描：解码 int8 向量与查询向量比余弦。
    private func search(targetType: KGVectorTargetType, vector: [Float], topK: Int,
                        similarityFloor: Double) throws -> [(String, Double)] {
        lock.lock(); defer { lock.unlock() }
        let query = VectorCodec.normalize(vector)
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
        return Array(scored.sorted { $0.1 > $1.1 }.prefix(max(0, topK)))
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
        for id in ids {
            let rows = try db.query("""
            SELECT id, text, source, heading FROM \(VectorStore.chunksTable) WHERE id = ?
            """, [.text(id)])
            for row in rows {
                guard let cid = row["id"]?.stringValue else { continue }
                out.append(ChunkRef(id: cid,
                                    text: row["text"]?.stringValue ?? "",
                                    source: row["source"]?.stringValue ?? "",
                                    heading: row["heading"]?.stringValue ?? ""))
            }
        }
        return out
    }

    // MARK: - 清空 / 重建

    /// 清空图谱（purge）：实体/关系/向量全部删除。重建路径会据此重算 weight。
    public func clear() throws {
        lock.lock(); defer { lock.unlock() }
        try db.transaction {
            try db.execute("DELETE FROM \(Self.kgVectorsTable)")
            try db.execute("DELETE FROM \(Self.entitiesTable)")
            try db.execute("DELETE FROM \(Self.relationshipsTable)")
        }
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
