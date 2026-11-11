import Foundation
import SwiftGraph

// MARK: - 内存知识图谱

/// 基于 SwiftGraph 的内存知识图谱。
///
/// - 顶点：实体名（`String`），一个实体一个顶点；多个切片抽到同一实体时聚合。
/// - 边：无向边（`directed: false`，等价于「正序 + 反序」两条有向 id），保证邻居
///   遍历与度数统计对称。
/// - 元数据：实体/关系的富信息（描述、权重、来源切片、文件路径）存放在并行字典
///   里，SwiftGraph 只承载邻接结构。
///
/// 快照缓存：用 Codable 序列化 `[KGEntity] + [KGRelationship]`，首次构建完成后
/// 落盘；下次启动直接反序列化重建邻接结构，免掉查库与重建。有新数据进入时增量
/// 更新内存图，并在「一段时间无更新」后防抖重存快照。
public final class KnowledgeGraph: @unchecked Sendable {

    /// 快照文件路径（`<ragDir>/knowledge_graph.snapshot.json`）。
    public let snapshotPath: String

    private var graph = UnweightedGraph<String>()
    /// 实体名（小写）→ 规范名。
    private var nameIndex: [String: String] = [:]
    /// 规范名 → 实体。
    private var entities: [String: KGEntity] = [:]
    /// 关系 id → 关系。
    private var relations: [String: KGRelationship] = [:]

    private let lock = NSRecursiveLock()
    private var dirty = false
    private var pendingSave: DispatchWorkItem?

    public init(snapshotPath: String) {
        self.snapshotPath = snapshotPath
    }

    // MARK: - 状态

    public var entityCount: Int {
        lock.lock(); defer { lock.unlock() }
        return entities.count
    }

    public var relationCount: Int {
        lock.lock(); defer { lock.unlock() }
        return relations.count
    }

    public var isEmpty: Bool {
        lock.lock(); defer { lock.unlock() }
        return entities.isEmpty && relations.isEmpty
    }

    // MARK: - 加载 / 恢复

    /// 优先从快照恢复；无快照时从存储层全量加载并建立图谱，随后保存快照。
    public func loadOrRestore(from store: KnowledgeGraphStore) throws {
        lock.lock(); defer { lock.unlock() }
        if restoreLocked() {
            AgentLog.info("知识图谱：从快照恢复（实体 \(entities.count)、关系 \(relations.count)）")
            return
        }
        let allEntities = try store.allEntities()
        let allRelationships = try store.allRelationships()
        rebuildLocked(entities: allEntities, relationships: allRelationships)
        AgentLog.info("知识图谱：从数据库加载并建立（实体 \(allEntities.count)、关系 \(allRelationships.count)）")
        saveLocked(force: true)
    }

    /// 全量重建（purge / 重建路径）。
    public func rebuild(entities: [KGEntity], relationships: [KGRelationship]) {
        lock.lock(); defer { lock.unlock() }
        rebuildLocked(entities: entities, relationships: relationships)
        saveLocked(force: true)
    }

    private func rebuildLocked(entities: [KGEntity], relationships: [KGRelationship]) {
        graph = UnweightedGraph<String>()
        nameIndex = [:]
        self.entities = [:]
        self.relations = [:]
        for entity in entities { addEntityLocked(entity) }
        for rel in relationships { addRelationshipLocked(rel) }
    }

    // MARK: - 增量更新

    /// 增加/合并实体节点。
    @discardableResult
    public func upsertEntity(_ entity: KGEntity) -> KGEntity {
        lock.lock(); defer { lock.unlock() }
        let merged = addEntityLocked(entity)
        markDirtyLocked()
        return merged
    }

    /// 增加/合并关系边。端点不存在时现场创建 `UNKNOWN` 实体节点。
    @discardableResult
    public func upsertRelationship(_ rel: KGRelationship) -> KGRelationship {
        lock.lock(); defer { lock.unlock() }
        let merged = addRelationshipLocked(rel)
        markDirtyLocked()
        return merged
    }

    private func canonicalName(_ name: String) -> String {
        name.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    @discardableResult
    private func addEntityLocked(_ entity: KGEntity) -> KGEntity {
        let canonical = canonicalName(entity.name)
        guard !canonical.isEmpty else { return entity }
        let key = canonical.lowercased()
        if let existingName = nameIndex[key] {
            // 同名合并：保留先出现的规范名，聚合来源切片/类型/文件路径。
            if var existing = entities[existingName] {
                var sourceIDs = existing.sourceIDs
                for id in entity.sourceIDs where !sourceIDs.contains(id) {
                    sourceIDs.append(id)
                }
                var props = existing.properties
                for (k, v) in entity.properties where props[k] == nil {
                    props[k] = v
                }
                var type = existing.type
                if type.isEmpty || type == "UNKNOWN" {
                    if !entity.type.isEmpty && entity.type != "UNKNOWN" { type = entity.type }
                }
                var description = existing.description
                if description.isEmpty { description = entity.description }
                existing.sourceIDs = sourceIDs
                existing.properties = props
                existing.type = type
                existing.description = description
                existing.updatedAt = max(existing.updatedAt, entity.updatedAt)
                entities[existingName] = existing
                return existing
            }
        }
        // 新实体。
        if !graph.vertexInGraph(vertex: canonical) {
            _ = graph.addVertex(canonical)
        }
        nameIndex[key] = canonical
        entities[canonical] = entity
        return entity
    }

    @discardableResult
    private func addRelationshipLocked(_ rel: KGRelationship) -> KGRelationship {
        let src = canonicalName(rel.srcEntity)
        let dst = canonicalName(rel.dstEntity)
        guard !src.isEmpty, !dst.isEmpty, src != dst else { return rel }

        // 端点缺失时现场创建 UNKNOWN 实体。
        if nameIndex[src.lowercased()] == nil {
            let placeholder = KGEntity(id: "unknown-\(src.lowercased())", name: src,
                                       type: "UNKNOWN", description: "",
                                       properties: rel.properties, sourceIDs: rel.sourceIDs,
                                       createdAt: Date().timeIntervalSince1970)
            _ = addEntityLocked(placeholder)
        }
        if nameIndex[dst.lowercased()] == nil {
            let placeholder = KGEntity(id: "unknown-\(dst.lowercased())", name: dst,
                                       type: "UNKNOWN", description: "",
                                       properties: rel.properties, sourceIDs: rel.sourceIDs,
                                       createdAt: Date().timeIntervalSince1970)
            _ = addEntityLocked(placeholder)
        }
        let srcCanonical = nameIndex[src.lowercased()] ?? src
        let dstCanonical = nameIndex[dst.lowercased()] ?? dst

        // 无向边（等价正序 + 反序两个 id）。
        if !graph.edgeExists(from: srcCanonical, to: dstCanonical) {
            graph.addEdge(from: srcCanonical, to: dstCanonical, directed: false)
        }
        relations[rel.id] = rel
        return rel
    }

    // MARK: - 查询

    public func entity(named name: String) -> KGEntity? {
        lock.lock(); defer { lock.unlock() }
        let key = canonicalName(name).lowercased()
        guard let canonical = nameIndex[key] else { return nil }
        return entities[canonical]
    }

    public func relationship(id: String) -> KGRelationship? {
        lock.lock(); defer { lock.unlock() }
        return relations[id]
    }

    /// 节点度数（去重邻居数），作为重要性/中心性代理指标。
    public func degree(of name: String) -> Int {
        lock.lock(); defer { lock.unlock() }
        let key = canonicalName(name).lowercased()
        guard let canonical = nameIndex[key] else { return 0 }
        return graph.neighborsForVertex(canonical)?.count ?? 0
    }

    /// 一跳邻居（实体名列表）。
    public func neighbors(of name: String) -> [String] {
        lock.lock(); defer { lock.unlock() }
        let key = canonicalName(name).lowercased()
        guard let canonical = nameIndex[key] else { return [] }
        return graph.neighborsForVertex(canonical) ?? []
    }

    /// 与某实体相关的全部关系。
    public func incidentRelationships(of name: String) -> [KGRelationship] {
        lock.lock(); defer { lock.unlock() }
        let key = canonicalName(name).lowercased()
        guard let canonical = nameIndex[key] else { return [] }
        return relations.values.filter {
            $0.srcEntity.lowercased() == key || $0.dstEntity.lowercased() == key
                || canonicalName($0.srcEntity).lowercased() == key
                || canonicalName($0.dstEntity).lowercased() == key
        }
    }

    /// 关系 rank = 两端点度数和（用于查询排序的重要性代理）。
    public func relationRank(_ rel: KGRelationship) -> Int {
        degree(of: rel.srcEntity) + degree(of: rel.dstEntity)
    }

    public func allEntities() -> [KGEntity] {
        lock.lock(); defer { lock.unlock() }
        return Array(entities.values)
    }

    public func allRelationships() -> [KGRelationship] {
        lock.lock(); defer { lock.unlock() }
        return Array(relations.values)
    }

    // MARK: - 快照

    private struct Snapshot: Codable {
        var entities: [KGEntity]
        var relationships: [KGRelationship]
    }

    /// 立即落盘快照。
    public func flush() {
        lock.lock(); defer { lock.unlock() }
        saveLocked(force: true)
    }

    private func markDirtyLocked() {
        dirty = true
        guard pendingSave == nil else { return }
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            self.lock.lock()
            self.pendingSave = nil
            if self.dirty {
                self.dirty = false
                self.writeSnapshotLocked()
            }
            self.lock.unlock()
        }
        pendingSave = work
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 1.0, execute: work)
    }

    private func saveLocked(force: Bool) {
        guard force else {
            markDirtyLocked()
            return
        }
        dirty = false
        pendingSave?.cancel()
        pendingSave = nil
        writeSnapshotLocked()
    }

    private func writeSnapshotLocked() {
        let snapshot = Snapshot(entities: Array(entities.values),
                                relationships: Array(relations.values))
        do {
            let data = try JSONEncoder().encode(snapshot)
            try AtomicFile.write(data, to: snapshotPath)
        } catch {
            AgentLog.warning("知识图谱快照写入失败: \(error.localizedDescription)")
        }
    }

    private func restoreLocked() -> Bool {
        guard FileManager.default.fileExists(atPath: snapshotPath),
              let data = try? Data(contentsOf: URL(fileURLWithPath: snapshotPath)),
              let snapshot = try? JSONDecoder().decode(Snapshot.self, from: data) else {
            return false
        }
        rebuildLocked(entities: snapshot.entities, relationships: snapshot.relationships)
        return true
    }
}
