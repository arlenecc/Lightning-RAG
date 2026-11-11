import Foundation
import SwiftGraph

/// 寻路算法。
public enum KGPathAlgorithm: String, Codable, Sendable {
    case dfs, bfs, dijkstra
}

/// 子图结果（BFS 展开到指定深度得到的实体与关系集合）。
public struct KGSubgraph: Sendable {
    public let entities: [KGEntity]
    public let relationships: [KGRelationship]
}

/// 寻路结果：节点序列 + 途经关系（含中间节点与边的信息）。
public struct KGPath: Sendable {
    public let nodes: [String]
    public let relationships: [KGRelationship]
}

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
    /// 无向实体对 → 关系 id（同一对实体只保留一条边）。
    private var pairIndex: [String: String] = [:]

    /// 读取数据库当前数据版本，用于快照与数据库的一致性校验。
    public typealias RevisionProvider = @Sendable () -> Int?
    private let revisionProvider: RevisionProvider?

    private let lock = NSRecursiveLock()
    private var dirty = false
    private var pendingSave: DispatchWorkItem?

    public init(snapshotPath: String, revisionProvider: RevisionProvider? = nil) {
        self.snapshotPath = snapshotPath
        self.revisionProvider = revisionProvider
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

    /// 优先从快照恢复；无快照或快照落后于数据库时从存储层全量加载并建立图谱，
    /// 随后保存快照。
    public func loadOrRestore(from store: KnowledgeGraphStore) throws {
        lock.lock(); defer { lock.unlock() }
        let expected = revisionProvider?() ?? 0
        if restoreLocked(expectedRevision: expected) {
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
        pairIndex = [:]
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

    /// 用数据库中的最终结果覆盖实体（增量更新用，不再做二次合并）。
    public func replaceEntity(_ entity: KGEntity) {
        lock.lock(); defer { lock.unlock() }
        let canonical = canonicalName(entity.name)
        guard !canonical.isEmpty else { return }
        entities[canonical] = entity
        nameIndex[canonical.lowercased()] = canonical
        if !graph.vertexInGraph(vertex: canonical) {
            _ = graph.addVertex(canonical)
        }
        markDirtyLocked()
    }

    /// 用数据库中的最终结果覆盖关系（增量更新用，不再做二次合并）。
    /// 同一实体对若已存在另一条关系 id，会被新结果替换。
    public func replaceRelationship(_ rel: KGRelationship) {
        lock.lock(); defer { lock.unlock() }
        let src = canonicalName(rel.srcEntity)
        let dst = canonicalName(rel.dstEntity)
        guard !src.isEmpty, !dst.isEmpty else { return }
        for name in [src, dst] where nameIndex[name.lowercased()] == nil {
            let placeholder = KGEntity(id: UUID().uuidString, name: name,
                                       type: "UNKNOWN", description: "",
                                       properties: rel.properties, sourceIDs: rel.sourceIDs,
                                       createdAt: Date().timeIntervalSince1970)
            _ = addEntityLocked(placeholder)
        }
        let srcCanonical = nameIndex[src.lowercased()] ?? src
        let dstCanonical = nameIndex[dst.lowercased()] ?? dst
        if !graph.edgeExists(from: srcCanonical, to: dstCanonical) {
            graph.addEdge(from: srcCanonical, to: dstCanonical, directed: false)
        }
        let pair = KGRelationship.pairKey(srcCanonical, dstCanonical)
        if let existingID = pairIndex[pair], existingID != rel.id {
            relations.removeValue(forKey: existingID)
        }
        relations[rel.id] = rel
        pairIndex[pair] = rel.id
        markDirtyLocked()
    }

    /// 删除重复关系行（增量合并后保留的那条由 `replaceRelationship` 写入）。
    public func removeRelationship(id: String) {
        lock.lock(); defer { lock.unlock() }
        guard let removed = relations.removeValue(forKey: id) else { return }
        let src = canonicalName(removed.srcEntity)
        let dst = canonicalName(removed.dstEntity)
        let pair = KGRelationship.pairKey(src, dst)
        if pairIndex[pair] == id {
            pairIndex.removeValue(forKey: pair)
            // 理论上同一实体对已有存活关系；没有的话保留边，避免误删其它关系。
            if let survivor = relations.values.first(where: {
                KGRelationship.pairKey($0.srcEntity, $0.dstEntity) == pair
            }) {
                pairIndex[pair] = survivor.id
            }
        }
        markDirtyLocked()
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
                // 类型按 "/" 合并，去掉 UNKNOWN（仅当有更具体类型时）。
                var types = Set(existing.type.split(separator: "/").map(String.init))
                    .union(entity.type.split(separator: "/").map(String.init))
                    .filter { !$0.isEmpty }
                if types.contains("UNKNOWN") && types.count > 1 {
                    types.remove("UNKNOWN")
                }
                let type = types.isEmpty ? existing.type : types.sorted().joined(separator: "/")
                var description = existing.description
                let incoming = entity.description.trimmingCharacters(in: .whitespacesAndNewlines)
                if !incoming.isEmpty, !description.contains(incoming) {
                    description = description.isEmpty ? incoming : description + "\n" + incoming
                }
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
        // 同一对实体只保留一条关系边：不同切片措辞不同也合并 keywords/描述/来源。
        let pair = KGRelationship.pairKey(srcCanonical, dstCanonical)
        if let existingID = pairIndex[pair], let existing = relations[existingID] {
            let merged = Self.merge(existing, rel)
            relations[existingID] = merged
            return merged
        }
        relations[rel.id] = rel
        pairIndex[pair] = rel.id
        return rel
    }

    /// 合并同一对实体的两条关系：keywords/sourceIDs 取并集，描述去重拼接，
    /// weight 取较大值（累加由构建阶段的证据计数负责）。
    private static func merge(_ a: KGRelationship, _ b: KGRelationship) -> KGRelationship {
        var merged = a
        var keywords = Set(a.relationType.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
        }.filter { !$0.isEmpty })
        for kw in b.relationType.split(separator: ",") {
            let trimmed = kw.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { keywords.insert(trimmed) }
        }
        merged.relationType = keywords.sorted().joined(separator: ",")
        let incoming = b.description.trimmingCharacters(in: .whitespacesAndNewlines)
        if !incoming.isEmpty, !merged.description.contains(incoming) {
            merged.description = merged.description.isEmpty
                ? incoming : merged.description + "\n" + incoming
        }
        var sources = Set(a.sourceIDs)
        sources.formUnion(b.sourceIDs)
        merged.sourceIDs = sources.sorted()
        var props = a.properties
        for (k, v) in b.properties where props[k] == nil { props[k] = v }
        merged.properties = props
        merged.weight = max(a.weight, b.weight)
        if b.createdAt > 0 {
            merged.createdAt = merged.createdAt > 0
                ? min(merged.createdAt, b.createdAt) : b.createdAt
        }
        merged.updatedAt = max(a.updatedAt, b.updatedAt)
        return merged
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

    // MARK: - 图谱算法（删除 / 子图 / 寻路）

    /// 删除实体节点及其所有关联边（关系端点指向它的无向边一并删除）。
    public func deleteEntity(_ name: String) {
        lock.lock(); defer { lock.unlock() }
        let key = canonicalName(name).lowercased()
        guard let canonical = nameIndex[key] else { return }
        graph.removeVertex(canonical)
        entities.removeValue(forKey: canonical)
        nameIndex.removeValue(forKey: key)
        let removedIDs = relations.values
            .filter { $0.srcEntity.lowercased() == key || $0.dstEntity.lowercased() == key }
            .map(\.id)
        for id in removedIDs {
            if let rel = relations.removeValue(forKey: id) {
                pairIndex.removeValue(forKey: rel.pairKey)
            }
        }
        markDirtyLocked()
    }

    /// 以 root 为根、BFS 展开到指定跳数 `depth` 的子图。
    /// `edgeTypes` 非空时只保留关系类型命中的边（空表示不过滤）。
    public func subgraph(root: String, depth: Int,
                         edgeTypes: [String]? = nil) -> KGSubgraph {
        lock.lock(); defer { lock.unlock() }
        let rootKey = canonicalName(root).lowercased()
        guard let rootName = nameIndex[rootKey] else {
            return KGSubgraph(entities: [], relationships: [])
        }
        var visited = Set<String>([rootName])
        var queue: [(String, Int)] = [(rootName, 0)]
        var entityNames = [rootName]
        var rels: [KGRelationship] = []
        while !queue.isEmpty {
            let (node, d) = queue.removeFirst()
            guard d < depth else { continue }
            for nb in graph.neighborsForVertex(node) ?? [] {
                if !visited.contains(nb) {
                    visited.insert(nb)
                    entityNames.append(nb)
                    queue.append((nb, d + 1))
                }
                let a = node.lowercased(), b = nb.lowercased()
                for rel in relations.values where
                    (rel.srcEntity.lowercased() == a && rel.dstEntity.lowercased() == b)
                    || (rel.dstEntity.lowercased() == a && rel.srcEntity.lowercased() == b) {
                    if let types = edgeTypes, !types.isEmpty,
                       !types.contains(rel.relationType) { continue }
                    if !rels.contains(where: { $0.id == rel.id }) { rels.append(rel) }
                }
            }
        }
        let ents = entityNames.compactMap { entities[$0] }
        return KGSubgraph(entities: ents, relationships: rels)
    }

    /// 两实体间寻路：dfs（任意路径）/ bfs（最短跳数）/ dijkstra（按关系权重优先）。
    public func shortestPath(from a: String, to b: String,
                             algorithm: KGPathAlgorithm) -> KGPath {
        lock.lock(); defer { lock.unlock() }
        let aKey = canonicalName(a).lowercased()
        let bKey = canonicalName(b).lowercased()
        guard let sa = nameIndex[aKey], let sb = nameIndex[bKey] else {
            return KGPath(nodes: [], relationships: [])
        }
        switch algorithm {
        case .bfs: return bfsPath(sa, sb)
        case .dfs: return dfsPath(sa, sb)
        case .dijkstra: return dijkstraPath(sa, sb)
        }
    }

    private func bfsPath(_ src: String, _ dst: String) -> KGPath {
        guard src != dst else { return KGPath(nodes: [src], relationships: []) }
        var queue: [String] = [src]
        var visited = Set([src])
        var parent: [String: String] = [:]
        while !queue.isEmpty {
            let node = queue.removeFirst()
            if node == dst { break }
            for nb in graph.neighborsForVertex(node) ?? [] where !visited.contains(nb) {
                visited.insert(nb)
                parent[nb] = node
                queue.append(nb)
            }
        }
        guard parent[dst] != nil else { return KGPath(nodes: [], relationships: []) }
        var path = [dst]; var cur = dst
        while cur != src { cur = parent[cur]!; path.append(cur) }
        path.reverse()
        return KGPath(nodes: path, relationships: relationshipsOnPath(path))
    }

    private func dfsPath(_ src: String, _ dst: String) -> KGPath {
        guard src != dst else { return KGPath(nodes: [src], relationships: []) }
        var visited = Set<String>()
        var stack: [String] = []
        var found: [String]?
        func dfs(_ node: String) {
            if found != nil { return }
            visited.insert(node)
            stack.append(node)
            if node == dst { found = stack; return }
            for nb in graph.neighborsForVertex(node) ?? [] where !visited.contains(nb) {
                dfs(nb)
                if found != nil { return }
            }
            stack.removeLast()
            visited.remove(node)
        }
        dfs(src)
        guard let p = found else { return KGPath(nodes: [], relationships: []) }
        return KGPath(nodes: p, relationships: relationshipsOnPath(p))
    }

    private func dijkstraPath(_ src: String, _ dst: String) -> KGPath {
        guard src != dst else { return KGPath(nodes: [src], relationships: []) }
        var dist: [String: Double] = [src: 0]
        var parent: [String: String] = [:]
        var visited = Set<String>()
        while true {
            let u = dist.keys.filter { !visited.contains($0) }
                .min { (dist[$0] ?? .infinity) < (dist[$1] ?? .infinity) }
            guard let u else { break }
            if u == dst { break }
            visited.insert(u)
            for nb in graph.neighborsForVertex(u) ?? [] {
                let cost = edgeCost(u, nb)
                let nd = (dist[u] ?? .infinity) + cost
                if nd < (dist[nb] ?? .infinity) {
                    dist[nb] = nd
                    parent[nb] = u
                }
            }
        }
        guard parent[dst] != nil else { return KGPath(nodes: [], relationships: []) }
        var path = [dst]; var cur = dst
        while cur != src { cur = parent[cur]!; path.append(cur) }
        path.reverse()
        return KGPath(nodes: path, relationships: relationshipsOnPath(path))
    }

    /// 边代价：关系权重越高越「近」（cost = 1 / (weight + 1)），优先走高证据边。
    private func edgeCost(_ a: String, _ b: String) -> Double {
        var minCost = Double.infinity
        let la = a.lowercased(), lb = b.lowercased()
        for rel in relations.values where
            (rel.srcEntity.lowercased() == la && rel.dstEntity.lowercased() == lb)
            || (rel.dstEntity.lowercased() == la && rel.srcEntity.lowercased() == lb) {
            let w = max(0.0, rel.weight)
            minCost = min(minCost, 1.0 / (w + 1.0))
        }
        return minCost == .infinity ? 1.0 : minCost
    }

    private func relationshipsOnPath(_ path: [String]) -> [KGRelationship] {
        var rels: [KGRelationship] = []
        for i in 0..<path.count - 1 {
            let a = path[i].lowercased(), b = path[i + 1].lowercased()
            for rel in relations.values where
                (rel.srcEntity.lowercased() == a && rel.dstEntity.lowercased() == b)
                || (rel.dstEntity.lowercased() == a && rel.srcEntity.lowercased() == b) {
                rels.append(rel)
            }
        }
        return rels
    }

    // MARK: - 快照

    private struct Snapshot: Codable {
        static let currentSchemaVersion = 2
        var schemaVersion: Int?
        var revision: Int?
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
        let snapshot = Snapshot(schemaVersion: Snapshot.currentSchemaVersion,
                                revision: revisionProvider?() ?? 0,
                                entities: Array(entities.values),
                                relationships: Array(relations.values))
        do {
            let data = try JSONEncoder().encode(snapshot)
            try AtomicFile.write(data, to: snapshotPath)
        } catch {
            AgentLog.warning("知识图谱快照写入失败: \(error.localizedDescription)")
        }
    }

    private func restoreLocked(expectedRevision: Int) -> Bool {
        guard FileManager.default.fileExists(atPath: snapshotPath),
              let data = try? Data(contentsOf: URL(fileURLWithPath: snapshotPath)),
              let snapshot = try? JSONDecoder().decode(Snapshot.self, from: data) else {
            return false
        }
        guard (snapshot.schemaVersion ?? 1) == Snapshot.currentSchemaVersion else {
            AgentLog.info("知识图谱快照 schema 版本不匹配，改为从数据库重建")
            return false
        }
        guard (snapshot.revision ?? 0) == expectedRevision else {
            AgentLog.info("知识图谱快照版本 \(snapshot.revision ?? 0) 落后于数据库 "
                + "\(expectedRevision)，改为从数据库重建")
            return false
        }
        rebuildLocked(entities: snapshot.entities, relationships: snapshot.relationships)
        return true
    }
}
