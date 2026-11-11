import Foundation

// MARK: - 数据模型

/// 图谱实体（与 Python `graph_memory.py` 的 entity 结构一致）。
public struct GraphEntity: Codable, Sendable, Equatable, Identifiable {
    public var name: String
    public var type: String
    public var observations: [String]
    public var createdAt: String
    /// 与 `observations` 平行的写入时间戳（epoch 秒）。旧库没有该字段时
    /// 全部按 0（时间未知、视为最旧）处理，保证向后兼容。
    public var observationDates: [Double]

    public var id: String { name.lowercased() }

    public init(name: String, type: String = "", observations: [String] = [],
                createdAt: String = "", observationDates: [Double] = []) {
        self.name = name
        self.type = type
        self.observations = observations
        self.createdAt = createdAt
        self.observationDates = Self.alignedDates(observationDates,
                                                  count: observations.count)
    }

    enum CodingKeys: String, CodingKey {
        case name, type, observations
        case createdAt = "created_at"
        case observationDates = "observation_dates"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        name = try c.decode(String.self, forKey: .name)
        type = try c.decode(String.self, forKey: .type)
        observations = try c.decode([String].self, forKey: .observations)
        createdAt = try c.decode(String.self, forKey: .createdAt)
        let dates = try c.decodeIfPresent([Double].self, forKey: .observationDates) ?? []
        observationDates = Self.alignedDates(dates, count: observations.count)
    }

    /// 工具输出（`get_entity` / `create_entity`）不暴露内部时间戳：
    /// 200 条观察会平白多出几千字符，撑大工具结果。持久化由
    /// `GraphMemoryStore.writeLocked()` 手写 `observation_dates` 字段完成。
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(name, forKey: .name)
        try c.encode(type, forKey: .type)
        try c.encode(observations, forKey: .observations)
        try c.encode(createdAt, forKey: .createdAt)
    }

    /// 时间戳数组与观察数组严格对齐：不足补 0，超出截断。
    static func alignedDates(_ dates: [Double], count: Int) -> [Double] {
        guard count > 0 else { return [] }
        if dates.count == count { return dates }
        if dates.count > count { return Array(dates.suffix(count)) }
        return dates + Array(repeating: 0, count: count - dates.count)
    }
}

/// 有向关系：source --[label]--> target。
public struct GraphRelation: Codable, Sendable, Equatable {
    public var source: String
    public var target: String
    public var label: String
    public var created: String?

    public init(source: String, target: String, label: String, created: String? = nil) {
        self.source = source
        self.target = target
        self.label = label
        self.created = created
    }
}

/// 一次观察检索命中。
public struct GraphObservationHit: Sendable, Equatable {
    public var entity: String
    public var text: String
    public var score: Double
    /// 观察写入时间（epoch 秒）；旧库无时间戳时为 0。
    public var timestamp: Double

    public init(entity: String, text: String, score: Double, timestamp: Double = 0) {
        self.entity = entity
        self.text = text
        self.score = score
        self.timestamp = timestamp
    }
}

/// 邻居查询结果（中心实体 + 半径内的实体与关系）。
public struct GraphNeighborhood: Sendable, Equatable {
    public var center: GraphEntity?
    public var entities: [GraphEntity]
    public var relations: [GraphRelation]

    public init(center: GraphEntity?, entities: [GraphEntity], relations: [GraphRelation]) {
        self.center = center
        self.entities = entities
        self.relations = relations
    }
}

// MARK: - 图谱存储

/// 线程安全的知识图谱存储：实体 + 关系 + 观察。
///
/// 持久化文件为单个 JSON：
/// `{"entities": {"<lower-name>": {name,type,observations,created_at}},
///   "relations": [{source,target,label,created}]}`
/// 与 Python 版本格式兼容。观察向量只缓存在内存中，首次检索时懒计算；
/// embedding 不可用或失败时自动回退到关键词子串匹配。
public final class GraphMemoryStore {
    public let path: String

    /// 单实体观察上限（保留最新）。
    public var maxObservationsPerEntity: Int = 200
    /// 实体总数上限（按 created_at 淘汰最旧）。
    public var maxEntities: Int = 5_000
    /// 关系总数上限（淘汰最旧）。
    public var maxRelations: Int = 20_000
    /// 实体消歧余弦阈值：新实体名与已有实体名的向量余弦 ≥ 该值才判定为同一实体
    /// 并合并（保守默认，宁漏勿错——误合并是持久性伤害，会逐轮复利）。
    public var entityResolutionThreshold: Double = 0.90

    private let embedding: EmbeddingProvider?
    private let lock = NSRecursiveLock()
    private var dirty = false
    private var pendingSave: DispatchWorkItem?
    private var entities: [String: GraphEntity] = [:]   // key = name.lowercased()
    private var relations: [GraphRelation] = []
    private var vectors: [String: [Float]] = [:]        // key = vectorKey(entity, text)
    private var nameVectors: [String: [Float]] = [:]    // key = name.lowercased()（实体名，懒计算）
    private var embeddingDisabled = false

    public init(path: String, embedding: EmbeddingProvider? = nil) {
        self.path = path
        self.embedding = embedding
        load()
    }

    /// 便捷构造：`<workspace>/.agent/long_memory.graph.json`。
    public convenience init(workspace: String, embedding: EmbeddingProvider? = nil) {
        self.init(path: (workspace as NSString)
            .appendingPathComponent(".agent")
            .appending("/long_memory.graph.json"), embedding: embedding)
    }

    private static let sharedLock = NSLock()
    private static var sharedStores: [String: GraphMemoryStore] = [:]

    /// 同一 workspace 的默认共享实例，确保多个工具看到同一份内存状态。
    public static func shared(workspace: String,
                              embedding: EmbeddingProvider? = nil) -> GraphMemoryStore {
        let path = (workspace as NSString)
            .appendingPathComponent(".agent")
            .appending("/long_memory.graph.json")
        sharedLock.lock(); defer { sharedLock.unlock() }
        if let existing = sharedStores[path] { return existing }
        let store = GraphMemoryStore(path: path, embedding: embedding)
        sharedStores[path] = store
        return store
    }

    /// 进程退出/切换 workspace 前确保共享图谱已落盘。
    public static func flushShared() {
        sharedLock.lock()
        let stores = Array(sharedStores.values)
        sharedLock.unlock()
        for store in stores { store.flush() }
    }

    /// 关闭 Agent 时释放该 workspace 的共享实例。
    ///
    /// 只从注册表移除并落盘：仍持有该实例的工具可以继续使用，
    /// 但下次 `shared()` 会重新从磁盘加载，避免跨 workspace 切换后
    /// 旧图谱常驻内存。
    public static func releaseShared(workspace: String) {
        let path = (workspace as NSString)
            .appendingPathComponent(".agent")
            .appending("/long_memory.graph.json")
        sharedLock.lock()
        let store = sharedStores.removeValue(forKey: path)
        sharedLock.unlock()
        store?.flush()
    }

    // MARK: - 实体消歧

    /// 把候选实体名消歧到库内已有实体的**规范名**。
    ///
    /// 判定顺序：
    /// 1. 精确匹配（大小写不敏感）→ 直接返回已有实体的 `name`；
    /// 2. 语义匹配：候选名与所有已有实体名统一用 `embedDocuments` 编码后做余弦，
    ///    取最高分，只有 ≥ `entityResolutionThreshold` 才合并（保守，宁漏勿错）；
    /// 3. type 冲突保护：候选 type 与命中实体 type 都非空且不同时拒绝合并；
    /// 4. embedding 不可用 / 失败 → 回退到精确匹配（即返回原名，不语义合并）。
    ///
    /// 返回的规范名随后交给 `upsertEntity` / `addObservations` / `createRelation`，
    /// 确保「同一实体」只保留一条记录，新抽取的内容以追加观察 / 更新关系的方式并入。
    /// - Parameter semantic: 是否启用向量语义消歧。插入阶段（事实抽取）默认 `false`，
    ///   只做字符全匹配——避免每次入库都重算全库实体名向量（大图下开销不可接受）；
    ///   向量语义合并交给**手动触发**的离线流程，详见 `mergeFactDraft`。
    public func resolveEntity(name: String, type: String = "",
                              semantic: Bool = false) async -> String {
        let normalized = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return name }

        // 1) 精确命中 + 候选名快照（同步方法内加锁，避免 async 上下文直接持有锁）。
        let snapshot = exactMatchAndCandidates(normalized.lowercased())
        if let exact = snapshot.exact { return exact }
        // 字符全匹配优先（插入阶段默认路径）：未命中也直接返回原名，不跑向量。
        guard semantic else { return normalized }
        let candidateNames = snapshot.candidates
        guard !candidateNames.isEmpty, let embedding, !isEmbeddingDisabled else {
            return normalized
        }

        // 2) 语义匹配（embedding 失败则回退精确匹配，不阻断入库）。
        // 实体名消歧是「名字 vs 名字」的对称比较，候选与已有名字统一用
        // `embedDocuments` 编码，避免 query/document 前缀造成系统性偏差。
        do {
            let queryVector = (try await embedding.embedDocuments([normalized])).first ?? []
            guard !queryVector.isEmpty else { return normalized }
            try await ensureNameVectors(candidateNames, embedding: embedding)
            if let best = bestSemanticMatch(queryVector: queryVector,
                                            candidates: candidateNames, type: type) {
                return best
            }
        } catch {
            AgentLog.warning("graph_memory: 实体消歧不可用，回退精确匹配: \(error.localizedDescription)")
        }
        return normalized
    }

    /// 同步：一次加锁返回精确命中的规范名与全部已有实体名快照。
    private func exactMatchAndCandidates(_ key: String)
        -> (exact: String?, candidates: [String]) {
        lock.lock(); defer { lock.unlock() }
        let exact = entities[key]?.name
        return (exact, entities.values.map(\.name))
    }

    /// 同步：在锁内完成余弦比较，返回最高分且过阈值的规范名。
    private func bestSemanticMatch(queryVector: [Float], candidates: [String],
                                   type: String) -> String? {
        lock.lock(); defer { lock.unlock() }
        var bestName: String?
        var bestScore = -1.0
        for candidate in candidates {
            guard let vector = nameVectors[candidate.lowercased()] else { continue }
            // type 冲突：person vs project 这类不同类实体，即便名字相似也不合并。
            if let existing = entities[candidate.lowercased()],
               !type.isEmpty, !existing.type.isEmpty,
               type.lowercased() != existing.type.lowercased() {
                continue
            }
            let score = Self.cosine(queryVector, vector)
            if score >= entityResolutionThreshold, score > bestScore {
                bestName = candidate
                bestScore = score
            }
        }
        return bestName
    }

    /// 懒计算实体名向量并缓存（key = name.lowercased()）。
    /// 锁操作封装在同步辅助方法里，避免 async 上下文直接持有锁。
    private func ensureNameVectors(_ names: [String], embedding: EmbeddingProvider) async throws {
        let missing = missingNameVectorNames(names)
        guard !missing.isEmpty else { return }

        var embeddedByName: [String: [Float]] = [:]
        var offset = 0
        while offset < missing.count {
            try Task.checkCancellation()
            let batch = Array(missing[offset..<min(offset + 32, missing.count)])
            let embedded = try await embedding.embedDocuments(batch)
            guard embedded.count == batch.count else {
                throw AgentError.storage("graph_memory 实体名 embedding 数量不一致")
            }
            for (name, vector) in zip(batch, embedded) {
                embeddedByName[name.lowercased()] = vector
            }
            offset += batch.count
        }
        cacheNameVectors(embeddedByName)
    }

    /// 同步：返回尚未缓存向量的实体名（去重）。
    private func missingNameVectorNames(_ names: [String]) -> [String] {
        lock.lock(); defer { lock.unlock() }
        var missing: [String] = []
        var seen = Set<String>()
        for name in names {
            let key = name.lowercased()
            if nameVectors[key] == nil, seen.insert(key).inserted {
                missing.append(name)
            }
        }
        return missing
    }

    /// 同步：写回实体名向量缓存（不覆盖已有值）。
    private func cacheNameVectors(_ vectors: [String: [Float]]) {
        lock.lock(); defer { lock.unlock() }
        for (key, vector) in vectors where nameVectors[key] == nil {
            nameVectors[key] = vector
        }
    }

    // MARK: - 事实抽取入库（消歧 + 合并）

    /// 一批三元组入库的统一入口：先对每个实体名消歧到规范名，再写入。
    ///
    /// 核心保证（对应「相同实体不重复入库」）：
    /// - 精确同名或语义同指的实体，合并到**已有**记录：只追加观察、不新建实体；
    /// - 关系端点直接查本批「原名 → 规范名」映射，保证无论 embedding 是否可用，
    ///   关系都指向合并后的实体，且不会触发 `createRelation` 补建空实体；
    /// - 两端若被消歧到同一实体，跳过该自环关系。
    ///
    /// 调用前应已过 `FactExtractionValidator.sanitize`（关系端点必落在本批实体内）。
    ///
    /// - Parameter semantic: 是否对关系端点做向量语义消歧。`false`（默认）时只做
    ///   字符全匹配，保证插入阶段不重算向量、不过度合并（宁漏勿错，且快）；
    ///   语义合并应走手动触发的离线流程。
    @discardableResult
    public func mergeFactDraft(_ draft: FactExtractionDraft,
                               semantic: Bool = false) async -> (observations: Int,
                                                                 relations: Int) {
        var canonicalNames: [String: String] = [:]
        var observationCount = 0
        for entity in draft.entities {
            let canonical = await resolveEntity(name: entity.name, type: entity.type,
                                                semantic: semantic)
            canonicalNames[entity.name.lowercased()] = canonical
            // upsert 保留已有 type（addObservations 只建空 type 实体），两者都幂等。
            upsertEntity(canonical, type: entity.type)
            addObservations(canonical, entity.observations)
            observationCount += entity.observations.count
        }

        var relationCount = 0
        for relation in draft.relations {
            let source = canonicalNames[relation.source.lowercased()] ?? relation.source
            let target = canonicalNames[relation.target.lowercased()] ?? relation.target
            guard source.lowercased() != target.lowercased() else { continue }
            if createRelation(source: source, target: target, label: relation.label) {
                relationCount += 1
            }
        }
        return (observationCount, relationCount)
    }

    // MARK: - 实体

    /// 创建实体；同名（大小写不敏感）时合并观察。
    @discardableResult
    public func upsertEntity(_ name: String, type entityType: String = "",
                             observations: [String] = []) -> GraphEntity {
        let normalized = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let cleaned = observations
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !normalized.isEmpty else {
            return GraphEntity(name: "", type: entityType, observations: [], createdAt: Self.now())
        }

        lock.lock()
        let key = normalized.lowercased()
        var entity: GraphEntity
        if var existing = entities[key] {
            if entityType != "" && existing.type.isEmpty {
                existing.type = entityType
            }
            var known = Set(existing.observations)
            var appended = 0
            for obs in cleaned where !known.contains(obs) {
                existing.observations.append(obs)
                known.insert(obs)
                appended += 1
            }
            if appended > 0 {
                existing.observationDates.append(contentsOf: Self.stampSequence(appended))
            }
            entity = existing
            entities[key] = existing
        } else {
            var known = Set<String>()
            var unique: [String] = []
            for obs in cleaned where !known.contains(obs) {
                unique.append(obs)
                known.insert(obs)
            }
            entity = GraphEntity(name: normalized, type: entityType,
                                 observations: unique, createdAt: Self.now(),
                                 observationDates: Self.stampSequence(unique.count))
            entities[key] = entity
        }
        pruneLocked()
        saveLocked()
        let result = entity
        lock.unlock()
        return result
    }

    @discardableResult
    public func deleteEntity(_ name: String) -> Bool {
        let key = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !key.isEmpty else { return false }
        lock.lock()
        guard entities.removeValue(forKey: key) != nil else {
            lock.unlock()
            return false
        }
        relations.removeAll {
            $0.source.lowercased() == key || $0.target.lowercased() == key
        }
        vectors = vectors.filter { !$0.key.hasPrefix(key + "|") }
        nameVectors.removeValue(forKey: key)
        saveLocked()
        lock.unlock()
        return true
    }

    public func getEntity(_ name: String) -> GraphEntity? {
        let key = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !key.isEmpty else { return nil }
        lock.lock(); defer { lock.unlock() }
        return entities[key]
    }

    public func listEntities() -> [GraphEntity] {
        lock.lock(); defer { lock.unlock() }
        return entities.values.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    // MARK: - 观察

    /// 添加观察；实体不存在时自动创建。返回实际新增条数。
    @discardableResult
    public func addObservations(_ name: String, _ observations: [String]) -> Int {
        let normalized = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let cleaned = observations
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !normalized.isEmpty, !cleaned.isEmpty else { return 0 }

        lock.lock()
        let key = normalized.lowercased()
        var entity: GraphEntity
        if let existing = entities[key] {
            entity = existing
        } else {
            entity = GraphEntity(name: normalized, type: "", observations: [],
                                 createdAt: Self.now())
        }
        var known = Set(entity.observations)
        var added = 0
        for obs in cleaned where !known.contains(obs) {
            entity.observations.append(obs)
            known.insert(obs)
            added += 1
        }
        if added > 0 {
            entity.observationDates.append(contentsOf: Self.stampSequence(added))
        }
        entities[key] = entity
        pruneLocked()
        saveLocked()
        lock.unlock()
        return added
    }

    // MARK: - 关系

    /// 创建有向边；两端实体不存在时自动创建。重复边返回 false。
    @discardableResult
    public func createRelation(source: String, target: String, label: String) -> Bool {
        let src = source.trimmingCharacters(in: .whitespacesAndNewlines)
        let tgt = target.trimmingCharacters(in: .whitespacesAndNewlines)
        let lbl = label.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !src.isEmpty, !tgt.isEmpty, !lbl.isEmpty else { return false }

        lock.lock()
        for name in [src, tgt] where entities[name.lowercased()] == nil {
            entities[name.lowercased()] = GraphEntity(name: name, type: "",
                                                      observations: [], createdAt: Self.now())
        }
        let duplicate = relations.contains {
            $0.source.lowercased() == src.lowercased()
                && $0.target.lowercased() == tgt.lowercased()
                && $0.label.lowercased() == lbl.lowercased()
        }
        if duplicate {
            lock.unlock()
            return false
        }
        relations.append(GraphRelation(source: src, target: tgt, label: lbl))
        pruneLocked()
        saveLocked()
        lock.unlock()
        return true
    }

    public func listRelations() -> [GraphRelation] {
        lock.lock(); defer { lock.unlock() }
        return relations
    }

    /// 返回中心实体及其 `depth` 跳内的邻居（默认 1 跳）。
    public func neighbors(of name: String, depth: Int = 1) -> GraphNeighborhood {
        let key = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !key.isEmpty else {
            return GraphNeighborhood(center: nil, entities: [], relations: [])
        }
        let limit = max(1, min(depth, 3))

        lock.lock(); defer { lock.unlock() }
        guard let center = entities[key] else {
            return GraphNeighborhood(center: nil, entities: [], relations: [])
        }
        var visited: Set<String> = [key]
        var frontier: Set<String> = [key]
        var collectedRelations: [GraphRelation] = []
        var seenRelations = Set<String>()

        for _ in 0..<limit {
            var next: Set<String> = []
            for relation in relations {
                let src = relation.source.lowercased()
                let tgt = relation.target.lowercased()
                guard frontier.contains(src) || frontier.contains(tgt) else { continue }
                let relKey = "\(src)|\(tgt)|\(relation.label.lowercased())"
                if seenRelations.insert(relKey).inserted {
                    collectedRelations.append(relation)
                }
                if !visited.contains(src) { next.insert(src) }
                if !visited.contains(tgt) { next.insert(tgt) }
            }
            if next.isEmpty { break }
            visited.formUnion(next)
            frontier = next
        }

        let found = visited.compactMap { entities[$0] }
            .sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
        return GraphNeighborhood(center: center, entities: found, relations: collectedRelations)
    }

    // MARK: - 检索

    /// 语义检索观察；向量不可用时回退为关键词子串匹配。
    public func search(query: String, topK: Int = 8) async -> [GraphObservationHit] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return [] }
        let limit = max(1, topK)

        let pairs: [(entity: String, text: String, timestamp: Double)] = {
            lock.lock(); defer { lock.unlock() }
            return entities.values
                .sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
                .flatMap { entity in
                    entity.observations.enumerated().map { index, text in
                        (entity: entity.name, text: text,
                         timestamp: index < entity.observationDates.count
                            ? entity.observationDates[index] : 0)
                    }
                }
        }()
        guard !pairs.isEmpty else { return [] }

        if let embedding, !isEmbeddingDisabled {
            do {
                try await ensureVectors(pairs, embedding: embedding)
                let queryVector = try await embedding.embedQuery(q)
                var hits: [GraphObservationHit] = []
                lock.lock()
                for pair in pairs {
                    guard let vector = vectors[Self.vectorKey(pair.entity, pair.text)] else { continue }
                    let score = Self.cosine(queryVector, vector)
                    if score >= 0.5 {
                        hits.append(GraphObservationHit(entity: pair.entity,
                                                        text: pair.text,
                                                        score: score,
                                                        timestamp: pair.timestamp))
                    }
                }
                lock.unlock()
                return Array(hits.sorted { $0.score > $1.score }.prefix(limit))
            } catch {
                AgentLog.warning("graph_memory: 语义检索不可用，回退关键词匹配: \(error.localizedDescription)")
                // 不永久禁用：模型可能随后加载成功（例如用户刚下载/切换模型），
                // 下次检索应重新尝试语义通道。
            }
        }
        return keywordSearch(pairs, query: q, topK: limit)
    }

    /// `search` 的语义化别名（任务契约中的 query）。
    public func query(_ text: String, topK: Int = 8) async -> [GraphObservationHit] {
        await search(query: text, topK: topK)
    }

    private func keywordSearch(_ pairs: [(entity: String, text: String, timestamp: Double)],
                               query: String, topK: Int) -> [GraphObservationHit] {
        let needle = query.lowercased()
        var hits: [GraphObservationHit] = []
        for pair in pairs where pair.text.lowercased().contains(needle) {
            hits.append(GraphObservationHit(entity: pair.entity, text: pair.text,
                                            score: 0.5, timestamp: pair.timestamp))
            if hits.count >= topK { break }
        }
        return hits
    }

    private var isEmbeddingDisabled: Bool {
        lock.lock(); defer { lock.unlock() }
        return embeddingDisabled
    }

    private func ensureVectors(_ pairs: [(entity: String, text: String, timestamp: Double)],
                               embedding: EmbeddingProvider) async throws {
        lock.lock()
        var missingTexts: [String] = []
        var seen = Set<String>()
        for pair in pairs where vectors[Self.vectorKey(pair.entity, pair.text)] == nil {
            // 同一段观察文本可能挂在多个实体下，按文本去重只嵌入一次。
            if seen.insert(pair.text).inserted { missingTexts.append(pair.text) }
        }
        lock.unlock()
        guard !missingTexts.isEmpty else { return }

        var embeddedByText: [String: [Float]] = [:]
        var offset = 0
        while offset < missingTexts.count {
            try Task.checkCancellation()
            let batch = Array(missingTexts[offset..<min(offset + 32, missingTexts.count)])
            let embedded = try await embedding.embedDocuments(batch)
            guard embedded.count == batch.count else {
                throw AgentError.storage("graph_memory embedding 数量不一致")
            }
            for (text, vector) in zip(batch, embedded) { embeddedByText[text] = vector }
            offset += batch.count
        }
        lock.lock()
        for pair in pairs where vectors[Self.vectorKey(pair.entity, pair.text)] == nil {
            if let vector = embeddedByText[pair.text] {
                vectors[Self.vectorKey(pair.entity, pair.text)] = vector
            }
        }
        lock.unlock()
    }

    // MARK: - 摘要

    /// 用于 prompt 注入的紧凑摘要。
    public func snapshot(maxItems: Int = 10) -> String {
        lock.lock()
        let entityList = entities.values.sorted {
            if $0.observations.count != $1.observations.count {
                return $0.observations.count > $1.observations.count
            }
            return $0.name.localizedStandardCompare($1.name) == .orderedAscending
        }
        let relationList = relations
        lock.unlock()

        guard !entityList.isEmpty || !relationList.isEmpty else { return "" }
        var lines: [String] = []
        var shown = 0
        for entity in entityList {
            if shown >= maxItems { break }
            let observations = entity.observations
            guard !observations.isEmpty else { continue }
            let type = entity.type.isEmpty ? "" : " (\(entity.type))"
            lines.append("- \(entity.name)\(type): \(observations.prefix(3).joined(separator: "; "))")
            shown += 1
        }
        for relation in relationList.prefix(max(0, maxItems / 2)) {
            lines.append("- \(relation.source) --[\(relation.label)]--> \(relation.target)")
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - 生命周期

    public func clear() {
        lock.lock()
        entities = [:]
        relations = []
        vectors = [:]
        nameVectors = [:]
        saveLocked()
        lock.unlock()
    }

    /// 立即落盘（本实现每次修改都已落盘）。
    public func flush() {
        lock.lock(); defer { lock.unlock() }
        saveLocked(force: true)
    }

    /// 释放资源；本实现无外部句柄，仅确保落盘。
    public func close() {
        flush()
    }

    // MARK: - 裁剪

    private func pruneLocked() {
        var pruned = false

        // 1) 每实体观察上限：保留最新（先收集 key，避免边遍历边改字典）。
        let oversized = entities.compactMap { (key, entity) -> String? in
            entity.observations.count > maxObservationsPerEntity ? key : nil
        }
        if !oversized.isEmpty {
            pruned = true
            for key in oversized {
                guard var entity = entities[key] else { continue }
                let dropCount = entity.observations.count - maxObservationsPerEntity
                let dropped = entity.observations.prefix(dropCount)
                entity.observations.removeFirst(dropCount)
                // 时间戳与观察数组严格对齐：一起裁掉最旧的头部。
                entity.observationDates = GraphEntity.alignedDates(
                    Array(entity.observationDates.dropFirst(min(dropCount, entity.observationDates.count))),
                    count: entity.observations.count)
                entities[key] = entity
                for text in dropped {
                    vectors.removeValue(forKey: Self.vectorKey(entity.name, text))
                }
            }
        }
        // 2) 实体总数上限：按 created_at 淘汰最旧。
        if entities.count > maxEntities {
            pruned = true
            let ordered = entities.sorted {
                if $0.value.createdAt != $1.value.createdAt {
                    return $0.value.createdAt < $1.value.createdAt
                }
                return $0.key < $1.key
            }
            let excess = entities.count - maxEntities
            for (key, _) in ordered.prefix(excess) {
                entities.removeValue(forKey: key)
                vectors = vectors.filter { !$0.key.hasPrefix(key + "|") }
                nameVectors.removeValue(forKey: key)
                relations.removeAll {
                    $0.source.lowercased() == key || $0.target.lowercased() == key
                }
            }
        }
        // 3) 关系上限：淘汰最旧。
        if relations.count > maxRelations {
            pruned = true
            relations.removeFirst(relations.count - maxRelations)
        }
        // 向量缓存只在确实裁剪过时重建存活集合：每次写入都全量扫描
        // 会让 upsert/addObservations 随着图谱变大而线性变慢。
        if pruned, !vectors.isEmpty {
            var alive = Set<String>()
            for entity in entities.values {
                for text in entity.observations {
                    alive.insert(Self.vectorKey(entity.name, text))
                }
            }
            vectors = vectors.filter { alive.contains($0.key) }
        }
    }

    // MARK: - 持久化

    private func load() {
        lock.lock(); defer { lock.unlock() }
        let fm = FileManager.default
        guard fm.fileExists(atPath: path),
              let raw = try? String(contentsOfFile: path, encoding: .utf8),
              let root = JSONValue.parse(raw)?.objectValue else {
            entities = [:]
            relations = []
            return
        }

        var loadedEntities: [String: GraphEntity] = [:]
        if let rawEntities = root["entities"]?.objectValue {
            for (key, value) in rawEntities {
                guard let object = value.objectValue else { continue }
                let name = (object["name"]?.stringValue ?? key)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !name.isEmpty else { continue }
                let type = object["type"]?.stringValue ?? ""
                let observations = (object["observations"]?.arrayValue ?? [])
                    .compactMap(\.stringValue)
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                let createdAt = object["created_at"]?.stringValue ?? ""
                // 旧库没有 observation_dates：全部按 0（时间未知／最旧）处理。
                let dates = (object["observation_dates"]?.arrayValue ?? [])
                    .compactMap(\.doubleValue)
                loadedEntities[name.lowercased()] = GraphEntity(
                    name: name, type: type, observations: observations,
                    createdAt: createdAt.isEmpty ? Self.now() : createdAt,
                    observationDates: dates)
            }
        }

        var loadedRelations: [GraphRelation] = []
        if let rawRelations = root["relations"]?.arrayValue {
            for item in rawRelations {
                guard let object = item.objectValue else { continue }
                let source = (object["source"]?.stringValue ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let target = (object["target"]?.stringValue ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let label = (object["label"]?.stringValue ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !source.isEmpty, !target.isEmpty, !label.isEmpty else { continue }
                let created = object["created"]?.stringValue
                loadedRelations.append(GraphRelation(source: source, target: target,
                                                     label: label, created: created))
            }
        }

        entities = loadedEntities
        relations = loadedRelations
        pruneLocked()
    }

    /// 写入落盘。默认防抖 0.5s：事实抽取一次会写 5 个实体 + 3 条关系，
    /// 每次都全量序列化大 JSON 会造成明显写放大。`force=true` 用于 flush/close。
    private func saveLocked(force: Bool = false) {
        dirty = true
        guard !force else {
            pendingSave?.cancel()
            pendingSave = nil
            dirty = false
            writeLocked()
            return
        }
        guard pendingSave == nil else { return }
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            self.lock.lock()
            self.pendingSave = nil
            if self.dirty {
                self.dirty = false
                self.writeLocked()
            }
            self.lock.unlock()
        }
        pendingSave = work
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 0.5, execute: work)
    }

    private func writeLocked() {
        var entityObjects: [String: JSONValue] = [:]
        for (key, entity) in entities {
            var object: [String: JSONValue] = [
                "name": .string(entity.name),
                "type": .string(entity.type),
                "observations": .array(entity.observations.map { .string($0) }),
                "observation_dates": .array(entity.observationDates.map { .double($0) }),
                "created_at": .string(entity.createdAt),
            ]
            if !entity.createdAt.isEmpty { object["created_at"] = .string(entity.createdAt) }
            entityObjects[key] = .object(object)
        }
        let relationValues: [JSONValue] = relations.map { relation in
            var object: [String: JSONValue] = [
                "source": .string(relation.source),
                "target": .string(relation.target),
                "label": .string(relation.label),
            ]
            if let created = relation.created { object["created"] = .string(created) }
            return .object(object)
        }
        let payload: JSONValue = .object([
            "entities": .object(entityObjects),
            "relations": .array(relationValues),
        ])
        do {
            let text = payload.jsonString(pretty: true) + "\n"
            try AtomicFile.write(Data(text.utf8), to: path)
        } catch {
            AgentLog.warning("graph_memory: 写入失败 \(path): \(error.localizedDescription)")
        }
    }

    // MARK: - 工具函数

    private static func now() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        return formatter.string(from: Date())
    }

    /// 一批新观察的时间戳：同批内按顺序微增，保证「数组尾部 = 更新」
    /// 这一既有语义在时间排序下依然成立。
    private static func stampSequence(_ count: Int) -> [Double] {
        guard count > 0 else { return [] }
        let base = Date().timeIntervalSince1970
        return (0..<count).map { base + Double($0) * 0.001 }
    }

    private static func vectorKey(_ entity: String, _ text: String) -> String {
        "\(entity.lowercased())|\(text)"
    }

    private static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0..<a.count {
            let x = Double(a[i]), y = Double(b[i])
            dot += x * y
            na += x * x
            nb += y * y
        }
        let denom = na.squareRoot() * nb.squareRoot()
        return denom > 0 ? max(0, min(1, dot / denom)) : 0
    }
}
