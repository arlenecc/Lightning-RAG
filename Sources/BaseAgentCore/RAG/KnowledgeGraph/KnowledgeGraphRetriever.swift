import Foundation

// MARK: - 知识图谱检索

/// 五种检索策略的图谱召回实现：先抽关键词，再按策略把对应文本嵌入，产出
/// 实体 / 关系 / 原文三类候选，去重合并后交给 reranker 精排。
///
/// - `naive` 不经过本类（原文召回走原有混合检索）。
/// - `local`：命中实体拉邻居边与关联实体。
/// - `global`：命中关系反查两边实体及度数。
/// - `hybrid`：实体 + 关系两个子图轮询交织。
/// - `mix`：实体 + 关系 + 原文切片全做。
public final class KnowledgeGraphRetriever {

    /// 查询语句直接召回的原文切片数（mix）。
    private static let directChunkTopK = 20
    /// 反查原文时最多读取的 chunk id 数，避免超大图谱一次拉爆。
    private static let maxReverseLookupIDs = 1_000

    private let kgStore: KnowledgeGraphStore
    private let graph: KnowledgeGraph
    private let builder: KnowledgeGraphBuilder
    private let embedding: EmbeddingProvider
    private let config: AgentConfig
    private let chunkStore: VectorStore

    public init(kgStore: KnowledgeGraphStore, graph: KnowledgeGraph,
                builder: KnowledgeGraphBuilder, embedding: EmbeddingProvider,
                config: AgentConfig, chunkStore: VectorStore) {
        self.kgStore = kgStore
        self.graph = graph
        self.builder = builder
        self.embedding = embedding
        self.config = config
        self.chunkStore = chunkStore
    }

    // MARK: - 主入口

    public func retrieve(query: String, source: String? = nil) async throws -> KGRetrievalCandidates {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return KGRetrievalCandidates()
        }
        let strategy = config.ragRetrievalStrategy
        let floor = config.ragKGSimilarityFloor
        let entityTopK = config.ragKGEntityTopK
        let relationTopK = config.ragKGRelationTopK
        let perHitChunkLimit = max(1, config.ragKGMaxChunksPerHit)
        let sourceNeedle = source?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let hasSourceFilter = !sourceNeedle.isEmpty

        // 1) 关键词抽取（失败回退：直接用原 query）。
        let keywords: KGKeywords
        if strategy.usesKnowledgeGraph {
            keywords = (try? await builder.extractKeywords(query: trimmed))
                ?? KGKeywords(highLevel: [trimmed], lowLevel: [trimmed])
        } else {
            keywords = KGKeywords(highLevel: [trimmed], lowLevel: [trimmed])
        }

        // 2) 三路向量（ll→实体、hl→关系、原 query→原文）一次批量计算。
        let lowTexts = normalizedKeywords(keywords.lowLevel, fallback: trimmed)
        let highTexts = normalizedKeywords(keywords.highLevel, fallback: trimmed)
        let vectors = try await embedQueries(lowTexts + highTexts + [trimmed])
        guard vectors.count == lowTexts.count + highTexts.count + 1 else {
            throw AgentError.storage("批量查询向量数量不一致")
        }
        let entityVector = averageVector(Array(vectors[0..<lowTexts.count]))
        let relationVector = averageVector(
            Array(vectors[lowTexts.count..<(lowTexts.count + highTexts.count)]))
        let chunkVector = vectors[vectors.count - 1]

        // 3) 三路召回。
        async let entityHits = kgStore.searchEntities(vector: entityVector, topK: entityTopK,
                                                      similarityFloor: floor)
        async let relationHits = kgStore.searchRelationships(vector: relationVector, topK: relationTopK,
                                                             similarityFloor: floor)
        let (entities, relations) = try await (entityHits, relationHits)

        // 4) 按策略组装实体/关系候选 + 图谱扩展。
        var entityPassages: [KGPassage] = []
        var relationPassages: [KGPassage] = []

        switch strategy {
        case .naive:
            break
        case .local:
            // 命中实体 + 一跳邻居实体；关联边一并拉出。
            for hit in entities {
                entityPassages.append(entityPassage(hit.entity, score: hit.score))
                for neighbor in graph.neighbors(of: hit.entity.name) {
                    if let e = graph.entity(named: neighbor),
                       !contains(entityPassages, name: e.name) {
                        entityPassages.append(entityPassage(e, score: hit.score * 0.9))
                    }
                }
            }
            for hit in entities {
                for rel in graph.incidentRelationships(of: hit.entity.name) {
                    relationPassages.append(relationPassage(rel, score: hit.score))
                }
            }
        case .global:
            // 命中关系 + 两端实体及度数。
            for hit in relations {
                relationPassages.append(relationPassage(hit.relationship, score: hit.score))
                if let s = graph.entity(named: hit.relationship.srcEntity) {
                    entityPassages.append(entityPassage(s, score: hit.score))
                }
                if let t = graph.entity(named: hit.relationship.dstEntity) {
                    entityPassages.append(entityPassage(t, score: hit.score))
                }
            }
        case .hybrid, .mix:
            let local = localExpansion(entities)
            let global = globalExpansion(relations)
            (entityPassages, relationPassages) = interleave(local, global)
        }

        // 5) 去重 + 统一排序（余弦 + 度数/权重）。
        entityPassages = KGCandidateRanking.entities(dedupEntities(entityPassages))
        relationPassages = KGCandidateRanking.relations(dedupRelations(relationPassages))

        // 6) 反查原文所需的 chunk 引用（一次查表，供来源过滤与限量复用）。
        let candidateSourceIDs = Set(entityPassages.flatMap(\.sourceChunkIDs)
            + relationPassages.flatMap(\.sourceChunkIDs))
        var refs = try kgStore.chunkRefs(
            ids: Array(candidateSourceIDs.sorted().prefix(Self.maxReverseLookupIDs)))
        if hasSourceFilter {
            refs = refs.filter { $0.source.localizedCaseInsensitiveContains(sourceNeedle) }
            let allowed = Set(refs.map(\.id))
            // 来源过滤：实体/关系至少要有一条归因切片落在目标来源。
            entityPassages = entityPassages.filter {
                !$0.sourceChunkIDs.allSatisfy { !allowed.contains($0) }
            }
            relationPassages = relationPassages.filter {
                !$0.sourceChunkIDs.allSatisfy { !allowed.contains($0) }
            }
            refs = refs.filter { ref in
                entityPassages.contains { $0.sourceChunkIDs.contains(ref.id) }
                    || relationPassages.contains { $0.sourceChunkIDs.contains(ref.id) }
            }
        }
        let refMap = Dictionary(uniqueKeysWithValues: refs.map { ($0.id, $0) })

        // 7) 实体/关系 → chunk：每条最多带回 perHitChunkLimit 片。
        let entityChunkStreams = chunkStreams(entityPassages, refMap: refMap,
                                              limit: perHitChunkLimit)
        let relationChunkStreams = chunkStreams(relationPassages, refMap: refMap,
                                                limit: perHitChunkLimit)

        // 8) 原文直接召回（mix；相似度低于下限直接丢弃）。
        var directChunks: [KGPassage] = []
        if strategy == .mix {
            let hits = try chunkStore.search(
                vector: chunkVector, topK: Self.directChunkTopK,
                sourceContains: hasSourceFilter ? sourceNeedle : nil)
            directChunks = hits.filter { $0.score >= floor }.map { hit in
                let chunkID = "\(VectorStore.hashSource(hit.source))_\(hit.chunkIndex)"
                return KGPassage(text: hit.text, kind: .chunk,
                                 label: (hit.source as NSString).lastPathComponent,
                                 score: hit.score, sourceChunkIDs: [chunkID])
            }
        }

        // 9) chunk 合并：直接召回在前，实体/关系反查按轮询交错去重（chunk id）。
        var chunkPassages: [KGPassage] = []
        var seenChunkIDs = Set<String>()
        func append(_ passage: KGPassage) {
            guard let id = passage.sourceChunkIDs.first,
                  seenChunkIDs.insert(id).inserted else { return }
            chunkPassages.append(passage)
        }
        directChunks.forEach(append)
        roundRobin(entityChunkStreams.flatMap { $0 },
                   relationChunkStreams.flatMap { $0 }).forEach(append)

        return KGRetrievalCandidates(entities: entityPassages,
                                     relations: relationPassages,
                                     chunks: chunkPassages)
    }

    // MARK: - 子图扩展

    private func localExpansion(_ entities: [KnowledgeGraphStore.EntityHit])
        -> (entities: [KGPassage], relations: [KGPassage]) {
        var ePassages: [KGPassage] = []
        var rPassages: [KGPassage] = []
        for hit in entities {
            ePassages.append(entityPassage(hit.entity, score: hit.score))
            for neighbor in graph.neighbors(of: hit.entity.name) {
                if let e = graph.entity(named: neighbor), !contains(ePassages, name: e.name) {
                    ePassages.append(entityPassage(e, score: hit.score * 0.9))
                }
            }
            for rel in graph.incidentRelationships(of: hit.entity.name) {
                rPassages.append(relationPassage(rel, score: hit.score))
            }
        }
        return (ePassages, rPassages)
    }

    private func globalExpansion(_ relations: [KnowledgeGraphStore.RelationHit])
        -> (entities: [KGPassage], relations: [KGPassage]) {
        var ePassages: [KGPassage] = []
        var rPassages: [KGPassage] = []
        for hit in relations {
            rPassages.append(relationPassage(hit.relationship, score: hit.score))
            if let s = graph.entity(named: hit.relationship.srcEntity) {
                ePassages.append(entityPassage(s, score: hit.score))
            }
            if let t = graph.entity(named: hit.relationship.dstEntity) {
                ePassages.append(entityPassage(t, score: hit.score))
            }
        }
        return (ePassages, rPassages)
    }

    /// 实体 + 关系两个子图轮询交织合并。
    private func interleave(_ a: (entities: [KGPassage], relations: [KGPassage]),
                            _ b: (entities: [KGPassage], relations: [KGPassage]))
        -> ([KGPassage], [KGPassage]) {
        var entities: [KGPassage] = []
        var relations: [KGPassage] = []
        let maxE = max(a.entities.count, b.entities.count)
        let maxR = max(a.relations.count, b.relations.count)
        for i in 0..<maxE {
            if i < a.entities.count { entities.append(a.entities[i]) }
            if i < b.entities.count { entities.append(b.entities[i]) }
        }
        for i in 0..<maxR {
            if i < a.relations.count { relations.append(a.relations[i]) }
            if i < b.relations.count { relations.append(b.relations[i]) }
        }
        return (entities, relations)
    }

    // MARK: - 候选构造

    private func entityPassage(_ entity: KGEntity, score: Double) -> KGPassage {
        let type = entity.type.isEmpty ? "" : "（\(entity.type)）"
        let text = "实体：\(entity.name)\(type)\n\(entity.description)"
        return KGPassage(text: text, kind: .entity, label: entity.name,
                         score: score, degree: graph.degree(of: entity.name),
                         sourceChunkIDs: entity.sourceIDs)
    }

    private func relationPassage(_ rel: KGRelationship, score: Double) -> KGPassage {
        let text = "关系：\(rel.srcEntity) --\(rel.relationType)--> \(rel.dstEntity)\n\(rel.description)"
        return KGPassage(text: text, kind: .relation,
                         label: "\(rel.srcEntity) --\(rel.relationType)--> \(rel.dstEntity)",
                         score: score, degree: graph.relationRank(rel),
                         weight: rel.weight, sourceChunkIDs: rel.sourceIDs,
                         srcEntity: rel.srcEntity, dstEntity: rel.dstEntity,
                         relationType: rel.relationType)
    }

    // MARK: - 去重

    private func dedupEntities(_ passages: [KGPassage]) -> [KGPassage] {
        var seen = Set<String>()
        var out: [KGPassage] = []
        for p in passages {
            let key = p.label.lowercased()
            if seen.insert(key).inserted { out.append(p) }
        }
        return out
    }

    /// 关系按三元组去重（无向边归一化）。
    private func dedupRelations(_ passages: [KGPassage]) -> [KGPassage] {
        var seen = Set<String>()
        var out: [KGPassage] = []
        for p in passages {
            if seen.insert(p.tripleKey()).inserted { out.append(p) }
        }
        return out
    }

    // MARK: - 原文反查

    /// 每个实体/关系各自最多 `limit` 片，按候选顺序保留。
    private func chunkStreams(_ passages: [KGPassage],
                              refMap: [String: KnowledgeGraphStore.ChunkRef],
                              limit: Int) -> [[KGPassage]] {
        passages.map { passage in
            var seen = Set<String>()
            var out: [KGPassage] = []
            for id in passage.sourceChunkIDs where seen.insert(id).inserted {
                guard out.count < limit, let ref = refMap[id] else { continue }
                out.append(KGPassage(text: ref.text, kind: .chunk,
                                     label: (ref.source as NSString).lastPathComponent,
                                     score: 0, sourceChunkIDs: [ref.id]))
            }
            return out
        }
    }

    /// 两组候选按位轮询交错（a[0], b[0], a[1], b[1], …）。
    private func roundRobin(_ a: [KGPassage], _ b: [KGPassage]) -> [KGPassage] {
        var out: [KGPassage] = []
        out.reserveCapacity(a.count + b.count)
        let maxCount = max(a.count, b.count)
        for i in 0..<maxCount {
            if i < a.count { out.append(a[i]) }
            if i < b.count { out.append(b[i]) }
        }
        return out
    }

    // MARK: - 工具

    private func contains(_ passages: [KGPassage], name: String) -> Bool {
        passages.contains { $0.label.lowercased() == name.lowercased() }
    }

    private func normalizedKeywords(_ keywords: [String], fallback: String) -> [String] {
        let filtered = keywords.filter {
            !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        return filtered.isEmpty ? [fallback] : filtered
    }

    /// 多路查询文本一次批量嵌入；provider 不支持批量时退回逐条 `embedQuery`。
    private func embedQueries(_ texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        if let batching = embedding as? EmbeddingQueryBatching {
            let vectors = try await batching.embedQueries(texts)
            if vectors.count == texts.count { return vectors }
        }
        var vectors: [[Float]] = []
        vectors.reserveCapacity(texts.count)
        for text in texts {
            vectors.append(try await embedding.embedQuery(text))
        }
        return vectors
    }

    /// 多个查询向量取均值并重新归一化。
    private func averageVector(_ vectors: [[Float]]) -> [Float] {
        guard let first = vectors.first else { return [] }
        var sum = [Float](repeating: 0, count: first.count)
        for vector in vectors {
            for i in 0..<min(sum.count, vector.count) { sum[i] += vector[i] }
        }
        let count = Float(vectors.count)
        for i in sum.indices { sum[i] /= count }
        return VectorCodec.normalize(sum)
    }
}
