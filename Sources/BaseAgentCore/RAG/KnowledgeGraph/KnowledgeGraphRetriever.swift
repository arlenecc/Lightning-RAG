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

    public func retrieve(query: String) async throws -> KGRetrievalCandidates {
        guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return KGRetrievalCandidates()
        }
        let strategy = config.ragRetrievalStrategy
        let floor = config.ragKGSimilarityFloor
        let entityTopK = config.ragKGEntityTopK
        let relationTopK = config.ragKGRelationTopK

        // 1) 关键词抽取（失败回退：直接用原 query）。
        let keywords: KGKeywords
        if strategy.usesKnowledgeGraph {
            keywords = (try? await builder.extractKeywords(query: query))
                ?? KGKeywords(highLevel: [query], lowLevel: [query])
        } else {
            keywords = KGKeywords(highLevel: [query], lowLevel: [query])
        }

        // 2) 三路向量：ll→实体、hl→关系、原 query→原文（可并发计算）。
        async let entityVec = averageQueryVector(keywords.lowLevel, fallback: query)
        async let relationVec = averageQueryVector(keywords.highLevel, fallback: query)
        async let chunkVec = embedding.embedQuery(query)
        let (entityVector, relationVector, chunkVector) = try await (entityVec, relationVec, chunkVec)

        // 3) 三路召回。
        async let entityHits = kgStore.searchEntities(vector: entityVector, topK: entityTopK,
                                                      similarityFloor: floor)
        async let relationHits = kgStore.searchRelationships(vector: relationVector, topK: relationTopK,
                                                             similarityFloor: floor)
        let (entities, relations) = try await (entityHits, relationHits)

        // 4) 按策略组装实体/关系候选 + 图谱扩展。
        var entityPassages: [KGPassage] = []
        var relationPassages: [KGPassage] = []
        var extraChunkIDs: [String] = []

        switch strategy {
        case .naive:
            break
        case .local:
            // 命中实体 + 一跳邻居实体；关联边一并拉出。
            for hit in entities {
                entityPassages.append(entityPassage(hit.entity, score: hit.score))
                for neighbor in graph.neighbors(of: hit.entity.name) {
                    if let e = graph.entity(named: neighbor), !contains(entityPassages, name: e.name) {
                        entityPassages.append(entityPassage(e, score: hit.score * 0.9))
                    }
                }
            }
            for hit in entities {
                for rel in graph.incidentRelationships(of: hit.entity.name) {
                    relationPassages.append(relationPassage(rel, score: hit.score))
                }
            }
            extraChunkIDs = entities.flatMap { $0.entity.sourceIDs }
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
            extraChunkIDs = relations.flatMap { $0.relationship.sourceIDs }
        case .hybrid:
            let local = localExpansion(entities)
            let global = globalExpansion(relations)
            (entityPassages, relationPassages) = interleave(local, global)
            extraChunkIDs = entities.flatMap { $0.entity.sourceIDs }
                + relations.flatMap { $0.relationship.sourceIDs }
        case .mix:
            let local = localExpansion(entities)
            let global = globalExpansion(relations)
            (entityPassages, relationPassages) = interleave(local, global)
            extraChunkIDs = entities.flatMap { $0.entity.sourceIDs }
                + relations.flatMap { $0.relationship.sourceIDs }
        }

        // 5) 去重。
        entityPassages = dedupEntities(entityPassages)
        relationPassages = dedupRelations(relationPassages)

        // 6) 实体/关系反查原文（每条最多 5 片）。
        let reverseChunks = try reverseLookupChunks(extraChunkIDs, limit: config.ragKGMaxChunksPerHit)

        // 7) 原文直接召回（mix；优先级最高）。
        var directChunks: [KGPassage] = []
        if strategy == .mix {
            let hits = try chunkStore.search(vector: chunkVector, topK: 20)
            directChunks = hits.map { hit in
                let chunkID = "\(VectorStore.hashSource(hit.source))_\(hit.chunkIndex)"
                return KGPassage(text: hit.text, kind: .chunk,
                                 label: (hit.source as NSString).lastPathComponent,
                                 score: hit.score, sourceChunkIDs: [chunkID])
            }
        }

        // 8) chunk 去重（按 chunk id），直接召回在前。
        var chunkPassages = directChunks
        for chunk in reverseChunks where !contains(chunkPassages, chunkID: chunk.sourceChunkIDs.first ?? "") {
            chunkPassages.append(chunk)
        }

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
                         weight: rel.weight, sourceChunkIDs: rel.sourceIDs)
    }

    // MARK: - 去重

    private func dedupEntities(_ passages: [KGPassage]) -> [KGPassage] {
        var seen = Set<String>()
        var out: [KGPassage] = []
        for p in passages {
            let key = p.label.lowercased()
            if seen.insert(key).inserted { out.append(p) }
        }
        return out.sorted {
            if $0.score != $1.score { return $0.score > $1.score }
            return $0.degree > $1.degree
        }
    }

    /// 关系按三元组去重（无向边归一化），再按 rank（度数）+ weight 降序。
    private func dedupRelations(_ passages: [KGPassage]) -> [KGPassage] {
        var seen = Set<String>()
        var out: [KGPassage] = []
        for p in passages {
            // label 形如 "src --type--> dst"；拆回三元组做无向归一。
            let key = canonicalTripleKey(p.label)
            if seen.insert(key).inserted { out.append(p) }
        }
        return out.sorted {
            if $0.degree != $1.degree { return $0.degree > $1.degree }
            if $0.weight != $1.weight { return $0.weight > $1.weight }
            return $0.score > $1.score
        }
    }

    private func canonicalTripleKey(_ label: String) -> String {
        let parts = label.components(separatedBy: "--")
        guard parts.count >= 3 else { return label.lowercased() }
        let src = parts[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let dst = parts[2].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let type = parts[1].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let (lo, hi) = src <= dst ? (src, dst) : (dst, src)
        return "\(lo)|\(hi)|\(type)"
    }

    // MARK: - 反查

    /// 实体/关系反查原文，每条最多带回 `limit` 片。
    private func reverseLookupChunks(_ chunkIDs: [String],
                                     limit: Int) throws -> [KGPassage] {
        guard !chunkIDs.isEmpty else { return [] }
        let deduped = Array(Set(chunkIDs))
        // 每个来源实体/关系最多带回 limit 片：这里整体去重后再限量。
        let refs = try kgStore.chunkRefs(ids: Array(deduped.prefix(1_000)))
        return refs.prefix(limit * 2).map { ref in
            KGPassage(text: ref.text, kind: .chunk,
                      label: (ref.source as NSString).lastPathComponent,
                      score: 0, sourceChunkIDs: [ref.id])
        }
    }

    // MARK: - 工具

    private func contains(_ passages: [KGPassage], name: String) -> Bool {
        passages.contains { $0.label.lowercased() == name.lowercased() }
    }

    private func contains(_ passages: [KGPassage], chunkID: String) -> Bool {
        passages.contains { $0.sourceChunkIDs.first == chunkID }
    }

    /// 多个关键词查询向量取均值并重新归一化。
    private func averageQueryVector(_ texts: [String],
                                    fallback: String) async throws -> [Float] {
        var texts = texts.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        if texts.isEmpty { texts = [fallback] }
        var vectors: [[Float]] = []
        for text in texts {
            vectors.append(try await embedding.embedQuery(text))
        }
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
