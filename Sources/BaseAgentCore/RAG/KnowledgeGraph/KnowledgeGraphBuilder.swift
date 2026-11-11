import Foundation

// MARK: - 知识图谱构建 worker

/// 用辅助模型在已完成切片上抽取实体/关系，聚合后入库并建立 SwiftGraph。
///
/// 全量构建流程：清空旧图谱 → 逐切片抽取 → 内存聚合（同名实体合并、同三元组
/// 关系合并与 weight 累加）→ 描述归纳（多段描述交 LLM 合并）→ 落库 + 向量化 →
/// 建图 + 快照缓存。重建路径 weight 随证据向下重算，不保留旧标量。
public final class KnowledgeGraphBuilder {

    /// 描述超过该数量时交给 LLM 归纳。
    private static let summarizeThreshold = 2
    /// 抽取温度（低温度保证一致性）。
    private static let extractionTemperature = 0.1
    /// 描述归纳温度。
    private static let summarizeTemperature = 0.1

    private let store: KnowledgeGraphStore
    private let graph: KnowledgeGraph
    private let llm: LLMService

    public init(store: KnowledgeGraphStore, graph: KnowledgeGraph, llm: LLMService) {
        self.store = store
        self.graph = graph
        self.llm = llm
    }

    // MARK: - 累加器

    private struct EntityAcc {
        var name: String
        var types: Set<String> = []
        var descriptions: [String] = []
        var sourceIDs: Set<String> = []
        var filePaths: Set<String> = []
    }

    private struct RelationAcc {
        var src: String
        var dst: String
        var keywords: Set<String> = []
        var descriptions: [String] = []
        var sourceIDs: Set<String> = []
        var filePaths: Set<String> = []
        var weightSum: Int = 0
    }

    public struct BuildStats: Sendable {
        public var chunksProcessed = 0
        public var entities = 0
        public var relationships = 0
        public var failedChunks = 0
    }

    // MARK: - 全量构建

    public func build(progress: ((Int, Int, String) -> Void)? = nil,
                      log: ((String) -> Void)? = nil) async throws -> BuildStats {
        // 1) 清空旧图谱（purge）：weight 会在下面随证据重算。
        try store.clear()
        graph.rebuild(entities: [], relationships: [])
        await store.loadTokenCounter()

        // 2) 逐切片抽取，聚合。
        var entityAcc: [String: EntityAcc] = [:]
        var relationAcc: [String: RelationAcc] = [:]
        let chunkRefs = try store.listChunkRefs()
        let total = chunkRefs.count
        var stats = BuildStats()

        log?("[kg] 开始抽取知识图谱，共 \(total) 个切片")
        for (index, chunk) in chunkRefs.enumerated() {
            try Task.checkCancellation()
            let chunkText = chunk.text
            guard !chunkText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                continue
            }
            do {
                let extraction = try await extract(chunk: chunkText)
                accumulate(extraction, chunkID: chunk.id, filePath: chunk.source,
                           entities: &entityAcc, relations: &relationAcc)
            } catch {
                stats.failedChunks += 1
                AgentLog.warning("[kg] 切片抽取失败（\(chunk.id)）: \(error.localizedDescription)")
            }
            stats.chunksProcessed += 1
            if index % 10 == 0 || index == total - 1 {
                progress?(index + 1, total, "抽取实体/关系 \(index + 1)/\(total)")
            }
        }

        // 3) 描述归纳 + 落库 + 向量化 + 建图。
        log?("[kg] 聚合完成：实体 \(entityAcc.count)、关系 \(relationAcc.count)，开始归纳与入库")
        let finalizedEntities = try await finalizeEntities(entityAcc, log: log)
        let finalizedRelations = try await finalizeRelations(relationAcc, log: log)
        stats.entities = finalizedEntities.count
        stats.relationships = finalizedRelations.count

        // 3a) 向量化（实体 + 关系各一条）。
        var targets: [(KGVectorTargetType, String, String)] = []
        for entity in finalizedEntities {
            targets.append((.entity, entity.id, store.entityEmbeddingText(entity)))
        }
        for rel in finalizedRelations {
            targets.append((.relationship, rel.id, store.relationshipEmbeddingText(rel)))
        }
        log?("[kg] 计算 \(targets.count) 条向量")
        var lastProgress = -1
        try await store.storeVectors(targets: targets) { done, count in
            let pct = count > 0 ? done * 100 / count : 0
            if pct != lastProgress {
                lastProgress = pct
                progress?(done, count, "向量化 \(pct)%")
            }
        }

        // 4) 建图 + 快照。
        graph.rebuild(entities: finalizedEntities, relationships: finalizedRelations)
        log?("[kg] 知识图谱构建完成：实体 \(stats.entities)、关系 \(stats.relationships)")
        return stats
    }

    // MARK: - 抽取

    /// 单个切片 → 实体/关系抽取结果。
    func extract(chunk: String) async throws -> KGExtractionResult {
        let system = KnowledgeGraphPrompts.entityExtractionSystem
        let prompt = KnowledgeGraphPrompts.entityExtractionTask(chunk: chunk)
        let output = try await complete(system: system, user: prompt,
                                        temperature: Self.extractionTemperature)
        return Self.parseExtraction(output)
    }

    /// 查询 → 高层/低层关键词（供检索使用）。
    func extractKeywords(query: String) async throws -> KGKeywords {
        let prompt = KnowledgeGraphPrompts.keywordExtractionPrompt(query: query)
        let output = try await complete(system: "", user: prompt,
                                        temperature: Self.extractionTemperature)
        return Self.parseKeywords(output)
    }

    /// 实体/关系描述归纳。
    func summarize(name: String, descriptions: [String]) async throws -> String {
        let list = descriptions.enumerated()
            .map { "\($0.offset + 1). \($0.element)" }
            .joined(separator: "\n")
        let prompt = KnowledgeGraphPrompts.summarizeEntityPrompt(name: name, list: list)
        let output = try await complete(system: "", user: prompt,
                                        temperature: Self.summarizeTemperature)
        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? descriptions.joined(separator: "\n") : trimmed
    }

    // MARK: - 单次直答

    private func complete(system: String, user: String,
                          temperature: Double) async throws -> String {
        var messages: [ChatMessage] = []
        if !system.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            messages.append(ChatMessage(role: .system, content: system))
        }
        messages.append(ChatMessage(role: .user, content: user))
        let options = LLMRequestOptions(enableThinking: false, isolated: true)
        var output = ""
        let stream = llm.chatStream(messages: messages, tools: [],
                                    temperature: temperature, options: options)
        for try await event in stream {
            if event.kind == .content { output += event.content }
            if event.kind == .done, event.truncated {
                throw AgentError.llm("辅助模型输出被截断")
            }
        }
        return output
    }

    // MARK: - 聚合

    private func accumulate(_ result: KGExtractionResult, chunkID: String, filePath: String,
                            entities: inout [String: EntityAcc],
                            relations: inout [String: RelationAcc]) {
        for e in result.entities {
            let name = Self.normalizeName(e.name)
            guard !name.isEmpty else { continue }
            let key = name.lowercased()
            if entities[key] == nil {
                entities[key] = EntityAcc(name: name)
            }
            var acc = entities[key]!
            if !e.type.isEmpty { acc.types.insert(e.type) }
            let desc = e.description.trimmingCharacters(in: .whitespacesAndNewlines)
            if !desc.isEmpty, !acc.descriptions.contains(desc) {
                acc.descriptions.append(desc)
            }
            acc.sourceIDs.insert(chunkID)
            if !filePath.isEmpty { acc.filePaths.insert(filePath) }
            entities[key] = acc
        }
        for r in result.relations {
            let src = Self.normalizeName(r.source)
            let dst = Self.normalizeName(r.target)
            guard !src.isEmpty, !dst.isEmpty, src != dst else { continue }
            let canonical = KGRelationship.canonicalKey(src: src, dst: dst,
                                                        type: r.keywords)
            if relations[canonical] == nil {
                // 无向：端点按字典序归一，保证正序/反序抽到同一对时合并为一条。
                let (lo, hi) = src.lowercased() <= dst.lowercased()
                    ? (src, dst) : (dst, src)
                relations[canonical] = RelationAcc(src: lo, dst: hi)
            }
            var acc = relations[canonical]!
            for kw in r.keywords.split(separator: ",") {
                let trimmed = kw.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { acc.keywords.insert(trimmed) }
            }
            let desc = r.description.trimmingCharacters(in: .whitespacesAndNewlines)
            if !desc.isEmpty, !acc.descriptions.contains(desc) {
                acc.descriptions.append(desc)
            }
            acc.sourceIDs.insert(chunkID)
            if !filePath.isEmpty { acc.filePaths.insert(filePath) }
            acc.weightSum += 1
            relations[canonical] = acc
        }
    }

    // MARK: - 归纳落库

    private func finalizeEntities(_ acc: [String: EntityAcc],
                                  log: ((String) -> Void)?) async throws -> [KGEntity] {
        var result: [KGEntity] = []
        var index = 0
        for (_, item) in acc.sorted(by: { $0.key < $1.key }) {
            try Task.checkCancellation()
            let type = item.types.contains("UNKNOWN") && item.types.count > 1
                ? item.types.filter { $0 != "UNKNOWN" }.first ?? "Other"
                : item.types.first ?? "Other"
            let description: String
            if item.descriptions.count == 1 {
                description = item.descriptions[0]
            } else if item.descriptions.count >= Self.summarizeThreshold {
                description = (try? await summarize(name: item.name,
                                                   descriptions: item.descriptions))
                    ?? item.descriptions.joined(separator: "\n")
            } else {
                description = item.descriptions.joined(separator: "\n")
            }
            var properties: [String: String] = [:]
            if !item.filePaths.isEmpty {
                properties["file_path"] = item.filePaths.sorted().joined(separator: ",")
            }
            let now = Date().timeIntervalSince1970
            let entity = KGEntity(id: UUID().uuidString, name: item.name, type: type,
                                  description: description, properties: properties,
                                  sourceIDs: item.sourceIDs.sorted(),
                                  createdAt: now, updatedAt: now)
            try store.upsertEntity(entity)
            _ = graph.upsertEntity(entity)
            result.append(entity)
            index += 1
            if index % 200 == 0 {
                log?("[kg] 已入库实体 \(index)/\(acc.count)")
            }
        }
        return result
    }

    private func finalizeRelations(_ acc: [String: RelationAcc],
                                   log: ((String) -> Void)?) async throws -> [KGRelationship] {
        var result: [KGRelationship] = []
        var index = 0
        for (_, item) in acc.sorted(by: { $0.key < $1.key }) {
            try Task.checkCancellation()
            let relationType = item.keywords.sorted().joined(separator: ",")
            let description: String
            if item.descriptions.count == 1 {
                description = item.descriptions[0]
            } else if item.descriptions.count >= Self.summarizeThreshold {
                description = (try? await summarize(name: "\(item.src)-\(item.dst)",
                                                   descriptions: item.descriptions))
                    ?? item.descriptions.joined(separator: "\n")
            } else {
                description = item.descriptions.joined(separator: "\n")
            }
            // weight = max(累加权重, 去重后的来源切片数量)；缺失兜底 1。
            let evidence = max(1, item.sourceIDs.count)
            let weight = Double(max(item.weightSum, evidence))
            var properties: [String: String] = [:]
            if !item.filePaths.isEmpty {
                properties["file_path"] = item.filePaths.sorted().joined(separator: ",")
            }
            let now = Date().timeIntervalSince1970
            let rel = KGRelationship(id: UUID().uuidString, srcEntity: item.src,
                                     dstEntity: item.dst, relationType: relationType,
                                     description: description, weight: weight,
                                     properties: properties,
                                     sourceIDs: item.sourceIDs.sorted(),
                                     createdAt: now, updatedAt: now)
            try store.upsertRelationship(rel)
            _ = graph.upsertRelationship(rel)
            result.append(rel)
            index += 1
            if index % 500 == 0 {
                log?("[kg] 已入库关系 \(index)/\(acc.count)")
            }
        }
        return result
    }

    // MARK: - 解析

    static func normalizeName(_ name: String) -> String {
        name.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func parseExtraction(_ output: String) -> KGExtractionResult {
        var entities: [KGExtractedEntity] = []
        var relations: [KGExtractedRelation] = []
        for rawLine in output.split(separator: "\n") {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty || line == "<|COMPLETE|>" { continue }
            let parts = line.components(separatedBy: "<|#|>")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            if parts.count >= 4, parts[0] == "entity" {
                let name = normalizeName(parts[1])
                guard !name.isEmpty else { continue }
                entities.append(KGExtractedEntity(name: name, type: parts[2],
                                                 description: parts[3]))
            } else if parts.count >= 5, parts[0] == "relation" {
                let src = normalizeName(parts[1])
                let dst = normalizeName(parts[2])
                guard !src.isEmpty, !dst.isEmpty else { continue }
                relations.append(KGExtractedRelation(source: src, target: dst,
                                                    keywords: parts[3],
                                                    description: parts[4]))
            }
        }
        return KGExtractionResult(entities: entities, relations: relations)
    }

    static func parseKeywords(_ output: String) -> KGKeywords {
        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let obj = JSONValue.parse(trimmed)?.objectValue else {
            return KGKeywords()
        }
        func strings(_ key: String) -> [String] {
            obj[key]?.arrayValue?.compactMap { $0.stringValue }
                .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty } ?? []
        }
        return KGKeywords(highLevel: strings("high_level_keywords"),
                          lowLevel: strings("low_level_keywords"))
    }
}
