import Foundation

// MARK: - 知识图谱构建 worker

/// 用辅助模型在已完成切片上抽取实体/关系，聚合后入库并建立 SwiftGraph。
///
/// - 全量构建：逐切片抽取（原始输出落断点缓存）→ 内存聚合（同名实体合并、
///   同实体对关系合并、weight 随证据累加）→ 描述归纳 → **原子替换**数据库
///   全部实体/关系 → 向量化 → 建图 + 快照。构建失败不会破坏旧图谱。
/// - 增量构建：只抽取尚未进入图谱的新切片，与数据库既有实体/关系合并，
///   仅对变化的节点/边重新向量化，然后重建内存图并保存快照。
public final class KnowledgeGraphBuilder {

    /// 描述超过该数量时交给 LLM 归纳。
    private static let summarizeThreshold = 2
    /// 抽取温度（低温度保证一致性）。
    private static let extractionTemperature = 0.1
    /// 描述归纳温度。
    private static let summarizeTemperature = 0.1
    /// 抽取提示词版本：提示词变化时作废旧抽取缓存。
    private static let extractionCacheVersion = "3"
    /// 归纳提示词版本：参与归纳缓存 key，提示词变化时自然失效。
    private static let summaryCacheVersion = "2"
    /// 抽取输出的重复关系行容忍上限。小模型收尾阶段会反复输出同一实体对
    /// （提示词明确禁止重复），继续消费只会白跑生成；超过该值即提前收流。
    /// 取 12 而不是更小值：新实体对可能夹在重复行之间出现，过早截断会丢边。
    private static let maxDuplicateRelationRows = 12

    private let store: KnowledgeGraphStore
    private let graph: KnowledgeGraph
    private let llm: LLMService
    private let limits: KGExtractionLimits
    private let format: KGExtractionFormat
    private let maxConcurrency: Int

    public init(store: KnowledgeGraphStore, graph: KnowledgeGraph, llm: LLMService,
                limits: KGExtractionLimits = .default,
                format: KGExtractionFormat = .delimited,
                maxConcurrency: Int = 1) {
        self.store = store
        self.graph = graph
        self.llm = llm
        self.limits = limits
        self.format = format
        self.maxConcurrency = max(1, min(4, maxConcurrency))
    }

    // MARK: - 累加器

    private struct EntityAcc {
        var id: String
        var name: String
        var types: Set<String> = []
        var descriptions: [String] = []
        var sourceIDs: Set<String> = []
        var filePaths: Set<String> = []
        var createdAt: Double = 0
    }

    private struct RelationAcc {
        var id: String
        var src: String
        var dst: String
        var keywords: Set<String> = []
        var descriptions: [String] = []
        var sourceIDs: Set<String> = []
        var filePaths: Set<String> = []
        var weightSum: Int = 0
        /// 同一 chunk 内重复抽到同一条关系只累加一次 weight。
        var countedChunks: Set<String> = []
        var createdAt: Double = 0
    }

    public struct BuildStats: Sendable {
        public var chunksProcessed = 0
        public var entities = 0
        public var relationships = 0
        public var failedChunks = 0
        /// 成功（或无需）抽取的切片 → 内容哈希；只有这些可以被标记为已处理。
        public var succeededChunkHashes: [String: String] = [:]
        /// 本次命中抽取缓存的切片数（重建时正文未变即可复用）。
        public var cacheHits = 0
        /// 抽取阶段耗时（秒）。
        public var extractionSeconds: Double = 0
    }

    /// 增量构建结果：调用方据此在内存图上做局部 upsert，而不是整图重建。
    public struct IncrementalResult: Sendable {
        public let stats: BuildStats
        public let changedEntities: [KGEntity]
        public let changedRelationships: [KGRelationship]
        public let removedRelationshipIDs: [String]
    }

    // MARK: - 全量构建

    public func build(progress: ((Int, Int, String) -> Void)? = nil,
                      log: ((String) -> Void)? = nil) async throws -> BuildStats {
        try store.markBuildState(.building)
        try store.prepareExtractionCache(version: Self.extractionCacheVersion)
        store.pruneSummaryCache()
        await store.loadTokenCounter()

        let chunkRefs = try store.listChunkRefs()
        log?("[kg] 开始抽取知识图谱，共 \(chunkRefs.count) 个切片")
        let extractStart = Date()
        var (entityAcc, relationAcc, stats) = try await extractAll(
            chunkRefs, progress: progress, log: log)
        stats.extractionSeconds = Date().timeIntervalSince(extractStart)
        log?("[kg] 抽取完成：\(chunkRefs.count) 片，耗时 "
            + String(format: "%.1fs", stats.extractionSeconds)
            + "，缓存命中 \(stats.cacheHits)，失败 \(stats.failedChunks)")

        let finalizeStart = Date()
        log?("[kg] 聚合完成：实体 \(entityAcc.count)、关系 \(relationAcc.count)，开始归纳与入库")
        let finalizedEntities = try await finalizeEntities(entityAcc)
        let finalizedRelations = try await finalizeRelations(relationAcc)
        stats.entities = finalizedEntities.count
        stats.relationships = finalizedRelations.count
        log?("[kg] 归纳完成，耗时 "
            + String(format: "%.1fs", Date().timeIntervalSince(finalizeStart)))

        // 原子替换：此前的数据库/快照在失败时保持可用。
        try store.replaceAll(entities: finalizedEntities, relationships: finalizedRelations)

        let targets = store.vectorTargets(entities: finalizedEntities,
                                          relationships: finalizedRelations)
        log?("[kg] 计算 \(targets.count) 条向量")
        let vectorStart = Date()
        var lastProgress = -1
        try await store.storeVectors(targets: targets) { done, count in
            let pct = count > 0 ? done * 100 / count : 0
            if pct != lastProgress {
                lastProgress = pct
                progress?(done, count, "向量化 \(pct)%")
            }
        }

        graph.rebuild(entities: finalizedEntities, relationships: finalizedRelations)
        try store.markChunksProcessed(hashes: stats.succeededChunkHashes)
        try store.markBuildState(.complete)
        log?("[kg] 向量化耗时 " + String(format: "%.1fs", Date().timeIntervalSince(vectorStart)))
        log?("[kg] 知识图谱构建完成：实体 \(stats.entities)、关系 \(stats.relationships)")
        return stats
    }

    // MARK: - 增量构建

    /// 只处理不在 `kg_processed_chunks` 里的新切片，并与既有图谱合并。
    public func buildIncremental(progress: ((Int, Int, String) -> Void)? = nil,
                                 log: ((String) -> Void)? = nil)
        async throws -> IncrementalResult {
        guard try store.buildState() == .complete else {
            throw AgentError.storage("知识图谱尚未构建完成，无法增量更新")
        }
        await store.loadTokenCounter()

        let allRefs = try store.listChunkRefs()
        let processed = store.processedChunkIDs()
        let newRefs = allRefs.filter { !processed.contains($0.id) }
        var stats = BuildStats()
        guard !newRefs.isEmpty else {
            return IncrementalResult(stats: stats, changedEntities: [],
                                     changedRelationships: [], removedRelationshipIDs: [])
        }

        let loaded = try loadAccumulators()
        var entityAcc = loaded.entities
        var relationAcc = loaded.relations

        log?("[kg] 增量抽取 \(newRefs.count) 个新切片")
        let extracted = try await extractAll(newRefs, progress: progress, log: log,
                                             entities: entityAcc,
                                             relations: relationAcc)
        entityAcc = extracted.entities
        relationAcc = extracted.relations
        stats = extracted.stats

        let existingEntities = Dictionary(
            uniqueKeysWithValues: try store.allEntities().map { ($0.id, $0) })
        let existingRelations = Dictionary(
            uniqueKeysWithValues: try store.allRelationships().map { ($0.id, $0) })

        let allEntities = try await finalizeEntities(entityAcc)
        let allRelations = try await finalizeRelations(relationAcc)
        let changedEntities = allEntities.filter {
            Self.isChanged($0, comparedTo: existingEntities[$0.id])
        }
        let changedRelations = allRelations.filter {
            Self.isChanged($0, comparedTo: existingRelations[$0.id])
        }

        for entity in changedEntities { try store.upsertEntity(entity) }
        for rel in changedRelations { try store.upsertRelationship(rel) }
        for id in loaded.duplicateRelationIDs { try store.deleteRelationship(id: id) }

        let targets = store.vectorTargets(entities: changedEntities,
                                          relationships: changedRelations)
        if !targets.isEmpty {
            try await store.storeVectors(targets: targets, progress: { done, count in
                progress?(done, count, "增量向量化 \(count > 0 ? done * 100 / count : 0)%")
            })
        }

        try store.markChunksProcessed(hashes: stats.succeededChunkHashes)
        stats.entities = changedEntities.count
        stats.relationships = changedRelations.count
        log?("[kg] 增量更新完成：新增/变更实体 \(stats.entities)、关系 \(stats.relationships)")
        return IncrementalResult(stats: stats, changedEntities: changedEntities,
                                 changedRelationships: changedRelations,
                                 removedRelationshipIDs: loaded.duplicateRelationIDs)
    }

    // MARK: - 抽取

    /// 单个切片 → 原始输出（内部可见，供真机抽取质量测试复用同一路径）。
    ///
    /// 首轮输出不完整（无 `<|COMPLETE|>`、末行残缺、空结果或被重复行保护截断）
    /// 时追加一次补抽，只请求漏掉/需要纠正的行，再与首轮合并交给解析。
    func extractRaw(chunk: String, sectionContext: String = "") async throws -> String {
        let system: String
        let prompt: String
        if format == .json {
            system = KnowledgeGraphPrompts.entityExtractionJSONSystem(limits: limits)
            prompt = KnowledgeGraphPrompts.entityExtractionJSONTask(
                chunk: chunk, sectionContext: sectionContext, limits: limits)
        } else {
            system = KnowledgeGraphPrompts.entityExtractionSystem(limits: limits)
            prompt = KnowledgeGraphPrompts.entityExtractionTask(
                chunk: chunk, sectionContext: sectionContext, limits: limits)
        }
        let first = try await complete(system: system, user: prompt,
                                       temperature: Self.extractionTemperature,
                                       stopOnDuplicateRows: format == .delimited,
                                       jsonGuard: format == .json)
        // JSON 输出自带结构边界，解析成功即完整；解析失败不做二次补抽，
        // 直接按空结果处理（由调用方计入失败/空切片）。
        // 收流截断的 JSON 由 parseExtractionJSON 的配平兜底恢复。
        guard format == .delimited else { return first.text }
        let parsed = Self.parseExtraction(first.text, limits: limits)
        guard shouldGlean(first: first, parsed: parsed) else { return first.text }
        let continuePrompt = KnowledgeGraphPrompts.entityContinueExtractionTask(
            firstOutput: first.text, chunk: chunk,
            sectionContext: sectionContext, limits: limits)
        let second = try await complete(system: system, user: continuePrompt,
                                        temperature: Self.extractionTemperature,
                                        stopOnDuplicateRows: true)
        return first.text + "\n" + second.text
    }

    /// 是否需要二次补抽。只在「可能缺内容」时触发，避免完好的切片白跑一轮：
    /// 命中完成标记直接结束；已达实体/关系上限也无需再补；空结果或末行字段
    /// 残缺视为不完整。重复行触发的收流通常是模型退化复读，补抽只会再复读
    /// 一遍同样的内容（2B 实测两轮实体对完全一致），因此不再触发补抽。
    private func shouldGlean(
        first: (text: String, stoppedByDuplicate: Bool),
        parsed: KGExtractionResult
    ) -> Bool {
        if first.text.contains("<|COMPLETE|>") { return false }
        if parsed.entities.count >= limits.maxEntitiesPerChunk,
           parsed.relations.count >= limits.maxRelationsPerChunk {
            return false
        }
        // 重复行收流通常意味着模型进入复读退化，补抽只会再复读一遍
        //（2B 实测两轮实体对完全一致、耗时翻倍），因此不再补抽。
        if first.stoppedByDuplicate { return false }
        return parsed.isEmpty || Self.lastRowLooksIncomplete(first.text)
    }

    /// 末行是否是残缺记录（流被 token 上限截断时常见）。
    private static func lastRowLooksIncomplete(_ output: String) -> Bool {
        guard let line = output.split(separator: "\n", omittingEmptySubsequences: true)
            .last?.trimmingCharacters(in: .whitespacesAndNewlines) else { return true }
        if line == "<|COMPLETE|>" { return false }
        let parts = line.components(separatedBy: "<|#|>")
        // relation 行允许 4 段（缺 description 时解析器用 keywords 兜底），
        // 因此只有少于 4 段才算残缺。
        if parts.first == "entity" { return parts.count < 4 }
        if parts.first == "relation" { return parts.count < 4 }
        return true
    }

    /// 按配置的格式解析抽取输出。
    private func parse(_ raw: String) -> KGExtractionResult {
        format == .json
            ? Self.parseExtractionJSON(raw, limits: limits)
            : Self.parseExtraction(raw, limits: limits)
    }

    /// 单个切片 → 实体/关系抽取结果。
    func extract(chunk: String) async throws -> KGExtractionResult {
        parse(try await extractRaw(chunk: chunk))
    }

    /// 查询 → 高层/低层关键词（供检索使用）。
    func extractKeywords(query: String) async throws -> KGKeywords {
        let prompt = KnowledgeGraphPrompts.keywordExtractionPrompt(query: query)
        // system 必须非空且稳定：本地后端的 isolated 前缀副本池只在
        // systemPrompt 非空时生效，否则每次检索都全量 prefill（约 1.5s）。
        let output = try await complete(
            system: "你是知识图谱检索的关键词提取器。严格按用户给出的格式输出，不要解释。",
            user: prompt,
            temperature: Self.extractionTemperature).text
        return Self.parseKeywords(output)
    }

    /// 实体/关系描述归纳。
    func summarize(name: String, descriptions: [String],
                   descriptionType: String = "Entity") async throws -> String {
        let list = descriptions.enumerated()
            .map { "\($0.offset + 1). \($0.element)" }
            .joined(separator: "\n")
        let prompt = KnowledgeGraphPrompts.summarizeEntityDescriptionsPrompt(
            name: name, descriptionType: descriptionType, list: list,
            maxCharacters: limits.maxDescriptionCharacters)
        let output = try await complete(system: "", user: prompt,
                                        temperature: Self.summarizeTemperature).text
        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? descriptions.joined(separator: "\n") : trimmed
    }

    // MARK: - 单次直答

    /// JSON 抽取输出的收流保护。
    ///
    /// JSON 模式没有 `<|COMPLETE|>` 标记与逐行结构，模型一旦进入复读
    ///（重复输出实体数组/关系数组）会一直跑到 token 上限。这里做两级检测：
    /// - 快停：同一个 `"name"` 值出现多次（正常输出里每个实体名只出现一次，
    ///   实体数组复读会立刻让所有名字翻倍）；
    /// - 停滞：连续 stallChars 字符没有新增唯一实体名/关系端点值，
    ///   说明输出不再产生新信息（关系复读、纯文本空转都会命中）。
    /// 关系端点值允许合法重复（枢纽实体会被多条关系引用），因此只参与
    /// 停滞统计、不参与快停；实体名在 schema 上只会出现一次，可安全快停。
    struct JSONExtractionGuard {
        /// 每增长多少字符重新扫描一次（输出仅几 KB，全量重扫开销可忽略）。
        let rescanInterval: Int
        /// 同名实体出现次数阈值：达到即判定复读。
        let maxNameOccurrences: Int
        /// 无新增唯一值的最长容忍字符数。
        let stallChars: Int

        init(rescanInterval: Int = 256, maxNameOccurrences: Int = 3,
             stallChars: Int = 3_072) {
            self.rescanInterval = rescanInterval
            self.maxNameOccurrences = maxNameOccurrences
            self.stallChars = stallChars
        }

        private var lastScannedCount = 0
        private var nameCounts: [String: Int] = [:]
        private var progressTokens: Set<String> = []
        private var lastProgressAt = 0
        private(set) var stoppedReason = ""

        /// 返回 true 表示应收流停止。
        mutating func shouldStop(output: String) -> Bool {
            guard stoppedReason.isEmpty else { return true }
            // 统一用 UTF-16 偏移：跨多次重扫（字符串在增长）保持可比。
            let length = (output as NSString).length
            guard length - lastScannedCount >= rescanInterval else { return false }
            lastScannedCount = length
            rescan(output)
            if let (name, count) = nameCounts.first(where: { $0.value >= maxNameOccurrences }) {
                stoppedReason = "实体「\(name)」重复出现 \(count) 次（疑似复读）"
                return true
            }
            if length - lastProgressAt > stallChars {
                stoppedReason = "连续 \(stallChars) 字符未产生新的实体或关系"
                return true
            }
            return false
        }

        private static let namePattern = try! NSRegularExpression(
            pattern: #"\"name\"\s*:\s*\"((?:[^"\\]|\\.)*)\""#)
        private static let sourcePattern = try! NSRegularExpression(
            pattern: #"\"source(?:_entity)?\"\s*:\s*\"((?:[^"\\]|\\.)*)\""#)
        private static let targetPattern = try! NSRegularExpression(
            pattern: #"\"target(?:_entity)?\"\s*:\s*\"((?:[^"\\]|\\.)*)\""#)

        private mutating func rescan(_ output: String) {
            let ns = output as NSString
            let full = NSRange(location: 0, length: ns.length)
            func captures(_ regex: NSRegularExpression) -> [(String, Int)] {
                regex.matches(in: output, range: full).compactMap { match in
                    guard match.range(at: 1).location != NSNotFound else { return nil }
                    return (ns.substring(with: match.range(at: 1)),
                            match.range(at: 1).upperBound)
                }
            }
            let names = captures(Self.namePattern)
            nameCounts = names.reduce(into: [:]) { $0[$1.0, default: 0] += 1 }
            var tokens = Set(names.map { "n:\($0.0)" })
            // 进度位置取「最后一个匹配值的结束偏移」：新值出现在哪里，
            // 停滞计时就从哪里重新起算，而不是重扫发生的时刻。
            var lastMatchEnd = 0
            for (value, end) in names {
                lastMatchEnd = max(lastMatchEnd, end)
                tokens.insert("n:\(value)")
            }
            for (value, end) in captures(Self.sourcePattern) {
                lastMatchEnd = max(lastMatchEnd, end)
                tokens.insert("s:\(value)")
            }
            for (value, end) in captures(Self.targetPattern) {
                lastMatchEnd = max(lastMatchEnd, end)
                tokens.insert("t:\(value)")
            }
            if tokens.count > progressTokens.count {
                lastProgressAt = max(lastProgressAt, lastMatchEnd)
            }
            progressTokens = tokens
        }
    }

    private func complete(system: String, user: String,
                          temperature: Double,
                          stopOnDuplicateRows: Bool = false,
                          jsonGuard: Bool = false)
        async throws -> (text: String, stoppedByDuplicate: Bool) {
        var messages: [ChatMessage] = []
        if !system.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            messages.append(ChatMessage(role: .system, content: system))
        }
        messages.append(ChatMessage(role: .user, content: user))
        let options = LLMRequestOptions(enableThinking: false, isolated: true)
        var output = ""
        var scannedLines = 0
        var seenPairs = Set<String>()
        var duplicateRows = 0
        var stoppedByDuplicate = false
        var shouldStop = false
        var jsonGuardState = JSONExtractionGuard()
        let stream = llm.chatStream(messages: messages, tools: [],
                                    temperature: temperature, options: options)
        for try await event in stream {
            if event.kind == .content {
                output += event.content
                if stopOnDuplicateRows {
                    duplicateRows += Self.countDuplicateRelationRows(
                        in: output, scannedLines: &scannedLines,
                        seenPairs: &seenPairs)
                } else if jsonGuard, jsonGuardState.shouldStop(output: output) {
                    stoppedByDuplicate = true
                    shouldStop = true
                    AgentLog.info("[kg] JSON 收流：\(jsonGuardState.stoppedReason)")
                }
            }
            // 模型给出完成标记后立即停止消费，避免标记之后继续生成浪费 token。
            if output.contains("<|COMPLETE|>") { shouldStop = true }
            if duplicateRows >= Self.maxDuplicateRelationRows {
                stoppedByDuplicate = true
                shouldStop = true
            }
            if event.kind == .done, event.truncated {
                throw AgentError.llm("辅助模型输出被截断")
            }
            if shouldStop { break }
        }
        return (output, stoppedByDuplicate)
    }

    /// 扫描尚未处理过的完整行，返回新增的「重复实体对」关系行数。
    ///
    /// 只统计已换行结束的行，避免把流式输出中的半截行误判为重复。
    private static func countDuplicateRelationRows(
        in output: String, scannedLines: inout Int,
        seenPairs: inout Set<String>
    ) -> Int {
        let lines = output.components(separatedBy: "\n")
        guard lines.count > 1 else { return 0 }
        var duplicates = 0
        // 最后一段没有换行符，可能是未输出完的半行，留到下次再处理。
        while scannedLines < lines.count - 1 {
            let parts = lines[scannedLines]
                .components(separatedBy: "<|#|>")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            scannedLines += 1
            guard parts.count >= 3, parts[0] == "relation" else { continue }
            let pair = [parts[1].lowercased(), parts[2].lowercased()]
                .sorted().joined(separator: "|")
            guard !pair.isEmpty else { continue }
            if seenPairs.contains(pair) {
                duplicates += 1
            } else {
                seenPairs.insert(pair)
            }
        }
        return duplicates
    }

    // MARK: - 批量抽取

    private func extractAll(_ chunks: [KnowledgeGraphStore.ChunkListItem],
                            progress: ((Int, Int, String) -> Void)?,
                            log: ((String) -> Void)?,
                            entities: [String: EntityAcc] = [:],
                            relations: [String: RelationAcc] = [:])
        async throws -> (entities: [String: EntityAcc],
                         relations: [String: RelationAcc],
                         stats: BuildStats) {
        var entityAcc = entities
        var relationAcc = relations
        var stats = BuildStats()
        let total = chunks.count
        // 缓存键同时覆盖提示词版本、输出格式、限额与章节上下文：
        // 切换 JSON/分隔符或调整限额后不会误用旧格式的抽取结果。
        // 而 kg_processed_chunks 里记录的是纯正文哈希，用于判断切片是否被改写
        //（与缓存作用域无关，避免改配置时把整库误判为已修改）。
        let cacheScope = "\(Self.extractionCacheVersion)|\(format.rawValue)"
            + "|\(limits.maxEntitiesPerChunk)|\(limits.maxRelationsPerChunk)"
            + "|\(limits.maxDescriptionCharacters)"
        let contentHashes = chunks.map { Self.contentHash($0.text) }
        let cacheHashes = chunks.map {
            Self.contentHash(cacheScope + "|" + $0.heading + "|" + $0.text)
        }
        var rawOutputs = [String?](repeating: nil, count: total)
        var failures = [String?](repeating: nil, count: total)
        var pending: [(index: Int, chunk: KnowledgeGraphStore.ChunkListItem)] = []

        for (index, chunk) in chunks.enumerated() {
            if chunk.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                rawOutputs[index] = ""
                stats.succeededChunkHashes[chunk.id] = contentHashes[index]
                continue
            }
            if let cached = store.cachedExtraction(chunkID: chunk.id,
                                                   contentHash: cacheHashes[index]) {
                rawOutputs[index] = cached
                stats.cacheHits += 1
            } else {
                pending.append((index, chunk))
            }
        }

        // 串行后端并发只会堆在内部锁前，退化为逐条；支持独立会话时并发抽取。
        let concurrency = llm.supportsConcurrentIsolatedRequests
            ? min(maxConcurrency, max(1, pending.count)) : 1
        var done = total - pending.count
        if done > 0 {
            progress?(done, total, "抽取实体/关系 \(done)/\(total)")
        }
        var slow: [(String, Double)] = []

        var cursor = 0
        while cursor < pending.count {
            try Task.checkCancellation()
            let end = min(cursor + concurrency, pending.count)
            let batch = Array(pending[cursor..<end])
            let results: [(index: Int, raw: String?, error: String?, seconds: Double)]
            if batch.count == 1 {
                let item = batch[0]
                results = [await extractOne(index: item.index, chunk: item.chunk,
                                            contentHash: cacheHashes[item.index])]
            } else {
                results = await withTaskGroup(
                    of: (Int, String?, String?, Double).self) { group in
                    for item in batch {
                        group.addTask { [self] in
                            let result = await extractOne(index: item.index, chunk: item.chunk,
                                                          contentHash: cacheHashes[item.index])
                            return (result.index, result.raw, result.error, result.seconds)
                        }
                    }
                    var collected: [(Int, String?, String?, Double)] = []
                    for await result in group { collected.append(result) }
                    return collected
                }
            }
            for result in results {
                rawOutputs[result.index] = result.raw
                failures[result.index] = result.error
                if result.seconds > 3 {
                    slow.append((chunks[result.index].id, result.seconds))
                }
            }
            cursor = end
            done += batch.count
            if done % 10 == 0 || done == total || cursor >= pending.count {
                progress?(done, total, "抽取实体/关系 \(done)/\(total)")
            }
        }

        // 按切片原顺序聚合，保证实体/关系 id 与来源顺序稳定。
        for (index, chunk) in chunks.enumerated() {
            stats.chunksProcessed += 1
            if let error = failures[index] {
                stats.failedChunks += 1
                AgentLog.warning("[kg] 切片抽取失败（\(chunk.id)）: \(error)")
                continue
            }
            let raw = rawOutputs[index] ?? ""
            let extraction = parse(raw)
            accumulate(extraction, chunkID: chunk.id, filePath: chunk.source,
                       entities: &entityAcc, relations: &relationAcc)
            stats.succeededChunkHashes[chunk.id] = contentHashes[index]
        }
        if !slow.isEmpty {
            let top = slow.sorted { $0.1 > $1.1 }.prefix(3)
                .map { "\($0.0)(\(String(format: "%.1fs", $0.1)))" }
                .joined(separator: ", ")
            log?("[kg] 较慢切片：\(top)")
        }
        return (entityAcc, relationAcc, stats)
    }

    /// 单个切片抽取 + 落缓存；错误转为字符串，便于批量并发收集。
    private func extractOne(index: Int, chunk: KnowledgeGraphStore.ChunkListItem,
                            contentHash: String)
        async -> (index: Int, raw: String?, error: String?, seconds: Double) {
        let start = Date()
        do {
            let raw = try await extractRaw(chunk: chunk.text, sectionContext: chunk.heading)
            try? store.cacheExtraction(chunkID: chunk.id, contentHash: contentHash, output: raw)
            return (index, raw, nil, Date().timeIntervalSince(start))
        } catch {
            return (index, nil, error.localizedDescription,
                    Date().timeIntervalSince(start))
        }
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
                entities[key] = EntityAcc(id: UUID().uuidString, name: name,
                                          createdAt: Date().timeIntervalSince1970)
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
            // 关系身份按实体对归一：不同措辞的 keywords 作为同一条件的属性合并。
            let pair = KGRelationship.pairKey(src, dst)
            if relations[pair] == nil {
                let (lo, hi) = src.lowercased() <= dst.lowercased() ? (src, dst) : (dst, src)
                relations[pair] = RelationAcc(id: UUID().uuidString, src: lo, dst: hi,
                                              createdAt: Date().timeIntervalSince1970)
            }
            var acc = relations[pair]!
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
            if acc.countedChunks.insert(chunkID).inserted {
                acc.weightSum += 1
            }
            relations[pair] = acc

            // 关系引用了尚未出现的实体：现场补 UNKNOWN 占位节点（会一并落库）。
            for endpoint in [src, dst] {
                let key = endpoint.lowercased()
                if entities[key] == nil {
                    entities[key] = EntityAcc(id: "unknown-\(key)", name: endpoint,
                                              types: ["UNKNOWN"],
                                              sourceIDs: [chunkID],
                                              filePaths: filePath.isEmpty ? [] : [filePath],
                                              createdAt: Date().timeIntervalSince1970)
                } else {
                    var acc = entities[key]!
                    acc.sourceIDs.insert(chunkID)
                    if !filePath.isEmpty { acc.filePaths.insert(filePath) }
                    entities[key] = acc
                }
            }
        }
    }

    // MARK: - 已有图谱装载（增量）

    private func loadAccumulators() throws
        -> (entities: [String: EntityAcc],
            relations: [String: RelationAcc],
            duplicateRelationIDs: [String]) {
        var entities: [String: EntityAcc] = [:]
        for e in try store.allEntities() {
            let key = e.name.lowercased()
            let acc = EntityAcc(
                id: e.id, name: e.name,
                types: Set(e.type.split(separator: "/").map(String.init)),
                descriptions: e.description.isEmpty ? [] : [e.description],
                sourceIDs: Set(e.sourceIDs),
                filePaths: Set((e.properties["file_path"] ?? "")
                    .split(separator: ",").map(String.init)),
                createdAt: e.createdAt)
            if let existing = entities[key] {
                if existing.types == ["UNKNOWN"], acc.types != ["UNKNOWN"] {
                    entities[key] = acc
                }
            } else {
                entities[key] = acc
            }
        }

        var relations: [String: RelationAcc] = [:]
        var duplicates: [String] = []
        for r in try store.allRelationships() {
            let key = r.pairKey
            var acc = RelationAcc(
                id: r.id, src: r.srcEntity, dst: r.dstEntity,
                keywords: Set(r.relationType.split(separator: ",")
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }),
                descriptions: r.description.isEmpty ? [] : [r.description],
                sourceIDs: Set(r.sourceIDs),
                filePaths: Set((r.properties["file_path"] ?? "")
                    .split(separator: ",").map(String.init)),
                weightSum: Int(r.weight.rounded()),
                createdAt: r.createdAt)
            if let existing = relations[key] {
                acc.keywords.formUnion(existing.keywords)
                acc.sourceIDs.formUnion(existing.sourceIDs)
                acc.filePaths.formUnion(existing.filePaths)
                acc.descriptions.append(contentsOf: existing.descriptions
                    .filter { !acc.descriptions.contains($0) })
                acc.weightSum = max(acc.weightSum, existing.weightSum)
                acc.createdAt = min(acc.createdAt, existing.createdAt)
                duplicates.append(r.id)
            }
            relations[key] = acc
        }
        return (entities, relations, duplicates)
    }

    // MARK: - 归纳落库

    private func finalizeEntities(_ acc: [String: EntityAcc]) async throws -> [KGEntity] {
        let items = acc.sorted(by: { $0.key < $1.key }).map(\.value)
        var requests: [(key: String, name: String, descriptions: [String], type: String)] = []
        for item in items where item.descriptions.count >= Self.summarizeThreshold {
            requests.append((summaryKey(item.name, item.descriptions), item.name,
                             item.descriptions, "Entity"))
        }
        let summaries = await resolveSummaries(requests)
        var result: [KGEntity] = []
        result.reserveCapacity(items.count)
        for item in items {
            try Task.checkCancellation()
            var types = item.types.filter { !$0.isEmpty }
            if types.contains("UNKNOWN") && types.count > 1 {
                types.remove("UNKNOWN")
            }
            let type = types.isEmpty ? "Other" : types.sorted().joined(separator: "/")
            let description: String
            if item.descriptions.count == 1 {
                description = item.descriptions[0]
            } else if let summary = summaries[summaryKey(item.name, item.descriptions)] {
                description = summary
            } else {
                description = item.descriptions.joined(separator: "\n")
            }
            var properties: [String: String] = [:]
            if !item.filePaths.isEmpty {
                properties["file_path"] = item.filePaths.sorted().joined(separator: ",")
            }
            properties["source_chunk_ids"] = item.sourceIDs.sorted().joined(separator: ",")
            let now = Date().timeIntervalSince1970
            result.append(KGEntity(id: item.id, name: item.name, type: type,
                                   description: Self.truncateDescription(
                                       description, limit: limits.maxDescriptionCharacters),
                                   properties: properties,
                                   sourceIDs: item.sourceIDs.sorted(),
                                   createdAt: item.createdAt > 0 ? item.createdAt : now,
                                   updatedAt: now))
        }
        return result
    }

    private func finalizeRelations(_ acc: [String: RelationAcc]) async throws -> [KGRelationship] {
        let items = acc.sorted(by: { $0.key < $1.key }).map(\.value)
        var requests: [(key: String, name: String, descriptions: [String], type: String)] = []
        for item in items where item.descriptions.count >= Self.summarizeThreshold {
            let name = "\(item.src)-\(item.dst)"
            requests.append((summaryKey(name, item.descriptions), name,
                             item.descriptions, "Relation"))
        }
        let summaries = await resolveSummaries(requests)
        var result: [KGRelationship] = []
        result.reserveCapacity(items.count)
        for item in items {
            try Task.checkCancellation()
            let relationType = item.keywords.sorted().joined(separator: ",")
            let summaryKeyValue = summaryKey("\(item.src)-\(item.dst)", item.descriptions)
            let description: String
            if item.descriptions.count == 1 {
                description = item.descriptions[0]
            } else if let summary = summaries[summaryKeyValue] {
                description = summary
            } else {
                description = item.descriptions.joined(separator: "\n")
            }
            // weight = max(累加权重, 去重后的来源切片数量)；缺失兜底 1。
            let evidence = max(1, item.sourceIDs.count)
            let weight = Double(max(item.weightSum, evidence))
            let sources = item.sourceIDs.sorted()
            let finalDescription = Self.truncateDescription(
                description, limit: limits.maxDescriptionCharacters)
            var properties: [String: String] = [:]
            if !item.filePaths.isEmpty {
                properties["file_path"] = item.filePaths.sorted().joined(separator: ",")
            }
            // 边属性字典里带齐查询/溯源所需信息。
            properties["description"] = finalDescription
            properties["keywords"] = relationType
            properties["weight"] = String(weight)
            properties["source_chunk_ids"] = sources.joined(separator: ",")
            let now = Date().timeIntervalSince1970
            result.append(KGRelationship(id: item.id, srcEntity: item.src,
                                         dstEntity: item.dst, relationType: relationType,
                                         description: finalDescription, weight: weight,
                                         properties: properties, sourceIDs: sources,
                                         createdAt: item.createdAt > 0 ? item.createdAt : now,
                                         updatedAt: now))
        }
        return result
    }

    private func summaryKey(_ name: String, _ descriptions: [String]) -> String {
        Self.contentHash(Self.summaryCacheVersion + "\u{0}" + name + "\u{0}"
            + descriptions.joined(separator: "\u{1}"))
    }

    /// 归纳去重：先查缓存，只对缺失项调用模型；后端支持独立会话时并发归纳。
    private func resolveSummaries(
        _ requests: [(key: String, name: String, descriptions: [String], type: String)])
        async -> [String: String] {
        var summaries: [String: String] = [:]
        var missing: [(key: String, name: String, descriptions: [String], type: String)] = []
        for request in requests {
            if let cached = store.cachedSummary(key: request.key) {
                summaries[request.key] = cached
            } else {
                missing.append(request)
            }
        }
        guard !missing.isEmpty else { return summaries }

        let limit = llm.supportsConcurrentIsolatedRequests
            ? min(maxConcurrency, max(1, missing.count)) : 1
        var cursor = 0
        while cursor < missing.count {
            let end = min(cursor + limit, missing.count)
            let batch = Array(missing[cursor..<end])
            var produced: [(String, String)] = []
            if batch.count == 1 {
                let item = batch[0]
                if let summary = try? await summarize(name: item.name,
                                                      descriptions: item.descriptions,
                                                      descriptionType: item.type) {
                    produced.append((item.key, summary))
                }
            } else {
                let results = await withTaskGroup(of: (String, String?).self) { group in
                    for item in batch {
                        group.addTask { [self] in
                            let summary = try? await summarize(name: item.name,
                                                               descriptions: item.descriptions,
                                                               descriptionType: item.type)
                            return (item.key, summary)
                        }
                    }
                    var output: [(String, String?)] = []
                    for await result in group { output.append(result) }
                    return output
                }
                produced = results.compactMap { key, summary in
                    summary.map { (key, $0) }
                }
            }
            for (key, summary) in produced {
                summaries[key] = summary
                store.cacheSummary(key: key, summary: summary)
            }
            cursor = end
        }
        return summaries
    }

    private static func isChanged(_ new: KGEntity, comparedTo old: KGEntity?) -> Bool {
        guard let old else { return true }
        return new.type != old.type
            || new.description != old.description
            || new.properties != old.properties
            || Set(new.sourceIDs) != Set(old.sourceIDs)
    }

    private static func isChanged(_ new: KGRelationship, comparedTo old: KGRelationship?) -> Bool {
        guard let old else { return true }
        return new.relationType != old.relationType
            || new.description != old.description
            || new.weight != old.weight
            || new.properties != old.properties
            || Set(new.sourceIDs) != Set(old.sourceIDs)
    }

    // MARK: - 解析

    static func normalizeName(_ name: String) -> String {
        name.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// 稳定的内容指纹（FNV-1a 64），跨进程一致，用于判断切片内容是否变化。
    static func contentHash(_ text: String) -> String {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x100000001b3
        }
        return String(hash, radix: 16)
    }

    /// 描述截断到指定字符数（按 Unicode 字符计数，中文按字算）。
    static func truncateDescription(_ text: String, limit: Int) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count > limit else { return trimmed }
        return String(trimmed.prefix(limit))
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func parseExtraction(_ output: String,
                                limits: KGExtractionLimits = .default) -> KGExtractionResult {
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
                // 纯数字/年份不是有效实体（模型偶尔会从描述里造出 “2010” 节点）。
                guard !name.allSatisfy({ $0.isNumber }) else { continue }
                entities.append(KGExtractedEntity(
                    name: name, type: parts[2],
                    description: truncateDescription(
                        parts[3], limit: limits.maxDescriptionCharacters)))
            } else if parts.count >= 4, parts[0] == "relation" {
                let src = normalizeName(parts[1])
                let dst = normalizeName(parts[2])
                guard !src.isEmpty, !dst.isEmpty else { continue }
                // 纯数字端点（如年份 2010）不是有效实体，避免生成无意义的占位节点。
                guard !src.allSatisfy({ $0.isNumber }),
                      !dst.allSatisfy({ $0.isNumber }) else { continue }
                let keywords = parts[3]
                // 小模型偶尔只输出到 keywords（4 段），缺 description；
                // 用 keywords 兜底，避免整条有效关系被丢弃。
                let description = parts.count >= 5 && !parts[4].isEmpty
                    ? parts[4] : keywords
                relations.append(KGExtractedRelation(
                    source: src, target: dst, keywords: keywords,
                    description: truncateDescription(
                        description, limit: limits.maxDescriptionCharacters)))
            }
        }
        // 模型未严格遵守提示词时兜底：每个切片只保留前 N 个实体/关系。
        if entities.count > limits.maxEntitiesPerChunk {
            entities = Array(entities.prefix(limits.maxEntitiesPerChunk))
        }
        if relations.count > limits.maxRelationsPerChunk {
            relations = Array(relations.prefix(limits.maxRelationsPerChunk))
        }
        return KGExtractionResult(entities: entities, relations: relations)
    }

    /// 解析 JSON 格式的抽取输出（允许模型加代码围栏或少量前后说明）。
    static func parseExtractionJSON(_ output: String,
                                    limits: KGExtractionLimits = .default) -> KGExtractionResult {
        guard let object = extractionJSONObject(in: output)?.objectValue else {
            return KGExtractionResult()
        }
        var entities: [KGExtractedEntity] = []
        var relations: [KGExtractedRelation] = []
        for item in object["entities"]?.arrayValue ?? [] {
            guard let obj = item.objectValue else { continue }
            let name = normalizeName(obj["name"]?.stringValue ?? "")
            guard !name.isEmpty, !name.allSatisfy({ $0.isNumber }) else { continue }
            let type = obj["type"]?.stringValue ?? "Other"
            let description = obj["description"]?.stringValue ?? ""
            entities.append(KGExtractedEntity(
                name: name, type: type,
                description: truncateDescription(
                    description, limit: limits.maxDescriptionCharacters)))
        }
        let relationItems = object["relationships"]?.arrayValue
            ?? object["relations"]?.arrayValue ?? []
        for item in relationItems {
            guard let obj = item.objectValue else { continue }
            let src = normalizeName(obj["source"]?.stringValue
                ?? obj["source_entity"]?.stringValue ?? "")
            let dst = normalizeName(obj["target"]?.stringValue
                ?? obj["target_entity"]?.stringValue ?? "")
            guard !src.isEmpty, !dst.isEmpty else { continue }
            guard !src.allSatisfy({ $0.isNumber }),
                  !dst.allSatisfy({ $0.isNumber }) else { continue }
            let keywords = jsonKeywords(obj["keywords"] ?? obj["relationship_keywords"])
            let rawDescription = obj["description"]?.stringValue
                ?? obj["relationship_description"]?.stringValue ?? ""
            let description = rawDescription.isEmpty ? keywords : rawDescription
            relations.append(KGExtractedRelation(
                source: src, target: dst, keywords: keywords,
                description: truncateDescription(
                    description, limit: limits.maxDescriptionCharacters)))
        }
        if entities.count > limits.maxEntitiesPerChunk {
            entities = Array(entities.prefix(limits.maxEntitiesPerChunk))
        }
        if relations.count > limits.maxRelationsPerChunk {
            relations = Array(relations.prefix(limits.maxRelationsPerChunk))
        }
        return KGExtractionResult(entities: entities, relations: relations)
    }

    /// 从可能包含代码围栏/说明文字的模型输出中提取 JSON 对象。
    private static func jsonObject(in output: String) -> JSONValue? {
        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        if let value = JSONValue.parse(trimmed) { return value }
        guard let start = trimmed.firstIndex(of: "{"),
              let end = trimmed.lastIndex(of: "}"), start < end else { return nil }
        return JSONValue.parse(String(trimmed[start...end]))
    }

    /// 收流保护可能产生「完整对象 + 复读尾巴」或「半截 JSON」，
    /// 因此解析按四级兜底：整段 → 首{}截取 → 第一个配平对象（丢弃复读）
    /// → 补齐缺失括号（截断恢复，尽量保留已输出的实体/关系）。
    static func extractionJSONObject(in output: String) -> JSONValue? {
        if let value = jsonObject(in: output) { return value }
        if let balanced = firstBalancedObject(in: output),
           let value = JSONValue.parse(balanced) { return value }
        if let closed = closedTruncatedObject(in: output),
           let value = JSONValue.parse(closed) { return value }
        return nil
    }

    /// 返回第一个配平的顶层 `{...}` 子串（字符串内的括号不计深度）。
    /// 复读场景：完整对象后面跟着重复内容，取第一个即丢弃复读。
    static func firstBalancedObject(in output: String) -> String? {
        guard let start = output.firstIndex(of: "{") else { return nil }
        var depth = 0
        var inString = false
        var escaped = false
        var index = start
        while index < output.endIndex {
            let ch = output[index]
            if escaped {
                escaped = false
            } else if inString && ch == "\\" {
                escaped = true
            } else if ch == "\"" {
                inString.toggle()
            } else if !inString {
                if ch == "{" {
                    depth += 1
                } else if ch == "}" {
                    depth -= 1
                    if depth == 0 { return String(output[start...index]) }
                }
            }
            index = output.index(after: index)
        }
        return nil
    }

    /// 截断 JSON 兜底：补齐缺失的 `]` `}` 使其可解析。
    /// 先尝试直接补齐；失败则丢弃最后一个不完整项（字符串外的最后一个
    /// 逗号之后的内容）再补齐。字符串截半、悬空键名都由后者处理。
    static func closedTruncatedObject(in output: String) -> String? {
        guard let start = output.firstIndex(of: "{") else { return nil }
        func close(_ prefix: Substring) -> String? {
            var stack: [Character] = []
            var inString = false
            var escaped = false
            var index = prefix.startIndex
            while index < prefix.endIndex {
                let ch = prefix[index]
                if escaped {
                    escaped = false
                } else if inString && ch == "\\" {
                    escaped = true
                } else if ch == "\"" {
                    inString.toggle()
                } else if !inString {
                    if ch == "{" { stack.append("}") }
                    else if ch == "[" { stack.append("]") }
                    else if ch == "}" || ch == "]" { _ = stack.popLast() }
                }
                index = output.index(after: index)
            }
            var closed = String(prefix)
            if inString { closed += "\"" }
            closed += String(stack.reversed())
            return JSONValue.parse(closed) == nil ? nil : closed
        }
        if let closed = close(output[start...]) { return closed }
        // 找最后一个字符串外的逗号，丢弃其后可能残缺的键值对/元素。
        var lastComma: String.Index?
        var inString = false
        var escaped = false
        var index = start
        while index < output.endIndex {
            let ch = output[index]
            if escaped {
                escaped = false
            } else if inString && ch == "\\" {
                escaped = true
            } else if ch == "\"" {
                inString.toggle()
            } else if !inString && ch == "," {
                lastComma = index
            }
            index = output.index(after: index)
        }
        guard let comma = lastComma else { return nil }
        return close(output[start..<comma])
    }

    private static func jsonKeywords(_ value: JSONValue?) -> String {
        if let string = value?.stringValue {
            return string.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let array = value?.arrayValue {
            return array.compactMap { $0.stringValue }
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
                .joined(separator: ", ")
        }
        return ""
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
