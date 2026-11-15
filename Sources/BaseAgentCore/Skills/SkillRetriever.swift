import Foundation

/// 语义/混合检索返回的候选技能。
public struct RetrievedSkill: Sendable, Equatable {
    public var dir: String
    public var name: String
    public var tags: [String]
    public var description: String
    public var score: Double
    public var bm25Score: Double

    public init(dir: String, name: String, tags: [String], description: String,
                score: Double, bm25Score: Double) {
        self.dir = dir
        self.name = name
        self.tags = tags
        self.description = description
        self.score = score
        self.bm25Score = bm25Score
    }
}

/// 技能语义检索器。
///
/// 召回策略与 Python `skill_retriever.py` 对齐：
/// 1. 向量通道：description（权重 0.6）+ examples（0.4），单通道阈值 0.5；
/// 2. BM25 关键词通道；
/// 3. 混合时向量得分优先，BM25 命中只作为补充（保证词面匹配不漏）；
/// 4. 未注入 embedding 时仅使用 BM25，完全离线可用。
public final class SkillRetriever {
    public let manager: SkillManager
    public var skillsDirectory: String { manager.skillsDirectory }
    public var builtinDirectories: [String] { manager.builtinDirectories }

    private let embedding: EmbeddingProvider?
    private let lock = NSRecursiveLock()

    private struct SkillVector: Codable {
        var fingerprint: String
        var description: [Float]
        var examples: [Float]
        var examplesEmpty: Bool
    }

    /// 技能向量磁盘缓存。按技能绝对路径 + 内容指纹校验，技能文件变化时
    /// 只重算变化项；embedding 模型或维度改变时整份作废。
    private struct VectorCacheFile: Codable {
        var version: Int
        var embedding: String
        var dimension: Int
        var skills: [String: SkillVector]
    }

    private struct BM25Doc {
        var skill: Skill
        var termFrequency: [String: Int]
        var length: Int
    }

    private var vectors: [String: SkillVector] = [:]   // key = 技能绝对路径
    private var bm25Docs: [BM25Doc] = []
    private var idf: [String: Double] = [:]
    private var averageLength: Double = 0
    private var bm25Signature = ""
    private var forceScan = false

    /// 向量缓存文件放在技能目录的上一级（App 下即 `.agent/skill_vectors.json`），
    /// 避免被技能扫描当成技能目录。
    private let cachePath: String
    private let cachePathLock = NSLock()
    private var cacheLoaded = false
    private var cacheDirty = false
    private static let cacheVersion = 3
    /// 可选精排器：向量+BM25 先取短名单，再用多语 reranker 按名称/描述精排，
    /// 显著改善中文任务 ↔ 英文技能名/描述的跨语言召回。
    private let reranker: RerankProvider?

    private let k1 = 1.5
    private let b = 0.75
    /// 向量同步串行化：`skill_tools` 的 semantic_search 可并行执行，首次
    /// （或缓存失效后）并发检索若各自重算同一批 2000+ 技能向量，会把
    /// embedding 开销翻倍。这里用单许可信号量让后来者等待，并在获得许可后
    /// 重新计算缺失集合，直接复用前一次的结果。
    private let syncGate = AsyncSemaphore(value: 1)

    public init(manager: SkillManager, embedding: EmbeddingProvider? = nil,
                reranker: RerankProvider? = nil) {
        self.manager = manager
        self.embedding = embedding
        self.reranker = reranker
        let parent = (manager.skillsDirectory as NSString).deletingLastPathComponent
        self.cachePath = (parent as NSString).appendingPathComponent("skill_vectors.json")
    }

    public convenience init(skillsDirectory: String,
                            builtinDirectories: [String] = [],
                            embedding: EmbeddingProvider? = nil) {
        self.init(manager: SkillManager(skillsDirectory: skillsDirectory,
                                        builtinDirectories: builtinDirectories),
                  embedding: embedding)
    }

    /// 便捷构造：主目录 `<workspace>/skills`。
    public convenience init(workspace: String,
                            builtinDirectories: [String] = [],
                            embedding: EmbeddingProvider? = nil) {
        self.init(manager: SkillManager(workspace: workspace,
                                        builtinDirectories: builtinDirectories),
                  embedding: embedding)
    }

    public func markStale() {
        lock.lock(); forceScan = true; lock.unlock()
        manager.markStale()
    }

    /// 读取技能目录（拒绝 `..` / 绝对路径；允许指向技能目录的外部软链接）。
    public func readSkillDir(_ path: String) -> String? {
        manager.resolveSkillDirectory(path)
    }

    /// 混合检索，返回带目录的候选。
    public func retrieve(query: String, topK: Int = 3) async -> [RetrievedSkill] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return [] }
        let limit = max(1, min(topK, 5))
        let skills = await refreshIndex()

        let queryTokens = SkillTextTokenizer.tokens(q)
        let rawBM25 = bm25Score(for: queryTokens)
        // 有 reranker 时扩大召回窗口，先粗排再精排；没有时沿用前 3。
        let recall = reranker == nil ? limit : max(limit * 10, 40)
        let bm25Top = rawBM25.sorted {
            if $0.value != $1.value { return $0.value > $1.value }
            return $0.key < $1.key
        }.prefix(recall)
        let bm25Scores = Dictionary(uniqueKeysWithValues: bm25Top.map { ($0.key, $0.value) })

        var queryVector: [Float]?
        if let embedding {
            do {
                let vector = try await embedding.embedQuery(q)
                if vector.isEmpty {
                    AgentLog.warning("[skill] 查询向量为空，回退 BM25")
                } else {
                    queryVector = vector
                }
            } catch {
                AgentLog.warning("[skill] 查询向量生成失败，回退 BM25: "
                    + error.localizedDescription)
            }
        }

        var candidates: [String: RetrievedSkill] = [:]
        var byPath: [String: Skill] = [:]
        for skill in skills { byPath[skill.absolutePath] = skill }

        // 1) 关键词通道
        for (path, score) in bm25Scores where score > 0 {
            guard let skill = byPath[path] else { continue }
            candidates[path] = RetrievedSkill(
                dir: skill.directory, name: skill.name, tags: skill.tags,
                description: skill.description,
                score: queryVector == nil ? score : 0,
                bm25Score: score)
        }

        // 2) 向量通道（含对关键词命中的分数补齐）；取融合分前 topK 个候选。
        //    不能用固定小窗口截断：技能少时会把语义正确的候选整个滤掉
        //    （实测 4 技能取 3 时正确技能被截，BM25 候选又因 score 置 0 沉底）。
        if let queryVector {
            lock.lock(); let snapshot = vectors; lock.unlock()
            var ranked: [(path: String, candidate: RetrievedSkill)] = []
            for skill in skills {
                guard let vector = snapshot[skill.absolutePath],
                      vector.fingerprint == skill.fingerprint else { continue }
                let descriptionScore = Self.cosine(queryVector, vector.description)
                let fused: Double
                if vector.examplesEmpty {
                    fused = descriptionScore
                } else {
                    let examplesScore = Self.cosine(queryVector, vector.examples)
                    fused = 0.6 * descriptionScore + 0.4 * examplesScore
                }
                let keywordScore = bm25Scores[skill.absolutePath] ?? 0
                ranked.append((path: skill.absolutePath, candidate: RetrievedSkill(
                    dir: skill.directory, name: skill.name, tags: skill.tags,
                    description: skill.description,
                    score: fused,
                    bm25Score: keywordScore)))
            }
            ranked.sort {
                if $0.candidate.score != $1.candidate.score {
                    return $0.candidate.score > $1.candidate.score
                }
                return $0.path < $1.path
            }
            // 有 reranker 时放宽阈值，让它对更大的短名单做精排。
            let vectorThreshold = reranker == nil ? 0.5 : 0
            for item in ranked.prefix(recall) {
                let keywordScore = bm25Scores[item.path] ?? 0
                guard item.candidate.score >= vectorThreshold || keywordScore > 0 else { continue }
                let existing = candidates[item.path]
                candidates[item.path] = RetrievedSkill(
                    dir: item.candidate.dir, name: item.candidate.name,
                    tags: item.candidate.tags, description: item.candidate.description,
                    score: max(item.candidate.score, existing?.score ?? 0),
                    bm25Score: keywordScore)
            }
        }

        // 同名技能可能来自不同目录（如 writing-skills 与 skills-codex 的同名副本），
        // 只保留分数最高的一份，避免重复项占满备选名额。
        var seenNames = Set<String>()
        let unique = candidates.values.sorted {
            if $0.score != $1.score { return $0.score > $1.score }
            if $0.bm25Score != $1.bm25Score { return $0.bm25Score > $1.bm25Score }
            return $0.name.localizedStandardCompare($1.name) == .orderedAscending
        }.filter { seenNames.insert($0.name.lowercased()).inserted }
        // 有关键词命中但没有进入向量 topK 的候选分数为 0：有正向语义分时
        // 不再展示它们，避免出现 0.000 的干扰项；全部为 0 时仍回退展示。
        let positive = unique.filter { $0.score > 0 }
        var ranked = positive.isEmpty ? unique : positive

        // 向量模型很小，跨语言（中文任务 ↔ 英文技能名/描述）时排序容易接近。
        // 用多语 reranker 对前 12 个候选按「名称 + 描述」精排，只影响排序。
        if let reranker {
            let shortlist = Array(ranked.prefix(recall))
            let documents = shortlist.map { "\($0.name). \($0.description)" }
            do {
                let scores = try await reranker.rerank(query: q, documents: documents)
                if scores.count == shortlist.count {
                    var reranked = shortlist
                    for index in reranked.indices {
                        reranked[index].score = scores[index]
                    }
                    ranked = reranked.sorted {
                        if $0.score != $1.score { return $0.score > $1.score }
                        return $0.name.localizedStandardCompare($1.name) == .orderedAscending
                    }
                } else {
                    AgentLog.warning("[skill] reranker 返回数量不一致，回退向量排序")
                }
            } catch {
                AgentLog.warning("[skill] reranker 精排失败，回退向量排序: "
                    + error.localizedDescription)
            }
        }
        return Array(ranked.prefix(limit))
    }

    /// 转换成共享契约中的 `SkillSuggestion`。
    public func suggestions(query: String, topK: Int = 3) async -> [SkillSuggestion] {
        await retrieve(query: query, topK: topK).map {
            SkillSuggestion(name: $0.name,
                            description: $0.description,
                            score: $0.score)
        }
    }

    /// 释放内存索引（检索器仍可继续使用，下次会重建）。
    /// 向量已持久化到磁盘，这里只清内存，下次检索直接重新加载缓存。
    public func close() {
        lock.lock()
        vectors = [:]
        bm25Docs = []
        idf = [:]
        bm25Signature = ""
        cacheLoaded = false
        cacheDirty = false
        lock.unlock()
    }

    // MARK: - 索引同步

    private func refreshIndex() async -> [Skill] {
        lock.lock()
        let force = forceScan
        forceScan = false
        lock.unlock()

        ensureVectorCacheLoaded()
        let skills = manager.scan(force: force)
        rebuildBM25(skills, force: force)

        if let embedding {
            await syncVectors(skills, embedding: embedding)
        }
        persistVectorsIfNeeded()
        return skills
    }

    // MARK: - 向量缓存

    /// 启动/首次检索时把磁盘向量加载进内存；模型或维度不匹配则整份丢弃。
    private func ensureVectorCacheLoaded() {
        guard let embedding else { return }
        cachePathLock.lock()
        let alreadyLoaded = cacheLoaded
        cachePathLock.unlock()
        guard !alreadyLoaded else { return }

        let dimension = embedding.dimension
        var loaded: [String: SkillVector] = [:]
        var invalid = false
        if let data = FileManager.default.contents(atPath: cachePath),
           let file = try? JSONDecoder().decode(VectorCacheFile.self, from: data) {
            if file.version == Self.cacheVersion,
               file.embedding == embedding.identifier,
               file.dimension == dimension {
                loaded = file.skills.filter {
                    $0.value.description.count == dimension
                        && $0.value.examples.count == dimension
                }
            } else {
                invalid = true
            }
        }

        lock.lock()
        if !cacheLoaded {
            // 只补内存里还没有的路径：并发首次检索已经算出的结果优先。
            for (path, vector) in loaded where vectors[path] == nil {
                vectors[path] = vector
            }
            cacheLoaded = true
            if invalid { cacheDirty = true }
        }
        lock.unlock()
        if invalid {
            AgentLog.warning("[skill] 向量缓存失效（版本/模型/维度变化），将重建")
        } else if !loaded.isEmpty {
            AgentLog.info("[skill] 向量缓存已加载 n=\(loaded.count)")
        }
    }

    /// 把最新向量快照原子写回磁盘。无变化时不写，避免每次检索都重写文件。
    private func persistVectorsIfNeeded() {
        guard let embedding else { return }
        lock.lock()
        guard cacheDirty else { lock.unlock(); return }
        let snapshot = vectors
        cacheDirty = false
        lock.unlock()

        let payload = VectorCacheFile(version: Self.cacheVersion,
                                      embedding: embedding.identifier,
                                      dimension: embedding.dimension,
                                      skills: snapshot)
        guard let data = try? JSONEncoder().encode(payload) else { return }
        let directory = (cachePath as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(atPath: directory,
                                                 withIntermediateDirectories: true)
        do {
            try AtomicFile.write(data, to: cachePath)
        } catch {
            AgentLog.warning("skill_retriever: 向量缓存写入失败: "
                + error.localizedDescription)
        }
    }

    private func rebuildBM25(_ skills: [Skill], force: Bool) {
        // 技能未变化时跳过全文分词：每次检索都对所有技能重新 tokenize
        // 是纯粹的重复开销。
        let signature = skills.map { "\($0.absolutePath)|\($0.fingerprint)" }
            .joined(separator: "\n")
        lock.lock()
        let unchanged = !force && signature == bm25Signature && !bm25Docs.isEmpty
        lock.unlock()
        if unchanged { return }

        var docs: [BM25Doc] = []
        var documentFrequency: [String: Int] = [:]
        var totalLength = 0

        for skill in skills {
            let text = String([
                skill.name,
                skill.tags.joined(separator: " "),
                skill.description,
                skill.whenToUse,
                String(skill.examples.prefix(3_000)),
            ].joined(separator: "\n").prefix(6_000))
            let tokens = SkillTextTokenizer.tokens(text)
            guard !tokens.isEmpty else { continue }
            var tf: [String: Int] = [:]
            for token in tokens { tf[token, default: 0] += 1 }
            for token in Set(tokens) { documentFrequency[token, default: 0] += 1 }
            docs.append(BM25Doc(skill: skill, termFrequency: tf, length: tokens.count))
            totalLength += tokens.count
        }

        let count = Double(max(docs.count, 1))
        var computedIDF: [String: Double] = [:]
        for (token, df) in documentFrequency {
            computedIDF[token] = log((count - Double(df) + 0.5) / (Double(df) + 0.5) + 1)
        }

        lock.lock()
        bm25Docs = docs
        idf = computedIDF
        averageLength = docs.isEmpty ? 0 : Double(totalLength) / Double(docs.count)
        bm25Signature = signature
        // 清理已删除技能的向量，避免内存无界增长；有删除时标记缓存待落盘。
        let alive = Set(skills.map(\.absolutePath))
        let before = vectors.count
        vectors = vectors.filter { alive.contains($0.key) }
        if vectors.count != before { cacheDirty = true }
        lock.unlock()
    }

    private func bm25Score(for queryTokens: [String]) -> [String: Double] {
        guard !queryTokens.isEmpty else { return [:] }
        lock.lock(); let docs = bm25Docs; let idfSnapshot = idf
        let avg = averageLength; lock.unlock()
        guard avg > 0 else { return [:] }

        var scores: [String: Double] = [:]
        for doc in docs {
            var score = 0.0
            for token in queryTokens {
                guard let tf = doc.termFrequency[token], let tokenIDF = idfSnapshot[token] else {
                    continue
                }
                let denominator = Double(tf) + k1 * (1 - b + b * Double(doc.length) / avg)
                score += tokenIDF * Double(tf) * (k1 + 1) / denominator
            }
            if score > 0 {
                scores[doc.skill.absolutePath] = score
            }
        }
        return scores
    }

    private func syncVectors(_ skills: [Skill], embedding: EmbeddingProvider) async {
        guard await syncGate.acquire() else { return }
        defer { syncGate.release() }
        let dimension = embedding.dimension
        var missing: [Skill] = []
        lock.lock()
        for skill in skills {
            let cached = vectors[skill.absolutePath]
            let valid = cached?.fingerprint == skill.fingerprint
                && cached?.description.count == dimension
                && cached?.examples.count == dimension
            if !valid { missing.append(skill) }
        }
        lock.unlock()
        guard !missing.isEmpty else { return }

        let descriptionTexts = missing.map { skill in
            String((skill.description.isEmpty ? skill.name : skill.description).prefix(2_000))
        }
        let exampleTexts = missing.enumerated().map { index, skill -> String in
            // 没有 examples 时用「名称 + tags + keywords」作为第二路向量：
            // 名称往往就是任务名（test-driven-development / meeting-insights-analyzer），
            // 只嵌入长描述会把这些强信号稀释掉。
            guard skill.examples.isEmpty else {
                return String(skill.examples.prefix(3_000))
            }
            let header = ([skill.name] + skill.tags + skill.keywords)
                .filter { !$0.isEmpty }
                .joined(separator: " ")
            return String((header.isEmpty ? skill.name : header).prefix(2_000))
        }

        do {
            // 描述与示例经常完全相同（无 examples 时回退 description）：
            // 去重后只 embed 一次，长技能列表下能省掉近一半推理。
            var order: [String] = []
            var seen = Set<String>()
            for text in descriptionTexts + exampleTexts where seen.insert(text).inserted {
                order.append(text)
            }
            var cache: [String: [Float]] = [:]
            var offset = 0
            while offset < order.count {
                try Task.checkCancellation()
                let batch = Array(order[offset..<min(offset + 32, order.count)])
                let embedded = try await embedding.embedDocuments(batch)
                guard embedded.count == batch.count else {
                    throw AgentError.storage("skill_retriever embedding 数量不一致")
                }
                for (text, vector) in zip(batch, embedded) { cache[text] = vector }
                offset += batch.count
            }
            lock.lock()
            for (index, skill) in missing.enumerated() {
                guard let descriptionVector = cache[descriptionTexts[index]],
                      let exampleVector = cache[exampleTexts[index]] else { continue }
                vectors[skill.absolutePath] = SkillVector(
                    fingerprint: skill.fingerprint,
                    description: descriptionVector,
                    examples: exampleVector,
                    examplesEmpty: false)
            }
            cacheDirty = true
            lock.unlock()
            lock.lock(); let total = vectors.count; lock.unlock()
            AgentLog.info("[skill] 向量同步完成 added=\(missing.count) total=\(total)")
        } catch {
            AgentLog.warning("skill_retriever: 语义索引不可用，回退 BM25: \(error.localizedDescription)")
        }
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
        let denominator = na.squareRoot() * nb.squareRoot()
        return denominator > 0 ? max(0, min(1, dot / denominator)) : 0
    }
}
