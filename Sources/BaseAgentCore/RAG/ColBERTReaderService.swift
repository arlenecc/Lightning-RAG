import Foundation
import CryptoKit
import ColBERTReader

/// 检索阅读模式的服务封装。
///
/// 负责 ColBERT 模型的自动下载（hf-mirror.com）与加载、文档入库、
/// 检索召回、向量索引构建，并把结果映射为 `RAGCitation` 召回切片气泡。
///
/// 启用该模式后，得分最高的第 1 条分片原文进入模型上下文供 LLM 总结
/// （`isPrimary == true`），其余召回分片作为引用气泡供用户点开阅读。
public final class ColBERTReaderService: @unchecked Sendable {
    private let config: AgentConfig
    private let dbPath: String
    private let lock = NSLock()
    /// 串行化底层 SQLite / ONNX 的阻塞操作（单连接非线程安全）。
    private let opLock = NSRecursiveLock()
    private var engine: ColBERTReaderEngine?
    private var loadTask: Task<ColBERTReaderEngine, Error>?
    /// 加载代数：cancelLoading 会递增，旧 task 完成/失败时凭代数判断是否仍属自己，
    /// 避免无条件清空 loadTask 把 cancel 后新起的 task「遗忘」造成并发下载。
    private var loadGeneration: UInt64 = 0
    /// 当前下载器（供 cancelLoading 中断进行中的大文件下载，.part 断点保留）。
    private var activeDownloader: HFModelDownloader?
    /// 模型下载状态上抛（供 UI 显示进度提示；节流后的关键节点 + 阶段变化）。
    /// 读写都经 lock：写端在 RAGEngine（持 colbertLock）经 colbertService() 幂等同步，
    /// 读端在下载 Task 内，裸属性会构成跨对象数据竞争。
    private var downloadStatusHandlerStorage: (@Sendable (String) -> Void)?

    public var downloadStatusHandler: (@Sendable (String) -> Void)? {
        get {
            lock.lock()
            defer { lock.unlock() }
            return downloadStatusHandlerStorage
        }
        set {
            lock.lock()
            defer { lock.unlock() }
            downloadStatusHandlerStorage = newValue
        }
    }

    public init(config: AgentConfig, dbPath: String) {
        self.config = config
        self.dbPath = dbPath
    }

    private var model: RequiredModel {
        RequiredModel(
            kind: "检索阅读",
            repo: config.colbertModel,
            directory: RequiredModelCatalog.colbertModelDirectory(for: config),
            allowedWeightNames: ["model_int8.onnx"],
            requiresTokenizer: true)
    }

    private var options: ColBERTReaderOptions {
        ColBERTReaderOptions(
            quantBits: config.colbertQuantBits,
            coarseCandidates: config.colbertCoarseCandidates,
            topN: config.colbertTopN,
            useANN: config.colbertUseANN,
            lengthNorm: config.colbertLengthNorm,
            queryPrune: config.colbertQueryPrune,
            rrfVectorWeight: config.ragRRFVectorWeight)
    }

    /// 由来源路径生成稳定的 doc_id 前缀（16 位十六进制，避免特殊字符干扰 LIKE）。
    private func sourceKey(_ source: String) -> String {
        let digest = SHA256.hash(data: Data(source.utf8))
        return digest.prefix(8).map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - 模型下载与加载（单飞 single-flight）

    private func ensureEngine() async throws -> ColBERTReaderEngine {
        lock.lock()
        // 快检必须在锁内：engine 写入都在 lock 下，锁外读构成数据竞争。
        if let engine, engine.isModelLoaded {
            lock.unlock()
            return engine
        }
        if let loadTask {
            lock.unlock()
            return try await loadTask.value
        }
        let generation = loadGeneration
        let task = Task<ColBERTReaderEngine, Error> { [config, dbPath, weak self] in
            guard let self else { throw AgentError.cancelled }
            let model = RequiredModel(
                kind: "检索阅读",
                repo: config.colbertModel,
                directory: RequiredModelCatalog.colbertModelDirectory(for: config),
                allowedWeightNames: ["model_int8.onnx"],
                requiresTokenizer: true)
            if !RequiredModelCatalog.isComplete(model) {
                try await self.download(model)
            }
            let engine = try ColBERTReaderEngine(dbPath: dbPath, modelDir: model.directory.path)
            try await engine.loadModel()
            return engine
        }
        loadTask = task
        lock.unlock()

        do {
            let engine = try await task.value
            lock.lock()
            // 仅当代数未变（未被 cancelLoading 打断并另起 task）才采纳结果、
            // 清空 loadTask 与下载器引用：否则 cancel 后新起的 T2 会被误清。
            if generation == loadGeneration {
                self.engine = engine
                loadTask = nil
                activeDownloader = nil
            }
            lock.unlock()
            return engine
        } catch {
            lock.lock()
            if generation == loadGeneration {
                loadTask = nil
                activeDownloader = nil
            }
            lock.unlock()
            throw error
        }
    }

    /// 打开 schema（不加载模型），用于删除/计数等纯 db 操作。
    private func ensureEngineInstance() throws -> ColBERTReaderEngine {
        lock.lock()
        defer { lock.unlock() }
        if let engine { return engine }
        let e = try ColBERTReaderEngine(dbPath: dbPath, modelDir: model.directory.path)
        engine = e
        return e
    }

    /// 下载缺失模型；状态经 `downloadStatusHandler` 上抛（UI 可显示进度）。
    private func download(_ model: RequiredModel) async throws {
        let handler = downloadStatusHandler
        let report: @Sendable (String) -> Void = { handler?($0) }
        report("正在下载检索阅读模型（\(model.repo)）…")
        let downloader = HFModelDownloader()
        lock.lock()
        activeDownloader = downloader
        lock.unlock()
        defer {
            lock.lock()
            if activeDownloader === downloader { activeDownloader = nil }
            lock.unlock()
        }
        let files = try await downloader.listNeededFiles(repo: model.repo)
        let scoped = RequiredModelCatalog.scopedFiles(for: model, in: files)
        let totalBytes = scoped.reduce(Int64(0)) { $0 + $1.size }
        var doneBytes: Int64 = 0
        for file in scoped {
            // 停止同步等场景可经 cancelLoading() 取消下载：URLSession data(for:)
            // 响应 Task 取消抛 CancellationError，loadTask 失败后可重试。
            try Task.checkCancellation()
            let label = file.path
            let started = Date()
            var lastLogAt = Date.distantPast
            AgentLog.info("[colbert] 开始下载 \(file.path)（\(file.size / 1_048_576)MB）")
            report("下载 \(label)（\(file.size / 1_048_576)MB，共 "
                + "\(totalBytes / 1_048_576)MB）")
            let baseBytes = doneBytes
            try await downloader.download(repo: model.repo, file: file,
                                          into: model.directory) { received, total in
                // 大文件下载可达数十分钟，每 10% 或 30 秒输出一次，避免用户以为同步卡死。
                let now = Date()
                let ratio = total > 0 ? Double(received) / Double(total) : 0
                let enoughTime = now.timeIntervalSince(lastLogAt) >= 30
                if enoughTime || (total > 0 && received >= total) {
                    lastLogAt = now
                    let percent = Int(ratio * 100)
                    let overall = baseBytes + received
                    AgentLog.info("[colbert] 下载 \(label) \(received / 1_048_576)MB"
                        + "/\(total / 1_048_576)MB（\(percent)%）")
                    report("下载 \(label) \(percent)%（累计 \(overall / 1_048_576)"
                        + "/\(totalBytes / 1_048_576)MB）")
                }
            }
            doneBytes += file.size
            AgentLog.info("[colbert] 下载完成 \(file.path)，耗时 "
                + String(format: "%.1f", Date().timeIntervalSince(started)) + "s")
        }
        report("模型下载完成，正在加载…")
        guard RequiredModelCatalog.isComplete(model) else {
            throw HFModelDownloader.HFError.badResponse("检索阅读模型下载后仍不完整")
        }
    }

    // MARK: - 入库 / 检索 / 索引

    /// 入库一段 markdown 文本，返回写入的切片数。
    /// `shouldCancel` 在每片编码前检查；取消时抛 `CancellationError`，整个文件回滚。
    @discardableResult
    public func ingest(markdown: String, source: String, title: String? = nil,
                       shouldCancel: (() -> Bool)? = nil) async throws -> Int {
        let engine = try await ensureEngine()
        opLock.lock()
        defer { opLock.unlock() }
        // 长文档切片编码可能持续数十秒：按 30 秒节流输出进度，避免看起来「卡住」。
        let encoded = markdown.count
        let started = Date()
        var lastLogAt = Date.distantPast
        // 语义切片：与普通 RAG 共用 potion-multilingual-128M（int4 ONNX）静态词向量，
        // 在相邻句相似度低谷处断开；模型不可用时切片器自动回退词法相似度。
        let similarity = PotionSentenceSimilarity().adjacentCosines
        return try engine.ingest(
            markdown: markdown, source: source, title: title,
            sourceKey: sourceKey(source), options: options,
            similarity: similarity,
            shouldCancel: shouldCancel,
            progress: { done, total in
                let now = Date()
                guard now.timeIntervalSince(lastLogAt) >= 30 || done == total else { return }
                lastLogAt = now
                AgentLog.info("[colbert] 编码 \(done)/\(total) 分片（\(encoded) 字符，已用 "
                    + String(format: "%.0f", now.timeIntervalSince(started)) + "s）")
            })
    }

    /// 删除指定来源的全部切片（不加载模型）。
    @discardableResult
    /// 删除指定来源的全部切片。
    /// - Parameter source: 来源原始路径；doc_id 前缀删不到时按 document.source 列
    ///   兜底（兼容 doc_search 独立工具建的旧库，其 doc_id 前缀非 SHA256 hex）。
    public func delete(source: String) throws -> Int {
        let engine = try ensureEngineInstance()
        opLock.lock()
        defer { opLock.unlock() }
        return try engine.delete(sourceKey: sourceKey(source), fallbackSource: source)
    }

    /// 指定来源的切片数（不加载模型）。
    /// - Throws: colbert.db 打不开/查询失败时上抛，调用方不得把错误当成 0 条。
    public func chunkCount(source: String) throws -> Int {
        let engine = try ensureEngineInstance()
        opLock.lock()
        defer { opLock.unlock() }
        return try engine.chunkCount(sourceKey: sourceKey(source), fallbackSource: source)
    }

    /// 全库切片总数（不加载模型）。
    /// - Throws: colbert.db 打不开/查询失败时上抛，调用方不得把错误当成 0 条。
    public func totalChunkCount() throws -> Int {
        let engine = try ensureEngineInstance()
        opLock.lock()
        defer { opLock.unlock() }
        return try engine.totalChunkCount()
    }

    /// 全库切片正文（供知识图谱抽取，不加载模型）。
    public func listAllChunkTexts() throws -> [(id: String, text: String, source: String, heading: String)] {
        let engine = try ensureEngineInstance()
        opLock.lock()
        defer { opLock.unlock() }
        return try engine.listAllChunkTexts()
    }

    /// 预加载模型并常驻，让首次检索不再现加载。
    public func preload() async throws {
        let engine = try await ensureEngine()
        // 只加载 ONNX session 不够：ORT 的算子 kernel 选择 / 内存 arena 分配 /
        // 线程绑定都惰性发生在首次 session.run()。若不在预加载阶段触发，
        // 首次检索的 query_center 就会承担这笔开销——真实 App 里主模型已占
        // 数 GB 内存，首次 arena 分配被放大到秒级（实测日志 1538ms）。
        // 用固定短句跑一次完整 query 编码，把冷启动成本挪到后台预加载。
        // 持 opLock 与 ingest/ANN 串行，避免并发触发多次「首次推理」互相争抢。
        opLock.lock()
        defer { opLock.unlock() }
        engine.warmupQueryEncoding()
    }

    /// 只写正文 + FTS，编码转后台。供同步时「文本入库即完成」。
    /// 不触发模型下载/加载：ensureEngineInstance 只打开 schema，文本入库纯 DB 操作。
    public func ingestTextsOnly(markdown: String, source: String, title: String? = nil,
                                sourceKey sourceKeyOverride: String? = nil,
                                shouldCancel: (() -> Bool)? = nil) throws -> Int {
        let engine = try ensureEngineInstance()
        opLock.lock()
        defer { opLock.unlock() }
        let similarity = PotionSentenceSimilarity().adjacentCosines
        // doc_id 前缀必须与 delete/chunkCount 的 sourceKey(source) 一致（SHA256 前 8 字节），
        // 否则按来源删除/计数永远匹配不到，孤儿残留 + 未变化文件反复重做。
        let key = sourceKey(source)
        return try engine.ingestTextsOnly(markdown: markdown, source: source,
                                          title: title, sourceKey: key,
                                          bits: config.colbertQuantBits,
                                          similarity: similarity,
                                          shouldCancel: shouldCancel)
    }

    /// 待回填编码的切片数（不触发模型加载/下载，供状态栏与后台循环查询）。
    public func pendingEncodeCount() -> Int {
        (try? ensureEngineInstance().pendingEncodeCount()) ?? 0
    }

    /// 轻量正文命中探查（不加载模型/下载），供多库联合检索前选库。
    public func ftsHitCount(query: String, cap: Int = 8) -> Int {
        (try? ensureEngineInstance().ftsHitCount(query: query, cap: cap)) ?? 0
    }

    /// 多库联合检索-极速模式粗筛。
    public func coarseCandidates(query: String, coarse: Int = 20,
                                 rrfVectorWeight: Double = 0.7, source: String? = nil)
        throws -> [(docID: String, rrf: Double, content: String,
                    source: String, title: String)] {
        try ensureEngineInstance().coarseCandidates(query: query, coarse: coarse,
                                                    rrfVectorWeight: rrfVectorWeight,
                                                    source: source)
    }

    /// 多库联合检索-极速模式按 doc_id 精排（MaxSim）。
    public func rerankByDocIDs(query: String, docIDs: [String],
                               lengthNorm: Bool = false) throws -> [ColBERTReaderHit] {
        try ensureEngineInstance().rerankByDocIDs(query: query, docIDs: docIDs,
                                                   lengthNorm: lengthNorm)
    }

    /// ColBERT 编码完备状态（不触发模型加载/下载）。
    public func encodingCompleteness() -> (total: Int, filled: Int, pending: Int) {
        (try? ensureEngineInstance().encodingCompleteness()) ?? (-1, 0, 0)
    }

    /// 后台回填一批待编码切片。确保模型已加载后委托 engine。
    /// 持 opLock 与 ingest/delete/shutdown 串行（texts-only 与回填都写 colbert.db）。
    public func backfillPending(limit: Int,
                                shouldCancel: (() -> Bool)? = nil) async throws -> Int {
        let engine = try await ensureEngine()
        opLock.lock()
        defer { opLock.unlock() }
        return try engine.backfillPending(limit: limit, shouldCancel: shouldCancel,
                                          onError: { message in
            AgentLog.warning("[colbert] 回填失败（保留待重试）：\(message)")
        })
    }

    /// 把 top-1 分片原文格式化为供 LLM 总结的上下文。
    /// 分片按 7500~8000 字符的大分片入库，单条体量在主模型 30k 预算内，完整给出。
    static func primaryContext(citations: [RAGCitation]) -> String {
        guard let first = citations.first else {
            return "（知识库中未找到相关内容）"
        }
        let heading = first.heading.isEmpty ? "" : "\n章节: \(first.heading)"
        let header = citations.count == 1
            ? "本次检索召回 1 条相关切片，已全部进入上下文："
            : "本次共召回 \(citations.count) 条相关分片。得分最高的 1 条已进入上下文供你总结，"
                + "其余 \(citations.count - 1) 条已作为引用气泡展示给用户"
                + "（用户可自行点开查看）。请基于下面的原文作答，不要臆造原文没有的内容："
        return header + "\n\n### 来源: \(first.source) (相关度: "
            + String(format: "%.4f", first.score) + ")\(heading)\n\(first.text)"
    }

    /// 检索并返回召回切片：得分最高的第 1 条进入模型上下文供 LLM 总结，
    /// 全部召回切片（含 top-1）作为引用气泡展示。
    /// `source` 非空时只返回该来源文件/标题内的切片。
    ///
    /// 检索路径只读（SQLite WAL 快照隔离 + FULLMUTEX 句柄串行），不持 opLock：
    /// ANN 构建/同步入库期间发起的对话检索可立即执行（读到构建前的索引快照），
    /// 不会在 opLock 上排队等待长任务结束。
    public func search(query: String, source: String? = nil,
                       topK: Int? = nil) async throws -> RAGRetrievalOutcome {
        let engine = try await ensureEngine()
        // 显式 topK（来自 rag_search 的 top_k / 公开 API）覆盖面板 colbertTopN：
        // 工具契约是「top_k 决定气泡数量」，不能静默失效。clamp 到工具声明的 1...10，
        // 未传时沿用配置面板的 colbertTopN。
        var effectiveOptions = options
        if let topK { effectiveOptions.topN = min(max(topK, 1), 10) }
        // 分阶段耗时接线：每次检索输出一行汇总，定位慢查询（编码/粗筛/精排）。
        let timingStarted = Date()
        var stageTimings: [(String, Double)] = []
        let hits = try engine.search(query: query, options: effectiveOptions, source: source) {
            key, seconds in
            stageTimings.append((key, seconds))
            // 写入统一收集器，入口「检索汇总」行也能看到 ColBERT 各阶段。
            RAGSearchTiming.current?.set("cb_\(key)", seconds: seconds)
        }

        var citations: [RAGCitation] = []
        var rank = 1
        for hit in hits {
            // source 用入库时的文件相对路径（气泡上的「来源」）；
            // title 是切片所在章节标题，作 heading 展示。旧库没有 source
            // 列时回退到 title，保持原有展示。
            let source = hit.source.isEmpty ? hit.title : hit.source
            citations.append(RAGCitation(
                id: hit.docID,
                rank: rank,
                source: source,
                heading: hit.title == source ? "" : hit.title,
                text: hit.content,
                score: hit.score,
                isPrimary: rank == 1))
            rank += 1
        }
        // 得分最高的 1 条进入模型上下文供 LLM 总结；其余切片仍以引用气泡
        // 展示。这里给出上下文文本：直接用 top-1 原文，引导模型据此作答、
        // 不臆造内容；其余切片提示用户点开气泡查看。
        let guide = Self.primaryContext(citations: citations)
        // 分阶段耗时日志：编码/粗筛/融合/精排各阶段毫秒 + 候选数。
        let totalMs = Date().timeIntervalSince(timingStarted) * 1000
        let stages = stageTimings
            .filter { !["shortlist", "search_total"].contains($0.0) }
            .map { "\($0.0)=\(Int($0.1 * 1000))ms" }
            .joined(separator: " ")
        AgentLog.info("[colbert] 检索 \(Int(totalMs))ms 候选=\(hits.count) \(stages)")
        return RAGRetrievalOutcome(context: guide, citations: citations)
    }

    /// 显式重建 ColBERT 库（清全部切片/向量/ANN），返回清掉的切片数。
    @discardableResult
    public func rebuildDatabase() throws -> Int {
        let engine = try ensureEngineInstance()
        opLock.lock()
        defer { opLock.unlock() }
        return try engine.rebuildDatabase()
    }

    /// 重置 ANN 索引（force 全量同步前调用；不依赖模型加载）。
    public func resetANN() throws {
        let engine = try ensureEngineInstance()
        opLock.lock()
        defer { opLock.unlock() }
        try engine.resetANN()
    }

    /// 构建 / 续建向量索引，返回摘要。
    /// - Parameter shouldCancel: 块间取消检查；中断后已写块保留，续建从水位继续。
    @discardableResult
    public func buildANN(shouldCancel: (() -> Bool)? = nil) async throws -> String {
        let engine = try await ensureEngine()
        opLock.lock()
        defer { opLock.unlock() }
        return try engine.buildANN(shouldCancel: shouldCancel)
    }

    /// ANN 索引状态摘要（读路径；opLock 与长任务互斥，避免读到中间状态）。
    /// 忙碌（ingest/构建持锁）时返回 nil：状态栏刷新不应在 opLock 上排队停摆。
    public func annStatus() -> String? {
        // engine 写入在 lock 下（ensureEngine 84/ensureEngineInstance 102），
        // 锁外读构成数据竞争；先取 lock 快照，再判 opLock 是否空闲。
        lock.lock()
        let current = engine
        lock.unlock()
        guard opLock.try() else { return nil }
        defer { opLock.unlock() }
        return current?.annStatus() ?? "未加载"
    }

    /// 结构化 ANN 状态，语义与 annStatus() 一致（非阻塞，忙碌返回 nil）。
    public func annStatusDetail() -> ColBERTReaderEngine.ANNStatus? {
        lock.lock()
        let current = engine
        lock.unlock()
        guard opLock.try() else { return nil }
        defer { opLock.unlock() }
        return current?.annStatusDetail()
    }

    /// 取消进行中的模型加载/下载（如停止同步时）；任务失败后下次调用重新加载。
    public func cancelLoading() {
        lock.lock()
        let task = loadTask
        let downloader = activeDownloader
        loadTask = nil
        activeDownloader = nil
        loadGeneration &+= 1
        lock.unlock()
        // 先取消 URLSession 下载：否则 await loadTask.value 会挂到网络自然结束
        // （Task.cancel 只被 Task.checkCancellation 在文件边界检查，传输中不生效）。
        downloader?.cancelCurrent()
        task?.cancel()
    }

    /// 释放模型与数据库句柄：供 RAGEngine close/reload/deinit 显式调用，
    /// 避免切换配置后旧模型内存（几百 MB）靠 ARC 延迟回收、与新模型双份驻留。
    /// 释放后再次使用会经 ensureEngine 重新下载/加载（若模型文件完整则直接从磁盘加载）。
    public func shutdown() {
        cancelLoading()
        opLock.lock()
        lock.lock()
        let current = engine
        engine = nil
        loadTask = nil
        activeDownloader = nil
        loadGeneration &+= 1
        lock.unlock()
        // db.close 可能阻塞在 SQLite 锁上，放在 opLock 内但 lock 外执行。
        current?.shutdown()
        opLock.unlock()
    }
}
