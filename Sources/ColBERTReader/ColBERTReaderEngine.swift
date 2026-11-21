import Foundation

/// 对外暴露的单条检索结果。
public struct ColBERTReaderHit: Sendable {
    public let docID: String
    public let title: String
    /// 文档来源（入库时的文件相对路径/文件名）；旧库可能为空。
    public let source: String
    public let content: String
    public let score: Double
    public let rank: Int

    public init(docID: String, title: String, source: String = "",
                content: String, score: Double, rank: Int) {
        self.docID = docID
        self.title = title
        self.source = source
        self.content = content
        self.score = score
        self.rank = rank
    }
}

/// 检索阅读模式的运行参数。
public struct ColBERTReaderOptions: Sendable {
    public var quantBits: Int
    public var coarseCandidates: Int
    public var topN: Int
    public var useANN: Bool
    /// MaxSim 精排是否按分片 token 数做长度归一化（A/B 对比用）。
    public var lengthNorm: Bool
    /// 查询侧 token 轻量剪枝（默认开：丢标点/空白/核心功能词，保留疑问词与形容词）。
    /// 关闭则查询侧完全不剪枝。检索期参数，不进 ingest 签名。
    public var queryPrune: Bool
    /// Weighted RRF 向量路权重（0...1，默认 0.7）；BM25 权重 = 1 - 该值。
    public var rrfVectorWeight: Double

    public init(quantBits: Int = 4, coarseCandidates: Int = 50,
                topN: Int = 3, useANN: Bool = true, lengthNorm: Bool = false,
                queryPrune: Bool = true, rrfVectorWeight: Double = 0.7) {
        self.quantBits = quantBits
        self.coarseCandidates = coarseCandidates
        self.topN = topN
        self.useANN = useANN
        self.lengthNorm = lengthNorm
        self.queryPrune = queryPrune
        self.rrfVectorWeight = rrfVectorWeight
    }
}

/// 检索阅读模式（ColBERT 多向量大分片检索）引擎。
///
/// 自包含、相对独立：接管文档解析生成 markdown 之后的切片、词向量计算、
/// 残差量化入库、检索召回与 MaxSim 精排全流程。模型下载由上层负责，
/// 本引擎只从给定目录加载模型。
public final class ColBERTReaderEngine {
    /// 跨所有 engine 共享 ColBERT 模型：多知识分区各有独立 db，但加载同一份
    /// model_int8.onnx 时只保留一份 ORTSession/arena，避免 N 库 N 份模型 OOM。
    private static let sharedLock = NSLock()
    private static var sharedModels: [String: ColBERT] = [:]
    private static var sharedLoadingTasks: [String: Task<ColBERT, Error>] = [:]

    private let db: SQLiteDB
    private let modelDir: String
    private var colbert: ColBERT?
    private var searchEngine: SearchEngine?
    private var didWarmup = false

    public init(dbPath: String, modelDir: String) throws {
        self.modelDir = modelDir
        self.db = try SQLiteDB(path: dbPath)
        try SearchEngine.createSchema(db)
    }

    /// 从 `modelDir` 加载模型（幂等；同 modelDir 跨 engine 共享同一实例）。
    public func loadModel(threads: Int = 4) async throws {
        if colbert != nil { return }
        let model = try await Self.sharedColbert(modelDir: modelDir, threads: threads)
        colbert = model
        searchEngine = SearchEngine(db: db, colbert: model)
    }

    private static func sharedColbert(modelDir: String, threads: Int) async throws -> ColBERT {
        sharedLock.lock()
        if let cached = sharedModels[modelDir] { sharedLock.unlock(); return cached }
        if let task = sharedLoadingTasks[modelDir] {
            sharedLock.unlock()
            return try await task.value
        }
        let task = Task<ColBERT, Error> {
            try await ColBERT.load(modelDir: modelDir, threads: threads)
        }
        sharedLoadingTasks[modelDir] = task
        sharedLock.unlock()
        do {
            let model = try await task.value
            sharedLock.lock()
            sharedModels[modelDir] = model
            sharedLoadingTasks[modelDir] = nil
            sharedLock.unlock()
            return model
        } catch {
            sharedLock.lock()
            sharedLoadingTasks[modelDir] = nil
            sharedLock.unlock()
            throw error
        }
    }

    public var isModelLoaded: Bool { colbert != nil }

    /// 预热一次查询编码前向：把 ORT 首次 session.run 的惰性初始化成本
    /// （kernel 选择 / arena 分配 / 线程绑定）从首次检索挪到后台预加载阶段。
    /// 模型未加载时为空操作；推理失败不抛出，交由首次检索按需重试。
    public func warmupQueryEncoding() {
        guard let colbert, !didWarmup else { return }
        _ = try? colbert.queryVectors("检索预热 query warmup", prune: true)
        didWarmup = true
    }

    /// 释放模型与数据库句柄：置 nil 让 ORTSession / tokenizer / SQLiteDB 经 ARC 立即回收，
    /// 而不是等整个引擎对象 deinit。供上层 close/reload 显式调用，避免切换配置后
    /// 旧模型内存（几百 MB）长时间与新模型双份驻留。释放后可再次 loadModel 重建。
    public func shutdown() {
        searchEngine = nil
        colbert = nil
        db.close()
    }

    private func engine() throws -> SearchEngine {
        guard let engine = searchEngine else {
            throw DocSearchError.model("模型尚未加载")
        }
        return engine
    }

    /// 入库一段 markdown 文本。返回写入的切片数。
    /// `similarity` 用于语义切片（相邻文本块余弦），由上层注入；为空时回退词法相似度。
    /// `shouldCancel` 在每片编码前检查，取消时整文件事务回滚。
    @discardableResult
    public func ingest(markdown: String, source: String, title: String? = nil,
                       sourceKey: String? = nil,
                       options: ColBERTReaderOptions,
                       similarity: (([String]) -> [Double]?)? = nil,
                       shouldCancel: (() -> Bool)? = nil,
                       progress: ((Int, Int) -> Void)? = nil) throws -> Int {
        try engine().addMarkdown(markdown, source: source, title: title,
                                 bits: options.quantBits, sourceKey: sourceKey,
                                 similarity: similarity, shouldCancel: shouldCancel,
                                 progress: progress)
    }

    /// 只写正文 + FTS + pending，不编码、不依赖模型。供同步时「文本入库即完成」。
    /// 直接走静态 DB 路径，模型未下载/未加载也能入库。
    @discardableResult
    public func ingestTextsOnly(markdown: String, source: String, title: String? = nil,
                                sourceKey: String? = nil, bits: Int = 4,
                                similarity: (([String]) -> [Double]?)? = nil,
                                shouldCancel: (() -> Bool)? = nil) throws -> Int {
        try SearchEngine.addMarkdownTextsOnly(
            db, text: markdown, source: source, title: title,
            bits: bits, sourceKey: sourceKey, similarity: similarity,
            shouldCancel: shouldCancel)
    }

    /// 后台回填一批待编码切片：取 pending → 编码 → 量化 → 写向量表并出队。
    /// 需要模型已加载；每片独立事务，单片失败不影响其它片（留 pending 下次重试）。
    /// 返回本批实际回填条数。
    @discardableResult
    public func backfillPending(limit: Int,
                                shouldCancel: (() -> Bool)? = nil,
                                onError: ((String) -> Void)? = nil) throws -> Int {
        let se = try engine()
        guard let model = colbert else { throw DocSearchError.model("模型尚未加载") }
        let batch = se.nextPendingEncode(limit: limit)
        guard !batch.isEmpty else { return 0 }
        var done = 0
        var failures = 0
        for item in batch {
            if shouldCancel?() == true { break }
            do {
                let vectors = try model.encodeDocument(item.content)
                guard !vectors.isEmpty else {
                    // 整段被剪枝为空：移除幽灵切片（正文/FTS/count/pending 一起清）。
                    try se.dropPendingGhost(docID: item.docID, rowid: item.rowid)
                    done += 1
                    continue
                }
                let record = try ResidualCodec.quantize(
                    docID: item.docID, vectors: vectors,
                    dim: ColBERT.dimension, bits: item.bits)
                try se.fillPendingEncode(docID: item.docID, rowid: item.rowid,
                                         record: record)
                done += 1
            } catch {
                // 单条失败退避重试：排到队尾，既不永久丢弃也不饿死后续健康切片。
                failures += 1
                try? se.retryPending(docID: item.docID)
                onError?("\(item.docID): \(error.localizedDescription)")
            }
        }
        return done
    }

    /// 显式重建 ColBERT 库（清全部切片/向量/ANN），返回清掉的切片数。
    @discardableResult
    public func rebuildDatabase() throws -> Int {
        try SearchEngine.rebuildDatabase(db)
    }

    /// 待回填编码的切片数（不依赖模型加载）。
    public func pendingEncodeCount() -> Int {
        SearchEngine.pendingEncodeCount(db)
    }

    /// ColBERT 编码完备状态：(切片总数, 已编码数, 待编码数)（不依赖模型加载）。
    /// total = -1 表示库打不开/查询失败——调用方不得据此判定「完备」。
    public func encodingCompleteness() -> (total: Int, filled: Int, pending: Int) {
        guard let total = try? SearchEngine.totalChunkCount(db),
              let filled = try? SearchEngine.encodedChunkCount(db) else {
            return (-1, 0, 0)
        }
        let pending = SearchEngine.pendingEncodeCount(db)
        return (total, filled, pending)
    }

    /// 删除指定来源的全部切片（不依赖模型加载）。
    @discardableResult
    public func delete(sourceKey: String, fallbackSource: String? = nil) throws -> Int {
        try SearchEngine.deleteSource(db, sourceKey, fallbackSource: fallbackSource)
    }

    /// 指定来源的切片数（不依赖模型加载）。
    public func chunkCount(sourceKey: String, fallbackSource: String? = nil) throws -> Int {
        try SearchEngine.chunkCount(db, sourceKey, fallbackSource: fallbackSource)
    }

    /// 全库切片总数（不依赖模型加载）。
    public func totalChunkCount() throws -> Int {
        try SearchEngine.totalChunkCount(db)
    }

    /// 全库切片正文（供知识图谱抽取，不加载模型）。
    public func listAllChunkTexts() throws -> [(id: String, text: String, source: String, heading: String)] {
        try SearchEngine.listAllChunkTexts(db)
    }

    /// 轻量正文命中探查（不加载模型），供多库联合检索前判断是否相关。
    public func ftsHitCount(query: String, cap: Int = 8) -> Int {
        SearchEngine.ftsHitCount(db, query: query, cap: cap)
    }

    /// 多库联合检索-极速模式粗筛（粗筛模型走 queryVectors，无需完整检索）。
    public func coarseCandidates(query: String, coarse: Int = 20,
                                 rrfVectorWeight: Double = 0.7, source: String? = nil)
        throws -> [(docID: String, rrf: Double, content: String,
                    source: String, title: String)] {
        guard let searchEngine else { return [] }
        return try searchEngine.coarseCandidates(query: query, coarse: coarse,
                                                 rrfVectorWeight: rrfVectorWeight,
                                                 source: source)
    }

    /// 多库联合检索-极速模式按 doc_id 精排（MaxSim）。
    public func rerankByDocIDs(query: String, docIDs: [String],
                               lengthNorm: Bool = false) throws -> [ColBERTReaderHit] {
        guard let searchEngine else { return [] }
        let hits = try searchEngine.rerankByDocIDs(query: query, docIDs: docIDs,
                                                    lengthNorm: lengthNorm)
        return hits.enumerated().map { i, h in
            ColBERTReaderHit(docID: h.docID, title: h.title, source: h.source,
                             content: h.content, score: Double(h.score), rank: i + 1)
        }
    }

    /// 检索并返回命中切片（含原文，供 UI 气泡展示）。
    /// `source` 非空时只返回来源文件名/标题匹配的切片。
    public func search(query: String, options: ColBERTReaderOptions,
                       source: String? = nil,
                       timing: ((String, Double) -> Void)? = nil) throws -> [ColBERTReaderHit] {
        let hits = try engine().search(query: query,
                                       topN: options.topN,
                                       coarse: options.coarseCandidates,
                                       useANN: options.useANN,
                                       lengthNorm: options.lengthNorm,
                                       queryPrune: options.queryPrune,
                                       rrfVectorWeight: options.rrfVectorWeight,
                                       // 开关开启即视为允许绕过「小库 < 2 万行不建 ANN」
                                       // 的隐藏门槛：UI 的「使用 ANN」才真正生效。
                                       // 无可用索引时 candidates 返回 nil，自然回退精确，
                                       // 因此小库开启也不会因缺索引而检索失败。
                                       forceANN: options.useANN,
                                       source: source,
                                       timing: timing)
        return hits.enumerated().map { index, hit in
            ColBERTReaderHit(docID: hit.docID, title: hit.title,
                             source: hit.source, content: hit.content,
                             score: Double(hit.score), rank: index + 1)
        }
    }

    /// 与检索共享同一 ANN 实例，保证 build 后簇心缓存随之失效。
    private var sharedANN: ANNIndex { searchEngine?.ann ?? ANNIndex(db: db) }

    /// 重置 ANN 索引（force 全量同步前调用：旧 rowid 全部失效，续建无意义）。
    /// drop + 立即重建空表，避免 status/candidates 因缺表吞错而静默失效。
    public func resetANN() throws {
        let ann = sharedANN
        try ann.drop()
        try ann.createSchema()
    }

    /// 构建 / 续建向量索引。返回摘要信息。
    /// - Parameter shouldCancel: 块间取消检查；中断后已写块保留，续建从水位继续。
    @discardableResult
    public func buildANN(shouldCancel: (() -> Bool)? = nil) throws -> String {
        try sharedANN.build(isCancelled: shouldCancel).summary
    }

    public func annStatus() -> String {
        sharedANN.status().summary
    }

    /// 结构化 ANN 状态（供上层 status 快照填 ann_usable / ann_needs_rebuild）。
    public struct ANNStatus: Sendable {
        public var summary: String
        public var usable: Bool
        public var needsRebuild: Bool
        public var indexedRows: Int
        public var tailRows: Int
        public var totalRows: Int
    }

    public func annStatusDetail() -> ANNStatus {
        let s = sharedANN.status()
        return ANNStatus(summary: s.summary, usable: s.usable,
                         needsRebuild: s.needsRebuild, indexedRows: s.indexedRows,
                         tailRows: s.tailRows, totalRows: s.totalRows)
    }
}
