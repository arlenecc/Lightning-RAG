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

    public init(quantBits: Int = 4, coarseCandidates: Int = 50,
                topN: Int = 3, useANN: Bool = true, lengthNorm: Bool = false,
                queryPrune: Bool = true) {
        self.quantBits = quantBits
        self.coarseCandidates = coarseCandidates
        self.topN = topN
        self.useANN = useANN
        self.lengthNorm = lengthNorm
        self.queryPrune = queryPrune
    }
}

/// 检索阅读模式（ColBERT 多向量大分片检索）引擎。
///
/// 自包含、相对独立：接管文档解析生成 markdown 之后的切片、词向量计算、
/// 残差量化入库、检索召回与 MaxSim 精排全流程。模型下载由上层负责，
/// 本引擎只从给定目录加载模型。
public final class ColBERTReaderEngine {
    private let db: SQLiteDB
    private let modelDir: String
    private var colbert: ColBERT?
    private var searchEngine: SearchEngine?

    public init(dbPath: String, modelDir: String) throws {
        self.modelDir = modelDir
        self.db = try SQLiteDB(path: dbPath)
        try SearchEngine.createSchema(db)
    }

    /// 从 `modelDir` 加载模型（幂等；已加载则直接返回）。
    /// 并发加载由上层单飞（single-flight）保证；这里只在加载完成后
    /// 以「先检查再赋值」的方式写入实例状态。
    public func loadModel(threads: Int = 4) async throws {
        if colbert != nil { return }
        let model = try await ColBERT.load(modelDir: modelDir, threads: threads)
        if colbert == nil {
            colbert = model
            searchEngine = SearchEngine(db: db, colbert: model)
        }
    }

    public var isModelLoaded: Bool { colbert != nil }

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
