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

    public init(quantBits: Int = 4, coarseCandidates: Int = 20,
                topN: Int = 3, useANN: Bool = true) {
        self.quantBits = quantBits
        self.coarseCandidates = coarseCandidates
        self.topN = topN
        self.useANN = useANN
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
    @discardableResult
    public func ingest(markdown: String, source: String, title: String? = nil,
                       sourceKey: String? = nil,
                       options: ColBERTReaderOptions) throws -> Int {
        try engine().addMarkdown(markdown, source: source, title: title,
                                 bits: options.quantBits, sourceKey: sourceKey)
    }

    /// 删除指定来源的全部切片（不依赖模型加载）。
    @discardableResult
    public func delete(sourceKey: String) throws -> Int {
        try SearchEngine.deleteSource(db, sourceKey)
    }

    /// 指定来源的切片数（不依赖模型加载）。
    public func chunkCount(sourceKey: String) -> Int {
        SearchEngine.chunkCount(db, sourceKey)
    }

    /// 全库切片总数（不依赖模型加载）。
    public func totalChunkCount() -> Int {
        SearchEngine.totalChunkCount(db)
    }

    /// 检索并返回命中切片（含原文，供 UI 气泡展示）。
    public func search(query: String, options: ColBERTReaderOptions) throws -> [ColBERTReaderHit] {
        let hits = try engine().search(query: query,
                                       topN: options.topN,
                                       coarse: options.coarseCandidates,
                                       useANN: options.useANN)
        return hits.enumerated().map { index, hit in
            ColBERTReaderHit(docID: hit.docID, title: hit.title,
                             source: hit.source, content: hit.content,
                             score: Double(hit.score), rank: index + 1)
        }
    }

    /// 构建 / 续建向量索引。返回摘要信息。
    @discardableResult
    public func buildANN() throws -> String {
        try ANNIndex(db: db).build().summary
    }

    public func annStatus() -> String {
        ANNIndex(db: db).status().summary
    }
}
