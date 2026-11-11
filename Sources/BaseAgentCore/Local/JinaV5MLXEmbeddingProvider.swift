import Foundation

/// jina-embeddings-v5-text-nano 的本地 MLX Embedding provider。
///
/// V5 的任务类型通过输入前缀表达（`Query: ` / `Document: `），provider 按
/// Matryoshka 配置输出对应维度的单位向量：内部已完成截断与重新 L2 归一化，
/// 外层 `MatryoshkaEmbedding` 不会再包装它。落盘元素类型由用户配置决定。
final class JinaV5MLXEmbeddingProvider: EmbeddingProvider,
                                       EmbeddingQueryBatching,
                                       EmbeddingTokenCounting,
                                       EmbeddingBinarySigning,
                                       @unchecked Sendable {
    let identifier: String
    let modelName: String
    let directory: URL
    let prefixStrategy: EmbeddingPrefixStrategy

    private let maxLength: Int
    private let state = JinaV5MLXLoadState()

    /// Matryoshka 输出维度：模型声明的六档之一，默认 128。
    let dimension: Int

    /// MLX 受全局 Metal 队列限制，固定单实例串行。
    var maxConcurrentRuns: Int { 1 }

    init(directory: URL, modelName: String,
         dimension: Int = JinaV5Profile.defaultRetrievalDimension) {
        self.directory = directory
        self.modelName = modelName
        self.identifier = "local:mlx-embedding:\(modelName)"
        // 只服务 jina-v5：即使目录被改名，也必须保留 V5 的大写前缀。
        self.prefixStrategy = EmbeddingPrefixStrategy(family: .jinaV5)
        self.maxLength = Self.window(directory: directory)
        // 防御非法值：只接受模型训练过的维度，其余回落到默认 128。
        self.dimension = JinaV5Profile.validDimension(dimension)
    }

    var binaryDimension: Int { JinaV5Profile.nativeDimension }
    var binarySourceIdentifier: String { identifier }

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        try await embedDocumentsWithSignatures(texts).vectors
    }

    func embedDocumentsWithSignatures(_ texts: [String]) async throws
        -> BinaryEmbeddingOutput {
        guard !texts.isEmpty else {
            return BinaryEmbeddingOutput(vectors: [], signatures: [])
        }
        let prefixed = texts.map {
            prefixStrategy.apply($0, task: .retrievalDocument)
        }
        return try await embedWithSignatures(prefixed)
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        try await embedQueryWithSignature(text).vector
    }

    /// 批量 query 嵌入：多条检索文本共用一次 encode，前缀与单条 `embedQuery` 一致。
    func embedQueries(_ texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        let prefixed = texts.map {
            prefixStrategy.apply($0, task: .retrievalQuery)
        }
        return try await embedWithSignatures(prefixed).vectors
    }

    func embedQueryWithSignature(_ text: String) async throws
        -> (vector: [Float], signature: Data) {
        let output = try await embedWithSignatures([
            prefixStrategy.apply(text, task: .retrievalQuery),
        ])
        return (output.vectors.first ?? [], output.signatures.first ?? Data())
    }

    private func embedWithSignatures(_ texts: [String]) async throws
        -> BinaryEmbeddingOutput {
        let (model, tokenizer) = try await state.handles(directory: directory)
        let batch = tokenizer.encodeBatch(texts, maxLength: maxLength)
        let result = model.encode(
            inputIDs: batch.inputIDs, attentionMask: batch.attentionMask,
            dimension: dimension)
        return BinaryEmbeddingOutput(
            vectors: result.vectors,
            signatures: result.nativeVectors.map {
                VectorEncoder.encodeSignature($0, bitDimension: binaryDimension)
            })
    }

    /// 释放权重；provider 保留，下次 embed 时按需重载。
    func unload() async {
        await state.unload()
    }

    /// 单条输入上限：窗口减去 [CLS]/[SEP] 与 V5 document 前缀预留。
    var maxContentTokens: Int {
        Self.contentTokenLimit(directory: directory, modelName: modelName)
    }

    static func contentTokenLimit(directory: URL, modelName: String) -> Int {
        let reserve = EmbeddingPrefixStrategy(
            modelName: modelName,
            modelPath: directory.lastPathComponent).documentPrefixTokenReserve
        return max(16, window(directory: directory) - 2 - reserve)
    }

    /// 模型声明窗口 8192，但推理前统一压到 1024 token（默认切片仅 800）。
    private static func window(directory: URL) -> Int {
        EmbeddingInputLimits.cappedWindow(max(LocalModelRegistry.resolveMaxSequenceLength(
            directory: directory, defaultValue: 8_192), 128))
    }

    func makeTokenCounter() async -> (@Sendable (String) -> Int)? {
        await TokenizerCounterCache.shared.counter(for: directory)
    }
}

private actor JinaV5MLXLoadState {
    private var model: JinaV5MLXModel?
    private var tokenizer: LocalTokenizer?
    private var failure: Error?

    func handles(directory: URL) async throws -> (JinaV5MLXModel, LocalTokenizer) {
        if let model, let tokenizer { return (model, tokenizer) }
        if let failure { throw failure }

        do {
            let model = try JinaV5MLXModel(directory: directory)
            let tokenizer = try await LocalTokenizer.load(directory: directory)
            self.model = model
            self.tokenizer = tokenizer
            return (model, tokenizer)
        } catch {
            let wrapped = error as? LocalModelError ?? LocalModelError.runtime(
                "Jina v5 MLX 加载失败（\(directory.path)）：\(error.localizedDescription)")
            failure = wrapped
            throw wrapped
        }
    }

    func unload() {
        model = nil
        tokenizer = nil
        failure = nil
    }
}
