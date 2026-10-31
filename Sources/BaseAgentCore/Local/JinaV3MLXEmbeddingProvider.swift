import Foundation

/// jina-embeddings-v3 的本地 MLX Embedding provider。
///
/// 与其它 provider 保持同样的接口：document/query 前缀、原生长度池化 + L2
/// 归一化；Matryoshka 截断与 int8 量化由外层统一处理。
final class JinaV3MLXEmbeddingProvider: EmbeddingProvider,
                                       EmbeddingTokenCounting,
                                       EmbeddingBinarySigning,
                                       @unchecked Sendable {
    let identifier: String
    let modelName: String
    let directory: URL

    private let maxLength: Int
    let prefixStrategy: EmbeddingPrefixStrategy
    private let state = JinaV3MLXLoadState()
    private let dimensionLock = NSLock()
    private var storedDimension = JinaV3Profile.nativeDimension

    var dimension: Int {
        dimensionLock.lock()
        defer { dimensionLock.unlock() }
        return storedDimension
    }

    /// MLX 受全局 Metal 队列限制，固定单实例串行。
    var maxConcurrentRuns: Int { 1 }

    init(directory: URL, modelName: String) {
        self.directory = directory
        self.modelName = modelName
        self.identifier = "local:mlx-embedding:\(modelName)"
        // 本 provider 只服务 jina-v3：即使权重目录被改名，也必须使用
        // passage: / query: 前缀，不能只靠目录名识别模型族。
        self.prefixStrategy = EmbeddingPrefixStrategy(family: .jina)
        self.maxLength = Self.window(directory: directory)

        if let config = JinaV3Profile.readConfig(directory: directory),
           let hidden = config["hidden_size"]?.intValue, hidden > 0 {
            self.storedDimension = hidden
        }
    }

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        let prefixed = texts.map {
            prefixStrategy.apply($0, task: .retrievalDocument)
        }
        return try await embed(prefixed, task: .retrievalPassage)
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        let vector = try await embed(
            [prefixStrategy.apply(text, task: .retrievalQuery)],
            task: .retrievalQuery
        )
        return vector.first ?? []
    }

    private func embed(_ texts: [String], task: JinaV3Task) async throws -> [[Float]] {
        let (model, tokenizer) = try await state.handles(directory: directory)
        let batch = tokenizer.encodeBatch(texts, maxLength: maxLength)
        let vectors = model.encode(
            inputIDs: batch.inputIDs,
            attentionMask: batch.attentionMask,
            task: task
        )
        if let hidden = vectors.first?.count, hidden > 0 {
            updateDimension(hidden)
        }
        return vectors
    }

    /// 同步更新维度，避免在 async 上下文直接持有 NSLock。
    private func updateDimension(_ hidden: Int) {
        dimensionLock.lock()
        defer { dimensionLock.unlock() }
        storedDimension = hidden
    }

    /// 释放权重；provider 保留，下次 embed 时按需重载。
    func unload() async {
        await state.unload()
    }

    /// 单条输入上限：窗口减去 [CLS]/[SEP] 与 jina document 前缀预留。
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

private actor JinaV3MLXLoadState {
    private var model: JinaV3MLXModel?
    private var tokenizer: LocalTokenizer?
    private var failure: Error?

    func handles(directory: URL) async throws -> (JinaV3MLXModel, LocalTokenizer) {
        if let model, let tokenizer { return (model, tokenizer) }
        if let failure { throw failure }

        do {
            let model = try JinaV3MLXModel(directory: directory)
            let tokenizer = try await LocalTokenizer.load(directory: directory)
            self.model = model
            self.tokenizer = tokenizer
            return (model, tokenizer)
        } catch {
            let wrapped = error as? LocalModelError ?? LocalModelError.runtime(
                "Jina v3 MLX 加载失败（\(directory.path)）：\(error.localizedDescription)")
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
