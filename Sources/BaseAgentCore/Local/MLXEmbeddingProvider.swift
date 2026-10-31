import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon

/// 基于 MLXEmbedders 的本地 Embedding 实现。
///
/// 与 ONNX 实现保持同样的接口：document/query 前缀、pooling + L2 归一化。
/// 前缀按模型族统一处理（`EmbeddingPrefixStrategy`）。
public final class MLXEmbeddingProvider: EmbeddingProvider, EmbeddingTokenCounting,
                                         EmbeddingBinarySigning, @unchecked Sendable {
    public static let bgeQueryInstruction = EmbeddingPrefixStrategy.bgeQueryInstruction

    public let identifier: String
    public let modelName: String
    public let directory: URL

    private let maxLength: Int
    private let prefixStrategy: EmbeddingPrefixStrategy
    private let state = MLXEmbeddingLoadState()
    private let dimensionLock = NSLock()
    private var storedDimension = 768

    public var dimension: Int {
        dimensionLock.lock()
        defer { dimensionLock.unlock() }
        return storedDimension
    }

    public init(directory: URL, modelName: String) {
        self.directory = directory
        self.modelName = modelName
        self.identifier = "local:mlx-embedding:\(modelName)"
        // bge 只有 bge-zh 在 query 侧加官方检索指令；en 版与 m3 不加。
        self.prefixStrategy = EmbeddingPrefixStrategy(
            modelName: modelName,
            modelPath: directory.lastPathComponent)
        self.maxLength = Self.window(directory: directory)

        if let data = FileManager.default.contents(
            atPath: directory.appendingPathComponent("config.json").path),
           let config = JSONValue.parse(String(decoding: data, as: UTF8.self))?.objectValue {
            let hidden = config["hidden_size"]?.intValue
                ?? config["text_config"]?.objectValue?["hidden_size"]?.intValue
            if let hidden, hidden > 0 {
                storedDimension = hidden
            }
        }
    }

    public func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        let prefixed = texts.map {
            prefixStrategy.apply($0, task: .retrievalDocument)
        }
        return try await embed(prefixed)
    }

    public func embedQuery(_ text: String) async throws -> [Float] {
        return try await embed([
            prefixStrategy.apply(text, task: .retrievalQuery)
        ]).first ?? []
    }

    private func embed(_ texts: [String]) async throws -> [[Float]] {
        let container = try await state.container(directory: directory)
        let maxLength = self.maxLength
        let vectors = try await container.perform { context -> [[Float]] in
            let tokenizer = context.tokenizer
            let eos = tokenizer.eosTokenId ?? 0
            let encoded = texts.map { text -> [Int] in
                Array(tokenizer.encode(text: text, addSpecialTokens: true)
                    .prefix(maxLength))
            }
            let width = max(16, encoded.map(\.count).max() ?? 16)
            let padded = stacked(encoded.map { ids in
                MLXArray(ids + Array(repeating: eos, count: width - ids.count))
            })
            let mask = padded .!= eos
            let tokenTypes = MLXArray.zeros(like: padded)
            let output = context.model(padded, positionIds: nil,
                                       tokenTypeIds: tokenTypes,
                                       attentionMask: mask)
            let pooled = context.pooling(output, normalize: true,
                                         applyLayerNorm: true)
            pooled.eval()
            return pooled.map { $0.asArray(Float.self) }
        }
        if let hidden = vectors.first?.count, hidden > 0 {
            dimensionLock.lock()
            storedDimension = hidden
            dimensionLock.unlock()
        }
        return vectors
    }

    /// 释放 embedding 权重；provider 保留，下次 embed 时按需重载。
    public func unload() async {
        await state.unload()
    }

    /// 单条输入上限：窗口减去 [CLS]/[SEP] 与当前模型族 document 前缀预留。
    /// bge-zh 的 document 不加指令前缀（query 才加），内容预算仍为 512 - 2 = 510。
    public var maxContentTokens: Int {
        Self.contentTokenLimit(directory: directory, modelName: modelName)
    }

    /// 与 `maxContentTokens` 同源的静态换算：设置界面不加载权重也能算出
    /// 「切片大小实际生效值」，两处口径不会漂移。
    public static func contentTokenLimit(directory: URL, modelName: String) -> Int {
        let reserve = EmbeddingPrefixStrategy(
            modelName: modelName,
            modelPath: directory.lastPathComponent).documentPrefixTokenReserve
        return max(16, window(directory: directory) - 2 - reserve)
    }

    /// 模型声明的最大序列长度（min 128；推理前统一压到 1024 token）。
    private static func window(directory: URL) -> Int {
        EmbeddingInputLimits.cappedWindow(max(LocalModelRegistry.resolveMaxSequenceLength(
            directory: directory, defaultValue: 512), 128))
    }

    /// 切片用的精确 token 计数（不含特殊 token）：只加载 tokenizer，不牵动模型权重。
    public func makeTokenCounter() async -> (@Sendable (String) -> Int)? {
        await TokenizerCounterCache.shared.counter(for: directory)
    }
}

private actor MLXEmbeddingLoadState {
    private var container: EmbedderModelContainer?
    private var failure: Error?

    func container(directory: URL) async throws -> EmbedderModelContainer {
        if let container { return container }
        if let failure { throw failure }
        do {
            let container = try await EmbedderModelFactory.shared.loadContainer(
                from: directory, using: LocalTokenizerLoader())
            self.container = container
            return container
        } catch {
            let wrapped = LocalModelError.runtime(
                "MLX Embedding 加载失败（\(directory.path)）：\(error.localizedDescription)")
            failure = wrapped
            throw wrapped
        }
    }

    func unload() {
        container = nil
        failure = nil
    }
}
