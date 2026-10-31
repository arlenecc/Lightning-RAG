import Foundation

/// Matryoshka 截断维的合法取值。
public enum MatryoshkaDimensions {
    /// 上一版通用三档（nomic / jina-v3 的设置面板仍使用这三档）。
    public static let standard = [128, 256, 512]

    /// jina-v5 模型声明的全部 Matryoshka 维度。
    public static let jinaV5 = [32, 64, 128, 256, 512, 768]

    /// 推荐默认检索维。
    public static let defaultValue = 128
}

/// Matryoshka（俄罗斯套娃）截断策略。
///
/// - `auto`：对已知支持该特性的模型启用（nomic、jina-embeddings-v3），
///   截断维跟随设置中的下拉框；jina-v5 由 provider 内部按同一配置裁剪，
///   不经过外层包装；
/// - `on`：任何 embedding 都按配置维截断（超过原生维时按原生维处理）；
///   jina-v5 provider 自身已按配置维输出，wrap 时会跳过他，避免二次截断；
/// - `off`：不截断（jina-v5 输出原生 768 维）。
public enum MatryoshkaMode: String, CaseIterable, Sendable {
    case auto
    case on
    case off

    public var displayName: String {
        switch self {
        case .auto: return "自动（nomic / jina-v3 / jina-v5）"
        case .on: return "始终启用"
        case .off: return "关闭"
        }
    }

    /// 当前模型是否应截断。
    func enabled(for identifier: String) -> Bool {
        switch self {
        case .on: return true
        case .off: return false
        case .auto:
            return identifier.lowercased().contains("nomic")
                || JinaV3Profile.matches(identifier)
        }
    }

    /// 实际截断维：统一使用设置中的下拉框值（auto / on 均如此）。
    func effectiveDimension(for identifier: String, configured: Int) -> Int {
        configured
    }
}

/// Embedding 装饰器：截断向量 → **重新归一化**。
///
/// 截断会破坏 L2 范数（Matryoshka 训练允许前 N 维近似原向量，但不再是单位
/// 向量），必须重新归一化，否则余弦与内积打分都会偏；这里不复用底层池化时
/// 对完整向量做的归一化，而是在截断之后重新算一遍。
///
/// 入库与查询共用同一个装饰器实例，保证两侧变换完全一致。
public final class MatryoshkaEmbedding: EmbeddingProvider, EmbeddingTokenCounting,
                                        EmbeddingVectorElementConstraining,
                                        @unchecked Sendable {
    private let base: EmbeddingProvider
    private let tokenSource: EmbeddingTokenCounting?

    /// 目标截断维（模型原生维更小时按原生维）。
    public let truncateDimension: Int

    /// 透传底层模型的落盘类型约束（未来模型可能要求特定元素类型）。
    public var requiredVectorElementType: VectorElementType? {
        (base as? EmbeddingVectorElementConstraining)?.requiredVectorElementType
    }

    public init(base: EmbeddingProvider, truncateDimension: Int) {
        self.base = base
        self.truncateDimension = max(1, truncateDimension)
        self.tokenSource = base as? EmbeddingTokenCounting
    }

    /// 按配置包装 provider；无需截断时原样返回（不额外分配）。
    public static func wrap(_ base: EmbeddingProvider,
                            mode: MatryoshkaMode,
                            dimension: Int) -> EmbeddingProvider {
        // jina-v5 provider 已在内部按配置维截断并重新归一化；
        // 再套一层会二次截断，必须直接放行。
        if JinaV5Profile.matches(base.identifier) { return base }
        guard mode.enabled(for: base.identifier) else { return base }
        let target = max(1, mode.effectiveDimension(
            for: base.identifier, configured: dimension))
        guard target < base.dimension else {
            // 目标维不小于原生维：没有可截掉的部分，但若模型未归一则仍需
            // 一个归一化步骤；底层 provider 都保证归一化，这里直接复用。
            return base
        }
        return MatryoshkaEmbedding(base: base, truncateDimension: target)
    }

    public var identifier: String { "\(base.identifier)#mrl\(truncateDimension)" }

    public var dimension: Int { min(truncateDimension, base.dimension) }

    public var maxConcurrentRuns: Int { base.maxConcurrentRuns }

    public func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        let vectors = try await base.embedDocuments(texts)
        return vectors.map { Self.truncateAndNormalize($0, dimension: truncateDimension) }
    }

    public func embedQuery(_ text: String) async throws -> [Float] {
        let vector = try await base.embedQuery(text)
        return Self.truncateAndNormalize(vector, dimension: truncateDimension)
    }

    public func unload() async {
        await base.unload()
    }

    // MARK: EmbeddingTokenCounting

    /// 底层不支持精确计时返回极大值：等价于「不收口」，与未包装时一致。
    public var maxContentTokens: Int {
        tokenSource?.maxContentTokens ?? Int.max
    }

    public func makeTokenCounter() async -> (@Sendable (String) -> Int)? {
        await tokenSource?.makeTokenCounter()
    }

    // MARK: 变换

    /// 截前 `dimension` 维后重新归一化。
    static func truncateAndNormalize(_ vector: [Float], dimension: Int) -> [Float] {
        guard !vector.isEmpty else { return vector }
        var output = vector.count > dimension
            ? Array(vector.prefix(dimension))
            : vector
        var norm: Float = 0
        for value in output { norm += value * value }
        norm = norm.squareRoot()
        guard norm > 0 else { return output }
        for index in output.indices { output[index] /= norm }
        return output
    }
}

// MARK: - EmbeddingBinarySigning

extension MatryoshkaEmbedding: EmbeddingBinarySigning {
    /// 签名位宽与截断维无关。
    public var binaryDimension: Int {
        if let signing = base as? EmbeddingBinarySigning {
            return signing.binaryDimension
        }
        return ((base.dimension + 7) / 8) * 8
    }

    /// 签名来源不能带 `#mrl`：改截断维时 bin 表仍然有效。
    public var binarySourceIdentifier: String {
        (base as? EmbeddingBinarySigning)?.binarySourceIdentifier ?? base.identifier
    }

    public func embedDocumentsWithSignatures(_ texts: [String]) async throws
        -> BinaryEmbeddingOutput {
        if let signing = base as? EmbeddingBinarySigning {
            let output = try await signing.embedDocumentsWithSignatures(texts)
            return BinaryEmbeddingOutput(
                vectors: output.vectors.map {
                    Self.truncateAndNormalize($0, dimension: truncateDimension)
                },
                signatures: output.signatures)
        }

        let vectors = try await base.embedDocuments(texts)
        let bitDimension = binaryDimension
        return BinaryEmbeddingOutput(
            vectors: vectors.map {
                Self.truncateAndNormalize($0, dimension: truncateDimension)
            },
            signatures: vectors.map {
                VectorEncoder.encodeSignature($0, bitDimension: bitDimension)
            })
    }

    public func embedQueryWithSignature(_ text: String) async throws
        -> (vector: [Float], signature: Data) {
        if let signing = base as? EmbeddingBinarySigning {
            let output = try await signing.embedQueryWithSignature(text)
            return (Self.truncateAndNormalize(output.vector, dimension: truncateDimension),
                    output.signature)
        }

        let vector = try await base.embedQuery(text)
        return (Self.truncateAndNormalize(vector, dimension: truncateDimension),
                VectorEncoder.encodeSignature(vector, bitDimension: binaryDimension))
    }
}
