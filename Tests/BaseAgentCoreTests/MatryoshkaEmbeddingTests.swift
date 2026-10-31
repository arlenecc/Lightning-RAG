import XCTest
@testable import BaseAgentCore

private final class MatryoshkaStubEmbedding: EmbeddingProvider,
                                            EmbeddingTokenCounting, @unchecked Sendable {
    let identifier: String
    let dimension: Int
    private let source: [Float]

    init(identifier: String, dimension: Int, source: [Float]) {
        self.identifier = identifier
        self.dimension = dimension
        self.source = source
    }

    var maxContentTokens: Int { 8192 }
    func makeTokenCounter() async -> (@Sendable (String) -> Int)? { nil }

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map { _ in source }
    }

    func embedQuery(_ text: String) async throws -> [Float] { source }
}

private final class MatryoshkaSigningStub: EmbeddingProvider,
                                            EmbeddingBinarySigning,
                                            @unchecked Sendable {
    let identifier: String
    let dimension: Int
    let binaryDimension: Int
    let vector: [Float]
    let signature: Data

    init(identifier: String, dimension: Int, binaryDimension: Int,
         vector: [Float], signature: Data) {
        self.identifier = identifier
        self.dimension = dimension
        self.binaryDimension = binaryDimension
        self.vector = vector
        self.signature = signature
    }

    var binarySourceIdentifier: String { identifier }

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map { _ in vector }
    }

    func embedQuery(_ text: String) async throws -> [Float] { vector }

    func embedDocumentsWithSignatures(_ texts: [String]) async throws -> BinaryEmbeddingOutput {
        BinaryEmbeddingOutput(vectors: texts.map { _ in vector },
                              signatures: texts.map { _ in signature })
    }

    func embedQueryWithSignature(_ text: String) async throws
        -> (vector: [Float], signature: Data) {
        (vector, signature)
    }
}

final class MatryoshkaEmbeddingTests: XCTestCase {
    /// 截断后必须重新归一化：不能复用原完整向量的归一化结果。
    func testTruncateAndNormalizeRefreshesNorm() {
        let output = MatryoshkaEmbedding.truncateAndNormalize([3, 4, 5, 6], dimension: 2)
        XCTAssertEqual(output.count, 2)
        XCTAssertEqual(output[0], 0.6, accuracy: 1e-6)
        XCTAssertEqual(output[1], 0.8, accuracy: 1e-6)
        XCTAssertEqual((output[0] * output[0] + output[1] * output[1]).squareRoot(),
                       1, accuracy: 1e-6)
    }

    func testTruncateKeepsShortVectorAndNormalizes() {
        let output = MatryoshkaEmbedding.truncateAndNormalize([2, 0], dimension: 8)
        XCTAssertEqual(output, [1, 0])
    }

    func testWrapEnableRules() {
        let nomic = MatryoshkaStubEmbedding(
            identifier: "local:mlx-embedding:nomic-embed-text-v1.5-mlx-int8",
            dimension: 8, source: [1, 0, 0, 0, 0, 0, 0, 0])
        let other = MatryoshkaStubEmbedding(
            identifier: "rag-fake", dimension: 8, source: [1, 0, 0, 0, 0, 0, 0, 0])

        let autoWrapped = MatryoshkaEmbedding.wrap(nomic, mode: .auto, dimension: 4)
        XCTAssertTrue(autoWrapped is MatryoshkaEmbedding)
        XCTAssertEqual(autoWrapped.identifier,
                       "local:mlx-embedding:nomic-embed-text-v1.5-mlx-int8#mrl4")
        XCTAssertEqual(autoWrapped.dimension, 4)

        XCTAssertTrue(MatryoshkaEmbedding.wrap(other, mode: .auto, dimension: 4) === other,
                      "auto 只对已知支持 Matryoshka 的模型启用")
        XCTAssertTrue(MatryoshkaEmbedding.wrap(nomic, mode: .off, dimension: 4) === nomic)
        XCTAssertTrue(MatryoshkaEmbedding.wrap(other, mode: .on, dimension: 4) is MatryoshkaEmbedding,
                      "on 对任何模型都截断")
        XCTAssertTrue(MatryoshkaEmbedding.wrap(nomic, mode: .on, dimension: 8) === nomic,
                      "截断维不小于原生维时无需包装")
    }

    func testJinaV5ProviderIsNeverWrappedByMatryoshka() {
        let base = MatryoshkaStubEmbedding(
            identifier: "local:mlx-embedding:jina-embeddings-v5-text-nano-mlx-int4",
            dimension: 128, source: [1, 0, 0, 0])

        // V5 provider 已在内部固定 128 维并重新归一化，任何 Matryoshka
        // 模式（包括强制 on）都不能再二次截断。
        XCTAssertTrue(MatryoshkaEmbedding.wrap(base, mode: .on, dimension: 64) === base)
        XCTAssertTrue(MatryoshkaEmbedding.wrap(base, mode: .auto, dimension: 64) === base)
        let wrapped = MatryoshkaEmbedding(base: base, truncateDimension: 64)
        XCTAssertEqual(wrapped.dimension, 64,
                       "显式装饰器仍按传入维截断（绕过 wrap 的兜底行为）")
    }

    func testDocumentsAndQueryUseIdenticalTransform() async throws {
        let base = MatryoshkaStubEmbedding(
            identifier: "nomic-test", dimension: 4, source: [3, 4, 0, 0])
        let wrapped = MatryoshkaEmbedding(base: base, truncateDimension: 2)

        let documents = try await wrapped.embedDocuments(["a", "b"])
        XCTAssertEqual(documents.count, 2)
        for vector in documents {
            XCTAssertEqual(vector.count, 2)
            XCTAssertEqual(vector[0], 0.6, accuracy: 1e-6)
            XCTAssertEqual(vector[1], 0.8, accuracy: 1e-6)
        }

        let query = try await wrapped.embedQuery("q")
        XCTAssertEqual(query, documents[0], "入库与查询必须共用同一套截断 + 归一化")
        XCTAssertEqual(wrapped.maxContentTokens, 8192, "token 上限应透传底层模型")
    }

    /// 截断只作用于主向量：底层签名必须原样透传，不能从截断后的向量重算。
    func testSignaturesPassThroughWhileVectorsTruncate() async throws {
        let base = MatryoshkaSigningStub(
            identifier: "native-signing", dimension: 8, binaryDimension: 8,
            vector: [3, 4, 0, 0, -1, 1, -1, 1],
            signature: Data([0b0000_1111]))
        let wrapped = MatryoshkaEmbedding(base: base, truncateDimension: 2)

        let documents = try await wrapped.embedDocumentsWithSignatures(["a", "b"])
        XCTAssertEqual(documents.vectors.count, 2)
        XCTAssertEqual(documents.vectors[0], [0.6, 0.8])
        XCTAssertEqual(documents.signatures,
                       [Data([0b0000_1111]), Data([0b0000_1111])],
                       "签名应来自底层原生输出，而不是 [3, 4] 截断后的向量")
        XCTAssertEqual(wrapped.binaryDimension, 8)
        XCTAssertEqual(wrapped.binarySourceIdentifier, "native-signing")
        XCTAssertFalse(wrapped.binarySourceIdentifier.contains("#mrl"),
                       "签名来源不能带截断维后缀")

        let query = try await wrapped.embedQueryWithSignature("q")
        XCTAssertEqual(query.vector, [0.6, 0.8])
        XCTAssertEqual(query.signature, Data([0b0000_1111]))
    }

    /// 底层不会签名时，包装器必须从截断前的完整向量生成签名。
    func testWrapperSignsFullVectorWhenBaseCannotSign() async throws {
        let base = MatryoshkaStubEmbedding(
            identifier: "nomic-test", dimension: 8,
            source: [1, 0, 1, -1, 1, -1, 1, -1])
        let wrapped = MatryoshkaEmbedding(base: base, truncateDimension: 2)

        let documents = try await wrapped.embedDocumentsWithSignatures(["a", "b"])
        XCTAssertEqual(documents.vectors[0], [1, 0])
        XCTAssertEqual(documents.signatures,
                       [Data([0b01010101]), Data([0b01010101])],
                       "签名必须是完整 8 维的 0x55；从 [1, 0] 重算只会得到 0x01")
        XCTAssertEqual(wrapped.binaryDimension, 8)
        XCTAssertEqual(wrapped.binarySourceIdentifier, "nomic-test")

        let query = try await wrapped.embedQueryWithSignature("q")
        XCTAssertEqual(query.vector, [1, 0])
        XCTAssertEqual(query.signature, Data([0b01010101]))
    }
}
