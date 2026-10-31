import XCTest
import Foundation
@testable import BaseAgentCore

/// 真实 jina-embeddings-v5-text-nano MLX 模型验证。
///
/// 与 v3 不同，V5 provider 必须直接输出 128 维单位向量：截取原生 768 维的
/// 前 128 维并重新做 L2 归一化，外层 Matryoshka 不再二次处理。
final class JinaV5MLXEmbeddingE2ETests: XCTestCase {
    private static let modelName = "jina-embeddings-v5-text-nano-mlx-int4"

    /// Python 参考实现（`models/jina-embeddings-v5-text-nano-mlx-int4/model.py`）
    /// 的 128 维检索分数：苹果新手机 0.8292，机器学习 0.2910，红苹果 0.4801。
    private static let expectedPhoneScore: Float = 0.829

    override func setUpWithError() throws {
        EmbeddingBenchmarkTests.configureMetallib()
    }

    func testEmbeds128DimensionalNormalizedVectorsAndRanksRelevantDocumentFirst() async throws {
        let directory = EmbeddingBenchmarkTests.resolveModelsRoot()
            .appendingPathComponent(Self.modelName)
        try XCTSkipUnless(
            JinaV5Profile.isJinaV5(directory: directory)
                && JinaV5Profile.weightsURL(in: directory) != nil,
            "模型缺失：\(directory.path)")

        let provider = JinaV5MLXEmbeddingProvider(
            directory: directory, modelName: Self.modelName)
        addTeardownBlock { await provider.unload() }

        XCTAssertEqual(provider.identifier, "local:mlx-embedding:\(Self.modelName)")
        XCTAssertEqual(provider.dimension, 128, "v5 provider 直接输出 128 维")

        let query = try await provider.embedQuery("苹果发布了新手机")
        XCTAssertEqual(query.count, 128)
        XCTAssertEqual(norm(query), 1, accuracy: 1e-3, "截断后必须重新 L2 归一化")

        let documents = try await provider.embedDocuments([
            "苹果公司推出了新款智能手机。",
            "机器学习通过数据训练模型，让计算机自动改进任务表现。",
            "我想吃一个红苹果。",
        ])
        XCTAssertEqual(documents.map(\.count), [128, 128, 128])
        for vector in documents {
            XCTAssertEqual(norm(vector), 1, accuracy: 1e-3)
        }

        let scores = documents.map { cosine(query, $0) }
        print("[jina-v5-e2e] scores=" + scores.map { String(format: "%.4f", $0) }.description)
        XCTAssertEqual(scores[0], Self.expectedPhoneScore, accuracy: 0.08,
                       "Swift 前向应与 model.py 参考实现一致")
        XCTAssertGreaterThan(scores[0], scores[1], "手机语义应比机器学习更相关")
        XCTAssertGreaterThan(scores[0], scores[2], "手机语义应比红苹果更相关")
    }

    func testQueryAndDocumentUseDistinctJinaV5Prefixes() async throws {
        let directory = EmbeddingBenchmarkTests.resolveModelsRoot()
            .appendingPathComponent(Self.modelName)
        try XCTSkipUnless(
            JinaV5Profile.isJinaV5(directory: directory)
                && JinaV5Profile.weightsURL(in: directory) != nil,
            "模型缺失：\(directory.path)")

        let provider = JinaV5MLXEmbeddingProvider(
            directory: directory, modelName: Self.modelName)
        addTeardownBlock { await provider.unload() }

        XCTAssertEqual(provider.prefixStrategy.family, .jinaV5)
        XCTAssertEqual(provider.prefixStrategy.prefix(for: .retrievalDocument), "Document: ")
        XCTAssertEqual(provider.prefixStrategy.prefix(for: .retrievalQuery), "Query: ")

        // 同一文本在不同任务下应得到不同向量（前缀确实生效）。
        let asQuery = try await provider.embedQuery("苹果")
        let asDocument = try await provider.embedDocuments(["苹果"])[0]
        XCTAssertGreaterThan(cosine(asQuery, asDocument), 0.5,
                             "同一文本的 query/doc 表示应保持较高相似度")
        XCTAssertLessThan(cosine(asQuery, asDocument), 0.999,
                          "前缀必须真正改变输入，不能是空操作")
    }

    func testSignaturesUseNative768DimensionNotTruncatedOutput() async throws {
        let directory = EmbeddingBenchmarkTests.resolveModelsRoot()
            .appendingPathComponent(Self.modelName)
        try XCTSkipUnless(
            JinaV5Profile.isJinaV5(directory: directory)
                && JinaV5Profile.weightsURL(in: directory) != nil,
            "模型缺失：\(directory.path)")

        let narrow = JinaV5MLXEmbeddingProvider(
            directory: directory, modelName: Self.modelName, dimension: 128)
        let wide = JinaV5MLXEmbeddingProvider(
            directory: directory, modelName: Self.modelName, dimension: 256)
        addTeardownBlock {
            await narrow.unload()
            await wide.unload()
        }

        XCTAssertEqual(narrow.binaryDimension, 768)
        XCTAssertEqual(narrow.binarySourceIdentifier, narrow.identifier)

        let texts = ["苹果公司发布了新款智能手机。", "机器学习通过数据训练模型。"]
        let narrowOutput = try await narrow.embedDocumentsWithSignatures(texts)
        let wideOutput = try await wide.embedDocumentsWithSignatures(texts)

        XCTAssertEqual(narrowOutput.vectors.map(\.count), [128, 128])
        XCTAssertEqual(wideOutput.vectors.map(\.count), [256, 256])
        XCTAssertEqual(narrowOutput.signatures.count, texts.count)
        XCTAssertEqual(narrowOutput.signatures.map(\.count), [96, 96],
                       "768 bit = 96 字节")
        XCTAssertEqual(narrowOutput.signatures, wideOutput.signatures,
                       "签名必须来自截断前的同一份原生输出")

        let query = try await narrow.embedQueryWithSignature("苹果发布了新手机")
        XCTAssertEqual(query.vector.count, 128)
        XCTAssertEqual(query.signature.count, 96)
    }

    // MARK: - 工具

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        for index in a.indices {
            dot += a[index] * b[index]
            normA += a[index] * a[index]
            normB += b[index] * b[index]
        }
        let denominator = normA.squareRoot() * normB.squareRoot()
        return denominator > 0 ? dot / denominator : 0
    }

    private func norm(_ vector: [Float]) -> Float {
        vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
    }
}
