import XCTest
import Foundation
@testable import BaseAgentCore

/// 真实 jina-embeddings-v5-text-nano ONNX int8 模型验证（可选后端）。
///
/// V5 ONNX 图内已完成 last-token pooling + L2，provider 只需按检索约定
/// 截取前 128 维并重新归一化。目录可用 `JINA_V5_ONNX_DIR` 覆盖。
final class JinaV5ONNXEmbeddingE2ETests: XCTestCase {
    private static let modelName = "jina-embeddings-v5-text-nano-onnx-int8"

    private func modelDirectory() throws -> URL {
        let override = ProcessInfo.processInfo.environment["JINA_V5_ONNX_DIR"]
        let directory = override.map { URL(fileURLWithPath: $0) }
            ?? EmbeddingBenchmarkTests.resolveModelsRoot()
                .appendingPathComponent(Self.modelName)
        guard JinaV5Profile.isJinaV5(directory: directory) else {
            throw XCTSkip("模型缺失或不是 jina-v5：\(directory.path)")
        }
        let onnx = directory.appendingPathComponent("model_quantized.onnx")
        let onnxAlt = directory.appendingPathComponent("model.onnx")
        guard FileManager.default.fileExists(atPath: onnx.path)
                || FileManager.default.fileExists(atPath: onnxAlt.path) else {
            throw XCTSkip("ONNX 权重缺失：\(directory.path)")
        }
        return directory
    }

    func testOnnxEmbeds128DimensionalNormalizedVectors() async throws {
        let directory = try modelDirectory()
        let provider = ONNXEmbeddingProvider(
            directory: directory, modelName: Self.modelName)
        addTeardownBlock { await provider.unload() }

        XCTAssertEqual(provider.identifier, "local:onnx:\(Self.modelName)")
        XCTAssertEqual(provider.dimension, 128, "v5 必须直接对外暴露 128 维")
        XCTAssertEqual(provider.prefixStrategy.family, .jinaV5)
        XCTAssertEqual(provider.prefixStrategy.prefix(for: .retrievalDocument), "Document: ")
        XCTAssertEqual(provider.prefixStrategy.prefix(for: .retrievalQuery), "Query: ")
        XCTAssertEqual(provider.maxContentTokens, 1_018, "1024 - [CLS]/[SEP] - V5 前缀预留")

        let query = try await provider.embedQuery("苹果发布了新手机")
        XCTAssertEqual(query.count, 128)
        XCTAssertEqual(norm(query), 1, accuracy: 1e-3)

        let documents = try await provider.embedDocuments([
            "苹果公司发布了新款智能手机。",
            "机器学习通过数据训练模型自动改进任务表现。",
            "红烧肉的关键在于煸炒糖色和慢火炖煮。",
            "我想吃一个红苹果。",
        ])
        XCTAssertEqual(documents.map(\.count), [128, 128, 128, 128])
        for vector in documents {
            XCTAssertEqual(norm(vector), 1, accuracy: 1e-3)
        }

        let scores = documents.map { cosine(query, $0) }
        print("[jina-v5-onnx-e2e] scores="
            + scores.map { String(format: "%.4f", $0) }.description)
        XCTAssertEqual(scores[0], 0.796, accuracy: 0.02,
                       "应与 example.py 的 128 维参考分数一致")
        XCTAssertGreaterThan(scores[0], scores[1], "手机语义应比机器学习更相关")
        XCTAssertGreaterThan(scores[0], scores[3], "手机语义应比红苹果更相关")
    }

    func testOnnxQueryAndDocumentUseDistinctPrefixes() async throws {
        let directory = try modelDirectory()
        let provider = ONNXEmbeddingProvider(
            directory: directory, modelName: Self.modelName)
        addTeardownBlock { await provider.unload() }

        let asQuery = try await provider.embedQuery("苹果")
        let asDocument = try await provider.embedDocuments(["苹果"])[0]
        XCTAssertLessThan(cosine(asQuery, asDocument), 0.999,
                          "Query: / Document: 前缀必须真正改变输入")
    }

    func testOnnxSignaturesUseNative768Dimension() async throws {
        let directory = try modelDirectory()
        let narrow = ONNXEmbeddingProvider(
            directory: directory, modelName: Self.modelName, v5Dimension: 128)
        let wide = ONNXEmbeddingProvider(
            directory: directory, modelName: Self.modelName, v5Dimension: 256)
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
        XCTAssertEqual(narrowOutput.signatures.map(\.count), [96, 96])
        XCTAssertEqual(narrowOutput.signatures, wideOutput.signatures,
                       "ONNX v5 也必须在截断前取签名")

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
