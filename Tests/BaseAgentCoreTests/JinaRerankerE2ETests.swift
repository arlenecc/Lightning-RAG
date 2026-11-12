import XCTest
import Foundation
@testable import BaseAgentCore

/// 真实 jina reranker：验证内置模型可加载、pair 编码正确、能稳定排序相关段落。
final class JinaRerankerE2ETests: XCTestCase {
    private static let modelName = "jina-reranker-v2-base-multilingual-onnx"

    override func setUp() {
        super.setUp()
        unsetenv("BASE_AGENT_MODELS_DIR")
        LocalModelFactory.shared.clearCaches()
    }

    override func tearDown() {
        unsetenv("BASE_AGENT_MODELS_DIR")
        LocalModelFactory.shared.clearCaches()
        super.tearDown()
    }

    private func modelDirectory() throws -> URL {
        let root = EmbeddingBenchmarkTests.resolveModelsRoot()
        let directory = root.appendingPathComponent(Self.modelName, isDirectory: true)
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("model_quantized.onnx").path),
            "模型缺失：\(directory.path)")
        return directory
    }

    private func makeProvider(directory: URL) throws -> RerankProvider {
        var config = AgentConfig()
        config.localRerankerBackend = "onnx"
        config.localRerankerModel = Self.modelName
        config.localRerankerModelPath = directory.path
        return try XCTUnwrap(LocalModelFactory.shared.makeReranker(config: config))
    }

    /// XLM-R pair 必须是 `<s> query </s></s> document </s>`。
    /// 该结构错误（首 token 为 `<unk>`、query 后只有一个 `</s>`）会让重排分数明显失真。
    func testXLMRPairEncodingMatchesReferenceStructure() async throws {
        let directory = try modelDirectory()
        let tokenizer = try await LocalTokenizer.load(directory: directory)
        XCTAssertTrue(tokenizer.usesXLMRPairFormat)
        let bos = try XCTUnwrap(tokenizer.base.bosTokenId
            ?? tokenizer.base.convertTokenToId("<s>"))
        let eos = try XCTUnwrap(tokenizer.base.eosTokenId
            ?? tokenizer.base.convertTokenToId("</s>"))
        XCTAssertNotEqual(bos, tokenizer.base.unknownTokenId)
        XCTAssertNotEqual(eos, tokenizer.base.unknownTokenId)

        let target = EmbeddingBenchmarkTests.targets[5]
        let queryIDs = tokenizer.encode(target.query, addSpecialTokens: false)
        let documentIDs = tokenizer.encode(target.passage, addSpecialTokens: false)
        let ids = tokenizer.encodePair(
            query: target.query, document: target.passage, maxLength: 1024)

        XCTAssertEqual(Array(ids.prefix(queryIDs.count + 3)),
                       [bos] + queryIDs + [eos, eos],
                       "query 段结构错误")
        XCTAssertEqual(Array(ids.suffix(documentIDs.count + 2)),
                       [eos] + documentIDs + [eos],
                       "document 段结构错误")
        XCTAssertEqual(ids.count, queryIDs.count + documentIDs.count + 4)
        XCTAssertEqual(ids.first, bos)

        // 截断时必须保留 document 段末尾的 `</s>`。
        let truncated = tokenizer.encodePair(
            query: target.query, document: target.passage, maxLength: 10)
        XCTAssertEqual(truncated.count, 10)
        XCTAssertEqual(truncated.first, bos)
        XCTAssertEqual(truncated.last, eos)
    }

    /// 自动下载的 HF ONNX 仓库布局：权重在 `onnx/` 子目录、tokenizer 在根目录。
    /// 旧实现把 tokenizer 目录取成 `onnx/`，加载报「缺少 tokenizer.json」。
    func testNestedONNXLayoutLoadsTokenizerFromModelRoot() async throws {
        let source = try modelDirectory()
        let temp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("nested-onnx-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: temp) }
        let onnxDir = temp.appendingPathComponent("onnx", isDirectory: true)
        try FileManager.default.createDirectory(at: onnxDir, withIntermediateDirectories: true)
        let fm = FileManager.default
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json"] {
            let src = source.appendingPathComponent(name)
            if fm.fileExists(atPath: src.path) {
                try fm.copyItem(at: src, to: temp.appendingPathComponent(name))
            }
        }
        // 权重用符号链接，避免复制 280MB。
        try fm.createSymbolicLink(
            atPath: onnxDir.appendingPathComponent("model_quantized.onnx").path,
            withDestinationPath: source.appendingPathComponent("model_quantized.onnx").path)

        let reranker = ONNXReranker(directory: temp, modelName: "nested-layout")
        XCTAssertEqual(reranker.directory.path, temp.path,
                       "tokenizer 应回到模型根目录读取")
        let scores = try await reranker.rerank(
            query: "光合作用需要什么",
            documents: ["植物需要阳光进行光合作用", "今天股市上涨"])
        XCTAssertEqual(scores.count, 2)
        XCTAssertGreaterThan(scores[0], scores[1])
    }

    /// 真实 hard negative 对照。q4-doc04 的 logit 恰为正数且落在 [0,1]，
    /// 若把 logit 误当概率直接返回，分数会从 0.55 掉到 0.18。
    func testHardNegativePassagesRankCorrectly() async throws {
        let directory = try modelDirectory()
        let provider = try makeProvider(directory: directory)
        defer { LocalModelFactory.shared.clearCaches() }

        let targets = EmbeddingBenchmarkTests.targets
        let q6 = targets[5]
        let q4 = targets[3]
        let unrelated = targets[0]

        let q6Scores = try await provider.rerank(
            query: q6.query, documents: [q6.passage, unrelated.passage])
        XCTAssertEqual(q6Scores.count, 2)
        XCTAssertGreaterThan(q6Scores[0], q6Scores[1],
                             "相关段落应高于无关段落：\(q6Scores)")
        // 整段原文接近 900 token，分数会被长度稀释；0.45 阈值针对 RAG 的 800 字切片，
        // 在 RAGPipelineVerificationTests 中单独验证，这里只保证相对排序与绝对值合理。
        XCTAssertGreaterThan(q6Scores[0], 0.3, "相关原文分数不应塌陷：\(q6Scores)")

        let q4Scores = try await provider.rerank(
            query: q4.query, documents: [q4.passage, unrelated.passage])
        XCTAssertEqual(q4Scores.count, 2)
        XCTAssertGreaterThan(q4Scores[0], q4Scores[1],
                             "相关段落应高于无关段落：\(q4Scores)")
        XCTAssertGreaterThan(q4Scores[0], 0.5,
                             "正 logit 必须做 sigmoid，不能直接当概率返回：\(q4Scores)")

        for score in q6Scores + q4Scores {
            XCTAssertTrue(score >= 0 && score <= 1, "分数应归一化到 [0,1]，实际 \(score)")
        }
        print("[jina-reranker-e2e] q6=\(q6Scores) q4=\(q4Scores)")
    }

    /// int8 模型对 padding 宽度有轻微数值敏感，但批大小不能改变排序结果。
    func testBatchSizeDoesNotChangeTopRanking() async throws {
        let directory = try modelDirectory()
        let batch8 = try makeProvider(directory: directory)
        let batch1 = ONNXReranker(directory: directory, modelName: Self.modelName + "-b1",
                                  batchSize: 1)
        defer {
            LocalModelFactory.shared.clearCaches()
            Task { await batch1.unload() }
        }

        let targets = EmbeddingBenchmarkTests.targets
        let q6 = targets[5]
        let documents = targets.map(\.passage)

        let scores8 = try await batch8.rerank(query: q6.query, documents: documents)
        let scores1 = try await batch1.rerank(query: q6.query, documents: documents)
        XCTAssertEqual(scores8.count, documents.count)
        XCTAssertEqual(scores1.count, documents.count)

        let top8 = scores8.indices.max { scores8[$0] < scores8[$1] }
        let top1 = scores1.indices.max { scores1[$0] < scores1[$1] }
        XCTAssertEqual(top8, 5, "batch8 应把 doc06 排第一：\(scores8)")
        XCTAssertEqual(top1, 5, "batch1 应把 doc06 排第一：\(scores1)")
        XCTAssertLessThan(abs(scores8[5] - scores1[5]), 0.15,
                          "批大小带来的分数漂移应可控：\(scores8[5]) vs \(scores1[5])")
        print("[jina-reranker-e2e] batch8 doc06=\(scores8[5]) "
            + "batch1 doc06=\(scores1[5])")
    }
}
