import XCTest
import Foundation
@testable import BaseAgentCore

/// 工厂接线与 App bundle 模型目录解析。
final class LocalModelFactoryTests: XCTestCase {
    private static let modelName = "jina-embeddings-v5-text-nano-mlx-int4"

    private var temporaryRoot: URL!

    override func setUpWithError() throws {
        temporaryRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("local-factory-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temporaryRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let temporaryRoot { try? FileManager.default.removeItem(at: temporaryRoot) }
    }

    func testFactoryBuildsJinaV5MLXProviderWith128Dimensions() throws {
        let modelsRoot = EmbeddingBenchmarkTests.resolveModelsRoot()
        let directory = modelsRoot.appendingPathComponent(Self.modelName)
        try XCTSkipUnless(
            JinaV5Profile.isJinaV5(directory: directory)
                && JinaV5Profile.weightsURL(in: directory) != nil,
            "模型缺失：\(directory.path)")

        setenv("BASE_AGENT_MODELS_DIR", modelsRoot.path, 1)
        defer {
            unsetenv("BASE_AGENT_MODELS_DIR")
            LocalModelFactory.shared.clearCaches()
        }

        var config = AgentConfig()
        config.localEmbeddingBackend = "mlx"
        config.localEmbeddingModel = Self.modelName
        config.localEmbeddingModelPath = ""

        let provider = try XCTUnwrap(
            LocalModelFactory.shared.makeEmbedding(config: config),
            "V5 MLX 应能通过工厂按名称解析")
        XCTAssertTrue(provider is JinaV5MLXEmbeddingProvider,
                      "jina-v5-mlx 不能落到 MLXEmbedders 默认工厂：\(provider)")
        XCTAssertEqual(provider.identifier, "local:mlx-embedding:\(Self.modelName)")
        XCTAssertEqual(provider.dimension, 128, "V5 provider 必须直接输出 128 维")
    }

    /// reranker 目录解析失败不缓存：模型下载/补全到位后，下一次 makeReranker
    /// 必须能重新解析成功，而不是一直拿到缓存的 Failing 实例（必须重启）。
    func testRerankerFailureIsNotCachedAndRecovers() throws {
        let dir = temporaryRoot.appendingPathComponent("late-reranker", isDirectory: true)
        var config = AgentConfig()
        config.workspace = temporaryRoot.path
        config.rerankerProvider = "local"
        config.localRerankerBackend = "onnx"
        config.localRerankerModel = ""
        config.ragRerankModel = ""
        config.localRerankerModelPath = dir.path
        defer { LocalModelFactory.shared.clearCaches() }

        // 目录尚不存在：首次解析失败，拿不到真实 provider。
        XCTAssertFalse(LocalModelFactory.shared.makeReranker(config: config)
            is ONNXReranker)

        // 模型补全（权重 + tokenizer.json）后同一份配置应恢复。
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        XCTAssertTrue(FileManager.default.createFile(
            atPath: dir.appendingPathComponent("model_quantized.onnx").path,
            contents: Data(repeating: 1, count: 8_192)))
        XCTAssertTrue(FileManager.default.createFile(
            atPath: dir.appendingPathComponent("tokenizer.json").path,
            contents: Data("{}".utf8)))

        let recovered = LocalModelFactory.shared.makeReranker(config: config)
        XCTAssertTrue(recovered is ONNXReranker,
                      "目录就位后不应继续返回缓存的 Failing：\(String(describing: recovered))")
    }

    func testFactoryDefaultsMLXEmbeddingToJinaV5() throws {
        let modelsRoot = EmbeddingBenchmarkTests.resolveModelsRoot()
        let directory = modelsRoot.appendingPathComponent(Self.modelName)
        try XCTSkipUnless(
            JinaV5Profile.isJinaV5(directory: directory)
                && JinaV5Profile.weightsURL(in: directory) != nil,
            "模型缺失：\(directory.path)")

        setenv("BASE_AGENT_MODELS_DIR", modelsRoot.path, 1)
        defer {
            unsetenv("BASE_AGENT_MODELS_DIR")
            LocalModelFactory.shared.clearCaches()
        }

        // 用户未指定任何 embedding 模型：MLX 后端应落到 V5 int4 默认值。
        var config = AgentConfig()
        config.localEmbeddingBackend = "mlx"
        config.localEmbeddingModel = ""
        config.localEmbeddingModelPath = ""
        config.ragEmbeddingModel = ""

        let provider = try XCTUnwrap(
            LocalModelFactory.shared.makeEmbedding(config: config))
        XCTAssertTrue(provider is JinaV5MLXEmbeddingProvider, "\(provider)")
        XCTAssertEqual(provider.identifier, "local:mlx-embedding:\(Self.modelName)")
        XCTAssertEqual(provider.dimension, 128)
    }

    /// jina-v5 解锁后：工厂按 Matryoshka 配置把维度传给 provider，
    /// 并且缓存 key 随维度变化，避免配置切换后复用旧实例。
    func testFactoryPassesJinaV5ConfiguredDimensionToMLX() throws {
        let directory = try makeFakeV5Directory()
        defer { LocalModelFactory.shared.clearCaches() }

        var config = AgentConfig()
        config.localEmbeddingBackend = "mlx"
        config.localEmbeddingModel = "jina-embeddings-v5-text-nano-test"
        config.localEmbeddingModelPath = directory.path
        config.ragMatryoshka = .auto

        config.ragMatryoshkaDim = 512
        let wide = try XCTUnwrap(
            LocalModelFactory.shared.makeEmbedding(config: config))
        XCTAssertTrue(wide is JinaV5MLXEmbeddingProvider, "\(wide)")
        XCTAssertEqual(wide.dimension, 512)
        XCTAssertNil((wide as? EmbeddingVectorElementConstraining)?
            .requiredVectorElementType, "jina-v5 不应再强制 int8")

        config.ragMatryoshkaDim = 128
        let narrow = try XCTUnwrap(
            LocalModelFactory.shared.makeEmbedding(config: config))
        XCTAssertEqual(narrow.dimension, 128)
        XCTAssertFalse((wide as AnyObject) === (narrow as AnyObject),
                       "维度变化不能复用旧维度 provider")

        config.ragMatryoshka = .off
        let native = try XCTUnwrap(
            LocalModelFactory.shared.makeEmbedding(config: config))
        XCTAssertEqual(native.dimension, JinaV5Profile.nativeDimension)
    }

    func testFactoryPassesJinaV5ConfiguredDimensionToONNX() throws {
        let directory = try makeFakeV5Directory()
        defer { LocalModelFactory.shared.clearCaches() }

        var config = AgentConfig()
        config.localEmbeddingBackend = "onnx"
        config.localEmbeddingModel = "jina-embeddings-v5-text-nano-test"
        config.localEmbeddingModelPath = directory.path
        config.ragMatryoshka = .auto
        config.ragMatryoshkaDim = 256

        let provider = try XCTUnwrap(
            LocalModelFactory.shared.makeEmbedding(config: config))
        XCTAssertTrue(provider is ONNXEmbeddingProvider, "\(provider)")
        XCTAssertEqual(provider.dimension, 256)
    }

    /// 构造只含 config.json 的 V5 目录：足以识别模型族，无需真实权重即可建 provider。
    private func makeFakeV5Directory() throws -> URL {
        let directory = temporaryRoot.appendingPathComponent(
            "jina-embeddings-v5-text-nano-test", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let modelConfig: [String: Any] = [
            "model_type": "eurobert",
            "base_model": "jinaai/jina-embeddings-v5-text-nano",
            "hidden_size": 768,
        ]
        try JSONSerialization.data(withJSONObject: modelConfig)
            .write(to: directory.appendingPathComponent("config.json"))
        return directory
    }

    // MARK: - Bundle 模型目录

    func testBundledModelsDirectoryPrefersResources() throws {
        let app = temporaryRoot.appendingPathComponent("Fake.app", isDirectory: true)
        let resources = app.appendingPathComponent("Contents/Resources", isDirectory: true)
        let models = resources.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)

        let resolved = LocalModelRegistry.bundledModelsDirectory(
            resourceURL: resources, bundleURL: app)
        XCTAssertEqual(resolved?.standardizedFileURL.path,
                       models.standardizedFileURL.path,
                       "打包后的模型应优先从 Contents/Resources/models 解析")
    }

    func testBundledModelsDirectoryFallsBackToContentsResourcesPath() throws {
        let app = temporaryRoot.appendingPathComponent("Fake.app", isDirectory: true)
        let models = app.appendingPathComponent("Contents/Resources/models", isDirectory: true)
        try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)

        let resolved = LocalModelRegistry.bundledModelsDirectory(
            resourceURL: nil, bundleURL: app)
        XCTAssertEqual(resolved?.standardizedFileURL.path,
                       models.standardizedFileURL.path)
    }

    func testBundledModelsDirectoryReturnsNilWhenAbsent() {
        let app = temporaryRoot.appendingPathComponent("Fake.app", isDirectory: true)
        XCTAssertNil(LocalModelRegistry.bundledModelsDirectory(
            resourceURL: app.appendingPathComponent("Contents/Resources"), bundleURL: app))
    }
}
