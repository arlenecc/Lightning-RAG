import XCTest
import Foundation
@testable import BaseAgentCore

/// 内置 reranker 的打包清单与默认配置解析。
final class BundledRerankerTests: XCTestCase {
    private static let modelName = "jina-reranker-v2-base-multilingual-onnx"

    private var temporaryRoot: URL!

    override func setUpWithError() throws {
        // 全量测试中其它用例可能已按不同模型目录缓存过 FailingRerankProvider；
        // 先清空缓存与环境，避免命中旧失败实例。
        unsetenv("BASE_AGENT_MODELS_DIR")
        LocalModelFactory.shared.clearCaches()
        temporaryRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("bundled-reranker-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temporaryRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        unsetenv("BASE_AGENT_MODELS_DIR")
        LocalModelFactory.shared.clearCaches()
        if let temporaryRoot { try? FileManager.default.removeItem(at: temporaryRoot) }
    }

    /// HF ONNX 仓库常把权重放在 `onnx/` 子目录、tokenizer 与 config 放在根目录。
    /// 读取目录必须回到根目录，否则加载时报「缺少 tokenizer：<root>/onnx/tokenizer.json」。
    func testONNXSupportDirectoryHandlesNestedWeightLayout() throws {
        let nested = temporaryRoot.appendingPathComponent("nested-reranker", isDirectory: true)
        let onnxDir = nested.appendingPathComponent("onnx", isDirectory: true)
        try FileManager.default.createDirectory(at: onnxDir, withIntermediateDirectories: true)
        let nestedWeight = onnxDir.appendingPathComponent("model_quantized.onnx")
        FileManager.default.createFile(
            atPath: nestedWeight.path, contents: Data(repeating: 1, count: 64))
        FileManager.default.createFile(
            atPath: nested.appendingPathComponent("config.json").path,
            contents: Data("{}".utf8))
        FileManager.default.createFile(
            atPath: nested.appendingPathComponent("tokenizer.json").path,
            contents: Data("{}".utf8))

        XCTAssertEqual(
            LocalModelRegistry.onnxSupportDirectory(modelURL: nestedWeight, modelRoot: nested).path,
            nested.path, "权重在 onnx/ 时应回到模型根目录读 tokenizer")

        // provider 自身也必须用根目录，避免真实加载时仍报缺少 tokenizer。
        let reranker = ONNXReranker(directory: nested, modelName: "nested-layout")
        XCTAssertEqual(reranker.directory.path, nested.path)

        // 平铺布局：权重与 tokenizer 同在根目录。
        let flat = temporaryRoot.appendingPathComponent("flat-reranker", isDirectory: true)
        try FileManager.default.createDirectory(at: flat, withIntermediateDirectories: true)
        let flatWeight = flat.appendingPathComponent("model_quantized.onnx")
        FileManager.default.createFile(
            atPath: flatWeight.path, contents: Data(repeating: 1, count: 64))
        FileManager.default.createFile(
            atPath: flat.appendingPathComponent("tokenizer.json").path,
            contents: Data("{}".utf8))
        XCTAssertEqual(
            LocalModelRegistry.onnxSupportDirectory(modelURL: flatWeight, modelRoot: flat).path,
            flat.path)

        // tokenizer 也放在 onnx/ 子目录的布局要能继续用。
        let nestedTokenizer = temporaryRoot.appendingPathComponent("onnx-tokenizer", isDirectory: true)
        let nestedOnnx = nestedTokenizer.appendingPathComponent("onnx", isDirectory: true)
        try FileManager.default.createDirectory(at: nestedOnnx, withIntermediateDirectories: true)
        let nestedOnnxWeight = nestedOnnx.appendingPathComponent("model_quantized.onnx")
        FileManager.default.createFile(
            atPath: nestedOnnxWeight.path, contents: Data(repeating: 1, count: 64))
        FileManager.default.createFile(
            atPath: nestedOnnx.appendingPathComponent("tokenizer.json").path,
            contents: Data("{}".utf8))
        XCTAssertEqual(
            LocalModelRegistry.onnxSupportDirectory(
                modelURL: nestedOnnxWeight, modelRoot: nestedTokenizer).path,
            nestedOnnx.path)

        // 显式传单个 .onnx 文件：root 归一到其父目录。
        XCTAssertEqual(
            LocalModelRegistry.onnxSupportDirectory(
                modelURL: flatWeight, modelRoot: flatWeight).path,
            flat.path)
    }

    /// 打包脚本必须复制 reranker 的全部运行时文件，避免发布包缺权重。
    func testPackageScriptBundlesRerankerRuntimeFiles() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let script = try String(
            contentsOf: repoRoot.appendingPathComponent("scripts/package-app.sh"),
            encoding: .utf8)

        XCTAssertTrue(
            script.contains("RERANKER_NAME=\"\(Self.modelName)\""),
            "scripts/package-app.sh 未声明内置 reranker 目录")
        for file in [
            "model_quantized.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config.json",
        ] {
            XCTAssertTrue(
                script.contains(file),
                "scripts/package-app.sh 未复制 reranker 运行时文件：\(file)")
        }
    }

    /// 模拟 App bundle 的 Contents/Resources/models，默认配置应命中内置 reranker。
    func testDefaultConfigResolvesBundledReranker() throws {
        let (modelsRoot, directory) = try makeFakeBundledReranker()
        setenv("BASE_AGENT_MODELS_DIR", modelsRoot.path, 1)

        let config = AgentConfig()
        let resolved = try LocalModelRegistry.resolveDirectory(
            kind: .reranker,
            value: config.localRerankerModelPath,
            config: config)
        XCTAssertEqual(resolved.standardizedFileURL.path,
                       directory.standardizedFileURL.path)

        let provider = try XCTUnwrap(
            LocalModelFactory.shared.makeReranker(config: config))
        let onnx = try XCTUnwrap(provider as? ONNXReranker,
                                 "默认配置应命中内置 ONNX reranker：\(provider)")
        XCTAssertEqual(onnx.directory.standardizedFileURL.path,
                       directory.standardizedFileURL.path)
    }

    /// 用户完全没有填写 reranker 路径/模型名时，应回落到内置默认模型。
    func testEmptyRerankerConfigUsesBundledDefault() throws {
        let (modelsRoot, directory) = try makeFakeBundledReranker()
        setenv("BASE_AGENT_MODELS_DIR", modelsRoot.path, 1)

        var config = AgentConfig()
        config.localRerankerModel = ""
        config.localRerankerModelPath = ""
        config.ragRerankModel = ""

        XCTAssertEqual(LocalModelRegistry.defaultName(for: .reranker), Self.modelName)
        let resolved = try LocalModelRegistry.resolveDirectory(
            kind: .reranker, value: "", config: config)
        XCTAssertEqual(resolved.standardizedFileURL.path,
                       directory.standardizedFileURL.path)

        let provider = try XCTUnwrap(
            LocalModelFactory.shared.makeReranker(config: config))
        XCTAssertTrue(provider is ONNXReranker, "空配置应回落到内置 reranker：\(provider)")
    }

    private func makeFakeBundledReranker() throws -> (modelsRoot: URL, directory: URL) {
        let modelsRoot = temporaryRoot.appendingPathComponent(
            "Contents/Resources/models", isDirectory: true)
        let directory = modelsRoot.appendingPathComponent(Self.modelName, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let files = [
            "model_quantized.onnx": "fake onnx",
            "tokenizer.json": "{}",
            "tokenizer_config.json": "{}",
            "special_tokens_map.json": "{}",
            "config.json": "{}",
        ]
        for (name, content) in files {
            try Data(content.utf8).write(to: directory.appendingPathComponent(name))
        }
        return (modelsRoot, directory)
    }
}
