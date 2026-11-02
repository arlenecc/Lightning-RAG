import XCTest
@testable import BaseAgentCore

/// HF 镜像模型下载：文件过滤、必需模型清单解析、（门控的）真实下载。
final class HFModelDownloaderTests: XCTestCase {

    func testNeededFilePredicate() {
        // 权重与配置 / tokenizer 必须拉取。
        XCTAssertTrue(HFModelDownloader.isNeededFile("config.json"))
        XCTAssertTrue(HFModelDownloader.isNeededFile("model.safetensors"))
        XCTAssertTrue(HFModelDownloader.isNeededFile("weights.safetensors"))
        XCTAssertTrue(HFModelDownloader.isNeededFile("sub/dir/model_quantized.onnx"))
        XCTAssertTrue(HFModelDownloader.isNeededFile("chat_template.jinja"))
        XCTAssertTrue(HFModelDownloader.isNeededFile("tokenizer.model"))
        XCTAssertTrue(HFModelDownloader.isNeededFile("merges.txt"))
        XCTAssertTrue(HFModelDownloader.isNeededFile("model.gguf"))
        // README / 脚本 / 隐藏文件不拉取。
        XCTAssertFalse(HFModelDownloader.isNeededFile("README.md"))
        XCTAssertFalse(HFModelDownloader.isNeededFile("example.py"))
        XCTAssertFalse(HFModelDownloader.isNeededFile(".gitattributes"))
    }

    func testCatalogResolvesDefaultModels() {
        // 把应用模型目录锚定到一个空目录：默认 "models/..." 值在该目录下
        // 都不存在（判定应全部缺失），同时避免测试进程解析到项目根的
        // 真实 models 目录。
        let anchor = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("hf-anchor-\(UUID().uuidString)", isDirectory: true)
        try! FileManager.default.createDirectory(at: anchor, withIntermediateDirectories: true)
        setenv("BASE_AGENT_MODELS_DIR", anchor.path, 1)
        defer {
            unsetenv("BASE_AGENT_MODELS_DIR")
            try? FileManager.default.removeItem(at: anchor)
        }

        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory()
            + "/hf-catalog-\(UUID().uuidString)"
        defer { try? FileManager.default.removeItem(atPath: config.workspace) }

        let repos = RequiredModelCatalog.models(for: config).map(\.repo)
        // 默认配置的主模型 / Embedding / Reranker / OCR 都应指向 arlenecc 仓库。
        XCTAssertTrue(repos.contains("arlenecc/qwen3.5-4b-agentic-coder-v4-4bit-mlx"))
        XCTAssertTrue(repos.contains("arlenecc/jina-embeddings-v5-text-nano-mlx-int4"))
        XCTAssertTrue(repos.contains("arlenecc/jina-reranker-v2-base-multilingual-onnx"))
        XCTAssertTrue(repos.contains("arlenecc/OvisOCR3-MLX-4bit-mixed46"))
        // 辅助模型默认关闭，不应出现。
        XCTAssertFalse(repos.contains("arlenecc/Qwen3.5-2B-MLX-4bit"))

        // 其余模型目录都为空 → 全部缺失；语义切片模型在开发/测试环境可能
        // 命中源码资源 bundle（视为就绪），按其就绪状态折算期望值。
        let all = RequiredModelCatalog.models(for: config)
        let missing = RequiredModelCatalog.missingModels(for: config)
        let potion = try! XCTUnwrap(all.first { $0.kind == "语义切片" })
        XCTAssertEqual(potion.repo, "arlenecc/potion-multilingual-128M-onnx")
        XCTAssertEqual(potion.subpath, "int4")
        let potionReady = RequiredModelCatalog.isComplete(potion.directory)
        XCTAssertEqual(missing.count, all.count - (potionReady ? 1 : 0))

        // 目录里出现任一权重文件即视为就绪（与 registry 同源判定）。
        let embedding = try! XCTUnwrap(
            RequiredModelCatalog.models(for: config).first { $0.kind == "Embedding" })
        try! FileManager.default.createDirectory(
            at: embedding.directory, withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: embedding.directory.appendingPathComponent("model.safetensors").path,
            contents: Data([1]))
        let after = RequiredModelCatalog.missingModels(for: config)
        XCTAssertFalse(after.contains { $0.kind == "Embedding" })
        XCTAssertEqual(after.count, missing.count - 1)
    }

    /// lite 打包把模型放进 App bundle 的 `Contents/Resources/models`：
    /// 判定必须能识别 bundle 内模型（应用模型目录锚定含 bundle），不能误报缺失。
    func testCatalogFindsBundledModels() {
        // 用临时目录模拟 bundle 布局：bundleURL/Contents/Resources/models/<name>。
        let bundleRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("hf-bundle-\(UUID().uuidString)", isDirectory: true)
        let resourcesModels = bundleRoot
            .appendingPathComponent("Contents/Resources/models", isDirectory: true)
        let embeddingDir = resourcesModels
            .appendingPathComponent("jina-embeddings-v5-text-nano-mlx-int4", isDirectory: true)
        try! FileManager.default.createDirectory(at: embeddingDir, withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: embeddingDir.appendingPathComponent("model.safetensors").path,
            contents: Data([1]))
        defer { try? FileManager.default.removeItem(at: bundleRoot) }

        // bundledModelsDirectory 参数化接口本身就是 bundle 搜索的实现，
        // 这里验证它能定位到模拟布局，并让 resolveDirectory 命中。
        let found = LocalModelRegistry.bundledModelsDirectory(
            resourceURL: resourcesModels.deletingLastPathComponent(),
            bundleURL: bundleRoot)
        XCTAssertEqual(found?.path, resourcesModels.path)

        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory()
            + "/hf-bundle-ws-\(UUID().uuidString)"
        defer { try? FileManager.default.removeItem(atPath: config.workspace) }
        config.localEmbeddingModelPath = resourcesModels
            .appendingPathComponent("jina-embeddings-v5-text-nano-mlx-int4").path
        // 显式路径指向 bundle 内已就绪模型 → 不得判缺失。
        XCTAssertFalse(RequiredModelCatalog.missingModels(for: config)
            .contains { $0.kind == "Embedding" })
    }

    /// 安装到 /Applications 等应用目录不可写时：下载落到配置目录
    /// `~/.base-agent/models`，且加载侧能在同位置回退找到。
    func testDownloadFallsBackToConfigModelsWhenAppModelsReadOnly() throws {
        // 应用 models 目录存在但只读，模拟装到 /Applications 的 bundle。
        let readOnly = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("hf-ro-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: readOnly, withIntermediateDirectories: true)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o555], ofItemAtPath: readOnly.path)
        setenv("BASE_AGENT_MODELS_DIR", readOnly.path, 1)
        defer {
            unsetenv("BASE_AGENT_MODELS_DIR")
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o755], ofItemAtPath: readOnly.path)
            try? FileManager.default.removeItem(at: readOnly)
        }

        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory()
            + "/hf-ro-ws-\(UUID().uuidString)"
        defer { try? FileManager.default.removeItem(atPath: config.workspace) }

        // 下载目标 = ~/.base-agent/models/<name>（只断言路径，不实际写 home）。
        let dir = LocalModelRegistry.downloadDirectory(
            kind: .llm, value: "models/test-download-model", config: config)
        let expected = (("~/.base-agent/models/test-download-model") as NSString)
            .expandingTildeInPath
        XCTAssertEqual(dir.standardizedFileURL.path,
                       URL(fileURLWithPath: expected).standardizedFileURL.path)

        // 加载侧回退：把就绪模型放进配置目录 models 后，resolveDirectory
        // 对 "models/<name>" 值必须能找到它。
        let name = "test-cfg-model-\(UUID().uuidString)"
        let cfgModelDir = LocalModelRegistry.configModelsDirectory()
            .appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: cfgModelDir, withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: cfgModelDir.appendingPathComponent("model.safetensors").path,
            contents: Data([1]))
        defer { try? FileManager.default.removeItem(at: cfgModelDir) }

        let resolved = try LocalModelRegistry.resolveDirectory(
            kind: .llm, value: "models/\(name)", config: config)
        XCTAssertEqual(resolved.standardizedFileURL.path,
                       cfgModelDir.standardizedFileURL.path)
        // 判定同步：不判缺失。
        XCTAssertFalse(RequiredModelCatalog.missingModels(for: config)
            .contains { $0.repo == "arlenecc/\(name)" })
    }

    /// 用户自配模型的各种形态都不得被误判为缺失：
    /// 显式单权重文件、onnx/ 子目录权重、分片 safetensors、仅含权重的目录。
    func testCatalogRecognizesUserConfiguredModels() {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("hf-user-models-\(UUID().uuidString)", isDirectory: true)
        try! FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        var config = AgentConfig()
        config.workspace = root.path

        // 1) 显式指定单个 .onnx 权重文件（registry 直接返回文件 URL）。
        let onnxFile = root.appendingPathComponent("my-reranker.onnx")
        FileManager.default.createFile(atPath: onnxFile.path, contents: Data([1]))
        config.localRerankerModelPath = onnxFile.path
        config.localRerankerBackend = "onnx"
        XCTAssertFalse(RequiredModelCatalog.missingModels(for: config)
            .contains { $0.kind == "Reranker" },
                       "显式指定的 .onnx 权重文件不应判缺失")

        // 2) 权重位于 onnx/ 子目录（registry 的候选搜索范围）。
        let onnxDir = root.appendingPathComponent("onnx-model", isDirectory: true)
        try! FileManager.default.createDirectory(
            at: onnxDir.appendingPathComponent("onnx", isDirectory: true),
            withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: onnxDir.appendingPathComponent("onnx/model_quantized.onnx").path,
            contents: Data([1]))
        config.localEmbeddingModelPath = onnxDir.path
        config.localEmbeddingBackend = "onnx"
        XCTAssertFalse(RequiredModelCatalog.missingModels(for: config)
            .contains { $0.kind == "Embedding" },
                       "onnx/ 子目录内的权重应判就绪")

        // 3) 分片 safetensors（无 model.safetensors 单文件）。
        let shardedDir = root.appendingPathComponent("sharded-mlx", isDirectory: true)
        try! FileManager.default.createDirectory(at: shardedDir, withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: shardedDir.appendingPathComponent("model-00001-of-00002.safetensors").path,
            contents: Data([1]))
        FileManager.default.createFile(
            atPath: shardedDir.appendingPathComponent("model.safetensors.index.json").path,
            contents: Data([1]))
        XCTAssertTrue(RequiredModelCatalog.isComplete(shardedDir))

        // 4) 显式路径为空目录（用户预创建）→ 仍判缺失，下载补到该路径。
        let emptyDir = root.appendingPathComponent("empty-model", isDirectory: true)
        try! FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)
        config.localRerankerModelPath = emptyDir.path
        XCTAssertTrue(RequiredModelCatalog.missingModels(for: config)
            .contains { $0.kind == "Reranker" })
        // 下载目标应就是用户指定的空目录。
        let reranker = try! XCTUnwrap(
            RequiredModelCatalog.models(for: config).first { $0.kind == "Reranker" })
        XCTAssertEqual(reranker.directory.standardizedFileURL.path,
                       emptyDir.standardizedFileURL.path)
    }

    func testCatalogSkipsAPIMainModel() {
        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory()
            + "/hf-catalog-api-\(UUID().uuidString)"
        defer { try? FileManager.default.removeItem(atPath: config.workspace) }
        config.llmProvider = "api"
        let kinds = RequiredModelCatalog.models(for: config).map(\.kind)
        XCTAssertFalse(kinds.contains("主模型"))
        // 本地 Embedding / Reranker / OCR 仍需要。
        XCTAssertTrue(kinds.contains("Embedding"))
        XCTAssertTrue(kinds.contains("Reranker"))
    }

    /// clean 版语义切片模型走 arlenecc/potion-multilingual-128M-onnx 的
    /// int4/ 子目录：scopedFiles 只取该档位并剥前缀，fp16/int8 不进清单。
    func testPotionScopedFilesStripQuantumPrefix() {
        let model = RequiredModel(
            kind: "语义切片",
            repo: "arlenecc/potion-multilingual-128M-onnx",
            directory: URL(fileURLWithPath: "/tmp/potion"),
            subpath: "int4")
        let files: [HFModelDownloader.FileEntry] = [
            .init(path: "int4/model.onnx", size: 80_066_323),
            .init(path: "int4/tokenizer.json", size: 18_616_131),
            .init(path: "int4/config.json", size: 922),
            .init(path: "fp16/model.onnx", size: 256_182_971),
            .init(path: "int8/model.onnx", size: 136_100_744),
            .init(path: "int4/example.py", size: 8_393),
            .init(path: "README.md", size: 11_937),
        ]
        let scoped = RequiredModelCatalog.scopedFiles(for: model, in: files)
        XCTAssertEqual(scoped.map(\.path),
                       ["model.onnx", "tokenizer.json", "config.json"])

        // 无子目录的普通模型：原样返回，不做任何过滤。
        let plain = RequiredModel(
            kind: "Embedding", repo: "arlenecc/some-model",
            directory: URL(fileURLWithPath: "/tmp/some"), subpath: nil)
        XCTAssertEqual(
            RequiredModelCatalog.scopedFiles(for: plain, in: files).count,
            files.count)
    }

    /// 语义切片条目跟随 rag_semantic_chunking 开关：关闭时不出现在清单里。
    func testCatalogOmitsPotionWhenSemanticChunkingDisabled() {
        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory()
            + "/hf-potion-off-\(UUID().uuidString)"
        defer { try? FileManager.default.removeItem(atPath: config.workspace) }
        config.ragSemanticChunking = false
        XCTAssertFalse(RequiredModelCatalog.models(for: config)
            .contains { $0.kind == "语义切片" })
    }

    /// 真实网络下载与镜像连通性验证：需要设置 HF_MIRROR_LIVE_TEST=1 才运行，
    /// 避免常规测试依赖外网。
    func testLiveMirrorListingAndDownload() async throws {
        guard ProcessInfo.processInfo.environment["HF_MIRROR_LIVE_TEST"] == "1" else {
            throw XCTSkip("设置 HF_MIRROR_LIVE_TEST=1 才运行真实网络下载测试")
        }
        let repo = "arlenecc/jina-embeddings-v5-text-nano-mlx-int4"
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("hf-live-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        let downloader = HFModelDownloader()
        let files = try await downloader.listNeededFiles(repo: repo)
        XCTAssertTrue(files.contains { $0.path == "config.json" })
        XCTAssertTrue(files.contains { $0.path == "model.safetensors" })
        // README 不在拉取清单里。
        XCTAssertFalse(files.contains { $0.path == "README.md" })

        let config = try XCTUnwrap(files.first { $0.path == "config.json" })
        var progressCalls = 0
        try await downloader.download(repo: repo, file: config, into: dir) { _, _ in
            progressCalls += 1
        }
        let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        XCTAssertFalse(data.isEmpty)
        // 再下载一次应直接跳过（文件已完整）。
        try await downloader.download(repo: repo, file: config, into: dir) { _, _ in }
        XCTAssertTrue(progressCalls > 0)
    }
}
