import XCTest
import BaseAgentCore

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

        // 默认目录都为空 → 全部缺失。
        let missing = RequiredModelCatalog.missingModels(for: config)
        XCTAssertEqual(missing.count, RequiredModelCatalog.models(for: config).count)

        // 目录里出现任一权重文件即视为就绪。
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
