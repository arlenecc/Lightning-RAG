import XCTest
import BaseAgentCore

/// 真实下载并应用模型的全流程验证：
/// hf-mirror 列文件 → 带断点续传的下载 → 落盘到 models 目录 →
/// RequiredModelCatalog 判定就绪 → LocalModelFactory 加载并实际推理。
/// 需要外网，设置 HF_MIRROR_LIVE_TEST=1 才运行。
final class HFModelDownloadApplyTests: XCTestCase {

    /// 纯色测试不需要真实下载；这里下载 embedding（约 132MB 权重），
    /// 下载完成后必须能被本地 MLX 后端加载并产出向量。
    func testDownloadEmbeddingModelAndApplyIt() async throws {
        guard ProcessInfo.processInfo.environment["HF_MIRROR_LIVE_TEST"] == "1" else {
            throw XCTSkip("设置 HF_MIRROR_LIVE_TEST=1 才运行真实下载测试")
        }
        let env = ProcessInfo.processInfo.environment
        let repo = env["HF_APPLY_TEST_REPO"]
            ?? "arlenecc/jina-embeddings-v5-text-nano-mlx-int4"
        let modelName = (repo as NSString).lastPathComponent

        // 1) 构造一个空 workspace：默认配置下 embedding 目录缺失。
        let workspace = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("hf-apply-\(UUID().uuidString)")
        let modelDir = workspace
            .appendingPathComponent("models")
            .appendingPathComponent(modelName)
        defer { try? FileManager.default.removeItem(at: workspace) }

        var config = AgentConfig()
        config.workspace = workspace.path
        config.localEmbeddingBackend = "mlx"
        config.localEmbeddingModelPath = modelDir.path

        let missingBefore = RequiredModelCatalog.missingModels(for: config)
        XCTAssertTrue(missingBefore.contains { $0.repo == repo },
                      "下载前 embedding 目录应缺失")

        let downloader = HFModelDownloader()
        let files = try await downloader.listNeededFiles(repo: repo)
        let weights = try XCTUnwrap(files.first { $0.path == "model.safetensors" })
        XCTAssertGreaterThan(weights.size, 50_000_000, "权重文件应存在且有实际体积")

        // 2) 断点续传：下载权重中途主动取消，验证 .part 残留与偏移量。
        let partial = modelDir.appendingPathComponent("model.safetensors.part")
        do {
            var cancelled = false
            let cancelLock = NSLock()
            _ = try await downloader.download(
                repo: repo, file: weights, into: modelDir
            ) { received, _ in
                cancelLock.lock()
                let shouldCancel = !cancelled && received > 5_000_000
                if shouldCancel { cancelled = true }
                cancelLock.unlock()
                if shouldCancel { downloader.cancelCurrent() }
            }
            XCTFail("中途取消不应正常完成")
        } catch is CancellationError {
            // 预期路径
        }
        let partialSize = HFModelDownloader.fileSize(at: partial)
        XCTAssertGreaterThan(partialSize, 1_000_000,
                             "取消后应保留已下载的断点（.part）")
        XCTAssertLessThan(partialSize, weights.size, "断点应小于完整文件")

        // 3) 续传剩余部分并补齐全部所需文件；进度回调必须发生。
        var progressCalls = 0
        for file in files {
            try await downloader.download(repo: repo, file: file, into: modelDir) { _, _ in
                progressCalls += 1
            }
        }
        XCTAssertGreaterThan(progressCalls, 0)
        // 续传后完整落位：.part 消失，最终大小与远端清单一致。
        XCTAssertFalse(FileManager.default.fileExists(atPath: partial.path),
                       "下载完成后 .part 应被移除")
        XCTAssertEqual(HFModelDownloader.fileSize(
            at: modelDir.appendingPathComponent(weights.path)), weights.size)

        // 4) 目录就绪判定翻转：不再出现在缺失清单里。
        let missingAfter = RequiredModelCatalog.missingModels(for: config)
        XCTAssertFalse(missingAfter.contains { $0.repo == repo },
                       "下载完成后 embedding 目录应判定就绪")

        // 5) 应用：本地 MLX 后端加载刚下载的模型并实际产出向量。
        let provider = try XCTUnwrap(
            LocalModelFactory.shared.makeEmbedding(config: config),
            "下载完成的模型应能被工厂构建为可用 provider")
        let vectors = try await provider.embedQuery("你好，世界")
        XCTAssertFalse(vectors.isEmpty, "embedding 返回空向量")
        XCTAssertGreaterThanOrEqual(vectors.count, 64)
        for value in vectors {
            XCTAssertTrue(value.isFinite, "embedding 含非有限值")
        }
        let norm = vectors.reduce(0) { $0 + Double($1) * Double($1) }.squareRoot()
        XCTAssertGreaterThan(norm, 0.01, "embedding 范数异常：\(norm)")
    }
}
