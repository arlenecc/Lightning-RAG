import XCTest
import Foundation
@testable import BaseAgentCore

/// 真实语料端到端同步基准（临时，不提交）。
///
/// 环境变量：
/// - `RAG_INGEST_BENCH_LIST`：文件名列表（每行一个），相对 `RAG_INGEST_BENCH_DIR`
/// - `RAG_INGEST_BENCH_DIR`：语料目录，默认 `/Volumes/SSD/book-clean`
/// - `RAG_INGEST_BENCH_LABEL`：结果标签（用于前后对比）
final class IngestBenchmarkTests: XCTestCase {
    func testRealCorpusIngest() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let listPath = env["RAG_INGEST_BENCH_LIST"] else {
            throw XCTSkip("未设置 RAG_INGEST_BENCH_LIST")
        }
        let corpusDir = env["RAG_INGEST_BENCH_DIR"] ?? "/Volumes/SSD/book-clean"
        let label = env["RAG_INGEST_BENCH_LABEL"] ?? "run"
        let names = try String(contentsOfFile: listPath, encoding: .utf8)
            .split(separator: "\n")
            .map { String($0).trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard !names.isEmpty else { throw XCTSkip("语料列表为空") }

        LocalModelFactory.shared.clearCaches()
        let modelsRoot = EmbeddingBenchmarkTests.resolveModelsRoot()
        setenv("BASE_AGENT_MODELS_DIR", modelsRoot.path, 1)
        EmbeddingBenchmarkTests.configureMetallib()

        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rag-ingest-bench-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        for name in names {
            let src = (corpusDir as NSString).appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: src) else {
                throw XCTSkip("语料缺失：\(src)")
            }
            let dst = (kb.path as NSString).appendingPathComponent(name)
            try? FileManager.default.removeItem(atPath: dst)
            // SSD 与临时目录不同卷，硬链接不可用；文件只有几百 KB，直接复制。
            try FileManager.default.copyItem(atPath: src, toPath: dst)
        }

        var config = AgentConfig()
        config.workspace = root.path
        config.ragANNEnabled = false
        config.ragANNAutoBuild = false
        config.ragRerankEnabled = false
        config.ragIngestConcurrency = 6
        let embedding = try XCTUnwrap(LocalModelFactory.shared.makeEmbedding(config: config))

        // 预热把模型加载/首次 kernel 初始化排除在同步计时之外。
        let warmStarted = Date()
        _ = try await embedding.embedDocuments(["预热文本：检索增强生成"])
        print(String(format: "[bench] warmup %.1fs", Date().timeIntervalSince(warmStarted)))

        let engine = RAGEngine(config: config, knowledgeBase: kb.path, embedding: embedding)
        defer { engine.close() }

        let printLock = NSLock()
        let outPath = env["RAG_INGEST_BENCH_OUT"] ?? "/tmp/rag_ingest_bench.log"
        FileManager.default.createFile(atPath: outPath, contents: nil)
        let outHandle = FileHandle(forWritingAtPath: outPath)
        defer { try? outHandle?.close() }
        func emit(_ line: String) {
            printLock.lock()
            print(line)
            try? outHandle?.write(contentsOf: Data((line + "\n").utf8))
            printLock.unlock()
        }

        let started = Date()
        let stats = await engine.ingest(force: true, log: { line in
            emit(String(format: "[bench +%.1fs] ", Date().timeIntervalSince(started)) + line)
        })
        let elapsed = Date().timeIntervalSince(started)
        emit(String(format: "[bench] RESULT label=%@ files=%d processed=%d chunks=%d "
            + "elapsed=%.2fs errors=%d",
            label, names.count, stats.processed, stats.chunks, elapsed, stats.errors.count))
        XCTAssertTrue(stats.errors.isEmpty, stats.errors.joined(separator: "; "))
        XCTAssertEqual(stats.processed, names.count)

        await embedding.unload()
        try? FileManager.default.removeItem(at: root)
    }
}
