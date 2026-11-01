import Foundation
import XCTest
@testable import BaseAgentCore

private final class PerformanceFakeEmbedding: EmbeddingProvider {
    let identifier = "perf-fake-embedding"
    let dimension = 8

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map { text in
            var vector = [Float](repeating: 0, count: dimension)
            for (index, scalar) in text.unicodeScalars.enumerated() {
                vector[Int(scalar.value + UInt32(index)) % dimension] += 1
            }
            return VectorCodec.normalize(vector)
        }
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        try await embedDocuments([text])[0]
    }
}

/// 记录每次 rerank 收到的候选数量，用来核对候选池倍数。
private final class CountingReranker: RerankProvider {
    let identifier = "perf-counting-rerank"
    private let lock = NSLock()
    private var counts: [Int] = []

    func rerank(query: String, documents: [String]) async throws -> [Double] {
        lock.lock()
        counts.append(documents.count)
        lock.unlock()
        return documents.enumerated().map { index, _ in 0.9 - Double(index) * 0.01 }
    }

    var documentCounts: [Int] {
        lock.lock(); defer { lock.unlock() }
        return counts
    }
}

/// 记录 preload 调用次数；`failing` 时模拟模型缺失。
private final class PreloadTrackingEmbedding: EmbeddingProvider {
    let identifier: String
    let dimension = 8
    let failing: Bool
    private let lock = NSLock()
    private var preloads = 0

    init(identifier: String = "perf-preload-embedding", failing: Bool = false) {
        self.identifier = identifier
        self.failing = failing
    }

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        try await PerformanceFakeEmbedding().embedDocuments(texts)
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        try await PerformanceFakeEmbedding().embedQuery(text)
    }

    func preload() async throws {
        if failing { throw AgentError.storage("embedding 预加载失败") }
        lock.lock()
        preloads += 1
        lock.unlock()
    }

    var preloadCount: Int {
        lock.lock(); defer { lock.unlock() }
        return preloads
    }
}

private final class PreloadTrackingReranker: RerankProvider {
    let identifier: String
    let failing: Bool
    private let lock = NSLock()
    private var preloads = 0

    init(identifier: String = "perf-preload-rerank", failing: Bool = false) {
        self.identifier = identifier
        self.failing = failing
    }

    func rerank(query: String, documents: [String]) async throws -> [Double] {
        documents.enumerated().map { index, _ in 0.9 - Double(index) * 0.01 }
    }

    func preload() async throws {
        if failing { throw AgentError.storage("reranker 预加载失败") }
        lock.lock()
        preloads += 1
        lock.unlock()
    }

    var preloadCount: Int {
        lock.lock(); defer { lock.unlock() }
        return preloads
    }
}

/// 模拟慢 reranker：可被取消，用来验证超时降级。
private final class SlowReranker: RerankProvider {
    let identifier = "perf-slow-rerank"
    let delayNanos: UInt64

    init(delaySeconds: Double) {
        delayNanos = UInt64(delaySeconds * 1_000_000_000)
    }

    func rerank(query: String, documents: [String]) async throws -> [Double] {
        try await Task.sleep(nanoseconds: delayNanos)
        return documents.enumerated().map { index, _ in 0.9 - Double(index) * 0.01 }
    }
}

final class RAGPerformanceTests: XCTestCase {
    private var root: URL!
    private var knowledgeBase: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-perf-\(UUID().uuidString)")
        knowledgeBase = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: knowledgeBase,
                                                withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func baseConfig() -> AgentConfig {
        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = knowledgeBase.path
        config.ragChunkSize = 40
        config.ragChunkOverlap = 5
        config.ragRerankThreshold = 0
        return config
    }

    /// 写入 `sections` 个小节，保证切片数明显多于 30。
    private func makeCorpus(sections: Int) throws {
        var markdown = "# 检索语料\n"
        for index in 1...sections {
            markdown += "\n## 片段\(index)\n\n这是第\(index)段用于验证候选池倍数的正文内容。\n"
        }
        try markdown.write(to: knowledgeBase.appendingPathComponent("corpus.md"),
                           atomically: true, encoding: .utf8)
    }

    // MARK: - 默认值

    func testPerformanceDefaults() {
        let config = AgentConfig()
        XCTAssertEqual(config.ragRerankTopN, 3)
        XCTAssertEqual(config.ragRerankCandidateMultiplier, 3)
        XCTAssertEqual(config.ragRerankTimeoutSeconds, 5)
    }

    /// 候选池倍数可配置，并持久化到配置文件、加载时夹到 1...5。
    func testRerankCandidateMultiplierPersistsAndClamps() throws {
        let path = root.appendingPathComponent("config.json").path
        var config = baseConfig()
        config.ragRerankCandidateMultiplier = 9
        try config.save(path)

        let loaded = AgentConfig.load(path)
        XCTAssertEqual(loaded.ragRerankCandidateMultiplier, 5,
                       "手改配置越界时应夹到 1...5")
    }

    // MARK: - 超时降级

    /// reranker 超过 5s（测试里用 0.2s）时必须立即返回 RRF 排序，
    /// 不能把整个检索拖到几十秒。
    func testRerankTimeoutFallsBackToRRF() async throws {
        try makeCorpus(sections: 20)

        var config = baseConfig()
        config.ragRerankEnabled = false
        let baseline = RAGEngine(config: config, knowledgeBase: knowledgeBase.path,
                                 embedding: PerformanceFakeEmbedding(), reranker: nil)
        _ = await baseline.ingest()
        let rrfHits = try await baseline.search(query: "片段", topK: 3)
        baseline.close()

        config.ragRerankEnabled = true
        config.ragRerankTopN = 3
        config.ragRerankTimeoutSeconds = 0.2
        let slow = RAGEngine(config: config, knowledgeBase: knowledgeBase.path,
                             embedding: PerformanceFakeEmbedding(),
                             reranker: SlowReranker(delaySeconds: 5))
        defer { slow.close() }

        let started = Date()
        let degraded = try await slow.search(query: "片段", topK: 10)
        XCTAssertLessThan(Date().timeIntervalSince(started), 2,
                          "超时后必须立即返回，不能等 reranker 跑完")
        XCTAssertEqual(degraded.count, 3)
        XCTAssertEqual(degraded.map(\.rowID), rrfHits.map(\.rowID),
                       "超时应回退到 RRF 排序")
    }

    /// TopN 调大后超时回退不能把整个候选池都返回：只取 RRF 最高的 3 条，
    /// 避免一次塞给会话过多上下文。
    func testRerankTimeoutFallbackCapsAtThreeResults() async throws {
        try makeCorpus(sections: 20)

        var config = baseConfig()
        config.ragRerankEnabled = false
        let baseline = RAGEngine(config: config, knowledgeBase: knowledgeBase.path,
                                 embedding: PerformanceFakeEmbedding(), reranker: nil)
        _ = await baseline.ingest()
        let rrfHits = try await baseline.search(query: "片段", topK: 3)
        baseline.close()

        config.ragRerankEnabled = true
        config.ragRerankTopN = 5
        config.ragRerankTimeoutSeconds = 0.2
        let slow = RAGEngine(config: config, knowledgeBase: knowledgeBase.path,
                             embedding: PerformanceFakeEmbedding(),
                             reranker: SlowReranker(delaySeconds: 5))
        defer { slow.close() }

        let degraded = try await slow.search(query: "片段", topK: 10)
        XCTAssertEqual(degraded.count, 3,
                       "超时回退只返回 RRF 最高的 3 条，不跟随 TopN=5")
        XCTAssertEqual(degraded.map(\.rowID), rrfHits.map(\.rowID),
                       "回退结果仍应是 RRF 排序的前三名")
    }

    // MARK: - 启动预加载

    /// 启动预加载必须把 embedding 与 reranker 的权重真正拉起来并常驻。
    func testPreloadWarmsEmbeddingAndReranker() async throws {
        var config = baseConfig()
        config.ragRerankEnabled = true
        let embedding = PreloadTrackingEmbedding()
        let reranker = PreloadTrackingReranker()
        let engine = RAGEngine(config: config, knowledgeBase: knowledgeBase.path,
                               embedding: embedding, reranker: reranker)
        defer { engine.close() }

        let outcome = await engine.preloadAuxiliaryModels()

        XCTAssertEqual(embedding.preloadCount, 1)
        XCTAssertEqual(reranker.preloadCount, 1)
        XCTAssertEqual(outcome.loaded, ["perf-preload-embedding", "perf-preload-rerank"])
        XCTAssertTrue(outcome.failures.isEmpty)
    }

    /// 关闭重排序时不应白占内存加载 reranker。
    func testPreloadSkipsRerankerWhenDisabled() async throws {
        var config = baseConfig()
        config.ragRerankEnabled = false
        let embedding = PreloadTrackingEmbedding()
        let reranker = PreloadTrackingReranker()
        let engine = RAGEngine(config: config, knowledgeBase: knowledgeBase.path,
                               embedding: embedding, reranker: reranker)
        defer { engine.close() }

        let outcome = await engine.preloadAuxiliaryModels()

        XCTAssertEqual(embedding.preloadCount, 1)
        XCTAssertEqual(reranker.preloadCount, 0)
        XCTAssertEqual(outcome.loaded, ["perf-preload-embedding"])
    }

    /// 预加载失败要带回可诊断信息，供启动日志展示，而不是静默吞掉。
    func testPreloadReportsFailures() async throws {
        var config = baseConfig()
        config.ragRerankEnabled = true
        let engine = RAGEngine(config: config, knowledgeBase: knowledgeBase.path,
                               embedding: PreloadTrackingEmbedding(failing: true),
                               reranker: PreloadTrackingReranker(failing: true))
        defer { engine.close() }

        let outcome = await engine.preloadAuxiliaryModels()

        XCTAssertTrue(outcome.loaded.isEmpty)
        XCTAssertEqual(outcome.failures.count, 2)
        XCTAssertTrue(outcome.failures[0].contains("perf-preload-embedding"))
        XCTAssertTrue(outcome.failures[1].contains("perf-preload-rerank"))
    }

    // MARK: - 候选池

    /// rerank 候选池 = max(调用方 top_k, TopN) × 3，而不是旧的 × 4。
    func testRerankCandidatePoolIsThreeTimesRequested() async throws {
        try makeCorpus(sections: 60)

        var config = baseConfig()
        config.ragRerankEnabled = true
        config.ragRerankTopN = 3
        let reranker = CountingReranker()
        let engine = RAGEngine(config: config, knowledgeBase: knowledgeBase.path,
                               embedding: PerformanceFakeEmbedding(), reranker: reranker)
        defer { engine.close() }

        let stats = await engine.ingest()
        XCTAssertGreaterThan(stats.chunks, 30, "语料需要足够多的切片")

        _ = try await engine.search(query: "片段", topK: 10)
        XCTAssertEqual(reranker.documentCounts, [30],
                       "top_k=10 + TopN=3 时候选池应为 30")
    }

    /// 面板上的候选池倍数生效：2 倍时 top_k=10 只送 20 条进 reranker。
    func testRerankCandidatePoolUsesConfiguredMultiplier() async throws {
        try makeCorpus(sections: 60)

        var config = baseConfig()
        config.ragRerankEnabled = true
        config.ragRerankTopN = 3
        config.ragRerankCandidateMultiplier = 2
        let reranker = CountingReranker()
        let engine = RAGEngine(config: config, knowledgeBase: knowledgeBase.path,
                               embedding: PerformanceFakeEmbedding(), reranker: reranker)
        defer { engine.close() }

        _ = await engine.ingest()
        _ = try await engine.search(query: "片段", topK: 10)
        XCTAssertEqual(reranker.documentCounts, [20],
                       "top_k=10 + 倍数 2 时候选池应为 20")
    }
}
