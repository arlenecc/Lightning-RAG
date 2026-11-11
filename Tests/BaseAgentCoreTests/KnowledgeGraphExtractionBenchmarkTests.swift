import Foundation
import XCTest
@testable import BaseAgentCore

/// 定量验证：抽取阶段耗时构成、缓存复用、并发收益。
private final class CountingLLM: LLMService {
    let identifier = "kg-bench"
    let supportsTools = false
    let contextWindow = 8192
    let supportsConcurrentIsolatedRequests: Bool
    private let extractionDelay: UInt64
    private let summaryDelay: UInt64
    private let lock = NSLock()
    private(set) var extractionCalls = 0
    private(set) var summaryCalls = 0

    init(concurrent: Bool, extractionDelayMS: UInt64 = 40, summaryDelayMS: UInt64 = 20) {
        self.supportsConcurrentIsolatedRequests = concurrent
        self.extractionDelay = extractionDelayMS * 1_000_000
        self.summaryDelay = summaryDelayMS * 1_000_000
    }

    func chatStream(messages: [ChatMessage], tools: [JSONValue],
                    temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        chatStream(messages: messages, tools: tools, temperature: temperature,
                   options: .default)
    }

    func chatStream(messages: [ChatMessage], tools: [JSONValue], temperature: Double,
                    options: LLMRequestOptions) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        let prompt = messages.last?.content ?? ""
        let answer = AsyncThrowingStream<LLMStreamEvent, Error> { continuation in
            Task {
                if prompt.contains("Extract entities and relationships") {
                    self.bumpExtraction()
                    try? await Task.sleep(nanoseconds: self.extractionDelay)
                    continuation.yield(LLMStreamEvent(kind: .content,
                                                      content: Self.extraction(prompt)))
                } else if prompt.contains("synthesize a list of descriptions") {
                    self.bumpSummary()
                    try? await Task.sleep(nanoseconds: self.summaryDelay)
                    continuation.yield(LLMStreamEvent(kind: .content,
                                                      content: "归纳后的描述"))
                } else if prompt.contains("expert keyword extractor") {
                    continuation.yield(LLMStreamEvent(
                        kind: .content,
                        content: "{\"high_level_keywords\":[\"topic\"],"
                            + "\"low_level_keywords\":[\"Topic1\"]}"))
                }
                continuation.yield(LLMStreamEvent(kind: .done))
                continuation.finish()
            }
        }
        return answer
    }

    private func bumpExtraction() {
        lock.lock(); extractionCalls += 1; lock.unlock()
    }

    private func bumpSummary() {
        lock.lock(); summaryCalls += 1; lock.unlock()
    }

    private static func extraction(_ prompt: String) -> String {
        let topic = prompt.range(of: "Topic[0-9]+", options: .regularExpression)
            .map { String(prompt[$0]) } ?? "Topic1"
        return """
        entity<|#|>\(topic)<|#|>Concept<|#|>\(topic) 是一个示例主题。
        entity<|#|>Shared A<|#|>Concept<|#|>Shared A 与 \(topic) 相关。
        entity<|#|>Shared B<|#|>Concept<|#|>Shared B 与 \(topic) 相关。
        entity<|#|>Shared C<|#|>Concept<|#|>Shared C 与 \(topic) 相关。
        relation<|#|>\(topic)<|#|>Shared A<|#|>包含<|#|>\(topic) 包含 Shared A。
        <|COMPLETE|>
        """
    }
}

private final class BenchEmbedding: EmbeddingProvider {
    let identifier = "kg-bench-embedding"
    let dimension = 64
    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map { _ in (0..<dimension).map { _ in Float.random(in: -1...1) } }
    }
    func embedQuery(_ text: String) async throws -> [Float] {
        (0..<dimension).map { _ in Float.random(in: -1...1) }
    }
}

final class KnowledgeGraphExtractionBenchmarkTests: XCTestCase {

    private func run(concurrent: Bool, documentCount: Int = 8)
        async throws -> (first: Double, second: Double, extraction: Int, summary: Int,
                         cacheHits: Int) {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("kg-bench-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        for i in 1...documentCount {
            try """
            # 主题 \(i)

            Topic\(i) 是一个用于性能测试的示例主题。Topic\(i) 与 Shared A、Shared B、
            Shared C 都存在关联。这里补充一些无关但足够长的正文，避免同步阶段因为
            正文过短而丢弃该文件。Topic\(i) 的描述会重复出现若干次，用于触发归纳。
            Topic\(i) 还包含更多细节内容，保证切片可以正常生成并进入图谱抽取流程。
            """.write(to: kb.appendingPathComponent("doc\(i).md"),
                      atomically: true, encoding: .utf8)
        }

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 1200
        config.ragChunkOverlap = 0

        let llm = CountingLLM(concurrent: concurrent)
        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: BenchEmbedding(), reranker: nil)
        _ = await engine.ingest()
        let t1 = Date()
        let first = await engine.buildKnowledgeGraph(auxLLM: llm)
        let firstSeconds = Date().timeIntervalSince(t1)
        let extractionAfterFirst = llm.extractionCalls
        let summaryAfterFirst = llm.summaryCalls

        // 第二次构建：正文不变，抽取与归纳都应命中缓存。
        let t2 = Date()
        let second = await engine.buildKnowledgeGraph(auxLLM: llm)
        let secondSeconds = Date().timeIntervalSince(t2)
        print("BENCH concurrent=\(concurrent) chunks=\(first.chunksProcessed) "
            + "first=\(String(format: "%.3f", firstSeconds))s "
            + "second=\(String(format: "%.3f", secondSeconds))s "
            + "extractionCalls=\(llm.extractionCalls) "
            + "summaryCalls=\(llm.summaryCalls) "
            + "secondCacheHits=\(second.cacheHits)")
        XCTAssertEqual(llm.extractionCalls, extractionAfterFirst)
        XCTAssertEqual(llm.summaryCalls, summaryAfterFirst)
        XCTAssertEqual(second.cacheHits, second.chunksProcessed)
        engine.close()
        return (firstSeconds, secondSeconds, llm.extractionCalls, llm.summaryCalls,
                second.cacheHits)
    }

    func testSerialBackendTiming() async throws {
        _ = try await run(concurrent: false)
    }

    func testConcurrentBackendTiming() async throws {
        _ = try await run(concurrent: true)
    }
}
