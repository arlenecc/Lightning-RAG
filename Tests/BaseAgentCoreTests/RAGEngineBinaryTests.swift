import XCTest
@testable import BaseAgentCore

private func fakeVector(_ text: String) -> [Float] {
    var vector = [Float](repeating: 0, count: 8)
    for (index, scalar) in text.unicodeScalars.enumerated() {
        vector[(Int(scalar.value) + index) % 8] += 1
    }
    return VectorCodec.normalize(vector)
}

/// 引擎与二值粗筛的集成：入库签名、查询签名、状态五态与 bin 主存储旁路。
final class RAGEngineBinaryTests: XCTestCase {
    private var root: URL!
    private var kb: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("binary-engine-\(UUID().uuidString)")
        kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private final class SigningEmbedding: EmbeddingProvider, EmbeddingBinarySigning {
        let identifier = "binary-engine-fake"
        let dimension = 8

        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map(fakeVector)
        }

        func embedQuery(_ text: String) async throws -> [Float] {
            fakeVector(text)
        }
    }

    private final class PlainEmbedding: EmbeddingProvider {
        let identifier = "plain-engine-fake"
        let dimension = 8

        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map(fakeVector)
        }

        func embedQuery(_ text: String) async throws -> [Float] {
            fakeVector(text)
        }
    }

    private func makeConfig(_ adjust: (inout AgentConfig) -> Void = { _ in }) -> AgentConfig {
        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 5
        config.ragVectorElementType = .int8
        adjust(&config)
        return config
    }

    private func writeDocs(_ count: Int) throws {
        for index in 1...count {
            let text = """
            # 文档 \(index)

            Apple banana orange fruit \(index). 苹果 香蕉 橙子 水果 \(index)。
            涡轮 发动机 机械 设备 编号 \(index)。turbine engine machine \(index)。
            Knowledge retrieval pipeline stores vectors for semantic search.
            The quick brown fox jumps over the lazy dog and keeps running.
            """
            try text.write(to: kb.appendingPathComponent("doc\(index).md"),
                           atomically: true, encoding: .utf8)
        }
    }

    private func makeEngine(_ config: AgentConfig,
                            embedding: EmbeddingProvider) -> RAGEngine {
        RAGEngine(config: config, knowledgeBase: kb.path,
                  embedding: embedding, reranker: nil)
    }

    func testIngestWritesBinarySignaturesAndStatusReady() async throws {
        try writeDocs(2)
        let engine = makeEngine(makeConfig(), embedding: SigningEmbedding())
        let stats = await engine.ingest()
        XCTAssertGreaterThan(stats.chunks, 0)

        let status = engine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(status["binary_coarse"]?.stringValue, "ready")
        XCTAssertEqual(status["binary_dimension"]?.intValue, 8)
        XCTAssertEqual(status["binary_rows"]?.intValue,
                       status["chunks_stored"]?.intValue)
        XCTAssertEqual(status["binary_coverage"]?.doubleValue ?? 0, 1.0,
                       accuracy: 0.001)
        XCTAssertEqual(status["binary_backend"]?.stringValue, "sqlite-vec")
        XCTAssertEqual(status["ann_bypassed"]?.boolValue, true)
        XCTAssertEqual(status["binary_note"]?.stringValue, "")

        let binary = try XCTUnwrap(engine.binaryIndexStatus())
        XCTAssertTrue(binary.storageEnabled)
        XCTAssertTrue(binary.isComplete)
        XCTAssertEqual(binary.dimension, 8)
        engine.close()
    }

    func testSearchUsesBinaryQueryAndPopulatesBinaryScore() async throws {
        try writeDocs(2)
        let engine = makeEngine(makeConfig(), embedding: SigningEmbedding())
        _ = await engine.ingest()
        let hits = try await engine.search(query: "apple fruit turbine", topK: 3)
        XCTAssertFalse(hits.isEmpty)
        XCTAssertTrue(hits.contains { $0.binaryScore != nil },
                      "开启二值粗筛时向量候选应带汉明相似度")
        engine.close()
    }

    /// 开关关闭只影响检索：入库仍维护签名，查询不再产生 binaryScore。
    func testDisabledBinaryCoarseStillWritesSignatures() async throws {
        try writeDocs(2)
        let config = makeConfig { $0.ragBinaryCoarseEnabled = false }
        let engine = makeEngine(config, embedding: SigningEmbedding())
        _ = await engine.ingest()

        let status = engine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(status["binary_coarse"]?.stringValue, "off")
        XCTAssertEqual(status["binary_rows"]?.intValue,
                       status["chunks_stored"]?.intValue)
        XCTAssertEqual(status["ann_bypassed"]?.boolValue, false)

        let hits = try await engine.search(query: "apple fruit turbine", topK: 3)
        XCTAssertFalse(hits.isEmpty)
        XCTAssertTrue(hits.allSatisfy { $0.binaryScore == nil })
        engine.close()
    }

    func testPlainProviderReportsUnsupportedWithoutBinaryRows() async throws {
        try writeDocs(1)
        let engine = makeEngine(makeConfig(), embedding: PlainEmbedding())
        _ = await engine.ingest()

        let status = engine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(status["binary_coarse"]?.stringValue, "unsupported")
        XCTAssertEqual(status["binary_rows"]?.intValue, 0)
        XCTAssertTrue(status["binary_note"]?.stringValue?.contains("不支持") == true,
                      status["binary_note"]?.stringValue ?? "")
        XCTAssertEqual(status["ann_bypassed"]?.boolValue, false)

        let binary = try XCTUnwrap(engine.binaryIndexStatus())
        XCTAssertFalse(binary.storageEnabled)
        engine.close()
    }

    func testBinPrimaryStorageBypassesBinaryCoarse() async throws {
        try writeDocs(1)
        let config = makeConfig { $0.ragVectorElementType = .bin }
        let engine = makeEngine(config, embedding: SigningEmbedding())
        _ = await engine.ingest()

        let status = engine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(status["binary_coarse"]?.stringValue, "bypassed_bin")
        XCTAssertTrue(status["binary_note"]?.stringValue?.contains("bin 主存储") == true,
                      status["binary_note"]?.stringValue ?? "")
        XCTAssertEqual(status["binary_rows"]?.intValue, 0)
        XCTAssertEqual(status["ann_bypassed"]?.boolValue, false)
        engine.close()
    }
}
