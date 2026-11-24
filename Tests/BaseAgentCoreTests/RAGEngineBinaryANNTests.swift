import XCTest
@testable import BaseAgentCore

final class RAGEngineBinaryANNTests: XCTestCase {
    private var root: URL!
    private var kb: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("binary-ann-engine-\(UUID().uuidString)")
        kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    private final class SigningEmbedding: EmbeddingProvider, EmbeddingBinarySigning {
        let identifier = "binary-ann-fake"
        let dimension = 8
        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map { text in
                var value = [Float](repeating: 0, count: 8)
                for (index, scalar) in text.unicodeScalars.enumerated() {
                    value[(Int(scalar.value) + index) % 8] += 1
                }
                return VectorCodec.normalize(value)
            }
        }
        func embedQuery(_ text: String) async throws -> [Float] {
            try await embedDocuments([text])[0]
        }
    }

    private func writeDocs(_ count: Int) throws {
        for index in 1...count {
            let text = String(repeating: "apple banana turbine engine \(index) ", count: 20)
                + "\nKnowledge retrieval pipeline stores vectors for semantic search."
            try text.write(to: kb.appendingPathComponent("doc\(index).md"),
                           atomically: true, encoding: .utf8)
        }
    }

    private func makeEngine(_ adjust: (inout AgentConfig) -> Void = { _ in }) -> RAGEngine {
        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 5
        config.ragVectorElementType = .int8
        config.ragANNMinChunks = 1
        config.ragANNClusters = 4
        config.ragANNClusterSize = 8
        adjust(&config)
        return RAGEngine(config: config, knowledgeBase: kb.path,
                         embedding: SigningEmbedding(), reranker: nil)
    }

    func testBuildTargetsBinaryWhenCoverageComplete() async throws {
        try writeDocs(2)
        let engine = makeEngine()
        _ = await engine.ingest()
        // 二值覆盖判定需要全部签名回填完成（文本先行入库），先等回填结束。
        try await awaitVectorsFilled(engine)
        XCTAssertEqual(engine.resolvedANNIndexTarget(), .binary)
        let result = try XCTUnwrap(engine.buildANNIndex())
        XCTAssertTrue(result.summary.contains("二值 ANN"), result.summary)
        XCTAssertTrue(try XCTUnwrap(engine.binaryANNIndexStatus()).isReady)
        XCTAssertFalse(try XCTUnwrap(engine.annIndexStatus()).isReady)
        engine.close()
    }

    func testBuildFallsBackToDenseWhenBinaryDisabled() async throws {
        try writeDocs(2)
        let engine = makeEngine { $0.ragBinaryCoarseEnabled = false }
        _ = await engine.ingest()
        try await awaitVectorsFilled(engine)
        XCTAssertEqual(engine.resolvedANNIndexTarget(), .dense)
        _ = try XCTUnwrap(engine.buildANNIndex())
        XCTAssertTrue(try XCTUnwrap(engine.annIndexStatus()).isReady)
        XCTAssertFalse(try XCTUnwrap(engine.binaryANNIndexStatus()).isReady)
        engine.close()
    }

    func testStatusExposesBinaryANNFields() async throws {
        try writeDocs(2)
        let engine = makeEngine()
        _ = await engine.ingest()
        try await awaitVectorsFilled(engine)
        _ = try XCTUnwrap(engine.buildANNIndex())
        let status = engine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(status["binary_ann"]?.stringValue, "ready")
        XCTAssertEqual(status["binary_backend"]?.stringValue, "bann")
        XCTAssertEqual(status["stage2_backend"]?.stringValue, "rerank")
        XCTAssertEqual(status["binary_ann_building"]?.boolValue, false)
        engine.close()
    }

    func testDropRemovesBothIndexes() async throws {
        try writeDocs(2)
        let engine = makeEngine()
        _ = await engine.ingest()
        _ = try XCTUnwrap(engine.buildANNIndex())
        try engine.dropANNIndex()
        XCTAssertFalse(try XCTUnwrap(engine.binaryANNIndexStatus()).isReady)
        XCTAssertFalse(try XCTUnwrap(engine.annIndexStatus()).isReady)
        engine.close()
    }

    func testSearchUsesBinaryANN() async throws {
        try writeDocs(2)
        let engine = makeEngine()
        _ = await engine.ingest()
        try await awaitVectorsFilled(engine)
        _ = try XCTUnwrap(engine.buildANNIndex())
        let hits = try await engine.search(query: "apple turbine", topK: 3)
        XCTAssertFalse(hits.isEmpty)
        XCTAssertTrue(hits.contains { $0.binaryScore != nil })
        engine.close()
    }

    // MARK: 回填等待辅助

    /// 等待全部切片向量/签名回填完成（文本先行入库，二值覆盖与 ANN 断言需要完整数据）。
    private func awaitVectorsFilled(_ engine: RAGEngine) async throws {
        for _ in 0..<300 {
            if engine.vectorCompleteness().pending == 0 { return }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
    }
}
