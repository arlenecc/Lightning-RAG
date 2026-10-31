import XCTest
@testable import BaseAgentCore

private func toolFakeVector(_ text: String) -> [Float] {
    var vector = [Float](repeating: 0, count: 8)
    for (index, scalar) in text.unicodeScalars.enumerated() {
        vector[(Int(scalar.value) + index) % 8] += 1
    }
    return VectorCodec.normalize(vector)
}

/// `rag_status` 对二值粗筛五态的输出。
final class RAGToolBinaryStatusTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("binary-tool-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private final class SigningEmbedding: EmbeddingProvider, EmbeddingBinarySigning {
        let identifier = "binary-tool-fake"
        let dimension = 8

        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map(toolFakeVector)
        }

        func embedQuery(_ text: String) async throws -> [Float] {
            toolFakeVector(text)
        }
    }

    private final class PlainEmbedding: EmbeddingProvider {
        let identifier = "plain-tool-fake"
        let dimension = 8

        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map(toolFakeVector)
        }

        func embedQuery(_ text: String) async throws -> [Float] {
            toolFakeVector(text)
        }
    }

    private func writeDoc(to kb: URL) throws {
        let text = """
        # 文档

        Apple banana orange fruit. 苹果 香蕉 橙子 水果。
        Knowledge retrieval pipeline stores vectors for semantic search.
        The quick brown fox jumps over the lazy dog and keeps running.
        涡轮 发动机 机械 设备。turbine engine machine equipment.
        """
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        try text.write(to: kb.appendingPathComponent("doc.md"),
                       atomically: true, encoding: .utf8)
    }

    private func statusOutput(
        name: String,
        adjust: (inout AgentConfig) -> Void = { _ in },
        embedding: EmbeddingProvider
    ) async throws -> String {
        let workspace = root.appendingPathComponent(name)
        let kb = workspace.appendingPathComponent("kb")
        try writeDoc(to: kb)

        var config = AgentConfig()
        config.workspace = workspace.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 5
        config.ragVectorElementType = .int8
        adjust(&config)

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: embedding, reranker: nil)
        _ = await engine.ingest()
        _ = engine.status(sourceLimit: 0, maxAge: 0)
        let tool = RagStatusTool(config: engine.config, engine: engine)
        let result = try await tool.run([:])
        engine.close()
        XCTAssertTrue(result.success, result.error)
        return result.output
    }

    func testReadyOutputExplainsBinaryReplacesANN() async throws {
        let output = try await statusOutput(name: "ready",
                                            embedding: SigningEmbedding())
        XCTAssertTrue(
            output.contains("二值粗筛: 可用（8 bit，覆盖 100%，本次取代 ANN）"),
            output)
        XCTAssertTrue(output.contains("（本次由二值粗筛取代）"), output)
    }

    func testOffUnsupportedAndBypassedOutputs() async throws {
        let off = try await statusOutput(
            name: "off",
            adjust: { $0.ragBinaryCoarseEnabled = false },
            embedding: SigningEmbedding())
        XCTAssertTrue(off.contains("二值粗筛: 关闭"), off)

        let unsupported = try await statusOutput(
            name: "unsupported", embedding: PlainEmbedding())
        XCTAssertTrue(
            unsupported.contains("二值粗筛: 不可用（embedding 不支持二值签名）"),
            unsupported)

        let bypassed = try await statusOutput(
            name: "bypassed",
            adjust: { $0.ragVectorElementType = .bin },
            embedding: SigningEmbedding())
        XCTAssertTrue(
            bypassed.contains("二值粗筛: 停用（bin 主存储仅支持单阶段检索）"),
            bypassed)
    }
}
