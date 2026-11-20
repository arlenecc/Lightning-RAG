import XCTest
@testable import BaseAgentCore

private final class StatusTestEmbedding: EmbeddingProvider {
    let identifier = "status-test"
    let dimension = 8
    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map { _ in [Float](repeating: 0.5, count: 8) }
    }
    func embedQuery(_ text: String) async throws -> [Float] { [Float](repeating: 0.5, count: 8) }
}

/// rag status 多分区只返回简洁概览，不拼默认库详情。
final class RagStatusMultiZoneTests: XCTestCase {
    private func makeEngine() -> RAGEngine {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("status-mz-\(UUID().uuidString)")
        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = root.path
        return RAGEngine(config: config, knowledgeBase: root.path,
                         embedding: StatusTestEmbedding(), reranker: nil)
    }

    func testMultiZoneReturnsOnlyOverview() async throws {
        let engine = makeEngine()
        defer { engine.close() }
        let tool = RagStatusTool(
            config: engine.config, engine: engine,
            zoneSummaryProvider: {
                [
                    (name: "社会", chunks: 553,
                     sources: ["地址的故事.md", "未来能源.md", "泥土.md"]),
                    (name: "自然", chunks: 687,
                     sources: ["寄生虫星球.md", "病毒星球.md"]),
                    (name: "历史", chunks: 435,
                     sources: ["伊朗四千年.md"]),
                ]
            })
        let result = try await tool.run([:])
        XCTAssertTrue(result.success)
        let out = result.output
        XCTAssertTrue(out.contains("3 个分区"), out)
        XCTAssertTrue(out.contains("合计：1675 切片 / 6 文档"), out)
        XCTAssertTrue(out.contains("【社会】553 切片"), out)
        XCTAssertTrue(out.contains("【自然】687 切片"), out)
        XCTAssertTrue(out.contains("【历史】435 切片"), out)
        XCTAssertTrue(out.contains("病毒星球.md"), out)
        // 不应有默认库详情字段
        XCTAssertFalse(out.contains("向量后端"), out)
        XCTAssertFalse(out.contains("二值粗筛"), out)
        XCTAssertFalse(out.contains("ANN 索引"), out)
        XCTAssertFalse(out.contains("RAG 目录"), out)
    }

    /// 单分区仍保留完整详情（不回归）。
    func testSingleZoneKeepsDetails() async throws {
        let engine = makeEngine()
        defer { engine.close() }
        let tool = RagStatusTool(config: engine.config, engine: engine,
                                 zoneSummaryProvider: nil)
        let result = try await tool.run([:])
        XCTAssertTrue(result.success)
        XCTAssertTrue(result.output.contains("知识库状态:"), result.output)
        XCTAssertTrue(result.output.contains("已存储切片数"), result.output)
    }
}
