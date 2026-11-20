import XCTest
@testable import BaseAgentCore

private final class ZoneOutlineEmbedding: EmbeddingProvider {
    let identifier = "zone-outline"
    let dimension = 8
    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map { text in
            var v = [Float](repeating: 0, count: dimension)
            for (i, s) in text.unicodeScalars.enumerated() {
                v[Int(s.value + UInt32(i)) % dimension] += 1
            }
            return VectorCodec.normalize(v)
        }
    }
    func embedQuery(_ text: String) async throws -> [Float] {
        try await embedDocuments([text])[0]
    }
}

/// 多分区 outline 回归：书在非默认分区时，rag_tools(op=outline) 必须能跨分区找到。
/// 之前只查默认库 engine →《病毒星球》在自然分区、默认库是社会分区 → 报"未找到文档"→
/// 模型被迫改用 search。
final class RagOutlineMultiZoneTests: XCTestCase {
    func testOutlineFindsDocumentInExtraEngine() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("outline-multizone-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let kbDefault = root.appendingPathComponent("default")
        let kbOther = root.appendingPathComponent("other")
        try FileManager.default.createDirectory(at: kbDefault, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: kbOther, withIntermediateDirectories: true)

        var config = AgentConfig()
        config.workspace = root.path
        config.ragChunkSize = 60
        config.ragChunkOverlap = 10
        config.ragRerankEnabled = false

        // 默认库：空
        var cfgDefault = config
        cfgDefault.knowledgeBase = kbDefault.path
        let defaultEngine = RAGEngine(config: cfgDefault, knowledgeBase: kbDefault.path,
                                      embedding: ZoneOutlineEmbedding(), reranker: nil)
        _ = await defaultEngine.ingest()

        // 其它分区：有《病毒星球》
        var cfgOther = config
        cfgOther.knowledgeBase = kbOther.path
        let md = """
        # 病毒星球
        ## 第一章 巨型病毒
        巨型病毒的发现改变了我们对生命边界的认识。
        ## 第二章 病毒与人类共生
        病毒与人类长期共生，驱动基因演化。
        Additional content to ensure the document is indexed in the knowledge base properly.
        """
        try md.write(to: kbOther.appendingPathComponent("病毒星球.md"),
                     atomically: true, encoding: .utf8)
        let otherEngine = RAGEngine(config: cfgOther, knowledgeBase: kbOther.path,
                                    embedding: ZoneOutlineEmbedding(), reranker: nil)
        _ = await otherEngine.ingest()

        let tool = RagOutlineTool(config: config, engine: defaultEngine,
                                  extraEngines: [otherEngine])
        let result = try await tool.run(["book_name": .string("病毒星球")])
        XCTAssertTrue(result.success, result.error)
        XCTAssertTrue(result.output.contains("病毒星球"), result.output)
        XCTAssertTrue(result.output.contains("第一章"), result.output)
    }

    /// 不传 book_name 时，列表应包含所有分区的文档。
    func testListIncludesAllZones() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("outline-list-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let kbA = root.appendingPathComponent("a")
        let kbB = root.appendingPathComponent("b")
        try FileManager.default.createDirectory(at: kbA, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: kbB, withIntermediateDirectories: true)

        var config = AgentConfig()
        config.workspace = root.path
        config.ragChunkSize = 60
        config.ragRerankEnabled = false
        let longText = String(repeating: "内容足够长以便被索引并被大纲工具列出。", count: 5)
        try "《社会书》\(longText)".write(to: kbA.appendingPathComponent("社会书.md"),
                                          atomically: true, encoding: .utf8)
        try "《自然书》\(longText)".write(to: kbB.appendingPathComponent("自然书.md"),
                                          atomically: true, encoding: .utf8)

        var cfgA = config; cfgA.knowledgeBase = kbA.path
        var cfgB = config; cfgB.knowledgeBase = kbB.path
        let eA = RAGEngine(config: cfgA, knowledgeBase: kbA.path,
                           embedding: ZoneOutlineEmbedding(), reranker: nil)
        let eB = RAGEngine(config: cfgB, knowledgeBase: kbB.path,
                           embedding: ZoneOutlineEmbedding(), reranker: nil)
        _ = await eA.ingest(); _ = await eB.ingest()

        let tool = RagOutlineTool(config: config, engine: eA, extraEngines: [eB])
        let result = try await tool.run([:])
        XCTAssertTrue(result.success)
        XCTAssertTrue(result.output.contains("社会书"), result.output)
        XCTAssertTrue(result.output.contains("自然书"), result.output)
        XCTAssertTrue(result.output.contains("跨 2 个分区"), result.output)
    }
}
