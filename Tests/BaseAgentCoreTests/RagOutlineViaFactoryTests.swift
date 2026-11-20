import XCTest
@testable import BaseAgentCore

/// 端到端验证 AgentToolFactory 把其它分区 engine 接到了 rag_tools(op=outline)：
/// 默认库没有《从黎明到衰落》，历史分区有；通过 registry.execute 调 outline 必须命中。
/// 只测 RagOutlineTool 本身不够——接线（ToolFactory/Agent）漏了一样会失败。
final class RagOutlineViaFactoryTests: XCTestCase {
    private final class FakeEmbedding: EmbeddingProvider {
        let identifier = "outline-factory-fake"
        let dimension = 8
        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map { text in
                var v = [Float](repeating: 0.01, count: dimension)
                for scalar in text.unicodeScalars { v[Int(scalar.value) % dimension] += 1 }
                return VectorCodec.normalize(v)
            }
        }
        func embedQuery(_ text: String) async throws -> [Float] { try await embedDocuments([text])[0] }
    }

    func testOutlineViaRegistryFindsBookInNonDefaultZone() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("outline-factory-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let kbDefault = root.appendingPathComponent("social")
        let kbHistory = root.appendingPathComponent("history")
        try FileManager.default.createDirectory(at: kbDefault, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: kbHistory, withIntermediateDirectories: true)
        // 默认库：社会书
        try String(repeating: "社会学内容足够长以便被索引。", count: 10)
            .write(to: kbDefault.appendingPathComponent("社会书.md"), atomically: true, encoding: .utf8)
        // 历史分区：目标书（书名带全角冒号）
        let bookName = "从黎明到衰落：西方文化生活五百年"
        let md = """
        # \(bookName)
        ## 第一章 西方文化的黎明
        本章讲述西方文化的早期形成，从中世纪到文艺复兴的思想脉络，以及科学革命如何改变了人类的世界观。
        ## 第二章 文化衰落与现代转型
        本章讨论20世纪的文化变迁，包括两次世界大战对西方文明的冲击，以及现代主义艺术的兴起和大众文化的普及。
        补充足够长的内容确保文档被索引进知识库并被大纲工具识别，这段文字详细描述了西方文化生活五百年的发展历程。
        """
        try md.write(to: kbHistory.appendingPathComponent("\(bookName).md"),
                     atomically: true, encoding: .utf8)

        func makeEngine(kb: URL) -> RAGEngine {
            var config = AgentConfig()
            config.workspace = root.path
            config.knowledgeBase = kb.path
            config.ragRerankEnabled = false
            config.ragSemanticChunking = false
            config.ragChunkSize = 100
            config.ragChunkOverlap = 0
            return RAGEngine(config: config, embedding: FakeEmbedding(), reranker: nil)
        }
        let engDefault = makeEngine(kb: kbDefault)
        let engHistory = makeEngine(kb: kbHistory)
        _ = await engDefault.ingest()
        _ = await engHistory.ingest()

        var config = AgentConfig()
        config.workspace = root.path
        let callbacks = NoopCallbacks()
        let session = Session(budget: ContextBudget(total: 30_000))
        let skillsDir = root.appendingPathComponent("skills")
        try FileManager.default.createDirectory(at: skillsDir, withIntermediateDirectories: true)
        let skills = SkillManager(skillsDirectory: skillsDir.path, builtinDirectories: [])
        let skillIndex = SkillIndex(manager: skills)
        let retriever = SkillRetriever(manager: skills, embedding: FakeEmbedding(), reranker: nil)

        let registry = AgentToolFactory.build(
            config: config, callbacks: callbacks, session: session,
            ragEngine: engDefault, skillManager: skills, skillIndex: skillIndex,
            skillRetriever: retriever,
            extraOutlineEngines: [engHistory])
        let args: [String: JSONValue] = [
            "op": .string("outline"),
            "book_name": .string(bookName),
        ]
        let result = try await registry.execute("rag_tools", args)
        XCTAssertTrue(result.success, result.error)
        XCTAssertTrue(result.output.contains("从黎明到衰落"), result.output)
        XCTAssertTrue(result.output.contains("第一章"), result.output)
    }
}

private final class NoopCallbacks: AgentCallbacks {}
