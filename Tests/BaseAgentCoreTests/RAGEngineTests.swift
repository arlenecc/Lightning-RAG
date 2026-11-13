import XCTest
@testable import BaseAgentCore

/// 模拟 OCR 有失败页的解析器：doc1 永远报告 1 页失败，doc2 正常。
private final class PartialFailureParser: DocumentParser {
    let name = "PartialFailure"
    let supportedExtensions = ["md", "markdown"]

    func parse(url: URL,
               options: ParserOptions,
               progress: ((ParserProgress) -> Void)?) async throws -> ParsedDocument {
        let text = try String(contentsOf: url, encoding: .utf8)
        let failed = url.lastPathComponent.contains("doc1") ? "1" : "0"
        var parsed = ParsedDocument(text: text,
                                    source: url.lastPathComponent,
                                    metadata: ["engine": "partial",
                                               "failed_pages": failed])
        // 模拟 OCR 路径：走激进清洗档。
        parsed = finish(parsed, options: options, profile: .ocr)
        return parsed
    }
}

private final class RagFakeEmbedding: EmbeddingProvider {
    let identifier = "rag-fake"
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

/// 模型缺失时的占位 embedding：维度/标识都不代表真实配置。
private final class RagUnavailableEmbedding: EmbeddingProvider {
    let identifier = "rag-unavailable"
    let dimension = 768
    let isAvailable = false

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        throw AgentError.storage("embedding 模型未就绪")
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        throw AgentError.storage("embedding 模型未就绪")
    }
}

/// 声明精确 token 计数的假 embedding：英文按 4 字符/token，
/// 用于构造「一个切片超过旧 1500 字符呈现上限」的真实场景。
private final class RagExactTokenEmbedding: EmbeddingProvider, EmbeddingTokenCounting {
    let identifier = "rag-exact-token"
    let dimension = 8

    var maxContentTokens: Int { 8_000 }

    func makeTokenCounter() async -> (@Sendable (String) -> Int)? {
        { max(1, $0.count / 4) }
    }

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

/// 关键词假 embedding：把 apple / zebra / volcano 映射到互不相关的基向量，
/// 用于验证定向来源检索时不受哈希碰撞干扰。
private final class RagKeywordEmbedding: EmbeddingProvider {
    let identifier = "rag-keyword"
    let dimension = 8

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map { text in
            var vector = [Float](repeating: 0, count: dimension)
            if text.localizedCaseInsensitiveContains("apple") { vector[0] = 1 }
            if text.localizedCaseInsensitiveContains("zebra") { vector[5] = 1 }
            if text.localizedCaseInsensitiveContains("volcano") { vector[6] = 1 }
            if vector.allSatisfy({ $0 == 0 }) { vector[7] = 1 }
            return VectorCodec.normalize(vector)
        }
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        try await embedDocuments([text])[0]
    }
}

/// 确定性的假 reranker：按候选顺序给递减分数，全部高于默认阈值，
/// 用于验证「重排序后的 TopN 截断」而不是分数过滤。
private final class RagFakeReranker: RerankProvider {
    let identifier = "rag-fake-rerank"

    func rerank(query: String, documents: [String]) async throws -> [Double] {
        documents.enumerated().map { index, _ in 0.9 - Double(index) * 0.01 }
    }
}

final class RAGEngineTests: XCTestCase {
    func testIngestAndHybridSearchEndToEnd() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let markdown = """
        # 第一章 水果

        Apple and banana are common fruits. 苹果和香蕉都是常见水果。
        Apple is a common fruit in daily meals. 苹果是日常饮食中的常见水果.

        # 第二章 机械

        Turbines and engines are machines. 涡轮和发动机属于机械。
        Apple sentences keep the document clear of the short-body guard.
        """
        try markdown.write(to: kb.appendingPathComponent("fruits.md"),
                           atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 5

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagFakeEmbedding(), reranker: nil)
        let stats = await engine.ingest()
        XCTAssertEqual(stats.processed, 1)
        XCTAssertGreaterThan(stats.chunks, 0)
        XCTAssertTrue(stats.errors.isEmpty, stats.errors.joined(separator: "; "))

        XCTAssertEqual(engine.listDocuments().count, 1)
        let outline = engine.getDocumentOutline(docName: "fruits.md")
        XCTAssertNotNil(outline)
        XCTAssertTrue(outline?.digest.contains("第一章") == true)

        let hits = try await engine.search(query: "apple fruit", topK: 3)
        XCTAssertFalse(hits.isEmpty)
        XCTAssertTrue(hits[0].source.hasSuffix("fruits.md"))
        XCTAssertTrue(hits[0].text.contains("Apple") || hits[0].text.contains("苹果"))

        // 无变更时增量同步应全部跳过。
        let second = await engine.ingest()
        XCTAssertEqual(second.processed, 0)
        XCTAssertEqual(second.skipped, 1)

        let formatted = try await engine.searchFormatted(query: "turbine", topK: 2)
        XCTAssertTrue(formatted.contains("fruits.md"))

        // 删除源文件后再次同步：切片与 documents 摘要都必须清理。
        try FileManager.default.removeItem(at: kb.appendingPathComponent("fruits.md"))
        let third = await engine.ingest()
        XCTAssertEqual(third.removed, 1)
        XCTAssertTrue(engine.listDocuments().isEmpty)
        let afterDelete = try await engine.search(query: "apple", topK: 3)
        XCTAssertTrue(afterDelete.isEmpty)
        engine.close()
    }

    /// embedding 模型未就绪（占位 provider）时不能用占位维度做索引一致性校验：
    /// 旧行为会把占位 768 与库里的真实维度相比，误报「向量库需要重建」。
    func testUnavailableEmbeddingDoesNotFlagRebuild() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-unavailable-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let markdown = """
        # 第一章 水果

        Apple and banana are common fruits. 苹果和香蕉都是常见水果。
        Apple is a common fruit in daily meals. 苹果是日常饮食中的常见水果.
        Bananas are rich in potassium and are popular among athletes.
        Oranges provide vitamin C and are widely grown in warm regions.
        Grapes can be eaten fresh or used to make wine in many countries.
        """
        try markdown.write(to: kb.appendingPathComponent("fruits.md"),
                           atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 5

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagFakeEmbedding(), reranker: nil)
        let stats = await engine.ingest()
        XCTAssertGreaterThan(stats.chunks, 0)
        engine.close()

        // 模型缺失：provider 是占位实现，维度 768 不代表真实配置。
        let reopened = RAGEngine(config: config, knowledgeBase: kb.path,
                                 embedding: RagUnavailableEmbedding(), reranker: nil)
        let snapshot = reopened.status(maxAge: 0)
        XCTAssertNotEqual(snapshot["vector_rebuild_required"], .bool(true),
                          "占位 embedding 不能让已有索引被判成需要重建")
        XCTAssertNil(snapshot["rebuild_reason"]?.stringValue)
        reopened.close()
    }

    func testShortDocumentIsDiscardedAndSkippedNextRun() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-short-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        // 只有书名与目录名：去掉标题/目录行后正文几乎为空。
        try "# 书名\n\n目录\n\n第一章 绪论 1\n第二章 方法 5\n"
            .write(to: kb.appendingPathComponent("junk.md"),
                   atomically: true, encoding: .utf8)
        try """
        # 第一章 水果

        Apple and banana are common fruits. 苹果和香蕉都是常见水果。
        Turbines and engines are machines. 涡轮和发动机属于机械。
        Additional sentences make the body long enough for indexing, so the document is kept in the knowledge base.
        """
            .write(to: kb.appendingPathComponent("fruits.md"),
                   atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 5

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagFakeEmbedding(), reranker: nil)
        let first = await engine.ingest()
        XCTAssertEqual(first.discarded, 1)
        XCTAssertEqual(first.processed, 1)
        XCTAssertTrue(first.errors.isEmpty, first.errors.joined(separator: "; "))
        XCTAssertEqual(engine.listDocuments(), ["fruits.md"])

        // 内容未变化时不再重新解析，直接按 manifest 标记跳过。
        let second = await engine.ingest()
        XCTAssertEqual(second.discarded, 0)
        XCTAssertEqual(second.processed, 0)
        XCTAssertEqual(second.skipped, 2)
        engine.close()
    }

    func testShortenedDocumentRemovesPreviousIndex() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-shorten-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let file = kb.appendingPathComponent("book.md")
        try """
        # 第一章 水果

        Apple and banana are common fruits. 苹果和香蕉都是常见水果。
        Turbines and engines are machines. 涡轮和发动机属于机械。
        Additional sentences make the body long enough for indexing, so the document is kept in the knowledge base.
        """
            .write(to: file, atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 5

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagFakeEmbedding(), reranker: nil)
        let first = await engine.ingest()
        XCTAssertEqual(first.processed, 1)
        XCTAssertEqual(engine.listDocuments().count, 1)
        let indexedHits = try await engine.search(query: "apple", topK: 3)
        XCTAssertFalse(indexedHits.isEmpty)

        // 文件被改短（大小变化触发重新解析）：旧切片/文档/缓存必须一起清掉。
        try "# 书名\n\n目录\n".write(to: file, atomically: true, encoding: .utf8)
        let second = await engine.ingest()
        XCTAssertEqual(second.discarded, 1)
        XCTAssertTrue(engine.listDocuments().isEmpty)
        XCTAssertNil(engine.getDocumentOutline(docName: "book.md"))
        let remainingHits = try await engine.search(query: "apple", topK: 3)
        XCTAssertTrue(remainingHits.isEmpty)
        engine.close()
    }

    func testShortBodyThresholdKeepsOneHundredCharacters() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-threshold-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        // 标题行不计入正文：正文 99 字丢弃，100 字正常入库。
        try "# 标题\n\n\(String(repeating: "a", count: 99))\n"
            .write(to: kb.appendingPathComponent("drop.md"),
                   atomically: true, encoding: .utf8)
        try "# 标题\n\n\(String(repeating: "a", count: 100))\n"
            .write(to: kb.appendingPathComponent("keep.md"),
                   atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 5

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagFakeEmbedding(), reranker: nil)
        let stats = await engine.ingest()
        XCTAssertEqual(stats.discarded, 1)
        XCTAssertEqual(stats.processed, 1)
        XCTAssertEqual(engine.listDocuments(), ["keep.md"])
        engine.close()
    }

    func testCancelledFullRebuildResumesFromUnfinishedFiles() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-resume-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        for index in 1...3 {
            let text = """
            # 文档 \(index)

            Apple banana orange fruit \(index). 苹果 香蕉 橙子 水果 \(index)。
            这是一段用于验证中断续跑的文件内容，编号 \(index)。
            Additional sentences make the body long enough for indexing, so the document is kept in the knowledge base.
            """
            try text.write(to: kb.appendingPathComponent("doc\(index).md"),
                           atomically: true, encoding: .utf8)
        }

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2

        let firstEngine = RAGEngine(config: config, knowledgeBase: kb.path,
                                    embedding: RagFakeEmbedding(), reranker: nil)
        let initial = await firstEngine.ingest()
        XCTAssertEqual(initial.processed, 3)
        firstEngine.close()

        // 修改切片参数触发逐文件重建；处理完第一个文件后停止。
        // 并发 ingest 下停止时会保留已完成的文件，这里固定为单文件便于断言。
        config.ragChunkSize = 60
        config.ragChunkOverlap = 3
        config.ragIngestConcurrency = 1
        let partialEngine = RAGEngine(config: config, knowledgeBase: kb.path,
                                      embedding: RagFakeEmbedding(), reranker: nil)
        var cancelled = false
        let partial = await partialEngine.ingest(log: { line in
            if !cancelled, line.contains("✓") {
                cancelled = true
                partialEngine.cancel()
            }
        })
        XCTAssertTrue(partial.cancelled)
        XCTAssertEqual(partial.processed, 1)
        partialEngine.close()

        // 同一新配置再次同步：已完成的 1 个文件跳过，只处理剩余 2 个。
        let resumeEngine = RAGEngine(config: config, knowledgeBase: kb.path,
                                     embedding: RagFakeEmbedding(), reranker: nil)
        let resumed = await resumeEngine.ingest()
        XCTAssertFalse(resumed.cancelled)
        XCTAssertEqual(resumed.skipped, 1)
        XCTAssertEqual(resumed.processed, 2)
        XCTAssertTrue(resumed.errors.isEmpty, resumed.errors.joined(separator: "; "))
        resumeEngine.close()
    }

    func testIdenticalFilesAcrossDirectoriesAreParsedOnce() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-dup-raw-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        let dirA = kb.appendingPathComponent("a")
        let dirB = kb.appendingPathComponent("b")
        try FileManager.default.createDirectory(at: dirA, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: dirB, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let content = """
        # 同一本书

        Apple banana orange fruit. 苹果 香蕉 橙子 水果。
        这是用于验证跨目录重复文件的正文。
        Additional sentences make the body long enough for indexing, so the document is kept in the knowledge base.
        """
        try content.write(to: dirA.appendingPathComponent("book.md"),
                          atomically: true, encoding: .utf8)
        try content.write(to: dirB.appendingPathComponent("book.md"),
                          atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagFakeEmbedding(), reranker: nil)
        let first = await engine.ingest()
        XCTAssertEqual(first.processed, 1, "相同文件只应解析一次")
        XCTAssertEqual(first.deduplicated, 0)
        XCTAssertEqual(engine.status()["sources"]?.arrayValue?.count, 1)
        XCTAssertEqual(engine.listDocuments().count, 1)

        let second = await engine.ingest()
        XCTAssertEqual(second.processed, 0)
        XCTAssertEqual(second.skipped, 1)
        XCTAssertEqual(second.deduplicated, 0)
        XCTAssertEqual(engine.status()["sources"]?.arrayValue?.count, 1)
        engine.close()
    }

    /// 默认信任 mtime + 大小：同尺寸、同 mtime 的内容变化不会被发现；
    /// 打开 `rag_verify_fingerprint` 后才重新采样指纹并重建。
    func testIncrementalSyncTrustsMtimeAndSizeUnlessVerifyFingerprint() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-verify-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let path = kb.appendingPathComponent("note.md")
        // 两个版本长度一致（同尺寸改动），正文都超过过短丢弃阈值。
        let contentA = "# 标题\n\n" + String(repeating: "版本一正文。", count: 20)
        let contentB = "# 标题\n\n" + String(repeating: "版本二正文。", count: 20)
        XCTAssertEqual(contentA.utf8.count, contentB.utf8.count,
                       "构造同尺寸改动，才需要指纹兜底")
        try contentA.write(to: path, atomically: true, encoding: .utf8)
        let fixed = Date(timeIntervalSince1970: 1_700_000_000)
        try FileManager.default.setAttributes([.modificationDate: fixed],
                                              ofItemAtPath: path.path)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2
        config.ragVerifyFingerprint = false

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagFakeEmbedding(), reranker: nil)
        let first = await engine.ingest()
        XCTAssertEqual(first.processed, 1)

        try contentB.write(to: path, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.modificationDate: fixed],
                                              ofItemAtPath: path.path)
        let second = await engine.ingest()
        XCTAssertEqual(second.processed, 0, "默认信任 mtime + size，不再读盘采样")
        XCTAssertEqual(second.skipped, 1)
        engine.close()

        var verifyConfig = config
        verifyConfig.ragVerifyFingerprint = true
        let verifying = RAGEngine(config: verifyConfig, knowledgeBase: kb.path,
                                  embedding: RagFakeEmbedding(), reranker: nil)
        let third = await verifying.ingest()
        XCTAssertEqual(third.processed, 1, "verify_fingerprint 应发现同尺寸同 mtime 的改动")
        XCTAssertEqual(third.skipped, 0)
        verifying.close()
    }

    func testContentHashDedupeRemovesDuplicateVectorsAndSkipsNextRun() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-dup-content-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        let dirA = kb.appendingPathComponent("a")
        let dirB = kb.appendingPathComponent("b")
        try FileManager.default.createDirectory(at: dirA, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: dirB, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        // 原始字节不同（空行/结尾不同），清洗后的正文完全一致。
        let body = String(repeating: "正文内容。", count: 30)
        try "# 同一本书\n\n\(body)".write(
            to: dirA.appendingPathComponent("book.md"),
            atomically: true, encoding: .utf8)
        try "# 同一本书\n\n\n\(body)\n".write(
            to: dirB.appendingPathComponent("book.md"),
            atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagFakeEmbedding(), reranker: nil)
        let first = await engine.ingest()
        // 解析/清洗后即命中内容哈希：副本只写 tombstone，不再切片/embedding。
        XCTAssertEqual(first.processed, 1)
        XCTAssertEqual(first.skipped, 1)
        XCTAssertEqual(first.deduplicated, 1, "content_hash 相同的副本应提前跳过")
        XCTAssertEqual(engine.status()["sources"]?.arrayValue?.count, 1)
        XCTAssertEqual(engine.listDocuments().count, 1)

        // 副本写入了 tombstone，下一次同步不再重复解析。
        let second = await engine.ingest()
        XCTAssertEqual(second.processed, 0)
        XCTAssertEqual(second.skipped, 2)
        XCTAssertEqual(second.deduplicated, 0)
        XCTAssertEqual(engine.status()["sources"]?.arrayValue?.count, 1)
        engine.close()
    }

    func testManualVectorDedupeEmptiesDuplicateSourceAndSkipsNextRun() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-manual-dedupe-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try "# 甲\n\n\(String(repeating: "这是甲文档独有的正文。", count: 12))".write(
            to: kb.appendingPathComponent("a.md"),
            atomically: true, encoding: .utf8)
        try "# 乙\n\n\(String(repeating: "这是乙文档独有的正文。", count: 12))".write(
            to: kb.appendingPathComponent("b.md"),
            atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        // 每个文档保持单切片，便于按切片数验证重复清理。
        config.ragChunkSize = 500
        config.ragChunkOverlap = 2

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagFakeEmbedding(), reranker: nil)
        let first = await engine.ingest()
        XCTAssertEqual(first.processed, 2)
        engine.close()

        // 模拟历史遗留/外部写入：乙文档已被写入与甲文档完全相同的切片。
        let store = try VectorStore(dbPath: config.dbPath, dimension: 8,
                                    enableSQLiteVec: config.ragSQLiteVecEnabled,
                                    elementType: config.ragVectorElementType,
                                    indexFingerprint: engine.vectorIndexFingerprint)
        let sources = store.listSources()
        guard let sourceA = sources.first(where: { $0.hasSuffix("/a.md") }),
              let sourceB = sources.first(where: { $0.hasSuffix("/b.md") }) else {
            store.close()
            XCTFail("未找到 a.md/b.md 的切片来源")
            return
        }
        let rows = try store.db.query(
            "SELECT text FROM chunks WHERE source = ? ORDER BY created_at ASC, rowid ASC LIMIT 1",
            [.text(sourceA)])
        guard let textA = rows.first?["text"]?.stringValue else {
            store.close()
            XCTFail("甲文档没有切片")
            return
        }
        let vector = try await RagFakeEmbedding().embedDocuments([textA])[0]
        _ = try store.add(chunks: [Chunk(text: textA, source: sourceB, chunkIndex: 0)],
                          vectors: [vector])
        store.close()

        let preview = try engine.previewVectorDedupe()
        XCTAssertEqual(preview.totalChunks, 2)
        XCTAssertEqual(preview.duplicateChunks, 1)
        XCTAssertEqual(preview.affectedSources, 1)

        let stats = try engine.deduplicateVectors()
        XCTAssertEqual(stats.removedChunks, 1)
        XCTAssertEqual(stats.affectedSources, 1)
        XCTAssertEqual(stats.emptiedSources, 1)
        XCTAssertEqual(stats.remainingChunks, 1)
        XCTAssertEqual(engine.status()["sources"]?.arrayValue?.count, 1)
        XCTAssertEqual(engine.status()["source_count"]?.intValue, 1)
        XCTAssertEqual(engine.listDocuments().count, 1)

        // manifest 中乙文档写入 tombstone，指向保留来源。
        let manifestURL = URL(fileURLWithPath: config.ragIndexDirectory)
            .appendingPathComponent("manifest.json")
        let object = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as? [String: Any] ?? [:]
        let files = object["files"] as? [String: Any] ?? [:]
        let entryB = files[sourceB] as? [String: Any]
        XCTAssertEqual(entryB?["duplicateOf"] as? String, sourceA)
        XCTAssertEqual(entryB?["chunkCount"] as? Int, 0)

        // 下次同步直接跳过副本，不会重新写回重复向量。
        let second = await engine.ingest()
        XCTAssertEqual(second.processed, 0)
        XCTAssertEqual(second.skipped, 2)
        XCTAssertEqual(engine.status()["sources"]?.arrayValue?.count, 1)
        engine.close()
    }

    func testIncrementalReindexesFileWhenChunksDisappear() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-missing-chunks-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        for index in 1...2 {
            try """
            # 文档 \(index)

            这是用于验证切片缺失后增量同步会重建的第 \(index) 个文件。
            Additional sentences make the body long enough for indexing, so the document is kept in the knowledge base.
            """.write(to: kb.appendingPathComponent("doc\(index).md"),
                      atomically: true, encoding: .utf8)
        }

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2

        let first = RAGEngine(config: config, knowledgeBase: kb.path,
                              embedding: RagFakeEmbedding(), reranker: nil)
        let initial = await first.ingest()
        XCTAssertEqual(initial.processed, 2)
        first.close()

        // 模拟「manifest 说完成、库里却没有切片」：外部清理/上次写入中断。
        let store = try VectorStore(dbPath: config.dbPath, dimension: 8,
                                    enableSQLiteVec: config.ragSQLiteVecEnabled,
                                    elementType: config.ragVectorElementType,
                                    indexFingerprint: first.vectorIndexFingerprint)
        guard let doc1Source = store.listSources().first(where: {
            $0.hasSuffix("/doc1.md")
        }) else {
            XCTFail("未找到 doc1.md 的切片来源")
            store.close()
            return
        }
        _ = try store.deleteBySource(doc1Source)
        store.close()

        let resumed = RAGEngine(config: config, knowledgeBase: kb.path,
                                embedding: RagFakeEmbedding(), reranker: nil)
        let stats = await resumed.ingest()
        XCTAssertEqual(stats.processed, 1, "切片缺失的文件必须重建")
        XCTAssertEqual(stats.skipped, 1)
        XCTAssertTrue(stats.errors.isEmpty, stats.errors.joined(separator: "; "))
        resumed.close()
    }

    func testFailedPagesAreRetriedOnNextIncrementalSync() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-partial-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        for index in 1...2 {
            try """
            # 文档 \(index)

            这是用于验证失败页重试的第 \(index) 个文件。
            Additional sentences make the body long enough for indexing, so the document is kept in the knowledge base.
            """.write(to: kb.appendingPathComponent("doc\(index).md"),
                      atomically: true, encoding: .utf8)
        }

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2

        let registry = DocumentParserRegistry(parsers: [PartialFailureParser()], config: config)
        let first = RAGEngine(config: config, knowledgeBase: kb.path,
                              embedding: RagFakeEmbedding(), reranker: nil,
                              parserRegistry: registry)
        let initial = await first.ingest()
        XCTAssertEqual(initial.processed, 2)
        first.close()

        let resumed = RAGEngine(config: config, knowledgeBase: kb.path,
                                embedding: RagFakeEmbedding(), reranker: nil,
                                parserRegistry: DocumentParserRegistry(
                                    parsers: [PartialFailureParser()], config: config))
        let stats = await resumed.ingest()
        XCTAssertEqual(stats.processed, 1, "有失败页的文件必须重试")
        XCTAssertEqual(stats.skipped, 1)
        resumed.close()
    }

    /// Excel 部分工作表损坏时仍保留可解析内容，但 manifest 必须标记未完成，
    /// 下次增量同步重新解析，而不是被 `complete=true` 永久跳过。
    func testUnreadableExcelSheetMarksIncompleteAndRetries() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-excel-partial-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let builder = BIFF8TestBuilder(
            sheets: [
                .init(name: "有效数据", cells: [
                    .init(row: 0, column: 0, value: .label("产品")),
                    .init(row: 0, column: 1, value: .label("说明")),
                    .init(row: 1, column: 0, value: .label("保险")),
                    .init(row: 1, column: 1, value: .label("用于验证部分工作表损坏时仍能入库")),
                    .init(row: 2, column: 0, value: .label("医疗")),
                    .init(row: 2, column: 1, value: .label("继续补充正文确保超过入库长度阈值")),
                    .init(row: 3, column: 0, value: .label("出行")),
                    .init(row: 3, column: 1, value: .label("更多条目让表格内容足够被切片和向量化")),
                    .init(row: 4, column: 0, value: .label("财产")),
                    .init(row: 4, column: 1, value: .label("继续补充足够多的有效正文，避免被当成无效碎片丢弃")),
                    .init(row: 5, column: 0, value: .label("责任")),
                    .init(row: 5, column: 1, value: .label("这一行同样用于增加表格正文长度并保持可读性")),
                    .init(row: 6, column: 0, value: .label("工程")),
                    .init(row: 6, column: 1, value: .label("工程险条目继续扩展内容确保总量稳定超过阈值")),
                    .init(row: 7, column: 0, value: .label("保证")),
                    .init(row: 7, column: 1, value: .label("保证保险条目用于验证多行表格正文的完整处理")),
                ]),
                .init(name: "损坏表", cells: [], rawPayload: Data()),
            ]
        )
        try builder.build().data.write(to: kb.appendingPathComponent("partial.xls"))

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2

        let first = RAGEngine(config: config, knowledgeBase: kb.path,
                              embedding: RagFakeEmbedding(), reranker: nil)
        let initial = await first.ingest()
        XCTAssertEqual(initial.processed, 1)
        XCTAssertTrue(initial.errors.isEmpty, initial.errors.joined(separator: "; "))
        first.close()

        let resumed = RAGEngine(config: config, knowledgeBase: kb.path,
                                embedding: RagFakeEmbedding(), reranker: nil)
        let stats = await resumed.ingest()
        XCTAssertEqual(stats.processed, 1, "有损坏工作表的 Excel 必须重试")
        XCTAssertEqual(stats.skipped, 0)
        resumed.close()
    }

    func testIncrementalReindexesLegacyManifestWithoutCompletionFields() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-legacy-manifest-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        for index in 1...2 {
            try """
            # 文档 \(index)

            旧版 manifest 缺少完成标记时，增量同步应重新处理该文件。
            Additional sentences make the body long enough for indexing, so the document is kept in the knowledge base.
            """.write(to: kb.appendingPathComponent("doc\(index).md"),
                      atomically: true, encoding: .utf8)
        }

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2

        let first = RAGEngine(config: config, knowledgeBase: kb.path,
                              embedding: RagFakeEmbedding(), reranker: nil)
        let initial = await first.ingest()
        XCTAssertEqual(initial.processed, 2)
        first.close()

        let manifestURL = URL(fileURLWithPath: config.ragIndexDirectory)
            .appendingPathComponent("manifest.json")
        var object = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as? [String: Any] ?? [:]
        var files = object["files"] as? [String: Any] ?? [:]
        guard let legacyKey = files.keys.first(where: { $0.hasSuffix("/doc1.md") }) else {
            XCTFail("manifest 中未找到 doc1.md")
            return
        }
        var entry = files[legacyKey] as? [String: Any] ?? [:]
        entry.removeValue(forKey: "chunkCount")
        entry.removeValue(forKey: "complete")
        files[legacyKey] = entry
        object["files"] = files
        try JSONSerialization.data(withJSONObject: object).write(to: manifestURL)

        let resumed = RAGEngine(config: config, knowledgeBase: kb.path,
                                embedding: RagFakeEmbedding(), reranker: nil)
        let stats = await resumed.ingest()
        XCTAssertEqual(stats.processed, 1, "旧版 manifest 条目必须重建")
        XCTAssertEqual(stats.skipped, 1)
        resumed.close()
    }

    func testSwitchingKnowledgeBasePreservesEachIndex() async throws {
        // 回归：历史版本所有知识库共用 rag_data 根目录，切换知识库目录后
        // 同步会把对方条目当「已删除文件」清空，切回后只能全量重建。
        // 修复后索引按知识库目录隔离，切回后增量同步仍然有效。
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-kb-switch-\(UUID().uuidString)")
        let kbA = root.appendingPathComponent("kbA")
        let kbB = root.appendingPathComponent("kbB")
        try FileManager.default.createDirectory(at: kbA, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: kbB, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try "# 甲\n\n甲库独有正文，用于验证切库后索引保留。\nAdditional sentences make the body long enough for indexing, so the document is kept in the knowledge base."
            .write(to: kbA.appendingPathComponent("a.md"), atomically: true, encoding: .utf8)
        try "# 乙\n\n乙库独有正文，用于验证切库后索引保留。\nAdditional sentences make the body long enough for indexing, so the document is kept in the knowledge base."
            .write(to: kbB.appendingPathComponent("b.md"), atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kbA.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2

        let dirA = config.ragIndexDirectory
        config.knowledgeBase = kbB.path
        let dirB = config.ragIndexDirectory
        XCTAssertNotEqual(dirA, dirB, "不同知识库目录必须使用不同的索引子目录")

        // 库 A 建索引 → 切到库 B 同步 → 切回库 A 必须增量跳过而不是全量重建。
        config.knowledgeBase = kbA.path
        let engineA = RAGEngine(config: config, knowledgeBase: kbA.path,
                                embedding: RagFakeEmbedding(), reranker: nil)
        let firstA = await engineA.ingest()
        XCTAssertEqual(firstA.processed, 1)
        engineA.close()

        config.knowledgeBase = kbB.path
        let engineB = RAGEngine(config: config, knowledgeBase: kbB.path,
                                embedding: RagFakeEmbedding(), reranker: nil)
        let firstB = await engineB.ingest()
        XCTAssertEqual(firstB.processed, 1)
        engineB.close()

        // 库 B 同步后，库 A 的 manifest/向量必须原样保留。
        let manifestA = try JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: dirA)
                .appendingPathComponent("manifest.json"))) as? [String: Any] ?? [:]
        XCTAssertEqual((manifestA["files"] as? [String: Any])?.count, 1,
                       "切库同步后库 A 的 manifest 不得被清空")

        config.knowledgeBase = kbA.path
        let backA = RAGEngine(config: config, knowledgeBase: kbA.path,
                              embedding: RagFakeEmbedding(), reranker: nil)
        let secondA = await backA.ingest()
        XCTAssertEqual(secondA.processed, 0, "切回库 A 后增量同步必须生效")
        XCTAssertEqual(secondA.skipped, 1)
        backA.close()
    }

    func testLongChunkIsNotTruncatedWhenChunkSizeGrows() async throws {
        // 回归：检索呈现曾固定按 1500 字符/条截断（rag_search_max_chars_per_result），
        // 用户把切片调大后仍只能看到切片前半段——即「设置面板改大了切片，
        // 检索环节却仍按短块呈现」。呈现上限必须随生效切片放大。
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-long-chunk-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        // 一段无标点的长正文（≈1.7k 字符，中文 1 字 ≈ 1 token），
        // 单章节、单切片，尾部带唯一标记用于判断是否被截断。
        let body = String(repeating: "甲", count: 1_700) + "尾部唯一标记XYZ"
        try body.write(to: kb.appendingPathComponent("long.md"),
                       atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        // 长切片（大于默认 1500 字符呈现上限）；测试用 embedding 不声明窗口，
        // 因此不会被收口，等价于用户显式设置了长切片。
        config.ragChunkSize = 2_000
        config.ragChunkOverlap = 100

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagFakeEmbedding(), reranker: nil)
        let stats = await engine.ingest()
        XCTAssertEqual(stats.processed, 1)
        XCTAssertEqual(stats.chunks, 1, "长正文应作为完整切片入库")

        let formatted = try await engine.searchFormatted(query: "甲", topK: 1)
        XCTAssertTrue(formatted.contains("尾部唯一标记XYZ"),
                      "长切片不得被固定字符上限截短")
        XCTAssertGreaterThan(formatted.count, config.ragSearchMaxCharsPerResult)
        engine.close()
    }

    func testChunkSizeAndOverlapChangesTriggerRebuild() async throws {
        // 切片大小 / overlap 都进 ingest 签名：改任一项都必须逐文件重建，
        // 不能因为「文件本身没变」而继续沿用旧切片。
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-chunk-signature-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try """
        # 切片签名

        这是用于验证切片参数变化会触发重建的正文内容，多个句子。
        第二句用于让切片边界随参数变化。第三句补齐长度。
        Additional sentences make the body long enough for indexing, so the document is kept in the knowledge base.
        """.write(to: kb.appendingPathComponent("doc.md"),
                  atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2

        let first = RAGEngine(config: config, knowledgeBase: kb.path,
                              embedding: RagFakeEmbedding(), reranker: nil)
        let initial = await first.ingest()
        XCTAssertEqual(initial.processed, 1)
        first.close()

        // 只改切片大小：必须重建。
        config.ragChunkSize = 80
        let resized = RAGEngine(config: config, knowledgeBase: kb.path,
                                embedding: RagFakeEmbedding(), reranker: nil)
        let afterSize = await resized.ingest()
        XCTAssertEqual(afterSize.processed, 1, "切片大小变化必须重建")
        XCTAssertEqual(afterSize.skipped, 0)
        resized.close()

        // 只改 overlap：同样必须重建。
        config.ragChunkOverlap = 8
        let overlapped = RAGEngine(config: config, knowledgeBase: kb.path,
                                   embedding: RagFakeEmbedding(), reranker: nil)
        let afterOverlap = await overlapped.ingest()
        XCTAssertEqual(afterOverlap.processed, 1, "overlap 变化必须重建")
        XCTAssertEqual(afterOverlap.skipped, 0)
        overlapped.close()

        // 只切语义切片开关：不能因为「没改大小」就跳过。
        config.ragSemanticChunking.toggle()
        let switched = RAGEngine(config: config, knowledgeBase: kb.path,
                                 embedding: RagFakeEmbedding(), reranker: nil)
        let afterSemantic = await switched.ingest()
        XCTAssertEqual(afterSemantic.processed, 1, "语义切片开关变化必须重建")
        XCTAssertEqual(afterSemantic.skipped, 0)
        switched.close()
    }

    func testRerankTopNControlsResultCount() async throws {
        // 设置面板的 TopN 决定重排序后最终返回几条：TopN=3 时即使调用方
        // 请求 top_k=10，也只返回 3 条；关闭重排序后恢复按调用方 top_k。
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-topn-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        var markdown = "# 检索片段\n\n"
        for index in 1...8 {
            markdown += "\n## 片段\(index)\n\n这是第\(index)段用于验证重排序 TopN 的正文内容。\n"
        }
        try markdown.write(to: kb.appendingPathComponent("topn.md"),
                           atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragChunkSize = 40
        config.ragChunkOverlap = 5
        config.ragRerankEnabled = true
        config.ragRerankTopN = 3
        // 阈值降到 0，避免分数过滤影响条数断言。
        config.ragRerankThreshold = 0

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagFakeEmbedding(),
                               reranker: RagFakeReranker())
        let stats = await engine.ingest()
        XCTAssertGreaterThanOrEqual(stats.chunks, 4, "需要多条候选才能验证 TopN")

        let three = try await engine.search(query: "片段", topK: 10)
        XCTAssertEqual(three.count, 3, "TopN 决定重排序后的返回条数")
        engine.close()

        // 同一份库换 TopN=6：返回条数随设置放大（仍受库内条数限制）。
        config.ragRerankTopN = 6
        let wider = RAGEngine(config: config, knowledgeBase: kb.path,
                              embedding: RagFakeEmbedding(),
                              reranker: RagFakeReranker())
        let six = try await wider.search(query: "片段", topK: 10)
        XCTAssertEqual(six.count, min(6, stats.chunks), "TopN 放大后返回条数随之增加")
        wider.close()

        // 关闭重排序：回到调用方 top_k，TopN 不参与。
        config.ragRerankEnabled = false
        config.ragRerankTopN = 3
        let plain = RAGEngine(config: config, knowledgeBase: kb.path,
                              embedding: RagFakeEmbedding(),
                              reranker: RagFakeReranker())
        let two = try await plain.search(query: "片段", topK: 2)
        XCTAssertEqual(two.count, 2, "未开启重排序时按调用方 top_k 返回")
        plain.close()
    }

    func testStatusSnapshotCachesAndInvalidatesOnIngest() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-status-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try "# 标题\n\n用于验证状态快照失效的正文。\nAdditional sentences make the body long enough for indexing, so the document is kept in the knowledge base."
            .write(to: kb.appendingPathComponent("a.md"),
                   atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagFakeEmbedding(), reranker: nil)
        // 首轮提示词只读快照：还没有快照时不能打开数据库。
        XCTAssertNil(engine.cachedStatus())

        let empty = engine.status()
        XCTAssertEqual(empty["chunks_stored"]?.intValue, 0)
        XCTAssertEqual(engine.cachedStatus()?["chunks_stored"]?.intValue, 0)

        let stats = await engine.ingest()
        XCTAssertGreaterThan(stats.chunks, 0)
        XCTAssertNil(engine.cachedStatus(), "写操作必须主动让快照失效")

        let refreshed = engine.status()
        XCTAssertEqual(refreshed["chunks_stored"]?.intValue, stats.chunks)
        XCTAssertEqual(refreshed["source_count"]?.intValue, 1)
        XCTAssertEqual(refreshed["sources"]?.arrayValue?.count, 1)
        engine.close()
    }

    /// source 子串必须先解析成真实来源，再在该来源内部定向检索；
    /// 不能拿子串去全局召回后过滤——大库上全局候选被其他来源占满就返回空。
    func testSourceSubstringUsesExactSourceForScopedRecall() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-scoped-source-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        // 其他来源制造超过 maxVectorK(4096) 的全局候选，把目标来源挤出
        // 精确 KNN 的候选池。每行带唯一编号：清洗会折叠完全重复的行，
        // 纯重复文本只剩一个切片。
        let other = (0..<5_000).map { "apple banana fruit \($0).\n" }.joined()
        try other.write(to: kb.appendingPathComponent("other.md"),
                        atomically: true, encoding: .utf8)
        try String(repeating: "zebra quokka ", count: 10)
            .write(to: kb.appendingPathComponent("target-a.md"),
                   atomically: true, encoding: .utf8)
        try String(repeating: "volcano glacier ", count: 10)
            .write(to: kb.appendingPathComponent("target-b.md"),
                   atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragSemanticChunking = false
        config.ragChunkSize = 20
        config.ragChunkOverlap = 0

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagKeywordEmbedding(), reranker: nil)
        let stats = await engine.ingest()
        XCTAssertEqual(stats.processed, 3)
        XCTAssertGreaterThan(stats.chunks, 4_096,
                             "需要超过 maxVectorK 的其他来源切片挤占全局召回")

        // 唯一匹配：直接解析成精确来源，绕开全局 KNN 截断。
        let scoped = try await engine.search(query: "apple", topK: 1, source: "target-a")
        XCTAssertEqual(scoped.count, 1, "来源子串必须在该来源内定向召回")
        XCTAssertEqual(scoped.first.map { ($0.source as NSString).lastPathComponent },
                       "target-a.md")
        XCTAssertTrue(scoped.first?.text.contains("zebra") == true)

        // 多个匹配：每个来源各自定向检索，再按 RRF 合并。
        let grouped = try await engine.search(query: "apple", topK: 2, source: "target")
        XCTAssertEqual(Set(grouped.map { ($0.source as NSString).lastPathComponent }),
                       ["target-a.md", "target-b.md"],
                       "多来源子串必须覆盖所有匹配来源")
        engine.close()
    }

    /// 英文正文按更安全的 4 字符/token 换算：旧实现按 1.5 字符/token
    /// 估算，会把 2k+ 字符的长切片截在 1500 字符处。
    func testEnglishChunkNotTruncatedAtLegacyCharLimit() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-english-limit-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let body = String(repeating: "retrieval augmented generation ", count: 70)
            + "TAILMARKERXYZ"
        XCTAssertGreaterThan(body.count, 1_500, "正文必须超过旧的 1500 字符上限")
        XCTAssertLessThan(body.count, 800 * 4, "正文必须能放进一个 800 token 切片")
        try body.write(to: kb.appendingPathComponent("english.md"),
                       atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragSemanticChunking = false
        config.ragChunkSize = 800
        config.ragChunkOverlap = 0

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagExactTokenEmbedding(), reranker: nil)
        let stats = await engine.ingest()
        XCTAssertEqual(stats.chunks, 1, "英文正文应完整落入一个切片")

        let formatted = try await engine.searchFormatted(query: "retrieval", topK: 1,
                                                         maxTotalChars: 20_000)
        XCTAssertTrue(formatted.contains("TAILMARKERXYZ"),
                      "长英文切片不得被固定的 1500 字符上限截短")
        engine.close()
    }

    /// 旧库/外部清理可能没有 manifest：来源子串解析要能回退到索引内的来源名单，
    /// 不能因为读不到 manifest 就把真实存在的来源判成不存在。
    func testSourceSearchFallsBackToIndexWhenManifestMissing() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-source-fallback-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try String(repeating: "zebra quokka ", count: 10)
            .write(to: kb.appendingPathComponent("target-a.md"),
                   atomically: true, encoding: .utf8)
        try String(repeating: "apple banana ", count: 10)
            .write(to: kb.appendingPathComponent("other.md"),
                   atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragSemanticChunking = false
        // 每个来源单切片，便于直接按切片数断言。
        config.ragChunkSize = 500
        config.ragChunkOverlap = 0

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagKeywordEmbedding(), reranker: nil)
        let stats = await engine.ingest()
        XCTAssertEqual(stats.chunks, 2)

        // 模拟旧库：库内切片仍在，manifest 完全缺失。
        let manifestURL = URL(fileURLWithPath: config.ragIndexDirectory)
            .appendingPathComponent("manifest.json")
        try FileManager.default.removeItem(at: manifestURL)
        XCTAssertFalse(FileManager.default.fileExists(atPath: manifestURL.path))

        let hit = try await engine.search(query: "zebra", topK: 1, source: "target-a")
        XCTAssertEqual(hit.count, 1, "manifest 缺失时必须回退到索引来源名单")
        XCTAssertEqual(hit.first.map { ($0.source as NSString).lastPathComponent },
                       "target-a.md")
        engine.close()
    }

    /// #9：辅助模型只在「多文档」或「接近上限的超长结果」时精简，
    /// 单文档中等长度结果原样返回，避免每次 rag_search 都多一次模型调用。
    func testAuxCondenseOnlyTriggersForMultiDocumentOrVeryLongResults() {
        var config = AgentConfig()
        config.ragAuxSummarizeMinChars = 5_000
        config.ragAuxSummarizeMaxChars = 20_000

        XCTAssertFalse(RAGEngine.shouldCondense(outputChars: 4_999, documentCount: 3,
                                                config: config))
        XCTAssertFalse(RAGEngine.shouldCondense(outputChars: 5_000, documentCount: 1,
                                                config: config),
                       "单文档中等长度结果不调用辅助模型")
        XCTAssertTrue(RAGEngine.shouldCondense(outputChars: 5_000, documentCount: 2,
                                               config: config))
        XCTAssertFalse(RAGEngine.shouldCondense(outputChars: 7_499, documentCount: 1,
                                                config: config))
        XCTAssertTrue(RAGEngine.shouldCondense(outputChars: 7_500, documentCount: 1,
                                               config: config),
                      "接近上限的超长结果即使单文档也摘要")
        XCTAssertTrue(RAGEngine.shouldCondense(outputChars: 20_000, documentCount: 1,
                                               config: config))
        XCTAssertFalse(RAGEngine.shouldCondense(outputChars: 20_001, documentCount: 2,
                                                config: config),
                       "超过上限不做精简")
    }

    /// 分片必须严格不超过上限，且保持原有换行语义；单行超限时硬切。
    func testCondenseSplitByLinesRespectsLimit() {
        let text = (0..<40).map { "第\($0)行内容" }.joined(separator: "\n")
        let chunks = RAGEngine.splitByLines(text, limit: 50)
        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks {
            XCTAssertLessThanOrEqual(chunk.count, 50)
        }
        XCTAssertEqual(chunks.joined(separator: "\n"), text, "切分不丢内容")

        let longLine = String(repeating: "字", count: 120)
        let hardCut = RAGEngine.splitByLines(longLine, limit: 50)
        XCTAssertEqual(hardCut.count, 3)
        XCTAssertTrue(hardCut.allSatisfy { $0.count <= 50 })
        XCTAssertEqual(hardCut.joined(), longLine)
    }

    /// 知识图谱输出按 `【标题】` 分节，超长节再按行切。
    func testCondenseSectionsFromHeadings() {
        let text = """
        从知识图谱检索到 3 个实体、2 条关系
        【实体】
        居鲁士
        大流士
        【关系】
        居鲁士 --建立--> 阿契美尼德
        【原文片段】
        \(String(repeating: "片段内容", count: 2000))
        """
        let sections = RAGEngine.sectionsFromHeadings(text, limit: 800)
        let titles = Set(sections.map(\.title))
        XCTAssertTrue(titles.contains("实体"))
        XCTAssertTrue(titles.contains("关系"))
        XCTAssertTrue(titles.contains("原文片段"))
        XCTAssertGreaterThan(sections.filter { $0.title == "原文片段" }.count, 1,
                             "超长节必须切成多片")
        XCTAssertTrue(sections.allSatisfy { $0.text.count <= 800 })
    }

    /// 章节聚合：同一文档同一 heading 归为一章，不同文档/heading 分开。
    func testCondenseSectionsGroupsByDocumentAndHeading() {
        let hits = [
            SearchHit(rowID: 1, text: "A1", source: "/kb/a.md", chunkIndex: 0,
                      heading: "第一章"),
            SearchHit(rowID: 2, text: "A2", source: "/kb/a.md", chunkIndex: 1,
                      heading: "第一章"),
            SearchHit(rowID: 3, text: "B1", source: "/kb/b.md", chunkIndex: 0),
            SearchHit(rowID: 4, text: "A3", source: "/kb/a.md", chunkIndex: 2,
                      heading: "第二章"),
        ]
        let sections = RAGEngine.condenseSections(hits: hits, limit: 4_000)
        XCTAssertEqual(sections.map(\.title),
                       ["a.md / 第一章", "b.md", "a.md / 第二章"])
        XCTAssertEqual(sections[0].text, "A1\nA2")
        XCTAssertEqual(sections[1].text, "B1")
    }

    /// 命中多文档且长度达阈值时，`searchFormatted` 必须把输出交给注入的整理器。
    func testSearchFormattedUsesAuxCondenserWhenEligible() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-condense-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        // 两份文档正文必须不同：ingest 会按内容哈希去重，完全相同的副本只入库一份。
        try String(repeating: "apple banana ", count: 40)
            .write(to: kb.appendingPathComponent("a.md"), atomically: true, encoding: .utf8)
        try String(repeating: "apple cherry ", count: 40)
            .write(to: kb.appendingPathComponent("b.md"), atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragSemanticChunking = false
        config.ragChunkSize = 800
        config.ragChunkOverlap = 0
        config.ragSearchMaxCharsPerResult = 1_000
        config.ragSearchMaxTotalChars = 2_000
        config.ragAuxSummarizeMinChars = 100
        config.ragAuxSummarizeMaxChars = 2_000

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagFakeEmbedding(), reranker: nil)
        _ = await engine.ingest()
        var calls = 0
        var stages: [RAGCondenseStage] = []
        engine.auxCondenser = { input, stage in
            calls += 1
            stages.append(stage)
            return "摘要:\n\(input.prefix(20))\n原文片段:\n---"
        }
        let output = try await engine.searchFormatted(query: "apple", topK: 2)
        // 两个文档 → 2 次章节摘要（map）+ 1 次整合（reduce）。
        XCTAssertEqual(calls, 3)
        XCTAssertEqual(stages.filter { $0 == .chapter }.count, 2)
        XCTAssertEqual(stages.filter { $0 == .merge }.count, 1)
        XCTAssertTrue(output.hasPrefix("摘要:"))
        engine.close()
    }

    /// 检索即展示：全部召回切片进 citations（top-1 标记 isPrimary），
    /// 但上下文只含 1 条切片正文；空结果不产 citations。
    func testSearchWithCitationsKeepsTopOneInContext() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-citations-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        // 两份文档正文必须不同：ingest 会按内容哈希去重。
        try String(repeating: "apple banana ", count: 40)
            .write(to: kb.appendingPathComponent("a.md"), atomically: true, encoding: .utf8)
        try String(repeating: "apple cherry ", count: 40)
            .write(to: kb.appendingPathComponent("b.md"), atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragSemanticChunking = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 5

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagFakeEmbedding(), reranker: nil)
        _ = await engine.ingest()
        let outcome = try await engine.searchWithCitations(query: "apple", topK: 5)
        // 全部召回切片都作为气泡卡片返回，只有第一个标记 isPrimary。
        XCTAssertGreaterThan(outcome.citations.count, 1)
        XCTAssertEqual(outcome.citations.filter(\.isPrimary).count, 1)
        XCTAssertTrue(outcome.citations[0].isPrimary)
        for (index, citation) in outcome.citations.enumerated() {
            XCTAssertEqual(citation.rank, index + 1)
            if index > 0 { XCTAssertFalse(citation.isPrimary) }
        }
        // 上下文只有一条切片正文：不出现第二个 "### 来源:"。
        XCTAssertEqual(outcome.context.components(separatedBy: "### 来源:").count - 1, 1)
        XCTAssertTrue(outcome.context.contains("检索即展示"))
        XCTAssertTrue(outcome.context.contains(outcome.citations[0].text.prefix(10)))

        // 指定来源无匹配（明确返回空，不退化成全局检索）：兜底文案，无气泡。
        let empty = try await engine.searchWithCitations(query: "apple",
                                                         source: "not-exist.md")
        XCTAssertEqual(empty.context, "（知识库中未找到相关内容）")
        XCTAssertTrue(empty.citations.isEmpty)
        engine.close()
    }

    /// rag_search 工具：召回切片通过 onRetrievedChunks 推给 UI，
    /// 工具输出（进上下文）只含 top-1 一条切片正文。
    func testRagSearchToolEmitsCitationsAndSingleChunkContext() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-tool-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try String(repeating: "apple banana ", count: 40)
            .write(to: kb.appendingPathComponent("a.md"), atomically: true, encoding: .utf8)
        try String(repeating: "apple cherry ", count: 40)
            .write(to: kb.appendingPathComponent("b.md"), atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragSemanticChunking = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 5

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagFakeEmbedding(), reranker: nil)
        _ = await engine.ingest()

        final class CitationRecorder: AgentCallbacks {
            var received: [[RAGCitation]] = []
            func onRetrievedChunks(_ chunks: [RAGCitation]) { received.append(chunks) }
        }
        let recorder = CitationRecorder()
        let tool = RagSearchTool(config: config, callbacks: recorder, registry: nil,
                                 engine: engine)
        let result = try await tool.run(["query": .string("apple"), "top_k": .int(5)])
        XCTAssertTrue(result.success)
        // 一批切片回调给 UI，含 top-1。
        XCTAssertEqual(recorder.received.count, 1)
        XCTAssertGreaterThan(recorder.received[0].count, 1)
        XCTAssertTrue(recorder.received[0][0].isPrimary)
        // 进上下文的输出只有一条切片正文（不含其余切片）。
        let body = result.toMessage()
        XCTAssertEqual(body.components(separatedBy: "### 来源:").count - 1, 1)
        engine.close()
    }
}
