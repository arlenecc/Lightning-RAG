import XCTest
import PDFKit
import CoreText
import AppKit
@testable import BaseAgentCore

private final class ScriptedAskLLM: LLMService, @unchecked Sendable {
    let identifier = "scripted-ask"
    private let lock = NSLock()
    private var call = 0

    func chatStream(messages: [ChatMessage], tools: [JSONValue],
                    temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        lock.lock(); let index = call; call += 1; lock.unlock()
        return AsyncThrowingStream { continuation in
            if index == 0 {
                continuation.yield(LLMStreamEvent(
                    kind: .done,
                    toolCalls: [ToolCall(id: "ask_1", name: "ask_user",
                                         arguments: ["prompt": .string("which value?")],
                                         rawArguments: "{\"prompt\":\"which value?\"}")]))
            } else {
                continuation.yield(LLMStreamEvent(kind: .content, content: "finished"))
                continuation.yield(LLMStreamEvent(kind: .done))
            }
            continuation.finish()
        }
    }
}

private final class AnswerCallbacks: AgentCallbacks {
    func askUser(_ prompt: String) -> String? { "42" }
}

final class RegressionTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-reg-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testL3FoldDeduplicatesByBodyIgnoringTurnPrefix() {
        let lines = (1...4).map { "[T\($0)] 重复的摘要" }
        let folded = l3Fold(lines, maxLines: 40, dedupeWindow: 10)
        XCTAssertTrue(folded.contains("4 turns"), folded)
        XCTAssertEqual(folded.components(separatedBy: "\n").count, 1)
    }

    func testOutlineSkipsFencedCodeAndSentence() {
        let text = """
        # 真标题

        ```swift
        # 不是标题
        1. 也不是标题
        ```

        第二章介绍方法。这里继续正文。
        """
        let nodes = DocumentOutline.extract(text)
        XCTAssertEqual(nodes.map(\.title), ["真标题"])
    }

    func testCheckpointSynchronizesSession() async throws {
        var config = AgentConfig()
        config.workspace = directory.path
        let session = Session(budget: ContextBudget(total: 30_000))
        let registry = ToolRegistry(config: config)
        registry.session = session
        let tool = UpdateWorkingCheckpointTool(config: config, callbacks: nil, registry: registry)

        let result = try await tool.run(["key_info": .string("port=8080"), "persist": .bool(true)])
        XCTAssertTrue(result.success)
        XCTAssertEqual(session.getKeyInfo(), "port=8080")
        XCTAssertEqual(CheckpointStore.readPersisted(config), "port=8080")

        _ = try await tool.run(["key_info": .string(""), "persist": .bool(false)])
        XCTAssertTrue(session.getKeyInfo().isEmpty)
        XCTAssertTrue(CheckpointStore.readPersisted(config).isEmpty)
    }

    /// 超长便签不再直接截断：先走注入的压缩器（LLM 压缩）；压缩器不可用
    /// 或输出仍超长时，保头保尾并显式标注中间省略，总长仍受 maxKeyInfo 约束。
    func testCheckpointOverlongCompressesInsteadOfTruncating() async throws {
        var config = AgentConfig()
        config.workspace = directory.path
        let session = Session(budget: ContextBudget(total: 30_000))
        let registry = ToolRegistry(config: config)
        registry.session = session
        let long = String(repeating: "A", count: 3_000) + "B"
            + String(repeating: "C", count: 2_000)

        // 压缩器正常：直接采用压缩结果，不丢任何内容。
        let compressed = UpdateWorkingCheckpointTool(
            config: config, callbacks: nil, registry: registry,
            compressor: { _ in "compressed-summary" })
        let result = try await compressed.run(["key_info": .string(long)])
        XCTAssertTrue(result.success)
        XCTAssertEqual(session.getKeyInfo(), "compressed-summary")

        // 压缩器不可用（nil）：保头保尾，中间显式省略。
        let fallback = UpdateWorkingCheckpointTool(config: config, callbacks: nil, registry: registry)
        let result2 = try await fallback.run(["key_info": .string(long)])
        XCTAssertTrue(result2.success)
        let stored = session.getKeyInfo()
        XCTAssertLessThanOrEqual(stored.count, CheckpointStore.maxKeyInfo)
        XCTAssertTrue(stored.hasPrefix(String(repeating: "A", count: 100)))
        XCTAssertTrue(stored.contains("[…中间已省略]"))
        XCTAssertTrue(stored.hasSuffix("CCC"))

        // 压缩器输出仍超长：同样退回保头保尾。
        let badCompressor = UpdateWorkingCheckpointTool(
            config: config, callbacks: nil, registry: registry,
            compressor: { _ in String(repeating: "X", count: 5_000) })
        let result3 = try await badCompressor.run(["key_info": .string(long)])
        XCTAssertTrue(result3.success)
        XCTAssertLessThanOrEqual(session.getKeyInfo().count, CheckpointStore.maxKeyInfo)
    }

    func testAskUserAnswerIsInjectedAndRunContinues() async {
        var config = AgentConfig()
        config.workspace = directory.path
        config.llmProvider = "api"
        config.maxIterations = 3
        let agent = Agent(llm: ScriptedAskLLM(), config: config, callbacks: AnswerCallbacks())
        let answer = await agent.run("need value")
        XCTAssertTrue(answer.hasPrefix("finished"), "实际答复：\(answer)")
        agent.close()
    }

    func testCodeRunBlocksWritesOutsideWorkspace() async throws {
        var config = AgentConfig()
        config.workspace = directory.path
        let tool = CodeRunTool(config: config, callbacks: NoopCallbacks())
        let outside = directory.deletingLastPathComponent()
            .appendingPathComponent("outside-\(UUID().uuidString).txt").path
        let escaped = outside.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        let code = "open(\"\(escaped)\", \"w\").write(\"bad\")"
        let result = try await tool.run(["code": .string(code), "timeout": .double(20)])
        XCTAssertFalse(result.success)
        XCTAssertFalse(FileManager.default.fileExists(atPath: outside))
    }

    func testCodeRunAllowsWorkspaceWriteAndSubprocess() async throws {
        var config = AgentConfig()
        config.workspace = directory.path
        let tool = CodeRunTool(config: config, callbacks: NoopCallbacks())
        let code = """
        import subprocess
        open("inside.txt", "w").write("ok")
        print(subprocess.run(["echo", "child-ok"], capture_output=True, text=True).stdout.strip())
        """
        let result = try await tool.run(["code": .string(code), "timeout": .double(20)])
        XCTAssertTrue(result.success, result.error)
        XCTAssertTrue(result.output.contains("child-ok"))
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("inside.txt").path))
    }

    func testSessionDigestOverflowKeepsRealLinesAndMarker() {
        let session = Session(budget: ContextBudget(total: 4_000),
                              rawTurns: 1, maxDigestLines: 5)
        for index in 1...7 {
            session.appendDigest("line-\(index)")
        }
        XCTAssertLessThanOrEqual(session.extraDigest.count, 5)
        XCTAssertTrue(session.extraDigest.first?.contains("已折叠省略") == true,
                      session.extraDigest.first ?? "")
        // 前 3 条被省略，4~7 必须完整保留；不能因为写入标记再多丢一条。
        XCTAssertEqual(Array(session.extraDigest.suffix(4)),
                       ["line-4", "line-5", "line-6", "line-7"])
    }

    func testDocumentCleanerPreservesCodeFenceCJKAndBodyNumbers() {
        let input = """
        <!-- page 1 -->
        标题
        第一行
        第二行
        2024
        第三行
        第四行
        第五行

        ```swift
        let value = 1
        print("这是一个示例")
        ```

        <!-- page 2 -->
        """
        let cleaned = DocumentCleaner.clean(input)
        XCTAssertTrue(cleaned.contains("```swift"), cleaned)
        XCTAssertTrue(cleaned.contains("let value = 1"), cleaned)
        XCTAssertFalse(cleaned.contains("CODE0"), cleaned)
        XCTAssertTrue(cleaned.contains("第一行第二行"), cleaned)
        XCTAssertTrue(cleaned.contains("2024"), "正文中的年份不应被当作页码删除")
    }

    func testSkillIndexFallsBackToSkillMarkdownEntry() throws {
        let skillsDirectory = directory.appendingPathComponent("skills")
        let skillDirectory = skillsDirectory.appendingPathComponent("demo")
        try FileManager.default.createDirectory(at: skillDirectory, withIntermediateDirectories: true)
        try """
        {"name":"demo","description":"demo skill","entry":"prompt.md"}
        """.write(to: skillDirectory.appendingPathComponent("skill.json"),
                  atomically: true, encoding: .utf8)
        try "# Demo\nbody".write(to: skillDirectory.appendingPathComponent("SKILL.md"),
                                 atomically: true, encoding: .utf8)

        let index = SkillIndex(skillsDirectory: skillsDirectory.path)
        _ = index.scan(force: true)
        let entries = index.listAll()
        XCTAssertEqual(entries.count, 1)
        XCTAssertEqual(entries.first?.entry, "SKILL.md")
        XCTAssertTrue(index.readSkillContent(path: "demo")?.contains("body") == true)
    }

    func testVectorStoreRebuildsEmptyStoreForLateDiscoveredDimension() throws {
        let store = try VectorStore(dbPath: directory.appendingPathComponent("late.sqlite").path,
                                    dimension: 768)
        let chunks = [Chunk(text: "hello world", source: "a.md", chunkIndex: 0)]
        XCTAssertEqual(try store.add(chunks: chunks, vectors: [[1, 0, 0]]), 1)
        XCTAssertEqual(store.dimension, 3)
        XCTAssertTrue(store.didResetForDimensionChange)
        let hits = try store.search(vector: [1, 0, 0], topK: 1)
        XCTAssertEqual(hits.first?.text, "hello world")
    }

    func testOvisUsesTextLayerWithoutCallingVLM() throws {
        let url = directory.appendingPathComponent("text.pdf")
        var mediaBox = CGRect(x: 0, y: 0, width: 612, height: 792)
        guard let consumer = CGDataConsumer(url: url as CFURL),
              let context = CGContext(consumer: consumer, mediaBox: &mediaBox, nil) else {
            return XCTFail("cannot create pdf context")
        }
        context.beginPDFPage(nil)
        let text = String(repeating: "Digital text layer paragraph. ", count: 10)
        let attributed = NSAttributedString(string: text, attributes: [
            .font: NSFont.systemFont(ofSize: 20),
        ])
        let framesetter = CTFramesetterCreateWithAttributedString(attributed)
        let path = CGPath(rect: CGRect(x: 40, y: 60, width: 532, height: 660), transform: nil)
        let frame = CTFramesetterCreateFrame(framesetter, CFRange(location: 0, length: 0), path, nil)
        CTFrameDraw(frame, context)
        context.endPDFPage()
        context.closePDF()

        let fake = CountingVLM()
        let parser = OvisPDFParser(vlm: fake)
        let sem = DispatchSemaphore(value: 0)
        var parsed: ParsedDocument?
        Task {
            parsed = try? await parser.parse(url: url, options: ParserOptions(pdfEngine: .ovis), progress: nil)
            sem.signal()
        }
        _ = sem.wait(timeout: .now() + 30)
        XCTAssertNotNil(parsed)
        XCTAssertTrue(parsed?.text.contains("Digital text layer") == true)
        XCTAssertEqual(fake.calls, 0, "文本层充足时不应调用 Ovis")
    }

    func testSummaryStreamFilterSeparatesSplitTags() {
        var filter = SummaryStreamFilter()
        var content = ""
        var summary = ""
        let chunks = ["<sum", "mary>用户要求确认任务需求", "，这是计划的第一步</sum",
                      "mary>好的，我来确认。"]
        for chunk in chunks {
            let out = filter.feed(chunk)
            content += out.content
            summary += out.summary
        }
        let tail = filter.finish()
        content += tail.content
        summary += tail.summary

        XCTAssertEqual(content, "好的，我来确认。")
        XCTAssertEqual(summary.trimmingCharacters(in: .whitespacesAndNewlines),
                       "用户要求确认任务需求，这是计划的第一步")
    }

    func testSummaryStreamFilterHandlesMultipleAndUnclosedSummaries() {
        var filter = SummaryStreamFilter()
        let first = filter.feed("<summary>第一步</summary>正文A<summary>第二步")
        XCTAssertEqual(first.content, "正文A")
        XCTAssertEqual(first.summary, "第一步\n")
        let tail = filter.finish()
        XCTAssertEqual(tail.summary, "第二步\n")
        XCTAssertTrue(tail.content.isEmpty)
    }

    /// 回归：几十个 <summary> 块被剥离后，块间换行不得在正文里累积成大段空白。
    func testSummaryStreamFilterCollapsesWhitespaceAroundSummaries() {
        var filter = SummaryStreamFilter()
        var content = ""
        var summary = ""
        let chunk = "概览。\n\n---\n\n"
            + (1...30).map { "<summary>结构要点\($0)</summary>\n\n" }.joined()
            + "---\n\n## 标题\n"
        let out = filter.feed(chunk)
        content += out.content
        summary += out.summary
        let tail = filter.finish()
        content += tail.content
        summary += tail.summary

        XCTAssertEqual(content, "概览。\n\n---\n\n---\n\n## 标题")
        XCTAssertEqual(summary.split(separator: "\n").count, 30)
    }

    func testSkillRequestClassificationSkipsGreetingsAndSimpleQueries() {
        XCTAssertTrue(SkillManager.isTrivialRequest("你好"))
        XCTAssertTrue(SkillManager.isTrivialRequest("今天天气怎么样？"))
        XCTAssertTrue(SkillManager.isTrivialRequest("你是谁"))
        XCTAssertFalse(SkillManager.isTrivialRequest("帮我写一个脚本"))
        XCTAssertFalse(SkillManager.isTrivialRequest("修复这个文件的编译错误"))
        XCTAssertFalse(SkillManager.isTrivialRequest("你好，帮我分析这个项目"))
    }

    func testGreetingClassificationForToolFreeRounds() {
        for text in ["你好", "你好呀！", "谢谢", "thanks", "在吗", "再见~", "好的", "嗯呢"] {
            XCTAssertTrue(SkillManager.isGreeting(text), "应为寒暄: \(text)")
        }
        for text in ["你好，帮我分析这个项目", "查一下 index.sqlite 有多少行",
                     "把测试跑一遍", "写个快速排序"] {
            XCTAssertFalse(SkillManager.isGreeting(text), "不应视为寒暄: \(text)")
        }
    }
}

private final class CountingVLM: VLMService {
    let identifier = "counting"
    let isAvailable = true
    var calls = 0
    func recognize(image: CGImage, prompt: String, maxTokens: Int) async throws -> String {
        calls += 1
        return "vlm"
    }
}
