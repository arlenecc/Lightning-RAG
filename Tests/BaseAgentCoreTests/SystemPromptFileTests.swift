import XCTest
@testable import BaseAgentCore

private final class PromptTestLLM: LLMService, @unchecked Sendable {
    let identifier = "prompt-test"
    let supportsTools = true
    let supportsConcurrentIsolatedRequests = false
    func chatStream(messages: [ChatMessage], tools: [JSONValue],
                    temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        AsyncThrowingStream { $0.yield(.init(kind: .done)); $0.finish() }
    }
}

private final class PromptTestCallbacks: AgentCallbacks {}

/// `<workspace>/assets/system_prompt.txt` 的首启写出 / 每次启动加载契约。
final class SystemPromptFileTests: XCTestCase {
    private var workspace: String!
    private var file: String { SystemPromptFile.path(in: workspace) }

    override func setUpWithError() throws {
        workspace = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("system-prompt-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: workspace,
                                                withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(atPath: workspace)
    }

    func testPathIsUnderWorkspaceAssets() {
        XCTAssertEqual(file, workspace + "/assets/system_prompt.txt")
    }

    func testFirstLaunchCreatesEditableFileWithBuiltinPrompt() throws {
        let first = SystemPromptFile.resolve(workspace: workspace)
        XCTAssertEqual(first.status, .created)
        XCTAssertEqual(first.path, file)
        XCTAssertEqual(first.text, BASE_SYSTEM_PROMPT)
        XCTAssertEqual(try String(contentsOfFile: file, encoding: .utf8),
                       BASE_SYSTEM_PROMPT + "\n")
        XCTAssertTrue(FileManager.default.fileExists(atPath: workspace + "/assets"))
    }

    /// 模拟「首启 → 用户编辑 → 重启」：第二次必须读到编辑后的内容，且不覆盖文件。
    func testSecondLaunchLoadsEditedTextWithoutOverwriting() throws {
        _ = SystemPromptFile.resolve(workspace: workspace)
        let custom = "You are 猫图鉴助手.\n只用中文回答，禁止调用工具。"
        try custom.write(toFile: file, atomically: true, encoding: .utf8)

        let second = SystemPromptFile.resolve(workspace: workspace)
        XCTAssertEqual(second.status, .loaded)
        XCTAssertEqual(second.text, custom)
        XCTAssertEqual(try String(contentsOfFile: file, encoding: .utf8), custom)
        XCTAssertFalse(second.text.contains("You are base-agent"))
    }

    /// 编辑器常在末尾补换行；裁剪首尾空白才能让 system 前缀逐字节稳定，
    /// 否则每次启动都会让 KV cache 全量重建。
    func testTrailingWhitespaceDoesNotChangeEffectivePrompt() throws {
        _ = SystemPromptFile.resolve(workspace: workspace)
        let custom = "# 我的规则\n回答一律用粤语。"
        try (custom + "\n\n  ").write(toFile: file, atomically: true, encoding: .utf8)
        XCTAssertEqual(SystemPromptFile.resolve(workspace: workspace).text, custom)
    }

    func testBlankFileFallsBackToBuiltinAndKeepsUserFile() throws {
        _ = SystemPromptFile.resolve(workspace: workspace)
        let blank = "  \n\n  "
        try blank.write(toFile: file, atomically: true, encoding: .utf8)

        let resolved = SystemPromptFile.resolve(workspace: workspace)
        XCTAssertEqual(resolved.status, .fallback)
        XCTAssertEqual(resolved.text, BASE_SYSTEM_PROMPT)
        XCTAssertFalse(resolved.note.isEmpty)
        XCTAssertEqual(try String(contentsOfFile: file, encoding: .utf8), blank,
                       "回退不得覆盖用户文件")
    }

    func testRepeatedLaunchWithUnchangedFileYieldsIdenticalPrefix() {
        let first = SystemPromptFile.resolve(workspace: workspace)
        let second = SystemPromptFile.resolve(workspace: workspace)
        XCTAssertEqual(first.status, .created)
        XCTAssertEqual(second.status, .loaded)
        XCTAssertEqual(first.text, second.text)
    }

    /// 完整启动链路：加载文件 → 注入 Agent → `systemPrompt()` 用它替换内置主体，
    /// 但 L0/L1 多级记忆段仍照常追加。
    func testAgentUsesLoadedPromptAndStillAppendsMemoryLayers() throws {
        _ = SystemPromptFile.resolve(workspace: workspace)
        let custom = "# 自定义行为规则\n始终用简体中文回答。"
        try custom.write(toFile: file, atomically: true, encoding: .utf8)
        let loaded = SystemPromptFile.resolve(workspace: workspace)

        var config = AgentConfig()
        config.workspace = workspace
        let agent = Agent(llm: PromptTestLLM(), config: config,
                          callbacks: PromptTestCallbacks(),
                          baseSystemPrompt: loaded.text)
        defer { agent.close() }

        let system = agent.systemPrompt()
        XCTAssertTrue(system.hasPrefix(custom))
        XCTAssertFalse(system.contains("You are base-agent"))
        XCTAssertTrue(system.contains("# [L0 META-SOP: memory_management_sop]"))
        XCTAssertTrue(system.contains("# [L1 Global Memory Insight]"))
        // 段内冻结：同一份前缀逐字节稳定，KV 才能复用。
        XCTAssertEqual(agent.systemPrompt(), system)
    }

    /// Core 层默认仍用内置提示词：不注入时不读磁盘，避免测试受用户真实文件影响。
    func testAgentDefaultsToBuiltinPrompt() {
        var config = AgentConfig()
        config.workspace = workspace
        let agent = Agent(llm: PromptTestLLM(), config: config,
                          callbacks: PromptTestCallbacks())
        defer { agent.close() }
        XCTAssertTrue(agent.systemPrompt().hasPrefix(BASE_SYSTEM_PROMPT))
        XCTAssertFalse(FileManager.default.fileExists(atPath: file))
    }
}
