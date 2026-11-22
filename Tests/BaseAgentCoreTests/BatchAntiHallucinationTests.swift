import XCTest
@testable import BaseAgentCore

/// 批量任务防幻觉：system prompt 范式 + 工具层完成凭证/相近文件名提示。
final class BatchAntiHallucinationTests: XCTestCase {
    private final class PlainLLM: LLMService {
        let identifier = "anti-hallu"
        let supportsTools = false
        let contextWindow = 4096
        func chatStream(messages: [ChatMessage], tools: [JSONValue],
                        temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
            AsyncThrowingStream { c in c.yield(LLMStreamEvent(kind: .done)); c.finish() }
        }
    }

    private func makeDir() throws -> URL {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("anti-hallu-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func makeCodeTool(_ workspace: URL) -> CodeRunTool {
        var config = AgentConfig()
        config.workspace = workspace.path
        config.llmProvider = "api"
        return CodeRunTool(config: config, callbacks: NoopCallbacks())
    }

    private func makeFileTool(_ workspace: URL) -> FileReadTool {
        var config = AgentConfig()
        config.workspace = workspace.path
        config.llmProvider = "api"
        return FileReadTool(config: config, callbacks: NoopCallbacks())
    }

    /// 空输出：明确提示未产生完成确认，禁止声称完成。
    func testCodeRunEmptyOutputWarns() async throws {
        let root = try makeDir(); defer { try? FileManager.default.removeItem(at: root) }
        let result = try await makeCodeTool(root).run(["code": .string("pass")])
        XCTAssertTrue(result.success)
        XCTAssertTrue(result.output.contains("没有任何输出"))
        XCTAssertTrue(result.output.contains("禁止据此声称完成"))
    }

    /// 成功退出但无完成计数：追加轻量提示。
    func testCodeRunNoCountWarns() async throws {
        let root = try makeDir(); defer { try? FileManager.default.removeItem(at: root) }
        let result = try await makeCodeTool(root).run([
            "code": .string("print('hello')"),
        ])
        XCTAssertTrue(result.success)
        XCTAssertTrue(result.output.contains("没有完成计数"))
    }

    /// 有明确完成计数：保持原样，不打扰。
    func testCodeRunWithCountNoWarning() async throws {
        let root = try makeDir(); defer { try? FileManager.default.removeItem(at: root) }
        let result = try await makeCodeTool(root).run([
            "code": .string("print('DONE count=120')"),
        ])
        XCTAssertTrue(result.output.contains("DONE count=120"))
        XCTAssertFalse(result.output.contains("没有完成计数"))
    }

    /// 中文文件名拼错：给出真实相近文件名供自纠。
    func testFileReadErrorSuggestsNearestName() async throws {
        let root = try makeDir(); defer { try? FileManager.default.removeItem(at: root) }
        let real = "兴亡的世界史 18 - 平野聪.md"
        try Data("内容".utf8).write(to: root.appendingPathComponent(real))
        let result = try await makeFileTool(root).run([
            "path": .string(real.replacingOccurrences(of: "平野聪", with: "平野 聰")),
        ])
        XCTAssertFalse(result.success)
        XCTAssertTrue(result.error.contains("最接近的真实文件名"), "error=\(result.error)")
        XCTAssertTrue(result.error.contains(real), "error=\(result.error)")
    }

    /// system prompt 内置常量含大批量任务范式与完成判定规则。
    func testSystemPromptHasBatchProtocol() throws {
        XCTAssertTrue(BASE_SYSTEM_PROMPT.contains("大批量任务"))
        XCTAssertTrue(BASE_SYSTEM_PROMPT.contains("禁止逐文件调用工具"))
        XCTAssertTrue(BASE_SYSTEM_PROMPT.contains("只有脚本输出明确的完成计数"))
        XCTAssertTrue(BASE_SYSTEM_PROMPT.contains("禁止凭记忆编造中文路径"))
    }
}

private final class NoopCallbacks: AgentCallbacks {}
