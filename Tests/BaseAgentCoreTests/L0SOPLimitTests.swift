import XCTest
@testable import BaseAgentCore

/// L0 META-SOP 常驻 system 的字符上限（当前 3000），含超长文件截断验证。
final class L0SOPLimitTests: XCTestCase {
    private final class PlainLLM: LLMService {
        let identifier = "l0-limit-llm"
        let supportsTools = false
        let contextWindow = 4096
        func chatStream(messages: [ChatMessage], tools: [JSONValue],
                        temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
            AsyncThrowingStream { c in c.yield(LLMStreamEvent(kind: .done)); c.finish() }
        }
    }

    private func makeWorkspace(withSOP sop: String?) throws -> (Agent, URL) {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("l0-limit-\(UUID().uuidString)")
        let mem = root.appendingPathComponent("memory")
        try FileManager.default.createDirectory(at: mem, withIntermediateDirectories: true)
        if let sop {
            try sop.write(to: mem.appendingPathComponent("memory_management_sop.md"),
                          atomically: true, encoding: .utf8)
        }
        var config = AgentConfig()
        config.workspace = root.path
        config.llmProvider = "api"
        let agent = Agent(llm: PlainLLM(), config: config, callbacks: NoopCallbacks())
        return (agent, root)
    }

    /// 从完整 system 里取出纯 L0 段（L0 标题之后、L1 标题之前）。
    private func l0Section(_ system: String) -> String? {
        let marker = "# [L0 META-SOP: memory_management_sop]\n"
        guard let range = system.range(of: marker) else { return nil }
        let after = String(system[range.upperBound...])
        if let l1 = after.range(of: "\n\n# [L1 Global Memory Insight]") {
            return String(after[..<l1.lowerBound])
        }
        return after
    }

    func testLimitIs3000() {
        XCTAssertEqual(SystemPromptLimits.maxSOPChars, 3_000)
    }

    /// 超长 L0 文件：常驻 system 里的 L0 段必须被截到 3000 字符，
    /// 且 BASE 与 L1 仍在正确位置（拼装顺序不变）。
    func testLongSOPTruncatedTo3000() throws {
        let longSOP = String(repeating: "记忆规范内容甲乙丙丁。", count: 500)  // 远超 3000
        let (agent, root) = try makeWorkspace(withSOP: longSOP)
        defer { try? FileManager.default.removeItem(at: root) }
        let system = agent.systemPrompt()
        guard let l0 = l0Section(system) else {
            return XCTFail("缺少 L0 段标题")
        }
        XCTAssertLessThanOrEqual(l0.count, 3_000,
            "L0 段不得超过 3000 字符，实际 \(l0.count)")
        XCTAssertEqual(l0.count, 3_000, "应精确截断到上限")
        XCTAssertTrue(system.hasPrefix("You are base-agent"), "BASE 仍应在最前")
    }

    /// 本机常见情形：L0 短于上限时原样保留，不被截断。
    func testShortSOPKeptIntact() throws {
        let shortSOP = "## 核心公理\n行动验证优先。"
        let (agent, root) = try makeWorkspace(withSOP: shortSOP)
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertTrue(agent.systemPrompt().contains(shortSOP))
    }

    /// 文件缺失时回退内置常量，同样受 3000 上限约束。
    func testMissingSOPFallsBackWithinLimit() throws {
        let (agent, root) = try makeWorkspace(withSOP: nil)
        defer { try? FileManager.default.removeItem(at: root) }
        let system = agent.systemPrompt()
        guard let l0 = l0Section(system) else {
            return XCTFail("缺少内置 L0 回退段")
        }
        XCTAssertLessThanOrEqual(l0.count, 3_000)
    }
}

private final class NoopCallbacks: AgentCallbacks {}
