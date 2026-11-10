import XCTest
@testable import BaseAgentCore

/// 轻量轮（寒暄 / 能力问询）回归：v1.2.22 曾出现发 `hi` 后模型把
/// system prompt 里的 L0 META-SOP 重写成一份 `# [SYSTEM]` 开头的假提示
/// 当作回答。修复要点：轻量轮只注入极简 system + 复述输出本地兜底。
final class LightweightTurnTests: XCTestCase {

    /// 完整 system 里诱发复述的常驻段落，测试里作为「不应出现」的标记。
    private let fullSystem = """
    You are base-agent, a local autonomous agent.
    # [L0 META-SOP: memory_management_sop]
    ## 0. 核心公理 (Core Axioms - 最高优先级)
    1. 行动验证原则 (Action-Verified Only)
    2. 神圣不可删改性 (Sanctity of Verified Data)
    """

    /// 真实事故中模型输出的前几行（raw_session 2026-11-07 摘录）。
    private let echo = """
    # [SYSTEM]
    ## 1. 身份
    你是一名本地自主智能体（Base Agent），具备文件读写、脚本执行、系统命令等物理操作权限。
    ## 2. 核心公理 (Core Axioms - 最高优先级)
    1. **行动验证原则 (Action-Verified Only)**
    """

    private final class Captured: @unchecked Sendable {
        private let lock = NSLock()
        private var storedMessages: [ChatMessage] = []
        private var storedTools: [JSONValue] = []
        var messages: [ChatMessage] {
            lock.lock(); defer { lock.unlock() }; return storedMessages
        }
        var tools: [JSONValue] {
            lock.lock(); defer { lock.unlock() }; return storedTools
        }
        func capture(_ messages: [ChatMessage], _ tools: [JSONValue]) {
            lock.lock()
            storedMessages = messages
            storedTools = tools
            lock.unlock()
        }
    }

    private func toolSchema(_ name: String) -> JSONValue {
        .object([
            "type": .string("function"),
            "function": .object([
                "name": .string(name),
                "description": .string("测试工具 \(name)"),
                "parameters": .object([:]),
            ]),
        ])
    }

    private func makeRuntime(_ captured: Captured, reply: String,
                             schemas: [JSONValue] = []) -> KernelRuntime {
        KernelRuntime(
            stream: { messages, tools, _ in
                captured.capture(messages, tools)
                return AsyncThrowingStream { continuation in
                    continuation.yield(LLMStreamEvent(kind: .content, content: reply))
                    continuation.yield(LLMStreamEvent(kind: .done, rawContent: reply))
                    continuation.finish()
                }
            },
            execute: { calls in calls.map { _ in ToolResult.ok("done") } },
            schemas: { schemas },
            log: { _ in }
        )
    }

    // MARK: - 纯函数

    func testLightweightSystemPromptExcludesMETA_SOP() {
        let prompt = KernelLoop.lightweightSystemPrompt
        XCTAssertLessThan(prompt.count, 400)
        for marker in ["[L0 META-SOP", "行动验证原则", "神圣不可删改性", "<|im_start|>system"] {
            XCTAssertFalse(prompt.contains(marker), "轻量 system 不得含 \(marker)")
        }
    }

    func testLooksLikeSystemPromptEcho() {
        XCTAssertTrue(KernelLoop.looksLikeSystemPromptEcho(echo))
        XCTAssertTrue(KernelLoop.looksLikeSystemPromptEcho(
            "这是正文，但里面夹了 [L0 META-SOP: memory_management_sop] 全文"))
        XCTAssertTrue(KernelLoop.looksLikeSystemPromptEcho(String(repeating: "复述", count: 400)))
        XCTAssertFalse(KernelLoop.looksLikeSystemPromptEcho("你好！有什么我可以帮你的吗？"))
        XCTAssertFalse(KernelLoop.looksLikeSystemPromptEcho(
            "我可以读写文件、执行命令、运行代码、检索本地知识库。"))
    }

    func testLightweightFallbackMatchesInputKind() {
        XCTAssertEqual(KernelLoop.lightweightFallback(for: "hi"),
                       "你好！有什么我可以帮你的吗？")
        XCTAssertTrue(KernelLoop.lightweightFallback(for: "你能干什么").contains("读写文件"))
    }

    // MARK: - 内核行为

    func testLightweightTurnUsesMinimalSystemAndRejectsEcho() async {
        let captured = Captured()
        let loop = KernelLoop(runtime: makeRuntime(captured, reply: echo),
                              config: AgentConfig(),
                              systemPrompt: { self.fullSystem },
                              maxTurns: 1)
        // 寒暄/能力问询已改为本地直答；这里用普通轻量输入覆盖「仍然走
        // LLM 的轻量轮」：极简 system + 拒绝 system prompt 复述。
        let result = await loop.run("讲个冷笑话", toolsEnabled: false)

        XCTAssertEqual(result.reason, .done)
        XCTAssertEqual(result.text, "你好！有什么我可以帮你的吗？")
        XCTAssertFalse(result.text.contains("[SYSTEM]"))
        XCTAssertFalse(result.text.contains("行动验证原则"))
        XCTAssertTrue(captured.tools.isEmpty, "轻量轮不得提供工具 schema")
        let first = captured.messages.first
        XCTAssertEqual(first?.role, .system)
        XCTAssertEqual(first?.content, KernelLoop.lightweightSystemPrompt)
        XCTAssertFalse(captured.messages.contains { $0.content.contains("L0 META-SOP") },
                       "完整 system 的 SOP 段绝不能进轻量轮上下文")
    }

    func testCapabilityQueryUsesRealToolListWithoutLLM() async {
        let captured = Captured()
        let schemas = ["file_tools", "rag_tools", "web_scan"].map(toolSchema)
        let loop = KernelLoop(runtime: makeRuntime(captured, reply: echo, schemas: schemas),
                              config: AgentConfig(),
                              systemPrompt: { self.fullSystem },
                              maxTurns: 1)
        let result = await loop.run("你能干什么", toolsEnabled: false)
        XCTAssertTrue(result.text.contains("file_tools"))
        XCTAssertTrue(result.text.contains("rag_tools"))
        XCTAssertFalse(result.text.contains("[SYSTEM]"))
        XCTAssertTrue(captured.messages.isEmpty, "能力问询不得调用 LLM")
    }

    func testNormalTurnStillReceivesFullSystemPrompt() async {
        let captured = Captured()
        let loop = KernelLoop(runtime: makeRuntime(captured, reply: "已完成"),
                              config: AgentConfig(),
                              systemPrompt: { self.fullSystem },
                              maxTurns: 1)
        _ = await loop.run("写个脚本", toolsEnabled: true)
        XCTAssertEqual(captured.messages.first?.content, fullSystem,
                       "普通任务轮必须继续拿到完整 system prompt")
    }

    func testCapabilityAnswerListsRealTools() {
        let names = ["file_tools", "code_run", "rag_tools", "long_memory", "update_plan",
                     "web_scan", "skill_tools", "ask_user", "mcp_custom"]
        let answer = KernelLoop.capabilityAnswer(from: names.map(toolSchema))
        XCTAssertTrue(answer.contains("共 9 个"), answer)
        for name in names { XCTAssertTrue(answer.contains(name), "缺少工具 \(name)") }
        XCTAssertTrue(answer.contains("文件与执行"))
        XCTAssertTrue(answer.contains("知识库检索"))
        XCTAssertTrue(answer.contains("mcp_custom"), "未知前缀的工具必须落到「其它」")
        XCTAssertFalse(answer.contains("回答问题"))
        XCTAssertFalse(answer.contains("提供信息"))

        let empty = KernelLoop.capabilityAnswer(from: [])
        XCTAssertTrue(empty.contains("工具清单还在加载"), empty)
    }

    /// 用户实测：「你能干什么 / 你有什么工具 / 有什么可用工具」三种问法
    /// 都得到同一段空泛套话。修复后三种都给出真实工具清单，且完全不调模型。
    func testAllCapabilityQueryPhrasesAnswerLocally() async {
        let schemas = ["file_tools", "rag_tools", "web_scan"].map(toolSchema)
        for phrase in ["你能干什么", "你有什么工具", "有什么可用工具"] {
            let captured = Captured()
            let loop = KernelLoop(runtime: makeRuntime(captured, reply: echo,
                                                       schemas: schemas),
                                  config: AgentConfig(),
                                  systemPrompt: { self.fullSystem },
                                  maxTurns: 1)
            let result = await loop.run(phrase, toolsEnabled: false)
            XCTAssertEqual(result.reason, .done, phrase)
            XCTAssertTrue(result.text.contains("file_tools"), phrase)
            XCTAssertTrue(result.text.contains("rag_tools"), phrase)
            XCTAssertFalse(result.text.contains("提供信息"), phrase)
            XCTAssertFalse(result.text.contains("[SYSTEM]"), phrase)
            XCTAssertTrue(captured.messages.isEmpty,
                          "\(phrase) 不应创建 LLM stream")
        }
    }

    /// 寒暄本地直答：不创建 LLM stream，也就不会换掉预热好的 system/tools
    /// 前缀（tools-changed 全量重建）或在启动预热期间让用户干等 20s。
    func testGreetingAnswersLocallyWithoutLLM() async {
        let captured = Captured()
        let loop = KernelLoop(runtime: makeRuntime(captured, reply: "你好呀"),
                              config: AgentConfig(),
                              systemPrompt: { self.fullSystem },
                              maxTurns: 1)
        let result = await loop.run("hi", toolsEnabled: false)
        XCTAssertEqual(result.reason, .done)
        XCTAssertEqual(result.text, KernelLoop.greetingAnswer(for: "hi"))
        XCTAssertTrue(captured.messages.isEmpty, "寒暄不得调用 LLM")
        XCTAssertTrue(captured.tools.isEmpty, "寒暄不得提供工具 schema")
    }
}
