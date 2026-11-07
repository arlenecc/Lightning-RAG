import Foundation
import XCTest
@testable import BaseAgentCore

private final class RecordingLLM: LLMService, @unchecked Sendable {
    let identifier = "recording"
    let supportsTools = true
    let supportsConcurrentIsolatedRequests: Bool

    private let reply: String
    private let lock = NSLock()
    private var defaults = 0
    private var options: [LLMRequestOptions] = []

    init(supportsIsolated: Bool, reply: String) {
        self.supportsConcurrentIsolatedRequests = supportsIsolated
        self.reply = reply
    }

    func chatStream(messages: [ChatMessage], tools: [JSONValue],
                    temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        lock.lock()
        defaults += 1
        lock.unlock()
        return makeStream()
    }

    func chatStream(messages: [ChatMessage], tools: [JSONValue],
                    temperature: Double,
                    options: LLMRequestOptions) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        lock.lock()
        self.options.append(options)
        lock.unlock()
        return makeStream()
    }

    var defaultCallCount: Int {
        lock.lock(); defer { lock.unlock() }
        return defaults
    }

    var optionCallCount: Int {
        lock.lock(); defer { lock.unlock() }
        return options.count
    }

    var recordedOptions: [LLMRequestOptions] {
        lock.lock(); defer { lock.unlock() }
        return options
    }

    private func makeStream() -> AsyncThrowingStream<LLMStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(LLMStreamEvent(kind: .content, content: reply))
            continuation.yield(LLMStreamEvent(kind: .done))
            continuation.finish()
        }
    }
}

/// KV 前缀复用回归：Session 窗口策略、内核折叠节奏与后台任务隔离。
///
/// 本地 MLX 的跨轮复用只认「完整消息前缀」：`<earlier_context>` 挂在消息索引 1，
/// 一旦摘要区每轮变化，system 之后的前缀就全部失效，只能全量 prefill。
/// 这些测试锁住「同一折叠周期内消息前缀不变」这个不变式。
final class KVPrefixReuseTests: XCTestCase {
    // MARK: - 辅助

    private func commonPrefix(_ lhs: [ChatMessage], _ rhs: [ChatMessage]) -> Int {
        var index = 0
        while index < lhs.count, index < rhs.count, lhs[index] == rhs[index] {
            index += 1
        }
        return index
    }

    /// 跑一轮完整生命周期：开轮 → 组装消息 → 收轮。
    ///
    /// 默认每轮 3200 + 3200 个 ASCII 字符，估算约 1829 token，
    /// 8 轮刚好超过 raw 软配额（12k），但远低于硬顶（约 23.7k）。
    @discardableResult
    private func addTurn(_ session: Session,
                         promptChars: Int = 3_200,
                         answerChars: Int = 3_200) -> TurnRecord {
        let rec = session.startTurn(prompt: String(repeating: "u", count: promptChars))
        _ = session.buildMessages(systemPrompt: "SYS")
        session.finishTurn(rec, assistantText: String(repeating: "a", count: answerChars))
        return rec
    }

    /// 一个只回复固定文本、没有任何工具调用的内核运行时。
    private func makeRuntime(_ onInvalidate: @escaping () -> Void) -> KernelRuntime {
        KernelRuntime(
            stream: { _, _, _ in
                AsyncThrowingStream { continuation in
                    continuation.yield(LLMStreamEvent(kind: .content, content: "ok"))
                    continuation.yield(LLMStreamEvent(kind: .done))
                    continuation.finish()
                }
            },
            execute: { calls in calls.map { _ in ToolResult.fail("no tools") } },
            schemas: { [] },
            invalidateKV: onInvalidate,
            log: { _ in }
        )
    }

    private func makeLoop(session: Session,
                          onInvalidate: @escaping () -> Void) -> KernelLoop {
        var config = AgentConfig()
        config.kernelConsolidateEvery = 8
        return KernelLoop(runtime: makeRuntime(onInvalidate),
                          config: config,
                          systemPrompt: { "SYS" },
                          session: session,
                          maxTurns: 1)
    }

    private func makeAgent(llm: LLMService) -> Agent {
        let workspace = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("kv-reuse-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: workspace) }

        var config = AgentConfig()
        config.workspace = workspace.path
        config.llmProvider = "api"
        config.embeddingProvider = "local"
        config.maxIterations = 3
        // 每 2 轮折叠：第 3 次 run 的循环开头就会命中一次周期性折叠，
        // 让后台任务测试能走真实的 didFold → onFoldBoundary 链路。
        config.kernelConsolidateEvery = 2
        return Agent(llm: llm, config: config, callbacks: NoopCallbacks())
    }

    private func waitForBackgroundIdle(_ agent: Agent,
                                       timeout: TimeInterval = 3) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if agent.isBackgroundIdle() { return true }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        return agent.isBackgroundIdle()
    }

    // MARK: - raw 窗口

    /// raw 窗口只应按「硬上下文余量」收缩；软配额（40%）超了也不该每轮裁剪历史。
    func testRawWindowDoesNotShrinkBySoftQuota() {
        let session = Session(budget: ContextBudget(total: 30_000), rawTurns: 8)
        for _ in 1...8 { addTurn(session) }

        let rec = session.startTurn(prompt: String(repeating: "u", count: 3_200))
        let messages = session.buildMessages(systemPrompt: "SYS")

        // 8 个已完成轮（每轮 user+assistant）+ 当前轮（只有 user）+ system。
        XCTAssertEqual(messages.count, 1 + 8 * 2 + 1)
        XCTAssertEqual(session.stats()["raw_window"]?.intValue, 9,
                       session.statsDescription())
        XCTAssertTrue(session.digestLines().isEmpty,
                      session.digestLines().joined(separator: " | "))
        session.finishTurn(rec, assistantText: "done")
    }

    // MARK: - 折叠策略

    /// CONSOLIDATE 应一次折到只剩最近一轮：给下一个周期留满 raw 窗口，
    /// 而不是只折掉超出部分、每 2~3 轮又重新开始滑窗。
    func testConsolidationFoldsAllButRecentTurn() {
        let session = Session(budget: ContextBudget(total: 30_000), rawTurns: 8)
        for _ in 1...8 { addTurn(session) }
        XCTAssertEqual(session.turns.count, 8)

        let folded = session.foldForConsolidation()

        XCTAssertEqual(folded, 7)
        XCTAssertEqual(session.turns.count, 1, "只保留最近一轮")
        XCTAssertEqual(session.extraDigest.count, 7)
    }

    /// 连续 24 个模型轮里，消息前缀只应在折叠边界（第 9、17 轮前）塌陷一次。
    /// 中间任何一次塌陷都意味着 `<earlier_context>` 变了，KV 缓存全量重建。
    func testPrefixCollapseOnlyAtConsolidationBoundaries() {
        let session = Session(budget: ContextBudget(total: 30_000), rawTurns: 8)
        let every = 8
        var previous: [ChatMessage] = []
        var collapsed: [Int] = []

        for turn in 1...24 {
            if turn > 1 && (turn - 1) % every == 0 {
                session.foldForConsolidation()
            }
            let rec = session.startTurn(prompt: "u\(turn)")
            let messages = session.buildMessages(systemPrompt: "SYS")
            if !previous.isEmpty, commonPrefix(previous, messages) <= 1 {
                collapsed.append(turn)
            }
            previous = messages
            session.finishTurn(rec, assistantText: "a\(turn)")
        }

        XCTAssertEqual(collapsed, [9, 17], "每个折叠周期只允许一次前缀塌陷")
    }

    // MARK: - 内核折叠节奏

    /// 只有 raw 软配额超支、整体仍在硬顶内时，内核不应折叠，更不应丢弃 KV。
    func testSoftRawOverAloneDoesNotConsolidate() async {
        let session = Session(budget: ContextBudget(total: 30_000), rawTurns: 8)
        var invalidations = 0
        let loop = makeLoop(session: session, onInvalidate: { invalidations += 1 })

        for _ in 1...4 {
            // 每轮约 4k token：3 轮就会超过 raw 软配额 12k。
            _ = await loop.run(String(repeating: "u", count: 14_000))
        }

        XCTAssertTrue(session.budget.rawOver, session.budget.snapshot())
        XCTAssertFalse(session.budget.totalOver)
        XCTAssertEqual(invalidations, 0, "软配额超支不应触发折叠/丢 KV")
        XCTAssertEqual(session.turns.count, 4)
        XCTAssertEqual(session.stats()["raw_window"]?.intValue, 4,
                       session.statsDescription())
    }

    /// 折叠周期不能超过 raw 窗口：否则窗口每轮都把最老轮滑进摘要区。
    func testConsolidateEveryClampedToRawWindow() {
        let session = Session(budget: ContextBudget(total: 30_000), rawTurns: 2)
        var config = AgentConfig()
        config.kernelConsolidateEvery = 8
        let loop = KernelLoop(runtime: makeRuntime {},
                              config: config,
                              systemPrompt: { "SYS" },
                              session: session)
        XCTAssertEqual(loop.consolidateEvery, 2)
    }

    // MARK: - 后台事实抽取（4h 文件游标 / 仅辅助模型）

    /// 不支持独立会话的主模型（且无辅助模型）：后台 LLM 工作整体跳过，
    /// 不能改写主会话 KV。
    func testBackgroundWorkSkippedWhenMainModelCannotIsolate() async {
        let llm = RecordingLLM(supportsIsolated: false, reply: "Hello there!")
        let agent = makeAgent(llm: llm)
        defer { agent.close() }

        for _ in 1...3 { _ = await agent.run("please remember my editor is vim") }
        let idle = await waitForBackgroundIdle(agent)
        XCTAssertTrue(idle, "后台任务应在超时前结束")

        XCTAssertEqual(llm.defaultCallCount, 3, "后台任务不应再调用主模型")
        XCTAssertEqual(llm.optionCallCount, 0, "不支持独立会话时不应发起后台请求")
    }

    /// 事实抽取**只允许辅助模型**：没有 aux 时即便主模型支持 isolated，
    /// 也不得回退主模型做全量 prefill（那会与前台抢 GPU）。
    func testFactExtractionRequiresAuxModel() async {
        let llm = RecordingLLM(supportsIsolated: true, reply: "Hello there!")
        let agent = makeAgent(llm: llm)
        defer { agent.close() }

        for _ in 1...3 { _ = await agent.run("please remember my editor is vim") }
        let idle = await waitForBackgroundIdle(agent)
        XCTAssertTrue(idle)

        XCTAssertEqual(llm.defaultCallCount, 3, "主模型只服务前台对话")
        XCTAssertEqual(llm.optionCallCount, 0, "不得回退主模型做事实抽取")
        XCTAssertNil(agent.workMemory.get("__auto_facts__"),
                     "当日工作记忆不再维护")
    }

    /// 有辅助模型时按 4h 文件游标抽取：多轮（含周期性折叠）只跑一次，
    /// 且只写 memory graph，不写当日工作记忆。
    func testFactExtractionUsesFileCursorAndAuxOnly() async {
        let llm = RecordingLLM(supportsIsolated: true, reply: "Hello there!")
        let aux = RecordingLLM(supportsIsolated: false, reply: "facts")
        let agent = makeAgent(llm: llm)
        defer { agent.close() }
        agent.auxLLM = aux

        // 每轮之间等后台空闲，模拟用户正常的交互间隔；否则前台新请求会
        // 立刻取消上一轮的抽取任务。
        for _ in 1...4 {
            _ = await agent.run("please remember my editor is vim")
            let idle = await waitForBackgroundIdle(agent)
            XCTAssertTrue(idle, "后台任务应在超时前结束")
        }

        XCTAssertEqual(llm.defaultCallCount, 4, "主模型只服务前台对话")
        XCTAssertEqual(llm.optionCallCount, 0, "辅助可用时主模型不得被后台调用")
        XCTAssertEqual(aux.defaultCallCount + aux.optionCallCount, 1,
            "4 小时内只抽一次，折叠边界不额外触发")
        XCTAssertNil(agent.workMemory.get("__auto_facts__"),
                     "事实只写 memory graph，不再维护当日工作记忆")
    }

    /// 上下文 >90% 的自动收缩也必须零 LLM 调用：被折掉的每一轮自带
    /// `TurnRecord.summary`，直接折成 `<earlier_context>` 即可。旧实现在
    /// 内核循环里同步 await 一次摘要模型，会直接卡住当前轮响应。
    func testShrinkSessionUsesPerTurnSummariesWithoutLLM() async {
        let llm = RecordingLLM(supportsIsolated: true, reply: "Hello there!")
        let agent = makeAgent(llm: llm)
        defer { agent.close() }

        for index in 1...3 {
            let rec = agent.session.startTurn(prompt: "u\(index)")
            agent.session.finishTurn(rec, assistantText: "a\(index)",
                                     summary: "要点-\(index)")
        }
        let shrunk = await agent.shrinkSession(reason: "context-90")

        XCTAssertTrue(shrunk)
        XCTAssertEqual(llm.defaultCallCount, 0, "收缩不得再调用主模型")
        XCTAssertEqual(llm.optionCallCount, 0, "收缩不得再发起后台模型请求")
        XCTAssertEqual(agent.session.turns.count, 1, "旧轮应被丢弃")
        let digest = agent.session.extraDigest.joined(separator: "\n")
        XCTAssertTrue(digest.contains("要点-1"), digest)
        XCTAssertTrue(digest.contains("要点-2"), digest)
        XCTAssertEqual(agent.session.turns.last?.summary, "要点-3",
                       "最近一轮保留原文，不需要回注自己的摘要")
    }

    // MARK: - 工具预算

    /// 软 raw 配额超支时，工具结果不应被一刀切压到 600 字符；
    /// 只有真正吃完硬上下文余量（侵入 reserve）才退化为最小额度。
    func testToolBudgetUsesHardContextLeft() {
        let budget = ContextBudget(total: 30_000)
        budget.setUsed("system", tokens: 3_900)
        budget.setUsed("earlier", tokens: 900)
        budget.setUsed("tools", tokens: 5_100)
        budget.setUsed("raw", tokens: 13_000)  // 软配额 12k 已超，硬顶未超

        XCTAssertTrue(budget.rawOver)
        XCTAssertFalse(budget.totalOver)
        XCTAssertGreaterThan(budget.remainingForTool(), 600)

        budget.setUsed("raw", tokens: 20_000)  // 吃进 reserve
        XCTAssertTrue(budget.totalOver)
        XCTAssertEqual(budget.remainingForTool(), 600)
    }
}
