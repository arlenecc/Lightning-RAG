import XCTest
@testable import BaseAgentCore

/// 折叠策略：每 8 轮保留最近 2 轮原文；段内禁止滑窗；只有占用 >90% 才回注摘要。
final class ContextFoldingPolicyTests: XCTestCase {

    private final class Captured: @unchecked Sendable {
        private let lock = NSLock()
        private var storedMessages: [ChatMessage] = []
        private var storedReasons: [String] = []
        private var storedFoldReasons: [String] = []
        private var storedInvalidations = 0

        var messages: [ChatMessage] {
            lock.lock(); defer { lock.unlock() }; return storedMessages
        }
        var reasons: [String] {
            lock.lock(); defer { lock.unlock() }; return storedReasons
        }
        var foldReasons: [String] {
            lock.lock(); defer { lock.unlock() }; return storedFoldReasons
        }
        var invalidations: Int {
            lock.lock(); defer { lock.unlock() }; return storedInvalidations
        }
        func setMessages(_ value: [ChatMessage]) {
            lock.lock(); storedMessages = value; lock.unlock()
        }
        func addReason(_ value: String) {
            lock.lock(); storedReasons.append(value); lock.unlock()
        }
        func addFoldReason(_ value: String) {
            lock.lock(); storedFoldReasons.append(value); lock.unlock()
        }
        func didInvalidate() {
            lock.lock(); storedInvalidations += 1; lock.unlock()
        }
    }

    private final class RecordingHooks: KernelHooks, @unchecked Sendable {
        private let captured: Captured
        init(_ captured: Captured) { self.captured = captured }
        func consolidated(_ session: Session, reason: String) async -> Bool {
            captured.addReason(reason)
            return false
        }
        func didFold(_ session: Session, reason: String, tools: [JSONValue]) async {
            captured.addFoldReason(reason)
        }
    }

    private func makeRuntime(_ captured: Captured) -> KernelRuntime {
        KernelRuntime(
            stream: { messages, _, _ in
                captured.setMessages(messages)
                return AsyncThrowingStream { continuation in
                    continuation.yield(LLMStreamEvent(kind: .content, content: "ok"))
                    continuation.yield(LLMStreamEvent(kind: .done))
                    continuation.finish()
                }
            },
            execute: { calls in calls.map { _ in ToolResult.ok("done") } },
            schemas: { [] },
            invalidateKV: { captured.didInvalidate() },
            log: { _ in }
        )
    }

    // MARK: - 配置与阈值

    func testDefaultIntervalIsEightAndLoopHonorsIt() {
        let config = AgentConfig()
        XCTAssertEqual(config.kernelConsolidateEvery, 8)
        XCTAssertEqual(config.kernelRawTurns, 10,
                       "8 轮周期 + 折叠保留 2 轮，段内才不会滑窗")
        let loop = KernelLoop(runtime: makeRuntime(Captured()),
                              config: config,
                              systemPrompt: { "SYS" },
                              maxTurns: 1)
        XCTAssertEqual(loop.consolidateEvery, 8)
    }

    func testReserveAndShrinkThresholdsAreIndependent() {
        let budget = ContextBudget(total: 30_000)
        // 输出预留恢复 21%（30k × 0.21 = 6300），收缩触发线独立为 90%（27000）。
        XCTAssertEqual(ContextBudget.reserveRatio, 0.21)
        XCTAssertEqual(ContextBudget.autoShrinkRatio, 0.90)
        XCTAssertEqual(budget.quota.reserve, 6_300)
        XCTAssertEqual(budget.autoShrinkThreshold, 27_000)

        // 79%~90% 之间：已侵入输出预留，但没到收缩线。
        budget.setUsed("raw", tokens: 25_000)
        XCTAssertTrue(budget.totalOver)
        XCTAssertFalse(budget.needsAutoShrink)

        budget.setUsed("raw", tokens: 27_000)
        XCTAssertFalse(budget.needsAutoShrink)
        budget.setUsed("raw", tokens: 27_001)
        XCTAssertTrue(budget.needsAutoShrink)
    }

    // MARK: - 常规折叠：确定性折叠回注 + 从 system 重启

    func testRegularConsolidationKeepsLastTwoTurns() async {
        let session = Session(budget: ContextBudget(total: 30_000), rawTurns: 10)
        for index in 1...8 {
            let rec = session.startTurn(prompt: "u\(index)")
            session.finishTurn(rec, assistantText: "a\(index)", summary: "s\(index)")
        }
        // 模拟上一周期残留的摘要：折叠后随新摘要行一起保留在 earlier_context。
        session.appendDigest("stale-digest")
        let captured = Captured()
        let loop = KernelLoop(runtime: makeRuntime(captured),
                              config: AgentConfig(),
                              systemPrompt: { "SYS" },
                              session: session,
                              hooks: RecordingHooks(captured),
                              maxTurns: 1)

        _ = await loop.run("hello")

        XCTAssertTrue(captured.reasons.isEmpty,
                      "周期性折叠不应调用摘要收缩钩子：\(captured.reasons)")
        // 折叠掉的 6 轮摘要行回注 earlier_context，残留的 stale-digest 一并保留。
        XCTAssertEqual(session.extraDigest.count, 7)
        XCTAssertTrue(session.extraDigest.contains("stale-digest"))
        XCTAssertGreaterThanOrEqual(captured.invalidations, 1, "前缀已变，KV 需失效一次")
        let messages = captured.messages
        // system + earlier_context + 保留的 2 轮原文（user/assistant ×2）+ 本轮 user。
        XCTAssertEqual(messages.count, 7,
                       "折叠后保留最近 2 轮原文 + 回注摘要：\(messages.count)")
        XCTAssertEqual(messages.first?.content, "SYS")
        XCTAssertTrue(messages.contains { $0.content.contains("earlier_context") },
                      "被折叠轮的摘要行回注 <earlier_context>")
        XCTAssertTrue(messages.contains { $0.content.contains("stale-digest") })
        XCTAssertTrue(messages.contains { $0.content.contains("[T1] s1") },
                      "被折叠轮的逐轮摘要行仍在上下文中")
        XCTAssertTrue(messages.contains { $0.content == "u7" })
        XCTAssertTrue(messages.contains { $0.content == "u8" })
        for index in 1...6 {
            XCTAssertFalse(messages.contains { $0.content == "u\(index)" },
                           "第 \(index) 轮已属于旧历史，原文不再发送")
        }
    }

    func testDiscardHistoryKeepsExactlyTwoRecentTurns() {
        let session = Session(budget: ContextBudget(total: 30_000), rawTurns: 10)
        for index in 1...8 {
            let rec = session.startTurn(prompt: "u\(index)")
            session.finishTurn(rec, assistantText: "a\(index)")
        }
        session.appendDigest("stale")
        XCTAssertEqual(session.discardHistory(keepingRecent: 2), 6)
        XCTAssertEqual(session.turns.map(\.prompt), ["u7", "u8"])
        // 摘要行回注：残留 digest + 6 条被折叠轮摘要行（无 summary 回退首句）。
        XCTAssertEqual(session.extraDigest.count, 7)
        XCTAssertTrue(session.extraDigest.contains("stale"))
        XCTAssertTrue(session.extraDigest.contains("[T1] a1"))
    }

    func testSlidOutOldTurnsAreDiscardedNotSummarized() {
        let session = Session(budget: ContextBudget(total: 30_000), rawTurns: 1)
        for index in 1...4 {
            let rec = session.startTurn(prompt: "u\(index)")
            session.finishTurn(rec, assistantText: "a\(index)", summary: "s\(index)")
        }
        _ = session.startTurn(prompt: "now")
        let messages = session.buildMessages(systemPrompt: "SYS")
        XCTAssertFalse(messages.contains { $0.content.contains("earlier_context") },
                       "old 轮直接丢弃，不再折成摘要回注")
        XCTAssertFalse(messages.contains { $0.content.contains("[T1]") })
    }

    func testOnlyExplicitDigestCreatesEarlierContext() {
        let session = Session(budget: ContextBudget(total: 30_000), rawTurns: 8)
        session.appendDigest("[摘要 90%] 旧上下文要点")
        _ = session.startTurn(prompt: "now")
        let injected = session.buildMessages(systemPrompt: "SYS")
        XCTAssertTrue(injected.contains { $0.content.contains("<earlier_context>") })
        XCTAssertTrue(injected.contains { $0.content.contains("旧上下文要点") })

        XCTAssertEqual(session.resetForFreshContext(), 1)
        XCTAssertTrue(session.extraDigest.isEmpty)
        let fresh = session.buildMessages(systemPrompt: "SYS", userText: "now")
        XCTAssertFalse(fresh.contains { $0.content.contains("<earlier_context>") })
    }

    // MARK: - 例外：占用 >90% 才走摘要收缩钩子

    func testReservePressureDoesNotTriggerSummaryReinjection() async {
        let session = Session(budget: ContextBudget(total: 30_000), rawTurns: 8)
        let rec = session.startTurn(prompt: "u1")
        session.finishTurn(rec, assistantText: "a1")
        // 已侵入 21% 输出预留（79% 线），但未到独立的 90% 收缩线。
        session.budget.setUsed("system", tokens: 25_000)
        let captured = Captured()
        let loop = KernelLoop(runtime: makeRuntime(captured),
                              config: AgentConfig(),
                              systemPrompt: { "SYS" },
                              session: session,
                              hooks: RecordingHooks(captured),
                              maxTurns: 1)

        _ = await loop.run("hello")

        XCTAssertTrue(captured.reasons.isEmpty,
                      "79% 预算线只做裁剪，不应触发摘要回注：\(captured.reasons)")
        XCTAssertEqual(captured.invalidations, 0)
        XCTAssertEqual(session.turns.count, 2, "历史应保持，不该被折叠清空")
    }

    func testContextPressureOver90TriggersSummaryHook() async {
        let session = Session(budget: ContextBudget(total: 30_000), rawTurns: 8)
        let rec = session.startTurn(prompt: "u1")
        session.finishTurn(rec, assistantText: "a1")
        // 只把记账推到 90% 以上，避免构造真实的大 prompt。
        session.budget.setUsed("system", tokens: 28_000)
        let captured = Captured()
        let loop = KernelLoop(runtime: makeRuntime(captured),
                              config: AgentConfig(),
                              systemPrompt: { "SYS" },
                              session: session,
                              hooks: RecordingHooks(captured),
                              maxTurns: 1)

        _ = await loop.run("hello")

        XCTAssertEqual(captured.reasons.first, "context-90",
                       "超过 90% 才允许走摘要回注路径：\(captured.reasons)")
    }

    // MARK: - A/B/E：段内 KV 稳定性

    /// B：预算按段内剩余轮数均摊，段首工具结果上限应小于段末。
    func testToolBudgetReservesFutureTurnsInSegment() {
        let budget = ContextBudget(total: 30_000)
        budget.setUsed("system", tokens: 5_000)
        budget.setUsed("tools", tokens: 5_000)
        let last = budget.remainingForTool(1, maxChars: 8_000, reserveTurns: 1)
        let first = budget.remainingForTool(1, maxChars: 8_000, reserveTurns: 8)
        XCTAssertLessThan(first, last)
        XCTAssertGreaterThanOrEqual(first, 600)
    }

    /// A：越顶判定是纯函数，不修改会话状态。
    func testRawCeilingProbeIsPure() {
        let session = Session(budget: ContextBudget(total: 4_000), rawTurns: 10)
        for index in 1...3 {
            let rec = session.startTurn(prompt: "u\(index)")
            session.finishTurn(rec, assistantText: "a\(index)")
        }
        _ = session.startTurn(prompt: "now")
        XCTAssertFalse(session.rawWouldExceedCeiling(systemTokens: 10))
        XCTAssertTrue(session.rawWouldExceedCeiling(systemTokens: 3_500))
        XCTAssertEqual(session.turns.count, 4, "探测不得改动历史")
    }

    /// A：超顶时触发一次提前折叠（而不是 windows 逐轮滑窗），并通知边界钩子。
    func testEarlyFoldTriggersOnceAndKeepsLastTwoTurns() async {
        let session = Session(budget: ContextBudget(total: 4_000), rawTurns: 10)
        for index in 1...5 {
            let rec = session.startTurn(prompt: "u\(index)")
            session.finishTurn(rec, assistantText: "a\(index)")
        }
        let captured = Captured()
        // 超长 system 把 raw 硬顶压到极小，模拟工具密集段越顶。
        let longSystem = String(repeating: "S", count: 12_000)
        let loop = KernelLoop(runtime: makeRuntime(captured),
                              config: AgentConfig(),
                              systemPrompt: { longSystem },
                              session: session,
                              hooks: RecordingHooks(captured),
                              maxTurns: 1)

        _ = await loop.run("hello")

        XCTAssertEqual(captured.foldReasons, ["early-raw-ceiling"],
                       "应只折叠一次：\(captured.foldReasons)")
        XCTAssertGreaterThanOrEqual(captured.invalidations, 1)
        XCTAssertEqual(session.turns.count, 3, "保留 2 个已完成轮 + 当前轮")
        XCTAssertEqual(session.extraDigest.count, 3, "提前折叠同样回注摘要行")
        XCTAssertTrue(captured.messages.contains { $0.content.contains("earlier_context") })
    }

    /// C/D：周期性折叠也会通知边界钩子（刷新 system 前缀 + 预热 KV）。
    func testPeriodicFoldNotifiesBoundaryHooks() async {
        let session = Session(budget: ContextBudget(total: 30_000), rawTurns: 10)
        for index in 1...8 {
            let rec = session.startTurn(prompt: "u\(index)")
            session.finishTurn(rec, assistantText: "a\(index)")
        }
        let captured = Captured()
        let loop = KernelLoop(runtime: makeRuntime(captured),
                              config: AgentConfig(),
                              systemPrompt: { "SYS" },
                              session: session,
                              hooks: RecordingHooks(captured),
                              maxTurns: 1)
        _ = await loop.run("hello")
        XCTAssertEqual(captured.foldReasons.first, "periodic")
    }

    /// 折叠边界预热用的 earlier 前缀必须与正式请求的消息逐字节一致，
    /// 否则 warmup 的 KV 前缀无法命中，折叠后首轮又变回全量重建。
    func testEarlierContextTextMatchesRenderedMessage() {
        let session = Session(budget: ContextBudget(total: 30_000), rawTurns: 8)
        XCTAssertTrue(session.earlierContextText.isEmpty, "无折叠时不应有前缀")

        session.appendDigest("[T1] 要点一")
        _ = session.startTurn(prompt: "now")
        let messages = session.buildMessages(systemPrompt: "SYS")
        let earlier = session.earlierContextText

        XCTAssertFalse(earlier.isEmpty)
        XCTAssertEqual(messages.count > 1 ? messages[1].role : nil, .user)
        XCTAssertEqual(messages.count > 1 ? messages[1].content : nil, earlier,
                       "预热前缀必须与正式请求的第 2 条消息一致")
        XCTAssertTrue(earlier.hasPrefix("<earlier_context>"))
    }
}
