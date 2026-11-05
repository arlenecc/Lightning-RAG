import Foundation
import XCTest
import ZIPFoundation
@testable import BaseAgentCore

private final class SilentCallbacks: AgentCallbacks {}

/// 改进 1–8：多级记忆（L0–L4）分层落盘、注入与门控的回归测试。
final class MemoryLayeringTests: XCTestCase {

    private var workspace: String!
    private var dir: MemoryDirectory!

    override func setUp() {
        super.setUp()
        workspace = NSTemporaryDirectory() + "/memory-layering-\(UUID().uuidString)"
        dir = MemoryDirectory(workspace: workspace)
        dir.ensureLayout()
    }

    override func tearDown() {
        try? FileManager.default.removeItem(atPath: workspace)
        dir = nil
        workspace = nil
        super.tearDown()
    }

    // MARK: 布局与读写

    func testEnsureLayoutSeedsAllLayersWithoutOverwriting() {
        let fm = FileManager.default
        XCTAssertTrue(fm.fileExists(atPath: dir.root))
        XCTAssertTrue(fm.fileExists(atPath: dir.rawSessionDir))
        XCTAssertTrue(fm.fileExists(atPath: dir.sopFile))
        XCTAssertTrue(fm.fileExists(atPath: dir.l1File))
        XCTAssertTrue(fm.fileExists(atPath: dir.l2File))
        XCTAssertTrue(fm.fileExists(atPath: dir.allHistoryFile))

        // 人工修改 L1 后再次 ensureLayout 不得覆盖（记忆神圣不可删改）。
        dir.write("# CUSTOM L1\n", to: dir.l1File)
        dir.ensureLayout()
        XCTAssertEqual(dir.read(dir.l1File), "# CUSTOM L1\n")
    }

    func testAppendDoesNotOverwriteAndSignatureChanges() {
        let file = dir.path("raw_session/probe.txt")
        XCTAssertTrue(dir.append("A\n", to: file))
        let before = dir.read(file)
        XCTAssertTrue(dir.append("B\n", to: file))
        XCTAssertTrue(dir.append("C\n", to: file))
        XCTAssertEqual(dir.read(file), "A\nB\nC\n")
        XCTAssertNotNil(before)
    }

    func testLoadSOPPrefersFileAndL1CapsAt30Lines() {
        dir.write("# MY SOP\n", to: dir.sopFile)
        XCTAssertEqual(dir.loadSOP(maxChars: 12_000), "# MY SOP\n")

        let many = (1...45).map { "line\($0)" }.joined(separator: "\n")
        dir.write(many, to: dir.l1File)
        let loaded = dir.loadL1(maxChars: 4_000)
        XCTAssertLessThanOrEqual(loaded.split(separator: "\n").count, 31)
        XCTAssertTrue(loaded.contains("整理后再写入"))
    }

    func testPatchRequiresUniqueMatch() {
        let file = dir.l2File
        dir.write("alpha\nbeta\nalpha\n", to: file)
        XCTAssertFalse(dir.patch(file: file, old: "alpha", new: "X"))  // 非唯一 → 拒绝
        XCTAssertEqual(dir.read(file), "alpha\nbeta\nalpha\n")
        XCTAssertTrue(dir.patch(file: file, old: "beta", new: "Y"))
        XCTAssertEqual(dir.read(file), "alpha\nY\nalpha\n")
    }

    // MARK: 改进1/2：全文与摘要落盘

    func testAppendSessionWritesFullTranscriptAndDigests() {
        let store = RawSessionStore(directory: dir)
        let rec = TurnRecord(index: 7, prompt: "anchor-prompt-with-working-memory")
        rec.assistantText = "assistant middle reply"
        rec.summary = "确认了端口 8080"
        rec.toolResults = [ToolResultRecord(id: "1", name: "file_read", content: "file body")]

        let id = store.appendSession(systemText: "SYSTEM TEXT",
                                     userText: "帮我查端口",
                                     turns: [rec],
                                     finalText: "端口是 8080")
        XCTAssertFalse(id.isEmpty)

        let day = MemoryDirectory.dayString()
        let raw = dir.read(dir.rawFile(day: day)) ?? ""
        XCTAssertTrue(raw.contains("@@SESSION-BEGIN"))
        // SYSTEM 全文只存一份到独立文件，SESSION 块里是引用行（避免重复 5k token）。
        XCTAssertTrue(raw.contains("@@SYSTEM "))
        XCTAssertTrue(raw.contains("\(day)-system.txt"))
        XCTAssertEqual(dir.read(dir.path("raw_session/\(day)-system.txt")), "SYSTEM TEXT")
        XCTAssertTrue(raw.contains("帮我查端口"))
        XCTAssertTrue(raw.contains("anchor-prompt-with-working-memory"))
        XCTAssertTrue(raw.contains("assistant middle reply"))
        XCTAssertTrue(raw.contains("file body"))
        XCTAssertTrue(raw.contains("端口是 8080"))

        let digest = dir.read(dir.digestFile(day: day)) ?? ""
        XCTAssertTrue(digest.contains("确认了端口 8080"))
    }

    func testAppendSessionIsAdditiveAcrossRequests() {
        let store = RawSessionStore(directory: dir)
        store.appendSession(systemText: "S", userText: "U1", turns: [], finalText: "A1")
        store.appendSession(systemText: "S", userText: "U2", turns: [], finalText: "A2")
        let raw = dir.read(dir.rawFile(day: MemoryDirectory.dayString())) ?? ""
        XCTAssertEqual(raw.components(separatedBy: "@@SESSION-BEGIN").count - 1, 2)
        XCTAssertTrue(raw.contains("A1"))
        XCTAssertTrue(raw.contains("A2"))
    }

    // MARK: 改进7：剥离 + 归档

    func testStripToSkeletonKeepsOnlyUserAndAgent() {
        let body = """
        @@SYSTEM
        secret system blob
        @@USER
        user question
        @@TURN-PROMPT turn=1
        anchor prompt injected
        @@ASSISTANT-RESPONSE turn=1
        thinking blob
        @@TOOL file_read
        tool output blob
        @@AGENT
        final answer
        """
        let skeleton = RawSessionStore.stripToSkeleton(body)
        XCTAssertTrue(skeleton.contains("user question"))
        XCTAssertTrue(skeleton.contains("final answer"))
        XCTAssertFalse(skeleton.contains("secret system blob"))
        XCTAssertFalse(skeleton.contains("anchor prompt injected"))
        XCTAssertFalse(skeleton.contains("thinking blob"))
        XCTAssertFalse(skeleton.contains("tool output blob"))
    }

    func testSplitSessionsHandlesUnclosedTail() {
        let raw = """
        @@SESSION-BEGIN 1101_0900 | x
        @@USER
        first
        @@SESSION-END 1101_0900 | y
        @@SESSION-BEGIN 1101_1000 | x
        @@USER
        second
        """
        let blocks = RawSessionStore.splitSessions(raw)
        XCTAssertEqual(blocks.count, 2)
        XCTAssertEqual(blocks[0].sessionId, "1101_0900")
        XCTAssertEqual(blocks[1].sessionId, "1101_1000")
        XCTAssertTrue(blocks[1].body.contains("second"))
    }

    func testArchivePastSessionsCreatesZipAndRemovesOriginals() throws {
        // 造一个"昨天"的原始记录
        let yesterday = Calendar.current.date(byAdding: .day, value: -1, to: Date())!
        let day = MemoryDirectory.dayString(yesterday)
        let raw = """
        @@SESSION-BEGIN 1101_0900 | 2026-11-01 09:00:00
        @@SYSTEM
        huge system blob
        @@USER
        帮我做记忆改造
        @@ASSISTANT-RESPONSE turn=1
        internal reasoning
        @@TOOL file_read
        noisy tool output
        @@AGENT
        已完成改造
        @@SESSION-END 1101_0900 | 2026-11-01 09:05:00
        """
        XCTAssertTrue(dir.write(raw, to: dir.rawFile(day: day)))
        XCTAssertTrue(dir.write("@@SESSION 1101_0900\n[T7] 摘要\n", to: dir.digestFile(day: day)))

        let archives = RawSessionStore(directory: dir).archivePastSessions()
        XCTAssertEqual(archives.count, 1)
        XCTAssertEqual(archives.first?.sessionIds, ["1101_0900"])

        let fm = FileManager.default
        XCTAssertFalse(fm.fileExists(atPath: dir.rawFile(day: day)), "原文应已删除")
        XCTAssertFalse(fm.fileExists(atPath: dir.digestFile(day: day)), "摘要原文应已删除")
        XCTAssertTrue(fm.fileExists(atPath: dir.archiveFile(day: day)), "应生成 zip")

        let history = dir.read(dir.allHistoryFile) ?? ""
        XCTAssertTrue(history.contains("帮我做记忆改造"))
        XCTAssertTrue(history.contains("已完成改造"))
        XCTAssertFalse(history.contains("huge system blob"))
        XCTAssertFalse(history.contains("noisy tool output"))

        let archive = try Archive(url: URL(fileURLWithPath: dir.archiveFile(day: day)),
                                  accessMode: .read)
        XCTAssertTrue(archive.contains { $0.path.contains("1101_0900") })
    }

    // MARK: 改进6：start_long_term_update 门控

    func testLongTermUpdateRejectedBeforeTurn5() async throws {
        var config = AgentConfig()
        config.workspace = workspace
        let session = Session(budget: ContextBudget(total: 30_000))
        let registry = ToolRegistry(config: config)
        registry.session = session
        let tool = StartLongTermUpdateTool(config: config, callbacks: nil, registry: registry)

        let rejected = try await tool.run([:])
        XCTAssertFalse(rejected.success, "轮数 <5 应直接拒绝")
        XCTAssertTrue(rejected.error.contains("拒绝"))

        // 推进到 5 轮后放行，并返回提纯 prompt / L1 当前内容 / L2-L3 写入指引。
        for _ in 0..<StartLongTermUpdateTool.minTurns { _ = session.startTurn(prompt: "x") }
        session.setKeyInfo("port=8080 已由 file_read 确认")
        let ready = try await tool.run(["focus": .string("env")])
        XCTAssertTrue(ready.success)
        XCTAssertTrue(ready.output.contains("DISTILL_READY"))
        XCTAssertTrue(ready.output.contains("【记忆提纯】"))
        XCTAssertTrue(ready.output.contains("global_mem.txt"))
        XCTAssertTrue(ready.output.contains("l1_current"))
        XCTAssertTrue(ready.output.contains("port=8080"))
    }

    // MARK: 改进3/4：L0/L1 拼接与按签名缓存

    func testSystemPromptIncludesL0ThenL1AndRefreshesOnFileChange() {
        // 用一个最小 LLMService；Agent 构造不发起请求，仅验证 systemPrompt 装配。
        final class NoopLLM: LLMService, @unchecked Sendable {
            let identifier = "noop"
            let supportsTools = true
            let supportsConcurrentIsolatedRequests = false
            func chatStream(messages: [ChatMessage], tools: [JSONValue],
                            temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
                AsyncThrowingStream { $0.yield(.init(kind: .done)); $0.finish() }
            }
        }
        var config = AgentConfig()
        config.workspace = workspace
        let agent = Agent(llm: NoopLLM(), config: config, callbacks: SilentCallbacks())
        defer { agent.close() }

        let first = agent.systemPrompt()
        XCTAssertTrue(first.contains("# [L0 META-SOP: memory_management_sop]"))
        XCTAssertTrue(first.contains("# [L1 Global Memory Insight]"))
        // L0 必须在 L1 之前。
        let l0 = first.range(of: "# [L0 META-SOP")!.lowerBound
        let l1 = first.range(of: "# [L1 Global Memory Insight]")!.lowerBound
        XCTAssertLessThan(l0, l1)
        // 签名未变 → 返回缓存的同一份，保证 KV 前缀逐字节稳定。
        XCTAssertEqual(agent.systemPrompt(), first)

        // L1 被 patch → 下一轮应重新注入（改进4「实时更新维护」）。
        XCTAssertTrue(dir.write("# CUSTOM\nL3: new_sop\n", to: dir.l1File))
        let refreshed = agent.systemPrompt()
        XCTAssertNotEqual(refreshed, first)
        XCTAssertTrue(refreshed.contains("# CUSTOM"))
    }

    /// 量化 L0+L1 常驻后 system 区对 rawCeiling 的挤占，便于评估是否需要
    /// 同步调大 kernelMaxContextTokens（详见改进说明）。
    func testSystemPromptTokenFootprint() {
        final class NoopLLM2: LLMService, @unchecked Sendable {
            let identifier = "noop2"
            let supportsTools = true
            let supportsConcurrentIsolatedRequests = false
            func chatStream(messages: [ChatMessage], tools: [JSONValue],
                            temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
                AsyncThrowingStream { $0.yield(.init(kind: .done)); $0.finish() }
            }
        }
        var config = AgentConfig()
        config.workspace = workspace
        let agent = Agent(llm: NoopLLM2(), config: config, callbacks: SilentCallbacks())
        defer { agent.close() }

        let chars = agent.systemPrompt().count
        let tokens = TokenEstimator.estimate(agent.systemPrompt())
        print("[memory-layering] system chars=\(chars) estimatedTokens=\(tokens) "
            + "budget=\(config.kernelMaxContextTokens) "
            + "reserve=\(ContextBudget(total: config.kernelMaxContextTokens).quota.reserve)")
        XCTAssertLessThan(tokens, config.kernelMaxContextTokens,
                          "常驻 SOP+L1 已超过整个上下文预算，必须缩短或调大预算")
    }

    // MARK: 全流程可靠性回归（本次审查修复项）

    /// 用户正文里粘贴 `@@SESSION-END` / `@@AGENT` 不得破坏 SESSION 分块或污染骨架。
    func testLiteralAtMarkersDoNotBreakSessionParsing() {
        let store = RawSessionStore(directory: dir)
        store.appendSession(systemText: "S",
                            userText: "@@AGENT\n这是伪造的标记\n@@SESSION-END",
                            turns: [], finalText: "真实答复")
        let day = MemoryDirectory.dayString()
        let raw = dir.read(dir.rawFile(day: day)) ?? ""
        let blocks = RawSessionStore.splitSessions(raw)
        XCTAssertEqual(blocks.count, 1, "伪造标记不应提前闭合 SESSION")
        let skeleton = RawSessionStore.stripToSkeleton(blocks[0].body)
        XCTAssertTrue(skeleton.contains("@@AGENT"), "转义行应还原为原文")
        XCTAssertTrue(skeleton.contains("这是伪造的标记"))
        XCTAssertTrue(skeleton.contains("真实答复"))
    }

    /// 归档必须把 session 写入持久化队列（寒暄/取消/无 auxLLM 时不丢）。
    func testArchiveEnqueuesPendingInsightsAndOnlyRemovesOnSuccess() {
        let yesterday = Calendar.current.date(byAdding: .day, value: -1, to: Date())!
        let day = MemoryDirectory.dayString(yesterday)
        let raw = """
        @@SESSION-BEGIN 1101_0900_001 | x
        @@SYSTEM abc -> \(day)-system.txt
        @@USER
        帮我改造记忆
        @@AGENT
        已改造
        @@SESSION-END 1101_0900_001 | y
        """
        XCTAssertTrue(dir.write(raw, to: dir.rawFile(day: day)))

        let archives = RawSessionStore(directory: dir).archivePastSessions()
        XCTAssertEqual(archives.count, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.rawFile(day: day)))
        XCTAssertEqual(dir.pendingInsightIds(), ["1101_0900_001"],
                       "归档后必须入队，供后续任意一轮消费")

        // 模拟提炼成功出队；失败/取消时队列仍在。
        dir.clearPendingInsights(["1101_0900_001"])
        XCTAssertTrue(dir.pendingInsightIds().isEmpty)
    }

    /// 同一天内多个 session 的 id 必须唯一（zip 条目重名会丢全文）。
    func testSessionIdsAreUniqueWithinSameMinute() {
        let a = MemoryDirectory.sessionId()
        let b = MemoryDirectory.sessionId()
        XCTAssertNotEqual(a, b)
    }

    /// L1 内容签名对等长 patch 也必须变化（mtime+size 会漏判）。
    func testSignatureDetectsEqualLengthPatch() {
        let before = dir.signature()
        // 等长替换：字符数相同，mtime 可能同秒
        XCTAssertTrue(dir.write(dir.read(dir.l1File)!.replacingOccurrences(
            of: "L2: 现空", with: "L2: 已填"), to: dir.l1File))
        XCTAssertNotEqual(dir.signature(), before)
    }

    /// 整理建议不得改写 L1（META-SOP 红线 + KV 前缀稳定）。
    func testTidyMemoryDoesNotOverwriteL1() async {
        let original = dir.read(dir.l1File) ?? ""
        let replies = [
            "=== L1 ===\n# [Global Memory Insight]\nL2: memory/global_mem.txt\n[RULES]\n1. 搜索先行\n"
            + "=== PROPOSAL ===\n- L2 瘦身建议\n",
        ]
        final class StubLLM: LLMService, @unchecked Sendable {
            let identifier = "stub"
            let supportsTools = false
            let supportsConcurrentIsolatedRequests = true
            let reply: String
            init(_ reply: String) { self.reply = reply }
            func chatStream(messages: [ChatMessage], tools: [JSONValue],
                            temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
                AsyncThrowingStream { c in
                    c.yield(.init(kind: .content, content: reply))
                    c.yield(.init(kind: .done)); c.finish()
                }
            }
        }
        let llm = StubLLM(replies[0])
        let maintenance = MemoryMaintenance(
            directory: dir,
            log: { _ in },
            collect: { msgs, _ in
                var out = ""
                do {
                    for try await e in llm.chatStream(messages: msgs, tools: [], temperature: 0.2) {
                        if e.kind == .content { out += e.content }
                        if e.kind == .done { break }
                    }
                } catch { return "" }
                return out
            })
        let changed = await maintenance.tidyMemory()
        XCTAssertTrue(changed)
        XCTAssertEqual(dir.read(dir.l1File), original, "L1 绝不能被后台整文件覆盖")
        XCTAssertTrue(dir.exists(dir.tidyProposalFile), "建议应落到 proposal 文件")
        XCTAssertTrue((dir.read(dir.tidyProposalFile) ?? "").contains("未自动执行"))
    }

    /// tracker 用归一化路径去重，且 reset 后重新视为首次。
    func testSOPTrackerCanonicalKeyAndReset() {
        let tracker = SOPReadTracker()
        XCTAssertTrue(tracker.markIfFirst("/tmp/ws/memory/x_sop.md"))
        XCTAssertFalse(tracker.markIfFirst("/tmp/ws/memory/x_sop.md"))
        tracker.reset()
        XCTAssertTrue(tracker.markIfFirst("/tmp/ws/memory/x_sop.md"))
        XCTAssertEqual(tracker.loadedCount, 1)
    }

    // MARK: 端到端：完整 Agent.run() 串联

    /// 模拟一次真实用户请求，验证「L0/L1 注入 → 全文落盘 → 摘要落盘 →
    /// 请求内前缀冻结」全链路，且用**真实 system prompt** 校验预算不越界。
    func testFullRunPersistsTranscriptAndKeepsPrefixStable() async {
        final class E2ELLM: LLMService, @unchecked Sendable {
            let identifier = "e2e"
            let supportsTools = false
            let supportsConcurrentIsolatedRequests = true
            let lock = NSLock()
            private var seenSystems: [String] = []
            func chatStream(messages: [ChatMessage], tools: [JSONValue],
                            temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
                let sys = messages.first(where: { $0.role == .system })?.content ?? ""
                lock.lock(); seenSystems.append(sys); lock.unlock()
                return AsyncThrowingStream { c in
                    c.yield(.init(kind: .content, content: "这是本次任务的完整答复内容。"))
                    c.yield(.init(kind: .done)); c.finish()
                }
            }
            var systems: [String] { lock.lock(); defer { lock.unlock() }; return seenSystems }
        }
        var config = AgentConfig()
        config.workspace = workspace
        config.maxIterations = 2
        let llm = E2ELLM()
        let agent = Agent(llm: llm, config: config, callbacks: SilentCallbacks())
        defer { agent.close() }

        let answer = await agent.run("请帮我确认服务端口并完成配置检查")
        XCTAssertFalse(answer.isEmpty)

        // 1) 模型实际看到的前缀包含 L0 与 L1，且整次请求内逐字节一致。
        let systems = llm.systems
        XCTAssertFalse(systems.isEmpty)
        XCTAssertTrue(systems[0].contains("# [L0 META-SOP"))
        XCTAssertTrue(systems[0].contains("# [L1 Global Memory Insight]"))
        XCTAssertTrue(Set(systems).count == 1, "同一请求内 system 前缀必须完全一致（KV 复用）")

        // 2) 预算：真实 system 已被记账，rawCeiling 不漏算；总用量未越硬顶。
        let budget = agent.session.budget
        XCTAssertEqual(budget.used("system"), TokenEstimator.estimate(systems[0]))
        XCTAssertLessThan(budget.used(), budget.total - budget.quota.reserve,
                          "首轮不得吃进 reserve（否则会触发 totalOver 紧急折叠）")

        // 3) 全文与摘要都落盘，且 system 引用指向的那份与本轮实发一致。
        let day = MemoryDirectory.dayString()
        let raw = dir.read(dir.rawFile(day: day)) ?? ""
        XCTAssertTrue(raw.contains("请帮我确认服务端口并完成配置检查"))
        XCTAssertTrue(raw.contains("这是本次任务的完整答复内容。"))
        XCTAssertEqual(dir.read(dir.path("raw_session/\(day)-system.txt")), systems[0],
                       "落盘 system 必须是本轮模型实际看到的那一份")
        let digest = dir.read(dir.digestFile(day: day)) ?? ""
        XCTAssertTrue(digest.contains("@@SESSION"))
    }
}
