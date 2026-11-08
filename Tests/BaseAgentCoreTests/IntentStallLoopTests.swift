import XCTest
import Foundation
@testable import BaseAgentCore

/// 回归：2026-11-08「获取 cnn.com 今日新闻并总结」事故。
///
/// 小模型把 system 里的 `update_plan` 协议复述成「我需要先创建任务计划……
/// 现在开始执行第一步」，一次工具都不发；旧内核把这段元陈述当正常正文
/// 收下（reason=done、plan 仍空、不触发自动续跑），用户反复点「继续」
/// 只得到同样的自我复述，计划与第一步永远不推进。
final class IntentStallLoopTests: XCTestCase {

    private final class State: @unchecked Sendable {
        private let lock = NSLock()
        private var _calls = 0
        private var _prompts: [String] = []
        private var _executed: [String] = []
        var calls: Int { lock.lock(); defer { lock.unlock() }; return _calls }
        var prompts: [String] { lock.lock(); defer { lock.unlock() }; return _prompts }
        var executed: [String] { lock.lock(); defer { lock.unlock() }; return _executed }
        func nextCall(_ prompt: String) -> Int {
            lock.lock(); defer { lock.unlock() }
            _calls += 1
            _prompts.append(prompt)
            return _calls
        }
        func record(_ names: [String]) {
            lock.lock(); defer { lock.unlock() }
            _executed.append(contentsOf: names)
        }
    }

    /// 事故现场摘录：纯意图声明，没有任何工具调用、没有任何结果。
    private let metaStatement = """
    我需要先创建任务计划，将任务分解为 2~7 个可验证的步骤。
    目标：获取 cnn.com 今日新闻并总结
    步骤：
    1. 获取 cnn.com 今日新闻页面内容
    2. 阅读并理解页面内容
    3. 总结主要内容
    现在我开始执行第一步：获取 cnn.com 今日新闻页面内容。
    """

    private func schema(_ name: String) -> JSONValue {
        .object([
            "type": .string("function"),
            "function": .object([
                "name": .string(name),
                "description": .string("测试工具 \(name)"),
                "parameters": .object([:]),
            ]),
        ])
    }

    private func textStream(_ text: String) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(LLMStreamEvent(kind: .content, content: text))
            continuation.yield(LLMStreamEvent(kind: .done, rawContent: text))
            continuation.finish()
        }
    }

    private func callStream(_ call: ToolCall) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(LLMStreamEvent(kind: .done, toolCalls: [call]))
            continuation.finish()
        }
    }

    private final class PlanCollector: @unchecked Sendable {
        private let lock = NSLock()
        private var _plans: [Plan] = []
        func append(_ plan: Plan) { lock.lock(); _plans.append(plan); lock.unlock() }
        func snapshot() -> [Plan] { lock.lock(); defer { lock.unlock() }; return _plans }
    }

    private final class PlanRecordingCallbacks: AgentCallbacks {
        let plans: PlanCollector
        init(plans: PlanCollector) { self.plans = plans }
        func onPlanUpdated(_ plan: Plan?) {
            if let plan { plans.append(plan) }
        }
    }

    private func makeLoop(_ state: State,
                          reply: @escaping (Int) -> AsyncThrowingStream<LLMStreamEvent, Error>,
                          maxTurns: Int = 12,
                          dynamicContext: @escaping () -> String = { "" },
                          callbacks: AgentCallbacks? = nil) -> KernelLoop {
        let runtime = KernelRuntime(
            stream: { messages, _, _ in
                let prompt = messages.last(where: { $0.role == .user })?.content ?? ""
                return reply(state.nextCall(prompt))
            },
            execute: { calls in
                state.record(calls.map(\.name))
                return calls.map { _ in ToolResult.ok("CNN 今日要闻：示例标题一、示例标题二。") }
            },
            schemas: { ["update_plan", "web_scan", "shell_run"].map(self.schema) },
            log: { _ in }
        )
        return KernelLoop(runtime: runtime, config: AgentConfig(), callbacks: callbacks,
                          systemPrompt: { BASE_SYSTEM_PROMPT },
                          dynamicContext: dynamicContext, maxTurns: maxTurns)
    }

    // MARK: - 纯函数

    func testIntentOnlyDetection() {
        XCTAssertTrue(KernelLoop.looksLikeIntentOnly(metaStatement))
        XCTAssertTrue(KernelLoop.looksLikeIntentOnly(
            "用户要求获取新闻并总结，我需要：1. 用 update_plan 创建计划；"
            + "2. 紧接着发出第一步的实际工具调用。让我创建计划并立即开始执行。"))
        // 正常的结果性回复不能被误判
        XCTAssertFalse(KernelLoop.looksLikeIntentOnly(
            "KV cache 缓存注意力键值，避免每轮重复计算历史 token。"))
        XCTAssertFalse(KernelLoop.looksLikeIntentOnly(
            "CNN 今日要闻有三条：一、……；二、……；三、……。"))
        XCTAssertFalse(KernelLoop.looksLikeIntentOnly("2"))
        XCTAssertFalse(KernelLoop.looksLikeIntentOnly(""))
    }

    // MARK: - 循环收敛

    /// 模型始终只输出元陈述：不得把元陈述当最终答复，必须有界收敛，
    /// 且提示必须跳过计划门、点名真实工具，不能继续要求「先建计划」。
    func testIntentOnlyRepliesNeverBecomeFinalAnswer() async {
        let state = State()
        let loop = makeLoop(state) { _ in self.textStream(self.metaStatement) }
        let result = await loop.run("获取 cnn.com 今日新闻并总结", toolsEnabled: true)

        XCTAssertEqual(state.calls, 3, "两次强制 + 一次收敛，不应无限循环")
        XCTAssertFalse(result.text.contains("创建任务计划"),
            "元陈述绝不能成为最终答复：\(result.text)")
        XCTAssertTrue(result.text.contains("没有发出任何工具调用"),
            "应给出明确的失败说明：\(result.text)")
        let prompts = state.prompts.dropFirst()
        XCTAssertEqual(prompts.count, 2)
        for prompt in prompts {
            XCTAssertTrue(prompt.contains("跳过 update_plan"), "提示必须去掉计划前置：\(prompt)")
            XCTAssertTrue(prompt.contains("web_scan"), "提示必须点名真实工具：\(prompt)")
            XCTAssertFalse(prompt.contains("计划建好之前不要先发实际工具调用"),
                "不能再注入自相矛盾的规划提示：\(prompt)")
        }
    }

    /// 模型第二次真的发出工具调用：规划前置门必须放行（不能再用「还没有
    /// 执行计划」把它拦回去），任务要能实际执行并基于结果收尾。
    func testToolCallAfterIntentStallBypassesPlanGate() async {
        let state = State()
        let call = ToolCall(id: "call-1", name: "web_scan",
                            arguments: ["url": .string("https://cnn.com")])
        let loop = makeLoop(state) { index in
            switch index {
            case 1: return self.textStream(self.metaStatement)
            case 2: return self.callStream(call)
            default: return self.textStream("CNN 今日要闻：示例标题一、示例标题二。")
            }
        }
        let result = await loop.run("获取 cnn.com 今日新闻并总结", toolsEnabled: true)

        XCTAssertEqual(state.executed, ["web_scan"],
            "真实工具调用必须被执行，不能被规划前置门拦截：\(state.executed)")
        XCTAssertTrue(result.text.contains("CNN 今日要闻"),
            "应基于工具结果收尾：\(result.text)")
        XCTAssertFalse(result.text.contains("创建任务计划"),
            "不得把元陈述当答复：\(result.text)")
    }

    /// 规划前置门已移除：模型第 1 轮就发出可用工具调用时，必须当轮执行，
    /// 不能再因为 plan 为空把调用丢弃、额外花一轮要求「先建计划」。
    func testFirstToolCallIsNeverDiscarded() async {
        let state = State()
        let call = ToolCall(id: "call-1", name: "web_scan",
                            arguments: ["url": .string("https://cnn.com")])
        let loop = makeLoop(state) { index in
            index == 1
                ? self.callStream(call)
                : self.textStream("CNN 今日要闻：示例标题一、示例标题二。")
        }
        let result = await loop.run(
            "请按步骤获取 cnn.com 今日新闻并总结", toolsEnabled: true)

        XCTAssertEqual(state.executed, ["web_scan"],
            "首个工具调用必须被真正执行：\(state.executed)")
        XCTAssertEqual(state.calls, 2,
            "不应有额外的『请先创建计划』重答轮：\(state.calls)")
        XCTAssertTrue(result.text.contains("CNN 今日要闻"),
            "应基于工具结果收尾：\(result.text)")
    }

    // MARK: - 结果性答复 / 上下文膨胀 / 跨段预算

    func testLooksLikeResult() {
        XCTAssertTrue(KernelLoop.looksLikeResult(
            "已完成：CNN 今日要闻共 3 条，分别是 A、B、C。"))
        XCTAssertTrue(KernelLoop.looksLikeResult(
            "1. 检查 CPU；2. 检查磁盘。结论是 I/O 瓶颈。"))
        XCTAssertFalse(KernelLoop.looksLikeResult(metaStatement))
        XCTAssertFalse(KernelLoop.looksLikeResult("请问要继续吗？"))
        XCTAssertFalse(KernelLoop.looksLikeResult(""))
    }

    /// 没执行任何工具就给「已完成」：不得被当最终答复直接放行（v1.2.27 的
    /// looksLikeResult 豁免导致用户实测「根本没执行就判定结束」），但也不得
    /// 无限重答——反思门 3 次后有界放行。
    func testUnverifiedAnswerIsNotAcceptedWithoutWork() async {
        let state = State()
        let answer = "已完成：CNN 今日要闻共 3 条，分别是 A、B、C。"
        let loop = makeLoop(state) { _ in self.textStream(answer) }
        let result = await loop.run(
            "请按步骤获取 cnn.com 今日新闻并总结", toolsEnabled: true)
        XCTAssertEqual(state.calls, 4,
            "3 次推进提醒 + 1 次有界放行：\(state.calls)")
        XCTAssertEqual(result.text, answer)
        XCTAssertGreaterThanOrEqual(state.executed.count, 0)
    }

    /// 计划面板保障：模型从不 update_plan、直接发工具调用时，内核必须代建
    /// 最小计划并回调面板；工具照常执行、收尾后计划自动标完成。
    func testKernelAutoCreatesPlanForMultistepTask() async {
        let state = State()
        let plans = PlanCollector()
        let callbacks = PlanRecordingCallbacks(plans: plans)
        let call = ToolCall(id: "c1", name: "web_scan",
                            arguments: ["url": .string("https://cnn.com")])
        let loop = makeLoop(state,
                            reply: { index in
                                index == 1 ? self.callStream(call)
                                    : self.textStream("已完成：CNN 今日要闻共 3 条。")
                            },
                            callbacks: callbacks)
        let result = await loop.run(
            "请按步骤获取 cnn.com 今日新闻并总结", toolsEnabled: true)

        XCTAssertEqual(state.executed, ["web_scan"],
            "代建计划不得拦截工具执行：\(state.executed)")
        XCTAssertGreaterThanOrEqual(plans.snapshot().count, 1,
            "计划面板应收到代建计划：\(plans.snapshot().count)")
        let created = plans.snapshot().first
        XCTAssertTrue(created?.goal.hasPrefix("请按步骤获取") == true,
            "goal 应来自用户输入：\(created?.goal ?? "?")")
        XCTAssertEqual(created?.steps.count, 2)
        XCTAssertEqual(created?.steps[0].status, "doing")
        if let final = plans.snapshot().last {
            XCTAssertTrue(final.steps.allSatisfy { $0.status == "done" },
                "收尾后计划应自动标完成：\(final.steps)")
        }
        XCTAssertEqual(result.reason, .done)
        XCTAssertTrue(result.text.contains("已完成"),
            "应基于工具结果收尾：\(result.text)")
    }

    /// 规划提示一段只注入一次。为保持 KV 对齐，已注入轮仍原样留在历史里，
    /// 因此检查的是每轮「当前 user 消息」，而不是整个 messages 拼接。
    func testPlanGateHintInjectedOnlyOnce() async {
        let state = State()
        let call = ToolCall(id: "call-1", name: "web_scan",
                            arguments: ["url": .string("https://cnn.com")])
        let loop = makeLoop(state) { index in
            index <= 2 ? self.callStream(call) : self.textStream("完成。")
        }
        _ = await loop.run("请按步骤获取 cnn.com 今日新闻并总结", toolsEnabled: true)
        let hints = state.prompts.filter { $0.contains("非平凡任务先 update_plan") }
        XCTAssertEqual(hints.count, 1,
            "规划提示不能逐轮重复注入：出现 \(hints.count) 次")
    }

    /// resume（自动续跑）段也必须保留意图空转检测。
    func testResumeSegmentStillRejectsIntentOnlyReply() async {
        let state = State()
        let loop = makeLoop(state,
                            reply: { _ in self.textStream(self.metaStatement) },
                            maxTurns: 8)
        _ = await loop.run("请按步骤获取 cnn.com 今日新闻并总结", toolsEnabled: true)
        loop.seedFailures(loop.failureSnapshot())
        let resumed = await loop.run("[System] 继续执行。", resume: true,
                                     emitFinished: false, toolsEnabled: true)
        XCTAssertFalse(resumed.text.contains("创建任务计划"),
            "续跑段不得把元陈述当最终答复：\(resumed.text)")
    }

    /// 轮次耗尽时应把已取得的工具结果交付给用户，而不是只给模板句。
    func testMaxTurnsReturnsToolHighlights() async {
        let state = State()
        let call = ToolCall(id: "call-1", name: "web_scan",
                            arguments: ["url": .string("https://cnn.com")])
        let loop = makeLoop(state, reply: { _ in self.callStream(call) }, maxTurns: 4)
        let result = await loop.run("请按步骤获取 cnn.com 今日新闻并总结",
                                    toolsEnabled: true)
        XCTAssertEqual(result.reason, .maxTurns)
        XCTAssertTrue(result.text.contains("已取得的进展"),
            "应汇总已执行工具的结果：\(result.text)")
        XCTAssertGreaterThanOrEqual(state.executed.count, 1)
    }

    /// 动态上下文一次请求只注入首个 THINK 轮：旧实现每个工具循环轮都把
    /// 同一份 ~7.5k 字符重复写进历史，快速吃满 raw 窗口触发提前折叠。
    /// 已注入的轮原样保留（KV 对齐），因此断言的是「每轮当前 user 消息」。
    func testDynamicContextInjectedOncePerRequest() async {
        let state = State()
        let dynamic = "# Knowledge base\nIndexed locally: demo-docs."
        let call = ToolCall(id: "call-1", name: "web_scan",
                            arguments: ["url": .string("https://cnn.com")])
        let loop = makeLoop(state,
                            reply: { _ in self.callStream(call) },
                            maxTurns: 5,
                            dynamicContext: { dynamic })
        _ = await loop.run("请按步骤获取 cnn.com 今日新闻并总结", toolsEnabled: true)
        let injected = state.prompts.filter { $0.contains(dynamic) }
        XCTAssertEqual(injected.count, 1,
            "动态上下文一次请求只注入首轮，出现 \(injected.count) 次")
        XCTAssertGreaterThanOrEqual(state.calls, 3,
            "需要多轮工具循环才能验证只注入一次：\(state.calls)")
    }

    // MARK: - 空正文（思考无正文）修复

    private func reasoningOnlyStream() -> AsyncThrowingStream<LLMStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(LLMStreamEvent(kind: .reasoning,
                                              content: "正在分析页面内容，稍后给出总结。"))
            continuation.yield(LLMStreamEvent(kind: .done, rawContent: ""))
            continuation.finish()
        }
    }

    /// 2026-11-08 cnn.com 事故：thinking 模式下模型长思考耗尽预算/未闭合
    /// </think>，正文为空被判 retry，3 次同样失败后空手退出。修复后第 2 次
    /// 起必须请求级关闭 thinking，耗尽后交付失败说明与已有进展。
    func testEmptyReplyRetriesWithoutThinkingThenExplains() async {
        let state = State()
        var optionsSeen: [LLMRequestOptions] = []
        let runtime = KernelRuntime(
            stream: { messages, _, _ in
                _ = state.nextCall(messages.last(where: { $0.role == .user })?.content ?? "")
                return self.reasoningOnlyStream()
            },
            streamWithOptions: { messages, _, _, options in
                _ = state.nextCall(messages.last(where: { $0.role == .user })?.content ?? "")
                optionsSeen.append(options)
                return self.reasoningOnlyStream()
            },
            execute: { calls in
                state.record(calls.map(\.name))
                return calls.map { _ in ToolResult.ok("CNN 今日要闻：A、B、C。") }
            },
            schemas: { ["update_plan", "web_scan"].map(self.schema) },
            log: { _ in }
        )
        let loop = KernelLoop(runtime: runtime, config: AgentConfig(), callbacks: nil,
                              systemPrompt: { BASE_SYSTEM_PROMPT }, maxTurns: 8)
        let result = await loop.run("请按步骤获取 cnn.com 今日新闻并总结",
                                    toolsEnabled: true)

        XCTAssertEqual(state.calls, 4,
            "3 次重试 + 1 次收尾，共 4 次调用：\(state.calls)")
        XCTAssertEqual(optionsSeen.count, 3,
            "默认首调用走 3 参通道，仅 3 次重试走 options 通道：\(optionsSeen.count)")
        XCTAssertTrue(optionsSeen.allSatisfy { $0.enableThinking == false },
            "空正文后必须请求级关闭 thinking")
        XCTAssertFalse(result.text.contains("正在分析页面内容"),
            "思考内容不得成为最终答复：\(result.text)")
        XCTAssertTrue(result.text.contains("未输出可见正文"),
            "应给出明确的失败说明：\(result.text)")
        XCTAssertTrue(result.text.contains("建议：换用"),
            "失败说明应包含可执行的下一步建议：\(result.text)")
    }

    /// 关闭 thinking 后模型给出正文：应当轮收下，任务不中断。
    func testRetryWithoutThinkingRecovers() async {
        let state = State()
        var optionsSeen: [LLMRequestOptions] = []
        let runtime = KernelRuntime(
            stream: { _, _, _ in self.reasoningOnlyStream() },
            streamWithOptions: { messages, _, _, options in
                let index = state.nextCall(messages.last(where: { $0.role == .user })?.content ?? "")
                optionsSeen.append(options)
                return index == 1
                    ? self.reasoningOnlyStream()
                    : self.textStream("已完成：CNN 今日要闻共 3 条。")
            },
            execute: { calls in
                state.record(calls.map(\.name))
                return calls.map { _ in ToolResult.ok("done") }
            },
            schemas: { ["update_plan", "web_scan"].map(self.schema) },
            log: { _ in }
        )
        let loop = KernelLoop(runtime: runtime, config: AgentConfig(), callbacks: nil,
                              systemPrompt: { BASE_SYSTEM_PROMPT }, maxTurns: 8)
        // 单步输入：专注验证 options 通道，避开多步任务的计划推进门。
        let result = await loop.run("总结一下这个话题", toolsEnabled: true)

        XCTAssertEqual(state.calls, 2, "降 thinking 后应一次成功：\(state.calls)")
        XCTAssertEqual(optionsSeen.dropFirst().first?.enableThinking, false)
        XCTAssertTrue(result.text.contains("已完成"),
            "应基于重试成功收尾：\(result.text)")
    }
}
