import XCTest
import Foundation
@testable import BaseAgentCore

/// 2026-11-09 cnn.com 真机事故回归。
///
/// 会话轨迹显示两次失败：
/// 1) 模型连续 3 轮只输出 `<summary>…</summary>` 元标签（既无工具调用也无
///    正文），旧内核把它当「空正文」重试 3 次，注入的提示只要求「输出简短
///    正文」，反而诱导模型继续复述元陈述，用户看到 3 次失败后空手结束；
/// 2) 另一次已经 web_scan 成功并写出总结，却被成果核验轮的
///    `<review>fix</review>` 元标记整份覆盖，最终答复只剩一句 fix。
final class CnnSummaryRegressionTests: XCTestCase {

    private final class State: @unchecked Sendable {
        private let lock = NSLock()
        private var _calls = 0
        private var _prompts: [String] = []
        private var _executed: [String] = []
        private var _urls: [String] = []
        var calls: Int { lock.lock(); defer { lock.unlock() }; return _calls }
        var prompts: [String] { lock.lock(); defer { lock.unlock() }; return _prompts }
        var executed: [String] { lock.lock(); defer { lock.unlock() }; return _executed }
        var urls: [String] { lock.lock(); defer { lock.unlock() }; return _urls }
        func nextCall(_ prompt: String) -> Int {
            lock.lock(); defer { lock.unlock() }
            _calls += 1
            _prompts.append(prompt)
            return _calls
        }
        func record(_ calls: [ToolCall]) {
            lock.lock(); defer { lock.unlock() }
            _executed.append(contentsOf: calls.map(\.name))
            _urls.append(contentsOf: calls.compactMap {
                $0.arguments["url"]?.stringValue
            })
        }
    }

    private let summaryOnly = """

    <summary>直接调用 web_scan 抓取 cnn.com 最新新闻</summary>
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

    private func makeLoop(
        _ state: State,
        reply: @escaping (Int) -> AsyncThrowingStream<LLMStreamEvent, Error>,
        toolResult: @escaping () -> ToolResult = {
            .ok("CNN 今日要闻：示例标题一、示例标题二。")
        },
        maxTurns: Int = 12
    ) -> KernelLoop {
        let runtime = KernelRuntime(
            stream: { messages, _, _ in
                let prompt = messages.last(where: { $0.role == .user })?.content ?? ""
                return reply(state.nextCall(prompt))
            },
            execute: { calls in
                state.record(calls)
                return calls.map { _ in toolResult() }
            },
            schemas: { ["update_plan", "web_scan", "shell_run"].map(self.schema) },
            log: { _ in }
        )
        return KernelLoop(runtime: runtime, config: AgentConfig(), callbacks: nil,
                          systemPrompt: { BASE_SYSTEM_PROMPT }, maxTurns: maxTurns)
    }

    // MARK: - URL 规范化（工具层）

    func testBareDomainIsNormalized() {
        XCTAssertEqual(WebURLPolicy.normalizedURLString("cnn.com"),
                       "https://cnn.com")
        XCTAssertEqual(WebURLPolicy.normalizedURLString(" www.cnn.com/world "),
                       "https://www.cnn.com/world")
        XCTAssertEqual(WebURLPolicy.normalizedURLString("https://cnn.com/x"),
                       "https://cnn.com/x")
        XCTAssertNil(WebURLPolicy.normalizedURLString("file:///etc/passwd"))
        XCTAssertNil(WebURLPolicy.normalizedURLString("javascript:alert(1)"))
        XCTAssertNil(WebURLPolicy.normalizedURLString("not a url"))
        XCTAssertNil(WebRenderTool.validateURL("cnn.com"),
                     "web_render 也必须接受裸域名")
        XCTAssertNotNil(WebRenderTool.validateURL("ftp://example.com/x"))
    }

    func testFirstWebURLFindsBareDomainInChineseSentence() {
        XCTAssertEqual(KernelLoop.firstWebURL(in: "去cnn.com获取最新的新闻，并进行总结"),
                       "https://cnn.com")
        XCTAssertEqual(KernelLoop.firstWebURL(in: "抓取 https://www.cnn.com/2026/11/09 的新闻"),
                       "https://www.cnn.com/2026/11/09")
        XCTAssertNil(KernelLoop.firstWebURL(in: "总结一下这个话题"))
        XCTAssertNil(KernelLoop.firstWebURL(in: "读一下 config.js 并总结"),
                     "文件名不能被误判成网址")
    }

    // MARK: - 只有 <summary> 的轮次

    /// 元标签不能再走「空正文重试」：提示必须要求真实工具调用，且给出可
    /// 照抄的 URL 参数，不能出现只要求「输出简短正文」的诱导文案。
    func testSummaryOnlyReplyForcesRealToolCallInsteadOfEmptyRetry() async {
        let state = State()
        let loop = makeLoop(state) { _ in self.textStream(self.summaryOnly) }
        _ = await loop.run("去cnn.com获取最新的新闻，并进行总结", toolsEnabled: true)

        let prompts = state.prompts.dropFirst()
        XCTAssertFalse(prompts.isEmpty, "至少要给出一轮强制工具调用提示")
        for prompt in prompts.prefix(2) {
            XCTAssertTrue(prompt.contains("web_scan"), "必须点名真实工具：\(prompt)")
            XCTAssertTrue(prompt.contains("https://cnn.com"),
                          "必须给出可照抄的规范化 URL：\(prompt)")
            XCTAssertFalse(prompt.contains("请直接输出 <summary>"),
                           "不能再诱导模型继续输出元标签：\(prompt)")
        }
    }

    /// 3 轮元标签后内核代发 web_scan，用户至少拿到真实页面内容，
    /// 绝不「什么都没做就结束」。
    func testSummaryOnlyStallFallsBackToKernelFetch() async {
        let state = State()
        let loop = makeLoop(state) { _ in self.textStream(self.summaryOnly) }
        let result = await loop.run("去cnn.com获取最新的新闻，并进行总结",
                                    toolsEnabled: true)

        XCTAssertTrue(state.executed.contains("web_scan"),
                      "意图空转耗尽后必须由内核代发 web_scan：\(state.executed)")
        XCTAssertEqual(state.urls.first, "https://cnn.com",
                       "兜底调用必须使用规范化 URL：\(state.urls)")
        XCTAssertTrue(result.text.contains("CNN 今日要闻"),
                      "即使模型仍不给正文，也要交付已抓取的真实内容：\(result.text)")
        XCTAssertFalse(result.text.contains("未输出可见正文"),
                       "不能再以『连续 3 轮空正文』收尾：\(result.text)")
    }

    /// 兜底拿到数据后，模型下一轮给出总结：直接收下，不被门控再拖一轮。
    func testFallbackFetchThenModelSummarizes() async {
        let state = State()
        let loop = makeLoop(state) { index in
            index <= 3
                ? self.textStream(self.summaryOnly)
                : self.textStream("CNN 今日要闻有两条：示例标题一、示例标题二。")
        }
        let result = await loop.run("去cnn.com获取最新的新闻，并进行总结",
                                    toolsEnabled: true)

        XCTAssertTrue(state.executed.contains("web_scan"))
        XCTAssertTrue(result.text.contains("CNN 今日要闻有两条"),
                      "应基于真实数据收下总结：\(result.text)")
        XCTAssertFalse(result.text.contains("<summary>"),
                       "元标签不得成为最终答复：\(result.text)")
        XCTAssertEqual(result.reason, .done)
    }

    // MARK: - 成果核验反思

    func testReviewMetaOnlyDetection() {
        XCTAssertTrue(KernelLoop.isReviewMetaOnly("<review>fix</review>"))
        XCTAssertTrue(KernelLoop.isReviewMetaOnly("<review>pass</review>"))
        XCTAssertTrue(KernelLoop.isReviewMetaOnly("fix"))
        XCTAssertTrue(KernelLoop.isReviewMetaOnly(""))
        XCTAssertFalse(KernelLoop.isReviewMetaOnly(
            "CNN 今日要闻有两条：示例标题一、示例标题二。"))
    }

    /// 真机事故：核验轮连吐两次 `<review>fix</review>`，旧内核把元标记
    /// 当最终答复，整份新闻总结被覆盖。
    func testReviewFixTagNeverReplacesFinalAnswer() async {
        let state = State()
        let call = ToolCall(id: "c1", name: "web_scan",
                            arguments: ["url": .string("https://cnn.com")])
        let answer = "CNN 今日要闻有两条：示例标题一、示例标题二。"
        let loop = makeLoop(state) { index in
            switch index {
            case 1: return self.callStream(call)
            case 2: return self.textStream(answer)
            default: return self.textStream("<review>fix</review>")
            }
        }
        let result = await loop.run("去cnn.com获取最新的新闻，并进行总结",
                                    toolsEnabled: true)

        XCTAssertTrue(result.text.hasPrefix(answer),
                      "元标记必须让位给候选答复：\(result.text)")
        XCTAssertFalse(result.text.contains("fix"), "fix 不得成为答案")
    }

    /// 工具全部失败时不进入成果核验：没有真实数据就没有「审视结果」。
    func testReviewIsSkippedWithoutSuccessfulToolResult() async {
        let state = State()
        let call = ToolCall(id: "c1", name: "web_scan",
                            arguments: ["url": .string("https://cnn.com")])
        let loop = makeLoop(
            state,
            reply: { index in
                if index == 1 { return self.callStream(call) }
                return self.textStream("<review>pass</review>")
            },
            toolResult: { .fail("HTTP 403") })
        _ = await loop.run("去cnn.com获取最新的新闻，并进行总结",
                           toolsEnabled: true)

        XCTAssertLessThan(state.calls, 4,
                          "无成功工具结果时不应再消耗核验轮：\(state.calls)")
    }
}
