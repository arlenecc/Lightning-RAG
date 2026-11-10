import XCTest
@testable import BaseAgentCore

/// 验证「RAG 大返回不进 KV 历史」的收紧逻辑：与 `KernelLoop` 真实路径共用
/// 同一个纯函数 `trimRagHistory`，因此直接证明方案 B 的核心行为。
final class KernelLoopRagHistoryTests: XCTestCase {

    /// 大体积 RAG 结果应被截到 limit 以内，并附上截断标记（含原始长度）。
    func testLargeRagResultIsCapped() {
        let big = String(repeating: "猫", count: 5_000)
        let out = KernelLoop.trimRagHistory("rag_tools", big, limit: 2_000)

        XCTAssertLessThanOrEqual(out.count, 2_000 + 80, "应截到上限附近")
        XCTAssertTrue(out.hasPrefix(String(repeating: "猫", count: 2_000)),
                      "应保留最相关头部")
        XCTAssertTrue(out.contains("已截断至 2000 字符"))
        XCTAssertTrue(out.contains("原始 5000 字符"))
    }

    /// 小体积 RAG 结果不应被改动（避免无谓截断影响质量）。
    func testSmallRagResultUntouched() {
        let small = String(repeating: "x", count: 500)
        let out = KernelLoop.trimRagHistory("rag_tools", small, limit: 2_000)
        XCTAssertEqual(out, small)
    }

    /// 非 RAG 工具（如 file_tools）即使很大也不能被这套逻辑收紧。
    func testNonRagToolNotTrimmed() {
        let big = String(repeating: "y", count: 5_000)
        let out = KernelLoop.trimRagHistory("file_tools", big, limit: 2_000)
        XCTAssertEqual(out, big, "非 rag_tools 工具不应被收紧")
    }

    /// limit <= 0 视为关闭：原样返回。
    func testDisabledWhenLimitZero() {
        let big = String(repeating: "z", count: 5_000)
        let out = KernelLoop.trimRagHistory("rag_tools", big, limit: 0)
        XCTAssertEqual(out, big)
    }

    /// 其它 RAG 子命令（rag_tools(op=status/outline/ingest) 等）也应被收紧。
    func testOtherRagToolsTrimmed() {
        let big = String(repeating: "w", count: 5_000)
        let out = KernelLoop.trimRagHistory("rag_tools", big, limit: 2_000)
        XCTAssertLessThanOrEqual(out.count, 2_000 + 80)
        XCTAssertTrue(out.contains("已截断至 2000 字符"))
    }

    // MARK: - 端到端：整条工具执行链路

    /// 真让 `KernelLoop` 跑一轮「模型发出 rag_search → 执行大返回 → finishTurn 进历史」，
    /// 断言进历史的内容被截到 `kernelRagHistoryChars` 以内（接线 + 逻辑一起验证）。
    func testRagResultTrimmedInHistoryThroughLoop() async {
        let session = Session(budget: ContextBudget(total: 60_000), rawTurns: 8)
        var config = AgentConfig()
        config.kernelConsolidateEvery = 8
        config.kernelRagHistoryChars = 2_000

        let runtime = KernelRuntime(
            stream: { messages, _, _ in
                // 第二轮（历史里已含 .tool 结果）：给出最终答复；否则第一轮回工具调用。
                if messages.contains(where: { $0.role == .tool }) {
                    return AsyncThrowingStream { c in
                        c.yield(.init(kind: .content, content: "ok"))
                        c.yield(.init(kind: .done))
                        c.finish()
                    }
                }
                return AsyncThrowingStream { c in
                    c.yield(.init(kind: .done, toolCalls: [
                        ToolCall(id: "k1", name: "rag_tools",
                                 arguments: ["op": .string("search"),
                                             "query": .string("欧洲猫"), "top_k": .int(3)])
                    ]))
                    c.finish()
                }
            },
            execute: { calls in
                calls.map { _ in .ok(String(repeating: "猫", count: 5_000)) }
            },
            schemas: { [] },
            log: { _ in }
        )

        let loop = KernelLoop(runtime: runtime, config: config,
                              systemPrompt: { "SYS" }, session: session, maxTurns: 3)
        _ = await loop.run("猫图鉴有哪些欧洲猫")  // 短输入，绕过规划前置门

        let ragTurn = session.turns.first {
            $0.toolResults.contains { $0.name == "rag_tools" }
        }
        XCTAssertNotNil(ragTurn, "应有一轮包含 rag_tools 结果")
        let rec = ragTurn!.toolResults.first { $0.name == "rag_tools" }!
        XCTAssertLessThanOrEqual(rec.content.count, 2_000 + 80,
                                 "进历史的内容应被截到 2000 附近")
        XCTAssertTrue(rec.content.contains("已截断至 2000 字符"))
        XCTAssertTrue(rec.content.contains("原始 5000 字符"))
    }
}
