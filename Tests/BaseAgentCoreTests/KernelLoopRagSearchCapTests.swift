import XCTest
@testable import BaseAgentCore

/// 验证「每次请求内 rag_tools(op="search") 硬上限 = 5」：
/// 模型反复请求检索时，只有前 5 次真正到达工具执行层，超出的被内核预置为失败，
/// 逼模型基于已有结果 + 大纲摘要综合作答，杜绝无效重复检索。
final class KernelLoopRagSearchCapTests: XCTestCase {

    /// 跑一轮 loop，假模型在第一轮发 `firstTool`，之后每轮发带序号的 rag_tools(op="search")。
    /// 返回：`executed`（真正到达 execute 的 ToolCall 序列，证明硬上限）、
    /// `emitted`（模型实际发出的检索次数）、`session`（用于查历史）。
    /// `rawTurns` 设大以避免滑动窗口把早期轮折出 `turns`，便于断言历史。
    private func runWith(firstTool: ToolCall, maxTurns: Int = 15) async
        -> (executed: [ToolCall], emitted: Int, session: Session) {
        var executed = [ToolCall]()
        var emitted = 0
        var callCounter = [0]

        let session = Session(budget: ContextBudget(total: 60_000), rawTurns: 50)
        var config = AgentConfig()
        config.kernelConsolidateEvery = 24
        config.kernelRawTurns = 50  // 大窗口：本测试验证 RAG 上限，期间不应触发 CONSOLIDATE 折掉历史

        let runtime = KernelRuntime(
            stream: { _, _, _ in
                callCounter[0] += 1
                let n = callCounter[0]
                return AsyncThrowingStream { c in
                    if n == 1 {
                        if firstTool.name == "rag_tools",
                           firstTool.arguments["op"]?.stringValue == "search" {
                            emitted += 1
                        }
                        c.yield(.init(kind: .done, toolCalls: [firstTool]))
                    } else {
                        emitted += 1
                        c.yield(.init(kind: .done, toolCalls: [
                            ToolCall(id: "s\(n)", name: "rag_tools",
                                     arguments: ["op": .string("search"),
                                                 "query": .string("问题\(n)"),
                                                 "source": .string("测试书")])
                        ]))
                    }
                    c.finish()
                }
            },
            execute: { (calls: [ToolCall]) -> [ToolResult] in
                executed.append(contentsOf: calls)
                return calls.map { _ in .ok("命中内容") }
            },
            schemas: { [] },
            log: { _ in }
        )
        let loop = KernelLoop(runtime: runtime, config: config,
                              systemPrompt: { "SYS" }, session: session,
                              maxTurns: maxTurns)
        _ = await loop.run("测试书讲了什么")  // 短输入，绕过规划前置门
        return (executed, emitted, session)
    }

    /// 核心断言：模型发了 >5 次检索，但只有 5 次真正执行，其余被预置 cap 失败。
    private func assertCap(enforcedIn executed: [ToolCall], emitted: Int,
                           session: Session, outlineCount: Int,
                           file: StaticString = #file, line: UInt = #line) {
        let executedSearches = executed.filter {
            $0.name == "rag_tools" && $0.arguments["op"]?.stringValue == "search"
        }
        XCTAssertEqual(executedSearches.count, 5,
                       "硬上限应只允许 5 次 rag_tools(op=\"search\") 真正执行",
                       file: file, line: line)
        XCTAssertEqual(executed.filter {
            $0.name == "rag_tools" && $0.arguments["op"]?.stringValue == "outline"
        }.count, outlineCount,
                       "rag_tools(op=\"outline\") 执行次数不符", file: file, line: line)
        XCTAssertGreaterThan(emitted, 5,
                             "模型应发出了超过 5 次检索", file: file, line: line)

        let allSearchResults = session.turns
            .flatMap { $0.toolResults }
            .filter { $0.name == "rag_tools" }
        let capped = allSearchResults.filter {
            $0.content.contains("本次请求的检索上限")
        }
        // 发出的检索 = 执行的 5 次 + 被拦截的若干次；被拦截的都应带 cap 提示。
        XCTAssertEqual(capped.count, emitted - 5,
                       "超出部分应全部被预置为 cap 失败", file: file, line: line)
        XCTAssertGreaterThanOrEqual(capped.count, 1,
                                    "至少应有一次被拦截", file: file, line: line)
    }

    /// 场景一：先 rag_tools(op="outline")，再反复 rag_tools(op="search") → 只执行 5 次。
    func testRagSearchCappedAfterOutline() async {
        let first = ToolCall(id: "o1", name: "rag_tools",
                             arguments: ["op": .string("outline"),
                                        "book_name": .string("测试书")])
        let (executed, emitted, session) = await runWith(firstTool: first)
        assertCap(enforcedIn: executed, emitted: emitted, session: session,
                  outlineCount: 1)
    }

    /// 场景二：没有 outline，直接反复 rag_tools(op="search") → 仍按每次 run() 的预算限 5 次。
    func testRagSearchCappedWithoutOutline() async {
        let first = ToolCall(id: "s1", name: "rag_tools",
                             arguments: ["op": .string("search"),
                                         "query": .string("问题1"),
                                         "source": .string("测试书")])
        let (executed, emitted, session) = await runWith(firstTool: first)
        assertCap(enforcedIn: executed, emitted: emitted, session: session,
                  outlineCount: 0)
    }

    /// 场景三：两轮 rag_tools(op="outline") 之间各发多轮检索，验证 outline 不再重置预算，
    /// 整个请求仍只执行 5 次 search（而不是每段各 5 次）。
    func testOutlineResetsBudget() async {
        var executed = [ToolCall]()
        var emitted = 0
        var callCounter = [0]

        let session = Session(budget: ContextBudget(total: 60_000), rawTurns: 50)
        var config = AgentConfig()
        config.kernelConsolidateEvery = 24
        config.kernelRawTurns = 50  // 大窗口：本测试验证 RAG 上限，期间不应触发 CONSOLIDATE 折掉历史
        let runtime = KernelRuntime(
            stream: { _, _, _ in
                callCounter[0] += 1
                let n = callCounter[0]
                // 第 1、5 轮发 outline；其余发 search（query 含轮次避开重复检测）。
                if n == 1 || n == 5 {
                    return AsyncThrowingStream { c in
                        c.yield(.init(kind: .done, toolCalls: [
                            ToolCall(id: "o\(n)", name: "rag_tools",
                                     arguments: ["op": .string("outline"),
                                                 "book_name": .string("测试书")])
                        ]))
                        c.finish()
                    }
                }
                emitted += 1
                return AsyncThrowingStream { c in
                    c.yield(.init(kind: .done, toolCalls: [
                        ToolCall(id: "s\(n)", name: "rag_tools",
                                 arguments: ["op": .string("search"),
                                             "query": .string("问题\(n)"),
                                             "source": .string("测试书")])
                    ]))
                    c.finish()
                }
            },
            execute: { (calls: [ToolCall]) -> [ToolResult] in
                executed.append(contentsOf: calls)
                return calls.map { _ in .ok("命中内容") }
            },
            schemas: { [] },
            log: { _ in }
        )
        let loop = KernelLoop(runtime: runtime, config: config,
                              systemPrompt: { "SYS" }, session: session, maxTurns: 15)
        _ = await loop.run("测试书讲了什么")

        let searchExecuted = executed.filter {
            $0.name == "rag_tools" && $0.arguments["op"]?.stringValue == "search"
        }
        // outline 不再重置预算：整个请求最多 5 次 search，避免同一任务反复检索。
        XCTAssertEqual(searchExecuted.count, 5,
                       "outline 不得重置检索预算，全程仍受 5 次上限约束")
        XCTAssertEqual(executed.filter {
            $0.name == "rag_tools" && $0.arguments["op"]?.stringValue == "outline"
        }.count, 2)
        XCTAssertGreaterThan(emitted, 10,
                             "模型应发出了超过 10 次检索")
    }
}
