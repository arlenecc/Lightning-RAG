import XCTest
@testable import BaseAgentCore

/// 验证「大纲（rag_outline）之后 rag_search 硬上限 = 3」：
/// 模型反复请求检索时，只有前 3 次真正到达工具执行层，超出的被内核预置为失败，
/// 逼模型基于已有结果 + 大纲摘要综合作答，杜绝无效重复检索。
final class KernelLoopRagSearchCapTests: XCTestCase {

    /// 跑一轮 loop，假模型在第一轮发 `firstCall`，之后每轮发带序号的 rag_search。
    /// 返回：`executed`（真正到达 execute 的工具名序列，证明硬上限）、
    /// `emitted`（模型实际发出的 rag_search 次数）、`session`（用于查历史）。
    /// `rawTurns` 设大以避免滑动窗口把早期轮折出 `turns`，便于断言历史。
    private func runWith(firstCall: String, maxTurns: Int = 15) async
        -> (executed: [String], emitted: Int, session: Session) {
        var executed = [String]()
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
                        if firstCall == "rag_search" { emitted += 1 }
                        c.yield(.init(kind: .done, toolCalls: [
                            ToolCall(id: "o1", name: firstCall,
                                     arguments: ["book_name": .string("测试书")])
                        ]))
                    } else {
                        emitted += 1
                        c.yield(.init(kind: .done, toolCalls: [
                            ToolCall(id: "s\(n)", name: "rag_search",
                                     arguments: ["query": .string("问题\(n)"),
                                                 "source": .string("测试书")])
                        ]))
                    }
                    c.finish()
                }
            },
            execute: { (calls: [ToolCall]) -> [ToolResult] in
                for c in calls { executed.append(c.name) }
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

    /// 核心断言：模型发了 >3 次检索，但只有 3 次真正执行，其余被预置 cap 失败。
    private func assertCap(enforcedIn executed: [String], emitted: Int,
                           session: Session, outlineCount: Int,
                           file: StaticString = #file, line: UInt = #line) {
        let executedSearches = executed.filter { $0 == "rag_search" }
        XCTAssertEqual(executedSearches.count, 3,
                       "硬上限应只允许 3 次 rag_search 真正执行", file: file, line: line)
        XCTAssertEqual(executed.filter { $0 == "rag_outline" }.count, outlineCount,
                       "rag_outline 执行次数不符", file: file, line: line)
        XCTAssertGreaterThan(emitted, 3,
                             "模型应发出了超过 3 次检索", file: file, line: line)

        let allSearchResults = session.turns
            .flatMap { $0.toolResults }
            .filter { $0.name == "rag_search" }
        let capped = allSearchResults.filter {
            $0.content.contains("单次大纲后的检索上限")
        }
        // 发出的检索 = 执行的 3 次 + 被拦截的若干次；被拦截的都应带 cap 提示。
        XCTAssertEqual(capped.count, emitted - 3,
                       "超出部分应全部被预置为 cap 失败", file: file, line: line)
        XCTAssertGreaterThanOrEqual(capped.count, 1,
                                    "至少应有一次被拦截", file: file, line: line)
    }

    /// 场景一：先 rag_outline 开预算，再反复 rag_search → 只执行 3 次。
    func testRagSearchCappedAfterOutline() async {
        let (executed, emitted, session) = await runWith(firstCall: "rag_outline")
        assertCap(enforcedIn: executed, emitted: emitted, session: session,
                  outlineCount: 1)
    }

    /// 场景二：没有 rag_outline，直接反复 rag_search → 仍按每次 run() 的预算限 3 次。
    func testRagSearchCappedWithoutOutline() async {
        let (executed, emitted, session) = await runWith(firstCall: "rag_search")
        assertCap(enforcedIn: executed, emitted: emitted, session: session,
                  outlineCount: 0)
    }

    /// 场景三：两轮 rag_outline 之间各发 3+ 次检索，第二次 outline 应重置预算，
    /// 使得第二段又能再执行 3 次（验证「每次 outline 重置」而不仅是「整个 run 限 3」）。
    func testOutlineResetsBudget() async {
        var executed = [String]()
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
                // 第 1、5 轮发 rag_outline；其余发 rag_search（query 含轮次避开重复检测）。
                if n == 1 || n == 5 {
                    return AsyncThrowingStream { c in
                        c.yield(.init(kind: .done, toolCalls: [
                            ToolCall(id: "o\(n)", name: "rag_outline",
                                     arguments: ["book_name": .string("测试书")])
                        ]))
                        c.finish()
                    }
                }
                emitted += 1
                return AsyncThrowingStream { c in
                    c.yield(.init(kind: .done, toolCalls: [
                        ToolCall(id: "s\(n)", name: "rag_search",
                                 arguments: ["query": .string("问题\(n)"),
                                             "source": .string("测试书")])
                    ]))
                    c.finish()
                }
            },
            execute: { (calls: [ToolCall]) -> [ToolResult] in
                for c in calls { executed.append(c.name) }
                return calls.map { _ in .ok("命中内容") }
            },
            schemas: { [] },
            log: { _ in }
        )
        let loop = KernelLoop(runtime: runtime, config: config,
                              systemPrompt: { "SYS" }, session: session, maxTurns: 15)
        _ = await loop.run("测试书讲了什么")

        let searchExecuted = executed.filter { $0 == "rag_search" }
        // 两轮 outline 各重置一次预算，每段最多 3 次 → 共应执行 6 次。
        XCTAssertEqual(searchExecuted.count, 6,
                       "两次 outline 应各允许 3 次检索（预算被重置）")
        XCTAssertEqual(executed.filter { $0 == "rag_outline" }.count, 2)
        XCTAssertGreaterThan(emitted, 6,
                             "模型应发出了超过 6 次检索")
    }
}
