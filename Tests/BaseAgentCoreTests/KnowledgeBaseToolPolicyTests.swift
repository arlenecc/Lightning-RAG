import XCTest
@testable import BaseAgentCore

/// 知识库问答的工具选择回归。
///
/// 真实事故：用户问「猫图鉴讲什么」，`rag_search` 已返回两个章节片段，模型却把
/// `来源: 猫图鉴.pdf` + `章节: 6 猫图鉴` 当成磁盘上的章节文件，臆造
/// `memory/猫图鉴.pdf/6 猫图鉴.md` 用 file_read / shell_run cat 连读三次全失败，
/// 最后还把「读取文件」标成 done、反过来问用户能不能改用 rag_search。
/// 修复三层：RAG 结果贴使用说明 / 系统提示与动态上下文写明硬规则 /
/// 内核跳过已失败过的相同调用。
final class KnowledgeBaseToolPolicyTests: XCTestCase {

    // MARK: - RAG 结果文本契约

    func testRagUsageNoteDeclaresPassagesAreTheContent() {
        XCTAssertTrue(ragUsageNote.contains("以上即知识库返回的全部可用内容"))
        XCTAssertTrue(ragUsageNote.contains("不是磁盘文件路径"))
        XCTAssertTrue(ragUsageNote.contains("禁止用 `file_read`"))
        XCTAssertTrue(ragUsageNote.contains("`shell_run`"))
        XCTAssertTrue(ragUsageNote.contains("`cat`"))
    }

    func testRagToolOKAppendsUsageNoteToEveryResult() {
        let result = ragToolOK("### [1] 来源: 猫图鉴.pdf\n章节: 6 猫图鉴\n# 6 猫图鉴")
        XCTAssertTrue(result.success)
        XCTAssertTrue(result.output.hasPrefix("### [1] 来源: 猫图鉴.pdf"))
        XCTAssertTrue(result.output.contains("不是磁盘文件路径"))
    }

    func testSystemPromptForbidsFileToolsForKnowledgeBaseQueries() {
        XCTAssertTrue(BASE_SYSTEM_PROMPT.contains("知识库问答只用 rag_ 系列工具"))
        XCTAssertTrue(BASE_SYSTEM_PROMPT.contains("不是磁盘文件路径"))
        XCTAssertTrue(BASE_SYSTEM_PROMPT.contains("禁止用 `file_read`/`shell_run`/`code_run`/`cat`"))
    }

    // MARK: - 内核：已失败的相同调用不得重发执行

    func testRepeatedFailedFileReadIsSkippedInsteadOfReexecuted() async {
        var executed: [String] = []
        var streamCount = [0]

        let runtime = KernelRuntime(
            stream: { _, _, _ in
                streamCount[0] += 1
                let n = streamCount[0]
                return AsyncThrowingStream { c in
                    // id 每轮不同，但 name + arguments 完全相同 —— callSig 只看后者，
                    // 正是事故中「原样重试」的形态。
                    c.yield(.init(kind: .done, toolCalls: [
                        ToolCall(id: "f\(n)", name: "file_read",
                                 arguments: ["path": .string("memory/猫图鉴.pdf/6 猫图鉴.md")])
                    ]))
                    c.finish()
                }
            },
            execute: { (calls: [ToolCall]) -> [ToolResult] in
                executed.append(contentsOf: calls.map(\.name))
                return calls.map { _ in .fail("File does not exist: memory/猫图鉴.pdf/6 猫图鉴.md") }
            },
            schemas: { [] },
            log: { _ in }
        )
        let loop = KernelLoop(runtime: runtime, config: AgentConfig(),
                              systemPrompt: { "SYS" }, maxTurns: 6)
        _ = await loop.run("猫图鉴讲什么")

        XCTAssertGreaterThan(streamCount[0], 1, "假模型应重复发起了同一调用")
        XCTAssertEqual(executed.count, 1,
                       "同一个失败调用只能真正执行一次，后续由内核预置失败，不得再耗一轮")
    }
}
