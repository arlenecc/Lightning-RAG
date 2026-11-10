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
        XCTAssertTrue(ragUsageNote.contains("以上即本次检索返回的全部可用内容"))
        XCTAssertTrue(ragUsageNote.contains("不是磁盘文件路径"))
        XCTAssertTrue(ragUsageNote.contains("仅适用于读取已索引文档"))
        XCTAssertTrue(ragUsageNote.contains("本规则不限制其它工具"))
    }

    func testRagToolOKAppendsUsageNoteToEveryResult() {
        let result = ragToolOK("### [1] 来源: 猫图鉴.pdf\n章节: 6 猫图鉴\n# 6 猫图鉴")
        XCTAssertTrue(result.success)
        XCTAssertTrue(result.output.hasPrefix("### [1] 来源: 猫图鉴.pdf"))
        XCTAssertTrue(result.output.contains("不是磁盘文件路径"))
    }

    func testSystemPromptKeepsKnowledgeBaseRuleNarrow() {
        // 窄规则现已落在 KNOWLEDGE_BASE_HINT（动态上下文注入）；BASE 保留「不替代联网」声明。
        XCTAssertTrue(KNOWLEDGE_BASE_HINT.contains("不是磁盘"))
        XCTAssertTrue(KNOWLEDGE_BASE_HINT.contains("本说明只针对知识库文档"))
        XCTAssertTrue(KNOWLEDGE_BASE_HINT.contains("给网址或要实时信息请用 `web_scan`"))
        XCTAssertTrue(BASE_SYSTEM_PROMPT.contains("不要用知识库检索代替联网"))
        XCTAssertFalse(BASE_SYSTEM_PROMPT.contains("知识库问答只用 rag_"),
                       "全局禁令会让模型连联网工具都不敢调用")
        XCTAssertFalse(KNOWLEDGE_BASE_HINT.contains("知识库问答只用"),
                       "全局禁令会让模型连联网工具都不敢调用")
    }

    /// 用户实测：问 CNN / 招聘网站，模型回「作为本地 agent 我没有访问外部
    /// 网站的权限」，并把知识库规则当成全局能力边界。所有注入文本都不得
    /// 再出现「只用 rag_* / 禁止用文件工具」这类可被泛化的措辞。
    func testKnowledgeBaseRulesNeverDisableOtherTools() {
        let texts = [
            ("system", BASE_SYSTEM_PROMPT),
            ("dynamic context", KNOWLEDGE_BASE_HINT),
            ("rag result", ragUsageNote),
        ]
        let globalBans = ["知识库问答只用", "只用 rag_", "禁止用文件工具", "不能改用文件工具绕过"]
        for (label, text) in texts {
            for ban in globalBans {
                XCTAssertFalse(text.contains(ban), "\(label) 不得含全局禁令：\(ban)")
            }
            XCTAssertTrue(text.contains("web_scan"), "\(label) 必须正向提到联网工具")
            let affirmsOtherTools = text.contains("不受限制") || text.contains("不限制其它工具")
                || text.contains("均不受限制") || text.contains("照常用")
                || text.contains("不要用知识库检索代替联网")
            XCTAssertTrue(affirmsOtherTools,
                          "\(label) 必须显式声明其它工具不受限制（或「不替代联网」）")
        }
    }

    func testSystemPromptAffirmsNetworkAccess() {
        XCTAssertTrue(BASE_SYSTEM_PROMPT.contains("联网抓取"))
        XCTAssertTrue(BASE_SYSTEM_PROMPT.contains("无法访问网络"))
        XCTAssertTrue(BASE_SYSTEM_PROMPT.contains("不要用知识库检索代替联网"))
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
                        ToolCall(id: "f\(n)", name: "file_tools",
                                 arguments: ["op": .string("read"),
                                             "path": .string("memory/猫图鉴.pdf/6 猫图鉴.md")])
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
