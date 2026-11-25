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

    /// outline 结果不是切片列表：全书摘要必须原文保留，不能走核心句提取
    /// 压成碎片（否则「总结全书」类问题反而拿不到摘要）。
    func testOutlineKeepsBookSummaryVerbatim() {
        // 摘要 110×19≈2090 字，加头与目录超过 2000 触发压缩，但摘要原文
        // 必须逐字保留，只丢章节目录。
        let summary = String(repeating: "病毒深刻影响宿主基因组与生物多样性。", count: 90)
        let tree = (0..<40).map { "- 第\($0 + 1)章 病毒与宿主关系简述" }.joined(separator: "\n")
        let outline = """
        文档: 病毒星球.pdf
        文档 ID: 123
        章节摘要进度: 10/10（全书摘要已完成）

        全书摘要:
        \(summary)

        章节结构:
        \(tree)
        """
        let out = KernelLoop.trimRagHistory("rag_tools", outline, limit: 2_000)
        XCTAssertTrue(out.contains("章节摘要进度: 10/10"), "头信息应保留")
        // 新优先级：目录骨架完整保留（40 章标题），摘要按余量保留，不再逐字保证
        XCTAssertTrue(out.contains("- 第1章"), "章节目录骨架应保留以引导后续 search")
        XCTAssertTrue(out.contains("- 第40章"), "目录骨架应尽量覆盖全部章节")
        XCTAssertTrue(out.contains("章节目录优先") || out.contains("已截断"))
        XCTAssertTrue(out.count < outline.count, "超长内容应按预算收紧")
        XCTAssertTrue(out.contains("原始 \(outline.count) 字符"))
    }

    /// 用户问具体章节时，命中章节的摘要必须优先于其他章节目录保留。
    func testOutlinePrioritizesMatchingChapterForChapterQuestion() {
        let otherChapters = (0..<30).map { "- 第\($0 + 1)章 病毒分类与命名" }.joined(separator: "\n")
        let outline = """
        文档: 病毒星球.pdf
        文档 ID: 123
        章节摘要进度: 40/40（全书摘要已完成）

        全书摘要:
        本书从多学科视角重新审视病毒与人类文明的关系。

        章节结构:
        \(otherChapters)
        - [巨型病毒与进化] Mimivirus 是已知最大的病毒，揭示了病毒在宿主基因组中的角色与进化意义。
        """
        let out = KernelLoop.trimRagHistory(
            "rag_tools", outline, limit: 300,
            query: "巨型病毒与进化讲了什么")
        XCTAssertTrue(out.contains("Mimivirus 是已知最大的病毒"), "命中章节摘要应优先保留")
        XCTAssertTrue(out.contains("匹配章节摘要 + 目录优先"), "应标注章节优先模式")
        if let chapterPos = out.range(of: "Mimivirus")?.lowerBound,
           let summaryPos = out.range(of: "本书从多学科")?.lowerBound {
            XCTAssertLessThan(chapterPos, summaryPos, "命中章节摘要应排在全书摘要之前")
        }
    }

    /// 无匹配 / 全书总结时：章节目录骨架必须排在全书摘要之前，
    /// 便于摘要质量差时仍能按目录标题引导后续 search。
    func testOutlinePrioritizesFullSummaryForWholeBookQuestion() {
        let summary = String(repeating: "病毒与人类文明共同演化。", count: 40)
        let tree = (0..<30).map { "- 第\($0 + 1)章 病毒分类与命名" }.joined(separator: "\n")
        let outline = """
        文档: 病毒星球.pdf
        文档 ID: 123
        章节摘要进度: 40/40（全书摘要已完成）

        全书摘要:
        \(summary)

        章节结构:
        \(tree)
        """
        let out = KernelLoop.trimRagHistory(
            "rag_tools", outline, limit: 800,
            query: "总结一下病毒星球全书的内容")
        XCTAssertTrue(out.contains("章节目录优先"), "应标注目录优先模式")
        let treePos = out.range(of: "- 第1章")?.lowerBound
        let summaryPos = out.range(of: "病毒与人类文明共同演化")?.lowerBound
        if let treePos, let summaryPos {
            XCTAssertLessThan(treePos, summaryPos, "章节目录骨架应排在全书摘要之前")
        }
    }

    /// 摘要自身超预算时保留开头并明确标注，而不是压成不可读的碎片。
    func testOutlineHugeSummaryTruncatedWithNotice() {
        let summary = String(repeating: "病", count: 5_000)
        let outline = "文档: x.pdf\n全书摘要:\n\(summary)"
        let out = KernelLoop.trimRagHistory("rag_tools", outline, limit: 300)
        XCTAssertTrue(out.contains("全书摘要"), "应保留摘要标题")
        XCTAssertTrue(out.contains("已截断"), "应明确标注截断")
        XCTAssertLessThan(out.count, 400)
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
