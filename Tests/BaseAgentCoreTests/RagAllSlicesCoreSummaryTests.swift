import XCTest
@testable import BaseAgentCore

/// RAG 结果回灌：所有切片都提取核心句后交给模型，长度恒定、覆盖不丢失。
final class RagAllSlicesCoreSummaryTests: XCTestCase {
    private func ragOutput(sliceCount: Int, fillerPerSlice: Int = 120) -> String {
        var lines = ["从知识库检索到 \(sliceCount) 条相关内容：", ""]
        for i in 1...sliceCount {
            lines.append("### [\(i)] 来源: book\(i).pdf (相关度: 0.9\(i % 10))")
            lines.append("核心观点认为第 \(i) 个切片说明了一个关键事实：编号 \(i) 的结论在 2026-11-\(String(format: "%02d", i)) 生效，阈值 \(i)0%，上限 \(i)00 元。")
            lines.append(String(repeating: "背景细节补充，不影响结论。", count: fillerPerSlice / 12))
            lines.append("")
        }
        return lines.joined(separator: "\n")
    }

    func testAllSlicesCoveredUnderBudget() {
        let raw = ragOutput(sliceCount: 10)
        let trimmed = KernelLoop.trimRagHistory("rag_tools", raw, limit: 1_500)
        // 10 个切片的标题全部保留 → 覆盖所有切片，不是只取 top1
        for i in 1...10 {
            XCTAssertTrue(trimmed.contains("### [\(i)] 来源"), "切片 \(i) 丢失")
        }
        XCTAssertLessThan(trimmed.count, raw.count, "核心句应压缩原文")
    }

    func testLengthBoundedRegardlessOfInputLength() {
        // 同样 5 个切片， filler 从 40 增到 400 字符，核心句输出应基本恒定
        let short = ragOutput(sliceCount: 5, fillerPerSlice: 40)
        let long = ragOutput(sliceCount: 5, fillerPerSlice: 400)
        let cShort = KernelLoop.trimRagHistory("rag_tools", short, limit: 500)
        let cLong = KernelLoop.trimRagHistory("rag_tools", long, limit: 500)
        // 的核心句集合几乎相同（关键事实句一样，filler 都被提掉），
        // 不再随原始切片文本长度线性增长。
        let diff = abs(cLong.count - cShort.count)
        XCTAssertLessThan(diff, 100,
            "输入 filler 从 \(short.count) 增到 \(long.count) 字，核心句长度差应很小，实际 \(diff)")
    }

    func testFactSlicesPreserved() {
        let raw = ragOutput(sliceCount: 6, fillerPerSlice: 200)
        let trimmed = KernelLoop.trimRagHistory("rag_tools", raw, limit: 2_500)
        // 每个切片的关键数字/日期应保留（核心句保真席位）
        for i in 1...6 {
            XCTAssertTrue(trimmed.contains("2026-11-\(String(format: "%02d", i))"), "切片 \(i) 日期丢失")
        }
    }

    func testTinyBudgetStillCoversAllSlices() {
        let raw = ragOutput(sliceCount: 8, fillerPerSlice: 120)
        let trimmed = KernelLoop.trimRagHistory("rag_tools", raw, limit: 600)
        for i in 1...8 {
            XCTAssertTrue(trimmed.contains("### [\(i)] 来源"), "极小预算下切片 \(i) 仍应保留标题")
        }
    }

    /// 预算很小时自动降级到每路 top5/top3，仍覆盖所有切片且不超预算。
    func testBudgetExceededDegradesToTop5() {
        let raw = ragOutput(sliceCount: 12, fillerPerSlice: 120)
        let trimmed = KernelLoop.trimRagHistory("rag_tools", raw, limit: 1_000)
        for i in 1...12 {
            XCTAssertTrue(trimmed.contains("### [\(i)] 来源"), "切片 \(i) 丢失")
        }
        XCTAssertTrue(trimmed.contains("每路 top5") || trimmed.contains("每路 top3"),
                      "应自适应降级到 top5/3: \(trimmed.suffix(120))")
        XCTAssertLessThanOrEqual(trimmed.count, 1_000 + 100)
    }

    func testNonRagUnchanged() {
        let text = String(repeating: "普通工具结果内容。", count: 1000)
        XCTAssertEqual(KernelLoop.trimRagHistory("shell_run", text, limit: 100), text)
    }
}
