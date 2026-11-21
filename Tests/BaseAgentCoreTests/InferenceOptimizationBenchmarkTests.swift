import XCTest
import Foundation
@testable import BaseAgentCore

/// 推理路径 CPU 侧优化的对比基准：证明「预编译正则」「短路扫描」有正向收益。
/// 用固定输入在同一进程内对比旧实现（每次重建 NSRegularExpression）与新实现，
/// 断言新实现更快，并校验两者输出完全一致（行为不回归）。
final class InferenceOptimizationBenchmarkTests: XCTestCase {

    private let sample = """
    这是一段普通回答文本，包含 < 数学符号 > 与代码 if x < 10 { ... }。
    知识库检索结果显示相关切片 3 条。用户问题涉及多个分区。
    """

    private func oldStrip(_ text: String) -> String {
        var result = text
        for pattern in [
            #"<function(?:\s*name\s*=|=)[^>]*>[\s\S]*?</function>"#,
            #"<tool_call>[\s\S]*?</tool_call>"#,
        ] {
            guard let regex = try? NSRegularExpression(
                pattern: pattern, options: [.dotMatchesLineSeparators]) else { continue }
            let range = NSRange(result.startIndex..<result.endIndex, in: result)
            result = regex.stringByReplacingMatches(in: result, options: [], range: range,
                                                   withTemplate: "")
        }
        return result
    }

    /// 对比：流式净化器每 token 调一次，模拟 2000 个 chunk。
    func testPrecompiledRegexIsFasterAndIdentical() {
        let iterations = 2_000
        // 预热，排除首次编译/lazy 初始化噪声
        _ = oldStrip(sample)
        _ = ToolTagSplitter.stripToolBlocks(sample)

        let oldStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations { _ = oldStrip(sample) }
        let oldSeconds = CFAbsoluteTimeGetCurrent() - oldStart

        let newStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations { _ = ToolTagSplitter.stripToolBlocks(sample) }
        let newSeconds = CFAbsoluteTimeGetCurrent() - newStart

        let oldMs = String(format: "%.1f", oldSeconds * 1000)
        let newMs = String(format: "%.1f", newSeconds * 1000)
        let speedup = String(format: "%.2f", oldSeconds / max(newSeconds, 1e-9))
        let line = "[opt-bench] stripToolBlocks old=" + oldMs + "ms new=" + newMs
            + "ms speedup=" + speedup + "x\n"
        FileHandle.standardError.write(Data(line.utf8))
        // 行为一致
        XCTAssertEqual(oldStrip(sample), ToolTagSplitter.stripToolBlocks(sample))
        XCTAssertEqual(oldStrip("<function name=\"x\"><param name=\"k\">v</param></function>后"),
                       ToolTagSplitter.stripToolBlocks("<function name=\"x\"><param name=\"k\">v</param></function>后"))
        // 正向收益：预编译至少快 2 倍（保守阈值，避免机器噪声导致假失败）
        XCTAssertLessThan(newSeconds * 2, oldSeconds,
            "预编译正则应显著更快（old=\(oldSeconds)s new=\(newSeconds)s）")
    }

    /// 普通正文短路的收益：模拟 300 个 token chunk 的流式拼接，
    /// 与「每 chunk 都跑正则 + 全 buffer 扫描」的旧实现对比。
    func testPlainTextShortCircuitFasterAndIdentical() {
        let chunks = Array(repeating: "普通正文片段，", count: 300)
        func oldFeed() -> String {
            var buffer = ""
            var out = ""
            for chunk in chunks {
                buffer += chunk
                let cleaned = ToolTagSplitter.stripFunctionBlocks(buffer)
                guard let lastLt = cleaned.lastIndex(of: "<") else {
                    out += cleaned; buffer = ""; continue
                }
                let after = cleaned[cleaned.index(after: lastLt)...]
                if after.contains(">") { out += cleaned; buffer = "" }
                else {
                    out += String(cleaned[..<lastLt])
                    buffer = String(cleaned[lastLt...])
                }
            }
            return out
        }
        func newFeed() -> String {
            var s = ContentSanitizer()
            var out = ""
            for chunk in chunks { out += s.feed(chunk).joined() }
            out += s.flush().joined()
            return out
        }
        _ = oldFeed(); _ = newFeed()
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<40 { _ = oldFeed() }
        let oldSeconds = CFAbsoluteTimeGetCurrent() - t0
        let t1 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<40 { _ = newFeed() }
        let newSeconds = CFAbsoluteTimeGetCurrent() - t1
        let line = "[opt-bench] plainFeed old=" + String(format: "%.1f", oldSeconds * 1000)
            + "ms new=" + String(format: "%.1f", newSeconds * 1000) + "ms\n"
        FileHandle.standardError.write(Data(line.utf8))
        XCTAssertEqual(oldFeed(), newFeed(), "短路不改变输出")
        XCTAssertLessThan(newSeconds * 2, oldSeconds,
            "普通正文短路应显著更快（old=\(oldSeconds)s new=\(newSeconds)s）")
    }

    /// DateFormatter 复用收益：日志每轮 5~20 条，旧实现每条新建 2 个格式化器。
    func testReusedDateFormattersFasterAndIdentical() {
        func oldStamp(_ date: Date) -> (String, String) {
            let a = DateFormatter(); a.dateFormat = "yyyyMMdd"
            let b = DateFormatter(); b.dateFormat = "HH:mm:ss"
            return (a.string(from: date), b.string(from: date))
        }
        let day = DateFormatter(); day.dateFormat = "yyyyMMdd"
        let time = DateFormatter(); time.dateFormat = "HH:mm:ss"
        func newStamp(_ date: Date) -> (String, String) {
            (day.string(from: date), time.string(from: date))
        }
        let date = Date()
        XCTAssertEqual(oldStamp(date).0, newStamp(date).0)
        XCTAssertEqual(oldStamp(date).1, newStamp(date).1)
        let n = 3_000
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<n { _ = oldStamp(date) }
        let oldSeconds = CFAbsoluteTimeGetCurrent() - t0
        let t1 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<n { _ = newStamp(date) }
        let newSeconds = CFAbsoluteTimeGetCurrent() - t1
        let line = "[opt-bench] dateFmt old=" + String(format: "%.1f", oldSeconds * 1000)
            + "ms new=" + String(format: "%.1f", newSeconds * 1000) + "ms\n"
        FileHandle.standardError.write(Data(line.utf8))
        XCTAssertLessThan(newSeconds * 3, oldSeconds,
            "复用格式化器应显著更快（old=\(oldSeconds)s new=\(newSeconds)s）")
    }

    /// F10 短路的行为等价性：模拟 AppModel.reclassifyStrayThinking，
    /// 对多种 chunk 切分（含跨 chunk/跨 flush 的 </think>）比较旧逻辑与新逻辑结果。
    func testStrayThinkShortCircuitNeverMissesTag() {
        func classify(_ chunks: [String], shortCircuit: Bool) -> (bubble: String, moved: Bool) {
            var onScreen = ""
            var buffer = ""
            var sawAngle = false
            var moved = false
            for (i, chunk) in chunks.enumerated() {
                buffer += chunk
                if chunk.contains("<") { sawAngle = true }
                if shortCircuit && !sawAngle { continue }
                let combined = onScreen + buffer
                guard let range = combined.range(of: "</think>") else { continue }
                let prefix = String(combined[..<range.lowerBound])
                guard !prefix.contains("<think>") else { continue }
                let tail = String(combined[range.upperBound...])
                if !moved {
                    moved = true
                    onScreen = tail
                } else {
                    onScreen = prefix + tail
                }
                buffer = ""
                // 模拟确定性 50ms flush（每隔一个 chunk 刷一次）
                if i % 2 == 0, !buffer.isEmpty { onScreen += buffer; buffer = "" }
            }
            return (onScreen + buffer, moved)
        }
        let cases: [[String]] = [
            ["普通正文", "没有标签", "继续回答"],               // 无标签
            ["思考内容", "</thin", "k>", "正式回答"],           // 跨 chunk
            ["甲", "</think>", "乙"],                          // 单 chunk 完整
            ["前文", "<think>完整块", "</think>", "后文"],       // 有开标签（应跳过）
            ["数学 a < b", " 且 c > d", " 结束"],                // 普通尖括号
            ["</", "think", ">"],                              // 三段切分
        ]
        for chunks in cases {
            let old = classify(chunks, shortCircuit: false)
            let new = classify(chunks, shortCircuit: true)
            XCTAssertEqual(old.bubble, new.bubble, "气泡内容必须一致: \(chunks)")
            XCTAssertEqual(old.moved, new.moved, "重分类判定必须一致: \(chunks)")
        }
    }

    /// 未闭合工具标签检测的短路：无 '<' 时不应付全 buffer 扫描成本。
    func testContentSanitizerPlainChunksUnchanged() {
        var sanitizer = ContentSanitizer()
        var out = ""
        for chunk in ["普通", "回答", "文本", "没有", "标签"] {
            out += sanitizer.feed(chunk).joined()
        }
        out += sanitizer.flush().joined()
        XCTAssertEqual(out, "普通回答文本没有标签")
    }
}
