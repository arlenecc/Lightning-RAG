import XCTest
@testable import BaseAgentCore

/// 真机验证「混合核心句提取」用于整段文本判定时的效果、得分与耗时。
/// 对照：同一 state 原文直通（maxStateChars=全文长度） vs 核心句压缩（=长度-1）。
final class CoreOpinionJudgmentBenchmarkTests: XCTestCase {
    private let modelDir = "/Users/arlenecc/codebase/agent-app/models/PKU-KCL-Meteor-0.8B-mlx-int4"

    private func requireModel() throws -> MeteorDecisionScorer {
        guard FileManager.default.fileExists(atPath: modelDir + "/readout.safetensors") else {
            throw XCTSkip("本机无决策模型，跳过真机基准")
        }
        return MeteorDecisionScorer(modelDir: modelDir)
    }

    /// 构造长文本：含明确结论（充分）/ 大量细节但无结论（不足）。
    private func longState(sufficient: Bool) -> String {
        var parts: [String] = []
        for i in 0..<40 {
            parts.append("第\(i)条资料记录了相关的背景细节、补充说明与示例，用于支撑后续分析。")
        }
        if sufficient {
            parts.append("综上所述，现有证据已经完整覆盖问题的所有要点，结论明确：信息充分，可以作答。")
            parts.append("因此核心观点认为必须据此给出最终答复，这意味着不需要继续检索。")
        } else {
            parts.append("不过关于关键结论部分，现有资料仍缺少直接证据，需要进一步确认。")
        }
        return parts.joined(separator: "\n")
    }

    func testCompressionStats() {
        for sufficient in [true, false] {
            let state = longState(sufficient: sufficient)
            let core = CoreOpinionExtractor.prepareForScoring(state, maxChars: state.count - 1)
            let line = "[core-bench] sufficient=\(sufficient) "
                + "原文=\(state.count)字 核心句=\(core.count)字 "
                + "压缩率=\(String(format: "%.1f%%", Double(core.count) / Double(state.count) * 100)) "
                + "句数=\(CoreOpinionExtractor.extract(state).count)\n"
            FileHandle.standardError.write(Data(line.utf8))
            XCTAssertLessThan(core.count, state.count)
            XCTAssertGreaterThan(core.count, 0)
        }
    }

    /// 真机：同一 state 原文 vs 核心句，判定概率与耗时。
    func testRealModelScoresOriginalVsCompressed() async throws {
        let scorer = try requireModel()
        let question = "Is the collected evidence sufficient to fully answer the task? "
            + "(A: insufficient, B: sufficient)"
        for sufficient in [true, false] {
            let state = longState(sufficient: sufficient)
            let t0 = CFAbsoluteTimeGetCurrent()
            let original = try await scorer.judgeBinary(
                state: state, question: question,
                maxStateChars: state.count)  // 直通，不压缩
            let t1 = CFAbsoluteTimeGetCurrent()
            let compressed = try await scorer.judgeBinary(
                state: state, question: question,
                maxStateChars: state.count - 1)  // 触发核心句压缩
            let t2 = CFAbsoluteTimeGetCurrent()
            let line = "[core-bench] sufficient=\(sufficient) "
                + "原文prob=\(String(format: "%.3f", original)) 核心句prob=\(String(format: "%.3f", compressed)) "
                + "差=\(String(format: "%+.3f", compressed - original)) "
                + "原文耗时=\(String(format: "%.2f", t1 - t0))s 压缩耗时=\(String(format: "%.2f", t2 - t1))s\n"
            FileHandle.standardError.write(Data(line.utf8))
            // 判定方向一致：充分文本的 prob 应高于不足文本（分别记录，不强制绝对阈值）
            XCTAssertTrue(original >= 0 && original <= 1)
            XCTAssertTrue(compressed >= 0 && compressed <= 1)
        }
    }

    /// 真实规模：多切片 evidence（5000+ 字）的压缩率与耗时。
    func testLargeEvidenceCompressionAndScore() async throws {
        let scorer = try requireModel()
        var parts: [String] = []
        for i in 0..<120 {
            parts.append("切片\(i)：本段记录了书籍内容的相关背景细节、作者观点与示例说明，可供参考。")
        }
        parts.append("综上所述，现有资料已完整覆盖核心问题，结论明确，信息充分可以作答。")
        let state = parts.joined(separator: "\n")
        let question = "Is the collected evidence sufficient? (A: insufficient, B: sufficient)"
        let core = CoreOpinionExtractor.prepareForScoring(state, maxChars: state.count - 1)
        let line0 = "[core-bench-large] 原文=\(state.count)字 核心句=\(core.count)字 "
            + "压缩率=\(String(format: "%.1f%%", Double(core.count) / Double(state.count) * 100)) "
            + "句数=\(CoreOpinionExtractor.extract(state).count)\n"
        FileHandle.standardError.write(Data(line0.utf8))
        let t0 = CFAbsoluteTimeGetCurrent()
        let pRaw = try await scorer.judgeBinary(state: state, question: question,
                                                 maxStateChars: state.count)
        let t1 = CFAbsoluteTimeGetCurrent()
        let pCore = try await scorer.judgeBinary(state: state, question: question,
                                                  maxStateChars: state.count - 1)
        let t2 = CFAbsoluteTimeGetCurrent()
        let line = "[core-bench-large] 原文prob=\(String(format: "%.3f", pRaw)) "
            + "核心句prob=\(String(format: "%.3f", pCore)) "
            + "原文耗时=\(String(format: "%.2f", t1 - t0))s 压缩耗时=\(String(format: "%.2f", t2 - t1))s\n"
        FileHandle.standardError.write(Data(line.utf8))
        XCTAssertLessThan(core.count, state.count)
        XCTAssertGreaterThan(pCore, 0.5, "明确充分的长 evidence 压缩后仍应判充分")
    }

    /// 真机：充分 vs 不足两组文本，核心句压缩后仍能区分（prob 排序正确）。
    func testCompressedKeepsSufficientVsInsufficientOrdering() async throws {
        let scorer = try requireModel()
        let question = "Is the collected evidence sufficient? (A: insufficient, B: sufficient)"
        let sufficient = longState(sufficient: true)
        let insufficient = longState(sufficient: false)
        let pSuff = try await scorer.judgeBinary(
            state: sufficient, question: question, maxStateChars: sufficient.count - 1)
        let pInsuff = try await scorer.judgeBinary(
            state: insufficient, question: question, maxStateChars: insufficient.count - 1)
        let line = "[core-bench] 压缩后 充分=\(String(format: "%.3f", pSuff)) "
            + "不足=\(String(format: "%.3f", pInsuff)) "
            + "可区分=\(pSuff > pInsuff)\n"
        FileHandle.standardError.write(Data(line.utf8))
        XCTAssertGreaterThan(pSuff, pInsuff,
            "核心句压缩应保留「充分 vs 不足」的判定区分度")
    }
}
