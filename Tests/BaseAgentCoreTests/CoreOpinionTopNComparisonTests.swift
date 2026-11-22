import XCTest
@testable import BaseAgentCore

/// 同进程对照：核心句 top20（共 40 句）vs top10（共 20 句）的
/// 压缩率、判定得分与耗时差异（同一模型、同一文本，公平对比）。
final class CoreOpinionTopNComparisonTests: XCTestCase {
    private let modelDir = "/Users/arlenecc/codebase/agent-app/models/PKU-KCL-Meteor-0.8B-mlx-int4"

    private func requireModel() throws -> MeteorDecisionScorer {
        guard FileManager.default.fileExists(atPath: modelDir + "/readout.safetensors") else {
            throw XCTSkip("本机无决策模型，跳过真机对照")
        }
        return MeteorDecisionScorer(modelDir: modelDir)
    }

    private func evidence(sufficient: Bool, count: Int = 120) -> String {
        var parts: [String] = []
        for i in 0..<count {
            parts.append("切片\(i)：记录相关背景细节、作者举例与说明，供参考。")
        }
        parts.append(sufficient
            ? "综上所述，现有资料已完整覆盖核心问题，结论明确，信息充分可以作答。"
            : "不过关键结论仍缺少直接证据，需要进一步确认。")
        return parts.joined(separator: "\n")
    }

    func testTop20VsTop10ScoresAndTiming() async throws {
        let scorer = try requireModel()
        let question = "Is the collected evidence sufficient? (A: insufficient, B: sufficient)"

        for sufficient in [true, false] {
            let state = evidence(sufficient: sufficient)
            let c20 = CoreOpinionExtractor.compressed(state, topN: 20)
            let c10 = CoreOpinionExtractor.compressed(state, topN: 10)

            func judge(_ text: String) async throws -> (Double, Double) {
                let t = CFAbsoluteTimeGetCurrent()
                // text 已压缩且 < 原文长度，传原文长度让 judgeBinary 直通不二次压缩
                let p = try await scorer.judgeBinary(state: text, question: question,
                                                     maxStateChars: state.count)
                return (p, CFAbsoluteTimeGetCurrent() - t)
            }
            let (p20, t20) = try await judge(c20)
            let (p10, t10) = try await judge(c10)
            let line = "[topn-compare] sufficient=\(sufficient) "
                + "原文=\(state.count)字 "
                + "top20=\(c20.count)字 prob=\(String(format: "%.3f", p20)) t=\(String(format: "%.2f", t20))s "
                + "top10=\(c10.count)字 prob=\(String(format: "%.3f", p10)) t=\(String(format: "%.2f", t10))s "
                + "prob差=\(String(format: "%+.3f", p10 - p20))\n"
            FileHandle.standardError.write(Data(line.utf8))
            XCTAssertTrue(p10 >= 0 && p10 <= 1)
        }
    }

    /// top10 是否仍能区分充分 vs 不足。
    func testTop10KeepsSufficientOrdering() async throws {
        let scorer = try requireModel()
        let question = "Is the collected evidence sufficient? (A: insufficient, B: sufficient)"
        let suff = CoreOpinionExtractor.compressed(evidence(sufficient: true), topN: 10)
        let insuff = CoreOpinionExtractor.compressed(evidence(sufficient: false), topN: 10)
        let pSuff = try await scorer.judgeBinary(state: suff, question: question,
                                                 maxStateChars: 100_000)
        let pInsuff = try await scorer.judgeBinary(state: insuff, question: question,
                                                   maxStateChars: 100_000)
        let line = "[topn-compare] top10区分度 充分=\(String(format: "%.3f", pSuff)) "
            + "不足=\(String(format: "%.3f", pInsuff)) 差=\(String(format: "%.3f", pSuff - pInsuff))\n"
        FileHandle.standardError.write(Data(line.utf8))
        XCTAssertGreaterThan(pSuff, pInsuff)
    }
}
