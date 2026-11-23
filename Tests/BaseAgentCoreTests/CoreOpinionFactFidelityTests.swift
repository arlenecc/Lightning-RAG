import XCTest
@testable import BaseAgentCore

/// 数字/实体保真：W_FACT=0.25 加分 + FACT_GUARANTEE_TOP=3 强制席位。
final class CoreOpinionFactFidelityTests: XCTestCase {
    func testFactScoreDetectsAllPatternCategories() {
        let cases: [(String, String)] = [
            ("预算 500万元用于采购。", "数字+单位"),
            ("转化率达到 12.5%。", "百分比"),
            ("审批周期为 3~5 个工作日。", "区间"),
            ("该条款于 2026-11-23 生效，截止日期 12/31。", "日期时间"),
            ("当前版本 v1.4.13 已发布。", "版本号"),
            ("配置 max_tokens=4096 需要调整。", "配置键值"),
            ("第 3 条规定阈值不低于 0.8，必须审批。", "条款/约束"),
        ]
        for (sentence, label) in cases {
            let score = CoreOpinionExtractor.factScore(sentence)
            XCTAssertGreaterThan(score, 0, "应识别\(label): \(sentence)")
        }
        XCTAssertEqual(CoreOpinionExtractor.factScore("今天天气很好适合散步。"), 0)
    }

    /// 关键数字句（低观点/低中心度）必须被保真席位强制纳入。
    func testFactGuaranteeIncludesNumericSentenceOutsideTop10() {
        var parts: [String] = []
        // 大量无数字的观点句，抢占启发式/TextRank 的 top10
        for i in 0..<40 {
            parts.append("核心观点认为第\(i)个背景说明实际上意味着必须继续深入分析这个问题。")
        }
        // 一句含关键数字但观点词少、位于中间，正常两路 top10 不会选它
        let factSentence = "合同违约金按每天 0.05% 计算，上限 30 万元，截止 2026-12-31。"
        parts.insert(factSentence, at: 20)
        let text = parts.joined(separator: "\n")

        let extracted = CoreOpinionExtractor.extract(text, topN: 10)
        XCTAssertTrue(extracted.contains(factSentence),
            "含金额/百分比/日期的关键句必须被 FACT_GUARANTEE_TOP 强制纳入: \(extracted)")
    }

    /// 保真句数量上限 5（与两路 topN 去重后总数可能 >20）。
    func testFactGuaranteeTopIsFive() {
        XCTAssertEqual(CoreOpinionExtractor.factGuaranteeTop, 3)
        let sentences = (0..<20).map { "第 \($0) 条规定费用为 \($0)0 元。" }
        let extracted = CoreOpinionExtractor.extract(sentences.joined(separator: "\n"), topN: 1)
        let numeric = extracted.filter { CoreOpinionExtractor.factScore($0) > 0 }
        XCTAssertLessThanOrEqual(numeric.count, CoreOpinionExtractor.perMethodTopN * 2 + CoreOpinionExtractor.factGuaranteeTop)
    }
}
