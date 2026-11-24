import XCTest
@testable import BaseAgentCore

/// 融合核心观点句提取器：启发式 + TextRank 各 top10（默认），合并去重。
final class CoreOpinionExtractorTests: XCTestCase {
    private func sampleText() -> String {
        var parts: [String] = []
        for i in 0..<60 {
            if i % 5 == 0 {
                parts.append("第\(i)句：核心观点认为，这个方法必须从根本上改变现状，因此值得推广。")
            } else if i % 5 == 1 {
                parts.append("第\(i)句补充了一些次要的细节与背景描述，用于说明环境。")
            } else if i % 5 == 2 {
                parts.append("实际上，第\(i)句的证据意味着结论可靠，所以我们强调这一点。")
            } else {
                parts.append("第\(i)句只是普通的叙述内容，包含一些常见词语和说明。")
            }
        }
        return parts.joined(separator: "。")
    }

    func testExtractReturnsCoreOpinionSentences() {
        let text = sampleText()
        let result = CoreOpinionExtractor.extract(text)
        XCTAssertFalse(result.isEmpty)
        XCTAssertLessThanOrEqual(result.count, CoreOpinionExtractor.perMethodTopN * 2)
        // 观点句（含「核心观点」「必须」「因此」）应被优先选中。
        let opinionCount = result.filter { $0.contains("核心观点") || $0.contains("实际上") }.count
        XCTAssertGreaterThan(opinionCount, 0, "应选出至少一条观点句")
        // 保持原文顺序（编号递增）。
        let indices = result.compactMap { s -> Int? in
            guard s.hasPrefix("第") else { return nil }
            return Int(s.dropFirst().prefix(while: { $0.isNumber }))
        }
        XCTAssertEqual(indices, indices.sorted(), "结果应按原文顺序返回")
    }

    func testDeduplicatesOverlappingTop20() {
        let text = sampleText()
        let result = CoreOpinionExtractor.extract(text)
        XCTAssertEqual(Set(result).count, result.count, "合并后不得有重复句")
    }

    func testShortTextAndEmptyInput() {
        XCTAssertTrue(CoreOpinionExtractor.extract("").isEmpty)
        XCTAssertEqual(CoreOpinionExtractor.extract("只有一句话。"), ["只有一句话。"])
        let two = CoreOpinionExtractor.extract("第一句很短。第二句是核心观点必须执行。")
        XCTAssertEqual(Set(two).count, 2)
    }

    /// 中心度信号：query 提供时应更聚焦相关句。
    func testQueryFocusesExtraction() {
        let text = """
        python 环境配置需要安装依赖。今天天气很好适合散步。\
        核心观点认为，python 的虚拟环境是必须的。晚饭吃了米饭和菜。\
        因此 python 项目应始终使用隔离环境。猫在沙发上睡觉。
        """
        let focused = CoreOpinionExtractor.extract(text, query: "python 环境配置")
        XCTAssertTrue(focused.contains { $0.contains("python") })
        XCTAssertLessThan(focused.count, 20)
    }

    func testCompressedIsShorterThanOriginal() {
        let text = sampleText()
        let compressed = CoreOpinionExtractor.compressed(text)
        XCTAssertLessThan(compressed.count, text.count)
        XCTAssertFalse(compressed.isEmpty)
    }

    /// 超长文本必须全文扫描：位于尾部（窗口之外）的核心句也要参与抽取，
    /// 不能只截取头部 3000 字。
    func testLongTextScanCoversTail() {
        let filler = (0..<200).map {
            "第\($0)句是普通的背景描述，介绍环境与用途，语气平缓没有观点。"
        }.joined()
        let tail = "综上所述，尾部的核心结论是必须启用双写校验，因此数据完整性才有保障。"
        let text = filler + tail

        let result = CoreOpinionExtractor.extractScanningAll(text, topN: 5)
        XCTAssertFalse(result.isEmpty)
        XCTAssertTrue(result.contains { $0.contains("双写校验") },
                      "尾部核心句应参与抽取，不能因只看头部而遗漏")

        // 不截断：结果必须来自原文中的真实句子。
        XCTAssertTrue(result.allSatisfy { text.contains($0) })
    }

    /// 短文本（单窗口以内）直接复用原提取逻辑，行为不变。
    func testScanningAllMatchesExtractForShortText() {
        let text = (0..<30).map {
            "第\($0)句是普通叙述内容，用于说明背景与环境。"
        }.joined()
        XCTAssertEqual(CoreOpinionExtractor.splitSentences(text).count, 30)
        XCTAssertEqual(CoreOpinionExtractor.extractScanningAll(text, topN: 5),
                       CoreOpinionExtractor.extract(text, topN: 5))
    }
}
