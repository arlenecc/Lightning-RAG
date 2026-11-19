import XCTest
@testable import BaseAgentCore

final class CharacterNgramBM25Tests: XCTestCase {
    func testSentenceSplitting() {
        let text = "第一句。第二句！第三句？\n第四句;"
        XCTAssertEqual(CharacterNgramBM25.sentences(text).count, 4)
    }

    func testScoresRankRelevantSentenceFirst() {
        let sents = [
            "今天的天气很好，适合外出散步。",
            "合同违约责任需要依据民法典相关条款认定。",
            "这是一段无关的技术文档介绍。",
        ]
        let scores = CharacterNgramBM25.scores(query: "合同违约责任如何认定", sentences: sents)
        XCTAssertEqual(scores.count, 3)
        XCTAssertGreaterThan(scores[1], scores[0], "含查询关键词的句子应得分更高")
        XCTAssertGreaterThan(scores[1], scores[2])
    }

    func testCoreSentencesReturnInOriginalOrder() {
        let text = """
        合同当事人应当遵循诚实信用原则。
        今天的天气很好适合外出散步。
        当事人一方不履行合同义务应当承担违约责任。
        这是一段完全无关的介绍文字。
        """
        let core = CharacterNgramBM25.coreSentences(query: "合同违约责任", text: text, count: 3)
        XCTAssertEqual(core.count, 3)
        XCTAssertTrue(core.contains { $0.contains("违约责任") })
        // 按原顺序：选中的句子索引必须递增（这里验证第一句与违约句的相对顺序）
        if let first = core.first, let second = core.dropFirst().first {
            let i1 = (text as NSString).range(of: first).location
            let i2 = (text as NSString).range(of: second).location
            XCTAssertLessThan(i1, i2)
        }
    }

    func testEmptyInputs() {
        XCTAssertTrue(CharacterNgramBM25.sentences("").isEmpty)
        XCTAssertTrue(CharacterNgramBM25.coreSentences(query: "x", text: "", count: 3).isEmpty)
        XCTAssertTrue(CharacterNgramBM25.scores(query: "", sentences: ["a"]).allSatisfy { $0 == 0 })
    }
}
