import XCTest
@testable import ColBERTReader

/// 文档侧 token 剪枝判定（标点/空白/停用词剔除，实词保留）的回归测试。
/// 只验证判定规则本身，不依赖真实 ONNX 模型。
final class ColbertTokenPruneTests: XCTestCase {
    private func isDiscardable(_ token: String) -> Bool {
        TokenPruner.shared.isDiscardableToken(token)
    }

    /// 纯标点、SentencePiece 空白占位（▁）应被丢弃。
    func testPunctuationAndWhitespaceDiscarded() {
        for token in ["▁", ",", "(", ")", "!", "?", "。", "，"] {
            XCTAssertTrue(isDiscardable(token), "token \(token) 应作为标点/空白被剪掉")
        }
    }

    /// 带 ▁ 前缀的英文停用词应被丢弃。
    func testEnglishStopwordsDiscarded() {
        for token in ["▁the", "▁of", "▁a", "▁and", "▁is", "▁what"] {
            XCTAssertTrue(isDiscardable(token), "停用词 \(token) 应被剪掉")
        }
    }

    /// 纯中文虚词单字应被丢弃。
    func testChineseFunctionWordsDiscarded() {
        for token in ["的", "了", "着", "和", "在"] {
            XCTAssertTrue(isDiscardable(token), "虚词 \(token) 应被剪掉")
        }
    }

    /// 完整停用词表（哈工大）里的中文停用词也应丢弃，如「应当」「当」「应用」。
    func testChineseStopwordListDiscarded() {
        for token in ["应当", "当", "应用"] {
            XCTAssertTrue(isDiscardable(token), "停用词 \(token) 应被剪掉")
        }
    }

    /// 中文实词（含单字实词）、英文实词、数字必须保留，避免误伤有效信息。
    func testContentTokensKept() {
        for token in ["合同", "违约", "责任", "方", "赔偿", "▁123", "quality"] {
            XCTAssertFalse(isDiscardable(token), "实词 \(token) 不应被误剪")
        }
    }
}
