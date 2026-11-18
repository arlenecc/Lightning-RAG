import XCTest
@testable import ColBERTReader

/// 查询侧轻量 token 剪枝判定回归测试：只丢标点/空白/核心功能词，
/// 保留疑问词、程度/形容词、实词与中文疑问词。不依赖真实 ONNX 模型。
final class ColbertQueryPruneTests: XCTestCase {
    private func isDiscardable(_ token: String) -> Bool {
        TokenPruner.shared.isLightweightDiscardableToken(token)
    }

    /// 纯标点、SentencePiece 空白占位应被丢弃。
    func testPunctuationAndWhitespaceDiscarded() {
        for token in ["▁", ",", "!", "?", "，", "。"] {
            XCTAssertTrue(isDiscardable(token), "token \(token) 应作为标点/空白被剪掉")
        }
    }

    /// 核心功能词应被丢弃，但疑问词与程度词即使在核心表里也保留。
    func testFunctionWordsDiscardedButInterrogativesKept() {
        for token in ["▁the", "▁of", "▁a", "▁is", "▁and"] {
            XCTAssertTrue(isDiscardable(token), "功能词 \(token) 应被剪掉")
        }
        for token in ["▁what", "▁which", "▁who", "▁how", "▁why", "▁when", "▁where"] {
            XCTAssertFalse(isDiscardable(token), "疑问词 \(token) 必须保留")
        }
        for token in ["▁very", "▁more", "▁most"] {
            XCTAssertFalse(isDiscardable(token), "程度词 \(token) 必须保留")
        }
    }

    /// 形容词、实词、数字与中文疑问词必须保留。
    func testContentAndChineseInterrogativesKept() {
        for token in ["▁better", "▁best", "▁important", "▁quality", "合同", "违约", "▁123"] {
            XCTAssertFalse(isDiscardable(token), "实词/形容词 \(token) 不应被误剪")
        }
        for token in ["什么", "怎么", "为什么", "哪些", "谁", "多少"] {
            XCTAssertFalse(isDiscardable(token), "中文疑问词 \(token) 必须保留")
        }
    }

    /// 中文纯虚词单字丢弃，但单字实词保留。
    func testChineseFunctionCharsVsContent() {
        for token in ["的", "了", "着"] {
            XCTAssertTrue(isDiscardable(token), "虚词 \(token) 应被剪掉")
        }
        for token in ["方", "合", "约"] {
            XCTAssertFalse(isDiscardable(token), "单字实词 \(token) 不应被误剪")
        }
    }
}
