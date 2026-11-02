import XCTest
@testable import BaseAgentCore

/// Token 级前缀匹配（形态①②③统一复用的核心判定）的纯函数语义验证。
final class CommonTokenPrefixTests: XCTestCase {

    func testStrictPrefixExtension() {
        // 形态①：缓存序列是新 prompt 的严格前缀 → 复用整段缓存，只算增量。
        let cached = [10, 20, 30, 40]
        XCTAssertEqual(MLXLLMService.commonTokenPrefix(cached + [50, 60], cached), 4)
    }

    func testDivergenceMidway() {
        // 形态③：中途分叉 → 只复用到分叉点。
        let cached = [10, 20, 30, 40, 50]
        let next = [10, 20, 30, 99, 50]
        XCTAssertEqual(MLXLLMService.commonTokenPrefix(next, cached), 3)
    }

    func testNewPromptShorter() {
        // 形态②：新 prompt 比缓存短（回退/收缩重试）→ 复用公共部分。
        let cached = [10, 20, 30, 40, 50]
        let next = [10, 20]
        XCTAssertEqual(MLXLLMService.commonTokenPrefix(next, cached), 2)
    }

    func testNoCommonPrefix() {
        XCTAssertEqual(MLXLLMService.commonTokenPrefix([1, 2], [9, 8, 7]), 0)
    }

    func testEmptySequences() {
        XCTAssertEqual(MLXLLMService.commonTokenPrefix([], [1, 2]), 0)
        XCTAssertEqual(MLXLLMService.commonTokenPrefix([1, 2], []), 0)
        XCTAssertEqual(MLXLLMService.commonTokenPrefix([], []), 0)
    }

    func testIdenticalSequences() {
        // 完全相等：没有新增量，调用方应放弃复用走重建（无新消息语义）。
        let same = [1, 2, 3]
        XCTAssertEqual(MLXLLMService.commonTokenPrefix(same, same), 3)
    }
}
