import XCTest
import MLX
import MLXLMCommon
import MLXGuidedGeneration
@testable import BaseAgentCore

/// 约束处理器本身的行为：触发前完全放行，触发后才施加掩码。
/// 用合成词表与假 tokenizer，不依赖真实模型。
final class GrammarLogitProcessorTests: XCTestCase {
    private static let vocab = [
        "<|end|>", "<function=", "rag_search", "bogus", "<tool_call>",
    ]

    private struct FakeTokenizer: MLXLMCommon.Tokenizer {
        let vocab: [String]
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            [vocab.firstIndex(of: text) ?? 0]
        }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map { vocab[$0] }.joined()
        }
        func convertTokenToId(_ token: String) -> Int? { vocab.firstIndex(of: token) }
        func convertIdToToken(_ id: Int) -> String? { vocab[id] }
        var bosToken: String? { nil }
        var eosToken: String? { "<|end|>" }
        var unknownToken: String? { nil }
        func applyChatTemplate(messages: [[String: any Sendable]],
                               tools: [[String: any Sendable]]?,
                               additionalContext: [String: any Sendable]?) throws -> [Int] { [] }
    }

    private func makeTools() -> [BaseAgentCore.JSONValue] {
        [BaseAgentCore.JSONValue.object([
            "function": BaseAgentCore.JSONValue.object([
                "name": BaseAgentCore.JSONValue.string("rag_search"),
                "parameters": BaseAgentCore.JSONValue.object([
                    "type": BaseAgentCore.JSONValue.string("object"),
                    "properties": BaseAgentCore.JSONValue.object([:]),
                ]),
            ]),
        ])]
    }

    func testMaskOnlyAfterTrigger() throws {
        let source = try XCTUnwrap(QwenXMLToolGrammar.make(tools: makeTools()))
        let grammarTokenizer = try GrammarTokenizer(
            vocab: Self.vocab, vocabType: .raw, eosTokenId: 0)
        let constraint = try GrammarConstraint(
            tokenizer: grammarTokenizer, grammar: source, rootRule: "root")
        let state = GrammarProcessorState(
            constraint: constraint,
            tokenizer: UnguardedTokenizer(FakeTokenizer(vocab: Self.vocab)),
            vocabSize: Self.vocab.count,
            trigger: QwenXMLToolGrammar.trigger)
        var processor = GrammarLogitProcessor(state: state)

        let logits = MLXArray([Float](repeating: 0, count: Self.vocab.count))
        // 走官方 API 的 didSample 路径（token.item 读取 → commit）。
        processor.didSample(token: MLXArray(3)) // "bogus"
        let free = processor.process(logits: logits).asArray(Float.self)
        XCTAssertEqual(free, [Float](repeating: 0, count: Self.vocab.count))

        // 触发串：下一位起才套掩码，干扰 token 应被屏蔽，工具名前缀应放行。
        processor.didSample(token: MLXArray(4)) // "<tool_call>"
        let constrained = processor.process(logits: logits).asArray(Float.self)
        XCTAssertTrue(constrained[3].isInfinite && constrained[3] < 0, "干扰 token 应被屏蔽")
        XCTAssertEqual(constrained[1], 0, "<function= 应放行")

        // 下一轮 prompt() 必须清掉状态，保证普通对话不复用上一轮的掩码。
        processor.prompt(MLXArray([1, 2, 3]))
        let restored = processor.process(logits: logits).asArray(Float.self)
        XCTAssertEqual(restored, [Float](repeating: 0, count: Self.vocab.count))
    }
}
