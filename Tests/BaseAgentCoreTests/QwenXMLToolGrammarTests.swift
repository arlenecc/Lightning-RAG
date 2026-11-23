import XCTest
import MLXGuidedGeneration
@testable import BaseAgentCore

/// Qwen XML 工具调用 EBNF：生成内容、编译、逐 token 掩码的基础验证。
/// 用合成 RAW 词表，不依赖真实模型。
final class QwenXMLToolGrammarTests: XCTestCase {
    private static let vocab = [
        "<|end|>",     // 0 EOS
        "\n",          // 1
        "<function=",  // 2
        "rag_search",  // 3
        ">",           // 4
        "<parameter=", // 5
        "query",       // 6
        "hello",       // 7
        "</parameter>",// 8
        "</function>", // 9
        "</tool_call>",// 10
        "bogus",       // 11
    ]

    private let tools: [JSONValue] = [
        .object([
            "type": .string("function"),
            "function": .object([
                "name": .string("rag_search"),
                "description": .string("search"),
                "parameters": .object([
                    "type": .string("object"),
                    "properties": .object([
                        "query": .object(["type": .string("string")]),
                    ]),
                    "required": .array([.string("query")]),
                ]),
            ]),
        ]),
    ]

    private func bit(_ mask: [Int32], _ id: Int) -> Bool {
        mask[id / 32] & (1 << Int32(id % 32)) != 0
    }

    func testGeneratedGrammarContainsToolAndParameterNames() throws {
        let source = try XCTUnwrap(QwenXMLToolGrammar.make(tools: tools))
        XCTAssertTrue(source.contains("root ::="))
        XCTAssertTrue(source.contains("rag_search"))
        XCTAssertTrue(source.contains("query"))
        XCTAssertTrue(source.contains("</tool_call>"))
        XCTAssertFalse(source.contains("\"type\":\"object\""), "不应把 JSON Schema 直接当语法")
        XCTAssertNil(QwenXMLToolGrammar.make(tools: []), "无工具时不生成语法")
    }

    func testGBNFIncludesTriggerForLazySampler() throws {
        let source = try XCTUnwrap(QwenXMLToolGrammar.makeGBNF(tools: tools))
        XCTAssertTrue(source.contains("root ::= \"<tool_call>\""), "GBNF 需从触发串开始")
        XCTAssertTrue(source.contains("rag_search"))
        XCTAssertTrue(source.contains("query"))
        XCTAssertFalse(source.contains("json_schema"))
    }

    func testGrammarCompilesAndMasksStepByStep() throws {
        let source = try XCTUnwrap(QwenXMLToolGrammar.make(tools: tools))
        let tokenizer = try GrammarTokenizer(vocab: Self.vocab, vocabType: .raw, eosTokenId: 0)
        let constraint = try GrammarConstraint(
            tokenizer: tokenizer, grammar: source, rootRule: "root")

        var mask = try constraint.computeMask()
        XCTAssertTrue(mask.needsApply)
        XCTAssertTrue(bit(mask.mask, 2), "首步应允许 <function=")
        XCTAssertFalse(bit(mask.mask, 11), "干扰 token 应被屏蔽")

        _ = try constraint.commitToken(2) // <function=
        mask = try constraint.computeMask()
        XCTAssertTrue(bit(mask.mask, 3), "应只允许声明过的工具名")

        _ = try constraint.commitToken(3) // rag_search
        mask = try constraint.computeMask()
        XCTAssertTrue(bit(mask.mask, 4), "工具名后应允许 >")

        _ = try constraint.commitToken(4) // >
        mask = try constraint.computeMask()
        XCTAssertTrue(bit(mask.mask, 5), "应允许 <parameter=")
        XCTAssertTrue(bit(mask.mask, 9), "零参数时也应允许直接闭合")

        _ = try constraint.commitToken(5) // <parameter=
        mask = try constraint.computeMask()
        XCTAssertTrue(bit(mask.mask, 6), "应只允许 schema 声明的参数名")
        XCTAssertFalse(bit(mask.mask, 11))
    }

    func testRollbackRestoresToolCallState() throws {
        let source = try XCTUnwrap(QwenXMLToolGrammar.make(tools: tools))
        let tokenizer = try GrammarTokenizer(vocab: Self.vocab, vocabType: .raw, eosTokenId: 0)
        let constraint = try GrammarConstraint(
            tokenizer: tokenizer, grammar: source, rootRule: "root")
        let initial = try constraint.computeMask()
        _ = try constraint.commitToken(2)
        _ = try constraint.commitToken(3)
        try constraint.rollback(2)
        let restored = try constraint.computeMask()
        XCTAssertEqual(initial.mask, restored.mask, "回滚后应回到生成起点，供下一轮复用状态机")
    }
}
