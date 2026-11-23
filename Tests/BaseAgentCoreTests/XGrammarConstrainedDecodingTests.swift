import XCTest
import MLXGuidedGeneration
@testable import BaseAgentCore

/// 验证 mlx-swift-lm 内置的 MLXCXGrammar（xgrammar C++ + Swift 桥接）可用：
/// JSON Schema 编译 → 逐 token 位掩码 → 提交推进 → EOS 终止，全链路走通。
/// 用合成 RAW 词表（不依赖真实模型），词表覆盖 `{"name":"Alice"}` 的完整序列。
final class XGrammarConstrainedDecodingTests: XCTestCase {

    /// id 0 是 EOS（注册为 stop token）；id 10/11 是语法不允许的干扰 token。
    private static let vocab = [
        "<|end|>",  // 0 EOS
        "{",        // 1
        "\"",       // 2
        "name",     // 3
        "\"",       // 4
        ":",        // 5
        "\"",       // 6
        "Alice",    // 7
        "\"",       // 8
        "}",        // 9
        "x",        // 10 干扰
        "y",        // 11 干扰
    ]

    private func makeTokenizer() throws -> GrammarTokenizer {
        try GrammarTokenizer(
            vocab: Self.vocab, vocabType: .raw, eosTokenId: 0)
    }

    private func makeConstraint(_ tokenizer: GrammarTokenizer) throws -> GrammarConstraint {
        try GrammarConstraint(
            tokenizer: tokenizer,
            jsonSchema: """
            {"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}
            """)
    }

    private func bit(_ mask: [Int32], _ id: Int) -> Bool {
        mask[id / 32] & (1 << Int32(id % 32)) != 0
    }

    /// 首步掩码：只允许 `{`，干扰 token 被屏蔽。
    func testInitialMaskFiltersTokens() throws {
        let tokenizer = try makeTokenizer()
        let constraint = try makeConstraint(tokenizer)
        let result = try constraint.computeMask()
        XCTAssertTrue(result.needsApply, "语法掩码必须排除部分 token")
        XCTAssertTrue(bit(result.mask, 1), "首位应允许 {")
        XCTAssertFalse(bit(result.mask, 10), "干扰 token x 应被屏蔽")
        XCTAssertFalse(bit(result.mask, 11), "干扰 token y 应被屏蔽")
        XCTAssertFalse(bit(result.mask, 9), "首位不应允许 }")
        XCTAssertFalse(result.isTerminated)
    }

    /// 提交语义：合法 token 推进状态，非法 token 被拒绝且状态不变。
    func testCommitAcceptsAndRejects() throws {
        let tokenizer = try makeTokenizer()

        // 非法首 token：拒绝并抛错。
        let rejecting = try makeConstraint(tokenizer)
        XCTAssertThrowsError(try rejecting.commitToken(10))

        // 合法首 token：接受。
        let accepting = try makeConstraint(tokenizer)
        let commit = try accepting.commitToken(1)
        XCTAssertFalse(commit.isTerminated)
        // 提交 { 后应允许 "（id 2/4/6/8 任一）。
        let next = try accepting.computeMask()
        XCTAssertTrue(bit(next.mask, 2), "{ 后应允许 \"")
        XCTAssertFalse(bit(next.mask, 1), "{ 后不应再允许 {")
    }

    /// 全序列走通：{ " name " : " Alice " } EOS → 终止。
    /// 每步先掩码（断言期望 token 可用），再提交。
    func testFullSequenceTerminates() throws {
        let tokenizer = try makeTokenizer()
        let constraint = try makeConstraint(tokenizer)

        // 每步允许集合：字符串内容步（id 7 之后）允许多 token，做宽松断言。
        let steps: [(expectAllowed: Int, commit: Int32)] = [
            (1, 1),  // {
            (2, 2),  // "
            (3, 3),  // name
            (4, 4),  // "
            (5, 5),  // :
            (6, 6),  // "
            (7, 7),  // Alice
            (8, 8),  // "
            (9, 9),  // }
            (0, 0),  // EOS
        ]
        for (index, step) in steps.enumerated() {
            let mask = try constraint.computeMask()
            XCTAssertTrue(bit(mask.mask, Int(step.expectAllowed)),
                          "第 \(index) 步应允许 token \(step.expectAllowed)")
            let result = try constraint.commitToken(step.commit)
            if index == steps.count - 1 {
                XCTAssertTrue(result.isTerminated, "EOS 提交后语法应判定终止")
            } else {
                XCTAssertFalse(result.isTerminated)
            }
        }
    }

    /// EBNF 入口：`root ::= "{" ...` 编译 + 掩码可用。
    func testEBNFGrammarCompiles() throws {
        let tokenizer = try makeTokenizer()
        let constraint = try GrammarConstraint(
            tokenizer: tokenizer,
            grammar: "root ::= \"{\" \"name\" \":\" \"Alice\" \"}\"",
            rootRule: "root")
        let mask = try constraint.computeMask()
        XCTAssertTrue(bit(mask.mask, 1), "EBNF 语法首位应允许 {")
        XCTAssertFalse(bit(mask.mask, 10))
    }

    /// 回滚：提交一步后回滚，掩码恢复到初始状态。
    func testRollbackRestoresState() throws {
        let tokenizer = try makeTokenizer()
        let constraint = try makeConstraint(tokenizer)
        let initial = try constraint.computeMask()
        _ = try constraint.commitToken(1)
        try constraint.rollback(1)
        let restored = try constraint.computeMask()
        XCTAssertEqual(initial.mask, restored.mask, "回滚后掩码应与初始位一致")
    }

    /// vendored xgrammar 为 v0.1.30：`clone()`（fork）不可用（0.2.x 才引入），
    /// 投机解码/beam 场景暂不可用；普通逐步约束解码只需要 mask+commit+rollback。
    func testCloneUnavailableInVendoredVersion() throws {
        let tokenizer = try makeTokenizer()
        let constraint = try makeConstraint(tokenizer)
        XCTAssertThrowsError(try constraint.clone())
    }
}
