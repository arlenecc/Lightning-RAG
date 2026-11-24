import XCTest
import MLXGuidedGeneration
@testable import BaseAgentCore

/// 多步计划落地链路的三层保障：
/// 1) 工具层兼容缺 op 的旧格式；
/// 2) grammar 按 schema required 强制必填参数；
/// 3) 内核纠正优先于代建（由 KernelLoop 集成测试覆盖，见下方纯函数校验）。
final class PlanCreationRobustnessTests: XCTestCase {
    private func temporaryWorkspace() -> String {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("plan-robustness-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.path
    }

    private func makeConfig(_ workspace: String) -> AgentConfig {
        var config = AgentConfig()
        config.workspace = workspace
        return config
    }

    // MARK: - 工具层兼容旧格式

    func testLegacyCreateCallWithoutOpCreatesPlan() async throws {
        let config = makeConfig(temporaryWorkspace())
        let tool = UpdatePlanTool(config: config, callbacks: nil)
        let args: [String: JSONValue] = [
            "goal": .string("分析加元汇率"),
            "steps": .array([
                .object(["title": .string("收集汇率数据"), "status": .string("pending")]),
                .object(["title": .string("核对宏观因素"), "status": .string("pending")]),
            ]),
        ]
        let result = try await tool.run(args)
        XCTAssertTrue(result.success, result.error)
        let plan = try XCTUnwrap(PlanStore.load(config))
        XCTAssertEqual(plan.goal, "分析加元汇率")
        XCTAssertEqual(plan.steps.count, 2)
        XCTAssertEqual(plan.steps[0].status, "doing")
    }

    func testLegacyPatchCallWithoutOpUpdatesPlan() async throws {
        let config = makeConfig(temporaryWorkspace())
        let tool = UpdatePlanTool(config: config, callbacks: nil)
        let createArgs: [String: JSONValue] = [
            "op": .string("create"),
            "goal": .string("目标"),
            "steps": .array([.string("第一步"), .string("第二步")]),
        ]
        let created = try await tool.run(createArgs)
        XCTAssertTrue(created.success, created.error)

        let patchArgs: [String: JSONValue] = [
            "step_id": .string("1"),
            "status": .string("done"),
            "note": .string("已完成"),
        ]
        let patched = try await tool.run(patchArgs)
        XCTAssertTrue(patched.success, patched.error)
        let plan = try XCTUnwrap(PlanStore.load(config))
        XCTAssertEqual(plan.steps[0].status, "done")
        XCTAssertEqual(plan.steps[0].note, "已完成")
    }

    func testPatchWithOnlyNoteFallsBackToDoingStep() async throws {
        let config = makeConfig(temporaryWorkspace())
        let tool = UpdatePlanTool(config: config, callbacks: nil)
        let createArgs: [String: JSONValue] = [
            "op": .string("create"),
            "goal": .string("目标"),
            "steps": .array([.string("第一步"), .string("第二步")]),
        ]
        let created = try await tool.run(createArgs)
        XCTAssertTrue(created.success, created.error)

        let noteArgs: [String: JSONValue] = ["op": .string("patch"), "note": .string("记一笔")]
        let result = try await tool.run(noteArgs)
        XCTAssertTrue(result.success, result.error)
        let plan = try XCTUnwrap(PlanStore.load(config))
        XCTAssertEqual(plan.steps[0].note, "记一笔")
    }

    func testEmptyCallStillReportsMissingOp() async throws {
        let config = makeConfig(temporaryWorkspace())
        let tool = UpdatePlanTool(config: config, callbacks: nil)
        let emptyArgs: [String: JSONValue] = [:]
        let result = try await tool.run(emptyArgs)
        XCTAssertFalse(result.success)
        XCTAssertTrue(result.error.contains("op"))
    }

    // MARK: - 文档名容错

    func testDocumentKeyNormalizesSpaceDashAndFullWidth() {
        let key = RAGEngine.normalizedDocumentKey("46 亿年的地球物语 - 地球起源到今天的全部历史.pdf")
        let stored = RAGEngine.normalizedDocumentKey("46亿年的地球物语-地球起源到今天的全部历史.pdf")
        XCTAssertEqual(key, stored)
        XCTAssertEqual(RAGEngine.normalizedDocumentKey("借地而生 - 写给地球人的土壤生命史"),
                       RAGEngine.normalizedDocumentKey("借地而生：写给地球人的土壤生命史.pdf"))
        XCTAssertEqual(RAGEngine.normalizedDocumentKey("ＡＢＣ　ＤＥＦ"), "abcdef")
    }

    // MARK: - grammar 强制必填参数

    private func toolSchema(required: [String]) -> JSONValue {
        .object([
            "type": .string("function"),
            "function": .object([
                "name": .string("update_plan"),
                "description": .string("plan"),
                "parameters": .object([
                    "type": .string("object"),
                    "properties": .object([
                        "op": .object(["type": .string("string")]),
                        "goal": .object(["type": .string("string")]),
                    ]),
                    "required": .array(required.map { .string($0) }),
                ]),
            ]),
        ])
    }

    func testGrammarForcesRequiredParameter() throws {
        let source = try XCTUnwrap(QwenXMLToolGrammar.make(tools: [toolSchema(required: ["op"])]))
        XCTAssertTrue(source.contains("req0_0 ::= \"<parameter=\" \"op\""),
                      "必填 op 应有独立且强制的产生式: \(source)")
        XCTAssertTrue(source.contains("call0 ::= \"update_plan\" ws \">\" ws req0_0"),
                      "call 规则必须展开必填项: \(source)")
        XCTAssertFalse(source.contains("param0 ::= \"<parameter=\" (\"goal\" | \"op\")"),
                       "必填项不应再出现在可重复的可选参数里: \(source)")
    }

    func testGrammarWithRequiredCompilesAndMasksFirstStep() throws {
        let source = try XCTUnwrap(QwenXMLToolGrammar.make(tools: [toolSchema(required: ["op"])]))
        let vocab = [
            "<|end|>", "<function=", "update_plan", ">", "<parameter=",
            "op", "goal", "create", "done", "</parameter>", "</function>",
            "</tool_call>", "\n",
        ]
        let tokenizer = try GrammarTokenizer(vocab: vocab, vocabType: .raw, eosTokenId: 0)
        let constraint = try GrammarConstraint(
            tokenizer: tokenizer, grammar: source, rootRule: "root")
        func allowed(_ id: Int) -> Bool {
            guard let mask = try? constraint.computeMask() else { return false }
            return mask.mask[id / 32] & (1 << Int32(id % 32)) != 0
        }

        // 首步只允许 <function=，干扰 token 被屏蔽（语法能被 xgrammar 编译）。
        XCTAssertTrue(allowed(1))
        XCTAssertFalse(allowed(3))
        _ = try constraint.commitToken(1)
        XCTAssertTrue(allowed(2), "应允许声明过的工具名")
        _ = try constraint.commitToken(2)
        XCTAssertTrue(allowed(3), "工具名后应允许 >")
    }

    func testGrammarWithoutRequiredKeepsOptionalParameters() throws {
        let source = try XCTUnwrap(QwenXMLToolGrammar.make(tools: [toolSchema(required: [])]))
        XCTAssertTrue(source.contains("call0 ::= \"update_plan\" ws \">\" ws param0*"),
                      "无必填项时保持旧的可选参数结构: \(source)")
    }
}
