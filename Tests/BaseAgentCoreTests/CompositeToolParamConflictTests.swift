import XCTest
@testable import BaseAgentCore

/// 工具合并（v1.3.5）的参数冲突回归：组合工具独占 `op` 键，任何子工具若
/// 仍声明 `op` 参数，或不同子工具间同名参数类型/枚举不一致，都会让「先到先得」
/// 的 schema 合并静默丢定义、`run` 前删除 `op` 让子工具永远收不到动作参数。
private final class ParamConflictLLM: LLMService, @unchecked Sendable {
    let identifier = "param-conflict-stub"
    func chatStream(messages: [ChatMessage], tools: [JSONValue],
                    temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }
}

final class CompositeToolParamConflictTests: XCTestCase {

    private func makeAgent() -> Agent {
        var config = AgentConfig()
        config.workspace = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("composite-param-\(UUID().uuidString)").path
        return Agent(llm: ParamConflictLLM(), config: config, callbacks: NoopCallbacks())
    }

    private func composites(_ agent: Agent) -> [CompositeTool] {
        agent.tools.allTools.compactMap { $0 as? CompositeTool }
    }

    /// 所有组合工具都不得存在保留键冲突：子工具声明 `op` 会导致其动作参数
    /// 被组合层吞掉（历史 bug：skill_search 的 op=list 永远收不到）。
    func testNoSubtoolDeclaresReservedOp() {
        let agent = makeAgent()
        defer { agent.close() }
        let all = composites(agent)
        XCTAssertEqual(Set(all.map(\.name)),
                       ["file_tools", "web_scan", "long_memory", "skill_tools", "rag_tools"])
        for tool in all {
            XCTAssertTrue(tool.reservedKeyCollisions.isEmpty,
                          "\(tool.name) 的子工具仍声明 op 参数：\(tool.reservedKeyCollisions)")
        }
    }

    /// 同名参数（path/query/top_k/url/glob 等）跨子命令的类型与枚举必须一致，
    /// 否则合并时只保留第一个定义，模型拿到的 schema 与实际行为不符。
    func testNoSameNameParameterTypeCollision() {
        let agent = makeAgent()
        defer { agent.close() }
        for tool in composites(agent) {
            XCTAssertTrue(tool.parameterTypeCollisions.isEmpty,
                          "\(tool.name) 同名参数类型/枚举冲突：\(tool.parameterTypeCollisions)")
        }
    }

    /// `skill_tools` 的 op 枚举必须完整，且 `list` 是真实可路由的动作。
    func testSkillToolsListOpIsExposedAndRunnable() async throws {
        let agent = makeAgent()
        defer { agent.close() }
        guard let tool = agent.tools.get("skill_tools") as? CompositeTool else {
            return XCTFail("缺少 skill_tools")
        }
        let ops = tool.parameters.objectValue?["properties"]?.objectValue?["op"]?
            .objectValue?["enum"]?.arrayValue?.compactMap(\.stringValue) ?? []
        XCTAssertEqual(Set(ops), ["search", "semantic_search", "load", "save", "list"])

        let listed = try await tool.run(["op": .string("list")])
        XCTAssertTrue(listed.success, "op=list 应可路由：\(listed.error)")
        XCTAssertFalse(listed.output.contains("unknown op"))
        XCTAssertFalse(listed.output.contains("缺少参数: query"),
                       "op=list 不应落进关键词检索分支：\(listed.output)")

        let searched = try await tool.run(["op": .string("search"),
                                           "query": .string("部署")])
        XCTAssertTrue(searched.success, "op=search 仍应可用：\(searched.error)")
    }

    /// 原子工具的 schema 不再声明 `op`；`required` 用的是改名后的动作键。
    func testAtomicToolsSchemaKeys() {
        let agent = makeAgent()
        defer { agent.close() }
        let search = SkillSearchTool(config: agent.config)
        XCTAssertNil(search.parameters.objectValue?["properties"]?.objectValue?["op"],
                     "skill_search 不得声明 op（会被组合层删除）")
        XCTAssertEqual(search.parameters.objectValue?["required"]?.arrayValue?
            .compactMap(\.stringValue), ["query"])

        let graph = MemoryGraphTool(config: agent.config)
        XCTAssertNil(graph.parameters.objectValue?["properties"]?.objectValue?["op"])
        XCTAssertEqual(graph.parameters.objectValue?["required"]?.arrayValue?
            .compactMap(\.stringValue), ["graph_op"])
        let graphProps = graph.parameters.objectValue?["properties"]?.objectValue ?? [:]
        XCTAssertNotNil(graphProps["graph_op"], "required 的键必须真实存在")
    }

    /// 组合工具的 `op` 参数始终覆盖全部子命令值（不被子工具 schema 覆盖）。
    func testCompositeOpEnumAlwaysPresent() {
        let agent = makeAgent()
        defer { agent.close() }
        for tool in composites(agent) {
            let props = tool.parameters.objectValue?["properties"]?.objectValue ?? [:]
            let enums = props["op"]?.objectValue?["enum"]?.arrayValue?
                .compactMap(\.stringValue) ?? []
            XCTAssertFalse(enums.isEmpty, "\(tool.name) 缺少 op 枚举")
            XCTAssertEqual(Set(enums).count, enums.count, "\(tool.name) op 枚举有重复值")
            XCTAssertEqual(tool.parameters.objectValue?["required"]?.arrayValue?
                .compactMap(\.stringValue), ["op"])
        }
    }
}
