import XCTest
@testable import BaseAgentCore

/// 技能加载判定回归：检索后加载前由决策模型选技能/判合适性，失败静默放行。
final class SkillDecisionGateTests: XCTestCase {
    private final class StubJudge: SkillDecisionJudge, @unchecked Sendable {
        let pickResult: Int?
        let appropriateResult: Bool?
        private let lock = NSLock()
        private(set) var pickCalls = 0
        private(set) var fitCalls = 0
        init(pick: Int? = nil, appropriate: Bool? = nil) {
            pickResult = pick; appropriateResult = appropriate
        }
        func pickSkill(question: String, conversationState: String,
                       candidates: [(name: String, description: String, tags: [String])]) async -> Int? {
            lock.lock(); pickCalls += 1; lock.unlock()
            return pickResult
        }
        func isAppropriate(question: String, conversationState: String,
                           skill: (name: String, description: String, tags: [String])) async -> Bool? {
            lock.lock(); fitCalls += 1; lock.unlock()
            return appropriateResult
        }
    }

    private final class Emb: EmbeddingProvider {
        let identifier = "skill-gate-emb"; let dimension = 4
        func embedDocuments(_ t: [String]) async throws -> [[Float]] { t.map { _ in [1,0,0,0] } }
        func embedQuery(_ t: String) async throws -> [Float] { [1,0,0,0] }
    }

    private func makeRegistry(judge: SkillDecisionJudge?) throws -> (ToolRegistry, URL, SkillIndex) {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("skill-gate-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let skillsDir = root.appendingPathComponent("skills")
        try FileManager.default.createDirectory(at: skillsDir, withIntermediateDirectories: true)
        // 造两个技能
        for (name, desc) in [("alpha", "Alpha skill for testing"), ("beta", "Beta skill for testing")] {
            let dir = skillsDir.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try """
            ---
            name: \(name)
            description: \(desc)
            tags: [test]
            ---
            # \(name) instructions
            Use \(name) carefully.
            """.write(to: dir.appendingPathComponent("SKILL.md"), atomically: true, encoding: .utf8)
        }
        var config = AgentConfig()
        config.workspace = root.path
        let manager = SkillManager(skillsDirectory: skillsDir.path, builtinDirectories: [])
        let index = SkillIndex(manager: manager)
        let retriever = SkillRetriever(manager: manager, embedding: Emb(), reranker: nil)
        let registry = AgentToolFactory.build(
            config: config, callbacks: NoopCallbacks(),
            session: Session(budget: ContextBudget(total: 30_000)),
            ragEngine: RAGEngine(config: config, knowledgeBase: root.path,
                                 embedding: Emb(), reranker: nil),
            skillManager: manager, skillIndex: index, skillRetriever: retriever,
            skillJudge: judge,
            skillContextProvider: { ("测试问句", "Plan: test") })
        return (registry, root, index)
    }

    /// 多技能：判定选中第 0 个，检索结果只返回选中的技能。
    func testPickAmongMultipleSkills() async throws {
        let judge = StubJudge(pick: 0)
        let (registry, root, _) = try makeRegistry(judge: judge)
        defer { try? FileManager.default.removeItem(at: root) }
        let result = try await registry.execute("skill_tools", [
            "op": .string("search"), "query": .string("test")
        ])
        XCTAssertTrue(result.success)
        XCTAssertEqual(judge.pickCalls, 1)
        XCTAssertTrue(result.output.contains("alpha"), "只应返回判中的 alpha")
        XCTAssertFalse(result.output.contains("\"name\" : \"beta\""), "不应返回 beta")
    }

    /// 单/多技能判定为"都不合适"（负 index）→ 禁止返回技能给模型加载。
    func testRejectAllSkills() async throws {
        let judge = StubJudge(pick: -1)
        let (registry, root, _) = try makeRegistry(judge: judge)
        defer { try? FileManager.default.removeItem(at: root) }
        let result = try await registry.execute("skill_tools", [
            "op": .string("search"), "query": .string("test")
        ])
        XCTAssertTrue(result.success)
        XCTAssertTrue(result.output.contains("没有合适的技能"))
    }

    /// judge 不可用（nil）→ 既有行为不变，返回多个技能。
    func testNoJudgePassthrough() async throws {
        let (registry, root, _) = try makeRegistry(judge: nil)
        defer { try? FileManager.default.removeItem(at: root) }
        let result = try await registry.execute("skill_tools", [
            "op": .string("search"), "query": .string("test")
        ])
        XCTAssertTrue(result.success)
        XCTAssertTrue(result.output.contains("alpha"))
        XCTAssertTrue(result.output.contains("beta"), "无判定器时保持原行为")
    }

    /// load 兜底：判定不合适时禁止加载内容（直接 load 也被拦）。
    func testLoadBlockedWhenInappropriate() async throws {
        let judge = StubJudge(appropriate: false)
        let (registry, root, _) = try makeRegistry(judge: judge)
        defer { try? FileManager.default.removeItem(at: root) }
        let result = try await registry.execute("skill_tools", [
            "op": .string("load"), "path": .string("alpha")
        ])
        XCTAssertFalse(result.success, "判定不合适应禁止 load")
        XCTAssertEqual(judge.fitCalls, 1)
        XCTAssertTrue(result.error.contains("不合适"))
    }

    /// 配置导出包含技能判定开关（读取路径由 Config.apply 统一处理）。
    func testConfigExportContainsSkillGate() {
        let c = AgentConfig()
        XCTAssertTrue(c.skillDecisionGateEnabled)
        XCTAssertEqual(c.skillDecisionGateThreshold, 0.5)
        let dict = c.export()
        XCTAssertEqual(dict["skill_decision_gate_enabled"] as? Bool, true)
        XCTAssertEqual(dict["skill_decision_gate_threshold"] as? Double, 0.5)
    }

    /// judge 判定不可用（nil 三态）→ load 放行，保持原行为。
    func testLoadPassthroughWhenJudgeUnavailable() async throws {
        let judge = StubJudge(appropriate: nil)
        let (registry, root, _) = try makeRegistry(judge: judge)
        defer { try? FileManager.default.removeItem(at: root) }
        let result = try await registry.execute("skill_tools", [
            "op": .string("load"), "path": .string("alpha")
        ])
        XCTAssertTrue(result.success, "判定不可用时应放行")
        XCTAssertTrue(result.output.contains("Alpha skill") || result.output.contains("alpha"))
    }
}

private final class NoopCallbacks: AgentCallbacks {}
