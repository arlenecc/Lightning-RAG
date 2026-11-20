import XCTest
@testable import BaseAgentCore

/// 内核决策门（complete 收尾拦截 + plan_step 计划评审）单元测试。
///
/// 用 stub 判定器驱动 KernelLoop，验证：
/// - 信息不足时拦截收尾并注入提示，预算耗尽后放行；
/// - 信息足够 / 判定不可用（nil）时静默放行，不额外消耗轮次；
/// - 计划创建后评审步骤，判「不必要」注入一次修订提示（同一计划只评一次）；
/// - 未装配判定器时行为与未接入前完全一致（回归保障）。
final class KernelDecisionGateTests: XCTestCase {

    // MARK: - Stub

    /// 决策判定 stub：按预设返回，同时记录调用内容供断言。
    private final class StubJudge: KernelDecisionJudge, @unchecked Sendable {
        let stepNecessaryResult: Bool?
        let informationSufficientResult: Bool?
        private(set) var stepCalls: [(goal: String, step: String)] = []
        private(set) var completeCalls: [(task: String, evidence: String)] = []
        private let lock = NSLock()

        init(stepNecessaryResult: Bool? = nil,
             informationSufficientResult: Bool? = nil) {
            self.stepNecessaryResult = stepNecessaryResult
            self.informationSufficientResult = informationSufficientResult
        }

        func stepNecessary(goal: String, step: String) async -> Bool? {
            lock.lock(); defer { lock.unlock() }
            stepCalls.append((goal, step))
            return stepNecessaryResult
        }

        func informationSufficient(task: String, evidence: String) async -> Bool? {
            lock.lock(); defer { lock.unlock() }
            completeCalls.append((task, evidence))
            return informationSufficientResult
        }
    }

    // MARK: - 复用工具

    /// 脚本化假模型：第 1 轮按 `firstTurn` 发事件，之后每轮输出纯文本答复。
    private final class ScriptedModel {
        let firstTurn: () -> [LLMStreamEvent]
        let textAnswer: String
        var callCount = 0
        private(set) var userPrompts: [String] = []
        var kernelLogs: [String] = []

        init(firstTurn: @escaping () -> [LLMStreamEvent],
             textAnswer: String = "查询完成：北京今日晴，气温 5 到 15 摄氏度，西北风三级，空气质量优。") {
            self.firstTurn = firstTurn
            self.textAnswer = textAnswer
        }

        func stream(_ messages: [ChatMessage]) -> AsyncThrowingStream<LLMStreamEvent, Error> {
            callCount += 1
            let lastUser = messages.last { $0.role == .user }
            userPrompts.append(lastUser?.content ?? "")
            let events: [LLMStreamEvent] = callCount == 1 ? firstTurn() : [
                .init(kind: .content, content: textAnswer),
                .init(kind: .done),
            ]
            return AsyncThrowingStream { c in
                for e in events { c.yield(e) }
                c.finish()
            }
        }
    }

    private func makeRuntime(model: ScriptedModel,
                             session: Session,
                             onExecute: (([ToolCall]) -> Void)? = nil) -> KernelRuntime {
        KernelRuntime(
            stream: { messages, _, _ in model.stream(messages) },
            execute: { calls in
                onExecute?(calls)
                return calls.map { _ in .ok("工具结果：部分信息……") }
            },
            schemas: { [] },
            log: { model.kernelLogs.append($0) }
        )
    }

    private func makeConfig(maxFinishBlocks: Int = 2,
                            maxPlanReviews: Int = 1) -> AgentConfig {
        var config = AgentConfig()
        config.kernelConsolidateEvery = 24
        config.kernelRawTurns = 50
        config.kernelDecisionGateMaxFinishBlocks = maxFinishBlocks
        config.kernelDecisionGateMaxPlanReviews = maxPlanReviews
        return config
    }

    private static func toolTurnEvents() -> [LLMStreamEvent] {
        [.init(kind: .done, toolCalls: [
            ToolCall(id: "t1", name: "web_search",
                     arguments: ["query": .string("测试查询")]),
        ])]
    }

    private static func planCreateCall() -> ToolCall {
        ToolCall(id: "p1", name: "update_plan",
                 arguments: ["op": .string("create"),
                             "goal": .string("完成测试任务"),
                             "steps": .string("1. 查资料; 2. 总结")])
    }

    /// 在 execute 里真实落一个两步计划（模拟 PlanTool 的行为）。
    private static func installPlan(session: Session) {
        session.setPlan(Plan(goal: "完成测试任务", steps: [
            PlanStep(id: "1", title: "检索相关资料", status: "doing"),
            PlanStep(id: "2", title: "总结给出答复", status: "pending"),
        ]))
    }

    // MARK: - complete 门

    /// 信息不足：拦截收尾并注入提示，预算耗尽后放行。
    func testCompleteGateBlocksInsufficientFinish() async {
        let judge = StubJudge(informationSufficientResult: false)
        let model = ScriptedModel(firstTurn: { Self.events(Self.toolTurnEvents()) })
        let session = Session(budget: ContextBudget(total: 60_000), rawTurns: 50)
        let loop = KernelLoop(runtime: makeRuntime(model: model, session: session),
                              config: makeConfig(maxFinishBlocks: 2),
                              systemPrompt: { "SYS" }, session: session, maxTurns: 8)
        loop.decisionJudge = judge
        let result = await loop.run("北京今天的天气怎么样")
        print("=== kernel logs ===")
        model.kernelLogs.forEach { print("LOG: \($0)") }
        model.userPrompts.enumerated().forEach { i, p in
            print("PROMPT[\(i)]: \(p.prefix(600))")
        }

        // 两次拦截后第三次文本放行。
        XCTAssertEqual(judge.completeCalls.count, 2,
                       "预算内应收尾判定 2 次（第 3 轮预算耗尽直接放行）")
        XCTAssertTrue(model.userPrompts.contains {
            $0.contains("决策模型判定当前收集的信息可能不足")
        }, "拦截提示应作为 pending 注入下一轮 user 提示")
        // 首轮工具 + 2 次被拦文本 + 1 次放行文本 = 4 次调用。
        XCTAssertEqual(model.callCount, 4)
    }

    /// 信息足够：判定放行，不额外消耗轮次。
    func testCompleteGatePassesWhenSufficient() async {
        let judge = StubJudge(informationSufficientResult: true)
        let model = ScriptedModel(firstTurn: { Self.events(Self.toolTurnEvents()) })
        let session = Session(budget: ContextBudget(total: 60_000), rawTurns: 50)
        let loop = KernelLoop(runtime: makeRuntime(model: model, session: session),
                              config: makeConfig(),
                              systemPrompt: { "SYS" }, session: session, maxTurns: 8)
        loop.decisionJudge = judge
        let result = await loop.run("北京今天的天气怎么样")

        XCTAssertEqual(judge.completeCalls.count, 1)
        XCTAssertEqual(model.callCount, 2, "足够时应直接放行：工具轮 + 文本轮")
        XCTAssertTrue(result.text.contains("北京今日晴"),
                      "最终文本应原样交付")
    }

    /// 判定不可用（nil）：静默放行，主流程不受影响。
    func testCompleteGateNilJudgePassesThrough() async {
        let judge = StubJudge(informationSufficientResult: nil)
        let model = ScriptedModel(firstTurn: { Self.events(Self.toolTurnEvents()) })
        let session = Session(budget: ContextBudget(total: 60_000), rawTurns: 50)
        let loop = KernelLoop(runtime: makeRuntime(model: model, session: session),
                              config: makeConfig(),
                              systemPrompt: { "SYS" }, session: session, maxTurns: 8)
        loop.decisionJudge = judge
        let result = await loop.run("北京今天的天气怎么样")

        XCTAssertEqual(judge.completeCalls.count, 1)
        XCTAssertEqual(model.callCount, 2, "nil 判定应放行：工具轮 + 文本轮")
        XCTAssertTrue(result.text.contains("北京今日晴"))
    }

    /// 未装配判定器：行为与未接入决策门前完全一致。
    func testNoJudgeKeepsLegacyBehavior() async {
        let model = ScriptedModel(firstTurn: { Self.events(Self.toolTurnEvents()) })
        let session = Session(budget: ContextBudget(total: 60_000), rawTurns: 50)
        let loop = KernelLoop(runtime: makeRuntime(model: model, session: session),
                              config: makeConfig(),
                              systemPrompt: { "SYS" }, session: session, maxTurns: 8)
        let result = await loop.run("北京今天的天气怎么样")

        XCTAssertEqual(model.callCount, 2)
        XCTAssertTrue(result.text.contains("北京今日晴"))
    }

    // MARK: - plan_step 门

    /// 计划评审发现不必要步骤：注入一次修订提示（同一计划只评一次）。
    func testPlanReviewFlagsUnnecessaryStep() async {
        let judge = StubJudge(stepNecessaryResult: false)
        let model = ScriptedModel(firstTurn: { Self.events([.init(kind: .done, toolCalls: [Self.planCreateCall()])]) })
        let session = Session(budget: ContextBudget(total: 60_000), rawTurns: 50)
        let runtime = makeRuntime(model: model, session: session) { calls in
            if calls.contains(where: {
                $0.name == "update_plan"
                    && $0.arguments["op"]?.stringValue == "create"
            }) {
                Self.installPlan(session: session)
            }
        }
        let loop = KernelLoop(runtime: runtime, config: makeConfig(),
                              systemPrompt: { "SYS" }, session: session, maxTurns: 8)
        loop.decisionJudge = judge
        _ = await loop.run("完成测试任务")

        XCTAssertEqual(judge.stepCalls.count, 2, "两个未完成步骤都应被评审")
        XCTAssertTrue(model.userPrompts.contains {
            $0.contains("决策模型认为计划中以下步骤可能对目标并非必要")
        }, "评审提示应作为 pending 注入下一轮 user 提示")
    }

    /// 步骤均必要：不注入提示，正常推进。
    func testPlanReviewSilentWhenNecessary() async {
        let judge = StubJudge(stepNecessaryResult: true)
        let model = ScriptedModel(firstTurn: { Self.events([.init(kind: .done, toolCalls: [Self.planCreateCall()])]) })
        let session = Session(budget: ContextBudget(total: 60_000), rawTurns: 50)
        let runtime = makeRuntime(model: model, session: session) { calls in
            if calls.contains(where: {
                $0.name == "update_plan"
                    && $0.arguments["op"]?.stringValue == "create"
            }) {
                Self.installPlan(session: session)
            }
        }
        let loop = KernelLoop(runtime: runtime, config: makeConfig(),
                              systemPrompt: { "SYS" }, session: session, maxTurns: 8)
        loop.decisionJudge = judge
        _ = await loop.run("完成测试任务")

        XCTAssertEqual(judge.stepCalls.count, 2)
        XCTAssertFalse(model.userPrompts.contains {
            $0.contains("决策模型认为计划中以下步骤")
        }, "全部必要时不注入修订提示")
    }

    // MARK: - 真机集成（模型文件缺失时自动跳过）

    /// judgeBinary 真机验证：加载仓库内 Meteor 决策模型，断言二元判定方向正确。
    func testJudgeBinaryWithRealModel() async throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // .../Tests/BaseAgentCoreTests/
            .deletingLastPathComponent()  // .../Tests/
            .deletingLastPathComponent()  // repo root
        let modelDir = repoRoot.appendingPathComponent("models/PKU-KCL-Meteor-0.8B-mlx-int4")
        guard FileManager.default.fileExists(
            atPath: modelDir.appendingPathComponent("readout.safetensors").path) else {
            throw XCTSkip("决策模型文件不存在，跳过真机集成测试")
        }
        let scorer = MeteorDecisionScorer(modelDir: modelDir.path)
        // 足够的证据 → sufficient 概率应偏 true。
        let enough = try await scorer.judgeBinary(
            state: "User task: 查询今天北京的天气\n\nCollected tool results (evidence):\n"
                + "北京今日晴，气温 5 到 15 摄氏度，西北风三级，空气质量优。",
            question: "Is the collected evidence sufficient to fully answer the user's task?",
            trueOption: "sufficient", falseOption: "insufficient")
        // 证据只覆盖任务所需 3 项信息中的 1 项 → sufficient 概率应偏 false
        // （对应模拟实验 premature 场景：信息不齐就收尾）。
        let partial = try await scorer.judgeBinary(
            state: "User task: 查询今天北京的天气（需要气温、风力、空气质量三项信息）\n\n"
                + "Collected tool results (evidence):\n"
                + "北京今日晴，气温 5 到 15 摄氏度。",
            question: "Is the collected evidence sufficient to fully answer the user's task?",
            trueOption: "sufficient", falseOption: "insufficient")
        XCTAssertGreaterThan(enough, 0.5, "证据充分时 sufficient 概率应 > 0.5（实测 \(enough)）")
        XCTAssertLessThan(partial, 0.5, "证据缺失时 sufficient 概率应 < 0.5（实测 \(partial)）")
    }
}

private extension KernelDecisionGateTests {
    /// 便捷构造事件数组（保持类型推断一致）。
    static func events(_ list: [LLMStreamEvent]) -> [LLMStreamEvent] { list }
}
