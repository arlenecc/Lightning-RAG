import XCTest
@testable import BaseAgentCore

final class ScheduledTaskTests: XCTestCase {

    private func temporaryDirectory() -> String {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("scheduled-task-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.path
    }

    private func storePath() -> String {
        (temporaryDirectory() as NSString).appendingPathComponent("schedules.json")
    }

    // MARK: 规则

    func testDailyRuleRollsOverToNextDay() {
        let calendar = Calendar(identifier: .gregorian)
        var components = DateComponents()
        components.year = 2026
        components.month = 11
        components.day = 13
        components.hour = 8
        components.minute = 0
        guard let base = calendar.date(from: components) else {
            return XCTFail("无法构造基准时间")
        }

        let rule = ScheduleRule.daily(hour: 9, minute: 30)
        let today = rule.nextFireDate(after: base, calendar: calendar)
        XCTAssertEqual(calendar.component(.day, from: today), 13)
        XCTAssertEqual(calendar.component(.hour, from: today), 9)
        XCTAssertEqual(calendar.component(.minute, from: today), 30)

        let tomorrow = rule.nextFireDate(after: today, calendar: calendar)
        XCTAssertEqual(calendar.component(.day, from: tomorrow), 14)
        XCTAssertEqual(calendar.component(.hour, from: tomorrow), 9)
        XCTAssertEqual(calendar.component(.minute, from: tomorrow), 30)
    }

    func testIntervalRuleAndValidity() {
        let base = Date(timeIntervalSince1970: 1_700_000_000)
        let rule = ScheduleRule.everyMinutes(15)
        XCTAssertTrue(rule.isValid)
        XCTAssertEqual(rule.nextFireDate(after: base).timeIntervalSince(base),
                       900, accuracy: 0.001)
        XCTAssertFalse(ScheduleRule(kind: .everyMinutes, minutes: 0).isValid)
        XCTAssertFalse(ScheduleRule.daily(hour: 24, minute: 0).isValid)
    }

    func testDailyRuleCrossesMonthBoundary() {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        var components = DateComponents()
        components.year = 2026
        components.month = 12
        components.day = 31
        components.hour = 23
        components.minute = 0
        guard let base = calendar.date(from: components) else {
            return XCTFail("无法构造基准时间")
        }
        let next = ScheduleRule.daily(hour: 9, minute: 0)
            .nextFireDate(after: base, calendar: calendar)
        XCTAssertEqual(calendar.component(.year, from: next), 2027)
        XCTAssertEqual(calendar.component(.month, from: next), 1)
        XCTAssertEqual(calendar.component(.day, from: next), 1)
        XCTAssertEqual(calendar.component(.hour, from: next), 9)
    }

    // MARK: 存储

    func testStoreRoundTripAndPersistence() throws {
        let path = storePath()
        let store = ScheduleStore(path: path)
        let created = try store.create(title: "每日摘要",
                                       prompt: "总结今天的日志",
                                       rule: .daily(hour: 9, minute: 0))
        XCTAssertEqual(store.all().count, 1)
        XCTAssertTrue(FileManager.default.fileExists(atPath: path))

        let reloaded = ScheduleStore(path: path)
        XCTAssertEqual(reloaded.all().first?.id, created.id)
        XCTAssertEqual(reloaded.all().first?.rule.summary, "每天 09:00")

        let stamp = Date(timeIntervalSince1970: 1_700_000_000)
        try reloaded.markRun(id: created.id, at: stamp)
        try reloaded.setLastResult(id: created.id, summary: "完成")
        _ = try reloaded.setEnabled(id: created.id, false)

        let again = ScheduleStore(path: path)
        XCTAssertEqual(again.all().first?.lastRunAt?.timeIntervalSince1970 ?? 0,
                       stamp.timeIntervalSince1970, accuracy: 1)
        XCTAssertEqual(again.all().first?.lastResult, "完成")
        XCTAssertEqual(again.all().first?.enabled, false)

        _ = try again.remove(id: created.id)
        XCTAssertTrue(again.all().isEmpty)
        XCTAssertTrue(ScheduleStore(path: path).all().isEmpty)
    }

    func testStoreToleratesMissingOptionalFields() throws {
        let path = storePath()
        let json = """
        [{"id":"abc","title":"t","prompt":"p",
          "rule":{"kind":"daily","hour":9,"minute":0},
          "createdAt":"2026-11-13T00:00:00Z"}]
        """
        try json.write(toFile: path, atomically: true, encoding: .utf8)
        let store = ScheduleStore(path: path)
        XCTAssertEqual(store.all().count, 1)
        XCTAssertEqual(store.all().first?.enabled, true, "缺省应视为启用")
        XCTAssertNil(store.all().first?.lastRunAt)
    }

    // MARK: 调度器

    func testRunnerFiresOverdueTaskExactlyOnce() throws {
        let store = ScheduleStore(path: storePath())
        let created = try store.create(title: "tick", prompt: "ping", rule: .everyMinutes(1))
        // 把计时锚点挪到 2 分钟前，让任务立即到期。
        try store.markRun(id: created.id, at: Date().addingTimeInterval(-120))

        let runner = ScheduleRunner(store: store)
        runner.tickInterval = 0.05
        let fired = expectation(description: "任务触发")
        fired.assertForOverFulfill = true
        runner.onTaskDue = { task in
            XCTAssertEqual(task.id, created.id)
            fired.fulfill()
        }

        runner.start()
        wait(for: [fired], timeout: 3)
        // 触发时已先落盘 lastRunAt，后续 tick 不应重复触发。
        Thread.sleep(forTimeInterval: 0.2)
        runner.stop()

        let reloaded = ScheduleStore(path: store.filePath)
        XCTAssertEqual(reloaded.all().first?.lastRunAt?.timeIntervalSinceNow ?? -999,
                       0, accuracy: 5)
    }

    func testRunnerIgnoresDisabledTask() throws {
        let store = ScheduleStore(path: storePath())
        let created = try store.create(title: "disabled", prompt: "ping", rule: .everyMinutes(1))
        try store.markRun(id: created.id, at: Date().addingTimeInterval(-120))
        _ = try store.setEnabled(id: created.id, false)

        let runner = ScheduleRunner(store: store)
        runner.tickInterval = 0.05
        let notFired = expectation(description: "停用任务不触发")
        notFired.isInverted = true
        runner.onTaskDue = { _ in notFired.fulfill() }
        runner.start()
        wait(for: [notFired], timeout: 0.5)
        runner.stop()
    }

    func testTriggerNowFiresImmediatelyAndRecordsRun() throws {
        let store = ScheduleStore(path: storePath())
        let created = try store.create(title: "手动", prompt: "ping",
                                       rule: .daily(hour: 23, minute: 59))
        let runner = ScheduleRunner(store: store)
        let fired = expectation(description: "立即执行")
        fired.assertForOverFulfill = true
        runner.onTaskDue = { task in
            XCTAssertEqual(task.id, created.id)
            fired.fulfill()
        }
        runner.triggerNow(created)
        wait(for: [fired], timeout: 1)
        XCTAssertNotNil(store.all().first?.lastRunAt,
                        "立即执行后应更新运行时间，避免下一 tick 重复触发")
        XCTAssertEqual(store.all().count, 1)
    }

    func testStoreNotifiesChanges() throws {
        let store = ScheduleStore(path: storePath())
        let notified = expectation(description: "变更通知")
        notified.expectedFulfillmentCount = 2
        notified.assertForOverFulfill = true
        store.onChange = { _ in notified.fulfill() }
        let task = try store.create(title: "a", prompt: "b", rule: .everyMinutes(5))
        try store.markRun(id: task.id, at: Date())
        wait(for: [notified], timeout: 1)
    }

    // MARK: 工具

    func testToolCreateListDisableDelete() async throws {
        var config = AgentConfig()
        config.workspace = temporaryDirectory()
        let path = (config.workspace as NSString)
            .appendingPathComponent(".agent/schedules.json")
        let store = ScheduleStore(path: path)
        let runner = ScheduleRunner(store: store)
        let tool = ScheduleTaskTool(config: config, callbacks: NoopCallbacks(),
                                    store: store, runner: runner)

        let created = try await tool.run([
            "op": .string("create"),
            "title": .string("晨报"),
            "prompt": .string("汇总今天的工作日志"),
            "time": .string("09:30"),
        ])
        XCTAssertTrue(created.success, created.error)
        XCTAssertEqual(store.all().count, 1)
        XCTAssertEqual(store.all().first?.rule.summary, "每天 09:30")

        let listed = try await tool.run(["op": .string("list")])
        XCTAssertTrue(listed.success)
        XCTAssertTrue(listed.output.contains("晨报"))

        let fired = expectation(description: "run_now 触发")
        fired.assertForOverFulfill = true
        runner.onTaskDue = { task in
            XCTAssertEqual(task.title, "晨报")
            fired.fulfill()
        }
        let ranNow = try await tool.run([
            "op": .string("run_now"),
            "id": .string("晨报"),
        ])
        XCTAssertTrue(ranNow.success, ranNow.error)
        await fulfillment(of: [fired], timeout: 2)
        XCTAssertEqual(store.all().count, 1)
        XCTAssertNotNil(store.all().first?.lastRunAt)

        let disabled = try await tool.run([
            "op": .string("disable"),
            "id": .string("晨报"),
        ])
        XCTAssertTrue(disabled.success, disabled.error)
        XCTAssertEqual(store.all().first?.enabled, false)

        let deleted = try await tool.run([
            "op": .string("delete"),
            "id": .string("晨报"),
        ])
        XCTAssertTrue(deleted.success, deleted.error)
        XCTAssertTrue(store.all().isEmpty)
    }

    func testToolRejectsInvalidArguments() async throws {
        var config = AgentConfig()
        config.workspace = temporaryDirectory()
        let store = ScheduleStore(path: (config.workspace as NSString)
            .appendingPathComponent(".agent/schedules.json"))
        let tool = ScheduleTaskTool(config: config, callbacks: NoopCallbacks(),
                                    store: store,
                                    runner: ScheduleRunner(store: store))

        let missingPrompt = try await tool.run([
            "op": .string("create"),
            "every_minutes": .int(10),
        ])
        XCTAssertFalse(missingPrompt.success)

        let badTime = try await tool.run([
            "op": .string("create"),
            "prompt": .string("x"),
            "time": .string("25:00"),
        ])
        XCTAssertFalse(badTime.success)

        let bothRules = try await tool.run([
            "op": .string("create"),
            "prompt": .string("x"),
            "time": .string("09:00"),
            "every_minutes": .int(5),
        ])
        XCTAssertFalse(bothRules.success)
        XCTAssertTrue(store.all().isEmpty)
    }

    func testToolParsesTimeWithFullWidthColon() {
        let parsed = ScheduleTaskTool.parseTime("9：05")
        XCTAssertEqual(parsed?.hour, 9)
        XCTAssertEqual(parsed?.minute, 5)
        XCTAssertNil(ScheduleTaskTool.parseTime("09:60"))
    }

    func testToolSchemaAndConfirmationPolicy() {
        let store = ScheduleStore(path: storePath())
        let tool = ScheduleTaskTool(config: AgentConfig(), callbacks: NoopCallbacks(),
                                    store: store, runner: ScheduleRunner(store: store))
        let schema = tool.parameters.objectValue
        XCTAssertEqual(schema?["type"]?.stringValue, "object")
        let properties = schema?["properties"]?.objectValue
        for key in ["op", "title", "prompt", "every_minutes", "time", "id"] {
            XCTAssertNotNil(properties?[key], "缺少参数 \(key)")
        }
        XCTAssertEqual(schema?["required"]?.arrayValue?.compactMap(\.stringValue), ["op"])

        // 只有会改变系统状态的操作用户确认；查看/启停不打扰。
        XCTAssertTrue(tool.shouldConfirm(["op": .string("create")]))
        XCTAssertTrue(tool.shouldConfirm(["op": .string("delete")]))
        XCTAssertTrue(tool.shouldConfirm(["op": .string("run_now")]))
        XCTAssertFalse(tool.shouldConfirm(["op": .string("list")]))
        XCTAssertFalse(tool.shouldConfirm(["op": .string("enable")]))
    }

    // MARK: 与 Agent 生命周期接线

    func testEndToEndCreateThenRunnerFiresAndResultIsListed() async throws {
        var config = AgentConfig()
        config.workspace = temporaryDirectory()
        let store = ScheduleStore(path: (config.workspace as NSString)
            .appendingPathComponent(".agent/schedules.json"))
        let runner = ScheduleRunner(store: store)
        let tool = ScheduleTaskTool(config: config, callbacks: NoopCallbacks(),
                                    store: store, runner: runner)

        let created = try await tool.run([
            "op": .string("create"),
            "prompt": .string("生成今天的摘要"),
            "every_minutes": .int(5),
        ])
        XCTAssertTrue(created.success, created.error)
        guard let task = store.all().first else {
            return XCTFail("任务未落盘")
        }

        // 模拟已经到期（把计时锚点挪到过去）。
        try store.markRun(id: task.id, at: Date().addingTimeInterval(-600))

        let fired = expectation(description: "到期触发")
        fired.assertForOverFulfill = true
        runner.onTaskDue = { due in
            XCTAssertEqual(due.id, task.id)
            fired.fulfill()
        }
        runner.tickInterval = 0.05
        runner.start()
        await fulfillment(of: [fired], timeout: 2)
        runner.stop()

        // App 层执行完成后回写摘要；工具列应能看到。
        try store.setLastResult(id: task.id, summary: "摘要已生成")
        let listed = try await tool.run(["op": .string("list")])
        XCTAssertTrue(listed.output.contains("上次结果：摘要已生成"))
    }

    func testRunIsolatedHidesAndRestoresUserPlan() async {
        var config = AgentConfig()
        config.workspace = temporaryDirectory()
        let llm = PlanProbeLLM()
        let agent = Agent(llm: llm, config: config, callbacks: NoopCallbacks())
        llm.agent = agent

        let plan = Plan(goal: "用户未完成计划",
                        steps: [PlanStep(title: "容易在定时任务里被误续跑的步骤",
                                         status: "doing")])
        agent.session.setPlan(plan)

        _ = await agent.runIsolated("生成今日摘要")

        XCTAssertNil(llm.planDuringCall, "定时任务运行期间不应看到用户旧计划")
        XCTAssertEqual(agent.session.getPlan()?.goal, plan.goal,
                       "运行结束后应恢复用户计划")
        XCTAssertEqual(PlanStore.load(config)?.goal, plan.goal,
                       "落盘计划也应恢复")
        agent.close()
    }

    func testAgentRegistersScheduleToolAndStopsRunnerOnClose() {
        var config = AgentConfig()
        config.workspace = temporaryDirectory()
        let agent = Agent(llm: UnloadedLLMService(), config: config, callbacks: NoopCallbacks())

        XCTAssertNotNil(agent.tools.get("schedule_task"), "工具应注册进 ToolRegistry")
        XCTAssertNotNil(agent.scheduleStore)
        guard let runner = agent.scheduleRunner else {
            agent.close()
            return XCTFail("缺少 ScheduleRunner")
        }
        // 调度器由 App 显式启动，构造 Agent 时不应自动扫描。
        XCTAssertFalse(runner.isRunning)
        runner.start()
        XCTAssertTrue(runner.isRunning)
        agent.close()
        XCTAssertFalse(runner.isRunning, "Agent 关闭时必须停止调度器")
    }
}

/// 记录调用期间 Agent 会话里看到的计划，用于验证 `runIsolated` 的计划隔离。
private final class PlanProbeLLM: LLMService, @unchecked Sendable {
    let identifier = "plan-probe"
    var planDuringCall: Plan?
    weak var agent: Agent?

    func chatStream(messages: [ChatMessage],
                    tools: [JSONValue],
                    temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        planDuringCall = agent?.session.getPlan()
        return AsyncThrowingStream { continuation in
            continuation.yield(LLMStreamEvent(kind: .content, content: "done"))
            continuation.yield(LLMStreamEvent(kind: .done, content: "done"))
            continuation.finish()
        }
    }
}
