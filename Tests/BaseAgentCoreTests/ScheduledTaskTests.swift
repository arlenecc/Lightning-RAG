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

    func testWeeklyRuleNextFire() {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        func date(_ year: Int, _ month: Int, _ day: Int,
                  _ hour: Int, _ minute: Int) -> Date {
            calendar.date(from: DateComponents(year: year, month: month, day: day,
                                               hour: hour, minute: minute))!
        }

        // 2026-11-13 是周五；下一次「周一 09:00」应为 11-16。
        let friday = date(2026, 11, 13, 10, 0)
        XCTAssertEqual(calendar.component(.weekday, from: friday), 6)
        let rule = ScheduleRule.weekly(weekday: 1, hour: 9, minute: 0)
        XCTAssertEqual(rule.nextFireDate(after: friday, calendar: calendar),
                       date(2026, 11, 16, 9, 0))
        // 周一当天尚未到点 → 当天 09:00；已过点 → 下周一。
        XCTAssertEqual(rule.nextFireDate(after: date(2026, 11, 16, 8, 0), calendar: calendar),
                       date(2026, 11, 16, 9, 0))
        XCTAssertEqual(rule.nextFireDate(after: date(2026, 11, 16, 10, 0), calendar: calendar),
                       date(2026, 11, 23, 9, 0))
        XCTAssertEqual(rule.summary, "每周一 09:00")
    }

    func testOneShotRuleDueOnlyUntilRun() {
        let future = Date().addingTimeInterval(3_600)
        var task = ScheduledTask(title: "一次", prompt: "ping", rule: .once(at: future))
        XCTAssertFalse(task.isDue(), "未到时间不应触发")
        XCTAssertFalse(task.isCompleted)

        task.lastRunAt = Date()
        XCTAssertFalse(task.isDue(), "一次性任务执行后不应再次触发")
        XCTAssertTrue(task.isCompleted)
        XCTAssertEqual(task.rule.summary.hasPrefix("一次性 · "), true)
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

    func testOneShotRunnerFiresOnceThenDisables() throws {
        let store = ScheduleStore(path: storePath())
        let created = try store.create(title: "一次性",
                                       prompt: "ping",
                                       rule: .once(at: Date().addingTimeInterval(-1)))
        let runner = ScheduleRunner(store: store)
        runner.tickInterval = 0.03
        let fired = expectation(description: "一次性触发")
        fired.assertForOverFulfill = true
        let counter = NSLock()
        var fireCount = 0
        runner.onTaskDue = { _ in
            counter.lock(); fireCount += 1; counter.unlock()
            fired.fulfill()
        }
        runner.start()
        wait(for: [fired], timeout: 2)
        Thread.sleep(forTimeInterval: 0.2)
        runner.stop()
        counter.lock()
        let finalCount = fireCount
        counter.unlock()
        XCTAssertEqual(finalCount, 1, "一次性任务只能触发一次")

        let reloaded = ScheduleStore(path: store.filePath)
        XCTAssertEqual(reloaded.all().first?.id, created.id)
        XCTAssertEqual(reloaded.all().first?.enabled, false, "执行后应自动停用")
        XCTAssertEqual(reloaded.all().first?.isCompleted, true)
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

    func testToolCreatesWeeklyDateAndDelayRules() async throws {
        var config = AgentConfig()
        config.workspace = temporaryDirectory()
        let store = ScheduleStore(path: (config.workspace as NSString)
            .appendingPathComponent(".agent/schedules.json"))
        let tool = ScheduleTaskTool(config: config, callbacks: NoopCallbacks(),
                                    store: store, runner: ScheduleRunner(store: store))

        let weekly = try await tool.run([
            "op": .string("create"),
            "prompt": .string("生成周报"),
            "weekday": .int(1),
            "time": .string("09:00"),
        ])
        XCTAssertTrue(weekly.success, weekly.error)
        XCTAssertEqual(store.all().last?.rule.kind, .weekly)
        XCTAssertEqual(store.all().last?.rule.summary, "每周一 09:00")

        let delay = try await tool.run([
            "op": .string("create"),
            "prompt": .string("提醒"),
            "in_minutes": .int(30),
        ])
        XCTAssertTrue(delay.success, delay.error)
        XCTAssertEqual(store.all().last?.rule.kind, .once)
        XCTAssertEqual(store.all().last?.rule.at?.timeIntervalSinceNow ?? 0,
                       1_800, accuracy: 10)

        let dated = try await tool.run([
            "op": .string("create"),
            "prompt": .string("指定日期任务"),
            "date": .string("2027-01-05"),
            "time": .string("08:15"),
        ])
        XCTAssertTrue(dated.success, dated.error)
        guard let at = store.all().last?.rule.at else {
            return XCTFail("缺少一次性任务时间")
        }
        let comps = Calendar.current.dateComponents(
            [.year, .month, .day, .hour, .minute], from: at)
        XCTAssertEqual(comps.year, 2027)
        XCTAssertEqual(comps.month, 1)
        XCTAssertEqual(comps.day, 5)
        XCTAssertEqual(comps.hour, 8)
        XCTAssertEqual(comps.minute, 15)
        XCTAssertTrue(dated.output.contains("一次性任务"), "create 输出应说明一次性语义")
    }

    func testToolRejectsInvalidRuleCombinations() async throws {
        var config = AgentConfig()
        config.workspace = temporaryDirectory()
        let store = ScheduleStore(path: (config.workspace as NSString)
            .appendingPathComponent(".agent/schedules.json"))
        let tool = ScheduleTaskTool(config: config, callbacks: NoopCallbacks(),
                                    store: store, runner: ScheduleRunner(store: store))

        let cases: [[String: JSONValue]] = [
            // date 缺 time
            ["op": .string("create"), "prompt": .string("p"),
             "date": .string("2027-01-01")],
            // date 与 weekday 互斥
            ["op": .string("create"), "prompt": .string("p"),
             "date": .string("2027-01-01"), "time": .string("09:00"), "weekday": .int(1)],
            // 一次性延迟与 time 互斥
            ["op": .string("create"), "prompt": .string("p"),
             "time": .string("09:00"), "in_minutes": .int(5)],
            // weekday 越界
            ["op": .string("create"), "prompt": .string("p"),
             "weekday": .int(8), "time": .string("09:00")],
            // 延迟为 0
            ["op": .string("create"), "prompt": .string("p"),
             "in_hours": .int(0)],
        ]
        for args in cases {
            let result = try await tool.run(args)
            XCTAssertFalse(result.success, "应拒绝非法组合：\(args)")
        }
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
        for key in ["op", "title", "prompt", "every_minutes", "time",
                    "weekday", "date", "in_minutes", "in_hours", "id"] {
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
