import Foundation
import XCTest
@testable import BaseAgentCore

/// 后台事实抽取的文件游标：4h 节流、按完整 SESSION 块增量、跨天归零。
final class FactExtractionWorkerTests: XCTestCase {

    private var workspace: URL!
    private var dir: MemoryDirectory!
    private var worker: FactExtractionWorker!
    private var now: Date!

    override func setUp() {
        super.setUp()
        workspace = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("fact-worker-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        dir = MemoryDirectory(workspace: workspace.path)
        dir.ensureLayout()
        worker = FactExtractionWorker(directory: dir)
        now = Date()
        addTeardownBlock { [workspace] in
            if let workspace { try? FileManager.default.removeItem(at: workspace) }
        }
    }

    // MARK: - 辅助

    private func sessionBlock(_ id: String, user: String, agent: String) -> String {
        "@@SESSION-BEGIN \(id) | 2026-11-07 10:00:00\n"
        + "@@SYSTEM deadbeef -> 2026-11-07-system.txt\n"
        + "@@USER\n" + user + "\n"
        + "@@TOOL file_read\nsecret tool output\n"
        + "@@AGENT\n" + agent + "\n"
        + "@@SESSION-END \(id) | 2026-11-07 10:00:01\n\n"
    }

    @discardableResult
    private func append(_ text: String, at date: Date? = nil) -> String {
        let day = MemoryDirectory.dayString(date ?? now)
        XCTAssertTrue(dir.append(text, to: dir.rawFile(day: day)))
        return day
    }

    // MARK: - 增量 / 节流

    /// 游标只推进到最后一个完整 SESSION-END；后续追加只读新增部分。
    func testIncrementalCursorOnlyReadsNewSessions() {
        append(sessionBlock("0001", user: "我的编辑器是 vim", agent: "记住了"))
        append(sessionBlock("0002", user: "我的 shell 是 zsh", agent: "好的"))

        let first = worker.takePending(now: now)
        XCTAssertEqual(first?.sessionCount, 2)
        XCTAssertTrue(first?.transcript.contains("我的编辑器是 vim") == true)
        XCTAssertFalse(first?.transcript.contains("secret tool output") == true,
                       "骨架化应剥掉 TOOL 段")
        XCTAssertFalse(first?.transcript.contains("deadbeef") == true,
                       "骨架化应剥掉 SYSTEM 段")
        guard let first else { return }
        worker.commit(first, now: now)

        XCTAssertNil(worker.takePending(now: now), "游标已推进，不应重复读取")
        XCTAssertFalse(worker.isDue(now: now), "4h 内不应再次触发")

        append(sessionBlock("0003", user: "新事实", agent: "收到"))
        let second = worker.takePending(now: now)
        XCTAssertEqual(second?.sessionCount, 1)
        XCTAssertTrue(second?.transcript.contains("新事实") == true)
        XCTAssertFalse(second?.transcript.contains("我的编辑器是 vim") == true,
                       "不得把已消费的旧 session 再喂一遍")
    }

    /// 未闭合的尾部留到下次，避免把半个 session 当成事实。
    func testUnclosedTailIsDeferred() {
        append("@@SESSION-BEGIN 0001 | 2026-11-07 10:00:00\n"
            + "@@USER\n半截内容\n")
        XCTAssertNil(worker.takePending(now: now), "没有 SESSION-END 不应产出批次")

        append("@@AGENT\n补完\n@@SESSION-END 0001 | 2026-11-07 10:00:01\n\n")
        let batch = worker.takePending(now: now)
        XCTAssertEqual(batch?.sessionCount, 1)
        XCTAssertTrue(batch?.transcript.contains("半截内容") == true)
        XCTAssertTrue(batch?.transcript.contains("补完") == true)
    }

    /// 4 小时窗口 + 没有新增内容时不空跑。
    func testFourHourThrottleRequiresNewContent() {
        append(sessionBlock("0001", user: "第一条事实", agent: "ok"))
        XCTAssertTrue(worker.isDue(now: now), "首次有内容即到期")
        guard let batch = worker.takePending(now: now) else { return XCTFail("应有批次") }
        worker.commit(batch, now: now)

        let within = now.addingTimeInterval(3_600)
        append(sessionBlock("0002", user: "一小时后", agent: "ok"))
        XCTAssertFalse(worker.isDue(now: within), "距上次成功不足 4h")

        let later = now.addingTimeInterval(FactExtractionWorker.interval + 1)
        XCTAssertTrue(worker.isDue(now: later), "满 4h 且有新增即可再抽")
        XCTAssertEqual(worker.takePending(now: later)?.sessionCount, 1)

        worker.commit(worker.takePending(now: later)!, now: later)
        XCTAssertFalse(worker.isDue(now: later), "无新增内容时不必空跑")
    }

    /// 被前台取消的那次调度也要计入 4h 节流，但游标不推进，积压不丢。
    func testMarkAttemptBlocksRapidRetries() {
        append(sessionBlock("0001", user: "第一条事实", agent: "ok"))
        XCTAssertTrue(worker.isDue(now: now))
        worker.markAttempt(now: now)          // 模拟后台任务刚启动就被取消
        XCTAssertFalse(worker.isDue(now: now), "4h 内不得立刻重试")

        let later = now.addingTimeInterval(FactExtractionWorker.interval + 1)
        XCTAssertTrue(worker.isDue(now: later), "下个窗口仍会重试")
        XCTAssertEqual(worker.takePending(now: later)?.sessionCount, 1,
                       "游标未推进，积压内容不丢")
    }

    /// 跨天自动把 offset 归零，读取新一天的当天文件。
    func testDayRolloverResetsOffset() {
        append(sessionBlock("0001", user: "今天的事实", agent: "ok"))
        guard let today = worker.takePending(now: now) else { return XCTFail("应有批次") }
        worker.commit(today, now: now)

        let tomorrow = Calendar.current.startOfDay(for: now)
            .addingTimeInterval(25 * 3_600)
        let day2 = append(sessionBlock("0002", user: "明天的事实", agent: "ok"),
                          at: tomorrow)
        XCTAssertNotEqual(day2, today.day)

        let batch = worker.takePending(now: tomorrow)
        XCTAssertEqual(batch?.day, day2)
        XCTAssertEqual(batch?.sessionCount, 1)
        XCTAssertTrue(batch?.transcript.contains("明天的事实") == true)
        XCTAssertTrue(worker.isDue(now: tomorrow), "跨天且满 4h 应可抽取")
    }
}
