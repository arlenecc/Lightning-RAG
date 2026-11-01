import XCTest
@testable import BaseAgentCore

/// OCR 模型生命周期：任务开始占用，最后一个任务结束时释放权重。
final class OCRModelLifecycleTests: XCTestCase {
    private actor UnloadCounter {
        private var count = 0

        func increment() {
            count += 1
        }

        func value() -> Int {
            count
        }
    }

    /// 单个 OCR 任务结束时应立即释放，不等空闲回收计时。
    func testSingleJobEndReleasesWeights() async {
        let counter = UnloadCounter()
        let lifecycle = OCRModelLifecycle {
            await counter.increment()
        }

        await lifecycle.begin()
        await lifecycle.end()

        let count = await counter.value()
        XCTAssertEqual(count, 1)
    }

    /// 同步与转换等任务重叠时不能提前释放，最后一个任务结束才释放。
    func testOverlappingJobsReleaseAfterLastEnd() async {
        let counter = UnloadCounter()
        let lifecycle = OCRModelLifecycle {
            await counter.increment()
        }

        await lifecycle.begin()
        await lifecycle.begin()
        await lifecycle.end()

        let afterFirstEnd = await counter.value()
        XCTAssertEqual(afterFirstEnd, 0, "仍有 OCR 任务在跑时不能释放权重")

        await lifecycle.end()
        let afterLastEnd = await counter.value()
        XCTAssertEqual(afterLastEnd, 1)
    }

    /// 每次新的按需任务结束后都应再次释放，不能只释放第一次。
    func testNextJobReleasesAgain() async {
        let counter = UnloadCounter()
        let lifecycle = OCRModelLifecycle {
            await counter.increment()
        }

        await lifecycle.begin()
        await lifecycle.end()
        await lifecycle.begin()
        await lifecycle.end()

        let count = await counter.value()
        XCTAssertEqual(count, 2)
    }

    /// 释放权重期间新的 OCR 任务必须等待，不能开始加载模型后又被旧的
    /// 卸载流程清掉。
    func testBeginWaitsForReleaseToFinish() async {
        let gate = ReleaseGate()
        let order = EventRecorder()
        let lifecycle = OCRModelLifecycle {
            await order.record("unload-start")
            await gate.waitInsideUnload()
            await order.record("unload-end")
        }

        await lifecycle.begin()
        let endTask = Task { await lifecycle.end() }
        await gate.waitUntilEntered()

        let beginTask = Task {
            await lifecycle.begin()
            await order.record("begin-done")
        }

        await gate.release()
        _ = await endTask.value
        _ = await beginTask.value

        let events = await order.events()
        XCTAssertEqual(events, ["unload-start", "unload-end", "begin-done"])
    }

    /// 空闲回收计时器触发时，若仍有 OCR 任务在跑，不能卸载正在使用的权重。
    func testIdleUnloadDoesNotReleaseWhileJobActive() async {
        let counter = UnloadCounter()
        let lifecycle = OCRModelLifecycle {
            await counter.increment()
        }

        await lifecycle.begin()
        await lifecycle.unloadIfIdle()

        let afterIdle = await counter.value()
        XCTAssertEqual(afterIdle, 0)

        await lifecycle.end()
        let afterEnd = await counter.value()
        XCTAssertEqual(afterEnd, 1)
    }

    /// 未 begin 的 end 不能把计数减成负数或误触发释放。
    func testUnbalancedEndDoesNotRelease() async {
        let counter = UnloadCounter()
        let lifecycle = OCRModelLifecycle {
            await counter.increment()
        }

        await lifecycle.end()

        let count = await counter.value()
        XCTAssertEqual(count, 0)
    }

    private actor EventRecorder {
        private var values: [String] = []

        func record(_ event: String) {
            values.append(event)
        }

        func events() -> [String] {
            values
        }
    }

    private actor ReleaseGate {
        private var entered = false
        private var released = false
        private var enteredWaiters: [CheckedContinuation<Void, Never>] = []
        private var releaseContinuation: CheckedContinuation<Void, Never>?

        func waitUntilEntered() async {
            if entered { return }
            await withCheckedContinuation { enteredWaiters.append($0) }
        }

        func waitInsideUnload() async {
            entered = true
            let waiters = enteredWaiters
            enteredWaiters.removeAll()
            for waiter in waiters { waiter.resume() }
            if released { return }
            await withCheckedContinuation { releaseContinuation = $0 }
        }

        func release() {
            released = true
            releaseContinuation?.resume()
            releaseContinuation = nil
        }
    }
}
