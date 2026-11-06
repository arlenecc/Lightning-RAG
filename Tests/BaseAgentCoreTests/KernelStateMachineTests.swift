import XCTest
@testable import BaseAgentCore

/// P1-c：自动机中断信号不再被吞掉。
final class KernelStateMachineTests: XCTestCase {

    func testNormalToolResultKeepsTheLoopAlive() {
        let outcome = KernelLoop.outcome(data: "ok", result: .ok("done"))
        XCTAssertFalse(outcome.shouldExit)
        XCTAssertEqual(outcome.nextPrompt, "\n")
        XCTAssertFalse(outcome.finished)
    }

    func testInterruptSignalIsPropagatedToTheLoop() {
        let result = ToolResult(success: true, output: "需要用户补充信息",
                               shouldExit: true)
        let outcome = KernelLoop.outcome(data: "需要用户补充信息", result: result)
        XCTAssertTrue(outcome.shouldExit,
                      "toOutcomes 曾恒传 false，中断分支永远不可达")
        XCTAssertNil(outcome.nextPrompt)
        XCTAssertTrue(outcome.finished)
    }
}
