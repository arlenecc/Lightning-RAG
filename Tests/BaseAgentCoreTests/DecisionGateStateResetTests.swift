import XCTest
@testable import BaseAgentCore

/// 决策门记账必须按「用户请求」重置：否则一次判定放行后，后续所有请求
/// 永久跳过生成前门/收尾门/计划评审（日志里"多次检索无判定"的根因）。
final class DecisionGateStateResetTests: XCTestCase {
    private func loopSource() throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(contentsOf: root
            .appendingPathComponent("Sources/BaseAgentCore/Kernel/Loop.swift"), encoding: .utf8)
    }

    func testBeginUserRequestResetsDecisionGate() throws {
        let source = try loopSource()
        // beginUserRequest 必须调用 resetDecisionGateState
        let marker = "public func beginUserRequest() {"
        guard let range = source.range(of: marker) else {
            return XCTFail("缺少 beginUserRequest")
        }
        let body = String(source[range.upperBound...].prefix(400))
        XCTAssertTrue(body.contains("resetDecisionGateState()"),
            "beginUserRequest 必须重置决策门状态: \(body)")
        // helper 必须重置 4 个变量
        guard let hr = source.range(of: "private func resetDecisionGateState() {") else {
            return XCTFail("缺少 resetDecisionGateState")
        }
        let helper = String(source[hr.upperBound...].prefix(300))
        for key in ["decisionFinishBlocks", "decisionPrechecked",
                    "decisionPlanReviews", "decisionReviewedPlanKey"] {
            XCTAssertTrue(helper.contains("\(key) = "), "helper 未重置 \(key): \(helper)")
        }
    }

    /// Agent 每条用户消息都调 kernel.beginUserRequest()（源码级确认接线）。
    func testAgentCallsBeginUserRequestPerTurn() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let agent = try String(contentsOf: root
            .appendingPathComponent("Sources/BaseAgentCore/Agent/Agent.swift"), encoding: .utf8)
        XCTAssertTrue(agent.contains("kernel.beginUserRequest()"))
    }
}
