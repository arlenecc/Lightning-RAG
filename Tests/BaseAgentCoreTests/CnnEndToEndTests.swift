import XCTest
import Foundation
@testable import BaseAgentCore

/// 端到端真机回归：从 cnn.com 获取今天的新闻并进行总结。
///
/// 验证「计划全链路」：
/// - 多步任务必须进入计划面板（模型自建或内核代建，`onPlanUpdated` 至少一次）；
/// - 工具（web_scan）必须被真实执行；
/// - 任务必须推进到有界收尾：要么给出总结，要么给出明确失败说明 +
///   已取得进展汇总，绝不「没执行就判定结束」或空手退出。
final class CnnEndToEndTests: XCTestCase {

    private final class LogCollector: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var lines: [String] = []
        func append(_ line: String) {
            lock.lock(); lines.append(line); lock.unlock()
        }
        func contains(_ fragment: String) -> Bool {
            lock.lock(); defer { lock.unlock() }
            return lines.contains { $0.contains(fragment) }
        }
    }

    private final class PlanCollector: @unchecked Sendable {
        private let lock = NSLock()
        private var _plans: [Plan] = []
        func append(_ plan: Plan) { lock.lock(); _plans.append(plan); lock.unlock() }
        func snapshot() -> [Plan] { lock.lock(); defer { lock.unlock() }; return _plans }
    }

    private final class Callbacks: AgentCallbacks {
        let logs: LogCollector
        let plans: PlanCollector
        init(logs: LogCollector, plans: PlanCollector) {
            self.logs = logs
            self.plans = plans
        }
        func onLog(_ line: String) { logs.append(line) }
        func onPlanUpdated(_ plan: Plan?) {
            if let plan { plans.append(plan) }
        }
    }

    private func modelDirectory() throws -> URL {
        let path = ProcessInfo.processInfo.environment["MLX_KV_TEST_MODEL_DIR"]
            ?? "models/Qwen3.5-2B-MLX-4bit"
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(
            atPath: url.appendingPathComponent("model.safetensors").path) else {
            throw XCTSkip("本地缺少 MLX 测试模型：\(path)")
        }
        return url
    }

    private func log(_ text: String) {
        FileHandle.standardError.write(Data((text + "\n").utf8))
    }

    func testPlanDrivenNewsTaskEndToEnd() async throws {
        let dir = try modelDirectory()
        let service = MLXLLMService(
            directory: dir, modelName: dir.lastPathComponent,
            maxTokens: 1024, topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false, contextWindow: 32_768)
        defer { service.close() }

        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory() + "/cnn-e2e-\(UUID().uuidString)"
        config.maxIterations = 8
        let logs = LogCollector()
        let plans = PlanCollector()
        let agent = Agent(llm: service, config: config,
                          callbacks: Callbacks(logs: logs, plans: plans))
        defer { agent.close() }

        let start = Date()
        let answer = await agent.run("从 cnn.com 获取今天的新闻并进行总结")
        let elapsed = Int(Date().timeIntervalSince(start))

        log("[e2e] elapsed=\(elapsed)s plans=\(plans.snapshot().count) "
            + "textChars=\(answer.count)")
        for plan in plans.snapshot() {
            log("[e2e] plan: goal=\(plan.goal.prefix(40)) "
                + "steps=" + plan.steps.map { "\($0.id):\($0.status)" }
                    .joined(separator: ","))
        }
        log("[e2e] answer=" + String(answer.prefix(600))
            .replacingOccurrences(of: "\n", with: "⏎"))

        // 1) 多步任务必须进计划面板。
        XCTAssertFalse(plans.snapshot().isEmpty,
            "多步任务没有进计划面板（模型未建计划且内核未代建）")
        // 2) 工具必须被真实执行（web_scan 是首选途径；code_run 爬虫也是
        //    合法替代，但 2B 模型应优先用内置 web_scan——schema 已强化引导）。
        XCTAssertTrue(logs.lines.contains { $0.contains("[tool] ") },
            "没有任何工具被执行")
        // 3) 有界收尾：要么答复，要么明确失败说明；不得空手退出，
        //    也不得以「达到轮次上限」模板句草草结束。
        XCTAssertFalse(answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "最终答复为空")
        XCTAssertFalse(answer.contains("已达到本轮最大轮次上限"),
            "任务以轮次上限模板句结束：\(String(answer.prefix(200)))")
        // 4) 计划收尾状态：done 或 blocked 都算「知道结局」；不能停在 doing。
        if let final = plans.snapshot().last {
            XCTAssertFalse(final.steps.contains { $0.status == "doing" },
                "任务结束后计划仍有进行中步骤：\(final.steps)")
        }
    }
}
