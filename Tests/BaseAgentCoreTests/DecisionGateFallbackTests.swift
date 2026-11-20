import XCTest
@testable import BaseAgentCore

/// 决策门降级回归：模型文件缺失/判定失败时静默放行且不刷屏。
final class DecisionGateFallbackTests: XCTestCase {
    func testMissingReadoutReportsUnavailable() {
        let missing = NSTemporaryDirectory() + "/no-such-decision-\(UUID().uuidString)"
        let scorer = MeteorDecisionScorer(modelDir: missing)
        XCTAssertFalse(scorer.isModelAvailable,
                       "缺少 readout.safetensors 必须判定为不可用，避免装配必然失败的 judge")
    }

    func testExistingModelReportsAvailable() {
        let fm = FileManager.default
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("decision-avail-\(UUID().uuidString)")
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }
        // 只放 readout 即可判定可用（backbone 分片/单文件由 load 路径处理）
        try? Data("stub".utf8).write(to: dir.appendingPathComponent("readout.safetensors"))
        let scorer = MeteorDecisionScorer(modelDir: dir.path)
        XCTAssertTrue(scorer.isModelAvailable)
    }

    /// autoDownload=false 时缺模型必须立即失败（不触发网络），保持既有降级行为。
    func testLoadFailsFastWithoutAutoDownload() async {
        let missing = NSTemporaryDirectory() + "/no-such-decision-\(UUID().uuidString)"
        let scorer = MeteorDecisionScorer(modelDir: missing, autoDownload: false)
        do {
            try await scorer.load()
            XCTFail("缺模型应抛错")
        } catch {
            XCTAssertTrue(String(describing: error).contains("readout"),
                          "应报缺 readout，而非触发下载: \(error)")
        }
    }

    func testDefaultDownloadRepo() {
        XCTAssertEqual(MeteorDecisionScorer.defaultDownloadRepo,
                       "arlenecc/PKU-KCL-Meteor-v1.5-0.8B-mlx-int4")
    }

    func testJudgeFailsSilentlyAfterFirstFailure() async {
        let missing = NSTemporaryDirectory() + "/no-such-decision-\(UUID().uuidString)"
        let scorer = MeteorDecisionScorer(modelDir: missing)
        let judge = MeteorKernelJudge(scorer: scorer, timeoutSeconds: 1)
        // 模型缺失：两次判定都必须 nil（静默放行），不得抛错阻塞主流程。
        let first = await judge.informationSufficient(task: "t", evidence: "e")
        let second = await judge.stepNecessary(goal: "g", step: "s")
        XCTAssertNil(first)
        XCTAssertNil(second)
    }
}
