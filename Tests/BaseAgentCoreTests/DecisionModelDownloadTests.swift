import XCTest
@testable import BaseAgentCore

/// 决策模型自动下载清单与就绪判定回归。
final class DecisionModelDownloadTests: XCTestCase {
    private func makeConfig() -> AgentConfig {
        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory() + "/decision-dl-\(UUID().uuidString)"
        return config
    }

    func testDecisionModelInCatalogWhenGateEnabled() {
        var config = makeConfig()
        config.kernelDecisionGateEnabled = true
        let models = RequiredModelCatalog.models(for: config)
        let decision = models.first { $0.kind == "决策模型" }
        XCTAssertNotNil(decision, "决策门启用时决策模型必须进入下载清单")
        XCTAssertEqual(decision?.repo,
                       "arlenecc/PKU-KCL-Meteor-v1.5-0.8B-mlx-int4")
        XCTAssertTrue(decision?.requiredFileNames.contains("readout.safetensors") ?? false)
    }

    func testDecisionModelAbsentWhenAllDisabled() {
        var config = makeConfig()
        config.kernelDecisionGateEnabled = false
        config.multiZoneUseDecisionJudge = false
        config.ragMode = "vector"
        let models = RequiredModelCatalog.models(for: config)
        XCTAssertFalse(models.contains { $0.kind == "决策模型" })
    }

    func testDecisionModelInCatalogForDecisionRagMode() {
        var config = makeConfig()
        config.kernelDecisionGateEnabled = false
        config.ragMode = "decision"
        XCTAssertTrue(RequiredModelCatalog.models(for: config)
            .contains { $0.kind == "决策模型" })
    }

    func testDirectoryResolvesRelativeAgainstWorkspace() {
        var config = makeConfig()
        config.decisionModelPath = "models/foo"
        let dir = RequiredModelCatalog.decisionModelDirectory(for: config)
        XCTAssertEqual(dir.standardizedFileURL.path,
                       URL(fileURLWithPath: (config.workspace as NSString)
                        .appendingPathComponent("models/foo")).standardizedFileURL.path)
    }

    func testDirectoryAbsoluteUnchanged() {
        var config = makeConfig()
        config.decisionModelPath = "/tmp/absolute-decision"
        XCTAssertEqual(RequiredModelCatalog.decisionModelDirectory(for: config).path,
                       "/tmp/absolute-decision")
    }

    /// 关键回归：backbone 在但 readout 缺失必须判为未完成（触发下载），
    /// 否则正是用户遇到的「load 时报缺少 readout」。
    func testNotCompleteWithoutReadout() throws {
        let config = makeConfig()
        let dir = RequiredModelCatalog.decisionModelDirectory(for: config)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        // 造一个"看起来有 backbone"的目录
        try Data(repeating: 0, count: 8192).write(
            to: dir.appendingPathComponent("model.safetensors"))
        try Data("stub".utf8).write(to: dir.appendingPathComponent("tokenizer.json"))
        var config2 = config
        config2.kernelDecisionGateEnabled = true
        let model = RequiredModelCatalog.models(for: config2).first { $0.kind == "决策模型" }!
        XCTAssertFalse(RequiredModelCatalog.isComplete(model),
                       "缺 readout.safetensors 必须判未完成，触发自动下载")
    }

    func testCompleteWithReadout() throws {
        let config = makeConfig()
        let dir = RequiredModelCatalog.decisionModelDirectory(for: config)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(repeating: 0, count: 8192).write(
            to: dir.appendingPathComponent("model.safetensors"))
        try Data(repeating: 0, count: 8192).write(
            to: dir.appendingPathComponent("readout.safetensors"))
        try Data("stub".utf8).write(to: dir.appendingPathComponent("tokenizer.json"))
        var config2 = config
        config2.kernelDecisionGateEnabled = true
        let model = RequiredModelCatalog.models(for: config2).first { $0.kind == "决策模型" }!
        XCTAssertTrue(RequiredModelCatalog.isComplete(model))
    }
}
