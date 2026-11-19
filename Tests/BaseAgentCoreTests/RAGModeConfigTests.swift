import XCTest
@testable import BaseAgentCore

final class RAGModeConfigTests: XCTestCase {
    func testDefaultsAreVectorMode() {
        let config = AgentConfig()
        XCTAssertEqual(config.ragMode, "vector")
        XCTAssertFalse(config.isColbertMode)
        XCTAssertFalse(config.isDecisionMode)
        XCTAssertEqual(config.decisionCoarseCandidates, 50)
        XCTAssertTrue(config.decisionUseCoreSentences)
        XCTAssertEqual(config.decisionCoreSentenceCount, 3)
        XCTAssertEqual(config.decisionTopN, 5)
    }

    func testMigrationFromLegacyColbertFlag() {
        var config = AgentConfig()
        config.colbertReaderEnabled = true
        config.ragMode = "" // 模拟旧配置无 rag_mode
        config.migrate()
        XCTAssertEqual(config.ragMode, "colbert")
        XCTAssertTrue(config.isColbertMode)
        XCTAssertTrue(config.colbertReaderEnabled, "兼容字段应与 ragMode 一致")
    }

    func testDecisionModeAndClamping() {
        var config = AgentConfig()
        config.ragMode = "decision"
        config.decisionCoarseCandidates = 7
        config.decisionCoreSentenceCount = 9
        config.decisionTopN = 99
        config.decisionModelBackend = "bogus"
        config.migrate()
        XCTAssertTrue(config.isDecisionMode)
        XCTAssertFalse(config.colbertReaderEnabled)
        XCTAssertEqual(config.decisionCoarseCandidates, 50)
        XCTAssertEqual(config.decisionCoreSentenceCount, 3)
        XCTAssertEqual(config.decisionTopN, 5)
        XCTAssertEqual(config.decisionModelBackend, "mlx-vlm")
    }

    /// 向量/决策共用 index.sqlite 的 chunks，必须共用 manifest；
    /// 极速用 colbert.db，manifest 独立——切换模式才不互相触发全量重建。
    func testManifestGroupingByStorage() {
        var vector = AgentConfig(); vector.ragMode = "vector"
        var decision = AgentConfig(); decision.ragMode = "decision"
        var colbert = AgentConfig(); colbert.ragMode = "colbert"; colbert.colbertReaderEnabled = true
        XCTAssertEqual(RAGEngine.manifestFileName(for: vector),
                       RAGEngine.manifestFileName(for: decision),
                       "向量/决策共用同一 manifest")
        XCTAssertNotEqual(RAGEngine.manifestFileName(for: colbert),
                          RAGEngine.manifestFileName(for: vector),
                          "极速模式 manifest 独立")
        // 两组的 colbertReaderEnabled 必须相反：这是签名分流的前提。
        XCTAssertFalse(vector.colbertReaderEnabled)
        XCTAssertFalse(decision.colbertReaderEnabled)
        XCTAssertTrue(colbert.colbertReaderEnabled)
    }

    func testRRFPayloadWeightClampAndDerivedBM25() {
        var config = AgentConfig()
        XCTAssertEqual(config.ragRRFVectorWeight, 0.7)
        XCTAssertEqual(config.ragRRFBM25Weight, 0.3, accuracy: 1e-9)
        config.ragRRFVectorWeight = 5
        config.migrate()
        XCTAssertEqual(config.ragRRFVectorWeight, 0.7, "非法权重回落默认")
        config.ragRRFVectorWeight = 0.3
        XCTAssertEqual(config.ragRRFBM25Weight, 0.7, accuracy: 1e-9, "BM25 权重派生")
    }

    func testModeRoundtripViaDictionary() throws {
        var config = AgentConfig()
        config.ragMode = "decision"
        config.decisionModelPath = "/tmp/decision-model"
        config.decisionTopN = 10
        let dict = config.export()
        let data = try JSONSerialization.data(withJSONObject: dict)
        let path = NSTemporaryDirectory() + "ragmode-\(UUID().uuidString).json"
        try data.write(to: URL(fileURLWithPath: path))
        defer { try? FileManager.default.removeItem(atPath: path) }
        let restored = AgentConfig.load(path)
        XCTAssertEqual(restored.ragMode, "decision")
        XCTAssertEqual(restored.decisionModelPath, "/tmp/decision-model")
        XCTAssertEqual(restored.decisionTopN, 10)
    }
}
