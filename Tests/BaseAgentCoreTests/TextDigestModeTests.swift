import XCTest
@testable import BaseAgentCore

/// RAG 浓缩 / 章节全书摘要的默认方式与配置切换。
final class TextDigestModeTests: XCTestCase {
    func testDefaultsToCoreSentences() {
        let config = AgentConfig()
        XCTAssertEqual(config.ragCondenseMode, .coreSentences)
        XCTAssertEqual(config.bookSummaryMode, .coreSentences)

        let exported = config.export() as? [String: Any]
        XCTAssertEqual(exported?["rag_condense_mode"] as? String, "core_sentences")
        XCTAssertEqual(exported?["book_summary_mode"] as? String, "core_sentences")
    }

    func testJSONCanSwitchToAuxModel() throws {
        let path = NSTemporaryDirectory() + "agent-config-\(UUID().uuidString).json"
        defer { try? FileManager.default.removeItem(atPath: path) }
        try #"{"rag_condense_mode":"aux_model","book_summary_mode":"aux_model"}"#
            .write(toFile: path, atomically: true, encoding: .utf8)

        let config = AgentConfig.load(path)
        XCTAssertEqual(config.ragCondenseMode, .auxModel)
        XCTAssertEqual(config.bookSummaryMode, .auxModel)
    }

    func testUnknownValueFallsBackToCoreSentences() throws {
        let path = NSTemporaryDirectory() + "agent-config-\(UUID().uuidString).json"
        defer { try? FileManager.default.removeItem(atPath: path) }
        try #"{"rag_condense_mode":"unknown","book_summary_mode":"unknown"}"#
            .write(toFile: path, atomically: true, encoding: .utf8)

        let config = AgentConfig.load(path)
        XCTAssertEqual(config.ragCondenseMode, .coreSentences)
        XCTAssertEqual(config.bookSummaryMode, .coreSentences)
    }
}
