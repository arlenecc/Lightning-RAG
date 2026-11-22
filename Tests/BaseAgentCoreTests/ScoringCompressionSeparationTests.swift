import XCTest
@testable import BaseAgentCore

/// 两套压缩机制必须分开：
/// - 分片打分（score）→ CharacterNgramBM25（query-aware，3~5 句，速度优先）
/// - 整段判定（judgeBinary）→ CoreOpinionExtractor（抛开 query，各 top20 全文核心句）
final class ScoringCompressionSeparationTests: XCTestCase {
    private func scorerSource(_ name: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/BaseAgentCore/RAG/Retrieval/\(name).swift")
        return try String(contentsOf: root, encoding: .utf8)
    }

    func testScorePathStillUsesQueryAwareBM25() throws {
        for name in ["MeteorDecisionScorer", "ValenDecisionScorer"] {
            let source = try scorerSource(name)
            XCTAssertTrue(source.contains("CharacterNgramBM25.coreSentences(query: query"),
                "\(name).score 必须保留 query-aware BM25（3~5 句，分片速度优先）")
            XCTAssertFalse(source.contains("CoreOpinionExtractor.extract(\n                    passage"),
                "\(name).score 不得改用全文核心句提取器")
        }
    }

    func testJudgeBinaryUsesExtractorWithoutQuery() throws {
        for name in ["MeteorDecisionScorer", "ValenDecisionScorer"] {
            let source = try scorerSource(name)
            XCTAssertTrue(source.contains("CoreOpinionExtractor.prepareForScoring("),
                "\(name).judgeBinary 应使用全文核心句压缩")
            XCTAssertTrue(source.contains("prepareForScoring(\n            state, maxChars: maxStateChars)"),
                "\(name).judgeBinary 不得传 query（抛开问句提全文核心句）")
        }
    }

    /// Config 不再有分片场景专属的哨兵开关。
    func testNoLegacyFullCoreOpinionsConfig() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/BaseAgentCore/Core/Config.swift")
        let source = try String(contentsOf: root, encoding: .utf8)
        XCTAssertFalse(source.contains("decisionUseFullCoreOpinions"))
    }
}
