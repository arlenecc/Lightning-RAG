import XCTest
import MLX
import MLXLMCommon
@testable import BaseAgentCore

/// ValenDecisionScorer 真机集成测试：模型目录（外置 SSD）缺失时自动跳过。
///
/// 验收基准（与 Python 侧对齐）：
/// 1. 提示构建：Swift 分片编码拼接的 token 序列与官方 example_decision.build_prompt
///    逐位一致（fixture 由 Python 导出）；
/// 2. 前向对齐：bbox 掩码 + MixerHead 的概率与 Python MLX 复现结果（mlx_report）
///    top-1 一致、偏差 < 0.05；
/// 3. judgeBinary 方向性：信息充足判定 > 0.5，关键信息缺失 < 0.5。
final class ValenDecisionScorerTests: XCTestCase {
    static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    static let fixtureURL = repoRoot
        .appendingPathComponent("Tests/BaseAgentCoreTests/Fixtures/valen_baseline.json")

    private func makeScorer() throws -> ValenDecisionScorer {
        let fixture = try Data(contentsOf: Self.fixtureURL)
        let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: fixture) as? [String: Any])
        let dir = try XCTUnwrap(obj["model_dir"] as? String)
        guard FileManager.default.fileExists(
            atPath: (dir as NSString).appendingPathComponent("decision_head.safetensors")) else {
            throw XCTSkip("Valen 决策模型目录不存在（外置 SSD 未挂载）: \(dir)")
        }
        return ValenDecisionScorer(modelDir: dir)
    }

    // MARK: - 提示构建逐位对齐

    func testBuildPromptMatchesPythonReference() async throws {
        _ = try makeScorer()
        let fixture = try Data(contentsOf: Self.fixtureURL)
        let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: fixture) as? [String: Any])
        let canonical = try XCTUnwrap(obj["canonical"] as? [String: Any])
        let pythonIds = try XCTUnwrap(canonical["ids"] as? [Int])
        let state = try XCTUnwrap(canonical["state"] as? String)
        let instructions = try XCTUnwrap(canonical["instructions"] as? String)
        let keys = try XCTUnwrap(canonical["keys"] as? [String])
        let criteria = try XCTUnwrap(canonical["criteria"] as? [String: String])

        let tokenizer = try await LocalTokenizerLoader().load(
            from: URL(fileURLWithPath: ValenDecisionScorer.defaultModelDir, isDirectory: true))
        let prompt = try ValenDecisionScorer.buildPrompt(
            tokenizer: tokenizer, state: state,
            question: .init(instructions: instructions, keys: keys,
                            descs: keys.map { criteria[$0] ?? "" }))

        XCTAssertEqual(prompt.tokens, pythonIds,
                       "Swift 分片编码应与官方 build_prompt 逐 token 等价")
        XCTAssertEqual(prompt.prefixLen, try XCTUnwrap(canonical["prefix"] as? Int))
        let span = try XCTUnwrap(canonical["instruction_span"] as? [Int])
        XCTAssertEqual([prompt.instructionSpan.0, prompt.instructionSpan.1], span)
        let candSpans = try XCTUnwrap(canonical["candidate_spans"] as? [[Int]])
        XCTAssertEqual(prompt.candidateSpans.map { [$0.0, $0.1] }, candSpans)
        XCTAssertEqual(prompt.decisionPosition,
                       try XCTUnwrap(canonical["decision_position"] as? Int))
    }

    // MARK: - 前向对齐（bbox 掩码 + MixerHead vs Python MLX）

    func testForwardMatchesPythonMLXReference() async throws {
        let scorer = try makeScorer()
        let fixture = try Data(contentsOf: Self.fixtureURL)
        let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: fixture) as? [String: Any])
        let requests = try XCTUnwrap(obj["requests"] as? [[String: Any]])
        try await scorer.load()

        for (ri, req) in requests.enumerated() {
            let ids = try XCTUnwrap(req["ids"] as? [Int])
            let prefixLen = try XCTUnwrap(req["prefix"] as? Int)
            let segments = try XCTUnwrap(req["segments"] as? [Int])
            let questions = try XCTUnwrap(req["questions"] as? [[String: Any]])
            // 官方非 bbox 题后缀共用 block 1 的重映射（verify_mlx.py 同款）。
            let remapped = segments.map { $0 > 0 ? 1 : $0 }
            let mask = ValenDecisionScorer.bboxMask(segments: remapped,
                                                    prefixLen: prefixLen, length: ids.count)
            let (layerCount, faIdx) = ValenDecisionScorer.layerLayout(
                configText: try String(contentsOf: URL(fileURLWithPath:
                    ValenDecisionScorer.defaultModelDir + "/config.json"), encoding: .utf8))

            for q in questions {
                let keys = try XCTUnwrap(q["keys"] as? [String])
                let qid = (q["qid"] as? String) ?? "?"
                let mlxProbs = try XCTUnwrap(q["mlx_probs"] as? [Double])
                let ctx = try XCTUnwrap(q["context_span"] as? [Int])
                let ins = try XCTUnwrap(q["instruction_span"] as? [Int])
                let cands = try XCTUnwrap(q["candidate_spans"] as? [[Int]])
                let dec = try XCTUnwrap(q["decision_position"] as? Int)
                let prompt = ValenDecisionScorer.ValenPrompt(
                    tokens: ids, prefixLen: prefixLen,
                    contextSpan: (ctx[0], ctx[1]),
                    instructionSpan: (ins[0], ins[1]),
                    candidateSpans: cands.map { ($0[0], $0[1]) },
                    decisionPosition: dec)
                let customCache = ValenDecisionScorer.makeCacheArray(
                    prompt: prompt, layerCount: layerCount, faIdx: faIdx, maskOverride: mask)
                let probs = try await scorer.forwardProbs(prompt: prompt, cacheArray: customCache)

                XCTAssertEqual(probs.count, keys.count, "req\(ri) 候选概率数量不符")
                let ourTop = probs.enumerated().max { $0.element < $1.element }!.offset
                let refTop = mlxProbs.enumerated().max { $0.element < $1.element }!.offset
                XCTAssertEqual(ourTop, refTop, "req\(ri) qid=\(qid) top-1 不一致：swift=\(probs) python=\(mlxProbs)")
                let maxDiff = zip(probs, mlxProbs).map { abs($0 - $1) }.max() ?? 1
                XCTAssertLessThan(maxDiff, 0.05,
                                  "req\(ri) qid=\(qid) 概率偏差过大：swift=\(probs) python=\(mlxProbs)")
            }
        }
    }

    // MARK: - judgeBinary 方向性

    func testJudgeBinaryDirectionality() async throws {
        let scorer = try makeScorer()
        let sufficient = """
        Task: 查询北京今天的天气
        Tool results: 北京今日晴，气温 5 到 15 摄氏度，空气质量良，湿度 40%。
        """
        let partial = "Tool results: 北京今日有云。"
        let q = "Is the collected evidence sufficient to fully answer the user's task?"
        let yes = try await scorer.judgeBinary(state: sufficient, question: q)
        let no = try await scorer.judgeBinary(state: partial, question: q)
        XCTAssertGreaterThan(yes, 0.5, "信息充足应判定 true（实际 \(yes)）")
        XCTAssertLessThan(no, 0.5, "关键信息缺失应判定 false（实际 \(no)）")
    }
}
