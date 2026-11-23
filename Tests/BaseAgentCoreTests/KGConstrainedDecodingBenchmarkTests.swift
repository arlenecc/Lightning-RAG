import XCTest
import MLX
import MLXLLM
import MLXLMCommon
import MLXGuidedGeneration
@testable import BaseAgentCore

/// 知识图谱抽取：无约束解码 vs XGrammar 约束解码 对比基准。
///
/// 同一辅助模型（Qwen3.5-2B）、同一抽取提示词（KG JSON 格式）、同一段真实
/// 书稿切片，分别用普通 TokenIterator 采样和 GuidedGenerationLoop（xgrammar
/// 位掩码）生成，对比：
/// - 速度：墙钟耗时 / 生成 token 数 / tok/s（约束模式掩码计算与 GPU 前向重叠）
/// - 准确性：JSON 解析成功率、实体/关系数量、对人工标注基准的 F1、两种模式实体集重合度
///
/// 需要真机模型（models/Qwen3.5-2B-MLX-4bit）与知识库 Markdown，缺失时跳过。
final class KGConstrainedDecodingBenchmarkTests: XCTestCase {

    private static let modelDir = URL(fileURLWithPath:
        "/Users/arlenecc/codebase/agent-app/models/Qwen3.5-2B-MLX-4bit")
    private static let bookPath = "/Volumes/SSD/app-space/rag_data/test2-c7fc2b3d/"
        + "documents/水的密码.pdf_a01caab2730a.md"
    /// 两种模式共用同一生成预算。900 时两种模式都会截断（关系还没开始输出），
    /// 2000 才能覆盖「20 实体 + 若干关系」的完整 JSON。
    private static let maxTokens = Int(
        ProcessInfo.processInfo.environment["KG_BENCH_MAX_TOKENS"] ?? "") ?? 2_000

    /// 切片 1 的基准（书首元数据段，实体密集）：人工标注。
    private static let truthEntities = [
        "水的密码", "特里斯坦·古利", "许丹", "译林出版社",
        "刘华杰", "电讯报", "坪山自然博物图书奖",
    ]
    private static let truthRelations: [(String, String)] = [
        ("水的密码", "特里斯坦·古利"),
        ("水的密码", "许丹"),
        ("水的密码", "译林出版社"),
    ]

    /// 与 KG JSON 抽取提示词对应的 JSON Schema。
    /// 数组上限与描述长度必须写进 schema：掩码只保证语法，不保证提示词里的
    /// 数量限额；不设 maxItems 时模型会在 entities 数组里一直写下去，
    /// relationships 永远轮不到。
    private static let schema = """
    {"type":"object","properties":{"entities":{"type":"array","maxItems":20,"items":{"type":"object","properties":{"name":{"type":"string","maxLength":60},"type":{"type":"string","maxLength":20},"description":{"type":"string","maxLength":100}},"required":["name","type","description"]}},"relationships":{"type":"array","maxItems":50,"items":{"type":"object","properties":{"source":{"type":"string","maxLength":60},"target":{"type":"string","maxLength":60},"keywords":{"type":"string","maxLength":60},"description":{"type":"string","maxLength":100}},"required":["source","target","keywords","description"]}}},"required":["entities","relationships"]}
    """

    // MARK: - 数据

    private func loadChunks() throws -> [(label: String, text: String)] {
        let md = try String(contentsOfFile: Self.bookPath, encoding: .utf8)
        // 切片 1：书首元数据/内容简介（基准对应段）；切片 2：跳过目录后的第一章正文。
        let rest = md.dropFirst(1_600)
        return [("元数据段", String(md.prefix(800))), ("第一章正文", String(rest.prefix(1_100)))]
    }

    // MARK: - 模型

    private func loadContainer() async throws -> ModelContainer {
        try await LLMModelFactory.shared.loadContainer(
            from: Self.modelDir, using: LocalTokenizerLoader())
    }

    private func renderPromptTokens(container: ModelContainer,
                                    system: String, user: String) async throws -> [Int] {
        try await container.perform { context in
            try context.tokenizer.applyChatTemplate(
                messages: [["role": "system", "content": system],
                           ["role": "user", "content": user]],
                tools: nil,
                additionalContext: ["enable_thinking": false])
        }
    }

    // MARK: - 两种解码模式

    private struct RunOutcome {
        var text: String
        var tokens: Int
        var seconds: Double
        /// 撞上 maxTokens 上限（输出不完整）。
        var truncated = false
        /// 约束模式：建掩码状态机耗时（词表编译 + schema 编译），与生成分开计。
        var setupSeconds: Double = 0
    }

    /// 词表提取 + GrammarTokenizer 构建是「每模型一次」的开销，
    /// 生产里可复用；这里单独计量，不摊进单次生成的耗时。
    private struct GrammarSetup {
        var tokenizer: GrammarTokenizer
        var vocabSize: Int
        var vocabSeconds: Double
        var initSeconds: Double
    }

    private func prepareGrammar(container: ModelContainer) async throws -> GrammarSetup {
        try await container.perform { context in
            let t0 = Date()
            let vocab = TokenizerVocabExtractor.extractForGrammar(from: context.tokenizer)
            let vocabSeconds = Date().timeIntervalSince(t0)
            let t1 = Date()
            let eosId = Int32(context.tokenizer.convertTokenToId("<|im_end|>") ?? -1)
            let tokenizer = try GrammarTokenizer(
                vocab: vocab.vocab, vocabType: vocab.vocabType, eosTokenId: eosId)
            return GrammarSetup(tokenizer: tokenizer, vocabSize: vocab.vocab.count,
                                vocabSeconds: vocabSeconds,
                                initSeconds: Date().timeIntervalSince(t1))
        }
    }

    /// 无约束：与生产路径同款 TokenIterator（temperature 0.1）。
    private func runUnconstrained(container: ModelContainer,
                                  promptTokens: [Int]) async throws -> RunOutcome {
        let params = GenerateParameters(temperature: 0.1)
        let start = Date()
        var out = ""
        var count = 0
        try await container.perform { context in
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(promptTokens)),
                model: context.model,
                cache: try context.model.newCache(parameters: params),
                state: nil,
                parameters: params)
            while let token = iterator.next() {
                count += 1
                out += context.tokenizer.decode(tokenIds: [token])
                if count >= Self.maxTokens { break }
            }
        }
        return RunOutcome(text: out, tokens: count,
                          seconds: Date().timeIntervalSince(start),
                          truncated: count >= Self.maxTokens)
    }

    /// XGrammar 约束：JSON Schema 位掩码，掩码计算与 GPU 前向重叠。
    /// 注意 `GuidedGenerationLoop` 在施加掩码后是贪心 argmax（无温度采样）。
    private func runConstrained(container: ModelContainer,
                                promptTokens: [Int],
                                grammar: GrammarSetup) async throws -> RunOutcome {
        var out = ""
        var count = 0
        var truncated = false
        var setupSeconds = 0.0
        var genSeconds = 0.0
        try await container.perform { context in
            // 每个请求一份独立语法状态机；编译 schema 的耗时单独计。
            let t0 = Date()
            let constraint = try GrammarConstraint(tokenizer: grammar.tokenizer,
                                                   jsonSchema: Self.schema)
            setupSeconds = Date().timeIntervalSince(t0)
            let genStart = Date()
            do {
                count = try GuidedGenerationLoop.run(
                    input: LMInput(tokens: MLXArray(promptTokens)),
                    context: context,
                    constraint: constraint,
                    maxTokens: Self.maxTokens,
                    vocabSize: grammar.vocabSize,
                    emit: { delta in
                        out += delta
                        return true
                    })
            } catch GuidedGenerationError.incompleteOutput {
                // 循环仅在 tokenCount >= maxTokens 时抛此错误。
                truncated = true
                count = Self.maxTokens
            }
            genSeconds = Date().timeIntervalSince(genStart)
        }
        return RunOutcome(text: out, tokens: count, seconds: genSeconds,
                          truncated: truncated, setupSeconds: setupSeconds)
    }

    // MARK: - 指标

    private struct Metrics {
        var parseOK = false
        /// 原始输出能否被标准 JSON 解析器整体接受（截断/多余文本都算失败）。
        var jsonValid = false
        var entityNames: [String] = []
        var relationPairs: [(String, String)] = []
        var entityF1 = 0.0
        var relationF1 = 0.0
    }

    private static func normalized(_ s: String) -> String {
        s.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    private static func entityHit(_ truth: String, in names: [String]) -> Bool {
        let t = normalized(truth)
        return names.contains { n in n.contains(t) || t.contains(n) }
    }

    private static func pairHit(_ truth: (String, String),
                                in pairs: [(String, String)]) -> Bool {
        entityHit(truth.0, in: pairs.map(\.0)) && entityHit(truth.1, in: pairs.map(\.1))
            || entityHit(truth.1, in: pairs.map(\.0)) && entityHit(truth.0, in: pairs.map(\.1))
    }

    private func evaluate(_ outcome: RunOutcome) -> Metrics {
        var m = Metrics()
        let trimmed = outcome.text.trimmingCharacters(in: .whitespacesAndNewlines)
        m.jsonValid = (try? JSONSerialization.jsonObject(
            with: Data(trimmed.utf8))) != nil
        let result = KnowledgeGraphBuilder.parseExtractionJSON(
            outcome.text, limits: .default)
        m.entityNames = result.entities.map { Self.normalized($0.name) }
        m.relationPairs = result.relations.map { (Self.normalized($0.source),
                                                   Self.normalized($0.target)) }
        m.parseOK = !m.entityNames.isEmpty || !m.relationPairs.isEmpty
        // 实体 F1：命中=truePositives，抽取总数=预测数，标注数=7。
        let entityTP = Self.truthEntities.filter { Self.entityHit($0, in: m.entityNames) }.count
        if entityTP > 0 {
            let p = Double(entityTP) / Double(max(1, m.entityNames.count))
            let r = Double(entityTP) / Double(Self.truthEntities.count)
            m.entityF1 = 2 * p * r / (p + r)
        }
        let relTP = Self.truthRelations.filter { Self.pairHit($0, in: m.relationPairs) }.count
        if relTP > 0 {
            let p = Double(relTP) / Double(max(1, m.relationPairs.count))
            let r = Double(relTP) / Double(Self.truthRelations.count)
            m.relationF1 = 2 * p * r / (p + r)
        }
        return m
    }

    // MARK: - 基准主测试

    func testBenchmarkConstrainedVsUnconstrained() async throws {
        guard FileManager.default.fileExists(atPath: Self.modelDir.path),
              FileManager.default.fileExists(atPath: Self.bookPath) else {
            throw XCTSkip("缺少真机模型或知识库 Markdown，跳过基准")
        }
        let limits = KGExtractionLimits.default
        let system = KnowledgeGraphPrompts.entityExtractionJSONSystem(limits: limits)
        let container = try await loadContainer()
        // KG_BENCH_CHUNKS=1/2 只跑指定切片（完整输出的长预算验证用）。
        let chunkFilter = ProcessInfo.processInfo.environment["KG_BENCH_CHUNKS"] ?? "all"
        let chunks = try loadChunks().enumerated().filter { index, _ in
            chunkFilter == "all" || chunkFilter == String(index + 1)
        }.map(\.element)
        let grammar = try await prepareGrammar(container: container)
        print("\n[kg-bench] 词表提取 \(String(format: "%.2f", grammar.vocabSeconds))s，"
            + "GrammarTokenizer 构建 \(String(format: "%.2f", grammar.initSeconds))s，"
            + "词表 \(grammar.vocabSize) token（每模型一次，可复用）")

        struct Row {
            var chunk: String
            var mode: String
            var seconds: Double
            var tokens: Int
            var parseOK: Bool
            var jsonValid: Bool
            var truncated: Bool
            var setupSeconds: Double
            var entities: Int
            var relations: Int
            var entityF1: Double
            var relationF1: Double
        }
        var rows: [Row] = []
        var entitySets: [String: Set<String>] = [:]
        var outputs: [String: String] = [:]

        for chunk in chunks {
            let user = KnowledgeGraphPrompts.entityExtractionJSONTask(
                chunk: chunk.text, sectionContext: "", limits: limits)
            let promptTokens = try await renderPromptTokens(
                container: container, system: system, user: user)

            let free = try await runUnconstrained(container: container,
                                                  promptTokens: promptTokens)
            let freeMetrics = evaluate(free)
            rows.append(Row(chunk: chunk.label, mode: "无约束",
                            seconds: free.seconds, tokens: free.tokens,
                            parseOK: freeMetrics.parseOK,
                            jsonValid: freeMetrics.jsonValid,
                            truncated: free.truncated,
                            setupSeconds: 0,
                            entities: freeMetrics.entityNames.count,
                            relations: freeMetrics.relationPairs.count,
                            entityF1: freeMetrics.entityF1,
                            relationF1: freeMetrics.relationF1))
            entitySets["无约束|\(chunk.label)"] = Set(freeMetrics.entityNames)
            outputs["\(chunk.label)|无约束"] = free.text

            let constrained = try await runConstrained(container: container,
                                                       promptTokens: promptTokens,
                                                       grammar: grammar)
            let consMetrics = evaluate(constrained)
            rows.append(Row(chunk: chunk.label, mode: "XGrammar",
                            seconds: constrained.seconds, tokens: constrained.tokens,
                            parseOK: consMetrics.parseOK,
                            jsonValid: consMetrics.jsonValid,
                            truncated: constrained.truncated,
                            setupSeconds: constrained.setupSeconds,
                            entities: consMetrics.entityNames.count,
                            relations: consMetrics.relationPairs.count,
                            entityF1: consMetrics.entityF1,
                            relationF1: consMetrics.relationF1))
            entitySets["XGrammar|\(chunk.label)"] = Set(consMetrics.entityNames)
            outputs["\(chunk.label)|XGrammar"] = constrained.text
        }

        print("\n===== KG 抽取 解码方式对比（模型：Qwen3.5-2B, maxTokens=\(Self.maxTokens)）=====")
        print("注：无约束=生产路径 temperature 0.1 采样；XGrammar=掩码后 argmax 贪心。")
        print("    耗时为纯生成时间，不含词表提取/GrammarTokenizer 构建；建约束=单次 schema 编译。")
        print(String(format: "%@ | %@ | %@ | %@ | %@ | %@ | %@ | %@ | %@",
                     "切片".padding(toLength: 10, withPad: " ", startingAt: 0),
                     "模式".padding(toLength: 9, withPad: " ", startingAt: 0),
                     "生成s".padding(toLength: 7, withPad: " ", startingAt: 0),
                     "tokens".padding(toLength: 7, withPad: " ", startingAt: 0),
                     "tok/s".padding(toLength: 6, withPad: " ", startingAt: 0),
                     "建约束s".padding(toLength: 7, withPad: " ", startingAt: 0),
                     "解析".padding(toLength: 4, withPad: " ", startingAt: 0),
                     "JSON".padding(toLength: 4, withPad: " ", startingAt: 0),
                     "实体/关系").padding(toLength: 9, withPad: " ", startingAt: 0))
        for r in rows {
            let tps = r.tokens > 0 ? String(format: "%.1f", Double(r.tokens) / r.seconds) : "-"
            let setup = r.mode == "XGrammar" ? String(format: "%.2f", r.setupSeconds) : "-"
            let parse = r.parseOK ? "OK" : "FAIL"
            let json = r.jsonValid ? "OK" : "FAIL"
            let flag = r.truncated ? " [截断]" : ""
            print("\(r.chunk.padding(toLength: 10, withPad: " ", startingAt: 0))"
                + " | \(r.mode.padding(toLength: 9, withPad: " ", startingAt: 0))"
                + " | \(String(format: "%6.1f", r.seconds))"
                + " | \(String(r.tokens).padding(toLength: 6, withPad: " ", startingAt: 0))"
                + " | \(tps.padding(toLength: 6, withPad: " ", startingAt: 0))"
                + " | \(setup.padding(toLength: 7, withPad: " ", startingAt: 0))"
                + " | \(parse.padding(toLength: 4, withPad: " ", startingAt: 0))"
                + " | \(json.padding(toLength: 4, withPad: " ", startingAt: 0))"
                + " | \(r.entities)/\(r.relations)\(flag)")
            print(String(format: "             F1 实体 %.2f / 关系 %.2f",
                         r.entityF1, r.relationF1))
            if let text = outputs["\(r.chunk)|\(r.mode)"] {
                let tail = String(text.suffix(80)).replacingOccurrences(of: "\n", with: " ")
                print("             输出 \(text.count) 字，尾部：…\(tail)")
            }
        }
        print("----- 两种模式实体集重合度（Jaccard）-----")
        for chunk in chunks {
            let a = entitySets["无约束|\(chunk.label)"] ?? []
            let b = entitySets["XGrammar|\(chunk.label)"] ?? []
            let union = a.union(b)
            let jac = union.isEmpty ? 0.0 : Double(a.intersection(b).count) / Double(union.count)
            print("\(chunk.label): 仅无约束 \(a.subtracting(b).count) 个，仅约束 \(b.subtracting(a).count) 个，Jaccard=\(String(format: "%.2f", jac))")
        }
        print("=====================================================================\n")

        // 最低限度的正确性断言：约束模式必须产出可解析且非空的结果（掩码保证 JSON 合法）。
        for r in rows where r.mode == "XGrammar" {
            XCTAssertTrue(r.parseOK, "\(r.chunk) 约束模式应解析出非空实体/关系")
            XCTAssertLessThanOrEqual(r.entities, limits.maxEntitiesPerChunk)
        }
    }
}
