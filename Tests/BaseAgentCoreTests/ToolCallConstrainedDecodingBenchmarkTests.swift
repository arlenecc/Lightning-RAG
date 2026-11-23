import XCTest
import MLX
import MLXLLM
import MLXLMCommon
import MLXGuidedGeneration
@testable import BaseAgentCore

/// 工具调用约束解码 A/B 基准：同一模型 / 同一提示词，对比普通采样与
/// Qwen XML 约束解码的「工具名 / 参数名 / 可解析率 / 速度」。
///
/// 真机测试，默认跳过。启用：
/// `TOOL_GRAMMAR_BENCH=1 TOOL_GRAMMAR_MODEL=<模型目录> swift test --filter ToolCallConstrainedDecodingBenchmarkTests`
final class ToolCallConstrainedDecodingBenchmarkTests: XCTestCase {
    private static let modelPath = ProcessInfo.processInfo.environment["TOOL_GRAMMAR_MODEL"]
        ?? "/Users/arlenecc/codebase/agent-app/models/Qwen3.5-2B-MLX-4bit"
    private static let maxTokens = Int(
        ProcessInfo.processInfo.environment["TOOL_BENCH_MAX_TOKENS"] ?? "") ?? 200
    private static let enabled = ProcessInfo.processInfo.environment["TOOL_GRAMMAR_BENCH"] == "1"

    private let toolName = "rag_search"
    private let queryKey = "query"

    private var toolSpecs: [[String: any Sendable]] {
        [[
            "type": "function",
            "function": [
                "name": toolName,
                "description": "Search the knowledge base for relevant passages.",
                "parameters": [
                    "type": "object",
                    "properties": [
                        queryKey: ["type": "string", "description": "Search query"],
                    ],
                    "required": [queryKey],
                ],
            ] as [String: any Sendable],
        ] as [String: any Sendable]]
    }

    private var jsonTools: [BaseAgentCore.JSONValue] {
        [BaseAgentCore.JSONValue.object([
            "type": BaseAgentCore.JSONValue.string("function"),
            "function": BaseAgentCore.JSONValue.object([
                "name": BaseAgentCore.JSONValue.string(toolName),
                "description": BaseAgentCore.JSONValue.string("Search the knowledge base."),
                "parameters": BaseAgentCore.JSONValue.object([
                    "type": BaseAgentCore.JSONValue.string("object"),
                    "properties": BaseAgentCore.JSONValue.object([
                        queryKey: BaseAgentCore.JSONValue.object([
                            "type": BaseAgentCore.JSONValue.string("string"),
                        ]),
                    ]),
                    "required": BaseAgentCore.JSONValue.array([
                        BaseAgentCore.JSONValue.string(queryKey),
                    ]),
                ]),
            ]),
        ])]
    }

    private struct Outcome {
        var text = ""
        var tokens = 0
        var seconds = 0.0
    }

    private struct Metrics {
        var callCount = 0
        var nameOK = false
        var keysOK = false
        var hasToolCallTag = false
    }

    private func evaluate(_ outcome: Outcome) -> Metrics {
        let calls = LocalToolCallParser.extract(
            raw: outcome.text, allowedNames: [toolName])
        var keysOK = false
        if let first = calls.first {
            keysOK = Set(first.arguments.keys).isSubset(of: [queryKey])
                && first.arguments[queryKey] != nil
        }
        return Metrics(
            callCount: calls.count,
            nameOK: calls.first?.name == toolName,
            keysOK: keysOK,
            hasToolCallTag: outcome.text.contains(QwenXMLToolGrammar.trigger))
    }

    func testConstrainedVsUnconstrainedToolCall() async throws {
        guard Self.enabled else { throw XCTSkip("设置 TOOL_GRAMMAR_BENCH=1 启用真机基准") }
        let directory = URL(fileURLWithPath: Self.modelPath)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("模型不存在：\(Self.modelPath)")
        }
        let container = try await LLMModelFactory.shared.loadContainer(
            from: directory, using: LocalTokenizerLoader())
        let system = """
        You have access to the rag_search tool. When the user asks for information, \
        you MUST call rag_search with a concise query instead of answering in plain text.
        """
        let promptTokens = try await container.perform { context in
            try context.tokenizer.applyChatTemplate(
                messages: [
                    ["role": "system", "content": system],
                    ["role": "user", "content": "Search for 水的密码 and its author."],
                ],
                tools: toolSpecs,
                additionalContext: ["enable_thinking": false])
        }

        let free = try await generate(container: container, promptTokens: promptTokens,
                                      constrained: false)
        let guided = try await generate(container: container, promptTokens: promptTokens,
                                        constrained: true)
        let freeMetrics = evaluate(free)
        let guidedMetrics = evaluate(guided)

        print("\n===== 工具调用解码方式对比（\(directory.lastPathComponent), maxTokens=\(Self.maxTokens)）=====")
        for (label, outcome, metrics) in [
            ("无约束", free, freeMetrics), ("XGrammar", guided, guidedMetrics),
        ] {
            let tps = outcome.seconds > 0
                ? String(format: "%.1f", Double(outcome.tokens) / outcome.seconds) : "-"
            print("\(label): \(outcome.seconds)s / \(outcome.tokens) tokens / \(tps) tok/s "
                + "| 工具调用 \(metrics.callCount) | 名称正确 \(metrics.nameOK) "
                + "| 参数正确 \(metrics.keysOK) | 含触发串 \(metrics.hasToolCallTag)")
            print("  原文前缀: \(String(outcome.text.prefix(240)).replacingOccurrences(of: "\n", with: "\\n"))")
        }
        print("==========================================================\n")

        // 约束模式下，只要模型输出了触发串，就必须产出可解析且名称合法的工具调用。
        if guidedMetrics.hasToolCallTag {
            XCTAssertGreaterThanOrEqual(guidedMetrics.callCount, 1, "约束模式应解析出工具调用")
            XCTAssertTrue(guidedMetrics.nameOK)
            XCTAssertTrue(guidedMetrics.keysOK)
        }
    }

    private func generate(container: ModelContainer,
                          promptTokens: [Int],
                          constrained: Bool) async throws -> Outcome {
        let params = GenerateParameters(maxTokens: Self.maxTokens, temperature: 0.1)
        var outcome = Outcome()
        let start = Date()
        try await container.perform { context in
            var grammarState: GrammarProcessorState?
            if constrained {
                let grammar = try XCTUnwrap(QwenXMLToolGrammar.make(tools: jsonTools))
                let vocab = TokenizerVocabExtractor.extractForGrammar(from: context.tokenizer)
                let grammarTokenizer = try GrammarTokenizer(
                    vocab: vocab.vocab, vocabType: vocab.vocabType,
                    eosTokenId: Int32(context.tokenizer.eosTokenId ?? -1))
                let constraint = try GrammarConstraint(
                    tokenizer: grammarTokenizer, grammar: grammar, rootRule: "root")
                let state = GrammarProcessorState(
                    constraint: constraint,
                    tokenizer: UnguardedTokenizer(context.tokenizer),
                    vocabSize: vocab.vocab.count,
                    trigger: QwenXMLToolGrammar.trigger)
                grammarState = state
            }
            let components: GenerationComponents
            if let grammarState {
                components = GenerationComponents(logitProcessorFactory: {
                    GrammarLogitProcessor(state: grammarState)
                })
            } else {
                components = GenerationComponents()
            }
            let iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(promptTokens)),
                model: context.model,
                cache: try context.model.newCache(parameters: params),
                state: nil,
                parameters: params,
                components: components)

            func consume(_ token: Int) -> Bool {
                outcome.text += context.tokenizer.decode(tokenIds: [token])
                outcome.tokens += 1
                return outcome.tokens < Self.maxTokens
            }
            var plain = iterator
            while let token = plain.next(), consume(token) {}
        }
        outcome.seconds = Date().timeIntervalSince(start)
        return outcome
    }
}
