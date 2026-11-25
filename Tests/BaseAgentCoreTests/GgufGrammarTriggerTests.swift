import XCTest
import llama
@testable import BaseAgentCore

/// GGUF 约束解码的 lazy trigger 回归：必须用 trigger token，而不是 regex。
/// llama.cpp 的 regex trigger 会回放 trigger_buffer，把 token id 与截断 piece
/// 一起喂给 grammar，可能抛 C++ 异常并在无 catch 边界处 abort。
final class GgufGrammarTriggerTests: XCTestCase {
    private func smallModelPath() throws -> String? {
        let p = "/Volumes/SSD/gguf/Qwen3.5-0.8B-Q4_K_M.gguf"
        return FileManager.default.fileExists(atPath: p) ? p : nil
    }

    private func makeTools() -> [JSONValue] {
        [
            .object(["function": .object([
                "name": .string("rag_tools"),
                "parameters": .object([
                    "properties": .object([
                        "op": .object(["type": .string("string")]),
                        "query": .object(["type": .string("string")]),
                    ]),
                    "required": .array([.string("op")]),
                ]),
            ])]),
        ]
    }

    func testTriggerStringIsSingleToken() throws {
        guard let path = try smallModelPath() else { throw XCTSkip("无小模型") }
        let model = try LlamaCppModel(path: path, gpuLayers: 0)
        let tokens = try model.tokenize(QwenXMLToolGrammar.trigger, addSpecial: false)
        XCTAssertEqual(tokens.count, 1, "Qwen GGUF 的 <tool_call> 应为单 token")
    }

    func testTriggerTokenGrammarAcceptsTrigger() throws {
        guard let path = try smallModelPath() else { throw XCTSkip("无小模型") }
        let model = try LlamaCppModel(path: path, gpuLayers: 0)
        let trigger = try model.tokenize(QwenXMLToolGrammar.trigger, addSpecial: false)
        XCTAssertFalse(trigger.isEmpty)
        let grammar = try XCTUnwrap(QwenXMLToolGrammar.makeGBNF(tools: makeTools()))
        let config = LlamaCppSampling(
            temperature: 0, topK: 0, topP: 1, minP: 0, repetitionPenalty: 1,
            grammar: grammar, triggerTokens: trigger, triggerPatterns: [])
        let sampler = LlamaCppSampler(vocab: model.vocab, vocabSize: model.vocabSize,
                                      config: config)
        sampler.accept(trigger[0])
    }
}
