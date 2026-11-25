import XCTest
import llama
@testable import BaseAgentCore

/// 真实 llama.cpp vocab 上的 GBNF 编译回归：约束解码的 GBNF 必须能被
/// llama_sampler_init_grammar_lazy_patterns 接受，否则 GGUF 后端会退化成
/// 无约束（日志 `failed to parse grammar` / `grammar 编译失败`）。
final class GbnfLlamaCompileTests: XCTestCase {
    func testGeneratedGBNFCompilesWithLlamaCpp() throws {
        let modelPath = "/Volumes/SSD/gguf/Qwen3.5-0.8B-Q4_K_M.gguf"
        guard FileManager.default.fileExists(atPath: modelPath) else { throw XCTSkip("无小模型") }
        LlamaCppBackend.ensureInitialized()
        var params = llama_model_default_params()
        params.n_gpu_layers = 0
        guard let model = llama_model_load_from_file(modelPath, params),
              let vocab = llama_model_get_vocab(model) else {
            XCTFail("模型/vocab 加载失败"); return
        }
        defer { llama_model_free(model) }

        let tools: [JSONValue] = [
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
            .object(["function": .object([
                "name": .string("update_plan"),
                "parameters": .object([
                    "properties": .object([
                        "op": .object(["type": .string("string")]),
                        "goal": .object(["type": .string("string")]),
                    ]),
                    "required": .array([.string("op")]),
                ]),
            ])]),
        ]
        let gbnf = try XCTUnwrap(QwenXMLToolGrammar.makeGBNF(tools: tools))
        let pattern = QwenXMLToolGrammar.lazyTriggerPattern
        let compiled = pattern.withCString { pat in
            var patterns: [UnsafePointer<CChar>?] = [pat]
            return patterns.withUnsafeMutableBufferPointer { buf in
                gbnf.withCString { g in
                    "root".withCString { root in
                        guard let sampler = llama_sampler_init_grammar_lazy_patterns(
                            vocab, g, root, buf.baseAddress, 1, nil, 0) else { return false }
                        llama_sampler_free(sampler)
                        return true
                    }
                }
            }
        }
        XCTAssertTrue(compiled, "GBNF 必须能被 llama.cpp 编译：\(gbnf)")
    }

    /// 规则名不得含下划线（llama.cpp GBNF 的硬限制）。
    func testGeneratedRuleNamesHaveNoUnderscore() throws {
        let tools: [JSONValue] = [.object(["function": .object([
            "name": .string("rag_tools"),
            "parameters": .object([
                "properties": .object(["op": .object(["type": .string("string")])]),
                "required": .array([.string("op")]),
            ]),
        ])])]
        let gbnf = try XCTUnwrap(QwenXMLToolGrammar.makeGBNF(tools: tools))
        for line in gbnf.split(separator: "\n") where line.contains("::=") {
            let name = line.split(separator: ":", maxSplits: 1).first ?? ""
            XCTAssertFalse(name.contains("_"), "规则名不能含下划线：\(name)")
        }
    }
}
