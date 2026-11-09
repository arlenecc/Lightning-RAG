import XCTest
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
@testable import BaseAgentCore

/// Spark-X2.6-4B（config.json 的 `model_type = spark2_5`）自定义架构回归。
///
/// mlx-swift-lm 内置表没有这个架构，必须在 `loadContainer` 之前
/// `SparkModel.register()`；权重键、逐层 RoPE/sliding-window 与 gated
/// attention 由 `Sources/BaseAgentCore/Local/SparkModel.swift` 实现。
/// 真机用例默认指向 `/Volumes/SSD/Spark-X2.6-4B-MLX-4bit`，
/// 可用 `SPARK_X26_MODEL_DIR` 覆盖，缺失时跳过。
final class SparkX26ModelTests: XCTestCase {

    private func modelDirectory() throws -> URL {
        let path = ProcessInfo.processInfo.environment["SPARK_X26_MODEL_DIR"]
            ?? "/Volumes/SSD/Spark-X2.6-4B-MLX-4bit"
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(
            atPath: url.appendingPathComponent("config.json").path),
              FileManager.default.fileExists(
                atPath: url.appendingPathComponent("model.safetensors").path) else {
            throw XCTSkip("本地缺少 Spark-X2.6 模型：\(path)")
        }
        return url
    }

    // MARK: - 配置 / 权重键（无需加载权重）

    func testConfigurationAndWeightKeysMatchSwiftModule() throws {
        let dir = try modelDirectory()
        let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        let cfg = try JSONDecoder().decode(Spark2_5Configuration.self, from: data)

        XCTAssertEqual(cfg.hiddenSize, 2560)
        XCTAssertEqual(cfg.hiddenLayers, 36)
        XCTAssertEqual(cfg.intermediateSize, 10240)
        XCTAssertEqual(cfg.attentionHeads, 16)
        XCTAssertEqual(cfg.kvHeads, 4)
        XCTAssertEqual(cfg.headDim, 256)
        XCTAssertEqual(cfg.vocabularySize, 131_072)
        XCTAssertEqual(cfg.slidingWindow, 512)
        XCTAssertTrue(cfg.tieWordEmbeddings)
        XCTAssertEqual(cfg.layerTypes.count, cfg.hiddenLayers)
        XCTAssertEqual(cfg.ropeParameters["full_attention"]?.partialRotaryFactor ?? 0,
                       0.25, accuracy: 1e-9)
        XCTAssertEqual(cfg.ropeParameters["sliding_attention"]?.partialRotaryFactor ?? 0,
                       1.0, accuracy: 1e-9)
        XCTAssertEqual(cfg.ropeParameters["full_attention"]?.ropeTheta ?? 0,
                       5_000_000, accuracy: 1)

        let indexData = try Data(
            contentsOf: dir.appendingPathComponent("model.safetensors.index.json"))
        let index = try XCTUnwrap(
            JSONSerialization.jsonObject(with: indexData) as? [String: Any])
        let weightMap = try XCTUnwrap(index["weight_map"] as? [String: String])
        // Module 路径必须逐一对上：model.embedding（量化）、每层
        // self_attn.q_k_v_proj / g_proj（未量化）/ out_proj、mlp 三件套、norm。
        let required = [
            "model.embedding.weight", "model.embedding.scales", "model.embedding.biases",
            "model.layers.0.self_attn.q_k_v_proj.weight",
            "model.layers.0.self_attn.q_k_v_proj.scales",
            "model.layers.0.self_attn.g_proj.weight",
            "model.layers.0.self_attn.out_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.norm.weight",
        ]
        for key in required {
            XCTAssertNotNil(weightMap[key], "权重索引缺少 Module 需要的键：\(key)")
        }
        XCTAssertNil(weightMap["model.layers.0.self_attn.g_proj.scales"],
                     "g_proj 在 checkpoint 里未量化，Swift 侧必须是普通 Linear")
        XCTAssertNil(weightMap["lm_head.weight"],
                     "tie_word_embeddings=true，不应有独立 lm_head")
    }

    /// 配置窗口直接填外部卷绝对路径时必须解析到模型目录本身。
    func testExplicitModelPathIsResolvable() throws {
        let dir = try modelDirectory()
        let resolved = try LocalModelRegistry.resolveDirectory(
            kind: .llm, value: dir.path, config: AgentConfig())
        XCTAssertEqual(resolved.standardizedFileURL.path,
                       dir.standardizedFileURL.path)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: resolved.appendingPathComponent("config.json").path))
    }

    // MARK: - chat template 兼容补丁 / GLM47 工具文本兜底

    func testChatTemplateCompatibilityPatch() throws {
        let dir = try modelDirectory()
        let patched = try XCTUnwrap(
            ChatTemplateCompatibility.overrideTemplate(in: dir),
            "Spark-X2.6 模板含 tojson(separators=...)，必须生成补丁模板")
        XCTAssertFalse(patched.contains("separators"))
        XCTAssertTrue(patched.contains("tojson"))
        XCTAssertEqual(
            ChatTemplateCompatibility.strippingSeparators(
                "{{ x | tojson(separators=(',', ':')) }}"),
            "{{ x | tojson }}")
        XCTAssertEqual(
            ChatTemplateCompatibility.strippingSeparators(
                "{{ x | tojson(separators=(', ', ': ')) }}"),
            "{{ x | tojson }}")

        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("spark-template-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        try JSONSerialization.data(withJSONObject: [
            "chat_template": "{{ messages | tojson }}",
        ]).write(to: tmp.appendingPathComponent("tokenizer_config.json"))
        XCTAssertNil(ChatTemplateCompatibility.overrideTemplate(in: tmp),
                     "普通模板不应被覆盖，避免影响其它模型")
    }

    func testGLM4TextToolCallFallback() {
        let allowed: Set<String> = ["get_current_time", "rag_status"]
        let withArgs = "<tool_call>get_current_time<arg_key>timezone</arg_key>"
            + "<arg_value>Asia/Shanghai</arg_value></tool_call>"
        let calls = MLXLLMService.extractTextToolCalls(withArgs, allowedNames: allowed)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.name, "get_current_time")
        XCTAssertEqual(calls.first?.arguments["timezone"]?.stringValue,
                       "Asia/Shanghai")

        // 库的 GLM4ToolCallParser 要求至少一个 <arg_key>，零参数工具会被拒；
        // 应用侧文本兜底必须能接住。
        let zero = MLXLLMService.extractTextToolCalls(
            "<tool_call>rag_status</tool_call>", allowedNames: allowed)
        XCTAssertEqual(zero.count, 1)
        XCTAssertEqual(zero.first?.name, "rag_status")
        XCTAssertTrue(zero.first?.arguments.isEmpty ?? false)

        XCTAssertTrue(MLXLLMService.extractTextToolCalls(
            "<tool_call>evil</tool_call>", allowedNames: allowed).isEmpty,
            "未声明的工具名不能凭空触发调用")
        XCTAssertEqual(MLXLLMService.extractTextToolCalls(
            "<tool_call>{\"name\":\"rag_status\",\"arguments\":{}}</tool_call>",
            allowedNames: allowed).count, 1, "JSON 方言仍走原路径")
    }

    // MARK: - 注册 + tiny 前向（验证整条自定义计算链，不加载 2.3GB 权重）

    func testRegisteredArchitectureRunsTinyForward() async throws {
        let dir = try modelDirectory()
        _ = dir
        await SparkModel.register()
        let registered = await LLMTypeRegistry.shared.contains("spark2_5")
        XCTAssertTrue(registered, "spark2_5 必须注册进 LLMTypeRegistry")

        let tiny: [String: Any] = [
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "rms_norm_eps": 1e-6,
            "vocab_size": 128,
            "tie_word_embeddings": true,
            "sliding_window": 8,
            "layer_types": ["sliding_attention", "full_attention"],
            "rope_parameters": [
                "full_attention": ["partial_rotary_factor": 0.25, "rope_theta": 500_000],
                "sliding_attention": ["partial_rotary_factor": 1.0, "rope_theta": 10_000],
            ],
            "max_position_embeddings": 128,
        ]
        let data = try JSONSerialization.data(withJSONObject: tiny)
        let cfg = try JSONDecoder().decode(Spark2_5Configuration.self, from: data)
        let model = SparkModel(cfg)
        let caches = try model.newCache(parameters: nil)
        XCTAssertEqual(caches.count, 2, "每层一个 KV cache（sliding + full）")

        let ids = MLXArray([1, 2, 3, 4] as [Int32]).reshaped(1, 4)
        let logits = model(ids, cache: caches)
        XCTAssertEqual(logits.shape, [1, 4, 128],
                       "tied embedding 直接产出词表 logits：\(logits.shape)")
        let head = logits.asType(.float32)[0, 0, 0 ..< 8].asArray(Float.self)
        XCTAssertEqual(head.count, 8)
        XCTAssertTrue(head.allSatisfy { $0.isFinite },
                      "gated attention / 双 RoPE / GEGLU 计算出现非有限值：\(head)")
    }

    // MARK: - 真机：加载 4bit 权重并生成

    func testRealModelLoadsAndGenerates() async throws {
        let dir = try modelDirectory()
        let service = MLXLLMService(
            directory: dir, modelName: dir.lastPathComponent,
            maxTokens: 48, topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false, contextWindow: 4096)
        defer { service.close() }

        var text = ""
        var sawDone = false
        let stream = service.chatStream(
            messages: [
                ChatMessage(role: .system, content: "You are a helpful assistant."),
                ChatMessage(role: .user, content: "Reply with exactly: SPARK_X26_OK"),
            ],
            tools: [], temperature: 0)
        for try await event in stream {
            switch event.kind {
            case .content: text += event.content
            case .done: sawDone = true
            case .reasoning: break
            }
        }
        FileHandle.standardError.write(
            Data(("[spark-x26] 输出: " + text + "\n").utf8))
        XCTAssertTrue(sawDone, "流必须正常结束而不是抛错")
        XCTAssertFalse(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                       "模型必须产出可见正文")
    }

    // MARK: - 真机：混合 sliding/full KV cache 的跨轮对话

    func testRealModelHandlesTwoTurnConversation() async throws {
        let dir = try modelDirectory()
        let service = MLXLLMService(
            directory: dir, modelName: dir.lastPathComponent,
            maxTokens: 32, topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false, contextWindow: 4096)
        defer { service.close() }

        func generate(_ messages: [ChatMessage]) async throws -> String {
            var text = ""
            for try await event in service.chatStream(
                messages: messages, tools: [], temperature: 0) {
                if case .content = event.kind { text += event.content }
            }
            return text
        }
        let system = ChatMessage(role: .system, content: "You are a helpful assistant.")
        let first = try await generate([
            system, ChatMessage(role: .user, content: "Say A."),
        ])
        let second = try await generate([
            system,
            ChatMessage(role: .user, content: "Say A."),
            ChatMessage(role: .assistant, content: "A"),
            ChatMessage(role: .user, content: "Say B."),
        ])
        XCTAssertFalse(first.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        XCTAssertFalse(second.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                       "混合 sliding/full KV cache 的第二轮必须仍能生成")
    }

    // MARK: - 真机：工具调用（Spark 模板是 GLM47 的 <tool_call>/<arg_key> 格式）

    func testRealModelToolCallParsing() async throws {
        let dir = try modelDirectory()
        let service = MLXLLMService(
            directory: dir, modelName: dir.lastPathComponent,
            maxTokens: 256, topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: true, contextWindow: 4096)
        defer { service.close() }
        let tool: BaseAgentCore.JSONValue = .object([
            "type": .string("function"),
            "function": .object([
                "name": .string("get_current_time"),
                "description": .string("Return the current local time for a timezone."),
                "parameters": .object([
                    "type": .string("object"),
                    "properties": .object([
                        "timezone": .object([
                            "type": .string("string"),
                            "description": .string("IANA timezone name"),
                        ]),
                    ]),
                    "required": .array([.string("timezone")]),
                ]),
            ]),
        ])
        var text = ""
        var reasoning = ""
        var raw = ""
        var truncated = false
        var calls: [BaseAgentCore.ToolCall] = []
        for try await event in service.chatStream(
            messages: [
                ChatMessage(role: .system,
                            content: "You are a helpful assistant. Always use the provided tool."),
                ChatMessage(role: .user,
                            content: "Call get_current_time with timezone Asia/Shanghai. Output only the tool call."),
            ],
            tools: [tool], temperature: 0) {
            switch event.kind {
            case .content: text += event.content
            case .reasoning: reasoning += event.content
            case .done:
                calls = event.toolCalls
                raw = event.rawContent
                truncated = event.truncated
            }
        }
        FileHandle.standardError.write(Data(
            ("[spark-x26][tool] thinking=\(reasoning)\ncontent=\(text)\n"
             + "raw=\(raw)\ntruncated=\(truncated)\ncalls=\(calls)\n").utf8))
        XCTAssertFalse(calls.isEmpty, "GLM47 工具调用必须被解析成结构化 ToolCall")
        XCTAssertEqual(calls.first?.name, "get_current_time")
        XCTAssertEqual(calls.first?.arguments["timezone"]?.stringValue,
                       "Asia/Shanghai")
    }

    /// 真机复现「你知识库里有什么」：零参数工具 `rag_status` 曾被上游
    /// GLM4ToolCallParser 误判为 malformedSyntax，导致整轮「工具调用未完成」
    /// 并连续三轮全量重建。恢复后必须是正常的、非截断的工具调用。
    func testZeroArgumentToolCallIsRecovered() async throws {
        let dir = try modelDirectory()
        let service = MLXLLMService(
            directory: dir, modelName: dir.lastPathComponent,
            maxTokens: 256, topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: true, contextWindow: 4096)
        defer { service.close() }
        let tool: BaseAgentCore.JSONValue = .object([
            "type": .string("function"),
            "function": .object([
                "name": .string("rag_status"),
                "description": .string("检查知识库索引状态。"),
                "parameters": .object([
                    "type": .string("object"),
                    "properties": .object([:]),
                ]),
            ]),
        ])
        var calls: [BaseAgentCore.ToolCall] = []
        var truncated = true
        for try await event in service.chatStream(
            messages: [
                ChatMessage(role: .system,
                            content: "You are a helpful assistant. Always use the provided tool."),
                ChatMessage(role: .user,
                            content: "检查知识库状态，只输出 rag_status 工具调用。"),
            ],
            tools: [tool], temperature: 0) {
            if case .done = event.kind {
                calls = event.toolCalls
                truncated = event.truncated
            }
        }
        FileHandle.standardError.write(Data(
            ("[spark-x26][zero-arg] calls=\(calls) truncated=\(truncated)\n").utf8))
        XCTAssertFalse(truncated, "零参数调用恢复后不能再被判为截断")
        XCTAssertFalse(calls.isEmpty, "零参数工具必须恢复成可执行调用")
        XCTAssertEqual(calls.first?.name, "rag_status")
        XCTAssertTrue(calls.first?.arguments.isEmpty ?? false)
    }
}
