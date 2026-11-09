import XCTest
import Foundation
import llama
@testable import BaseAgentCore

/// Spark-X2.6 GGUF 适配的「不回归」护栏。
///
/// 两处补丁都作用在**所有** GGUF 模型的公共路径上：
/// - `LlamaCppModel` 加载 chat template 时统一过 `strippingSeparators`；
/// - `LocalToolCallParser` 增加 GLM47 方言兜底（GGUF / ONNX / MLX 共用）。
///
/// 这里证明：非 Spark 模板逐字不变、既有工具方言解析结果不变、
/// 对照 GGUF 模型的元数据模板也未被动过。
final class GGUFCompatibilitySafetyTests: XCTestCase {

    // MARK: - 模板补丁：未命中时必须是恒等变换

    func testTemplatePatchIsNoOpForOtherModels() {
        let templates = [
            // Qwen ChatML
            "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
            // Llama 3
            "{{ bos_token }}{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n' + message['content'] + '<|eot_id|>' }}{% endfor %}",
            // Gemma / 通用 turn 模板
            "{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\\n' + message['content'] + '<end_of_turn>\\n' }}{% endfor %}",
            // 合法 tojson 参数（indent）不能被误删
            "{{ tools | tojson(indent=2) }}",
            // 无参数 tojson
            "{{ tool.function | tojson }}",
            // 普通 JSON 文本
            "{\"name\":\"foo\",\"arguments\":{}}",
            // 含 separators 字样但不是 filter 参数
            "Use separators=(',', ':') when formatting JSON.",
            "",
        ]
        for template in templates {
            XCTAssertEqual(
                ChatTemplateCompatibility.strippingSeparators(template), template,
                "非 Spark 模板必须原样返回：\(template.prefix(100))")
        }

        // 只有 HF 的 separators 语法被改写，且幂等。
        let spark = "{{ tool.function | tojson(separators=(',', ':')) }}"
        let once = ChatTemplateCompatibility.strippingSeparators(spark)
        XCTAssertEqual(once, "{{ tool.function | tojson }}")
        XCTAssertEqual(ChatTemplateCompatibility.strippingSeparators(once), once)
        let spark2 = "{{ v | tojson(separators=(', ', ': ')) }}"
        XCTAssertEqual(ChatTemplateCompatibility.strippingSeparators(spark2),
                       "{{ v | tojson }}")
    }

    // MARK: - 工具解析：既有方言结果不变，新增方言不误伤

    func testExistingToolDialectsUnchanged() {
        let allowed: Set<String> = ["get_weather", "rag_status"]

        let json = #"<tool_call>{"name":"get_weather","arguments":{"city":"Beijing"}}</tool_call>"#
        let jsonCalls = LocalToolCallParser.extract(raw: json, allowedNames: allowed)
        XCTAssertEqual(jsonCalls.count, 1)
        XCTAssertEqual(jsonCalls.first?.name, "get_weather")
        XCTAssertEqual(jsonCalls.first?.arguments["city"]?.stringValue, "Beijing")

        let xml = "<tool_call><function=get_weather>"
            + "<parameter=city>Shanghai</parameter></function></tool_call>"
        let xmlCalls = LocalToolCallParser.extract(raw: xml, allowedNames: allowed)
        XCTAssertEqual(xmlCalls.count, 1)
        XCTAssertEqual(xmlCalls.first?.name, "get_weather")
        XCTAssertEqual(xmlCalls.first?.arguments["city"]?.stringValue, "Shanghai")

        let fenced = "```json\n{\"name\":\"rag_status\",\"arguments\":{}}\n```"
        XCTAssertEqual(LocalToolCallParser.extract(
            raw: fenced, allowedNames: allowed).count, 1)

        // 合法 JSON 但工具名未声明：不能被 GLM47 兜底「救活」。
        let unknown = #"<tool_call>{"name":"evil","arguments":{}}</tool_call>"#
        XCTAssertTrue(LocalToolCallParser.extract(
            raw: unknown, allowedNames: allowed).isEmpty)

        // 白名单词出现在普通说明里，不能被误判成工具调用。
        let prose = "例如调用 <tool_call>rag_status 这个工具可以查询状态</tool_call>"
        XCTAssertTrue(LocalToolCallParser.extract(
            raw: prose, allowedNames: allowed).isEmpty)
        XCTAssertTrue(LocalToolCallParser.extract(
            raw: "这是一段普通回答，没有工具调用。", allowedNames: allowed).isEmpty)

        // GLM47 零参数形式仍然只允许精确的工具名。
        XCTAssertEqual(LocalToolCallParser.extract(
            raw: "<tool_call>rag_status</tool_call>",
            allowedNames: allowed).count, 1)
        XCTAssertEqual(LocalToolCallParser.extract(
            raw: "<tool_call>get_weather<arg_key>city</arg_key>"
                + "<arg_value>Beijing</arg_value></tool_call>",
            allowedNames: allowed).first?.arguments["city"]?.stringValue, "Beijing")
    }

    // MARK: - 真机：对照 GGUF 模型的元数据模板逐字不变

    func testOtherGGUFChatTemplateIsUntouched() throws {
        let path = ProcessInfo.processInfo.environment["OTHER_GGUF_MODEL"]
            ?? "/Volumes/SSD/gguf/Qwen3.5-2B-Q4_K_M.gguf"
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("本地缺少对照 GGUF 模型：\(path)")
        }
        let model = try LlamaCppModel(path: path, gpuLayers: 0)
        let raw = LlamaCppModel.metadataString(
            model.raw, key: "tokenizer.chat_template")
        let patched = try XCTUnwrap(model.chatTemplate, "对照模型必须能加载模板")
        XCTAssertFalse(patched.contains("separators"))
        if let raw {
            XCTAssertEqual(patched, raw,
                           "对照模型的 chat template 必须逐字保持不变")
        }
        XCTAssertGreaterThan(model.contextTrain, 0)
        XCTAssertGreaterThan(model.vocabSize, 0)
    }
}
