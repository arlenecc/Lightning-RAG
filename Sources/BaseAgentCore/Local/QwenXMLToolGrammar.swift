import Foundation

/// 从 OpenAI 风格的工具 schema 生成 Qwen3.5 XML 工具调用的 EBNF 约束。
///
/// Qwen3.5 的 chat template 要求形如：
/// ```
/// <tool_call>
/// <function=tool_name>
/// <parameter=key>
/// value
/// </parameter>
/// </function>
/// </tool_call>
/// ```
/// 约束从 `<tool_call>` 之后开始（触发串本身由调用方检测），因此 `root`
/// 直接以可选的空白 + `<function=` 开头。参数值放行任意文本（不含 `<`）：
/// 约束只能保证**结构**正确，不能保证参数值的语义正确。
enum QwenXMLToolGrammar {
    /// 延迟激活的触发标记。检测到它之后才开始施加掩码，之前完全放行，
    /// 因此普通文本回答（即便本轮携带工具 schema）不受任何影响。
    static let trigger = "<tool_call>"
    /// llama.cpp lazy sampler 要求捕获组指定从哪个位置开始喂语法。
    static let lazyTriggerPattern = "(\(trigger))"

    private struct ToolSpec {
        let name: String
        let keys: [String]
    }

    static func make(tools: [JSONValue]) -> String? {
        let specs: [ToolSpec] = tools.compactMap { tool in
            guard let function = tool["function"]?.objectValue,
                  let name = function["name"]?.stringValue, !name.isEmpty else { return nil }
            var keys: [String] = []
            if let parameters = function["parameters"]?.objectValue,
               let properties = parameters["properties"]?.objectValue {
                keys = properties.keys.sorted()
            }
            return ToolSpec(name: name, keys: keys)
        }
        guard !specs.isEmpty else { return nil }

        var lines: [String] = [
            "root ::= ws \"<function=\" (\(specs.indices.map { "call\($0)" }.joined(separator: " | ")))",
            "ws ::= [ \\t\\n]*",
            "value ::= [^<]*",
        ]
        for (index, spec) in specs.enumerated() {
            let paramRule = "param\(index)"
            let alternatives = spec.keys.map { key in
                "\"\(escape(key))\""
            }.joined(separator: " | ")
            lines.append("call\(index) ::= \"\(escape(spec.name))\" ws \">\" ws "
                + "\(paramRule)* \"</function>\" ws \"</tool_call>\"")
            if alternatives.isEmpty {
                // 无参数工具：给一个不可达的产生式，保持规则集合合法。
                lines.append("\(paramRule) ::= \"\\u0000\"")
            } else {
                lines.append("\(paramRule) ::= \"<parameter=\" (\(alternatives)) ws \">\" ws "
                    + "value \"</parameter>\" ws")
            }
        }
        return lines.joined(separator: "\n")
    }

    /// GBNF 版本（llama.cpp 的 lazy grammar sampler）。与 EBNF 的差别是
    /// 触发串仍在生成序列里——lazy sampler 从 `<tool_call>` 开始喂语法，
    /// 因此 root 要把触发串包含进来。
    static func makeGBNF(tools: [JSONValue]) -> String? {
        guard let source = make(tools: tools) else { return nil }
        return source.replacingOccurrences(
            of: "root ::= ws \"<function=\"",
            with: "root ::= \"<tool_call>\" ws \"<function=\"")
    }

    private static func escape(_ text: String) -> String {
        var out = ""
        for ch in text {
            switch ch {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default: out.append(ch)
            }
        }
        return out
    }
}
