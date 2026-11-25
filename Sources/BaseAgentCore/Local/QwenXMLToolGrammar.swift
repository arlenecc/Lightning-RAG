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
        /// 必须出现的参数名（schema 的 `required`，按声明顺序）。xgrammar 的
        /// 工具调用语法若不强制必填项，小模型常输出缺 `op` 的旧格式调用，
        /// 工具层虽然兼容，但这里补齐能让模型第一下就生成正确结构。
        let required: [String]
    }

    static func make(tools: [JSONValue]) -> String? {
        let specs: [ToolSpec] = tools.compactMap { tool in
            guard let function = tool["function"]?.objectValue,
                  let name = function["name"]?.stringValue, !name.isEmpty else { return nil }
            var keys: [String] = []
            var required: [String] = []
            if let parameters = function["parameters"]?.objectValue,
               let properties = parameters["properties"]?.objectValue {
                keys = properties.keys.sorted()
                if let declared = parameters["required"]?.arrayValue {
                    let valid = Set(keys)
                    required = declared.compactMap { $0.stringValue }
                        .filter { valid.contains($0) }
                }
            }
            return ToolSpec(name: name, keys: keys, required: required)
        }
        guard !specs.isEmpty else { return nil }

        // GBNF（llama.cpp）对括号分组较严格：`("x")` 单候选括号会被判为
        // 语法错误（XGrammar/XGrammar 的 EBNF 则接受）。用 helper 统一生成
        // 候选表达式：单个直接写字面量，多个才用括号包起来。
        func alternation(_ keys: [String]) -> String {
            let parts = keys.map { "\"\(escape($0))\"" }
            return parts.count == 1 ? parts[0] : "(\(parts.joined(separator: " | ")))"
        }
        // 注意：call0/call1 是规则名（不是字面量），绝不能加引号。
        let callAlternatives = specs.indices.map { "call\($0)" }.joined(separator: " | ")
        let callChoice = specs.count == 1 ? callAlternatives : "(\(callAlternatives))"
        var lines: [String] = [
            "root ::= ws \"<function=\" \(callChoice)",
            "ws ::= [ \\t\\n]*",
            "value ::= [^<]*",
        ]
        for (index, spec) in specs.enumerated() {
            let paramRule = "param\(index)"
            // 必填项独立成产生式并在 call 里强制展开一次；其余仍走
            // 可重复的 paramRule。必填项自身也从 paramRule 的候选里排除，
            // 避免同一参数被输出两次（如两个 <parameter=op>）。
            let optionalKeys = spec.keys.filter { !spec.required.contains($0) }
            let optionalAlternatives = alternation(optionalKeys)
            var callBody = "\"\(escape(spec.name))\" ws \">\" ws"
            for (reqIndex, key) in spec.required.enumerated() {
                // llama.cpp GBNF 不接受规则名里的下划线（实测 req0_0 编译
                // 失败），用纯字母数字命名（req00 / req01 ...）。
                let reqRule = "req\(index)\(reqIndex)"
                callBody += " \(reqRule)"
                lines.append("\(reqRule) ::= \"<parameter=\" \"\(escape(key))\" ws \">\" ws "
                    + "value \"</parameter>\" ws")
            }
            if optionalAlternatives.isEmpty {
                // 无可选参数：不给 paramRule 任何可产出分支；若连必填项也没有，
                // 用不可达产生式保持规则集合合法。
                if spec.required.isEmpty {
                    lines.append("\(paramRule) ::= \"\\u0000\"")
                }
            } else {
                lines.append("\(paramRule) ::= \"<parameter=\" \(optionalAlternatives) ws \">\" ws "
                    + "value \"</parameter>\" ws")
                callBody += " \(paramRule)*"
            }
            lines.append("call\(index) ::= \(callBody) \"</function>\" ws \"</tool_call>\"")
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
