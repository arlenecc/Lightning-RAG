import Foundation

/// 组合工具：把同一领域的多个原子工具收敛成一个「子命令」工具，用 `op` 参数选择动作。
///
/// 目的：减少暴露给模型的工具数量（如把 file_read/file_write/file_modify/… 合并为
/// `file_tools`），但内部仍复用原有 `AgentTool` 实现，行为零变化。
/// - `run` 按 `op` 分发到对应子工具，并把 `op` 之外的参数原样透传。
/// - `shouldConfirm` 同样按 `op` 转发，保留原工具的确认语义（fail-closed 由 registry 统一处理）。
/// - `op` 为空时若配置了 `defaultOp` 则走默认动作（如 `web_scan` 默认 `scan`），否则报错。
public final class CompositeTool: AgentTool {
    public let name: String
    public let description: String
    public let parameters: JSONValue
    /// `op` 缺失时使用的默认动作；nil 表示必填。
    public let defaultOp: String?

    /// 组合工具独占的参数键：分发前会被删除，子工具不得声明同名参数。
    private static let reservedKeys: Set<String> = ["op"]

    /// 诊断：声明了保留键（或把它列入 required）的子工具名。非空即表示该
    /// 子工具的动作参数会被组合层吞掉（早期 `skill_search` 的 `op=list`）。
    /// 正常应始终为空，测试据此回归。
    public private(set) var reservedKeyCollisions: [String] = []

    /// 诊断：不同子工具间同名参数但 schema 类型/枚举不一致的键。非空即表示
    /// 合并时「先到先得」会静默丢掉后者的定义。正常应始终为空。
    public private(set) var parameterTypeCollisions: [String] = []

    private let subcommands: [String: AgentTool]
    private let orderedOps: [String]

    public init(name: String,
                description: String,
                subcommands: [(op: String, tool: AgentTool)],
                defaultOp: String? = nil) {
        self.name = name
        self.description = description
        self.defaultOp = defaultOp

        var map: [String: AgentTool] = [:]
        var order: [String] = []
        for (op, tool) in subcommands where map[op] == nil {
            map[op] = tool
            order.append(op)
        }
        self.subcommands = map
        self.orderedOps = order

        // 合并各子命令的参数 schema；`op` 枚举覆盖所有子命令值。
        // 同时校验参数冲突：保留键冲突会让子工具永远收不到动作参数，
        // 同名参数类型/枚举冲突会让「先到先得」静默丢弃一方定义。
        var properties: [String: JSONValue] = [:]
        var signatures: [String: String] = [:]
        var reserved: [String] = []
        var typeCollisions: [String] = []
        for op in order {
            guard let tool = map[op],
                  let schema = tool.parameters.objectValue,
                  let props = schema["properties"]?.objectValue else { continue }
            let required = Set(schema["required"]?.arrayValue?
                .compactMap(\.stringValue) ?? [])
            if props["op"] != nil || required.contains("op"), !reserved.contains(tool.name) {
                reserved.append(tool.name)
            }
            for (key, value) in props where !Self.reservedKeys.contains(key) {
                let object = value.objectValue ?? [:]
                let type = object["type"]?.stringValue ?? ""
                let enums = object["enum"]?.arrayValue?.compactMap(\.stringValue) ?? []
                let signature = type + "|" + enums.joined(separator: ",")
                if let existing = signatures[key] {
                    if existing != signature && !typeCollisions.contains(key) {
                        typeCollisions.append(key)
                    }
                } else {
                    signatures[key] = signature
                }
                if properties[key] == nil { properties[key] = value }
            }
        }
        self.reservedKeyCollisions = reserved
        self.parameterTypeCollisions = typeCollisions
        properties["op"] = ToolSupport.stringProperty(
            "选择要执行的动作：\(order.joined(separator: " / "))",
            enumValues: order)
        self.parameters = ToolSupport.objectSchema(
            properties: properties, required: ["op"])
    }

    public func shouldConfirm(_ args: [String: JSONValue]) -> Bool {
        guard let tool = resolvedTool(args) else { return false }
        return tool.shouldConfirm(forwardedArgs(args))
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let op = resolvedOp(args)
        guard let tool = subcommands[op] else {
            return .fail(
                "unknown op: \"\(op)\"（可用: \(orderedOps.joined(separator: "/"))）")
        }
        return try await tool.run(forwardedArgs(args))
    }

    /// 分发给子工具的参数：移除组合层独占的 `op`，其余参数原样透传。
    private func forwardedArgs(_ args: [String: JSONValue]) -> [String: JSONValue] {
        var rest = args
        for key in Self.reservedKeys { rest.removeValue(forKey: key) }
        return rest
    }

    private func resolvedOp(_ args: [String: JSONValue]) -> String {
        let raw = (args["op"]?.stringValue ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return raw.isEmpty ? (defaultOp ?? "") : raw
    }

    private func resolvedTool(_ args: [String: JSONValue]) -> AgentTool? {
        let op = resolvedOp(args)
        return op.isEmpty ? nil : subcommands[op]
    }
}
