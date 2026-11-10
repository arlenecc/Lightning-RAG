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
        var properties: [String: JSONValue] = [:]
        for tool in subcommands.map({ $0.tool }) {
            guard let props = tool.parameters.objectValue?["properties"]?.objectValue else {
                continue
            }
            for (key, value) in props where properties[key] == nil {
                properties[key] = value
            }
        }
        properties["op"] = ToolSupport.stringProperty(
            "选择要执行的动作：\(order.joined(separator: " / "))",
            enumValues: order)
        self.parameters = ToolSupport.objectSchema(
            properties: properties, required: ["op"])
    }

    public func shouldConfirm(_ args: [String: JSONValue]) -> Bool {
        guard let tool = resolvedTool(args) else { return false }
        return tool.shouldConfirm(args)
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let op = resolvedOp(args)
        guard let tool = subcommands[op] else {
            return .fail(
                "unknown op: \"\(op)\"（可用: \(orderedOps.joined(separator: "/"))）")
        }
        var rest = args
        rest.removeValue(forKey: "op")
        return try await tool.run(rest)
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
