import Foundation

// MARK: - 参数与编码辅助

private extension Dictionary where Key == String, Value == JSONValue {
    func trimmedString(_ key: String) -> String {
        (self[key]?.stringValue ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func positiveInt(_ key: String, default defaultValue: Int, range: ClosedRange<Int>) -> Int {
        let value = self[key]?.intValue ?? defaultValue
        return Swift.min(Swift.max(value, range.lowerBound), range.upperBound)
    }
}

private func encodeJSON<T: Encodable>(_ value: T, pretty: Bool = false) -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = pretty
        ? [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        : [.sortedKeys, .withoutEscapingSlashes]
    guard let data = try? encoder.encode(value),
          let text = String(data: data, encoding: .utf8) else {
        return "{}"
    }
    return text
}

private func roundedScore(_ value: Double) -> Double {
    (value * 10_000).rounded() / 10_000
}

// MARK: - work_memory

/// 工作记忆工具：等价于 Python `WorkMemoryTool`。
public final class WorkMemoryTool: AgentTool {
    public let name = "work_memory"
    public let description = "管理当前任务的工作记忆（workspace 内持久化）。op=set 保存键值，get 读取，list 列出全部，clear 清空。"
    public let parameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([
            "op": .object([
                "type": .string("string"),
                "enum": .array([.string("set"), .string("get"), .string("list"), .string("clear")]),
                "description": .string("操作类型"),
            ]),
            "key": .object([
                "type": .string("string"),
                "description": .string("键名（set/get 必填）"),
            ]),
            "value": .object([
                "type": .string("string"),
                "description": .string("保存的文本值（set 必填）"),
            ]),
        ]),
        "required": .array([.string("op")]),
    ])

    private let workMemory: WorkMemory

    public init(config: AgentConfig,
                callbacks: AgentCallbacks? = nil,
                registry: ToolRegistry? = nil,
                workMemory: WorkMemory? = nil) {
        self.workMemory = workMemory ?? registry?.workMemory
            ?? WorkMemory(workspace: config.workspace)
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let op = args.trimmedString("op")
        switch op {
        case "set":
            let key = args.trimmedString("key")
            guard !key.isEmpty else { return .fail("缺少参数: key") }
            guard let value = args["value"]?.stringValue else {
                return .fail("缺少参数: value")
            }
            workMemory.set(key, value)
            return .ok("set \(key)")
        case "get":
            let key = args.trimmedString("key")
            guard !key.isEmpty else { return .fail("缺少参数: key") }
            return .ok(workMemory.get(key) ?? "")
        case "list":
            return .ok(encodeJSON(workMemory.list(), pretty: true))
        case "clear":
            workMemory.clear()
            return .ok("cleared")
        default:
            return .fail("未知 op: \(op)（可用: set/get/list/clear）")
        }
    }
}

// MARK: - memory_graph

/// 知识图谱工具：等价于 Python `MemoryGraphTool`。
public final class MemoryGraphTool: AgentTool {
    public let name = "memory_graph"
    public let description = "跨任务长期记忆（实体-关系-观察知识图谱）。graph_op=create_entity/add_observations/create_relation/list_entities/get_entity/delete_entity/snapshot/clear。"
    public let parameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([
            "graph_op": .object([
                "type": .string("string"),
                "enum": .array([
                    .string("create_entity"), .string("add_observations"),
                    .string("create_relation"), .string("list_entities"),
                    .string("get_entity"), .string("delete_entity"),
                    .string("snapshot"), .string("clear"),
                ]),
                "description": .string("操作类型"),
            ]),
            "name": .object([
                "type": .string("string"),
                "description": .string("实体名称"),
            ]),
            "entity_type": .object([
                "type": .string("string"),
                "description": .string("实体类型，如 person / project / preference（create_entity 可选）"),
            ]),
            "observations": .object([
                "type": .string("array"),
                "items": .object(["type": .string("string")]),
                "description": .string("要记录的观察事实（create_entity/add_observations）"),
            ]),
            "source": .object([
                "type": .string("string"),
                "description": .string("关系起点实体（create_relation）"),
            ]),
            "target": .object([
                "type": .string("string"),
                "description": .string("关系终点实体（create_relation）"),
            ]),
            "label": .object([
                "type": .string("string"),
                "description": .string("关系标签，如 works_on / likes（create_relation）"),
            ]),
            "max_items": .object([
                "type": .string("integer"),
                "description": .string("snapshot 返回的实体条数，默认 10，范围 1-50"),
            ]),
        ]),
        "required": .array([.string("op")]),
    ])

    private let graph: GraphMemoryStore

    public init(config: AgentConfig,
                callbacks: AgentCallbacks? = nil,
                registry: ToolRegistry? = nil,
                graphMemory: GraphMemoryStore? = nil) {
        self.graph = graphMemory ?? GraphMemoryStore.shared(
            workspace: config.workspace,
            embedding: LocalModelFactory.shared.makeEmbedding(config: config))
    }

    public func shouldConfirm(_ args: [String: JSONValue]) -> Bool {
        let op = args.trimmedString("graph_op")
        return op == "clear" || op == "delete_entity"
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let op = args.trimmedString("graph_op")
        switch op {
        case "create_entity":
            let name = args.trimmedString("name")
            guard !name.isEmpty else { return .fail("缺少参数: name") }
            guard let observations = parseObservations(args) else {
                return .fail("'observations' 必须是字符串数组")
            }
            let entity = graph.upsertEntity(name,
                                            type: args.trimmedString("entity_type"),
                                            observations: observations)
            return .ok(encodeJSON(entity))
        case "add_observations":
            let name = args.trimmedString("name")
            guard !name.isEmpty else { return .fail("缺少参数: name") }
            guard let observations = parseObservations(args),
                  !observations.isEmpty else {
                return .fail("op 'add_observations' requires 'name' and 'observations'")
            }
            let added = graph.addObservations(name, observations)
            return .ok("added \(added) observation(s)")
        case "create_relation":
            let source = args.trimmedString("source")
            let target = args.trimmedString("target")
            let label = args.trimmedString("label")
            guard !source.isEmpty, !target.isEmpty, !label.isEmpty else {
                return .fail("缺少参数: source / target / label")
            }
            let created = graph.createRelation(source: source, target: target, label: label)
            return .ok(created ? "created" : "duplicate or invalid")
        case "list_entities":
            let entities = graph.listEntities()
            var summary = entities.prefix(200).map { entity -> String in
                let type = entity.type.isEmpty ? "?" : entity.type
                return "\(entity.name) (\(type)) [\(entity.observations.count) obs]"
            }
            if entities.count > 200 {
                summary.append("… (共 \(entities.count) 个实体，仅显示前 200 个；用 get_entity 查询具体实体)")
            }
            return .ok(encodeJSON(summary, pretty: true))
        case "get_entity":
            let name = args.trimmedString("name")
            guard !name.isEmpty else { return .fail("缺少参数: name") }
            guard let entity = graph.getEntity(name) else {
                return .ok("not found")
            }
            return .ok(encodeJSON(entity))
        case "delete_entity":
            let name = args.trimmedString("name")
            guard !name.isEmpty else { return .fail("缺少参数: name") }
            return .ok(graph.deleteEntity(name) ? "deleted" : "not found")
        case "snapshot":
            let text = graph.snapshot(maxItems: args.positiveInt("max_items", default: 10,
                                                                 range: 1...50))
            return .ok(text.isEmpty ? "(empty)" : text)
        case "clear":
            graph.clear()
            return .ok("cleared")
        default:
            return .fail("未知 graph_op: \(op)")
        }
    }

    private func parseObservations(_ args: [String: JSONValue]) -> [String]? {
        guard let raw = args["observations"] else { return [] }
        if raw.stringValue != nil { return nil }
        guard let array = raw.arrayValue else { return nil }
        let strings = array.compactMap(\.stringValue)
        guard strings.count == array.count else { return nil }
        return strings.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }
}

// MARK: - memory_search

/// 语义记忆检索工具：等价于 Python `MemorySearchTool`。
public final class MemorySearchTool: AgentTool {
    public let name = "memory_search"
    public let description = "检索长期记忆（实体观察）中的相关事实。query 请使用用户原始措辞，不要翻译或音译；top_k 控制返回条数（默认 5）。"
    public let parameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([
            "query": .object([
                "type": .string("string"),
                "description": .string("检索关键词或问题"),
            ]),
            "top_k": .object([
                "type": .string("integer"),
                "description": .string("返回条数，默认 5，范围 1-20"),
            ]),
        ]),
        "required": .array([.string("query")]),
    ])

    private let graph: GraphMemoryStore

    public init(config: AgentConfig,
                callbacks: AgentCallbacks? = nil,
                registry: ToolRegistry? = nil,
                graphMemory: GraphMemoryStore? = nil) {
        self.graph = graphMemory ?? GraphMemoryStore.shared(
            workspace: config.workspace,
            embedding: LocalModelFactory.shared.makeEmbedding(config: config))
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let query = args.trimmedString("query")
        guard !query.isEmpty else { return .fail("缺少参数: query") }
        let topK = args.positiveInt("top_k", default: 5, range: 1...20)
        let hits = await graph.search(query: query, topK: topK)
        let payload: JSONValue = .array(hits.map { hit in
            .object([
                "entity": .string(hit.entity),
                "text": .string(hit.text),
                "score": .double(roundedScore(hit.score)),
            ])
        })
        return .ok(payload.jsonString(pretty: true))
    }
}
