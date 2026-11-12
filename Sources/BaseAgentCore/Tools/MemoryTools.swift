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
        // 组合工具 long_memory 独占 `op`，本工具的动作键已改名为 `graph_op`；
        // required 必须与之一致，否则 schema 会要求一个永远被删掉的键。
        "required": .array([.string("graph_op")]),
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
    public let description = """
    检索长期记忆，返回固定 1000 字符以内的 JSON 对象。
    query 请使用用户原始措辞，不要翻译或音译；top_k 控制召回候选数（默认 5，范围 1-20）。
    优先级：① 时间最近的 5 条观察（hits）→ ② 这 5 条观察直接关联的实体与关系（entities / relations）
    → ③ 仅当还有余额时才做一跳邻居扩展（role=neighbor 的实体、扩展关系、related_facts），余额不足则放弃扩展。
    """
    public let parameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([
            "query": .object([
                "type": .string("string"),
                "description": .string("检索关键词或问题"),
            ]),
            "top_k": .object([
                "type": .string("integer"),
                "description": .string("召回观察条数，默认 5，范围 1-20"),
            ]),
        ]),
        "required": .array([.string("query")]),
    ])

    /// 单次返回的硬预算（字符）。工具结果进历史会撑大 KV 前缀，收紧到 1000。
    static let maxOutputChars = 1_000
    /// 优先保留的观察条数（按写入时间倒序）。
    static let maxRecentObservations = 5
    /// 核心关系上限：hub 实体的边不能吃光预算。
    static let maxCoreRelations = 8
    /// 单条观察文本硬上限（防一条超长观察撑爆整次结果）。
    static let maxTextChars = 160
    /// 预算削减时允许把观察文本压到的最小长度（正常观察 8–80 字，不再压短）。
    static let minTextChars = 80
    /// 扩展段至少要有这么多余额才尝试，否则直接放弃扩展。
    static let minExpansionRoom = 80

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
        let payload = Self.buildPayload(hits: hits, graph: graph,
                                        budget: Self.maxOutputChars)
        return .ok(payload.jsonString())
    }

    // MARK: - 分层装配

    /// 三层优先级：① 时间最近的 5 条观察 → ② 这 5 条直接关联的实体与关系
    /// → ③ 仅当仍有余额才做一跳邻居扩展（放不下即放弃剩余扩展）。
    /// 全程保证输出 ≤ budget 且是完整 JSON（不会被内核 `l1Trim` 切成半截）。
    static func buildPayload(hits: [GraphObservationHit],
                             graph: GraphMemoryStore,
                             budget: Int = maxOutputChars) -> JSONValue {
        // ① 按写入时间倒序取最近 5 条；旧库时间戳为 0 时按语义分数兜底。
        let recent: [GraphObservationHit] = Array(hits
            .sorted { left, right in
                if left.timestamp != right.timestamp { return left.timestamp > right.timestamp }
                return left.score > right.score
            }
            .prefix(maxRecentObservations))

        // ② 这 5 条观察直接关联的实体，以及这些实体直接参与的关系。
        var coreEntities: [GraphEntity] = []
        var seenNames = Set<String>()
        for hit in recent where seenNames.insert(hit.entity.lowercased()).inserted {
            if let entity = graph.getEntity(hit.entity) { coreEntities.append(entity) }
        }
        let coreNames = Set(coreEntities.map { $0.name.lowercased() })
        var coreRelations: [GraphRelation] = []
        var seenRelations = Set<String>()
        for relation in graph.listRelations() {
            guard coreNames.contains(relation.source.lowercased())
                || coreNames.contains(relation.target.lowercased()) else { continue }
            let key = Self.relationKey(relation)
            if seenRelations.insert(key).inserted { coreRelations.append(relation) }
            if coreRelations.count >= maxCoreRelations { break }
        }

        // ③ 削减核心直到装得下。顺序按「信息价值」从低到高：
        //    先压短过长文本（保住 5 条观察 + 实体 + 关系），再丢尾部关系、
        //    再丢尾部实体，最后才减少观察条数。
        var textCap = maxTextChars
        var hitCount = recent.count
        var entityCount = coreEntities.count
        var relationCount = coreRelations.count
        func coreObject() -> JSONValue {
            Self.payload(
                hits: Array(recent.prefix(hitCount)).map { Self.hitPayload($0, cap: textCap) },
                entities: Array(coreEntities.prefix(entityCount)).map {
                    Self.entityPayload($0, role: "hit")
                },
                relations: Array(coreRelations.prefix(relationCount)).map(Self.relationPayload),
                facts: [])
        }
        while coreObject().jsonString().count > budget {
            if textCap > minTextChars {
                textCap = max(minTextChars, textCap - 20)
                continue
            }
            if relationCount > 0 { relationCount -= 1; continue }
            if entityCount > 0 { entityCount -= 1; continue }
            if hitCount > 1 { hitCount -= 1; continue }
            break
        }

        var finalHits = Array(recent.prefix(hitCount)).map { Self.hitPayload($0, cap: textCap) }
        var finalEntities = Array(coreEntities.prefix(entityCount)).map {
            Self.entityPayload($0, role: "hit")
        }
        var finalRelations = Array(coreRelations.prefix(relationCount)).map(Self.relationPayload)
        var finalFacts: [JSONValue] = []
        func current() -> JSONValue {
            Self.payload(hits: finalHits, entities: finalEntities,
                         relations: finalRelations, facts: finalFacts)
        }
        func fits() -> Bool { current().jsonString().count <= budget }

        // ④ 余额充足才扩展。按「邻居实体 → 扩展关系 → 相关事实」逐条尝试，
        //    任一条放不下就回滚并放弃剩余全部扩展。
        if budget - current().jsonString().count >= minExpansionRoom {
            var entityExpansion: [JSONValue] = []
            var relationExpansion: [JSONValue] = []
            var factExpansion: [JSONValue] = []
            var seenNeighborNames = Set(coreNames)
            var seenFactKeys = Set(recent.map { Self.factKey($0.entity, $0.text) })

            for entity in coreEntities.prefix(entityCount) {
                let neighborhood = graph.neighbors(of: entity.name, depth: 1)
                for neighbor in neighborhood.entities
                    where !coreNames.contains(neighbor.name.lowercased()) {
                    let key = neighbor.name.lowercased()
                    guard seenNeighborNames.insert(key).inserted else { continue }
                    entityExpansion.append(Self.entityPayload(neighbor, role: "neighbor"))
                    var perEntity = 0
                    for observation in neighbor.observations where perEntity < 2 {
                        let factKey = Self.factKey(neighbor.name, observation)
                        guard seenFactKeys.insert(factKey).inserted else { continue }
                        perEntity += 1
                        factExpansion.append(Self.factPayload(neighbor.name, observation,
                                                              cap: textCap))
                    }
                }
                for relation in neighborhood.relations {
                    let key = Self.relationKey(relation)
                    guard seenRelations.insert(key).inserted else { continue }
                    relationExpansion.append(Self.relationPayload(relation))
                }
            }

            let expansion = entityExpansion.map { (section: "entities", item: $0) }
                + relationExpansion.map { (section: "relations", item: $0) }
                + factExpansion.map { (section: "related_facts", item: $0) }
            for candidate in expansion {
                switch candidate.section {
                case "entities": finalEntities.append(candidate.item)
                case "relations": finalRelations.append(candidate.item)
                default: finalFacts.append(candidate.item)
                }
                if !fits() {
                    switch candidate.section {
                    case "entities": finalEntities.removeLast()
                    case "relations": finalRelations.removeLast()
                    default: finalFacts.removeLast()
                    }
                    break
                }
            }
        }

        return current()
    }

    // MARK: - 组装辅助

    private static func payload(hits: [JSONValue], entities: [JSONValue],
                                relations: [JSONValue], facts: [JSONValue]) -> JSONValue {
        .object([
            "hits": .array(hits),
            "entities": .array(entities),
            "relations": .array(relations),
            "related_facts": .array(facts),
        ])
    }

    private static func hitPayload(_ hit: GraphObservationHit, cap: Int) -> JSONValue {
        .object([
            "entity": .string(hit.entity),
            "text": .string(capped(hit.text, cap: cap)),
            "score": .double(roundedScore(hit.score)),
        ])
    }

    private static func relationPayload(_ relation: GraphRelation) -> JSONValue {
        .object([
            "source": .string(relation.source),
            "target": .string(relation.target),
            "label": .string(relation.label),
        ])
    }

    private static func factPayload(_ entity: String, _ text: String, cap: Int) -> JSONValue {
        .object([
            "entity": .string(entity),
            "text": .string(capped(text, cap: cap)),
        ])
    }

    /// 实体只回传定位所需的最小字段；1000 字符预算下把体积留给观察与关系。
    private static func entityPayload(_ entity: GraphEntity, role: String) -> JSONValue {
        .object([
            "name": .string(entity.name),
            "role": .string(role),
        ])
    }

    private static func relationKey(_ relation: GraphRelation) -> String {
        "\(relation.source.lowercased())|\(relation.target.lowercased())|\(relation.label.lowercased())"
    }

    private static func factKey(_ entity: String, _ text: String) -> String {
        "\(entity.lowercased())|\(text.lowercased())"
    }

    private static func capped(_ text: String, cap: Int) -> String {
        text.count > cap ? String(text.prefix(cap)) + "…" : text
    }
}
