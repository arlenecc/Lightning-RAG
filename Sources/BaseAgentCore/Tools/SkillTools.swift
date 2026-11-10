import Foundation

// MARK: - 参数辅助

private extension Dictionary where Key == String, Value == JSONValue {
    func trimmedString(_ key: String) -> String {
        (self[key]?.stringValue ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func boundedInt(_ key: String, default defaultValue: Int, range: ClosedRange<Int>) -> Int {
        let value = self[key]?.intValue ?? defaultValue
        return Swift.min(Swift.max(value, range.lowerBound), range.upperBound)
    }

    func stringList(_ key: String) -> [String] {
        guard let raw = self[key] else { return [] }
        if let array = raw.arrayValue {
            return array.compactMap(\.stringValue)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        }
        if let text = raw.stringValue {
            return text.split(whereSeparator: { $0 == "," || $0 == "，" || $0 == "\n" })
                .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        }
        return []
    }
}

private func rounded(_ value: Double) -> Double {
    (value * 10_000).rounded() / 10_000
}

// MARK: - skill_search

/// 关键词技能检索工具：等价于 Python `SkillSearchTool`。
///
/// 注意：本工具**不得**声明 `op` 参数。它是 `skill_tools` 组合工具的子命令，
/// 组合工具独占 `op` 键并在分发前将其删除；早期版本自带的 `op`(search/list)
/// 因此永远收不到，`list` 动作实际失效。现在 `list` 由 `SkillListTool` 承担，
/// 这里只做关键词检索。
public final class SkillSearchTool: AgentTool {
    public let name = "skill_search"
    public let description = "关键词搜索 .agent/skills 下已保存的技能；按 query 打分（名称 > 关键词/tags > 描述）。"
    public let parameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([
            "query": .object([
                "type": .string("string"),
                "description": .string("搜索关键词"),
            ]),
            "top_k": .object([
                "type": .string("integer"),
                "description": .string("返回条数，默认 5，范围 1-20"),
            ]),
        ]),
        "required": .array([.string("query")]),
    ])

    private let index: SkillIndex

    public init(config: AgentConfig,
                callbacks: AgentCallbacks? = nil,
                registry: ToolRegistry? = nil,
                index: SkillIndex? = nil) {
        self.index = index ?? SkillIndex(workspace: config.workspace)
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        // 兼容旧调用方：若仍传 op=list / skill_op=list，转发给共享实现。
        let legacy = args.trimmedString("skill_op").isEmpty
            ? args.trimmedString("op") : args.trimmedString("skill_op")
        if legacy == "list" { return listAllSkills(index) }

        let query = args.trimmedString("query")
        guard !query.isEmpty else { return .fail("缺少参数: query") }
        let topK = args.boundedInt("top_k", default: 5, range: 1...20)
        let results = index.search(query: query, topK: topK)
        if results.isEmpty {
            return .ok("找不到匹配的技能。")
        }
        let payload: JSONValue = .array(results.map { entryJSON($0) })
        return .ok(payload.jsonString(pretty: true))
    }

    private func entryJSON(_ entry: SkillIndexEntry) -> JSONValue {
        .object([
            "name": .string(entry.name),
            "description": .string(entry.description),
            "keywords": .array(entry.keywords.map { .string($0) }),
            "tags": .array(entry.tags.map { .string($0) }),
            "path": .string(entry.path),
            "entry": .string(entry.entry),
        ])
    }
}

// MARK: - skill_list

/// 列出全部已保存技能。作为 `skill_tools(op="list")` 的子命令独立存在，
/// 避免任何子工具再声明与组合工具 `op` 同名的参数。
public final class SkillListTool: AgentTool {
    public let name = "skill_list"
    public let description = "列出全部已保存技能（名称、路径、描述）。"
    public let parameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([:]),
        "required": .array([]),
    ])

    private let index: SkillIndex

    public init(config: AgentConfig,
                callbacks: AgentCallbacks? = nil,
                registry: ToolRegistry? = nil,
                index: SkillIndex? = nil) {
        self.index = index ?? SkillIndex(workspace: config.workspace)
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        listAllSkills(index)
    }
}

/// `list` 动作的共享实现（原 `SkillSearchTool(op="list")` 分支）。
private func listAllSkills(_ index: SkillIndex) -> ToolResult {
    index.scan(force: true)
    let entries = index.listAll()
    if entries.isEmpty {
        return .ok("暂无已保存的技能。")
    }
    let compact: [JSONValue] = entries.prefix(50).map { entry in
        .object([
            "name": .string(entry.name),
            "path": .string(entry.path),
            "description": .string(String(entry.description.prefix(100))),
        ])
    }
    var output = JSONValue.array(compact).jsonString(pretty: true)
    if entries.count > 50 {
        output += "\n… (共 \(entries.count) 个技能，仅显示前 50 个；用 skill_tools(op=\"search\") + query 精确查找)"
    }
    return .ok(output)
}

// MARK: - skill_semantic_search

/// 语义/混合技能检索工具：等价于 Python `SkillSemanticSearchTool`。
public final class SkillSemanticSearchTool: AgentTool {
    public let name = "skill_semantic_search"
    public let description = "用自然语言语义检索 skills 目录下的 SKILL.md（向量 + BM25 混合）。query 请使用用户原始措辞，不要翻译或音译。返回技能目录、名称、tags、描述与匹配分数。"
    public let parameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([
            "query": .object([
                "type": .string("string"),
                "description": .string("自然语言任务描述"),
            ]),
            "top_k": .object([
                "type": .string("integer"),
                "description": .string("返回条数，默认 5，范围 1-20"),
            ]),
        ]),
        "required": .array([.string("query")]),
    ])

    private let retriever: SkillRetriever

    public init(config: AgentConfig,
                callbacks: AgentCallbacks? = nil,
                registry: ToolRegistry? = nil,
                retriever: SkillRetriever? = nil) {
        self.retriever = retriever ?? SkillRetriever(
            workspace: config.workspace,
            embedding: LocalModelFactory.shared.makeEmbedding(config: config))
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let query = args.trimmedString("query")
        guard !query.isEmpty else { return .fail("缺少参数: query") }
        let topK = args.boundedInt("top_k", default: 5, range: 1...20)
        let skills = await retriever.retrieve(query: query, topK: topK)
        guard !skills.isEmpty else {
            return .ok("找不到匹配的技能。")
        }
        let payload: JSONValue = .array(skills.map { skill in
            .object([
                "dir": .string(skill.dir),
                "name": .string(skill.name),
                "tags": .array(skill.tags.map { .string($0) }),
                "description": .string(String(skill.description.prefix(200))),
                "score": .double(rounded(skill.score)),
            ])
        })
        return .ok(payload.jsonString(pretty: true))
    }
}

// MARK: - skill_load

/// 技能加载工具：等价于 Python `SkillLoadTool`。
public final class SkillLoadTool: AgentTool {
    public let name = "skill_load"
    public let description = "按目录加载技能的完整说明（默认 SKILL.md）。path 必须是技能目录相对路径。"
    public let parameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([
            "path": .object([
                "type": .string("string"),
                "description": .string("技能目录相对路径，例如 my-skill 或 nested/my-skill"),
            ]),
            "entry": .object([
                "type": .string("string"),
                "description": .string("入口文件名，默认 SKILL.md"),
            ]),
        ]),
        "required": .array([.string("path")]),
    ])

    private let retriever: SkillRetriever
    private let index: SkillIndex

    public init(config: AgentConfig,
                callbacks: AgentCallbacks? = nil,
                registry: ToolRegistry? = nil,
                retriever: SkillRetriever? = nil,
                index: SkillIndex? = nil) {
        self.retriever = retriever ?? SkillRetriever(workspace: config.workspace)
        self.index = index ?? SkillIndex(workspace: config.workspace)
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let rawPath = args.trimmedString("path")
        guard !rawPath.isEmpty else { return .fail("缺少参数: path") }
        guard !(rawPath as NSString).isAbsolutePath,
              !(rawPath as NSString).pathComponents.contains("..") else {
            return .fail("path 必须是技能目录相对路径")
        }

        var entry = args.trimmedString("entry")
        if entry.isEmpty { entry = "SKILL.md" }
        guard !(entry as NSString).isAbsolutePath,
              !(entry as NSString).pathComponents.contains("..") else {
            return .fail("entry 必须是技能目录内的相对文件名")
        }

        if let directory = retriever.readSkillDir(rawPath) {
            var candidates = [entry]
            if entry == "SKILL.md" { candidates.append("prompt.md") }
            for name in candidates {
                if let content = retriever.manager.readFile(inDirectory: directory, entry: name) {
                    let base = URL(fileURLWithPath: retriever.skillsDirectory)
                        .resolvingSymlinksInPath().path
                    let external = !(directory == base || directory.hasPrefix(base + "/"))
                    return .ok(external ? "[external skill: \(directory)]\n\n\(content)" : content)
                }
            }
        }

        if let content = index.readSkillContent(path: rawPath, entry: entry) {
            return .ok(content)
        }
        if entry != "SKILL.md", let content = index.readSkillContent(path: rawPath, entry: "SKILL.md") {
            return .ok(content)
        }
        return .fail("技能不存在或入口文件缺失: \(rawPath)")
    }
}

// MARK: - skill_save

/// 技能保存工具：等价于 Python `SkillSaveTool`。
public final class SkillSaveTool: AgentTool {
    public let name = "skill_save"
    public let description = "把可复用的操作流程保存为技能（写入 workspace/skills/<slug>/SKILL.md，并同步到 .agent/skills 索引目录）。"
    public let parameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([
            "name": .object([
                "type": .string("string"),
                "description": .string("技能名称（会转成目录 slug）"),
            ]),
            "description": .object([
                "type": .string("string"),
                "description": .string("一句话说明技能用途"),
            ]),
            "when_to_use": .object([
                "type": .string("string"),
                "description": .string("适用场景"),
            ]),
            "steps": .object([
                "type": .string("string"),
                "description": .string("可复用的操作步骤"),
            ]),
            "pitfalls": .object([
                "type": .string("string"),
                "description": .string("注意事项 / 踩坑记录"),
            ]),
            "tags": .object([
                "type": .string("string"),
                "description": .string("逗号分隔的检索标签（可选）"),
            ]),
        ]),
        "required": .array([.string("name"), .string("description"), .string("steps")]),
    ])

    private let manager: SkillManager
    private let retriever: SkillRetriever
    private let index: SkillIndex
    private let legacySkillsDirectory: String

    public init(config: AgentConfig,
                callbacks: AgentCallbacks? = nil,
                registry: ToolRegistry? = nil,
                manager: SkillManager? = nil,
                retriever: SkillRetriever? = nil,
                index: SkillIndex? = nil) {
        let resolvedManager = manager ?? SkillManager(workspace: config.workspace)
        self.manager = resolvedManager
        self.legacySkillsDirectory = URL(fileURLWithPath: config.workspace)
            .appendingPathComponent(".agent")
            .appendingPathComponent("skills")
            .path
        self.retriever = retriever ?? SkillRetriever(manager: resolvedManager,
                                                     embedding: LocalModelFactory.shared
                                                        .makeEmbedding(config: config))
        self.index = index ?? SkillIndex(workspace: config.workspace)
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let name = args.trimmedString("name")
        let description = args.trimmedString("description")
        guard !name.isEmpty else { return .fail("缺少参数: name") }
        guard !description.isEmpty else { return .fail("缺少参数: description") }

        let whenToUse = args.trimmedString("when_to_use")
        let steps = args.trimmedString("steps")
        let pitfalls = args.trimmedString("pitfalls")
        let tags = args.stringList("tags")
        guard !steps.isEmpty else { return .fail("steps is required") }

        do {
            let skill = try manager.save(name: name,
                                         description: description,
                                         whenToUse: whenToUse,
                                         steps: steps,
                                         pitfalls: pitfalls,
                                         tags: tags)
            retriever.markStale()
            index.scan(force: true)
            let mirrored = URL(fileURLWithPath: legacySkillsDirectory)
                .appendingPathComponent(skill.directory)
                .appendingPathComponent("SKILL.md")
                .path
            var note = "下次可用 skill_tools(op=\"semantic_search\") 或 skill_tools(op=\"search\") 召回（索引会自动增量同步）"
            if FileManager.default.fileExists(atPath: mirrored) {
                note += "；已镜像到 \(mirrored) 以便关键词检索命中"
            }
            return .ok("skill saved: \(skill.absolutePath)\n\(note)。")
        } catch {
            return .fail("技能保存失败: \(error.localizedDescription)")
        }
    }
}
