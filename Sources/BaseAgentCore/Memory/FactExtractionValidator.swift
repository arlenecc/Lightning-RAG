import Foundation

// MARK: - 事实抽取草稿

/// 辅助模型产出的实体草稿（写入 memory graph 前的中间形态）。
public struct FactEntityDraft: Sendable, Equatable {
    public var name: String
    public var type: String
    public var observations: [String]

    public init(name: String, type: String = "", observations: [String] = []) {
        self.name = name
        self.type = type
        self.observations = observations
    }
}

/// 辅助模型产出的关系草稿。
public struct FactRelationDraft: Sendable, Equatable {
    public var source: String
    public var target: String
    public var label: String

    public init(source: String, target: String, label: String) {
        self.source = source
        self.target = target
        self.label = label
    }
}

public struct FactExtractionDraft: Sendable, Equatable {
    public var entities: [FactEntityDraft]
    public var relations: [FactRelationDraft]

    public init(entities: [FactEntityDraft] = [], relations: [FactRelationDraft] = []) {
        self.entities = entities
        self.relations = relations
    }

    public var isEmpty: Bool { entities.isEmpty && relations.isEmpty }
}

/// 校验层丢弃统计，供日志诊断抽取质量。
public struct FactExtractionSanitizeReport: Sendable, Equatable {
    public var droppedEntities = 0
    public var droppedObservations = 0
    public var droppedRelations = 0

    public var totalDropped: Int {
        droppedEntities + droppedObservations + droppedRelations
    }
}

// MARK: - 事实抽取校验

/// `FACT_EXTRACT_SYSTEM` 准入规则的确定性兜底。
///
/// 辅助模型默认是 2B 级小模型，即使 prompt 写满约束仍会概率性犯规；而写入侧
/// `GraphMemoryStore.createRelation` 会自动补齐缺失端点、`addObservations` 会
/// 自动创建实体，一旦放进来就是**永久噪声**（还会被语义检索召回、挤占后续
/// prompt）。因此模型输出必须先过这一层，只把「有具体观察的实体」与「两端
/// 都在本批实体里的具体关系」放行进库。
///
/// 规则与 `FACT_EXTRACT_SYSTEM` 一一对应，改 prompt 时须同步这里。
public enum FactExtractionValidator {

    public static let maxEntities = 5
    public static let maxObservationsPerEntity = 3
    public static let maxTotalObservations = 8
    public static let maxRelations = 3

    /// 清洗并返回可直接写入的草稿 + 丢弃统计。纯函数，便于单测固化。
    public static func sanitize(_ draft: FactExtractionDraft)
        -> (draft: FactExtractionDraft, report: FactExtractionSanitizeReport) {
        var report = FactExtractionSanitizeReport()
        var candidates: [FactEntityDraft] = []
        var observationBudget = maxTotalObservations

        for entity in draft.entities {
            guard isValidEntityName(entity.name) else {
                report.droppedEntities += 1
                report.droppedObservations += entity.observations.count
                continue
            }
            let name = normalizedName(entity.name)
            var kept: [String] = []
            var seen = Set<String>()
            for raw in entity.observations {
                let observation = normalizedObservation(raw)
                let key = observationKey(observation)
                guard isValidObservation(observation, entityName: name),
                      seen.insert(key).inserted else {
                    report.droppedObservations += 1
                    continue
                }
                kept.append(observation)
            }
            if kept.count > maxObservationsPerEntity {
                report.droppedObservations += kept.count - maxObservationsPerEntity
                kept = Array(kept.prefix(maxObservationsPerEntity))
            }
            if observationBudget <= 0 {
                report.droppedObservations += kept.count
                kept = []
            } else if kept.count > observationBudget {
                report.droppedObservations += kept.count - observationBudget
                kept = Array(kept.prefix(observationBudget))
            }
            observationBudget -= kept.count
            // 没有合格观察的实体绝不入库：写进去也无法被语义检索到。
            guard !kept.isEmpty else {
                report.droppedEntities += 1
                continue
            }
            candidates.append(FactEntityDraft(name: name,
                                              type: normalizedType(entity.type),
                                              observations: kept))
        }

        if candidates.count > maxEntities {
            let removed = candidates.suffix(candidates.count - maxEntities)
            report.droppedEntities += removed.count
            report.droppedObservations += removed.reduce(0) { $0 + $1.observations.count }
            candidates = Array(candidates.prefix(maxEntities))
        }

        // 关系端点必须落在本批最终保留的实体里：否则 createRelation 会
        // 现场补建一个空观察实体，正是「无意义实体」的主要来源。
        var canonicalNames: [String: String] = [:]
        for entity in candidates { canonicalNames[entity.name.lowercased()] = entity.name }

        var relations: [FactRelationDraft] = []
        var seenRelations = Set<String>()
        for relation in draft.relations {
            guard let source = canonicalNames[normalizedName(relation.source).lowercased()],
                  let target = canonicalNames[normalizedName(relation.target).lowercased()],
                  source.lowercased() != target.lowercased() else {
                report.droppedRelations += 1
                continue
            }
            let label = normalizedObservation(relation.label)
            guard isValidLabel(label) else {
                report.droppedRelations += 1
                continue
            }
            let key = "\(source.lowercased())|\(target.lowercased())|\(label.lowercased())"
            guard seenRelations.insert(key).inserted else {
                report.droppedRelations += 1
                continue
            }
            relations.append(FactRelationDraft(source: source, target: target, label: label))
        }
        if relations.count > maxRelations {
            report.droppedRelations += relations.count - maxRelations
            relations = Array(relations.prefix(maxRelations))
        }

        return (FactExtractionDraft(entities: candidates, relations: relations), report)
    }

    // MARK: - 实体名

    private static let genericEntityNames: Set<String> = [
        // 中文泛指名词（没有具体指代，写进去无法被有效召回）
        "项目", "代码", "问题", "方案", "工具", "文件", "数据", "任务", "内容", "系统",
        "模型", "对话", "结果", "错误", "方法", "步骤", "信息", "东西", "事情", "时间",
        "地点", "名称", "类型", "列表", "输出", "输入", "请求", "回答", "助手", "智能体",
        "测试", "配置", "环境", "状态", "当前", "目标", "需求", "功能", "模块", "接口",
        "参数", "变量", "函数", "类", "库", "框架", "版本", "文档", "命令", "操作",
        "处理", "修改", "更新", "新增", "删除", "查询", "检索", "分析", "总结", "其他",
        "其它", "示例", "例子", "样例", "上文", "这段", "数据库", "日志", "脚本",
        // 英文泛指
        "project", "code", "issue", "problem", "tool", "file", "data", "task", "content",
        "system", "model", "conversation", "result", "error", "method", "step", "info",
        "item", "thing", "stuff", "unknown", "none", "null", "name", "type", "value",
        "text", "example", "test", "config", "environment", "status", "other", "others",
        "database", "log", "script",
    ]

    /// 助手自身工具/能力名，不是用户事实。
    private static let assistantToolNames: Set<String> = [
        "rag_tools", "file_tools", "shell_run", "code_run", "web_scan", "long_memory",
        "update_plan", "update_working_checkpoint", "start_long_term_update",
        "ask_user", "skill_tools", "memory_graph", "rag_search", "code_interpreter", "mcp",
    ]

    private static let placeholderNames: Set<String> = [
        "xxx", "todo", "tbd", "placeholder", "example", "sample", "示例", "占位",
    ]

    private static func isValidEntityName(_ raw: String) -> Bool {
        let name = normalizedName(raw)
        guard name.count >= 2, name.count <= 40 else { return false }
        guard !name.contains("@@"), !name.contains("<"), !name.contains(">"),
              !name.contains("{"), !name.contains("}") else { return false }
        // 纯数字/纯符号没有指代价值。
        guard name.unicodeScalars.contains(where: { CharacterSet.letters.contains($0) })
            else { return false }
        let lower = name.lowercased()
        return !genericEntityNames.contains(lower)
            && !assistantToolNames.contains(lower)
            && !placeholderNames.contains(lower)
    }

    private static func isUserEntity(_ name: String) -> Bool {
        let lower = name.lowercased()
        return lower == "user" || name == "用户"
    }

    // MARK: - 观察

    /// 相对时间、过程性对话动词、对话引用：都会让观察脱离上下文后失去意义。
    private static let forbiddenObservationSubstrings: [String] = [
        "@@", "<entity", "<observation", "placeholder",
        "在对话中", "这段对话", "上文", "上述", "用户说", "助手说", "模型推测",
        "今天", "昨天", "明天", "刚才", "之前", "上次", "本次", "这轮", "目前",
        "现在", "正在", "接下来", "稍后", "待会", "暂时", "临时",
        "讨论了", "提到了", "询问了", "建议", "可能", "也许", "应该", "计划",
        "打算", "准备", "希望可以", "不确定",
    ]

    /// 没有可复述信息的空话句式（长度看似达标，但脱离上下文后毫无价值）。
    private static let emptyObservationPhrases: [String] = [
        "是一个", "是一种", "是一件", "很重要", "很有用", "非常重要", "比较好",
        "还不错", "没关系", "是一个项目", "是一个工具", "被使用", "被讨论",
        "is a ", "is an ", "is important",
    ]

    private static let pronounPrefixes: [String] = [
        "他", "她", "它", "他们", "这个", "那个", "其", "此", "该", "这", "那",
        "上述", "前者", "后者",
        "he ", "she ", "it ", "they ", "this ", "that ", "these ", "those ",
    ]

    private static func isValidObservation(_ observation: String,
                                           entityName: String) -> Bool {
        guard observation.count >= 8, observation.count <= 160 else { return false }
        let lower = observation.lowercased()
        for forbidden in forbiddenObservationSubstrings
            where lower.contains(forbidden.lowercased()) { return false }
        for phrase in emptyObservationPhrases
            where lower.contains(phrase.lowercased()) { return false }
        for prefix in pronounPrefixes where lower.hasPrefix(prefix.lowercased()) {
            return false
        }
        // 自包含：必须带明确主语（实体名），用户本人实体允许「用户」这一中文写法。
        let nameLower = entityName.lowercased()
        let userAlias = isUserEntity(entityName)
            && (observation.hasPrefix("用户") || lower.hasPrefix("user"))
        guard lower.hasPrefix(nameLower) || lower.contains(nameLower) || userAlias
            else { return false }
        // 去掉实体名后仍有实质内容，挡住「是一个项目」「很重要」这类空话。
        let residue = observation
            .replacingOccurrences(of: entityName, with: "", options: .caseInsensitive)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return residue.count >= 6
    }

    // MARK: - 关系

    private static let genericRelationLabels: Set<String> = [
        "related", "related_to", "related to", "associated", "associated with",
        "association", "has", "have", "is", "are", "with", "and", "of", "to",
        "unknown", "none", "null",
        "有关", "关联", "相关", "涉及", "包含", "是", "有", "关系", "联系", "无",
        "其他", "其它",
    ]

    private static func isValidLabel(_ label: String) -> Bool {
        guard label.count >= 2, label.count <= 30 else { return false }
        return !genericRelationLabels.contains(label.lowercased())
    }

    private static let allowedTypes: Set<String> = [
        "person", "organization", "project", "place", "tool", "concept", "other",
    ]

    private static func normalizedType(_ raw: String) -> String {
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return allowedTypes.contains(value) ? value : "other"
    }

    // MARK: - 归一化

    private static func normalizedName(_ raw: String) -> String {
        collapseWhitespace(raw)
    }

    private static func normalizedObservation(_ raw: String) -> String {
        collapseWhitespace(raw)
    }

    private static func collapseWhitespace(_ raw: String) -> String {
        raw.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private static func observationKey(_ observation: String) -> String {
        observation.lowercased().filter { !$0.isWhitespace && !$0.isPunctuation }
    }
}
