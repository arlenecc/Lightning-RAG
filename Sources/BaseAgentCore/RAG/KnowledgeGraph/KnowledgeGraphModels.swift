import Foundation

// MARK: - 知识图谱数据模型

/// 知识图谱实体（对应 `entities` 表）。
///
/// 同名实体（大小写不敏感）在多个切片中被抽到时聚合为一条：
/// `description` 视情况交 LLM 归纳，`sourceIDs` 去重拼接，`properties` 合并。
public struct KGEntity: Codable, Sendable, Equatable, Identifiable {
    /// uuid 字符串。
    public var id: String
    public var name: String
    /// 实体类型/分类（Person / Organization / ... / Other / UNKNOWN）。
    public var type: String
    /// 实体的详细描述（核心字段）。
    public var description: String
    /// 扩展属性（文件路径等，通常存 JSON）。
    public var properties: [String: String]
    /// 该实体提取自哪些原始切片（chunk id，去重）。
    public var sourceIDs: [String]
    public var createdAt: Double
    public var updatedAt: Double

    public init(id: String, name: String, type: String = "",
                description: String = "", properties: [String: String] = [:],
                sourceIDs: [String] = [], createdAt: Double = 0, updatedAt: Double = 0) {
        self.id = id
        self.name = name
        self.type = type
        self.description = description
        self.properties = properties
        self.sourceIDs = sourceIDs
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}

/// 知识图谱关系（对应 `relationships` 表）。
///
/// 关系是**无向**的：查询时按三元组（无向边归一化）去重；内存图谱中每条关系
/// 生成正序 + 反序两条有向边，保证邻居遍历/度数统计对称。
public struct KGRelationship: Codable, Sendable, Equatable, Identifiable {
    /// uuid 字符串（有向边唯一标识；同一无向关系只存一条）。
    public var id: String
    /// 源实体名（起点）。
    public var srcEntity: String
    /// 目标实体名（终点）。
    public var dstEntity: String
    /// 关系名称/谓词（抽取产出的 keywords，逗号分隔）。
    public var relationType: String
    /// 关系的详细描述（核心字段）。
    public var description: String
    /// 权重 = 支持该关系的证据切片数量（缺失兜底 1）。
    public var weight: Double
    /// 边的属性（时间戳、条件、文件路径等）。
    public var properties: [String: String]
    /// 该关系提取自哪些原始切片（chunk id，去重）。
    public var sourceIDs: [String]
    public var createdAt: Double
    public var updatedAt: Double

    public init(id: String, srcEntity: String, dstEntity: String,
                relationType: String = "", description: String = "",
                weight: Double = 1, properties: [String: String] = [:],
                sourceIDs: [String] = [], createdAt: Double = 0, updatedAt: Double = 0) {
        self.id = id
        self.srcEntity = srcEntity
        self.dstEntity = dstEntity
        self.relationType = relationType
        self.description = description
        self.weight = weight
        self.properties = properties
        self.sourceIDs = sourceIDs
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    /// 无向三元组归一化键：端点按字典序排列后拼接谓词，用于去重。
    public var canonicalKey: String {
        Self.canonicalKey(src: srcEntity, dst: dstEntity, type: relationType)
    }

    /// 无向实体对键（不含谓词）。
    ///
    /// 同一对实体在不同切片里可能用不同措辞描述同一条关系，关系身份应按
    /// 实体对归一，keywords 作为同一条边的属性合并，而不是产生平行边。
    public var pairKey: String { Self.pairKey(srcEntity, dstEntity) }

    public static func pairKey(_ src: String, _ dst: String) -> String {
        let a = src.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let b = dst.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let (lo, hi) = a <= b ? (a, b) : (b, a)
        return "\(lo)|\(hi)"
    }

    /// 反向向量的 target_id 后缀：关系无向，正序与反序各存一条向量。
    public static let reverseVectorSuffix = "#rev"

    /// 反序向量 id（关键词相同，源/目标互换）。
    public var reverseVectorID: String { id + Self.reverseVectorSuffix }

    public static func canonicalKey(src: String, dst: String, type: String) -> String {
        let a = src.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let b = dst.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let t = type.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let (lo, hi) = a <= b ? (a, b) : (b, a)
        return "\(lo)|\(hi)|\(t)"
    }
}

// MARK: - 向量目标类型

/// kg_vectors 表里每条向量指向的对象类型。
public enum KGVectorTargetType: String, Codable, Sendable, Equatable {
    case entity
    case relationship
}

// MARK: - 抽取结果

/// 单个切片抽取出的实体。
public struct KGExtractedEntity: Sendable, Equatable {
    public var name: String
    public var type: String
    public var description: String

    public init(name: String, type: String, description: String) {
        self.name = name
        self.type = type
        self.description = description
    }
}

/// 单个切片抽取出的关系（无向）。
public struct KGExtractedRelation: Sendable, Equatable {
    public var source: String
    public var target: String
    /// 关系关键词（逗号分隔）。
    public var keywords: String
    public var description: String

    public init(source: String, target: String, keywords: String, description: String) {
        self.source = source
        self.target = target
        self.keywords = keywords
        self.description = description
    }
}

/// 一次抽取的完整产出。
public struct KGExtractionResult: Sendable, Equatable {
    public var entities: [KGExtractedEntity]
    public var relations: [KGExtractedRelation]

    public init(entities: [KGExtractedEntity] = [], relations: [KGExtractedRelation] = []) {
        self.entities = entities
        self.relations = relations
    }

    public var isEmpty: Bool { entities.isEmpty && relations.isEmpty }
}

// MARK: - 抽取格式

/// 实体/关系抽取的输出格式。
///
/// - `delimited`：LightRAG 风格的分隔符行（每行一条记录，`<|#|>` 分隔字段），
///   带 `<|COMPLETE|>` 与重复行收流，适合更小或只认行格式的模型。
/// - `json`：单个 JSON 对象（`entities` / `relationships` 数组），结构清晰、
///   解析歧义少，2B 实测实体与关系更完整，默认值；解析失败时该切片按空结果处理。
public enum KGExtractionFormat: String, Codable, Sendable, CaseIterable {
    case delimited
    case json

    public init(stored: String) {
        self = stored.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "json"
            ? .json : .delimited
    }
}

// MARK: - 抽取限额

/// 单个切片的抽取上限（由 config.json 的 `rag_kg_max_*` 参数控制）。
public struct KGExtractionLimits: Sendable, Equatable {
    /// 每个切片最多保留的实体数。
    public var maxEntitiesPerChunk: Int
    /// 每个切片最多保留的关系数。
    public var maxRelationsPerChunk: Int
    /// 实体/关系描述的最大字符数。
    public var maxDescriptionCharacters: Int

    public init(maxEntitiesPerChunk: Int = 20,
                maxRelationsPerChunk: Int = 50,
                maxDescriptionCharacters: Int = 100) {
        self.maxEntitiesPerChunk = max(1, maxEntitiesPerChunk)
        self.maxRelationsPerChunk = max(1, maxRelationsPerChunk)
        self.maxDescriptionCharacters = max(1, maxDescriptionCharacters)
    }

    public static let `default` = KGExtractionLimits()
}

// MARK: - 关键词抽取

/// 查询的关键词分解：高层主题（用于关系检索）与低层实体（用于实体检索）。
public struct KGKeywords: Sendable, Equatable {
    public var highLevel: [String]
    public var lowLevel: [String]

    public init(highLevel: [String] = [], lowLevel: [String] = []) {
        self.highLevel = highLevel
        self.lowLevel = lowLevel
    }
}

// MARK: - 检索候选与结果

/// 检索候选的类别。
public enum KGPassageKind: String, Sendable, Equatable {
    case entity
    case relation
    case chunk
}

/// 一个检索候选段（实体描述 / 关系三元组 / 原文切片）。
public struct KGPassage: Sendable, Equatable {
    /// 送入 reranker / 拼进 context 的正文。
    public var text: String
    public var kind: KGPassageKind
    /// 展示标签：实体名 / "src --type--> dst" / 文件名。
    public var label: String
    /// 向量余弦相似度（chunk 侧为 RRF/检索分）。
    public var score: Double
    /// 图谱度数（实体为节点度数；关系为两端点度数和；chunk 为 0）。
    public var degree: Int
    /// 关系权重（实体/chunk 为 0）。
    public var weight: Double
    /// 该候选关联的原始切片 id（归因反查）。
    public var sourceChunkIDs: [String]
    /// 关系候选的结构化三元组字段；实体/chunk 为空。
    public var srcEntity: String?
    public var dstEntity: String?
    public var relationType: String?

    public init(text: String, kind: KGPassageKind, label: String,
                score: Double, degree: Int = 0, weight: Double = 0,
                sourceChunkIDs: [String] = [],
                srcEntity: String? = nil, dstEntity: String? = nil,
                relationType: String? = nil) {
        self.text = text
        self.kind = kind
        self.label = label
        self.score = score
        self.degree = degree
        self.weight = weight
        self.sourceChunkIDs = sourceChunkIDs
        self.srcEntity = srcEntity
        self.dstEntity = dstEntity
        self.relationType = relationType
    }

    /// 无向归一的三元组键；优先用结构化字段，避免实体名含 `--` 时解析错位。
    func tripleKey() -> String {
        let src: String
        let dst: String
        let type: String
        if let s = srcEntity, let d = dstEntity {
            src = s.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            dst = d.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            type = (relationType ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        } else {
            let parts = label.components(separatedBy: "--")
            guard parts.count >= 3 else { return label.lowercased() }
            src = parts[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            dst = parts[2].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            type = parts[1].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        }
        let (lo, hi) = src <= dst ? (src, dst) : (dst, src)
        return "\(lo)|\(hi)|\(type)"
    }
}

/// 知识图谱候选的最终排序：向量余弦 + 图谱度数（中心性代理）+ 关系权重。
///
/// 度数/权重先按候选集合内的最大值归一化，再加权求和，避免大图度数悬殊时
/// 直接把余弦淹没。
public enum KGCandidateRanking {
    static let entityCosineWeight = 0.8
    static let entityDegreeWeight = 0.2
    static let relationCosineWeight = 0.7
    static let relationRankWeight = 0.2
    static let relationEvidenceWeight = 0.1

    public static func entities(_ passages: [KGPassage]) -> [KGPassage] {
        let maxDegree = max(1, passages.map(\.degree).max() ?? 1)
        return passages.sorted { lhs, rhs in
            let ls = score(lhs, cosineWeight: entityCosineWeight,
                           centralityWeight: entityDegreeWeight,
                           centrality: Double(lhs.degree) / Double(maxDegree))
            let rs = score(rhs, cosineWeight: entityCosineWeight,
                           centralityWeight: entityDegreeWeight,
                           centrality: Double(rhs.degree) / Double(maxDegree))
            if ls != rs { return ls > rs }
            if lhs.score != rhs.score { return lhs.score > rhs.score }
            return lhs.label < rhs.label
        }
    }

    public static func relations(_ passages: [KGPassage]) -> [KGPassage] {
        let maxRank = max(1, passages.map(\.degree).max() ?? 1)
        let maxWeight = max(1, passages.map(\.weight).max() ?? 1)
        return passages.sorted { lhs, rhs in
            let ls = relationScore(lhs, maxRank: maxRank, maxWeight: maxWeight)
            let rs = relationScore(rhs, maxRank: maxRank, maxWeight: maxWeight)
            if ls != rs { return ls > rs }
            if lhs.degree != rhs.degree { return lhs.degree > rhs.degree }
            if lhs.weight != rhs.weight { return lhs.weight > rhs.weight }
            return lhs.label < rhs.label
        }
    }

    private static func score(_ p: KGPassage, cosineWeight: Double,
                              centralityWeight: Double, centrality: Double) -> Double {
        cosineWeight * p.score + centralityWeight * centrality
    }

    private static func relationScore(_ p: KGPassage, maxRank: Int,
                                      maxWeight: Double) -> Double {
        relationCosineWeight * p.score
            + relationRankWeight * (Double(p.degree) / Double(maxRank))
            + relationEvidenceWeight * (p.weight / maxWeight)
    }
}

/// 知识图谱检索的原始召回（reranker 精排前的候选池）。
public struct KGRetrievalCandidates: Sendable {
    /// 实体候选（已按余弦 + 度数 + 权重排序、去重、预算截断）。
    public var entities: [KGPassage]
    /// 关系候选。
    public var relations: [KGPassage]
    /// 原文切片候选（查询向量直接召回，优先级最高）。
    public var chunks: [KGPassage]

    public init(entities: [KGPassage] = [], relations: [KGPassage] = [],
                chunks: [KGPassage] = []) {
        self.entities = entities
        self.relations = relations
        self.chunks = chunks
    }

    /// 三路按实体/关系/原文顺序合并后的统一候选池（供 reranker 消费）。
    public var allPassages: [KGPassage] {
        entities + relations + chunks
    }

    public var isEmpty: Bool { entities.isEmpty && relations.isEmpty && chunks.isEmpty }
}
