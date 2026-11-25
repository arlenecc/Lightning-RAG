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
}

// MARK: - RAG 结果使用说明

/// 贴在每条 RAG 结果尾部：知识库问答是「向量库检索」，不是「读文件」。
/// 旧版模型把 `来源: 猫图鉴.pdf` + `章节: 6 猫图鉴` 当成磁盘上的章节文件，
/// 臆造 `memory/猫图鉴.pdf/6 猫图鉴.md` 反复用 file_read / shell_run cat 读取。
/// 但规则必须**精确限定在读取已索引文档这一件事**：曾有一版写成
/// 「知识库问答只用 rag_*，禁止用文件工具」，模型泛化成全局能力边界，
/// 连 `web_scan` 抓外部网站都不敢调用，反问用户「我没有访问外网的权限」。
let ragUsageNote = """

---
【知识库读取规则（仅适用于读取已索引文档）】以上即本次检索返回的全部可用内容，
请直接据此作答（需要更多细节就再用 `rag_tools(op="search")` 换措辞/换章节，或
`rag_tools(op="outline")` 看大纲）。「来源 / 章节 / 文档名 / 文档 ID」是向量库索引标识、
不是磁盘文件路径；源文件（PDF、扫描件等）已切片入库，磁盘上不存在由它们拼出的 `.md`，
去读必然失败、无需尝试。本规则不限制其它工具：访问外部网站用 `web_scan`，读写本地文件、
执行命令用 `file_tools`/`shell_run`/`code_run`，均照常可用。
"""

func ragToolOK(_ text: String, metrics: [String: Double] = [:]) -> ToolResult {
    .init(success: true, output: text + ragUsageNote, metrics: metrics)
}

// MARK: - rag_search

/// 知识库检索工具：等价于 Python `RagSearchTool`。
///
/// 检索即展示（Retrieval-as-Presentation）：默认检索与极速检索模式下，召回切片
/// 先与问句做二元相关度判定，只有相关的切片进入上下文；每条切片以「问句相关
/// top5 句 + 核心代表句」压缩后提供，不相关切片被过滤。全部采纳切片通过
/// `AgentCallbacks.onRetrievedChunks` 推给 UI，作为气泡附在回复后供用户点开。
public final class RagSearchTool: AgentTool {
    public let name = "rag_search"
    public let description = "从本地知识库（RAG）检索与 query 相关的内容，可限定来源文件。query 请使用用户原始措辞，不要翻译或音译。默认检索与极速检索模式会先判定切片与问句的相关性，只把相关切片以「问句相关句 + 核心代表句」压缩进上下文，不相关切片会被过滤；其它检索模式只返回最相关的 1 条切片全文。如判定结果不足以回答，请换措辞再次检索。来源/章节只是索引标识、不是磁盘文件路径，无需再找原文件。此说明只针对知识库文档，不影响 web_scan / file_tools / shell_run 等其它工具。"
    public let parameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([
            "query": .object([
                "type": .string("string"),
                "description": .string("检索问题或关键词"),
            ]),
            "source": .object([
                "type": .string("string"),
                "description": .string("只检索指定来源文件（可选，支持文件名子串）"),
            ]),
            "top_k": .object([
                "type": .string("integer"),
                "description": .string("召回候选切片数（决定展示给用户的气泡数量），"
                    + "默认 3，范围 1-10；开启重排序时由设置里的 TopN 决定"),
            ]),
        ]),
        "required": .array([.string("query")]),
    ])

    private let engine: RAGEngine
    /// 多分区联合检索处理器：非空时优先使用它（多分区挂载），否则退回单 engine。
    public typealias SearchHandler = (_ query: String, _ topK: Int, _ source: String?) async throws -> RAGRetrievalOutcome
    private let searchHandler: SearchHandler?
    /// 极速检索模式：完成度依据是「判定存在相关切片」而非召回最高分。
    private let usesRelevanceConfirmation: Bool
    /// 检索即展示：召回切片通过回调推给 UI 气泡；无 UI 调用方为 nil。
    private weak var callbacks: AgentCallbacks?

    public init(config: AgentConfig,
                callbacks: AgentCallbacks? = nil,
                registry: ToolRegistry? = nil,
                engine: RAGEngine,
                searchHandler: SearchHandler? = nil) {
        self.engine = engine
        self.searchHandler = searchHandler
        self.usesRelevanceConfirmation = config.colbertReaderEnabled
        self.callbacks = callbacks
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let query = args.trimmedString("query")
        guard !query.isEmpty else { return .fail("缺少参数: query") }
        let topK = args.boundedInt("top_k", default: 3, range: 1...10)
        let source = args.trimmedString("source")
        do {
            let outcome: RAGRetrievalOutcome
            if let handler = searchHandler {
                outcome = try await handler(query, topK, source.isEmpty ? nil : source)
            } else {
                outcome = try await engine.searchWithCitations(
                    query: query,
                    topK: topK,
                    source: source.isEmpty ? nil : source)
            }
            if !outcome.citations.isEmpty {
                callbacks?.onRetrievedChunks(outcome.citations)
            }
            // 完成度确认指标（内核按「任一成立」放行）：
            // - 向量检索：召回最高分 > 0.5，或判定确认存在相关切片；
            // - 极速检索：只认判定确认（MaxSim 分数与阈值不可比，不输出
            //   rag_max_score，避免全部不相关时仍被最高分误放行）。
            let confirmed: Double = outcome.relevanceConfirmed ? 1 : 0
            // 没有任何采纳切片时 maxScore 必须归零：过滤前的召回最高分不能
            // 证明「材料已就位」，否则内核会凭它放行收尾、模型反复空搜。
            let exportedMaxScore = outcome.citations.isEmpty ? 0 : outcome.maxScore
            let metrics: [String: Double] = usesRelevanceConfirmation
                ? ["rag_relevance_confirmed": confirmed]
                : ["rag_max_score": exportedMaxScore,
                   "rag_relevance_confirmed": confirmed]
            return ragToolOK(outcome.context, metrics: metrics)
        } catch {
            return .fail("知识库检索失败: \(error.localizedDescription)")
        }
    }
}

// MARK: - rag_outline

/// 单文档概览工具：只看某一本已索引文档的目录结构、逐章摘要与全书摘要。
/// 文档清单/工作区状态由 RagStatusTool 负责，不在本工具内提供。
public final class RagOutlineTool: AgentTool {
    public let name = "rag_outline"
    public let description = "快速概览**某一本已索引文档**：目录结构、各章摘要、全书摘要。必须传具体 book_name（支持部分匹配，跨全部分区查找）。不负责文档清单/工作区状态——列文档、看分区收录了哪些书请用 rag_tools(op=\"status\")。章节标题/摘要是向量库索引内容、不是磁盘文件路径，无需再找原文件。此说明只针对知识库文档，不影响 web_scan / file_tools / shell_run 等其它工具。"
    public let parameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([
            "book_name": .object([
                "type": .string("string"),
                "description": .string("必填：要概览的具体文档名（支持部分匹配，跨所有已挂载分区查找）；列文档清单请用 rag_tools(op=\"status\")"),
            ]),
        ]),
        "required": .array([.string("book_name")]),
    ])

    private let engine: RAGEngine
    /// 其它挂载分区的引擎：默认库查不到 book_name 时遍历它们（多分区联合大纲）。
    private let extraEngines: [RAGEngine]

    public init(config: AgentConfig,
                callbacks: AgentCallbacks? = nil,
                registry: ToolRegistry? = nil,
                engine: RAGEngine,
                extraEngines: [RAGEngine] = []) {
        self.engine = engine
        self.extraEngines = extraEngines
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let name = args.trimmedString("book_name")
        // 本工具只做单文档概览，不再提供"不传就列全部文档"能力——清单/状态
        // 完全交给 rag_tools(op="status")，避免模型把两个 op 的职责混起来。
        guard !name.isEmpty else {
            return .fail("缺少必填参数 book_name：请提供要概览的具体文档名；"
                + "若只是想列出/清点已索引文档，请改用 rag_tools(op=\"status\")。")
        }

        // 多分区惰性查找：先查默认库，命中即返回，不再对其它分区做
        // 归一化全表兜底（旧实现用 map 提前算完所有分区，命中时也白扫）。
        func collectAmbiguous(_ list: [DocumentRecord], into lines: inout [String]) {
            lines.append(contentsOf: list.prefix(20).map {
                "  - \($0.docName)（ID: \($0.docID)）"
            })
        }
        var ambiguousLines: [String] = []
        let firstLookup = engine.lookupDocument(name)
        if case .found(let record) = firstLookup,
           let enriched = engine.getEnrichedOutline(for: record) {
            return ragToolOK(Self.format(enriched))
        }
        if case .ambiguous(let list) = firstLookup {
            collectAmbiguous(list, into: &ambiguousLines)
        }
        // 默认库未命中/仅歧义时才遍历其它挂载分区（book 可能在非默认库，
        // 如《病毒星球》在自然分区而默认库是社会分区）。
        for eng in extraEngines {
            switch eng.lookupDocument(name) {
            case .found(let record):
                if let enriched = eng.getEnrichedOutline(for: record) {
                    return ragToolOK(Self.format(enriched))
                }
                // 该分区有此文档但没有大纲，继续找下一个。
            case .ambiguous(let list):
                collectAmbiguous(list, into: &ambiguousLines)
            case .notFound:
                break
            }
        }
        if !ambiguousLines.isEmpty {
            var lines = ["匹配到多个文档（\(ambiguousLines.count) 个），请给 rag_tools(op=\"outline\") 传更完整的文档名:"]
            lines.append(contentsOf: ambiguousLines)
            return ragToolOK(lines.joined(separator: "\n"))
        }
        // 失败时也给出路：模型常把分区名（社会/自然/历史）或「全部文档」
        // 这类清单意图当成 book_name 传进来，直接 .fail 会让它反复重试
        // outline 直到轮次耗尽。明确指向 status，一次就能自纠。
        return .fail("未找到文档: \(name)。若你想列出/清点已索引文档（含分区名如「\(name)」），"
            + "请改用 rag_tools(op=\"status\")，它返回每个分区的文档名清单；"
            + "本工具只接具体文档名看单本大纲。")
    }

    // MARK: - 输出

    private static func format(_ enriched: EnrichedDocumentOutline) -> String {
        let record = enriched.record
        let parsedChapters = parseChapters(record.chapters)
        // 任务记录的是当前目录的章节总数；越界的旧摘要（目录收缩遗留）
        // 不能算进进度，也不能出现在输出里。
        let total = max(enriched.jobProgress.chapterCount, parsedChapters.count)
        let summaries = total > 0
            ? enriched.chapterSummaries.filter { $0.sectionIndex < total }
            : []
        let completed = summaries.filter { !$0.summary.isEmpty }.count
        let hasBookSummary = !(enriched.bookSummary?.summary.isEmpty ?? true)
        let state: String
        if hasBookSummary {
            state = "全书摘要已完成"
        } else if total == 0 {
            state = "无可识别章节"
        } else if completed < total {
            state = "后台仍在补充"
        } else {
            state = "等待生成全书摘要"
        }

        var lines = ["文档: \(record.docName)", "文档 ID: \(record.docID)"]
        lines.append("章节摘要进度: \(completed)/\(total)（\(state)）")

        if let book = enriched.bookSummary, !book.summary.isEmpty {
            lines.append("")
            lines.append("全书摘要:")
            lines.append(book.summary)
        } else if !summaries.isEmpty {
            lines.append("")
            lines.append("内容概览（基于已完成章节摘要）:")
            for summary in summaries.prefix(20) where !summary.summary.isEmpty {
                lines.append("- [\(summary.heading)] \(summary.summary)")
            }
            if summaries.count > 20 {
                lines.append("…（仅显示前 20 章摘要）")
            }
        } else {
            let digest = enriched.repairedTOC
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !digest.isEmpty {
                lines.append("")
                lines.append("内容摘要:")
                lines.append(String(digest.prefix(20_000)))
            }
        }

        let tree = chapterTree(enriched, summaries: summaries)
        if !tree.isEmpty {
            lines.append("")
            lines.append("章节结构:")
            lines.append(tree)
        }
        return lines.joined(separator: "\n")
    }

    private static func chapterTree(_ enriched: EnrichedDocumentOutline,
                                    summaries visibleSummaries: [ChapterSummaryRecord]) -> String {
        let nodes = parseChapters(enriched.record.chapters)
        let summaries = Dictionary(
            visibleSummaries.map { ($0.sectionIndex, $0) },
            uniquingKeysWith: { first, _ in first })
        if !nodes.isEmpty {
            let lines = nodes.enumerated().map { index, node -> String in
                let indent = String(repeating: "  ", count: max(0, min(node.level - 1, 5)))
                if let summary = summaries[index], !summary.summary.isEmpty {
                    return "\(indent)- [\(node.title)] \(summary.summary)"
                }
                return "\(indent)- \(node.title)"
            }
            return String(lines.joined(separator: "\n").prefix(20_000))
        }
        let summaryLines = visibleSummaries
            .sorted { $0.sectionIndex < $1.sectionIndex }
            .map { summary -> String in
                let indent = String(repeating: "  ", count: max(0, min(summary.level - 1, 5)))
                return summary.summary.isEmpty
                    ? "\(indent)- \(summary.title)"
                    : "\(indent)- [\(summary.title)] \(summary.summary)"
            }
        if !summaryLines.isEmpty {
            return String(summaryLines.joined(separator: "\n").prefix(20_000))
        }
        let raw = enriched.record.chapters
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return raw.isEmpty ? "" : String(raw.prefix(20_000))
    }

    private static func parseChapters(_ raw: String) -> [(level: Int, title: String)] {
        guard let data = raw.data(using: .utf8),
              let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            return []
        }
        return array.compactMap { item in
            guard let title = item["title"] as? String,
                  !title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                return nil
            }
            return (level: (item["level"] as? Int) ?? 1, title: title)
        }
    }
}

// MARK: - rag_status

/// 知识库状态工具：等价于 Python `RagStatusTool`。
public final class RagStatusTool: AgentTool {
    public let name = "rag_status"
    public let description = "查看知识库清单与索引状态：**任何罗列/清点已索引文档的问句（有哪些文档、列出所有文档、各分区收录了什么、具体都有哪些书）一律用本工具**，它返回每个分区的名字、切片数和全部文档名清单；另含 RAG 目录、向量后端等状态。这是清单类问题的唯一正确入口，不要用 op=\"search\"（只返回片段）或 op=\"outline\"（只看单本大纲）。"
    public let parameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([:]),
        "required": .array([]),
    ])

    private let engine: RAGEngine
    /// 多分区概览提供者：非空时在每个分区 status 后追加总览（名字/切片/文档）。
    public typealias ZoneSummaryProvider = () -> [(name: String, chunks: Int, sources: [String])]
    private let zoneSummaryProvider: ZoneSummaryProvider?

    public init(config: AgentConfig,
                callbacks: AgentCallbacks? = nil,
                registry: ToolRegistry? = nil,
                engine: RAGEngine,
                zoneSummaryProvider: ZoneSummaryProvider? = nil) {
        self.engine = engine
        self.zoneSummaryProvider = zoneSummaryProvider
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        // 只查一次分区总览：标题的「默认库」标注与文末总览共用。
        let zoneSummaries = zoneSummaryProvider?() ?? []
        let status = engine.status()
        let knowledgeBase = status["knowledge_base"]?.stringValue ?? ""
        let ragDir = status["rag_dir"]?.stringValue ?? ""
        let chunks = status["chunks_stored"]?.intValue ?? 0
        let backend = status["vector_backend"]?.stringValue ?? ""
        let sources = status["sources"]?.arrayValue?.compactMap(\.stringValue) ?? []
        let sourceCount = status["source_count"]?.intValue ?? sources.count
        let annIndex = status["ann_index"]?.stringValue ?? ""
        let annBuilding = status["ann_building"]?.boolValue ?? false
        let annBuildProgress = status["ann_build_progress"]?.doubleValue ?? 0
        let annBuildMessage = status["ann_build_message"]?.stringValue ?? ""
        let annStaleRows = status["ann_stale_rows"]?.intValue ?? 0
        let annEffectiveCoverage = status["ann_effective_coverage"]?.doubleValue ?? 0
        let elementType = status["vector_element_type"]?.stringValue ?? ""
        let matryoshkaDim = status["matryoshka_dim"]?.intValue ?? 0
        let rebuildRequired = status["vector_rebuild_required"]?.boolValue ?? false
        let rebuildReason = status["rebuild_reason"]?.stringValue ?? ""
        let binaryCoarse = status["binary_coarse"]?.stringValue ?? ""
        let binaryDimension = status["binary_dimension"]?.intValue ?? 0
        let binaryCoverage = status["binary_coverage"]?.doubleValue ?? 0
        let binaryNote = status["binary_note"]?.stringValue ?? ""
        let binaryANN = status["binary_ann"]?.stringValue ?? ""
        let binaryANNIndex = status["binary_ann_index"]?.stringValue ?? ""
        let binaryANNNote = status["binary_ann_note"]?.stringValue ?? ""
        let binaryBackend = status["binary_backend"]?.stringValue ?? ""
        let annBypassed = status["ann_bypassed"]?.boolValue ?? false

        // 多分区只返回简洁概览：每个分区的名字、规模与文档名，不再拼默认库的
        // 一长串详细状态（切片数/向量后端/ANN 等只反映单库，多库下既冗长又易误导）。
        if zoneSummaries.count > 1 {
            var overview: [String] = ["知识库状态：\(zoneSummaries.count) 个分区（默认跨全部分区联合检索）"]
            var totalChunks = 0
            var totalDocs = 0
            for s in zoneSummaries {
                totalChunks += s.chunks
                totalDocs += s.sources.count
                overview.append("")
                overview.append("【\(s.name)】\(s.chunks) 切片，\(s.sources.count) 文档")
                for source in s.sources.prefix(10) {
                    overview.append("  - \((source as NSString).lastPathComponent)")
                }
                if s.sources.count > 10 {
                    overview.append("  …（仅显示前 10 个，共 \(s.sources.count) 个）")
                }
            }
            overview.insert("合计：\(totalChunks) 切片 / \(totalDocs) 文档", at: 1)
            return .ok(overview.joined(separator: "\n"))
        }

        var lines: [String] = ["知识库状态:"]
        lines.append("  知识库目录: \(knowledgeBase.isEmpty ? "（未设置）" : knowledgeBase)")
        lines.append("  RAG 目录: \(ragDir)")
        lines.append("  已存储切片数: \(chunks)")
        if !backend.isEmpty { lines.append("  向量后端: \(backend)") }
        if !elementType.isEmpty {
            lines.append("  向量元素类型: \(elementType)"
                + (matryoshkaDim > 0 ? "，Matryoshka 截断 \(matryoshkaDim) 维" : ""))
        }
        if rebuildRequired {
            lines.append("  向量库待重建: \(rebuildReason.isEmpty ? "配置已变化" : rebuildReason)")
            lines.append("    提示: 向量检索暂停（全文检索仍可用），请在设置中执行「重建向量库」")
        }
        if !binaryCoarse.isEmpty {
            switch binaryCoarse {
            case "ready":
                let percent = Int((min(1, max(0, binaryCoverage)) * 100).rounded())
                let backend = binaryBackend.isEmpty ? "" : "\(binaryBackend)，"
                lines.append("  二值粗筛: 可用（\(backend)\(binaryDimension) bit，覆盖 \(percent)%"
                    + (annBypassed ? "，本次取代 ANN" : "") + "）")
            case "incomplete":
                let percent = Int((min(1, max(0, binaryCoverage)) * 100).rounded())
                let note = binaryNote.isEmpty
                    ? "覆盖 \(percent)%，请重新同步"
                    : binaryNote
                lines.append("  二值粗筛: 不可用（\(note)）")
            case "bypassed_bin":
                lines.append("  二值粗筛: 停用（bin 主存储仅支持单阶段检索）")
            case "off":
                lines.append("  二值粗筛: 关闭")
            default:
                lines.append("  二值粗筛: 不可用（"
                    + "\(binaryNote.isEmpty ? "原因未知" : binaryNote)）")
            }
        }
        if !binaryANN.isEmpty {
            switch binaryANN {
            case "ready", "tail":
                lines.append("  二值 ANN: \(binaryANNIndex)")
            case "not_built":
                lines.append("  二值 ANN: 未构建（本次回退 bin 表暴力汉明）")
            case "building":
                lines.append("  二值 ANN: 构建中")
            case "needs_rebuild":
                lines.append("  二值 ANN: 建议重建（\(binaryANNNote)）")
            case "incomplete", "unusable":
                lines.append("  二值 ANN: 不可用（"
                    + "\(binaryANNNote.isEmpty ? "原因未知" : binaryANNNote)）")
            default:
                break
            }
        }
        if !annIndex.isEmpty {
            lines.append("  ANN 索引: \(annIndex)"
                + (annBypassed ? "（本次由二值粗筛取代）" : ""))
        }
        if annStaleRows > 0 {
            lines.append(String(format: "  ANN 陈旧切片: %ld（有效覆盖 %.1f%%）",
                                annStaleRows, annEffectiveCoverage * 100))
        }
        if annBuilding {
            let percent = Int((min(1, max(0, annBuildProgress)) * 100).rounded())
            lines.append("  ANN 后台构建: 进行中 \(percent)%"
                + (annBuildMessage.isEmpty ? "" : " — \(annBuildMessage)"))
            lines.append("    提示: 构建期间检索/状态查询不受影响，已写完的块会在取消后保留")
        } else if !annBuildMessage.isEmpty {
            lines.append("  ANN 构建: \(annBuildMessage)")
        }
        lines.append("  已索引文件数: \(sourceCount)")
        if !sources.isEmpty {
            lines.append("  已索引文件:")
            for source in sources.prefix(50) {
                lines.append("    - \(source)")
            }
            if sourceCount > min(50, sources.count) {
                lines.append("    …（仅显示前 \(min(50, sources.count)) 个，共 \(sourceCount) 个）")
            }
        }
        if let error = status["error"]?.stringValue, !error.isEmpty {
            lines.append("  错误: \(error)")
        }
        return .ok(lines.joined(separator: "\n"))
    }
}

// MARK: - rag_ingest

/// 知识库构建工具：等价于 Python `RagIngestTool`。
public final class RagIngestTool: AgentTool {
    public let name = "rag_ingest"
    public let description = "把知识库目录中的文档构建/更新到 RAG 索引。增量处理，force=true 时强制重处理全部文件。"
    public let parameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([
            "force": .object([
                "type": .string("boolean"),
                "description": .string("是否强制重建全部索引（默认 false，仅处理变更文件）"),
            ]),
        ]),
        "required": .array([]),
    ])

    private let engine: RAGEngine

    public init(config: AgentConfig,
                callbacks: AgentCallbacks? = nil,
                registry: ToolRegistry? = nil,
                engine: RAGEngine) {
        self.engine = engine
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let force = args["force"]?.boolValue ?? false
        let stats = await engine.ingest(force: force)

        if stats.processed == 0, stats.skipped == 0, stats.chunks == 0,
           !stats.errors.isEmpty {
            return .fail(stats.errors.joined(separator: "\n"))
        }
        if stats.cancelled {
            return .fail("RAG 同步已取消。\(stats.summary)")
        }

        var lines: [String] = ["知识库索引完成:"]
        lines.append("  新处理文件数: \(stats.processed)")
        lines.append("  跳过(未变更): \(stats.skipped)")
        lines.append("  删除文件数: \(stats.removed)")
        lines.append("  去重副本数: \(stats.deduplicated)")
        lines.append("  生成切片数: \(stats.chunks)")
        if engine.annAutoBuildState.isRunning {
            lines.append("  ANN 索引: 正在后台构建（可用 rag_status 查看进度）")
        }
        if !stats.errors.isEmpty {
            lines.append("  失败: \(stats.errors.count) 个")
            for error in stats.errors.prefix(5) {
                lines.append("    - \(error)")
            }
        }
        return .ok(lines.joined(separator: "\n"))
    }
}
