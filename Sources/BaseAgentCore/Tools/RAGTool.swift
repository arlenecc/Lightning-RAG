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
/// 臆造 `memory/猫图鉴.pdf/6 猫图鉴.md` 反复用 file_read / shell_run cat 读取，
/// 全失败后仍不作答。规则贴着结果给，比只写在 system prompt 有效得多。
let ragUsageNote = """

---
【知识库读取规则】以上即知识库返回的全部可用内容，请直接据此作答（需要更多细节就再用
`rag_search` 换措辞/换章节，或用 `rag_outline` 看大纲）。「来源 / 章节 / 文档名 / 文档 ID」
只是向量库的索引标识，**不是磁盘文件路径**；源文件（PDF、扫描件等）已切片入库，磁盘上
不存在由它们拼出的 `.md` 文本。禁止用 `file_read`、`file_patch`、`shell_run`、`code_run`
或 `cat` 去读知识库文档或臆造路径——这类调用必然失败，也不能改用文件工具绕过。
"""

func ragToolOK(_ text: String) -> ToolResult {
    .ok(text + ragUsageNote)
}

// MARK: - rag_search

/// 知识库检索工具：等价于 Python `RagSearchTool`。
public final class RagSearchTool: AgentTool {
    public let name = "rag_search"
    public let description = "从本地知识库（RAG）检索与 query 相关的内容，可限定来源文件。query 请使用用户原始措辞，不要翻译或音译。返回带来源的 Markdown 片段，片段即正文、请直接据此作答；来源/章节只是索引标识、不是磁盘文件路径，禁止用 file_read/shell_run/code_run 读取源文件。"
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
                "description": .string("候选片段数，默认 3，范围 1-10；开启重排序时"
                    + "最终条数由设置里的 TopN 决定"),
            ]),
        ]),
        "required": .array([.string("query")]),
    ])

    private let engine: RAGEngine

    public init(config: AgentConfig,
                callbacks: AgentCallbacks? = nil,
                registry: ToolRegistry? = nil,
                engine: RAGEngine) {
        self.engine = engine
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        let query = args.trimmedString("query")
        guard !query.isEmpty else { return .fail("缺少参数: query") }
        let topK = args.boundedInt("top_k", default: 3, range: 1...10)
        let source = args.trimmedString("source")
        do {
            let output = try await engine.searchFormatted(
                query: query,
                topK: topK,
                source: source.isEmpty ? nil : source)
            return ragToolOK(output)
        } catch {
            return .fail("知识库检索失败: \(error.localizedDescription)")
        }
    }
}

// MARK: - rag_outline

/// 文档大纲工具：等价于 Python `RagOutlineTool`。
public final class RagOutlineTool: AgentTool {
    public let name = "rag_outline"
    public let description = "查看已索引文档的章节结构（目录）与内容摘要，用于先整体了解一本书再定位细节；不传 book_name 时列出全部已索引文档。章节标题/摘要是向量库索引内容，不是磁盘文件路径，禁止用文件工具读取。"
    public let parameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([
            "book_name": .object([
                "type": .string("string"),
                "description": .string("文档名称（支持部分匹配）；留空则列出所有文档"),
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
        let name = args.trimmedString("book_name")
        if name.isEmpty {
            let list = engine.listDocuments(limit: 100)
            guard list.total > 0 else {
                return ragToolOK("知识库中还没有已索引文档。")
            }
            var lines = ["已索引文档（共 \(list.total) 个）:"]
            lines.append(contentsOf: list.names.map {
                "  - \((($0 as NSString).lastPathComponent))"
            })
            if list.total > list.names.count {
                lines.append("  …（仅显示前 \(list.names.count) 个）")
            }
            return ragToolOK(lines.joined(separator: "\n"))
        }

        switch engine.lookupDocument(name) {
        case .notFound:
            return .fail("未找到文档: \(name)")
        case .ambiguous(let candidates):
            var lines = ["匹配到多个文档（\(candidates.count) 个），请给 rag_outline 传更完整的文档名:"]
            for candidate in candidates.prefix(20) {
                lines.append("  - \(candidate.docName)（ID: \(candidate.docID)）")
            }
            return ragToolOK(lines.joined(separator: "\n"))
        case .found(let record):
            guard let enriched = engine.getEnrichedOutline(for: record) else {
                return .fail("文档 \(record.docName) 没有可用大纲")
            }
            return ragToolOK(Self.format(enriched))
        }
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
    public let description = "查看知识库索引状态：知识库目录、RAG 目录、切片数、已索引文件列表。"
    public let parameters: JSONValue = .object([
        "type": .string("object"),
        "properties": .object([:]),
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
