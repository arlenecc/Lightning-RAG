import Foundation
import BaseAgentCore

// RAGBenchmark：用合成评估集对比「普通 RAG（embedding+reranker）」与
// 「检索阅读模式（ColBERT）」两种检索链路的准确率（hit@1 / hit@k / MRR）
// 与总耗时。两阶段顺序切换、复用同一 workspace，从而复用已下载/分发的模型。

// MARK: - 命令行参数

struct BenchmarkOptions {
    var corpus: String
    var topK: Int = 3
    var workspace: String
    var limit: Int?
    /// 模拟普通 embedding 模型缺失（验证检索阅读模式与 embedding 解耦）。
    var disableEmbedding: Bool = false
}

func parseArguments() -> BenchmarkOptions? {
    let args = CommandLine.arguments
    var corpus: String?
    var topK = 3
    var workspace: String?
    var limit: Int?
    var disableEmbedding = false

    var i = 1
    while i < args.count {
        let arg = args[i]
        switch arg {
        case "--corpus":
            i += 1
            if i < args.count { corpus = args[i] }
        case "--topk":
            i += 1
            if i < args.count, let n = Int(args[i]), n > 0 { topK = n }
        case "--workspace":
            i += 1
            if i < args.count { workspace = args[i] }
        case "--limit":
            i += 1
            if i < args.count, let n = Int(args[i]), n > 0 { limit = n }
        case "--no-embedding":
            disableEmbedding = true
        case "-h", "--help":
            printUsage()
            return nil
        default:
            break
        }
        i += 1
    }

    guard let corpus else {
        fputs("错误：缺少 --corpus <语料目录>\n", stderr)
        printUsage()
        return nil
    }
    let ws = workspace ?? "/tmp/rag_benchmark_\(Int(Date().timeIntervalSince1970))"
    return BenchmarkOptions(corpus: corpus, topK: topK, workspace: ws,
                            limit: limit, disableEmbedding: disableEmbedding)
}

func printUsage() {
    print("""
    用法: RAGBenchmark --corpus <语料目录> [选项]

    选项:
      --corpus <dir>     语料目录（扫描 .md/.markdown/.txt），必填
      --topk <k>         召回条数，默认 3
      --workspace <dir>  工作目录（索引/模型），默认 /tmp/rag_benchmark_<时间戳>
      --limit <n>        只取前 n 个文档（用于快速验证）
      --no-embedding     把普通 embedding 配置成不可用，验证检索阅读模式
                         在不依赖 embedding 模型时仍能正常入库与检索
      -h, --help         显示帮助

    说明:
      对每个文档自动生成 1~2 条合成查询（标题 + 首段摘录），ground truth 为
      文件名。分别用普通 RAG 与检索阅读模式入库并逐条检索，统计 hit@1 /
      hit@k / MRR 与平均、中位查询耗时，最后输出对比表格并写
      <workspace>/benchmark_report.json。
    """)
}

// MARK: - 查询集生成

struct Query {
    let text: String
    let gtSource: String
}

/// 来源归一化：取最后一段路径、去扩展名、转小写。
/// 普通 RAG 的 `SearchHit.source` 是绝对路径，检索阅读的 `RAGCitation.source`
/// 是入库时的相对路径（`file.relativePath`），两者 lastPathComponent 都是文件名。
func normalizeSource(_ s: String) -> String {
    let last = (s as NSString).lastPathComponent
    return (last as NSString).deletingPathExtension.lowercased()
}

func scanCorpus(_ dir: String, limit: Int?) -> [URL] {
    let fm = FileManager.default
    let base = URL(fileURLWithPath: dir)
    guard let enumerator = fm.enumerator(
        at: base,
        includingPropertiesForKeys: [.isRegularFileKey],
        options: [.skipsHiddenFiles, .skipsPackageDescendants]) else {
        return []
    }
    var files: [URL] = []
    for case let url as URL in enumerator {
        let ext = url.pathExtension.lowercased()
        guard ext == "md" || ext == "markdown" || ext == "txt" else { continue }
        files.append(url)
        if let limit, files.count >= limit { break }
    }
    files.sort { $0.path < $1.path }
    return files
}

/// 从单个文档生成合成查询：标题（首个 `# ` 行）+ 首段正文摘录。
func generateQueries(for url: URL) -> [Query] {
    guard let content = try? String(contentsOf: url, encoding: .utf8) else { return [] }
    let gt = normalizeSource(url.lastPathComponent)

    var title: String?
    var paragraphs: [String] = []
    var current = ""
    for line in content.components(separatedBy: .newlines) {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if title == nil, trimmed.hasPrefix("# ") {
            title = String(trimmed.dropFirst(2)).trimmingCharacters(in: .whitespaces)
            continue
        }
        if trimmed.isEmpty || trimmed.hasPrefix("#") {
            if !current.isEmpty {
                paragraphs.append(current)
                current = ""
            }
        } else {
            current += (current.isEmpty ? "" : " ") + trimmed
        }
    }
    if !current.isEmpty { paragraphs.append(current) }

    var queries: [Query] = []
    if let title, !title.isEmpty {
        queries.append(Query(text: title, gtSource: gt))
    }
    if let first = paragraphs.first(where: { $0.count >= 20 }) {
        queries.append(Query(text: String(first.prefix(80)), gtSource: gt))
    }
    return queries
}

// MARK: - 指标统计

struct QueryResult {
    let query: Query
    let hit: Bool
    let hitRank: Int?
    let elapsed: Double
    let error: String?
}

struct PhaseMetrics {
    let mode: String
    let queries: [QueryResult]
    let ingestSeconds: Double
    let ingestChunks: Int
    let ingestErrors: [String]

    var total: Int { queries.count }
    var hits: Int { queries.filter { $0.hit }.count }
    var hitAt1: Int { queries.filter { ($0.hitRank ?? 0) == 1 }.count }
    var hitAtK: Int { hits }
    var hitAt1Rate: Double { total > 0 ? Double(hitAt1) / Double(total) : 0 }
    var hitAtKRate: Double { total > 0 ? Double(hits) / Double(total) : 0 }
    var mrr: Double {
        guard total > 0 else { return 0 }
        let sum = queries.reduce(0.0) { acc, r in
            r.hitRank.map { acc + 1.0 / Double($0) } ?? acc
        }
        return sum / Double(total)
    }
    var avgQueryMs: Double {
        let ok = queries.filter { $0.error == nil }
        guard !ok.isEmpty else { return 0 }
        return ok.reduce(0.0) { $0 + $1.elapsed } / Double(ok.count) * 1000
    }
    var medianQueryMs: Double {
        let ok = queries.filter { $0.error == nil }.map { $0.elapsed }.sorted()
        guard !ok.isEmpty else { return 0 }
        let mid = ok.count / 2
        if ok.count % 2 == 0 {
            return (ok[mid - 1] + ok[mid]) / 2 * 1000
        }
        return ok[mid] * 1000
    }
    var errorCount: Int { queries.filter { $0.error != nil }.count }
}

func evaluateHits(_ sources: [String], gt: String) -> (hit: Bool, rank: Int?) {
    for (i, s) in sources.enumerated() {
        if normalizeSource(s) == gt {
            return (true, i + 1)
        }
    }
    return (false, nil)
}

// MARK: - 阶段执行

func makeConfig(corpus: String, options: BenchmarkOptions, colbert: Bool) -> AgentConfig {
    var config = AgentConfig()
    config.workspace = options.workspace
    config.knowledgeBase = corpus
    config.colbertReaderEnabled = colbert
    // 检索阅读返回条数 = 普通 RAG 的 topK，保证 hit@k 口径一致。
    config.colbertTopN = options.topK
    config.colbertCoarseCandidates = max(config.colbertCoarseCandidates, options.topK)
    if options.disableEmbedding {
        // "unsupported" 后端让 LocalModelFactory 返回 FailingEmbeddingProvider，
        // 用于验证检索阅读模式不依赖普通 embedding 模型。
        config.localEmbeddingBackend = "unsupported"
    }
    return config
}

func runNormalRAG(corpus: String, queries: [Query],
                  options: BenchmarkOptions) async -> PhaseMetrics {
    let engine = RAGEngine(config: makeConfig(corpus: corpus, options: options, colbert: false))

    print("\n[阶段 1] 普通 RAG（embedding + 向量 + FTS + RRF + reranker）入库中…")
    let ingestStart = Date()
    let stats = await engine.ingest(force: true)
    let ingestSeconds = Date().timeIntervalSince(ingestStart)
    print("  -> \(stats.summary)（入库耗时 \(formatSeconds(ingestSeconds))）")
    for err in stats.errors { print("  ! \(err)") }

    var results: [QueryResult] = []
    for (idx, q) in queries.enumerated() {
        let start = Date()
        do {
            let hits = try await engine.search(query: q.text, topK: options.topK)
            let elapsed = Date().timeIntervalSince(start)
            let (hit, rank) = evaluateHits(hits.map(\.source), gt: q.gtSource)
            results.append(QueryResult(query: q, hit: hit, hitRank: rank, elapsed: elapsed, error: nil))
        } catch {
            let elapsed = Date().timeIntervalSince(start)
            results.append(QueryResult(query: q, hit: false, hitRank: nil, elapsed: elapsed,
                                       error: error.localizedDescription))
        }
        if (idx + 1) % 20 == 0 || idx + 1 == queries.count {
            print("  -> 检索进度 \(idx + 1)/\(queries.count)")
        }
    }
    return PhaseMetrics(mode: "普通 RAG（embedding+reranker）", queries: results,
                        ingestSeconds: ingestSeconds, ingestChunks: stats.chunks,
                        ingestErrors: stats.errors)
}

func runColbertRAG(corpus: String, queries: [Query],
                   options: BenchmarkOptions) async -> PhaseMetrics {
    let engine = RAGEngine(config: makeConfig(corpus: corpus, options: options, colbert: true))

    print("\n[阶段 2] 检索阅读模式（ColBERT 多向量大分片）入库中…")
    let ingestStart = Date()
    let stats = await engine.ingest(force: true)
    let ingestSeconds = Date().timeIntervalSince(ingestStart)
    print("  -> \(stats.summary)（入库耗时 \(formatSeconds(ingestSeconds))）")
    for err in stats.errors { print("  ! \(err)") }

    var results: [QueryResult] = []
    for (idx, q) in queries.enumerated() {
        let start = Date()
        do {
            let outcome = try await engine.searchWithCitations(query: q.text, topK: options.topK)
            let elapsed = Date().timeIntervalSince(start)
            let (hit, rank) = evaluateHits(outcome.citations.map(\.source), gt: q.gtSource)
            results.append(QueryResult(query: q, hit: hit, hitRank: rank, elapsed: elapsed, error: nil))
        } catch {
            let elapsed = Date().timeIntervalSince(start)
            results.append(QueryResult(query: q, hit: false, hitRank: nil, elapsed: elapsed,
                                       error: error.localizedDescription))
        }
        if (idx + 1) % 20 == 0 || idx + 1 == queries.count {
            print("  -> 检索进度 \(idx + 1)/\(queries.count)")
        }
    }
    return PhaseMetrics(mode: "检索阅读（ColBERT）", queries: results,
                        ingestSeconds: ingestSeconds, ingestChunks: stats.chunks,
                        ingestErrors: stats.errors)
}

// MARK: - 报告

func formatSeconds(_ s: Double) -> String {
    s < 60 ? String(format: "%.1fs", s) : String(format: "%.1fm", s / 60)
}

func percent(_ v: Double) -> String { String(format: "%.1f%%", v * 100) }

func printReport(normal: PhaseMetrics, colbert: PhaseMetrics, options: BenchmarkOptions) {
    let k = options.topK
    print("\n========== RAG 检索对比报告 ==========")
    print("语料: \(options.corpus)")
    print("查询数: \(normal.total)   topK: \(k)   工作目录: \(options.workspace)")
    print()

    let rows: [(String, String, String)] = [
        ("命中 / 总数", "\(normal.hits) / \(normal.total)", "\(colbert.hits) / \(colbert.total)"),
        ("hit@1", percent(normal.hitAt1Rate), percent(colbert.hitAt1Rate)),
        ("hit@\(k)", percent(normal.hitAtKRate), percent(colbert.hitAtKRate)),
        ("MRR", String(format: "%.4f", normal.mrr), String(format: "%.4f", colbert.mrr)),
        ("平均查询耗时", String(format: "%.0fms", normal.avgQueryMs),
         String(format: "%.0fms", colbert.avgQueryMs)),
        ("中位查询耗时", String(format: "%.0fms", normal.medianQueryMs),
         String(format: "%.0fms", colbert.medianQueryMs)),
        ("入库耗时", formatSeconds(normal.ingestSeconds), formatSeconds(colbert.ingestSeconds)),
        ("入库切片数", "\(normal.ingestChunks)", "\(colbert.ingestChunks)"),
        ("查询失败数", "\(normal.errorCount)", "\(colbert.errorCount)"),
    ]

    let header = "| 指标 | 普通 RAG | 检索阅读(ColBERT) |"
    let sep = "|---|---|---|"
    print(header)
    print(sep)
    for (name, a, b) in rows {
        print("| \(name) | \(a) | \(b) |")
    }

    if normal.errorCount > 0 || colbert.errorCount > 0 {
        print("\n提示：存在查询失败，常见原因是 embedding/reranker/colbert 模型未就绪，")
        print("请确认对应模型已下载或随 App 分发，并检查上方的入库/错误日志。")
    }
}

func writeJSON(normal: PhaseMetrics, colbert: PhaseMetrics,
               options: BenchmarkOptions) {
    struct ModeReport: Codable {
        var mode: String
        var total: Int
        var hits: Int
        var hitAt1: Int
        var hitAtK: Int
        var hitAt1Rate: Double
        var hitAtKRate: Double
        var mrr: Double
        var avgQueryMs: Double
        var medianQueryMs: Double
        var ingestSeconds: Double
        var ingestChunks: Int
        var errorCount: Int
    }
    struct Report: Codable {
        var corpus: String
        var topK: Int
        var workspace: String
        var generatedAt: String
        var normal: ModeReport
        var colbert: ModeReport
    }

    func modeReport(_ m: PhaseMetrics) -> ModeReport {
        ModeReport(mode: m.mode, total: m.total, hits: m.hits,
                   hitAt1: m.hitAt1, hitAtK: m.hitAtK,
                   hitAt1Rate: m.hitAt1Rate, hitAtKRate: m.hitAtKRate,
                   mrr: m.mrr, avgQueryMs: m.avgQueryMs, medianQueryMs: m.medianQueryMs,
                   ingestSeconds: m.ingestSeconds, ingestChunks: m.ingestChunks,
                   errorCount: m.errorCount)
    }

    let report = Report(corpus: options.corpus, topK: options.topK,
                        workspace: options.workspace,
                        generatedAt: ISO8601DateFormatter().string(from: Date()),
                        normal: modeReport(normal), colbert: modeReport(colbert))
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    guard let data = try? encoder.encode(report) else { return }
    let path = (options.workspace as NSString).appendingPathComponent("benchmark_report.json")
    try? FileManager.default.createDirectory(atPath: options.workspace,
                                             withIntermediateDirectories: true)
    try? data.write(to: URL(fileURLWithPath: path))
    print("\n报告已写入: \(path)")
}

// MARK: - main

@main
struct RAGBenchmark {
    static func main() async {
        // --llm-bench 子命令：推理基线测量（阶段 0，见 .trae/documents/mlx-gguf-inference-analysis.md）
        if let benchIndex = CommandLine.arguments.firstIndex(of: "--llm-bench") {
            let rest = Array(CommandLine.arguments[(benchIndex + 1)...])
            do {
                try await LLMBench.run(rest)
            } catch {
                fputs("LLMBench 失败：\(error)\n", stderr)
                exit(1)
            }
            exit(0)
        }

        guard let options = parseArguments() else {
            exit(1)
        }

        let files = scanCorpus(options.corpus, limit: options.limit)
        guard !files.isEmpty else {
            fputs("错误：语料目录 \(options.corpus) 下未找到 .md/.markdown/.txt 文件\n", stderr)
            exit(1)
        }

        var queries: [Query] = []
        for file in files {
            queries.append(contentsOf: generateQueries(for: file))
        }
        guard !queries.isEmpty else {
            fputs("错误：无法从语料生成任何查询（文件为空或格式不符）\n", stderr)
            exit(1)
        }
        print("扫描到 \(files.count) 个文档，生成 \(queries.count) 条合成查询")

        let normal = await runNormalRAG(corpus: options.corpus, queries: queries, options: options)
        let colbert = await runColbertRAG(corpus: options.corpus, queries: queries, options: options)

        printReport(normal: normal, colbert: colbert, options: options)
        writeJSON(normal: normal, colbert: colbert, options: options)
    }
}
