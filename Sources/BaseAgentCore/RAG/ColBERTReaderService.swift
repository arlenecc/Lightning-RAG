import Foundation
import CryptoKit
import ColBERTReader

/// 检索阅读模式的服务封装。
///
/// 负责 ColBERT 模型的自动下载（hf-mirror.com）与加载、文档入库、
/// 检索召回、向量索引构建，并把结果映射为 `RAGCitation` 召回切片气泡。
///
/// 启用该模式后，分片不再进入模型上下文（`context` 置空、全部
/// `isPrimary == false`），模型只回复引导性信息，用户点击气泡阅读原文。
public final class ColBERTReaderService: @unchecked Sendable {
    private let config: AgentConfig
    private let dbPath: String
    private let lock = NSLock()
    /// 串行化底层 SQLite / ONNX 的阻塞操作（单连接非线程安全）。
    private let opLock = NSRecursiveLock()
    private var engine: ColBERTReaderEngine?
    private var loadTask: Task<ColBERTReaderEngine, Error>?
    /// 模型下载状态上抛（供 UI 显示进度提示；节流后的关键节点 + 阶段变化）。
    public var downloadStatusHandler: (@Sendable (String) -> Void)?

    public init(config: AgentConfig, dbPath: String) {
        self.config = config
        self.dbPath = dbPath
    }

    private var model: RequiredModel {
        RequiredModel(
            kind: "检索阅读",
            repo: config.colbertModel,
            directory: RequiredModelCatalog.colbertModelDirectory(for: config),
            allowedWeightNames: ["model_int8.onnx"],
            requiresTokenizer: true)
    }

    private var options: ColBERTReaderOptions {
        ColBERTReaderOptions(
            quantBits: config.colbertQuantBits,
            coarseCandidates: config.colbertCoarseCandidates,
            topN: config.colbertTopN,
            useANN: config.colbertUseANN)
    }

    /// 由来源路径生成稳定的 doc_id 前缀（16 位十六进制，避免特殊字符干扰 LIKE）。
    private func sourceKey(_ source: String) -> String {
        let digest = SHA256.hash(data: Data(source.utf8))
        return digest.prefix(8).map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - 模型下载与加载（单飞 single-flight）

    private func ensureEngine() async throws -> ColBERTReaderEngine {
        lock.lock()
        // 快检必须在锁内：engine 写入都在 lock 下，锁外读构成数据竞争。
        if let engine, engine.isModelLoaded {
            lock.unlock()
            return engine
        }
        if let loadTask {
            lock.unlock()
            return try await loadTask.value
        }
        let task = Task<ColBERTReaderEngine, Error> { [config, dbPath, weak self] in
            guard let self else { throw AgentError.cancelled }
            let model = RequiredModel(
                kind: "检索阅读",
                repo: config.colbertModel,
                directory: RequiredModelCatalog.colbertModelDirectory(for: config),
                allowedWeightNames: ["model_int8.onnx"],
                requiresTokenizer: true)
            if !RequiredModelCatalog.isComplete(model) {
                try await self.download(model)
            }
            let engine = try ColBERTReaderEngine(dbPath: dbPath, modelDir: model.directory.path)
            try await engine.loadModel()
            return engine
        }
        loadTask = task
        lock.unlock()

        do {
            let engine = try await task.value
            lock.lock()
            self.engine = engine
            lock.unlock()
            return engine
        } catch {
            lock.lock()
            loadTask = nil
            lock.unlock()
            throw error
        }
    }

    /// 打开 schema（不加载模型），用于删除/计数等纯 db 操作。
    private func ensureEngineInstance() throws -> ColBERTReaderEngine {
        lock.lock()
        defer { lock.unlock() }
        if let engine { return engine }
        let e = try ColBERTReaderEngine(dbPath: dbPath, modelDir: model.directory.path)
        engine = e
        return e
    }

    /// 下载缺失模型；状态经 `downloadStatusHandler` 上抛（UI 可显示进度）。
    private func download(_ model: RequiredModel) async throws {
        let handler = downloadStatusHandler
        let report: @Sendable (String) -> Void = { handler?($0) }
        report("正在下载检索阅读模型（\(model.repo)）…")
        let downloader = HFModelDownloader()
        let files = try await downloader.listNeededFiles(repo: model.repo)
        let scoped = RequiredModelCatalog.scopedFiles(for: model, in: files)
        let totalBytes = scoped.reduce(Int64(0)) { $0 + $1.size }
        var doneBytes: Int64 = 0
        for file in scoped {
            // 停止同步等场景可经 cancelLoading() 取消下载：URLSession data(for:)
            // 响应 Task 取消抛 CancellationError，loadTask 失败后可重试。
            try Task.checkCancellation()
            let label = file.path
            let started = Date()
            var lastLogAt = Date.distantPast
            AgentLog.info("[colbert] 开始下载 \(file.path)（\(file.size / 1_048_576)MB）")
            report("下载 \(label)（\(file.size / 1_048_576)MB，共 "
                + "\(totalBytes / 1_048_576)MB）")
            try await downloader.download(repo: model.repo, file: file,
                                          into: model.directory) { received, total in
                // 大文件下载可达数十分钟，每 10% 或 30 秒输出一次，避免用户以为同步卡死。
                let now = Date()
                let ratio = total > 0 ? Double(received) / Double(total) : 0
                let enoughTime = now.timeIntervalSince(lastLogAt) >= 30
                if enoughTime || (total > 0 && received >= total) {
                    lastLogAt = now
                    let percent = Int(ratio * 100)
                    let overall = doneBytes + received
                    AgentLog.info("[colbert] 下载 \(label) \(received / 1_048_576)MB"
                        + "/\(total / 1_048_576)MB（\(percent)%）")
                    report("下载 \(label) \(percent)%（累计 \(overall / 1_048_576)"
                        + "/\(totalBytes / 1_048_576)MB）")
                }
            }
            doneBytes += file.size
            AgentLog.info("[colbert] 下载完成 \(file.path)，耗时 "
                + String(format: "%.1f", Date().timeIntervalSince(started)) + "s")
        }
        report("模型下载完成，正在加载…")
        guard RequiredModelCatalog.isComplete(model) else {
            throw HFModelDownloader.HFError.badResponse("检索阅读模型下载后仍不完整")
        }
    }

    // MARK: - 入库 / 检索 / 索引

    /// 入库一段 markdown 文本，返回写入的切片数。
    /// `shouldCancel` 在每片编码前检查；取消时抛 `CancellationError`，整个文件回滚。
    @discardableResult
    public func ingest(markdown: String, source: String, title: String? = nil,
                       shouldCancel: (() -> Bool)? = nil) async throws -> Int {
        let engine = try await ensureEngine()
        opLock.lock()
        defer { opLock.unlock() }
        // 长文档切片编码可能持续数十秒：按 30 秒节流输出进度，避免看起来「卡住」。
        let encoded = markdown.count
        let started = Date()
        var lastLogAt = Date.distantPast
        // 语义切片：与普通 RAG 共用 potion-multilingual-128M（int4 ONNX）静态词向量，
        // 在相邻句相似度低谷处断开；模型不可用时切片器自动回退词法相似度。
        let similarity = PotionSentenceSimilarity().adjacentCosines
        return try engine.ingest(
            markdown: markdown, source: source, title: title,
            sourceKey: sourceKey(source), options: options,
            similarity: similarity,
            shouldCancel: shouldCancel,
            progress: { done, total in
                let now = Date()
                guard now.timeIntervalSince(lastLogAt) >= 30 || done == total else { return }
                lastLogAt = now
                AgentLog.info("[colbert] 编码 \(done)/\(total) 分片（\(encoded) 字符，已用 "
                    + String(format: "%.0f", now.timeIntervalSince(started)) + "s）")
            })
    }

    /// 删除指定来源的全部切片（不加载模型）。
    @discardableResult
    /// 删除指定来源的全部切片。
    /// - Parameter source: 来源原始路径；doc_id 前缀删不到时按 document.source 列
    ///   兜底（兼容 doc_search 独立工具建的旧库，其 doc_id 前缀非 SHA256 hex）。
    public func delete(source: String) throws -> Int {
        let engine = try ensureEngineInstance()
        opLock.lock()
        defer { opLock.unlock() }
        return try engine.delete(sourceKey: sourceKey(source), fallbackSource: source)
    }

    /// 指定来源的切片数（不加载模型）。
    /// - Throws: colbert.db 打不开/查询失败时上抛，调用方不得把错误当成 0 条。
    public func chunkCount(source: String) throws -> Int {
        let engine = try ensureEngineInstance()
        opLock.lock()
        defer { opLock.unlock() }
        return try engine.chunkCount(sourceKey: sourceKey(source), fallbackSource: source)
    }

    /// 全库切片总数（不加载模型）。
    /// - Throws: colbert.db 打不开/查询失败时上抛，调用方不得把错误当成 0 条。
    public func totalChunkCount() throws -> Int {
        let engine = try ensureEngineInstance()
        opLock.lock()
        defer { opLock.unlock() }
        return try engine.totalChunkCount()
    }

    /// 预加载模型并常驻，让首次检索不再现加载。
    public func preload() async throws {
        _ = try await ensureEngine()
    }

    /// 检索并返回召回切片（全部不进模型上下文）。
    /// `source` 非空时只返回该来源文件/标题内的切片。
    ///
    /// 检索路径只读（SQLite WAL 快照隔离 + FULLMUTEX 句柄串行），不持 opLock：
    /// ANN 构建/同步入库期间发起的对话检索可立即执行（读到构建前的索引快照），
    /// 不会在 opLock 上排队等待长任务结束。
    public func search(query: String, source: String? = nil) async throws -> RAGRetrievalOutcome {
        let engine = try await ensureEngine()
        // 分阶段耗时接线：每次检索输出一行汇总，定位慢查询（编码/粗筛/精排）。
        let timingStarted = Date()
        var stageTimings: [(String, Double)] = []
        let hits = try engine.search(query: query, options: options, source: source) {
            key, seconds in
            stageTimings.append((key, seconds))
        }

        var citations: [RAGCitation] = []
        var rank = 1
        for hit in hits {
            // source 用入库时的文件相对路径（气泡上的「来源」）；
            // title 是切片所在章节标题，作 heading 展示。旧库没有 source
            // 列时回退到 title，保持原有展示。
            let source = hit.source.isEmpty ? hit.title : hit.source
            citations.append(RAGCitation(
                id: hit.docID,
                rank: rank,
                source: source,
                heading: hit.title == source ? "" : hit.title,
                text: hit.content,
                score: hit.score,
                isPrimary: false))
            rank += 1
        }
        // 检索阅读模式：分片不进入模型上下文，但必须明确告诉模型「已召回、
        // 内容在气泡里」，否则模型只看到空上下文会声称「没找到」或臆造内容。
        let guide = "本次检索在知识库中召回 \(citations.count) 条原文切片，"
            + "已作为引用气泡展示给用户。检索阅读模式不把切片正文放进模型上下文："
            + "不要猜测或复述切片内容，用一两句话说明已找到相关资料，"
            + "并引导用户点击下方引用气泡阅读原文。"
        // 分阶段耗时日志：编码/粗筛/融合/精排各阶段毫秒 + 候选数。
        let totalMs = Date().timeIntervalSince(timingStarted) * 1000
        let stages = stageTimings
            .filter { !["shortlist", "search_total"].contains($0.0) }
            .map { "\($0.0)=\(Int($0.1 * 1000))ms" }
            .joined(separator: " ")
        AgentLog.info("[colbert] 检索 \(Int(totalMs))ms 候选=\(hits.count) \(stages)")
        return RAGRetrievalOutcome(context: guide, citations: citations)
    }

    /// 构建 / 续建向量索引，返回摘要。
    /// - Parameter shouldCancel: 块间取消检查；中断后已写块保留，续建从水位继续。
    @discardableResult
    public func buildANN(shouldCancel: (() -> Bool)? = nil) async throws -> String {
        let engine = try await ensureEngine()
        opLock.lock()
        defer { opLock.unlock() }
        return try engine.buildANN(shouldCancel: shouldCancel)
    }

    /// ANN 索引状态摘要（读路径；opLock 与长任务互斥，避免读到中间状态）。
    /// 忙碌（ingest/构建持锁）时返回 nil：状态栏刷新不应在 opLock 上排队停摆。
    public func annStatus() -> String? {
        guard opLock.try() else { return nil }
        defer { opLock.unlock() }
        return engine?.annStatus() ?? "未加载"
    }

    /// 取消进行中的模型加载/下载（如停止同步时）；任务失败后下次调用重新加载。
    public func cancelLoading() {
        lock.lock()
        let task = loadTask
        loadTask = nil
        lock.unlock()
        task?.cancel()
    }
}
