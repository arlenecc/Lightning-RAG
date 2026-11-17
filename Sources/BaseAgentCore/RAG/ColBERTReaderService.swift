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
        if let engine, engine.isModelLoaded { return engine }

        lock.lock()
        if let loadTask {
            lock.unlock()
            return try await loadTask.value
        }
        let task = Task<ColBERTReaderEngine, Error> { [config, dbPath] in
            let model = RequiredModel(
                kind: "检索阅读",
                repo: config.colbertModel,
                directory: RequiredModelCatalog.colbertModelDirectory(for: config),
                allowedWeightNames: ["model_int8.onnx"],
                requiresTokenizer: true)
            if !RequiredModelCatalog.isComplete(model) {
                try await Self.download(model)
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

    private static func download(_ model: RequiredModel) async throws {
        let downloader = HFModelDownloader()
        let files = try await downloader.listNeededFiles(repo: model.repo)
        let scoped = RequiredModelCatalog.scopedFiles(for: model, in: files)
        for file in scoped {
            let label = file.path
            let started = Date()
            var lastLogAt = Date.distantPast
            AgentLog.info("[colbert] 开始下载 \(file.path)（\(file.size / 1_048_576)MB）")
            try await downloader.download(repo: model.repo, file: file,
                                          into: model.directory) { received, total in
                // 大文件下载可达数十分钟，每 10% 或 30 秒输出一次，避免用户以为同步卡死。
                let now = Date()
                let ratio = total > 0 ? Double(received) / Double(total) : 0
                let enoughTime = now.timeIntervalSince(lastLogAt) >= 30
                if enoughTime || (total > 0 && received >= total) {
                    lastLogAt = now
                    let percent = Int(ratio * 100)
                    AgentLog.info("[colbert] 下载 \(label) \(received / 1_048_576)MB"
                        + "/\(total / 1_048_576)MB（\(percent)%）")
                }
            }
            AgentLog.info("[colbert] 下载完成 \(file.path)，耗时 "
                + String(format: "%.1f", Date().timeIntervalSince(started)) + "s")
        }
        guard RequiredModelCatalog.isComplete(model) else {
            throw HFModelDownloader.HFError.badResponse("检索阅读模型下载后仍不完整")
        }
    }

    // MARK: - 入库 / 检索 / 索引

    /// 入库一段 markdown 文本，返回写入的切片数。
    @discardableResult
    public func ingest(markdown: String, source: String, title: String? = nil) async throws -> Int {
        let engine = try await ensureEngine()
        opLock.lock()
        defer { opLock.unlock() }
        // 长文档切片编码可能持续数十秒：按 30 秒节流输出进度，避免看起来「卡住」。
        let encoded = markdown.count
        let started = Date()
        var lastLogAt = Date.distantPast
        return try engine.ingest(
            markdown: markdown, source: source, title: title,
            sourceKey: sourceKey(source), options: options,
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
    public func delete(source: String) throws -> Int {
        let engine = try ensureEngineInstance()
        opLock.lock()
        defer { opLock.unlock() }
        return try engine.delete(sourceKey: sourceKey(source))
    }

    /// 指定来源的切片数（不加载模型）。
    public func chunkCount(source: String) -> Int {
        guard let engine = try? ensureEngineInstance() else { return 0 }
        opLock.lock()
        defer { opLock.unlock() }
        return engine.chunkCount(sourceKey: sourceKey(source))
    }

    /// 全库切片总数（不加载模型）。
    public func totalChunkCount() -> Int {
        guard let engine = try? ensureEngineInstance() else { return 0 }
        opLock.lock()
        defer { opLock.unlock() }
        return engine.totalChunkCount()
    }

    /// 预加载模型并常驻，让首次检索不再现加载。
    public func preload() async throws {
        _ = try await ensureEngine()
    }

    /// 检索并返回召回切片（全部不进模型上下文）。
    public func search(query: String) async throws -> RAGRetrievalOutcome {
        let engine = try await ensureEngine()
        opLock.lock()
        let hits: [ColBERTReaderHit]
        do {
            hits = try engine.search(query: query, options: options)
        } catch {
            opLock.unlock()
            throw error
        }
        opLock.unlock()

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
        // 检索阅读模式：分片不进入模型上下文，模型只回复引导性信息。
        return RAGRetrievalOutcome(context: "", citations: citations)
    }

    /// 构建 / 续建向量索引，返回摘要。
    @discardableResult
    public func buildANN() async throws -> String {
        let engine = try await ensureEngine()
        opLock.lock()
        defer { opLock.unlock() }
        return try engine.buildANN()
    }

    public func annStatus() -> String {
        lock.lock()
        defer { lock.unlock() }
        return engine?.annStatus() ?? "未加载"
    }
}
