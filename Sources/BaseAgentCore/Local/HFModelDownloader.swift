import Foundation

// MARK: - HF 镜像模型下载

/// 从 HuggingFace 兼容镜像（默认 hf-mirror.com）按需拉取模型文件。
///
/// - 文件列表走 REST API：`GET /api/models/{repo}/tree/main?recursive=true`；
/// - 单文件下载：`GET /{repo}/resolve/main/{path}`，HTTP `Range` 请求配合
///   `.part` 临时文件实现断点续传（重启应用后从已下载偏移继续）；
/// - 只拉取推理运行时会读取的文件（权重 / 配置 / tokenizer），忽略 README、
///   示例脚本等无关内容。
public final class HFModelDownloader: @unchecked Sendable {
    public struct FileEntry: Sendable, Equatable {
        public let path: String
        public let size: Int64

        public init(path: String, size: Int64) {
            self.path = path
            self.size = size
        }
    }

    public enum HFError: LocalizedError {
        case badResponse(String)
        case httpStatus(Int)

        public var errorDescription: String? {
            switch self {
            case .badResponse(let message): return message
            case .httpStatus(let code): return "下载请求失败（HTTP \(code)）"
            }
        }
    }

    /// 默认镜像：国内网络可直接访问的 HuggingFace 反代。
    public static let defaultMirrorBase = "https://hf-mirror.com"
    /// 默认模型统一托管在 arlenecc/<模型名> 仓库。
    public static let repoOwner = "arlenecc"

    private let baseURL: URL
    private let session: URLSession
    private let lock = NSLock()
    private var activeTask: URLSessionTask?

    /// `HF_MIRROR_BASE` 环境变量可覆盖镜像地址（测试/内网镜像）。
    public init(base: String = HFModelDownloader.defaultMirrorBase) {
        let envBase = ProcessInfo.processInfo.environment["HF_MIRROR_BASE"] ?? ""
        self.baseURL = URL(string: envBase.isEmpty ? base : envBase)
            ?? URL(string: base)!
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 60
        // 大模型下载可达数 GB，资源级超时必须放开。
        cfg.timeoutIntervalForResource = 7 * 24 * 3_600
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        self.session = URLSession(configuration: cfg)
    }

    deinit {
        lock.lock()
        activeTask?.cancel()
        lock.unlock()
    }

    // MARK: 文件清单

    /// 列出仓库中推理需要的文件（过滤 README / 脚本 / 图片等）。
    public func listNeededFiles(repo: String) async throws -> [FileEntry] {
        var comps = URLComponents(
            url: baseURL.appendingPathComponent("api/models/\(repo)/tree/main"),
            resolvingAgainstBaseURL: false)
        comps?.queryItems = [URLQueryItem(name: "recursive", value: "true")]
        guard let url = comps?.url else {
            throw HFError.badResponse("无效的仓库地址：\(repo)")
        }
        let (data, response) = try await session.data(for: URLRequest(url: url))
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200 else { throw HFError.httpStatus(code) }
        guard let list = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw HFError.badResponse("模型文件列表格式异常：\(repo)")
        }
        var entries: [FileEntry] = []
        for item in list {
            guard (item["type"] as? String) == "file",
                  let path = item["path"] as? String,
                  Self.isNeededFile(path) else { continue }
            let size = (item["size"] as? NSNumber)?.int64Value ?? 0
            entries.append(FileEntry(path: path, size: size))
        }
        return entries
    }

    /// 推理运行时会读取的文件类型：权重（safetensors/onnx/gguf）、配置与
    /// tokenizer（json/jinja/model/txt）。README（md）、示例脚本、图片等排除。
    public static func isNeededFile(_ path: String) -> Bool {
        let name = (path as NSString).lastPathComponent
        guard !name.hasPrefix(".") else { return false }
        let ext = (name as NSString).pathExtension.lowercased()
        switch ext {
        case "json", "jinja", "model", "txt", "safetensors", "onnx", "gguf":
            return true
        default:
            return false
        }
    }

    /// 权重文件扩展名：目录里存在任意一个即视为模型就绪。
    public static func isWeightFile(_ name: String) -> Bool {
        let ext = (name as NSString).pathExtension.lowercased()
        return ext == "safetensors" || ext == "onnx" || ext == "gguf"
    }

    // MARK: 单文件下载（断点续传）

    /// 下载仓库内的一个文件到 `directory/<path>`。
    ///
    /// 目标文件已存在且大小一致时直接返回；存在 `.part` 临时文件时从其
    /// 当前偏移发 Range 请求续传。失败自动重试 3 次（每次从最新偏移继续）。
    public func download(repo: String, file: FileEntry, into directory: URL,
                         progress: @escaping @Sendable (_ received: Int64, _ total: Int64) -> Void
    ) async throws {
        let fm = FileManager.default
        let destination = directory.appendingPathComponent(file.path)
        if Self.fileSize(at: destination) == file.size {
            progress(file.size, file.size)
            return
        }
        try fm.createDirectory(at: destination.deletingLastPathComponent(),
                               withIntermediateDirectories: true)
        let partial = destination.appendingPathExtension("part")

        var lastError: Error?
        for _ in 0..<3 {
            try Task.checkCancellation()
            do {
                try await downloadOnce(repo: repo, file: file, partial: partial,
                                       progress: progress)
                // 大小校验：不符则报错（.part 保留，下次从断点继续）。
                if file.size > 0 {
                    let got = Self.fileSize(at: partial)
                    guard got == file.size else {
                        throw HFError.badResponse(
                            "\(file.path) 下载不完整（\(got)/\(file.size) 字节）")
                    }
                }
                if fm.fileExists(atPath: destination.path) {
                    try fm.removeItem(at: destination)
                }
                try fm.moveItem(at: partial, to: destination)
                progress(file.size, file.size)
                return
            } catch {
                if error is CancellationError { throw error }
                lastError = error
                try? await Task.sleep(nanoseconds: 1_500_000_000)
            }
        }
        throw lastError ?? HFError.badResponse("下载失败：\(file.path)")
    }

    private func downloadOnce(repo: String, file: FileEntry, partial: URL,
                              progress: @escaping @Sendable (Int64, Int64) -> Void
    ) async throws {
        let fm = FileManager.default
        var offset = Self.fileSize(at: partial)
        if !fm.fileExists(atPath: partial.path) {
            fm.createFile(atPath: partial.path, contents: Data())
            offset = 0
        }
        let handle = try FileHandle(forWritingTo: partial)
        defer { try? handle.close() }
        if offset > 0 {
            try handle.seek(toOffset: UInt64(offset))
        } else {
            try handle.truncate(atOffset: 0)
        }

        var request = URLRequest(url: resolveURL(repo: repo, path: file.path))
        if offset > 0 {
            request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range")
        }

        let sink = RangeDownloadSink(expectedSize: file.size, resumedAt: offset,
                                     handle: handle, progress: progress)
        let task = session.dataTask(with: request)
        task.delegate = sink
        lock.lock()
        activeTask = task
        lock.unlock()
        defer {
            lock.lock()
            if activeTask === task { activeTask = nil }
            lock.unlock()
        }

        let failure: Error? = await withCheckedContinuation { continuation in
            sink.onFinish = { continuation.resume(returning: $0) }
            task.resume()
        }
        if let failure { throw failure }
    }

    /// 取消当前正在下载的文件（.part 保留，下次从断点续传）。
    public func cancelCurrent() {
        lock.lock()
        let task = activeTask
        lock.unlock()
        task?.cancel()
    }

    private func resolveURL(repo: String, path: String) -> URL {
        var url = baseURL
            .appendingPathComponent(repo)
            .appendingPathComponent("resolve")
            .appendingPathComponent("main")
        for segment in path.split(separator: "/") {
            url = url.appendingPathComponent(String(segment))
        }
        return url
    }

    public static func fileSize(at url: URL) -> Int64 {
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (attrs?[.size] as? NSNumber)?.int64Value ?? 0
    }
}

/// 把 dataTask 的字节流写入 FileHandle 并上报进度的委托。
private final class RangeDownloadSink: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    let expectedSize: Int64
    let resumedAt: Int64
    let handle: FileHandle
    let progress: @Sendable (Int64, Int64) -> Void
    /// 完成回调：nil 表示成功（由 didCompleteWithError / 内部失败判定决定）。
    var onFinish: (@Sendable (Error?) -> Void)?
    private var received: Int64
    private var failure: Error?
    private var lastReport = Date.distantPast
    private let lock = NSLock()

    init(expectedSize: Int64, resumedAt: Int64, handle: FileHandle,
         progress: @escaping @Sendable (Int64, Int64) -> Void) {
        self.expectedSize = expectedSize
        self.resumedAt = resumedAt
        self.handle = handle
        self.progress = progress
        self.received = resumedAt
    }

    private func finish(_ error: Error?) {
        lock.lock()
        let alreadyDone = (onFinish == nil)
        let callback = onFinish
        onFinish = nil
        lock.unlock()
        guard !alreadyDone else { return }
        callback?(failure ?? error)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200...299).contains(code) else {
            if code == 416 {
                // Range 不可满足（本地残留多于远端文件）：删除 .part 重来。
                try? handle.truncate(atOffset: 0)
                received = 0
                failure = HFModelDownloader.HFError.httpStatus(code)
                completionHandler(.cancel)
                finish(nil)
                return
            }
            failure = HFModelDownloader.HFError.httpStatus(code)
            completionHandler(.cancel)
            finish(nil)
            return
        }
        if code == 200 && resumedAt > 0 {
            // 服务器忽略 Range 返回全量：从头重写。
            try? handle.truncate(atOffset: 0)
            received = 0
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive data: Data) {
        do {
            try handle.write(contentsOf: data)
        } catch {
            failure = error
            dataTask.cancel()
            return
        }
        received += Int64(data.count)
        // 进度回调限流到 ~10/s，避免 GB 级文件每块都刷新 UI。
        let now = Date()
        if now.timeIntervalSince(lastReport) >= 0.1 {
            lastReport = now
            progress(received, max(expectedSize, received))
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        if let nsError = error as NSError?, nsError.code == NSURLErrorCancelled {
            // 主动取消：不算失败，也不重试。
            finish(CancellationError())
            return
        }
        if let error, failure == nil { failure = error }
        if failure == nil, expectedSize > 0, received != expectedSize {
            failure = HFModelDownloader.HFError.badResponse(
                "连接中断（\(received)/\(expectedSize) 字节），将从断点续传")
        }
        finish(nil)
    }
}

// MARK: - 默认配置的必需模型清单

/// 默认配置实际引用的本地模型（按当前配置解析目录与 HF 仓库名）。
public struct RequiredModel: Sendable, Equatable, Identifiable {
    public var id: String { repo }
    /// 用途显示名：主模型 / 辅助模型 / Embedding / Reranker / OCR。
    public let kind: String
    /// HF 仓库名：arlenecc/<模型名>。
    public let repo: String
    /// 本地目标目录（缺失时从这里补齐）。
    public let directory: URL

    public init(kind: String, repo: String, directory: URL) {
        self.kind = kind
        self.repo = repo
        self.directory = directory
    }
}

public enum RequiredModelCatalog {
    /// 解析当前配置需要的全部本地模型（按目录去重）。
    public static func models(for config: AgentConfig) -> [RequiredModel] {
        var seen = Set<String>()
        var out: [RequiredModel] = []
        func add(kind: String, value: String) {
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return }
            let dir = directory(for: trimmed, config: config)
            guard seen.insert(dir.standardizedFileURL.path).inserted else { return }
            let repoName: String
            if trimmed.hasSuffix(".gguf") {
                repoName = ((trimmed as NSString).lastPathComponent as NSString)
                    .deletingPathExtension
            } else {
                repoName = (trimmed as NSString).lastPathComponent
            }
            out.append(RequiredModel(
                kind: kind,
                repo: "\(HFModelDownloader.repoOwner)/\(repoName)",
                directory: dir))
        }

        if config.llmProvider == "local" {
            add(kind: "主模型", value: config.localLLMModel)
        }
        if config.auxEnabled, config.auxLLMProvider == "local" {
            add(kind: "辅助模型", value: config.auxLocalLLMModel)
        }
        if config.embeddingProvider == "local" {
            add(kind: "Embedding", value: embeddingValue(config))
        }
        if config.rerankerProvider == "local" {
            add(kind: "Reranker", value: rerankerValue(config))
        }
        if config.ragOCREngine == "mlx-vlm" {
            let value = !config.ragOCRModelPath.isEmpty
                ? config.ragOCRModelPath
                : (!config.localVLMModel.isEmpty
                    ? config.localVLMModel : config.ragOCRModelName)
            add(kind: "OCR", value: value)
        }
        return out
    }

    /// 缺失（目录不存在或没有任何权重文件）的必需模型。
    public static func missingModels(for config: AgentConfig) -> [RequiredModel] {
        models(for: config).filter { !isComplete($0.directory) }
    }

    /// 模型目录是否就绪：存在至少一个权重文件（safetensors / onnx / gguf）。
    public static func isComplete(_ directory: URL) -> Bool {
        let items = (try? FileManager.default.contentsOfDirectory(
            atPath: directory.path)) ?? []
        return items.contains { HFModelDownloader.isWeightFile($0) }
    }

    /// 配置值 → 本地目录。相对路径（"models/xxx" 或裸名）解析到
    /// workspace/models 下；GGUF 单文件取其所在目录。
    static func directory(for value: String, config: AgentConfig) -> URL {
        if value.hasSuffix(".gguf") {
            let dirPath = (value as NSString).deletingLastPathComponent
            if dirPath.hasPrefix("/") {
                return URL(fileURLWithPath: dirPath)
            }
            if dirPath.isEmpty {
                return URL(fileURLWithPath: config.modelsDirectory)
            }
            return URL(fileURLWithPath: config.workspace)
                .appendingPathComponent(dirPath)
        }
        if value.hasPrefix("/") {
            return URL(fileURLWithPath: value)
        }
        let name = (value as NSString).lastPathComponent
        return URL(fileURLWithPath: config.modelsDirectory)
            .appendingPathComponent(name)
    }

    static func embeddingValue(_ config: AgentConfig) -> String {
        if !config.localEmbeddingModelPath.isEmpty {
            return config.localEmbeddingModelPath
        }
        return config.localEmbeddingModel.isEmpty
            ? config.ragEmbeddingModel : config.localEmbeddingModel
    }

    static func rerankerValue(_ config: AgentConfig) -> String {
        if !config.localRerankerModelPath.isEmpty {
            return config.localRerankerModelPath
        }
        return config.localRerankerModel.isEmpty
            ? config.ragRerankModel : config.localRerankerModel
    }
}
