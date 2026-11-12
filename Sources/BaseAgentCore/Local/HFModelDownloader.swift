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
        /// 远程仓库内路径（含子目录前缀），用于拼接下载 URL。
        /// 子目录模型（如 potion 的 `int4/`）本地落点需剥掉前缀，但远程 URL
        /// 必须保留前缀，否则会指向仓库根下不存在的文件而 404。为空时等同 `path`。
        public let remotePath: String

        public init(path: String, size: Int64, remotePath: String? = nil) {
            self.path = path
            self.size = size
            self.remotePath = remotePath ?? path
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
    ///
    /// 与单文件下载同样的 3 次重试：TLS/网络瞬时抖动（首次很快失败）
    /// 不应直接把错误抛给启动下载流程。
    public func listNeededFiles(repo: String) async throws -> [FileEntry] {
        var lastError: Error?
        for attempt in 0..<3 {
            if attempt > 0 {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
            }
            do {
                return try await listNeededFilesOnce(repo: repo)
            } catch {
                if error is CancellationError { throw error }
                lastError = error
            }
        }
        throw lastError ?? HFError.badResponse("获取模型文件列表失败：\(repo)")
    }

    private func listNeededFilesOnce(repo: String) async throws -> [FileEntry] {
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
            // 远端清单不可信：先挡住 `../`、绝对路径、盘符等，否则会被
            // appendingPathComponent 落到模型目录之外（zip-slip 式写入）。
            guard (item["type"] as? String) == "file",
                  let path = item["path"] as? String,
                  Self.isSafeRelativePath(path),
                  Self.isNeededFile(path) else { continue }
            let size = (item["size"] as? NSNumber)?.int64Value ?? 0
            entries.append(FileEntry(path: path, size: size))
        }
        return entries
    }

    /// 远端清单里的路径能否安全落到模型目录内。
    ///
    /// 拒绝：绝对路径、`~`、空/`.`/`..` 段、Windows 反斜杠与盘符、URL scheme、
    /// NUL 和超长路径。`appendingPathComponent` 不会规范化 `..`，因此必须在
    /// 拼路径之前就过滤；`download` 里还有一次 containment 兜底。
    public static func isSafeRelativePath(_ path: String) -> Bool {
        guard !path.isEmpty, path.utf8.count <= 1024,
              !path.contains("\0"), !path.contains("\\"),
              !path.hasPrefix("/"), !path.hasPrefix("~") else { return false }
        let segments = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !segments.isEmpty else { return false }
        for segment in segments {
            let part = String(segment)
            if part.isEmpty || part == "." || part == ".." { return false }
            if part.contains(":") { return false }
        }
        return true
    }

    /// 解析符号链接后，target 是否仍在 root 目录内（root 本身不算）。
    static func isContained(_ target: URL, in root: URL) -> Bool {
        let resolved = canonicalPath(target)
        let rootPath = canonicalPath(root)
        let prefix = rootPath.hasSuffix("/") ? rootPath : rootPath + "/"
        return resolved.hasPrefix(prefix)
    }

    /// 展开路径里所有**已存在**的符号链接，再拼回尚未创建的尾部组件。
    ///
    /// `URL.resolvingSymlinksInPath()` 在目标文件不存在时不会展开中间的
    /// symlink（下载前正是这种情形），直接用它会让 `root/link/model` 中的
    /// `link -> /elsewhere` 逃过前缀比对。所以从最深的不存在组件往上找到
    /// 第一个真实目录，对它做 realpath 后再拼回剩余组件。
    private static func canonicalPath(_ url: URL) -> String {
        let fm = FileManager.default
        var missing: [String] = []
        var cursor = url.standardizedFileURL
        while cursor.path != "/" {
            var isDirectory: ObjCBool = false
            if fm.fileExists(atPath: cursor.path, isDirectory: &isDirectory) { break }
            missing.insert(cursor.lastPathComponent, at: 0)
            cursor = cursor.deletingLastPathComponent()
        }
        var resolved = cursor.resolvingSymlinksInPath().standardizedFileURL
        for component in missing {
            // 已解析部分本身可能是 symlink 目标；继续逐段 realpath，
            // 防止新拼上的组件里又出现指向外部的符号链接。
            resolved = resolved.appendingPathComponent(component)
                .resolvingSymlinksInPath().standardizedFileURL
        }
        return resolved.path
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

    // MARK: 单文件下载（断点续传）

    /// 下载仓库内的一个文件到 `directory/<path>`。
    ///
    /// 目标文件已存在且大小一致时直接返回；存在 `.part` 临时文件时从其
    /// 当前偏移发 Range 请求续传。失败自动重试 3 次（每次从最新偏移继续）。
    public func download(repo: String, file: FileEntry, into directory: URL,
                         progress: @escaping @Sendable (_ received: Int64, _ total: Int64) -> Void
    ) async throws {
        // FileEntry 也可能由打包清单/子目录模型转换构造，不只来自远端 tree API；
        // 因此落盘前再校验一次，并确认解析符号链接后仍在目标目录内。
        guard Self.isSafeRelativePath(file.path),
              Self.isSafeRelativePath(file.remotePath) else {
            throw HFError.badResponse("非法的模型文件路径：\(file.path)")
        }
        let fm = FileManager.default
        let destination = directory.appendingPathComponent(file.path).standardizedFileURL
        guard Self.isContained(destination, in: directory) else {
            throw HFError.badResponse("模型文件路径越出目标目录：\(file.path)")
        }
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

        var request = URLRequest(url: resolveURL(repo: repo, path: file.remotePath))
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
    /// 用途显示名：主模型 / 辅助模型 / Embedding / Reranker / OCR / 语义切片。
    public let kind: String
    /// HF 仓库名：arlenecc/<模型名>。
    public let repo: String
    /// 本地目标目录（缺失时从这里补齐）。
    public let directory: URL
    /// 仓库内子目录：只下载该子目录下的文件并剥掉前缀（如 potion 的 `int4`）。
    public let subpath: String?
    /// 运行时真正会加载的权重文件名（basename 白名单）。
    /// `nil` = 不限制（按扩展名宽松判定，适用于分片 safetensors / gguf）。
    /// ONNX 后端只认 `resolveONNXModel` 候选名单里的名字，避免把仓库里
    /// `model_fp16.onnx` / 原版 `model.safetensors` 这类永远不读的权重拉下来。
    public let allowedWeightNames: Set<String>?
    /// 是否必须存在 tokenizer 文件（MLX / ONNX 模型的加载前置）。
    /// GGUF 的 tokenizer 内嵌在权重里，目录下载不带 tokenizer 文件，
    /// 这类模型必须为 false，否则会被永远判定缺失。
    public let requiresTokenizer: Bool

    public init(kind: String, repo: String, directory: URL,
                subpath: String? = nil,
                allowedWeightNames: Set<String>? = nil,
                requiresTokenizer: Bool = true) {
        self.kind = kind
        self.repo = repo
        self.directory = directory
        self.subpath = subpath
        self.allowedWeightNames = allowedWeightNames
        self.requiresTokenizer = requiresTokenizer
    }
}

public enum RequiredModelCatalog {
    /// 与 `LocalModelRegistry.resolveONNXModel` 同源的可加载权重文件名。
    /// 取 basename，这样 `onnx/model_int8.onnx` 与根目录同名文件都能命中。
    public static let onnxWeightNames: Set<String> = Set(
        LocalModelRegistry.onnxCandidateNames.map { ($0 as NSString).lastPathComponent }
    )
    /// 解析当前配置需要的全部本地模型（按目录去重）。
    public static func models(for config: AgentConfig) -> [RequiredModel] {
        var seen = Set<String>()
        var out: [RequiredModel] = []
        func add(kind: String, registryKind: LocalModelKind, value: String) {
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return }
            // 与实际加载完全同源的目录解析：registry 能定位到（应用目录 /
            // App bundle / workspace / 用户目录 / HF 缓存）就用它判定就绪，
            // 避免「加载器找得到、判定却说缺失」的误报；全部缺失时下载
            // 目标指向 registry 的加载锚点，保证下载完成后一定能被加载。
            let dir: URL
            if let resolved = try? LocalModelRegistry.resolveDirectory(
                kind: registryKind, value: trimmed, config: config) {
                dir = resolved
            } else {
                dir = LocalModelRegistry.downloadDirectory(
                    kind: registryKind, value: trimmed, config: config)
            }
            guard seen.insert(dir.standardizedFileURL.path).inserted else { return }
            let repoName: String
            if trimmed.hasSuffix(".gguf") {
                repoName = ((trimmed as NSString).lastPathComponent as NSString)
                    .deletingPathExtension
            } else {
                repoName = (trimmed as NSString).lastPathComponent
            }
            // ONNX 后端的 embedding / reranker 只加载候选名单里的权重文件，
            // 下载清单与就绪判定都按这份白名单走；MLX / GGUF 不限制（分片权重）。
            let backend: String
            switch registryKind {
            case .reranker: backend = config.localRerankerBackend
            case .embedding: backend = config.localEmbeddingBackend
            case .llm, .aux: backend = config.localLLMBackend
            default: backend = ""
            }
            // GGUF 的 tokenizer 内嵌在权重里，下载目录不含 tokenizer.json；
            // 这类模型不能要求 tokenizer，否则会被判成永远缺失。
            let requiresTokenizer = !trimmed.lowercased().hasSuffix(".gguf")
                && !backend.lowercased().contains("gguf")
            // 用户显式选定了一个具体权重文件（config 里写的是文件路径）时，
            // registry 会把它归一成父目录，但运行时按用户选的文件直接采用，
            // 文件名不必在候选名单里 —— 这类不能套白名单，否则永远判缺失。
            var explicitIsDirectory: ObjCBool = false
            let explicitWeightFile = ["onnx", "safetensors", "gguf"].contains(
                (trimmed as NSString).pathExtension.lowercased())
                && FileManager.default.fileExists(
                    atPath: (trimmed as NSString).expandingTildeInPath,
                    isDirectory: &explicitIsDirectory)
                && !explicitIsDirectory.boolValue
            let weightNames: Set<String>? = backend.lowercased() == "onnx"
                && !explicitWeightFile ? onnxWeightNames : nil
            out.append(RequiredModel(
                kind: kind,
                repo: "\(HFModelDownloader.repoOwner)/\(repoName)",
                directory: dir,
                allowedWeightNames: weightNames,
                requiresTokenizer: requiresTokenizer))
        }

        if config.llmProvider == "local" {
            add(kind: "主模型", registryKind: .llm, value: config.localLLMModel)
        }
        if config.auxEnabled, config.auxLLMProvider == "local" {
            add(kind: "辅助模型", registryKind: .aux, value: config.auxLocalLLMModel)
        }
        if config.embeddingProvider == "local" {
            add(kind: "Embedding", registryKind: .embedding,
                value: embeddingValue(config))
        }
        if config.rerankerProvider == "local" {
            add(kind: "Reranker", registryKind: .reranker,
                value: rerankerValue(config))
        }
        if config.ragOCREngine == "mlx-vlm" {
            let value = !config.ragOCRModelPath.isEmpty
                ? config.ragOCRModelPath
                : (!config.localVLMModel.isEmpty
                    ? config.localVLMModel : config.ragOCRModelName)
            add(kind: "OCR", registryKind: .vlm, value: value)
        }
        if config.ragSemanticChunking {
            // 语义切片模型：lite 版随资源 bundle 内置（modelDirectory 直接
            // 命中 bundle，isComplete 为真不触发下载）；clean 版不带，从
            // 仓库 int4/ 子目录下载并剥前缀，落点与加载查找保持一致。
            let dir = PotionSentenceSimilarity.modelDirectory()
                ?? PotionSentenceSimilarity.downloadDirectory()
            out.append(RequiredModel(
                kind: "语义切片",
                repo: "\(HFModelDownloader.repoOwner)/potion-multilingual-128M-onnx",
                directory: dir,
                subpath: "int4"))
        }
        return out
    }

    /// 下载清单过滤：按仓库子目录剥前缀 + 按运行时可加载权重白名单筛选。
    ///
    /// potion 等多量化档仓库（fp16/int8/int4 平级）只取目标档位；ONNX 后端
    /// 只取 `resolveONNXModel` 候选名单里的权重，避免把 fp16 / 原版
    /// safetensors 这类运行时永远不读的数百 MB 文件拉下来。
    public static func scopedFiles(
        for model: RequiredModel,
        in files: [HFModelDownloader.FileEntry]
    ) -> [HFModelDownloader.FileEntry] {
        let subpath = model.subpath?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        // 普通模型（无子目录、无白名单）保持原样返回：清单在
        // listNeededFiles 阶段已按 isNeededFile 过滤过一次，不重复处理。
        guard !subpath.isEmpty || model.allowedWeightNames != nil else {
            return files
        }
        let prefix = subpath.isEmpty ? "" : (subpath.hasSuffix("/") ? subpath : subpath + "/")
        var out: [HFModelDownloader.FileEntry] = []
        for file in files {
            let localPath: String
            if prefix.isEmpty {
                localPath = file.path
            } else {
                guard file.path.hasPrefix(prefix) else { continue }
                localPath = String(file.path.dropFirst(prefix.count))
            }
            guard !localPath.isEmpty,
                  HFModelDownloader.isNeededFile(localPath) else { continue }
            // 权重文件必须在运行时可加载名单内（非权重文件如 config/tokenizer 放行）。
            if let allowed = model.allowedWeightNames,
               isWeightPath(localPath),
               !allowed.contains((localPath as NSString).lastPathComponent) {
                continue
            }
            out.append(HFModelDownloader.FileEntry(path: localPath,
                                                  size: file.size,
                                                  remotePath: file.remotePath))
        }
        return out
    }

    /// 缺失（目录不存在或没有可加载的权重文件）的必需模型。
    public static func missingModels(for config: AgentConfig) -> [RequiredModel] {
        models(for: config).filter { !isComplete($0) }
    }

    /// 与运行时同源的就绪判定：只认可加载文件名，且体积必须达标；
    /// MLX / ONNX 还必须有 tokenizer 文件，否则「权重已下完、tokenizer 还没下」
    /// 会被误判为就绪，启动加载时才报缺少 tokenizer。
    public static func isComplete(_ model: RequiredModel) -> Bool {
        guard isComplete(model.directory, allowedWeightNames: model.allowedWeightNames)
        else { return false }
        return !model.requiresTokenizer || hasTokenizerFile(in: model.directory)
    }

    /// 加载必需的 tokenizer 文件是否存在。
    ///
    /// 只认 `tokenizer.json`：运行时 `LocalTokenizer.load` 明确要求它，
    /// 只有 `tokenizer.model` / `spiece.model` 时模型同样加载失败。
    /// ONNX 仓库可能把 tokenizer 放在 `onnx/` 子目录，两处都查。
    static func hasTokenizerFile(in directory: URL) -> Bool {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: directory.path, isDirectory: &isDirectory) else {
            return false
        }
        // 显式指定的单文件模型（如部分 GGUF）不需要目录内 tokenizer。
        guard isDirectory.boolValue else { return true }
        let scopes = [directory, directory.appendingPathComponent("onnx", isDirectory: true)]
        for scope in scopes {
            let url = scope.appendingPathComponent("tokenizer.json")
            if fm.fileExists(atPath: url.path), HFModelDownloader.fileSize(at: url) > 0 {
                return true
            }
        }
        return false
    }

    /// 宽松版（不限制权重文件名），供按目录查询使用。
    public static func isComplete(_ directory: URL) -> Bool {
        isComplete(directory, allowedWeightNames: nil)
    }

    /// 必须覆盖用户自配模型的全部形态，否则会被永远判定缺失：
    /// - registry 对显式选定的 `.onnx` / `.safetensors` / `.gguf` 文件
    ///   直接返回**文件 URL** 而非目录；
    /// - ONNX 权重可能位于 `onnx/` 子目录（registry 的候选搜索范围）；
    /// - 分片权重为 `model-00001-of-0000N.safetensors`（按扩展名命中）。
    ///
    /// 体积校验是必需的：下载中断会留下 1 字节占位文件，若算作就绪，
    /// 启动会跳过补全、真正加载时才失败（实测 embedding 就踩过这个坑）。
    private static func isComplete(_ directory: URL,
                                   allowedWeightNames: Set<String>?) -> Bool {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: directory.path, isDirectory: &isDirectory) else {
            return false
        }
        // 权重文件最小也在数十 KB；4KB 阈值足够保守，只挡空文件/占位符。
        let minWeightBytes: Int64 = 4_096
        func usable(_ name: String, at url: URL) -> Bool {
            guard isWeightPath(name) else { return false }
            if let allowed = allowedWeightNames,
               !allowed.contains((name as NSString).lastPathComponent) {
                return false
            }
            return HFModelDownloader.fileSize(at: url) >= minWeightBytes
        }
        // 用户显式指定的单个权重文件：运行时 `resolveONNXModel` 直接采用它、
        // 不走候选名单，所以这里也不能套白名单（只校验体积与扩展名）。
        if !isDirectory.boolValue {
            return isWeightPath(directory.lastPathComponent)
                && HFModelDownloader.fileSize(at: directory) >= minWeightBytes
        }
        // 目录：顶层与 onnx/ 子目录（与 registry 的 ONNX 候选范围一致）。
        let scopes = [directory, directory.appendingPathComponent("onnx", isDirectory: true)]
        for scope in scopes {
            let items = (try? fm.contentsOfDirectory(atPath: scope.path)) ?? []
            for item in items where usable(item, at: scope.appendingPathComponent(item)) {
                return true
            }
        }
        return false
    }

    static func isWeightPath(_ path: String) -> Bool {
        let ext = (path as NSString).pathExtension.lowercased()
        return ext == "safetensors" || ext == "onnx" || ext == "gguf" || ext == "npz"
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
