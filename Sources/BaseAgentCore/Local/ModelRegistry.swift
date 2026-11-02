import Foundation

/// 本地模型类别。不同类别的默认目录名与查找顺序略有差异。
public enum LocalModelKind: String, Sendable {
    case embedding
    case reranker
    case llm
    case aux
    case vlm

    /// 错误信息中展示的中文名称。
    public var displayName: String {
        switch self {
        case .embedding: return "Embedding 模型"
        case .reranker: return "Reranker 模型"
        case .llm: return "主 LLM 模型"
        case .aux: return "辅助 LLM 模型"
        case .vlm: return "VLM（OCR）模型"
        }
    }
}

/// 本地模型层统一错误。所有错误都带可执行的修复提示，避免静默降级。
public enum LocalModelError: LocalizedError, Sendable {
    case notConfigured(LocalModelKind, hint: String)
    case pathNotFound(LocalModelKind, path: String)
    case notFound(LocalModelKind, value: String, searched: [String], hint: String)
    case missingONNXModel(LocalModelKind, directory: String, expected: [String])
    case missingTokenizer(directory: String, file: String)
    case invalidConfiguration(String)
    case unsupportedArchitecture(String)
    case runtime(String)

    public var errorDescription: String? {
        switch self {
        case .notConfigured(let kind, let hint):
            return "未配置\(kind.displayName)：\(hint)"
        case .pathNotFound(let kind, let path):
            return "\(kind.displayName)的显式路径不存在：\(path)"
        case .notFound(let kind, let value, let searched, let hint):
            let list = searched.map { "  - \($0)" }.joined(separator: "\n")
            return "找不到\(kind.displayName)：\(value)\n已检查以下位置：\n\(list)\n\(hint)"
        case .missingONNXModel(let kind, let directory, let expected):
            return "\(kind.displayName)目录中缺少 ONNX 权重：\(directory)\n期望文件名（按优先级）：\(expected.joined(separator: "、"))"
        case .missingTokenizer(let directory, let file):
            return "模型目录缺少 tokenizer 文件：\(directory)/\(file)。请确保 tokenizer.json 与模型权重位于同一目录（tokenizer_config.json 可选）。"
        case .invalidConfiguration(let message):
            return "本地模型配置无效：\(message)"
        case .unsupportedArchitecture(let message):
            return "本地模型架构不受当前 mlx-swift-lm 支持：\(message)"
        case .runtime(let message):
            return "本地模型运行失败：\(message)"
        }
    }
}

/// 模型文件发现与路径解析。
///
/// 解析优先级：
/// 1. 配置中显式填写的路径（绝对路径、`~` 或 `./` 开头）——不存在时直接报错；
/// 2. `models/<name>` 锚定应用所在目录的 `models`，随后回退到配置目录
///    `~/.base-agent/models`（应用目录不可写时模型下载的落点）；
/// 3. 应用目录 `models/<name>`、`<workspace>/models/<name>`、`~/base-agent/models/<name>`；
/// 4. HuggingFace / FastEmbed 缓存目录（含 `models--org--name/snapshots/<rev>` 布局）。
public enum LocalModelRegistry {

    /// 兼容 FastEmbed 的特殊命名：配置里是量化别名，但缓存/仓库使用上游真实模型名。
    private static let repoAliases: [String: [String]] = [
        "nomic-ai/nomic-embed-text-v1.5-Q": ["nomic-ai/nomic-embed-text-v1.5"],
        "BAAI/bge-reranker-v2-m3-int8": [
            "BAAI/bge-reranker-v2-m3",
            "onnx-community/bge-reranker-v2-m3-ONNX",
        ],
        "BAAI/bge-reranker-v2-m3": [
            "BAAI/bge-reranker-v2-m3-int8",
            "onnx-community/bge-reranker-v2-m3-ONNX",
        ],
    ]

    /// 当调用方没有给出任何模型名时使用的兜底名称。
    public static func defaultName(for kind: LocalModelKind) -> String? {
        switch kind {
        case .embedding: return "jina-embeddings-v5-text-nano-mlx-int4"
        case .reranker: return "jina-reranker-v2-base-multilingual-onnx"
        case .llm: return "qwen3.5-4b-agentic-coder-v4-4bit-mlx"
        case .aux: return "Qwen3.5-2B-MLX-4bit"
        case .vlm: return nil
        }
    }

    /// 应用所在目录下的 `models` 目录。
    ///
    /// 打包后从 `Lightning-RAG.app` 所在目录向上查找（如 `dist/Lightning-RAG.app` →
    /// `dist/models`，再向上到项目根的 `models`）；裸可执行文件从可执行文件
    /// 所在目录向上查找。`BASE_AGENT_MODELS_DIR` 可显式覆盖模型根目录。
    public static func applicationModelsDirectory() -> URL? {
        let fm = FileManager.default
        if let override = ProcessInfo.processInfo.environment["BASE_AGENT_MODELS_DIR"],
           !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let url = URL(fileURLWithPath: expandPath(override), isDirectory: true)
            if fm.fileExists(atPath: url.path) {
                return url
            }
        }

        // 打包后的 App 把模型放在 `Contents/Resources/models`；
        // 优先于 App 所在目录外的 models，确保分发版使用内置权重。
        if let bundled = bundledModelsDirectory(
            resourceURL: Bundle.main.resourceURL,
            bundleURL: Bundle.main.bundleURL) {
            return bundled
        }

        var directory = Bundle.main.bundleURL.standardizedFileURL
        if directory.pathExtension.lowercased() == "app" {
            // `.app` 包本身不能作为模型目录的宿主，从它所在目录开始找。
            directory = directory.deletingLastPathComponent()
        }
        for _ in 0..<6 {
            let candidate = directory.appendingPathComponent("models", isDirectory: true)
            if fm.fileExists(atPath: candidate.path) {
                return candidate
            }
            let parent = directory.deletingLastPathComponent()
            if parent.path == directory.path {
                break
            }
            directory = parent
        }
        return nil
    }

    /// App bundle 内的模型目录（`Contents/Resources/models`）。
    ///
    /// 参数化 resource/bundle URL 便于用临时目录模拟 bundle 布局；
    /// 找不到时返回 nil，调用方继续按 App 所在目录搜索。
    static func bundledModelsDirectory(resourceURL: URL?, bundleURL: URL) -> URL? {
        let fm = FileManager.default
        var candidates: [URL] = []
        if let resourceURL {
            candidates.append(resourceURL.appendingPathComponent("models", isDirectory: true))
        }
        candidates.append(bundleURL.appendingPathComponent(
            "Contents/Resources/models", isDirectory: true))
        for candidate in candidates where fm.fileExists(atPath: candidate.path) {
            return candidate
        }
        return nil
    }

    /// 配置文件目录（`~/.base-agent`）下的 `models` 子目录。
    ///
    /// App 安装到 `/Applications` 等位置时应用 models 目录（bundle 内）不可写，
    /// 模型下载补全落到这里；加载侧在应用 models 目录之后回退查找，
    /// 保证下载完成后的模型一定能被加载。
    public static func configModelsDirectory() -> URL {
        URL(fileURLWithPath: expandPath("~/.base-agent/models"), isDirectory: true)
    }

    /// 下载补全时模型应放置的目录：与 `resolveDirectory` 的加载锚点保持一致，
    /// 保证下载完成后加载路径一定能找到它。
    ///
    /// - 显式路径（绝对 / `~` / `./`）：原样展开；
    /// - `models/<name>`：应用 models 目录**可写**时放那里（App bundle /
    ///   可执行文件旁，`BASE_AGENT_MODELS_DIR` 优先）；应用目录不存在或
    ///   不可写（如安装到 `/Applications`）时放配置目录 `~/.base-agent/models`
    ///   ——加载侧在同位置回退查找；
    /// - 裸名：`<workspace>/models/<name>`（与裸名的候选搜索一致）。
    public static func downloadDirectory(
        kind: LocalModelKind,
        value: String,
        config: AgentConfig
    ) -> URL {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        let name = (trimmed as NSString).lastPathComponent
        if isExplicitPath(trimmed) {
            var url = URL(fileURLWithPath: expandPath(trimmed))
            if url.pathExtension.lowercased() == "gguf" {
                url = url.deletingLastPathComponent()
            }
            return url
        }
        if trimmed.hasPrefix("models/") {
            let fm = FileManager.default
            if let appModels = applicationModelsDirectory(),
               fm.isWritableFile(atPath: appModels.path) {
                return appModels.appendingPathComponent(name, isDirectory: true)
            }
            // 应用目录缺失或不可写（如装到 /Applications）：
            // 落到配置文件目录的 models 子目录。
            return configModelsDirectory().appendingPathComponent(name, isDirectory: true)
        }
        return URL(fileURLWithPath: config.modelsDirectory, isDirectory: true)
            .appendingPathComponent(name, isDirectory: true)
    }

    /// 解析模型目录。
    /// - Parameters:
    ///   - kind: 模型类别，仅用于错误提示与默认值。
    ///   - value: 配置中的路径 / 目录名 / HF repo id。
    ///   - config: 运行时配置，提供 `<workspace>/models`。
    public static func resolveDirectory(
        kind: LocalModelKind,
        value: String,
        config: AgentConfig,
        aliasFilter: ((String) -> Bool)? = nil
    ) throws -> URL {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        let effective: String
        if trimmed.isEmpty {
            guard let fallback = defaultName(for: kind) else {
                throw LocalModelError.notConfigured(
                    kind,
                    hint: "请在配置中填写 local_vlm_model 或 rag_ocr_model_path，并把模型目录放到应用所在目录的 models 目录下。"
                )
            }
            effective = fallback
        } else {
            effective = trimmed
        }

        let fm = FileManager.default
        let expanded = expandPath(effective)

        // 1) 显式路径：用户意图明确，路径不存在时不做猜测。
        if isExplicitPath(effective) {
            let url = URL(fileURLWithPath: expanded)
            guard fm.fileExists(atPath: url.path) else {
                throw LocalModelError.pathNotFound(kind, path: expanded)
            }
            return try normalizedModelDirectory(url)
        }

        // 2) `models/...` 锚定到应用所在目录的 models 目录（App bundle /
        //    可执行文件旁 / BASE_AGENT_MODELS_DIR），不回退到 workspace；
        //    随后回退到配置目录的 models（应用目录不可写时模型下载的落点），
        //    保证下载补全的模型一定能被加载。
        if effective.hasPrefix("models/") {
            var roots = applicationModelsDirectory().map { [$0] } ?? []
            roots.append(configModelsDirectory())
            var searched: [String] = []
            for candidate in candidateDirectories(
                kind: kind, value: effective, config: config,
                aliasFilter: aliasFilter, roots: roots, includeCaches: false) {
                searched.append(candidate.path)
                guard fm.fileExists(atPath: candidate.path) else { continue }
                if let resolved = try? normalizedModelDirectory(candidate) {
                    return resolved
                }
            }
            let rootPath = applicationModelsDirectory()?.path
                ?? LocalModelRegistry.configModelsDirectory().path
            throw LocalModelError.notFound(
                kind, value: effective, searched: searched,
                hint: "请将模型放到应用所在目录的 models 目录，或配置目录的 "
                    + "models 子目录（\(LocalModelRegistry.configModelsDirectory().path)）：\(rootPath)/<name>。")
        }

        // 3) 相对路径 / 直接写目录名且当前工作目录下存在。
        let relative = URL(fileURLWithPath: expanded)
        if fm.fileExists(atPath: relative.path) {
            return try normalizedModelDirectory(relative)
        }

        // 4) 应用目录 models、workspace/models、~/base-agent/models 与 HF/FastEmbed 缓存。
        var searched: [String] = []
        for candidate in candidateDirectories(
            kind: kind, value: effective, config: config, aliasFilter: aliasFilter) {
            searched.append(candidate.path)
            guard fm.fileExists(atPath: candidate.path) else { continue }
            if let resolved = try? normalizedModelDirectory(candidate) {
                return resolved
            }
        }

        let appRoot = applicationModelsDirectory()?.path ?? "<应用目录>/models"
        let hint = "请将模型放到应用所在目录的 models 目录：\(appRoot)/<name>，"
            + "或在配置中填写 local_embedding_model / local_reranker_model / local_llm_model / local_vlm_model 的显式路径。"
        throw LocalModelError.notFound(kind, value: effective, searched: searched, hint: hint)
    }

    /// 解析 ONNX 模型位置，供 LLM / Embedding / Reranker 的 ONNX 后端使用。
    ///
    /// - 若 `value` 显式指向一个存在的 `.onnx` 文件（用户在设置窗口选定了具体文件），
    ///   直接返回该文件 URL——尊重用户的明确选择，不再在目录内搜索。
    /// - 否则按目录语义调用 `resolveDirectory`（含 `models/` 锚定、相对路径、
    ///   应用/workspace/HF 缓存等回退）。
    ///
    /// 返回的文件或目录随后交给 `resolveONNXModel(in:)` 进一步定位权重文件。
    public static func resolveONNXDirectory(
        kind: LocalModelKind,
        value: String,
        config: AgentConfig,
        aliasFilter: ((String) -> Bool)? = nil
    ) throws -> URL {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty, trimmed.lowercased().hasSuffix(".onnx") {
            let url = URL(fileURLWithPath: expandPath(trimmed))
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir),
               !isDir.boolValue {
                return url
            }
            // 是目录或不存在，落到下方目录解析（不存在时由 resolveDirectory 报错）。
        }
        return try resolveDirectory(
            kind: kind, value: value, config: config, aliasFilter: aliasFilter)
    }

    /// 按约定顺序发现 ONNX 权重文件。
    ///
    /// 优先级（量化/小体积优先，因为 `model.onnx` 多为 FP16、体积大且不一定更快）：
    /// `model_quantized.onnx` → `model_int8.onnx` → `model_int4.onnx` → `model.onnx`，
    /// 先在当前目录查找，再在 `onnx/` 子目录查找。
    ///
    /// 若传入的路径本身就是一个存在的 `.onnx` 文件（用户在设置窗口显式选定），
    /// 则直接采用该文件，不再搜索——尊重用户的明确选择。
    public static func resolveONNXModel(
        in directory: URL,
        kind: LocalModelKind = .embedding
    ) throws -> URL {
        let fm = FileManager.default
        // 显式指向单个 .onnx 文件时直接采用，不再搜索。
        if directory.pathExtension.lowercased() == "onnx", fm.fileExists(atPath: directory.path) {
            return directory
        }
        guard fm.fileExists(atPath: directory.path) else {
            throw LocalModelError.missingONNXModel(
                kind, directory: directory.path, expected: onnxCandidateNames)
        }

        for relative in onnxCandidateNames {
            let url = directory.appendingPathComponent(relative)
            if fm.fileExists(atPath: url.path) {
                return url
            }
        }

        throw LocalModelError.missingONNXModel(
            kind, directory: directory.path, expected: onnxCandidateNames)
    }

    /// ONNX 权重候选文件名（按优先级）：量化/int8 > int4 > 未量化(FP16)。
    /// 先当前目录，再 `onnx/` 子目录。
    private static let onnxCandidateNames: [String] = [
        "model_quantized.onnx",
        "model_int8.onnx",
        "model_int4.onnx",
        "model.onnx",
        "onnx/model_quantized.onnx",
        "onnx/model_int8.onnx",
        "onnx/model_int4.onnx",
        "onnx/model.onnx",
    ]

    /// 从 config.json 推断最大序列长度，供 tokenizer 截断使用。
    public static func resolveMaxSequenceLength(directory: URL, defaultValue: Int = 8192) -> Int {
        guard let config = readJSON(directory.appendingPathComponent("config.json")) else {
            return defaultValue
        }
        if let n = config["n_positions"] as? Int, n > 0 {
            return max(128, n)
        }
        if let n = config["max_position_embeddings"] as? Int, n > 0 {
            return max(128, n)
        }
        return defaultValue
    }

    /// 从 tokenizer_config.json 读取 `model_max_length`（HF 官方声明的输入 token 上限）。
    ///
    /// XLM-R 系模型图内按 `cumsum(mask) + padding_idx` 计算位置索引，
    /// 有效输入长度是 `max_position_embeddings - 2` 而非 514/8194 本身，
    /// 因此 ONNX 推理的截断长度必须与 tokenizer 声明的 `model_max_length` 对齐。
    public static func tokenizerModelMaxLength(directory: URL) -> Int? {
        guard let config = readJSON(directory.appendingPathComponent("tokenizer_config.json")),
              let n = config["model_max_length"] as? Int, n > 0 else {
            return nil
        }
        return n
    }

    /// 读取 config.json 的 `model_type`（部分模型只有 `architectures`）。
    public static func modelType(in directory: URL) -> String? {
        guard let config = readJSON(directory.appendingPathComponent("config.json")) else {
            return nil
        }
        if let type = config["model_type"] as? String, !type.isEmpty {
            return type
        }
        if let architectures = config["architectures"] as? [String],
           let first = architectures.first {
            return first
        }
        return nil
    }

    /// 读取 tokenizer_config.json 的 tokenizer_class / model_type，用于判断 pair 编码风格。
    public static func tokenizerClass(in directory: URL) -> String? {
        guard let config = readJSON(directory.appendingPathComponent("tokenizer_config.json")) else {
            return nil
        }
        if let value = config["tokenizer_class"] as? String, !value.isEmpty {
            return value
        }
        if let value = config["model_type"] as? String, !value.isEmpty {
            return value
        }
        return nil
    }

    /// 判断是否为 Ovis 架构（当前 mlx-swift-lm 未注册该架构）。
    public static func isOvisArchitecture(directory: URL) -> Bool {
        guard let type = modelType(in: directory)?.lowercased() else { return false }
        return type.contains("ovis")
    }

    // MARK: - 内部工具

    private static func isExplicitPath(_ value: String) -> Bool {
        value.hasPrefix("/") || value.hasPrefix("~")
            || value.hasPrefix("./") || value.hasPrefix("../")
    }

    /// 若传入的是缓存仓库目录或快照父目录，则定位到真正包含权重的目录。
    private static func normalizedModelDirectory(_ url: URL) throws -> URL {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            throw LocalModelError.runtime("路径不存在：\(url.path)")
        }

        if !isDirectory.boolValue {
            let ext = url.pathExtension.lowercased()
            guard ["onnx", "safetensors", "gguf"].contains(ext) else {
                throw LocalModelError.invalidConfiguration("模型路径不是目录也不是已知权重文件：\(url.path)")
            }
            let parent = url.deletingLastPathComponent()
            return try snapshotDirectory(for: parent)
        }

        return try snapshotDirectory(for: url)
    }

    /// 处理 `models--org--name/{snapshots/<rev>, refs/main}` 布局。
    private static func snapshotDirectory(for directory: URL) throws -> URL {
        let fm = FileManager.default
        if hasModelContent(directory) {
            return directory
        }

        // refs/main 指向具体 revision 时优先使用。
        let refFile = directory.appendingPathComponent("refs/main")
        if let ref = try? String(contentsOf: refFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !ref.isEmpty {
            let pinned = directory.appendingPathComponent("snapshots/\(ref)")
            if hasModelContent(pinned) {
                return pinned
            }
        }

        let snapshots = directory.appendingPathComponent("snapshots")
        guard fm.fileExists(atPath: snapshots.path) else {
            throw LocalModelError.invalidConfiguration(
                "目录中没有模型文件（config.json / tokenizer.json / *.onnx / *.safetensors）：\(directory.path)"
            )
        }

        let entries = (try? fm.contentsOfDirectory(
            at: snapshots,
            includingPropertiesForKeys: [.contentModificationDateKey, .isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []

        let candidates = entries
            .filter { ((try? $0.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false) }
            .sorted { lhs, rhs in
                let l = ((try? lhs.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate) ?? .distantPast
                let r = ((try? rhs.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate) ?? .distantPast
                return l > r
            }
        for candidate in candidates where hasModelContent(candidate) {
            return candidate
        }

        throw LocalModelError.invalidConfiguration("HF 缓存快照中没有可用模型文件：\(snapshots.path)")
    }

    private static func hasModelContent(_ directory: URL) -> Bool {
        let fm = FileManager.default
        let markers = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        for marker in markers where fm.fileExists(atPath: directory.appendingPathComponent(marker).path) {
            return true
        }
        let weights = ["onnx", "safetensors", "gguf", "npz"]
        if let entries = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) {
            return entries.contains { weights.contains($0.pathExtension.lowercased()) }
        }
        return false
    }

    private static func candidateDirectories(
        kind: LocalModelKind,
        value: String,
        config: AgentConfig,
        aliasFilter: ((String) -> Bool)? = nil,
        roots: [URL]? = nil,
        includeCaches: Bool = true
    ) -> [URL] {
        let aliases = (repoAliases[value] ?? []).filter { aliasFilter?($0) ?? true }
        // 旧配置习惯写 `models/<name>`（相对应用根目录）。模型根目录本身已经
        // 是模型目录，此处去掉前缀后再拼接，避免解析成 `models/models/<name>`。
        let names = ([value] + aliases).flatMap { name -> [String] in
            if name.hasPrefix("models/") {
                return [name, String(name.dropFirst("models/".count))]
            }
            return [name]
        }
        // 应用所在目录的 models 优先，保证实际加载的是随应用放置的模型。
        let localRoots = roots ?? [
            applicationModelsDirectory(),
            URL(fileURLWithPath: expandPath(config.modelsDirectory), isDirectory: true),
            URL(fileURLWithPath: expandPath("~/base-agent/models"), isDirectory: true),
        ].compactMap { $0 }

        var result: [URL] = []
        var seen = Set<String>()
        func append(_ url: URL) {
            let path = url.standardizedFileURL.path
            if seen.insert(path).inserted {
                result.append(url)
            }
        }

        let lastComponents = names.compactMap { name -> String? in
            let component = (name as NSString).lastPathComponent
            return component.isEmpty ? nil : component
        }

        for root in localRoots {
            for name in names {
                if name.contains("/") {
                    // 支持 models/<org>/<name> 以及 models/<name> 两种本地布局。
                    append(root.appendingPathComponent(name, isDirectory: true))
                    append(root.appendingPathComponent((name as NSString).lastPathComponent, isDirectory: true))
                } else {
                    append(root.appendingPathComponent(name, isDirectory: true))
                }
            }
            // 量化别名与真实目录名共存的常见情况。
            for component in lastComponents {
                append(root.appendingPathComponent(component, isDirectory: true))
            }
        }

        for cacheRoot in includeCaches ? cacheRoots() : [] {
            for name in names where name.contains("/") {
                let repoDirName = "models--" + name.replacingOccurrences(of: "/", with: "--")
                append(cacheRoot.appendingPathComponent(repoDirName, isDirectory: true))
            }
            for component in lastComponents {
                let repoDirName = "models--" + component.replacingOccurrences(of: "/", with: "--")
                append(cacheRoot.appendingPathComponent(repoDirName, isDirectory: true))
            }
        }

        return result
    }

    private static func cacheRoots() -> [URL] {
        var roots: [URL] = []
        let env = ProcessInfo.processInfo.environment

        if let hubCache = env["HF_HUB_CACHE"], !hubCache.isEmpty {
            roots.append(URL(fileURLWithPath: expandPath(hubCache), isDirectory: true))
        }
        if let hfHome = env["HF_HOME"], !hfHome.isEmpty {
            roots.append(URL(fileURLWithPath: expandPath(hfHome), isDirectory: true)
                .appendingPathComponent("hub", isDirectory: true))
        }
        roots.append(URL(fileURLWithPath: expandPath("~/.cache/huggingface/hub"), isDirectory: true))
        roots.append(URL(fileURLWithPath: expandPath("~/Library/Caches/huggingface/hub"), isDirectory: true))
        roots.append(URL(fileURLWithPath: expandPath("~/.cache/fastembed"), isDirectory: true))
        return roots
    }

    private static func readJSON(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data),
              let dictionary = object as? [String: Any] else {
            return nil
        }
        return dictionary
    }
}
