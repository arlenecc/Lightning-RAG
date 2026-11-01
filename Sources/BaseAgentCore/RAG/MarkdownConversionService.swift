import Foundation

// MARK: - 请求与进度

/// 独立文件转 Markdown 的请求。解析/OCR 设置全部来自 `config`，
/// 不读写知识库 manifest，也不触发 embedding/入库。
public struct MarkdownConversionRequest: Sendable {
    public var sourceURL: URL
    public var outputDirectory: URL
    public var config: AgentConfig

    public init(sourceURL: URL, outputDirectory: URL, config: AgentConfig) {
        self.sourceURL = sourceURL
        self.outputDirectory = outputDirectory
        self.config = config
    }
}

public struct MarkdownConversionProgress: Sendable {
    public enum Phase: Sendable, Equatable {
        case scanning
        case parsing
        case writing
        case finished
    }

    public var phase: Phase
    /// 1-based；扫描阶段为 0。
    public var fileIndex: Int
    public var fileTotal: Int
    public var fileName: String
    public var stage: String
    public var pageIndex: Int?
    public var pageTotal: Int?
    public var completedFiles: Int
    public var failedFiles: Int
    /// 0...1 总体进度。
    public var overallFraction: Double

    public init(phase: Phase,
                fileIndex: Int = 0,
                fileTotal: Int = 0,
                fileName: String = "",
                stage: String,
                pageIndex: Int? = nil,
                pageTotal: Int? = nil,
                completedFiles: Int = 0,
                failedFiles: Int = 0,
                overallFraction: Double = 0) {
        self.phase = phase
        self.fileIndex = fileIndex
        self.fileTotal = fileTotal
        self.fileName = fileName
        self.stage = stage
        self.pageIndex = pageIndex
        self.pageTotal = pageTotal
        self.completedFiles = completedFiles
        self.failedFiles = failedFiles
        self.overallFraction = min(1, max(0, overallFraction))
    }
}

public struct MarkdownConversionResult: Sendable, Identifiable {
    public var id: String { source.path }
    public let source: URL
    public let outputURL: URL?
    public let error: String?
    public let elapsed: TimeInterval

    public init(source: URL, outputURL: URL?, error: String?, elapsed: TimeInterval) {
        self.source = source
        self.outputURL = outputURL
        self.error = error
        self.elapsed = elapsed
    }

    public var succeeded: Bool { error == nil && outputURL != nil }
}

public struct MarkdownConversionReport: Sendable {
    public let results: [MarkdownConversionResult]
    public let wasCancelled: Bool
    public let elapsed: TimeInterval

    public init(results: [MarkdownConversionResult],
                wasCancelled: Bool,
                elapsed: TimeInterval) {
        self.results = results
        self.wasCancelled = wasCancelled
        self.elapsed = elapsed
    }

    public var succeededCount: Int { results.filter(\.succeeded).count }
    public var failedCount: Int { results.count - succeededCount }
}

public enum MarkdownConversionError: LocalizedError, Sendable {
    case sourceNotFound(String)
    case sourceNotSupported(String)
    case outputNotDirectory(String)
    case noSupportedFiles(String)
    case cannotPrepareOutput(String)

    public var errorDescription: String? {
        switch self {
        case .sourceNotFound(let path):
            return "待转换路径不存在：\(path)"
        case .sourceNotSupported(let name):
            return "不支持的文件格式：\(name)"
        case .outputNotDirectory(let path):
            return "保存目录不是文件夹：\(path)"
        case .noSupportedFiles(let path):
            return "目录中没有可转换的文件：\(path)"
        case .cannotPrepareOutput(let message):
            return "无法准备保存目录：\(message)"
        }
    }
}

// MARK: - 转换服务

/// 复用 RAG 的 `DocumentParserRegistry`，把单个文件或整个目录转换成
/// 独立的 Markdown 文件。解析、OCR、图片归一化、清洗与入库使用同一套
/// 解析器，唯一区别是不做切片、embedding 和向量库写入。
///
/// 执行策略与主流程一致：轻文件按 `ragIngestConcurrency`（最多 6）分批并发，
/// 大文件单独成批且串行；PDF 内部再按文件体积收紧 worker / 在途页 / 批大小，
/// 并受全局页面图像闸门约束。这样目录转换既有并发加速，也不会因多份大文件
/// 同时解析把内存顶爆。
public actor MarkdownConversionService {
    private struct SourceFile: Sendable {
        let url: URL
        let relativePath: String
    }

    /// 一个待转换文件及其预先分配的输出路径。
    private struct FileTask: Sendable {
        let index: Int
        let file: SourceFile
        let outputURL: URL
    }

    private struct FileOutcome: Sendable {
        let index: Int
        /// 取消的批次不产生结果（与串行版本一致）。
        let result: MarkdownConversionResult?
        let cancelled: Bool
    }

    /// 大文件阈值：达到该体积的 PDF/文档单独成批，不与任何文件并发。
    /// 大文件内部的页级并发已由 `PDFProcessingPlan` 按体积收紧。
    static let heavyFileBytes: Int64 = 32 * 1024 * 1024

    public init() {}

    public func convert(
        request: MarkdownConversionRequest,
        progress: ((MarkdownConversionProgress) -> Void)? = nil,
        onResult: ((MarkdownConversionResult) -> Void)? = nil
    ) async throws -> MarkdownConversionReport {
        let startedAt = Date()
        // 转换工具复用 OCR 解析器：整批转换期间保持 Ovis 实例，批次结束
        // 立即释放，避免转换完成后权重常驻。
        await LocalModelFactory.shared.beginOCRJob()
        defer { Task { await LocalModelFactory.shared.endOCRJob() } }
        let fm = FileManager.default
        let source = request.sourceURL.standardizedFileURL
        let output = request.outputDirectory.standardizedFileURL

        var sourceIsDirectory: ObjCBool = false
        guard fm.fileExists(atPath: source.path, isDirectory: &sourceIsDirectory) else {
            throw MarkdownConversionError.sourceNotFound(source.path)
        }

        var outputIsDirectory: ObjCBool = false
        if fm.fileExists(atPath: output.path, isDirectory: &outputIsDirectory) {
            guard outputIsDirectory.boolValue else {
                throw MarkdownConversionError.outputNotDirectory(output.path)
            }
        } else {
            do {
                try fm.createDirectory(at: output, withIntermediateDirectories: true)
            } catch {
                throw MarkdownConversionError.cannotPrepareOutput(error.localizedDescription)
            }
        }

        progress?(MarkdownConversionProgress(
            phase: .scanning,
            stage: "扫描待转换文件",
            overallFraction: 0))

        let files = try Self.discover(source: source,
                                      sourceIsDirectory: sourceIsDirectory.boolValue,
                                      output: output)
        guard !files.isEmpty else {
            throw MarkdownConversionError.noSupportedFiles(source.path)
        }

        let total = files.count
        progress?(MarkdownConversionProgress(
            phase: .scanning,
            fileTotal: total,
            stage: "发现 \(total) 个待转换文件",
            overallFraction: 0))

        let config = request.config
        // 与主流程一致的文件级并发：默认 10 会夹到 6；单文件或大文件
        // 批次内实际上只有 1 个任务。转换工具不做 embedding/入库，
        // 因此这里只保留解析阶段的并发。
        let baseConcurrency = min(max(1, config.ragIngestConcurrency), 6, max(1, total))
        let fileSizes = files.map { PDFProcessingPlan.fileSize(of: $0.url) }
        let batches = Self.batchIndices(fileSizes: fileSizes,
                                        baseConcurrency: baseConcurrency)

        // 输出路径在并发开始前按发现顺序分配，保证 a.txt/a.html 的
        // a.md / a-2.md 与串行版本完全一致。
        var usedOutputs: Set<String> = []
        let outputURLs = files.map { file in
            Self.uniqueOutputURL(
                Self.desiredOutputURL(for: file,
                                      sourceIsDirectory: sourceIsDirectory.boolValue,
                                      output: output),
                used: &usedOutputs)
        }

        let registry = DocumentParserRegistry.makeDefault(config: config)
        var results: [MarkdownConversionResult] = []
        var succeeded = 0
        var failed = 0
        var cancelled = false

        for batch in batches {
            if Task.isCancelled {
                cancelled = true
                break
            }
            let baseCompleted = succeeded
            let baseFailed = failed
            let tasks = batch.map {
                FileTask(index: $0, file: files[$0], outputURL: outputURLs[$0])
            }

            // 同一批内并发解析；大文件批次只有一个元素，等价于串行。
            // 批结束时才合并结果，保证 results 顺序与发现顺序一致。
            let outcomes = await withTaskGroup(of: FileOutcome.self) { group in
                for task in tasks {
                    group.addTask {
                        await Self.convertFile(
                            task,
                            config: config,
                            registry: registry,
                            total: total,
                            baseCompleted: baseCompleted,
                            baseFailed: baseFailed,
                            progress: progress)
                    }
                }
                var collected: [FileOutcome] = []
                for await outcome in group { collected.append(outcome) }
                return collected.sorted { $0.index < $1.index }
            }

            for outcome in outcomes {
                if outcome.cancelled {
                    cancelled = true
                    continue
                }
                guard let result = outcome.result else { continue }
                results.append(result)
                if result.succeeded { succeeded += 1 } else { failed += 1 }
                onResult?(result)
                progress?(MarkdownConversionProgress(
                    phase: .writing,
                    fileIndex: outcome.index + 1,
                    fileTotal: total,
                    fileName: files[outcome.index].relativePath,
                    stage: result.succeeded
                        ? "已写入 \(result.outputURL?.lastPathComponent ?? "Markdown")"
                        : "失败：\(result.error ?? "未知错误")",
                    completedFiles: succeeded,
                    failedFiles: failed,
                    overallFraction: Double(succeeded + failed) / Double(total)))
            }

            if cancelled { break }
        }

        let processed = succeeded + failed
        progress?(MarkdownConversionProgress(
            phase: .finished,
            fileTotal: total,
            stage: cancelled ? "已取消" : "转换完成",
            completedFiles: succeeded,
            failedFiles: failed,
            overallFraction: cancelled ? Double(processed) / Double(total) : 1))

        return MarkdownConversionReport(
            results: results,
            wasCancelled: cancelled,
            elapsed: Date().timeIntervalSince(startedAt))
    }

    // MARK: - 单文件转换（供并发批次调用）

    /// 解析、清洗并写出单个文件的 Markdown。
    ///
    /// 该方法是 `nonisolated` 的静态函数：由 `convert` 的 task group 并发调用，
    /// 不触碰 actor 状态；每个文件通过 `registry.freshParser` 拿到独立解析器，
    /// 与主流程的文件级并发完全一致。
    private static func convertFile(
        _ task: FileTask,
        config: AgentConfig,
        registry: DocumentParserRegistry,
        total: Int,
        baseCompleted: Int,
        baseFailed: Int,
        progress: ((MarkdownConversionProgress) -> Void)?
    ) async -> FileOutcome {
        guard !Task.isCancelled else {
            return FileOutcome(index: task.index, result: nil, cancelled: true)
        }

        let fileIndex = task.index + 1
        let fileName = task.file.relativePath
        let fileStarted = Date()
        let baseFraction = Double(baseCompleted + baseFailed) / Double(total)

        progress?(MarkdownConversionProgress(
            phase: .parsing,
            fileIndex: fileIndex,
            fileTotal: total,
            fileName: fileName,
            stage: "解析与提取",
            completedFiles: baseCompleted,
            failedFiles: baseFailed,
            overallFraction: baseFraction))

        do {
            let options = ParserOptions(config: config)
            guard let parser = registry.freshParser(for: task.file.url,
                                                    options: options) else {
                throw MarkdownConversionError.sourceNotSupported(fileName)
            }

            let parsed = try await parser.parse(url: task.file.url,
                                                options: options) { event in
                switch event {
                case .page(let pageIndex, let pageTotal, let rawStage):
                    // 解析占单文件进度的 90%，留出清洗/写盘时间。
                    let pageFraction = Double(max(0, pageIndex))
                        / Double(max(1, pageTotal))
                    let intra = 0.9 * min(1, max(0, pageFraction))
                    progress?(MarkdownConversionProgress(
                        phase: .parsing,
                        fileIndex: fileIndex,
                        fileTotal: total,
                        fileName: fileName,
                        stage: stageText(rawStage),
                        pageIndex: pageIndex,
                        pageTotal: pageTotal,
                        completedFiles: baseCompleted,
                        failedFiles: baseFailed,
                        overallFraction: (Double(baseCompleted + baseFailed) + intra)
                            / Double(total)))
                case .message(let message):
                    progress?(MarkdownConversionProgress(
                        phase: .parsing,
                        fileIndex: fileIndex,
                        fileTotal: total,
                        fileName: fileName,
                        stage: message,
                        completedFiles: baseCompleted,
                        failedFiles: baseFailed,
                        overallFraction: baseFraction))
                }
            }

            if Task.isCancelled {
                return FileOutcome(index: task.index, result: nil, cancelled: true)
            }

            let text = parsed.metadata["cleaned"] == "1"
                ? parsed.text
                : DocumentCleaner.clean(
                    parsed.text,
                    keepHeadersFooters: false,
                    profile: DocumentProfile.defaultProfile(for: task.file.url))
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw AgentError.parse("解析结果为空")
            }

            progress?(MarkdownConversionProgress(
                phase: .writing,
                fileIndex: fileIndex,
                fileTotal: total,
                fileName: fileName,
                stage: "结果清洗 / Markdown 转换",
                completedFiles: baseCompleted,
                failedFiles: baseFailed,
                overallFraction: (Double(baseCompleted + baseFailed) + 0.95)
                    / Double(total)))

            try FileManager.default.createDirectory(
                at: task.outputURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            let body = text.hasSuffix("\n") ? text : text + "\n"
            try autoreleasepool {
                try body.write(to: task.outputURL, atomically: true, encoding: .utf8)
            }

            return FileOutcome(
                index: task.index,
                result: MarkdownConversionResult(
                    source: task.file.url,
                    outputURL: task.outputURL,
                    error: nil,
                    elapsed: Date().timeIntervalSince(fileStarted)),
                cancelled: false)
        } catch {
            if Task.isCancelled || isCancellation(error) {
                return FileOutcome(index: task.index, result: nil, cancelled: true)
            }
            return FileOutcome(
                index: task.index,
                result: MarkdownConversionResult(
                    source: task.file.url,
                    outputURL: nil,
                    error: error.localizedDescription,
                    elapsed: Date().timeIntervalSince(fileStarted)),
                cancelled: false)
        }
    }

    // MARK: - 分批

    /// 按文件体积分批：达到 `heavyFileBytes` 的大文件单独成批（批间串行，
    /// 因此永远不会多份大 PDF 同时解析）；轻文件按 `baseConcurrency`
    /// （最多 6）打包，批内并发。返回的下标始终保持发现顺序。
    static func batchIndices(fileSizes: [Int64], baseConcurrency: Int) -> [[Int]] {
        PDFProcessingPlan.batchIndices(fileSizes: fileSizes,
                                       baseConcurrency: baseConcurrency,
                                       heavyFileBytes: heavyFileBytes)
    }

    // MARK: - 文件发现与输出路径

    private static func discover(source: URL,
                                 sourceIsDirectory: Bool,
                                 output: URL) throws -> [SourceFile] {
        let fm = FileManager.default
        let sourcePath = source.path

        if !sourceIsDirectory {
            let ext = source.pathExtension.lowercased()
            guard RAGEngine.supportedExtensions.contains(ext) else {
                throw MarkdownConversionError.sourceNotSupported(source.lastPathComponent)
            }
            return [SourceFile(url: source, relativePath: source.lastPathComponent)]
        }

        guard let enumerator = fm.enumerator(
            at: source,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]) else {
            throw MarkdownConversionError.sourceNotFound(sourcePath)
        }

        let outputPath = output.path
        let prefix = sourcePath.hasSuffix("/") ? sourcePath : sourcePath + "/"
        let skipOutput = outputPath != sourcePath && outputPath.hasPrefix(prefix)
        var files: [SourceFile] = []

        for case let url as URL in enumerator {
            let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey])
            if values?.isDirectory == true {
                // 输出目录位于待转换目录内时跳过，避免把上次结果再次转换。
                if skipOutput, url.standardizedFileURL.path == outputPath {
                    enumerator.skipDescendants()
                }
                continue
            }
            guard values?.isRegularFile == true else { continue }
            let ext = url.pathExtension.lowercased()
            guard RAGEngine.supportedExtensions.contains(ext) else { continue }

            let path = url.standardizedFileURL.path
            let relative = path.hasPrefix(prefix)
                ? String(path.dropFirst(prefix.count))
                : url.lastPathComponent
            files.append(SourceFile(url: url, relativePath: relative))
        }

        files.sort {
            $0.relativePath.localizedStandardCompare($1.relativePath) == .orderedAscending
        }
        return files
    }

    private static func desiredOutputURL(for file: SourceFile,
                                         sourceIsDirectory: Bool,
                                         output: URL) -> URL {
        let relative = sourceIsDirectory
            ? file.relativePath
            : file.url.lastPathComponent
        // 用 NSString 处理相对路径：URL(fileURLWithPath:) 会把没有目录的
        // 文件名解析到当前工作目录，导致输出路径被污染。
        let nsRelative = relative as NSString
        let directory = nsRelative.deletingLastPathComponent
        // 只对文件名去扩展名，避免把目录一起带进新文件名（sub/sub/b.md）。
        let base = (nsRelative.lastPathComponent as NSString).deletingPathExtension
        let dir = (directory.isEmpty || directory == ".")
            ? output
            : output.appendingPathComponent(directory)
        return dir.appendingPathComponent(base + ".md")
    }

    /// 同一批里 `a.txt` 与 `a.html` 都会得到 `a.md`，后者自动改为 `a-2.md`。
    private static func uniqueOutputURL(_ desired: URL,
                                        used: inout Set<String>) -> URL {
        let standardized = desired.standardizedFileURL
        if used.insert(standardized.path).inserted {
            return desired
        }
        let dir = desired.deletingLastPathComponent()
        let base = desired.deletingPathExtension().lastPathComponent
        var index = 2
        while true {
            let candidate = dir.appendingPathComponent("\(base)-\(index).md")
            if used.insert(candidate.standardizedFileURL.path).inserted {
                return candidate
            }
            index += 1
        }
    }

    private static func stageText(_ stage: String) -> String {
        switch stage {
        case "render": return "页面渲染 / 图片归一化"
        case "text": return "提取文本层"
        case "ocr": return "图片识别"
        case "normalize": return "图片归一化"
        default: return stage.isEmpty ? "解析" : stage
        }
    }

    private static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        if let agentError = error as? AgentError,
           case .cancelled = agentError {
            return true
        }
        return false
    }
}
