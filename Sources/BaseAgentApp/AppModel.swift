import Foundation
import AppKit
import BaseAgentCore

// MARK: - UI 模型

struct ChatItem: Identifiable, Equatable {
    enum Role: String { case user, assistant, system }
    let id = UUID()
    var role: Role
    var content: String
    /// 用户消息携带的图像附件（JPEG/PNG 字节），仅用于 UI 缩略图展示。
    var images: [Data]?
}

struct ToolEvent: Identifiable {
    let id = UUID()
    var name: String
    var arguments: String
    var success: Bool?
    var output: String
    var timestamp = Date()
}

struct LogLine: Identifiable {
    let id = UUID()
    var text: String
    var timestamp = Date()
}

/// 协作式取消标志：Agent 回调在后台线程读取，UI 在主线程写入。
final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var isCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return value
    }
    func set(_ newValue: Bool) {
        lock.lock(); value = newValue; lock.unlock()
    }
}

// MARK: - Callbacks

/// 把 AgentCallbacks 的事件桥接到主线程 UI。
final class UICallbacks: AgentCallbacks, @unchecked Sendable {
    private weak var model: AppViewModel?
    private let cancel: CancelFlag

    init(model: AppViewModel, cancel: CancelFlag) {
        self.model = model
        self.cancel = cancel
    }

    func isCancelled() -> Bool { cancel.isCancelled }

    func onContent(_ text: String) {
        DispatchQueue.main.async { [weak self] in self?.model?.appendContent(text) }
    }

    func onReasoning(_ text: String) {
        DispatchQueue.main.async { [weak self] in self?.model?.appendReasoning(text) }
    }

    func onToolStart(_ name: String, _ args: [String: JSONValue]) {
        DispatchQueue.main.async { [weak self] in
            self?.model?.beginTool(name: name, args: args)
        }
    }

    func onToolEnd(_ name: String, _ result: ToolResult) {
        DispatchQueue.main.async { [weak self] in
            self?.model?.finishTool(name: name, result: result)
        }
    }

    func onLog(_ line: String) {
        DispatchQueue.main.async { [weak self] in self?.model?.appendLog(line) }
    }

    func onUsage(_ usage: [String: Int]) {
        DispatchQueue.main.async { [weak self] in
            self?.model?.totalTokens = usage["total_tokens"] ?? usage["total"] ?? 0
        }
    }

    func onTokenSpeed(_ totalTokens: Int, _ speed: Double) {
        DispatchQueue.main.async { [weak self] in
            self?.model?.tokenSpeed = speed
        }
    }

    func onPlanUpdated(_ plan: Plan?) {
        DispatchQueue.main.async { [weak self] in self?.model?.plan = plan }
    }

    func onError(_ message: String) {
        DispatchQueue.main.async { [weak self] in
            self?.model?.appendLog("ERROR \(message)")
            self?.model?.appendSystem(message)
        }
    }

    func onContextShrunk(_ summary: String, _ reason: String) {
        DispatchQueue.main.async { [weak self] in
            self?.model?.appendLog("[context] 已收缩（\(reason)）：\(summary.prefix(200))")
        }
    }

    func onSkillSuggested(_ skill: SkillSuggestion) {
        DispatchQueue.main.async { [weak self] in
            self?.model?.appendLog("[skill] 重复请求，可固化为技能：\(skill.name)")
        }
    }

    func confirm(_ message: String) -> Bool {
        runOnMain {
            let alert = NSAlert()
            alert.messageText = "需要确认"
            alert.alertStyle = .warning
            // 默认按钮是「拒绝」：回车/误触不会自动放行高风险操作。
            alert.addButton(withTitle: "拒绝")
            alert.addButton(withTitle: "允许")

            // 确认文案现在包含完整实参（不能只显示摘要，否则「批准 A、执行 B」）。
            // 长参数放进可滚动附件视图，既不撑爆弹窗，也保证用户能看完。
            if message.count > 600 {
                alert.informativeText = "请核对下面实际执行的完整参数："
                let frame = NSRect(x: 0, y: 0, width: 560, height: 260)
                let textView = NSTextView(frame: frame)
                textView.string = message
                textView.isEditable = false
                textView.isRichText = false
                textView.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
                textView.textContainerInset = NSSize(width: 4, height: 4)
                let scroll = NSScrollView(frame: frame)
                scroll.documentView = textView
                scroll.hasVerticalScroller = true
                scroll.borderType = .bezelBorder
                alert.accessoryView = scroll
            } else {
                alert.informativeText = message
            }
            return alert.runModal() == .alertSecondButtonReturn
        }
    }

    func askUser(_ prompt: String) -> String? {
        runOnMain {
            let alert = NSAlert()
            alert.messageText = "需要补充信息"
            alert.informativeText = prompt
            alert.addButton(withTitle: "提交")
            alert.addButton(withTitle: "取消")
            let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 420, height: 24))
            field.placeholderString = "请输入…"
            alert.accessoryView = field
            guard alert.runModal() == .alertFirstButtonReturn else { return nil }
            let value = field.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
            return value.isEmpty ? nil : value
        }
    }

    func onFinished() {
        DispatchQueue.main.async { [weak self] in self?.model?.finishRun() }
    }

    private func runOnMain<T>(_ body: () -> T) -> T {
        if Thread.isMainThread { return body() }
        return DispatchQueue.main.sync(execute: body)
    }
}

// MARK: - AppViewModel

@MainActor
final class AppViewModel: ObservableObject {
    @Published var config: AgentConfig
    /// 识别测试窗口使用的临时配置（含尚未保存的自定义 OCR 设置）；为空时回退 `config`。
    @Published var ocrLabConfig: AgentConfig?
    /// 转换工具窗口使用的临时配置（含尚未保存的 OCR / PDF 设置）；为空时回退 `config`。
    /// 仅作为解析参数快照，不写回 config.json，也不重建 Agent。
    @Published var converterConfig: AgentConfig?
    @Published var chatItems: [ChatItem] = []
    @Published var reasoning = ""
    @Published var logs: [LogLine] = []
    @Published var toolEvents: [ToolEvent] = []
    @Published var plan: Plan?
    @Published var isRunning = false {
        didSet { if !isRunning { flushQueuedSends() } }
    }
    @Published var statusText = "就绪"
    @Published var totalTokens = 0
    @Published var tokenSpeed: Double = 0
    @Published var ragStatusText = "RAG 未就绪"
    @Published var isSyncing = false {
        didSet { if !isSyncing { flushQueuedSends() } }
    }
    /// 已请求停止、正在等当前解析/embedding 批次收尾。
    @Published var isStoppingSync = false
    /// 正在手动清理向量库内的重复切片。
    @Published var isDeduplicating = false {
        didSet { if !isDeduplicating { flushQueuedSends() } }
    }
    /// 正在构建/续建向量库的 ANN 索引（构建期间检索会按超时跳过）。
    @Published var isBuildingANN = false {
        didSet { if !isBuildingANN { flushQueuedSends() } }
    }
    /// 正在构建知识图谱（辅助模型抽取实体/关系）。
    @Published var isBuildingKG = false {
        didSet { if !isBuildingKG { flushQueuedSends() } }
    }
    /// 正在对知识图谱做实体消歧（向量语义相似度合并同指实体）。
    @Published var isResolvingEntities = false {
        didSet { if !isResolvingEntities { flushQueuedSends() } }
    }
    /// 知识图谱是否已构建（决定 4 种图谱检索策略是否可选）。
    @Published var knowledgeGraphBuilt = false
    @Published var knowledgeGraphEntityCount = 0
    @Published var knowledgeGraphRelationCount = 0
    /// 维护态期间用户发来的消息：旧实现直接 return 静默丢弃，改为排队，
    /// 等维护结束（或上一轮生成结束）后依次自动发送。
    private struct QueuedSend {
        let text: String
        let images: [Data]
    }
    private var queuedSends: [QueuedSend] = []
    /// ANN 索引摘要（设置窗口显示，来自 RAG status 快照）。
    @Published var annStatusText = "ANN 索引未构建"
    /// 二值 ANN 索引摘要（stage 1 自研 bann；未构建时显示回退提示）。
    @Published var binaryANNStatusText = "二值 ANN 索引未构建"
    /// 闲时后台章节摘要的进度（设置窗口显示）。
    @Published var bookSummaryStatusText = "章节摘要：空闲"
    /// 当前索引与配置不匹配，需要用户显式重建。
    @Published var vectorRebuildRequired = false
    @Published var vectorRebuildReason = ""
    @Published var isModelLoading = false
    /// 主模型加载完成后的 KV 预热阶段；期间禁止发送消息，避免和 prefill 争夺模型。
    @Published var isWarmingUp = false
    @Published var mainModelLoaded = false
    @Published var auxModelLoaded = false

    let cancelFlag = CancelFlag()
    /// 最近一次构建 Agent 时加载的 BASE system prompt
    /// （来自 `<workspace>/assets/system_prompt.txt`）。只在启动 / 应用配置时刷新，
    /// 会话进行中不重读，避免 system 前缀抖动导致 KV cache 全量重建；
    /// 编辑文件后重启（或重新应用配置）生效。
    private var baseSystemPrompt = BASE_SYSTEM_PROMPT
    private(set) var agent: Agent?
    private var streamingID: UUID?
    private var pendingToolIndices: [String: [Int]] = [:]
    private var runTask: Task<Void, Never>?
    private var syncTask: Task<Void, Never>?
    /// 空闲内存回收：无前台活动 3 分钟后释放可重建的 OCR 权重。
    private var idleUnloadTask: Task<Void, Never>?
    /// 闲时摘要计时：需要比 OCR 回收更长的空闲窗口，避免和前台抢 GPU。
    private var idleSummaryTask: Task<Void, Never>?
    /// 闲时章节摘要会话；只在该会话内使用辅助模型，结束后卸载临时模型。
    private var summarySessionTask: Task<Void, Never>?
    /// 当前会话的协调器。懒创建，避免启动时在同步路径上打开大向量库。
    private var summaryCoordinator: BookSummaryCoordinator?
    /// 会话收尾任务：同步/退出/配置切换等需要串行等待模型释放。
    private var summaryStopTask: Task<Void, Never>?
    /// 每次开始/停止会话递增，旧会话的迟到回调不能覆盖新状态。
    private var summarySessionToken = 0
    /// 每次同步/取消同步递增，旧同步的收尾不能清掉新同步的状态。
    private var syncGeneration = 0
    private var runGeneration = 0
    private var contentBuffer = ""
    private var reasoningBuffer = ""
    private var summaryFilter = SummaryStreamFilter()
    private var flushScheduled = false
    /// 本轮是否已把第一个 chunk 刷到 UI。首字节延迟是用户最敏感的指标，
    /// 不该为了 50ms 批处理再等一轮。
    private var didFirstFlush = false
    private var ragStatusRefreshing = false
    private var configGeneration = 0
    /// `applyConfig { sync }` 路径：记住应用配置前已加载的模型，同步结束后再恢复。
    private var pendingSyncRestore: (main: Bool, aux: Bool)?
    // MARK: RAG MCP 服务（供其它大模型应用对接，共享同一个知识库引擎）
    @Published var ragMCPRunning = false
    private var ragMCPServer: RAGMCPServer?

    /// 启动 RAG MCP 服务（MCP 2025-06-18 Streamable HTTP）。
    /// 与 App 内 RAG 共用同一个 `RAGEngine`：入库/检索/删除就是同一套既有流程。
    @MainActor
    func startRAGMCPService() {
        guard let engine = agent?.ragEngine else {
            appendLog("[rag-mcp] Agent 未就绪，无法启动 MCP 服务")
            return
        }
        stopRAGMCPService(persistDisabled: false)
        let options = RAGMCPServerOptions(
            listenLAN: config.ragMCPListenLan,
            port: config.ragMCPPort)
        let server = RAGMCPServer(engine: engine, options: options) { [weak self] line in
            Task { @MainActor [weak self] in self?.appendLog(line) }
        }
        do {
            try server.start()
            ragMCPServer = server
            ragMCPRunning = true
            config.ragMCPEnabled = true
            try? config.save(Self.configPath)
            appendLog("[rag-mcp] MCP 服务已启动："
                + "http://\(config.ragMCPListenLan ? "0.0.0.0" : "127.0.0.1"):\(config.ragMCPPort)/mcp")
        } catch {
            appendLog("[rag-mcp] 启动失败：\(error.localizedDescription)")
        }
    }

    /// 停止 RAG MCP 服务。`persistDisabled=false` 用于配置重建等场景：
    /// 只停实例，不把用户「启用」的开关落盘成关闭。
    @MainActor
    func stopRAGMCPService(persistDisabled: Bool = true) {
        guard ragMCPServer != nil || ragMCPRunning else {
            if persistDisabled, config.ragMCPEnabled {
                config.ragMCPEnabled = false
                try? config.save(Self.configPath)
            }
            return
        }
        ragMCPServer?.stop()
        ragMCPServer = nil
        ragMCPRunning = false
        if persistDisabled {
            config.ragMCPEnabled = false
            try? config.save(Self.configPath)
        }
    }

    /// 只更新监听配置（不重建 Agent）：成功返回 true，端口非法返回 false。
    @MainActor
    @discardableResult
    func updateRAGMCPSettings(listenLAN: Bool, port: Int) -> Bool {
        guard (1...65535).contains(port) else { return false }
        config.ragMCPListenLan = listenLAN
        config.ragMCPPort = port
        do {
            try config.save(Self.configPath)
        } catch {
            appendLog("[rag-mcp] 配置保存失败: \(error.localizedDescription)")
        }
        return true
    }

    static var configPath: String {
        (NSHomeDirectory() as NSString).appendingPathComponent(".base-agent/config.json")
    }

    init() {
        config = AgentConfig.load(Self.configPath)
        config.ensureWorkspace()
        // `AgentLog` 只写 stderr + sink；不接线的话 `[mlx-kv] rebuild/prefix-reuse`、
        // `[mlx] prompt=... tps` 这类关键诊断在 UI 日志里永远看不到。
        // 必须在 config 初始化之后：闭包捕获 self 前所有存储属性都要就绪。
        AgentLog.sink = { [weak self] line in
            Task { @MainActor [weak self] in self?.appendLog(line) }
        }
        buildAgent()
        checkDefaultModels()
        appendLog("base-agent 启动，workspace=\(config.workspace)")
        // 上次退出时未完成的计划不跨启动恢复：直接丢弃，避免开盘即自动续跑。
        if agent?.discardPersistedPlan() == true {
            appendLog("[plan] 已清除上次未完成的计划")
        }
        // 启动时是否预加载主模型由 auto_preload_main 控制；辅助模型需在设置中
        // 显式启用后才加载。模型缺失时只记录可诊断错误，不阻塞启动。
        let shouldPreloadMain = config.llmProvider == "local" && config.autoPreloadMain
        if !shouldPreloadMain, config.llmProvider == "local" {
            appendLog("[model] 已关闭启动预加载，主模型需在配置窗口手动加载")
        }
        if !config.auxEnabled {
            appendLog("[model] 辅助模型未启用，启动时不加载")
        }
        if missingDefaultModels.isEmpty {
            Task {
                await preloadModels(main: shouldPreloadMain,
                                    aux: config.auxEnabled)
            }
            // Embedding / Reranker 与主模型是否预加载无关：检索首问同样需要它们。
            Task {
                await preloadRAGModels()
            }
        } else {
            // 默认模型缺失：预加载必然失败，等下载补齐后由用户重新触发加载。
            appendLog("[models] 默认模型缺失，已跳过预加载："
                + missingDefaultModels.map(\.repo).joined(separator: ", "))
        }
        // 即使无需预加载模型（API 模式/未启用辅助模型），也要让闲时摘要
        // 在启动后按空闲计时自动开始。
        scheduleIdleUnload()
        // 按配置恢复 RAG MCP 服务（默认关闭；上次运行中才会自动重启）。
        if config.ragMCPEnabled {
            startRAGMCPService()
        }
    }

    // MARK: 默认模型下载（hf-mirror）

    /// 当前配置需要但本地尚未就绪的默认模型（arlenecc/<模型名>）。
    @Published var missingDefaultModels: [RequiredModel] = []
    @Published var isDownloadingModels = false
    /// 0...1，按「已就绪字节 / 需下载总字节」聚合全部缺失模型。
    @Published var modelDownloadProgress: Double = 0
    @Published var modelDownloadStatusText = ""
    private var modelDownloadTask: Task<Void, Never>?
    private var downloadPromptShown = false
    private let hfDownloader = HFModelDownloader()

    /// 检查当前配置引用的本地模型是否就绪（权重 + MLX/ONNX 必需的 tokenizer）。
    func checkDefaultModels() {
        missingDefaultModels = RequiredModelCatalog.missingModels(for: config)
    }

    /// 首启提醒：模型缺失时弹窗让用户选择「自动下载补全」或「自行下载指定」。
    /// 选择自行下载指定后写入配置持久化，不再自动下载；未就绪前对话与同步
    /// 仍不可用，直到用户在配置窗口指定好可用模型。
    func promptModelDownloadIfNeeded() {
        guard !missingDefaultModels.isEmpty, !isDownloadingModels,
              config.autoDownloadMissingModels,
              !downloadPromptShown else { return }
        downloadPromptShown = true
        let names = missingDefaultModels
            .map { "\($0.kind)：\($0.repo)" }
            .joined(separator: "\n")
        let alert = NSAlert()
        alert.messageText = "默认模型缺失，需要联网下载或自行指定"
        alert.informativeText = "以下默认模型尚未就绪：\n\(names)\n\n"
            + "· 自动下载补全：从 \(HFModelDownloader.defaultMirrorBase) 下载运行"
            + "所需文件（支持断点续传，中断后可继续），完成后才能正常使用软件；\n"
            + "· 自行下载指定：不再自动下载，由你自行获取模型并在配置窗口指定"
            + "各阶段使用的模型，配置就绪后自动解除限制。"
        alert.addButton(withTitle: "自动下载补全")
        alert.addButton(withTitle: "自行下载指定")
        alert.addButton(withTitle: "稍后再说")
        NSApp.activate(ignoringOtherApps: true)
        switch alert.runModal() {
        case .alertFirstButtonReturn:
            startModelDownload()
        case .alertSecondButtonReturn:
            config.autoDownloadMissingModels = false
            do {
                try config.save(Self.configPath)
                appendLog("[models] 已选择自行下载指定模型（auto_download_missing_models=false），"
                    + "跳过自动下载")
            } catch {
                appendLog("[models] 配置保存失败：\(error.localizedDescription)")
            }
        default:
            // 稍后再说：本次启动不再提示，状态栏保留手动下载入口。
            break
        }
    }

    /// 开始补齐缺失的默认模型：REST API 列文件 → URLSession Range 续传。
    func startModelDownload() {
        guard !isDownloadingModels else { return }
        checkDefaultModels()
        guard !missingDefaultModels.isEmpty else { return }
        isDownloadingModels = true
        modelDownloadProgress = 0
        modelDownloadStatusText = "正在获取模型文件列表…"
        let missing = missingDefaultModels
        modelDownloadTask = Task { [weak self] in
            do {
                try await self?.runModelDownload(missing: missing)
            } catch is CancellationError {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.isDownloadingModels = false
                    self.statusText = "模型下载已停止"
                    self.modelDownloadStatusText = "模型下载已停止（已下载部分保留，可续传）"
                    self.appendLog("[models] 模型下载已停止")
                }
            } catch {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.isDownloadingModels = false
                    self.statusText = "模型下载失败"
                    self.modelDownloadStatusText = "模型下载失败：\(error.localizedDescription)"
                    self.appendLog("[models] 模型下载失败：\(error.localizedDescription)")
                    self.showInfoAlert(title: "模型下载失败",
                                       message: error.localizedDescription
                                       + "\n\n已下载的部分会保留，可再次点击下载从断点继续。")
                }
            }
        }
    }

    func cancelModelDownload() {
        modelDownloadTask?.cancel()
        hfDownloader.cancelCurrent()
    }

    @MainActor
    private func runModelDownload(missing: [RequiredModel]) async throws {
        // 先列全部缺失模型的文件清单，统计总字节，得到聚合进度基数。
        var plan: [(model: RequiredModel, files: [HFModelDownloader.FileEntry])] = []
        var totalBytes: Int64 = 0
        for model in missing {
            try Task.checkCancellation()
            let listed = try await hfDownloader.listNeededFiles(repo: model.repo)
            let files = RequiredModelCatalog.scopedFiles(for: model, in: listed)
            let pending = files.filter { file in
                HFModelDownloader.fileSize(at: model.directory
                    .appendingPathComponent(file.path)) != file.size
            }
            plan.append((model, pending))
            totalBytes += pending.reduce(0) { $0 + $1.size }
        }
        guard totalBytes > 0 else {
            checkDefaultModels()
            // 清单里没有任何待下载文件却仍判缺失：远端仓库本身缺少运行必需
            // 的文件（通常是没有 tokenizer.json），必须报错，不能提示已就绪。
            if let still = missingDefaultModels.first {
                throw HFModelDownloader.HFError.badResponse(
                    "\(still.repo) 缺少运行必需的文件（权重或 tokenizer.json），"
                        + "镜像仓库可能不完整，请在配置窗口指定其它已就绪的模型")
            }
            isDownloadingModels = false
            statusText = "默认模型已就绪"
            return
        }

        var completedBytes: Int64 = 0
        for entry in plan {
            for file in entry.files {
                try Task.checkCancellation()
                let base = completedBytes
                let filePath = file.path
                appendLog("[models] 下载 \(entry.model.repo)/\(filePath)")
                try await hfDownloader.download(
                    repo: entry.model.repo, file: file, into: entry.model.directory
                ) { [weak self] received, total in
                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        let done = Double(base + received)
                        self.modelDownloadProgress = min(1, done / Double(totalBytes))
                        self.modelDownloadStatusText = String(
                            format: "%@ · %@ %.0f%%",
                            entry.model.repo, filePath,
                            min(1, done / Double(totalBytes)) * 100)
                    }
                }
                completedBytes += file.size
                modelDownloadProgress = Double(completedBytes) / Double(totalBytes)
            }
            // 逐模型校验：只有权重下完、tokenizer 还没下（或远端缺文件）时
            // 不能宣布就绪，否则启动加载会报「缺少 tokenizer」且不再触发补全。
            guard RequiredModelCatalog.isComplete(entry.model) else {
                throw HFModelDownloader.HFError.badResponse(
                    "\(entry.model.repo) 下载完成后仍缺少权重或 tokenizer.json，"
                        + "请再次点击下载（支持断点续传）或手动指定可用模型")
            }
            appendLog("[models] 已就绪：\(entry.model.repo)")
        }

        checkDefaultModels()
        if let still = missingDefaultModels.first {
            isDownloadingModels = false
            statusText = "模型下载未完成"
            modelDownloadStatusText = "\(still.repo) 仍未就绪"
            throw HFModelDownloader.HFError.badResponse(
                "\(still.repo) 仍未就绪（缺少权重或 tokenizer.json），请在配置窗口检查模型目录")
        }
        isDownloadingModels = false
        statusText = "默认模型已就绪"
        modelDownloadStatusText = ""
        appendLog("[models] 默认模型下载完成")
        showInfoAlert(title: "模型下载完成",
                      message: "默认模型已就绪，软件可以正常使用了。")
    }

    /// 默认模型未就绪时的统一拦截提示（区分自动下载与自行指定两种模式）。
    private func blockForMissingModels(_ operation: String) -> Bool {
        guard !missingDefaultModels.isEmpty else { return false }
        let models = missingDefaultModels.map(\.repo).joined(separator: "、")
        let hint = config.autoDownloadMissingModels
            ? "可点击状态栏「下载」从 hf-mirror.com 自动补全（支持断点续传），"
                + "或在配置窗口指定已就绪的模型。"
            : "你已选择自行下载指定模型：请自行获取后，在配置窗口为各阶段指定"
                + "已就绪的模型并应用，配置就绪后自动解除限制。"
        showInfoAlert(
            title: "默认模型未就绪",
            message: "\(operation)需要以下模型就绪：\(models)。\n\(hint)")
        return true
    }

    // MARK: Agent 生命周期

    /// 丢弃当前流式缓冲。配置切换/清空对话后，旧任务迟到的 flush 不应写进新气泡。
    private func clearStreamingState() {
        contentBuffer = ""
        reasoningBuffer = ""
        streamingID = nil
        flushScheduled = false
        didFirstFlush = false
        summaryFilter.reset()
    }

    private func buildAgent() {
        // 每次构建 Agent（启动 / 应用配置）都以当前 workspace 的可编辑文件为准：
        // 首启写出内置提示词；切换工作区时自动改用新目录下的文件。
        let promptFile = SystemPromptFile.resolve(workspace: config.workspace)
        baseSystemPrompt = promptFile.text
        switch promptFile.status {
        case .created:
            appendLog("[prompt] 首次启动已生成可编辑 system prompt："
                + "\(promptFile.path)（编辑后重启生效）")
        case .loaded:
            appendLog("[prompt] 已加载自定义 system prompt：\(promptFile.path)"
                + "（\(promptFile.text.count) 字符）")
        case .fallback:
            appendLog("[prompt] \(promptFile.note)；文件：\(promptFile.path)")
        }
        // 新 Agent 的模型服务尚未预加载；成功 preload 后再置为已加载。
        mainModelLoaded = false
        auxModelLoaded = false
        LocalModelFactory.shared.setCacheLimitMB(config.mlxCacheLimitMB)
        LocalModelFactory.shared.setMemoryLimitMB(config.mlxMemoryLimitMB)
        if config.effectiveMainContextTokens < config.maxContextTokens {
            appendLog("[memory] 主模型上下文按 KV 上限收紧到 "
                + "\(config.effectiveMainContextTokens) tokens")
        }
        if config.effectiveAuxContextTokens < config.auxMaxContextTokens {
            appendLog("[memory] 辅助模型上下文按 KV 上限收紧到 "
                + "\(config.effectiveAuxContextTokens) tokens")
        }
        let llm = LocalModelFactory.shared.makeLLM(config: config)
        let callbacks = UICallbacks(model: self, cancel: cancelFlag)
        let newAgent = Agent(llm: llm, config: config, callbacks: callbacks,
                             baseSystemPrompt: baseSystemPrompt)
        // 未启用时保持 auxLLM 为 nil：RAG 精简会直接跳过，后台任务回退主模型，
        // 也不会被同步前后的恢复逻辑误加载。
        newAgent.auxLLM = config.auxEnabled
            ? LocalModelFactory.shared.makeAuxLLM(config: config)
            : nil
        // 知识图谱抽取/关键词提取复用辅助模型实例。
        newAgent.ragEngine.auxLLM = newAgent.auxLLM
        agent = newAgent
        refreshRAGStatus()
        refreshKnowledgeGraphStatus()
        Task { [weak newAgent] in
            await newAgent?.connectMCP()
        }
    }

    /// 应用完整配置（知识库、Agent、模型等）。重建 Agent 会关闭旧模型实例；
    /// 这里记住此前的加载状态，应用配置后自动恢复。
    /// `afterApply` 用于「先应用知识库配置、再同步」这类链式操作。
    func applyConfig(_ newConfig: AgentConfig,
                     afterApply: (@MainActor () -> Void)? = nil) {
        // 先停掉在途对话，再关闭 Agent，避免旧 Task 继续使用已释放/已换配置的模型。
        let wasMainLoaded = mainModelLoaded
        let auxWasLoaded = auxModelLoaded
        // 新配置关闭辅助模型时，即使旧模型已加载也不恢复，避免误启动。
        let wasAuxLoaded = auxWasLoaded && newConfig.auxEnabled
        let oldTask = runTask
        runGeneration += 1
        configGeneration += 1
        let generation = configGeneration
        isModelLoading = false
        cancelFlag.set(true)
        oldTask?.cancel()
        Task { [weak self] in
            // 摘要会话可能正占用辅助模型与向量库；先等它收尾，再关闭旧 Agent。
            await self?.stopSummaryIdleSession()
            _ = await oldTask?.value
            await MainActor.run {
                guard let self else { return }
                // 连续修改配置时只应用最后一次，避免旧任务晚到把新 agent 覆盖掉。
                guard self.configGeneration == generation else { return }
                self.cancelIdleUnload()
                self.clearStreamingState()
                self.isWarmingUp = false
                self.config = newConfig
                do {
                    try self.config.save(Self.configPath)
                } catch {
                    self.appendLog("配置保存失败: \(error.localizedDescription)")
                }
                // 配置可能指向了用户自行准备好的模型：重检缺失清单并解除/
                // 维持使用限制；仍缺且允许自动下载时允许再次弹出选择提示。
                self.checkDefaultModels()
                self.downloadPromptShown = false
                // 取消旧知识库同步，避免旧 agent 的 ingest 在 close 时占着锁，
                // 导致连接迟迟不释放（close 会在同步结束后补关）。
                self.agent?.ragEngine.cancel()
                self.syncGeneration += 1
                self.syncTask?.cancel()
                self.syncTask = nil
                self.isStoppingSync = false
                // 旧 Agent 即将关闭：先停 MCP 服务（共享同一引擎），不改动开关。
                self.stopRAGMCPService(persistDisabled: false)
                self.agent?.close()
                LocalModelFactory.shared.clearCaches()
                self.buildAgent()
                // 新引擎就绪后按新配置恢复服务（监听范围/端口可能已变化）。
                if newConfig.ragMCPEnabled {
                    self.startRAGMCPService()
                }
                if auxWasLoaded && !newConfig.auxEnabled {
                    self.appendLog("[model] 辅助模型已关闭，不再加载")
                }
                self.plan = nil
                self.isRunning = false
                self.isSyncing = false
                self.pendingToolIndices.removeAll()
                self.toolEvents.removeAll()
                self.cancelFlag.set(false)
                self.appendLog("配置已应用："
                    + (self.config.llmProvider == "local"
                       ? self.config.localLLMModel : self.config.model))
                self.appendLog("知识库目录：\(self.config.resolvedKnowledgeBase)"
                    + (self.config.knowledgeBase.isEmpty ? "（跟随工作目录）" : ""))
                if let afterApply {
                    // 同步会先卸载模型、结束后再恢复；这里不要同时预加载，
                    // 否则同步期间模型又常驻内存，抵消资源释放。
                    self.pendingSyncRestore = (wasMainLoaded, wasAuxLoaded)
                    afterApply()
                } else {
                    Task { await self.preloadRAGModels() }
                    if wasMainLoaded || wasAuxLoaded {
                        Task {
                            await self.preloadModels(main: wasMainLoaded,
                                                     aux: wasAuxLoaded)
                        }
                    }
                }
                // 配置应用会先取消空闲计时；若本次不触发预加载/同步，也必须
                // 重新计时，否则闲时摘要要等到下一次前台活动才会恢复。
                self.scheduleIdleUnload()
            }
        }
    }

    // MARK: 单模型重载

    private enum ModelSlot { case main, aux }

    /// 只重载主模型，保留已加载的辅助模型与当前会话。
    func reloadMainModel(config newConfig: AgentConfig) {
        reloadModel(.main, config: newConfig)
    }

    /// 只重载辅助模型，保留已加载的主模型与当前会话。
    func reloadAuxModel(config newConfig: AgentConfig) {
        reloadModel(.aux, config: newConfig)
    }

    /// 「加载主/辅助模型」的统一入口：面板除了本槽位模型参数之外还改了别的
    /// （切片大小、语义切片、知识库目录、embedding / reranker 后端、OCR……）时，
    /// 不能只替换 LLM —— Agent 与 RAGEngine 都持有配置快照，必须整体重建，
    /// 否则会出现「面板设了长切片、运行时仍按短切片处理」这类不同步。
    private func reloadModel(_ slot: ModelSlot, config newConfig: AgentConfig) {
        guard !isSyncing, !isDeduplicating, !isBuildingANN else {
            isModelLoading = false
            appendLog("[model] 知识同步/索引构建进行中，暂不加载模型")
            return
        }
        let allowed = slot == .main
            ? AgentConfig.mainLLMOnlyKeys : AgentConfig.auxLLMOnlyKeys
        let drifted = newConfig.agentRebuildKeys(comparedTo: config, allowing: allowed)
        guard !drifted.isEmpty else {
            swapModel(slot, config: newConfig)
            return
        }
        let slotLabel = slot == .main ? "主模型" : "辅助模型"
        let sample = drifted.prefix(6).joined(separator: "、")
        appendLog("[model] 检测到 \(drifted.count) 项配置变化（\(sample)"
            + (drifted.count > 6 ? " 等" : "") + "），重建 Agent 后加载\(slotLabel)")
        let wasMain = mainModelLoaded
        let wasAux = auxModelLoaded
        applyConfig(newConfig) { [weak self] in
            guard let self else { return }
            // 这里不在同步流程内，清掉 applyConfig 留下的恢复标记，直接加载目标槽位。
            self.pendingSyncRestore = nil
            let main = wasMain || slot == .main
            let aux = (wasAux || slot == .aux) && newConfig.auxEnabled
            guard main || aux else { return }
            Task { await self.preloadModels(main: main, aux: aux) }
        }
    }

    /// 重建 Agent 会同时关闭主/辅模型；这里只替换目标槽位，另一个模型保持常驻。
    private func swapModel(_ slot: ModelSlot, config newConfig: AgentConfig) {
        guard !isSyncing, !isDeduplicating, !isBuildingANN else {
            isModelLoading = false
            appendLog("[model] 知识同步/索引构建进行中，暂不加载模型")
            return
        }
        let oldTask = runTask
        runGeneration += 1
        configGeneration += 1
        let generation = configGeneration
        cancelFlag.set(true)
        oldTask?.cancel()
        cancelIdleUnload()
        isRunning = false
        isModelLoading = true
        statusText = "模型加载中…"

        Task { [weak self] in
            // 替换模型前先让闲时摘要释放对辅助模型的占用。
            await self?.stopSummaryIdleSession()
            _ = await oldTask?.value
            guard let self, self.configGeneration == generation else { return }
            guard let agent = self.agent else {
                self.isModelLoading = false
                return
            }

            self.clearStreamingState()
            self.isWarmingUp = false
            self.config = newConfig
            do {
                try self.config.save(Self.configPath)
            } catch {
                self.appendLog("配置保存失败: \(error.localizedDescription)")
            }
            // 槽位切换同样可能指向用户自行准备好的模型：重检缺失清单。
            self.checkDefaultModels()
            self.downloadPromptShown = false
            // 温度等请求参数由 KernelLoop 读取 agent.config，必须同步。
            agent.config = newConfig
            self.pendingToolIndices.removeAll()
            self.cancelFlag.set(false)

            switch slot {
            case .main:
                // 先释放旧主模型，再按新参数创建；辅助模型不受影响。
                agent.switchLLM(UnloadedLLMService())
                LocalModelFactory.shared.unloadLLMCaches()
                let service = LocalModelFactory.shared.makeLLM(config: newConfig)
                do {
                    if let preloadable = service as? PreloadableLLMService {
                        try await preloadable.preload()
                    }
                    guard self.configGeneration == generation else { return }
                    agent.switchLLM(service)
                    self.mainModelLoaded = true
                    let label = newConfig.localLLMModel.isEmpty
                        ? "默认主模型" : newConfig.localLLMModel
                    self.appendLog("[model] 主模型已加载：\(label)")
                    self.statusText = "模型加载成功"
                    guard self.configGeneration == generation else { return }
                    if let warmable = service as? WarmupLLMService {
                        await self.warmupMainModel(warmable, generation: generation)
                    } else if !self.isRunning {
                        self.statusText = "就绪"
                    }
                } catch {
                    self.mainModelLoaded = false
                    self.appendLog("[model] 主模型加载失败：\(error.localizedDescription)")
                    if !self.isRunning { self.statusText = "模型加载失败" }
                }

            case .aux:
                // 配置关闭辅助模型时，卸载旧实例并直接返回，绝不创建新实例。
                guard newConfig.auxEnabled else {
                    agent.unloadAuxLLM()
                    LocalModelFactory.shared.unloadLLMCaches()
                    self.auxModelLoaded = false
                    self.appendLog("[model] 辅助模型未启用，跳过加载")
                    self.statusText = "辅助模型未启用"
                    break
                }
                // 先释放旧辅助模型，再按新参数创建；主模型不受影响。
                agent.unloadAuxLLM()
                LocalModelFactory.shared.unloadLLMCaches()
                let service = LocalModelFactory.shared.makeAuxLLM(config: newConfig)
                do {
                    if let preloadable = service as? PreloadableLLMService {
                        try await preloadable.preload()
                    }
                    guard self.configGeneration == generation else { return }
                    agent.auxLLM = service
                    self.auxModelLoaded = true
                    let label = newConfig.auxLocalLLMModel.isEmpty
                        ? "默认辅助模型" : newConfig.auxLocalLLMModel
                    self.appendLog("[model] 辅助模型已加载：\(label)")
                    self.statusText = "模型加载成功"
                    if !self.isRunning { self.statusText = "就绪" }
                } catch {
                    self.auxModelLoaded = false
                    self.appendLog("[model] 辅助模型加载失败：\(error.localizedDescription)")
                    if !self.isRunning { self.statusText = "模型加载失败" }
                }
            }

            if self.configGeneration == generation {
                self.isModelLoading = false
                // 单槽位重载前已取消空闲计时；重载结束后恢复闲时摘要调度。
                if !self.isRunning { self.scheduleIdleUnload() }
            }
        }
    }

    // MARK: 模型加载 / 卸载

    /// 主动把主/辅模型权重加载到内存。API 模式的主模型不需要预加载。
    func preloadModels(main: Bool, aux: Bool) async {
        // 未启用时忽略所有 aux 预加载请求，避免同步恢复/旧调用误启动。
        let shouldLoadAux = aux && config.auxEnabled
        guard main || shouldLoadAux else { return }
        let generation = configGeneration
        isModelLoading = true
        statusText = "模型加载中…"
        var hadError = false
        var loadedAny = false
        var warmable: WarmupLLMService?
        defer {
            // 配置已切换时，旧预加载无权清掉新一次加载的状态。
            if configGeneration == generation {
                isModelLoading = false
                isWarmingUp = false
            }
        }

        if main, config.llmProvider.lowercased() == "local",
           let service = agent?.llm as? PreloadableLLMService {
            do {
                try await service.preload()
                guard configGeneration == generation else { return }
                mainModelLoaded = true
                loadedAny = true
                warmable = service as? WarmupLLMService
                let label = config.localLLMModel.isEmpty
                    ? "默认主模型" : config.localLLMModel
                appendLog("[model] 主模型已加载：\(label)")
            } catch {
                guard configGeneration == generation else { return }
                hadError = true
                mainModelLoaded = false
                appendLog("[model] 主模型加载失败：\(error.localizedDescription)")
            }
        }

        if shouldLoadAux, let service = agent?.auxLLM as? PreloadableLLMService {
            do {
                try await service.preload()
                guard configGeneration == generation else { return }
                auxModelLoaded = true
                loadedAny = true
                let label = config.auxLocalLLMModel.isEmpty
                    ? "默认辅助模型" : config.auxLocalLLMModel
                appendLog("[model] 辅助模型已加载：\(label)")
            } catch {
                guard configGeneration == generation else { return }
                hadError = true
                auxModelLoaded = false
                appendLog("[model] 辅助模型加载失败：\(error.localizedDescription)")
            }
        }

        guard configGeneration == generation else { return }
        if loadedAny {
            statusText = "模型加载成功"
        }
        guard configGeneration == generation else { return }
        // 辅助模型失败不应阻止主模型预热。
        if let warmable {
            await warmupMainModel(warmable, generation: generation)
        } else if !isRunning {
            statusText = hadError ? "模型加载失败" : "就绪"
        }
        if !isRunning { scheduleIdleUnload() }
    }

    /// 启动/配置应用后把本地 Embedding / Reranker 权重拉起来并常驻。
    ///
    /// 首次检索现场加载 ONNX session 实测 10~15s；失败只记录日志，不阻塞启动，
    /// 下次检索仍会按需加载并重试。
    func preloadRAGModels() async {
        guard let agent else { return }
        let generation = configGeneration
        let outcome = await agent.ragEngine.preloadAuxiliaryModels()
        guard configGeneration == generation else { return }
        if !outcome.loaded.isEmpty {
            appendLog("[model] 已预加载并常驻：\(outcome.loaded.joined(separator: "、"))")
        }
        for failure in outcome.failures {
            appendLog("[model] 预加载失败：\(failure)")
        }
    }

    /// 用稳定的 system prompt + 工具清单预热主模型 KV cache，并更新状态栏。
    private func warmupMainModel(_ service: WarmupLLMService,
                                 generation: Int? = nil) async {
        guard let agent else { return }
        if let generation, configGeneration != generation { return }
        isWarmingUp = true
        defer {
            if generation == nil || configGeneration == generation {
                isWarmingUp = false
            }
        }
        statusText = "模型预热中…"
        // MCP 工具清单会进入 system 前缀，预热前先等连接完成。
        await agent.connectMCP()
        guard generation == nil || configGeneration == generation else { return }
        do {
            try await service.warmup(systemPrompt: agent.systemPrompt(),
                                     tools: agent.dispatcher.schemas())
            guard generation == nil || configGeneration == generation else { return }
            appendLog("[model] 主模型预热结束")
            // 预热期间已排队的用户请求此时开始生成，状态切回「思考中…」；
            // 否则直接就绪。旧实现固定睡 1.2s 只为让中间状态可见。
            statusText = isRunning ? "思考中…" : "就绪"
        } catch {
            guard generation == nil || configGeneration == generation else { return }
            appendLog("[model] 主模型预热失败：\(error.localizedDescription)")
            if !isRunning { statusText = "预热失败" }
        }
    }

    /// 卸载主模型，保留会话与 RAG；再次对话前需要在配置窗口重新加载。
    func unloadMainModel() {
        // 生成/加载/预热/维护期间切换 LLM 会让在途请求使用已关闭的模型。
        guard !isRunning, !isModelLoading, !isWarmingUp, !isDeduplicating,
              !isSyncing, !isBuildingANN else { return }
        // 取消待触发的空闲计时，避免刚卸载完又被后台摘要重新拉起模型。
        cancelIdleUnload()
        agent?.switchLLM(UnloadedLLMService())
        LocalModelFactory.shared.unloadLLMCaches()
        mainModelLoaded = false
        appendLog("[model] 主模型已卸载")
        statusText = "主模型未加载"
        scheduleIdleUnload()
    }

    /// 卸载辅助模型；主模型不受影响。
    func unloadAuxModel() {
        guard !isRunning, !isModelLoading, !isWarmingUp, !isDeduplicating,
              !isSyncing, !isBuildingANN else { return }
        // 取消待触发的空闲计时，并等当前摘要会话（可能正借用辅助模型）
        // 彻底结束后再 close，避免与在途生成并发释放同一份权重。
        cancelIdleUnload()
        Task { [weak self] in
            guard let self else { return }
            await self.stopSummaryIdleSession()
            // 等待期间用户可能已经开始对话/同步/维护；此时不能再卸载。
            guard !self.isRunning, !self.isModelLoading, !self.isWarmingUp,
                  !self.isDeduplicating, !self.isSyncing,
                  !self.isBuildingANN else {
                self.appendLog("[model] 前台任务已开始，暂不卸载辅助模型")
                self.scheduleIdleUnload()
                return
            }
            self.agent?.unloadAuxLLM()
            LocalModelFactory.shared.unloadLLMCaches()
            self.auxModelLoaded = false
            self.appendLog("[model] 辅助模型已卸载")
            self.scheduleIdleUnload()
        }
    }

    func resetConversation() {
        cancelIdleUnload()
        let oldTask = runTask
        runGeneration += 1
        cancelFlag.set(true)
        agent?.ragEngine.cancel()
        oldTask?.cancel()
        Task { [weak self] in
            _ = await oldTask?.value
            await MainActor.run {
                guard let self else { return }
                self.chatItems.removeAll()
                self.reasoning = ""
                self.toolEvents.removeAll()
                self.pendingToolIndices.removeAll()
                self.plan = nil
                self.clearStreamingState()
                self.agent?.reset()
                self.cancelFlag.set(false)
                self.isRunning = false
                self.totalTokens = 0
                self.tokenSpeed = 0
                self.statusText = "就绪"
                self.refreshRAGStatus()
                self.appendLog("对话已清空")
                self.scheduleIdleUnload()
            }
        }
    }

    func cancel() {
        guard isRunning else { return }
        cancelFlag.set(true)
        runTask?.cancel()
        statusText = "正在终止…"
        appendLog("用户请求终止")
        agent?.ragEngine.cancel()
    }

    /// 清空任务计划：运行中先请求终止，结束后再次清掉迟到的计划更新。
    func clearPlan() {
        cancelIdleUnload()
        let wasRunning = isRunning
        if wasRunning { cancel() }
        let task = runTask
        agent?.discardPersistedPlan()
        plan = nil
        appendLog(wasRunning ? "已请求终止，并清空任务计划" : "任务计划已清空")
        guard let task else {
            // 没有在途任务：本次操作结束即可恢复空闲计时。
            scheduleIdleUnload()
            return
        }
        Task { [weak self] in
            _ = await task.value
            await MainActor.run {
                guard let self else { return }
                self.agent?.discardPersistedPlan()
                self.plan = nil
                self.scheduleIdleUnload()
            }
        }
    }

    // MARK: 发送

    /// 维护态解除 / 上一轮结束后，把排队消息依次发出。每次只发一条，
    /// isRunning 的 didSet 会在本轮结束时触发下一次 flush。
    private func flushQueuedSends() {
        guard !queuedSends.isEmpty, !isRunning,
              !isSyncing, !isDeduplicating, !isBuildingANN else { return }
        let next = queuedSends.removeFirst()
        appendLog("[ui] 维护任务已结束，自动发送排队消息")
        send(next.text, images: next.images)
    }

    func send(_ input: String, images: [Data] = []) {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        // 模型加载/预热期间不再丢弃用户输入：寒暄/能力问询在内核本地直答
        // （零 GPU，启动即有响应）；正式任务排队，MLXLLMService.chatStream
        // 等预热结束后直接复用刚建好的 system+tools 前缀。
        guard !text.isEmpty || !images.isEmpty, !isRunning, let agent else { return }
        // 知识同步 / 向量去重 / ANN 构建期间不再静默丢弃输入。
        if isSyncing || isDeduplicating || isBuildingANN {
            if queuedSends.count >= 20 { queuedSends.removeFirst() }
            queuedSends.append(QueuedSend(text: text, images: images))
            statusText = "维护任务进行中，消息已排队（\(queuedSends.count)）…"
            appendLog("[ui] 知识同步/向量去重/索引构建进行中，消息已排队，结束后自动发送")
            return
        }
        if blockForMissingModels("发送消息") { return }
        cancelIdleUnload()
        runGeneration += 1
        let generation = runGeneration
        // 长会话只保留最近 1000 条消息，避免 UI 数组与布局无限增长。
        if chatItems.count > 1_000 { chatItems.removeFirst(chatItems.count - 1_000) }
        clearStreamingState()
        chatItems.append(ChatItem(role: .user, content: text, images: images.isEmpty ? nil : images))
        let item = ChatItem(role: .assistant, content: "")
        chatItems.append(item)
        streamingID = item.id
        reasoning = ""
        toolEvents.removeAll()
        pendingToolIndices.removeAll()
        isRunning = true
        // 预热尚未结束时正式任务会排队等前缀就绪（约 20s，只有启动这一次）。
        // 明确显示排队状态，避免「思考中…」却长时间无首个 token 像卡死。
        if isWarmingUp {
            statusText = "模型预热中，问题已排队…"
            appendLog("[model] 预热期间收到请求，前缀就绪后立即开始处理")
        } else {
            statusText = "思考中…"
        }
        totalTokens = 0
        tokenSpeed = 0
        cancelFlag.set(false)

        runTask = Task { [weak self, weak agent] in
            guard let self, let agent else { return }
            // 闲时摘要可能正借用同一个辅助模型；给它 1.5s 让出 GPU，
            // 超时也不再阻塞用户这一轮（卸载继续在后台完成）。
            await self.stopSummaryIdleSession(timeout: 1.5)
            // 等待期间用户可能已点「终止」，或配置切换已作废旧一轮。
            let answer: String
            if Task.isCancelled || self.runGeneration != generation {
                answer = ""
            } else {
                answer = await agent.run(text, images: images)
            }
            await MainActor.run {
                guard self.runGeneration == generation else { return }
                self.finishSummaryFilter()
                self.flushBuffers()
                self.finalizeStreaming(with: answer)
                self.isRunning = false
                self.statusText = "就绪"
                self.cancelFlag.set(false)
                self.refreshRAGStatus()
                self.scheduleIdleUnload()
            }
        }
    }

    /// 退出前的有序收尾：停止在途生成/同步，等待 GPU 工作结束并释放模型。
    /// 不做这步直接退出，MLX 的 asyncEval 会与进程退出并发而崩溃。
    func prepareForTermination() async {
        cancelIdleUnload()
        // 等待闲时摘要会话退出并卸载临时辅助模型，避免和退出流程并发。
        await stopSummaryIdleSession()
        cancelFlag.set(true)
        runGeneration += 1
        let running = runTask
        runTask = nil
        running?.cancel()
        agent?.ragEngine.cancel()
        syncTask?.cancel()
        _ = await running?.value
        if let agent {
            await agent.shutdownForTermination()
        }
        await LocalModelFactory.shared.unloadAllForTermination()
        LocalModelFactory.shared.unloadLLMCaches()
        LocalModelFactory.shared.synchronizeGPU()
    }

    // MARK: 空闲内存回收

    /// 延迟 3 分钟无前台活动后释放可重建的 OCR 权重。
    /// Embedding / Reranker 启动预加载后常驻，模型服务对象保留。
    private func scheduleIdleUnload() {
        idleUnloadTask?.cancel()
        idleUnloadTask = Task { [weak self] in
            guard let self else { return }
            var wait = IdleWorkPolicy.ocrUnloadDelayNanoseconds
            while !Task.isCancelled {
                do {
                    try await Task.sleep(nanoseconds: wait)
                } catch {
                    return
                }
                guard !Task.isCancelled else { return }
                guard self.canStartIdleWork else {
                    // 前台对话/加载/同步或 Agent 后台任务尚未结束：一分钟后
                    // 再检查，避免一次短暂忙碌让本次空闲窗口被永久跳过。
                    wait = 60_000_000_000
                    continue
                }
                await LocalModelFactory.shared.unloadIdleOCRWeights()
                return
            }
        }
        scheduleIdleSummary()
    }

    /// 空闲摘要使用更长的窗口（10 分钟）：会话一旦启动会持续占用辅助模型，
    /// 用户只是短暂离开时不能和下一次前台请求抢 GPU。
    private func scheduleIdleSummary(
        delay: UInt64 = IdleWorkPolicy.summaryDelayNanoseconds
    ) {
        idleSummaryTask?.cancel()
        idleSummaryTask = Task { [weak self] in
            guard let self else { return }
            var wait = delay
            while !Task.isCancelled {
                do {
                    try await Task.sleep(nanoseconds: wait)
                } catch {
                    return
                }
                guard !Task.isCancelled else { return }
                guard self.canStartIdleWork else {
                    wait = 60_000_000_000
                    continue
                }
                // 摘要会话在独立 Task 里继续运行（内部按 15/5 分钟节流逐章处理）。
                self.startSummaryIdleSession()
                return
            }
        }
    }

    /// 闲时摘要/回收的启动条件；忙碌时由上面的计时任务每分钟重试。
    private var canStartIdleWork: Bool {
        !isRunning && !isModelLoading && !isWarmingUp && !isSyncing
            && !isDeduplicating && !isBuildingANN
            && summaryStopTask == nil
            && (agent?.isBackgroundIdle() ?? true)
    }

    private func cancelIdleUnload() {
        idleUnloadTask?.cancel()
        idleUnloadTask = nil
        idleSummaryTask?.cancel()
        idleSummaryTask = nil
        requestSummaryStop()
    }

    // MARK: 闲时章节摘要

    /// 10 分钟空闲后开启一个摘要会话；一次只处理一个章节/一次全书摘要。
    /// 临时辅助模型由会话创建和卸载，常驻辅助模型只借用、不关闭。
    private func startSummaryIdleSession() {
        guard summarySessionTask == nil, summaryCoordinator == nil,
              summaryStopTask == nil, let agent else { return }
        summarySessionToken += 1
        let token = summarySessionToken
        // 常驻条件：辅助引擎已启用且当前可用。API 客户端无需预加载，实例存在
        // 即可复用；本地模型必须已成功预加载，否则交给临时会话加载并及时释放。
        let canUseResident: Bool
        if config.auxEnabled, let aux = agent.auxLLM {
            canUseResident = auxModelLoaded || !(aux is PreloadableLLMService)
        } else {
            canUseResident = false
        }
        let mode: BookSummaryMode = canUseResident ? .resident : .temporary
        bookSummaryStatusText = "章节摘要：准备中…"
        summarySessionTask = Task.detached(priority: .utility) { [weak self, weak agent] in
            guard let agent, !Task.isCancelled else { return }
            // 日志闭包会在会话期间多次逃逸执行；先取不可变弱引用，
            // 避免嵌套 Task 直接捕获外层并发闭包里的 var self。
            let model = self
            let coordinator: BookSummaryCoordinator
            do {
                coordinator = try agent.ragEngine.makeBookSummaryCoordinator(
                    residentAuxProvider: { agent.auxLLM },
                    transientAuxFactory: { config in
                        LocalModelFactory.shared.makeTransientAuxLLM(config: config)
                    },
                    log: { line in
                        Task { @MainActor in model?.appendLog(line) }
                    })
            } catch {
                let message = error.localizedDescription
                await MainActor.run { [weak self] in
                    guard let self, self.summarySessionToken == token else { return }
                    self.summarySessionTask = nil
                    self.bookSummaryStatusText = "章节摘要：不可用"
                    self.appendLog("章节摘要：初始化失败 — \(message)")
                }
                return
            }
            // 加载临时辅助模型前先检查候选：既避免为必然失败的 tick 空转一次
            // 大模型加载，也把 Markdown 已失联的候选轮转到队尾，防止它们长期
            // 占住队首导致后面的书永远排不到。
            switch coordinator.checkIdleWork() {
            case .noWork:
                await MainActor.run { [weak self] in
                    guard let self, self.summarySessionToken == token else { return }
                    self.summarySessionTask = nil
                    self.bookSummaryStatusText = "章节摘要：空闲"
                }
                return
            case .rotated:
                // 本轮只清理了失联文档；一分钟后再检查一次，避免为了必然
                // 失败的 tick 反复加载辅助模型。
                await MainActor.run { [weak self] in
                    guard let self, self.summarySessionToken == token else { return }
                    self.summarySessionTask = nil
                    self.bookSummaryStatusText = "章节摘要：稍后继续"
                    self.scheduleIdleSummary(delay: 60_000_000_000)
                }
                return
            case .ready:
                break
            }
            await MainActor.run { [weak self] in
                guard let self, self.summarySessionToken == token else { return }
                self.summaryCoordinator = coordinator
                self.bookSummaryStatusText = "章节摘要：准备辅助模型…"
            }
            guard await coordinator.beginSession(mode: mode) else {
                await coordinator.endSession()
                await MainActor.run { [weak self] in
                    guard let self, self.summarySessionToken == token else { return }
                    self.summaryCoordinator = nil
                    self.summarySessionTask = nil
                    self.bookSummaryStatusText = "章节摘要：辅助模型不可用"
                }
                return
            }
            await MainActor.run { [weak self] in
                guard let self, self.summarySessionToken == token else { return }
                self.bookSummaryStatusText = coordinator.statusSnapshot()
            }
            var consecutiveFailures = 0
            session: while !Task.isCancelled {
                let outcome = await coordinator.runTick(mode: mode)
                await MainActor.run { [weak self] in
                    guard let self, self.summarySessionToken == token else { return }
                    self.bookSummaryStatusText = coordinator.statusSnapshot()
                    if case .failed(let docName, let message) = outcome {
                        self.appendLog("章节摘要：\(docName) 处理失败 — \(message)")
                    }
                }
                switch outcome {
                case .processedChapter, .processedBook:
                    consecutiveFailures = 0
                    continue session
                case .throttled(let waitSeconds):
                    // wait=0 表示另一个 tick 正在收尾，短暂让出后重试。
                    let seconds = waitSeconds > 0 ? waitSeconds : 5
                    do {
                        try await Task.sleep(
                            nanoseconds: UInt64(seconds * 1_000_000_000))
                    } catch {
                        break session
                    }
                case .failed:
                    // 连续失败达到上限后及时结束会话；可能是模型配置/存储等
                    // 系统性故障，继续逐本重试只会空转并占用资源。
                    consecutiveFailures += 1
                    if consecutiveFailures >= 2 {
                        await MainActor.run { [weak self] in
                            guard let self, self.summarySessionToken == token else { return }
                            self.appendLog("章节摘要：连续处理失败，结束本次空闲会话")
                        }
                        break session
                    }
                    do { try await Task.sleep(nanoseconds: 30_000_000_000) }
                    catch { break session }
                case .noWork:
                    await MainActor.run { [weak self] in
                        guard let self, self.summarySessionToken == token else { return }
                        self.appendLog("章节摘要：本次空闲会话没有待处理章节")
                    }
                    break session
                }
            }
            await coordinator.endSession()
            await MainActor.run { [weak self] in
                guard let self, self.summarySessionToken == token else { return }
                self.summaryCoordinator = nil
                self.summarySessionTask = nil
                self.bookSummaryStatusText = "章节摘要：空闲"
            }
        }
    }

    /// 请求停止摘要会话；不等待完成，供前台操作即时取消。
    private func requestSummaryStop() {
        guard summaryStopTask == nil,
              summarySessionTask != nil || summaryCoordinator != nil else { return }
        summaryStopTask = Task { @MainActor [weak self] in
            await self?.performSummaryStop()
        }
    }

    /// 等待摘要会话彻底结束（含临时模型卸载）；可重复调用。
    ///
    /// `timeout` 只给前台发送请求用：闲时摘要的 `cancelCurrent()` +
    /// `unloadAndWait()` 可能耗时数十秒，不能让用户每轮都等它。超时后
    /// 直接放行，卸载仍在后台完成；配置切换/卸载模型/退出等必须真正等完
    /// 的调用点继续用默认的无限等待。
    private func stopSummaryIdleSession(timeout: TimeInterval? = nil) async {
        requestSummaryStop()
        guard summaryStopTask != nil else { return }
        guard let timeout else {
            if let task = summaryStopTask { _ = await task.value }
            return
        }
        let deadline = Date().addingTimeInterval(timeout)
        while summaryStopTask != nil, Date() < deadline {
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
    }

    private func performSummaryStop() async {
        let session = summarySessionTask
        let coordinator = summaryCoordinator
        summarySessionTask = nil
        summarySessionToken += 1
        session?.cancel()
        // runTick 内部是独立 Task，外层取消不会自动传递；先取消当前章节。
        if let coordinator {
            await coordinator.cancelCurrent()
        }
        if let session {
            _ = await session.value
        }
        summaryCoordinator = nil
        summaryStopTask = nil
        bookSummaryStatusText = "章节摘要：空闲"
    }

    // MARK: 流式更新

    func appendContent(_ text: String) {
        guard streamingID != nil else { return }
        // `<summary>` 是内核协议标签：正文照常进对话，摘要转进思考区。
        let parts = summaryFilter.feed(text)
        if !parts.content.isEmpty { contentBuffer += parts.content }
        if !parts.summary.isEmpty { reasoningBuffer += parts.summary }
        if !parts.content.isEmpty || !parts.summary.isEmpty {
            scheduleFlush()
        }
        statusText = "生成中…"
    }

    func appendReasoning(_ text: String) {
        reasoningBuffer += text
        scheduleFlush()
    }

    /// 每 50ms 合并一次流式增量，避免逐 token 修改整个数组触发重绘。
    private func scheduleFlush() {
        guard !flushScheduled else { return }
        // 首个 chunk 立即上屏；批处理只对后续 token 有意义。
        if !didFirstFlush {
            didFirstFlush = true
            flushBuffers()
            return
        }
        flushScheduled = true
        let generation = runGeneration
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 50_000_000)
            guard let self else { return }
            // 配置切换/清空对话后，旧任务不能再把缓冲写进新气泡。
            guard self.runGeneration == generation else {
                self.flushScheduled = false
                return
            }
            self.flushBuffers()
        }
    }

    /// 流结束时取出过滤器里尚未闭合的残留内容。
    private func finishSummaryFilter() {
        let tail = summaryFilter.finish()
        if !tail.content.isEmpty { contentBuffer += tail.content }
        if !tail.summary.isEmpty { reasoningBuffer += tail.summary }
    }

    func flushBuffers() {
        flushScheduled = false
        if !contentBuffer.isEmpty, let id = streamingID,
           let index = chatItems.firstIndex(where: { $0.id == id }) {
            let existing = chatItems[index].content
            if existing.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                // 工具调用前的中间回复可能只剩被剥掉 <tool_call> 后的空行；
                // 气泡第一个可见字符之前不应保留这些空白，否则会出现巨大的空背景框。
                chatItems[index].content = String(contentBuffer.drop(while: \.isWhitespace))
            } else {
                chatItems[index].content += contentBuffer
            }
            contentBuffer = ""
        }
        if !reasoningBuffer.isEmpty {
            reasoning += reasoningBuffer
            reasoningBuffer = ""
        }
    }

    func appendSystem(_ text: String) {
        chatItems.append(ChatItem(role: .system, content: text))
    }

    func appendLog(_ text: String) {
        let line = LogLine(text: text)
        logs.append(line)
        if logs.count > 2_000 { logs.removeFirst(logs.count - 2_000) }
    }

    /// 手动清空日志面板（日志与工具调用记录都是内存态，进程真正退出
    /// 后本就清空；此按钮用于不重启时手动清理）。
    func clearLogs() {
        logs.removeAll()
        toolEvents.removeAll()
    }

    func beginTool(name: String, args: [String: JSONValue]) {
        // 上一次流式回复到此结束；若模型在 <summary> 内被截断，先落盘并复位，
        // 避免把下一段回复的正文一直吞进思考区。
        finishSummaryFilter()
        flushBuffers()
        var arguments = JSONValue.object(args).jsonString()
        if arguments.count > 500 { arguments = String(arguments.prefix(500)) + "…" }
        let event = ToolEvent(name: name, arguments: arguments, success: nil, output: "")
        toolEvents.append(event)
        pendingToolIndices[name, default: []].append(toolEvents.count - 1)
        statusText = "调用工具 \(name)…"
    }

    func finishTool(name: String, result: ToolResult) {
        if var indices = pendingToolIndices[name], let index = indices.first,
           index < toolEvents.count {
            toolEvents[index].success = result.success
            let body = result.success ? result.output : result.error
            toolEvents[index].output = String(body.prefix(600))
            indices.removeFirst()
            pendingToolIndices[name] = indices.isEmpty ? nil : indices
        }
    }

    func finishRun() {
        isRunning = false
        statusText = "就绪"
        cancelFlag.set(false)
    }

    private func finalizeStreaming(with answer: String) {
        // 兜底答案同样剥掉协议标签，避免摘要回落到对话区。
        let cleaned = stripSummaryTags(answer)
        guard let id = streamingID,
              let index = chatItems.firstIndex(where: { $0.id == id }) else {
            if !cleaned.isEmpty { chatItems.append(ChatItem(role: .assistant, content: cleaned)) }
            return
        }
        if chatItems[index].content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            chatItems[index].content = cleaned
        }
        streamingID = nil
    }

    // MARK: RAG

    /// 点击「向量去重」：先扫描并把重复数量反馈给用户，确认后再清理。
    func requestVectorDedupe() {
        guard let agent else {
            showInfoAlert(title: "向量去重", message: "RAG 尚未就绪，请先在知识同步配置中应用配置。")
            return
        }
        if isSyncing {
            showInfoAlert(title: "暂不能向量去重", message: "知识同步正在进行，请等同步结束后再试。")
            return
        }
        if isDeduplicating {
            showInfoAlert(title: "向量去重进行中", message: "正在排查或清理重复向量，请稍候。")
            return
        }
        if isBuildingANN {
            // 去重会 DELETE chunks/向量行，而 ANN 构建按 rowid 扫描同一批表；
            // 两者并发可能产出引用已删除切片的索引，必须串行。
            showInfoAlert(title: "暂不能向量去重", message: "ANN 索引维护正在进行，请稍后再试。")
            return
        }
        if isModelLoading || isWarmingUp {
            showInfoAlert(title: "暂不能向量去重", message: "模型正在加载或预热，请稍后再试。")
            return
        }
        if isRunning {
            showInfoAlert(title: "暂不能向量去重", message: "对话正在生成，请等本轮结束后再试。")
            return
        }
        cancelIdleUnload()
        isDeduplicating = true
        statusText = "正在扫描重复向量…"
        let target = ObjectIdentifier(agent)
        Task.detached(priority: .utility) { [weak self] in
            // 摘要会话与向量维护共用同一个 SQLite 库；先等它退出，
            // 避免维护期间仍有摘要写入或在途模型调用。
            await self?.stopSummaryIdleSession()
            let stillValid: Bool = await MainActor.run { [weak self] in
                guard let self, self.isDeduplicating,
                      let current = self.agent,
                      ObjectIdentifier(current) == target else { return false }
                return true
            }
            guard stillValid else {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.isDeduplicating = false
                    if !self.isRunning, !self.isSyncing, !self.isModelLoading,
                       !self.isWarmingUp, !self.isBuildingANN {
                        self.statusText = "就绪"
                    }
                    self.scheduleIdleUnload()
                }
                return
            }
            let result: Result<VectorDedupPreview, Error>
            do {
                result = .success(try agent.ragEngine.previewVectorDedupe())
            } catch {
                result = .failure(error)
            }
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.isDeduplicating = false
                guard let current = self.agent, ObjectIdentifier(current) == target else {
                    self.statusText = "就绪"
                    self.scheduleIdleUnload()
                    return
                }
                switch result {
                case .failure(let error):
                    self.appendLog("RAG: 向量去重扫描失败 — \(error.localizedDescription)")
                    self.statusText = "向量去重失败"
                    self.showInfoAlert(title: "向量去重失败", message: error.localizedDescription)
                    self.scheduleIdleUnload()
                case .success(let preview):
                    if preview.duplicateChunks == 0 {
                        self.appendLog("RAG: 向量库共 \(preview.totalChunks) 条切片，未发现重复向量")
                        self.statusText = "未发现重复向量"
                        self.showInfoAlert(
                            title: "向量去重",
                            message: "向量库共 \(preview.totalChunks) 条切片，未发现重复向量。")
                        self.scheduleIdleUnload()
                        return
                    }
                    if self.confirmVectorDedupe(preview) {
                        self.performVectorDedupe(agent: agent, target: target)
                    } else {
                        self.appendLog("[rag] 已取消向量去重")
                        self.statusText = "就绪"
                        self.scheduleIdleUnload()
                    }
                }
            }
        }
    }

    /// 用户确认后执行实际删除。
    private func performVectorDedupe(agent: Agent, target: ObjectIdentifier) {
        guard !isSyncing, !isDeduplicating, !isBuildingANN,
              !isModelLoading, !isRunning, !isWarmingUp else {
            showInfoAlert(title: "暂不能向量去重", message: "当前有任务正在进行，请稍后再试。")
            // 扫描阶段已取消空闲计时；拒绝执行后要恢复，否则摘要要等下次前台活动。
            scheduleIdleUnload()
            return
        }
        cancelIdleUnload()
        isDeduplicating = true
        statusText = "正在清理重复向量…"
        appendLog("[rag] 开始向量去重：清理重复切片，保留入库时间最早的一份")
        Task.detached(priority: .utility) { [weak self] in
            // 确认弹窗期间配置可能已切换；真正删除前再等摘要会话退出并校验
            // 目标 Agent 仍然有效，避免旧引擎继续操作已关闭/已换配置的库。
            await self?.stopSummaryIdleSession()
            let stillValid: Bool = await MainActor.run { [weak self] in
                guard let self, self.isDeduplicating,
                      let current = self.agent,
                      ObjectIdentifier(current) == target else { return false }
                return true
            }
            guard stillValid else {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.isDeduplicating = false
                    if !self.isRunning, !self.isSyncing, !self.isModelLoading,
                       !self.isWarmingUp, !self.isBuildingANN {
                        self.statusText = "就绪"
                    }
                    self.scheduleIdleUnload()
                }
                return
            }
            let result: Result<VectorDedupStats, Error>
            do {
                result = .success(try agent.ragEngine.deduplicateVectors())
            } catch {
                result = .failure(error)
            }
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.isDeduplicating = false
                if let current = self.agent, ObjectIdentifier(current) == target {
                    self.refreshRAGStatus()
                }
                switch result {
                case .success(let stats):
                    self.appendLog("RAG: 向量去重完成 — \(stats.summary)")
                    self.statusText = "向量去重完成"
                    self.showInfoAlert(title: "向量去重完成", message: stats.summary + "。")
                case .failure(let error):
                    self.appendLog("RAG: 向量去重失败 — \(error.localizedDescription)")
                    self.statusText = "向量去重失败"
                    self.showInfoAlert(title: "向量去重失败", message: error.localizedDescription)
                }
                self.scheduleIdleUnload()
            }
        }
    }

    private func confirmVectorDedupe(_ preview: VectorDedupPreview) -> Bool {
        let alert = NSAlert()
        alert.messageText = "确认清理重复向量"
        alert.informativeText = """
        向量库共 \(preview.totalChunks) 条切片，检测到 \(preview.duplicateChunks) 条重复向量，涉及 \(preview.affectedSources) 个来源。
        每个重复分组只保留入库时间最早的一份，删除后不可恢复。
        """
        alert.alertStyle = .warning
        // 默认按钮是「取消」，避免误触删除。
        alert.addButton(withTitle: "取消")
        alert.addButton(withTitle: "清理")
        NSApp.activate(ignoringOtherApps: true)
        return alert.runModal() == .alertSecondButtonReturn
    }

    // MARK: ANN 索引

    /// 点击「构建 ANN 索引」：先取一次索引状态，已有索引时确认重建，再启动构建。
    ///
    /// 构建期间 store 被构建线程独占，检索会按只读超时跳过（知识库暂时「不可检索」），
    /// 因此只允许在空闲时启动；已在跑的同步/去重/对话都要先结束。
    func requestANNIndexBuild() {
        guard let agent else {
            showInfoAlert(title: "向量索引", message: "RAG 尚未就绪，请先在知识同步配置中应用配置。")
            return
        }
        if isSyncing {
            showInfoAlert(title: "暂不能构建向量索引", message: "知识同步正在进行，请等同步结束后再试。")
            return
        }
        if isDeduplicating {
            showInfoAlert(title: "暂不能构建向量索引", message: "向量去重正在进行，请稍后再试。")
            return
        }
        if isBuildingANN {
            showInfoAlert(title: "向量索引构建中",
                          message: "正在构建向量索引。可点「停止构建」中断，已写完的块会保留，下次从断点续建。")
            return
        }
        if isModelLoading || isWarmingUp {
            showInfoAlert(title: "暂不能构建向量索引", message: "模型正在加载或预热，请稍后再试。")
            return
        }
        if isRunning {
            showInfoAlert(title: "暂不能构建向量索引", message: "对话正在生成，请等本轮结束后再试。")
            return
        }
        cancelIdleUnload()
        isBuildingANN = true
        statusText = "正在读取向量索引状态…"
        let target = ObjectIdentifier(agent)
        Task.detached(priority: .utility) { [weak self] in
            // 摘要会话会周期性借用辅助模型；先让它退出，避免构建期间还有
            // 后台模型调用，也避免配置切换后旧引擎与新 Agent 同时访问库。
            await self?.stopSummaryIdleSession()
            let stillValid: Bool = await MainActor.run { [weak self] in
                guard let self, self.isBuildingANN,
                      let current = self.agent,
                      ObjectIdentifier(current) == target else { return false }
                return true
            }
            guard stillValid else {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.isBuildingANN = false
                    if !self.isRunning, !self.isSyncing, !self.isDeduplicating,
                       !self.isModelLoading, !self.isWarmingUp {
                        self.statusText = "就绪"
                    }
                    self.scheduleIdleUnload()
                }
                return
            }
            // 目标随签名覆盖率而定：覆盖完整 → 二值 ANN（bann），否则主 ANN。
            let indexTarget = agent.ragEngine.resolvedANNIndexTarget()
            let status = indexTarget == .binary
                ? agent.ragEngine.binaryANNIndexStatus()
                : agent.ragEngine.annIndexStatus()
            await MainActor.run { [weak self] in
                guard let self else { return }
                guard let current = self.agent, ObjectIdentifier(current) == target else {
                    self.isBuildingANN = false
                    self.statusText = "就绪"
                    self.scheduleIdleUnload()
                    return
                }
                if let status {
                    if indexTarget == .binary {
                        self.binaryANNStatusText = status.summary
                    } else {
                        self.annStatusText = status.summary
                    }
                    // 已有索引：重建要丢弃现有索引并整库重跑，先确认。
                    if status.blocks > 0, !self.confirmANNIndexRebuild(status, target: indexTarget) {
                        self.isBuildingANN = false
                        self.statusText = "就绪"
                        self.appendLog("[rag] 已取消索引构建")
                        self.scheduleIdleUnload()
                        return
                    }
                }
                self.performANNIndexBuild(agent: agent, target: target, indexTarget: indexTarget)
            }
        }
    }

    /// 用户确认后真正启动构建：进度写状态栏，逐块日志由引擎按 10s 限流输出。
    private func performANNIndexBuild(agent: Agent, target: ObjectIdentifier,
                                      indexTarget: RAGEngine.ANNIndexTarget) {
        guard !isSyncing, !isDeduplicating, !isModelLoading, !isRunning, !isWarmingUp else {
            isBuildingANN = false
            showInfoAlert(title: "暂不能构建索引", message: "当前有任务正在进行，请稍后再试。")
            // 读取状态阶段已取消空闲计时；拒绝执行后恢复闲时摘要调度。
            scheduleIdleUnload()
            return
        }
        let label = indexTarget == .binary ? "二值 ANN 索引" : "ANN 索引"
        cancelIdleUnload()
        isBuildingANN = true
        statusText = indexTarget == .binary
            ? "正在构建二值 ANN 索引…"
            : "正在构建 ANN 索引…"
        appendLog("[rag] 开始构建\(label)：分块 IVF，块内 k-means + 量化；"
            + "构建期间知识检索与状态查询不受影响")
        Task.detached(priority: .utility) { [weak self] in
            // 确认重建弹窗可能停留很久；真正开跑前再等摘要退出并校验目标
            // Agent，防止旧引擎在配置切换后继续写同一份索引。
            await self?.stopSummaryIdleSession()
            let stillValid: Bool = await MainActor.run { [weak self] in
                guard let self, self.isBuildingANN,
                      let current = self.agent,
                      ObjectIdentifier(current) == target else { return false }
                return true
            }
            guard stillValid else {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.isBuildingANN = false
                    if !self.isRunning, !self.isSyncing, !self.isDeduplicating,
                       !self.isModelLoading, !self.isWarmingUp {
                        self.statusText = "就绪"
                    }
                    self.scheduleIdleUnload()
                }
                return
            }
            let result = agent.ragEngine.buildANNIndex { update in
                Task { @MainActor [weak self] in
                    guard let self, self.isBuildingANN else { return }
                    self.statusText = update.message
                }
            } log: { line in
                Task { @MainActor [weak self] in self?.appendLog(line) }
            }
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.isBuildingANN = false
                if let current = self.agent, ObjectIdentifier(current) == target {
                    self.refreshRAGStatus()
                }
                switch result {
                case .none:
                    self.appendLog("RAG: \(label)构建未执行（被拒绝或失败，详见日志）")
                    self.statusText = "\(label)构建失败"
                    self.showInfoAlert(
                        title: "\(label)构建失败",
                        message: "构建未执行或被中断，详情见日志。已写完的块会保留，可再次构建从断点续建。")
                case .some(let build):
                    self.appendLog("RAG: \(label)构建结束 — \(build.summary)")
                    self.statusText = build.cancelled ? "\(label)构建已停止" : "\(label)构建完成"
                    self.showInfoAlert(
                        title: build.cancelled ? "\(label)构建已停止" : "\(label)构建完成",
                        message: build.summary + (build.cancelled ? "。已写完的块会保留，可再次构建续建。" : "。"))
                }
                self.scheduleIdleUnload()
            }
        }
    }

    /// 停止 ANN 索引构建。已写完的块保留，下次构建从断点（rowid 水位）继续。
    func stopANNIndexBuild() {
        guard isBuildingANN else { return }
        statusText = "正在停止 ANN 索引构建…"
        appendLog("用户请求停止 ANN 索引构建")
        // 引擎在每个块之间检查取消标志，块内不中断，保证已落盘数据一致。
        agent?.ragEngine.cancel()
    }

    // MARK: 知识图谱

    private var kgBuildTask: Task<Void, Never>?

    /// 刷新知识图谱构建状态（不加载模型，直接查 SQLite）。
    func refreshKnowledgeGraphStatus() {
        guard let agent else {
            knowledgeGraphBuilt = false
            knowledgeGraphEntityCount = 0
            knowledgeGraphRelationCount = 0
            return
        }
        let status = agent.ragEngine.knowledgeGraphStatus()
        knowledgeGraphBuilt = status.built
        knowledgeGraphEntityCount = status.entities
        knowledgeGraphRelationCount = status.relationships
    }

    /// 点击「构建知识图谱」：后台用辅助模型在已完成切片上抽取实体/关系。
    func requestKnowledgeGraphBuild() {
        guard let agent else {
            showInfoAlert(title: "知识图谱", message: "RAG 尚未就绪，请先在知识同步配置中应用配置。")
            return
        }
        if isSyncing {
            showInfoAlert(title: "暂不能构建知识图谱", message: "知识同步正在进行，请等同步结束后再试。")
            return
        }
        if isBuildingKG {
            showInfoAlert(title: "知识图谱构建中", message: "正在后台构建知识图谱，请稍候。")
            return
        }
        if isBuildingANN || isDeduplicating {
            showInfoAlert(title: "暂不能构建知识图谱", message: "向量索引/去重任务正在进行，请稍后再试。")
            return
        }
        if isModelLoading || isWarmingUp {
            showInfoAlert(title: "暂不能构建知识图谱", message: "模型正在加载或预热，请稍后再试。")
            return
        }
        if isRunning {
            showInfoAlert(title: "暂不能构建知识图谱", message: "对话正在生成，请等本轮结束后再试。")
            return
        }
        guard agent.auxLLM != nil else {
            showInfoAlert(title: "暂不能构建知识图谱",
                          message: "知识图谱抽取需要辅助模型，请先在配置中启用辅助模型。")
            return
        }
        cancelIdleUnload()
        isBuildingKG = true
        statusText = "正在构建知识图谱…"
        let target = ObjectIdentifier(agent)
        kgBuildTask = Task.detached(priority: .utility) { [weak self] in
            await self?.stopSummaryIdleSession()
            let stillValid: Bool = await MainActor.run { [weak self] in
                guard let self, self.isBuildingKG, let current = self.agent,
                      ObjectIdentifier(current) == target else { return false }
                return true
            }
            guard stillValid else {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.isBuildingKG = false
                    if !self.isRunning, !self.isSyncing, !self.isDeduplicating,
                       !self.isBuildingANN, !self.isModelLoading, !self.isWarmingUp {
                        self.statusText = "就绪"
                    }
                    self.scheduleIdleUnload()
                }
                return
            }
            let stats = await agent.ragEngine.buildKnowledgeGraph(auxLLM: agent.auxLLM) { current, total, message in
                Task { @MainActor [weak self] in
                    guard let self, self.isBuildingKG else { return }
                    self.statusText = total > 0 ? "\(message) \(current)/\(total)" : message
                }
            } log: { line in
                Task { @MainActor [weak self] in self?.appendLog(line) }
            }
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.isBuildingKG = false
                if let current = self.agent, ObjectIdentifier(current) == target {
                    self.refreshKnowledgeGraphStatus()
                    self.refreshRAGStatus()
                }
                self.appendLog("[kg] 知识图谱构建完成：实体 \(stats.entities)、关系 \(stats.relationships)"
                    + "（处理切片 \(stats.chunksProcessed)，失败 \(stats.failedChunks)）")
                self.statusText = "知识图谱构建完成"
                self.scheduleIdleUnload()
            }
        }
    }

    /// 停止知识图谱构建。
    func stopKnowledgeGraphBuild() {
        guard isBuildingKG else { return }
        statusText = "正在停止知识图谱构建…"
        appendLog("用户请求停止知识图谱构建")
        kgBuildTask?.cancel()
    }

    /// 点击「实体消歧」：用向量语义相似度对图谱内同指实体做进一步合并。
    /// 抽取/构建阶段只用字符全匹配，此方法用于事后收敛重复实体。
    func requestEntityResolution() {
        guard let agent else {
            showInfoAlert(title: "实体消歧", message: "RAG 尚未就绪，请先在知识同步配置中应用配置。")
            return
        }
        if isSyncing || isBuildingKG || isDeduplicating || isBuildingANN {
            showInfoAlert(title: "暂不能实体消歧",
                          message: "知识同步/图谱构建/去重/索引任务正在进行，请稍后再试。")
            return
        }
        if isResolvingEntities {
            showInfoAlert(title: "实体消歧进行中", message: "正在对图谱做向量语义合并，请稍候。")
            return
        }
        guard agent.ragEngine.knowledgeGraphStatus().built else {
            showInfoAlert(title: "暂不能实体消歧",
                          message: "请先构建知识图谱（抽取/构建阶段仅做字符全匹配）。")
            return
        }
        cancelIdleUnload()
        isResolvingEntities = true
        statusText = "正在对知识图谱做实体消歧…"
        let target = ObjectIdentifier(agent)
        Task.detached(priority: .utility) { [weak self] in
            await self?.stopSummaryIdleSession()
            let stillValid: Bool = await MainActor.run { [weak self] in
                guard let self, self.isResolvingEntities,
                      let current = self.agent,
                      ObjectIdentifier(current) == target else { return false }
                return true
            }
            guard stillValid else {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.isResolvingEntities = false
                    self.scheduleIdleUnload()
                }
                return
            }
            do {
                let stats = try await agent.ragEngine
                    .resolveKnowledgeGraphDuplicates(threshold: 0.90)
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.isResolvingEntities = false
                    self.refreshKnowledgeGraphStatus()
                    self.appendLog("[kg] 实体消歧完成：合并 \(stats.mergedGroups) 组、"
                        + "删除 \(stats.removedEntities) 个实体（现存 \(stats.entities) 实体"
                        + " / \(stats.relationships) 关系）")
                    self.statusText = "实体消歧完成"
                    self.scheduleIdleUnload()
                }
            } catch {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.isResolvingEntities = false
                    self.appendLog("[kg] 实体消歧失败: \(error.localizedDescription)")
                    self.showInfoAlert(title: "实体消歧失败", message: error.localizedDescription)
                    self.statusText = "就绪"
                    self.scheduleIdleUnload()
                }
            }
        }
    }

    private func confirmANNIndexRebuild(_ status: ANNIndexStatus,
                                        target: RAGEngine.ANNIndexTarget) -> Bool {
        let label = target == .binary ? "二值 ANN 索引" : "ANN 索引"
        let alert = NSAlert()
        alert.messageText = "确认重建\(label)"
        alert.informativeText = """
        当前索引：\(status.summary)。
        重建会先删除现有索引（含 \(status.indexedRows) 个已索引切片），再整库重新聚类写入；
        大库上这一步可能耗时数十分钟，期间知识检索仍可正常使用。
        """
        alert.alertStyle = .warning
        // 默认按钮是「取消」，避免误触长时间占用。
        alert.addButton(withTitle: "取消")
        alert.addButton(withTitle: "重建")
        NSApp.activate(ignoringOtherApps: true)
        return alert.runModal() == .alertSecondButtonReturn
    }

    /// 删除现有 ANN 索引，回到精确检索（不重跑构建）。
    func requestDropANNIndex() {
        guard let agent else {
            showInfoAlert(title: "向量索引", message: "RAG 尚未就绪，请先在知识同步配置中应用配置。")
            return
        }
        guard !isSyncing, !isBuildingANN, !isDeduplicating, !isRunning,
              !isModelLoading, !isWarmingUp else {
            showInfoAlert(title: "暂不能删除索引", message: "当前有任务正在进行，请稍后再试。")
            return
        }
        let target = ObjectIdentifier(agent)
        cancelIdleUnload()
        isBuildingANN = true
        statusText = "正在删除向量索引…"
        appendLog("[rag] 开始删除 ANN 与二值 ANN 索引（删除后检索回退精确扫描）")
        Task.detached(priority: .utility) { [weak self] in
            // 与构建/去重路径保持一致：先等摘要会话退出，再校验目标 Agent。
            await self?.stopSummaryIdleSession()
            let stillValid: Bool = await MainActor.run { [weak self] in
                guard let self, self.isBuildingANN,
                      let current = self.agent,
                      ObjectIdentifier(current) == target else { return false }
                return true
            }
            guard stillValid else {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.isBuildingANN = false
                    if !self.isRunning, !self.isSyncing, !self.isDeduplicating,
                       !self.isModelLoading, !self.isWarmingUp {
                        self.statusText = "就绪"
                    }
                    self.scheduleIdleUnload()
                }
                return
            }
            let result: Result<Void, Error>
            do {
                try agent.ragEngine.dropANNIndex()
                result = .success(())
            } catch {
                result = .failure(error)
            }
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.isBuildingANN = false
                if let current = self.agent, ObjectIdentifier(current) == target {
                    self.refreshRAGStatus()
                }
                switch result {
                case .success:
                    self.appendLog("RAG: ANN 与二值 ANN 索引已删除")
                    self.statusText = "向量索引已删除"
                case .failure(let error):
                    self.appendLog("RAG: 删除向量索引失败 — \(error.localizedDescription)")
                    self.statusText = "删除向量索引失败"
                    self.showInfoAlert(title: "删除向量索引失败",
                                       message: error.localizedDescription)
                }
                self.scheduleIdleUnload()
            }
        }
    }

    private func showInfoAlert(title: String, message: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.addButton(withTitle: "好")
        NSApp.activate(ignoringOtherApps: true)
        alert.runModal()
    }

    // MARK: 向量库重建

    /// 点击「重建向量库」：按当前元素类型/Matryoshka 配置重建向量表，
    /// 清空旧切片与 ANN 索引后立即全量重新同步。
    func requestVectorRebuild() {
        guard let agent else {
            showInfoAlert(title: "重建向量库",
                          message: "RAG 尚未就绪，请先在知识同步配置中应用配置。")
            return
        }
        if isSyncing {
            showInfoAlert(title: "暂不能重建向量库",
                          message: "知识同步/重建正在进行，请等结束后再试。")
            return
        }
        if isDeduplicating || isBuildingANN {
            showInfoAlert(title: "暂不能重建向量库",
                          message: "向量去重或 ANN 索引维护正在进行，请稍后再试。")
            return
        }
        if isModelLoading || isWarmingUp {
            showInfoAlert(title: "暂不能重建向量库", message: "模型正在加载或预热，请稍后再试。")
            return
        }
        if isRunning {
            showInfoAlert(title: "暂不能重建向量库", message: "对话正在生成，请等本轮结束后再试。")
            return
        }
        guard confirmVectorRebuild() else { return }
        cancelIdleUnload()
        isSyncing = true
        isStoppingSync = false
        syncGeneration += 1
        statusText = "正在重建向量库…"
        appendLog("[rag] 用户确认重建向量库：元素类型 \(config.ragVectorElementType.rawValue)"
            + "，Matryoshka \(config.ragMatryoshka.rawValue) \(config.ragMatryoshkaDim) 维")
        startIngest(agent: agent, force: true, restore: nil,
                    generation: syncGeneration, rebuildVector: true)
    }

    private func confirmVectorRebuild() -> Bool {
        let type = config.ragVectorElementType
        let mrl = config.ragMatryoshka == .off
            ? "关闭"
            : "\(config.ragMatryoshkaDim) 维（\(config.ragMatryoshka.rawValue)）"
        let alert = NSAlert()
        alert.messageText = "确认重建向量库"
        alert.informativeText = """
        将按当前配置重建向量表：元素类型 \(type.rawValue)，Matryoshka \(mrl)。
        旧切片、全文索引、文档缓存与 ANN 索引会被清空，随后全量重新解析并 embedding；
        千万级切片的大库可能耗时数小时，可随时点「停止同步」中断，下次从头继续。
        知识库原始文件不会被删除。
        """
        alert.alertStyle = .warning
        // 默认按钮是「取消」，避免误触长时间占用。
        alert.addButton(withTitle: "取消")
        alert.addButton(withTitle: "重建")
        NSApp.activate(ignoringOtherApps: true)
        return alert.runModal() == .alertSecondButtonReturn
    }

    func syncKnowledgeBase(force: Bool) {
        let restore = pendingSyncRestore
        pendingSyncRestore = nil
        guard let agent, !isSyncing, !isDeduplicating, !isBuildingANN else { return }
        if blockForMissingModels("知识库同步") { return }
        cancelIdleUnload()
        if isRunning {
            appendLog("[rag] 对话进行中，跳过本次同步")
            if let restore, restore.main || restore.aux {
                Task { await resumeModelsAfterSync(main: restore.main, aux: restore.aux) }
            }
            return
        }
        // 同步已被接受：立即进入同步态，按钮马上锁定（「同步知识」变灰、
        // 「停止同步」可用），不存在模型加载等待/卸载期间的空窗。
        isSyncing = true
        isStoppingSync = false
        syncGeneration += 1
        statusText = force ? "强制同步知识库…" : "同步知识库…"

        // 启动加载/预热是短暂状态：等它结束再按「先卸载、后同步」执行。
        // 不要直接丢弃用户点击。最多等 5 分钟，避免异常卡死时无限挂起。
        if isModelLoading || isWarmingUp {
            appendLog("[rag] 模型正在加载/预热，完成后自动开始同步")
            pendingSyncRestore = restore
            let generation = syncGeneration
            Task { @MainActor [weak self] in
                for _ in 0..<3_000 {
                    guard let self else { return }
                    if !self.isModelLoading && !self.isWarmingUp { break }
                    // 等待期间用户点了「停止同步」：放弃本次同步。
                    if self.isStoppingSync { break }
                    try? await Task.sleep(nanoseconds: 100_000_000)
                }
                guard let self else { return }
                self.pendingSyncRestore = nil
                guard self.syncGeneration == generation else { return }
                guard !self.isStoppingSync else {
                    self.appendLog("[rag] 同步已在模型就绪前被停止")
                    self.isSyncing = false
                    self.isStoppingSync = false
                    self.syncTask = nil
                    if let restore, restore.main || restore.aux {
                        Task { await self.resumeModelsAfterSync(main: restore.main, aux: restore.aux) }
                    }
                    // 本次同步在开始前就被取消，模型可能无需恢复；仍要重新
                    // 计时，否则闲时摘要要等到下一次前台活动才会恢复。
                    self.scheduleIdleUnload()
                    return
                }
                if self.isModelLoading || self.isWarmingUp {
                    self.appendLog("[rag] 模型加载/预热仍未结束，取消本次同步")
                    self.isSyncing = false
                    self.isStoppingSync = false
                    self.syncTask = nil
                    if let restore, restore.main || restore.aux {
                        Task { await self.resumeModelsAfterSync(main: restore.main, aux: restore.aux) }
                    }
                    // 同上：同步未真正开始也不能让空闲计时永久停摆。
                    self.scheduleIdleUnload()
                    return
                }
                self.startIngest(agent: agent, force: force,
                                 restore: restore, generation: generation)
            }
            return
        }
        startIngest(agent: agent, force: force, restore: restore, generation: syncGeneration)
    }

    /// 已处于同步态（isSyncing == true）：卸载本地模型并启动 ingest 任务。
    /// `rebuildVector` 为 true 时先按当前配置重建向量表，再执行全量同步。
    private func startIngest(agent: Agent, force: Bool,
                             restore: (main: Bool, aux: Bool)?, generation: Int,
                             rebuildVector: Bool = false) {
        // applyConfig 后的同步：模型尚未加载，按应用配置前的状态在同步结束后恢复。
        let wasMain = restore?.main ?? mainModelLoaded
        // 同步结束后的恢复也必须看当前开关：关闭时即使旧状态是已加载也不恢复。
        let wasAux = (restore?.aux ?? auxModelLoaded) && config.auxEnabled
        let modelNote = config.auxEnabled ? "主/辅助模型" : "主模型（辅助模型未启用）"
        if rebuildVector {
            appendLog("[rag] 重建向量库：先卸载\(modelNote)，按当前元素类型/Matryoshka 配置重建后全量同步")
        } else {
            appendLog("[rag] 同步开始：先卸载\(modelNote)，结束后重新加载并预热主模型")
        }
        statusText = rebuildVector ? "正在重建向量库…" : (force ? "强制同步知识库…" : "同步知识库…")
        let syncGen = generation
        syncTask = Task.detached(priority: .userInitiated) { [weak self, weak agent] in
            guard let agent else {
                await MainActor.run { [weak self] in
                    guard let self, self.syncGeneration == syncGen else { return }
                    self.isSyncing = false
                    self.isStoppingSync = false
                    self.syncTask = nil
                }
                return
            }
            // 先等闲时摘要会话退出并释放辅助模型，再卸载本地模型：
            // 避免「同步卸载」与摘要的在途生成并发释放同一份权重。
            await self?.stopSummaryIdleSession()
            await MainActor.run { [weak self] in
                self?.suspendModelsForSync()
            }
            // 给 MLX/GGUF 的异步卸载一点收尾时间，避免刚释放的权重与
            // RAG 的 embedding 模型抢内存。GGUF 是同步释放，这里几乎无感知。
            try? await Task.sleep(nanoseconds: 200_000_000)
            let stats: IngestStats
            var rebuildSummary: String?
            if rebuildVector {
                let result = await agent.ragEngine.rebuildVectorIndex { current, total, file in
                    Task { @MainActor [weak self] in
                        guard let self, self.syncGeneration == syncGen else { return }
                        self.statusText = "重建后同步 \(current)/\(total) \(file)"
                    }
                } log: { line in
                    Task { @MainActor [weak self] in self?.appendLog(line) }
                }
                if let result {
                    rebuildSummary = result.summary
                    stats = result.ingest ?? IngestStats()
                } else {
                    var failed = IngestStats()
                    failed.errors.append("向量库重建未执行或被中断，详情见日志")
                    stats = failed
                }
            } else {
                stats = await agent.ragEngine.ingest(force: force) { current, total, file in
                    Task { @MainActor [weak self] in
                        guard let self, self.syncGeneration == syncGen else { return }
                        self.statusText = "同步 \(current)/\(total) \(file)"
                    }
                } log: { line in
                    Task { @MainActor [weak self] in self?.appendLog(line) }
                }
            }
            // 只重载实际执行同步的引擎：配置切换后 self.agent 可能已经是新实例，
            // 绝不能把旧同步的结果作用到新 agent 上。
            agent.reloadRAG()
            // 先复制到不可变局部变量：下面的 MainActor 闭包是 Sendable，
            // 直接捕获 var 在 Swift 6 语言模式下会报错。
            let finishedRebuildSummary = rebuildSummary
            await MainActor.run { [weak self] in
                guard let self, self.syncGeneration == syncGen else { return }
                self.isSyncing = false
                self.isStoppingSync = false
                self.syncTask = nil
                if let finishedRebuildSummary {
                    self.appendLog("RAG: \(finishedRebuildSummary)")
                }
                self.appendLog("RAG: \(stats.summary)")
                for error in stats.errors.prefix(20) {
                    self.appendLog("RAG 错误: \(error)")
                }
                if self.agent === agent {
                    self.refreshRAGStatus()
                    // 同步可能让旧图谱失效（stale），刷新后 UI 自动回落到 naive。
                    self.refreshKnowledgeGraphStatus()
                } else {
                    self.statusText = "就绪"
                }
                self.scheduleIdleUnload()
            }
            // 同步结束（成功/失败/取消）后按同步前状态恢复模型。
            let shouldRestore = await MainActor.run(body: { [weak self] in
                guard let self else { return false }
                return self.syncGeneration == syncGen && self.agent === agent
            })
            if shouldRestore {
                // 同步 Task 可能刚被「停止同步」取消；恢复必须放到新 Task，
                // 否则 warmup 中的 Task.checkCancellation 会立即抛错显示「预热失败」。
                Task.detached { [weak self] in
                    await self?.resumeModelsAfterSync(main: wasMain, aux: wasAux)
                }
            }
        }
    }

    /// 知识同步前卸载主/辅模型，释放权重、KV cache 与 MLX 缓冲。
    private func suspendModelsForSync() {
        // 只卸载本地可预加载模型；API 模式的 OpenAI 客户端不占本地权重，保持原样。
        let hasMain = mainModelLoaded || agent?.llm is PreloadableLLMService
        // 辅助模型关闭时视为没有可卸载的辅助模型，避免同步路径误触。
        let hasAux = config.auxEnabled
            && (auxModelLoaded || agent?.auxLLM is PreloadableLLMService)
        guard hasMain || hasAux else {
            appendLog("[model] 知识同步前模型未加载，无需卸载")
            return
        }
        if hasMain {
            agent?.switchLLM(UnloadedLLMService())
            appendLog("[model] 已卸载主模型，释放权重与 KV cache")
        }
        if hasAux {
            agent?.unloadAuxLLM()
            appendLog("[model] 已卸载辅助模型，释放权重与 KV cache")
        }
        LocalModelFactory.shared.unloadLLMCaches()
        mainModelLoaded = false
        auxModelLoaded = false
        appendLog("[model] 主/辅助模型卸载完成，开始知识同步")
    }

    /// 同步结束后恢复同步前已加载的模型；主模型会重新预热。
    private func resumeModelsAfterSync(main: Bool, aux: Bool) async {
        // 恢复前再次确认开关：同步期间被关闭（或旧状态带 aux）时都不能误加载。
        let shouldLoadAux = aux && config.auxEnabled
        guard main || shouldLoadAux, let agent else { return }
        var parts: [String] = []
        if main { parts.append("主模型") }
        if shouldLoadAux { parts.append("辅助模型") }
        appendLog("[model] 知识同步结束：重新加载\(parts.joined(separator: "与"))"
            + (main ? "，主模型加载后自动预热" : ""))
        statusText = "同步结束，重新加载模型…"
        if main, agent.llm is UnloadedLLMService {
            agent.switchLLM(LocalModelFactory.shared.makeLLM(config: config))
        }
        if shouldLoadAux, agent.auxLLM == nil {
            agent.auxLLM = LocalModelFactory.shared.makeAuxLLM(config: config)
        }
        await preloadModels(main: main, aux: shouldLoadAux)
    }

    func stopSync() {
        guard isSyncing, !isStoppingSync else { return }
        isStoppingSync = true
        statusText = "正在停止同步…"
        appendLog("用户请求停止知识库同步")
        // 内部标志让 RAG 在文件/embedding 批次边界退出；Task 取消让
        // SemanticChunker 等支持 Task.isCancelled 的步骤立即返回。
        agent?.ragEngine.cancel()
        syncTask?.cancel()
    }

    /// 缓存 RAG 统计，避免 SwiftUI body 每次重绘都查库。
    /// 查库可能触发建库 + COUNT/DISTINCT，必须放到后台，不能卡主线程。
    func refreshRAGStatus() {
        guard let agent else {
            ragStatusText = "RAG 未就绪"
            return
        }
        guard !ragStatusRefreshing else { return }
        ragStatusRefreshing = true
        // 用 ObjectIdentifier 判新旧实例，避免在并发闭包里捕获会变化的 agent 引用。
        let targetID = ObjectIdentifier(agent)
        let engine = agent.ragEngine
        Task.detached(priority: .utility) { [weak self] in
            let status = engine.status()
            let chunks = status["chunks_stored"]?.intValue ?? 0
            let backend = status["vector_backend"]?.stringValue ?? "sqlite"
            let sources = status["source_count"]?.intValue
                ?? status["sources"]?.arrayValue?.count ?? 0
            let ann = status["ann_index"]?.stringValue ?? ""
            let binaryANN = status["binary_ann_index"]?.stringValue ?? ""
            let rebuildRequired = status["vector_rebuild_required"]?.boolValue ?? false
            let rebuildReason = status["rebuild_reason"]?.stringValue ?? ""
            // 超时/不可用时 status 带 error：状态栏给可读提示，不显示误导性的 0 切片。
            let text: String
            if let reason = status["error"]?.stringValue {
                text = "知识库\(reason)（上次统计：\(chunks) 切片 · \(sources) 文件）"
            } else {
                text = "\(chunks) 切片 · \(sources) 文件 · \(backend)"
            }
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.ragStatusRefreshing = false
                if let current = self.agent,
                   ObjectIdentifier(current) == targetID {
                    self.ragStatusText = text
                    if !ann.isEmpty { self.annStatusText = ann }
                    if !binaryANN.isEmpty { self.binaryANNStatusText = binaryANN }
                    self.vectorRebuildRequired = rebuildRequired
                    self.vectorRebuildReason = rebuildReason
                } else {
                    // agent 已切换：丢弃旧结果，重新为新实例取一次状态。
                    self.refreshRAGStatus()
                }
            }
        }
    }
}
