import SwiftUI
import AppKit
import UniformTypeIdentifiers
import BaseAgentCore

// MARK: - 窗口

/// 独立的文件转 Markdown 窗口：选择单个文件或整个目录，批量转换并写入指定目录。
/// 解析、OCR、图片归一化、清洗等全部复用知识同步使用的解析器，
/// 使用打开窗口时配置面板上的 OCR / PDF 设置快照，不改动运行中的 Agent。
struct MarkdownConvertView: View {
    @ObservedObject var model: AppViewModel
    @StateObject private var state = MarkdownConverterState()

    private var config: AgentConfig { model.converterConfig ?? model.config }

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            progressPane
            Divider()
            resultPane
        }
        .frame(minWidth: 1_060, minHeight: 560)
        .onDisappear {
            // 关闭窗口时取消仍在进行的批量转换，避免后台继续占用 OCR/GPU。
            state.cancel()
        }
    }

    // MARK: 顶部操作栏

    private var toolbar: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text(L("待转换文件")).font(.caption)
                TextField(L("选择文件或目录"), text: Binding(
                    get: { state.sourceURL?.path ?? "" },
                    set: { _ in }
                ))
                .textFieldStyle(.roundedBorder)
                .frame(minWidth: 240)
                Button(L("选择…")) { state.pickSource() }
                    .help(L("可选择一个文件，或选择整个目录递归转换\n")
                        + L("Office 支持 .docx / .xlsx / .pptx 与旧版 .doc / .xls / .ppt\n支持：")
                        + MarkdownConverterState.supportedExtensionHint)

                Text(L("保存目录")).font(.caption)
                TextField(L("转换结果保存位置"), text: Binding(
                    get: { state.outputDirectory?.path ?? "" },
                    set: { _ in }
                ))
                .textFieldStyle(.roundedBorder)
                .frame(minWidth: 180)
                Button(L("选择…")) { state.pickOutputDirectory() }
                    .disabled(state.isRunning)

                Button(state.isRunning ? L("转换中…") : L("开始转换")) {
                    state.start(config: config)
                }
                .buttonStyle(.borderedProminent)
                .disabled(state.sourceURL == nil || state.outputDirectory == nil
                    || state.isRunning)

                if state.isRunning {
                    Button(L("停止")) { state.cancel() }
                }

                Toggle(L("增量转换"), isOn: $state.incremental)
                    .toggleStyle(.checkbox)
                    .disabled(state.isRunning)
                    .help(L("开启时跳过上次已成功转换且源文件未变化的文件；\n")
                        + L("关闭后强制重新转换全部文件"))
            }

            HStack(spacing: 8) {
                Text(L("设置来源：\(engineLabel(config.ragPDFEngine)) PDF 引擎")
                    + " · EPUB \(engineLabel(config.ragEPUBOCREngine))"
                    + L(" · 图像归一化\(config.ragOCRPreprocess ? "开" : "关")")
                    + L(" · Otsu 二值化\(config.ragOCRBinarize ? "开" : "关")"))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer(minLength: 8)
                if !state.statusMessage.isEmpty {
                    Text(state.statusMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
        }
        .padding(12)
    }

    private func engineLabel(_ engine: PDFEngine) -> String {
        switch engine {
        case .vision: return "Apple Vision"
        case .ovis: return "OvisOCR3"
        case .hybrid: return L("混合")
        case .custom: return L("自定义")
        }
    }

    // MARK: 进度

    private var progressPane: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text(state.fileDescription)
                    .font(.caption)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer(minLength: 8)
                Text("\(state.percentText)%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            ProgressView(value: state.overallFraction)
                .progressViewStyle(.linear)

            HStack(spacing: 8) {
                Text(state.stageDescription)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                Spacer(minLength: 8)
                if state.fileTotal > 0 {
                    Text(L("已完成 \(state.completedFiles) · 失败 \(state.failedFiles)")
                        + L(" · 共 \(state.fileTotal)"))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
    }

    // MARK: 结果

    private var resultPane: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 6) {
                Text(L("转换结果")).font(.caption).bold()
                if let report = state.report {
                    Text(L("成功 \(report.succeededCount - report.skippedCount)")
                        + L(" · 跳过 \(report.skippedCount)")
                        + L(" · 失败 \(report.failedCount)"))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 4)
                Button(L("清空列表")) { state.clearResults() }
                    .buttonStyle(.borderless)
                    .font(.caption)
                    .disabled(state.isRunning || state.results.isEmpty)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)

            Divider()

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        if state.results.isEmpty {
                            Text(state.isRunning ? L("转换中…") : L("尚未开始转换。"))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .padding(10)
                        } else {
                            ForEach(state.results) { item in
                                resultRow(item)
                                    .id(item.id)
                            }
                        }
                    }
                    .padding(8)
                }
                .onChange(of: state.results.count) { _, _ in
                    if let last = state.results.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
    }

    private func resultRow(_ item: MarkdownConversionResult) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: item.skipped
                  ? "arrow.right.circle"
                  : (item.succeeded ? "checkmark.circle.fill" : "xmark.circle.fill"))
                .font(.caption2)
                .foregroundStyle(item.skipped
                    ? Color.secondary : (item.succeeded ? .green : .red))
                .padding(.top, 1)

            VStack(alignment: .leading, spacing: 1) {
                Text(item.source.lastPathComponent)
                    .font(.caption)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if let output = item.outputURL {
                    Text(output.path)
                        .font(.system(.caption2, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .lineLimit(1)
                        .truncationMode(.middle)
                } else if let error = item.error {
                    Text(error)
                        .font(.caption2)
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Spacer(minLength: 6)

            Text(item.skipped ? L("跳过") : String(format: "%.1fs", item.elapsed))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
        .padding(.horizontal, 4)
    }
}

// MARK: - 状态

@MainActor
final class MarkdownConverterState: ObservableObject {
    /// 与知识同步保持一致的可解析扩展名提示。
    static let supportedExtensionHint = RAGEngine.supportedExtensions.sorted()
        .joined(separator: "、")

    @Published var sourceURL: URL?
    @Published var outputDirectory: URL?
    @Published var isRunning = false
    @Published var currentFileName = ""
    @Published var currentStage = ""
    @Published var pageIndex: Int?
    @Published var pageTotal: Int?
    @Published var fileIndex = 0
    @Published var fileTotal = 0
    @Published var completedFiles = 0
    @Published var failedFiles = 0
    @Published var overallFraction = 0.0
    @Published var statusMessage = ""
    @Published var results: [MarkdownConversionResult] = []
    @Published var report: MarkdownConversionReport?
    /// 增量转换：跳过上次已成功且源文件未变化的文件。
    @Published var incremental = true

    private var task: Task<Void, Never>?
    /// UI 更新缓冲：批量转换 7000+ 文件时，逐条 Task 派发会淹没主线程 runloop，
    /// 造成界面假死；这里把结果/进度批量合并后再一次性送到 MainActor。
    private final class UIBuffer {
        private let lock = NSLock()
        private var pendingResults: [MarkdownConversionResult] = []
        private var pendingProgress: MarkdownConversionProgress?
        private var flushTask: Task<Void, Never>?
        private var consumer: (([MarkdownConversionResult], MarkdownConversionProgress?) -> Void)?

        func start(consumer: @escaping ([MarkdownConversionResult], MarkdownConversionProgress?) -> Void) {
            self.consumer = consumer
        }

        func addResult(_ result: MarkdownConversionResult, maxResults: Int = 2000) {
            lock.lock()
            pendingResults.append(result)
            // 列表上限：超出只保留最后 maxResults 条，统计由 report 提供
            if pendingResults.count > maxResults * 2 {
                pendingResults.removeFirst(pendingResults.count - maxResults)
            }
            let shouldFlush = pendingResults.count >= 20
            lock.unlock()
            if shouldFlush { scheduleFlush(immediate: true) }
            else { scheduleFlush(immediate: false) }
        }

        func addProgress(_ progress: MarkdownConversionProgress) {
            lock.lock()
            pendingProgress = progress
            lock.unlock()
            scheduleFlush(immediate: false)
        }

        private func scheduleFlush(immediate: Bool) {
            lock.lock()
            let need = flushTask == nil
            if need {
                flushTask = Task { [weak self] in
                    try? await Task.sleep(nanoseconds: immediate ? 30_000_000 : 120_000_000)
                    self?.flushNow()
                }
            }
            lock.unlock()
        }

        private func flushNow() {
            lock.lock()
            let results = pendingResults
            let progress = pendingProgress
            pendingResults.removeAll()
            pendingProgress = nil
            flushTask = nil
            let handler = consumer
            lock.unlock()
            guard !results.isEmpty || progress != nil else { return }
            Task { @MainActor in handler?(results, progress) }
        }

        func finish() {
            lock.lock()
            flushTask?.cancel()
            flushTask = nil
            lock.unlock()
            flushNow()
        }
    }
    private let uiBuffer = UIBuffer()

    var percentText: String {
        String(Int((overallFraction * 100).rounded()))
    }

    var fileDescription: String {
        guard fileTotal > 0 else {
            return isRunning ? L("正在扫描待转换文件…") : L("等待选择待转换文件")
        }
        let name = currentFileName.isEmpty ? "—" : currentFileName
        return L("第 \(max(1, fileIndex))/\(fileTotal) 个文件 · \(name)")
    }

    var stageDescription: String {
        if let pageIndex, let pageTotal, pageTotal > 0 {
            return L("当前环节：\(currentStage.isEmpty ? "解析" : currentStage)")
                + L("（第 \(pageIndex)/\(pageTotal) 页）")
        }
        return L("当前环节：\(currentStage.isEmpty ? "—" : currentStage)")
    }

    // MARK: 选择路径

    func pickSource() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        panel.prompt = L("选择")
        panel.message = L("可选择一个文件，或选择整个目录递归转换")
        NSApp.activate(ignoringOtherApps: true)
        guard panel.runModal() == .OK, let url = panel.url else { return }
        sourceURL = url.standardizedFileURL
        currentFileName = ""
        currentStage = ""
        fileIndex = 0
        fileTotal = 0
        overallFraction = 0
        statusMessage = ""
        if outputDirectory == nil {
            // 单文件默认输出到同目录；目录默认输出到其下的 markdown 子目录。
            outputDirectory = url.hasDirectoryPath
                ? url.appendingPathComponent("markdown")
                : url.deletingLastPathComponent()
        }
    }

    func pickOutputDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.prompt = L("选择保存目录")
        NSApp.activate(ignoringOtherApps: true)
        guard panel.runModal() == .OK, let url = panel.url else { return }
        outputDirectory = url.standardizedFileURL
    }

    // MARK: 转换

    func start(config: AgentConfig) {
        guard let source = sourceURL, !isRunning else { return }
        let output = outputDirectory
            ?? source.deletingLastPathComponent().appendingPathComponent("markdown")
        outputDirectory = output

        results.removeAll()
        report = nil
        isRunning = true
        currentFileName = ""
        currentStage = L("准备中")
        pageIndex = nil
        pageTotal = nil
        fileIndex = 0
        fileTotal = 0
        completedFiles = 0
        failedFiles = 0
        overallFraction = 0
        statusMessage = ""

        let request = MarkdownConversionRequest(sourceURL: source,
                                                outputDirectory: output,
                                                config: config,
                                                incremental: incremental)
        uiBuffer.start { [weak self] results, progress in
            guard let self else { return }
            if !results.isEmpty {
                self.results.append(contentsOf: results)
                if self.results.count > 2000 {
                    self.results.removeFirst(self.results.count - 2000)
                }
            }
            if let progress { self.apply(progress) }
        }
        task = Task { [weak self] in
            let service = MarkdownConversionService()
            do {
                let report = try await service.convert(
                    request: request,
                    progress: { [weak self] progress in
                        self?.uiBuffer.addProgress(progress)
                    },
                    onResult: { [weak self] result in
                        self?.uiBuffer.addResult(result)
                    })
                self?.uiBuffer.finish()
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.isRunning = false
                    self.report = report
                    self.completedFiles = report.succeededCount
                    self.failedFiles = report.failedCount
                    self.overallFraction = report.wasCancelled
                        ? self.overallFraction : 1
                    self.currentStage = report.wasCancelled ? L("已取消") : L("转换完成")
                    let skipped = report.skippedCount > 0
                        ? L(" · 跳过 \(report.skippedCount)") : ""
                    self.statusMessage = report.wasCancelled
                        ? L("已取消：成功 \(report.succeededCount)\(skipped)")
                            + L(" · 失败 \(report.failedCount)")
                        : L("转换完成：成功 \(report.succeededCount)\(skipped)")
                            + L(" · 失败 \(report.failedCount)")
                            + L("（用时 \(String(format: "%.1f", report.elapsed))s）")
                }
            } catch {
                self?.uiBuffer.finish()
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.isRunning = false
                    self.statusMessage = L("转换失败：\(error.localizedDescription)")
                }
            }
        }
    }

    func cancel() {
        task?.cancel()
        statusMessage = L("正在停止…")
    }

    func clearResults() {
        results.removeAll()
        report = nil
        statusMessage = ""
    }

    private func apply(_ progress: MarkdownConversionProgress) {
        // 收尾阶段由 start 的完成分支写入最终统计，这里只更新过程量。
        if progress.phase == .finished { return }
        if !progress.fileName.isEmpty {
            currentFileName = progress.fileName
        }
        currentStage = progress.stage
        pageIndex = progress.pageIndex
        pageTotal = progress.pageTotal
        if progress.fileIndex > 0 { fileIndex = progress.fileIndex }
        if progress.fileTotal > 0 { fileTotal = progress.fileTotal }
        completedFiles = progress.completedFiles
        failedFiles = progress.failedFiles
        overallFraction = progress.overallFraction
    }
}
