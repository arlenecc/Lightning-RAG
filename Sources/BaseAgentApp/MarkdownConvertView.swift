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
                Text("待转换文件").font(.caption)
                TextField("选择文件或目录", text: Binding(
                    get: { state.sourceURL?.path ?? "" },
                    set: { _ in }
                ))
                .textFieldStyle(.roundedBorder)
                .frame(minWidth: 240)
                Button("选择…") { state.pickSource() }
                    .help("可选择一个文件，或选择整个目录递归转换\n"
                        + "Office 支持 .docx / .xlsx / .pptx 与旧版 .doc / .xls / .ppt\n支持："
                        + MarkdownConverterState.supportedExtensionHint)

                Text("保存目录").font(.caption)
                TextField("转换结果保存位置", text: Binding(
                    get: { state.outputDirectory?.path ?? "" },
                    set: { _ in }
                ))
                .textFieldStyle(.roundedBorder)
                .frame(minWidth: 180)
                Button("选择…") { state.pickOutputDirectory() }
                    .disabled(state.isRunning)

                Button(state.isRunning ? "转换中…" : "开始转换") {
                    state.start(config: config)
                }
                .buttonStyle(.borderedProminent)
                .disabled(state.sourceURL == nil || state.outputDirectory == nil
                    || state.isRunning)

                if state.isRunning {
                    Button("停止") { state.cancel() }
                }

                Toggle("增量转换", isOn: $state.incremental)
                    .toggleStyle(.checkbox)
                    .disabled(state.isRunning)
                    .help("开启时跳过上次已成功转换且源文件未变化的文件；\n"
                        + "关闭后强制重新转换全部文件")
            }

            HStack(spacing: 8) {
                Text("设置来源：\(engineLabel(config.ragPDFEngine)) PDF 引擎"
                    + " · EPUB \(engineLabel(config.ragEPUBOCREngine))"
                    + " · 图像归一化\(config.ragOCRPreprocess ? "开" : "关")"
                    + " · Otsu 二值化\(config.ragOCRBinarize ? "开" : "关")")
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
        case .hybrid: return "混合"
        case .custom: return "自定义"
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
                    Text("已完成 \(state.completedFiles) · 失败 \(state.failedFiles)"
                        + " · 共 \(state.fileTotal)")
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
                Text("转换结果").font(.caption).bold()
                if let report = state.report {
                    Text("成功 \(report.succeededCount - report.skippedCount)"
                        + " · 跳过 \(report.skippedCount)"
                        + " · 失败 \(report.failedCount)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 4)
                Button("清空列表") { state.clearResults() }
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
                            Text(state.isRunning ? "转换中…" : "尚未开始转换。")
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

            Text(item.skipped ? "跳过" : String(format: "%.1fs", item.elapsed))
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

    var percentText: String {
        String(Int((overallFraction * 100).rounded()))
    }

    var fileDescription: String {
        guard fileTotal > 0 else {
            return isRunning ? "正在扫描待转换文件…" : "等待选择待转换文件"
        }
        let name = currentFileName.isEmpty ? "—" : currentFileName
        return "第 \(max(1, fileIndex))/\(fileTotal) 个文件 · \(name)"
    }

    var stageDescription: String {
        if let pageIndex, let pageTotal, pageTotal > 0 {
            return "当前环节：\(currentStage.isEmpty ? "解析" : currentStage)"
                + "（第 \(pageIndex)/\(pageTotal) 页）"
        }
        return "当前环节：\(currentStage.isEmpty ? "—" : currentStage)"
    }

    // MARK: 选择路径

    func pickSource() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        panel.prompt = "选择"
        panel.message = "可选择一个文件，或选择整个目录递归转换"
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
        panel.prompt = "选择保存目录"
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
        currentStage = "准备中"
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
        task = Task { [weak self] in
            let service = MarkdownConversionService()
            do {
                let report = try await service.convert(
                    request: request,
                    progress: { progress in
                        Task { @MainActor [weak self] in
                            self?.apply(progress)
                        }
                    },
                    onResult: { result in
                        Task { @MainActor [weak self] in
                            self?.results.append(result)
                        }
                    })
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.isRunning = false
                    self.report = report
                    self.completedFiles = report.succeededCount
                    self.failedFiles = report.failedCount
                    self.overallFraction = report.wasCancelled
                        ? self.overallFraction : 1
                    self.currentStage = report.wasCancelled ? "已取消" : "转换完成"
                    let skipped = report.skippedCount > 0
                        ? " · 跳过 \(report.skippedCount)" : ""
                    self.statusMessage = report.wasCancelled
                        ? "已取消：成功 \(report.succeededCount)\(skipped)"
                            + " · 失败 \(report.failedCount)"
                        : "转换完成：成功 \(report.succeededCount)\(skipped)"
                            + " · 失败 \(report.failedCount)"
                            + "（用时 \(String(format: "%.1f", report.elapsed))s）"
                }
            } catch {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.isRunning = false
                    self.statusMessage = "转换失败：\(error.localizedDescription)"
                }
            }
        }
    }

    func cancel() {
        task?.cancel()
        statusMessage = "正在停止…"
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
