import SwiftUI
import AppKit
import BaseAgentCore
import UniformTypeIdentifiers

/// 从终端直接运行裸可执行文件时，macOS 不会自动把窗口提到前台；
/// 启动后显式激活，避免用户找不到窗口或以为程序没运行。
final class BaseAgentAppDelegate: NSObject, NSApplicationDelegate {
    /// 退出收尾需要访问 App 模型；SwiftUI 创建 delegate 时拿不到 @StateObject，
    /// 由 MainView 出现时注入。
    @MainActor weak var model: AppViewModel?
    @MainActor private var isTerminating = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        // 裸可执行文件没有 .app bundle，运行时显式设置 Dock 图标。
        if let url = Bundle.module.url(forResource: "AppIcon", withExtension: "png"),
           let icon = NSImage(contentsOf: url) {
            NSApp.applicationIconImage = icon
        }
        NSApp.activate(ignoringOtherApps: true)
    }

    /// 退出前先停止在途生成并排空 GPU 队列，再真正退出。
    ///
    /// 直接退出会让 MLX 仍在执行的 asyncEval 与进程退出（销毁 command
    /// encoder）并发，触发 Metal 断言或 SIGSEGV。
    @MainActor
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        AgentLog.info("[app] 收到退出请求，先停止生成并排空 GPU 队列")
        guard let model, !isTerminating else {
            return isTerminating ? .terminateLater : .terminateNow
        }
        isTerminating = true
        Task { @MainActor in
            await model.prepareForTermination()
            AgentLog.info("[app] 收尾完成，退出进程")
            NSApp.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }

    func applicationWillTerminate(_ notification: Notification) {
        AgentLog.info("[app] 进程结束")
    }
}

@main
struct BaseAgentDesktopApp: App {
    @NSApplicationDelegateAdaptor(BaseAgentAppDelegate.self) private var appDelegate
    @StateObject private var model = AppViewModel()

    var body: some Scene {
        WindowGroup {
            MainView(model: model, delegate: appDelegate)
                .frame(minWidth: 1100, minHeight: 700)
        }
        .defaultSize(width: 1280, height: 820)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandMenu("对话") {
                Button("清空对话") { model.resetConversation() }
                    .keyboardShortcut("k", modifiers: [.command, .shift])
                Button("停止") { model.cancel() }
                    .keyboardShortcut(".", modifiers: [.command])
                    .disabled(!model.isRunning)
                Button("同步知识库") { model.syncKnowledgeBase(force: false) }
                    .disabled(model.isRunning || model.isModelLoading
                        || model.isWarmingUp || model.isSyncing || model.isBuildingANN)
                Button("强制全量同步") { model.syncKnowledgeBase(force: true) }
                    .disabled(model.isRunning || model.isModelLoading
                        || model.isWarmingUp || model.isSyncing || model.isBuildingANN)
                Button(model.isStoppingSync ? "正在停止同步…" : "停止知识库同步") {
                    model.stopSync()
                }
                .disabled(!model.isSyncing || model.isStoppingSync)
                Button("构建 ANN 索引") { model.requestANNIndexBuild() }
                    .disabled(model.isRunning || model.isModelLoading || model.isWarmingUp
                        || model.isSyncing || model.isBuildingANN)
                Button("停止 ANN 索引构建") { model.stopANNIndexBuild() }
                    .disabled(!model.isBuildingANN)
            }
        }

        Window("配置", id: "settings") {
            SettingsView(model: model)
        }
        .defaultSize(width: 940, height: 860)

        Window("日志", id: "logs") {
            LogsWindowView(model: model)
        }
        .defaultSize(width: 900, height: 620)

        Window("引用切片", id: "citation-viewer") {
            ChunkViewerWindowView(model: model)
        }
        .defaultSize(width: 780, height: 680)

        Window("OCR 识别测试", id: "ocr-lab") {
            OCRLabView(model: model)
                .frame(minWidth: 900, minHeight: 640)
        }
        .defaultSize(width: 1280, height: 880)

        Window("转换工具", id: "markdown-converter") {
            MarkdownConvertView(model: model)
        }
        .defaultSize(width: 1120, height: 680)
    }
}

// MARK: - 主窗口

/// 聊天输入框待发送的附件（图片等多模态内容）。
struct AttachmentItem: Identifiable {
    let id = UUID()
    let data: Data
    let thumbnail: NSImage
    let fileName: String
}

struct MainView: View {
    @ObservedObject var model: AppViewModel
    /// 退出收尾需要拿到当前 App 模型，由这里注入给 delegate。
    let delegate: BaseAgentAppDelegate
    @Environment(\.openWindow) private var openWindow
    @State private var input = ""
    /// 待发送的图片等附件；随首条用户消息传给多模态后端。
    @State private var attachments: [AttachmentItem] = []
    @State private var showAttachmentPicker = false

    var body: some View {
        // 左侧 3/4：对话区从窗口顶部到底部，tokens 与输入区固定在底部；
        // 右侧 1/4：任务计划（上）+ 思考过程（下）+ 状态栏。
        HSplitView {
            chatPane
                .frame(minWidth: 640, maxWidth: .infinity, maxHeight: .infinity)

            sidebar
                .frame(minWidth: 210, idealWidth: 230, maxWidth: 320,
                       maxHeight: .infinity)
        }
        .frame(minWidth: 1100, minHeight: 700)
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                // 与配置窗口内「开始同步」一致的快捷入口：目录已配置时无需进设置。
                Button {
                    model.syncKnowledgeBase(force: false)
                } label: {
                    Label("同步知识", systemImage: "arrow.triangle.2.circlepath")
                        .labelStyle(.titleAndIcon)
                }
                .disabled(model.isRunning || model.isModelLoading
                    || model.isWarmingUp || model.isSyncing || model.isBuildingANN
                    || !model.missingDefaultModels.isEmpty)
                .help("开始知识库同步（与配置窗口内「开始同步」一致）")

                Button {
                    model.stopSync()
                } label: {
                    Label(model.isStoppingSync ? "正在停止…" : "停止同步",
                          systemImage: "stop.circle")
                        .labelStyle(.titleAndIcon)
                }
                .disabled(!model.isSyncing || model.isStoppingSync)
                .help(model.isStoppingSync ? "正在停止同步…" : "停止知识库同步")

                Button {
                    openWindow(id: "settings")
                } label: {
                    Label("配置", systemImage: "gearshape")
                        .labelStyle(.titleAndIcon)
                }
                .help("打开配置窗口")

                Button {
                    openWindow(id: "logs")
                } label: {
                    Label("日志", systemImage: "list.bullet.rectangle")
                        .labelStyle(.titleAndIcon)
                }
                .help("打开日志窗口")
            }
        }
        .onAppear {
            delegate.model = model
            model.refreshRAGStatus()
            // 首启提醒：默认模型缺失时弹窗引导联网下载（可稍后，状态栏保留入口）。
            model.promptModelDownloadIfNeeded()
            if model.config.ragAutoIngest, model.missingDefaultModels.isEmpty {
                model.syncKnowledgeBase(force: false)
            }
        }
    }

    // MARK: 左侧

    private var sidebar: some View {
        VStack(spacing: 0) {
            VSplitView {
                planPane
                    .frame(minHeight: 200, maxHeight: .infinity)
                schedulePane
                    .frame(minHeight: 120, maxHeight: .infinity)
                thinkingPane
                    .frame(minHeight: 120, maxHeight: .infinity)
            }
            Divider()
            statusBar
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    // MARK: 定时任务

    private var schedulePane: some View {
        VStack(spacing: 0) {
            HStack(spacing: 6) {
                Text("定时任务").font(.caption).bold()
                if !model.scheduledTasks.isEmpty {
                    Text("\(model.scheduledTasks.filter(\.enabled).count)/\(model.scheduledTasks.count)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 4)
                Button {
                    model.refreshScheduledTasks()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .help("刷新定时任务")
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)

            Divider()

            ScrollView {
                if model.scheduledTasks.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("暂无定时任务。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text("在对话里说「每天早上 9 点帮我……」即可创建；"
                            + "到点后自动执行，结果显示在对话里。")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(model.scheduledTasks) { task in
                            scheduleRow(task)
                        }
                    }
                    .padding(8)
                }
            }
        }
    }

    private func scheduleRow(_ task: ScheduledTask) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(alignment: .firstTextBaseline, spacing: 5) {
                if task.isCompleted {
                    Image(systemName: "checkmark.seal.fill")
                        .font(.caption)
                        .foregroundStyle(Color.green)
                        .help("一次性任务已完成")
                } else {
                    Button {
                        model.setScheduledTaskEnabled(task, enabled: !task.enabled)
                    } label: {
                        Image(systemName: task.enabled
                              ? "checkmark.circle.fill" : "circle")
                            .font(.caption)
                            .foregroundStyle(task.enabled ? Color.accentColor : Color.secondary)
                    }
                    .buttonStyle(.borderless)
                    .help(task.enabled ? "停用任务" : "启用任务")
                }

                Text(task.title)
                    .font(.caption)
                    .fontWeight(task.enabled ? .semibold : .regular)
                    .foregroundStyle(task.enabled ? .primary : .secondary)
                    .lineLimit(2)
                Spacer(minLength: 2)
                Button {
                    model.runScheduledTaskNow(task)
                } label: {
                    Image(systemName: "play.circle").font(.caption)
                }
                .buttonStyle(.borderless)
                .help("立即执行一次")
                .disabled(model.isRunning || model.isModelLoading
                    || model.isWarmingUp || model.isSyncing || model.isBuildingANN)
                Button {
                    model.deleteScheduledTask(task)
                } label: {
                    Image(systemName: "trash").font(.caption)
                }
                .buttonStyle(.borderless)
                .help("删除任务")
            }
            Text(scheduleDetail(task))
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(3)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.leading, 18)
        }
    }

    private func scheduleDetail(_ task: ScheduledTask) -> String {
        var parts = [task.rule.summary]
        if task.isCompleted {
            parts.append("已完成（一次性）")
        } else if task.enabled {
            let next = task.nextRunDate()
            parts.append(Date() >= next
                ? "已到期（等待执行）"
                : "下次 \(scheduleLocalTimeText(next))")
        } else {
            parts.append("已停用")
        }
        if let result = task.lastResult, !result.isEmpty {
            parts.append("上次：\(result)")
        }
        return parts.joined(separator: " · ")
    }

    // MARK: 任务计划

    private var planPane: some View {
        VStack(spacing: 0) {
            HStack(spacing: 6) {
                Text("任务计划").font(.caption).bold()
                if let plan = model.plan {
                    Text(planProgress(plan))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 4)
                Button {
                    model.plan = model.agent?.session.getPlan()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .help("刷新计划")
                Button {
                    model.clearPlan()
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.borderless)
                .help("清空计划（运行中会先终止）")
                .disabled(model.plan == nil && !model.isRunning)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)

            Divider()

            ScrollView {
                if let plan = model.plan, !plan.steps.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        if !plan.goal.isEmpty {
                            Text(plan.goal)
                                .font(.caption)
                                .bold()
                                .fixedSize(horizontal: false, vertical: true)
                                .padding(.bottom, 2)
                        }
                        ForEach(plan.steps) { step in
                            planRow(step)
                        }
                    }
                    .padding(8)
                } else {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("暂无计划。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text("Agent 开始任务前会先规划步骤，这里会实时显示进度。")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }

    private func planProgress(_ plan: Plan) -> String {
        let finished = plan.steps.filter {
            ["done", "blocked", "skipped"].contains($0.status)
        }.count
        return "\(finished)/\(plan.steps.count)"
    }

    private func planRow(_ step: PlanStep) -> some View {
        let isCurrent = step.status == "doing"
        return HStack(alignment: .top, spacing: 6) {
            Image(systemName: planIcon(step.status))
                .font(.caption)
                .foregroundStyle(planColor(step.status))
                .frame(width: 14)

            VStack(alignment: .leading, spacing: 2) {
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text("\(step.id).")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(step.title)
                        .font(.caption)
                        .fontWeight(isCurrent ? .semibold : .regular)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if !step.note.isEmpty {
                    Text(step.note)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Spacer(minLength: 0)

            if isCurrent {
                Text("进行中")
                    .font(.caption2)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(Color.accentColor.opacity(0.18))
                    .foregroundStyle(Color.accentColor)
                    .cornerRadius(4)
            }
        }
        .padding(6)
        .background(isCurrent ? Color.accentColor.opacity(0.10) : Color.clear)
        .cornerRadius(6)
    }

    private func planIcon(_ status: String) -> String {
        switch status {
        case "done": return "checkmark.circle.fill"
        case "doing": return "play.circle.fill"
        case "blocked": return "xmark.octagon.fill"
        case "skipped": return "minus.circle"
        default: return "circle"
        }
    }

    private func planColor(_ status: String) -> Color {
        switch status {
        case "done": return .green
        case "doing": return .accentColor
        case "blocked": return .red
        case "skipped": return .secondary
        default: return .secondary
        }
    }

    // MARK: 思考过程

    private var thinkingPane: some View {
        VStack(spacing: 0) {
            HStack(spacing: 6) {
                Text("思考过程").font(.caption).bold()
                Spacer(minLength: 4)
                Text(model.reasoning.isEmpty ? "—" : "\(model.reasoning.count) 字")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)

            Divider()

            ScrollViewReader { proxy in
                ScrollView {
                    if model.reasoning.isEmpty {
                        Text("暂无思考内容。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(10)
                    } else {
                        Text(model.reasoning)
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(8)
                            .id("thinkingBottom")
                    }
                }
                .onChange(of: model.reasoning) { _, _ in
                    proxy.scrollTo("thinkingBottom", anchor: .bottom)
                }
            }
        }
    }

    // MARK: 状态栏

    private var statusBar: some View {
        VStack(alignment: .leading, spacing: 3) {
            if model.isDownloadingModels || !model.missingDefaultModels.isEmpty {
                modelDownloadBar
            }
            HStack(spacing: 6) {
                Text(modelLabel)
                    .font(.caption2)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(modelLabel)
                Spacer(minLength: 4)
                if model.isRunning || model.isModelLoading
                    || model.isWarmingUp || model.isSyncing || model.isBuildingANN {
                    ProgressView().controlSize(.mini)
                }
                Text(model.statusText)
                    .font(.caption2)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            HStack(spacing: 6) {
                Text(model.ragStatusText)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                Spacer(minLength: 4)
                Button {
                    model.syncKnowledgeBase(force: false)
                } label: {
                    Image(systemName: "arrow.triangle.2.circlepath")
                        .font(.caption)
                }
                .buttonStyle(.borderless)
                .disabled(model.isRunning || model.isModelLoading
                    || model.isWarmingUp || model.isSyncing || model.isBuildingANN
                    || !model.missingDefaultModels.isEmpty)
                .help("增量同步知识库")
                Button {
                    model.syncKnowledgeBase(force: true)
                } label: {
                    Image(systemName: "arrow.uturn.forward.circle")
                        .font(.caption)
                }
                .buttonStyle(.borderless)
                .disabled(model.isRunning || model.isModelLoading
                    || model.isWarmingUp || model.isSyncing || model.isBuildingANN
                    || !model.missingDefaultModels.isEmpty)
                .help("强制全量同步（忽略增量缓存，全部重新解析入库）")
                if model.isSyncing {
                    Button {
                        model.stopSync()
                    } label: {
                        Image(systemName: model.isStoppingSync
                              ? "hourglass" : "stop.circle")
                            .font(.caption)
                    }
                    .buttonStyle(.borderless)
                    .disabled(model.isStoppingSync)
                    .help(model.isStoppingSync ? "正在停止同步…" : "停止知识库同步")
                }
                if model.isBuildingANN {
                    Button {
                        model.stopANNIndexBuild()
                    } label: {
                        Image(systemName: "stop.circle").font(.caption)
                    }
                    .buttonStyle(.borderless)
                    .help("停止 ANN 索引构建（已写完的块会保留，可再次构建续建）")
                }
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    /// 默认模型下载：缺失时提示并可发起下载，下载中显示聚合进度。
    /// 未就绪前对话与知识同步均被拦截（下载完成才能正常使用软件）。
    @ViewBuilder
    private var modelDownloadBar: some View {
        VStack(alignment: .leading, spacing: 4) {
            if model.isDownloadingModels {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text(model.modelDownloadStatusText)
                        .font(.caption2)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .help(model.modelDownloadStatusText)
                    Spacer(minLength: 4)
                    Button {
                        model.cancelModelDownload()
                    } label: {
                        Image(systemName: "stop.circle").font(.caption)
                    }
                    .buttonStyle(.borderless)
                    .help("停止下载（已下载部分保留，下次从断点续传）")
                }
                ProgressView(value: model.modelDownloadProgress)
            } else if model.config.autoDownloadMissingModels {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                    Text("默认模型缺失，需联网下载后才能使用")
                        .font(.caption2)
                        .lineLimit(1)
                        .truncationMode(.tail)
                    Spacer(minLength: 4)
                    Button("下载") {
                        model.startModelDownload()
                    }
                    .buttonStyle(.borderless)
                    .controlSize(.small)
                    .help("从 hf-mirror.com 下载默认模型（支持断点续传）")
                }
            } else {
                // 自行下载指定模式：不自动下载，引导用户去配置窗口指定模型；
                // 保留手动触发自动下载的入口（仅显式点击才会启动）。
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                    Text("默认模型缺失（自行管理模式），请在配置中指定可用模型")
                        .font(.caption2)
                        .lineLimit(1)
                        .truncationMode(.tail)
                    Spacer(minLength: 4)
                    Button("打开配置") {
                        openWindow(id: "settings")
                    }
                    .buttonStyle(.borderless)
                    .controlSize(.small)
                    .help("在配置窗口为各阶段指定自行准备好的模型")
                    Button("自动下载") {
                        model.startModelDownload()
                    }
                    .buttonStyle(.borderless)
                    .controlSize(.small)
                    .help("手动触发一次自动下载补全（已选择自行管理，不会自动启动）")
                }
            }
        }
        .padding(.vertical, 2)
    }

    private var modelLabel: String {
        if model.config.llmProvider == "local" {
            return model.config.localLLMModel.isEmpty
                ? "本地 MLX（未配置模型）"
                : model.config.localLLMModel
        }
        return "\(model.config.model) · \(model.config.baseURL)"
    }

    // MARK: 对话

    private var chatPane: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 10) {
                        ForEach(Array(model.chatItems.enumerated()), id: \.element.id) { _, item in
                            chatBubble(item)
                                .id(item.id)
                        }
                    }
                    .padding()
                }
                .onChange(of: model.chatItems.count) { _, _ in
                    scrollToBottom(proxy, animated: true)
                }
                .onChange(of: model.chatItems.last?.content) { _, _ in
                    // 流式内容每 50ms 更新一次，逐次带动画会明显卡顿。
                    scrollToBottom(proxy, animated: false)
                }
            }

            Divider()

            Text(tokenSummary)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity)
                .multilineTextAlignment(.center)
                .padding(.top, 6)

            // 待发送附件缩略图；用户可在发送前逐个移除。
            if !attachments.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(attachments) { att in
                            ZStack(alignment: .topTrailing) {
                                Image(nsImage: att.thumbnail)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(height: 44)
                                    .cornerRadius(6)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 6)
                                            .strokeBorder(Color.secondary.opacity(0.3),
                                                          lineWidth: 1))
                                Button {
                                    attachments.removeAll { $0.id == att.id }
                                } label: {
                                    Image(systemName: "xmark.circle.fill")
                                        .font(.system(size: 15))
                                        .foregroundStyle(.white, Color.black.opacity(0.55))
                                }
                                .buttonStyle(.plain)
                                .offset(x: 5, y: -5)
                                .help("移除附件")
                            }
                        }
                    }
                    .padding(.horizontal, 10)
                }
                .frame(height: 52)
            }

            HStack(alignment: .bottom, spacing: 8) {
                TextField("输入消息，Enter 发送（Shift+Enter 换行）…",
                          text: $input, axis: .vertical)
                    .lineLimit(1...6)
                    .textFieldStyle(.plain)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    // 输入区高度约为原来单行输入框的两倍。
                    .frame(minHeight: 46, alignment: .topLeading)
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(Color.secondary.opacity(0.12)))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .strokeBorder(Color.secondary.opacity(0.25), lineWidth: 1))
                    // 多行 TextField 会吞掉 Return，onSubmit 不会触发；直接拦截按键。
                    .onKeyPress(.return, phases: [.down]) { press in
                        // 中文输入法正在组合候选词时，Enter 必须交给 IME 确认候选；
                        // 否则会先触发 send() 清空输入框，再被 IME 回填成残留拼音。
                        if press.modifiers.contains(.shift) || hasMarkedText {
                            return .ignored
                        }
                        guard !model.isRunning, !model.isModelLoading,
                              !model.isWarmingUp, !model.isSyncing,
                              !model.isBuildingANN else {
                            return .handled
                        }
                        send()
                        return .handled
                    }
                // 添加附件：图片等多模态内容随首条用户消息传给多模态后端。
                Button {
                    showAttachmentPicker = true
                } label: {
                    Image(systemName: "paperclip")
                        .font(.system(size: 18))
                        .foregroundStyle(attachments.isEmpty ? Color.secondary : Color.accentColor)
                        .frame(width: 36, height: 46)
                        .background(
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .fill(Color.secondary.opacity(0.12)))
                }
                .buttonStyle(.plain)
                .help("添加图片等附件（多模态）")
                .disabled(model.isRunning || model.isModelLoading || model.isWarmingUp)
                Button {
                    if model.isRunning {
                        model.cancel()
                    } else {
                        send()
                    }
                } label: {
                    Text(model.isRunning ? "停止" : "发送")
                        .font(.body.weight(.medium))
                        .foregroundStyle(.white)
                        // 与左侧消息输入框同高（46pt）、同圆角。
                        .frame(width: 64, height: 46)
                        .background(
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .fill(model.isRunning ? Color.red : Color.accentColor))
                }
                .buttonStyle(.plain)
                .disabled(sendBlocked)
                .opacity(sendBlocked ? 0.45 : 1)
                .help(model.isRunning ? "停止本轮生成" : "发送消息")
            }
            .padding(10)
        }
        .fileImporter(
            isPresented: $showAttachmentPicker,
            allowedContentTypes: [.image],
            allowsMultipleSelection: true
        ) { result in
            guard let urls = try? result.get() else { return }
            for url in urls {
                guard url.startAccessingSecurityScopedResource() else { continue }
                defer { url.stopAccessingSecurityScopedResource() }
                guard let data = try? Data(contentsOf: url),
                      let image = NSImage(contentsOf: url) else { continue }
                let thumb = Self.makeThumbnail(from: image)
                attachments.append(AttachmentItem(
                    data: data, thumbnail: thumb, fileName: url.lastPathComponent))
            }
        }
    }

    /// 把附件图片缩小到适合气泡/预览的缩略图，避免大图占用内存。
    private static func makeThumbnail(from image: NSImage, maxSize: CGFloat = 256) -> NSImage {
        let size = image.size
        guard size.width > 0, size.height > 0 else { return image }
        let scale = min(1, maxSize / max(size.width, size.height))
        let newSize = NSSize(width: size.width * scale, height: size.height * scale)
        let thumb = NSImage(size: newSize)
        thumb.lockFocus()
        NSGraphicsContext.current?.imageInterpolation = .high
        image.draw(in: NSRect(origin: .zero, size: newSize))
        thumb.unlockFocus()
        return thumb
    }

    /// 发送按钮不可用：默认模型未就绪，正在加载/预热/同步/构建索引，
    /// 且输入与附件都为空。
    private var sendBlocked: Bool {
        !model.isRunning
            && ((input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                 && attachments.isEmpty)
                || !model.missingDefaultModels.isEmpty
                || model.isModelLoading || model.isWarmingUp || model.isSyncing
                || model.isBuildingANN)
    }

    private var tokenSummary: String {
        if model.tokenSpeed > 0 {
            return "\(model.totalTokens) tokens · "
                + String(format: "%.1f", model.tokenSpeed) + " tok/s"
        }
        return "\(model.totalTokens) tokens"
    }

    private func chatBubble(_ item: ChatItem) -> some View {
        HStack {
            if item.role == .user { Spacer(minLength: 60) }
            VStack(alignment: .leading, spacing: 8) {
                if item.role == .user, let images = item.images, !images.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 6) {
                            ForEach(Array(images.enumerated()), id: \.offset) { _, data in
                                if let image = NSImage(data: data) {
                                    Image(nsImage: image)
                                        .resizable()
                                        .scaledToFit()
                                        .frame(height: 120)
                                        .cornerRadius(6)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 6)
                                                .strokeBorder(Color.secondary.opacity(0.3),
                                                              lineWidth: 1))
                                }
                            }
                        }
                    }
                    .frame(maxHeight: 130)
                }
                Text(item.content.isEmpty && item.role == .assistant ? "…" : item.content)
                    .textSelection(.enabled)
                // 检索即展示：全部召回切片（含模型参考的 top-1）以气泡跟在答案后，
                // 点击在独立窗口查看原文，可前后切换相邻切片。
                if item.role == .assistant, let citations = item.citations,
                   !citations.isEmpty {
                    CitationChipsRow(citations: citations, messageID: item.id,
                                     model: model) {
                        openWindow(id: "citation-viewer")
                    }
                }
            }
            .padding(10)
            .background(background(for: item.role))
            .cornerRadius(10)
            .frame(maxWidth: 760, alignment: .leading)
            if item.role == .assistant { Spacer(minLength: 160) }
        }
    }

    private func background(for role: ChatItem.Role) -> Color {
        switch role {
        case .user: return Color.accentColor.opacity(0.18)
        case .assistant: return Color(nsColor: .controlBackgroundColor)
        case .system: return Color.yellow.opacity(0.12)
        }
    }

    private var hasMarkedText: Bool {
        (NSApp.keyWindow?.firstResponder as? NSTextView)?.hasMarkedText() ?? false
    }

    private func send() {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty || !attachments.isEmpty else { return }
        let images = attachments.map(\.data)
        input = ""
        attachments.removeAll()
        model.send(text, images: images)
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy, animated: Bool) {
        guard let last = model.chatItems.last else { return }
        if animated {
            withAnimation(.easeOut(duration: 0.15)) {
                proxy.scrollTo(last.id, anchor: .bottom)
            }
        } else {
            proxy.scrollTo(last.id, anchor: .bottom)
        }
    }
}

// MARK: - 独立日志窗口

struct LogsWindowView: View {
    @ObservedObject var model: AppViewModel
    @State private var tab = 0

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Picker("", selection: $tab) {
                    Text("日志").tag(0)
                    Text("工具调用").tag(1)
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                Spacer(minLength: 8)

                Button("复制全部") {
                    let text = model.logs
                        .map { "\(timeText($0.timestamp))  \($0.text)" }
                        .joined(separator: "\n")
                    let pb = NSPasteboard.general
                    pb.clearContents()
                    pb.setString(text, forType: .string)
                }
                .disabled(model.logs.isEmpty)
                .help("复制全部日志到剪贴板")

                Button("清空") {
                    model.clearLogs()
                }
                .disabled(model.logs.isEmpty && model.toolEvents.isEmpty)
                .help("清空当前面板的日志记录")
            }
            .padding(8)

            Divider()

            if tab == 0 {
                logList
            } else {
                toolList
            }
        }
        .frame(minWidth: 720, minHeight: 420)
    }

    private var logList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    if model.logs.isEmpty {
                        Text("暂无日志。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .padding(10)
                    } else {
                        ForEach(model.logs) { log in
                            Text("\(timeText(log.timestamp))  \(log.text)")
                                .font(.system(.caption, design: .monospaced))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .id(log.id)
                        }
                        Color.clear.frame(height: 1).id("logBottom")
                    }
                }
                .padding(8)
            }
            // 用「最后一条日志的 id」做触发：日志超过 2000 条后裁剪旧条目，
            // 数组 count 恒定不变，按 count 触发会从此停滚；id 每条唯一。
            .onChange(of: model.logs.last?.id) { _, _ in
                scrollToLatest(proxy)
            }
            .onAppear {
                scrollToLatest(proxy)
            }
        }
    }

    private func scrollToLatest(_ proxy: ScrollViewProxy) {
        // 优先滚到最后一条真实日志（LazyVStack 对未渲染区域的
        // "logBottom" 锚点定位不稳）；空列表时兜底。
        if let last = model.logs.last {
            proxy.scrollTo(last.id, anchor: .bottom)
        } else {
            proxy.scrollTo("logBottom", anchor: .bottom)
        }
    }

    private var toolList: some View {
        Group {
            if model.toolEvents.isEmpty {
                Text("暂无工具调用记录。")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(model.toolEvents) { event in
                    ToolEventRow(event: event)
                }
            }
        }
    }

    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter
    }()

    private func timeText(_ date: Date) -> String {
        Self.timeFormatter.string(from: date)
    }
}

// MARK: - 检索即展示

/// 回复下方的召回切片气泡行：每个 chip 一条切片，点击在独立窗口查看原文。
struct CitationChipsRow: View {
    let citations: [RAGCitation]
    let messageID: UUID
    @ObservedObject var model: AppViewModel
    let onOpen: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("召回切片 \(citations.count) 条（点击查看原文）")
                .font(.caption2)
                .foregroundStyle(.secondary)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(Array(citations.enumerated()), id: \.element.id) { index, citation in
                        chip(index, citation)
                    }
                }
                .padding(.vertical, 1)
            }
        }
    }

    private func chip(_ index: Int, _ citation: RAGCitation) -> some View {
        HStack(spacing: 5) {
            Image(systemName: citation.isPrimary ? "star.fill" : "doc.text")
                .font(.caption2)
                .foregroundStyle(citation.isPrimary ? Color.accentColor : Color.secondary)
                .help(citation.isPrimary ? "模型参考本切片总结回复" : "召回切片，仅供用户查看")
            Text("#\(index + 1) \(citation.source)")
                .font(.caption)
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(maxWidth: 220, alignment: .leading)
            if !citation.heading.isEmpty {
                Text(citation.heading)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .frame(maxWidth: 160, alignment: .leading)
            }
            Text(String(format: "%.2f", citation.score))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(citation.isPrimary
                              ? Color.accentColor.opacity(0.55)
                              : Color.secondary.opacity(0.35),
                              lineWidth: citation.isPrimary ? 1.2 : 1))
        .contentShape(Rectangle())
        .onTapGesture {
            model.openCitation(messageID: messageID, index: index)
            onOpen()
        }
        .help("查看切片原文（⌘[ / ⌘] 切换相邻切片）")
    }
}

/// 引用切片查看窗口：显示当前切片原文，可切换同一回复内的前/后切片，
/// 避免切片截断带来的阅读困扰。
struct ChunkViewerWindowView: View {
    @ObservedObject var model: AppViewModel

    var body: some View {
        Group {
            if let state = model.citationViewer,
               let item = model.chatItems.first(where: { $0.id == state.messageID }),
               let citations = item.citations, !citations.isEmpty {
                viewer(state: state, citations: citations)
            } else {
                Text("没有正在查看的引用切片。\n点击对话中回复下方的切片气泡即可在此查看原文。")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    private func viewer(state: CitationViewerState,
                        citations: [RAGCitation]) -> some View {
        let index = min(max(state.index, 0), citations.count - 1)
        let citation = citations[index]
        return VStack(spacing: 0) {
            header(citation, index: index, total: citations.count, state: state)
            Divider()
            // 切片原文：入库切片本身有长度上限，整段直出，不二次截断。
            ScrollView {
                Text(citation.text)
                    .font(.system(size: 13))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
            }
        }
        .id(state)
    }

    private func header(_ citation: RAGCitation, index: Int, total: Int,
                        state: CitationViewerState) -> some View {
        VStack(spacing: 4) {
            HStack(spacing: 8) {
                Button {
                    model.openCitation(messageID: state.messageID, index: index - 1)
                } label: {
                    Image(systemName: "chevron.left")
                }
                .disabled(index == 0)
                // 用 ⌘[ / ⌘] 而不是 ⌘← / ⌘→：后者是输入框的行首/行尾快捷键，
                // 避免快捷键串窗时抢走主窗口输入框的编辑行为。
                .keyboardShortcut("[", modifiers: .command)
                .help("上一个切片（⌘[）")

                Text("切片 \(index + 1) / \(total)")
                    .font(.callout.monospacedDigit())

                Button {
                    model.openCitation(messageID: state.messageID, index: index + 1)
                } label: {
                    Image(systemName: "chevron.right")
                }
                .disabled(index >= total - 1)
                .keyboardShortcut("]", modifiers: .command)
                .help("下一个切片（⌘]）")

                if citation.isPrimary {
                    Label("模型参考", systemImage: "star.fill")
                        .font(.caption2)
                        .foregroundStyle(Color.accentColor)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.accentColor.opacity(0.14))
                        .cornerRadius(6)
                }
                Spacer(minLength: 8)
                Text("相关度 \(String(format: "%.4f", citation.score))")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 4) {
                Text(citation.source)
                    .font(.caption).bold()
                    .lineLimit(1)
                    .truncationMode(.middle)
                if !citation.heading.isEmpty {
                    Text("· \(citation.heading)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(nsColor: .controlBackgroundColor))
    }
}

// MARK: - 工具事件行

struct ToolEventRow: View {
    let event: ToolEvent

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(event.name).bold().font(.caption)
                Spacer()
                if let success = event.success {
                    Image(systemName: success ? "checkmark.circle" : "xmark.circle")
                        .foregroundStyle(success ? .green : .red)
                } else {
                    ProgressView().controlSize(.small)
                }
            }
            Text(event.arguments)
                .font(.system(.caption2, design: .monospaced))
                .lineLimit(2)
            if !event.output.isEmpty {
                Text(event.output)
                    .font(.system(.caption2, design: .monospaced))
                    .textSelection(.enabled)
                    .lineLimit(6)
            }
        }
        .padding(.vertical, 2)
    }
}
