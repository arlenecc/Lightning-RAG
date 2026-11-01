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
                    || model.isWarmingUp || model.isSyncing || model.isBuildingANN)
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
            if model.config.ragAutoIngest {
                model.syncKnowledgeBase(force: false)
            }
        }
    }

    // MARK: 左侧

    private var sidebar: some View {
        VStack(spacing: 0) {
            VSplitView {
                planPane
                    .frame(minHeight: 220, maxHeight: .infinity)
                thinkingPane
                    .frame(minHeight: 140, maxHeight: .infinity)
            }
            Divider()
            statusBar
        }
        .background(Color(nsColor: .windowBackgroundColor))
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
                    || model.isWarmingUp || model.isSyncing || model.isBuildingANN)
                .help("增量同步知识库")
                Button {
                    model.syncKnowledgeBase(force: true)
                } label: {
                    Image(systemName: "arrow.uturn.forward.circle")
                        .font(.caption)
                }
                .buttonStyle(.borderless)
                .disabled(model.isRunning || model.isModelLoading
                    || model.isWarmingUp || model.isSyncing || model.isBuildingANN)
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

    /// 发送按钮不可用：正在加载/预热/同步/构建索引，且输入与附件都为空。
    private var sendBlocked: Bool {
        !model.isRunning
            && ((input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                 && attachments.isEmpty)
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
