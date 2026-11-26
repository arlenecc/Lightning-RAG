import SwiftUI
import AppKit
import PDFKit
import WebKit
import UniformTypeIdentifiers
import BaseAgentCore

// MARK: - 窗口

struct OCRLabView: View {
    @ObservedObject var model: AppViewModel
    @StateObject private var state = OCRLabState()

    private var config: AgentConfig { model.ocrLabConfig ?? model.config }

    private var availableEngines: [OCRLabEngine] {
        let isPDF = state.fileURL.map(OCRLabService.isPDF) ?? true
        let hasCustom = !config.ragOCRModelPath
            .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        return OCRLabEngine.allCases.filter { engine in
            if engine.isPDFOnly && !isPDF { return false }
            if engine == .custom && !hasCustom { return false }
            return true
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            HSplitView {
                originalPane
                    .frame(minWidth: 280, idealWidth: 340, maxWidth: 520)
                resultPane
                    .frame(minWidth: 440, maxWidth: .infinity)
            }
        }
        .frame(minWidth: 960, minHeight: 640)
        .onChange(of: state.fileURL) { _, _ in
            if !availableEngines.contains(state.engine),
               let first = availableEngines.first {
                state.engine = first
            }
        }
    }

    // MARK: 顶部操作栏

    private var toolbar: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text(L("待识别文件")).font(.caption)
                TextField(L("选择图片或 PDF 文件"), text: Binding(
                    get: { state.fileURL?.path ?? "" },
                    set: { _ in }
                ))
                .textFieldStyle(.roundedBorder)
                .frame(minWidth: 220)
                Button(L("选择…")) { state.pickFile() }

                Text(L("识别模型")).font(.caption)
                Picker("", selection: $state.engine) {
                    ForEach(availableEngines, id: \.self) { engine in
                        Text(engineLabel(engine)).tag(engine)
                    }
                }
                .labelsHidden()
                .frame(width: 220)

                Button(state.isRunning ? L("识别中…") : L("开始识别")) {
                    state.start(config: config)
                }
                .buttonStyle(.borderedProminent)
                .disabled(state.fileURL == nil || state.isRunning
                    || !availableEngines.contains(state.engine))

                Button(L("保存结果")) { state.save() }
                    .disabled(state.result == nil)
            }

            HStack(spacing: 8) {
                if state.isRunning {
                    ProgressView().controlSize(.small)
                }
                Text(state.statusMessage.isEmpty ? state.progressText : state.statusMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                Spacer()
                if let result = state.result {
                    Text(L("\(result.engineLabel) · \(result.pageCount) 页"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(height: 18)
        }
        .padding(12)
    }

    private func engineLabel(_ engine: OCRLabEngine) -> String {
        engine == .custom ? L("自定义：\(state.customModelName(config: config))") : engine.title
    }

    // MARK: 左侧原文件

    private var originalPane: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(L("原文件")).font(.headline)
                Spacer()
                if state.pageCount > 1 {
                    Text(L("共 \(state.pageCount) 页")).font(.caption).foregroundStyle(.secondary)
                }
            }
            Group {
                if let url = state.fileURL {
                    if OCRLabService.isPDF(url) {
                        pdfPages
                    } else if let image = NSImage(contentsOf: url) {
                        ScrollView([.horizontal, .vertical]) {
                            Image(nsImage: image)
                                .resizable()
                                .scaledToFit()
                                .padding(8)
                        }
                    } else {
                        emptyHint(L("无法读取图片"))
                    }
                } else {
                    emptyHint(L("请选择图片或 PDF 文件"))
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(nsColor: .textBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        .padding(12)
    }

    private var pdfPages: some View {
        ScrollView {
            LazyVStack(spacing: 10) {
                ForEach(0..<max(1, state.pageCount), id: \.self) { index in
                    Button {
                        state.selectedPage = index
                    } label: {
                        VStack(spacing: 4) {
                            if let image = state.thumbnail(index) {
                                Image(nsImage: image)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(maxWidth: 200, maxHeight: 260)
                                    .background(Color.white)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 4)
                                            .stroke(state.selectedPage == index
                                                    ? Color.accentColor
                                                    : Color.gray.opacity(0.35),
                                                    lineWidth: state.selectedPage == index ? 2.5 : 1)
                                    )
                            } else {
                                RoundedRectangle(cornerRadius: 4)
                                    .fill(Color.gray.opacity(0.12))
                                    .frame(width: 180, height: 240)
                            }
                            Text(L("第 \(index + 1) 页"))
                                .font(.caption)
                                .foregroundStyle(state.selectedPage == index
                                                 ? Color.accentColor : .secondary)
                        }
                        .padding(4)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(8)
        }
    }

    private func emptyHint(_ text: String) -> some View {
        Text(text)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: 右侧识别结果

    private var resultPane: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(L("识别结果")).font(.headline)
                Spacer()
                if let result = state.result, result.pageCount > 1 {
                    Text(L("第 \(min(state.selectedPage + 1, result.pageCount)) / \(result.pageCount) 页"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Group {
                if let markdown = state.selectedMarkdown {
                    if markdown.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        emptyHint(L("本页未识别出内容"))
                    } else {
                        HTMLView(html: MarkdownHTML.document(markdown))
                    }
                } else {
                    emptyHint(L("点击「开始识别」后，这里显示识别结果"))
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(nsColor: .textBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        .padding(12)
    }
}

// MARK: - 状态

@MainActor
final class OCRLabState: ObservableObject {
    @Published var fileURL: URL?
    @Published var engine: OCRLabEngine = .apple
    @Published var isRunning = false
    @Published var result: OCRLabResult?
    @Published var selectedPage = 0
    @Published var pageCount = 0
    @Published var progressText = ""
    @Published var statusMessage = ""
    @Published var pdfDocument: PDFDocument?

    private var thumbnails: [Int: NSImage] = [:]
    private var task: Task<Void, Never>?

    func pickFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.pdf, .image]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        NSApp.activate(ignoringOtherApps: true)
        guard panel.runModal() == .OK, let url = panel.url else { return }
        load(url: url)
    }

    func load(url: URL) {
        task?.cancel()
        fileURL = url
        result = nil
        selectedPage = 0
        progressText = ""
        statusMessage = ""
        thumbnails.removeAll()

        if OCRLabService.isPDF(url) {
            let document = PDFDocument(url: url)
            pdfDocument = document
            pageCount = document?.pageCount ?? 0
        } else {
            pdfDocument = nil
            pageCount = 1
        }
        if engine.isPDFOnly, !OCRLabService.isPDF(url) {
            engine = .apple
        }
    }

    func thumbnail(_ index: Int) -> NSImage? {
        if let cached = thumbnails[index] { return cached }
        guard let page = pdfDocument?.page(at: index) else { return nil }
        let image = page.thumbnail(of: NSSize(width: 180, height: 240), for: .cropBox)
        thumbnails[index] = image
        return image
    }

    var selectedMarkdown: String? {
        guard let result else { return nil }
        guard selectedPage >= 0, selectedPage < result.pages.count else { return "" }
        return result.pages[selectedPage].markdown
    }

    func customModelName(config: AgentConfig) -> String {
        let path = config.ragOCRModelPath
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty else { return L("未配置") }
        let url = URL(fileURLWithPath: path)
        if config.localVLMBackend.lowercased() == "mlx" {
            return url.lastPathComponent
        }
        let name = url.deletingPathExtension().lastPathComponent
        return name.isEmpty ? url.lastPathComponent : name
    }

    func start(config: AgentConfig) {
        guard let fileURL, !isRunning else { return }
        isRunning = true
        result = nil
        selectedPage = 0
        statusMessage = ""
        progressText = L("准备识别…")
        let engine = self.engine

        task = Task {
            do {
                let result = try await OCRLabService.run(
                    url: fileURL,
                    engine: engine,
                    config: config
                ) { progress in
                    let text = Self.describe(progress)
                    Task { @MainActor in
                        self.progressText = text
                    }
                }
                isRunning = false
                self.result = result
                pageCount = max(pageCount, result.pageCount)
                selectedPage = 0
                progressText = ""
                statusMessage = L("识别完成：\(result.pages.count) 页")
            } catch {
                isRunning = false
                progressText = ""
                statusMessage = L("识别失败：\(error.localizedDescription)")
            }
        }
    }

    func save() {
        guard let result else { return }
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        let base = fileURL?.deletingPathExtension().lastPathComponent ?? "ocr"
        panel.nameFieldStringValue = base + ".md"
        if let markdownType = UTType(filenameExtension: "md") {
            panel.allowedContentTypes = [markdownType]
        }
        NSApp.activate(ignoringOtherApps: true)
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try result.markdown.write(to: url, atomically: true, encoding: .utf8)
            statusMessage = L("已保存到 \(url.path)")
        } catch {
            statusMessage = L("保存失败：\(error.localizedDescription)")
        }
    }

    private static func describe(_ progress: ParserProgress) -> String {
        switch progress {
        case .page(let index, let total, let stage):
            let name: String
            switch stage {
            case "text": name = L("提取文本层")
            case "ocr": name = L("OCR 识别")
            default: name = stage
            }
            return L("第 \(index)/\(total) 页 · \(name)")
        case .message(let message):
            return message
        }
    }
}

// MARK: - HTML 渲染

struct HTMLView: NSViewRepresentable {
    let html: String

    final class Coordinator {
        var lastHTML = ""
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> WKWebView {
        let view = WKWebView()
        context.coordinator.lastHTML = html
        view.loadHTMLString(html, baseURL: nil)
        return view
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        guard context.coordinator.lastHTML != html else { return }
        context.coordinator.lastHTML = html
        webView.loadHTMLString(html, baseURL: nil)
    }
}

/// 轻量 Markdown → HTML：覆盖 OCR 结果常见的标题、表格、列表、代码块与内联格式，
/// 原始 HTML 表格块原样透传，交给 WKWebView 渲染。
enum MarkdownHTML {
    static func document(_ markdown: String) -> String {
        """
        <!doctype html>
        <html><head><meta charset="utf-8">
        <style>
        :root { color-scheme: light; }
        body { font: 14px/1.65 -apple-system, "PingFang SC", "Helvetica Neue", sans-serif;
               color: #1d1d1f; margin: 0; padding: 18px 20px 28px; background: #fff;
               word-wrap: break-word; }
        h1 { font-size: 1.6em; margin: 0.8em 0 0.45em; }
        h2 { font-size: 1.35em; margin: 0.85em 0 0.4em; }
        h3 { font-size: 1.15em; margin: 0.8em 0 0.35em; }
        h4, h5, h6 { font-size: 1em; margin: 0.75em 0 0.3em; }
        p { margin: 0.55em 0; }
        table { border-collapse: collapse; margin: 0.8em 0; max-width: 100%; }
        th, td { border: 1px solid #d2d2d7; padding: 6px 9px; vertical-align: top;
                 text-align: left; }
        th { background: #f5f5f7; font-weight: 600; }
        pre { background: #f5f5f7; border-radius: 6px; padding: 10px 12px;
              overflow-x: auto; }
        code { font-family: "SF Mono", Menlo, monospace; font-size: 0.92em;
               background: #f5f5f7; border-radius: 4px; padding: 1px 4px; }
        pre code { background: transparent; padding: 0; }
        blockquote { margin: 0.7em 0; padding: 0.2em 0 0.2em 12px;
                     border-left: 3px solid #d2d2d7; color: #515154; }
        ul, ol { margin: 0.5em 0; padding-left: 1.6em; }
        li { margin: 0.22em 0; }
        hr { border: 0; border-top: 1px solid #d2d2d7; margin: 1.1em 0; }
        a { color: #0066cc; text-decoration: none; }
        img { max-width: 100%; }
        </style></head>
        <body>\(render(markdown))</body></html>
        """
    }

    static func render(_ markdown: String) -> String {
        let normalized = markdown
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        let lines = normalized.components(separatedBy: "\n")
        var html: [String] = []
        var index = 0

        while index < lines.count {
            let line = lines[index].trimmingCharacters(in: .whitespaces)
            if line.isEmpty {
                index += 1
                continue
            }

            if line.hasPrefix("```") {
                var code: [String] = []
                index += 1
                while index < lines.count {
                    let current = lines[index].trimmingCharacters(in: .whitespaces)
                    if current.hasPrefix("```") {
                        index += 1
                        break
                    }
                    code.append(lines[index])
                    index += 1
                }
                html.append("<pre><code>\(escape(code.joined(separator: "\n")))</code></pre>")
                continue
            }

            if isRawHTMLBlock(line) {
                var block: [String] = []
                if containsTableStart(line) {
                    // HTML 表格常跨多行且中间夹空行：一直收集到 </table>，
                    // 否则空行之后的 <tr>/<td> 会被当作正文转义成源码文本。
                    while index < lines.count {
                        let current = lines[index]
                        block.append(current)
                        index += 1
                        if current.range(of: "</table>",
                                         options: .caseInsensitive) != nil {
                            break
                        }
                    }
                } else {
                    while index < lines.count {
                        let current = lines[index]
                        if current.trimmingCharacters(in: .whitespaces).isEmpty { break }
                        block.append(current)
                        index += 1
                    }
                }
                html.append(block.joined(separator: "\n"))
                continue
            }

            if let level = headingLevel(line) {
                let text = String(line.dropFirst(level))
                    .trimmingCharacters(in: .whitespaces)
                html.append("<h\(level)>\(inline(text))</h\(level)>")
                index += 1
                continue
            }

            if isHorizontalRule(line) {
                html.append("<hr>")
                index += 1
                continue
            }

            if line.hasPrefix(">") {
                var quote: [String] = []
                while index < lines.count {
                    let current = lines[index].trimmingCharacters(in: .whitespaces)
                    guard current.hasPrefix(">") else { break }
                    quote.append(String(current.dropFirst())
                        .trimmingCharacters(in: .whitespaces))
                    index += 1
                }
                html.append("<blockquote>\(inline(quote.joined(separator: "\n")))</blockquote>")
                continue
            }

            if isTableStart(lines, at: index) {
                let (table, next) = renderTable(lines, from: index)
                html.append(table)
                index = next
                continue
            }

            if let kind = listKind(line) {
                let (list, next) = renderList(lines, from: index, kind: kind)
                html.append(list)
                index = next
                continue
            }

            var paragraph: [String] = []
            while index < lines.count {
                let current = lines[index].trimmingCharacters(in: .whitespaces)
                if current.isEmpty || isBlockStart(lines, at: index) { break }
                paragraph.append(current)
                index += 1
            }
            if paragraph.isEmpty {
                index += 1
                continue
            }
            html.append("<p>\(inline(paragraph.joined(separator: "\n")))</p>")
        }
        return html.joined(separator: "\n")
    }

    // MARK: 块识别

    private static func headingLevel(_ line: String) -> Int? {
        for level in 1...6 where line.hasPrefix(String(repeating: "#", count: level) + " ") {
            return level
        }
        return nil
    }

    private static func isHorizontalRule(_ line: String) -> Bool {
        let stripped = line.replacingOccurrences(of: " ", with: "")
        return stripped == "---" || stripped == "***" || stripped == "___"
    }

    private static func isRawHTMLBlock(_ line: String) -> Bool {
        // 表格起始标签可能出现在行内（如「结果：<table ...>」），同样按原始 HTML 处理。
        if containsTableStart(line) { return true }
        let lower = line.lowercased()
        for tag in ["<table", "<thead", "<tbody", "<tfoot", "<div", "<p>",
                    "<p ", "<h1", "<h2", "<h3", "<h4", "<h5", "<h6",
                    "<ul", "<ol", "<blockquote", "<pre", "<img"] {
            if lower.hasPrefix(tag) { return true }
        }
        return false
    }

    /// 行内是否出现 HTML 表格起始标签；内联代码里的 `<table>` 示例不算。
    private static func containsTableStart(_ line: String) -> Bool {
        guard let range = line.range(of: "<table", options: .caseInsensitive) else {
            return false
        }
        let prefix = line[..<range.lowerBound]
        return prefix.filter({ $0 == "`" }).count % 2 == 0
    }

    private static func isBlockStart(_ lines: [String], at index: Int) -> Bool {
        let line = lines[index].trimmingCharacters(in: .whitespaces)
        if line.hasPrefix("```") || isRawHTMLBlock(line) { return true }
        if headingLevel(line) != nil || isHorizontalRule(line) { return true }
        if line.hasPrefix(">") || listKind(line) != nil { return true }
        return isTableStart(lines, at: index)
    }

    // MARK: 列表

    private enum ListKind { case ordered, unordered }

    private static func listKind(_ line: String) -> ListKind? {
        if line.range(of: #"^[-*+]\s+"#, options: .regularExpression) != nil {
            return .unordered
        }
        if line.range(of: #"^\d{1,3}[.)]\s+"#, options: .regularExpression) != nil {
            return .ordered
        }
        return nil
    }

    private static func stripListMarker(_ line: String) -> String {
        line.replacingOccurrences(of: #"^[-*+]\s+"#, with: "",
                                  options: .regularExpression)
            .replacingOccurrences(of: #"^\d{1,3}[.)]\s+"#, with: "",
                                  options: .regularExpression)
    }

    private static func renderList(_ lines: [String], from start: Int,
                                   kind: ListKind) -> (String, Int) {
        let tag = kind == .ordered ? "ol" : "ul"
        var items: [String] = []
        var index = start
        while index < lines.count {
            let line = lines[index].trimmingCharacters(in: .whitespaces)
            guard let current = listKind(line), current == kind else { break }
            items.append("<li>\(inline(stripListMarker(line)))</li>")
            index += 1
        }
        return ("<\(tag)>\(items.joined())</\(tag)>", index)
    }

    // MARK: 表格

    private static func isTableStart(_ lines: [String], at index: Int) -> Bool {
        let header = lines[index].trimmingCharacters(in: .whitespaces)
        guard header.contains("|"), index + 1 < lines.count else { return false }
        return isTableSeparator(lines[index + 1].trimmingCharacters(in: .whitespaces))
    }

    private static func isTableSeparator(_ line: String) -> Bool {
        guard line.contains("|") || line.contains("-") else { return false }
        let stripped = line
            .replacingOccurrences(of: "|", with: "")
            .replacingOccurrences(of: ":", with: "")
            .replacingOccurrences(of: "-", with: "")
            .trimmingCharacters(in: .whitespaces)
        return stripped.isEmpty
    }

    private static func cells(_ line: String) -> [String] {
        var value = line.trimmingCharacters(in: .whitespaces)
        if value.hasPrefix("|") { value.removeFirst() }
        if value.hasSuffix("|") { value.removeLast() }
        return value.components(separatedBy: "|")
            .map { $0.trimmingCharacters(in: .whitespaces) }
    }

    private static func renderTable(_ lines: [String], from start: Int) -> (String, Int) {
        let headers = cells(lines[start])
        var index = start + 2
        var rows: [[String]] = []
        while index < lines.count {
            let line = lines[index].trimmingCharacters(in: .whitespaces)
            guard line.contains("|"), !line.isEmpty else { break }
            rows.append(cells(line))
            index += 1
        }
        var html = "<table><thead><tr>"
        html += headers.map { "<th>\(inline($0))</th>" }.joined()
        html += "</tr></thead><tbody>"
        for row in rows {
            html += "<tr>"
            html += row.map { "<td>\(inline($0))</td>" }.joined()
            html += "</tr>"
        }
        html += "</tbody></table>"
        return (html, index)
    }

    // MARK: 内联

    static func inline(_ text: String) -> String {
        var value = escape(text)
        value = replace(#"`([^`]+)`"#, in: value, template: "<code>$1</code>")
        value = replace(#"\*\*([^*]+)\*\*"#, in: value, template: "<strong>$1</strong>")
        value = replace(#"(?<!\*)\*([^*\n]+)\*(?!\*)"#, in: value, template: "<em>$1</em>")
        value = replace(#"\[([^\]]+)\]\((https?://[^)\s]+|mailto:[^)\s]+|#[^)\s]*)\)"#,
                        in: value, template: "<a href=\"$2\">$1</a>")
        return value.replacingOccurrences(of: "\n", with: "<br>")
    }

    private static func replace(_ pattern: String, in text: String,
                                template: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.stringByReplacingMatches(in: text, range: range,
                                               withTemplate: template)
    }

    private static func escape(_ text: String) -> String {
        text.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
    }
}
