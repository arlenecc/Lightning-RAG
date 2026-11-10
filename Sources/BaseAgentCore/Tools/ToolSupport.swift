import Foundation
import SwiftSoup

// MARK: - 工具层错误

/// 工具内部可预期的失败。工具通常把它转成 `ToolResult.fail`，不让异常穿透到
/// registry（与 Python 工具直接返回 ToolResult 的语义一致）。
public enum ToolError: Error, LocalizedError {
    case message(String)

    public var errorDescription: String? {
        switch self {
        case .message(let text): return text
        }
    }

    public var message: String {
        switch self {
        case .message(let text): return text
        }
    }
}

// MARK: - 公共辅助

/// 基础工具层共享的辅助函数：工作区路径安全解析、参数校验、体积/超时限制、
/// head+tail 截断、受限进程执行、HTML/Markdown 文本化与错误包装。
public enum ToolSupport {

    // MARK: 限制常量（对齐 Python tools）

    /// file_read 单次返回的最大字符数。
    public static let maxReadChars = 200_000
    /// file_write 单次写入的最大字符数（读取上限的 5 倍）。
    public static let maxWriteChars = 5 * maxReadChars
    /// file_modify 允许整文件读入的最大字节数。
    public static let maxModifyBytes = 10 * 1024 * 1024
    /// shell_run/code_run 单路 stdout/stderr 的字符上限。
    public static let maxOutputChars = 200_000
    /// 进程超时的上下限（秒）。
    public static let minTimeout: Double = 1
    public static let maxTimeout: Double = 600
    /// web_scan 单次抓取的最大字节数。
    public static let maxFetchBytes = 8 * 1024 * 1024
    /// 交给 HTML 解析器的最大字符数。
    public static let maxParseChars = 2 * 1024 * 1024
    /// 浏览器 UA：不少 CDN（Fastly 等）对空/非浏览器 UA 直接 403/404，
    /// URLSession 默认不带 UA，必须在所有 web 请求上显式设置。
    public static let browserUserAgent =
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        + "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"

    // MARK: JSON Schema 构造

    /// 构造工具参数 JSON Schema（type=object）。
    public static func objectSchema(properties: [String: JSONValue],
                                    required: [String] = []) -> JSONValue {
        .object([
            "type": .string("object"),
            "properties": .object(properties),
            "required": .array(required.map { .string($0) }),
        ])
    }

    /// `{"type":"string","description":...}` 参数项。
    public static func stringProperty(_ description: String,
                                      enumValues: [String]? = nil) -> JSONValue {
        var object: [String: JSONValue] = [
            "type": .string("string"),
            "description": .string(description),
        ]
        if let enumValues {
            object["enum"] = .array(enumValues.map { .string($0) })
        }
        return .object(object)
    }

    /// `{"type":"boolean","description":...}` 参数项。
    public static func boolProperty(_ description: String) -> JSONValue {
        .object([
            "type": .string("boolean"),
            "description": .string(description),
        ])
    }

    /// `{"type":"number","description":...}` 参数项。
    public static func numberProperty(_ description: String) -> JSONValue {
        .object([
            "type": .string("number"),
            "description": .string(description),
        ])
    }

    /// `{"type":"object","description":...}` 参数项（自由键值对，如 env 变量表）。
    public static func objectProperty(_ description: String) -> JSONValue {
        .object([
            "type": .string("object"),
            "description": .string(description),
        ])
    }

    /// `{"type":"array","description":...,"items":...}` 参数项。
    public static func arrayProperty(_ description: String,
                                     items: JSONValue) -> JSONValue {
        .object([
            "type": .string("array"),
            "description": .string(description),
            "items": items,
        ])
    }

    // MARK: 参数读取与校验

    /// 读取必填字符串参数；缺失/null 抛 `ToolError`。
    /// 非字符串标量按 Python `str()` 语义强制转换，容器类型转成紧凑 JSON。
    public static func requireString(_ args: [String: JSONValue],
                                     _ key: String) throws -> String {
        guard let value = args[key], !value.isNull else {
            throw ToolError.message("missing required argument: \(key)")
        }
        return coerceString(value)
    }

    /// 读取可选字符串参数；缺失/null 返回 nil。
    public static func optionalString(_ args: [String: JSONValue],
                                      _ key: String) -> String? {
        guard let value = args[key], !value.isNull else { return nil }
        return coerceString(value)
    }

    /// 读取字符串参数，缺失/null 返回默认值。
    public static func string(_ args: [String: JSONValue],
                              _ key: String,
                              default defaultValue: String = "") -> String {
        optionalString(args, key) ?? defaultValue
    }

    public static func optionalBool(_ args: [String: JSONValue],
                                    _ key: String) -> Bool? {
        guard let value = args[key] else { return nil }
        return value.boolValue
    }

    public static func bool(_ args: [String: JSONValue],
                            _ key: String,
                            default defaultValue: Bool = false) -> Bool {
        optionalBool(args, key) ?? defaultValue
    }

    public static func optionalDouble(_ args: [String: JSONValue],
                                      _ key: String) -> Double? {
        guard let value = args[key] else { return nil }
        return value.doubleValue
    }

    public static func double(_ args: [String: JSONValue],
                              _ key: String,
                              default defaultValue: Double) -> Double {
        optionalDouble(args, key) ?? defaultValue
    }

    public static func optionalInt(_ args: [String: JSONValue],
                                   _ key: String) -> Int? {
        guard let value = args[key] else { return nil }
        return value.intValue
    }

    /// 与 Python `str(value)` 等价的宽松转换。
    public static func coerceString(_ value: JSONValue) -> String {
        switch value {
        case .string(let s): return s
        case .int(let i): return String(i)
        case .double(let d): return String(d)
        case .bool(let b): return b ? "true" : "false"
        case .null: return ""
        case .array, .object: return value.jsonString()
        }
    }

    /// 把超时钳制到 [1, 600] 秒；畸形值回退默认值。
    public static func clampTimeout(_ value: Double?, default defaultValue: Double) -> Double {
        let candidate = value ?? defaultValue
        guard candidate.isFinite else { return defaultValue }
        return max(minTimeout, min(candidate, maxTimeout))
    }

    // MARK: 错误包装

    /// 把任意错误转成给模型看的稳定文案。
    public static func errorText(_ error: Error) -> String {
        if let toolError = error as? ToolError {
            return toolError.message
        }
        if let agentError = error as? AgentError {
            return agentError.errorDescription ?? agentError.localizedDescription
        }
        let nsError = error as NSError
        return "\(nsError.domain)(\(nsError.code)): \(nsError.localizedDescription)"
    }

    /// 统一把错误包装成失败结果。
    public static func failure(_ error: Error) -> ToolResult {
        .fail(errorText(error))
    }

    // MARK: 工作区路径安全

    /// 解析路径并锁定在 workspace 内。
    ///
    /// - 相对路径按 workspace 拼接；绝对路径必须落在 workspace 内。
    /// - 使用 `resolvingSymlinksInPath` 解析符号链接后再做前缀比对，避免
    ///   workspace 内的 symlink 指向外部目录时绕过检查。
    /// - 越界抛 `ToolError`（文案与 Python 一致）。
    public static func resolveWorkspacePath(_ path: String,
                                            workspace: String) throws -> String {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw ToolError.message("path is required")
        }
        let ws = realPath(workspace)
        let candidate: String
        if trimmed.hasPrefix("/") {
            candidate = trimmed
        } else {
            candidate = (ws as NSString).appendingPathComponent(trimmed)
        }
        let full = realPath(candidate)
        guard full == ws || full.hasPrefix(ws + "/") else {
            throw ToolError.message("Path '\(path)' is outside the workspace")
        }
        return full
    }

    /// 解析符号链接并标准化路径（等价 Python `os.path.realpath`）。
    ///
    /// `URL.resolvingSymlinksInPath()` 在**尾部组件不存在**时不会展开中间的
    /// symlink：`workspace/link/evil.txt` 会原样返回，看起来仍在 workspace 内，
    /// 但随后由 OS 落盘时却写到了 link 指向的外部目录。这里先上溯到真实存在
    /// 的祖先目录并 realpath，再逐段拼回缺失组件、每段重新解析 symlink。
    public static func realPath(_ path: String) -> String {
        let fm = FileManager.default
        var url = URL(fileURLWithPath: path).standardizedFileURL
        var missing: [String] = []
        while !fm.fileExists(atPath: url.path) {
            let parent = url.deletingLastPathComponent()
            if parent.path == url.path { break }
            missing.append(url.lastPathComponent)
            url = parent
        }
        var resolved = url.resolvingSymlinksInPath().standardizedFileURL
        for component in missing.reversed() {
            resolved.appendPathComponent(component)
            resolved = resolved.resolvingSymlinksInPath().standardizedFileURL
        }
        return resolved.path
    }

    /// workspace/.agent/<name> 状态文件路径。
    public static func stateFile(_ name: String, workspace: String) -> String {
        URL(fileURLWithPath: workspace)
            .appendingPathComponent(".agent", isDirectory: true)
            .appendingPathComponent(name)
            .path
    }

    public static func fileSize(_ path: String) -> Int {
        let attributes = try? FileManager.default.attributesOfItem(atPath: path)
        return (attributes?[.size] as? NSNumber)?.intValue ?? 0
    }

    // MARK: 多级记忆写保护

    /// `memory/` 根目录（realpath 归一化）。L0/L1 会被逐字拼进 system prompt，
    /// L2/L3/L4 由记忆引擎维护 —— 通用文件工具不得静默改写它们，否则一次
    /// 提示词注入就能把指令持久化到后续所有会话。
    public static func managedMemoryRoot(workspace: String) -> String {
        realPath((workspace as NSString).appendingPathComponent("memory"))
    }

    /// 是否落在 `<workspace>/memory/` 目录树内（含 symlink 解析后的越界排除）。
    public static func isManagedMemoryPath(_ fullPath: String, workspace: String) -> Bool {
        let root = managedMemoryRoot(workspace: workspace)
        let target = realPath(fullPath)
        return target == root || target.hasPrefix(root + "/")
    }

    /// 会被逐字注入 system prompt 的两个文件：L0 META-SOP 与 L1 索引。
    public static func isSystemPromptMemoryFile(_ fullPath: String, workspace: String) -> Bool {
        guard isManagedMemoryPath(fullPath, workspace: workspace) else { return false }
        let protected: Set<String> = ["memory_management_sop.md", "global_mem_insight.txt"]
        return protected.contains((fullPath as NSString).lastPathComponent)
    }

    /// `file_tools(op="write")` 写入前守卫。协议只允许 `file_tools(op="patch")` 少量追加，故：
    /// - L0/L1 一律不允许覆盖/重建（无论文件当前是否存在）；
    /// - `memory/` 内已存在的其它文件不允许覆盖；
    /// - 新建 L3 SOP 等文件允许，但由 `shouldConfirm` 强制人工复核。
    public static func guardMemoryWrite(_ fullPath: String, workspace: String) throws {
        guard isManagedMemoryPath(fullPath, workspace: workspace) else { return }
        if isSystemPromptMemoryFile(fullPath, workspace: workspace) {
            throw ToolError.message(
                "拒绝用 file_tools(op=\"write\") 写 `memory/` 的 L0/L1 文件：它们直接注入 system prompt。"
                + "若确需修改，请用 file_tools(op=\"patch\") 做词级修改（会请求用户确认）。")
        }
        if FileManager.default.fileExists(atPath: fullPath) {
            throw ToolError.message(
                "拒绝用 file_tools(op=\"write\") 覆盖 `memory/` 内已有文件；按 META-SOP 只能 "
                + "用 file_tools(op=\"patch\") 少量追加（会请求用户确认）。")
        }
    }

    // MARK: 工具确认（所见即所执行）

    /// 确认弹窗单条参数上限。超限时保留头尾，绝不把危险尾部藏起来。
    public static let maxConfirmChars = 12_000

    /// 构造人工确认文案。
    ///
    /// 用户看到的内容必须与实际传给 `tool.run(args)` 的参数一致。旧实现只显示
    /// 「首行注释/程序名摘要」，模型（或被网页/文档注入的模型）能把真正的危险
    /// 命令藏在后续行里 —— 用户批准 A、实际执行 B。这里改为逐行引用完整原文，
    /// 每行加 `| ` 前缀使内容里伪造的 UI 文本一眼可辨；超长时头尾截断。
    public static func confirmationMessage(name: String,
                                           args: [String: JSONValue]) -> String {
        let executionKey: String? = name == "code_run" ? "code"
            : (name == "shell_run" ? "command" : nil)
        let body: String
        if let executionKey {
            body = optionalString(args, executionKey) ?? ""
        } else {
            body = JSONValue.object(args).jsonString(pretty: true)
        }

        var message = "即将调用工具 `\(name)`，以下是实际执行的完整参数：\n"
        if executionKey != nil, let summary = executionSummary(body) {
            message += "摘要（取自内容本身，不可作为审批依据）：\(summary)\n"
        }
        if body.isEmpty { message += "（参数为空）\n" }
        message += quotedForConfirmation(body)

        if let timeout = optionalDouble(args, "timeout") {
            message += "\n超时：\(timeout)s"
        }
        if let executionKey {
            var others: [String: JSONValue] = [:]
            for (key, value) in args where key != executionKey && key != "timeout" {
                others[key] = value
            }
            if !others.isEmpty {
                message += "\n其它参数：" + JSONValue.object(others).jsonString()
            }
        }
        return message + "\n\n是否继续？"
    }

    private static func quotedForConfirmation(_ text: String) -> String {
        let truncated = text.count > maxConfirmChars
        let shown = truncated ? truncateHeadTail(text, limit: maxConfirmChars) : text
        let quoted = shown.split(separator: "\n", omittingEmptySubsequences: false)
            .map { "| " + $0 }
            .joined(separator: "\n")
        let warning = truncated
            ? "⚠ 参数共 \(text.count) 字符，超出确认框上限；下面只显示首尾片段。\n"
            : ""
        return warning + "----- 实际参数开始 -----\n"
            + quoted + "\n----- 实际参数结束 -----"
    }

    /// 仅用于标题提示，绝不能替代上面的完整原文。
    private static func executionSummary(_ text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "（空）" }
        if !trimmed.contains("\n") {
            let program = trimmed.split(separator: " ").first.map(String.init) ?? ""
            let known: [String: String] = [
                "pip": "安装/管理 Python 包", "npm": "安装/管理 Node 包",
                "brew": "安装/管理 macOS 软件", "git": "Git 版本控制操作",
                "curl": "发起网络请求", "wget": "下载文件",
                "python": "运行 Python 脚本", "python3": "运行 Python 脚本",
                "node": "运行 Node 脚本", "rm": "删除文件/目录",
                "mv": "移动/重命名文件", "cp": "复制文件",
                "docker": "Docker 容器操作", "ssh": "SSH 远程连接",
            ]
            if let label = known[program] { return label }
        }
        for line in trimmed.split(separator: "\n") {
            let value = line.trimmingCharacters(in: .whitespaces)
            if value.hasPrefix("#") {
                let comment = value.drop(while: { $0 == "#" }).trimmingCharacters(in: .whitespaces)
                if !comment.isEmpty, !comment.hasPrefix("!") { return String(comment.prefix(80)) }
            }
            if value.hasPrefix("\"\"\"") || value.hasPrefix("'''") {
                let doc = value.trimmingCharacters(in: CharacterSet(charactersIn: "\"' "))
                if !doc.isEmpty { return String(doc.prefix(80)) }
            }
        }
        let first = trimmed.split(separator: "\n").first.map(String.init) ?? trimmed
        return "执行代码片段：" + String(first.prefix(60))
    }

    // MARK: 输出截断

    /// 保留头 + 尾的截断：超长输出里最有价值的信息往往在末尾（报错摘要）。
    public static func truncateHeadTail(_ text: String,
                                        limit: Int = maxOutputChars,
                                        tailRatio: Double = 0.25) -> String {
        let count = text.count
        guard count > limit else { return text }
        let tailBudget = Int(Double(limit) * tailRatio)
        let headBudget = max(0, limit - tailBudget)
        return String(text.prefix(headBudget))
            + "\n… [输出超过 \(limit) 字符，中间已截断] …\n"
            + String(text.suffix(tailBudget))
    }

    /// 按行去空白并丢弃空行（web 正文清洗用）。
    public static func normalizeLines(_ text: String) -> String {
        text.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    /// 合并 stdout/stderr（与 Python `_combine` 格式一致）。
    public static func combineOutput(stdout: String, stderr: String) -> String {
        guard !stderr.isEmpty else { return stdout }
        return stdout + (stdout.isEmpty ? "" : "\n") + "[stderr]\n" + stderr
    }

    // MARK: UTF-8 解码

    /// 解码 UTF-8 前缀；若末尾截断了多字节字符，最多回退 3 字节再试。
    /// 返回 nil 表示内容不是合法 UTF-8（按二进制处理）。
    public static func decodeUTF8Prefix(_ data: Data) -> String? {
        if let text = String(data: data, encoding: .utf8) {
            return text
        }
        guard data.count > 1 else { return nil }
        for drop in 1...min(3, data.count - 1) {
            let trimmed = Data(data.dropLast(drop))
            if let text = String(data: trimmed, encoding: .utf8) {
                return text
            }
        }
        return nil
    }

    // MARK: 进程环境

    /// 为子进程准备一份干净的环境：去掉会污染解释器启动的变量，
    /// 固定 UTF-8 输出，关闭颜色/分页交互。
    public static func sanitizedEnvironment(overrides: [String: String] = [:]) -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        env.removeValue(forKey: "PYTHONSTARTUP")
        env.removeValue(forKey: "PYTHONHOME")
        env.removeValue(forKey: "PYTHONBREAKPOINT")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env["NO_COLOR"] = "1"
        env["TERM"] = env["TERM"] ?? "dumb"
        if env["LANG"] == nil { env["LANG"] = "en_US.UTF-8" }
        for (key, value) in overrides { env[key] = value }
        return env
    }

    // MARK: HTML / Markdown 文本化

    /// SwiftSoup 提取网页可见正文：移除脚本/样式等噪音标签，按内联样式与
    /// `aria-hidden` 启发式剔除隐藏子树，按块级元素换行，并把表单控件标注成
    /// `[input #id name= type= "placeholder"]` 形式——参考 simphtml.py 的可见性
    /// 与表单态保留思路，但只在静态解析能可靠判断的范围内做（真实布局可见性
    /// 交给 WebKit 渲染路径负责）。
    public static func htmlToText(_ html: String, maxChars: Int = maxParseChars) -> String {
        // 超长页面（如 cnn.com 首页 5.9MB）的开头是巨大的 <head>/预加载清单，
        // 直接按字符截断会把正文体整体截出解析窗口、提取结果为空，进而被
        // 误判为「动态加载」再花 20s+ 渲染兜底。跳过 head 从 <body> 起取窗口。
        var window = String(html.prefix(maxChars))
        if html.count > maxChars,
           let bodyRange = html.range(of: "<body", options: .caseInsensitive) {
            window = String(html[bodyRange.lowerBound...].prefix(maxChars))
        }
        guard let document = try? SwiftSoup.parse(window) else {
            return normalizeLines(stripTags(window))
        }
        pruneHTMLDocument(document)

        var lines: [String] = []
        var current = ""
        if let body = document.body() {
            collectHTMLText(body, lines: &lines, current: &current)
        } else {
            collectHTMLText(document, lines: &lines, current: &current)
        }
        flushHTMLText(&current, lines: &lines)

        let text = normalizeLines(lines.joined(separator: "\n"))
        if !text.isEmpty { return text }
        return normalizeLines((try? document.text()) ?? stripTags(window))
    }

    /// 解析前的 DOM 裁剪：移除噪音标签与隐藏子树，让正文提取少背垃圾。
    /// - 噪音标签：script/style/noscript/meta/link/template/colgroup/col/param/
    ///   source/head/title/svg（svg 内联路径/图形对文本提取无价值）；
    /// - 隐藏：`aria-hidden="true"`，或内联 `style` 里 `display:none` /
    ///   `visibility:hidden` / `opacity:0`（静态解析无法算布局，只认内联声明）。
    private static func pruneHTMLDocument(_ doc: Document) {
        let _ = try? doc.select(
            "script, style, noscript, meta, link, template, colgroup, col, param, source, "
            + "head, title, svg").remove()
        if let hidden = try? doc.select("[aria-hidden='true']") {
            for el in hidden { let _ = try? el.remove() }
        }
        // 内联隐藏：先收集候选再逐个移除（移除父节点会让子节点脱离，但
        // SwiftSoup 的 Elements 是独立快照，重复 remove 是安全的 no-op）。
        guard let all = try? doc.select("*") else { return }
        for el in all {
            guard let style = try? el.attr("style"), !style.isEmpty else { continue }
            let s = style.lowercased().replacingOccurrences(of: " ", with: "")
            if s.contains("display:none") || s.contains("visibility:hidden") {
                let _ = try? el.remove()
                continue
            }
            if let m = s.range(of: #"opacity:([0-9]*\.?[0-9]+)"#, options: .regularExpression),
               let value = Double(s[m].replacingOccurrences(of: "opacity", with: "")
                                        .replacingOccurrences(of: ":", with: "")
                                        .replacingOccurrences(of: " ", with: "")),
               value <= 0 {
                let _ = try? el.remove()
            }
        }
    }

    /// 列表页 / SPA 动态加载检测：提取正文过短，或正文虽长但主体只是加载占位时，
    /// 返回降级提示（分析数据接口直接调用 / 内置 WebKit 渲染 / 交互脚本）。
    /// 避免模型把「加载占位」误当有效内容或直接断定站点无信息。
    ///
    /// 触发条件（两者任一）：
    /// - 正文短（<300 字符）且存在 AJAX/前端框架/加载占位/iframe 特征；
    /// - 正文长但含加载占位文案，且同时存在 loader / AJAX / 框架挂载点——
    ///   招聘站等页面常靠导航/页脚把正文撑到几百字符，只看长度会漏判。
    ///
    /// 同时附上从页面和已抓取的外部 JS 里提取的数据源线索（候选接口、JS 引用、
    /// iframe），让模型少一轮摸索即可定位真正的数据来源。
    static func dynamicLoadingHint(html: String,
                                   extracted: String,
                                   externalScriptTexts: [String] = [],
                                   inspectedScriptURLs: [String] = []) -> String {
        let visible = extracted.trimmingCharacters(in: .whitespacesAndNewlines)
        let shortBody = visible.count < 300

        let lowered = html.lowercased()
        var signals: [String] = []
        // 前端框架挂载点：body 里只有挂载容器，正文必然是空壳。
        let hasFrameworkMount = lowered.contains("id=\"root\"") || lowered.contains("id=\"app\"")
            || lowered.contains("id=\"__next\"") || lowered.contains("data-reactroot")
            || lowered.contains("__nuxt__") || lowered.contains("ng-app")
        if hasFrameworkMount {
            signals.append("前端框架挂载点")
        }
        // 加载占位元素：数据由 AJAX 填充的典型标志。
        let hasLoaderElement = lowered.range(
            of: "class=\"[^\"]*(loader|loading|skeleton|spinner)",
            options: .regularExpression) != nil
        if hasLoaderElement {
            signals.append("加载占位元素（loader/skeleton/spinner）")
        }
        // 页面内 AJAX 调用代码。
        let hasAjaxCode = lowered.contains("xmlhttprequest") || lowered.contains("$.ajax")
            || lowered.contains("axios") || lowered.contains("fetch(")
        if hasAjaxCode {
            signals.append("AJAX 请求代码")
        }
        // 正文为空但内容在 iframe 里（主文档只是壳）。
        let iframes = extractIframeSources(html: html)

        // 正文里出现「Loading jobs / please wait / 加载中」等占位文案。
        // 长正文只靠这一条不足以判定（静态文章也可能写 please wait），
        // 必须同时有 loader 元素 / AJAX 代码 / 框架挂载点。
        let hasPlaceholderText = visible.range(
            of: #"loading\s+(jobs|data|results|content)|please\s+wait|loading(\.\.\.|…)|正在加载|加载中"#,
            options: [.regularExpression, .caseInsensitive]) != nil
        let longBodyDynamic = !shortBody && hasPlaceholderText
            && (hasLoaderElement || hasAjaxCode || hasFrameworkMount)
        if longBodyDynamic && !hasLoaderElement {
            signals.append("加载占位文案（loading/please wait）")
        }

        let shortBodyDynamic = shortBody && (!signals.isEmpty || !iframes.isEmpty)
        guard shortBodyDynamic || longBodyDynamic else { return "" }

        // 合并 HTML 内联接口与外部 JS 里的接口线索：先去重，再按接口优先级全局排序，
        // 避免 HTML 里的噪声先入为主、把外部 JS 中真正的 /api/ 端点挤掉。
        var merged: [(url: String, score: Int, order: Int)] = []
        var seenEndpoints = Set<String>()
        for source in [html] + externalScriptTexts {
            for endpoint in extractAPIEndpointHints(html: source, limit: 8) {
                guard seenEndpoints.insert(endpoint).inserted else { continue }
                merged.append((endpoint, apiEndpointPriority(endpoint), merged.count))
            }
        }
        let apiHints = merged
            .sorted { $0.score == $1.score ? $0.order < $1.order : $0.score > $1.score }
            .prefix(8)
            .map(\.url)

        var lines: [String] = []
        if !apiHints.isEmpty {
            lines.append("候选数据接口：\(apiHints.joined(separator: ", "))")
        }
        let scripts = extractScriptSources(html: html)
        if !scripts.isEmpty {
            lines.append("页面 JS 引用：\(scripts.joined(separator: ", "))")
        }
        if !iframes.isEmpty {
            lines.append("内嵌 iframe：\(iframes.joined(separator: ", "))")
        }
        if !inspectedScriptURLs.isEmpty {
            lines.append("已扫描 JS：\(inspectedScriptURLs.joined(separator: ", "))")
        }

        let summary = shortBody
            ? "本页提取正文仅 \(visible.count) 字符"
            : "本页提取正文 \(visible.count) 字符但主要是导航/页脚/静态文案"
        var hint = "\n\n[web_scan 提示] \(summary)，且检测到动态加载"
            + "特征（\(signals.joined(separator: "、"))）——页面主体内容很可能由 JS 动态加载，"
            + "静态 HTML 只能拿到加载占位。web_scan 已自动渲染并直取页面数据接口，结果见上方"
            + "数据段；若仍缺信息或需要翻页，用 shell_run/code_run 按数据段给出的分页参数"
            + "显式调用（不要对未知 POST 接口盲发写请求）；若需要点击/填表/滚动加载等交互，"
            + "用 web_scan(op=\"render\", script=...) 执行 JS。"
        if !lines.isEmpty {
            hint += "\n已定位的数据源线索：\n" + lines.joined(separator: "\n")
        }
        return hint
    }

    /// 提取页面内 iframe 的 src（按出现顺序去重，含相对路径原文）。
    static func extractIframeSources(html: String, limit: Int = 5) -> [String] {
        guard let document = try? SwiftSoup.parse(html),
              let frames = try? document.select("iframe[src]") else { return [] }
        var seen = Set<String>()
        var urls: [String] = []
        for frame in frames.array() {
            guard urls.count < limit else { break }
            guard let src = try? frame.attr("src"), !src.isEmpty else { continue }
            let trimmed = src.trimmingCharacters(in: .whitespaces)
            if seen.insert(trimmed).inserted { urls.append(trimmed) }
        }
        return urls
    }

    /// 提取页面 script[src] 引用（按出现顺序去重），供降级提示定位数据加载逻辑。
    static func extractScriptSources(html: String, limit: Int = 5) -> [String] {
        guard let document = try? SwiftSoup.parse(html),
              let scripts = try? document.select("script[src]") else { return [] }
        var seen = Set<String>()
        var urls: [String] = []
        for script in scripts.array() {
            guard urls.count < limit else { break }
            guard let src = try? script.attr("src"), !src.isEmpty else { continue }
            if seen.insert(src).inserted { urls.append(src) }
        }
        return urls
    }

    /// 挑选值得抓取分析的同源脚本：数据接口通常写在 search/settings/config 类脚本里，
    /// 而 jQuery/Bootstrap/统计/字体等基础设施脚本只增加流量。
    ///
    /// 规则：解析为绝对 URL 后只保留与页面同源的 http(s) 脚本，过滤已知基础设施库，
    /// 再按路径关键词评分降序排列（同分保持页面出现顺序），返回前 `limit` 个。
    static func selectScriptSourcesForInspection(html: String,
                                                 baseURL: URL,
                                                 limit: Int = 5) -> [URL] {
        guard limit > 0,
              let document = try? SwiftSoup.parse(html),
              let scripts = try? document.select("script[src]") else { return [] }
        let baseHost = baseURL.host?.lowercased() ?? ""

        var seen = Set<String>()
        var candidates: [(url: URL, score: Int, order: Int)] = []
        for script in scripts.array() {
            guard let raw = try? script.attr("src") else { continue }
            let src = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !src.isEmpty,
                  let resolved = URL(string: src, relativeTo: baseURL)?.absoluteURL,
                  let scheme = resolved.scheme?.lowercased(),
                  scheme == "http" || scheme == "https",
                  let host = resolved.host?.lowercased(),
                  host == baseHost,
                  !isInfrastructureScript(resolved) else { continue }
            guard seen.insert(resolved.absoluteString).inserted else { continue }
            candidates.append((resolved, scriptInspectionScore(resolved), candidates.count))
        }
        return candidates
            .sorted { $0.score == $1.score ? $0.order < $1.order : $0.score > $1.score }
            .prefix(limit)
            .map(\.url)
    }

    /// 基础设施/统计类脚本：对定位数据接口没有价值，抓取只会浪费流量。
    private static func isInfrastructureScript(_ url: URL) -> Bool {
        let lowered = url.absoluteString.lowercased()
        let blocked = ["jquery", "bootstrap", "webfont", "polyfill", "analytics",
                       "gtag", "doubleclick", "googletagmanager", "localize", "cdn"]
        return blocked.contains { lowered.contains($0) }
    }

    /// 脚本路径关键词评分：搜索/岗位/接口类脚本最可能藏有真实数据端点。
    private static func scriptInspectionScore(_ url: URL) -> Int {
        let path = url.path.lowercased()
        let weighted: [(String, Int)] = [
            ("search", 6), ("settings", 4), ("job", 4), ("api", 4),
            ("config", 3), ("data", 2), ("index", 2), ("bundle", 1), ("app", 1),
        ]
        return weighted.reduce(0) { score, item in
            path.contains(item.0) ? score + item.1 : score
        }
    }

    /// 从页面源码提取候选数据接口路径（含 api/ajax/ashx/asmx/search/query/json/graphql
    /// 特征的站内 URL），供降级提示直接给出可调用的端点。
    ///
    /// 会过滤静态资源误报：`jquery` 里的 "query"、`.css/.js/图片/字体` 引用都不是接口；
    /// 结果按「/api/ 优先，其次 ajax/ashx/search」排序，避免页面资源噪声挤掉真正的接口。
    static func extractAPIEndpointHints(html: String, limit: Int = 5) -> [String] {
        guard limit > 0 else { return [] }
        // 先匹配所有引号包裹的站内绝对路径，再做关键词/静态资源过滤。
        let pattern = "[\"'](/[A-Za-z0-9_./?=&%*,~-]+)[\"']"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { return [] }
        let range = NSRange(html.startIndex..., in: html)
        var seen = Set<String>()
        var hits: [(url: String, score: Int, order: Int)] = []
        var scanned = 0
        for match in regex.matches(in: html, range: range) {
            scanned += 1
            guard scanned <= 500 else { break }
            guard let r = Range(match.range(at: 1), in: html) else { continue }
            let url = String(html[r])
            guard isDataEndpointCandidate(url) else { continue }
            guard seen.insert(url).inserted else { continue }
            hits.append((url, apiEndpointPriority(url), hits.count))
        }
        return hits
            .sorted { $0.score == $1.score ? $0.order < $1.order : $0.score > $1.score }
            .prefix(limit)
            .map(\.url)
    }

    /// 站内路径是否像数据接口：排除 jquery/静态资源，并要求含接口特征关键词。
    private static func isDataEndpointCandidate(_ url: String) -> Bool {
        let lowered = url.lowercased()
        if lowered.contains("jquery") { return false }
        let path = lowered.split(separator: "?", maxSplits: 1,
                                  omittingEmptySubsequences: false)
            .first.map(String.init) ?? lowered
        let assetSuffixes = [".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg",
                             ".ico", ".woff", ".woff2", ".ttf", ".eot", ".map"]
        if assetSuffixes.contains(where: { path.hasSuffix($0) }) { return false }
        let keywords = ["api", "ajax", "ashx", "asmx", "search", "query", "json", "graphql"]
        return keywords.contains { lowered.contains($0) }
    }

    /// 接口优先级：/api/ 最高，其次 ajax/ashx/asmx/search 等明确的数据端点。
    private static func apiEndpointPriority(_ url: String) -> Int {
        let lowered = url.lowercased()
        var score = 0
        if lowered.contains("/api/") || lowered.hasPrefix("/api") { score += 10 }
        if lowered.contains("ajax") { score += 6 }
        if lowered.contains("ashx") || lowered.contains("asmx") { score += 5 }
        if lowered.contains("/search") || lowered.contains("getsearch") { score += 4 }
        if lowered.contains("query") { score += 3 }
        if lowered.contains("graphql") { score += 3 }
        if lowered.contains("json") { score += 2 }
        if lowered.contains("data") { score += 1 }
        return score
    }

    private static func collectHTMLText(_ node: Node,
                                        lines: inout [String],
                                        current: inout String) {
        if let textNode = node as? TextNode {
            current += textNode.text()
            return
        }
        guard let element = node as? Element else { return }
        let tag = element.tagName().lowercased()
        // 已在 pruneHTMLDocument 里清掉；这里再兜底一次，避免遗漏。
        if ["script", "style", "noscript", "meta", "link", "template",
            "colgroup", "col", "param", "source", "head", "title", "svg"].contains(tag) {
            return
        }
        if tag == "br" {
            current += "\n"
            return
        }
        if tag == "hr" {
            current += "\n----\n"
            return
        }
        // 表单控件：标注成单行 `[input #id name= type= "placeholder"]`，
        // 把运行态相关信息（勾选/选中值）保留给模型，而不是整段丢掉。
        if tag == "input" || tag == "textarea" || tag == "select" {
            if let annotation = formControlAnnotation(element) {
                flushHTMLText(&current, lines: &lines)
                lines.append(annotation)
            }
            if tag == "textarea",
               let value = try? element.text(), !value.isEmpty {
                lines.append(value)
            }
            return
        }
        let isBlock = htmlBlockTags.contains(tag)
        if isBlock { flushHTMLText(&current, lines: &lines) }
        for child in element.getChildNodes() {
            collectHTMLText(child, lines: &lines, current: &current)
        }
        if isBlock { flushHTMLText(&current, lines: &lines) }
    }

    /// 把表单控件压成一行标注；返回 nil 表示该控件无需输出（如隐藏域 type=hidden）。
    private static func formControlAnnotation(_ el: Element) -> String? {
        let tag = el.tagName().lowercased()
        // 隐藏域（CSRF token 等）对用户不可见、对模型无意义，直接跳过。
        let type = (tag == "input") ? ((try? el.attr("type"))?.lowercased() ?? "text") : nil
        if tag == "input", type == "hidden" { return nil }
        var parts: [String] = [tag]
        if let id = try? el.id(), !id.isEmpty { parts.append("#\(id)") }
        if let name = try? el.attr("name"), !name.isEmpty { parts.append("name=\(name)") }
        if tag == "input", let type {
            parts.append("type=\(type)")
        }
        if let ph = try? el.attr("placeholder"), !ph.isEmpty { parts.append("\"\(ph)\"") }
        if let disabled = try? el.attr("disabled"), !disabled.isEmpty {
            parts.append("disabled")
        }
        if tag == "select" {
            if let sel = try? el.attr("data-selected"), !sel.isEmpty {
                parts.append("=\"\(sel)\"")
            } else if let opts = try? el.select("option") {
                let labels = opts.compactMap { try? $0.text() }
                    .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
                    .prefix(12)
                if !labels.isEmpty {
                    parts.append("options=[\(labels.joined(separator: ", "))]")
                }
            }
        }
        if tag == "input",
           let type, (type == "checkbox" || type == "radio"),
           el.hasAttr("checked"),
           (try? el.attr("checked"))?.lowercased() != "false" {
            parts.append("checked")
        }
        return "[\(parts.joined(separator: " "))]"
    }

    private static func flushHTMLText(_ current: inout String, lines: inout [String]) {
        let collapsed = current.replacingOccurrences(of: #"[ \t\r\n]+"#,
                                                     with: " ",
                                                     options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !collapsed.isEmpty { lines.append(collapsed) }
        current = ""
    }

    private static let htmlBlockTags: Set<String> = [
        "p", "div", "section", "article", "header", "footer", "aside", "main", "nav",
        "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "dl", "dt", "dd",
        "table", "thead", "tbody", "tfoot", "tr", "td", "th", "pre", "blockquote",
        "figure", "figcaption", "form", "fieldset", "address",
        "details", "summary", "hr",
    ]

    /// 极简标签剥离（SwiftSoup 解析失败时的兜底）。
    private static func stripTags(_ html: String) -> String {
        var text = html
        for pattern in [#"<script.*?</script>"#, #"<style.*?</style>"#] {
            text = text.replacingOccurrences(of: pattern, with: " ",
                                             options: [.regularExpression, .caseInsensitive])
        }
        text = text.replacingOccurrences(of: #"<[^>]+>"#, with: " ",
                                         options: .regularExpression)
        return text
    }

    /// 去掉常见的 Markdown 语法标记，得到可直接阅读的纯文本。
    public static func markdownToPlainText(_ markdown: String) -> String {
        var text = markdown
        text = text.replacingOccurrences(of: #"(?m)^#{1,6}\s+"#, with: "",
                                         options: .regularExpression)
        text = text.replacingOccurrences(of: #"!\[([^\]]*)\]\([^)]*\)"#, with: "$1",
                                         options: .regularExpression)
        text = text.replacingOccurrences(of: #"\[([^\]]+)\]\([^)]*\)"#, with: "$1",
                                         options: .regularExpression)
        text = text.replacingOccurrences(of: #"[*_]{1,3}([^*_\n]+)[*_]{1,3}"#, with: "$1",
                                         options: .regularExpression)
        text = text.replacingOccurrences(of: #"`([^`]+)`"#, with: "$1",
                                         options: .regularExpression)
        text = text.replacingOccurrences(of: "\n{3,}", with: "\n\n",
                                         options: .regularExpression)
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - 有界输出收集器

/// 线程安全的有界输出收集器：保留头部与尾部，中间截断。
/// `readabilityHandler` 在内部队列回调，多个数据块会并发进入。
public final class BoundedOutputCollector: @unchecked Sendable {
    private let lock = NSLock()
    private let limit: Int
    private let headBudget: Int
    private let tailBudget: Int
    private var head = ""
    private var tail = ""
    private var truncated = false

    public init(limit: Int = ToolSupport.maxOutputChars, tailRatio: Double = 0.25) {
        self.limit = limit
        let tail = Int(Double(limit) * tailRatio)
        self.tailBudget = tail
        self.headBudget = max(0, limit - tail)
    }

    public func feed(_ text: String) {
        guard !text.isEmpty else { return }
        lock.lock()
        defer { lock.unlock() }

        if !truncated {
            let room = headBudget - head.count
            if text.count <= room {
                head += text
                return
            }
            if room > 0 {
                head += String(text.prefix(room))
                truncated = true
                appendTail(String(text.dropFirst(room)))
            } else {
                truncated = true
                appendTail(text)
            }
            return
        }
        appendTail(text)
    }

    private func appendTail(_ text: String) {
        guard !text.isEmpty else { return }
        tail += text
        if tail.count > tailBudget {
            tail = String(tail.suffix(tailBudget))
        }
    }

    public func value() -> String {
        lock.lock()
        defer { lock.unlock() }
        var text = head
        if truncated {
            text += "\n… [输出超过 \(limit) 字符，中间已截断] …"
        }
        return text + tail
    }
}

// MARK: - 受限进程执行

/// 一次进程执行的结果。
public struct ProcessOutcome: Sendable {
    public var stdout: String
    public var stderr: String
    public var exitCode: Int32?
    public var timedOut: Bool
    public var cancelled: Bool
    /// 进程未能启动时的错误文案；正常执行时为 nil。
    public var launchError: String?

    public init(stdout: String = "", stderr: String = "", exitCode: Int32? = nil,
                timedOut: Bool = false, cancelled: Bool = false,
                launchError: String? = nil) {
        self.stdout = stdout
        self.stderr = stderr
        self.exitCode = exitCode
        self.timedOut = timedOut
        self.cancelled = cancelled
        self.launchError = launchError
    }
}

/// Foundation.Process 的封装：受限输出、超时/取消、优雅终止。
///
/// 说明：Foundation 的 `Process` 不暴露 `start_new_session`，因此这里对
/// 子进程先发 SIGTERM、宽限期后 SIGKILL。只有在子进程组 id 等于自身 pid
/// （即它是组长）时才发 `kill(-pid)`，避免误杀与父进程同组的其他进程。
public enum ToolProcessRunner {

    /// 在指定目录执行进程并等待结束。
    public static func run(executable: String,
                           arguments: [String],
                           currentDirectory: String? = nil,
                           environment: [String: String]? = nil,
                           timeout: Double,
                           shouldCancel: @escaping () -> Bool = { false }) async -> ProcessOutcome {
        let outCollector = BoundedOutputCollector()
        let errCollector = BoundedOutputCollector()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        let drainGroup = DispatchGroup()

        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        if let currentDirectory {
            process.currentDirectoryURL = URL(fileURLWithPath: currentDirectory)
        }
        process.environment = environment ?? ToolSupport.sanitizedEnvironment()
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        do {
            try process.run()
        } catch {
            try? stdoutPipe.fileHandleForReading.close()
            try? stderrPipe.fileHandleForReading.close()
            return ProcessOutcome(launchError: "Failed to start \(executable): \(ToolSupport.errorText(error))")
        }

        // 启动成功后再挂读取回调：管道里已有的数据会在 handler 安装后立即到达。
        drainGroup.enter()
        drainGroup.enter()
        attachReader(stdoutPipe.fileHandleForReading, collector: outCollector, group: drainGroup)
        attachReader(stderrPipe.fileHandleForReading, collector: errCollector, group: drainGroup)

        let deadline = Date().addingTimeInterval(timeout)
        var timedOut = false
        var cancelled = false

        // 0.15s 粒度轮询：超时与 UI「终止对话」都要能及时打断。
        while process.isRunning {
            if Task.isCancelled || shouldCancel() {
                cancelled = true
                break
            }
            if Date() >= deadline {
                timedOut = true
                break
            }
            try? await Task.sleep(nanoseconds: 150_000_000)
        }

        if timedOut || cancelled {
            await terminate(process)
        }

        // 等待回收（SIGKILL 后极端情况下仍需几百毫秒）。
        let reapDeadline = Date().addingTimeInterval(5)
        while process.isRunning && Date() < reapDeadline {
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        if process.isRunning {
            sendSignal(process.processIdentifier, SIGKILL)
        }

        // 给读线程留时间排空管道；孙进程持有写端时最多等 2 秒就放弃。
        let drainDeadline = Date().addingTimeInterval(2)
        while drainGroup.wait(timeout: .now()) != .success && Date() < drainDeadline {
            try? await Task.sleep(nanoseconds: 20_000_000)
        }

        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        stderrPipe.fileHandleForReading.readabilityHandler = nil
        try? stdoutPipe.fileHandleForReading.close()
        try? stderrPipe.fileHandleForReading.close()

        let exitCode: Int32? = process.isRunning ? nil : process.terminationStatus
        return ProcessOutcome(stdout: outCollector.value(),
                              stderr: errCollector.value(),
                              exitCode: exitCode,
                              timedOut: timedOut,
                              cancelled: cancelled)
    }

    // MARK: 内部实现

    private static func attachReader(_ handle: FileHandle,
                                     collector: BoundedOutputCollector,
                                     group: DispatchGroup) {
        let eof = EOFGuard()
        let decoder = UTF8StreamDecoder()
        handle.readabilityHandler = { readable in
            let data = readable.availableData
            if data.isEmpty {
                if eof.markDone() {
                    readable.readabilityHandler = nil
                    collector.feed(decoder.finish())
                    group.leave()
                }
                return
            }
            collector.feed(decoder.decode(data))
        }
    }

    /// SIGTERM → 3 秒宽限 → SIGKILL。先递归终止子进程，避免 npx/shell 留下孤儿。
    private static func terminate(_ process: Process) async {
        guard process.isRunning else { return }
        let pid = process.processIdentifier
        terminateProcessTree(pid: pid, signal: SIGTERM)
        let deadline = Date().addingTimeInterval(3)
        while process.isRunning && Date() < deadline {
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        if process.isRunning {
            terminateProcessTree(pid: pid, signal: SIGKILL)
        }
    }

    /// 递归终止进程树（先子后父）。仅用于本进程启动的工具子进程。
    public static func terminateProcessTree(pid: pid_t, signal: Int32) {
        guard pid > 0 else { return }
        for child in descendants(of: pid).reversed() {
            sendSignal(child, signal)
        }
        sendSignal(pid, signal)
    }

    private static func descendants(of pid: pid_t) -> [pid_t] {
        var result: [pid_t] = []
        for child in childPIDs(of: pid) {
            result.append(contentsOf: descendants(of: child))
            result.append(child)
        }
        return result
    }

    private static func childPIDs(of pid: pid_t) -> [pid_t] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
        process.arguments = ["-P", String(pid)]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            return []
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return String(decoding: data, as: UTF8.self)
            .split(whereSeparator: \.isNewline)
            .compactMap { pid_t($0.trimmingCharacters(in: .whitespaces)) }
    }

    /// 只对「自身即进程组长」的子进程发组信号，避免误杀父进程所在的组。
    private static func sendSignal(_ pid: pid_t, _ signal: Int32) {
        guard pid > 0 else { return }
        let pgid = getpgid(pid)
        if pgid == pid {
            kill(-pid, signal)
        } else {
            kill(pid, signal)
        }
    }
}

/// 跨数据块的增量 UTF-8 解码器：管道读取可能把多字节字符切成两半，
/// 逐块 `String(decoding:)` 会在切分处产生替换字符。
private final class UTF8StreamDecoder: @unchecked Sendable {
    private var pending = Data()
    private var broken = false

    func decode(_ data: Data) -> String {
        guard !broken else { return String(decoding: data, as: UTF8.self) }

        var combined = pending
        combined.append(data)
        if let text = String(data: combined, encoding: .utf8) {
            pending = Data()
            return text
        }
        // 末尾可能是不完整的多字节序列：最多回退 3 字节，余下留到下一块。
        for drop in 1...min(3, combined.count) {
            if let text = String(data: Data(combined.dropLast(drop)), encoding: .utf8) {
                pending = Data(combined.suffix(drop))
                return text
            }
        }
        // 中间存在非法字节：降级为有损解码，后续不再尝试拼接。
        broken = true
        pending = Data()
        return String(decoding: combined, as: UTF8.self)
    }

    func finish() -> String {
        guard !pending.isEmpty else { return "" }
        let text = String(decoding: pending, as: UTF8.self)
        pending = Data()
        return text
    }
}

/// readabilityHandler 的 EOF 只处理一次（回调可能在内部队列重复触发）。
private final class EOFGuard: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false

    func markDone() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if done { return false }
        done = true
        return true
    }
}
