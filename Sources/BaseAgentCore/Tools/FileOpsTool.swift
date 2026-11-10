import Foundation

// MARK: - 文件读取

/// `file_read`：读取 workspace 内文本文件。
///
/// 支持按行分页（offset/limit）与行号输出（line_numbers），大文件无需再绕道
/// shell_run tail/sed；UTF-8 解码失败时尝试 GB18030（中文场景常见编码），
/// 仍失败才按二进制 base64 处理。
public final class FileReadTool: AgentTool {
    public let name = "file_read"
    public let description =
        "Read a text file inside the workspace. Optional line pagination: offset "
        + "(1-based first line) + limit (max lines) with line_numbers=true returns "
        + "editor-style numbered lines for precise file_modify targeting."

    private let config: AgentConfig
    /// 改进5：SOP 只读一次的去重记录器，由 ToolFactory 注入（与 Agent 同生命周期）。
    private let sopTracker: SOPReadTracker

    public var parameters: JSONValue {
        ToolSupport.objectSchema(
            properties: [
                "path": ToolSupport.stringProperty(
                    "Relative path within the workspace (or absolute inside it)."),
                "offset": ToolSupport.numberProperty(
                    "First line to return, 1-based (default: from the start)."),
                "limit": ToolSupport.numberProperty(
                    "Max number of lines to return (default: all, capped by size limits)."),
                "line_numbers": ToolSupport.boolProperty(
                    "Prefix each line with \"N: \" for precise patch targeting."),
            ],
            required: ["path"])
    }

    public init(config: AgentConfig, callbacks: AgentCallbacks?, registry: ToolRegistry? = nil,
                sopTracker: SOPReadTracker = SOPReadTracker()) {
        self.config = config
        self.sopTracker = sopTracker
        _ = callbacks
        _ = registry
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        do {
            let path = try ToolSupport.requireString(args, "path")
            let offset = Int(ToolSupport.optionalDouble(args, "offset") ?? 0)
            let limit = Int(ToolSupport.optionalDouble(args, "limit") ?? 0)
            let lineNumbers = ToolSupport.optionalBool(args, "line_numbers") ?? false
            let full = try ToolSupport.resolveWorkspacePath(path, workspace: config.workspace)

            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: full, isDirectory: &isDirectory),
                  !isDirectory.boolValue else {
                return .fail("File does not exist: \(path)")
            }

            let size = ToolSupport.fileSize(full)
            let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: full))
            defer { try? handle.close() }

            // 按字符上限读取：UTF-8 最宽 4 字节/字符，多读一个余量用于判定截断。
            let byteCap = ToolSupport.maxReadChars * 4 + 4
            let raw = (try handle.read(upToCount: byteCap)) ?? Data()

            // 编码判定：UTF-8 → GB18030（中文场景常见）→ 二进制。
            let utf8Text = ToolSupport.decodeUTF8Prefix(raw)
            let gbText = (utf8Text == nil) ? Self.decodeGB18030(raw) : nil
            let encodingText = utf8Text ?? gbText

            if let decoded = encodingText {
                var output = decoded
                // 两个维度都可能截断：字符数超过上限，或文件比实际读入的字节多
                // （宽字符场景下 20 万字符恰好用满了字节预算）。
                let truncatedByChars = output.count > ToolSupport.maxReadChars
                let truncatedByBytes = size > raw.count
                if truncatedByChars {
                    output = String(output.prefix(ToolSupport.maxReadChars))
                }

                // 按行分页：offset/limit/line_numbers 一起处理。
                let wantsPagination = offset > 0 || limit > 0 || lineNumbers
                if wantsPagination {
                    var lines = output.components(separatedBy: "\n")
                    // 逐行编号前先去掉可能带出的行尾 \r（CRLF 文件）。
                    lines = lines.map { $0.hasSuffix("\r") ? String($0.dropLast()) : $0 }
                    let start = max(0, offset - (offset > 0 ? 1 : 0))
                    let end = limit > 0 ? min(lines.count, start + limit) : lines.count
                    guard start < lines.count else {
                        return .ok("(offset \(offset) 超出文件行数 \(lines.count))")
                    }
                    let slice = Array(lines[start..<end])
                    var paged = slice.enumerated().map { idx, line in
                        lineNumbers ? "\(start + idx + 1): \(line)" : line
                    }.joined(separator: "\n")
                    if end < lines.count || truncatedByBytes {
                        paged += "\n… [文件共 \(lines.count) 行 / \(size) 字节，"
                            + "显示第 \(start + 1)–\(end) 行]"
                            + (truncatedByBytes && end >= lines.count
                               ? "（文件超出读取上限，尾部未载入）" : "")
                    }
                    output = paged
                } else if truncatedByChars || truncatedByBytes {
                    output += "\n… [文件共 \(size) 字节，仅显示前 "
                        + "\(ToolSupport.maxReadChars) 字符]"
                }

                var result = output
                if gbText != nil {
                    // GB18030 路径：提示编码，避免模型把乱码当内容。
                    result = "[encoding: GB18030]\n" + result
                }
                // 改进5：L1 索引定位到的 SOP 全文进入当前轮（内核 L1 会把它裁到
                // remainingForTool 的最大长度）。追加"只提炼一次"的指令，让模型
                // 立即把要点沉淀为 key_info，此后每轮靠 key_info 回注，不再重读全文。
                if isMemorySOP(full) {
                    let first = sopTracker.markIfFirst(full)
                    result += first
                        ? "\n\n[SOP 已全文载入] 立即调用 update_working_checkpoint(key_info=...) "
                            + "把本 SOP 的关键前置条件与典型坑提炼进去；key_info 每轮自动回注，"
                            + "正常情况下本次会话无需再次 file_tools(op=\"read\") 本文件。"
                        : "\n\n[SOP 重复载入] 本文件本次会话已读过，要点应已在 key_info 中；"
                            + "若 key_info 仍含要点请直接据此行动；若已被覆盖/清空导致要点丢失，"
                            + "应立即重新读取全文并再次提炼，不要凭印象继续。"
                }
                return .ok(result)
            }

            // 非 UTF-8/GB18030：按二进制处理，base64 同样限制体积（会再放大 4/3）。
            try handle.seek(toOffset: 0)
            let binary = (try handle.read(upToCount: ToolSupport.maxReadChars / 2)) ?? Data()
            return .ok("(binary) " + binary.base64EncodedString())
        } catch {
            return ToolSupport.failure(error)
        }
    }

    /// GB18030 解码（含 GBK/GB2312 超集）。仅当解码结果「看起来是文本」
    /// （含可打印 CJK/ASCII 内容且控制字符占比低）才返回，否则视为二进制。
    static func decodeGB18030(_ data: Data) -> String? {
        let encoding = String.Encoding(
            rawValue: CFStringConvertEncodingToNSStringEncoding(
                CFStringEncoding(CFStringEncodings.GB_18030_2000.rawValue)))
        guard let text = String(data: data, encoding: encoding) else { return nil }
        // 防误判：GB18030 是双字节编码，几乎任何字节流都能"解出来"。
        // 用控制字符占比与可打印率做二次校验。
        let sample = text.prefix(2000)
        let control = sample.filter { $0.unicodeScalars.allSatisfy {
            $0.value < 0x20 && $0 != "\n" && $0 != "\t" && $0 != "\r"
        } }.count
        return Double(control) / Double(max(1, sample.count)) < 0.05 ? text : nil
    }

    /// 只识别**本工作区 memory/ 根目录正下方**的 L3 SOP（`*_sop.md`）与 L0
    /// META-SOP。传入的是 `resolveWorkspacePath` 归一化后的绝对路径：
    /// 用「包含 memory/ 子串」会误伤 `docs/memory/foo_sop.md` 这类无关文件。
    private func isMemorySOP(_ fullPath: String) -> Bool {
        let memoryRoot = (MemoryDirectory(workspace: config.workspace).root as NSString)
            .standardizingPath
        let normalized = (fullPath as NSString).standardizingPath
        let prefix = memoryRoot + "/"
        guard normalized.hasPrefix(prefix) else { return false }
        // 必须是直接子文件（非嵌套目录），避免把 memory/raw_session/ 下的内容误判。
        let relative = String(normalized.dropFirst(prefix.count))
        guard !relative.contains("/") else { return false }
        let name = relative.lowercased()
        return name.hasSuffix("_sop.md") || name == "memory_management_sop.md"
    }
}

// MARK: - 文件写入

/// `file_write`：覆盖写入 workspace 内文件，自动创建父目录。
public final class FileWriteTool: AgentTool {
    public let name = "file_write"
    public let description =
        "Write text content to a file inside the workspace, creating parent directories."
    /// 锁定在 workspace 内，无法影响系统/其他文件。
    public let destructive = false

    private let config: AgentConfig

    public var parameters: JSONValue {
        ToolSupport.objectSchema(
            properties: [
                "path": ToolSupport.stringProperty("File path within the workspace."),
                "content": ToolSupport.stringProperty("Text content to write."),
            ],
            required: ["path", "content"])
    }

    public init(config: AgentConfig, callbacks: AgentCallbacks?, registry: ToolRegistry? = nil) {
        self.config = config
        _ = callbacks
        _ = registry
    }

    /// `memory/` 内的新建文件（如 L3 SOP）必须由人复核；L0/L1 更直接由
    /// `guardMemoryWrite` 硬拒，不给确认后覆盖的机会。
    public func shouldConfirm(_ args: [String: JSONValue]) -> Bool {
        guard let path = ToolSupport.optionalString(args, "path"),
              let full = try? ToolSupport.resolveWorkspacePath(path, workspace: config.workspace)
        else { return false }
        return ToolSupport.isManagedMemoryPath(full, workspace: config.workspace)
            && !ToolSupport.isSystemPromptMemoryFile(full, workspace: config.workspace)
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        do {
            let path = try ToolSupport.requireString(args, "path")
            let content = try ToolSupport.requireString(args, "content")

            // 与 file_read 的上限策略对齐；病态体积明确提示改用 shell_run。
            guard content.count <= ToolSupport.maxWriteChars else {
                return .fail(
                    "content too large (\(content.count) chars > \(ToolSupport.maxWriteChars) cap). "
                    + "Split the write into several smaller file_tools(op=\"write\") calls, or use "
                    + "shell_run with a heredoc / streaming write for large files.")
            }

            let full = try ToolSupport.resolveWorkspacePath(path, workspace: config.workspace)
            try ToolSupport.guardMemoryWrite(full, workspace: config.workspace)
            try AtomicFile.write(content, to: full)
            return .ok("Wrote \(content.count) chars to \(path)")
        } catch {
            return ToolSupport.failure(error)
        }
    }
}

// MARK: - 文件局部修改

/// `file_modify`：替换 workspace 文件中的 `old` 块（默认第一处，`replace_all=true` 全部）。
///
/// 与参考实现一致：默认替换**第一处**，不强制唯一；需要全部替换时显式设置
/// `replace_all=true`。
public final class FileModifyTool: AgentTool {
    public let name = "file_modify"
    public let description =
        "Replace the first occurrence of `old` with `new` in a workspace file. "
        + "`old` must match exactly (whitespace/indentation); use file_read first. "
        + "Set replace_all=true to replace every occurrence."
    public let destructive = false

    private let config: AgentConfig

    public var parameters: JSONValue {
        ToolSupport.objectSchema(
            properties: [
                "path": ToolSupport.stringProperty("File path within the workspace."),
                "old": ToolSupport.stringProperty("Exact text block to replace (first occurrence)."),
                "new": ToolSupport.stringProperty("Replacement text."),
                "replace_all": ToolSupport.boolProperty(
                    "Replace every occurrence (default false = first occurrence only)."),
            ],
            required: ["path", "old", "new"])
    }

    public init(config: AgentConfig, callbacks: AgentCallbacks?, registry: ToolRegistry? = nil) {
        self.config = config
        _ = callbacks
        _ = registry
    }

    /// `memory/` 下只允许 `file_patch` 少量追加，且每次都必须人工复核：
    /// 这些文件会被回注 system prompt / 长期记忆，静默改写等于持久化注入。
    public func shouldConfirm(_ args: [String: JSONValue]) -> Bool {
        guard let path = ToolSupport.optionalString(args, "path"),
              let full = try? ToolSupport.resolveWorkspacePath(path, workspace: config.workspace)
        else { return false }
        return ToolSupport.isManagedMemoryPath(full, workspace: config.workspace)
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        do {
            let path = try ToolSupport.requireString(args, "path")
            let old = try ToolSupport.requireString(args, "old")
            let new = try ToolSupport.requireString(args, "new")
            let replaceAll = ToolSupport.optionalBool(args, "replace_all") ?? false

            guard !old.isEmpty else {
                return .fail("'old' must be a non-empty string")
            }

            let full = try ToolSupport.resolveWorkspacePath(path, workspace: config.workspace)

            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: full, isDirectory: &isDirectory),
                  !isDirectory.boolValue else {
                return .fail("File does not exist: \(path)")
            }

            let size = ToolSupport.fileSize(full)
            guard size <= ToolSupport.maxModifyBytes else {
                return .fail(
                    "File too large for file_tools(op=\"modify\") (\(size) bytes > "
                    + "\(ToolSupport.maxModifyBytes)); use file_tools(op=\"write\") to rewrite it "
                    + "or shell_run (sed/python) instead")
            }

            let data = try Data(contentsOf: URL(fileURLWithPath: full))
            guard let text = String(data: data, encoding: .utf8) else {
                return .fail("File is not UTF-8 text: \(path)")
            }

            let occurrences = text.components(separatedBy: old).count - 1
            guard occurrences > 0 else {
                // 未命中是 file_modify 最常见失败：大概率是缩进/空白/CRLF 差异。
                // 给出可操作的排查路径，而不是让模型盲试。
                let hasCR = text.contains("\r\n")
                return .fail("'old' string not found in \(path). "
                    + "常见原因：① 缩进/空白与文件实际内容不一致；"
                    + "② 文件为 CRLF 行尾\(hasCR ? "（本文件确为 CRLF）" : "")；"
                    + "③ 内容已被之前的操作改过。"
                    + "请用 file_tools(op=\"read\", line_numbers=true) 重新确认精确文本后再试。")
            }

            let updated: String
            if replaceAll {
                updated = text.replacingOccurrences(of: old, with: new)
            } else if let range = text.range(of: old) {
                var copy = text
                copy.replaceSubrange(range, with: new)
                updated = copy
            } else {
                return .fail("'old' string not found in \(path)")
            }

            try AtomicFile.write(updated, to: full)
            // 报告替换处数：replace_all 时模型需要知道实际生效了几处。
            return .ok("Modified \(path)（替换 \(replaceAll ? occurrences : 1) 处）")
        } catch {
            return ToolSupport.failure(error)
        }
    }
}
