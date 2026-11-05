import Foundation

// MARK: - 文件读取

/// `file_read`：读取 workspace 内文本文件。
public final class FileReadTool: AgentTool {
    public let name = "file_read"
    public let description = "Read the textual contents of a file inside the workspace."

    private let config: AgentConfig
    /// 改进5：SOP 只读一次的去重记录器，由 ToolFactory 注入（与 Agent 同生命周期）。
    private let sopTracker: SOPReadTracker

    public var parameters: JSONValue {
        ToolSupport.objectSchema(
            properties: [
                "path": ToolSupport.stringProperty(
                    "Relative path within the workspace (or absolute inside it)."),
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

            if let text = ToolSupport.decodeUTF8Prefix(raw) {
                var output = text
                // 两个维度都可能截断：字符数超过上限，或文件比实际读入的字节多
                // （宽字符场景下 20 万字符恰好用满了字节预算）。
                let truncatedByChars = output.count > ToolSupport.maxReadChars
                let truncatedByBytes = size > raw.count
                if truncatedByChars {
                    output = String(output.prefix(ToolSupport.maxReadChars))
                }
                if truncatedByChars || truncatedByBytes {
                    output += "\n… [文件共 \(size) 字节，仅显示前 "
                        + "\(ToolSupport.maxReadChars) 字符]"
                }
                // 改进5：L1 索引定位到的 SOP 全文进入当前轮（内核 L1 会把它裁到
                // remainingForTool 的最大长度）。追加"只提炼一次"的指令，让模型
                // 立即把要点沉淀为 key_info，此后每轮靠 key_info 回注，不再重读全文。
                if isMemorySOP(full) {
                    let first = sopTracker.markIfFirst(full)
                    output += first
                        ? "\n\n[SOP 已全文载入] 立即调用 update_working_checkpoint(key_info=...) "
                            + "把本 SOP 的关键前置条件与典型坑提炼进去；key_info 每轮自动回注，"
                            + "正常情况下本次会话无需再次 file_read 本文件。"
                        : "\n\n[SOP 重复载入] 本文件本次会话已读过，要点应已在 key_info 中；"
                            + "若 key_info 仍含要点请直接据此行动；若已被覆盖/清空导致要点丢失，"
                            + "应立即重新读取全文并再次提炼，不要凭印象继续。"
                }
                return .ok(output)
            }

            // 非 UTF-8：按二进制处理，base64 同样限制体积（会再放大 4/3）。
            try handle.seek(toOffset: 0)
            let binary = (try handle.read(upToCount: ToolSupport.maxReadChars / 2)) ?? Data()
            return .ok("(binary) " + binary.base64EncodedString())
        } catch {
            return ToolSupport.failure(error)
        }
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

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        do {
            let path = try ToolSupport.requireString(args, "path")
            let content = try ToolSupport.requireString(args, "content")

            // 与 file_read 的上限策略对齐；病态体积明确提示改用 shell_run。
            guard content.count <= ToolSupport.maxWriteChars else {
                return .fail(
                    "content too large (\(content.count) chars > \(ToolSupport.maxWriteChars) cap). "
                    + "Split the write into several smaller file_write calls, or use "
                    + "shell_run with a heredoc / streaming write for large files.")
            }

            let full = try ToolSupport.resolveWorkspacePath(path, workspace: config.workspace)
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
                    "File too large for file_modify (\(size) bytes > "
                    + "\(ToolSupport.maxModifyBytes)); use file_write to rewrite it "
                    + "or shell_run (sed/python) instead")
            }

            let data = try Data(contentsOf: URL(fileURLWithPath: full))
            guard let text = String(data: data, encoding: .utf8) else {
                return .fail("File is not UTF-8 text: \(path)")
            }

            let occurrences = text.components(separatedBy: old).count - 1
            guard occurrences > 0 else {
                return .fail("'old' string not found in \(path)")
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
            return .ok("Modified \(path)")
        } catch {
            return ToolSupport.failure(error)
        }
    }
}
