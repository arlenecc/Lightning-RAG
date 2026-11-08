import Foundation

// MARK: - 文件枚举

/// `file_list`：枚举 workspace 内目录条目（只读）。
///
/// 结构性缺口补齐：此前工作区可见性完全依赖 `shell_run` 跑 `ls/grep`，
/// 无 shell 确认路径下 agent 对工作区结构完全失明。本工具 + `file_search`
/// 让 agent 无需 shell 即可完成「看到什么文件、找哪段内容」两类高频动作。
public final class FileListTool: AgentTool {
    public let name = "file_list"
    public let description =
        "List directory entries inside the workspace (read-only). Returns per-entry "
        + "type (dir/), size and last-modified time. Optional glob filter (e.g. *.swift) "
        + "and recursive mode with depth cap. Hidden entries (dot-prefixed) are skipped."
    /// 只读操作，不需要确认。
    public let destructive = false

    private let config: AgentConfig

    public var parameters: JSONValue {
        ToolSupport.objectSchema(
            properties: [
                "path": ToolSupport.stringProperty(
                    "Directory path relative to the workspace (default: workspace root)."),
                "glob": ToolSupport.stringProperty(
                    "Optional glob filter for file names, e.g. \"*.swift\" (dirs unaffected)."),
                "recursive": ToolSupport.boolProperty(
                    "Recurse into subdirectories (default false)."),
                "max_depth": ToolSupport.numberProperty(
                    "Max recursion depth when recursive=true (default 3)."),
                "max_entries": ToolSupport.numberProperty(
                    "Max entries to return (default 500)."),
            ],
            required: [])
    }

    /// 单次枚举的默认/硬上限：防止把巨型目录树整棵倒进上下文。
    static let defaultMaxEntries = 500
    static let hardMaxEntries = 5_000
    static let defaultMaxDepth = 3
    static let hardMaxDepth = 8

    public init(config: AgentConfig, callbacks: AgentCallbacks?, registry: ToolRegistry? = nil) {
        self.config = config
        _ = callbacks
        _ = registry
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        do {
            let path = ToolSupport.optionalString(args, "path") ?? "."
            let glob = ToolSupport.optionalString(args, "glob") ?? ""
            let recursive = ToolSupport.optionalBool(args, "recursive") ?? false
            let maxDepth = min(
                Int(ToolSupport.optionalDouble(args, "max_depth")
                    ?? Double(Self.defaultMaxDepth)),
                Self.hardMaxDepth)
            let maxEntries = min(
                Int(ToolSupport.optionalDouble(args, "max_entries")
                    ?? Double(Self.defaultMaxEntries)),
                Self.hardMaxEntries)

            let root = try ToolSupport.resolveWorkspacePath(path, workspace: config.workspace)
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: root, isDirectory: &isDirectory),
                  isDirectory.boolValue else {
                return .fail("Directory does not exist: \(path)")
            }

            let globRegex = Self.globRegex(glob)
            var lines: [String] = []
            var truncated = false

            func fmt(_ url: URL, isDir: Bool) -> String? {
                let name = url.lastPathComponent
                // 隐藏条目（.开头）默认跳过：.git/.build 等噪声占满输出。
                guard !name.hasPrefix(".") else { return nil }
                if !isDir, let regex = globRegex,
                   regex.firstMatch(in: name, range: NSRange(name.startIndex..., in: name))
                       == nil {
                    return nil
                }
                let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
                let size = (attrs?[.size] as? Int64) ?? 0
                let mtime = (attrs?[.modificationDate] as? Date) ?? Date()
                let df = DateFormatter()
                df.dateFormat = "yyyy-MM-dd HH:mm"
                let tag = isDir ? "dir " : "file"
                return "\(tag) \(String(format: "%8d", size)) "
                    + df.string(from: mtime) + "  "
                    + (isDir ? name + "/" : name)
            }

            if !recursive {
                for item in try FileManager.default.contentsOfDirectory(
                    at: URL(fileURLWithPath: root), includingPropertiesForKeys: nil) {
                    guard lines.count < maxEntries else { truncated = true; break }
                    var isDir: ObjCBool = false
                    _ = FileManager.default.fileExists(atPath: item.path, isDirectory: &isDir)
                    if let line = fmt(item, isDir: isDir.boolValue) { lines.append(line) }
                }
            } else {
                let enumerator = FileManager.default.enumerator(
                    at: URL(fileURLWithPath: root),
                    includingPropertiesForKeys: [.isDirectoryKey],
                    options: [.skipsHiddenFiles])
                while let item = enumerator?.nextObject() as? URL {
                    if lines.count >= maxEntries { truncated = true; break }
                    let depth = (enumerator?.level ?? 0)
                    if depth > maxDepth { enumerator?.skipDescendants(); continue }
                    let isDir = (try? item.resourceValues(forKeys: [.isDirectoryKey]))?
                        .isDirectory ?? false
                    if let line = fmt(item, isDir: isDir) { lines.append(line) }
                }
            }

            guard !lines.isEmpty else {
                return .ok("(empty directory \(recursive ? "tree" : ""): \(path))")
            }
            var output = lines.joined(separator: "\n")
            if truncated {
                output += "\n… [结果超过 \(maxEntries) 条已截断；可用 glob 收窄范围，"
                    + "或对子目录单独调用]"
            }
            return .ok(output)
        } catch {
            return ToolSupport.failure(error)
        }
    }

    /// glob → 正则（`*`→`[^/]*`，`?`→`[^/]`，其余字符按字面量转义）。
    static func globRegex(_ glob: String) -> NSRegularExpression? {
        let pattern = glob.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !pattern.isEmpty else { return nil }
        var regex = ""
        for ch in pattern {
            switch ch {
            case "*": regex += "[^/]*"
            case "?": regex += "[^/]"
            default: regex += NSRegularExpression.escapedPattern(for: String(ch))
            }
        }
        return try? NSRegularExpression(pattern: "^" + regex + "$")
    }
}

// MARK: - 内容搜索

/// `file_search`：在 workspace 内按正则搜索文件内容（只读，grep 语义）。
///
/// 输出 `path:line: text`，与主流编辑器/Grep 一致，模型可直接把行号用于
/// 后续 file_modify 定位。自动跳过隐藏目录、二进制与超大文件。
public final class FileSearchTool: AgentTool {
    public let name = "file_search"
    public let description =
        "Search file contents inside the workspace with a regex (read-only, grep-like). "
        + "Returns \"path:line: matched text\" per hit. Skips hidden dirs, binaries and "
        + "oversized files. Use glob (e.g. \"*.swift\") to narrow the file set."
    public let destructive = false

    private let config: AgentConfig

    public var parameters: JSONValue {
        ToolSupport.objectSchema(
            properties: [
                "pattern": ToolSupport.stringProperty("Regular expression to search for."),
                "path": ToolSupport.stringProperty(
                    "Base directory relative to the workspace (default: workspace root)."),
                "glob": ToolSupport.stringProperty(
                    "Optional glob to filter searched files, e.g. \"*.swift\"."),
                "case_sensitive": ToolSupport.boolProperty(
                    "Case-sensitive matching (default false)."),
                "context": ToolSupport.numberProperty(
                    "Context lines before/after each hit (default 0, max 3)."),
                "max_results": ToolSupport.numberProperty(
                    "Max matching lines to return (default 60)."),
            ],
            required: ["pattern"])
    }

    static let defaultMaxResults = 60
    static let hardMaxResults = 300
    /// 超过此大小的文件不读入（避免把日志/二进制整份倒进内存）。
    static let maxSearchFileBytes = 2 * 1024 * 1024
    /// 总扫描字节预算：防止巨型工作区把一次搜索拖成分钟级。
    static let totalScanByteBudget = 16 * 1024 * 1024

    public init(config: AgentConfig, callbacks: AgentCallbacks?, registry: ToolRegistry? = nil) {
        self.config = config
        _ = callbacks
        _ = registry
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        do {
            let pattern = try ToolSupport.requireString(args, "pattern")
            let path = ToolSupport.optionalString(args, "path") ?? "."
            let glob = ToolSupport.optionalString(args, "glob") ?? ""
            let caseSensitive = ToolSupport.optionalBool(args, "case_sensitive") ?? false
            let context = min(Int(ToolSupport.optionalDouble(args, "context") ?? 0), 3)
            let maxResults = min(
                Int(ToolSupport.optionalDouble(args, "max_results")
                    ?? Double(Self.defaultMaxResults)),
                Self.hardMaxResults)

            guard let regex = try? NSRegularExpression(
                pattern: pattern,
                options: caseSensitive ? [] : [.caseInsensitive]) else {
                return .fail("Invalid regular expression: \(pattern)")
            }
            let globRegex = FileListTool.globRegex(glob)

            let root = try ToolSupport.resolveWorkspacePath(path, workspace: config.workspace)
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: root, isDirectory: &isDirectory) else {
                return .fail("Path does not exist: \(path)")
            }

            var results: [String] = []
            var scannedBytes = 0
            var skippedBinary = 0
            var hitLimit = false

            func searchFile(_ url: URL) {
                guard results.count < maxResults, scannedBytes < Self.totalScanByteBudget
                else { return }
                let name = url.lastPathComponent
                guard !name.hasPrefix(".") else { return }
                if let g = globRegex,
                   g.firstMatch(in: name, range: NSRange(name.startIndex..., in: name)) == nil {
                    return
                }
                var isDir: ObjCBool = false
                guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir),
                      !isDir.boolValue else { return }
                let size = ToolSupport.fileSize(url.path)
                guard size <= Self.maxSearchFileBytes else { return }
                scannedBytes += size
                guard let data = FileManager.default.contents(atPath: url.path) else { return }
                // NUL 字节 ≈ 二进制；计入跳过数但不中断。
                if data.contains(0) {
                    skippedBinary += 1
                    return
                }
                guard let text = String(data: data, encoding: .utf8) else { return }
                let lines = text.components(separatedBy: "\n")
                var emitted: Set<Int> = []
                for (idx, line) in lines.enumerated() {
                    guard results.count < maxResults else {
                        hitLimit = true; return
                    }
                    let ns = line as NSString
                    guard regex.firstMatch(in: line,
                                           range: NSRange(location: 0, length: ns.length))
                        != nil else { continue }
                    // 相对路径展示：两边都过 realpath（/var → /private/var 等
                    // symlink 归一化必须一致，否则前缀比对失败退化成文件名）。
                    let relFull = ToolSupport.realPath(url.path)
                    let rel = relFull.hasPrefix(root + "/")
                        ? String(relFull.dropFirst(root.count + 1))
                        : url.lastPathComponent
                    if context == 0 {
                        results.append("\(rel):\(idx + 1): \(ns.trimmedPrefix(400))")
                        emitted.insert(idx)
                    } else {
                        for k in max(0, idx - context)...min(lines.count - 1, idx + context) {
                            guard !emitted.contains(k) else { continue }
                            emitted.insert(k)
                            let marker = k == idx ? ":" : "-"
                            results.append("\(rel):\(k + 1)\(marker) "
                                + (lines[k] as NSString).trimmedPrefix(400))
                        }
                    }
                }
            }

            if isDirectory.boolValue {
                let enumerator = FileManager.default.enumerator(
                    at: URL(fileURLWithPath: root),
                    includingPropertiesForKeys: nil,
                    options: [.skipsHiddenFiles])
                while let item = enumerator?.nextObject() as? URL {
                    if results.count >= maxResults { hitLimit = true; break }
                    if scannedBytes >= Self.totalScanByteBudget { break }
                    searchFile(item)
                }
            } else {
                searchFile(URL(fileURLWithPath: root))
            }

            guard !results.isEmpty else {
                var note = ""
                if skippedBinary > 0 { note += "（已跳过 \(skippedBinary) 个二进制文件）" }
                return .ok("(no matches for \"\(pattern)\"\(note))")
            }
            var output = results.joined(separator: "\n")
            if hitLimit {
                output += "\n… [命中数超过 \(maxResults) 已截断；建议加 glob 收窄文件范围"
                    + "或用更具体的 pattern]"
            }
            return .ok(output)
        } catch {
            return ToolSupport.failure(error)
        }
    }
}

extension NSString {
    /// 搜索输出单行截断：超长行（如压缩过的 JS）不把上下文撑爆。
    func trimmedPrefix(_ max: Int) -> String {
        let text = self as String
        return length <= max ? text : String(text.prefix(max)) + "…"
    }
}
