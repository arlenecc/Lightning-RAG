import Foundation

/// 用户可编辑的基础 system prompt：`<workspace>/assets/system_prompt.txt`。
///
/// - 首次启动：写出内置 `BASE_SYSTEM_PROMPT`，用户可直接用文本编辑器调整。
/// - 之后每次启动：由 `AppViewModel` 读取该文件并注入 `Agent`，替换内置提示词。
/// - 文件为空 / 不可读 / 首次写出失败：本次回退内置常量，**绝不覆盖用户文件**。
///
/// 文件内容就是提示词本身，不额外写注释行，避免把说明文字也发给模型。
/// 只替换协议主体；L0 META-SOP 与 L1 全局索引仍由 `Agent.buildStaticSystemPrompt()`
/// 按多级记忆机制追加在后面。
public enum SystemPromptFile {
    public static let directoryName = "assets"
    public static let fileName = "system_prompt.txt"

    public enum Status: String, Sendable, Equatable {
        /// 首次启动，已写出内置提示词。
        case created
        /// 读到了用户编辑后的内容。
        case loaded
        /// 本次回退内置常量（见 `note`）。
        case fallback
    }

    public struct Resolved: Sendable {
        public let text: String
        public let path: String
        public let status: Status
        /// `fallback` 时的人类可读原因，其余状态为空串。
        public let note: String
    }

    public static func path(in workspace: String) -> String {
        let dir = (workspace as NSString).appendingPathComponent(directoryName)
        return (dir as NSString).appendingPathComponent(fileName)
    }

    /// 启动钩子：文件不存在则创建，存在则以其内容为准。
    /// 同步且轻量；任何失败都只降级为内置提示词，不阻塞启动、不抛错。
    @discardableResult
    public static func resolve(workspace: String) -> Resolved {
        let file = path(in: workspace)
        let fm = FileManager.default
        func fallback(_ note: String) -> Resolved {
            Resolved(text: BASE_SYSTEM_PROMPT, path: file, status: .fallback, note: note)
        }

        guard fm.fileExists(atPath: file) else {
            do {
                try fm.createDirectory(
                    atPath: (file as NSString).deletingLastPathComponent,
                    withIntermediateDirectories: true)
                try (BASE_SYSTEM_PROMPT + "\n").write(
                    to: URL(fileURLWithPath: file), atomically: true, encoding: .utf8)
                return Resolved(text: BASE_SYSTEM_PROMPT, path: file,
                                status: .created, note: "")
            } catch {
                return fallback("无法写出默认文件：\(error.localizedDescription)")
            }
        }

        do {
            let raw = try String(contentsOf: URL(fileURLWithPath: file), encoding: .utf8)
            // 去掉首尾空白：编辑器常见的末尾换行不应让 system 前缀逐字节抖动，
            // 否则每次启动都会让 KV cache 全量重建。
            let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else {
                return fallback("文件内容为空，本次使用内置提示词（文件未改动）")
            }
            return Resolved(text: text, path: file, status: .loaded, note: "")
        } catch {
            return fallback("读取失败：\(error.localizedDescription)")
        }
    }
}
