import Foundation

// MARK: - 检查点状态文件

/// 工作记忆检查点的最小可行落盘实现。
///
/// Python 版把 checkpoint 写进内核 `Session.working`，跨会话部分交给
/// work memory。Swift 工具层不直接依赖 Memory 层，因此统一写入
/// `<workspace>/.agent/working_checkpoint.json`：文件天然跨会话存活，
/// 同时也能被 `start_long_term_update` 读取并作为蒸馏材料。
enum CheckpointStore {
    static let maxKeyInfo = 4_000

    static func path(_ config: AgentConfig) -> String {
        ToolSupport.stateFile("working_checkpoint.json", workspace: config.workspace)
    }

    static func read(_ config: AgentConfig) -> String {
        readRecord(config)?.text ?? ""
    }

    /// 仅返回标记了 `persist=true` 的跨会话检查点。
    static func readPersisted(_ config: AgentConfig) -> String {
        guard let record = readRecord(config), record.persist else { return "" }
        return record.text
    }

    static func readRecord(_ config: AgentConfig) -> (text: String, persist: Bool)? {
        guard let data = FileManager.default.contents(atPath: path(config)),
              let raw = try? JSONDecoder().decode(JSONValue.self, from: data),
              let text = raw["key_info"]?.stringValue else {
            return nil
        }
        return (text, raw["persist"]?.boolValue ?? false)
    }

    static func write(_ text: String, persist: Bool, config: AgentConfig) throws {
        let payload = JSONValue.object([
            "key_info": .string(text),
            "persist": .bool(persist),
            "updated_at": .string(ISO8601DateFormatter().string(from: Date())),
        ])
        try AtomicFile.write(payload.jsonString(pretty: true), to: path(config))
    }

    static func clear(_ config: AgentConfig) throws {
        let file = path(config)
        guard FileManager.default.fileExists(atPath: file) else { return }
        try FileManager.default.removeItem(atPath: file)
    }
}

// MARK: - update_working_checkpoint

/// `update_working_checkpoint`：短期工作记忆便签。
///
/// 只记「非显然、后续必须知道」的状态（已验证的路径/ID/端口、决策、已排除
/// 的方案），不记流水账。空字符串表示清空。
public final class UpdateWorkingCheckpointTool: AgentTool {
    public let name = "update_working_checkpoint"
    public let description =
        "Short-term working notepad, auto-injected into every later turn. "
        + "Call early/mid task ONLY when there is non-obvious state worth keeping: "
        + "verified paths/IDs/ports, decisions, approaches already ruled out. "
        + "Do NOT store transcripts or facts re-derivable in one tool call. "
        + "Empty key_info clears it."

    private let config: AgentConfig
    private let callbacks: AgentCallbacks?
    private weak var registry: ToolRegistry?

    public var parameters: JSONValue {
        ToolSupport.objectSchema(
            properties: [
                "key_info": ToolSupport.stringProperty(
                    "Compact notes (<=\(CheckpointStore.maxKeyInfo) chars). "
                    + "Empty string clears the notepad."),
                "persist": ToolSupport.boolProperty(
                    "Also persist across sessions (default false)."),
            ],
            required: ["key_info"])
    }

    public init(config: AgentConfig, callbacks: AgentCallbacks?, registry: ToolRegistry? = nil) {
        self.config = config
        self.callbacks = callbacks
        self.registry = registry
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        do {
            let keyInfo = try ToolSupport.requireString(args, "key_info")
            let persist = ToolSupport.optionalBool(args, "persist") ?? false
            let text = String(keyInfo
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .prefix(CheckpointStore.maxKeyInfo))

            guard let session = registry?.session else {
                return .fail("working checkpoint unavailable (no kernel session bound)")
            }
            session.setKeyInfo(text)

            if persist {
                try CheckpointStore.write(text, persist: true, config: config)
            }
            if text.isEmpty {
                try CheckpointStore.clear(config)
                return .ok("working checkpoint cleared")
            }
            let hint = session.currentTurn <= 2
                ? " (tip: skip re-calling unless new non-obvious facts appear)" : ""
            return .ok("working checkpoint updated (\(text.count) chars)\(hint)")
        } catch {
            return ToolSupport.failure(error)
        }
    }
}

// MARK: - start_long_term_update

/// `start_long_term_update`：**L2/L3 文件的唯一正式写入入口**（改进6）。
///
/// 本身不直接写文件，而是「门控 + 返回提纯 prompt 与材料」，由模型随后用
/// `file_patch` 少量追加。门控：当前轮数 < `minTurns`(5) 直接拒绝，短任务不许碰长期记忆。
///
/// 内核（`Loop.swift` 按工具名）仍会置 `distillRequested`，触发后台图谱蒸馏；
/// 但 L2 `memory/global_mem.txt` / L3 `memory/*_sop.md` 的写入必须经过这里。
public final class StartLongTermUpdateTool: AgentTool {
    public let name = "start_long_term_update"
    public let description =
        "ONLY gate for writing long-term memory files (L2/L3). Call after finishing a task "
        + "or reaching an important checkpoint; rejected before turn 5. Returns the purification "
        + "prompt and current L1 index. Then follow L0 decision tree: verified environment facts "
        + "-> append to memory/global_mem.txt; hard-won task pitfalls -> append/create "
        + "memory/xxx_sop.md. Use file_patch (small additions only), then sync the L1 index. "
        + "Skip if nothing durable was learned."

    /// 短任务门控：当前轮数小于此值直接拒绝。
    public static let minTurns = 5

    private let config: AgentConfig
    private let callbacks: AgentCallbacks?
    private weak var registry: ToolRegistry?
    private let memoryDir: MemoryDirectory

    public var parameters: JSONValue {
        ToolSupport.objectSchema(
            properties: [
                "focus": ToolSupport.stringProperty(
                    "What to distill, e.g. 'python env setup' (optional)."),
            ])
    }

    public init(config: AgentConfig, callbacks: AgentCallbacks?, registry: ToolRegistry? = nil) {
        self.config = config
        self.callbacks = callbacks
        self.registry = registry
        self.memoryDir = MemoryDirectory(workspace: config.workspace)
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        // 门控：轮数不足直接拒绝，短任务不碰长期记忆。
        let turn = registry?.session?.currentTurn ?? 0
        guard turn >= Self.minTurns else {
            return .fail("记忆提纯被拒绝：当前轮数 \(turn) < \(Self.minTurns)。"
                + "短任务不许写长期记忆，请直接完成任务；到达重要检查点且轮数足够后再调用。")
        }

        let focus = ToolSupport.optionalString(args, "focus") ?? ""
        // 材料优先取内存 Session 的 key_info（默认 persist=false 不落盘），
        // 为空再回退磁盘检查点；避免旧实现"读盘恒空"的问题。
        var transcript = registry?.session?.getKeyInfo() ?? ""
        if transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            transcript = CheckpointStore.read(config)
        }
        if !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            transcript = "[working_checkpoint]\n\(transcript)"
        } else {
            transcript = "(当前会话尚无有效记录)"
        }

        let l1 = memoryDir.read(memoryDir.l1File, limit: SystemPromptLimits.maxL1Chars) ?? ""

        let payload = JSONValue.object([
            "status": .string("DISTILL_READY"),
            "turn": .int(turn),
            "focus": .string(focus),
            "transcript": .string(String(transcript.suffix(3_000))),
            // 注入上下文：L0 META-SOP 与整份 L1 已常驻 system prompt（在其后），
            // 这里再回一份 L1 当前全文，供同步 patch 时精确匹配 old。
            "l1_current": .string(l1),
            "prompt": .string(MEMORY_PURIFY_PROMPT),
            "next": .array([
                .string("先 file_read 读 memory/global_mem.txt 与相关 xxx_sop.md 现状（先读后patch）"),
                .string("按 L0 决策树归类：环境特异性事实 → file_patch 追加到 memory/global_mem.txt"),
                .string("特定任务技术/坑 → file_patch 追加或新建 memory/xxx_sop.md"),
                .string("每次 L2/L3 变更后，用 file_patch 同步 L1 索引行（只放关键词/名称，≤30 行）"),
            ]),
            "rules": .array([
                .string("只写经工具验证成功的信息；禁止写入推测、临时状态、通用常识、易重建细节"),
                .string("合并重复、压缩冗述，不堆叠流水账；不得把模型推测记作用户要求"),
                .string("memory/ 下只能 file_patch 少量追加，禁止 file_write 覆盖与 code_run 批量改写"),
                .string("无提纯价值则跳过，不要为了调用而调用"),
            ]),
        ])
        return .ok(compactJSON(payload, maxChars: 3_600))
    }
}
