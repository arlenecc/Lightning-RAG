import Foundation

// MARK: - 检查点状态文件

/// 工作记忆检查点的最小可行落盘实现。
///
/// Python 版把 checkpoint 写进内核 `Session.working`，跨会话部分交给
/// work memory。Swift 工具层不直接依赖 Memory 层，因此统一写入
/// `<workspace>/.agent/working_checkpoint.json`：文件天然跨会话存活，
/// 同时也能被 `start_long_term_update` 读取并作为蒸馏材料。
enum CheckpointStore {
    /// 便签每轮都会回注，上限直接决定每轮固定开销（2000 字符 ≈ 1.2~1.5k token）。
    /// 超长先请辅助模型压缩，仍超长则保头保尾。
    static let maxKeyInfo = 2_000

    static func path(_ config: AgentConfig) -> String {
        ToolSupport.stateFile("working_checkpoint.json", workspace: config.workspace)
    }

    static func read(_ config: AgentConfig) -> String {
        readRecord(config)?.text ?? ""
    }

    /// 头尾截断到 ``maxKeyInfo``，显式标注中间省略。压缩器不可用或历史文件
    /// 超长（旧上限 4000）时用它兜底，保证每轮回注的固定开销可控。
    static func clamp(_ text: String) -> String {
        guard text.count > maxKeyInfo else { return text }
        let marker = "\n[…中间已省略]\n"
        let head = Int(Double(maxKeyInfo) * 0.6)
        let tail = max(200, maxKeyInfo - head - marker.count)
        return String(text.prefix(head)) + marker + String(text.suffix(tail))
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
        return (clamp(text), raw["persist"]?.boolValue ?? false)
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
    /// 超长便签的 LLM 压缩器（Agent 注入辅助模型）。nil 或压缩失败时退化保头保尾。
    public let compressor: ((String) async -> String?)?

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

    public init(config: AgentConfig, callbacks: AgentCallbacks?, registry: ToolRegistry? = nil,
                compressor: ((String) async -> String?)? = nil) {
        self.config = config
        self.callbacks = callbacks
        self.registry = registry
        self.compressor = compressor
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        do {
            let keyInfo = try ToolSupport.requireString(args, "key_info")
            let persist = ToolSupport.optionalBool(args, "persist") ?? false
            var text = keyInfo.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.count > CheckpointStore.maxKeyInfo {
                // 超长不直接截断丢信息：先请辅助模型压缩一次；压缩失败或
                // 仍超长时保头保尾（显式标注中间省略），损失可控可见。
                if let compressed = await compressor?(text),
                   !compressed.isEmpty, compressed.count <= CheckpointStore.maxKeyInfo {
                    text = compressed
                } else {
                    text = CheckpointStore.clamp(text)
                }
            }

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
/// `file_tools(op="patch")` 少量追加。门控：当前轮数 < `minTurns`(5) 直接拒绝，短任务不许碰长期记忆。
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
        + "memory/xxx_sop.md. Use file_tools(op=\"patch\") (small additions only), then sync the L1 index. "
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
        // L0/L1 常驻关闭时，模型看不到 L0 决策树：把分类要点内联进 payload，
        // 让记忆提纯在本模式下仍可独立完成（工具自身仍读 L1 全文，不受开关影响）。
        let hierarchyOn = config.memoryHierarchyInSystemPrompt
        let nextGuide = hierarchyOn
            ? "按 L0 决策树先 read 再 patch：事实 → memory/global_mem.txt，任务坑 → memory/xxx_sop.md，改完同步 L1；memory/ 禁止 write 覆盖"
            : "L0 未常驻，按此分类先 read 再 patch：验证过的环境事实 → memory/global_mem.txt；"
                + "难获的任务坑/SOP → memory/xxx_sop.md；偏好/实体 → long_memory(op=\"graph\")；"
                + "改完同步本返回的 l1_current（file_tools op=patch 少量追加，禁 write 覆盖）"
        // MEMORY_PURIFY_PROMPT 首句「按L0提取」在 L0 未常驻时模型看不到依据，
        // 替换为等价的自包含提纯指令（其余禁止项原样保留）。
        let purifyPrompt = hierarchyOn
            ? MEMORY_PURIFY_PROMPT
            : MEMORY_PURIFY_PROMPT.replacingOccurrences(
                of: "按L0提取", with: "按下方 next 的分类规则提取")

        let payload = JSONValue.object([
            "status": .string("DISTILL_READY"),
            "turn": .int(turn),
            "focus": .string(focus),
            "transcript": .string(String(transcript.suffix(1_800))),
            // 整份 L1 当前全文始终返回，供同步 patch 时精确匹配 old。
            "l1_current": .string(l1),
            "prompt": .string(purifyPrompt),
            "next": .string(nextGuide),
        ])
        return .ok(compactJSON(payload, maxChars: 4_000))
    }
}
