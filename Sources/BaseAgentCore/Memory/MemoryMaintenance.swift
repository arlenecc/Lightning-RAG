import Foundation

/// 改进7（历史语义提炼）与改进8（记忆整理）的后台执行体。
///
/// 两者都**只调用辅助引擎**（`collectChat`），在 `Agent.fireFinished` 的
/// `Task.detached(.utility)` 链里串接执行，不与主模型争前台 KV；若主模型
/// 不支持独立会话，调用方会跳过整个后台链，这里不会被触发。
///
/// 所有文件写入都遵守 META-SOP 的「只 patch、禁 overwrite、失败宁可不改」：
/// - 历史提炼产物 `history_insight.md` 是**追加**（只增不改的情绪事件）与
///   「活动知识层」的覆盖式写回，由辅助模型自行输出完整新状态；
/// - L1 整理结果先做 ≤30 行 / 无密钥 / 非空校验，再 `backup` + 写入；
///   校验不过直接丢弃，绝不落盘。
public final class MemoryMaintenance: @unchecked Sendable {

    private let dir: MemoryDirectory
    private let log: (String) -> Void
    /// 辅助模型一次性调用（复用 Agent.collectChat，已处理 isolated/cancel）。
    private let collect: ([ChatMessage], Double) async -> String

    public init(directory: MemoryDirectory,
                log: @escaping (String) -> Void,
                collect: @escaping ([ChatMessage], Double) async -> String) {
        self.dir = directory
        self.log = log
        self.collect = collect
    }

    // MARK: 改进7 —— 历史重点挖掘

    /// 对**持久化队列**里的待处理 session 做语义提炼，增量更新 `history_insight.md`。
    ///
    /// 队列由 `archivePastSessions()` 在删原文前入队、提炼成功后才出队，
    /// 因此寒暄轮 / 后台任务被下一次 run 取消 / 辅助模型暂不可用都不会丢数据：
    /// 下次任意一轮会重新拾起未完成项。数据源是 `all_history.txt`（骨架），
    /// 不依赖本次 `archives` 内存参数。
    public func mineHistoryInsights() async {
        let pending = dir.pendingInsightIds()
        guard !pending.isEmpty else { return }
        guard let fullHistory = dir.read(dir.allHistoryFile, limit: 2_000_000),
              !fullHistory.isEmpty else { return }

        let previous = dir.read(dir.historyInsightFile, limit: 200_000) ?? ""
        let lastMark = Self.lastSessionMarker(in: previous)
        // 只截取队列中尚未处理的骨架，按 session 边界对齐，不拆分同一 session。
        // 增量由「持久化队列」判定（allowed 集合），不依赖 id 字符串大小比较：
        // `MMdd_HHmmss_NNN` 跨年时字典序不可靠，且 lastMark 只作为给模型的提示。
        let increment = Self.skeletonAfter(nil, in: fullHistory, allowed: Set(pending))
        guard !increment.isEmpty else {
            // 队列里的 id 在 all_history 找不到（异常）：出队避免永久卡住。
            dir.clearPendingInsights(pending)
            return
        }

        let prompt = """
        按 system 中的《历史重点挖掘 SOP》执行本次增量扫描。

        --- 已处理到的 session 标识 ---
        \(lastMark.isEmpty ? "(无，首次运行)" : lastMark)

        --- 已有产物（活动知识层 / 情绪事件 / 增量标记，需读取并更新写回） ---
        \(String(previous.prefix(40_000)))

        --- 本次新增的历史 USER/Agent 骨架（含 session 标识） ---
        \(String(increment.prefix(60_000)))

        --- 输出要求 ---
        直接输出更新后的 `history_insight.md` 全文（纯文本，不要 markdown 代码围栏）：
        1. 「持续活动」区域 2. 「已消失事项」区域（必须两个独立区域）
        3. 「情绪事件」追加列表 4. 末尾一行严格为：
           <!-- LAST_SESSION: <本次处理到的最后一个 session 标识> -->
        每条发现必须含 session 标识、可 grep 的关键原文片段、发现类型标签。
        """
        let raw = await collect([
            ChatMessage(role: .system, content: "你是历史对话重点挖掘器。" + HISTORY_INSIGHT_SOP),
            ChatMessage(role: .user, content: prompt),
        ], 0.2)
        let cleaned = Self.stripFence(raw)
        guard !cleaned.isEmpty, cleaned.contains("LAST_SESSION:") else {
            log("[memory] 历史提炼输出不合规，已丢弃（不改动产物）")
            return
        }
        dir.backup(dir.historyInsightFile)
        guard dir.write(cleaned, to: dir.historyInsightFile) else {
            log("[memory] history_insight.md 写入失败，队列保留待重试")
            return
        }
        // 只在产物落盘成功后出队；失败/取消都保留，下一轮重试。
        dir.clearPendingInsights(pending)
        log("[memory] 历史提炼完成：处理 \(pending.count) 个 session，队列已出队")
    }

    // MARK: 改进8 —— 辅助引擎执行「记忆整理 SOP」

    /// 用辅助模型整理 L1/L2/L3，避免主引擎卡顿。
    ///
    /// **绝不自动改写记忆文件**：META-SOP 红线「L1 只能 patch 词级别修改，禁
    /// overwrite」，且覆盖会改 mtime → system 前缀变 → KV 全量失效。辅助引擎只
    /// 产出「整理建议 + 建议的 L1 全文」到 `memory_tidy_proposal.md`，由模型在
    /// `start_long_term_update` 门控后用 `file_tools(op="patch")` 逐词修改；无人复核不落盘。
    public func tidyMemory() async -> Bool {
        let l1 = dir.read(dir.l1File, limit: 8_000) ?? ""
        let l2 = dir.read(dir.l2File, limit: 20_000) ?? ""
        let l3Names = dir.list(dir.root)
            .filter { $0.hasSuffix("_sop.md") || $0.hasSuffix(".py") }
            .sorted().prefix(40).joined(separator: "\n")
        guard !l1.isEmpty else { return false }

        let prompt = """
        按 system 中的《记忆整理 SOP》执行本次整理。

        --- 当前 L1 (global_mem_insight.txt) ---
        \(l1)

        --- 当前 L2 (global_mem.txt) 开头 ---
        \(String(l2.prefix(12_000)))

        --- 当前 L3 文件名清单 ---
        \(l3Names)

        --- 输出要求 ---
        分两部分，用固定分隔行隔开：
        === L1 ===
        整理后的 L1 **全文**（纯文本；≤30 行；只保留存在性指针与高 ROI RULES；
        删翻译/内容描述/实现细节/冗余；保留原 [RULES] 标号体系）。
        === PROPOSAL ===
        L2 瘦身与 L3 审计的**建议清单**（纯文本，逐条写清改哪个文件、建议的
        file_tools(op="patch", old, new) 是什么；本部分不会自动执行，仅供后续按门控少量 patch）。
        """
        let raw = await collect([
            ChatMessage(role: .system, content: "你是记忆整理器。" + MEMORY_TIDY_SOP),
            ChatMessage(role: .user, content: prompt),
        ], 0.2)
        let cleaned = Self.stripFence(raw)
        guard let parts = Self.splitTidyOutput(cleaned) else {
            log("[memory] L1 整理输出格式不合规，已丢弃")
            return false
        }
        guard Self.validL1(parts.l1) else {
            log("[memory] L1 整理结果校验失败（>30行/空/疑似密钥），已丢弃")
            return false
        }
        // 只写建议文件（非记忆文件）；L1/L2/L3 一律不动，避免持久性伤害与 KV 抖动。
        let report = """
        # 记忆整理建议（辅助引擎产出，未自动执行）

        > 按 META-SOP：L1 只能 patch 词级别修改，禁 overwrite。
        > 请经 start_long_term_update 门控后，用 file_tools(op="patch") 逐条核对再修改。
        > 生成时间：\(MemoryDirectory.timestamp())

        ## 建议的 L1 全文（仅供对照，不要整文件覆盖）
        \(parts.l1)

        ## L2 / L3 整理建议
        \(parts.proposal)
        """
        guard dir.write(report, to: dir.tidyProposalFile) else { return false }
        log("[memory] 记忆整理建议已生成（L1 建议 \(parts.l1.split(separator: "\n").count) 行），"
            + "记忆文件未改动，见 memory_tidy_proposal.md")
        return parts.l1 != l1
    }

    // MARK: 校验 / 解析辅助

    /// 校验「建议的 L1 全文」是否可作为后续 patch 的参考（不落盘，但避免把明显
    /// 有害的建议交给模型）。口径与 `MemoryDirectory.loadL1` 一致：
    /// `omittingEmptySubsequences: false`，否则 30 非空行 + 空行会被注入侧截掉尾部。
    private static func validL1(_ text: String) -> Bool {
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        guard !lines.isEmpty, lines.count <= 30 else { return false }
        let banned = ["password=", "api_key=", "secret=", "-----BEGIN "]
        let lower = text.lowercased()
        guard banned.allSatisfy({ !lower.contains($0.lowercased()) }) else { return false }
        // 存在性指针与 RULES 不可丢（META-SOP「存在性不可丢」）。
        return text.contains("[RULES]") || text.contains("RULES")
            || text.contains("global_mem.txt") || text.contains("raw_session")
    }

    private static func splitTidyOutput(_ text: String) -> (l1: String, proposal: String)? {
        guard let l1Range = text.range(of: "=== L1 ==="),
              let propRange = text.range(of: "=== PROPOSAL ==="),
              l1Range.upperBound < propRange.lowerBound else { return nil }
        let l1 = String(text[l1Range.upperBound..<propRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let proposal = String(text[propRange.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !l1.isEmpty else { return nil }
        return (l1, proposal)
    }

    private static func lastSessionMarker(in text: String) -> String {
        var last = ""
        for line in text.split(separator: "\n") {
            let s = String(line)
            if s.hasPrefix("<!-- LAST_SESSION:"), let end = s.range(of: "-->") {
                let value = s[s.index(s.startIndex, offsetBy: "<!-- LAST_SESSION:".count)..<end.lowerBound]
                last = value.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        return last
    }

    /// 截取增量标记之后的 session 骨架，且只保留本次归档的 session，保证不拆分同一 session。
    private static func skeletonAfter(_ marker: String?, in history: String,
                                      allowed: Set<String>) -> String {
        var kept = ""
        var currentId: String?
        var current = ""
        var include = false
        func flush() {
            if include, let id = currentId, allowed.contains(id) {
                kept += "@@SESSION \(id)\n\(current)@@SESSION-END\n\n"
            }
            current = ""
        }
        for line in history.split(separator: "\n", omittingEmptySubsequences: false) {
            let s = String(line)
            if s.hasPrefix("@@SESSION ") && !s.hasPrefix("@@SESSION-END") {
                flush()
                currentId = String(s.dropFirst("@@SESSION ".count))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                include = true
                continue
            }
            if s.hasPrefix("@@SESSION-END") {
                flush()
                currentId = nil
                include = false
                continue
            }
            if include { current += s + "\n" }
        }
        flush()
        return kept
    }

    private static func stripFence(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("```") {
            s = s.split(separator: "\n", omittingEmptySubsequences: false).dropFirst().joined(separator: "\n")
            if let r = s.range(of: "```", options: .backwards) { s = String(s[..<r.lowerBound]) }
        }
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
