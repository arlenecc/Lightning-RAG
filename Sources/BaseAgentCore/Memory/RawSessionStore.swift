import Foundation
import ZIPFoundation

/// 改进 1 / 2 / 7：历史会话层（L4）的落盘、摘要追加与过期归档。
///
/// ## 落盘格式（每天一个文件，追加）
/// 一个「用户请求」= 一个 SESSION 块。块内用 `@@` 标记分段，改进7 归档时
/// 按标记剥离 `@@SYSTEM` 与 `@@ASSISTANT-RESPONSE`/`@@TOOL` 段，
/// 只保留 USER / AGENT 对话骨架。
///
/// ```
/// @@SESSION-BEGIN 1105_0930 | 2026-11-05 09:30:12
/// @@SYSTEM
/// <本轮实际发给模型的 system prompt 全文>
/// @@USER
/// <用户原文>
/// @@ASSISTANT-RESPONSE turn=1
/// <中间轮助手正文（含 summary / tool_calls）>
/// @@TOOL file_read
/// <工具结果全文>
/// @@AGENT
/// <给用户最终答复>
/// @@SESSION-END 1105_0930 | 2026-11-05 09:31:02
/// ```
///
/// 用 `@@` 而非 `---` 作标记，是因为 META-SOP 正文里本身有 `---` 分隔线，
/// 会与分段标记冲突。
///
/// ## 归档（改进7）
/// 对今天之前的 `<date>.txt`：按 SESSION 块切分 → 剥离 → 骨架追加进
/// `all_history.txt` → 每个 session 单独写进 `<date>.zip`（zip 内文件名即
/// session 标识，可 grep 回溯）→ 打包成功后才删除原 txt 与 digest。
public struct RawSessionArchive {
    public let day: String
    public let sessionIds: [String]
    public let archivedBytes: Int
}

public final class RawSessionStore: @unchecked Sendable {

    private let dir: MemoryDirectory
    private let fm = FileManager.default

    public init(directory: MemoryDirectory) {
        self.dir = directory
    }

    // MARK: 改进1/2 —— 每用户轮落盘

    /// 把一次用户请求的完整轨迹追加进 `<date>.txt`，各轮摘要追加进
    /// `<date>-digest.txt`。**无条件执行**：不依赖辅助模型、不受事实抽取闸门影响。
    ///
    /// - `systemText`：本轮实际注入的 system prompt（L0+L1+BASE），归档时可剥离。
    /// - `userText`：用户原文（`Agent.lastUserText`），非 anchorPrompt 加工后的 prompt。
    /// - `turns`：本次请求产生的轮（prompt/assistantText/toolResults/summary）。
    /// - `finalText`：最终答复（作为 `@@AGENT` 段）。
    @discardableResult
    public func appendSession(systemText: String,
                              userText: String,
                              turns: [TurnRecord],
                              finalText: String,
                              date: Date = Date()) -> String {
        let day = MemoryDirectory.dayString(date)
        let beginId = MemoryDirectory.sessionId(date)
        var out = ""
        out += "@@SESSION-BEGIN \(beginId) | \(MemoryDirectory.timestamp(date))\n"
        // 同一天只写一次完整 SYSTEM 全文，后续 session 用引用行，
        // 避免 5k token 的 L0+L1 每天重复 N 份（归档时 SYSTEM 段本就要剥离）。
        let systemHash = String(format: "%016llx", Self.stableHash(systemText))
        let systemRefFile = dir.path("raw_session/\(day)-system.txt")
        if !dir.exists(systemRefFile) {
            _ = dir.write(systemText, to: systemRefFile)
        }
        out += "@@SYSTEM \(systemHash) -> \(day)-system.txt\n"

        // 用户原文只在块首写一次；后续轮的 prompt 是内核加工过的 anchorPrompt，
        // 属于"系统提示段"，不重复写入（避免污染 USER 骨架）。
        out += "@@USER\n\(Self.escapeBody(userText))"

        for turn in turns {
            // 每轮发给模型的 user 消息（anchorPrompt 全文）也完整入库，
            // 保证".txt = 历史对话全文"；归档时按 @@ 标记剥离，只留骨架。
            if !turn.prompt.isEmpty {
                out += "@@TURN-PROMPT turn=\(turn.index)\n\(Self.escapeBody(turn.prompt))"
            }
            if !turn.assistantText.isEmpty {
                out += "@@ASSISTANT-RESPONSE turn=\(turn.index)\n\(Self.escapeBody(turn.assistantText))"
            }
            for result in turn.toolResults {
                out += "@@TOOL \(result.name)\n\(Self.escapeBody(result.content))"
            }
        }
        if !finalText.isEmpty {
            out += "@@AGENT\n\(Self.escapeBody(finalText))"
        }
        out += "@@SESSION-END \(beginId) | \(MemoryDirectory.timestamp())\n\n"

        dir.append(out, to: dir.rawFile(day: day))

        // 改进2：全部轮次摘要（含折叠摘要区）追加，带 session 标识便于关联。
        var digest = "@@SESSION \(beginId)\n"
        for line in turns.map({ $0.digestLine() }) {
            digest += Self.escapeBody(line) + "\n"
        }
        digest += "@@SESSION-END\n\n"
        dir.append(digest, to: dir.digestFile(day: day))
        return beginId
    }

    /// 正文里以 `@@` 开头的行会被误判成分段标记（用户粘贴日志极常见），
    /// 导致 SESSION 提前闭合、工具输出混进骨架。统一加 `@@LITERAL ` 前缀转义，
    /// 解析时 `unescapeLine` 还原；可逆、仍可 grep 定位原文。
    static func escapeBody(_ text: String) -> String {
        guard !text.isEmpty else { return "\n" }
        let escaped = text.split(separator: "\n", omittingEmptySubsequences: false)
            .map { line -> String in
                let s = String(line)
                return s.hasPrefix("@@") ? "@@LITERAL " + s : s
            }
            .joined(separator: "\n")
        return escaped.hasSuffix("\n") ? escaped : escaped + "\n"
    }

    static func unescapeLine(_ line: String) -> String {
        line.hasPrefix("@@LITERAL ") ? String(line.dropFirst("@@LITERAL ".count)) : line
    }

    private static func stableHash(_ text: String) -> UInt64 {
        var h: UInt64 = 14_695_981_039_346_656_037
        for byte in text.utf8 { h ^= UInt64(byte); h &*= 1_099_511_628_211 }
        return h
    }

    /// 追加 L3 折叠历史摘要（CONSOLIDATE 时产生的 extraDigest 行）。
    /// 与 `appendSession` 的逐轮摘要合并进同一个 digest 文件，保证"所有轮次"不丢。
    public func appendFoldedDigest(_ lines: [String], date: Date = Date()) {
        guard !lines.isEmpty else { return }
        let day = MemoryDirectory.dayString(date)
        let body = "@@FOLDED \(MemoryDirectory.timestamp(date))\n"
            + lines.joined(separator: "\n") + "\n@@FOLDED-END\n\n"
        dir.append(body, to: dir.digestFile(day: day))
    }

    // MARK: 改进7 —— 过期日期预处理 + 打包

    /// 处理"今天之前"的原始记录。**每次最多归档 1 天**，把耗时/内存封顶，
    /// 避免多天积压时在主请求 `return` 前做 N 次全量读 + zip 卡住回复。
    /// 剩余的日期由后续用户轮依次处理（幂等）。
    @discardableResult
    public func archivePastSessions(today: Date = Date()) -> [RawSessionArchive] {
        let todayString = MemoryDirectory.dayString(today)
        let names = dir.list(dir.rawSessionDir)
            .filter { Self.isDayFile($0) }
            .filter { String($0.prefix(10)) < todayString }
            .sorted()
            .prefix(1)

        var results: [RawSessionArchive] = []
        for name in names {
            let day = String(name.prefix(10))
            let zip = dir.archiveFile(day: day)
            // 上次打包成功但删原文前被中断：补齐入队 + 清理残留，不重复打包。
            if dir.exists(zip) {
                let residual = Self.splitSessions(
                    dir.read(dir.rawFile(day: day), limit: 20_000_000) ?? "")
                dir.enqueuePendingInsights(residual.map(\.sessionId))
                dir.remove(dir.rawFile(day: day))
                dir.remove(dir.digestFile(day: day))
                continue
            }
            guard let raw = dir.read(dir.rawFile(day: day), limit: 20_000_000) else { continue }

            let blocks = Self.splitSessions(raw)
            guard !blocks.isEmpty else { continue }

            var skeleton = ""
            var ids: [String] = []
            var staging: [(name: String, body: String)] = []

            for block in blocks {
                let id = block.sessionId
                ids.append(id)
                let clean = Self.stripToSkeleton(block.body)
                skeleton += "@@SESSION \(id)\n\(clean)@@SESSION-END\n\n"
                staging.append((name: id + ".txt", body: "@@SESSION \(id)\n\(clean)@@SESSION-END\n"))
            }
            // 摘要也一并入包，保留可追溯性。
            if let digest = dir.read(dir.digestFile(day: day), limit: 5_000_000), !digest.isEmpty {
                staging.append((name: "\(day)-digest.txt", body: digest))
            }

            // 顺序即安全性：zip 成功 → all_history 追加成功 → 入队成功 → 才删原文。
            // 任何一步失败都保留原始 .txt/-digest.txt，下次可重试，绝不静默丢数据。
            guard Self.zip(staging, to: zip), Self.zipHasEntries(zip, count: staging.count)
            else {
                dir.remove(zip)
                continue
            }
            guard dir.append(skeleton, to: dir.allHistoryFile) else {
                dir.remove(zip)
                continue
            }
            dir.enqueuePendingInsights(ids)
            let bytes = raw.utf8.count
            dir.remove(dir.rawFile(day: day))
            dir.remove(dir.digestFile(day: day))
            results.append(RawSessionArchive(day: day, sessionIds: ids, archivedBytes: bytes))
        }
        return results
    }

    private static func isDayFile(_ name: String) -> Bool {
        guard name.hasSuffix(".txt"), name.count == 14 else { return false }
        let stem = String(name.prefix(10))
        return stem.count == 10 && stem.split(separator: "-").count == 3
            && stem.allSatisfy({ $0.isNumber || $0 == "-" })
    }

    /// 按 `@@SESSION-BEGIN` / `@@SESSION-END` 切块；未闭合的尾部也算一块（异常中断保护）。
    public static func splitSessions(_ raw: String) -> [(sessionId: String, body: String)] {
        var result: [(String, String)] = []
        var currentId: String?
        var current: [String] = []
        for line in raw.split(separator: "\n", omittingEmptySubsequences: false) {
            let s = String(line)
            if s.hasPrefix("@@SESSION-BEGIN ") {
                if let id = currentId { result.append((id, current.joined(separator: "\n") + "\n")) }
                let rest = String(s.dropFirst("@@SESSION-BEGIN ".count))
                currentId = String(rest.split(separator: " ").first ?? Substring(rest))
                current = []
                continue
            }
            if s.hasPrefix("@@SESSION-END") {
                if let id = currentId {
                    result.append((id, current.joined(separator: "\n") + "\n"))
                    currentId = nil
                    current = []
                }
                continue
            }
            if currentId != nil { current.append(s) }
        }
        if let id = currentId { result.append((id, current.joined(separator: "\n") + "\n")) }
        return result
    }

    /// 剥离 SYSTEM 段、ASSISTANT-RESPONSE 段、TOOL 结果段；
    /// 仅保留 USER / AGENT 对话骨架与会话边界标记。
    public static func stripToSkeleton(_ body: String) -> String {
        var kept: [String] = []
        var skipping = false
        for line in body.split(separator: "\n", omittingEmptySubsequences: false) {
            let s = String(line)
            if s.hasPrefix("@@LITERAL ") {
                // 转义行是正文，不是分段标记；按当前 skipping 状态决定是否保留。
                if !skipping { kept.append(Self.unescapeLine(s)) }
                continue
            }
            if s.hasPrefix("@@") {
                if s.hasPrefix("@@SYSTEM") || s.hasPrefix("@@ASSISTANT-RESPONSE")
                    || s.hasPrefix("@@TOOL ") || s.hasPrefix("@@TURN-PROMPT") {
                    skipping = true
                    continue
                }
                if s.hasPrefix("@@USER") || s.hasPrefix("@@AGENT") {
                    skipping = false
                    kept.append(s)
                    continue
                }
                // 其它 @@ 标记（折叠摘要等）：不保留内容，但结束跳过态
                skipping = true
                continue
            }
            if !skipping { kept.append(s) }
        }
        // 压掉连续空行，保持骨架紧凑。
        var compact: [String] = []
        for line in kept {
            if line.isEmpty, compact.last?.isEmpty == true { continue }
            compact.append(line)
        }
        return compact.joined(separator: "\n")
    }

    /// 校验 zip 条目数与预期一致（打包成功但条目丢失时不能删原文）。
    private static func zipHasEntries(_ path: String, count: Int) -> Bool {
        guard let archive = try? Archive(url: URL(fileURLWithPath: path), accessMode: .read)
        else { return false }
        var found = 0
        for _ in archive { found += 1; if found >= count { return true } }
        return false
    }

    /// 用 ZIPFoundation 把各 session 片段打进一个 zip（zip 内文件名即 session 标识，
    /// 满足 SOP 的 grep 回溯要求）。相比外部 `/usr/bin/zip`，无子进程开销、不依赖系统命令。
    private static func zip(_ entries: [(name: String, body: String)], to path: String) -> Bool {
        guard !entries.isEmpty else { return false }
        let fm = FileManager.default
        let url = URL(fileURLWithPath: path)
        try? fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        // 先写临时文件再 rename，避免中途崩溃留下半截 zip 被误判为"已归档"。
        let tmp = path + ".tmp"
        try? fm.removeItem(atPath: tmp)
        do {
            guard let archive = Archive(url: URL(fileURLWithPath: tmp), accessMode: .create)
            else { return false }
            for entry in entries {
                let data = Data(entry.body.utf8)
                try archive.addEntry(with: entry.name, type: .file,
                                     uncompressedSize: UInt32(data.count),
                                     provider: { (position: Int, size: Int) -> Data in
                                         let start = Int(position)
                                         let end = min(start + Int(size), data.count)
                                         return data.subdata(in: start..<end)
                                     })
            }
            try? fm.removeItem(atPath: path)
            try fm.moveItem(atPath: tmp, toPath: path)
            return fm.fileExists(atPath: path)
        } catch {
            try? fm.removeItem(atPath: tmp)
            return false
        }
    }
}
