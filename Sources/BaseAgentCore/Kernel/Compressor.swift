import Foundation

// MARK: - 正则辅助

/// 内核共享的正则工具（与 Python 侧 compressor.py 的降噪/折叠规则一一对应）。
enum KernelRegex {
    static func search(_ pattern: String, _ text: String,
                       options: NSRegularExpression.Options = []) -> Bool {
        guard let re = try? NSRegularExpression(pattern: pattern, options: options) else {
            return false
        }
        return re.firstMatch(in: text, options: [],
                             range: NSRange(text.startIndex..., in: text)) != nil
    }

    static func replace(_ pattern: String, in text: String, with template: String,
                        options: NSRegularExpression.Options = []) -> String {
        guard let re = try? NSRegularExpression(pattern: pattern, options: options) else {
            return text
        }
        let range = NSRange(text.startIndex..., in: text)
        return re.stringByReplacingMatches(in: text, options: [], range: range,
                                           withTemplate: template)
    }

    static func matches(_ pattern: String, _ text: String,
                        options: NSRegularExpression.Options = []) -> [NSTextCheckingResult] {
        guard let re = try? NSRegularExpression(pattern: pattern, options: options) else {
            return []
        }
        return re.matches(in: text, options: [],
                          range: NSRange(text.startIndex..., in: text))
    }
}

// MARK: - 五级信息压缩（L0 → L4）

/// 协议标签（与 GenericAgent 一致）。
private let summaryRegex = "<summary>(.*?)</summary>"
private let codeFenceRegex = "```[\\s\\S]*?```"
private let protocolTagRegex = "<(thinking|think|tool_use|tool_result)>[\\s\\S]*?</\\1>"
/// 噪声标签：L1 降噪时整块剔除（保留内部文本）。
private let noiseHTMLRegex = "<(script|style|noscript|svg|head)\\b[\\s\\S]*?</\\1>"
/// HTML 标签白名单：只剥真正的 HTML 标签，保留 `std::vector<int>`、
/// `Map<String,Object>`、`a < b && c > d` 等源码/日志内容。
private let htmlTagRegex =
    "</?(?:div|span|p|a|br|img|table|tr|td|th|script|style|noscript|svg|head"
    + "|body|html|ul|ol|li|h[1-6]|pre|code|blockquote|em|strong|b|i|u|font"
    + "|form|input|button|label|section|article|header|footer|nav|main"
    + "|dl|dt|dd|caption|col|colgroup|thead|tbody|tfoot)\\b[^>]{0,200}>?"
private let multiBlankRegex = "(\\r?\\n){3,}"

/// 保头保尾的截断。错误/总结常在结尾，因此尾部必须留。
func headTail(_ text: String, maxChars: Int) -> String {
    if text.count <= maxChars {
        return text
    }
    let marker = "\n...[已裁剪 \(text.count) → \(maxChars) 字符；需完整内容请重新调用]\n"
    let head = maxChars / 2
    let tail = maxChars - head - marker.count
    if tail <= 0 {
        return String(text.prefix(maxChars))
    }
    return String(text.prefix(head)) + marker + String(text.suffix(tail))
}

// MARK: - L0：原始捕获

/// 工具返回的那一刻就上硬上限。
///
/// 工具内部通常已有自己的上限（file_read 200K、shell_run 200KB…），
/// 这里是内核侧的统一兜底，保证任何工具都不可能把 MB 级内容带进上下文。
public func l0Capture(_ raw: JSONValue?, hardCap: Int = 20_000) -> String {
    guard let raw else { return "" }
    let text: String
    if case .string(let s) = raw {
        text = s
    } else {
        text = raw.jsonString()
    }
    return headTail(text, maxChars: hardCap)
}

public func l0Capture(_ raw: String, hardCap: Int = 20_000) -> String {
    headTail(raw, maxChars: hardCap)
}

// MARK: - L1：工具级裁剪

/// 注入本轮前的裁剪 + 结构化降噪。
///
/// `maxChars` 由 `ContextBudget.remainingForTool()` 动态算出 —— 预算紧张时
/// 自动收紧，不用改任何工具代码。
public func l1Trim(_ text: String, maxChars: Int, denoise: Bool = true) -> String {
    if text.isEmpty { return "" }
    var s = text
    if denoise {
        s = KernelRegex.replace(noiseHTMLRegex, in: s, with: " ",
                                options: [.caseInsensitive])
        if s.contains("<"), s.contains(">") {
            s = KernelRegex.replace(htmlTagRegex, in: s, with: " ")
        }
        s = KernelRegex.replace(multiBlankRegex, in: s, with: "\n\n")
        s = s.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    return headTail(s, maxChars: maxChars)
}

/// 带工具名前缀的 L1 裁剪，便于模型区分多工具结果。
public func l1TrimResult(_ name: String, _ content: String, maxChars: Int) -> String {
    let body = l1Trim(content, maxChars: maxChars)
    return "<tool_result name=\(name)>\n\(body)\n</tool_result>"
}

// MARK: - L2：回合摘要

/// 抽取本轮的物理快照（≤30 字）。
///
/// 优先用模型按协议输出的 `<summary>`；缺失时回退到「剥掉代码/标签后的首句」，
/// 再不行就用工具名拼一句。**永不返回空串**。
public func l2Summary(_ content: String, toolNames: [String] = [], maxChars: Int = 60) -> String {
    if content.isEmpty {
        if !toolNames.isEmpty {
            return "调用 " + toolNames.prefix(3).joined(separator: ",")
        }
        return "（无文本输出）"
    }

    let matches = KernelRegex.matches(
        summaryRegex, content,
        options: [.dotMatchesLineSeparators, .caseInsensitive]
    )
    if let m = matches.first, m.numberOfRanges > 1,
       let range = Range(m.range(at: 1), in: content) {
        let s = String(content[range])
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\n", with: " ")
        if !s.isEmpty {
            return String(s.prefix(maxChars))
        }
    }

    var stripped = KernelRegex.replace(codeFenceRegex, in: content, with: " ")
    stripped = KernelRegex.replace(protocolTagRegex, in: stripped, with: " ",
                                   options: [.caseInsensitive])
    stripped = KernelRegex.replace("\\s+", in: stripped, with: " ")
        .trimmingCharacters(in: .whitespacesAndNewlines)
    if !stripped.isEmpty {
        // 取第一句：中文句末标点直接切；英文句点需跟空白，避免把 "1.5" 切断。
        let first = firstSentence(stripped)
        return String((first.isEmpty ? stripped : first).prefix(maxChars))
    }
    if !toolNames.isEmpty {
        return "调用 " + toolNames.prefix(3).joined(separator: ",")
    }
    return "（空回复）"
}

/// Python: `re.split(r"(?<=[。！？；!?])|(?<=[.])\s", s)[0]`。
private func firstSentence(_ text: String) -> String {
    let terminators: Set<Character> = ["。", "！", "？", "；", "!", "?"]
    for (offset, ch) in text.enumerated() {
        if terminators.contains(ch) {
            return String(text.prefix(offset + 1))
        }
    }
    if let range = text.range(of: ". ") {
        return String(text[..<range.lowerBound]) + "."
    }
    return text
}

/// 把 `<summary>` 从回复正文里剥掉（UI 展示 / 入库前用）。
public func stripSummaryTags(_ content: String) -> String {
    KernelRegex.replace(summaryRegex, in: content, with: "",
                        options: [.dotMatchesLineSeparators, .caseInsensitive])
        .trimmingCharacters(in: .whitespacesAndNewlines)
}

/// 模型是否**按协议**输出了 `<summary>`。
///
/// 不能用 `l2Summary` 的返回值判空：后者对任何非空 content 都保证返回非空。
public func hasSummaryTag(_ content: String) -> Bool {
    !content.isEmpty && KernelRegex.search(summaryRegex, content,
                                           options: [.dotMatchesLineSeparators, .caseInsensitive])
}

/// 流式过滤 `<summary>…</summary>`：把摘要从正文中分离，供 UI 放进思考区。
///
/// 模型可能把标签拆到多个 chunk 里，因此保留跨 chunk 缓冲；正文仍按原样
/// 即时输出，最多只暂存可能是标签前缀的尾部字符。
///
/// 块被剥离后，其前后换行若原样进正文，几十个块叠加就会在气泡里留下
/// 大段空白。因此这里用「空白悬挂」：每段正文的尾部空白先挂起，等下一个
/// 可见字符到来时收敛为最多一个空行（两个换行）再输出；若随后又是一个
/// summary 块或流结束，则整段丢弃。
public struct SummaryStreamFilter {
    private var pending = ""
    private var inSummary = false
    private var whitespaceHold = ""

    public init() {}

    public mutating func reset() {
        pending = ""
        inSummary = false
        whitespaceHold = ""
    }

    /// 输入一个流式 chunk，返回可展示的正文与摘要。
    public mutating func feed(_ chunk: String) -> (content: String, summary: String) {
        pending += chunk
        var content = ""
        var summary = ""

        while !pending.isEmpty {
            if inSummary {
                guard let range = pending.range(of: "</summary>",
                                                options: [.caseInsensitive]) else {
                    break
                }
                let inner = String(pending[..<range.lowerBound])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !inner.isEmpty {
                    summary += inner
                    if !summary.hasSuffix("\n") { summary += "\n" }
                }
                pending = String(pending[range.upperBound...])
                inSummary = false
            } else {
                guard let range = pending.range(of: "<summary>",
                                                options: [.caseInsensitive]) else {
                    // 暂存可能是 "<summary>" 前缀的尾部，避免标签被拆到两个 chunk。
                    let hold = Self.heldSuffixLength(pending)
                    if hold > 0 {
                        Self.route(String(pending.dropLast(hold)),
                                   into: &content, hold: &whitespaceHold)
                        pending = String(pending.suffix(hold))
                    } else {
                        Self.route(pending, into: &content, hold: &whitespaceHold)
                        pending = ""
                    }
                    break
                }
                // summary 块对正文不可见：块前空白整段丢弃，块前正文的尾部
                // 悬挂空白也一并丢弃，只保留可见文字。
                let before = String(pending[..<range.lowerBound])
                Self.route(before, into: &content, hold: &whitespaceHold)
                whitespaceHold = ""
                pending = String(pending[range.upperBound...])
                inSummary = true
            }
        }

        return (content, summary)
    }

    /// 流结束：未闭合的摘要按思考内容输出，其余按正文输出（不含尾部空白，
    /// 避免气泡末尾残留空行）。
    public mutating func finish() -> (content: String, summary: String) {
        var content = ""
        var summary = ""
        if inSummary {
            let inner = pending.trimmingCharacters(in: .whitespacesAndNewlines)
            if !inner.isEmpty {
                summary += inner
                if !summary.hasSuffix("\n") { summary += "\n" }
            }
        } else {
            content = pending.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        pending = ""
        inSummary = false
        whitespaceHold = ""
        return (content, summary)
    }

    /// 把一段正文并入输出：尾部空白挂起；此前悬挂的空白在下一个可见字符
    /// 到来时收敛为最多两个换行后一并输出。
    static func route(_ text: String, into content: inout String, hold: inout String) {
        guard !text.isEmpty else { return }
        var visible = Substring(text)
        while let last = visible.last, last.isWhitespace { visible = visible.dropLast() }
        if visible.isEmpty {
            hold += text
            return
        }
        content += String(hold.filter { $0 == "\n" }.prefix(2)) + visible
        hold = String(text.suffix(text.count - visible.count))
    }

    private static func heldSuffixLength(_ text: String) -> Int {
        let tag = "<summary>"
        let lower = text.lowercased()
        let limit = min(tag.count, lower.count)
        guard limit > 0 else { return 0 }
        for length in stride(from: limit, through: 1, by: -1) {
            if tag.hasPrefix(String(lower.suffix(length))) {
                return length
            }
        }
        return 0
    }
}

// MARK: - L3：会话折叠

/// 把摘要行折叠成 `<earlier_context>` 块。
///
/// 连续相同的摘要合并为 `（N turns）` —— 长任务里模型反复重试同一动作时，
/// 几十轮会塌缩成一行。
public func l3Fold(_ lines: [String], maxLines: Int = 40,
                   dedupeWindow: Int = 40) -> String {
    guard !lines.isEmpty else { return "" }
    let window = Array(lines.suffix(dedupeWindow))
    var folded: [String] = []
    var cnt = 0
    var last = ""
    var lastKey = ""
    for line in window {
        let key = foldKey(line)
        if key == lastKey && !key.isEmpty {
            cnt += 1
            continue
        }
        if cnt > 0 {
            folded.append("\(last)（\(cnt + 1) turns）")
        } else if !last.isEmpty {
            folded.append(last)
        }
        last = line
        lastKey = key
        cnt = 0
    }
    if !last.isEmpty {
        folded.append(cnt > 0 ? "\(last)（\(cnt + 1) turns）" : last)
    }
    return folded.suffix(maxLines).joined(separator: "\n")
}

/// 剥掉 `[T{n}] ` 轮次前缀后的正文，用于判断连续摘要是否真的相同。
private func foldKey(_ line: String) -> String {
    guard line.hasPrefix("[T"), let range = line.range(of: "] ") else { return line }
    return String(line[range.upperBound...])
}

// MARK: - L4：蒸馏提示词

/// 记忆提纯器的系统提示（从 Python 原样迁移）。
public let DISTILL_SYSTEM = """
你是记忆提纯器。只保留【经工具调用验证成功、且长期有效】的信息：
- 环境事实（路径、端口、版本、配置值）
- 用户偏好与红线
- 踩坑点与避坑方法
禁止写入：临时状态、推理过程、未验证信息、通用常识、易重建细节、模型推测（不得记作用户要求）。
以更少文字保留更高复用价值；无提纯价值则输出空列表。
"""

/// L4 蒸馏提示词模板（从 Python 原样迁移）。
///
/// Python 侧用 `DISTILL_PROMPT.format(transcript=...)`，因此源码里的 `{{` / `}}`
/// 在最终提示中会变成单层 JSON 花括号；这里直接保存格式化后的文本，
/// `{transcript}` 为唯一占位符，由 `makeDistillPrompt` 替换。
public let DISTILL_PROMPT = """
把下面这一段任务记录提纯为结构化条目，输出 JSON，不要任何前言：

{"facts": ["..."], "lessons": ["..."], "skill": {"name": "", "summary": "", "when_to_use": "", "steps": ""}}

- facts：验证过的环境事实/用户偏好（路径、端口、版本、配置值），每条 ≤ 30 字，最多 5 条
- lessons：踩坑与避坑方法，每条 ≤ 40 字，最多 3 条
- skill：若该流程**明确可复用**（同类任务下次能照做），给出
  - name：英文小写连字符 id，如 extract-pdf-tables
  - summary：一句话说明做什么（≤ 50 字）
  - when_to_use：触发场景（≤ 40 字）
  - steps：编号步骤（≤ 600 字，只写验证过的步骤）
  否则 skill.name 留空字符串
- 没有内容就输出空列表/空字符串，不要编造

任务记录：
{transcript}
"""

public func makeDistillPrompt(transcript: String) -> String {
    DISTILL_PROMPT.replacingOccurrences(of: "{transcript}", with: String(transcript.prefix(4000)))
}

// MARK: - 助手：把任意内容压到指定 token 数以内

/// 把文本压到 `maxTokens` 以内（先降噪，再 head+tail）。
public func fitTokens(_ text: String, maxTokens: Int) -> String {
    if text.isEmpty { return "" }
    if TokenEstimator.estimate(text) <= maxTokens {
        return text
    }
    return l1Trim(text, maxChars: max(200, maxTokens * 3))
}

/// 把 dict/list 压成紧凑字符串（工作记忆注入系统提示时用）。
public func compactJSON(_ obj: JSONValue, maxChars: Int = 400) -> String {
    let s = obj.jsonString()
    return s.count <= maxChars ? s : String(s.prefix(maxChars)) + "…"
}

// MARK: - 基于 LLMService 的异步摘要 / 蒸馏

/// 摘要与蒸馏接口。LLM 失败时一律回退到确定性折叠，保证信息不会凭空消失。
public final class Compressor {
    private static let summarySystem = "你是上下文压缩器。只保留继续任务所需的关键事实、决策、路径与结论，压缩为简洁摘要，不要前言。"

    /// L4 蒸馏：调用 LLM 提纯 facts / lessons / skill；失败或输出不可解析时
    /// 用确定性折叠兜底（至少保留轨迹的头部与尾部）。
    public static func distill(transcript: String,
                               llm: LLMService,
                               temperature: Double = 0.2) async -> DistillResult {
        guard !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return DistillResult()
        }
        let prompt = makeDistillPrompt(transcript: transcript)
        do {
            let raw = try await complete(
                messages: [
                    ChatMessage(role: .system, content: DISTILL_SYSTEM),
                    ChatMessage(role: .user, content: prompt),
                ],
                llm: llm, temperature: temperature
            )
            let result = l4Normalize(l4ExtractJSON(raw))
            if !result.facts.isEmpty || !result.lessons.isEmpty || !result.skill.isEmpty {
                return result
            }
            return deterministicDistill(transcript)
        } catch {
            return deterministicDistill(transcript)
        }
    }

    /// 上下文兜底摘要：优先 LLM（temperature=0.2），失败时用确定性折叠。
    /// 返回文本不会为空（输入非空时至少保留 head+tail）。
    public static func summarize(_ transcript: String,
                                 llm: LLMService?,
                                 maxChars: Int = 600,
                                 temperature: Double = 0.2) async -> String {
        let trimmed = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        let limit = max(200, maxChars)
        if let llm {
            do {
                let raw = try await complete(
                    messages: [
                        ChatMessage(role: .system, content: summarySystem),
                        ChatMessage(role: .user, content: String(trimmed.prefix(8000))),
                    ],
                    llm: llm, temperature: temperature
                )
                let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty {
                    return headTail(text, maxChars: limit)
                }
            } catch {
                // 落到确定性折叠。
            }
        }
        let folded = l3Fold(trimmed.components(separatedBy: .newlines), maxLines: 40)
        return headTail(folded.isEmpty ? trimmed : folded, maxChars: limit)
    }

    // MARK: 内部

    private static func complete(messages: [ChatMessage], llm: LLMService,
                                 temperature: Double) async throws -> String {
        // 逐 chunk 字符串拼接是 O(n²)；先用数组收集，结束后一次性合并。
        var chunks: [String] = []
        let stream = llm.chatStream(messages: messages, tools: [], temperature: temperature)
        loop: for try await event in stream {
            switch event.kind {
            case .content:
                chunks.append(event.content)
            case .reasoning:
                continue
            case .done:
                // break 只会跳出 switch；这里必须显式跳出整个 for-await。
                break loop
            }
        }
        return chunks.joined()
    }

    private static func deterministicDistill(_ transcript: String) -> DistillResult {
        let lines = transcript.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let folded = l3Fold(lines, maxLines: 40)
        // 没有结构化输出时也不能把轨迹丢光：保留前若干行作为 facts，
        // 折叠后的轨迹（≤ 600 字）作为一条 lesson，供后续人工/模型再提纯。
        let facts = lines.prefix(5).map { String($0.prefix(120)) }
        let lesson = String(folded.prefix(600))
        return DistillResult(
            facts: facts,
            lessons: lesson.isEmpty ? [] : [lesson],
            skill: DistilledSkill()
        )
    }
}
