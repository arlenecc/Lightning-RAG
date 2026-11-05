import CryptoKit
import Foundation

// MARK: - 技能模型

/// 一个已发现的技能。
public struct Skill: Sendable, Equatable, Identifiable {
    public var name: String
    public var description: String
    public var tags: [String]
    public var keywords: [String]
    /// 相对扫描根目录的路径（根目录内的技能）；外部软链接时为解析后的绝对路径。
    public var directory: String
    /// 技能目录绝对路径。
    public var absolutePath: String
    /// 入口文件名，通常为 `SKILL.md`（兼容旧的 `prompt.md`）。
    public var entry: String
    public var whenToUse: String
    public var whenNotToUse: String
    public var examples: String
    /// 内容摘要（SHA256），用于检索索引增量更新。
    public var fingerprint: String
    /// 入口文件全文（最多 200k 字符）。
    public var body: String

    public var id: String { absolutePath }
}

// MARK: - 技能管理

/// 技能发现、加载与保存。
///
/// - 扫描 `workspace/skills` 与调用方传入的内置技能目录（递归、跟随软链接、防环）；
/// - 解析 `SKILL.md` YAML frontmatter 的 `name` / `description` / `tags`，
///   并识别正文里的 `When to use` / `When not to use` / `Examples` 小节；
/// - 同时兼容旧式目录：`skill.json` + `prompt.md`。
public final class SkillManager {
    /// 主技能目录（`<workspace>/skills`），`save` 写入此处。
    public let skillsDirectory: String
    /// 额外的只读技能根目录（内置技能）。
    public let builtinDirectories: [String]

    private let lock = NSRecursiveLock()
    private var cached: [Skill] = []
    private var requestCounts: [String: Int] = [:]
    private var lastScanAt: TimeInterval = 0
    private var lastMTime: TimeInterval = 0
    private let scanInterval: TimeInterval = 300

    /// 递归扫描时跳过的目录名（同 Python 版）。
    private let skippedDirectories: Set<String> = [
        "tests", "test", ".git", ".svn", ".hg", "node_modules", "__pycache__",
        "build", "dist",
    ]

    public init(skillsDirectory: String, builtinDirectories: [String] = []) {
        self.skillsDirectory = skillsDirectory
        self.builtinDirectories = builtinDirectories
        loadRequestCounts()
    }

    /// 便捷构造：主目录为 `<workspace>/skills`。
    public convenience init(workspace: String, builtinDirectories: [String] = []) {
        self.init(skillsDirectory: (workspace as NSString)
            .appendingPathComponent("skills"),
                  builtinDirectories: builtinDirectories)
    }

    /// 触发下次扫描（save / 外部变更后调用）。
    public func markStale() {
        lock.lock(); lastScanAt = 0; lock.unlock()
    }

    // MARK: 意图匹配 / 重复请求建议固化

    private var requestsPath: String {
        (skillsDirectory as NSString).deletingLastPathComponent + "/skill_requests.json"
    }

    /// 按关键词命中度挑选最相关技能（名称 ×3、关键词 ×2、tags ×2、描述 ×1）。
    public func match(_ text: String) -> Skill? {
        let query = Set(TextAnalyzer.words(text))
        guard !query.isEmpty else { return nil }
        var best: Skill?
        var bestScore = 0
        for skill in list() {
            var score = 0
            let nameWords = Set(TextAnalyzer.words(skill.name))
            score += nameWords.intersection(query).count * 3
            let keywordWords = Set(skill.keywords.flatMap { TextAnalyzer.words($0) })
            score += keywordWords.intersection(query).count * 2
            let tagWords = Set(skill.tags.flatMap { TextAnalyzer.words($0) })
            score += tagWords.intersection(query).count * 2
            if skill.description.localizedCaseInsensitiveContains(text) ||
                text.localizedCaseInsensitiveContains(skill.description) {
                score += 1
            }
            if score > bestScore {
                bestScore = score
                best = skill
            }
        }
        return best
    }

    private static let suggestThreshold = 2
    private static let stopwords: Set<String> = [
        "the", "a", "an", "to", "of", "and", "or", "for", "in", "on", "with",
        "please", "帮我", "一下", "这个", "那个", "然后", "怎么", "如何",
    ]

    /// 纯寒暄/感谢/告别：连工具都不需要，直接文本回复。
    public static func isGreeting(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return true }
        if trimmed.contains("\n") { return false }
        let greetingPattern =
            "^(你好|您好|hi|hello|hey|嗨|哈喽|早上好|上午好|中午好|下午好|晚上好|晚安|"
            + "在吗|在不在|谢谢|多谢|感谢|thanks|thank you|再见|拜拜|bye|"
            + "好的|好|行|可以|嗯|哦|ok|okay)[呀啊哈呢吧嘛！!。.，,~～\\s]*$"
        return trimmed.range(of: greetingPattern,
                             options: [.regularExpression, .caseInsensitive]) != nil
    }

    /// 能力/身份问询：「你能做什么」「有哪些工具/能力」「介绍一下自己」等。
    ///
    /// 答案就写在 system prompt 的工具清单里，不需要调用任何工具。与 `isGreeting`
    /// 同级走轻量通道（无工具 schema、maxTurns=2、不输出 summary）；否则模型会
    /// 为了「列出可用技能 / 知识库文档 / RAG 索引状态」连调多个工具，并在每轮
    /// `<summary>` 里反复写「工具调用已完成」的元陈述，实测每轮拖 7-8s。
    public static func isCapabilityQuery(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !trimmed.contains("\n"), trimmed.count <= 40 else {
            return false
        }
        let lower = trimmed.lowercased()
        // 夹带具体动作/对象的请求不是能力问询（「你能帮我查下这个文件吗」）。
        let taskVerbs = [
            "查一下", "查下", "查查", "搜一下", "搜下", "读一下", "读下", "打开",
            "写一下", "写下", "改一下", "改下", "分析", "总结", "生成", "下载",
            "执行", "运行", "编译", "翻译", "整理", "读取", "写入", "列出文件",
            "fix", "debug", "write", "read", "open", "search for", "look up",
        ]
        if taskVerbs.contains(where: { lower.contains($0) }) { return false }
        // 「能/会/可以」后允许若干衬字，覆盖「你能帮我做什么」「你都能干啥」；
        // 具体任务请求已由上面的 taskVerbs + 40 字上限挡住，不会误放行。
        let pattern =
            "(能|会|可以).{0,6}(什么|啥|哪些|干啥|做啥)|"
            + "(有哪些|有什么|支持哪些|支持的?|列出|列一下|介绍).{0,8}"
            + "(能力|功能|工具|技能|命令|本领|用途|知识库|记忆)|"
            + "(能力|功能|工具|技能)(清单|列表|范围|介绍|概览)|"
            + "你是谁|你是什么|介绍(一下|下)?(你|自己|你自己)|"
            + "what can you|what are your|who are you|capabilities|features"
        // `range(of:.regularExpression)` 对非法正则会静默返回 nil（不抛错），
        // 因此这里用 NSRegularExpression 显式校验，避免括号不配对时判定恒为 false。
        guard let regex = try? NSRegularExpression(pattern: pattern,
                                                   options: .caseInsensitive) else {
            return false
        }
        let range = NSRange(trimmed.startIndex..., in: trimmed)
        return regex.firstMatch(in: trimmed, range: range) != nil
    }

    /// 寒暄、感谢、身份询问与短问询：不进入技能固化统计。
    ///
    /// 只放行「明显不需要计划」的短文本；含动作动词的请求一律按任务处理。
    public static func isTrivialRequest(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return true }
        if trimmed.contains("\n") { return false }
        if isGreeting(trimmed) { return true }

        let lower = trimmed.lowercased()
        let actionWords = [
            "写", "改", "修改", "创建", "新建", "删除", "移动", "复制", "重命名",
            "实现", "开发", "修复", "调试", "排查", "优化", "重构", "部署", "安装",
            "运行", "执行", "打开", "保存", "导出", "导入", "下载", "上传", "同步",
            "整理", "分析", "总结", "生成", "转换", "翻译", "搜索", "查找", "帮我",
            "请帮", "处理", "完成", "计划", "步骤", "方案", "设计", "测试", "编译",
            "构建", "提交", "发送", "调用", "读取", "写入", "爬取", "抓取",
            "write", "create", "modify", "edit", "fix", "debug", "implement",
            "build", "run", "execute", "deploy", "install", "analyze", "summarize",
            "generate", "convert", "translate", "search", "find", "save", "export",
            "import", "download", "upload", "sync", "plan", "design", "test", "compile",
        ]
        if actionWords.contains(where: { lower.contains($0) }) { return false }

        // 短问询/陈述视为简单请求；稍长但明确是问句的也放行。
        if trimmed.count <= 20 { return true }
        if trimmed.count <= 40, trimmed.hasSuffix("?") || trimmed.hasSuffix("？") {
            return true
        }
        return false
    }

    /// 统计相似意图；同一意图出现达到阈值后返回一个「待固化」技能草稿。
    public func recordRequest(_ text: String) -> Skill? {
        let words = TextAnalyzer.words(text)
            .filter { $0.count > 1 && !Self.stopwords.contains($0) }
        let key = Array(Set(words)).sorted().joined(separator: " ")
        guard !key.isEmpty else { return nil }
        lock.lock()
        let count = (requestCounts[key] ?? 0) + 1
        requestCounts[key] = count
        if requestCounts.count > 500 {
            // 简单容量治理：只保留最近的计数（按 key 排序，不追求 LRU）。
            requestCounts = Dictionary(uniqueKeysWithValues:
                Array(requestCounts.sorted { $0.value > $1.value }.prefix(500)))
        }
        saveRequestCountsLocked()
        lock.unlock()
        guard count >= Self.suggestThreshold else { return nil }

        let title = String(text.trimmingCharacters(in: .whitespacesAndNewlines).prefix(40))
        return Skill(
            name: title.isEmpty ? "recurring-task" : title,
            description: "重复出现 \(count) 次的请求，可固化为技能。",
            tags: words,
            keywords: words,
            directory: "",
            absolutePath: "",
            entry: "SKILL.md",
            whenToUse: text,
            whenNotToUse: "",
            examples: "",
            fingerprint: "",
            body: "当用户提出类似「\(title)」的请求时，按已验证步骤执行。"
        )
    }

    private func loadRequestCounts() {
        guard let data = FileManager.default.contents(atPath: requestsPath),
              let raw = try? JSONDecoder().decode([String: Int].self, from: data) else { return }
        requestCounts = raw
    }

    private func saveRequestCountsLocked() {
        guard let data = try? JSONEncoder().encode(requestCounts) else { return }
        try? AtomicFile.write(data, to: requestsPath)
    }

    /// 扫描所有根目录并返回技能列表（带 300s + mtime 去抖）。
    public func scan(force: Bool = false) -> [Skill] {
        lock.lock(); defer { lock.unlock() }
        let now = Date().timeIntervalSince1970
        let mtime = currentMTimeLocked()
        if !force, lastScanAt > 0, now - lastScanAt < scanInterval, mtime == lastMTime {
            return cached
        }
        var found: [Skill] = []
        var seen = Set<String>()
        for root in rootsLocked() {
            for skill in scanRootLocked(root) where seen.insert(skill.absolutePath).inserted {
                found.append(skill)
            }
        }
        cached = found.sorted {
            if $0.name != $1.name {
                return $0.name.localizedStandardCompare($1.name) == .orderedAscending
            }
            return $0.directory < $1.directory
        }
        lastScanAt = now
        lastMTime = mtime
        return cached
    }

    public func list() -> [Skill] {
        scan()
    }

    /// 按名称（大小写不敏感）或目录名加载技能。
    public func load(name: String) -> Skill? {
        let needle = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !needle.isEmpty else { return nil }
        let all = scan()
        if let exact = all.first(where: { $0.name.caseInsensitiveCompare(needle) == .orderedSame }) {
            return exact
        }
        return all.first { skill in
            let last = (skill.directory as NSString).lastPathComponent
            return last.caseInsensitiveCompare(needle) == .orderedSame
        }
    }

    /// 加载技能入口文件。
    public func loadContent(name: String, entry: String? = nil) throws -> String {
        guard let skill = load(name: name) else {
            throw AgentError.storage("技能不存在: \(name)")
        }
        let file = entry?.trimmingCharacters(in: .whitespacesAndNewlines) ?? skill.entry
        guard let content = readFile(inDirectory: skill.absolutePath, entry: file) else {
            throw AgentError.storage("技能入口文件不存在: \(skill.absolutePath)/\(file)")
        }
        return content
    }

    /// 在技能目录内安全读取文件（拒绝绝对路径、`..` 与目录逃逸）。
    public func readFile(inDirectory directory: String, entry: String) -> String? {
        guard let filePath = Self.resolveWithin(directory, entry),
              FileManager.default.fileExists(atPath: filePath),
              let text = Self.readLimited(filePath) else {
            return nil
        }
        return text
    }

    /// 把 `skill_load` 传入的相对路径解析成技能目录绝对路径。
    ///
    /// - 依次在 `skillsDirectory` 与内置技能根目录下查找；
    /// - 通过软链接指向根目录外部的目录，只要目标含技能标记文件也允许加载。
    public func resolveSkillDirectory(_ path: String) -> String? {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !(trimmed as NSString).isAbsolutePath,
              !(trimmed as NSString).pathComponents.contains("..") else {
            return nil
        }
        let fm = FileManager.default
        let markers = ["SKILL.md", "skill.json", "prompt.md"]

        lock.lock()
        let roots = rootsLocked()
        lock.unlock()

        for root in roots {
            let base = URL(fileURLWithPath: root).resolvingSymlinksInPath().path
            let lexical = URL(fileURLWithPath: root)
                .appendingPathComponent(trimmed).path
            var isDirectory: ObjCBool = false
            guard fm.fileExists(atPath: lexical, isDirectory: &isDirectory),
                  isDirectory.boolValue else { continue }
            let resolved = URL(fileURLWithPath: lexical).resolvingSymlinksInPath().path

            if resolved == base || resolved.hasPrefix(base + "/") {
                return resolved
            }
            // 允许根目录内的软链接指向外部技能目录。
            if markers.contains(where: {
                fm.fileExists(atPath: (resolved as NSString).appendingPathComponent($0))
            }) {
                return resolved
            }
        }
        return nil
    }

    // MARK: - 保存

    /// 保存技能到 `<skillsDirectory>/<slug>/SKILL.md`，并镜像到
    /// `<workspace>/.agent/skills/<slug>/`（保留已有文件，便于关键词索引复用）。
    @discardableResult
    public func save(name: String,
                     description: String,
                     whenToUse: String = "",
                     steps: String = "",
                     pitfalls: String = "",
                     tags: [String] = []) throws -> Skill {
        let slug = Self.slugify(name)
        guard !slug.isEmpty else {
            throw AgentError.invalidArguments("invalid skill name (need [a-z0-9-])")
        }

        let fm = FileManager.default
        let directory = (skillsDirectory as NSString).appendingPathComponent(slug)
        do {
            try fm.createDirectory(atPath: directory, withIntermediateDirectories: true)
        } catch {
            throw AgentError.storage("创建技能目录失败: \(error.localizedDescription)")
        }

        let cleanDescription = Self.oneLine(description, limit: 200)
        let cleanWhen = Self.oneLine(whenToUse, limit: 400)
        let cleanSteps = Self.truncate(steps, limit: 8_000)
        let cleanPitfalls = Self.truncate(pitfalls, limit: 2_000)
        let cleanTags = Array(tags
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .prefix(8))

        var lines: [String] = ["---"]
        lines.append("name: \(slug)")
        lines.append("description: \(cleanDescription)")
        lines.append("tags: [\(cleanTags.joined(separator: ", "))]")
        lines.append("---")
        lines.append("")
        lines.append("# \(slug)")
        lines.append("")
        lines.append(cleanDescription)
        lines.append("")
        lines.append("## When to use")
        lines.append("")
        lines.append(cleanWhen.isEmpty ? "（未指定）" : cleanWhen)
        lines.append("")
        lines.append("## Steps")
        lines.append("")
        lines.append(cleanSteps.isEmpty ? "（未填写）" : cleanSteps)
        lines.append("")
        lines.append("## Pitfalls")
        lines.append("")
        lines.append(cleanPitfalls.isEmpty ? "（暂无）" : cleanPitfalls)
        lines.append("")

        let body = lines.joined(separator: "\n")
        let skillPath = (directory as NSString).appendingPathComponent("SKILL.md")
        try AtomicFile.write(Data(body.utf8), to: skillPath)
        // 关键词索引（SkillIndex）只扫描 skill.json；主目录也必须写一份，
        // 否则新建技能只能被 SkillManager 的目录扫描发现。
        writeIndexMetadata(directory: directory, slug: slug,
                           description: cleanDescription, tags: cleanTags,
                           keywords: cleanTags)

        mirrorToLegacyIndex(slug: slug, body: body,
                            description: cleanDescription, tags: cleanTags,
                            keywords: cleanTags)

        markStale()
        if let parsed = parseSkillMarkdown(body,
                                           directory: slug,
                                           absolutePath: directory,
                                           entry: "SKILL.md",
                                           fingerprint: Self.fingerprint(body)) {
            return parsed
        }
        return Skill(name: slug, description: cleanDescription, tags: cleanTags,
                     keywords: cleanTags, directory: slug, absolutePath: directory,
                     entry: "SKILL.md", whenToUse: cleanWhen, whenNotToUse: "",
                     examples: "", fingerprint: Self.fingerprint(body), body: body)
    }

    // MARK: - 扫描实现

    private func rootsLocked() -> [String] {
        var roots: [String] = []
        var seen = Set<String>()
        for path in [skillsDirectory] + builtinDirectories {
            let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let resolved = URL(fileURLWithPath: trimmed).resolvingSymlinksInPath().path
            guard seen.insert(resolved).inserted,
                  FileManager.default.fileExists(atPath: resolved) else { continue }
            roots.append(trimmed)
        }
        return roots
    }

    private func scanRootLocked(_ root: String) -> [Skill] {
        var results: [Skill] = []
        var stack: [String] = [root]
        var visited = Set<String>()

        while let current = stack.popLast() {
            let real = URL(fileURLWithPath: current).resolvingSymlinksInPath().path
            guard visited.insert(real).inserted else { continue }
            guard let names = try? FileManager.default.contentsOfDirectory(atPath: current) else {
                continue
            }
            let nameSet = Set(names)
            let markdownPath = (current as NSString).appendingPathComponent("SKILL.md")

            if nameSet.contains("SKILL.md") {
                if let text = Self.readLimited(markdownPath),
                   let skill = parseSkillMarkdown(text,
                                                  directory: Self.relativeDirectory(current, root: root),
                                                  absolutePath: current,
                                                  entry: "SKILL.md",
                                                  fingerprint: Self.fingerprint(text)) {
                    results.append(skill)
                }
            } else if nameSet.contains("skill.json") {
                if let skill = parseLegacySkill(directory: current, root: root) {
                    results.append(skill)
                }
            }

            for name in names {
                if name.hasPrefix(".") || skippedDirectories.contains(name) { continue }
                let full = (current as NSString).appendingPathComponent(name)
                var isDirectory: ObjCBool = false
                guard FileManager.default.fileExists(atPath: full, isDirectory: &isDirectory),
                      isDirectory.boolValue else { continue }
                stack.append(full)
            }
        }
        return results
    }

    /// 解析 `SKILL.md`（frontmatter + 正文章节）。
    public func parseSkillMarkdown(_ text: String,
                                   directory: String,
                                   absolutePath: String,
                                   entry: String,
                                   fingerprint: String) -> Skill? {
        let content = Self.stripBOM(text)
        guard content.hasPrefix("---") else { return nil }
        let lines = content.components(separatedBy: .newlines)
        guard lines.count > 1 else { return nil }

        var index = 1
        var meta: [String: String] = [:]
        var key: String?
        var blockLines: [String] = []
        var inBlock = false
        var blockFolded = false

        func flushBlock() {
            guard let currentKey = key else { return }
            if blockFolded {
                meta[currentKey] = blockLines.filter { !$0.isEmpty }.joined(separator: " ")
            } else {
                meta[currentKey] = blockLines.joined(separator: "\n")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            key = nil
            blockLines = []
            inBlock = false
            blockFolded = false
        }

        while index < lines.count {
            let raw = lines[index]
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line == "---" || line == "..." {
                flushBlock()
                index += 1
                break
            }
            if raw.hasPrefix(" ") || raw.hasPrefix("\t") {
                if inBlock {
                    blockLines.append(raw.trimmingCharacters(in: .whitespaces))
                }
                index += 1
                continue
            }
            flushBlock()

            guard let colon = raw.firstIndex(of: ":") else {
                index += 1
                continue
            }
            let name = String(raw[..<colon]).trimmingCharacters(in: .whitespaces).lowercased()
            let value = String(raw[raw.index(after: colon)...])
                .trimmingCharacters(in: .whitespaces)
            if let scalarMarker = value.first,
               scalarMarker == "|" || scalarMarker == ">",
               value.dropFirst().allSatisfy({ $0 == "+" || $0 == "-" }) {
                inBlock = true
                blockFolded = scalarMarker == ">"
                blockLines = []
                key = name
                index += 1
                continue
            }
            meta[name] = Self.unquote(value)
            index += 1
        }

        let name = Self.unquote(meta["name"] ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }
        let description = Self.unquote(meta["description"] ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let tags = Self.parseList(meta["tags"] ?? meta["tag"] ?? "")
        let keywords = Self.parseList(meta["keywords"] ?? "")
        let body = index < lines.count ? lines[index...].joined(separator: "\n") : ""

        return Skill(name: name,
                     description: description,
                     tags: tags,
                     keywords: keywords,
                     directory: directory,
                     absolutePath: absolutePath,
                     entry: entry,
                     whenToUse: Self.section(body, names: [
                         "when to use this skill", "when to use", "use this skill when",
                         "when you should use", "使用场景", "何时使用",
                     ]),
                     whenNotToUse: Self.section(body, names: [
                         "when not to use", "when not to use this skill",
                         "do not use", "don't use", "不适用场景",
                     ]),
                     examples: Self.section(body, names: [
                         "examples", "example", "usage examples", "示例",
                     ]),
                     fingerprint: fingerprint,
                     body: String(content.prefix(200_000)))
    }

    private func parseLegacySkill(directory: String, root: String) -> Skill? {
        let jsonPath = (directory as NSString).appendingPathComponent("skill.json")
        guard let raw = try? String(contentsOfFile: jsonPath, encoding: .utf8),
              let object = JSONValue.parse(raw)?.objectValue else { return nil }
        let name = (object["name"]?.stringValue ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }
        let description = object["description"]?.stringValue ?? ""
        let keywords = (object["keywords"]?.arrayValue ?? []).compactMap(\.stringValue)
        let entry = object["entry"]?.stringValue?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? "prompt.md"
        let entryName = entry.isEmpty ? "prompt.md" : entry
        let entryPath = (directory as NSString).appendingPathComponent(entryName)
        let body = Self.readLimited(entryPath) ?? ""
        return Skill(name: name,
                     description: description,
                     tags: [],
                     keywords: keywords,
                     directory: Self.relativeDirectory(directory, root: root),
                     absolutePath: directory,
                     entry: entryName,
                     whenToUse: "",
                     whenNotToUse: "",
                     examples: "",
                     fingerprint: Self.fingerprint(body + "|" + description + "|" + keywords.joined(separator: ",")),
                     body: body)
    }

    /// 写入 SkillIndex 所需的元数据文件。
    private func writeIndexMetadata(directory: String, slug: String,
                                    description: String, tags: [String],
                                    keywords: [String]) {
        let payload: [String: JSONValue] = [
            "name": .string(slug),
            "description": .string(description),
            "keywords": .array((keywords.isEmpty ? tags : keywords).map { .string($0) }),
            "tags": .array(tags.map { .string($0) }),
            "version": .string("1.0"),
            "entry": .string("SKILL.md"),
        ]
        let text = JSONValue.object(payload).jsonString(pretty: true) + "\n"
        let metaPath = (directory as NSString).appendingPathComponent("skill.json")
        try? AtomicFile.write(Data(text.utf8), to: metaPath)
    }

    /// 把主技能镜像到 `<workspace>/.agent/skills/<slug>/`，供旧版关键词索引读取。
    ///
    /// 主目录本身就是 `.agent/skills` 时无需（也不能）再嵌套一层 `.agent`。
    private func mirrorToLegacyIndex(slug: String, body: String,
                                     description: String, tags: [String],
                                     keywords: [String]) {
        guard let workspace = Self.workspaceRoot(forSkillsDirectory: skillsDirectory) else {
            return
        }
        let legacyDir = URL(fileURLWithPath: workspace)
            .appendingPathComponent(".agent")
            .appendingPathComponent("skills")
            .appendingPathComponent(slug)
            .path
        let primaryDir = URL(fileURLWithPath: skillsDirectory)
            .appendingPathComponent(slug).path
        guard (legacyDir as NSString).standardizingPath
            != (primaryDir as NSString).standardizingPath else {
            return
        }
        let fm = FileManager.default
        try? fm.createDirectory(atPath: legacyDir, withIntermediateDirectories: true)

        // 覆盖写：技能更新后镜像也必须同步，否则索引会一直读旧内容。
        try? AtomicFile.write(Data(body.utf8),
                              to: (legacyDir as NSString).appendingPathComponent("SKILL.md"))
        writeIndexMetadata(directory: legacyDir, slug: slug,
                           description: description, tags: tags, keywords: keywords)
    }

    // MARK: - mtime

    private func currentMTimeLocked() -> TimeInterval {
        var latest: TimeInterval = 0
        let fm = FileManager.default
        for root in rootsLocked() {
            latest = max(latest, Self.modificationTime(root))
            guard let names = try? fm.contentsOfDirectory(atPath: root) else { continue }
            for name in names {
                let full = (root as NSString).appendingPathComponent(name)
                var isDirectory: ObjCBool = false
                guard fm.fileExists(atPath: full, isDirectory: &isDirectory),
                      isDirectory.boolValue else { continue }
                latest = max(latest, Self.modificationTime(full))
                for marker in ["SKILL.md", "skill.json"] {
                    latest = max(latest, Self.modificationTime(
                        (full as NSString).appendingPathComponent(marker)))
                }
            }
        }
        return latest
    }

    // MARK: - 解析工具

    private static func parseList(_ raw: String) -> [String] {
        var value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if value.hasPrefix("["), value.hasSuffix("]") {
            value = String(value.dropFirst().dropLast())
        }
        return value
            .split(whereSeparator: { $0 == "," || $0 == "\n" || $0 == ";" })
            .map { unquote(String($0)).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private static func section(_ body: String, names: [String]) -> String {
        let wanted = Set(names)
        var collecting = false
        var buffer: [String] = []
        for raw in body.components(separatedBy: .newlines) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("## ") {
                let title = String(line.dropFirst(3))
                    .trimmingCharacters(in: .whitespaces)
                    .lowercased()
                if collecting { break }
                collecting = wanted.contains(title)
                continue
            }
            if collecting { buffer.append(raw) }
        }
        return buffer.joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func slugify(_ raw: String) -> String {
        let lowered = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        var out = ""
        var lastDash = false
        for scalar in lowered.unicodeScalars {
            let isASCIIAlnum = (scalar.value >= 97 && scalar.value <= 122)
                || (scalar.value >= 48 && scalar.value <= 57)
            if isASCIIAlnum || scalar == "." {
                out.unicodeScalars.append(scalar)
                lastDash = false
            } else if !lastDash {
                out.append("-")
                lastDash = true
            }
        }
        let trimmed = out.trimmingCharacters(in: CharacterSet(charactersIn: "-."))
        return String(trimmed.prefix(64))
    }

    private static func oneLine(_ text: String, limit: Int) -> String {
        let flat = text.replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return truncate(flat, limit: limit)
    }

    private static func truncate(_ text: String, limit: Int) -> String {
        guard text.count > limit else { return text }
        return String(text.prefix(limit))
    }

    private static func unquote(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 2 else { return trimmed }
        if (trimmed.hasPrefix("\"") && trimmed.hasSuffix("\""))
            || (trimmed.hasPrefix("'") && trimmed.hasSuffix("'")) {
            return String(trimmed.dropFirst().dropLast())
        }
        return trimmed
    }

    private static func stripBOM(_ text: String) -> String {
        var result = text
        while let first = result.first, first == "\u{FEFF}" || first.isWhitespace {
            result.removeFirst()
        }
        return result
    }

    private static func fingerprint(_ text: String) -> String {
        let digest = SHA256.hash(data: Data(text.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private static func modificationTime(_ path: String) -> TimeInterval {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let date = attrs[.modificationDate] as? Date else { return 0 }
        return date.timeIntervalSince1970
    }

    private static func readLimited(_ path: String, limit: Int = 200_000) -> String? {
        guard let text = try? String(contentsOfFile: path, encoding: .utf8) else {
            return nil
        }
        return String(text.prefix(limit))
    }

    private static func relativeDirectory(_ directory: String, root: String) -> String {
        let base = (root as NSString).standardizingPath
        let path = (directory as NSString).standardizingPath
        guard path != base, path.hasPrefix(base + "/") else { return directory }
        return String(path.dropFirst(base.count + 1))
    }

    /// 由技能目录反推 workspace 根目录。
    ///
    /// `<ws>/skills` → `<ws>`；`<ws>/.agent/skills` → `<ws>`（不能返回 `.agent`，
    /// 否则镜像会落到 `.agent/.agent/skills`）。
    private static func workspaceRoot(forSkillsDirectory directory: String) -> String? {
        let path = (directory as NSString).standardizingPath
        guard (path as NSString).lastPathComponent == "skills" else { return nil }
        var parent = (path as NSString).deletingLastPathComponent
        guard !parent.isEmpty else { return nil }
        if (parent as NSString).lastPathComponent == ".agent" {
            parent = (parent as NSString).deletingLastPathComponent
        }
        return parent.isEmpty ? nil : parent
    }

    /// 在 `base` 内安全解析相对路径；绝对路径与 `..` 一律拒绝。
    static func resolveWithin(_ base: String, _ relative: String) -> String? {
        let trimmed = relative.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !(trimmed as NSString).isAbsolutePath else { return nil }
        let components = (trimmed as NSString).pathComponents
        guard !components.contains("..") else { return nil }

        let baseResolved = URL(fileURLWithPath: base).resolvingSymlinksInPath().path
        let candidate = URL(fileURLWithPath: base).appendingPathComponent(trimmed)
            .resolvingSymlinksInPath().path
        guard candidate == baseResolved || candidate.hasPrefix(baseResolved + "/") else {
            return nil
        }
        return candidate
    }
}
