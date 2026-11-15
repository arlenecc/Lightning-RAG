import Foundation

/// 关键词索引中的一条技能记录（与 Python `_index.json` 字段一致）。
public struct SkillIndexEntry: Codable, Sendable, Equatable {
    public var name: String
    public var description: String
    public var keywords: [String]
    public var tags: [String]
    public var path: String
    public var entry: String

    public init(name: String, description: String = "", keywords: [String] = [],
                tags: [String] = [], path: String, entry: String = "prompt.md") {
        self.name = name
        self.description = description
        self.keywords = keywords
        self.tags = tags
        self.path = path
        self.entry = entry
    }
}

/// 技能关键词索引（BM25 风格词频打分）。
///
/// 复用 `SkillManager` 的递归扫描（含 `SKILL.md` / 旧式 `skill.json`、
/// 内置与下载技能目录），把技能元数据写入 `<skillsDirectory>/_index.json`；
/// 检索打分规则与 Python 版一致：名称命中 ×3、关键词 ×2、tags ×2、描述子串 ×1。
public final class SkillIndex {
    public let skillsDirectory: String
    /// 技能扫描与解析的唯一来源：与语义检索共用同一份目录与解析规则，
    /// 避免关键词索引只看到 `.agent/skills` 顶层而漏掉下载技能。
    private let manager: SkillManager

    private struct IndexFile: Codable {
        var version: Int
        var updatedAt: String
        var updatedAtTs: Double
        var skills: [SkillIndexEntry]

        enum CodingKeys: String, CodingKey {
            case version
            case updatedAt = "updated_at"
            case updatedAtTs = "updated_at_ts"
            case skills
        }
    }

    private let lock = NSRecursiveLock()
    private var entries: [SkillIndexEntry] = []
    private let indexVersion = 1
    private let indexFileName = "_index.json"

    /// 与语义检索共用同一个 `SkillManager`：扫描去抖、递归目录、软链接、
    /// `need-deps` 跳过与 `SKILL.md` 解析规则完全一致。
    public init(manager: SkillManager) {
        self.manager = manager
        self.skillsDirectory = manager.skillsDirectory
        load()
    }

    public convenience init(skillsDirectory: String, builtinDirectories: [String] = []) {
        self.init(manager: SkillManager(skillsDirectory: skillsDirectory,
                                        builtinDirectories: builtinDirectories))
    }

    /// 便捷构造：`<workspace>/.agent/skills`（主目录）+
    /// `<workspace>/skills`（用户下载/内置技能目录），与 `Agent` 的扫描范围一致。
    public convenience init(workspace: String) {
        let root = URL(fileURLWithPath: workspace)
        self.init(skillsDirectory: root.appendingPathComponent(".agent")
            .appendingPathComponent("skills").path,
                  builtinDirectories: [root.appendingPathComponent("skills").path])
    }

    // MARK: - 扫描

    /// 返回索引中的技能数量；`force` 忽略 300s 去抖。
    @discardableResult
    public func scan(force: Bool = false) -> Int {
        lock.lock(); defer { lock.unlock() }
        let fm = FileManager.default
        let scanned: [SkillIndexEntry] = manager.scan(force: force).compactMap { skill in
            // 旧式 `skill.json` 声明的入口可能不存在：回退到实际存在的约定文件，
            // 否则索引里会留下一个无法加载的入口。
            var entry = skill.entry
            let directory = skill.absolutePath
            if !fm.fileExists(atPath: (directory as NSString).appendingPathComponent(entry)) {
                if fm.fileExists(atPath: (directory as NSString).appendingPathComponent("SKILL.md")) {
                    entry = "SKILL.md"
                } else if fm.fileExists(atPath: (directory as NSString)
                    .appendingPathComponent("prompt.md")) {
                    entry = "prompt.md"
                } else {
                    return nil
                }
            }
            return SkillIndexEntry(name: skill.name,
                                   description: skill.description,
                                   keywords: skill.keywords,
                                   tags: skill.tags,
                                   path: skill.directory,
                                   entry: entry)
        }
        guard scanned != entries else { return entries.count }
        entries = scanned
        saveLocked()
        AgentLog.info("[skill] 关键词索引已更新 total=\(entries.count)")
        return entries.count
    }

    /// 与 Python 版一致：名称匹配权重最高，其次关键词 / tags，描述子串兜底。
    public func search(query: String, topK: Int = 3) -> [SkillIndexEntry] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return [] }
        scan()
        lock.lock(); let snapshot = entries; lock.unlock()

        let queryTokens = Set(SkillTextTokenizer.tokens(q))
        let lowered = q.lowercased()
        let limit = max(1, min(topK, 5))

        var scored: [(entry: SkillIndexEntry, score: Int)] = []
        for entry in snapshot {
            var score = 0
            let nameTokens = Set(SkillTextTokenizer.tokens(entry.name))
            score += nameTokens.intersection(queryTokens).count * 3
            let keywordTokens = Set(entry.keywords.flatMap { SkillTextTokenizer.tokens($0) })
            score += keywordTokens.intersection(queryTokens).count * 2
            let tagTokens = Set(entry.tags.flatMap { SkillTextTokenizer.tokens($0) })
            score += tagTokens.intersection(queryTokens).count * 2
            if entry.description.lowercased().contains(lowered) { score += 1 }
            guard score > 0 else { continue }
            scored.append((entry, score))
        }
        scored.sort {
            if $0.score != $1.score { return $0.score > $1.score }
            return $0.entry.name.localizedStandardCompare($1.entry.name) == .orderedAscending
        }
        return scored.prefix(limit).map { $0.entry }
    }

    public func listAll() -> [SkillIndexEntry] {
        scan()
        lock.lock(); defer { lock.unlock() }
        return entries
    }

    /// 判断给定目录是否是合法技能目录（目录内存在 skill.json 或入口文件）。
    public func getSkillPath(_ path: String) -> String? {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        // 先按所有技能根目录（主目录 + 内置/下载目录）解析，保证索引里的
        // `others/cv-generator` 这类嵌套下载技能也能加载。
        if !(trimmed as NSString).isAbsolutePath,
           let resolved = manager.resolveSkillDirectory(trimmed) {
            return resolved
        }
        let directory: String
        if (trimmed as NSString).isAbsolutePath {
            directory = trimmed
        } else {
            guard let resolved = resolveWithin(skillsDirectory, trimmed) else { return nil }
            directory = resolved
        }
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: directory, isDirectory: &isDirectory),
              isDirectory.boolValue else { return nil }
        let meta = (directory as NSString).appendingPathComponent("skill.json")
        let prompt = (directory as NSString).appendingPathComponent("prompt.md")
        let markdown = (directory as NSString).appendingPathComponent("SKILL.md")
        guard fm.fileExists(atPath: meta) || fm.fileExists(atPath: prompt)
            || fm.fileExists(atPath: markdown) else { return nil }
        return directory
    }

    /// 读取技能入口文件（先声明入口，再回退 prompt.md / SKILL.md）。
    public func readSkillContent(path: String, entry: String = "prompt.md") -> String? {
        guard let directory = getSkillPath(path) else { return nil }
        var candidates: [String] = []
        let declared = entry.trimmingCharacters(in: .whitespacesAndNewlines)
        if !declared.isEmpty { candidates.append(declared) }
        if let metaRaw = try? String(contentsOfFile: (directory as NSString)
            .appendingPathComponent("skill.json"), encoding: .utf8),
           let meta = JSONValue.parse(metaRaw)?.objectValue,
           let metaEntry = meta["entry"]?.stringValue,
           !candidates.contains(metaEntry) {
            candidates.append(metaEntry)
        }
        for name in ["prompt.md", "SKILL.md"] where !candidates.contains(name) {
            candidates.append(name)
        }
        for name in candidates {
            guard let file = resolveWithin(directory, name),
                  let text = Self.readLimited(file) else { continue }
            return text
        }
        return nil
    }

    // MARK: - 持久化

    private func load() {
        let path = (skillsDirectory as NSString).appendingPathComponent(indexFileName)
        guard let raw = try? String(contentsOfFile: path, encoding: .utf8),
              let root = JSONValue.parse(raw)?.objectValue else {
            entries = []
            return
        }
        var loaded: [SkillIndexEntry] = []
        for item in root["skills"]?.arrayValue ?? [] {
            guard let object = item.objectValue else { continue }
            let name = (object["name"]?.stringValue ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let relativePath = object["path"]?.stringValue ?? ""
            guard !name.isEmpty, !relativePath.isEmpty else { continue }
            let entry = object["entry"]?.stringValue ?? "prompt.md"
            loaded.append(SkillIndexEntry(
                name: name,
                description: object["description"]?.stringValue ?? "",
                keywords: (object["keywords"]?.arrayValue ?? []).compactMap(\.stringValue),
                tags: (object["tags"]?.arrayValue ?? []).compactMap(\.stringValue),
                path: relativePath,
                entry: entry.isEmpty ? "prompt.md" : entry))
        }
        entries = loaded
    }

    private func saveLocked() {
        let fm = FileManager.default
        if !fm.fileExists(atPath: skillsDirectory) {
            try? fm.createDirectory(atPath: skillsDirectory, withIntermediateDirectories: true)
        }
        let now = Date()
        let formatter = ISO8601DateFormatter()
        let payload = IndexFile(
            version: indexVersion,
            updatedAt: formatter.string(from: now),
            updatedAtTs: now.timeIntervalSince1970,
            skills: entries)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(payload) else { return }
        do {
            try AtomicFile.write(data, to: (skillsDirectory as NSString)
                .appendingPathComponent(indexFileName))
        } catch {
            AgentLog.warning("skill_index: 写入索引失败: \(error.localizedDescription)")
        }
    }

    /// 在 `base` 内安全解析相对路径；拒绝绝对路径、`..` 与软链接逃逸。
    func resolveWithin(_ base: String, _ relative: String) -> String? {
        let trimmed = relative.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !(trimmed as NSString).isAbsolutePath else { return nil }
        guard !(trimmed as NSString).pathComponents.contains("..") else { return nil }
        let baseResolved = URL(fileURLWithPath: base).resolvingSymlinksInPath().path
        let candidate = URL(fileURLWithPath: base).appendingPathComponent(trimmed)
            .resolvingSymlinksInPath().path
        guard candidate == baseResolved || candidate.hasPrefix(baseResolved + "/") else {
            return nil
        }
        return candidate
    }

    private static func readLimited(_ path: String, limit: Int = 200_000) -> String? {
        guard let text = try? String(contentsOfFile: path, encoding: .utf8) else {
            return nil
        }
        return String(text.prefix(limit))
    }
}

// MARK: - 分词

/// 轻量分词：英文/数字按词，中文按连续片段 + 二元组。
enum SkillTextTokenizer {
    static func tokens(_ text: String) -> [String] {
        var out: [String] = []
        var word = ""
        var cjk = ""

        func flushWord() {
            if !word.isEmpty { out.append(word.lowercased()); word = "" }
        }
        func flushCJK() {
            guard !cjk.isEmpty else { return }
            out.append(cjk)
            let chars = Array(cjk)
            if chars.count > 1 {
                for i in 0..<(chars.count - 1) {
                    out.append(String(chars[i...i + 1]))
                }
            }
            cjk = ""
        }

        for scalar in text.unicodeScalars {
            if (0x4E00...0x9FFF).contains(scalar.value) {
                flushWord()
                cjk.unicodeScalars.append(scalar)
                continue
            }
            flushCJK()
            if CharacterSet.alphanumerics.contains(scalar) || scalar == "_" {
                word.unicodeScalars.append(scalar)
            } else {
                flushWord()
            }
        }
        flushWord()
        flushCJK()
        return out
    }
}
