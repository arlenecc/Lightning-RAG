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
/// 扫描 `<skillsDirectory>/<slug>/skill.json`，将元数据写入
/// `<skillsDirectory>/_index.json`；检索打分规则与 Python 版一致：
/// 名称命中 ×3、关键词 ×2、tags ×2、描述子串 ×1。
public final class SkillIndex {
    public let skillsDirectory: String

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
    private var lastScanAt: TimeInterval = 0
    /// 目录本身及其直接子目录、子目录内 skill.json 的最大 mtime。
    private var lastDirMTime: TimeInterval = 0
    private let scanInterval: TimeInterval = 300
    private let indexVersion = 1
    private let indexFileName = "_index.json"
    private let skipped: Set<String> = ["__pycache__", "node_modules", ".git"]

    public init(skillsDirectory: String) {
        self.skillsDirectory = skillsDirectory
        load()
    }

    /// 便捷构造：`<workspace>/.agent/skills`。
    public convenience init(workspace: String) {
        self.init(skillsDirectory: URL(fileURLWithPath: workspace)
            .appendingPathComponent(".agent")
            .appendingPathComponent("skills")
            .path)
    }

    // MARK: - 扫描

    /// 返回索引中的技能数量；`force` 忽略 300s 去抖。
    @discardableResult
    public func scan(force: Bool = false) -> Int {
        lock.lock(); defer { lock.unlock() }
        let now = Date().timeIntervalSince1970
        let mtime = directoryMTimeLocked()
        if !force, lastScanAt > 0, now - lastScanAt < scanInterval, mtime == lastDirMTime {
            return entries.count
        }

        let fm = FileManager.default
        guard fm.fileExists(atPath: skillsDirectory),
              let names = try? fm.contentsOfDirectory(atPath: skillsDirectory) else {
            entries = []
            lastScanAt = now
            lastDirMTime = mtime
            saveLocked()
            return 0
        }

        var scanned: [SkillIndexEntry] = []
        for name in names.sorted() {
            if name.hasPrefix(".") || name.hasPrefix("_") || skipped.contains(name) { continue }
            let directory = (skillsDirectory as NSString).appendingPathComponent(name)
            var isDirectory: ObjCBool = false
            guard fm.fileExists(atPath: directory, isDirectory: &isDirectory),
                  isDirectory.boolValue else { continue }
            let metaPath = (directory as NSString).appendingPathComponent("skill.json")
            guard let raw = try? String(contentsOfFile: metaPath, encoding: .utf8),
                  let object = JSONValue.parse(raw)?.objectValue else { continue }
            let skillName = (object["name"]?.stringValue ?? name)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !skillName.isEmpty else { continue }
            let declaredEntry = object["entry"]?.stringValue?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? "prompt.md"
            var entry = declaredEntry.isEmpty ? "prompt.md" : declaredEntry
            if !fm.fileExists(atPath: (directory as NSString).appendingPathComponent(entry)) {
                // 声明的入口缺失时回退到实际存在的约定文件，并同步修正 entry，
                // 否则索引里会留下一个指向不存在文件的入口。
                if fm.fileExists(atPath: (directory as NSString).appendingPathComponent("SKILL.md")) {
                    entry = "SKILL.md"
                } else if fm.fileExists(atPath: (directory as NSString).appendingPathComponent("prompt.md")) {
                    entry = "prompt.md"
                } else {
                    continue
                }
            }
            scanned.append(SkillIndexEntry(
                name: skillName,
                description: object["description"]?.stringValue ?? "",
                keywords: (object["keywords"]?.arrayValue ?? []).compactMap(\.stringValue),
                tags: (object["tags"]?.arrayValue ?? []).compactMap(\.stringValue),
                path: name,
                entry: entry))
        }

        entries = scanned
        lastScanAt = now
        lastDirMTime = mtime
        saveLocked()
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
        lastScanAt = root["updated_at_ts"]?.doubleValue ?? Date().timeIntervalSince1970
        lastDirMTime = directoryMTimeLocked()
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

    private func directoryMTimeLocked() -> TimeInterval {
        var latest = Self.modificationTime(skillsDirectory)
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: skillsDirectory) else {
            return latest
        }
        for name in names {
            let full = (skillsDirectory as NSString).appendingPathComponent(name)
            latest = max(latest, Self.modificationTime(full))
            latest = max(latest, Self.modificationTime(
                (full as NSString).appendingPathComponent("skill.json")))
        }
        return latest
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

    private static func modificationTime(_ path: String) -> TimeInterval {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let date = attrs[.modificationDate] as? Date else { return 0 }
        return date.timeIntervalSince1970
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
