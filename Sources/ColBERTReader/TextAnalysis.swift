import Foundation
import NaturalLanguage

/// 文本分析（借鉴主体 RAG 的 TextAnalyzer）：句子切分、词切分、FTS5 预分词与查询构造。
enum TextAnalysis {
    static func sentences(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        var result: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let s = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !s.isEmpty { result.append(s) }
            return true
        }
        if result.isEmpty {
            result = text.split(whereSeparator: \.isNewline)
                .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        }
        return result
    }

    static func words(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        var result: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let token = String(text[range]).lowercased()
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !token.isEmpty, token.unicodeScalars.contains(where: isWordScalar) {
                result.append(token)
            }
            return true
        }
        return result
    }

    /// FTS5 索引文本：英文原词 + CJK bigram，缓解中文整词不匹配。
    static func ftsIndexText(_ text: String) -> String {
        var output: [String] = []
        for token in words(text) {
            output.append(token)
            if containsCJK(token), token.count > 2 {
                let chars = Array(token)
                for i in 0..<(chars.count - 1) { output.append(String(chars[i...i + 1])) }
            }
        }
        return output.joined(separator: " ")
    }

    static func ftsQuery(_ text: String, mode: String = "OR", limit: Int = 24) -> String {
        var seen = Set<String>()
        var tokens: [String] = []
        for token in ftsTokens(text) {
            if TokenPruner.shared.isStopword(token) { continue }
            if containsCJK(token), token.count > 2 { continue }
            if seen.insert(token).inserted { tokens.append(token) }
        }
        tokens = tokens.enumerated()
            .sorted { $0.element.count == $1.element.count ? $0.offset < $1.offset : $0.element.count > $1.element.count }
            .prefix(limit).map(\.element)
        let joiner = mode == "AND" ? " AND " : " OR "
        return tokens.map { "\"" + $0.replacingOccurrences(of: "\"", with: "\"\"") + "\"" }
            .joined(separator: joiner)
    }

    private static func ftsTokens(_ text: String) -> [String] {
        ftsIndexText(text).split(separator: " ").map(String.init)
    }

    static func lexicalSimilarity(_ a: String, _ b: String) -> Double {
        let sa = Set(words(a)), sb = Set(words(b))
        guard !sa.isEmpty, !sb.isEmpty else { return 0 }
        return Double(sa.intersection(sb).count) / Double(sa.union(sb).count)
    }

    static func containsCJK(_ text: String) -> Bool {
        text.unicodeScalars.contains { scalar in
            switch scalar.value {
            case 0x3400...0x4DBF, 0x4E00...0x9FFF, 0x3040...0x30FF, 0xAC00...0xD7AF: return true
            default: return false
            }
        }
    }

    static func join(_ a: String, _ b: String) -> String {
        guard let last = a.last, let first = b.first, !last.isWhitespace, !first.isWhitespace else {
            return a + b
        }
        if isCJK(last) || isCJK(first) { return a + b }
        return a + " " + b
    }

    private static func isCJK(_ c: Character) -> Bool {
        guard let s = c.unicodeScalars.first else { return false }
        switch s.value {
        case 0x3000...0x303F, 0x3040...0x30FF, 0x3400...0x4DBF,
             0x4E00...0x9FFF, 0xAC00...0xD7AF, 0xFF00...0xFFEF: return true
        default: return false
        }
    }

    private static func isWordScalar(_ s: Unicode.Scalar) -> Bool {
        if CharacterSet.alphanumerics.contains(s) { return true }
        switch s.value {
        case 0x3400...0x4DBF, 0x4E00...0x9FFF, 0x3040...0x30FF, 0xAC00...0xD7AF: return true
        default: return false
        }
    }
}

/// Token Pruning：剔标点/空白，过滤停用词与低信息量词，只留核心名词、动词、专有名词（CJK 保留实词）。
final class TokenPruner {
    static let shared = TokenPruner()

    private let stopwords: Set<String>
    private let weakChars: Set<Character> = Set(
        "的 了 着 过 得 地 是 在 和 与 或 也 就 都 而 之 以 于 其 这 那 一 不 很 更 最 可 能 会 要 将 被 把 从 对 为 所 由 并 且 但 则 者 矣 焉 乎 哉 又 再 还 才 只 已 曾 乃 若 即 便 却 竟 偏"
            .filter { !$0.isWhitespace })

    /// 核心英文功能词（不合并哈工大表）。文档侧在 init 里再并大表；查询侧轻量
    /// 剪枝只用这份核心表，且从中剔除疑问词/程度词，避免误删回答查询必需的信息。
    private static let coreFunctionWords: Set<String> = [
        "a", "an", "the", "and", "or", "but", "if", "then", "than", "of", "to", "in",
        "on", "at", "by", "for", "from", "with", "as", "is", "are", "was", "were", "be",
        "been", "being", "do", "does", "did", "have", "has", "had", "can", "could",
        "should", "would", "will", "may", "might", "must", "this", "that", "these",
        "those", "it", "its", "we", "our", "you", "your", "they", "their", "he", "she",
        "his", "her", "what", "which", "who", "when", "where", "how", "why", "about",
        "into", "over", "under", "also", "very", "more", "most", "some", "any", "all",
    ]

    /// 查询侧轻量剪枝保留词：疑问词（查询语义核心）与程度/形容词性功能词。
    /// 这些词即使在核心功能词表里也不删——保留形容词与疑问词是查询侧硬要求。
    private static let queryKeepWords: Set<String> = [
        "what", "which", "who", "whom", "whose", "when", "where", "how", "why",
        "very", "more", "most", "better", "best",
    ]

    /// 查询侧轻量停用词表 = 核心功能词 - 保留词。刻意不读哈工大 2992 词大表：
    /// 那张表含疑问词与部分形容词，查询侧宁可少删，绝不能误删查询关键词。
    private static let lightStopwords: Set<String> =
        Set(coreFunctionWords).subtracting(queryKeepWords)

    private init() {
        var stops = Self.coreFunctionWords
        if let url = Bundle.module.url(forResource: "stopwords_full", withExtension: "txt"),
           let text = try? String(contentsOf: url, encoding: .utf8) {
            for line in text.split(whereSeparator: \.isNewline) {
                let w = String(line).trimmingCharacters(in: .whitespaces).lowercased()
                if !w.isEmpty { stops.insert(w) }
            }
        }
        stopwords = stops
    }

    func isStopword(_ token: String) -> Bool { stopwords.contains(token.lowercased()) }

    /// 逐 token 剪枝判定（文档侧 ColBERT 用）。与 `prune(_:)` 的区别：这里只做
    /// 「丢弃/保留」二值判断，不去重、不重排、不做词性/长度过滤——ColBERT 的
    /// 多向量表示与位置严格对齐，去重或词性筛选会破坏 token 序列语义。
    ///
    /// 丢弃三类：
    /// 1. 纯标点 / 空白 token（含 SentencePiece 的 `▁` 占位），清洗后为空；
    /// 2. 停用词 token（`▁the` → `the` 命中词表）；
    /// 3. 纯中文虚词 token（`的`/`了` 等全部落在 weakChars）。保留中文单字实词
    ///    （如「方」「合」）与数字，避免误伤有效信息。
    func isDiscardableToken(_ token: String) -> Bool {
        let cleaned = cleanScalars(token)
        guard !cleaned.isEmpty else { return true }
        if stopwords.contains(cleaned.lowercased()) { return true }
        if TextAnalysis.containsCJK(cleaned), cleaned.allSatisfy({ weakChars.contains($0) }) {
            return true
        }
        return false
    }

    /// 查询侧轻量剪枝判定：只丢「纯标点/空白」「核心功能词」「纯中文虚词单字」，
    /// 其余一律保留。与文档侧的差别：
    /// - 停用词只用精简的 `lightStopwords`（疑问词 what/which/… 与程度词
    ///   very/more/most、形容词 better/best 已剔除），不读哈工大 2992 词大表；
    /// - 不做词性/去重/长度筛选，形容词、疑问词、实词全部保留；
    /// - 中文疑问词（什么/怎么/为什么/哪些…）显式保护，即使单字落在虚词表也不删。
    func isLightweightDiscardableToken(_ token: String) -> Bool {
        let cleaned = cleanScalars(token)
        guard !cleaned.isEmpty else { return true }
        let lower = cleaned.lowercased()
        if Self.queryKeepWords.contains(lower) { return false }
        if Self.lightStopwords.contains(lower) { return true }
        if TextAnalysis.containsCJK(cleaned) {
            // 中文疑问词整词保护：即使组成字都在 weakChars 也不删。
            if Self.chineseInterrogatives.contains(cleaned) { return false }
            if cleaned.allSatisfy({ weakChars.contains($0) }) { return true }
        }
        return false
    }

    /// 中文疑问词（用于查询侧轻量剪枝保护）。
    private static let chineseInterrogatives: Set<String> = [
        "什么", "怎么", "怎样", "如何", "为什么", "哪些", "哪个", "哪里", "哪儿",
        "谁", "多少", "几时", "何时", "可否", "是否", "有无",
    ]

    /// 保留词的字符（汉字/字母/数字），彻底删除标点、空格、换行。
    private func cleanScalars(_ text: String) -> String {
        var out = ""
        for scalar in text.unicodeScalars {
            if CharacterSet.alphanumerics.contains(scalar) || TextAnalysis.containsCJK(String(scalar)) {
                out.unicodeScalars.append(scalar)
            }
        }
        return out
    }

    func prune(_ text: String, limit: Int = 4096) -> [String] {
        let raw = TextAnalysis.words(text)
        guard !raw.isEmpty else { return [] }

        // 拉丁词用词性标注保留 名词/动词/专有名词；标注不可用时退化为「非停用词」。
        var tags: [String: Set<NLTag>] = [:]
        let tagger = NLTagger(tagSchemes: [.lexicalClass])
        tagger.string = text
        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word,
                             scheme: .lexicalClass,
                             options: [.omitWhitespace, .omitPunctuation]) { tag, range in
            guard let tag else { return true }
            let key = String(text[range]).lowercased()
            if key.isEmpty { return true }
            if tags[key] == nil { tags[key] = [] }
            tags[key]?.insert(tag)
            return true
        }
        let keepTags: Set<NLTag> = [.noun, .verb, .organizationName, .personalName, .placeName]
        let hasTags = !tags.isEmpty

        var seen = Set<String>()
        var output: [String] = []
        for word in raw {
            let cleaned = cleanScalars(word)
            guard !cleaned.isEmpty else { continue }
            let lower = cleaned.lowercased()
            guard !stopwords.contains(lower), seen.insert(lower).inserted else { continue }
            if TextAnalysis.containsCJK(cleaned) {
                guard cleaned.count >= 2, !cleaned.allSatisfy({ weakChars.contains($0) }) else { continue }
            } else {
                guard cleaned.count >= 2 else { continue }
                if hasTags, let wordTags = tags[lower], !wordTags.isEmpty {
                    guard !wordTags.isDisjoint(with: keepTags) else { continue }
                }
            }
            output.append(cleaned)
            if output.count >= limit { break }
        }
        return output
    }
}
