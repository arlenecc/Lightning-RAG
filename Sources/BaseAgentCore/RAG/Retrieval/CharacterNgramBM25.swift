import Foundation

/// 字符级 N-gram BM25：用于「决策判定模式」从原文切片中抽取核心句。
///
/// 与 SQLite FTS5 的词级 BM25 不同，这里按字符 N-gram（默认 3）滑动窗口切分，
/// 对中文/无空格文本更稳健。评分对象是一个文本切片内的每个句子：
/// 把「句子集合」当作语料库，query 当作查询，用 BM25 给每个句子打分。
public enum CharacterNgramBM25 {
    public struct Config: Sendable {
        public var ngram: Int
        public var k1: Double
        public var b: Double
        public init(ngram: Int = 3, k1: Double = 1.6, b: Double = 0.4) {
            self.ngram = max(1, ngram)
            self.k1 = k1
            self.b = b
        }
    }

    public static let standard = Config()

    /// 从文本中切出句子（按中文/英文句末标点与换行），保留原文顺序。
    public static func sentences(_ text: String) -> [String] {
        var result: [String] = []
        var current = ""
        func flush() {
            let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { result.append(trimmed) }
            current = ""
        }
        for ch in text {
            current.append(ch)
            if ch == "。" || ch == "！" || ch == "？" || ch == ";" || ch == "；"
                || ch == "\n" || ch == "." || ch == "!" || ch == "?" {
                flush()
            }
        }
        flush()
        return result
    }

    /// 字符级 N-gram 序列。先把空白去掉，再按 Unicode 标量滑窗（中文单字即 1 gram 单位）。
    static func grams(_ text: String, n: Int) -> [String] {
        let chars = Array(text.filter { !$0.isWhitespace })
        guard chars.count >= n else { return chars.isEmpty ? [] : [String(chars)] }
        var out: [String] = []
        out.reserveCapacity(chars.count - n + 1)
        for i in 0...(chars.count - n) {
            out.append(String(chars[i..<(i + n)]))
        }
        return out
    }

    /// 计算 query 对一组句子的 BM25 分数，返回与输入句子一一对应的分数。
    public static func scores(query: String, sentences: [String],
                              config: Config = Config()) -> [Double] {
        guard !sentences.isEmpty else { return [] }
        let qGrams = grams(query, n: config.ngram)
        guard !qGrams.isEmpty else { return Array(repeating: 0, count: sentences.count) }

        let docGrams = sentences.map { grams($0, n: config.ngram) }
        let docLengths = docGrams.map { $0.count }
        let avgdl = docLengths.isEmpty ? 0 : Double(docLengths.reduce(0, +)) / Double(docLengths.count)

        // 文档频率：每个 gram 出现在多少个句子里
        var df: [String: Int] = [:]
        for g in Set(qGrams) {
            df[g] = docGrams.filter { $0.contains(g) }.count
        }
        let N = Double(sentences.count)

        return docGrams.enumerated().map { index, tokens in
            guard !tokens.isEmpty else { return 0.0 }
            var tf: [String: Int] = [:]
            for g in tokens { tf[g, default: 0] += 1 }
            let dl = Double(docLengths[index])
            var score = 0.0
            for g in Set(qGrams) {
                guard let freq = tf[g], freq > 0 else { continue }
                let f = Double(freq)
                let n = Double(df[g] ?? 0)
                // Robertson/Sparck Jones IDF，加 0.5 平滑并钳到非负，避免 Common gram 拉负分。
                let idf = log(1.0 + (N - n + 0.5) / (n + 0.5))
                guard idf > 0 else { continue }
                let lengthNorm: Double = avgdl > 0 ? dl / avgdl : 0
                let denom = f + config.k1 * (1.0 - config.b + config.b * lengthNorm)
                score += idf * (f * (config.k1 + 1.0)) / denom
            }
            return score
        }
    }

    /// 抽取核心句：取 BM25 分数最高的 `count` 句，按其在原文中出现的顺序返回。
    public static func coreSentences(query: String, text: String,
                                     count: Int = 3,
                                     config: Config = Config()) -> [String] {
        let sents = sentences(text)
        guard !sents.isEmpty, count > 0 else { return [] }
        let scores = scores(query: query, sentences: sents, config: config)
        let ranked = scores.enumerated()
            .sorted { $0.element > $1.element }
            .prefix(count)
            .map(\.offset)
        return ranked.sorted().map { sents[$0] }
    }
}
