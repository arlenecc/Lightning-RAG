import Foundation

/// 从长文本中提取核心观点句：两种独立算法各取 topN（默认 10），合并去重后作为最终核心句。
///
/// 用途：决策模型/辅助模型打分前把长文本压缩成「既能全面代表、又足够短」的句集，
/// 避免整段原文导致打分过慢或超上下文。纯函数，不依赖模型与网络。
///
/// 方法 A：多信号启发式打分
///   Score = w1·P_pos + w2·D_lexicon + w3·C_centrality + w4·L_length
/// 方法 B：TextRank + 观点词加权初始化（观点句 2.0 / 普通句 0.5）
public enum CoreOpinionExtractor {

    /// 观点指示词表（出现频率越高，越可能是观点句）。
    public static let opinionLexicon: [String] = [
        "认为", "必须", "应该", "核心", "本质", "实际上", "意味着", "导致",
        "因此", "所以", "综上所述", "指出", "强调", "不可忽视",
    ]

    /// 每种方法各自抽取的句数：经真机对照（top20 vs top10），
    /// top10（合计上限 20 句）判定概率差 ≤0.028、充分/不足区分度更好，
    /// 打分耗时进一步从 2.67s 降到 0.31s，故默认取 10。
    public static let perMethodTopN = 10

    /// 权重：位置 0.25 / 观点词 0.30 / 中心度 0.30 / 长度 0.15。
    public struct Weights: Sendable {
        public var position: Double = 0.25
        public var lexicon: Double = 0.30
        public var centrality: Double = 0.30
        public var length: Double = 0.15
        public init() {}
    }

    // MARK: - 主入口

    /// 方法 A + 方法 B 各取 topN，按原文顺序合并去重。
    /// - Parameters:
    ///   - text: 待压缩文本
    ///   - query: 可选查询（提供时中心度同时考虑与 query 的相关性，更聚焦）
    ///   - topN: 每种方法取几句（默认 20）
    /// - Returns: 核心句数组（按原文顺序），空文本返回 []
    public static func extract(_ text: String, query: String? = nil,
                               topN: Int = perMethodTopN) -> [String] {
        let sentences = splitSentences(text)
        guard !sentences.isEmpty else { return [] }
        let n = max(1, min(topN, sentences.count))

        let heuristic = heuristicTop(sentences, query: query, count: n)
        let textRank = textRankTop(sentences, count: n)

        // 合并去重：用句子原文（trim 后）作 key，保持启发式优先、再 TextRank 补充；
        // 最终按原文顺序返回，便于模型按叙述顺序理解。
        var picked = Set<String>()
        var merged: [String] = []
        for s in heuristic + textRank where picked.insert(s).inserted {
            merged.append(s)
        }
        let order = Dictionary(sentences.enumerated().map { ($0.element, $0.offset) },
                               uniquingKeysWith: { min($0, $1) })
        return merged.sorted { (order[$0] ?? 0) < (order[$1] ?? 0) }
    }

    /// 把核心句拼成送给模型的短文本（保留句子边界）。
    public static func compressed(_ text: String, query: String? = nil,
                                  topN: Int = perMethodTopN) -> String {
        extract(text, query: query, topN: topN).joined(separator: "\n")
    }

    /// 打分前的统一文本压缩：短文本直通（行为完全不变），超长文本用融合核心句
    /// 替代粗暴 prefix 截断。所有 judgeBinary / score 打分点共用这一个入口。
    /// - Parameters:
    ///   - text: 待压缩原文
    ///   - maxChars: 原文长度上限；超过才压缩（Int.max 表示永不直通）
    ///   - query: 可选查询。**整段判定（judgeBinary）场景必须传 nil**——判定要抛开问句、
    ///     提取代表全文的核心句；只有 query-aware 分片场景才需要它。
    ///   - topN: 每种方法取几句，默认各 10（合计最多 20）
    public static func prepareForScoring(_ text: String, maxChars: Int = 3_000,
                                         query: String? = nil,
                                         topN: Int = perMethodTopN) -> String {
        guard text.count > maxChars || maxChars == Int.max else { return text }
        let core = compressed(text, query: query, topN: topN)
        // 核心句集反而更长时（短句多）退回原文前缀，避免压缩变膨胀。
        return core.isEmpty ? String(text.prefix(maxChars))
            : (core.count < text.count ? core : String(text.prefix(maxChars)))
    }

    // MARK: - 切句

    /// 复用 CharacterNgramBM25 的切句（按中英文句末标点与换行）。
    public static func splitSentences(_ text: String) -> [String] {
        CharacterNgramBM25.sentences(text)
    }

    // MARK: - 方法 A：多信号启发式

    /// 位置启发式：第 1 句 1.0，最后 1 句 0.8，中间 0.2。
    private static func positionScore(index: Int, count: Int) -> Double {
        guard count > 1 else { return 1.0 }
        if index == 0 { return 1.0 }
        if index == count - 1 { return 0.8 }
        return 0.2
    }

    /// 观点词密度：命中词数 / 句长（字符数），做适度放大便于与 [0,1] 信号竞争。
    private static func lexiconScore(_ sentence: String) -> Double {
        guard !sentence.isEmpty else { return 0 }
        var hits = 0
        for word in opinionLexicon {
            var start = sentence.startIndex
            while let range = sentence.range(of: word, range: start..<sentence.endIndex) {
                hits += 1
                start = range.upperBound
                if start >= sentence.endIndex { break }
            }
        }
        // 命中/字符 × 10，单句一个强观点词约 0.1~0.3，多词累加。
        return min(1.0, Double(hits) * 10.0 / Double(sentence.count))
    }

    /// 中心度：句子与全文的字符 2-gram 重叠率（Jaccard），可选与 query 的 BM25 相关性加权。
    private static func centralityScores(_ sentences: [String], query: String?) -> [Double] {
        let global = Set(CharacterNgramBM25.grams(sentences.joined(separator: " "), n: 2))
        guard !global.isEmpty else { return Array(repeating: 0, count: sentences.count) }
        let queryRel = query.map {
            CharacterNgramBM25.scores(query: $0, sentences: sentences,
                                      config: CharacterNgramBM25.Config(ngram: 2))
        }
        let maxRel = queryRel?.max() ?? 0
        return sentences.enumerated().map { index, sentence in
            let grams = Set(CharacterNgramBM25.grams(sentence, n: 2))
            guard !grams.isEmpty else { return 0 }
            let overlap = Double(grams.intersection(global).count) / Double(grams.count)
            guard let rel = queryRel, maxRel > 0 else { return overlap }
            return overlap * 0.7 + (rel[index] / maxRel) * 0.3
        }
    }

    /// 长度惩罚：15~60 字满分；过短/过长按距离线性打折（下限 0.1）。
    private static func lengthScore(_ sentence: String) -> Double {
        let n = sentence.count
        if n >= 15 && n <= 60 { return 1.0 }
        if n < 15 { return max(0.1, Double(n) / 15.0) }
        return max(0.1, 1.0 - Double(n - 60) / 200.0)
    }

    private static func heuristicTop(_ sentences: [String], query: String?,
                                     count: Int) -> [String] {
        let w = Weights()
        let centrality = centralityScores(sentences, query: query)
        let scored = sentences.enumerated().map { index, sentence -> (Int, Double) in
            let score = w.position * positionScore(index: index, count: sentences.count)
                + w.lexicon * lexiconScore(sentence)
                + w.centrality * centrality[index]
                + w.length * lengthScore(sentence)
            return (index, score)
        }
        return scored.sorted { $0.1 > $1.1 }.prefix(count).map { sentences[$0.0] }
    }

    // MARK: - 方法 B：TextRank + 观点词加权

    /// TextRank：句间字符 2-gram Jaccard 相似度构建图，观点句初始分数 2.0 / 普通 0.5，
    /// 迭代 12 轮（damping 0.85）后按得分取 topN。
    private static func textRankTop(_ sentences: [String], count: Int) -> [String] {
        let n = sentences.count
        guard n > 0 else { return [] }
        if n == 1 { return sentences }

        let grams = sentences.map { Set(CharacterNgramBM25.grams($0, n: 2)) }
        var graph = Array(repeating: [Double](repeating: 0, count: n), count: n)
        for i in 0..<n {
            for j in (i + 1)..<n {
                let a = grams[i], b = grams[j]
                guard !a.isEmpty, !b.isEmpty else { continue }
                let sim = Double(a.intersection(b).count) / Double(a.union(b).count)
                guard sim > 0 else { continue }
                graph[i][j] = sim
                graph[j][i] = sim
            }
        }
        let outgoing = graph.map { row in max(1e-9, row.reduce(0, +)) }

        var score = sentences.map { lexiconScore($0) > 0 ? 2.0 : 0.5 }
        let damping = 0.85
        for _ in 0..<12 {
            var next = Array(repeating: 1.0 - damping, count: n)
            for i in 0..<n {
                var inflow = 0.0
                for j in 0..<n where i != j && graph[j][i] > 0 {
                    inflow += graph[j][i] / outgoing[j] * score[j]
                }
                next[i] += damping * inflow
            }
            score = next
        }
        return score.enumerated().sorted { $0.element > $1.element }
            .prefix(count).map { sentences[$0.offset] }
    }
}
