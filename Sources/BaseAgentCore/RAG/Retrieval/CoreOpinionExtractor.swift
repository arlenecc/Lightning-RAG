import Foundation

/// 从长文本中提取核心观点句：两种独立算法各取 topN（默认 10），合并去重后作为最终核心句。
///
/// 用途：决策模型/辅助模型打分前把长文本压缩成「既能全面代表、又足够短」的句集，
/// 避免整段原文导致打分过慢或超上下文。纯函数，不依赖模型与网络。
///
/// 方法 A：多信号启发式打分
///   Score = w1·P_pos + w2·D_lexicon + w3·C_centrality + w4·L_length + w5·F_fact
/// 方法 B：TextRank + 观点词加权初始化（观点句 2.0 / 普通句 0.5）
/// 保真席位：两路 topN 之外，按 F_fact 强制纳入前 3 个数字/实体句。
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

    /// 数字/实体保真席位：即使不在两路 topN 内，也按 F_fact 强制纳入前 3 句
    /// （席位少一些，避免核心句集合过长）。
    public static let factGuaranteeTop = 3

    /// 权重：位置 0.25 / 观点词 0.30 / 中心度 0.30 / 长度 0.15 / 事实保真 0.25。
    public struct Weights: Sendable {
        public var position: Double = 0.25
        public var lexicon: Double = 0.30
        public var centrality: Double = 0.30
        public var length: Double = 0.15
        /// 数字/实体保真加分权重（W_FACT）。
        public var fact: Double = 0.25
        public init() {}
    }

    /// 事实模式：命中越多类别，F_fact 越高（封顶 1.0）。
    /// 覆盖数字+单位、百分比/分数、区间、日期时间、版本号、配置键值、条款编号、
    /// 阈值/期限/审批类约束词 8 类。
    private static let factPatterns: [NSRegularExpression] = {
        let patterns = [
            // 数字 + 常见单位（个/万/亿/人/次/天/小时/年/月/日/元/%/kg/km/MB/GB…）
            "[0-9０-９一二三四五六七八九十百千万亿]+[ ]*(个|万|亿|人|次|天|日|小时|分钟|秒|年|月|周|元|美元|%|％|千克|公斤|克|米|公里|千米|MB|GB|TB|kbps|倍|层|条|页|章)",
            // 百分比 / 分数 / 比率
            "[0-9０-９]+[ ]*(%|％|/|∶|:)[ ]*[0-9０-９]*|百分之[0-9０-９一二三四五六七八九十]+",
            // 区间
            "[0-9０-９]+[ ]*(~|-|—|至|到)[ ]*[0-9０-９]+",
            // 日期 / 时间
            "[0-9０-９]{4}[年/-][0-9０-９]{1,2}[月/-][0-9０-９]{1,2}|[0-9０-９]{1,2}:[0-9０-９]{2}|[0-9０-９]{1,2}时[0-9０-９]{1,2}分",
            // 版本号 / 编号
            "v?[0-9０-９]+\\.[0-9０-９]+(\\.[0-9０-９]+)?|第[0-9０-９一二三四五六七八九十百]+[条款章节条]",
            // 配置键值 / 参数名 = 值
            "[A-Za-z_][A-Za-z0-9_]*[ ]*[=:][ ]*[\"']?[A-Za-z0-9_.-]+",
            // 阈值 / 期限 / 审批类约束词
            "阈值|上限|下限|不少于|不低于|不超过|至少|最多|截止|期限|必须|禁止|需审批|经批准|有效期|默认为|不得超过",
        ]
        return patterns.compactMap { try? NSRegularExpression(pattern: $0, options: .caseInsensitive) }
    }()

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

        // 保真席位：两路 topN 之外，按 F_fact 得分强制纳入前 K 个数字/实体句。
        // 这些句子即使观点/中心度不高，也可能含金额/日期/条款等关键细节，丢失代价高。
        let factQuota = min(factGuaranteeTop, sentences.count)
        let factTop = sentences.enumerated()
            .filter { factScore($0.element) > 0 }
            .sorted { factScore($0.element) > factScore($1.element)
                || (factScore($0.element) == factScore($1.element) && $0.offset < $1.offset) }
            .prefix(factQuota).map(\.element)

        // 合并去重：启发式 → TextRank → 保真席位；最终按原文顺序返回。
        var picked = Set<String>()
        var merged: [String] = []
        for s in heuristic + textRank + factTop where picked.insert(s).inserted {
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

    /// 复用 CharacterNgramBM25 的切句（按中英文句末标点与换行），
    /// 并修复小数点/版本号被误当句末的问题（"0.05%" 会被切成 "0." + "05%"），
    /// 否则数字事实句被拆散、保真信号失效。
    public static func splitSentences(_ text: String) -> [String] {
        let raw = CharacterNgramBM25.sentences(text)
        guard raw.count > 1 else { return raw }
        var merged: [String] = []
        var buffer = raw[0]
        for piece in raw.dropFirst() {
            let a = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
            let b = piece.trimmingCharacters(in: .whitespacesAndNewlines)
            let tail = a.last
            let numericTail = tail.map { $0.isNumber } ?? false
            let dotAfterDigit = (a.hasSuffix(".") || a.hasSuffix("．"))
                && (a.dropLast().last?.isNumber ?? false)
            let numericHead = b.first.map { $0.isNumber } ?? false
            if (numericTail || dotAfterDigit) && numericHead {
                buffer += piece
            } else {
                merged.append(buffer)
                buffer = piece
            }
        }
        merged.append(buffer)
        return merged
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

    /// 数字/实体保真分：按命中的事实模式类别数计分（0~1）。
    /// 每命中一类 +0.25，命中 4 类即满分——避免抽取时丢掉数字、日期、条款等关键细节。
    public static func factScore(_ sentence: String) -> Double {
        guard !sentence.isEmpty else { return 0 }
        let ns = sentence as NSString
        let range = NSRange(location: 0, length: ns.length)
        var hits = 0
        for pattern in factPatterns {
            if pattern.firstMatch(in: sentence, range: range) != nil { hits += 1 }
        }
        return min(1.0, Double(hits) * 0.25)
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
                + w.fact * factScore(sentence)
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
