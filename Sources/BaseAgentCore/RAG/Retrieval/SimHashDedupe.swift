import Foundation

/// SimHash(64) 近似文本去重（移植自 tools/SimHashDedupe.swift，精简为库内算法）。
///
/// - 字符级 3-gram + 2-gram shingle（中文无需分词），停用词/纯虚字过滤，取 TF topN；
/// - MurmurHash3 x64_128 低 64 位做特征哈希，按位加权累加成 64 位指纹；
/// - 4×16 位 LSH 分桶，汉明距离 ≤3 时鸽巢原理保证必召回；
/// - 可选 3-gram Jaccard 内容复核，剔除指纹碰撞导致的误判。
///
/// 用于「解析→清洗→markdown 之后、切片入库之前」的全局近似排重：与已入库文档
/// 高度相似的正文直接跳过，不再切片/向量化。与 VectorStore 的精确 content_hash
/// 去重互补（后者只挡完全相同的正文，SimHash 挡近似副本/轻微改写）。
public enum SimHashDedupe {
    public struct Config: Sendable {
        /// 头/尾各采样字符数（避免首尾拼接产生虚假跨段 shingle）。
        public var sampleChars: Int = 20_000
        /// 3-gram topN。
        public var top3: Int = 300
        /// 2-gram topN。
        public var top2: Int = 200
        /// 判定重复的汉明距离上限（≤3 有 4 段 LSH 必召回保证）。
        public var hammingThreshold: Int = 3
        /// 是否做 3-gram Jaccard 内容复核（距离 1..阈值时启用，防误判）。
        public var verifyContent: Bool = true
        /// Jaccard 下限。默认 0.7：挡近似副本；数字参与指纹后，
        /// 同模板但编号/日期不同的文档仍有足够区分度。
        public var jaccardThreshold: Double = 0.7
        /// 特征哈希 seed（必须稳定，否则跨进程指纹不可比）。
        public var hashSeed: UInt64 = 0

        public init() {}
    }

    // MARK: - MurmurHash3 x64_128（取低 64 位 h1）

    @inline(__always) private static func rotl(_ x: UInt64, _ r: Int) -> UInt64 {
        (x << UInt64(r)) | (x >> UInt64(64 - r))
    }

    @inline(__always) private static func fmix64(_ k0: UInt64) -> UInt64 {
        var k = k0
        k ^= k >> 33
        k &*= 0xff51afd7ed558ccd
        k ^= k >> 33
        k &*= 0xc4ceb9fe1a85ec53
        k ^= k >> 33
        return k
    }

    public static func hash64(_ s: String, seed: UInt64 = 0) -> UInt64 {
        let data = Array(s.utf8)
        let c1: UInt64 = 0x87c37b91114253d5
        let c2: UInt64 = 0x4cf5ad432745937f
        var h1 = seed, h2 = seed
        let blocks = data.count / 16
        for b in 0..<blocks {
            var k1: UInt64 = 0, k2: UInt64 = 0
            for j in 0..<8 { k1 |= UInt64(data[b * 16 + j]) << UInt64(8 * j) }
            for j in 0..<8 { k2 |= UInt64(data[b * 16 + 8 + j]) << UInt64(8 * j) }
            k1 &*= c1; k1 = rotl(k1, 31); k1 &*= c2; h1 ^= k1
            h1 = rotl(h1, 27); h1 &+= h2; h1 = h1 &* 5 &+ 0x52dce729
            k2 &*= c2; k2 = rotl(k2, 33); k2 &*= c1; h2 ^= k2
            h2 = rotl(h2, 31); h2 &+= h1; h2 = h2 &* 5 &+ 0x38495ab5
        }
        let off = blocks * 16
        let tail = data.count - off
        if tail > 0 {
            var k1: UInt64 = 0, k2: UInt64 = 0
            if tail >= 9 {
                for j in 8..<tail { k2 ^= UInt64(data[off + j]) << UInt64(8 * (j - 8)) }
                k2 &*= c2; k2 = rotl(k2, 33); k2 &*= c1; h2 ^= k2
            }
            if tail >= 1 {
                for j in 0..<min(tail, 8) { k1 ^= UInt64(data[off + j]) << UInt64(8 * j) }
                k1 &*= c1; k1 = rotl(k1, 31); k1 &*= c2; h1 ^= k1
            }
        }
        h1 ^= UInt64(data.count); h2 ^= UInt64(data.count)
        h1 &+= h2; h2 &+= h1
        h1 = fmix64(h1); h2 = fmix64(h2)
        h1 &+= h2; h2 &+= h1
        return h1
    }

    // MARK: - 文本清理与代表文本

    /// 汉字保留、A-Z 转小写、a-z 保留，其余剔除。
    @inline(__always) private static func kept(_ scalar: Unicode.Scalar) -> Character? {
        let v = scalar.value
        if (v >= 0x4E00 && v <= 0x9FFF) || (v >= 0x3400 && v <= 0x4DBF)
            || (v >= 0x20000 && v <= 0x2A6DF) || (v >= 0xF900 && v <= 0xFAFF) {
            return Character(scalar)
        } else if v >= 0x41 && v <= 0x5A {
            return UnicodeScalar(v + 32).map { Character($0) }
        } else if v >= 0x61 && v <= 0x7A {
            return Character(scalar)
        } else if v >= 0x30 && v <= 0x39 {
            // 数字必须保留：模板化文档（编号/日期/计数）常只靠数字区分，
            // 过滤数字会把「第1个文件」「第2个文件」误判为完全相同的副本。
            return Character(scalar)
        }
        return nil
    }

    private static func cleanAll(_ text: String) -> [Character] {
        var out: [Character] = []
        for scalar in text.unicodeScalars {
            if let c = kept(scalar) { out.append(c) }
        }
        return out
    }

    /// 头/尾各 sampleChars 两段；短文本全文单段。分段避免首尾拼接产生虚假跨段 shingle。
    private static func segments(_ text: String, sampleChars: Int) -> [[Character]] {
        if sampleChars <= 0 { return [cleanAll(text)] }
        var total = 0
        for scalar in text.unicodeScalars where kept(scalar) != nil { total += 1 }
        if total <= 2 * sampleChars { return [cleanAll(text)] }
        var head: [Character] = [], tail: [Character] = []
        head.reserveCapacity(sampleChars); tail.reserveCapacity(sampleChars)
        var idx = 0
        for scalar in text.unicodeScalars {
            guard let c = kept(scalar) else { continue }
            if idx < sampleChars { head.append(c) }
            if idx >= total - sampleChars { tail.append(c) }
            idx += 1
        }
        return [head, tail]
    }

    // MARK: - 停用词（精简内置：中文高频虚词/虚字，英文功能词）

    /// 只用于过滤 2/3-gram 中「整体由虚词构成」的片段，不追求覆盖全量词表。
    public static let builtinStopGrams: Set<String> = {
        let chinese = "的了和与及或在是有对于不为也都而就被把从这那其之以其所谓以及并且但是因此然后一个一种一些没有可以能够进行通过关于按照根据使用需要应该可能如果虽然即使或者以及以及"
        var set = Set<String>()
        let chars = Array(chinese)
        for c in chars { set.insert(String(c)) }
        // 常见双字虚词
        for g in ["算了", "并且", "而且", "但是", "因此", "然后", "或者", "虽然", "即使", "关于", "按照", "根据", "通过", "进行", "使用", "需要", "应该", "可能", "可以", "能够", "对于", "这个", "那个", "一个", "一种", "一些", "没有", "所谓", "以及", "而是", "就是", "不是", "而已", "等等"] {
            set.insert(g)
        }
        for w in ["the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "has", "have", "had", "but", "not", "you", "your", "its", "of ", " in", " to"] {
            set.insert(w.trimmingCharacters(in: .whitespaces))
        }
        return set
    }()

    /// 纯虚字字符（整体由这些字符构成的 gram 直接丢弃）。
    private static let weakChars: Set<Character> = Set("的了和与及或在是有都不为也都而就被把从这那其之以其")

    private static func grams(_ segs: [[Character]], k: Int, config: Config) -> [String: Int] {
        var counts: [String: Int] = [:]
        let stops = builtinStopGrams
        for seg in segs {
            guard seg.count >= k else { continue }
            for i in 0...(seg.count - k) {
                let g = String(seg[i..<(i + k)])
                if stops.contains(g) { continue }
                if g.allSatisfy({ weakChars.contains($0) }) { continue }
                counts[g, default: 0] += 1
            }
        }
        return counts
    }

    private static func top(_ counts: [String: Int], _ n: Int) -> [(String, Int)] {
        guard n > 0 else { return [] }
        return counts.sorted {
            if $0.value != $1.value { return $0.value > $1.value }
            return $0.key < $1.key
        }.prefix(n).map { ($0.key, $0.value) }
    }

    /// 提取特征（3-gram topN ∪ 2-gram topN，权重=TF）。
    public static func features(_ text: String, config: Config = Config()) -> [(String, Int)] {
        let segs = segments(text, sampleChars: config.sampleChars)
        return top(grams(segs, k: 3, config: config), config.top3)
            + top(grams(segs, k: 2, config: config), config.top2)
    }

    // MARK: - 指纹

    /// 由特征生成 64 位 SimHash（十六进制字符串便于持久化/比较）。
    public static func fingerprint(_ text: String, config: Config = Config()) -> String {
        let feats = features(text, config: config)
        var v = [Int64](repeating: 0, count: 64)
        for (word, weight) in feats {
            let h = hash64(word, seed: config.hashSeed)
            let w = Int64(weight)
            for i in 0..<64 {
                v[i] += ((h >> UInt64(i)) & 1 == 1) ? w : -w
            }
        }
        var fp: UInt64 = 0
        for i in 0..<64 where v[i] > 0 { fp |= (UInt64(1) << UInt64(i)) }
        return String(fp, radix: 16)
    }

    @inline(__always) public static func hamming(_ aHex: String, _ bHex: String) -> Int? {
        guard let a = UInt64(aHex, radix: 16), let b = UInt64(bHex, radix: 16) else { return nil }
        return (a ^ b).nonzeroBitCount
    }

    @inline(__always) private static func band(_ fp: UInt64, _ b: Int) -> UInt16 {
        UInt16(truncatingIfNeeded: fp >> UInt64(16 * b))
    }

    // MARK: - LSH 索引

    /// 从「文档 key → 指纹十六进制」构建 4×16 位 LSH 索引；内存态，查询 O(1) 粗排。
    public final class Index: @unchecked Sendable {
        private let lock = NSLock()
        private var docs: [(key: String, fp: UInt64)] = []
        private var tables: [[UInt16: [Int]]] = Array(repeating: [:], count: 4)

        public init() {}

        public func insert(key: String, fingerprintHex: String) {
            guard let fp = UInt64(fingerprintHex, radix: 16) else { return }
            lock.lock(); defer { lock.unlock() }
            let id = docs.count
            docs.append((key, fp))
            for b in 0..<4 { tables[b][band(fp, b), default: []].append(id) }
        }

        /// 返回与给定指纹最相似的已入库文档（汉明距离 ≤ threshold）。
        public func nearest(_ fingerprintHex: String, threshold: Int) -> (key: String, distance: Int)? {
            guard let fp = UInt64(fingerprintHex, radix: 16) else { return nil }
            lock.lock(); defer { lock.unlock() }
            var candidates = Set<Int>()
            for b in 0..<4 {
                if let ids = tables[b][band(fp, b)] { candidates.formUnion(ids) }
            }
            var best: (String, Int)?
            for id in candidates {
                let d = (fp ^ docs[id].fp).nonzeroBitCount
                if d <= threshold, best == nil || d < best!.1 { best = (docs[id].key, d) }
            }
            return best
        }
    }

    /// 内容复核：两段 markdown 的 3-gram Jaccard（用独立代表文本，抗模板陷阱）。
    public static func jaccard(_ a: String, _ b: String, config: Config = Config()) -> Double {
        func grams(_ text: String) -> Set<String> {
            var s = Set<String>()
            for seg in segments(text, sampleChars: config.sampleChars) {
                guard seg.count >= 3 else { continue }
                for i in 0...(seg.count - 3) { s.insert(String(seg[i..<(i + 3)])) }
            }
            return s
        }
        let ga = grams(a), gb = grams(b)
        let inter = ga.intersection(gb).count
        let union = ga.count + gb.count - inter
        guard union > 0 else { return 0 }
        return Double(inter) / Double(union)
    }
}
