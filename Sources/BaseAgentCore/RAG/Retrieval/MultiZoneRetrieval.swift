import Foundation

/// 多知识分区联合检索：把每个分区各自召回的 citations 做 Weighted RRF 融合。
///
/// 各分区是完全独立的 sqlite 库，rowID 会重复，所以不能直接用 RAGEngine.fuseSourceGroups
/// （它以 rowID 为键会把不同分区的命中错误合并）。这里用 `分区名:id` 复合键。
public enum MultiZoneRetrieval {
    /// 一个分区的召回结果。
    public struct ZoneResult: Sendable {
        /// 分区唯一 id（用于融合键，避免领域名重名误合并）。
        public let zoneID: String
        public let zoneName: String
        public let citations: [RAGCitation]
        /// 该分区融合权重（决策模式第一次判定的相关分；普通联合检索为 1）。
        public let weight: Double
        public init(zoneID: String, zoneName: String, citations: [RAGCitation],
                    weight: Double = 1) {
            self.zoneID = zoneID
            self.zoneName = zoneName
            self.citations = citations
            self.weight = weight
        }
        /// 兼容旧调用（仅有名字时用名字当 id）。
        public init(zoneName: String, citations: [RAGCitation], weight: Double = 1) {
            self.init(zoneID: zoneName, zoneName: zoneName,
                      citations: citations, weight: weight)
        }
    }

    /// 跨分区融合。rank 用各分区内部名次（从 1 起），与分区权重无关；
    /// 若 needsWeight 则由调用方对分区结果先做粗筛权重，这里只做等权 RRF 排名融合。
    /// - Parameters:
    ///   - results: 每个分区的召回（已按相关度排序）
    ///   - topK: 最终返回条数
    ///   - rrfK: RRF 常数（默认 60）
    ///   - vectorWeight: 不参与本函数（RRF 融合为跨分区名次融合，单库内融合已用权重）
    public static func fuse(_ results: [ZoneResult], topK: Int,
                            rrfK: Int = 60) -> [RAGCitation] {
        guard topK > 0, !results.isEmpty else { return [] }
        let k = Double(max(1, rrfK))
        struct Entry { var citation: RAGCitation; var score: Double }
        var entries: [String: Entry] = [:]
        for result in results {
            for (rank, citation) in result.citations.enumerated() {
                let key = "\(result.zoneID)|\(citation.id)"
                let contribution = max(0, result.weight) / (k + Double(rank + 1))
                if var existing = entries[key] {
                    existing.score += contribution
                    entries[key] = existing
                } else {
                    var copy = citation
                    copy.id = key // 跨分区唯一 id，避免 UI 气泡串台
                    entries[key] = Entry(citation: copy, score: contribution)
                }
            }
        }
        let sorted = entries.values.sorted { $0.score > $1.score }.prefix(topK)
        return sorted.enumerated().map { index, entry in
            var citation = entry.citation
            citation.rank = index + 1
            citation.score = entry.score
            citation.isPrimary = index == 0
            return citation
        }
    }

    /// 跨库融合的中间候选：保留 zoneID + 库内原始 id，供「融合后按来源路由精排」。
    public struct ZoneCandidate: Sendable {
        public let zoneID: String
        public let zoneName: String
        /// 库内原始 id（向量=rowID 字符串，ColBERT=docID），不含分区前缀。
        public let localID: String
        public let text: String
        public let source: String
        public let heading: String
        public var rrfScore: Double
        /// 库内向量/BM25 粗筛分（供无精排降级时排序）。
        public let baseScore: Double
    }

    /// 一库的粗筛候选组。
    public struct CandidateGroup: Sendable {
        public let zoneID: String
        public let zoneName: String
        public let weight: Double
        public let candidates: [ZoneCandidate]
        public init(zoneID: String, zoneName: String, weight: Double = 1,
                    candidates: [ZoneCandidate]) {
            self.zoneID = zoneID; self.zoneName = zoneName
            self.weight = weight; self.candidates = candidates
        }
    }

    /// 跨库粗筛候选 RRF 融合，返回带 zoneID 的中间候选（不丢库内 id）。
    public static func fuseCandidates(_ groups: [CandidateGroup], topK: Int,
                                      rrfK: Int = 60) -> [ZoneCandidate] {
        guard topK > 0 else { return [] }
        let k = Double(max(1, rrfK))
        struct Entry { var candidate: ZoneCandidate; var score: Double }
        var entries: [String: Entry] = [:]
        for group in groups {
            for (rank, c) in group.candidates.enumerated() {
                let key = "\(group.zoneID)|\(c.localID)"
                let contribution = max(0, group.weight) / (k + Double(rank + 1))
                if var existing = entries[key] {
                    existing.score += contribution
                    entries[key] = existing
                } else {
                    entries[key] = Entry(candidate: c, score: contribution)
                }
            }
        }
        return entries.values.sorted { $0.score > $1.score }
            .prefix(topK).map { entry in
                // rrfScore 用融合后的总分覆盖；baseScore 保留原粗筛分。
                var updated = entry.candidate
                updated.rrfScore = entry.score
                return updated
            }
    }

    /// 由融合后的 citations 生成进模型上下文的 top-1 文本。
    public static func primaryContext(_ citations: [RAGCitation]) -> String {
        guard let primary = citations.first(where: { $0.isPrimary }) ?? citations.first else {
            return "（知识库中未找到相关内容）"
        }
        let location = [primary.source, primary.heading]
            .filter { !$0.isEmpty }.joined(separator: " · ")
        let header = location.isEmpty ? "相关切片：" : "相关切片（\(location)）："
        return header + "\n" + primary.text
    }
}
