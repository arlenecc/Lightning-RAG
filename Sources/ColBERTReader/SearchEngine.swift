import Foundation

struct SearchHit {
    let docID: String
    let title: String
    /// 文档来源（文件相对路径/文件名），用于引用气泡展示与来源归属。
    let source: String
    let content: String
    let score: Float
    let vectorRank: Int?
    let bm25Rank: Int?
    let rrf: Double
}

struct Candidate {
    var docID: String
    var rrf: Double = 0
    var vectorRank: Int? = nil
    var bm25Rank: Int? = nil
    var vectorDistance: Double? = nil
    var bm25Score: Double? = nil
}

/// 建库 / 入库 / 检索。三张要求的表 + 一个 FTS5 BM25 索引（不是数据表）。
final class SearchEngine {
    let db: SQLiteDB
    let colbert: ColBERT
    let options: ChunkOptions
    private let rrfK = 60.0
    /// 最近一次检索的向量通道（ann / vec0-exact），供 CLI 输出。
    private(set) var lastVectorBackend = "vec0-exact"

    init(db: SQLiteDB, colbert: ColBERT, options: ChunkOptions = ChunkOptions()) {
        self.db = db
        self.colbert = colbert
        self.options = options
    }

    static func createSchema(_ db: SQLiteDB) throws {
        try db.exec("""
        CREATE TABLE IF NOT EXISTS document (
            doc_id TEXT PRIMARY KEY,
            title  TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL
        );
        """)
        // 旧库补列：早期版本没有 source 列，补上后旧行取默认空串，
        // 检索侧回退到 title，新入库的文档会写入真实来源。
        if !columnExists(db, table: "document", column: "source") {
            try db.exec("ALTER TABLE document ADD COLUMN source TEXT NOT NULL DEFAULT ''")
        }
        try db.exec("""
        CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(
            doc_id UNINDEXED,
            title_tokens,
            content_tokens,
            tokenize = 'unicode61 remove_diacritics 2'
        );
        """)
        // sqlite-vec 0.1.9 的 vec0 只支持向量列；用 rowid 与 document 行一一对齐。
        // embedding 列不能 SELECT 出原始字节，另建普通镜像表供 ANN/反量化读取。
        try db.exec("""
        CREATE VIRTUAL TABLE IF NOT EXISTS doc_summary_vec USING vec0(
            embedding float[\(ColBERT.dimension)] distance_metric=cosine
        );
        CREATE TABLE IF NOT EXISTS doc_vectors (
            rowid INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL
        );
        """)
        try db.exec("""
        CREATE TABLE IF NOT EXISTS doc_residuals (
            doc_id TEXT PRIMARY KEY,
            token_count INTEGER NOT NULL,
            dim INTEGER NOT NULL,
            quant_bits INTEGER NOT NULL,
            scale REAL NOT NULL,
            offset REAL NOT NULL,
            residual_blob BLOB NOT NULL
        );
        """)
        try ANNIndex(db: db).createSchema()
        if Self.meta(db, "doc_count") == nil {
            let count = (try? db.query("SELECT COUNT(*) AS n FROM document").first?["n"]?.intValue) ?? 0
            Self.setMeta(db, "doc_count", String(count))
        }
        if Self.meta(db, "doc_rowid_seq") == nil {
            let maxID = (try? db.query("SELECT COALESCE(MAX(rowid),0) AS n FROM document")
                .first?["n"]?.intValue) ?? 0
            Self.setMeta(db, "doc_rowid_seq", String(maxID))
        }
    }

    // MARK: - meta helpers

    /// 表是否已有指定列（用于旧库无损补列）。
    private static func columnExists(_ db: SQLiteDB, table: String, column: String) -> Bool {
        let rows = (try? db.query("PRAGMA table_info(\(table))")) ?? []
        return rows.contains { $0["name"]?.stringValue == column }
    }

    private static func meta(_ db: SQLiteDB, _ key: String) -> String? {
        try? db.query("SELECT value FROM doc_meta WHERE key = ?", [.text(key)])
            .first?["value"]?.stringValue
    }

    private static func setMeta(_ db: SQLiteDB, _ key: String, _ value: String) {
        try? db.execute("""
        INSERT INTO doc_meta(key,value) VALUES(?,?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, [.text(key), .text(value)])
    }

    // MARK: - 入库

    func addFile(path: String, title: String? = nil, bits: Int = 4,
                 progress: ((Int, Int) -> Void)? = nil,
                 timing: ((String, Double) -> Void)? = nil) throws -> Int {
        guard let data = FileManager.default.contents(atPath: path),
              let text = String(data: data, encoding: .utf8) else {
            throw DocSearchError.argument("无法读取 UTF-8 文本文件: \(path)")
        }
        let base = ((path as NSString).lastPathComponent as NSString).deletingPathExtension
        return try addMarkdown(text, source: base, title: title, bits: bits,
                               progress: progress, timing: timing)
    }

    /// 直接入库一段 markdown 文本（供主应用在文档解析后旁路接入）。
    /// `sourceKey` 非空时作为 doc_id 的稳定前缀（用于按来源删除/计数）；
    /// 为空时回退到「文件名 + 内容哈希」的旧式前缀（保持 CLI 兼容）。
    func addMarkdown(_ text: String, source: String, title: String? = nil, bits: Int = 4,
                     sourceKey: String? = nil,
                     progress: ((Int, Int) -> Void)? = nil,
                     timing: ((String, Double) -> Void)? = nil) throws -> Int {
        var base = (source as NSString).lastPathComponent
        if !base.isEmpty {
            base = (base as NSString).deletingPathExtension
        }
        if base.isEmpty { base = "doc" }
        let sourceID = sourceKey ?? (base + "-" + hash8(text))
        let docTitle = title ?? base
        let addStarted = Date()
        // 大分片不依赖模型做语义切分：每个 8000 字符片段的边界由结构与词法相似度决定。
        // 之前把 ColBERT 传入切片器，需要在整本书的每个句子上跑一次模型前向
        // （一本书数千句 × 24 层），是同步卡住的主因之一。
        let chunks = try Chunker.split(text, sourceName: docTitle, model: nil, options: options)
        let chunkingTime = Date().timeIntervalSince(addStarted)
        timing?("chunking", chunkingTime)
        guard !chunks.isEmpty else { throw DocSearchError.chunking("没有产生任何切片") }

        var encodeTotal = 0.0, quantTotal = 0.0, dbTotal = 0.0
        var termTotal = 0, tokenTotal = 0
        return try db.transaction {
            // 同一文件重复入库时先清理旧切片（前缀是确定性的 sourceID）。
            try Self.deleteChunks(db, sourceID)

            var rowSeq = Int64(Self.meta(db, "doc_rowid_seq") ?? "") ?? 0
            var added = 0
            for (i, chunk) in chunks.enumerated() {
                let id = String(format: "%@#%04d", sourceID, i)
                var stage = Date()
                // 整段编码：一次前向得到切片内每个内容 token 的向量。
                // 旧实现逐词编码，每个词都要重复一次完整 24 层前向（一本书数小时），
                // 且丢失词序；整段编码与标准 ColBERT 一致，入库降到秒级。
                let vectors = try colbert.encodeDocument(chunk.content)
                encodeTotal += Date().timeIntervalSince(stage)
                guard !vectors.isEmpty else { continue }
                termTotal += vectors.count
                tokenTotal += vectors.count

                stage = Date()
                let record = try ResidualCodec.quantize(docID: id, vectors: vectors,
                                                        dim: ColBERT.dimension, bits: bits)
                quantTotal += Date().timeIntervalSince(stage)

                stage = Date()
                rowSeq += 1
                try db.execute("INSERT INTO document(rowid,doc_id,title,source,content) VALUES(?,?,?,?,?)",
                               [.int(rowSeq), .text(id), .text(chunk.title),
                                .text(docTitle), .text(chunk.content)])
                try db.execute("INSERT INTO document_fts(doc_id,title_tokens,content_tokens) VALUES(?,?,?)",
                               [.text(id), .text(TextAnalysis.ftsIndexText(chunk.title)),
                                .text(TextAnalysis.ftsIndexText(chunk.content))])
                try db.execute("INSERT INTO doc_summary_vec(rowid,embedding) VALUES(?,?)",
                               [.int(rowSeq), .floats(record.center)])
                try db.execute("INSERT INTO doc_vectors(rowid,embedding) VALUES(?,?)",
                               [.int(rowSeq), .floats(record.center)])
                try db.execute("""
                    INSERT INTO doc_residuals(doc_id,token_count,dim,quant_bits,scale,offset,residual_blob)
                    VALUES(?,?,?,?,?,?,?)
                    """, [.text(id), .int(Int64(record.tokenCount)), .int(Int64(record.dim)), .int(Int64(record.bits)),
                          .double(Double(record.scale)), .double(Double(record.offset)), .blob(record.blob)])
                dbTotal += Date().timeIntervalSince(stage)
                added += 1
                progress?(i + 1, chunks.count)
            }
            let previous = Int(Self.meta(db, "doc_count") ?? "") ?? 0
            Self.setMeta(db, "doc_count", String(previous + added))
            Self.setMeta(db, "doc_rowid_seq", String(rowSeq))
            let total = Date().timeIntervalSince(addStarted)
            timing?("encode_total", encodeTotal)
            timing?("quantize_total", quantTotal)
            timing?("db_write_total", dbTotal)
            timing?("add_total", total)
            timing?("terms", Double(termTotal))
            timing?("tokens", Double(tokenTotal))
            timing?("chunks", Double(added))
            return chunks.count
        }
    }

    // MARK: - 按来源删除 / 计数

    /// 按稳定前缀删除某来源的全部切片（不开事务，调用方负责包裹）。
    @discardableResult
    private static func deleteChunks(_ db: SQLiteDB, _ sourceKey: String) throws -> Int {
        let ann = ANNIndex(db: db)
        let rows = try db.query("SELECT doc_id, rowid FROM document WHERE doc_id LIKE ?",
                                [.text(sourceKey + "#%")])
        var staleIDs: [Int64] = []
        var removed = 0
        for row in rows {
            guard let id = row["doc_id"]?.stringValue else { continue }
            if let rowID = row["rowid"]?.intValue {
                staleIDs.append(rowID)
                try db.execute("DELETE FROM doc_summary_vec WHERE rowid = ?", [.int(rowID)])
                try db.execute("DELETE FROM doc_vectors WHERE rowid = ?", [.int(rowID)])
            }
            try db.execute("DELETE FROM document WHERE doc_id = ?", [.text(id)])
            try db.execute("DELETE FROM document_fts WHERE doc_id = ?", [.text(id)])
            try db.execute("DELETE FROM doc_residuals WHERE doc_id = ?", [.text(id)])
            removed += 1
        }
        try ann.recordStale(rowIDs: staleIDs)
        return removed
    }

    /// 删除某来源的全部切片并同步 doc_count（事务包裹）。
    @discardableResult
    static func deleteSource(_ db: SQLiteDB, _ sourceKey: String) throws -> Int {
        try db.transaction {
            let removed = try deleteChunks(db, sourceKey)
            let previous = Int(meta(db, "doc_count") ?? "") ?? 0
            setMeta(db, "doc_count", String(max(0, previous - removed)))
            return removed
        }
    }

    /// 某来源的切片数（不依赖模型加载）。
    static func chunkCount(_ db: SQLiteDB, _ sourceKey: String) -> Int {
        let rows = try? db.query("SELECT COUNT(*) AS n FROM document WHERE doc_id LIKE ?",
                                 [.text(sourceKey + "#%")])
        return Int(rows?.first?["n"]?.intValue ?? 0)
    }

    /// 全库切片总数（不依赖模型加载）。
    static func totalChunkCount(_ db: SQLiteDB) -> Int {
        let rows = try? db.query("SELECT COUNT(*) AS n FROM document")
        return Int(rows?.first?["n"]?.intValue ?? 0)
    }

    // MARK: - 检索

    func search(query: String, topN: Int = 5, coarse: Int = 20,
                useANN: Bool = true, nprobe: Int = ANNIndex.defaultNprobe,
                rescore: Int = ANNIndex.defaultRescore, forceANN: Bool = false,
                timing: ((String, Double) -> Void)? = nil) throws -> [SearchHit] {
        guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }
        let searchStarted = Date()
        var stage = Date()
        let center = try colbert.queryCenter(query)
        timing?("query_center", Date().timeIntervalSince(stage))
        let fetchK = max(60, coarse * 3)
        var candidates: [String: Candidate] = [:]

        stage = Date()
        var pairs: [(id: String, score: Double, distance: Double?)] = []
        let ann = ANNIndex(db: db)
        let totalRows = Int(Self.meta(db, "doc_count") ?? "") ?? 0
        if useANN, (forceANN || totalRows >= ANNIndex.minRowsForANN), ann.status().usable,
           let found = ann.candidates(query: center, topK: fetchK,
                                      nprobe: nprobe, rescore: rescore), !found.isEmpty {
            lastVectorBackend = "ANN(nprobe=\(nprobe),rescore=\(rescore))"
            for item in found {
                guard let id = try db.query("SELECT doc_id FROM document WHERE rowid = ?",
                                            [.int(item.rowID)]).first?["doc_id"]?.stringValue else { continue }
                pairs.append((id, item.score, nil))
            }
        }
        if pairs.isEmpty {
            lastVectorBackend = "vec0-exact"
            let rows = try db.query("""
                SELECT rowid, distance FROM doc_summary_vec
                WHERE embedding MATCH ? AND k = ? ORDER BY distance
                """, [.floats(center), .int(Int64(fetchK))])
            for row in rows {
                guard let rowID = row["rowid"]?.intValue,
                      let id = try db.query("SELECT doc_id FROM document WHERE rowid = ?",
                                            [.int(rowID)]).first?["doc_id"]?.stringValue else { continue }
                pairs.append((id, 1.0 / (1.0 + (row["distance"]?.doubleValue ?? 2.0)),
                              row["distance"]?.doubleValue ?? nil))
            }
        }
        timing?("vector_coarse", Date().timeIntervalSince(stage))
        stage = Date()
        for (rank, pair) in pairs.enumerated() {
            var c = candidates[pair.id] ?? Candidate(docID: pair.id)
            c.vectorRank = rank + 1
            c.vectorDistance = pair.distance
            c.rrf += 1.0 / (rrfK + Double(rank + 1))
            candidates[pair.id] = c
        }

        var bm25Rows: [[String: SQLValue]] = []
        for mode in ["AND", "OR"] {
            let match = TextAnalysis.ftsQuery(query, mode: mode)
            guard !match.isEmpty else { continue }
            bm25Rows = try db.query("""
                SELECT doc_id, bm25(document_fts) AS score FROM document_fts
                WHERE document_fts MATCH ? ORDER BY score LIMIT ?
                """, [.text(match), .int(Int64(fetchK))])
            if !bm25Rows.isEmpty { break }
        }
        for (rank, row) in bm25Rows.enumerated() {
            guard let id = row["doc_id"]?.stringValue else { continue }
            var c = candidates[id] ?? Candidate(docID: id)
            c.bm25Rank = rank + 1
            c.bm25Score = row["score"]?.doubleValue ?? nil
            c.rrf += 1.0 / (rrfK + Double(rank + 1))
            candidates[id] = c
        }

        timing?("bm25_coarse", Date().timeIntervalSince(stage))
        stage = Date()
        let shortlist = candidates.values.sorted {
            if $0.rrf != $1.rrf { return $0.rrf > $1.rrf }
            return $0.docID < $1.docID
        }.prefix(max(1, coarse))
        timing?("rrf", Date().timeIntervalSince(stage))
        timing?("coarse_total", Date().timeIntervalSince(searchStarted))

        stage = Date()
        let queryTokens = try colbert.encode([query], kind: .query).first ?? []
        timing?("query_tokens", Date().timeIntervalSince(stage))
        guard !queryTokens.isEmpty else { return [] }

        var loadTotal = 0.0, maxsimTotal = 0.0
        var hits: [SearchHit] = []
        for candidate in shortlist {
            var itemStart = Date()
            guard let record = loadResidual(candidate.docID),
                  let meta = try loadMeta(candidate.docID) else { continue }
            loadTotal += Date().timeIntervalSince(itemStart)
            itemStart = Date()
            let score = try ResidualCodec.maxSim(query: queryTokens, record: record)
            maxsimTotal += Date().timeIntervalSince(itemStart)
            hits.append(SearchHit(docID: candidate.docID, title: meta.title,
                                  source: meta.source, content: meta.content,
                                  score: score, vectorRank: candidate.vectorRank,
                                  bm25Rank: candidate.bm25Rank, rrf: candidate.rrf))
        }
        hits.sort { $0.score != $1.score ? $0.score > $1.score : $0.docID < $1.docID }
        timing?("residual_load", loadTotal)
        timing?("maxsim", maxsimTotal)
        timing?("rerank", loadTotal + maxsimTotal)
        timing?("shortlist", Double(shortlist.count))
        timing?("search_total", Date().timeIntervalSince(searchStarted))
        return Array(hits.prefix(max(1, topN)))
    }

    private func loadMeta(_ docID: String) throws -> (title: String, source: String,
                                                      content: String)? {
        let rows = try db.query("SELECT title, source, content FROM document WHERE doc_id = ?",
                                [.text(docID)])
        guard let row = rows.first else { return nil }
        return (row["title"]?.stringValue ?? "",
                row["source"]?.stringValue ?? "",
                row["content"]?.stringValue ?? "")
    }

    private func loadResidual(_ docID: String) -> ResidualRecord? {
        guard let rows = try? db.query("""
            SELECT token_count, dim, quant_bits, scale, offset, residual_blob
            FROM doc_residuals WHERE doc_id = ?
            """, [.text(docID)]), let row = rows.first,
              let count = row["token_count"]?.intValue,
              let dim = row["dim"]?.intValue,
              let bits = row["quant_bits"]?.intValue,
              let scale = row["scale"]?.doubleValue,
              let offset = row["offset"]?.doubleValue,
              let blob = row["residual_blob"]?.dataValue,
              let rowRows = try? db.query("SELECT rowid FROM document WHERE doc_id = ?", [.text(docID)]),
              let rowID = rowRows.first?["rowid"]?.intValue,
              let vecRows = try? db.query("SELECT embedding FROM doc_vectors WHERE rowid = ?", [.int(rowID)]),
              let centerData = vecRows.first?["embedding"]?.dataValue else { return nil }
        guard centerData.count == Int(dim) * MemoryLayout<Float>.stride else { return nil }
        let centerFloats = centerData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let center = centerFloats.map { Float16($0) }
        guard center.count == dim else { return nil }
        return ResidualRecord(docID: docID, tokenCount: Int(count), dim: Int(dim), bits: Int(bits),
                              scale: Float(scale), offset: Float(offset),
                              centerFP16: center, blob: blob)
    }

    // MARK: - helpers

    private func blob16(_ values: [Float16]) -> Data {
        values.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private func decode16(_ data: Data) -> [Float16] {
        guard data.count % 2 == 0 else { return [] }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
    }

    private func hash8(_ text: String) -> String {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x100000001b3
        }
        return String(format: "%08llx", hash & 0xffff_ffff)
    }
}
