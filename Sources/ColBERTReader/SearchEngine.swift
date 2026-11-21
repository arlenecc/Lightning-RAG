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
    /// ANN 索引长期持有以复用簇心解码缓存（每次检索 new 会导致全量重解码）。
    /// centroidCache 用 blocks/clusters/dim 做失效校验，build/reset 后会自清。
    let ann: ANNIndex
    private let rrfK = 60.0
    /// 最近一次检索的向量通道（ann / vec0-exact），供 CLI 输出。
    /// 最近一次检索的粗筛后端（诊断展示用）。检索已无 opLock 并发进入，
    /// 用锁保护这个 engine 级可变状态；展示值允许竞态，读取方拿近似值即可。
    private let backendLock = NSLock()
    private var lastVectorBackendStorage = "vec0-exact"
    var lastVectorBackend: String {
        get { backendLock.lock(); defer { backendLock.unlock() }; return lastVectorBackendStorage }
        set { backendLock.lock(); defer { backendLock.unlock() }; lastVectorBackendStorage = newValue }
    }

    init(db: SQLiteDB, colbert: ColBERT, options: ChunkOptions = ChunkOptions()) {
        self.db = db
        self.colbert = colbert
        self.options = options
        self.ann = ANNIndex(db: db)
    }

    static func createSchema(_ db: SQLiteDB) throws {
        try createTables(db)
        try ANNIndex(db: db).createSchema()

        // 维度守卫：换用非 64 维 ColBERT 模型时，旧向量表与当前模型维度不符。
        // 向量/残差/文档共用同一 rowid 关联，必须整体重建，否则 vec0 查询与
        // 中心向量解码全部错乱。维度记录在 doc_meta.vec_dim。
        let recordedDim = Int(Self.meta(db, "vec_dim") ?? "") ?? 0
        if recordedDim != 0 && recordedDim != ColBERT.dimension {
            // ColBERTReader 是纯库（无日志设施）；重建通过 doc_count=0 体现，
            // 上层同步面板会显示切片数为 0，提示用户重新同步。
            try ANNIndex(db: db).drop()
            for table in ["document", "document_fts", "doc_summary_vec",
                          "doc_vectors", "doc_residuals", "pending_colbert"] {
                try db.execute("DROP TABLE IF EXISTS \(table)")
            }
            try createTables(db)
            // drop() 删掉了 ann_blocks/ann_cells，必须在本会话内重建空表，
            // 否则 status()/candidates() 查询缺表被吞错，ANN 静默失效到下次进程启动。
            try ANNIndex(db: db).createSchema()
            Self.setMeta(db, "doc_count", "0")
            Self.setMeta(db, "doc_rowid_seq", "0")
            // 必须回写新维度：doc_meta 不在 DROP 列表中，旧值残留会让下次
            // 打开再次触发本分支，把重新同步的数据整库清掉。
            Self.setMeta(db, "vec_dim", String(ColBERT.dimension))
        }
        // 空库首次打开（或刚重建）：记录当前模型维度；有数据的旧库（无维度
        // 记录）视为历史 64 维库，仅在维度确实一致时补记，避免误写。
        if Self.meta(db, "vec_dim") == nil {
            let hasData = ((try? db.query("SELECT COUNT(*) AS n FROM document"))?
                .first?["n"]?.intValue) ?? 0 > 0
            if !hasData || ColBERT.dimension == 64 {
                Self.setMeta(db, "vec_dim", String(ColBERT.dimension))
            }
        }
        if Self.meta(db, "doc_count") == nil {
            let count = (try? db.query("SELECT COUNT(*) AS n FROM document").first?["n"]?.intValue) ?? 0
            Self.setMeta(db, "doc_count", String(count))
        }
        if Self.meta(db, "doc_rowid_seq") == nil {
            let maxID = (try? db.query("SELECT COALESCE(MAX(rowid),0) AS n FROM document")
                .first?["n"]?.intValue) ?? 0
            Self.setMeta(db, "doc_rowid_seq", String(maxID))
        }
        // 旧库兼容：pending_colbert 新增 created_at 列（退避重试用）。
        let hasCreatedAt = (try? db.query("PRAGMA table_info(pending_colbert)")
            .contains { $0["name"]?.stringValue == "created_at" }) ?? true
        if !hasCreatedAt {
            try? db.execute("ALTER TABLE pending_colbert ADD COLUMN created_at REAL NOT NULL DEFAULT 0")
            try? db.execute("CREATE INDEX IF NOT EXISTS idx_pending_colbert_time ON pending_colbert(created_at)")
        }
    }

    /// 显式重建 ColBERT 库（用户点「重建向量库」时在极速模式调用）：
    /// 清掉全部切片/向量/pending/ANN，重建空表。返回被清掉的切片数。
    static func rebuildDatabase(_ db: SQLiteDB) throws -> Int {
        let removed = try totalChunkCount(db)
        try ANNIndex(db: db).drop()
        try db.transaction {
            for table in ["document", "document_fts", "doc_summary_vec",
                          "doc_vectors", "doc_residuals", "pending_colbert"] {
                try db.execute("DROP TABLE IF EXISTS \(table)")
            }
        }
        try createTables(db)
        try ANNIndex(db: db).createSchema()
        setMeta(db, "doc_count", "0")
        setMeta(db, "doc_rowid_seq", "0")
        setMeta(db, "vec_dim", String(ColBERT.dimension))
        return removed
    }

    /// 全部数据表（document/document_fts/doc_summary_vec/doc_vectors/doc_residuals）。
    /// 维度守卫重建与首次建表共用；doc_meta/ANN 表由调用方单独管理。
    private static func createTables(_ db: SQLiteDB) throws {
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
        -- 异步 ColBERT 编码回填队列：document 正文先行入库，残差/向量由后台补齐。
        CREATE TABLE IF NOT EXISTS pending_colbert (
            doc_id TEXT PRIMARY KEY,
            rowid INTEGER NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL,
            bits INTEGER NOT NULL DEFAULT 4,
            created_at REAL NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_pending_colbert_rowid ON pending_colbert(rowid);
        CREATE INDEX IF NOT EXISTS idx_pending_colbert_time ON pending_colbert(created_at);
        """)
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
                 similarity: (([String]) -> [Double]?)? = nil,
                 shouldCancel: (() -> Bool)? = nil,
                 progress: ((Int, Int) -> Void)? = nil,
                 timing: ((String, Double) -> Void)? = nil) throws -> Int {
        guard let data = FileManager.default.contents(atPath: path),
              let text = String(data: data, encoding: .utf8) else {
            throw DocSearchError.argument("无法读取 UTF-8 文本文件: \(path)")
        }
        let base = ((path as NSString).lastPathComponent as NSString).deletingPathExtension
        return try addMarkdown(text, source: base, title: title, bits: bits,
                               similarity: similarity, shouldCancel: shouldCancel,
                               progress: progress, timing: timing)
    }

    /// 直接入库一段 markdown 文本（供主应用在文档解析后旁路接入）。
    /// `sourceKey` 非空时作为 doc_id 的稳定前缀（用于按来源删除/计数）；
    /// 为空时回退到「文件名 + 内容哈希」的旧式前缀（保持 CLI 兼容）。
    /// `similarity` 是相邻文本块的语义相似度提供者（上层注入 potion 等
    /// 静态词向量模型）；为空或失败时切片器回退词法相似度。
    /// `shouldCancel` 在每片编码前检查，用户停止同步时整文件事务回滚。
    func addMarkdown(_ text: String, source: String, title: String? = nil, bits: Int = 4,
                     sourceKey: String? = nil,
                     similarity: (([String]) -> [Double]?)? = nil,
                     shouldCancel: (() -> Bool)? = nil,
                     progress: ((Int, Int) -> Void)? = nil,
                     timing: ((String, Double) -> Void)? = nil) throws -> Int {
        var base = (source as NSString).lastPathComponent
        if !base.isEmpty {
            base = (base as NSString).deletingPathExtension
        }
        if base.isEmpty { base = "doc" }
        let sourceID = sourceKey ?? (base + "-" + Self.hash8(text))
        let docTitle = title ?? base
        let addStarted = Date()
        // 边界优先落在语义低谷（相邻句向量余弦最低处），模型不可用时
        // 自动回退词法相似度；切片边界始终在句子/原子边界上。
        let chunks = try Chunker.split(text, sourceName: docTitle,
                                       similarity: similarity, options: options)
        let chunkingTime = Date().timeIntervalSince(addStarted)
        timing?("chunking", chunkingTime)
        guard !chunks.isEmpty else { throw DocSearchError.chunking("没有产生任何切片") }

        var encodeTotal = 0.0, quantTotal = 0.0, dbTotal = 0.0
        var termTotal = 0, tokenTotal = 0
        // 编码移出写事务：事务内逐片 ONNX 编码会让 SQLiteDB 递归锁持有分钟级，
        // 并发检索/状态查询全部被串行阻塞。改为分批「事务外编码 → 短事务写入」，
        // 写锁窗口降到毫秒级；中断时已写批保留（manifest 未标记完成，下次重做
        // 会先清旧前缀再写，与「上次未完整完成」重试机制一致，最终一致）。
        let encodeBatch = 32
        var rowSeq = Int64(Self.meta(db, "doc_rowid_seq") ?? "") ?? 0
        var added = 0
        // 清理同源旧切片必须在批次循环外独立事务完成：若首批编码结果全为空
        // （整篇被文档侧剪枝掉），循环内的写入事务会被跳过，旧切片残留、
        // doc_count 不扣减，导致新旧编码共存。取消检查放在清理前，避免取消时删掉旧数据。
        if shouldCancel?() == true { throw CancellationError() }
        try db.transaction {
            let removed = try Self.deleteChunks(db, sourceID)
            let previous = Int(Self.meta(db, "doc_count") ?? "") ?? 0
            Self.setMeta(db, "doc_count", String(max(0, previous - removed)))
        }
        for batchStart in stride(from: 0, to: chunks.count, by: encodeBatch) {
            let batchEnd = min(batchStart + encodeBatch, chunks.count)
            // —— 事务外：本批编码 + 量化（CPU/GPU 密集，不持任何锁）——
            var prepared: [(index: Int, record: ResidualRecord)] = []
            for i in batchStart..<batchEnd {
                if shouldCancel?() == true { throw CancellationError() }
                var stage = Date()
                // 整段编码：一次前向得到切片内每个内容 token 的向量。
                // 旧实现逐词编码，每个词都要重复一次完整 24 层前向（一本书数小时），
                // 且丢失词序；整段编码与标准 ColBERT 一致，入库降到秒级。
                let vectors = try colbert.encodeDocument(chunks[i].content)
                encodeTotal += Date().timeIntervalSince(stage)
                guard !vectors.isEmpty else { continue }
                termTotal += vectors.count
                tokenTotal += vectors.count

                stage = Date()
                let id = String(format: "%@#%04d", sourceID, i)
                let record = try ResidualCodec.quantize(docID: id, vectors: vectors,
                                                        dim: ColBERT.dimension, bits: bits)
                quantTotal += Date().timeIntervalSince(stage)
                prepared.append((i, record))
            }
            guard !prepared.isEmpty else {
                for i in batchStart..<batchEnd { progress?(i + 1, chunks.count) }
                continue
            }
            // —— 短事务：批量写入（毫秒级；同源旧切片已在循环外清理）——
            let writeStart = Date()
            try db.transaction {
                for (i, record) in prepared {
                    let id = String(format: "%@#%04d", sourceID, i)
                    rowSeq += 1
                    try db.execute("INSERT INTO document(rowid,doc_id,title,source,content) VALUES(?,?,?,?,?)",
                                   [.int(rowSeq), .text(id), .text(chunks[i].title),
                                    .text(docTitle), .text(chunks[i].content)])
                    try db.execute("INSERT INTO document_fts(doc_id,title_tokens,content_tokens) VALUES(?,?,?)",
                                   [.text(id), .text(TextAnalysis.ftsIndexText(chunks[i].title)),
                                    .text(TextAnalysis.ftsIndexText(chunks[i].content))])
                    try db.execute("INSERT INTO doc_summary_vec(rowid,embedding) VALUES(?,?)",
                                   [.int(rowSeq), .floats(record.center)])
                    try db.execute("INSERT INTO doc_vectors(rowid,embedding) VALUES(?,?)",
                                   [.int(rowSeq), .floats(record.center)])
                    try db.execute("""
                        INSERT INTO doc_residuals(doc_id,token_count,dim,quant_bits,scale,offset,residual_blob)
                        VALUES(?,?,?,?,?,?,?)
                        """, [.text(id), .int(Int64(record.tokenCount)), .int(Int64(record.dim)), .int(Int64(record.bits)),
                              .double(Double(record.scale)), .double(Double(record.offset)), .blob(record.blob)])
                    added += 1
                }
                // 旧切片的计数扣减已在循环外的清理事务完成，这里只累加本批写入数。
                let previous = Int(Self.meta(db, "doc_count") ?? "") ?? 0
                Self.setMeta(db, "doc_count", String(previous + prepared.count))
                Self.setMeta(db, "doc_rowid_seq", String(rowSeq))
            }
            dbTotal += Date().timeIntervalSince(writeStart)
            // 进度回调放锁外，避免外部回调拖长写锁窗口。
            for (i, _) in prepared { progress?(i + 1, chunks.count) }
        }
        let total = Date().timeIntervalSince(addStarted)
        timing?("encode_total", encodeTotal)
        timing?("quantize_total", quantTotal)
        timing?("db_write_total", dbTotal)
        timing?("add_total", total)
        timing?("terms", Double(termTotal))
        timing?("tokens", Double(tokenTotal))
        timing?("chunks", Double(added))
        // 返回实际写入数而非切片总数：文档侧剪枝会跳过不产生向量的切片，
        // 上层 manifest 登记实际数，避免下次同步误判「切片数不完整」而反复重做。
        return added
    }

    // MARK: - 异步 ColBERT 编码回填（正文先入库，向量后台补齐）

    /// 待回填编码的切片数。
    public func pendingEncodeCount() -> Int {
        Int((try? db.query("SELECT COUNT(*) AS n FROM pending_colbert")
            .first?["n"]?.intValue) ?? 0)
    }

    /// 只写正文 + FTS + pending 队列，不编码/写向量。供同步时「文本入库即完成」。
    /// 静态方法：不依赖 ColBERT 模型/实例，模型未下载时也能入库。
    /// rowid 序号在写事务内读取并递增（SQLite 写事务互斥），避免并发 worker 冲突。
    @discardableResult
    static func addMarkdownTextsOnly(
        _ db: SQLiteDB, text: String, source: String, title: String? = nil,
        bits: Int = 4, sourceKey: String? = nil, options: ChunkOptions = ChunkOptions(),
        similarity: (([String]) -> [Double]?)? = nil,
        shouldCancel: (() -> Bool)? = nil
    ) throws -> Int {
        var base = (source as NSString).lastPathComponent
        if !base.isEmpty { base = (base as NSString).deletingPathExtension }
        if base.isEmpty { base = "doc" }
        let sourceID = sourceKey ?? (base + "-" + hash8(text))
        let docTitle = title ?? base
        let chunks = try Chunker.split(text, sourceName: docTitle,
                                       similarity: similarity, options: options)
        guard !chunks.isEmpty else { throw DocSearchError.chunking("没有产生任何切片") }
        if shouldCancel?() == true { throw CancellationError() }
        var added = 0
        try db.transaction {
            let removed = try Self.deleteChunks(db, sourceID)
            let previous = Int(Self.meta(db, "doc_count") ?? "") ?? 0
            Self.setMeta(db, "doc_count", String(max(0, previous - removed)))
            // rowid 序号必须在事务内读取：并发 worker 的写事务互斥，事务内读到的
            // 是已提交的最新值，避免多文件同时入库分配相同 rowid。
            var rowSeq = Int64(Self.meta(db, "doc_rowid_seq") ?? "") ?? 0
            for (i, chunk) in chunks.enumerated() {
                let id = String(format: "%@#%04d", sourceID, i)
                rowSeq += 1
                // source 列统一存原始来源路径（供删除/计数按 source 兜底），
                // 不能存 docTitle（相对文件名）——否则与上层传入的原始 source 不匹配，
                // 按 doc_id 前缀（哈希）删不到时会连兜底也失效，孤儿永久残留。
                try db.execute("INSERT INTO document(rowid,doc_id,title,source,content) VALUES(?,?,?,?,?)",
                               [.int(rowSeq), .text(id), .text(chunk.title),
                                .text(source), .text(chunk.content)])
                try db.execute("INSERT INTO document_fts(doc_id,title_tokens,content_tokens) VALUES(?,?,?)",
                               [.text(id), .text(TextAnalysis.ftsIndexText(chunk.title)),
                                .text(TextAnalysis.ftsIndexText(chunk.content))])
                try db.execute("""
                    INSERT OR REPLACE INTO pending_colbert(doc_id,rowid,title,source,content,bits)
                    VALUES(?,?,?,?,?,?)
                    """, [.text(id), .int(rowSeq), .text(chunk.title),
                          .text(source), .text(chunk.content), .int(Int64(bits))])
                added += 1
            }
            let previousAfterDelete = Int(Self.meta(db, "doc_count") ?? "") ?? 0
            Self.setMeta(db, "doc_count", String(previousAfterDelete + added))
            Self.setMeta(db, "doc_rowid_seq", String(rowSeq))
        }
        return added
    }

    /// 取一批待回填切片的正文（供后台编码）。
    func nextPendingEncode(limit: Int) -> [(docID: String, rowid: Int64, title: String,
                                            source: String, content: String, bits: Int)] {
        let rows = (try? db.query("""
            SELECT doc_id, rowid, title, source, content, bits
            FROM pending_colbert
            WHERE created_at <= ?
            ORDER BY created_at, rowid LIMIT ?
            """, [.double(Date().timeIntervalSince1970),
                  .int(Int64(max(1, limit)))])) ?? []
        return rows.compactMap { row in
            guard let id = row["doc_id"]?.stringValue,
                  let rid = row["rowid"]?.intValue,
                  let content = row["content"]?.stringValue else { return nil }
            return (id, rid, row["title"]?.stringValue ?? "",
                    row["source"]?.stringValue ?? "", content,
                    Int(row["bits"]?.intValue ?? 4))
        }
    }

    /// 编码失败后退避重试：把 pending 的 created_at 推后，按时间排序时排到队尾。
    /// 既不永久丢弃（临时模型/DB 错误可恢复），也不让坏记录每批占队首。
    func retryPending(docID: String, backoffSeconds: Double = 2) throws {
        let retryAt = Date().timeIntervalSince1970 + backoffSeconds
        try db.execute("UPDATE pending_colbert SET created_at = ? WHERE doc_id = ?",
                       [.double(retryAt), .text(docID)])
    }

    /// 整段被剪枝为空（无有效向量）时移除该幽灵切片：正文/FTS/pending/d_count
    /// 一起清掉——它不产生向量，留在库里只会计入切片数却永远无法被检索命中。
    func dropPendingGhost(docID: String, rowid: Int64) throws {
        try db.transaction {
            try db.execute("DELETE FROM document WHERE doc_id = ?", [.text(docID)])
            try db.execute("DELETE FROM document_fts WHERE doc_id = ?", [.text(docID)])
            try db.execute("DELETE FROM doc_residuals WHERE doc_id = ?", [.text(docID)])
            try db.execute("DELETE FROM doc_summary_vec WHERE rowid = ?", [.int(rowid)])
            try db.execute("DELETE FROM doc_vectors WHERE rowid = ?", [.int(rowid)])
            try db.execute("DELETE FROM pending_colbert WHERE doc_id = ?", [.text(docID)])
            let previous = Int(Self.meta(db, "doc_count") ?? "") ?? 0
            Self.setMeta(db, "doc_count", String(max(0, previous - 1)))
        }
    }

    /// 写入一条编码结果（三张向量表）并从 pending 出队。幂等：先删同 rowid/doc_id 旧行。
    func fillPendingEncode(docID: String, rowid: Int64, record: ResidualRecord) throws {
        try db.transaction {
            try db.execute("DELETE FROM doc_summary_vec WHERE rowid = ?", [.int(rowid)])
            try db.execute("DELETE FROM doc_vectors WHERE rowid = ?", [.int(rowid)])
            try db.execute("INSERT INTO doc_summary_vec(rowid,embedding) VALUES(?,?)",
                           [.int(rowid), .floats(record.center)])
            try db.execute("INSERT INTO doc_vectors(rowid,embedding) VALUES(?,?)",
                           [.int(rowid), .floats(record.center)])
            try db.execute("""
                INSERT OR REPLACE INTO doc_residuals(doc_id,token_count,dim,quant_bits,scale,offset,residual_blob)
                VALUES(?,?,?,?,?,?,?)
                """, [.text(docID), .int(Int64(record.tokenCount)), .int(Int64(record.dim)),
                      .int(Int64(record.bits)), .double(Double(record.scale)),
                      .double(Double(record.offset)), .blob(record.blob)])
            try db.execute("DELETE FROM pending_colbert WHERE doc_id = ?", [.text(docID)])
        }
    }

    // MARK: - 按来源删除 / 计数

    /// 按稳定前缀删除某来源的全部切片（不开事务，调用方负责包裹）。
    @discardableResult
    private static func deleteChunks(_ db: SQLiteDB, _ sourceKey: String) throws -> Int {
        let rows = try db.query("SELECT doc_id, rowid FROM document WHERE doc_id LIKE ?",
                                [.text(sourceKey + "#%")])
        return try deleteRows(db, rows)
    }

    /// 按 document.source 列匹配删除（doc_search 旧库 doc_id 前缀不兼容时的兜底）。
    private static func deleteChunksBySourceColumn(_ db: SQLiteDB, _ source: String) throws -> Int {
        let rows = try db.query("SELECT doc_id, rowid FROM document WHERE source = ?",
                                [.text(source)])
        return try deleteRows(db, rows)
    }

    /// 删除选中行的全部关联数据（向量/FTS/残差/document），并登记 ANN 陈旧计数。
    private static func deleteRows(_ db: SQLiteDB, _ rows: [[String: SQLValue]]) throws -> Int {
        let ann = ANNIndex(db: db)
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
            try db.execute("DELETE FROM pending_colbert WHERE doc_id = ?", [.text(id)])
            removed += 1
        }
        try ann.recordStale(rowIDs: staleIDs)
        return removed
    }

    /// 删除某来源的全部切片并同步 doc_count（事务包裹）。
    @discardableResult
    static func deleteSource(_ db: SQLiteDB, _ sourceKey: String,
                             fallbackSource: String? = nil) throws -> Int {
        try db.transaction {
            var removed = try deleteChunks(db, sourceKey)
            // 回退：doc_search 独立工具建的旧库 doc_id 前缀是「文件名-fnv8」而非
            // SHA256 hex，按前缀匹配不到；document.source 列记录入库时的真实
            // 来源路径，用它兜底删除，避免旧数据永远无法按来源清理。
            if removed == 0, let fallbackSource, !fallbackSource.isEmpty {
                removed = try deleteChunksBySourceColumn(db, fallbackSource)
            }
            let previous = Int(meta(db, "doc_count") ?? "") ?? 0
            setMeta(db, "doc_count", String(max(0, previous - removed)))
            return removed
        }
    }

    /// 某来源的切片数（不依赖模型加载）。
    static func chunkCount(_ db: SQLiteDB, _ sourceKey: String,
                           fallbackSource: String? = nil) throws -> Int {
        // 不吞错：调用方（全量重建闸门）必须区分「真的 0 条」与「库打不开/查询失败」，
        // 否则瞬时 IO 错误会把增量同步误升级为整库重建。
        let rows = try db.query("SELECT COUNT(*) AS n FROM document WHERE doc_id LIKE ?",
                                [.text(sourceKey + "#%")])
        let count = Int(rows.first?["n"]?.intValue ?? 0)
        if count > 0 { return count }
        // 回退：doc_search 旧库按 source 列计数，避免有效来源被判 0 条而重复入库。
        if let fallbackSource, !fallbackSource.isEmpty {
            let alt = try db.query("SELECT COUNT(*) AS n FROM document WHERE source = ?",
                                   [.text(fallbackSource)])
            return Int(alt.first?["n"]?.intValue ?? 0)
        }
        return count
    }

    /// 全库切片总数（不依赖模型加载）。
    static func totalChunkCount(_ db: SQLiteDB) throws -> Int {
        let rows = try db.query("SELECT COUNT(*) AS n FROM document")
        return Int(rows.first?["n"]?.intValue ?? 0)
    }

    /// 全库切片文本（供知识图谱逐片抽取）。doc_id 作稳定 id、
    /// title 作章节上下文、source 作来源；只取正文，不加载模型。
    static func listAllChunkTexts(_ db: SQLiteDB) throws -> [(id: String, text: String, source: String, heading: String)] {
        let rows = try db.query("SELECT doc_id, content, source, title FROM document ORDER BY rowid")
        return rows.compactMap { row in
            guard let id = row["doc_id"]?.stringValue,
                  let text = row["content"]?.stringValue, !text.isEmpty else { return nil }
            return (id: id, text: text,
                    source: row["source"]?.stringValue ?? "",
                    heading: row["title"]?.stringValue ?? "")
        }
    }

    /// 多库联合检索-极速模式粗筛：ANN/vec0 + BM25 的 Weighted RRF，返回候选 doc_id
    /// 及其 rrf 分（不做残差加载/MaxSim）。供跨库统一融合。
    func coarseCandidates(query: String, coarse: Int = 20,
                          rrfVectorWeight: Double = 0.7,
                          source: String? = nil)
        throws -> [(docID: String, rrf: Double, content: String,
                    source: String, title: String)] {
        guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }
        let center = try colbert.queryVectors(query, prune: true).center
        let coarseCount = max(1, coarse)
        let fetchK = Self.coarseFetchK(coarse)
        var candidates: [String: Candidate] = [:]
        var coarseHits: [(rowID: Int64, score: Double, distance: Double?)] = []
        let ann = self.ann
        if ann.status().usable,
           let found = ann.candidates(query: center, topK: fetchK,
                                      nprobe: ANNIndex.defaultNprobe,
                                      rescore: ANNIndex.defaultRescore,
                                      knownStatus: ann.status()), !found.isEmpty {
            coarseHits = found.map { ($0.rowID, $0.score, nil) }
        }
        if coarseHits.isEmpty {
            let rows = try db.query("""
                SELECT rowid, distance FROM doc_summary_vec
                WHERE embedding MATCH ? AND k = ? ORDER BY distance
                """, [.floats(center), .int(Int64(fetchK))])
            for row in rows {
                guard let rowID = row["rowid"]?.intValue else { continue }
                coarseHits.append((rowID, 1.0 / (1.0 + (row["distance"]?.doubleValue ?? 2.0)),
                                   row["distance"]?.doubleValue ?? nil))
            }
        }
        let idByRow = try docIDsByRowID(coarseHits.map(\.rowID))
        let wVec = min(max(rrfVectorWeight, 0), 1), wFTS = 1.0 - wVec
        for (rank, item) in coarseHits.enumerated() {
            guard let id = idByRow[item.rowID] else { continue }
            var c = candidates[id] ?? Candidate(docID: id)
            c.vectorRank = rank + 1; c.vectorDistance = item.distance
            c.rrf += wVec / (rrfK + Double(rank + 1))
            candidates[id] = c
        }
        for mode in ["AND", "OR"] {
            let match = TextAnalysis.ftsQuery(query, mode: mode)
            guard !match.isEmpty else { continue }
            let rows = try db.query("""
                SELECT doc_id, bm25(document_fts) AS score FROM document_fts
                WHERE document_fts MATCH ? ORDER BY score LIMIT ?
                """, [.text(match), .int(Int64(fetchK))])
            if !rows.isEmpty {
                for (rank, row) in rows.enumerated() {
                    guard let id = row["doc_id"]?.stringValue else { continue }
                    var c = candidates[id] ?? Candidate(docID: id)
                    c.bm25Rank = rank + 1; c.bm25Score = row["score"]?.doubleValue ?? nil
                    c.rrf += wFTS / (rrfK + Double(rank + 1))
                    candidates[id] = c
                }
                break
            }
        }
        var shortlist = Array(candidates.values.sorted {
            $0.rrf != $1.rrf ? $0.rrf > $1.rrf : $0.docID < $1.docID
        }.prefix(coarseCount))
        if let source, !source.isEmpty {
            let totalCap = Self.sourceCandidateLimit(coarse)
            var seen = Set(shortlist.map(\.docID))
            if totalCap > shortlist.count {
                let escaped = Self.likePattern(source)
                let rows = try db.query("""
                    SELECT doc_id FROM document
                    WHERE source LIKE ? ESCAPE '\\' OR title LIKE ? ESCAPE '\\'
                    ORDER BY doc_id LIMIT ?
                    """, [.text(escaped), .text(escaped), .int(Int64(totalCap - shortlist.count))])
                for row in rows {
                    guard let id = row["doc_id"]?.stringValue, seen.insert(id).inserted else { continue }
                    shortlist.append(Candidate(docID: id))
                    if shortlist.count >= totalCap { break }
                }
            }
        }
        return shortlist.compactMap { c in
            guard let meta = try? loadMeta(c.docID) else { return nil }
            return (c.docID, c.rrf, meta.content, meta.source, meta.title)
        }
    }

    /// 多库联合检索-极速模式精排：对给定 doc_id 列表做残差加载 + MaxSim。
    func rerankByDocIDs(query: String, docIDs: [String],
                        lengthNorm: Bool = false) throws -> [SearchHit] {
        guard !docIDs.isEmpty else { return [] }
        let queryTokens = try colbert.queryVectors(query, prune: true).tokens
        let data = try loadShortlistData(docIDs)
        var hits: [SearchHit] = []
        for docID in docIDs {
            guard let meta = data.metas[docID] else { continue }
            guard let record = data.residuals[docID],
                  let score = try? ResidualCodec.maxSim(query: queryTokens, record: record,
                                                        normalizeLength: lengthNorm) else {
                // 未回填/残差损坏：BM25 兜底分（排位靠后）。
                hits.append(SearchHit(docID: docID, title: meta.title,
                                      source: meta.source, content: meta.content,
                                      score: 0, vectorRank: nil, bm25Rank: nil, rrf: 0))
                continue
            }
            hits.append(SearchHit(docID: docID, title: meta.title,
                                  source: meta.source, content: meta.content,
                                  score: score, vectorRank: nil, bm25Rank: nil, rrf: 0))
        }
        return hits.sorted { $0.score > $1.score }
    }

    /// 轻量正文命中探查（不加载模型）：对 document_fts 做 AND/OR MATCH，
    /// 返回命中数（上限 cap）。供多库联合检索前判断「该库是否与问句相关」。
    static func ftsHitCount(_ db: SQLiteDB, query: String, cap: Int = 8) -> Int {
        guard cap > 0 else { return 0 }
        for mode in ["AND", "OR"] {
            let match = TextAnalysis.ftsQuery(query, mode: mode)
            guard !match.isEmpty else { continue }
            let rows = (try? db.query("""
                SELECT doc_id FROM document_fts WHERE document_fts MATCH ? LIMIT ?
                """, [.text(match), .int(Int64(cap))])) ?? []
            if !rows.isEmpty { return rows.count }
        }
        return 0
    }

    /// 已编码（有残差向量）的切片数（不依赖模型加载）。
    static func encodedChunkCount(_ db: SQLiteDB) throws -> Int {
        let rows = try db.query("SELECT COUNT(*) AS n FROM doc_residuals")
        return Int(rows.first?["n"]?.intValue ?? 0)
    }

    /// 待编码队列长度（不依赖模型加载）。
    static func pendingEncodeCount(_ db: SQLiteDB) -> Int {
        Int((try? db.query("SELECT COUNT(*) AS n FROM pending_colbert")
            .first?["n"]?.intValue) ?? 0)
    }

    // MARK: - 检索

    func search(query: String, topN: Int = 5, coarse: Int = 50,
                useANN: Bool = true, lengthNorm: Bool = false, queryPrune: Bool = true,
                rrfVectorWeight: Double = 0.7,
                nprobe: Int = ANNIndex.defaultNprobe,
                rescore: Int = ANNIndex.defaultRescore, forceANN: Bool = false,
                source: String? = nil,
                timing: ((String, Double) -> Void)? = nil) throws -> [SearchHit] {
        guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }
        let searchStarted = Date()
        var stage = Date()
        // 一次前向同时拿到簇心（粗筛）与内容 token 向量（MaxSim），
        // 不再为两次用途重复编码查询。
        let encodedQuery = try colbert.queryVectors(query, prune: queryPrune)
        timing?("query_center", Date().timeIntervalSince(stage))
        let center = encodedQuery.center
        // 向量 / BM25 各取 coarse 的 3 倍做 RRF 融合（过召回冗余，融合后再
        // 截到 coarse 进精排）。严格按面板档位派生，无隐藏下限。
        let coarseCount = max(1, coarse)
        let fetchK = Self.coarseFetchK(coarse)
        var candidates: [String: Candidate] = [:]

        stage = Date()
        // 粗筛结果先按 rowID 收集，最后一次性 IN 查询取回 doc_id，避免逐候选点查。
        var coarseHits: [(rowID: Int64, score: Double, distance: Double?)] = []
        let ann = self.ann
        let annStatus = ann.status()
        let totalRows = Int(Self.meta(db, "doc_count") ?? "") ?? 0
        if useANN, (forceANN || totalRows >= ANNIndex.minRowsForANN), annStatus.usable,
           let found = ann.candidates(query: center, topK: fetchK,
                                      nprobe: nprobe, rescore: rescore,
                                      knownStatus: annStatus), !found.isEmpty {
            lastVectorBackend = "ANN(nprobe=\(nprobe),rescore=\(rescore))"
            coarseHits = found.map { ($0.rowID, $0.score, nil) }
        }
        if coarseHits.isEmpty {
            lastVectorBackend = "vec0-exact"
            let rows = try db.query("""
                SELECT rowid, distance FROM doc_summary_vec
                WHERE embedding MATCH ? AND k = ? ORDER BY distance
                """, [.floats(center), .int(Int64(fetchK))])
            for row in rows {
                guard let rowID = row["rowid"]?.intValue else { continue }
                coarseHits.append((rowID, 1.0 / (1.0 + (row["distance"]?.doubleValue ?? 2.0)),
                                   row["distance"]?.doubleValue ?? nil))
            }
        }
        let idByRow = try docIDsByRowID(coarseHits.map(\.rowID))
        let pairs = coarseHits.compactMap { item -> (id: String, score: Double, distance: Double?)? in
            guard let id = idByRow[item.rowID] else { return nil }
            return (id, item.score, item.distance)
        }
        timing?("vector_coarse", Date().timeIntervalSince(stage))
        stage = Date()
        // Weighted RRF：向量路 w_vec、BM25 路 (1 - w_vec)，权重和恒为 1。
        let wVec = min(max(rrfVectorWeight, 0), 1)
        let wFTS = 1.0 - wVec
        for (rank, pair) in pairs.enumerated() {
            var c = candidates[pair.id] ?? Candidate(docID: pair.id)
            c.vectorRank = rank + 1
            c.vectorDistance = pair.distance
            c.rrf += wVec / (rrfK + Double(rank + 1))
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
            c.rrf += wFTS / (rrfK + Double(rank + 1))
            candidates[id] = c
        }

        timing?("bm25_coarse", Date().timeIntervalSince(stage))
        stage = Date()
        var shortlist = Array(candidates.values.sorted {
            if $0.rrf != $1.rrf { return $0.rrf > $1.rrf }
            return $0.docID < $1.docID
        }.prefix(coarseCount))
        if let source, !source.isEmpty {
            // 指定来源时，RRF 粗筛的 top-N 可能根本不包含该来源：把该来源的
            // 切片直接并入候选，避免「明明有却检索不到」。
            // sourceLimit 是「注入后 shortlist 的总上限」而非额外增量：已在
            // shortlist 里的同来源候选先占配额，避免精排规模膨胀到 coarse+limit。
            // 按 doc_id 排序保证注入确定性（该批无 rrf，最终由 MaxSim 精排打分）。
            let totalCap = Self.sourceCandidateLimit(coarse)
            var seen = Set(shortlist.map(\.docID))
            // 已有 shortlist 无论是否来自该来源都占配额：总数硬上限 = totalCap。
            let injectBudget = max(0, totalCap - shortlist.count)
            if injectBudget > 0 {
                // LIKE 中 % / _ / \ 是通配符/转义符：来自工具参数的 source 必须转义，
                // 否则含这些字符的来源会放大注入候选集（后端还会再按 contains 过滤，
                // 结果虽不会错，但会无谓拉大批次做 MaxSim）。
                let escaped = Self.likePattern(source)
                let rows = try db.query("""
                    SELECT doc_id FROM document
                    WHERE source LIKE ? ESCAPE '\\' OR title LIKE ? ESCAPE '\\'
                    ORDER BY doc_id LIMIT ?
                    """, [.text(escaped), .text(escaped), .int(Int64(injectBudget))])
                for row in rows {
                    guard let id = row["doc_id"]?.stringValue, seen.insert(id).inserted else { continue }
                    shortlist.append(Candidate(docID: id))
                    if shortlist.count >= totalCap { break }
                }
            }
        }
        timing?("rrf", Date().timeIntervalSince(stage))
        timing?("coarse_total", Date().timeIntervalSince(searchStarted))

        stage = Date()
        let queryTokens = encodedQuery.tokens
        guard !queryTokens.isEmpty else { return [] }

        var loadTotal = 0.0, maxsimTotal = 0.0
        var hits: [SearchHit] = []
        // shortlist 批量加载：固定 3 次查询取齐全部候选的 meta/残差/中心向量，
        // 替代逐候选点查（coarse=100 时 N+1 点查 300+ 次）。
        // 批量查询失败（瞬时 busy/IO）时降级为逐候选 try? 加载：宁可召回降级，
        // 也不把单条候选的读取失败放大成整次检索报错。
        var itemStart = Date()
        let shortlistData: (metas: [String: ShortlistMeta],
                            residuals: [String: ResidualRecord])
        do {
            shortlistData = try loadShortlistData(shortlist.map(\.docID))
        } catch {
            // 纯库无日志设施；降级为逐候选读取（单候选失败只丢那条）。
            // 逐候选降级填充（等价旧实现语义：单候选失败只丢那一条）。
            var metas: [String: ShortlistMeta] = [:]
            var residuals: [String: ResidualRecord] = [:]
            for candidate in shortlist {
                if let record = loadResidual(candidate.docID),
                   let meta = try? loadMeta(candidate.docID) {
                    metas[candidate.docID] = ShortlistMeta(
                        rowID: 0, title: meta.title,
                        source: meta.source, content: meta.content)
                    residuals[candidate.docID] = record
                }
            }
            shortlistData = (metas, residuals)
        }
        loadTotal += Date().timeIntervalSince(itemStart)
        for candidate in shortlist {
            itemStart = Date()
            guard let meta = shortlistData.metas[candidate.docID] else { continue }
            // 来源过滤：匹配文件名/相对路径或章节标题，向后兼容未单独建索引的旧库。
            if let source, !source.isEmpty,
               !meta.source.localizedCaseInsensitiveContains(source),
               !meta.title.localizedCaseInsensitiveContains(source) {
                continue
            }
            loadTotal += Date().timeIntervalSince(itemStart)
            guard let record = shortlistData.residuals[candidate.docID] else {
                // 正文已入库但 ColBERT 编码尚未回填：用 BM25 命中做一个兜底分数，
                // 让未回填切片仍能被关键词检索到（排在已回填的 MaxSim 结果之后）。
                // 无 BM25 命中（仅来自向量粗筛却缺残差）则跳过。
                if candidate.bm25Rank != nil {
                    let fallbackScore = Float(1.0 / (60.0 + Double(candidate.bm25Rank ?? 999)))
                    hits.append(SearchHit(docID: candidate.docID, title: meta.title,
                                          source: meta.source, content: meta.content,
                                          score: fallbackScore, vectorRank: nil,
                                          bm25Rank: candidate.bm25Rank, rrf: candidate.rrf))
                }
                continue
            }
            itemStart = Date()
            // 单条残差损坏（blob 截断/维度不符）只丢弃该候选，不让整次检索失败。
            guard let score = try? ResidualCodec.maxSim(query: queryTokens, record: record,
                                                        normalizeLength: lengthNorm) else {
                continue
            }
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

    /// 批量把 rowid 映射到 doc_id（粗筛阶段 ANN/vec0 都返回 rowid，避免逐候选点查）。
    private func docIDsByRowID(_ rowIDs: [Int64]) throws -> [Int64: String] {
        guard !rowIDs.isEmpty else { return [:] }
        let placeholders = rowIDs.map { _ in "?" }.joined(separator: ",")
        let rows = try db.query("""
            SELECT rowid, doc_id FROM document WHERE rowid IN (\(placeholders))
            """, rowIDs.map { .int($0) })
        var map: [Int64: String] = [:]
        for row in rows {
            if let rid = row["rowid"]?.intValue, let id = row["doc_id"]?.stringValue {
                map[rid] = id
            }
        }
        return map
    }

    /// shortlist 候选的文档元信息（命名元组作字典值会让类型推断卡死，用 struct）。
    private struct ShortlistMeta {
        let rowID: Int64
        let title: String
        let source: String
        let content: String
    }

    /// shortlist 批量加载：固定 3 次 IN 查询取齐全部候选的 meta/rowid、残差、
    /// 中心向量，替代逐候选点查。缺失候选（残差/中心不全）不进结果。
    private func loadShortlistData(_ docIDs: [String]) throws -> (
        metas: [String: ShortlistMeta],
        residuals: [String: ResidualRecord]
    ) {
        guard !docIDs.isEmpty else { return ([:], [:]) }
        let placeholders = docIDs.map { _ in "?" }.joined(separator: ",")

        var metas: [String: ShortlistMeta] = [:]
        var rowIDByDoc: [String: Int64] = [:]
        let metaRows = try db.query("""
            SELECT doc_id, rowid, title, source, content FROM document
            WHERE doc_id IN (\(placeholders))
            """, docIDs.map { .text($0) })
        var rowIDs: [Int64] = []
        for row in metaRows {
            guard let id = row["doc_id"]?.stringValue,
                  let rowID = row["rowid"]?.intValue else { continue }
            metas[id] = ShortlistMeta(rowID: rowID, title: row["title"]?.stringValue ?? "",
                                      source: row["source"]?.stringValue ?? "",
                                      content: row["content"]?.stringValue ?? "")
            rowIDs.append(rowID)
            rowIDByDoc[id] = rowID
        }

        var centers: [Int64: Data] = [:]
        if !rowIDs.isEmpty {
            let rowPlaceholders = rowIDs.map { _ in "?" }.joined(separator: ",")
            let vecRows = try db.query("""
                SELECT rowid, embedding FROM doc_vectors WHERE rowid IN (\(rowPlaceholders))
                """, rowIDs.map { .int($0) })
            for row in vecRows {
                if let rowID = row["rowid"]?.intValue,
                   let data = row["embedding"]?.dataValue {
                    centers[rowID] = data
                }
            }
        }

        let residualRows = try db.query("""
            SELECT doc_id, token_count, dim, quant_bits, scale, offset, residual_blob
            FROM doc_residuals WHERE doc_id IN (\(placeholders))
            """, docIDs.map { .text($0) })
        var residuals: [String: ResidualRecord] = [:]
        for row in residualRows {
            guard let id = row["doc_id"]?.stringValue else { continue }
            guard let count = row["token_count"]?.intValue,
                  let dim = row["dim"]?.intValue,
                  let bits = row["quant_bits"]?.intValue,
                  let scale = row["scale"]?.doubleValue,
                  let offset = row["offset"]?.doubleValue,
                  let blob: Data = row["residual_blob"]?.dataValue else { continue }
            guard let rowID: Int64 = rowIDByDoc[id],
                  let centerData: Data = centers[rowID] else { continue }
            let expectedBytes = Int(dim) * MemoryLayout<Float>.stride
            guard centerData.count == expectedBytes else { continue }
            let centerFloats: [Float] = centerData.withUnsafeBytes {
                Array($0.bindMemory(to: Float.self))
            }
            let center: [Float16] = centerFloats.map { Float16($0) }
            guard center.count == Int(dim) else { continue }
            let record = ResidualRecord(docID: id, tokenCount: Int(count), dim: Int(dim),
                                        bits: Int(bits), scale: Float(scale),
                                        offset: Float(offset), centerFP16: center, blob: blob)
            residuals[id] = record
        }
        return (metas, residuals)
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

    /// 粗筛池大小：向量/BM25 各取 coarse 的 3 倍供 RRF 融合。纯函数，便于回归
    /// 「档位严格派生、无隐藏下限」的契约（配置面板 10/20/50/100 均成比例）。
    static func coarseFetchK(_ coarse: Int) -> Int { max(1, coarse) * 3 }

    /// 指定来源时并入候选的上限：随 coarse 放大，不低于 200 兜底，限制精排开销。
    static func sourceCandidateLimit(_ coarse: Int) -> Int { max(200, max(1, coarse) * 4) }

    /// 把字面量转成 LIKE 匹配串：转义 \ / % / _ 并包裹通配符，配合 `ESCAPE '\'` 使用。
    static func likePattern(_ literal: String) -> String {
        var out = ""
        out.reserveCapacity(literal.count + 2)
        for ch in literal {
            if ch == "\\" || ch == "%" || ch == "_" { out.append("\\") }
            out.append(ch)
        }
        return "%\(out)%"
    }

    private func blob16(_ values: [Float16]) -> Data {
        values.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private func decode16(_ data: Data) -> [Float16] {
        guard data.count % 2 == 0 else { return [] }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
    }

    private static func hash8(_ text: String) -> String {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x100000001b3
        }
        return String(format: "%08llx", hash & 0xffff_ffff)
    }
}
