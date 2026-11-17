import Foundation
import OnnxRuntimeBindings
import Tokenizers

/// ColBERT 词向量引擎：严格复用模型目录 swift-example 的前缀、query 增强与输出约定。
final class ColBERT {
    enum Kind { case query, document }

    static let queryPrefix = "[QueryMarker]"
    static let documentPrefix = "[DocumentMarker]"
    static let queryMarkerID: Int = 250002
    static let documentMarkerID: Int = 250003
    static let maskID: Int = 250001
    static let padID: Int = 1
    static let queryMaxLen = 32
    static let dimension = 64

    private static let envLock = NSLock()
    private static var sharedEnv: ORTEnv?

    private let tokenizer: any Tokenizers.Tokenizer
    private let session: ORTSession
    private let outputName: String

    private init(tokenizer: any Tokenizers.Tokenizer, session: ORTSession, outputName: String) {
        self.tokenizer = tokenizer
        self.session = session
        self.outputName = outputName
    }

    static func load(modelDir: String,
                     threads: Int = min(8, ProcessInfo.processInfo.activeProcessorCount)) async throws -> ColBERT {
        let dir = URL(fileURLWithPath: modelDir, isDirectory: true)
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("tokenizer.json").path) else {
            throw DocSearchError.model("缺少 tokenizer.json: \(modelDir)")
        }
        let model = dir.appendingPathComponent("model_int8.onnx").path
        guard FileManager.default.fileExists(atPath: model) else {
            throw DocSearchError.model("缺少 model_int8.onnx: \(modelDir)")
        }

        let tokenizer = try await Tokenizers.AutoTokenizer.from(modelFolder: dir, strict: false)
        // tokenizer.json 自带 truncation=max_length 300（模型作者的短文训练设置）。
        // 不能整段直接编码，否则 8000 字符的切片只会保留前 300 token，其余正文
        // 完全不参与检索；encodeDocument 按 300 token 切成多段后批量编码。
        let env: ORTEnv
        envLock.lock()
        if sharedEnv == nil { sharedEnv = try ORTEnv(loggingLevel: .warning) }
        let cached = sharedEnv!
        envLock.unlock()
        env = cached

        let options = try ORTSessionOptions()
        if threads > 0 { try options.setIntraOpNumThreads(Int32(threads)) }
        try options.setGraphOptimizationLevel(.all)
        let session = try ORTSession(env: env, modelPath: model, sessionOptions: options)
        guard let output = try session.outputNames().first else {
            throw DocSearchError.model("ONNX 图没有输出")
        }
        return ColBERT(tokenizer: tokenizer, session: session, outputName: output)
    }

    // MARK: - tokenisation

    private func encodeIDs(_ text: String, kind: Kind) -> [Int] {
        let prefix = kind == .query ? Self.queryPrefix : Self.documentPrefix
        return tokenizer.encode(text: prefix + " " + text, addSpecialTokens: true)
    }

    /// 返回每条文本的非特殊 token 位置（跳过 <s>/</s>/前缀标记）。
    private func contentIndices(_ ids: [Int], kind: Kind) -> [Int] {
        let marker = kind == .query ? Self.queryMarkerID : Self.documentMarkerID
        return ids.indices.filter { ids[$0] != 0 && ids[$0] != 2 && ids[$0] != marker }
    }

    // MARK: - ONNX

    private func run(ids: [[Int64]], mask: [[Int64]]) throws -> [[[Float]]] {
        guard let first = ids.first else { return [] }
        let seq = first.count
        func tensor(_ rows: [[Int64]]) throws -> ORTValue {
            var flat: [Int64] = []
            flat.reserveCapacity(rows.count * seq)
            for row in rows { flat.append(contentsOf: row) }
            let data = NSMutableData(capacity: flat.count * MemoryLayout<Int64>.stride)!
            flat.withUnsafeBytes { data.append($0.baseAddress!, length: $0.count) }
            return try ORTValue(tensorData: data,
                                elementType: .int64,
                                shape: [NSNumber(value: rows.count), NSNumber(value: seq)])
        }
        let outputs = try session.run(
            withInputs: ["input_ids": try tensor(ids), "attention_mask": try tensor(mask)],
            outputNames: [outputName],
            runOptions: nil)
        guard let value = outputs[outputName] else { throw DocSearchError.model("推理没有返回 \(outputName)") }
        let info = try value.tensorTypeAndShapeInfo()
        let shape = info.shape.map { $0.intValue }
        guard shape.count == 3, shape[2] == Self.dimension else {
            throw DocSearchError.model("输出形状异常: \(shape)")
        }
        let data = try value.tensorData()
        let count = shape[0] * shape[1] * shape[2]
        guard data.length >= count * MemoryLayout<Float>.stride else {
            throw DocSearchError.model("输出数据长度不足")
        }
        var flat = [Float](repeating: 0, count: count)
        flat.withUnsafeMutableBytes { data.getBytes($0.baseAddress!, length: count * MemoryLayout<Float>.stride) }
        var result = [[[Float]]]()
        result.reserveCapacity(shape[0])
        for b in 0..<shape[0] {
            var seqOut = [[Float]]()
            seqOut.reserveCapacity(shape[1])
            for s in 0..<shape[1] {
                let off = (b * shape[1] + s) * shape[2]
                seqOut.append(Array(flat[off..<(off + shape[2])]))
            }
            result.append(seqOut)
        }
        return result
    }

    /// 批量编码。query 用 <mask> 补齐到至少 32；doc 用 PAD_ID 补齐并屏蔽 padding。
    func encode(_ texts: [String], kind: Kind, batchSize: Int = 32) throws -> [[[Float]]] {
        guard !texts.isEmpty else { return [] }
        let sequences = texts.map { encodeIDs($0, kind: kind) }
        let target: Int
        if kind == .query {
            target = max(Self.queryMaxLen, sequences.map(\.count).max() ?? 0)
        } else {
            target = sequences.map(\.count).max() ?? 0
        }
        var out = [[[Float]]]()
        out.reserveCapacity(texts.count)
        var offset = 0
        while offset < sequences.count {
            let end = min(offset + max(1, batchSize), sequences.count)
            let ids = (offset..<end).map { i -> [Int64] in
                var row = sequences[i].map { Int64($0) }
                let pad = kind == .query ? Int64(Self.maskID) : Int64(Self.padID)
                row.append(contentsOf: Array(repeating: pad, count: target - row.count))
                return row
            }
            let mask = (offset..<end).map { i -> [Int64] in
                Array(repeating: Int64(1), count: sequences[i].count)
                    + Array(repeating: 0, count: target - sequences[i].count)
            }
            let batch = try run(ids: ids, mask: mask)
            for (j, seq) in batch.enumerated() {
                out.append(Array(seq.prefix(sequences[offset + j].count)))
            }
            offset = end
        }
        return out
    }

    /// 对一组词/短语分别编码，取其内容 token 向量均值作为「这个词的向量」。
    func encodeTerms(_ terms: [String], batchSize: Int = 128) throws -> [[Float]] {
        guard !terms.isEmpty else { return [] }
        let seqs = terms.map { encodeIDs($0, kind: .document) }
        let target = seqs.map(\.count).max() ?? 0
        var output = [[Float]]()
        output.reserveCapacity(terms.count)
        var offset = 0
        while offset < seqs.count {
            let end = min(offset + max(1, batchSize), seqs.count)
            let ids = (offset..<end).map { i -> [Int64] in
                seqs[i].map { Int64($0) } + Array(repeating: Int64(Self.padID), count: target - seqs[i].count)
            }
            let mask = (offset..<end).map { i -> [Int64] in
                Array(repeating: Int64(1), count: seqs[i].count)
                    + Array(repeating: 0, count: target - seqs[i].count)
            }
            let batch = try run(ids: ids, mask: mask)
            for (j, seq) in batch.enumerated() {
                let positions = contentIndices(seqs[offset + j], kind: .document)
                let chosen = positions.isEmpty ? Array(0..<seq.count) : positions
                var center = [Float](repeating: 0, count: Self.dimension)
                for p in chosen where p < seq.count {
                    for d in 0..<Self.dimension { center[d] += seq[p][d] }
                }
                let denom = Float(max(1, chosen.count))
                output.append(center.map { $0 / denom })
            }
            offset = end
        }
        return output
    }

    /// 整段编码：把切片正文切成若干 ≤300 token 的小段，批量前向后拼接各段内
    /// 容 token 的向量（标准 ColBERT 的 token 级表示）。
    ///
    /// 两个关键点：
    /// 1. tokenizer.json 写死了 truncation=300，超过 300 token 的部分会被静默丢弃，
    ///    所以必须先按 300 token 分段，否则一本书只有开头几百字参与检索；
    /// 2. 逐词编码（`encodeTerms`）每个词都要重复一次完整 24 层前向，一本书要数
    ///    小时；分段批量编码按 batch=8 分摊权重读取，入库耗时降到分钟级。
    func encodeDocument(_ text: String, maxTokens: Int = 300,
                        batchSize: Int = 8) throws -> [[Float]] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        let segments = Self.segment(trimmed, maxTokens: maxTokens,
                                    encodeIDs: { self.encodeIDs($0, kind: .document) })
        guard !segments.isEmpty else { return [] }
        let encoded = try encode(segments, kind: .document, batchSize: batchSize)
        var vectors: [[Float]] = []
        vectors.reserveCapacity(segments.count * 128)
        for (i, segment) in segments.enumerated() {
            guard i < encoded.count else { break }
            let seq = encoded[i]
            let ids = encodeIDs(segment, kind: .document)
            let positions = contentIndices(ids, kind: .document)
            if positions.isEmpty {
                vectors.append(contentsOf: seq)
            } else {
                for p in positions where p < seq.count { vectors.append(seq[p]) }
            }
        }
        return vectors
    }

    /// 按 token 预算切段：优先在句子边界断开，单句超长时按字符二分收缩。
    private static func segment(_ text: String, maxTokens: Int,
                                encodeIDs: (String) -> [Int]) -> [String] {
        var sentences = TextAnalysis.sentences(text)
        if sentences.isEmpty { sentences = [text] }
        var segments: [String] = []
        var current = ""
        func flush() {
            let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { segments.append(trimmed) }
            current = ""
        }
        for sentence in sentences {
            let candidate = current.isEmpty ? sentence : TextAnalysis.join(current, sentence)
            if encodeIDs(candidate).count <= maxTokens {
                current = candidate
                continue
            }
            flush()
            if encodeIDs(sentence).count <= maxTokens {
                current = sentence
                continue
            }
            // 单句超过预算：按字符二分找到最大可容纳前缀，剩余部分继续处理。
            var rest = Substring(sentence)
            while !rest.isEmpty {
                var lo = 1, hi = rest.count
                while lo < hi {
                    let mid = (lo + hi + 1) / 2
                    if encodeIDs(String(rest.prefix(mid))).count <= maxTokens {
                        lo = mid
                    } else {
                        hi = mid - 1
                    }
                }
                guard lo > 0 else { break }
                segments.append(String(rest.prefix(lo)))
                rest = rest.dropFirst(lo)
            }
        }
        flush()
        return segments
    }

    /// 查询中心向量：取 query 增强后所有「内容 token」（含 <mask>，排除 <s>/</s>/标记）的均值。
    func queryCenter(_ query: String) throws -> [Float] {
        let encoded = try encode([query], kind: .query)
        guard let seq = encoded.first, !seq.isEmpty else {
            throw DocSearchError.model("问句编码为空")
        }
        var center = [Float](repeating: 0, count: Self.dimension)
        let positions = 2..<(seq.count - 1)   // <s>, [QueryMarker] ... </s>
        for p in positions { for d in 0..<Self.dimension { center[d] += seq[p][d] } }
        let n = Float(max(1, positions.count))
        return VectorMath.normalize(center.map { $0 / n })
    }

    /// 相邻文本块的语义相似度（ColBERT 词向量中心余弦）；失败返回 nil 供切片器回退词法相似度。
    func adjacentSimilarities(_ texts: [String]) -> [Double]? {
        guard texts.count >= 2 else { return [] }
        do {
            let centers = try encodeTerms(texts)
            guard centers.count == texts.count else { return nil }
            return (0..<(texts.count - 1)).map {
                VectorMath.cosine(centers[$0], centers[$0 + 1])
            }
        } catch {
            return nil
        }
    }
}

enum VectorMath {
    static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in a.indices { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        let den = sqrt(na) * sqrt(nb)
        return den > 0 ? Double(dot / den) : 0
    }

    static func normalize(_ v: [Float]) -> [Float] {
        var n: Float = 0
        for x in v { n += x * x }
        n = sqrt(n)
        guard n > 0 else { return v }
        return v.map { $0 / n }
    }
}
