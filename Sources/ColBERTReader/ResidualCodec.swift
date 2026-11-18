import Foundation
import Accelerate

/// 4bit（兼容 2bit）残差量化：fp32 → fp16 → 减中心 → 线性量化 → 半字节打包。
struct ResidualRecord {
    let docID: String
    let tokenCount: Int
    let dim: Int
    let bits: Int
    let scale: Float
    let offset: Float
    let centerFP16: [Float16]
    let blob: Data

    var center: [Float] { centerFP16.map(Float.init) }
}

enum ResidualCodec {
    /// 量化：scale = 2*maxAbs/(2^bits-1)，offset = -maxAbs，两个 4bit 值打进一个字节。
    static func quantize(docID: String, vectors: [[Float]], dim: Int = 64,
                         bits: Int = 4) throws -> ResidualRecord {
        guard bits == 2 || bits == 4 else { throw DocSearchError.quant("仅支持 2/4 bit") }
        guard !vectors.isEmpty else { throw DocSearchError.quant("词向量为空") }
        guard vectors.allSatisfy({ $0.count == dim }) else {
            throw DocSearchError.quant("词向量维度不一致，期望 \(dim)")
        }
        let half = vectors.map { v in v.map { Float16($0) } }
        var center = [Float](repeating: 0, count: dim)
        for v in half { for d in 0..<dim { center[d] += Float(v[d]) } }
        let denom = Float(vectors.count)
        let center16 = center.map { Float16($0 / denom) }

        var maxAbs: Float = 0
        for v in half {
            for d in 0..<dim {
                let r = abs(Float(v[d] - center16[d]))
                if r > maxAbs { maxAbs = r }
            }
        }
        let levels = bits == 4 ? Float(15) : Float(3)
        let scale = maxAbs > 1e-12 ? (2 * maxAbs) / levels : 1e-8
        let offset = -maxAbs
        let perByte = 8 / bits
        let mask = (1 << bits) - 1
        var packed = [UInt8]()
        packed.reserveCapacity((vectors.count * dim + perByte - 1) / perByte)

        var accumulator: UInt8 = 0
        var shift = 0
        for v in half {
            for d in 0..<dim {
                let residual = Float(v[d] - center16[d])
                var q = Int((residual - offset) / scale + 0.5)
                q = max(0, min(Int(levels), q))
                accumulator |= UInt8(q & mask) << UInt8(shift)
                shift += bits
                if shift == 8 {
                    packed.append(accumulator)
                    accumulator = 0
                    shift = 0
                }
            }
        }
        if shift > 0 { packed.append(accumulator) }
        return ResidualRecord(docID: docID, tokenCount: vectors.count, dim: dim, bits: bits,
                              scale: scale, offset: offset, centerFP16: center16,
                              blob: Data(packed))
    }

    /// 标准 ColBERT 段落长度（约 300 token）：长度归一化的参考基准。
    /// 本项目的检索阅读分片把若干 300-token 段拼成一条 record（实测 166~5122
    /// token），直接用 sum-of-max 会让长片系统性碾压短片，故以此基准做温和归一化。
    static let referenceTokenCount: Int = 300

    /// 流式解压 + vDSP_mmul 矩阵乘：1 字节解 2 个 4bit 值，逐批算 MaxSim，不展开全部 token。
    ///
    /// `normalizeLength=true` 时对 sum-of-max 做平方根长度归一化：
    /// `sum / max(1, sqrt(tokenCount / 300))`。sqrt 而非线性，保留「内容多命中机会大」
    /// 的合理优势，同时消除 30 倍长度差带来的碾压（5122:166 的倍率从 30 压到 ~5.5）。
    static func maxSim(query: [[Float]], record: ResidualRecord,
                       queryTokens: [[Float]]? = nil, batchTokens: Int = 256,
                       normalizeLength: Bool = false) throws -> Float {
        let q = queryTokens ?? query
        guard !q.isEmpty, record.tokenCount > 0 else { return 0 }
        guard q.allSatisfy({ $0.count == record.dim }) else {
            throw DocSearchError.quant("query 维度与残差维度不一致")
        }
        let dim = record.dim
        let qCount = q.count
        // 损坏/截断记录防御：校验维度、量化位数与 blob 长度后再做裸下标访问，
        // 否则 bytes[(index*dim+d)/perByte] 越界会让整次检索 trap，破坏
        // 「单候选失败只丢一条」的容错语义。
        guard record.bits == 2 || record.bits == 4 else {
            throw DocSearchError.quant("非法量化位数 \(record.bits)")
        }
        guard record.centerFP16.count == dim else {
            throw DocSearchError.quant("残差中心维度不符：\(record.centerFP16.count) != \(dim)")
        }
        var queryFlat = [Float](repeating: 0, count: qCount * dim)
        for i in 0..<qCount {
            queryFlat.replaceSubrange((i * dim)..<((i + 1) * dim), with: q[i])
        }
        let bytes = [UInt8](record.blob)
        let perByte = 8 / record.bits
        let requiredBytes = (record.tokenCount * dim + perByte - 1) / perByte
        guard bytes.count >= requiredBytes else {
            throw DocSearchError.quant("残差 blob 截断：\(bytes.count) < \(requiredBytes)")
        }
        let mask = (1 << record.bits) - 1
        var maxima = [Float](repeating: -Float.greatestFiniteMagnitude, count: qCount)

        var token = 0
        while token < record.tokenCount {
            let count = min(batchTokens, record.tokenCount - token)
            // 行优先的 (dim x count) 矩阵：第 d 行第 j 列 = token[j][d]
            var docT = [Float](repeating: 0, count: dim * count)
            for j in 0..<count {
                let index = token + j
                for d in 0..<dim {
                    let bitPos = (index * dim + d) % perByte
                    let byte = bytes[(index * dim + d) / perByte]
                    let code = Int((byte >> UInt8(bitPos * record.bits)) & UInt8(mask))
                    let dequant = Float16(Float(code) * record.scale + record.offset)
                    let value = Float16(dequant + record.centerFP16[d])
                    docT[d * count + j] = Float(value)
                }
            }
            var scores = [Float](repeating: 0, count: qCount * count)
            vDSP_mmul(queryFlat, 1, docT, 1, &scores, 1,
                      vDSP_Length(qCount), vDSP_Length(count), vDSP_Length(dim))
            for qi in 0..<qCount {
                let base = qi * count
                for j in 0..<count where scores[base + j] > maxima[qi] {
                    maxima[qi] = scores[base + j]
                }
            }
            token += count
        }
        let total = maxima.reduce(0, +)
        guard normalizeLength else { return total }
        let ratio = Float(record.tokenCount) / Float(Self.referenceTokenCount)
        return total / max(1, sqrt(max(1, ratio)))
    }

    /// 朴素实现，供 vDSP 路径自测对照。
    static func naiveMaxSim(query: [[Float]], vectors: [[Float]]) -> Float {
        var total: Float = 0
        for q in query {
            var best = -Float.greatestFiniteMagnitude
            for d in vectors {
                var dot: Float = 0
                for i in q.indices { dot += q[i] * d[i] }
                if dot > best { best = dot }
            }
            total += best
        }
        return total
    }
}
