import Foundation
import Accelerate

/// 块内 int8 量化向量缓冲：评分统一 dot × scale / 127。
struct ANNBlockBuffer {
    private(set) var rowids: [Int64] = []
    private(set) var codes: [Int8] = []
    private(set) var scales: [Float] = []
    private(set) var lastRowID: Int64 = 0
    var count: Int { rowids.count }

    mutating func reset(maxRows: Int, dim: Int) {
        rowids.removeAll(keepingCapacity: true)
        codes.removeAll(keepingCapacity: true)
        scales.removeAll(keepingCapacity: true)
        rowids.reserveCapacity(maxRows)
        codes.reserveCapacity(maxRows * dim)
        scales.reserveCapacity(maxRows)
        lastRowID = 0
    }

    mutating func append(rowID: Int64, floats: [Float], dim: Int) {
        guard floats.count == dim else { return }
        var maxAbs: Float = 0
        for value in floats { maxAbs = max(maxAbs, abs(value)) }
        let scale = maxAbs > 0 ? maxAbs : 1
        let factor = 127 / scale
        for value in floats {
            let q = (value * factor).rounded()
            codes.append(Int8(max(-127, min(127, q))))
        }
        scales.append(scale)
        rowids.append(rowID)
        lastRowID = rowID
    }

    func dequantize(row: Int, dim: Int, into output: inout [Float]) {
        if output.count != dim { output = [Float](repeating: 0, count: dim) }
        guard row >= 0, (row + 1) * dim <= codes.count else {
            for i in 0..<dim { output[i] = 0 }
            return
        }
        let factor = (row < scales.count ? scales[row] : 1) / 127
        codes.withUnsafeBufferPointer { source in
            guard let base = source.baseAddress else { return }
            output.withUnsafeMutableBufferPointer { destination in
                guard let out = destination.baseAddress else { return }
                vDSP_vflt8(base + row * dim, 1, out, 1, vDSP_Length(dim))
                var f = factor
                vDSP_vsmul(out, 1, &f, out, 1, vDSP_Length(dim))
            }
        }
    }
}

/// 所有块的 int8 簇心缓存；nearest 用 vDSP 点积维护 nprobe 降序表。
final class ANNCentroidCache {
    let blocks: Int, clusters: Int, dim: Int
    private let codes: [Int8]
    private let scales: [Float]

    init(blocks: Int, clusters: Int, dim: Int, codes: [Int8], scales: [Float]) {
        self.blocks = blocks
        self.clusters = clusters
        self.dim = dim
        self.codes = codes
        self.scales = scales
    }

    func nearest(query: UnsafePointer<Float>, block: Int, nprobe: Int,
                 scratch: inout [Float]) -> [Int] {
        guard clusters > 0, dim > 0, nprobe > 0, block >= 0, block < blocks else { return [] }
        let base = block * clusters * dim
        guard base >= 0, base + clusters * dim <= codes.count else { return [] }
        if scratch.count < clusters * dim { scratch = [Float](repeating: 0, count: clusters * dim) }
        codes.withUnsafeBufferPointer { source in
            guard let p = source.baseAddress else { return }
            vDSP_vflt8(p + base, 1, &scratch, 1, vDSP_Length(clusters * dim))
        }
        var probe: [Int] = []
        var probeScores: [Float] = []
        probe.reserveCapacity(nprobe)
        probeScores.reserveCapacity(nprobe)
        scratch.withUnsafeBufferPointer { floats in
            guard let p = floats.baseAddress else { return }
            for cluster in 0..<clusters {
                var dot: Float = 0
                vDSP_dotpr(p + cluster * dim, 1, query, 1, &dot, vDSP_Length(dim))
                let score = dot * scales[block * clusters + cluster] / 127
                if probe.count == nprobe, score <= probeScores[probe.count - 1] { continue }
                var slot = probe.count
                while slot > 0, probeScores[slot - 1] < score { slot -= 1 }
                if probe.count == nprobe { probe.removeLast(); probeScores.removeLast() }
                probe.insert(cluster, at: slot)
                probeScores.insert(score, at: slot)
            }
        }
        return probe
    }
}

/// 容量受限 top-K 小顶堆。
struct ANNTopKCollector {
    let capacity: Int
    private var rowids: [Int64] = []
    private var scores: [Double] = []

    init(capacity: Int) { self.capacity = max(1, capacity) }

    var count: Int { rowids.count }

    mutating func offer(rowid: Int64, score: Double) {
        if rowids.count < capacity {
            rowids.append(rowid); scores.append(score)
            var index = rowids.count - 1
            while index > 0 {
                let parent = (index - 1) / 2
                if scores[parent] <= scores[index] { break }
                scores.swapAt(parent, index); rowids.swapAt(parent, index)
                index = parent
            }
            return
        }
        guard score > scores[0] else { return }
        scores[0] = score; rowids[0] = rowid
        var index = 0
        while true {
            let left = index * 2 + 1, right = left + 1
            var smallest = index
            if left < rowids.count, scores[left] < scores[smallest] { smallest = left }
            if right < rowids.count, scores[right] < scores[smallest] { smallest = right }
            if smallest == index { break }
            scores.swapAt(index, smallest); rowids.swapAt(index, smallest)
            index = smallest
        }
    }

    func sortedItems() -> [(Int64, Double)] {
        let items = Array(zip(rowids, scores))
        return items.sorted { $0.1 > $1.1 }
    }
}

enum ANNPacking {
    static func packCell(buffer: ANNBlockBuffer, order: [Int32],
                         range: Range<Int>, dim: Int) -> Data {
        let rowBytes = 12 + dim
        var bytes = [UInt8](repeating: 0, count: range.count * rowBytes)
        bytes.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return }
            buffer.codes.withUnsafeBufferPointer { codes in
                guard let codeBase = codes.baseAddress else { return }
                var offset = 0
                for slot in range {
                    let index = Int(order[slot])
                    raw.storeBytes(of: buffer.rowids[index], toByteOffset: offset, as: Int64.self)
                    raw.storeBytes(of: buffer.scales[index], toByteOffset: offset + 8, as: Float.self)
                    base.advanced(by: offset + 12)
                        .copyMemory(from: UnsafeRawPointer(codeBase + index * dim), byteCount: dim)
                    offset += rowBytes
                }
            }
        }
        return Data(bytes)
    }

    static func packCentroids(centroids: [Float], counts: [Int32],
                              clusters: Int, dim: Int) -> Data {
        var bytes = [UInt8](repeating: 0, count: 12 + clusters * (8 + dim))
        bytes.withUnsafeMutableBytes { raw in
            raw.storeBytes(of: UInt32(0x414E_4E31), toByteOffset: 0, as: UInt32.self) // ANN1
            raw.storeBytes(of: UInt32(clusters), toByteOffset: 4, as: UInt32.self)
            raw.storeBytes(of: UInt32(dim), toByteOffset: 8, as: UInt32.self)
            guard let base = raw.baseAddress else { return }
            var offset = 12
            for cluster in 0..<clusters {
                let start = cluster * dim
                var maxAbs: Float = 0
                for i in 0..<dim { maxAbs = max(maxAbs, abs(centroids[start + i])) }
                let scale = maxAbs > 0 ? maxAbs : 1
                let factor = 127 / scale
                raw.storeBytes(of: UInt32(bitPattern: counts[cluster]), toByteOffset: offset,
                               as: UInt32.self)
                raw.storeBytes(of: scale, toByteOffset: offset + 4, as: Float.self)
                let codes = base.advanced(by: offset + 8).assumingMemoryBound(to: Int8.self)
                for i in 0..<dim {
                    let q = (centroids[start + i] * factor).rounded()
                    codes[i] = Int8(max(-127, min(127, q)))
                }
                offset += 8 + dim
            }
        }
        return Data(bytes)
    }

    static func decodeCentroids(_ data: Data, clusters: Int, dim: Int) -> (codes: [Int8], scales: [Float])? {
        let rowBytes = 8 + dim
        guard data.count >= 12 + clusters * rowBytes else { return nil }
        var codes = [Int8](repeating: 0, count: clusters * dim)
        var scales = [Float](repeating: 0, count: clusters)
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            let magic = raw.loadUnaligned(fromByteOffset: 0, as: UInt32.self)
            guard magic == 0x414E_4E31 else { return }
            for cluster in 0..<clusters {
                let offset = 12 + cluster * rowBytes
                scales[cluster] = raw.loadUnaligned(fromByteOffset: offset + 4, as: Float.self)
                let src = base.advanced(by: offset + 8).assumingMemoryBound(to: Int8.self)
                for i in 0..<dim { codes[cluster * dim + i] = src[i] }
            }
        }
        return (codes, scales)
    }
}

enum ANNClustering {
    struct Result { var centroids: [Float]; var counts: [Int32] }

    static func normalize(_ vector: inout [Float], count: Int) {
        guard count > 0, vector.count >= count else { return }
        var norm: Float = 0
        vector.withUnsafeMutableBufferPointer { p in
            guard let base = p.baseAddress else { return }
            vDSP_svesq(base, 1, &norm, vDSP_Length(count))
            guard norm > 0 else { return }
            var scale = 1 / sqrt(norm)
            vDSP_vsmul(base, 1, &scale, base, 1, vDSP_Length(count))
        }
    }

    static func train(buffer: ANNBlockBuffer, rows: Int, dim: Int, clusters: Int,
                      iterations: Int, seeds: [Float]) -> Result? {
        guard rows > 0, clusters > 0, dim > 0, seeds.count >= clusters * dim else { return nil }
        let sampleRows = min(rows, 8192)
        var sample = [Float](repeating: 0, count: sampleRows * dim)
        var row = [Float](repeating: 0, count: dim)
        for index in 0..<sampleRows {
            let source = sampleRows >= rows ? min(index, rows - 1)
                : min(Int(Double(index) * Double(rows) / Double(sampleRows)), rows - 1)
            buffer.dequantize(row: source, dim: dim, into: &row)
            for i in 0..<dim { sample[index * dim + i] = row[i] }
        }
        var centroids = Array(seeds[0..<(clusters * dim)])
        for cluster in 0..<clusters {
            let offset = cluster * dim
            var slice = Array(centroids[offset..<(offset + dim)])
            normalize(&slice, count: dim)
            centroids.replaceSubrange(offset..<(offset + dim), with: slice)
        }
        var assignments = [Int32](repeating: 0, count: sampleRows)
        var counts = [Int32](repeating: 0, count: clusters)
        for _ in 0..<max(1, iterations) {
            assignMatrix(matrix: sample, rows: sampleRows, dim: dim, clusters: clusters,
                         centroids: centroids, output: &assignments)
            var sums = [Float](repeating: 0, count: clusters * dim)
            counts = [Int32](repeating: 0, count: clusters)
            for index in 0..<sampleRows {
                let c = Int(assignments[index])
                counts[c] += 1
                for i in 0..<dim { sums[c * dim + i] += sample[index * dim + i] }
            }
            for c in 0..<clusters where counts[c] > 0 {
                let offset = c * dim
                var slice = Array(sums[offset..<(offset + dim)])
                normalize(&slice, count: dim)
                for i in 0..<dim { centroids[offset + i] = slice[i] }
            }
        }
        assignMatrix(matrix: sample, rows: sampleRows, dim: dim, clusters: clusters,
                     centroids: centroids, output: &assignments)
        counts = [Int32](repeating: 0, count: clusters)
        for index in 0..<sampleRows { counts[Int(assignments[index])] += 1 }
        return Result(centroids: centroids, counts: counts)
    }

    static func assign(buffer: ANNBlockBuffer, rows: Int, dim: Int,
                       clusters: Int, centroids: [Float]) -> [Int32] {
        var output = [Int32](repeating: 0, count: rows)
        guard rows > 0, clusters > 0, dim > 0 else { return output }
        let batch = max(1, min(rows, min(8192, max(256, 4_000_000 / max(1, dim + clusters)))))
        var matrix = [Float](repeating: 0, count: batch * dim)
        var scratch = [Int32](repeating: 0, count: batch)
        var row = [Float](repeating: 0, count: dim)
        var start = 0
        while start < rows {
            let count = min(batch, rows - start)
            for index in 0..<count {
                buffer.dequantize(row: start + index, dim: dim, into: &row)
                for i in 0..<dim { matrix[index * dim + i] = row[i] }
            }
            var local = Array(scratch[0..<count])
            assignMatrix(matrix: matrix, rows: count, dim: dim, clusters: clusters,
                         centroids: centroids, output: &local)
            for index in 0..<count { output[start + index] = local[index] }
            start += count
        }
        return output
    }

    static func assignMatrix(matrix: [Float], rows: Int, dim: Int, clusters: Int,
                             centroids: [Float], output: inout [Int32]) {
        guard rows > 0, clusters > 0, dim > 0, matrix.count >= rows * dim,
              centroids.count >= clusters * dim else { return }
        if output.count != rows { output = [Int32](repeating: 0, count: rows) }
        var transposed = [Float](repeating: 0, count: dim * clusters)
        for cluster in 0..<clusters {
            let source = cluster * dim
            for i in 0..<dim { transposed[i * clusters + cluster] = centroids[source + i] }
        }
        var scores = [Float](repeating: 0, count: rows * clusters)
        matrix.withUnsafeBufferPointer { m in
            transposed.withUnsafeBufferPointer { c in
                scores.withUnsafeMutableBufferPointer { s in
                    guard let mb = m.baseAddress, let cb = c.baseAddress, let sb = s.baseAddress else { return }
                    vDSP_mmul(mb, 1, cb, 1, sb, 1,
                              vDSP_Length(rows), vDSP_Length(clusters), vDSP_Length(dim))
                }
            }
        }
        scores.withUnsafeBufferPointer { p in
            guard let base = p.baseAddress else { return }
            for row in 0..<rows {
                var best = 0
                var bestScore = -Float.greatestFiniteMagnitude
                let offset = row * clusters
                for cluster in 0..<clusters where base[offset + cluster] > bestScore {
                    bestScore = base[offset + cluster]
                    best = cluster
                }
                output[row] = Int32(best)
            }
        }
    }
}
