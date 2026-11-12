import Foundation

/// bge-reranker-v2-m3(-int8) 的 ONNX 交叉编码器实现。
///
/// 输入为 query/document pair，模型输出单个相关性 logit；
/// 这里统一做数值稳定的 sigmoid，返回 `0...1` 的分数。
public final class ONNXReranker: RerankProvider, @unchecked Sendable {
    public let identifier: String
    public let modelName: String
    public let directory: URL

    private let modelURL: URL?
    private let preflightError: Error?
    private let maxLength: Int
    private let batchSize: Int
    private let state = RerankerLoadState()

    public init(directory: URL, modelName: String, batchSize: Int = 8) {
        self.modelName = modelName
        self.identifier = "local:onnx:\(modelName)"
        self.batchSize = max(1, batchSize)

        var resolved: URL?
        var loadError: Error?
        do {
            resolved = try LocalModelRegistry.resolveONNXModel(in: directory, kind: .reranker)
        } catch {
            loadError = error
        }
        self.modelURL = resolved
        self.preflightError = loadError

        // config / tokenizer 的读取目录：权重在 `onnx/` 子目录、tokenizer 在
        // 根目录时不能只看权重父目录，否则会误报缺少 tokenizer.json。
        self.directory = resolved.map {
            LocalModelRegistry.onnxSupportDirectory(modelURL: $0, modelRoot: directory)
        } ?? directory
        let modelDir = self.directory
        // XLM-R 系 ONNX 图内自算 position id（cumsum + padding_idx=1），
        // 有效长度是 tokenizer 声明的 model_max_length（如 base 为 512），
        // 直接用 config 的 max_position_embeddings（514）会在长 pair 上 Gather 越界。
        self.maxLength = min(
            LocalModelRegistry.resolveMaxSequenceLength(directory: modelDir, defaultValue: 8192),
            LocalModelRegistry.tokenizerModelMaxLength(directory: modelDir) ?? 8_192)
    }

    // MARK: - RerankProvider

    public func rerank(query: String, documents: [String]) async throws -> [Double] {
        guard !documents.isEmpty else { return [] }
        if let preflightError { throw preflightError }
        guard let modelURL else {
            throw LocalModelError.runtime("Reranker 模型路径未解析：\(directory.path)")
        }

        let (session, tokenizer) = try await state.handles(
            modelURL: modelURL,
            directory: directory
        )
        let outputName = preferredOutputName(in: session)

        // 先整体编码再按 token 长度分桶：组内 pad 到组内最长，
        // 避免一段混进长文档时其余短文档陪跑满宽；分数按原始索引写回。
        let encodedPairs = documents.map {
            tokenizer.encodePair(query: query, document: $0, maxLength: maxLength)
        }
        let batches = Self.lengthBucketedBatches(
            lengths: encodedPairs.map(\.count), batchSize: batchSize)
        var scores = [Double](repeating: 0, count: encodedPairs.count)

        for indices in batches {
            try Task.checkCancellation()
            let encoded = tokenizer.pad(indices.map { encodedPairs[$0] })
            let inputs = try makeInputs(encoded, inputNames: session.inputNames)
            let outputs = try session.run(inputs: inputs, outputNames: [outputName])
            guard let output = outputs[outputName] else {
                throw ONNXRuntimeError.output("模型没有返回输出 \(outputName)")
            }
            let tensor = try output.floatTensor()
            let batchScores = try logitsToScores(tensor, batchSize: encoded.batchSize)
            for (offset, original) in indices.enumerated() {
                scores[original] = batchScores[offset]
            }
        }
        return scores
    }

    /// 长度分桶的批次规划：按 token 长度升序排列后，**按长度比例**聚桶——
    /// 桶内最长不超过桶内最短的 `ratioThreshold` 倍（默认 1.5），同时单桶不超过
    /// `batchSize` 上限。这样短文档只陪跑到「长度相近」的文档，避免被长尾文档
    /// 一把 pad 到超长、徒增计算；长度齐整时仍会自然并成大桶吃满 batching 收益。
    /// `ratioThreshold` 取 `.infinity` 时退化为固定 `batchSize` 切片（旧行为）。
    static func lengthBucketedBatches(
        lengths: [Int], batchSize: Int, ratioThreshold: Double = 1.5
    ) -> [[Int]] {
        let size = max(1, batchSize)
        let ordered = lengths.indices.sorted {
            lengths[$0] == lengths[$1] ? $0 < $1 : lengths[$0] < lengths[$1]
        }
        var batches: [[Int]] = []
        var current: [Int] = []
        var currentMin = Int.max
        for idx in ordered {
            let len = lengths[idx]
            if current.isEmpty {
                current = [idx]
                currentMin = len
            } else if current.count < size,
                      Double(len) <= Double(currentMin) * ratioThreshold {
                current.append(idx)
            } else {
                batches.append(current)
                current = [idx]
                currentMin = len
            }
        }
        if !current.isEmpty { batches.append(current) }
        return batches
    }

    // MARK: - 推理

    private func makeInputs(
        _ encoded: TokenizedBatch,
        inputNames: [String]
    ) throws -> [String: OrtValue] {
        let batch = encoded.batchSize
        let length = encoded.sequenceLength
        let inputIDs = encoded.inputIDs.flatMap { row in row.map(Int64.init) }
        let mask = encoded.attentionMask.flatMap { row in row.map(Int64.init) }

        var inputs: [String: OrtValue] = [:]
        for name in inputNames {
            switch name {
            case "input_ids":
                inputs[name] = try OrtValue(int64: inputIDs, shape: [batch, length])
            case "attention_mask":
                inputs[name] = try OrtValue(int64: mask, shape: [batch, length])
            case "token_type_ids":
                inputs[name] = try OrtValue(
                    int64: Array(repeating: 0, count: batch * length),
                    shape: [batch, length]
                )
            case "position_ids":
                inputs[name] = try OrtValue(
                    int64: Self.positionIDs(mask: encoded.attentionMask),
                    shape: [batch, length]
                )
            default:
                throw ONNXRuntimeError.input("Reranker 模型要求未知输入 `\(name)`，当前封装未实现")
            }
        }
        return inputs
    }

    /// `[B,1]` / `[B]` / `[1,B]` logits → sigmoid。
    private func logitsToScores(_ tensor: OrtTensor, batchSize: Int) throws -> [Double] {
        let shape = tensor.shape
        let values = tensor.values

        if shape.count == 1 {
            guard values.count >= batchSize else {
                throw ONNXRuntimeError.shape("logits 数量 \(values.count) 小于 batch \(batchSize)")
            }
            return activate(values.prefix(batchSize).map { Double($0) })
        }

        guard shape.count == 2 else {
            throw ONNXRuntimeError.shape("Reranker logits 维度应为 [B]、[B,1] 或 [1,B]，实际为 \(shape)")
        }

        let rows = shape[0]
        let columns = shape[1]
        if rows == batchSize {
            // 常规 [B, num_labels]，num_labels 通常为 1，取第 0 列。
            var result: [Double] = []
            result.reserveCapacity(batchSize)
            for row in 0..<batchSize {
                result.append(Double(values[row * columns]))
            }
            return activate(result)
        }
        if rows == 1 && columns >= batchSize {
            // 少数导出模型的 [1, B] 布局。
            return activate(values.prefix(batchSize).map { Double($0) })
        }
        throw ONNXRuntimeError.shape("logits shape=\(shape) 与 batch=\(batchSize) 不匹配")
    }

    /// 交叉编码器导出（jina / bge 系）输出单个相关性 logit，必须统一做 sigmoid。
    /// 不能按“值落在 [0,1]”猜测概率：正 logit 恰好在 [0,1] 时会被误当成概率，
    /// 导致相关文档分数被系统性压低（例如 logit 0.183 实际应为 0.546）。
    private func activate(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        return values.map { Self.sigmoid(Float($0)) }
    }

    private func preferredOutputName(in session: OrtSession) -> String {
        let preferred = ["logits", "scores", "output"]
        for name in preferred where session.outputNames.contains(name) {
            return name
        }
        return session.outputNames[0]
    }

    /// 数值稳定的 sigmoid，避免 exp 溢出。
    private static func sigmoid(_ value: Float) -> Double {
        let x = Double(value)
        if x >= 0 {
            return 1.0 / (1.0 + exp(-x))
        }
        let exponent = exp(x)
        return exponent / (1.0 + exponent)
    }

    private static func positionIDs(mask: [[Int]]) -> [Int64] {
        var result: [Int64] = []
        result.reserveCapacity(mask.reduce(0) { $0 + $1.count })
        for row in mask {
            var position: Int64 = -1
            for flag in row {
                if flag != 0 {
                    position += 1
                    result.append(max(position, 0))
                } else {
                    result.append(0)
                }
            }
        }
        return result
    }

    /// 释放 ONNX session/tokenizer；provider 保留，下次 rerank 时按需重载。
    public func unload() async {
        await state.unload()
    }
}

/// 会话与 tokenizer 只加载一次；失败缓存错误，便于定位模型缺失。
private actor RerankerLoadState {
    private var session: OrtSession?
    private var tokenizer: LocalTokenizer?
    private var failure: Error?

    func handles(modelURL: URL, directory: URL) async throws -> (OrtSession, LocalTokenizer) {
        if let session, let tokenizer {
            return (session, tokenizer)
        }
        if let failure {
            throw failure
        }

        do {
            let session = try OrtSession(modelPath: modelURL.path, intraOpThreads: 4)
            let tokenizer = try await LocalTokenizer.load(directory: directory)
            self.session = session
            self.tokenizer = tokenizer
            return (session, tokenizer)
        } catch {
            self.failure = error
            throw error
        }
    }

    func unload() {
        session = nil
        tokenizer = nil
        failure = nil
    }
}
