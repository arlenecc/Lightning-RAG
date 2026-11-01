import Foundation

/// 本地 ONNX Embedding 实现。
///
/// - 输入前缀按模型族统一处理（`EmbeddingPrefixStrategy`）；
/// - 输出做 attention-mask 加权 mean pooling，再 L2 归一化；
/// - 按子批执行，避免长文本批量推理一次性占用过多内存。
public final class ONNXEmbeddingProvider: EmbeddingProvider, EmbeddingTokenCounting,
                                          EmbeddingBinarySigning,
                                          @unchecked Sendable {
    public let identifier: String
    public let modelName: String
    public let directory: URL

    private let modelURL: URL?
    private let preflightError: Error?
    private let maxLength: Int
    private let batchSize: Int
    let prefixStrategy: EmbeddingPrefixStrategy
    private let isJinaV3: Bool
    private let isJinaV5: Bool
    private let state: EmbeddingLoadState

    /// ONNXRuntime session.run 线程安全：同实例（共享权重）可并发调用。
    /// 注意 ORT 的 intra-op 线程池是**每 session 一个**，并发 run 共用
    /// 同一个池，并不会各自创建 `intraOpThreads` 个线程；因此
    /// `maxConcurrentRuns × intraOpThreads` 不是实际线程数。
    public let maxConcurrentRuns: Int
    /// 该 session 的 intra-op 线程数（并发 run 共享）。
    public let intraOpThreads: Int

    private let dimensionLock = NSLock()
    private var storedDimension = 768
    /// jina-v5 的 Matryoshka 输出维度；其它模型不使用。
    private let v5OutputDimension: Int

    /// 标准输出维度；实际值会在首次推理后按输出张量更新。
    public var dimension: Int {
        dimensionLock.lock()
        defer { dimensionLock.unlock() }
        return storedDimension
    }

    /// - Parameters:
    ///   - concurrentRuns: 允许的并发 run 数（默认 2，覆盖多文件并发 +
    ///     语义切片同时请求 embedding）。并发 run 共用 session 线程池，
    ///     不增加线程数，但能让线程池在多个 run 之间保持饱和。
    ///   - intraOpThreads: 单 session 的 intra-op 线程数；0 表示按逻辑
    ///     核数自动分配。实测 8 核机器上 2×8 比 2×4 快约 10%（见
    ///     `ONNXThreadBenchTests`），说明并发 run 并没有成倍占用线程。
    ///   - v5Dimension: jina-v5 的 Matryoshka 输出维度（默认 128，仅接受
    ///     模型声明的六档）；其它模型忽略该参数。落盘元素类型由用户配置。
    public init(directory: URL, modelName: String, batchSize: Int = 32,
                concurrentRuns: Int = 2, intraOpThreads: Int = 0,
                v5Dimension: Int = MatryoshkaDimensions.defaultValue) {
        let runs = max(1, concurrentRuns)
        let threads = intraOpThreads > 0
            ? max(1, min(32, intraOpThreads))
            : max(1, min(32, ProcessInfo.processInfo.activeProcessorCount))
        self.modelName = modelName
        self.identifier = "local:onnx:\(modelName)"
        self.maxConcurrentRuns = runs
        self.intraOpThreads = threads
        self.state = EmbeddingLoadState(intraOpThreads: threads)
        self.batchSize = max(1, batchSize)

        // 先解析权重文件：若显式选了 .onnx 文件则直接用，否则在目录内按约定搜索。
        var resolved: URL?
        var loadError: Error?
        do {
            resolved = try LocalModelRegistry.resolveONNXModel(in: directory, kind: .embedding)
        } catch {
            loadError = error
        }
        self.modelURL = resolved
        self.preflightError = loadError

        // 权重文件的所在目录：显式选文件时取父目录，按目录解析时即目录本身。
        // config / tokenizer 均从此目录读取。
        let modelDir = resolved?.pathExtension.lowercased() == "onnx"
            ? resolved!.deletingLastPathComponent() : directory
        self.directory = modelDir

        // 模型窗口可能很大（jina-v3 8192），推理前统一压到 1024 token。
        self.maxLength = EmbeddingInputLimits.cappedWindow(
            LocalModelRegistry.resolveMaxSequenceLength(directory: modelDir, defaultValue: 8192))
        let isJina = JinaV3Profile.isJinaV3(directory: modelDir)
        let isV5 = JinaV5Profile.isJinaV5(directory: modelDir)
        self.isJinaV3 = isJina
        self.isJinaV5 = isV5
        self.v5OutputDimension = JinaV5Profile.validDimension(v5Dimension)
        // config 已认出 jina 时固定用模型族前缀，避免权重目录被改名后前缀退化。
        if isV5 {
            self.prefixStrategy = EmbeddingPrefixStrategy(family: .jinaV5)
        } else if isJina {
            self.prefixStrategy = EmbeddingPrefixStrategy(family: .jina)
        } else {
            self.prefixStrategy = EmbeddingPrefixStrategy(
                modelName: modelName, modelPath: modelDir.lastPathComponent)
        }

        // 从 config.json 预读真实维度，避免非 768 模型建库时维度错配。
        // Jina v3 的 ONNX config 没有 hidden_size，按模型族固定 1024；
        // Jina v5 原生 768 维但按 Matryoshka 配置截断，不能被 hidden_size 覆盖。
        if isV5 {
            self.storedDimension = self.v5OutputDimension
        } else if isJina {
            self.storedDimension = JinaV3Profile.nativeDimension
            if let data = FileManager.default.contents(
                atPath: modelDir.appendingPathComponent("config.json").path),
               let config = JSONValue.parse(String(decoding: data, as: UTF8.self))?.objectValue {
                let hidden = config["hidden_size"]?.intValue
                    ?? config["text_config"]?.objectValue?["hidden_size"]?.intValue
                if let hidden, hidden > 0 {
                    self.storedDimension = hidden
                }
            }
        } else if let data = FileManager.default.contents(
            atPath: modelDir.appendingPathComponent("config.json").path),
           let config = JSONValue.parse(String(decoding: data, as: UTF8.self))?.objectValue {
            let hidden = config["hidden_size"]?.intValue
                ?? config["text_config"]?.objectValue?["hidden_size"]?.intValue
            if let hidden, hidden > 0 {
                self.storedDimension = hidden
            }
        }
    }

    // MARK: - EmbeddingProvider

    public func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        let prefixed = texts.map {
            prefixStrategy.apply($0, task: .retrievalDocument)
        }
        return try await embed(prefixed, task: .retrievalDocument).vectors
    }

    public func embedQuery(_ text: String) async throws -> [Float] {
        let result = try await embed([
            prefixStrategy.apply(text, task: .retrievalQuery)
        ], task: .retrievalQuery)
        return result.vectors.first ?? []
    }

    public var binaryDimension: Int {
        isJinaV5 ? JinaV5Profile.nativeDimension : ((dimension + 7) / 8) * 8
    }
    public var binarySourceIdentifier: String { identifier }

    public func embedDocumentsWithSignatures(_ texts: [String]) async throws
        -> BinaryEmbeddingOutput {
        guard !texts.isEmpty else {
            return BinaryEmbeddingOutput(vectors: [], signatures: [])
        }
        let prefixed = texts.map { prefixStrategy.apply($0, task: .retrievalDocument) }
        let pooled = try await embed(prefixed, task: .retrievalDocument)
        return BinaryEmbeddingOutput(
            vectors: pooled.vectors,
            signatures: pooled.nativeVectors.map {
                VectorEncoder.encodeSignature($0, bitDimension: binaryDimension)
            })
    }

    public func embedQueryWithSignature(_ text: String) async throws
        -> (vector: [Float], signature: Data) {
        let pooled = try await embed([
            prefixStrategy.apply(text, task: .retrievalQuery)
        ], task: .retrievalQuery)
        return (pooled.vectors.first ?? [], pooled.nativeVectors.first.map {
            VectorEncoder.encodeSignature($0, bitDimension: binaryDimension)
        } ?? Data())
    }

    // MARK: - 推理

    private struct PooledEmbeddings {
        let vectors: [[Float]]
        let nativeVectors: [[Float]]
    }

    private func embed(_ texts: [String], task: EmbeddingTask) async throws
        -> PooledEmbeddings {
        if let preflightError { throw preflightError }
        guard let modelURL else {
            throw LocalModelError.runtime("Embedding 模型路径未解析：\(directory.path)")
        }

        let (session, tokenizer) = try await state.handles(
            modelURL: modelURL,
            directory: directory
        )
        let outputName = preferredOutputName(in: session)
        var vectors: [[Float]] = []
        var nativeVectors: [[Float]] = []
        vectors.reserveCapacity(texts.count)
        nativeVectors.reserveCapacity(texts.count)

        var start = 0
        while start < texts.count {
            try Task.checkCancellation()
            let end = min(start + batchSize, texts.count)
            let batch = Array(texts[start..<end])
            let encoded = tokenizer.encodeBatch(batch, maxLength: maxLength)
            let inputs = try makeInputs(
                encoded, inputNames: session.inputNames, task: task)
            let outputs = try session.run(inputs: inputs, outputNames: [outputName])
            guard let output = outputs[outputName] else {
                throw ONNXRuntimeError.output("模型没有返回输出 \(outputName)")
            }
            let tensor = try output.floatTensor()
            let pooled = try pool(tensor, mask: encoded)
            vectors.append(contentsOf: pooled.vectors)
            nativeVectors.append(contentsOf: pooled.nativeVectors)
            start = end
        }
        return PooledEmbeddings(vectors: vectors, nativeVectors: nativeVectors)
    }

    private func makeInputs(
        _ encoded: TokenizedBatch,
        inputNames: [String],
        task: EmbeddingTask
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
            case "task_id":
                // Jina v3 unified ONNX：0=retrieval.query，1=retrieval.passage；
                // 图声明为标量 int64（shape []）。
                let id = isJinaV3
                    ? JinaV3Task.id(for: task)
                    : (task == .retrievalQuery ? 0 : 1)
                inputs[name] = try OrtValue(int64: [id], shape: [])
            default:
                throw ONNXRuntimeError.input("Embedding 模型要求未知输入 `\(name)`，当前封装未实现")
            }
        }
        return inputs
    }

    /// 池化 + L2 归一化：Jina v5 用图内 last-token 输出，其余模型做 mean pooling。
    private func pool(_ tensor: OrtTensor, mask: TokenizedBatch) throws
        -> PooledEmbeddings {
        let shape = tensor.shape
        if shape.count == 2 {
            let batch = shape[0]
            let hidden = shape[1]
            guard batch == mask.batchSize else {
                throw ONNXRuntimeError.shape("输出 batch=\(batch) 与输入 batch=\(mask.batchSize) 不一致")
            }
            if isJinaV5 {
                // V5 图内已做 last-token pooling + L2；这里按配置维截断并重新归一化。
                updateDimension(v5OutputDimension)
                let rawRows = (0..<batch).map { row in
                    Array(tensor.values[(row * hidden)..<((row + 1) * hidden)])
                }
                return PooledEmbeddings(
                    vectors: rawRows.map {
                        JinaV5Profile.truncateAndNormalize(
                            $0, dimension: v5OutputDimension)
                    },
                    nativeVectors: rawRows)
            }
            updateDimension(hidden)
            let vectors = (0..<batch).map { row in
                var vector = Array(tensor.values[(row * hidden)..<((row + 1) * hidden)])
                Self.normalize(&vector)
                return vector
            }
            return PooledEmbeddings(vectors: vectors, nativeVectors: vectors)
        }

        guard shape.count == 3 else {
            throw ONNXRuntimeError.shape("Embedding 输出维度应为 [B,H] 或 [B,T,H]，实际为 \(shape)")
        }
        let batch = shape[0]
        let length = shape[1]
        let hidden = shape[2]
        guard batch == mask.batchSize, length == mask.sequenceLength else {
            throw ONNXRuntimeError.shape(
                "输出 shape=\(shape) 与输入 batch/token=\(mask.batchSize)/\(mask.sequenceLength) 不一致"
            )
        }
        updateDimension(isJinaV5 ? v5OutputDimension : hidden)

        var vectors: [[Float]] = []
        var nativeVectors: [[Float]] = []
        vectors.reserveCapacity(batch)
        nativeVectors.reserveCapacity(batch)
        for row in 0..<batch {
            if isJinaV5 {
                // [B,T,H] 输出时按 V5 约定做 last-token pooling，再按配置维截断重归一化。
                var lastToken = 0
                for token in 0..<length where mask.attentionMask[row][token] != 0 {
                    lastToken = token
                }
                let offset = (row * length + lastToken) * hidden
                let nativeRow = Array(tensor.values[offset..<(offset + hidden)])
                vectors.append(JinaV5Profile.truncateAndNormalize(
                    nativeRow, dimension: v5OutputDimension))
                nativeVectors.append(nativeRow)
                continue
            }
            var vector = [Float](repeating: 0, count: hidden)
            var count = 0
            for token in 0..<length where mask.attentionMask[row][token] != 0 {
                let offset = (row * length + token) * hidden
                for index in 0..<hidden {
                    vector[index] += tensor.values[offset + index]
                }
                count += 1
            }
            if count > 0 {
                let scale = Float(count)
                for index in 0..<hidden {
                    vector[index] /= scale
                }
            }
            Self.normalize(&vector)
            vectors.append(vector)
            nativeVectors.append(vector)
        }
        return PooledEmbeddings(vectors: vectors, nativeVectors: nativeVectors)
    }

    private func preferredOutputName(in session: OrtSession) -> String {
        let preferred = [
            "last_hidden_state", "token_embeddings", "sentence_embedding",
            "embedding", "embeddings",
        ]
        for name in preferred where session.outputNames.contains(name) {
            return name
        }
        return session.outputNames[0]
    }

    private func updateDimension(_ value: Int) {
        guard value > 0 else { return }
        dimensionLock.lock()
        storedDimension = value
        dimensionLock.unlock()
    }

    /// 由 attention_mask 生成 position_ids：每条有效 token 递增，padding 位置保持 0。
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

    private static func normalize(_ vector: inout [Float]) {
        var sum: Float = 0
        for value in vector {
            sum += value * value
        }
        guard sum > 0 else { return }
        let norm = sqrt(sum)
        for index in vector.indices {
            vector[index] /= norm
        }
    }

    /// 释放 ONNX session/tokenizer；provider 保留，下次 embed 时按需重载。
    public func unload() async {
        await state.unload()
    }

    /// 单条输入上限：窗口减去 [CLS]/[SEP] 与当前模型族 document 前缀的预留。
    public var maxContentTokens: Int {
        Self.contentTokenLimit(directory: directory, modelName: modelName)
    }

    /// 与 `maxContentTokens` 同源的静态换算：设置界面不加载权重也能算出
    /// 「切片大小实际生效值」，两处口径不会漂移。
    public static func contentTokenLimit(directory: URL, modelName: String) -> Int {
        let window = EmbeddingInputLimits.cappedWindow(
            LocalModelRegistry.resolveMaxSequenceLength(
                directory: directory, defaultValue: 8_192))
        // V5 即使目录被改名也保持 document 前缀预留，切片口径不漂移。
        let strategy: EmbeddingPrefixStrategy = JinaV5Profile.isJinaV5(directory: directory)
            ? EmbeddingPrefixStrategy(family: .jinaV5)
            : EmbeddingPrefixStrategy(
                modelName: modelName, modelPath: directory.lastPathComponent)
        return max(16, window - 2 - strategy.documentPrefixTokenReserve)
    }

    /// 切片用的精确 token 计数（不含特殊 token）：只加载 tokenizer，不牵动模型权重。
    public func makeTokenCounter() async -> (@Sendable (String) -> Int)? {
        await TokenizerCounterCache.shared.counter(for: directory)
    }
}

/// 会话与 tokenizer 只加载一次；失败后缓存错误，重复调用会得到同一明确错误。
private actor EmbeddingLoadState {
    private var session: OrtSession?
    private var tokenizer: LocalTokenizer?
    private var failure: Error?
    private let intraOpThreads: Int

    init(intraOpThreads: Int) {
        self.intraOpThreads = max(1, intraOpThreads)
    }

    func handles(modelURL: URL, directory: URL) async throws -> (OrtSession, LocalTokenizer) {
        if let session, let tokenizer {
            return (session, tokenizer)
        }
        if let failure {
            throw failure
        }

        do {
            // 线程数由 provider 统一收口：默认「核数 ÷ 并发数」，
            // 避免 2 路并发 × 8 线程在 8 核机器上互相抢占。
            let session = try OrtSession(modelPath: modelURL.path,
                                         intraOpThreads: intraOpThreads)
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
