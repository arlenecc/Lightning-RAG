import Foundation
import Hub
import Tokenizers

/// 语义切片的句子相似度：potion-multilingual-128M（int4 ONNX，纯 CPU）。
///
/// potion 是 model2vec 静态词向量模型：每个 token 查表后做 mean-pool + L2
/// 归一化，没有位置编码，因此不像 bge-small-zh 那样受 512 token 窗口限制，
/// 长句与长切片都能完整参与相似度计算。
/// lite 版随 App 资源 bundle 分发；clean 版不带，由启动时的模型自动下载
/// 流程补齐（落点与 `modelDirectory()` 的回退查找一致）。
struct PotionSentenceSimilarity {
    #if arch(x86_64)
    /// Intel 版随包分发 `models/potion-multilingual-128M-onnx-int4`（ONNX 加载）。
    static let resourceName = "potion-multilingual-128M-onnx-int4"
    #else
    static let resourceName = "potion-multilingual-128M-int4"
    #endif
    /// SwiftPM 资源 bundle 名（potion 是该 bundle 的唯一资源）。
    static let resourceBundleName = "BaseAgent_BaseAgentCore.bundle"
    static let modelFileName = "model.onnx"
    static let dimension = 256
    static let batchSize = 32
    static let intraOpThreads = 4
    /// `tokenizer.json` 的 unk id；官方 model2vec 会直接丢弃这些 token。
    static let unknownTokenID = 1
    /// 模型本身没有长度上限；这里只做防御性截断，避免异常超长文本
    /// （如无标点的巨型表格/编码串）一次性占用过多内存。8192 已远超
    /// 默认切片大小与 embedding 推理窗口。
    static let maxSequenceLength = 8_192
    /// 单个子批的 token 总量上限：按 token 数排序后切批，避免长句
    /// 把整批 padding 到极宽、放大 gather 后的浮点内存。
    static let maxBatchTokens = 32_768

    private static let runtime = PotionRuntime()

    /// 资源目录；环境变量仅用于开发/基准替换模型。
    ///
    /// 查找顺序：环境变量覆盖 → App 资源 bundle（lite 版内置）→
    /// 应用 models 目录 → 配置目录 `~/.base-agent/models`
    /// （clean 版启动自动下载的落点）。
    static func modelDirectory() -> URL? {
        let override = ProcessInfo.processInfo.environment["BASE_AGENT_SEMANTIC_MODEL"] ?? ""
        if !override.isEmpty {
            let url = URL(fileURLWithPath: override).standardizedFileURL
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        if let bundled = bundledModelDirectory() { return bundled }
        return downloadedModelCandidates().first {
            FileManager.default.fileExists(
                atPath: $0.appendingPathComponent(modelFileName).path)
        }
    }

    /// 启动自动下载时模型应放置的目录：与 `modelDirectory()` 的回退查找
    /// 保持一致，保证下载完成后一定能被找到。
    static func downloadDirectory() -> URL {
        let fm = FileManager.default
        if let appModels = LocalModelRegistry.applicationModelsDirectory(),
           fm.isWritableFile(atPath: appModels.path) {
            return appModels.appendingPathComponent(resourceName, isDirectory: true)
        }
        return LocalModelRegistry.configModelsDirectory()
            .appendingPathComponent(resourceName, isDirectory: true)
    }

    /// 下载模型的候选查找位置（应用 models 目录 → 配置目录）。
    private static func downloadedModelCandidates() -> [URL] {
        var urls: [URL] = []
        if let appModels = LocalModelRegistry.applicationModelsDirectory() {
            urls.append(appModels.appendingPathComponent(resourceName, isDirectory: true))
        }
        urls.append(LocalModelRegistry.configModelsDirectory()
            .appendingPathComponent(resourceName, isDirectory: true))
        return urls
    }

    /// 手动定位 SwiftPM 资源 bundle。
    ///
    /// 不能直接访问 `Bundle.module`：clean 分发版不带该 bundle，生成的
    /// 访问器找不到时会 fatalError；这里按同一组候选路径探测，缺失时
    /// 安全返回 nil（由启动下载补齐）。
    private static func bundledModelDirectory() -> URL? {
        let fm = FileManager.default
        let anchor = Bundle(for: PotionRuntime.self)
        var candidates: [URL] = []
        if let resourceURL = Bundle.main.resourceURL {
            candidates.append(resourceURL.appendingPathComponent(resourceBundleName))
        }
        candidates.append(Bundle.main.bundleURL.appendingPathComponent(resourceBundleName))
        candidates.append(anchor.bundleURL.appendingPathComponent(resourceBundleName))
        candidates.append(anchor.bundleURL.deletingLastPathComponent()
            .appendingPathComponent(resourceBundleName))
        for candidate in candidates where fm.fileExists(atPath: candidate.path) {
            if let bundle = Bundle(url: candidate),
               let url = bundle.url(forResource: resourceName, withExtension: nil) {
                return url
            }
        }
        return nil
    }

    /// HF `tokenizer.json` 编码结果（已丢弃 unk token）；供 golden 对齐测试使用。
    func tokenIDs(for sentence: String) -> [Int]? {
        Self.runtime.tokenIDs(for: sentence)
    }

    /// 每个句子的向量（已 L2 归一化）。任一句无法编码/推理失败时返回 nil。
    func sentenceVectors(_ sentences: [String]) -> [[Float]]? {
        Self.runtime.sentenceVectors(sentences)
    }

    /// 相邻句余弦（长度 = 句数 - 1）。不足两句或任一句无法编码时返回 nil。
    func adjacentCosines(_ sentences: [String]) -> [Double]? {
        guard sentences.count >= 2 else { return nil }
        guard let vectors = sentenceVectors(sentences) else { return nil }
        return (0..<(vectors.count - 1)).map {
            Self.cosine(vectors[$0], vectors[$0 + 1])
        }
    }

    private static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count else { return 0 }
        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        for index in a.indices {
            dot += Double(a[index]) * Double(b[index])
            normA += Double(a[index]) * Double(a[index])
            normB += Double(b[index]) * Double(b[index])
        }
        guard normA > 0, normB > 0 else { return 0 }
        return dot / (sqrt(normA) * sqrt(normB))
    }
}

/// 模型与分词器只加载一次；失败后缓存失败状态，避免每个章节重复尝试。
private final class PotionRuntime {
    private let lock = NSLock()
    private var state: State?
    private var failed = false
    private var warned = false

    private struct State {
        let session: OrtSession
        let tokenizer: any Tokenizers.Tokenizer
        let unknownTokenID: Int
    }

    func tokenIDs(for sentence: String) -> [Int]? {
        lock.lock()
        defer { lock.unlock() }
        guard let state = loadIfNeeded() else { return nil }
        return encode(sentence, state: state)
    }

    /// 批量编码并推理：按 token 数排序后切子批，失败统一返回 nil 让调用方回退。
    func sentenceVectors(_ sentences: [String]) -> [[Float]]? {
        guard !sentences.isEmpty else { return [] }
        if Task.isCancelled { return nil }
        lock.lock()
        defer { lock.unlock() }
        guard let state = loadIfNeeded() else { return nil }

        var encoded: [(index: Int, ids: [Int])] = []
        encoded.reserveCapacity(sentences.count)
        for (index, sentence) in sentences.enumerated() {
            guard let ids = encode(sentence, state: state) else { return nil }
            encoded.append((index, ids))
        }
        // 长句集中在同一批会把整批 padding 到最长句；先按长度排序再切批，
        // 推理完按原下标放回，结果顺序不变。
        let sorted = encoded.sorted { $0.ids.count < $1.ids.count }

        var result = [[Float]](repeating: [], count: sentences.count)
        var start = 0
        while start < sorted.count {
            if Task.isCancelled { return nil }
            var end = start
            var batchTokens = 0
            while end < sorted.count, end - start < PotionSentenceSimilarity.batchSize {
                let next = sorted[end].ids.count
                if end > start, batchTokens + next > PotionSentenceSimilarity.maxBatchTokens {
                    break
                }
                batchTokens += next
                end += 1
            }
            let batch = Array(sorted[start..<end])
            guard let vectors = run(batch.map(\.ids), state: state) else { return nil }
            for (item, vector) in zip(batch, vectors) {
                result[item.index] = vector
            }
            start = end
        }
        return result
    }

    // MARK: - 加载

    private func loadIfNeeded() -> State? {
        if let state { return state }
        guard !failed else { return nil }
        guard let directory = PotionSentenceSimilarity.modelDirectory() else {
            failed = true
            warnOnce("未找到内置语义分片模型")
            return nil
        }
        do {
            let tokenizer = try Self.loadTokenizer(directory: directory)
            // 静态词向量模型很小，纯 CPU 反向传播只需 4 线程，避免和 LLM 抢资源。
            let session = try OrtSession(
                modelPath: directory
                    .appendingPathComponent(PotionSentenceSimilarity.modelFileName).path,
                intraOpThreads: PotionSentenceSimilarity.intraOpThreads
            )
            let unknown = Self.readUnknownTokenID(directory: directory)
            let state = State(session: session, tokenizer: tokenizer,
                              unknownTokenID: unknown)
            self.state = state
            AgentLog.info("semantic_chunk: 语义切片模型已加载并使用：\(directory.path)")
            return state
        } catch {
            failed = true
            warnOnce("语义分片模型加载失败：\(error.localizedDescription)")
            return nil
        }
    }

    /// 直接按 `tokenizer.json` 构造 Unigram 分词器。
    ///
    /// potion 目录不附带 `tokenizer_config.json`，而 swift-transformers 的
    /// `AutoTokenizer.from(modelFolder:)` 要求该文件存在；这里只显式指定
    /// tokenizer class（Unigram 在库内由 XLM-R 类别承载），其余行为完全由
    /// `tokenizer.json` 的 normalizer / pre_tokenizer / post_processor 决定。
    private static func loadTokenizer(directory: URL) throws -> any Tokenizers.Tokenizer {
        let data = try Data(contentsOf: directory.appendingPathComponent("tokenizer.json"))
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw LocalModelError.invalidConfiguration(
                "tokenizer.json 不是合法 JSON：\(directory.path)")
        }
        var dictionary: [NSString: Any] = [:]
        dictionary.reserveCapacity(object.count)
        for (key, value) in object {
            dictionary[NSString(string: key)] = value
        }
        let tokenizerData = Config(dictionary)
        let tokenizerConfig: Config = ["tokenizer_class": "XLMRobertaTokenizer"]
        return try AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, strict: true)
    }

    private static func readUnknownTokenID(directory: URL) -> Int {
        guard let data = FileManager.default.contents(
            atPath: directory.appendingPathComponent("config.json").path),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let value = object["unk_token_id"] as? Int, value >= 0 else {
            return PotionSentenceSimilarity.unknownTokenID
        }
        return value
    }

    // MARK: - 分词

    private func encode(_ sentence: String, state: State) -> [Int]? {
        // 空句/纯空白在 metaspace 分词下仍可能产出一个 `▁` token，
        // 必须先挡掉，让调用方回退词法相似度。
        guard !sentence.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }
        var ids = state.tokenizer.encode(text: sentence, addSpecialTokens: false)
            .filter { $0 != state.unknownTokenID }
        guard !ids.isEmpty else { return nil }
        if ids.count > PotionSentenceSimilarity.maxSequenceLength {
            ids = Array(ids.prefix(PotionSentenceSimilarity.maxSequenceLength))
        }
        return ids
    }

    // MARK: - 推理

    private func run(_ batch: [[Int]], state: State) -> [[Float]]? {
        let width = batch.map(\.count).max() ?? 0
        guard width > 0 else { return nil }
        var ids: [Int64] = []
        var mask: [Int64] = []
        ids.reserveCapacity(batch.count * width)
        mask.reserveCapacity(batch.count * width)
        for row in batch {
            for index in 0..<width {
                // padding token 用 0：图以 attention_mask 做 mean-pool，
                // mask=0 的位置不会进入均值。
                ids.append(index < row.count ? Int64(row[index]) : 0)
                mask.append(index < row.count ? 1 : 0)
            }
        }
        do {
            var inputs: [String: OrtValue] = [:]
            for name in state.session.inputNames {
                switch name {
                case "input_ids":
                    inputs[name] = try OrtValue(int64: ids, shape: [batch.count, width])
                case "attention_mask":
                    inputs[name] = try OrtValue(int64: mask, shape: [batch.count, width])
                default:
                    throw ONNXRuntimeError.input("语义分片模型要求未知输入 \(name)")
                }
            }
            let outputName = state.session.outputNames.contains("embeddings")
                ? "embeddings" : state.session.outputNames[0]
            let outputs = try state.session.run(inputs: inputs, outputNames: [outputName])
            guard let output = outputs[outputName] else {
                throw ONNXRuntimeError.output("语义分片模型没有返回 \(outputName)")
            }
            let tensor = try output.floatTensor()
            return try pool(tensor, batch: batch, mask: mask)
        } catch {
            warnOnce("语义分片推理失败：\(error.localizedDescription)")
            return nil
        }
    }

    /// 模型输出 `[B,H]`（已 mean-pool + 归一化）；兼容 `[B,T,H]` 的通用输出。
    private func pool(_ tensor: OrtTensor, batch: [[Int]],
                      mask: [Int64]) throws -> [[Float]] {
        let shape = tensor.shape
        if shape.count == 2 {
            let rows = shape[0]
            let hidden = shape[1]
            guard rows == batch.count else {
                throw ONNXRuntimeError.shape("语义分片模型输出 batch=\(rows)")
            }
            return (0..<rows).map { row in
                var vector = Array(tensor.values[(row * hidden)..<((row + 1) * hidden)])
                Self.normalize(&vector)
                return vector
            }
        }
        guard shape.count == 3 else {
            throw ONNXRuntimeError.shape("语义分片模型输出 shape=\(shape)")
        }
        let rows = shape[0]
        let width = shape[1]
        let hidden = shape[2]
        guard rows == batch.count, width > 0 else {
            throw ONNXRuntimeError.shape("语义分片模型输出 shape=\(shape)")
        }
        var result: [[Float]] = []
        result.reserveCapacity(rows)
        for row in 0..<rows {
            var vector = [Float](repeating: 0, count: hidden)
            var count = 0
            for token in 0..<width where mask[row * width + token] != 0 {
                let offset = (row * width + token) * hidden
                for index in 0..<hidden {
                    vector[index] += tensor.values[offset + index]
                }
                count += 1
            }
            if count > 0 {
                let scale = Float(count)
                for index in vector.indices { vector[index] /= scale }
                Self.normalize(&vector)
            }
            result.append(vector)
        }
        return result
    }

    private static func normalize(_ vector: inout [Float]) {
        var sum: Float = 0
        for value in vector { sum += value * value }
        guard sum > 0 else { return }
        let norm = sqrt(sum)
        for index in vector.indices { vector[index] /= norm }
    }

    private func warnOnce(_ message: String) {
        guard !warned else { return }
        warned = true
        AgentLog.warning("semantic_chunk: \(message)，回退词法相似度")
    }
}
