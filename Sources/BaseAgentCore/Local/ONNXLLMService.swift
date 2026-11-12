import Foundation
import OnnxRuntimeBindings

/// ONNX 解码器专项错误，避免复用 GGUF 的错误文案。
enum ONNXLLMError: LocalizedError, Sendable {
    case unsupported(String)
    case decode(String)

    var errorDescription: String? {
        switch self {
        case .unsupported(let message): return "ONNX 模型不受支持：\(message)"
        case .decode(let message): return "ONNX 推理失败：\(message)"
        }
    }
}

/// 进程内 ONNX 解码器 LLM。
///
/// 直接使用 onnxruntime-swift 运行 `input_ids` / `attention_mask` /
/// `position_ids` + `past_key_values` 图，不依赖 onnxruntime-genai 或 Python。
public final class ONNXLLMService: LLMService, PreloadableLLMService,
                                   WarmupLLMService, ClosableLLMService,
                                   KVCacheInvalidating, @unchecked Sendable {
    public let identifier: String
    public let contextWindow: Int
    public let supportsTools = true
    /// `isolated` 请求使用独立的 past/logits，且 ONNX Runtime 的 Session.run 可并发。
    public var supportsConcurrentIsolatedRequests: Bool { true }

    private let directory: URL
    private let tokenizerDirectory: URL?
    private let modelName: String
    private let maxTokens: Int
    private let topP: Double
    private let minP: Double
    private let topK: Int
    private let repetitionPenalty: Double
    private let enableThinking: Bool

    private let gate = AsyncGate()
    private let stateLock = NSLock()
    private var loader: Task<ONNXLoaded, Error>?
    private var cache: ONNXCache?

    init(directory: URL,
         tokenizerDirectory: URL? = nil,
         modelName: String,
         contextWindow: Int,
         maxTokens: Int,
         topP: Double,
         minP: Double,
         topK: Int,
         repetitionPenalty: Double,
         enableThinking: Bool) {
        self.directory = directory
        self.tokenizerDirectory = tokenizerDirectory
        self.modelName = modelName
        self.identifier = "local:onnx:\(modelName)"
        self.contextWindow = max(1_024, contextWindow)
        self.maxTokens = max(1, maxTokens)
        self.topP = topP
        self.minP = minP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.enableThinking = enableThinking
    }

    // MARK: - LLMService

    public func chatStream(messages: [ChatMessage],
                           tools: [JSONValue],
                           temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        makeStream(messages: messages, tools: tools, temperature: temperature,
                   options: .default)
    }

    public func chatStream(messages: [ChatMessage],
                           tools: [JSONValue],
                           temperature: Double,
                           options: LLMRequestOptions) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        makeStream(messages: messages, tools: tools, temperature: temperature,
                   options: options)
    }

    private func makeStream(messages: [ChatMessage],
                            tools: [JSONValue],
                            temperature: Double,
                            options: LLMRequestOptions) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.run(messages: messages, tools: tools,
                                       temperature: temperature,
                                       enableThinking: options.enableThinking ?? self.enableThinking,
                                       isolated: options.isolated,
                                       continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: Self.mapError(error, modelName: modelName))
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public func preload() async throws {
        _ = try await loaded()
    }

    public func close() {
        stateLock.lock()
        loader = nil
        cache = nil
        stateLock.unlock()
    }

    public func invalidateKVCache() {
        stateLock.lock()
        cache = nil
        stateLock.unlock()
    }

    private func loaded() async throws -> ONNXLoaded {
        stateLock.lock()
        if let loader {
            stateLock.unlock()
            return try await loader.value
        }
        let task = Task {
            try await ONNXLoaded.load(directory: directory,
                                      tokenizerDirectory: tokenizerDirectory,
                                      modelName: modelName)
        }
        loader = task
        stateLock.unlock()
        do {
            return try await task.value
        } catch {
            stateLock.lock()
            if loader == task { loader = nil }
            stateLock.unlock()
            throw error
        }
    }

    // MARK: - 预热

    public func warmup(systemPrompt: String, tools: [JSONValue]) async throws {
        guard !systemPrompt.isEmpty else { return }
        try await gate.withLock {
            try Task.checkCancellation()
            let loaded = try await loaded()
            guard loaded.graph.canReusePast else {
                AgentLog.info("[onnx-warmup] 模型没有可复用的 past_key_values，跳过预热")
                return
            }
            // 部分模板要求至少有一条 user 消息；多渲染一条占位 user，
            // 只缓存到占位内容之前（system+tools+user 头），正式请求可精确续写。
            let marker = "__WARMUP_\(UUID().uuidString)__"
            let rendered: [Int]
            do {
                rendered = try loaded.tokenizer.applyChatTemplate(
                    messages: [ChatMessage(role: .system, content: systemPrompt),
                               ChatMessage(role: .user, content: marker)],
                    tools: tools,
                    enableThinking: enableThinking,
                    addGenerationPrompt: false)
            } catch {
                AgentLog.warning("[onnx-warmup] 模板不支持前缀预热，跳过："
                    + error.localizedDescription)
                return
            }
            let markerTokens = loaded.tokenizer.encode(marker, addSpecialTokens: false)
            var tokens = rendered
            if !markerTokens.isEmpty,
               let start = Self.firstIndex(of: markerTokens, in: rendered) {
                tokens = Array(rendered.prefix(start))
            }
            guard tokens.count >= 32, tokens.count <= contextWindow - 1_024 else {
                AgentLog.warning("[onnx-warmup] 跳过：前缀 token 数 \(tokens.count) 不适合预热")
                return
            }
            let result = try loaded.graph.prefill(
                tokens: tokens,
                initialPast: try loaded.graph.initialPast(),
                chunkSize: Self.chunkSize,
                contextLimit: contextWindow)
            stateLock.lock()
            cache = ONNXCache(tokens: tokens, past: result.past,
                              logits: result.logits, toolsKey: Self.toolsKey(tools))
            stateLock.unlock()
            AgentLog.info("[onnx-warmup] prefill tokens=\(tokens.count) tools=\(tools.count)")
        }
    }

    // MARK: - 生成

    private func run(messages: [ChatMessage],
                     tools: [JSONValue],
                     temperature: Double,
                     enableThinking: Bool,
                     isolated: Bool,
                     continuation: AsyncThrowingStream<LLMStreamEvent, Error>.Continuation) async throws {
        if isolated {
            // 一次性分片各自持有 past/logits，不读写共享 cache；ONNX Runtime 的
            // Session.run 可并发执行，因此不占用串行 gate，辅助模型分片可真正并行。
            try await generate(messages: messages, tools: tools,
                               temperature: temperature,
                               enableThinking: enableThinking,
                               isolated: true, continuation: continuation)
        } else {
            try await gate.withLock {
                try await generate(messages: messages, tools: tools,
                                   temperature: temperature,
                                   enableThinking: enableThinking,
                                   isolated: false, continuation: continuation)
            }
        }
    }

    /// 生成主体。共享 cache 的会话请求由调用方持有 gate；`isolated` 请求不碰 cache。
    private func generate(messages: [ChatMessage],
                          tools: [JSONValue],
                          temperature: Double,
                          enableThinking: Bool,
                          isolated: Bool,
                          continuation: AsyncThrowingStream<LLMStreamEvent, Error>.Continuation) async throws {
        try Task.checkCancellation()
        let loaded = try await loaded()
            let tokenizer = loaded.tokenizer
            let baseTokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: tools, enableThinking: enableThinking,
                addGenerationPrompt: false)
            let tokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: tools, enableThinking: enableThinking)
            let generationPrefix = Self.generationPrefix(
                base: baseTokens, full: tokens, tokenizer: tokenizer)
            guard !tokens.isEmpty else { throw AgentError.llm("ONNX prompt 为空") }
            guard tokens.count <= contextWindow - 64 else {
                throw AgentError.llm(
                    "ONNX prompt \(tokens.count) tokens 超过上下文预算 \(contextWindow)")
            }

            let toolsKey = Self.toolsKey(tools)
            stateLock.lock()
            let cached = cache
            stateLock.unlock()

            var reused = false
            var past: [String: OrtValue]
            var pastLength = 0
            var lastLogits: [Float]
            let promptTokenCount = tokens.count

            if !isolated, loaded.graph.canReusePast,
               let cached,
               cached.toolsKey == toolsKey,
               cached.tokens.count <= tokens.count,
               Array(tokens.prefix(cached.tokens.count)) == cached.tokens {
                reused = true
                past = cached.past
                pastLength = cached.tokens.count
                lastLogits = cached.logits
            } else {
                let initial = try loaded.graph.initialPast()
                past = initial
                lastLogits = []
            }

            let suffix = reused ? Array(tokens[pastLength...]) : tokens
            if !suffix.isEmpty || lastLogits.isEmpty {
                let result = try loaded.graph.prefill(
                    tokens: suffix,
                    initialPast: past,
                    chunkSize: Self.chunkSize,
                    contextLimit: contextWindow)
                past = result.past
                pastLength = reused ? pastLength + suffix.count : suffix.count
                lastLogits = result.logits
            }
            let cachedPromptTokens = reused ? tokens.count - suffix.count : 0

            let startInThink = tokenizer
                .decode(tokens, skipSpecialTokens: false)
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .hasSuffix("<think>")
            var thinkSplitter = ThinkSplitter(startInThink: startInThink)
            var toolSplitter = ToolTagSplitter()
            var rawContent = generationPrefix
            var displayTokens: [Int] = []
            var emitted = ""
            let allowedToolNames = Set(tools.compactMap {
                $0["function"]?["name"]?.stringValue
            })
            let sampler = ONNXSampler(config: LlamaCppSampling(
                                        temperature: temperature,
                                        topK: topK, topP: topP, minP: minP,
                                        repetitionPenalty: repetitionPenalty))
            let generatedBudget = max(1, min(maxTokens, contextWindow - tokens.count - 8))
            var generated: [Int] = []
            let generationStart = Date()

            func emit(_ text: String) {
                rawContent += text
                for piece in thinkSplitter.feed(text) {
                    if piece.reasoning {
                        if !piece.text.isEmpty {
                            continuation.yield(LLMStreamEvent(
                                kind: .reasoning, content: piece.text))
                        }
                    } else {
                        for visible in toolSplitter.feed(piece.text) where !visible.isEmpty {
                            continuation.yield(LLMStreamEvent(
                                kind: .content, content: visible))
                        }
                    }
                }
            }

            func emitDisplayToken(_ token: Int) {
                displayTokens.append(token)
                let text = tokenizer.decode(displayTokens, skipSpecialTokens: true)
                if text.hasPrefix(emitted) {
                    let delta = String(text.dropFirst(emitted.count))
                    if !delta.isEmpty { emit(delta) }
                    emitted = text
                } else {
                    let piece = tokenizer.decode([token], skipSpecialTokens: true)
                    if !piece.isEmpty { emit(piece) }
                    emitted += piece
                }
            }

            do {
                var logits = lastLogits
                while generated.count < generatedBudget {
                    try Task.checkCancellation()
                    if pastLength >= contextWindow { break }
                    let token = sampler.sample(logits: logits)
                    sampler.accept(token)
                    generated.append(token)
                    if loaded.eosTokens.contains(token) {
                        // 把 EOS 也解码进 KV，并更新 logits；
                        // 下一轮 prompt 的 EOS 才能精确命中前缀。
                        if loaded.graph.canReusePast {
                            logits = try loaded.graph.step(
                                token: token, past: &past, pastLength: &pastLength,
                                contextLimit: contextWindow)
                        }
                        break
                    }
                    if loaded.graph.canReusePast {
                        logits = try loaded.graph.step(
                            token: token, past: &past, pastLength: &pastLength,
                            contextLimit: contextWindow)
                    } else {
                        // 没有 past 输入的导出：每一步都重跑完整序列。
                        let all = tokens + generated
                        logits = try loaded.graph.logits(tokens: all,
                                                         contextLimit: contextWindow)
                        pastLength = all.count
                    }
                    emitDisplayToken(token)
                }

                for piece in thinkSplitter.flush() {
                    if piece.reasoning {
                        if !piece.text.isEmpty {
                            continuation.yield(LLMStreamEvent(
                                kind: .reasoning, content: piece.text))
                        }
                    } else {
                        for visible in toolSplitter.feed(piece.text) where !visible.isEmpty {
                            continuation.yield(LLMStreamEvent(kind: .content, content: visible))
                        }
                    }
                }
                for visible in toolSplitter.flush() where !visible.isEmpty {
                    continuation.yield(LLMStreamEvent(kind: .content, content: visible))
                }

                let toolCalls = LocalToolCallParser.extract(
                    raw: rawContent, allowedNames: allowedToolNames)
                let elapsed = max(0.001, Date().timeIntervalSince(generationStart))
                let usage: [String: Int] = [
                    "prompt_tokens": promptTokenCount,
                    "completion_tokens": generated.count,
                    "total_tokens": promptTokenCount + generated.count,
                    "tokens_per_second": Int(Double(generated.count) / elapsed),
                    "cached_prompt_tokens": cachedPromptTokens,
                ]
                AgentLog.info("[onnx] prompt=\(promptTokenCount) cached=\(cachedPromptTokens) "
                    + "generated=\(generated.count) reused=\(reused)")

                if isolated {
                    // 一次性请求不覆盖共享缓存；下一次会话请求仍走自己的缓存。
                } else if loaded.graph.canReusePast {
                    stateLock.lock()
                    cache = ONNXCache(tokens: tokens + generated, past: past,
                                      logits: logits, toolsKey: toolsKey)
                    stateLock.unlock()
                } else {
                    stateLock.lock()
                    cache = nil
                    stateLock.unlock()
                }
                continuation.yield(LLMStreamEvent(
                    kind: .done,
                    toolCalls: toolCalls,
                    usage: usage,
                    rawContent: rawContent))
            } catch {
                if !isolated {
                    stateLock.lock()
                    cache = nil
                    stateLock.unlock()
                }
                throw error
            }
    }

    private static let chunkSize = 512

    /// `add_generation_prompt` 追加的文本（去掉 assistant 角色标记），
    /// 并入 rawContent 让下一轮 token 前缀与缓存对齐。
    private static func generationPrefix(base: [Int], full: [Int],
                                         tokenizer: LocalTokenizer) -> String {
        guard base.count < full.count,
              Array(full.prefix(base.count)) == base else { return "" }
        var text = tokenizer.decode(Array(full.dropFirst(base.count)),
                                    skipSpecialTokens: true)
        for marker in ["<|im_start|>assistant", "<|assistant|>",
                       "<start_of_turn>model", "[/INST]"] {
            if let range = text.range(of: marker, options: .backwards) {
                text = String(text[range.upperBound...])
            }
        }
        return text
    }

    private static func firstIndex(of needle: [Int], in haystack: [Int]) -> Int? {
        guard !needle.isEmpty, haystack.count >= needle.count else { return nil }
        for index in 0...(haystack.count - needle.count)
        where Array(haystack[index..<(index + needle.count)]) == needle {
            return index
        }
        return nil
    }

    private static func toolsKey(_ tools: [JSONValue]) -> String {
        tools.map { $0.jsonString() }.joined(separator: "\n")
    }

    private static func mapError(_ error: Error, modelName: String) -> Error {
        if error is CancellationError { return AgentError.cancelled }
        if let agentError = error as? AgentError { return agentError }
        if let local = error as? LocalModelError { return AgentError.llm(local.localizedDescription) }
        if let onnx = error as? ONNXLLMError { return AgentError.llm(onnx.localizedDescription) }
        return AgentError.llm("ONNX 模型 \(modelName) 推理失败：\(error.localizedDescription)")
    }
}

// MARK: - 缓存

private struct ONNXCache {
    var tokens: [Int]
    var past: [String: OrtValue]
    var logits: [Float]
    var toolsKey: String
}

// MARK: - 模型加载

private final class ONNXLoaded: @unchecked Sendable {
    let session: OrtSession
    let tokenizer: LocalTokenizer
    let graph: ONNXDecoderGraph
    let eosTokens: Set<Int>

    private init(session: OrtSession, tokenizer: LocalTokenizer,
                 graph: ONNXDecoderGraph, eosTokens: Set<Int>) {
        self.session = session
        self.tokenizer = tokenizer
        self.graph = graph
        self.eosTokens = eosTokens
    }

    static func load(directory: URL,
                     tokenizerDirectory: URL?,
                     modelName: String) async throws -> ONNXLoaded {
        let modelURL = try LocalModelRegistry.resolveONNXModel(in: directory, kind: .llm)
        // 权重可能在 onnx/ 子目录而 tokenizer/config 在根目录，不能只看权重父目录。
        let modelDir = LocalModelRegistry.onnxSupportDirectory(
            modelURL: modelURL, modelRoot: directory)
        let tokenizer = try await LocalTokenizer.load(
            directory: tokenizerDirectory ?? modelDir)
        let session = try OrtSession(modelPath: modelURL.path)
        let graph = try ONNXDecoderGraph(session: session, directory: modelDir)
        var eos = Set<Int>()
        if let value = tokenizer.base.eosTokenId { eos.insert(value) }
        if let data = try? Data(contentsOf: modelDir.appendingPathComponent("generation_config.json")),
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let value = object["eos_token_id"] as? Int {
                eos.insert(value)
            } else if let values = object["eos_token_id"] as? [Int] {
                eos.formUnion(values)
            }
        }
        AgentLog.info("[onnx] 已加载 \(modelName)：\(modelURL.lastPathComponent) "
            + "inputs=\(session.inputNames.count) outputs=\(session.outputNames.count)")
        return ONNXLoaded(session: session, tokenizer: tokenizer,
                          graph: graph, eosTokens: eos)
    }
}

// MARK: - 图适配

private final class ONNXDecoderGraph {
    private let session: OrtSession
    private let inputIDs: String
    private let attentionMask: String?
    private let positionIDs: String?
    private let tokenTypeIDs: String?
    private let logitsOutput: String
    private let pastInputs: [String]
    private let presentByPast: [String: String]
    private let initialPastValues: [String: OrtValue]

    var canReusePast: Bool {
        !pastInputs.isEmpty && pastInputs.allSatisfy { presentByPast[$0] != nil }
    }

    init(session: OrtSession, directory: URL) throws {
        self.session = session
        guard let inputIDs = session.inputNames.first(where: {
            $0 == "input_ids" || $0.hasSuffix("input_ids")
        }) else {
            throw ONNXLLMError.unsupported(
                "ONNX 图没有 input_ids 输入（\(session.modelPath)）")
        }
        guard let logits = session.outputNames.first(where: {
            $0 == "logits" || $0.contains("logits")
        }) else {
            throw ONNXLLMError.unsupported(
                "ONNX 图没有 logits 输出（\(session.modelPath)）")
        }
        self.inputIDs = inputIDs
        self.logitsOutput = logits
        self.attentionMask = session.inputNames.first {
            $0 == "attention_mask" || $0.hasSuffix("attention_mask")
        }
        self.positionIDs = session.inputNames.first {
            $0 == "position_ids" || $0.hasSuffix("position_ids")
        }
        self.tokenTypeIDs = session.inputNames.first {
            $0 == "token_type_ids" || $0.hasSuffix("token_type_ids")
        }
        let past = session.inputNames.filter {
            $0.lowercased().contains("past_key_values") || $0.lowercased().hasPrefix("past")
        }
        let present = session.outputNames.filter {
            $0.lowercased().contains("present")
        }
        self.pastInputs = past
        var pairing: [String: String] = [:]
        for name in past {
            let suffix = Self.normalizedPastName(name)
            if let match = present.first(where: {
                Self.normalizedPastName($0) == suffix
            }) {
                pairing[name] = match
            }
        }
        self.presentByPast = pairing

        let architecture = ONNXLLMArchitecture(directory: directory)
        if !past.isEmpty {
            guard past.count == present.count else {
                throw ONNXLLMError.unsupported(
                    "ONNX past 输入 \(past.count) 个与 present 输出 \(present.count) 个不匹配")
            }
            guard let architecture else {
                throw ONNXLLMError.unsupported(
                    "ONNX 模型含 past_key_values，但缺少 config.json，无法推导空 KV 形状")
            }
            if past.contains(where: { $0.lowercased().contains("encoder") }) {
                throw ONNXLLMError.unsupported("暂不支持 ONNX encoder-decoder 模型")
            }
            let expected = architecture.layerCount * 2
            guard past.count == expected else {
                let names = past.prefix(4).joined(separator: ", ")
                throw ONNXLLMError.unsupported(
                    "暂不支持该 ONNX past 布局（\(past.count) 个输入，期望 \(expected) 个：\(names)…）")
            }
            var initial: [String: OrtValue] = [:]
            let shape = [1, architecture.kvHeads, 0, architecture.headDim]
            for name in past {
                initial[name] = try OrtValue(
                    float32: [Float](repeating: 0, count: 0), shape: shape)
            }
            self.initialPastValues = initial
        } else {
            self.initialPastValues = [:]
        }
    }

    func initialPast() throws -> [String: OrtValue] {
        initialPastValues
    }

    struct PrefillResult {
        var past: [String: OrtValue]
        var logits: [Float]
    }

    /// 从空 past 开始，分块 prefill。
    func prefill(tokens: [Int], initialPast: [String: OrtValue],
                 chunkSize: Int, contextLimit: Int) throws -> PrefillResult {
        var past = initialPast
        var logits: [Float] = []
        var index = 0
        let size = max(1, chunkSize)
        while index < tokens.count {
            let end = min(index + size, tokens.count)
            let chunk = Array(tokens[index..<end])
            logits = try run(chunk: chunk, past: &past,
                             pastLength: index, contextLimit: contextLimit)
            index = end
        }
        if tokens.isEmpty {
            // 空 prompt 不会被调用；这里只保留初始 past。
            logits = []
        }
        return PrefillResult(past: past, logits: logits)
    }

    /// 单个 token 增量解码，返回下一个 token 的 logits。
    func step(token: Int, past: inout [String: OrtValue],
              pastLength: inout Int, contextLimit: Int) throws -> [Float] {
        guard pastLength < contextLimit else {
            throw ONNXLLMError.decode("ONNX 上下文已满（\(contextLimit) tokens）")
        }
        let logits = try run(chunk: [token], past: &past,
                             pastLength: pastLength, contextLimit: contextLimit)
        pastLength += 1
        return logits
    }

    /// 无 past 图：每次重跑完整序列。
    func logits(tokens: [Int], contextLimit: Int) throws -> [Float] {
        guard tokens.count <= contextLimit else {
            throw ONNXLLMError.decode("ONNX 上下文已满（\(tokens.count)/\(contextLimit) tokens）")
        }
        var past: [String: OrtValue] = [:]
        return try run(chunk: tokens, past: &past, pastLength: 0,
                       contextLimit: contextLimit)
    }

    private func run(chunk: [Int], past: inout [String: OrtValue],
                     pastLength: Int, contextLimit: Int) throws -> [Float] {
        guard !chunk.isEmpty else { return [] }
        let length = pastLength + chunk.count
        guard length <= contextLimit else {
            throw ONNXLLMError.decode("ONNX 上下文已满（\(length)/\(contextLimit) tokens）")
        }
        var inputs: [String: OrtValue] = [:]
        inputs[inputIDs] = try OrtValue(
            int64: chunk.map(Int64.init), shape: [1, chunk.count])
        if let attentionMask {
            inputs[attentionMask] = try OrtValue(
                int64: Array(repeating: 1, count: length), shape: [1, length])
        }
        if let positionIDs {
            inputs[positionIDs] = try OrtValue(
                int64: (pastLength..<length).map(Int64.init), shape: [1, chunk.count])
        }
        if let tokenTypeIDs {
            inputs[tokenTypeIDs] = try OrtValue(
                int64: Array(repeating: 0, count: chunk.count), shape: [1, chunk.count])
        }
        for name in pastInputs {
            guard let value = past[name] else {
                throw ONNXLLMError.decode("缺少 past 张量 \(name)")
            }
            inputs[name] = value
        }

        let outputs = try session.run(inputs: inputs,
                                      outputNames: session.outputNames)
        guard let logitsValue = outputs[logitsOutput] else {
            throw ONNXLLMError.decode("ONNX 输出缺少 \(logitsOutput)")
        }
        let tensor = try logitsValue.floatTensor()
        guard let vocab = tensor.shape.last, vocab > 0 else {
            throw ONNXLLMError.decode("ONNX logits 形状异常：\(tensor.shape)")
        }
        let rows = tensor.values.count / vocab
        guard rows > 0 else {
            throw ONNXLLMError.decode("ONNX logits 为空")
        }
        let offset = (rows - 1) * vocab
        let row = Array(tensor.values[offset..<(offset + vocab)])

        if canReusePast {
            var next: [String: OrtValue] = [:]
            for name in pastInputs {
                guard let outputName = presentByPast[name],
                      let value = outputs[outputName] else {
                    throw ONNXLLMError.decode("ONNX 输出缺少 \(name) 对应的 present")
                }
                next[name] = value
            }
            past = next
        }
        return row
    }

    /// `past_key_values.0.key` 与 `present.0.key` 都归一化为 `.0.key`。
    private static func normalizedPastName(_ name: String) -> String {
        var value = name.lowercased()
        for prefix in ["past_key_values", "present", "past"] where value.hasPrefix(prefix) {
            value.removeFirst(prefix.count)
            break
        }
        return value
    }
}

// MARK: - 架构信息

private struct ONNXLLMArchitecture {
    var layerCount: Int
    var kvHeads: Int
    var headDim: Int

    init?(directory: URL) {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        let layers = (object["num_hidden_layers"] as? Int)
            ?? (object["num_layers"] as? Int)
            ?? (object["n_layer"] as? Int)
        let heads = (object["num_attention_heads"] as? Int)
            ?? (object["n_head"] as? Int)
        guard let layers, layers > 0, let heads, heads > 0 else { return nil }
        let kvHeads = (object["num_key_value_heads"] as? Int)
            ?? (object["n_head_kv"] as? Int)
            ?? heads
        let hidden = (object["hidden_size"] as? Int) ?? (object["n_embd"] as? Int) ?? 0
        let headDim = (object["head_dim"] as? Int)
            ?? (hidden > 0 ? hidden / heads : 0)
        guard kvHeads > 0, headDim > 0 else { return nil }
        self.layerCount = layers
        self.kvHeads = kvHeads
        self.headDim = headDim
    }
}

// MARK: - 采样

private final class ONNXSampler {
    private let config: LlamaCppSampling
    private var recent: [Int] = []

    init(config: LlamaCppSampling) {
        self.config = config
    }

    func accept(_ token: Int) {
        recent.append(token)
        if recent.count > 64 { recent.removeFirst(recent.count - 64) }
    }

    func sample(logits: [Float]) -> Int {
        guard !logits.isEmpty else { return 0 }
        var values = logits

        if config.repetitionPenalty > 1 {
            for token in recent where token >= 0 && token < values.count {
                if values[token] > 0 {
                    values[token] /= config.repetitionPenalty
                } else {
                    values[token] *= config.repetitionPenalty
                }
            }
        }

        if config.temperature <= 0 {
            var best = 0
            for index in values.indices where values[index] > values[best] {
                best = index
            }
            return best
        }

        let temperature = config.temperature
        for index in values.indices { values[index] /= temperature }
        let maxLogit = values.max() ?? 0
        var probabilities = [Float](repeating: 0, count: values.count)
        var total: Float = 0
        for index in values.indices {
            let value = exp(values[index] - maxLogit)
            probabilities[index] = value
            total += value
        }
        guard total > 0 else { return 0 }
        for index in probabilities.indices { probabilities[index] /= total }

        var order = Array(probabilities.indices)
        order.sort { probabilities[$0] > probabilities[$1] }
        let maxProbability = probabilities[order[0]]
        var limit = order.count
        if config.topK > 0 { limit = min(limit, config.topK) }
        if config.minP > 0 {
            var kept = 0
            for index in 0..<limit where probabilities[order[index]] >= config.minP * maxProbability {
                kept += 1
            }
            limit = max(1, kept)
        }
        if config.topP < 1 {
            var cumulative: Float = 0
            var kept = 0
            for index in 0..<limit {
                cumulative += probabilities[order[index]]
                kept += 1
                if cumulative >= config.topP { break }
            }
            limit = max(1, kept)
        }

        var mass: Float = 0
        for index in 0..<limit { mass += probabilities[order[index]] }
        guard mass > 0 else { return order[0] }
        var target = Float.random(in: 0..<mass)
        for index in 0..<limit {
            let candidate = order[index]
            target -= probabilities[candidate]
            if target <= 0 { return candidate }
        }
        return order[0]
    }
}
