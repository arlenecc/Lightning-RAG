import Foundation
import llama

/// 进程内 GGUF 主/辅 LLM。权重与 KV cache 常驻，不启动任何子进程。
public final class LlamaCppLLMService: LLMService, PreloadableLLMService,
                                       WarmupLLMService, ClosableLLMService,
                                       KVCacheInvalidating, @unchecked Sendable {
    public let identifier: String
    public let modelPath: String
    public let contextWindow: Int
    public let supportsTools = true

    private let maxTokens: Int
    private let topP: Double
    private let minP: Double
    private let topK: Int
    private let repetitionPenalty: Double
    private let enableThinking: Bool
    private let gpuLayers: Int
    /// 多模态投影文件（mmproj）。非空时以 mtmd 加载视觉能力，
    /// 聊天消息携带图像时走多模态 prefill。
    private let mmprojPath: String?
    /// 跨轮 KV/前缀复用开关（llm_kv_reuse）。关闭时每轮全量重建，
    /// 与 MLX 后端对齐。
    private let kvReuse: Bool
    /// KV cache int8 量化（ggml q8_0），KV 内存约减半。
    private let kvInt8: Bool
    /// 是否让 llama.cpp 强制启用 Flash Attention（可与 q8_0 KV 叠加）。
    private let flashAttention: Bool

    private let gate = AsyncGate()
    private let lock = NSLock()
    private var loaded: Loaded?
    private var loadFailure: Error?
    private var cachedTokens: [llama_token] = []
    private var cachedToolsKey: String?
    /// 上次成功预热的前缀签名。GGUF 无法像 MLX 那样复制 KV 快照，折叠后
    /// 必须重新 prefill 前缀；但缓存仍完好时的重复 warmup（配置切换/重复
    /// 调用）不该再算一遍整段 7k token。
    private var warmupSystemPrompt: String?
    private var warmupToolsKey: String?
    private var warmupTokens: [llama_token] = []
    /// `llama_memory_seq_rm` 是否可用。llama.h 明确：partial sequence 裁剪
    /// 在 hybrid/recurrent memory 上会返回 false（只有删除整个序列才保证
    /// 成功）。Qwen3.5 等混合 SSM 模型因此每轮生成后的 `trim` 都会走到
    /// 这里；旧实现直接整段重建（实测 7~8k token prefill 需 7~8s）。
    /// 现在失败后改走消息级增量 append（不重渲染 assistant、只追加新
    /// user/tool 增量），首次失败后记住能力，后续不再白试 seq_rm。
    private var seqRmSupported = true

    init(modelPath: String,
         contextWindow: Int,
         maxTokens: Int,
         topP: Double,
         minP: Double,
         topK: Int,
         repetitionPenalty: Double,
         enableThinking: Bool,
         gpuLayers: Int,
         mmprojPath: String? = nil,
         kvReuse: Bool = true,
         kvInt8: Bool = true,
         flashAttention: Bool = true) {
        self.modelPath = modelPath
        self.identifier = "llama.cpp:\((modelPath as NSString).lastPathComponent)"
        self.contextWindow = max(1_024, contextWindow)
        self.maxTokens = max(1, maxTokens)
        self.topP = topP
        self.minP = minP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.enableThinking = enableThinking
        self.gpuLayers = gpuLayers
        self.mmprojPath = mmprojPath
        self.kvReuse = kvReuse
        self.kvInt8 = kvInt8
        self.flashAttention = flashAttention
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
                    // 与 MLX 侧对齐：整轮被取消后不保留任何跨轮缓存，避免下一轮
                    // 从语义上已作废的会话前缀继续生成（取消可能发生在生成中途）。
                    if error is CancellationError {
                        invalidateKVCache()
                    }
                    continuation.finish(throwing: Self.mapError(error, modelPath: modelPath))
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    // MARK: - 加载

    public func preload() async throws {
        try await gate.withLock {
            lock.lock()
            defer { lock.unlock() }
            _ = try ensureLoadedLocked()
        }
    }

    public func close() {
        lock.lock()
        loaded = nil
        loadFailure = nil
        cachedTokens = []
        cachedToolsKey = nil
        warmupSystemPrompt = nil
        warmupToolsKey = nil
        warmupTokens = []
        // 换模型后要重新探测：旧实例是混合 SSM，新实例可能是标准 attention。
        seqRmSupported = true
        lock.unlock()
    }

    public func invalidateKVCache() {
        lock.lock()
        defer { lock.unlock() }
        cachedTokens = []
        cachedToolsKey = nil
        if let loaded, let memory = loaded.memory {
            llama_memory_clear(memory, true)
        }
    }

    private func ensureLoadedLocked() throws -> Loaded {
        if let loaded { return loaded }
        if let loadFailure { throw loadFailure }
        do {
            let value = try Loaded(modelPath: modelPath,
                                   contextWindow: contextWindow,
                                   gpuLayers: gpuLayers,
                                   mmprojPath: mmprojPath,
                                   kvInt8: kvInt8,
                                   flashAttention: flashAttention)
            loaded = value
            return value
        } catch {
            loadFailure = error
            throw error
        }
    }

    // MARK: - 预热

    /// 预填充 system + tools 前缀，不生成 token；正式请求可直接复用该段 KV。
    /// KV 复用关闭时预热无意义，直接跳过。
    public func warmup(systemPrompt: String, tools: [JSONValue]) async throws {
        guard kvReuse, !systemPrompt.isEmpty else { return }
        try await gate.withLock {
            try Task.checkCancellation()
            lock.lock()
            defer { lock.unlock() }

            let loaded = try ensureLoadedLocked()
            let toolsKey = Self.toolsKey(tools)
            // 前缀签名一致且缓存里仍完整保留该前缀 → 无需再 prefill。
            // 折叠/清空历史后 invalidation 会清掉 cachedTokens，此时条件
            // 不成立，warmup 必须重跑：GGUF 没有可复制的 KV 快照，重跑
            // 前缀正是让下一轮 mode=append 只算增量的唯一途径。
            if systemPrompt == warmupSystemPrompt,
               toolsKey == warmupToolsKey,
               !warmupTokens.isEmpty,
               cachedTokens.count >= warmupTokens.count,
               Array(cachedTokens.prefix(warmupTokens.count)) == warmupTokens {
                AgentLog.info("[gguf-warmup] 复用已就绪前缀 tokens=\(warmupTokens.count) "
                    + "（跳过全量 prefill）")
                return
            }
            cachedTokens = []
            cachedToolsKey = nil

            // 部分模板要求至少有一条 user 消息；多渲染一条占位 user，
            // system+tools 前缀仍然可复用，正式请求会裁掉占位尾巴。
            let marker = "__WARMUP_\(UUID().uuidString)__"
            let rendered: String
            do {
                rendered = try LlamaCppPromptRenderer.render(
                    template: loaded.model.chatTemplate,
                    model: loaded.model,
                    messages: [
                        ChatMessage(role: .system, content: systemPrompt),
                        ChatMessage(role: .user, content: marker),
                    ],
                    tools: tools,
                    enableThinking: enableThinking,
                    addGenerationPrompt: false)
            } catch {
                AgentLog.warning("[gguf-warmup] 模板不支持前缀预热，跳过："
                    + error.localizedDescription)
                return
            }
            // 只保留到占位 user 内容之前：system + tools + user 头。
            // 正式请求的第一条 user 消息天然以这段为前缀，可精确续写，
            // 对混合 SSM 模型也无需回滚 KV。
            guard let markerRange = rendered.range(of: marker) else {
                AgentLog.warning("[gguf-warmup] 渲染结果缺少占位标记，跳过")
                return
            }
            let prompt = String(rendered[..<markerRange.lowerBound])
            let tokens = try loaded.model.tokenize(prompt, addSpecial: true)
            guard tokens.count >= 32, tokens.count <= contextWindow - 1_024 else {
                AgentLog.warning("[gguf-warmup] 跳过：前缀 token 数 \(tokens.count) 不适合预热")
                return
            }
            llama_memory_clear(loaded.memory, true)
            try LlamaCppContext.decode(tokens: tokens,
                                       context: loaded.context,
                                       chunkSize: loaded.batchSize)
            cachedTokens = tokens
            cachedToolsKey = toolsKey
            warmupSystemPrompt = systemPrompt
            warmupToolsKey = toolsKey
            warmupTokens = tokens
            AgentLog.info("[gguf-warmup] prefill tokens=\(tokens.count) tools=\(tools.count)")
        }
    }

    // MARK: - 生成

    private func run(messages: [ChatMessage],
                     tools: [JSONValue],
                     temperature: Double,
                     enableThinking: Bool,
                     isolated: Bool,
                     continuation: AsyncThrowingStream<LLMStreamEvent, Error>.Continuation) async throws {
        try await gate.withLock {
            try Task.checkCancellation()
            lock.lock()
            defer { lock.unlock() }

            let loaded = try ensureLoadedLocked()
            let model = loaded.model

            // 多模态分支：消息携带图像且模型已加载 mmproj 时，走 mtmd 多模态 prefill。
            let hasImages = messages.contains { ($0.images?.isEmpty ?? true) == false }
            if hasImages {
                guard let vision = loaded.vision else {
                    throw LlamaCppError.unsupported(
                        "聊天消息包含图像，但 \(modelPath) 未配置 mmproj，无法处理视觉输入")
                }
                try runMultimodal(loaded: loaded, vision: vision, messages: messages,
                                  tools: tools, enableThinking: enableThinking,
                                  temperature: temperature, continuation: continuation)
                return
            }

            let basePrompt = try LlamaCppPromptRenderer.render(
                template: model.chatTemplate,
                model: model,
                messages: messages,
                tools: tools,
                enableThinking: enableThinking,
                addGenerationPrompt: false)
            let prompt = try LlamaCppPromptRenderer.render(
                template: model.chatTemplate,
                model: model,
                messages: messages,
                tools: tools,
                enableThinking: enableThinking,
                addGenerationPrompt: true)
            let tokens = try model.tokenize(prompt, addSpecial: true)
            let generationPrefix = LlamaCppPromptRenderer.generationPrefix(
                full: prompt, base: basePrompt)
            guard !tokens.isEmpty else {
                throw AgentError.llm("GGUF prompt 为空")
            }
            guard tokens.count <= contextWindow - 64 else {
                throw AgentError.llm(
                    "GGUF prompt \(tokens.count) tokens 超过上下文预算 \(contextWindow)")
            }

            let toolsKey = Self.toolsKey(tools)
            let reuseAllowed = kvReuse && !isolated
            if !reuseAllowed {
                // 关闭复用或一次性请求：丢弃上一轮缓存并整段重建，
                // 避免把其它分片/会话的 KV 当作当前 prompt 的前缀。
                llama_memory_clear(loaded.memory, true)
                cachedTokens = []
                cachedToolsKey = nil
            }
            let toolsMatch = reuseAllowed && cachedToolsKey == toolsKey
            var common = toolsMatch
                ? LlamaCppContext.commonPrefix(cachedTokens, tokens) : 0
            if common == tokens.count {
                common = max(0, tokens.count - 1)
            }
            // 形态标注（对齐 [mlx-kv]）：
            // append = 缓存是严格前缀（形态①）；trim = 裁到公共前缀
            // （形态②新 prompt 更短 / ③中途分叉）；rebuild = 全量重建。
            let mode: String
            if common == 0 {
                mode = "rebuild"
            } else if common >= cachedTokens.count {
                mode = "append"
            } else {
                mode = "trim"
            }
            AgentLog.info("[gguf-kv] mode=\(mode) common=\(common) "
                + "cached=\(cachedTokens.count) prompt=\(tokens.count) "
                + "tools_match=\(toolsMatch)")
            // 生成前缓存中实际存在的完整 token 序列；增量路径基于它续写。
            let prefixTokens = cachedTokens
            var deltaTokens: [llama_token]? = nil
            if mode == "trim" {
                var trimmed = false
                if seqRmSupported, let memory = loaded.memory {
                    let removed = llama_memory_seq_rm(memory, 0, llama_pos(common), -1)
                    trimmed = removed
                    if !removed {
                        // 混合 SSM 等结构不支持按位置裁剪 KV；记住能力，
                        // 后续不再白试，改走消息级增量。
                        AgentLog.warning("[gguf-kv] seq_rm(\(common)) 失败，"
                            + "改用消息级增量 append")
                        seqRmSupported = false
                    }
                }
                if !trimmed {
                    let stale = prefixTokens.count - common
                    if let delta = makeDeltaTokens(
                        messages: messages, loaded: loaded, tools: tools, stale: stale) {
                        deltaTokens = delta
                    } else {
                        common = 0
                        AgentLog.warning("[gguf-kv] 增量不可用，回退整段重建")
                    }
                }
            }
            if let delta = deltaTokens {
                // 不清内存：上一轮 assistant 的原始输出仍在缓存里，
                // 直接把新消息增量接在后面，等于形态① append。
                common = prefixTokens.count
                AgentLog.info("[gguf-kv] delta-append cached=\(common) "
                    + "delta=\(delta.count)（免整段重建）")
            } else if common == 0 {
                llama_memory_clear(loaded.memory, true)
            }

            let suffix = deltaTokens ?? Array(tokens[common...])
            let effectiveTokens = deltaTokens.map { prefixTokens + $0 } ?? tokens
            let prefillStart = Date()
            do {
                try LlamaCppContext.decode(tokens: suffix,
                                           context: loaded.context,
                                           chunkSize: loaded.batchSize)
            } catch {
                // prefill 中途失败/取消：内存已写入部分新后缀，但账目仍是旧前缀，
                // 不清会让下一轮前缀匹配撞上半截 KV（generateLoop 的失败路径
                // 已有同样清理，这里补上 prefill 段）。
                if let memory = loaded.memory {
                    llama_memory_clear(memory, true)
                }
                cachedTokens = []
                cachedToolsKey = nil
                throw error
            }
            let prefillMilliseconds = Int(Date().timeIntervalSince(prefillStart) * 1_000)
            let cachedPromptTokens = common

            // prompt 挤满上下文时生成预算趋近 0：继续生成只会产出空/半截
            // 正文，被内核判 retry 后反复空转（2026-11-08 cnn.com 事故）。
            // 抛 context 错误交给内核 resetForFreshContext 收缩历史后重试。
            guard contextWindow - effectiveTokens.count >= 200 else {
                throw AgentError.llm("GGUF 生成预算不足：prompt 已占 "
                    + "\(effectiveTokens.count)/\(contextWindow)（context window），"
                    + "需要收缩历史后重试")
            }
            let generationBudget = max(1, min(maxTokens,
                contextWindow - effectiveTokens.count - 8))
            let startInThink = prompt
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .hasSuffix("<think>")
            let allowedToolNames = Set(tools.compactMap {
                $0["function"]?["name"]?.stringValue
            })
            let (rawContent, generated): (String, [llama_token])
            // 计时起点必须在生成循环之前：旧实现把它放在 catch 之后，
            // elapsed 只剩工具解析的零点几毫秒，tps 被打成几万，性能日志
            // 完全失真，无法据此判断某轮是否真的变慢。
            let generationStart = Date()
            do {
                (rawContent, generated) = try generateLoop(
                    context: loaded.context, model: model,
                    temperature: temperature, startPast: llama_pos(effectiveTokens.count),
                    budget: generationBudget, startInThink: startInThink,
                    allowedToolNames: allowedToolNames, continuation: continuation)
            } catch {
                // prefill / 生成失败后缓存可能停在半截位置（生成循环内部已
                // 清过一次），这里统一清内存并重置账目，避免残留 KV 占内存、
                // 也避免下次前缀匹配撞上半截缓存。
                if let memory = loaded.memory {
                    llama_memory_clear(memory, true)
                }
                cachedTokens = []
                cachedToolsKey = nil
                throw error
            }

            let toolCalls = LocalToolCallParser.extract(
                raw: rawContent, allowedNames: allowedToolNames)
            let elapsed = max(0.001, Date().timeIntervalSince(generationStart))
            let usage: [String: Int] = [
                "prompt_tokens": effectiveTokens.count,
                "completion_tokens": generated.count,
                "total_tokens": effectiveTokens.count + generated.count,
                "tokens_per_second": Int(Double(generated.count) / elapsed),
                "cached_prompt_tokens": cachedPromptTokens,
            ]
            AgentLog.info("[gguf] prompt=\(effectiveTokens.count) cached=\(cachedPromptTokens) "
                + "generated=\(generated.count) prefill_ms=\(prefillMilliseconds) "
                + "tps=\(usage["tokens_per_second"] ?? 0)")

            if !isolated {
                cachedTokens = effectiveTokens + generated
                cachedToolsKey = toolsKey
            }
            continuation.yield(LLMStreamEvent(
                kind: .done,
                toolCalls: toolCalls,
                usage: usage,
                rawContent: rawContent))
        }
    }

    /// 混合 SSM / recurrent memory 无法按位置裁剪 KV（`llama_memory_seq_rm`
    /// 对 partial sequence 返回 false），旧路径只能整段重建 7~8k token。
    /// 这里改用消息级增量：缓存里已经完整保留上一轮 assistant 的原始输出
    /// （含 EOS），只需把本轮新追加的 user/tool 消息按同一 chat template
    /// 渲染成增量 token 接在后面，就能继续 `mode=append`，生成质量不变。
    ///
    /// 增量用「锚点差分」求：渲染 head（真实 system + 占位 user + 占位
    /// assistant）得 a，再渲染 head + 新消息得 b；a 去掉尾部空白后，b 剩下
    /// 的部分正是「消息尾 + 新消息 + generation prompt」。任何一步不匹配
    /// 都返回 nil，调用方回退到 seq_rm / 全量重建，不会破坏正确性。
    private func makeDeltaTokens(messages: [ChatMessage], loaded: Loaded,
                                 tools: [JSONValue], stale: Int) -> [llama_token]? {
        func reject(_ why: String, quiet: Bool = false) -> [llama_token]? {
            if !quiet { AgentLog.info("[gguf-kv] delta-reject \(why)") }
            return nil
        }
        // stale 是缓存里需要丢弃的尾部长度，应只是上一轮生成的几十~几百
        // token；大跨度改写说明历史被折叠/回滚，增量不可信。
        guard stale > 0, stale <= 1_024 else { return reject("stale=\(stale)") }
        guard messages.first?.role == .system else { return reject("no-system") }
        // 首轮还没有 assistant，本来就该走 rebuild，不必刷日志。
        guard let lastAssistant = messages.lastIndex(where: { $0.role == .assistant })
        else { return reject("no-assistant", quiet: true) }
        let delta = Array(messages[(lastAssistant + 1)...])
        guard !delta.isEmpty else { return reject("no-new-messages") }
        guard delta.allSatisfy({ $0.role == .user || $0.role == .tool })
        else { return reject("roles=" + delta.map(\.role.rawValue).joined(separator: ",")) }
        let model = loaded.model
        let sentinel = "__DELTA_\(UUID().uuidString)__"
        let head: [ChatMessage] = [
            messages[0],
            ChatMessage(role: .user, content: sentinel + "_u1"),
            ChatMessage(role: .assistant, content: "A"),
            // Qwen3.5 模板只给「最后一条 assistant」加 <think>…</think> 块。
            // 让占位 assistant 在 a / b 里都不是最后一条，两次渲染的 head
            // 才会逐字节相同；末尾这条占位 user 属于锚点，不会进增量。
            ChatMessage(role: .user, content: sentinel + "_u2"),
        ]
        do {
            let a = try LlamaCppPromptRenderer.render(
                template: model.chatTemplate, model: model, messages: head,
                tools: tools, enableThinking: enableThinking,
                addGenerationPrompt: false)
            let b = try LlamaCppPromptRenderer.render(
                template: model.chatTemplate, model: model, messages: head + delta,
                tools: tools, enableThinking: enableThinking,
                addGenerationPrompt: true)
            // a 末尾的换行是模板给 assistant 消息加的，但模型生成到
            // EOS(<|im_end|>) 就停，缓存里没有它；把它留给增量才能严格拼接。
            let anchor = a.replacingOccurrences(
                of: "\\s+$", with: "", options: .regularExpression)
            guard !anchor.isEmpty, b.hasPrefix(anchor)
            else { return reject("template-prefix-mismatch") }
            let tail = String(b.dropFirst(anchor.count))
            guard !tail.isEmpty else { return reject("empty-tail") }
            let ids = try model.tokenize(tail, addSpecial: false)
            return ids.isEmpty ? reject("empty-tokens") : ids
        } catch {
            return reject("render-error=\(error.localizedDescription)")
        }
    }

    // MARK: - 生成 / 多模态

    /// 自回归生成循环（文本与多模态 prefill 之后共用）。
    /// 从 `startPast` 位置继续 decode，返回拼接后的原文与生成的 token 序列。
    private func generateLoop(
        context: OpaquePointer,
        model: LlamaCppModel,
        temperature: Double,
        startPast: llama_pos,
        budget: Int,
        startInThink: Bool,
        allowedToolNames: Set<String>,
        continuation: AsyncThrowingStream<LLMStreamEvent, Error>.Continuation
    ) throws -> (rawContent: String, generated: [llama_token]) {
        let sampling = LlamaCppSampling(
            temperature: temperature, topK: topK, topP: topP,
            minP: minP, repetitionPenalty: repetitionPenalty)
        let sampler = LlamaCppSampler(vocabSize: model.vocabSize, config: sampling)
        var thinkSplitter = ThinkSplitter(startInThink: startInThink)
        var toolSplitter = ToolTagSplitter()
        var decoder = LlamaCppUTF8StreamDecoder()
        var rawContent = ""
        var generated: [llama_token] = []
        var nPast = startPast

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

        do {
            while generated.count < budget {
                try Task.checkCancellation()
                if Int(nPast) >= contextWindow { break }
                let token = sampler.sample(context: context)
                sampler.accept(token)
                generated.append(token)
                try LlamaCppContext.decode(token: token, context: context)
                nPast += 1
                if model.isEndOfGeneration(token) { break }

                let bytes = model.pieceBytes(token)
                if !bytes.isEmpty {
                    let text = decoder.feed(bytes)
                    if !text.isEmpty { emit(text) }
                }
            }

            let tail = decoder.flush()
            if !tail.isEmpty { emit(tail) }
            for piece in thinkSplitter.flush() {
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
            for visible in toolSplitter.flush() where !visible.isEmpty {
                continuation.yield(LLMStreamEvent(kind: .content, content: visible))
            }
        } catch {
            llama_memory_clear(llama_get_memory(context), true)
            throw error
        }
        return (rawContent, generated)
    }

    /// 多模态 prefill：把带图消息里的图像占位符经 mtmd 注入视觉特征，
    /// 再用 mtmd 分词 + 评估写入 KV，之后交给通用生成循环续写。
    /// 多模态请求禁用跨轮 KV 复用（mtmd 的视觉 prefill 与纯文本缓存不兼容）。
    private func runMultimodal(
        loaded: Loaded,
        vision: OpaquePointer,
        messages: [ChatMessage],
        tools: [JSONValue],
        enableThinking: Bool,
        temperature: Double,
        continuation: AsyncThrowingStream<LLMStreamEvent, Error>.Continuation
    ) throws {
        let model = loaded.model
        let marker = String(cString: mtmd_default_marker())

        // 在每条带图用户消息前插入与「成功解码的图像数」一致的占位符，并记录
        // bitmap。解码失败的图像连同占位符一起跳过——mtmd_tokenize 要求
        // 文本中的媒体标记数量与 bitmap 数组严格一致。
        var bitmaps: [OpaquePointer] = []
        defer { for bitmap in bitmaps { mtmd_bitmap_free(bitmap) } }
        var decodeFailures = 0
        let injected = messages.map { message -> ChatMessage in
            guard let images = message.images, !images.isEmpty else { return message }
            var markers = ""
            for data in images {
                let bitmap = data.withUnsafeBytes { buffer -> OpaquePointer? in
                    guard let base = buffer.bindMemory(to: UInt8.self).baseAddress else { return nil }
                    let opt = mtmd_helper_init_opt_default()
                    return mtmd_helper_bitmap_init_from_buf(
                        vision, base, buffer.count, false, opt).bitmap
                }
                if let bitmap {
                    bitmaps.append(bitmap)
                    markers += marker + "\n"
                } else {
                    decodeFailures += 1
                }
            }
            guard !markers.isEmpty else { return message }
            var copy = message
            copy.content = markers + message.content
            return copy
        }
        if decodeFailures > 0 {
            AgentLog.warning("[gguf-vlm] \(decodeFailures) 张图像解码失败，已跳过")
        }
        guard !bitmaps.isEmpty else {
            throw LlamaCppError.decode("图像解码失败（mtmd 未生成 bitmap）")
        }

        let formatted = try LlamaCppPromptRenderer.render(
            template: model.chatTemplate, model: model, messages: injected,
            tools: tools, enableThinking: enableThinking, addGenerationPrompt: true)
        guard formatted.contains(marker) else {
            throw LlamaCppError.template("chat template 丢失了媒体占位符 \(marker)")
        }

        llama_memory_clear(loaded.memory, true)
        guard let chunks = mtmd_input_chunks_init() else {
            throw LlamaCppError.decode("mtmd_input_chunks_init 失败")
        }
        defer { mtmd_input_chunks_free(chunks) }

        var text = mtmd_input_text(text: nil, text_len: 0, add_special: true, parse_special: true)
        var bitmapsArray = bitmaps.map { $0 as OpaquePointer? }
        let tokenizeResult: Int32 = formatted.withCString { pointer in
            text.text = pointer
            text.text_len = formatted.utf8.count
            return bitmapsArray.withUnsafeBufferPointer { buffer in
                mtmd_tokenize(vision, chunks, &text, buffer.baseAddress!, bitmapsArray.count)
            }
        }
        guard tokenizeResult == 0 else {
            throw LlamaCppError.decode("mtmd_tokenize 返回 \(tokenizeResult)")
        }

        var nPast: llama_pos = 0
        let evalResult = mtmd_helper_eval_chunks(
            vision, loaded.context, chunks, 0, 0, Int32(loaded.batchSize), true, &nPast)
        guard evalResult == 0 else {
            throw LlamaCppError.decode("mtmd 图像编码失败（\(evalResult)）")
        }

        let budget = max(1, min(maxTokens, contextWindow - Int(nPast) - 8))
        let allowedToolNames = Set(tools.compactMap {
            $0["function"]?["name"]?.stringValue
        })
        // 与文本路径一致：thinking 模板的渲染结果以 <think> 结尾时，
        // 生成内容从思考开始，避免推理内容漏进正文。
        let startInThink = formatted
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .hasSuffix("<think>")
        let generationStart = Date()
        let (rawContent, generated) = try generateLoop(
            context: loaded.context, model: model, temperature: temperature,
            startPast: nPast, budget: budget, startInThink: startInThink,
            allowedToolNames: allowedToolNames, continuation: continuation)

        let toolCalls = LocalToolCallParser.extract(
            raw: rawContent, allowedNames: allowedToolNames)
        let elapsed = max(0.001, Date().timeIntervalSince(generationStart))
        let usage: [String: Int] = [
            "prompt_tokens": Int(nPast),
            "completion_tokens": generated.count,
            "total_tokens": Int(nPast) + generated.count,
            "tokens_per_second": Int(Double(generated.count) / elapsed),
            "cached_prompt_tokens": 0,
        ]
        AgentLog.info("[gguf-vlm] prefill=\(nPast) images=\(bitmaps.count) "
            + "think=\(startInThink) generated=\(generated.count)")
        continuation.yield(LLMStreamEvent(
            kind: .done, toolCalls: toolCalls, usage: usage, rawContent: rawContent))
    }

    // MARK: - 状态

    private final class Loaded {
        let model: LlamaCppModel
        let context: OpaquePointer
        let memory: OpaquePointer?
        let batchSize: Int
        /// mtmd 多模态上下文（mmproj 投影）。无 mmproj 时为 nil。
        let vision: OpaquePointer?

        init(modelPath: String, contextWindow: Int, gpuLayers: Int,
             mmprojPath: String?, kvInt8: Bool, flashAttention: Bool) throws {
            model = try LlamaCppModel(path: modelPath, gpuLayers: gpuLayers)
            batchSize = min(2_048, max(64, contextWindow))
            context = try LlamaCppContext.make(
                model: model,
                contextWindow: contextWindow,
                batchSize: batchSize,
                embeddings: false,
                kvInt8: kvInt8,
                flashAttention: flashAttention)
            memory = llama_get_memory(context)
            let modelPointer = model.raw

            if let mmprojPath, !mmprojPath.isEmpty {
                var params = mtmd_context_params_default()
                params.use_gpu = llama_supports_gpu_offload()
                params.print_timings = false
                params.warmup = false
                params.n_threads = Int32(max(4, min(8,
                    ProcessInfo.processInfo.activeProcessorCount - 2)))
                guard let created = mmprojPath.withCString({ path in
                    mtmd_init_from_file(path, modelPointer, params)
                }) else {
                    llama_free(context)
                    throw LlamaCppError.mmproj(
                        "mtmd_init_from_file 返回空：\(mmprojPath)")
                }
                vision = created
            } else {
                vision = nil
            }
        }

        deinit {
            if let vision { mtmd_free(vision) }
            llama_free(context)
        }
    }

    private static func toolsKey(_ tools: [JSONValue]) -> String {
        tools.map { $0.jsonString() }.joined(separator: "\n")
    }

    private static func mapError(_ error: Error, modelPath: String) -> Error {
        if error is CancellationError { return AgentError.cancelled }
        if let agentError = error as? AgentError { return agentError }
        return AgentError.llm(
            "本地 GGUF 模型 \((modelPath as NSString).lastPathComponent) 推理失败："
            + error.localizedDescription)
    }
}
