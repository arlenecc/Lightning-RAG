import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// 本地 MLX LLM 服务（Qwen3 系）。
///
/// 通过 `ChatSession` 持有跨轮 KV cache：新一轮只把新增消息喂给会话，
/// 历史被折叠/回滚时由内核调用 `invalidateKVCache()` 丢弃缓存。
/// 模型权重通过 `ModelContainer` 单例复用；调用串行由 Agent 保证。
public final class MLXLLMService: LLMService, PreloadableLLMService,
    WarmupLLMService, ClosableLLMService, KVCacheInvalidating,
    AsyncUnloadableLLMService, @unchecked Sendable {
    public let identifier: String
    public let supportsTools = true
    public let contextWindow: Int
    /// `isolated` 请求会创建共享权重、独立 KV 的轻量副本，可安全并行。
    public var supportsConcurrentIsolatedRequests: Bool { true }
    public let directory: URL
    public let modelName: String

    private let maxTokens: Int
    private let topP: Float
    private let minP: Float
    private let topK: Int
    private let repetitionPenalty: Float
    private let enableThinking: Bool
    /// prefill 每次前向的 token 上限；0 表示使用 MLX 默认值。
    private let promptBatchSize: Int
    /// 跨轮 KV/前缀复用开关（llm_kv_reuse）。关闭时每轮全量重建，
    /// 与 GGUF 后端对齐。
    private let kvReuse: Bool
    /// KV cache int8 量化（QuantizedKVCache kvBits=8），KV 内存约减半。
    private let kvInt8: Bool
    /// 模型容器共享给同一模型的轻量副本；容器内是只读权重。
    private let state: LLMLoadState
    /// 跨轮 KV cache 与模型状态。MLX 调用由 Agent 串行驱动。
    private var rawCache: [KVCache]?
    private var rawState: LMOutput.State?
    /// 缓存中实际已处理的 token 数（含生成部分）；MambaCache 的 offset 不增长，
    /// 不能直接读 cache.first.offset。
    private var cachedTokenCount = 0
    /// 缓存当前表示的完整 token 序列（prompt + 生成），用于下一轮 token 级
    /// 严格前缀匹配。仅在缓存可裁剪（isTrimmable）时有意义；不可裁剪的
    /// 混合缓存（Qwen3.5 MambaCache）走消息级匹配，此数组保持为空。
    private var cachedTokens: [Int] = []
    /// 已经消费/生成的完整消息轨迹（含内核折叠掉的中间工具步）。
    private var sessionTranscript: [ChatMessage] = []
    /// 上一次生成是否产出了一个尚未在外部消息里消费的 assistant 消息。
    private var pendingGeneratedAssistant = false
    /// 缓存前缀对应的 tools 指纹；工具清单变化时必须重建，否则 system 前缀不一致。
    private var cachedToolsKey: String?
    /// 稳定 system + tools 前缀的原始快照。历史折叠/滑窗导致必须重建时，
    /// 从这里复制前缀，只 prefill 动态历史，避免重新计算 4~5k token。
    private var warmupCacheSnapshot: [KVCache]?
    private var warmupStateSnapshot: LMOutput.State?
    private var warmupTokens: [Int] = []
    private var warmupToolsKey: String?
    /// 上次成功预热对应的 system prompt。折叠边界会用新冻结的 prompt 再次
    /// 请求预热；逐字节相同时只需丢弃会话 KV，前缀快照可直接给下一轮的
    /// prefix-reuse 使用，不必重跑 7k token 的 GPU prefill（实测 18~21s）。
    private var warmupSystemPrompt: String?
    /// 正在进行的预热任务。启动期间发出的请求先等它完成，就能直接复用刚
    /// prefill 好的前缀；否则并发 prefill 会踩踏 GPU 并让首轮全量重建。
    private let warmupLock = NSLock()
    private var activeWarmup: Task<Void, Error>?
    /// 请求级 thinking 覆盖用的持久副本（按 thinking 值缓存）。
    /// 旧实现每次 options 调用都新建副本且 `kvReuse=false`：连续 3 次空正文
    /// 重试就是 3 次全量 prefill（4B 模型真机实测约 80s/次）。副本与主服务
    /// 各自持有独立 KV/轨迹，缓存它不会污染常驻会话。
    private let thinkingCopyLock = NSLock()
    private var thinkingCopies: [Bool: MLXLLMService] = [:]

    private func setActiveWarmup(_ task: Task<Void, Error>?) {
        warmupLock.lock(); activeWarmup = task; warmupLock.unlock()
    }

    private func currentWarmup() -> Task<Void, Error>? {
        warmupLock.lock(); defer { warmupLock.unlock() }
        return activeWarmup
    }

    public init(
        directory: URL,
        modelName: String,
        maxTokens: Int,
        topP: Double = 0.95,
        minP: Double = 0.05,
        topK: Int = 20,
        repetitionPenalty: Double = 1.0,
        enableThinking: Bool = true,
        contextWindow: Int = 32768,
        promptBatchSize: Int = 0,
        kvReuse: Bool = true,
        kvInt8: Bool = true
    ) {
        self.directory = directory
        self.modelName = modelName
        self.identifier = "local:mlx:\(modelName)"
        self.maxTokens = max(1, maxTokens)
        self.contextWindow = max(1024, contextWindow)
        self.topP = Float(max(0, min(1, topP)))
        self.minP = Float(max(0, min(1, minP)))
        self.topK = max(0, topK)
        self.repetitionPenalty = Float(max(0, repetitionPenalty))
        self.enableThinking = enableThinking
        self.promptBatchSize = max(0, promptBatchSize)
        self.kvReuse = kvReuse
        self.kvInt8 = kvInt8
        self.state = LLMLoadState()
    }

    /// 共享已加载权重的轻量副本：独立持有 KV/session，用于并行的一次性请求。
    ///
    /// 只有 `state` 共享；`rawCache`、会话轨迹等仍各自独立，因此不会互相污染。
    private init(sharing other: MLXLLMService, enableThinking: Bool,
                 kvReuse: Bool = false) {
        self.directory = other.directory
        self.modelName = other.modelName
        self.identifier = other.identifier
        self.maxTokens = other.maxTokens
        self.topP = other.topP
        self.minP = other.minP
        self.topK = other.topK
        self.repetitionPenalty = other.repetitionPenalty
        self.enableThinking = enableThinking
        self.contextWindow = other.contextWindow
        self.promptBatchSize = other.promptBatchSize
        // 一次性并行请求不能复用其它分片的 KV；持久的 thinking 副本则允许
        // 在自己独立的轨迹上跨轮复用。
        self.kvReuse = kvReuse
        self.kvInt8 = other.kvInt8
        self.state = other.state
    }

    // MARK: - LLMService

    public func chatStream(
        messages: [ChatMessage],
        tools: [JSONValue],
        temperature: Double
    ) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let directory = self.directory
            let enableThinking = self.enableThinking
            let maxTokens = self.maxTokens
            let topP = self.topP
            let minP = self.minP
            let topK = self.topK
            let repetitionPenalty = self.repetitionPenalty
            let modelName = self.modelName
            let contextWindow = self.contextWindow
            let kvReuse = self.kvReuse
            let kvInt8 = self.kvInt8
            let state = self.state
            // 文本兜底只接受本请求声明过的工具名，避免模型把普通 JSON
            // （例如回答里的 {"name": ...}）误判成工具调用。
            let allowedToolNames = Set(tools.compactMap {
                $0["function"]?["name"]?.stringValue
            })

            // 取消时直接终止底层 MLX 生成，避免用户点「终止」后还要等下一个 token。
            let innerTask = InnerTaskBox()
            let task = Task {
                // 生成中断抢救记账：截断/取消发生时，prompt 段的 KV 仍然
                // 有效（内核重试只在历史末尾追加纠错提示）。可裁剪缓存
                // 裁掉已生成部分保留 prompt，重试的 prefill 只剩增量；
                // 不可裁剪缓存（Qwen3.5 MambaCache）无法撤销生成段，
                // 只能整体丢弃（维持旧行为）。
                var activeCache: [KVCache]?
                var activePromptTokens = 0
                var activePromptTokenIDs: [Int]?
                var activeTranscript: [ChatMessage] = []
                var activeToolsKey: String?
                var activeIterator: StatefulTokenIterator?
                do {
                    try Task.checkCancellation()
                    // 启动期间发出的请求排队等预热完成：预热已把 system+tools
                    // 前缀算好，这里就能直接增量续写，不必全量 prefill 20s。
                    if let warmup = self.currentWarmup() {
                        try? await warmup.value
                    }
                    try Task.checkCancellation()
                    let container = try await state.container(directory: directory)
                    let probe = try await state.templateProbe(
                        container: container,
                        enableThinking: enableThinking
                    )

                    let mlxMessages = Self.makeChatMessages(messages)
                    let toolSpecs = Self.makeToolSpecs(tools)
                    let toolsKey = Self.toolsKey(tools)
                    let thinkingContext: [String: any Sendable] = [
                        "enable_thinking": enableThinking
                    ]

                    var transcriptAfter: [ChatMessage] = []
                    var deltaMlx: [Chat.Message] = []
                    var hasDelta = false
                    var common = 0
                    var rebuild = !kvReuse || rawCache == nil || sessionTranscript.isEmpty
                    var rebuildReason = !kvReuse ? "kv-reuse-disabled"
                        : rawCache == nil ? "no-cache" : "no-transcript"

                    var prepared: LMInput?
                    var totalPromptTokens = 0
                    var cachedPromptTokens = 0
                    var reuseCache = false
                    var prefixReuse = false

                    // ---- Token 级前缀匹配（可裁剪缓存：形态①②③统一） ----
                    // 把新 prompt 渲染成完整 token 序列，与缓存当前表示的序列
                    // （prompt + 生成）做严格前缀对比：缓存是新 prompt 的前缀 →
                    // 只预填增量（①，常态多轮追加）；新 prompt 更短（回退/收缩
                    // 重试）或中途分叉（动态上下文改写历史）→ 裁到公共前缀复用，
                    // 只重算其后的一小段（②③）。相比消息级匹配，assistant 段
                    // 逐 token 验证，不再依赖「生成内容已在缓存」的消息级假设。
                    // 不可裁剪的混合缓存（Qwen3.5 MambaCache 等）走下方消息级匹配。
                    var tokenReused = false
                    var prerenderedInput: LMInput?
                    var prerenderedTokens: [Int] = []
                    // 提交缓存时记录「本轮实际喂入的完整 prompt token 序列」，
                    // 供下一轮 token 级前缀匹配；消息级 delta 路径拿不到完整
                    // 序列，置空（该路径只用于不可裁剪缓存，序列本来就不参与）。
                    var promptTokensForCache: [Int]? = nil
                    if kvReuse, rawState == nil,
                       let existing = rawCache, !existing.isEmpty,
                       cachedToolsKey == toolsKey, !cachedTokens.isEmpty,
                       existing.allSatisfy({ $0.isTrimmable }) {
                        do {
                            let full = try await container.prepare(input: UserInput(
                                chat: mlxMessages, tools: toolSpecs,
                                additionalContext: thinkingContext))
                            let tokens = full.text.tokens.asArray(Int.self)
                            prerenderedInput = full
                            prerenderedTokens = tokens
                            let commonTokens = Self.commonTokenPrefix(tokens, cachedTokens)
                            // 完全相等 = 没有新消息，放弃复用走重建（保持既有语义）。
                            if commonTokens > 0, commonTokens < tokens.count {
                                let stale = cachedTokens.count - commonTokens
                                if stale > 0 { _ = existing.map { $0.trim(stale) } }
                                let reuseLen = existing.first?.offset ?? 0
                                // 裁剪结果与预期不符（窗口淘汰等）→ 弃用走重建。
                                if reuseLen == commonTokens,
                                   tokens.count <= contextWindow - 512 {
                                    prepared = LMInput(tokens: full.text.tokens[reuseLen...])
                                    totalPromptTokens = tokens.count
                                    cachedPromptTokens = reuseLen
                                    reuseCache = true
                                    tokenReused = true
                                    promptTokensForCache = tokens
                                    // 轨迹与消息列表同步：回退消息级匹配时前缀
                                    // 仍然成立，不会误判 unexpected-assistant。
                                    transcriptAfter = messages
                                    common = reuseLen
                                    AgentLog.info("[mlx-kv] token-reuse cached=\(reuseLen) "
                                        + "delta=\(tokens.count - reuseLen) "
                                        + "stale=\(stale)")
                                } else {
                                    AgentLog.info("[mlx-kv] token-trim-rejected "
                                        + "expect=\(commonTokens) offset=\(reuseLen)")
                                }
                            }
                        } catch {
                            // 渲染失败：回退消息级匹配/重建。
                            prerenderedInput = nil
                            prerenderedTokens = []
                        }
                    }

                    // ---- 消息级匹配（不可裁剪缓存的形态①） ----
                    // 判断能否在已有 raw KV cache 上继续追加。只追加 user/tool
                    // 消息：assistant 已经由上一轮生成并存在于缓存里。
                    if !tokenReused {
                    // 工具清单参与 template 的 system 前缀渲染；变化后不能复用旧缓存。
                    if !rebuild, let cachedToolsKey, cachedToolsKey != toolsKey {
                        rebuild = true
                        rebuildReason = "tools-changed"
                    }
                    if !rebuild {
                        common = Self.commonPrefixLength(messages, sessionTranscript)
                        if common < min(messages.count, sessionTranscript.count) {
                            // 前缀在轨迹中段分叉：本应走 delta 追加的调用被迫
                            // 全量重建。输出分叉点两侧消息便于定位改写历史的调用方。
                            let a = messages[common], b = sessionTranscript[common]
                            AgentLog.warning("[mlx-kv] prefix-mismatch at \(common): "
                                + "msg=\(a.role.rawValue)|\(String(a.content.prefix(120))) "
                                + "vs cached=\(b.role.rawValue)|\(String(b.content.prefix(120)))")
                        }
                        if common == 0 {
                            rebuild = true
                            rebuildReason = "no-common-prefix"
                        } else {
                            transcriptAfter = sessionTranscript
                            var feedStart = common
                            if pendingGeneratedAssistant,
                               common < messages.count,
                               messages[common].role == .assistant {
                                transcriptAfter.append(messages[common])
                                feedStart = common + 1
                            }
                            if feedStart < messages.count {
                                let rest = Array(messages[feedStart...])
                                // 出现未缓存的 assistant 说明轨迹错位；直接重建，
                                // 否则后续 tool 结果会失去对应的调用消息。
                                if rest.contains(where: { $0.role == .assistant }) {
                                    rebuild = true
                                    rebuildReason = "unexpected-assistant"
                                } else if let rewritten = Self.makeDeltaMessages(
                                    Array(mlxMessages[feedStart...])), !rewritten.isEmpty {
                                    deltaMlx = rewritten
                                    hasDelta = true
                                    transcriptAfter.append(contentsOf: rest)
                                    if cachedTokenCount + Self.estimateTranscript(rest)
                                        > contextWindow - 512 {
                                        rebuild = true
                                        rebuildReason = "context-limit"
                                    }
                                } else {
                                    rebuild = true
                                    rebuildReason = "empty-delta"
                                }
                            } else {
                                rebuild = true
                                rebuildReason = "no-new-messages"
                            }
                        }
                    }
                    }

                    let model = await container.perform {
                        UncheckedSendableBox($0.model)
                    }.value
                    let tokenizer = await container.tokenizer
                    let modelConfiguration = await container.configuration

                    if !rebuild, hasDelta, !tokenReused {
                        do {
                            let deltaInput = try await container.prepare(input: UserInput(
                                chat: deltaMlx,
                                tools: nil,
                                additionalContext: thinkingContext))
                            let deltaTokens = deltaInput.text.tokens.size
                            let cacheTokens = cachedTokenCount
                            if cacheTokens + deltaTokens <= contextWindow - 512 {
                                prepared = deltaInput
                                totalPromptTokens = cacheTokens + deltaTokens
                                cachedPromptTokens = cacheTokens
                                reuseCache = true
                            }
                        } catch {
                            reuseCache = false
                        }
                    }
                    if prepared == nil || !reuseCache {
                        // token 级匹配已渲染过完整输入时直接复用，避免二次渲染。
                        let fullInput: LMInput
                        let fullTokens: [Int]
                        if let pre = prerenderedInput, !prerenderedTokens.isEmpty,
                           pre.text.tokens.size == prerenderedTokens.count {
                            fullInput = pre
                            fullTokens = prerenderedTokens
                        } else {
                            let rendered = try await container.prepare(input: UserInput(
                                chat: mlxMessages,
                                tools: toolSpecs,
                                additionalContext: thinkingContext))
                            fullInput = rendered
                            fullTokens = rendered.text.tokens.asArray(Int.self)
                        }
                        transcriptAfter = messages
                        totalPromptTokens = fullTokens.count
                        promptTokensForCache = fullTokens
                        // 硬性上下文保护：宁可本轮明确报错，也不让 KV 无限增长。
                        guard fullTokens.count <= contextWindow - 512 else {
                            throw AgentError.llm(
                                "prompt \(fullTokens.count) tokens 超过上下文预算 \(contextWindow)")
                        }
                        cachedPromptTokens = 0
                        // 历史折叠/滑窗会改写消息前缀，但 system+tools 前缀没变。
                        // 命中快照时只 prefill 动态历史，避免再次花几十秒算 4~5k token。
                        if warmupCacheSnapshot != nil,
                           warmupToolsKey == toolsKey,
                           !warmupTokens.isEmpty,
                           fullTokens.count > warmupTokens.count,
                           Array(fullTokens.prefix(warmupTokens.count)) == warmupTokens {
                            prefixReuse = true
                            cachedPromptTokens = warmupTokens.count
                            prepared = LMInput(
                                tokens: fullInput.text.tokens[warmupTokens.count...])
                            AgentLog.info("[mlx-kv] prefix-reuse cached=\(warmupTokens.count) "
                                + "delta=\(fullTokens.count - warmupTokens.count)")
                        } else {
                            prepared = fullInput
                            // 真正的全量重建才报 rebuild；warmup 快照命中时只
                            // 预填动态历史，不算重建。
                            AgentLog.info("[mlx-kv] rebuild reason=\(rebuildReason) "
                                + "messages=\(messages.count) common=\(common) "
                                + "cached=\(cachedTokenCount) transcript=\(sessionTranscript.count)")
                        }
                    }
                    guard let input = prepared else {
                        throw AgentError.llm("本地输入准备失败")
                    }

                    // 生成预算不足时抛 context 错误交给内核收缩历史，
                    // 而不是用 max(256,…) 硬撑：KV 越界 + 空正文反复重试。
                    let remainingContext = contextWindow - totalPromptTokens - 64
                    guard remainingContext >= 256 else {
                        throw AgentError.llm("MLX 生成预算不足：prompt 已占 "
                            + "\(totalPromptTokens)/\(contextWindow)（context window），"
                            + "需要收缩历史后重试")
                    }
                    let tokenBudget = remainingContext
                    let parameters = Self.parameters(
                        maxTokens: min(maxTokens, tokenBudget),
                        temperature: temperature, topP: topP, topK: topK,
                        minP: minP, repetitionPenalty: repetitionPenalty,
                        promptBatchSize: promptBatchSize,
                        kvInt8: kvInt8)

                    let cache: [KVCache]
                    let state: LMOutput.State?
                    if prefixReuse, let snapshot = warmupCacheSnapshot {
                        // 每次重建都从原始快照复制，不能推进快照本身。
                        cache = snapshot.map { $0.copy() }
                        state = warmupStateSnapshot
                    } else if reuseCache, let existing = rawCache {
                        cache = existing
                        state = rawState
                    } else {
                        cache = Self.stabilizeQuantizedCache(
                            try model.newCache(parameters: parameters),
                            kvInt8: kvInt8)
                        state = nil
                    }

                    let iterator = try TokenIterator(
                        input: input, model: model, cache: cache, state: state,
                        parameters: parameters)
                    let wrapped = StatefulTokenIterator(iterator: iterator)
                    // 预填充后、生成前的状态，供下一次调用接续。
                    rawState = wrapped.state
                    activeCache = cache
                    activePromptTokens = totalPromptTokens
                    activePromptTokenIDs = promptTokensForCache
                    activeTranscript = transcriptAfter
                    activeToolsKey = toolsKey
                    activeIterator = wrapped
                    let (stream, generationTask) = MLXLMCommon.generateTask(
                        promptTokenCount: input.text.tokens.size,
                        modelConfiguration: modelConfiguration,
                        tokenizer: tokenizer,
                        iterator: wrapped,
                        tools: toolSpecs,
                        toolCallPolicy: parameters.toolCallPolicy)
                    innerTask.task = generationTask
                    // 竞态兜底：外层取消可能发生在生成任务挂载之前。
                    try Task.checkCancellation()
                    AgentLog.info("[mlx-kv] reuse=\(reuseCache || prefixReuse) cached=\(cachedPromptTokens) "
                        + "delta=\(totalPromptTokens - cachedPromptTokens) "
                        + "prefill_ms=\(Int(wrapped.promptPrefillTime * 1000)) "
                        + "common=\(common) transcript=\(sessionTranscript.count)")

                    var thinkSplitter = ThinkSplitter(startInThink: probe.startInThink)
                    var toolSplitter = ToolTagSplitter()
                    var rawContent = probe.generationPrefix
                    var collectedCalls: [ToolCall] = []
                    var rejectedDetail: String?
                    var usage: [String: Int] = [:]

                    for await generation in stream {
                        try Task.checkCancellation()
                        switch generation {
                        case .chunk(let text):
                            rawContent += text
                            for piece in thinkSplitter.feed(text) {
                                if piece.reasoning {
                                    if !piece.text.isEmpty {
                                        continuation.yield(LLMStreamEvent(
                                            kind: .reasoning,
                                            content: piece.text
                                        ))
                                    }
                                } else {
                                    for visible in toolSplitter.feed(piece.text) where !visible.isEmpty {
                                        continuation.yield(LLMStreamEvent(
                                            kind: .content,
                                            content: visible
                                        ))
                                    }
                                }
                            }

                        case .toolCall(let call):
                            collectedCalls.append(Self.toToolCall(call))

                        case .rejectedToolCall(let rejected):
                            // 不立即失败：部分模型的工具调用只出现在文本里，
                            // 先记录，流结束后再尝试文本解析兜底。
                            let name = rejected.toolName.map { "（工具：\($0)）" } ?? ""
                            rejectedDetail = name + (rejected.detail ?? rejected.reason.rawValue)

                        case .info(let info):
                            let promptTokens = cachedPromptTokens + info.promptTokenCount
                            let total = promptTokens + info.generationTokenCount
                            usage = [
                                "prompt_tokens": promptTokens,
                                "completion_tokens": info.generationTokenCount,
                                "total_tokens": total,
                                "tokens_per_second": Int(info.tokensPerSecond),
                                "cached_prompt_tokens": cachedPromptTokens,
                            ]
                            AgentLog.info(
                                "[mlx] prompt=\(promptTokens) "
                                + "cached=\(cachedPromptTokens) "
                                + "generated=\(info.generationTokenCount) "
                                + "tps=\(Int(info.tokensPerSecond))")
                        }
                    }

                    // 收尾：输出剩余缓冲；若仍处于 tool 标签内则丢弃未完成的块。
                    for piece in thinkSplitter.flush() {
                        let events: [LLMStreamEvent]
                        if piece.reasoning {
                            events = piece.text.isEmpty ? [] : [
                                LLMStreamEvent(kind: .reasoning, content: piece.text)
                            ]
                        } else {
                            events = toolSplitter.feed(piece.text).map {
                                LLMStreamEvent(kind: .content, content: $0)
                            }
                        }
                        for event in events where !event.content.isEmpty {
                            continuation.yield(event)
                        }
                    }
                    for visible in toolSplitter.flush() where !visible.isEmpty {
                        continuation.yield(LLMStreamEvent(kind: .content, content: visible))
                    }

                    // 框架没解析出结构化 tool call 时，尝试从原始文本提取
                    // <tool_call>{...}</tool_call> 或 JSON 围栏，避免直接失败。
                    var finalCalls = collectedCalls
                    if finalCalls.isEmpty {
                        finalCalls = Self.extractTextToolCalls(
                            rawContent, allowedNames: allowedToolNames)
                    }
                    // 框架拒绝了工具调用、文本兜底也提不出完整调用时，视为生成被
                    // 截断：不直接失败终止本轮，交给内核按「截断」重试。
                    let truncated = finalCalls.isEmpty && rejectedDetail != nil
                    if truncated {
                        AgentLog.warning("[mlx] 工具调用未完成（\(rejectedDetail ?? "?")），请求重新生成")
                    }

                    _ = await innerTask.task?.value
                    // 释放已完成的底层生成任务，避免服务长期持有其闭包与流。
                    innerTask.task = nil
                    if truncated {
                        // 半截工具调用不会进入消息轨迹，继续复用这段缓存会让
                        // 下一轮从残缺 payload 之后接着生成。但 prompt 段的 KV
                        // 仍然有效：可裁剪缓存裁掉已生成的部分保留 prompt，
                        // 内核重试（相同历史 + 末尾追加纠错提示）经 token 级
                        // 前缀匹配命中整段 prompt，重试 prefill 只剩增量。
                        let generated = activeIterator?.tokenCount ?? 0
                        if let salvage = activeCache, generated > 0,
                           salvage.allSatisfy({ $0.isTrimmable }) {
                            let trimmed = salvage.map { $0.trim(generated) }
                            if trimmed.allSatisfy({ $0 == generated }) {
                                rawCache = salvage
                                rawState = nil
                                cachedTokenCount = activePromptTokens
                                cachedTokens = activePromptTokenIDs ?? []
                                sessionTranscript = activeTranscript
                                pendingGeneratedAssistant = false
                                cachedToolsKey = activeToolsKey
                                AgentLog.info("[mlx-kv] truncated 抢救：保留 "
                                    + "prompt=\(activePromptTokens) 裁生成=\(generated)")
                            } else {
                                invalidateKVCache()
                            }
                        } else {
                            // 不可裁剪缓存（Qwen3.5 MambaCache）无法撤销生成段。
                            invalidateKVCache()
                        }
                    } else {
                        // 成功后才提交缓存与轨迹；下一次调用可直接在其后追加。
                        rawCache = cache
                        rawState = wrapped.state
                        cachedTokenCount = totalPromptTokens + wrapped.tokenCount
                        // token 级前缀匹配需要「prompt + 生成」的完整 token 序列；
                        // 拿不到完整序列（消息级 delta 路径）时置空，下一轮回退
                        // 消息级匹配。
                        if let promptTokens = promptTokensForCache {
                            cachedTokens = promptTokens + wrapped.generatedTokens
                        } else {
                            cachedTokens = []
                        }
                        sessionTranscript = transcriptAfter
                        pendingGeneratedAssistant = true
                        cachedToolsKey = toolsKey
                    }

                    continuation.yield(LLMStreamEvent(
                        kind: .done,
                        toolCalls: finalCalls,
                        usage: usage,
                        rawContent: rawContent,
                        truncated: truncated
                    ))
                    continuation.finish()
                } catch {
                    innerTask.cancel()
                    // 取消/失败时缓存已部分推进。prompt 段 KV 仍然有效：
                    // 可裁剪缓存裁掉已生成的部分保留 prompt（内核重试/下一轮
                    // 消息 = 原历史 + 新输入，token 级前缀命中 prompt，只补
                    // 增量）；不可裁剪缓存只能整体丢弃重建。
                    let generated = activeIterator?.tokenCount ?? 0
                    if let salvage = activeCache, generated > 0,
                       salvage.allSatisfy({ $0.isTrimmable }) {
                        let trimmed = salvage.map { $0.trim(generated) }
                        if trimmed.allSatisfy({ $0 == generated }) {
                            rawCache = salvage
                            rawState = nil
                            cachedTokenCount = activePromptTokens
                            cachedTokens = activePromptTokenIDs ?? []
                            sessionTranscript = activeTranscript
                            pendingGeneratedAssistant = false
                            cachedToolsKey = activeToolsKey
                            AgentLog.info("[mlx-kv] 中断抢救：保留 prompt=\(activePromptTokens) "
                                + "裁生成=\(generated)（\(String(describing: type(of: error))))")
                            continuation.finish(throwing: Self.mapError(error, modelName: modelName))
                            return
                        }
                    }
                    AgentLog.warning("[mlx-kv] 中断丢弃缓存重建: "
                        + "\(String(describing: type(of: error))) "
                        + "\(String(describing: error).prefix(160))")
                    rawCache = nil
                    rawState = nil
                    cachedTokenCount = 0
                    cachedTokens = []
                    sessionTranscript = []
                    pendingGeneratedAssistant = false
                    cachedToolsKey = nil
                    continuation.finish(throwing: Self.mapError(error, modelName: modelName))
                }
            }
            continuation.onTermination = { _ in
                innerTask.cancel()
                task.cancel()
            }
        }
    }

    public func chatStream(
        messages: [ChatMessage],
        tools: [JSONValue],
        temperature: Double,
        options: LLMRequestOptions
    ) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        let thinking = options.enableThinking ?? enableThinking
        // 覆盖 thinking 或要求独立会话时，用共享权重的副本跑一次完整请求：
        // 既不会改动常驻会话的 KV，也能让多个摘要分片真正并行。
        guard options.isolated || thinking != enableThinking else {
            return chatStream(messages: messages, tools: tools, temperature: temperature)
        }
        if options.isolated {
            // 并行摘要分片：一次性副本，不保留半截历史污染其它请求。
            let isolated = MLXLLMService(sharing: self, enableThinking: thinking)
            return isolated.chatStream(messages: messages, tools: tools,
                                       temperature: temperature)
        }
        // 仅覆盖 thinking（空正文重试等）：复用持久副本。它与主服务各自
        // 持有独立 KV/轨迹，但副本内部允许跨轮前缀复用，避免每次重试都
        // 重跑整段 prefill。
        thinkingCopyLock.lock()
        let copy: MLXLLMService
        if let existing = thinkingCopies[thinking] {
            copy = existing
        } else {
            copy = MLXLLMService(sharing: self, enableThinking: thinking,
                                 kvReuse: true)
            thinkingCopies[thinking] = copy
        }
        thinkingCopyLock.unlock()
        return copy.chatStream(messages: messages, tools: tools,
                               temperature: temperature)
    }

    // MARK: - 加载 / 卸载

    /// 主动加载权重与 chat template，供配置窗口的「加载模型」使用。
    public func preload() async throws {
        let container = try await state.container(directory: directory)
        _ = try await state.templateProbe(container: container,
                                         enableThinking: enableThinking)
    }

    /// 用稳定 system prompt（含工具清单）预热 KV cache。
    ///
    /// 只做 prefill、不进入解码；缓存的是正式请求的第一段前缀，因此不能包含
    /// generation prompt。工具清单参与该前缀渲染，所以也必须一起传入。
    public func warmup(systemPrompt: String, tools: [JSONValue]) async throws {
        // 预热放在独立 Task：它要跑完整段 prefill（首次还含 Metal shader 编译），
        // chatStream 通过 currentWarmup() 等它结束，就能让启动期间发出的第一句
        // 排队复用刚建好的前缀，而不是被 UI 丢弃或与预热并发踩踏 GPU。
        // 注册与创建放在同一把锁内：chatStream 可能在启动瞬间读取
        // currentWarmup()，先建 Task 后注册会漏掉这次等待，导致正式请求与
        // 预热并发踩踏 GPU。Task 体异步调度，不会在锁内同步执行。
        warmupLock.lock()
        let task = Task {
            try await self.performWarmup(systemPrompt: systemPrompt, tools: tools)
        }
        activeWarmup = task
        warmupLock.unlock()
        try await task.value
    }

    private func performWarmup(systemPrompt: String, tools: [JSONValue]) async throws {
        // 外层 preload Task 被取消时，这个 GPU 工作仍需跑完并释放引用；
        // 只在自身结束时清，保证在途请求一定会等到它。
        defer { setActiveWarmup(nil) }
        guard !systemPrompt.isEmpty else { return }
        let toolsKey = Self.toolsKey(tools)
        // 历史折叠/滑窗会丢弃会话 KV，但 system + tools 前缀通常没变。
        // 此时已有前缀快照可供 chatStream 的 prefix-reuse 复制，重跑整段
        // prefill 纯属重复劳动：每 8 轮折叠就多花 ~18s，正是「时快时慢」
        // 的主因。签名一致时只丢会话缓存、保留快照，本轮零 GPU 开销。
        if systemPrompt == warmupSystemPrompt,
           toolsKey == warmupToolsKey,
           warmupCacheSnapshot != nil, !warmupTokens.isEmpty {
            invalidateKVCache()
            AgentLog.info("[mlx-warmup] 复用已有前缀快照 tokens=\(warmupTokens.count) "
                + "（system/tools 未变，跳过全量 prefill）")
            return
        }
        let warmupStarted = Date()
        func stage(_ name: String) {
            AgentLog.info("[mlx-warmup] stage=\(name) "
                + "ms=\(Int(Date().timeIntervalSince(warmupStarted) * 1000))")
        }
        // 预热失败/取消时不能留下旧缓存，否则正式请求会接着旧前缀生成。
        invalidateKVCache()
        warmupCacheSnapshot = nil
        warmupStateSnapshot = nil
        warmupTokens = []
        warmupToolsKey = nil
        warmupSystemPrompt = nil
        try Task.checkCancellation()
        let container = try await state.container(directory: directory)
        let tokenizer = await container.tokenizer
        guard let bridge = tokenizer as? MLXTokenizerBridge else {
            // 诊断：container 的 tokenizer 若不是应用的 bridge 包装，预热静默失效。
            AgentLog.warning("[mlx-warmup] 跳过：tokenizer 类型 \(type(of: tokenizer)) "
                + "不是 MLXTokenizerBridge")
            return
        }

        let toolSpecs = Self.makeToolSpecs(tools)
        let context: [String: any Sendable] = ["enable_thinking": enableThinking]

        let model = await container.perform {
            UncheckedSendableBox($0.model)
        }.value
        let parameters = Self.parameters(
            maxTokens: 1, temperature: 0, topP: topP, topK: topK,
            minP: minP, repetitionPenalty: repetitionPenalty,
            promptBatchSize: promptBatchSize,
            kvInt8: kvInt8)
        try Task.checkCancellation()
        let cache = Self.stabilizeQuantizedCache(
            try model.newCache(parameters: parameters),
            kvInt8: kvInt8)

        // system-only 渲染被模板拒绝（Qwen3.5 等要求消息里有 user）时，
        // 用两条不同占位 user 渲染并差分求出 system 段 token 边界。
        // 【关键】prefill 只喂公共前缀段（system 段），从不 prefill 占位
        // 内容再裁剪——旧实现依赖 KVCache.trim，把 MambaCache（Qwen3.5 的
        // SSM 递归状态不可回退）整体挡在预热之外；改为按边界 prefill 后，
        // 可裁剪与不可裁剪缓存统一适用，且省掉占位段的多余计算。
        var tokens: [Int]
        var keepCount: Int
        do {
            let rawMessages = DefaultMessageGenerator().generate(
                messages: [.system(systemPrompt)])
            let rendered = try bridge.applyChatTemplate(
                messages: rawMessages, tools: toolSpecs,
                additionalContext: context, addGenerationPrompt: false)
            tokens = rendered
            keepCount = rendered.count
        } catch {
            func render(_ dummy: String) throws -> [Int] {
                try bridge.applyChatTemplate(
                    messages: DefaultMessageGenerator().generate(messages: [
                        .system(systemPrompt), .user(dummy),
                    ]),
                    tools: toolSpecs, additionalContext: context,
                    addGenerationPrompt: false)
            }
            let renderedA = try render("AA")
            let renderedB = try render("BBBB BBBB BBBB")
            tokens = renderedA
            keepCount = Self.commonTokenPrefix(renderedA, renderedB)
            if keepCount < renderedA.count {
                AgentLog.info("[mlx-warmup] 模板需要 user 消息：占位差分 system 段 "
                    + "\(keepCount)/\(renderedA.count) tokens")
            }
        }
        stage("render")
        guard keepCount >= 32, keepCount <= contextWindow - 1024 else {
            AgentLog.warning("[mlx-warmup] 跳过：前缀 token 数 \(keepCount) 不适合预热")
            return
        }

        // 只预填前缀段：缓存推进长度精确等于 keepCount，快照与正式请求
        // 的前缀逐 token 对齐，无需任何 trim。
        let prefixTokens = Array(tokens.prefix(keepCount))
        let iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(prefixTokens)),
            model: model, cache: cache, state: nil, parameters: parameters)
        let wrapped = StatefulTokenIterator(iterator: iterator)

        // MLX 惰性求值：TokenIterator.prepare()/next() 内部只用 asyncEval（异步提交），
        // cache.copy() 也只是惰性切片。若此时就返回，状态栏会提前显示「预热结束」，
        // 而首句要现场补做 prefill、并首次编译 decode/sampler 路径的 Metal kernel
        // （实测首句额外 20~30s，Metal 缓存到启动后 85s 才落盘）。
        // 因此预热必须把 GPU 工作真正做完再返回。
        // 1) 强制求值：让「已 prefill」的 KV 状态不再是欠条。
        eval(cache.flatMap { $0.state })
        stage("prefill-eval")

        // 2) 牺牲性 decode：从「已 prefill 的 system 前缀缓存」复制一份，追加
        //    一小段 user 形状的 token 再 decode 两步——编译的是首轮真实形状
        //    （长 KV 偏移 + 增量 prefill + decode/sampler）。旧实现用 2 token
        //    输入 + 独立空缓存，编译的是完全不同的 kernel，首句仍要现场编译
        //    20~30s。
        //    【必须 state: nil】主前缀的 LMOutput.State 里携带模型前向状态，
        //    MLXArray 是引用类型；把它交给 scratch 再跑几十步 decode 会写回并
        //    污染 warmupStateSnapshot / rawState，首轮虽用新缓存，模型侧残留
        //    状态仍可能把生成带偏。副本缓存 + 独立 state 才是真正的隔离。
        do {
            let inferParameters = Self.parameters(
                maxTokens: 2, temperature: 0.7, topP: topP, topK: topK,
                minP: minP, repetitionPenalty: repetitionPenalty,
                promptBatchSize: promptBatchSize, kvInt8: kvInt8)
            let decodeCache = cache.map { $0.copy() }
            let extra = Array(tokens.prefix(min(32, tokens.count)))
            var scratch = try TokenIterator(
                input: LMInput(tokens: MLXArray(extra)),
                model: model, cache: decodeCache, state: nil,
                parameters: inferParameters)
            var steps = 0
            while steps < extra.count + 2, scratch.next() != nil { steps += 1 }
            eval(decodeCache.flatMap { $0.state })
            AgentLog.info("[mlx-warmup] decode kernel 预热 shape=\(extra.count) "
                + "steps=\(steps)")
        } catch {
            // kernel 预热失败不该让整个预热失效：主前缀缓存仍可用。
            AgentLog.info("[mlx-warmup] decode kernel 预热跳过：\(error.localizedDescription)")
        }

        // 3) 等 GPU 队列排空（shader 编译、权重换入全部完成），状态栏才是真就绪。
        Stream.defaultStream.synchronize()
        try Task.checkCancellation()

        // 保留一份未推进的前缀快照；历史折叠/滑窗需要重建时从它复制，
        // 只 prefill 动态历史，避免再次计算整段 system + tools。
        // 快照本身是常驻 KV：前缀过大时不再复制，避免内存直接翻倍。
        if keepCount <= 8_192 {
            warmupCacheSnapshot = cache.map { $0.copy() }
            warmupStateSnapshot = wrapped.state
            warmupTokens = Array(tokens.prefix(keepCount))
            warmupToolsKey = toolsKey
            warmupSystemPrompt = systemPrompt
        } else {
            AgentLog.info("[mlx-warmup] 前缀 \(keepCount) tokens 超过快照上限，跳过副本")
        }

        rawCache = cache
        rawState = wrapped.state
        cachedTokenCount = keepCount
        cachedTokens = Array(tokens.prefix(keepCount))
        sessionTranscript = [ChatMessage(role: .system, content: systemPrompt)]
        pendingGeneratedAssistant = false
        cachedToolsKey = toolsKey
        AgentLog.info("[mlx-warmup] prefill tokens=\(keepCount) kept=\(keepCount) "
            + "tools=\(tools.count) "
            + "gpu_ready_ms=\(Int(Date().timeIntervalSince(warmupStarted) * 1000))")
    }

    /// 释放本地权重；实际释放发生在所有引用消失后，同时清空 MLX 缓存。
    public func close() {
        Task { await unloadAndWait() }
    }

    /// 等待权重卸载与 GPU 工作收尾。退出流程必须用它：`close()` 只是发起
    /// 卸载，若进程随后退出，仍在执行的 asyncEval 会与 command encoder
    /// 销毁并发，触发 Metal 断言或 SIGSEGV。
    public func unloadAndWait() async {
        invalidateKVCache()
        thinkingCopyLock.lock()
        thinkingCopies.removeAll()
        thinkingCopyLock.unlock()
        warmupCacheSnapshot = nil
        warmupStateSnapshot = nil
        warmupTokens = []
        warmupToolsKey = nil
        warmupSystemPrompt = nil
        // 先排空 GPU 队列，再释放容器，最后再同步一次收尾。
        MLX.Stream.gpu.synchronize()
        await state.unload()
        MLX.Stream.gpu.synchronize()
    }

    /// 历史被折叠/回滚或换模型时丢弃跨轮会话与 KV cache。
    public func invalidateKVCache() {
        // 缓存非空时被清空意味着下一轮要全量重建（可能十几秒），留痕便于
        // 事后从日志定位「每句回复都慢」的触发源。
        if rawCache != nil || !sessionTranscript.isEmpty {
            AgentLog.info("[mlx-kv] invalidate 丢弃缓存 cached=\(cachedTokenCount) "
                + "transcript=\(sessionTranscript.count)")
        }
        rawCache = nil
        rawState = nil
        cachedTokenCount = 0
        cachedTokens = []
        sessionTranscript = []
        pendingGeneratedAssistant = false
        cachedToolsKey = nil
        // 历史被折叠/回滚/换模型时，副本的轨迹同样失效；副本字典为空，
        // 不会递归回主服务。
        thinkingCopyLock.lock()
        let copies = Array(thinkingCopies.values)
        thinkingCopyLock.unlock()
        for copy in copies { copy.invalidateKVCache() }
    }

    // MARK: - 批量摘要

    /// 把一组摘要请求合成单个 batch 生成，单线程跑 batched prefill + decode。
    ///
    /// 只支持 SSM 混合架构（newCache 产出 MambaCache 的 Qwen35 系）；
    /// 其余架构返回 nil，调用方回退到逐条路径。与聊天路径一样假设由
    /// Agent 串行调用，不与 chatStream 并发共享模型权重。
    public func generateSummariesBatch(
        requests: [(system: String, prompt: String)],
        temperature: Double,
        maxTokensPerRow: Int
    ) async throws -> [String]? {
        let container = try await state.container(directory: directory)
        let model = await container.perform {
            UncheckedSendableBox($0.model)
        }.value
        let tokenizer = await container.tokenizer
        let configuration = await container.configuration

        // 仅 Qwen35 类 SSM 混合模型：批量生成器依赖 MambaCache 的 leftPadding
        // 掩码语义与注意力掩码替换，其它架构未验证。
        guard try model.newCache(parameters: nil).contains(where: { $0 is MambaCache }) else {
            return nil
        }
        // 停止集合与生产循环（buildStopTokenIds）一致：模型 config eos +
        // tokenizer eos + extraEOSTokens；左 padding 用 eos 占位。
        var stopTokenIds = configuration.eosTokenIds
        if let id = tokenizer.eosTokenId {
            stopTokenIds.insert(id)
        }
        for token in configuration.extraEOSTokens {
            if let id = tokenizer.convertTokenToId(token) {
                stopTokenIds.insert(id)
            }
        }
        guard
            let padToken = tokenizer.eosTokenId ?? configuration.eosTokenIds.first,
            !stopTokenIds.isEmpty
        else {
            return nil
        }

        let thinkingContext: [String: any Sendable] = ["enable_thinking": false]
        var rows: [[Int]] = []
        rows.reserveCapacity(requests.count)
        for request in requests {
            let chat: [Chat.Message] = [.system(request.system), .user(request.prompt)]
            let input = try await container.prepare(
                input: UserInput(
                    chat: chat,
                    tools: nil,
                    additionalContext: thinkingContext))
            rows.append(input.text.tokens.asArray(Int.self))
        }

        guard
            let plan = BatchGenerationPlan.leftPad(rows: rows, padToken: padToken)
        else {
            return []
        }

        let generated = try BatchedMLXGenerator.generate(
            tokens: plan,
            model: model,
            stopTokenIds: stopTokenIds,
            padToken: padToken,
            maxTokensPerRow: maxTokensPerRow,
            temperature: temperature > 0.02 ? Float(temperature) : nil)

        var results: [String] = []
        results.reserveCapacity(generated.count)
        for tokens in generated {
            if tokens.isEmpty {
                results.append("")
                continue
            }
            let text = await container.decode(tokenIds: tokens)
            results.append(text.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return results
    }

    // MARK: - 消息 / 工具 schema 转换

    static func makeChatMessages(_ messages: [ChatMessage]) -> [Chat.Message] {
        messages.map { message in
            switch message.role {
            case .system:
                return .system(message.content)
            case .user:
                return .user(message.content)
            case .assistant:
                let calls = message.toolCalls?.map { toMLXToolCall($0) }
                return .assistant(message.content, toolCalls: calls)
            case .tool:
                return .tool(message.content, id: message.toolCallID, name: message.name)
            }
        }
    }

    static func makeToolSpecs(_ tools: [JSONValue]) -> [[String: any Sendable]]? {
        guard !tools.isEmpty else { return nil }
        let specs: [[String: any Sendable]] = tools.compactMap { tool in
            guard let object = tool.objectValue else { return nil }
            var result: [String: any Sendable] = [:]
            for (key, value) in object {
                result[key] = toSendable(value)
            }
            return result
        }
        return specs.isEmpty ? nil : specs
    }

    private static func toMLXToolCall(_ call: ToolCall) -> MLXLMCommon.ToolCall {
        var arguments: [String: any Sendable] = [:]
        for (key, value) in call.arguments {
            arguments[key] = toSendable(value)
        }
        return MLXLMCommon.ToolCall(
            function: .init(name: call.name, arguments: arguments),
            id: call.id.isEmpty ? nil : call.id
        )
    }

    /// 从模型原始文本中提取 `<tool_call>{...}</tool_call>` 或 JSON 围栏。
    ///
    /// 只接受本次请求里声明过的工具名：模型在普通回答中输出
    /// `{"name": "...", ...}` 时不能凭空触发工具。
    static func extractTextToolCalls(_ raw: String,
                                     allowedNames: Set<String>) -> [ToolCall] {
        guard !raw.isEmpty, !allowedNames.isEmpty else { return [] }
        var payloads: [String] = []
        var seen = Set<String>()

        func collect(_ regex: String, group: Int = 1) {
            guard let re = try? NSRegularExpression(pattern: regex,
                                                    options: [.dotMatchesLineSeparators]) else { return }
            let range = NSRange(raw.startIndex..<raw.endIndex, in: raw)
            for match in re.matches(in: raw, options: [], range: range) {
                guard match.numberOfRanges > group,
                      let r = Range(match.range(at: group), in: raw) else { continue }
                let text = String(raw[r]).trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty, seen.insert(text).inserted { payloads.append(text) }
            }
        }

        collect("<tool_call>\\s*([\\s\\S]*?)\\s*</tool_call>")
        collect("```(?:json)?\\s*([\\s\\S]*?)```")
        if payloads.isEmpty, let value = JSONValue.parse(raw), value.objectValue != nil {
            payloads.append(raw)
        }

        var calls: [ToolCall] = []
        var jsonHandled = Set<String>()
        for payload in payloads {
            guard let object = JSONValue.parse(payload)?.objectValue else { continue }
            let function = object["function"]?.objectValue
            let name = (object["name"]?.stringValue
                ?? object["tool"]?.stringValue
                ?? function?["name"]?.stringValue ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty, allowedNames.contains(name) else { continue }
            var arguments: [String: JSONValue] = [:]
            if let rawArgs = object["arguments"] ?? object["parameters"] ?? function?["arguments"] {
                if let text = rawArgs.stringValue, let parsed = JSONValue.parse(text)?.objectValue {
                    arguments = parsed
                } else if let obj = rawArgs.objectValue {
                    arguments = obj
                }
            }
            let id = object["id"]?.stringValue
                ?? "call_\(UUID().uuidString.prefix(8))"
            jsonHandled.insert(payload)
            calls.append(ToolCall(id: id, name: name, arguments: arguments,
                                  rawArguments: JSONValue.object(arguments).jsonString()))
        }
        // GLM47 / GLM4 方言：`name<arg_key>k</arg_key><arg_value>v</arg_value>`。
        // Spark-X2.6 的 chat template 就是这套；库的结构化 parser 要求至少
        // 一个 <arg_key>，零参数工具会解析失败，文本兜底里一并兼容。
        // GGUF / ONNX 侧通过 `LocalToolCallParser` 复用同一实现。
        for payload in payloads where !jsonHandled.contains(payload) {
            guard let call = LocalToolCallParser.glm4Payload(
                payload, allowedNames: allowedNames) else { continue }
            calls.append(call)
        }
        return calls
    }

    static func toToolCall(_ call: MLXLMCommon.ToolCall) -> ToolCall {
        var arguments: [String: JSONValue] = [:]
        for (key, value) in call.function.arguments {
            arguments[key] = toJSON(value)
        }
        let id = call.id ?? "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12))"
        return ToolCall(
            id: id,
            name: call.function.name,
            arguments: arguments,
            rawArguments: JSONValue.object(arguments).jsonString()
        )
    }

    private static func toSendable(_ value: JSONValue) -> any Sendable {
        switch value {
        case .null:
            return Int?.none
        case .bool(let value):
            return value
        case .int(let value):
            return value
        case .double(let value):
            return value
        case .string(let value):
            return value
        case .array(let values):
            return values.map { toSendable($0) }
        case .object(let object):
            var result: [String: any Sendable] = [:]
            for (key, value) in object {
                result[key] = toSendable(value)
            }
            return result
        }
    }

    private static func toJSON(_ value: MLXLMCommon.JSONValue) -> JSONValue {
        switch value {
        case .null:
            return .null
        case .bool(let value):
            return .bool(value)
        case .int(let value):
            return .int(value)
        case .double(let value):
            return .double(value)
        case .string(let value):
            return .string(value)
        case .array(let values):
            return .array(values.map { toJSON($0) })
        case .object(let object):
            var result: [String: JSONValue] = [:]
            for (key, value) in object {
                result[key] = toJSON(value)
            }
            return .object(result)
        }
    }

    // MARK: KV cache 辅助

    private static func parameters(
        maxTokens: Int,
        temperature: Double,
        topP: Float,
        topK: Int,
        minP: Float,
        repetitionPenalty: Float,
        promptBatchSize: Int,
        kvInt8: Bool
    ) -> GenerateParameters {
        var prefill = PrefillParameters()
        if promptBatchSize > 0 {
            prefill.stepSize = max(64, promptBatchSize)
        }
        return GenerateParameters(
            maxTokens: maxTokens,
            // KV cache int8 量化：newCache 按 kvBits 建 QuantizedKVCache，
            // KV 内存约减半，生成质量几乎无损。
            kvBits: kvInt8 ? 8 : nil,
            temperature: Float(temperature),
            topP: topP,
            topK: topK,
            minP: minP,
            // 1.0 等价于无惩罚：传 1.0 只会多挂一个每 token 白跑的 logit processor。
            repetitionPenalty: repetitionPenalty > 1.0 ? repetitionPenalty : nil,
            prefill: prefill
        )
    }

    /// 把标准 KV 缓存预转换成量化缓存（与 kvBits=8 / groupSize=64 配置一致）。
    ///
    /// 库的动态压缩设计是「prefill 用 fp16 存储、首个 decode 步压缩」，压缩
    /// 会**替换缓存数组元素**——外部持有的引用从此过时，生成 token 的 KV 全部
    /// 写进新对象，跨轮 KV 复用（token 级前缀匹配依赖 offset 与 trim）失效。
    /// 预转换后压缩 plan 无对象可换（leaf 非 simple → no-op），引用全程有效。
    /// MambaCache 等非 simple 缓存不受影响，原样保留。
    private static func stabilizeQuantizedCache(
        _ cache: [KVCache], kvInt8: Bool
    ) -> [KVCache] {
        guard kvInt8 else { return cache }
        return cache.map { entry in
            if let simple = entry as? KVCacheSimple {
                return (try? simple.toQuantized(groupSize: 64, bits: 8)) ?? entry
            }
            return entry
        }
    }

    private static func commonPrefixLength(_ lhs: [ChatMessage],
                                           _ rhs: [ChatMessage]) -> Int {
        var index = 0
        while index < lhs.count, index < rhs.count, lhs[index] == rhs[index] {
            index += 1
        }
        return index
    }

    /// 两个 token 序列的严格公共前缀长度（token 级前缀匹配用）。
    /// internal：单元测试直接验证匹配语义。
    static func commonTokenPrefix(_ lhs: [Int], _ rhs: [Int]) -> Int {
        var index = 0
        let limit = min(lhs.count, rhs.count)
        while index < limit, lhs[index] == rhs[index] {
            index += 1
        }
        return index
    }

    /// tools 在 chat template 中的渲染顺序必须与缓存一致，因此按原顺序拼接。
    private static func toolsKey(_ tools: [JSONValue]) -> String {
        tools.map { $0.jsonString() }.joined(separator: "\n")
    }

    /// 把增量中的 tool 结果改写成 `<tool_response>` 包裹的 user 消息。
    ///
    /// 只渲染增量时，chat template 看不到前一条消息，`tool` 分支会漏掉
    /// `<|im_start|>user` 头，导致复用后的上下文错位、模型输出异常。全量渲染
    /// 中 tool 结果本来就渲染成 `user + <tool_response>`，这里按同样形态改写；
    /// 连续多个 tool 结果必须合成一条 user 消息，否则会多出 `<|im_end|>`。
    private static func makeDeltaMessages(_ messages: [Chat.Message]) -> [Chat.Message]? {
        var result: [Chat.Message] = []
        var toolRun: [String] = []

        func flushToolRun() {
            guard !toolRun.isEmpty else { return }
            let blocks = toolRun.enumerated().map { index, content in
                let lead = index == 0 ? "" : "\n"
                return "\(lead)<tool_response>\n\(content)\n</tool_response>"
            }
            result.append(.user(blocks.joined()))
            toolRun.removeAll()
        }

        for message in messages {
            switch message.role {
            case .tool:
                toolRun.append(message.content)
            case .assistant:
                return nil
            default:
                flushToolRun()
                result.append(message)
            }
        }
        flushToolRun()
        return result
    }

    /// 粗略估算会话轨迹的 token 数（用于判断累积缓存是否该回退到裁剪视图）。
    private static func estimateTranscript(_ messages: [ChatMessage]) -> Int {
        var text = ""
        for message in messages {
            text += message.content
            if let calls = message.toolCalls {
                for call in calls {
                    text += call.name
                    text += call.rawArguments
                }
            }
        }
        return TokenEstimator.estimate(text)
    }

    private static func mapError(_ error: Error, modelName: String) -> Error {
        if error is CancellationError {
            return AgentError.cancelled
        }
        if let agentError = error as? AgentError {
            return agentError
        }
        return AgentError.llm("本地 MLX 模型 \(modelName) 推理失败：\(error.localizedDescription)")
    }
}

// MARK: - 生成辅助

/// 跨 actor 传递非 Sendable 的模型引用。
final class UncheckedSendableBox<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}

/// 让 `onTermination`（@Sendable、同步）能够取消底层 MLX 生成任务。
private final class InnerTaskBox: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: Task<Void, Never>?

    var task: Task<Void, Never>? {
        get {
            lock.lock()
            defer { lock.unlock() }
            return storage
        }
        set {
            lock.lock()
            storage = newValue
            lock.unlock()
        }
    }

    func cancel() {
        lock.lock()
        let current = storage
        storage = nil
        lock.unlock()
        current?.cancel()
    }
}

/// `generateTask` 会消费 TokenIterator；用引用类型包一层，生成结束后仍能读到最终 state。
/// 同时记录每个生成 token 的 id：跨轮 token 级前缀匹配需要「prompt + 生成」
/// 的完整 token 序列才能与下一轮的模板渲染结果对齐。
private final class StatefulTokenIterator: TokenIteratorProtocol {
    private var iterator: TokenIterator
    private(set) var generatedTokens: [Int] = []

    init(iterator: TokenIterator) {
        self.iterator = iterator
    }

    var maxTokens: Int? { iterator.maxTokens }
    var tokenCount: Int { iterator.tokenCount }
    var promptPrefillTime: TimeInterval { iterator.promptPrefillTime }
    var state: LMOutput.State? { iterator.state }

    func makeIterator() -> StatefulTokenIterator { self }

    func next() -> Int? {
        guard let token = iterator.next() else { return nil }
        generatedTokens.append(token)
        return token
    }

    func discardGeneratedToken() {
        // 不移除 generatedTokens：stop token 在返回它的 next() 里已被 feed
        // 进缓存（step 先于返回执行），下一轮渲染的 assistant 段同样含终止
        // 符 —— 记录必须与缓存实际内容逐 token 一致。
        iterator.discardGeneratedToken()
    }
}

// MARK: - 流式标签切分

/// Qwen3 think 标签流式切分。`startInThink` 由 chat template 尾部是否已包含 `<think>` 决定。
struct ThinkSplitter {
    private let openTag = "<think>"
    private let closeTag = "</" + "think>"
    private var inThink: Bool
    private var buffer = ""

    init(startInThink: Bool) {
        self.inThink = startInThink
    }

    mutating func feed(_ chunk: String) -> [(reasoning: Bool, text: String)] {
        buffer += chunk
        var output: [(reasoning: Bool, text: String)] = []

        while true {
            let tag = inThink ? closeTag : openTag
            if let range = buffer.range(of: tag) {
                let before = String(buffer[..<range.lowerBound])
                if !before.isEmpty {
                    output.append((inThink, before))
                }
                buffer.removeSubrange(..<range.upperBound)
                inThink.toggle()
            } else {
                // 保留可能跨 chunk 的半截标签。
                let keep = tag.count - 1
                if buffer.count > keep {
                    let split = buffer.index(buffer.endIndex, offsetBy: -keep)
                    let ready = String(buffer[..<split])
                    if !ready.isEmpty {
                        output.append((inThink, ready))
                    }
                    buffer = String(buffer[split...])
                }
                break
            }
        }
        return output
    }

    mutating func flush() -> [(reasoning: Bool, text: String)] {
        let text = buffer
        buffer = ""
        return text.isEmpty ? [] : [(inThink, text)]
    }
}

/// 剥离 `<tool_call>...</tool_call>` 标记，避免结构化工具调用同时污染正文。
struct ToolTagSplitter {
    private let openTag = "<tool_call>"
    private let closeTag = "</tool_call>"
    private var inTool = false
    private var buffer = ""

    mutating func feed(_ chunk: String) -> [String] {
        buffer += chunk
        var output: [String] = []

        while true {
            let tag = inTool ? closeTag : openTag
            if let range = buffer.range(of: tag) {
                let before = String(buffer[..<range.lowerBound])
                if !inTool, !before.isEmpty {
                    output.append(before)
                }
                buffer.removeSubrange(..<range.upperBound)
                inTool.toggle()
            } else {
                let keep = tag.count - 1
                if buffer.count > keep {
                    let split = buffer.index(buffer.endIndex, offsetBy: -keep)
                    let ready = String(buffer[..<split])
                    if !inTool, !ready.isEmpty {
                        output.append(ready)
                    }
                    buffer = String(buffer[split...])
                }
                break
            }
        }
        return output
    }

    mutating func flush() -> [String] {
        let text = buffer
        buffer = ""
        return inTool || text.isEmpty ? [] : [text]
    }
}

// MARK: - 加载状态

/// chat template 探针结果。
private struct TemplateProbe: Sendable {
    /// 模板是否已经以 `<think>` 结尾（生成内容直接从思考开始）。
    var startInThink: Bool
    /// 最后一个 assistant 标记之后的模板文本，用于重建 rawContent。
    var generationPrefix: String
}

/// 模型容器只加载一次；失败后缓存错误，避免重复加载大模型。
private actor LLMLoadState {
    private var container: ModelContainer?
    private var failure: AgentError?
    /// 模板探针按 thinking 开关分别缓存：同一模型可能同时服务摘要（关闭）
    /// 与前台对话（开启），两种模板尾部的生成前缀不同。
    private var probes: [Bool: TemplateProbe] = [:]

    func container(directory: URL) async throws -> ModelContainer {
        if let container { return container }
        if let failure { throw failure }

        do {
            // Spark-X2.6（config.json 的 model_type = spark2_5）是自定义架构，
            // mlx-swift-lm 内置表里没有；必须在 loadContainer 之前注册 creator，
            // 否则 ModelTypeRegistry 会抛 unsupportedModelType。注册幂等，
            // 对其它 model_type 无副作用。
            await SparkModel.register()
            let container = try await LLMModelFactory.shared.loadContainer(
                from: directory,
                using: LocalTokenizerLoader()
            )
            self.container = container
            return container
        } catch {
            let wrapped = AgentError.llm(
                "MLX 模型加载失败（\(directory.path)）：\(error.localizedDescription)"
            )
            failure = wrapped
            throw wrapped
        }
    }

    func unload() {
        container = nil
        probes = [:]
        failure = nil
    }

    func templateProbe(container: ModelContainer, enableThinking: Bool) async throws -> TemplateProbe {
        if let probe = probes[enableThinking] { return probe }

        let tokens = try await container.perform { context in
            try context.tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": "ping"]],
                tools: nil,
                additionalContext: ["enable_thinking": enableThinking]
            )
        }
        let rendered = await container.decode(tokenIds: tokens)
        let startInThink = rendered
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .hasSuffix("<think>")

        var prefix = ""
        if let range = rendered.range(of: "<|im_start|>assistant", options: .backwards) {
            prefix = String(rendered[range.upperBound...])
        }

        let result = TemplateProbe(startInThink: startInThink, generationPrefix: prefix)
        probes[enableThinking] = result
        return result
    }
}
