import Foundation

// MARK: - 工具分发

/// 把内核的 ToolCall 列表变成 ToolResult 列表（顺序一致）。
///
/// 确认、取消、并发这些「如何执行」的细节属于能力层，不放进内核。
public final class AgentToolDispatcher {
    private weak var agent: Agent?
    private var schemaCache: [JSONValue]?
    private let lock = NSLock()

    private let maxParallel = 8

    /// 只读、无副作用、可并发的调用。组合工具需按 `op` 判定：`rag_tools(op=ingest)`
    /// 会写索引、`long_memory(op=graph)` 会写图谱，绝不能并行。
    private func isParallelSafe(_ call: ToolCall) -> Bool {
        let op = call.arguments["op"]?.stringValue ?? ""
        switch call.name {
        case "web_scan":
            return true
        case "rag_tools":
            return op == "search" || op == "outline"
        case "skill_tools":
            return op == "search" || op == "semantic_search" || op == "list"
        case "long_memory":
            return op == "search"
        default:
            return false
        }
    }

    public init(agent: Agent) { self.agent = agent }

    public func schemas() -> [JSONValue] {
        lock.lock(); defer { lock.unlock() }
        if let schemaCache { return schemaCache }
        let value = agent?.tools.schemas() ?? []
        schemaCache = value
        return value
    }

    public func invalidateSchemas() {
        lock.lock(); schemaCache = nil; lock.unlock()
    }

    public func dispatch(_ calls: [ToolCall]) async -> [ToolResult] {
        guard let agent else {
            return calls.map { _ in .fail("agent 已释放") }
        }
        var results = [Int: ToolResult]()
        var parallelJobs: [(Int, ToolCall)] = []

        for (index, call) in calls.enumerated() {
            agent.log("[tool] \(call.name) \(call.compact)")
            AgentLog.info("[tool] 开始 \(call.name) \(call.compact)")

            if call.arguments["__parse_error__"] != nil {
                let raw = call.arguments["__raw__"]?.stringValue ?? ""
                agent.callbacks.onToolStart(call.name, ["_raw_arguments": .string(raw)])
                let result = ToolResult.fail("工具参数解析失败: \(call.rawArguments.prefix(500))")
                agent.callbacks.onToolEnd(call.name, result)
                results[index] = result
                continue
            }

            if agent.callbacks.isCancelled() {
                // 每个 tool_call 都必须补结果，否则下一轮悬挂 tool_calls 会被 API 400。
                agent.log("[tool] \(call.name) 已因用户取消而跳过")
                agent.callbacks.onToolStart(call.name, call.arguments)
                let result = ToolResult.fail("用户取消了对话，工具未执行")
                agent.callbacks.onToolEnd(call.name, result)
                results[index] = result
                continue
            }

            guard let tool = agent.tools.get(call.name) else {
                agent.callbacks.onToolStart(call.name, call.arguments)
                let result = ToolResult.fail("unknown tool: \(call.name)")
                agent.callbacks.onToolEnd(call.name, result)
                results[index] = result
                continue
            }

            if tool.shouldConfirm(call.arguments) {
                let message = ToolSupport.confirmationMessage(name: call.name,
                                                              args: call.arguments)
                guard agent.callbacks.confirm(message) else {
                    agent.log("[tool] \(call.name) DENIED by user")
                    let result = ToolResult.fail("User denied \(call.name) (\(message))")
                    agent.callbacks.onToolEnd(call.name, result)
                    results[index] = result
                    continue
                }
                agent.callbacks.onToolStart(call.name, call.arguments)
                let result = await AgentToolDispatcher.run(tool, call.arguments)
                agent.logToolResult(call.name, result)
                agent.callbacks.onToolEnd(call.name, result)
                results[index] = result
                continue
            }

            if isParallelSafe(call) {
                parallelJobs.append((index, call))
                continue
            }

            agent.callbacks.onToolStart(call.name, call.arguments)
            let result = await AgentToolDispatcher.run(tool, call.arguments)
            agent.logToolResult(call.name, result)
            agent.callbacks.onToolEnd(call.name, result)
            results[index] = result
        }

        if !parallelJobs.isEmpty {
            await runParallel(parallelJobs, agent: agent, into: &results)
        }

        return (0..<calls.count).map {
            results[$0] ?? .fail("工具未执行（harness 内部错误）")
        }
    }

    private func runParallel(_ jobs: [(Int, ToolCall)], agent: Agent,
                             into results: inout [Int: ToolResult]) async {
        let workerCount = min(maxParallel, jobs.count)
        var offset = 0
        while offset < jobs.count {
            // 慢工具（MCP 300s / 渲染 45s）运行期间用户点了取消：
            // 剩余调用不再排队执行，直接补 fail 结果，避免整批继续空转。
            if agent.callbacks.isCancelled() {
                for (index, call) in jobs[offset...] {
                    agent.callbacks.onToolStart(call.name, call.arguments)
                    let result = ToolResult.fail("用户取消了对话，工具未执行")
                    agent.callbacks.onToolEnd(call.name, result)
                    results[index] = result
                }
                return
            }
            let slice = Array(jobs[offset..<min(offset + workerCount, jobs.count)])
            for (_, call) in slice {
                agent.callbacks.onToolStart(call.name, call.arguments)
            }
            let collected = await withTaskGroup(of: (Int, String, ToolResult).self,
                                                returning: [(Int, String, ToolResult)].self) { group in
                for (index, call) in slice {
                    let tool = agent.tools.get(call.name)
                    group.addTask {
                        if agent.callbacks.isCancelled() {
                            return (index, call.name,
                                    ToolResult.fail("用户取消了对话，工具未执行"))
                        }
                        if let tool {
                            return (index, call.name,
                                    await AgentToolDispatcher.run(tool, call.arguments))
                        }
                        return (index, call.name, ToolResult.fail("unknown tool: \(call.name)"))
                    }
                }
                var output: [(Int, String, ToolResult)] = []
                for await item in group { output.append(item) }
                return output
            }
            for (index, name, result) in collected {
                agent.logToolResult(name, result)
                agent.callbacks.onToolEnd(name, result)
                results[index] = result
            }
            offset += slice.count
        }
    }

    private static func run(_ tool: AgentTool,
                            _ args: [String: JSONValue]) async -> ToolResult {
        do {
            return try await tool.run(args)
        } catch {
            return ToolResult.fail("\(type(of: error)): \(error.localizedDescription)")
        }
    }

}

// MARK: - 内核 hooks

private final class AgentKernelHooks: KernelHooks {
    weak var agent: Agent?
    init(agent: Agent) { self.agent = agent }

    /// 只在「上下文占用 >90%」的自动收缩时折叠到 60% 回水线（逐轮自带
    /// summary 的确定性折叠，不调 LLM），摘要行回注 <earlier_context>。
    /// 每 8 轮的周期性折叠在内核内确定性折叠（同样回注摘要行），不经过这里。
    func consolidated(_ session: Session, reason: String) async -> Bool {
        guard reason == "context-90" else { return false }
        return await agent?.shrinkSession(reason: reason) ?? false
    }

    /// 折叠边界：刷新 system 前缀 + 预热 KV（周期性 / 90% / 提前折叠 / 异常重试）。
    func didFold(_ session: Session, reason: String, tools: [JSONValue]) async {
        await agent?.onFoldBoundary(reason: reason, tools: tools)
    }

    func finished(_ session: Session, text: String) {
        agent?.fireFinished(text: text)
    }

    func failed(_ session: Session, error: String) {
        agent?.callbacks.onError(error)
    }
}

// MARK: - Agent 门面

/// Agent 退化为门面：`run()` 只负责记录请求、恢复状态、委托内核、回调 UI。
/// 控制流在 `KernelLoop`（三态自动机 + 五级压缩），能力在 `ToolRegistry`，
/// 经验在事实抽取 / L4 蒸馏。
public final class Agent: @unchecked Sendable {
    public var config: AgentConfig
    public var llm: LLMService
    public var auxLLM: LLMService?
    public let callbacks: AgentCallbacks
    public let tools: ToolRegistry
    /// 定时任务存储/调度器（随 Agent 构建；App 侧负责启动与消费到期回调）。
    public var scheduleStore: ScheduleStore? { tools.scheduleStore }
    public var scheduleRunner: ScheduleRunner? { tools.scheduleRunner }
    public let skills: SkillManager
    public let skillIndex: SkillIndex
    public let skillRetriever: SkillRetriever
    public let ragEngine: RAGEngine
    /// 多知识分区：默认分区即 ragEngine；其余挂载分区各一个独立 engine（共享模型）。
    /// key = zone.id，value 为对应分区的 engine；默认分区的 id 也在此表中指向 ragEngine。
    public let zoneEngines: [String: RAGEngine]
    public let session: Session
    /// 协议主体（BASE）system prompt。默认内置常量；应用启动时可由
    /// `<workspace>/assets/system_prompt.txt` 替换，见 `SystemPromptFile`。
    public let baseSystemPrompt: String
    /// 多级记忆目录（L0–L4）。与 `.agent/` 下既有 GraphMemory 并存。
    public let memoryDirectory: MemoryDirectory
    public let rawSessionStore: RawSessionStore
    /// 历史归档串行链：避免 zip/大文件读取卡住用户请求，同时保证同一天
    /// 不会被两个后台任务重复归档。
    private let archiveLock = NSLock()
    private var archiveTask: Task<Void, Never>?
    /// 与 FileReadTool 共享的 SOP 去重记录器（reset() 时清空）。
    public let sopReadTracker: SOPReadTracker
    public lazy var dispatcher: AgentToolDispatcher = AgentToolDispatcher(agent: self)
    public private(set) var mcpClients: [MCPClient] = []
    private var mcpConnectTask: Task<Bool, Never>?
    private var mcpTaskGeneration = 0
    private let mcpTaskLock = NSLock()
    private var mcpClosed = false

    private var kernelStorage: KernelLoop?
    private var turnIndex = 0
    /// 计划被显式清空的次数：`runIsolated` 据此判断执行期间用户是否手动清空过计划，
    /// 有则不要再用旧快照覆盖。
    private var planEpoch = 0
    private var lastUserText = ""
    /// 动态上下文按「一次用户请求」冻结：同一轮内的多次工具循环复用同一段，
    /// 保证 system + 历史前缀稳定，KV cache 才能跨调用命中。
    private var cachedDynamicContext: String?
    /// L0/L1 常驻 system 前缀缓存。`systemPrompt()` 每个 THINK 轮都被调用，
    /// 必须按「文件签名 + 内容」缓存，否则每轮同步读盘且前缀可能抖动。
    private var cachedStaticSystem: String?
    private var cachedStaticSignature = ""
    private let staticSystemLock = NSLock()
    /// 冻结的 system prompt：两次折叠之间的所有 THINK 轮（含跨用户请求与
    /// 自动续跑）前缀逐字节一致 → KV 可复用；落盘时也写这一份，历史与实发一致。
    /// 不按请求重读 L0/L1，而是**跨请求冻结到折叠边界** —— 段内即使 L1 被
    /// patch，头部也不抖动；折叠时由 `onFoldBoundary` 刷新并预热 KV。
    private var requestSystemPrompt: String?
    private let requestSystemLock = NSLock()
    /// 折叠纪元：每次折叠 +1；`frozenSystemEpoch` 记录当前快照属于哪个纪元。
    private var systemFoldEpoch = 0
    private var frozenSystemEpoch = -1
    /// 改进1：本用户请求内 `finishTurn` 结束的轮引用快照。
    /// 用引用而非拷贝，且早于 CONSOLIDATE 折叠收集，保证被折掉的轮全文不丢。
    private var finishedTurnSnapshot: [TurnRecord] = []
    private let rawSessionSnapshotLock = NSLock()
    /// 改进8：上次**成功产出整理建议**的轮次，按压缩周期节流，避免每轮都调 LLM。
    private var lastMemoryTidyTurn = 0
    private let tidyStateLock = NSLock()
    private let backgroundLock = NSLock()
    private var backgroundRunning = false
    /// 上一轮结束后启动的辅助模型任务（事实抽取/蒸馏）。
    /// 用户发来新请求时先取消它，避免 2B 模型与主模型争抢同一块 GPU。
    private let backgroundTaskLock = NSLock()
    private var backgroundTask: Task<Void, Never>?
    private var backgroundGeneration = 0
    /// 后台事实抽取的文件游标：每 4 小时一次，增量读当天 raw_session 记录。
    /// 状态全部持久化在 cursor 文件里，不占用会话内存、与折叠边界无关。
    private let factExtractionWorker: FactExtractionWorker
    /// 自动续跑段数。每段都重跑完整 prefill 并带着累积的提示，实测 4 段
    /// 可把一次请求放大到 29 次 LLM 调用 / 108s；收缩到 2 段，配合
    /// KernelLoop 已收紧的反思门上限，避免「越提醒越慢」。
    private let autoContinueMax = 2
    public var autoContinue = true

    /// 兼容 Python Agent.max_iterations：始终以 config 为唯一真源。
    public var maxIterations: Int {
        get { max(1, config.maxIterations) }
        set { config.maxIterations = max(1, newValue) }
    }

    public init(llm: LLMService,
                config: AgentConfig,
                callbacks: AgentCallbacks,
                toolRegistry: ToolRegistry? = nil,
                skills: SkillManager? = nil,
                baseSystemPrompt: String = BASE_SYSTEM_PROMPT) {
        self.baseSystemPrompt = baseSystemPrompt
        self.config = config
        config.ensureWorkspace()
        self.llm = llm
        self.callbacks = callbacks

        let agentDir = (config.workspace as NSString).appendingPathComponent(".agent")
        let skillsDirectory = (agentDir as NSString).appendingPathComponent("skills")
        let builtin = (config.workspace as NSString).appendingPathComponent("skills")
        self.skills = skills ?? SkillManager(skillsDirectory: skillsDirectory,
                                             builtinDirectories: [builtin])
        self.skillIndex = SkillIndex(manager: self.skills)
        let embedding = LocalModelFactory.shared.makeEmbedding(config: config)
        let reranker = LocalModelFactory.shared.makeReranker(config: config)
        self.skillRetriever = SkillRetriever(manager: self.skills, embedding: embedding,
                                             reranker: reranker)
        self.ragEngine = RAGEngine(config: config, embedding: embedding, reranker: reranker)
        // 多知识分区：默认分区用 ragEngine；每个额外挂载分区独立建 engine，
        // 共享同一 embedding/reranker provider（工厂按 config 缓存，避免重复加载权重）。
        var zoneMap: [String: RAGEngine] = [:]
        if let defaultZone = config.defaultZone {
            zoneMap[defaultZone.id] = self.ragEngine
        }
        for zone in config.knowledgeZones where zone.enabled && !(config.defaultZone?.id == zone.id) {
            var zoneConfig = config
            zoneConfig.knowledgeBase = zone.source
            zoneMap[zone.id] = RAGEngine(config: zoneConfig, embedding: embedding,
                                         reranker: reranker)
        }
        self.zoneEngines = zoneMap
        let memoryDir = MemoryDirectory(workspace: config.workspace)
        memoryDir.ensureLayout()
        self.memoryDirectory = memoryDir
        self.rawSessionStore = RawSessionStore(directory: memoryDir)
        self.factExtractionWorker = FactExtractionWorker(directory: memoryDir)
        let sharedSOPTracker = SOPReadTracker()
        self.sopReadTracker = sharedSOPTracker
        // 内核预算不能超过模型实际生效的上下文，否则消息会撑破 KV 硬上限。
        let kernelBudget = min(
            config.kernelMaxContextTokens == 0 ? 30_000 : config.kernelMaxContextTokens,
            config.effectiveMainContextTokens)
        self.session = Session(
            budget: ContextBudget(total: kernelBudget),
            // raw 窗口至少覆盖「一个折叠周期 + 折叠时保留的 2 轮」：否则段内
            // 每新增一轮都会挤掉最老轮，KV cache 反复重建（“快两句又慢”）。
            // 挤出的旧轮直接丢弃、不回注摘要。
            rawTurns: max(8,
                          config.kernelRawTurns == 0 ? 20 : config.kernelRawTurns,
                          (config.kernelConsolidateEvery == 0 ? 18 : config.kernelConsolidateEvery) + 2),
            maxDigestLines: config.kernelMaxDigestLines == 0 ? 20 : config.kernelMaxDigestLines
        )

        if let toolRegistry {
            self.tools = toolRegistry
        } else {
            // init 未完成前不能在逃逸闭包里捕获 self（存储属性尚未全部初始化），
            // 先用局部转发闭包占位，注册完成后再指向真正的压缩实现。
            var compressorForward: ((String) async -> String?)?
            var multiZoneSearchForward: RagSearchTool.SearchHandler?
            let defaultEngine = self.ragEngine
            let zoneEngineMap = self.zoneEngines
            self.tools = AgentToolFactory.build(
                config: config, callbacks: callbacks, session: self.session,
                ragEngine: self.ragEngine, skillManager: self.skills,
                skillIndex: self.skillIndex, skillRetriever: self.skillRetriever,
                checkpointCompressor: { text in
                    await compressorForward?(text)
                },
                multiZoneSearch: { query, topK, source in
                    if let handler = multiZoneSearchForward {
                        return try await handler(query, topK, source)
                    }
                    return try await defaultEngine.searchWithCitations(
                        query: query, topK: topK, source: source)
                },
                zoneSummaryProvider: {
                    let zones = config.knowledgeZones.filter { $0.enabled }
                    guard zones.count > 1 else { return [] }
                    return zones.compactMap { zone in
                        guard let e = zoneEngineMap[zone.id] else { return nil }
                        // 直接查真实状态而非缓存：缓存只在 AppModel 刷新后才有，
                        // 若某分区从未被刷新（新配置/刚启动），cachedStatus 为 nil
                        // 会让该分区从总览中静默消失，模型就会编造分区名与文档数。
                        let st = e.status()
                        return (name: zone.name,
                                chunks: st["chunks_stored"]?.intValue ?? 0,
                                sources: st["sources"]?.arrayValue?.compactMap(\.stringValue) ?? [])
                    }
                },
                sopTracker: sharedSOPTracker)
            multiZoneSearchForward = { [weak self] query, topK, source in
                guard let self else {
                    throw AgentError.storage("知识分区不可用")
                }
                if self.config.isDecisionMode {
                    return try await self.decisionMultiZoneSearch(query: query,
                                                                   topK: topK, source: source)
                }
                return try await self.multiZoneSearch(query: query, topK: topK, source: source)
            }
            compressorForward = { [weak self] text in
                await self?.compressCheckpointText(text)
            }
        }
        self.tools.session = self.session
        self.tools.ragEngine = self.ragEngine
        // 改进1/2：每轮结束时收集轮引用，run() 末尾一次性追加落盘。
        session.onTurnFinished = { [weak self] rec in
            guard let self else { return }
            self.rawSessionSnapshotLock.lock()
            self.finishedTurnSnapshot.append(rec)
            self.rawSessionSnapshotLock.unlock()
        }
        // RAG 检索结果的辅助模型精简：动态读取当前 auxLLM（AppModel 可在
        // 运行中替换辅助模型），触发阈值在 RAGEngine.condenseIfNeeded 内，
        // 分片与整合由 RAGEngine 编排（map-reduce 章节摘要）。
        self.ragEngine.auxCondenser = { [weak self] text, stage in
            guard let self, self.auxLLM != nil else { return nil }
            return await self.condenseRAGResult(text, stage: stage)
        }
    }

    private lazy var kernel: KernelLoop = {
        let runtime = KernelRuntime(
            stream: { [weak self] messages, schemas, temperature in
                guard let self else {
                    return AsyncThrowingStream { $0.finish(throwing: AgentError.internalError("agent 已释放")) }
                }
                return self.llm.chatStream(messages: messages, tools: schemas,
                                           temperature: temperature)
            },
            streamWithOptions: { [weak self] messages, schemas, temperature, options in
                guard let self else {
                    return AsyncThrowingStream { $0.finish(throwing: AgentError.internalError("agent 已释放")) }
                }
                return self.llm.chatStream(messages: messages, tools: schemas,
                                           temperature: temperature, options: options)
            },
            execute: { [weak self] calls in
                guard let self else { return calls.map { _ in .fail("agent 已释放") } }
                return await self.dispatcher.dispatch(calls)
            },
            schemas: { [weak self] in self?.dispatcher.schemas() ?? [] },
            shouldCancel: { [weak self] in self?.callbacks.isCancelled() ?? false },
            invalidateKV: { [weak self] in self?.invalidateKVCache() },
            log: { [weak self] line in self?.log(line) }
        )
        let hooks = AgentKernelHooks(agent: self)
        return KernelLoop(runtime: runtime, config: config, callbacks: callbacks,
                          systemPrompt: { [weak self] in self?.systemPrompt() ?? BASE_SYSTEM_PROMPT },
                          dynamicContext: { [weak self] in self?.dynamicContextPrompt() ?? "" },
                          session: session, hooks: hooks, maxTurns: maxIterations)
    }()

    // MARK: 生命周期

    /// 连接配置中的 MCP 服务。幂等：并发调用共享同一个 Task。
    ///
    /// 全部连接失败时清空任务，使下一次 `run` 还能重试；
    /// `close()` 之后迟到的连接结果会被立即关闭，不会重新挂上。
    public func connectMCP() async {
        mcpTaskLock.lock()
        if let task = mcpConnectTask {
            mcpTaskLock.unlock()
            _ = await task.value
            return
        }
        if mcpClosed {
            mcpTaskLock.unlock()
            return
        }
        let task = Task<Bool, Never> { [weak self] in
            guard let self else { return false }
            let clients = await AgentToolFactory.connectMCP(
                config: self.config, callbacks: self.callbacks,
                registry: self.tools,
                onToolsChanged: { [weak self] in
                    guard let self else { return }
                    Task { @MainActor in
                        self.dispatcher.invalidateSchemas()
                        self.kernel.invalidateSchemas()
                    }
                })
            return await MainActor.run { [weak self] in
                guard let self else {
                    clients.forEach { $0.shutdown() }
                    return false
                }
                if self.mcpClosed {
                    clients.forEach { $0.shutdown() }
                    return false
                }
                self.mcpClients.append(contentsOf: clients)
                self.dispatcher.invalidateSchemas()
                self.kernel.invalidateSchemas()
                // 无服务配置视为成功；配置了服务却一个都没连上则允许下次重试。
                return self.config.mcpServers.isEmpty || !clients.isEmpty
            }
        }
        mcpTaskGeneration += 1
        let generation = mcpTaskGeneration
        mcpConnectTask = task
        mcpTaskLock.unlock()

        let ok = await task.value
        if !ok {
            mcpTaskLock.lock()
            if mcpTaskGeneration == generation { mcpConnectTask = nil }
            mcpTaskLock.unlock()
        }
    }

    public func reset() {
        session.reset()
        kernel.resetRuntimeState()
        PlanStore.clear(config)
        turnIndex = 0
        lastUserText = ""
        cachedDynamicContext = nil
        // Session.turnIndex 归零后，若不同步复位节流标记，
        // `turn >= lastTidy + 8` 会长期为 false → 清空对话后整理再也不跑。
        markMemoryTidyDone(turn: 0)
        // 清空对话 = 新段起点：重新读 L0/L1，别沿用上一段的冻结前缀。
        requestSystemLock.lock()
        requestSystemPrompt = nil
        systemFoldEpoch = 0
        frozenSystemEpoch = -1
        requestSystemLock.unlock()
        clearFinishedTurnSnapshot()
        sopReadTracker.reset()
        // 清空对话后 Session 已重置，但本地 LLM 仍持有旧会话的 KV cache；
        // 不失效会让下一轮继续复用已经“被清空”的历史。
        invalidateKVCache()
    }

    /// 把过期归档排到后台串行链。每轮只排一次，前一个没跑完就跟着它，
    /// 避免多轮请求同时压缩同一天的历史文件。
    private func scheduleArchive() {
        let store = rawSessionStore
        archiveLock.lock()
        let previous = archiveTask
        let task = Task.detached(priority: .utility) {
            _ = await previous?.value
            _ = store.archivePastSessions()
        }
        archiveTask = task
        archiveLock.unlock()
    }

    /// 丢弃计划（Session 内 + 落盘文件），不触碰对话历史。
    /// 用于启动时清理上次未完成的计划，以及计划面板的「清空计划」。
    @discardableResult
    public func discardPersistedPlan() -> Bool {
        let hadPersisted = PlanStore.load(config) != nil
        session.clearPlan()
        PlanStore.clear(config)
        planEpoch += 1
        return hadPersisted
    }

    /// 多分区联合检索：并行检索所有挂载分区，RRF 融合后返回单一 outcome。
    /// 只挂一个分区（或无多分区配置）时直接走默认 engine，保持既有行为。
    public func multiZoneSearch(query: String, topK: Int,
                                source: String? = nil) async throws -> RAGRetrievalOutcome {
        let zones = config.mountedZones
        guard zones.count > 1, zoneEngines.count > 1 else {
            return try await ragEngine.searchWithCitations(query: query,
                                                           topK: topK, source: source)
        }
        // source 语义：多分区时若 source 命中某分区名/id，视为「定向到该分区」；
        // 否则视为单库文档名过滤（与联合检索冲突）予以忽略，改查全部分区。
        let targetedZoneID: String? = {
            guard let s = source else { return nil }
            return zones.first { $0.id == s || $0.name == s }?.id
        }()
        let effectiveSource: String? = targetedZoneID == nil ? source : nil
        let searchZones = targetedZoneID.map { id in zones.filter { $0.id == id } } ?? zones
        let started = Date()
        // 前置选库：轻量 FTS 探查，零命中的无关库直接跳过（用户显式定向时不过滤）。
        // 若全部零命中则保守查全部，避免分词差异导致漏检。
        var selectedZones = searchZones
        if targetedZoneID == nil {
            let hits = Dictionary(uniqueKeysWithValues: searchZones.map { zone in
                (zone.id, (zoneEngines[zone.id]?.probeRelevance(query: query) ?? 0) > 0)
            })
            let probed = searchZones.filter { hits[$0.id] == true }
            if !probed.isEmpty { selectedZones = probed }
            let probeDesc = searchZones.map { z in
                z.name + "=" + (hits[z.id] == true ? "命中" : "跳过")
            }.joined(separator: ", ")
            let fallbackNote = probed.isEmpty ? "（全部零命中，保守查全部）" : ""
            AgentLog.info("[multi-zone] 选库探查：\(probeDesc)\(fallbackNote)")
        }
        AgentLog.info("[multi-zone] 联合检索开始 query=\(query.prefix(60)) topK=\(topK) "
            + "实际检索分区=\(selectedZones.map(\.name).joined(separator: "、"))"
            + (source != nil ? " source=\(source!)（\(targetedZoneID == nil ? "文档过滤已忽略" : "定向分区")）" : ""))
        // 判定模式选库：用判定模型决定去哪些库（对全部 searchZones 判定，
        // 不先经过 FTS 过滤，避免词法漏检的库进不了判定候选）。失败则沿用 FTS 结果。
        if config.multiZoneUseDecisionJudge, targetedZoneID == nil,
           let scorer = try? ragEngine.sharedDecisionScorer() {
            do {
                let budget = config.ragQueryTimeoutSeconds == 0
                    ? Double.greatestFiniteMagnitude : max(30, config.ragQueryTimeoutSeconds * 3)
                let descriptions = searchZones.map { z in
                    "知识分区【\(z.name)】：" +
                        ((try? zoneEngines[z.id]?.listDocuments(limit: 6).names) ?? []).joined(separator: "、")
                }
                let scores = try await RAGEngine.withTimeout(budget) {
                    try await scorer.score(query: "用户问句：\(query)", passages: descriptions,
                                           coreSentenceCount: 3, useCoreSentences: false)
                }
                let ranked = zip(searchZones, scores).sorted { $0.1 > $1.1 }
                let pickCount = min(max(1, config.decisionTopN), ranked.count)
                let picked = ranked.prefix(pickCount).map(\.0)
                if !picked.isEmpty { selectedZones = picked }
                // 日志用判定名次对应的分区与分数，避免张冠李戴（M2）。
                let desc = ranked.prefix(pickCount).map { pair in
                    pair.0.name + "=" + String(format: "%.3f", pair.1)
                }.joined(separator: ", ")
                AgentLog.info("[multi-zone] 判定选库：\(desc)")
            } catch {
                AgentLog.warning("[multi-zone] 判定选库失败，沿用 FTS 选库：\(error.localizedDescription)")
            }
        }

        let isColbert = config.colbertReaderEnabled
        // 每库「只粗筛 20 条」（向量=coarseSearch / 极速=coarseCandidates），
        // 返回带 zoneID 与库内原始 id 的中间候选，供跨库统一 RRF 融合。
        let coarsePerZone = 20
        let groups: [MultiZoneRetrieval.CandidateGroup] = await withTaskGroup(
            of: MultiZoneRetrieval.CandidateGroup?.self) { group in
            for zone in selectedZones {
                guard let engine = zoneEngines[zone.id] else { continue }
                group.addTask {
                    do {
                        var candidates: [MultiZoneRetrieval.ZoneCandidate] = []
                        if isColbert {
                            let coarse = await engine.colbertCoarse(query: query,
                                                                    coarse: coarsePerZone,
                                                                    source: effectiveSource)
                            candidates = coarse.map {
                                MultiZoneRetrieval.ZoneCandidate(
                                    zoneID: zone.id, zoneName: zone.name, localID: $0.docID,
                                    text: $0.content, source: $0.source, heading: $0.title,
                                    rrfScore: $0.rrf, baseScore: $0.rrf)
                            }
                        } else {
                            let hits = try await engine.coarseSearch(query: query,
                                                                     topK: coarsePerZone,
                                                                     source: effectiveSource)
                            candidates = hits.map { h in
                                MultiZoneRetrieval.ZoneCandidate(
                                    zoneID: zone.id, zoneName: zone.name,
                                    localID: String(h.rowID), text: h.text,
                                    source: h.source, heading: h.heading,
                                    rrfScore: h.rrfScore ?? h.score,
                                    baseScore: h.vectorScore ?? h.ftsScore ?? h.score)
                            }
                        }
                        AgentLog.info("[multi-zone] 分区 \(zone.name) 粗筛 \(candidates.count) 条")
                        return MultiZoneRetrieval.CandidateGroup(zoneID: zone.id,
                                                                 zoneName: zone.name,
                                                                 candidates: candidates)
                    } catch {
                        AgentLog.warning("[multi-zone] 分区 \(zone.name) 粗筛失败："
                            + error.localizedDescription)
                        return nil
                    }
                }
            }
            var collected: [MultiZoneRetrieval.CandidateGroup] = []
            for await g in group { if let g { collected.append(g) } }
            return collected
        }
        let validGroups = groups.filter { !$0.candidates.isEmpty }
        guard !validGroups.isEmpty else {
            AgentLog.warning("[multi-zone] 所有分区均无有效粗筛候选，降级为默认库检索")
            return try await ragEngine.searchWithCitations(query: query, topK: topK, source: source)
        }
        let fused20 = Array(MultiZoneRetrieval.fuseCandidates(validGroups,
                                                               topK: 20,
                                                               rrfK: config.ragRRFK).prefix(20))
        AgentLog.info("[multi-zone] 跨库 RRF 融合：\(groups.count) 分区 → \(fused20.count) 条候选，"
            + "总耗时 \(Int(Date().timeIntervalSince(started) * 1000))ms")
        guard !fused20.isEmpty else {
            return RAGRetrievalOutcome(context: "（知识库中未找到相关内容）", citations: [])
        }

        // 汇总阶段：判定模式开启时用判定模型对候选核心句打分取 topN；
        // 否则沿用各库原精排（向量 reranker 一次性；极速 MaxSim 按库路由）。
        var citations: [RAGCitation] = []
        var usedDecision = false
        if config.multiZoneUseDecisionJudge,
           let scorer = try? ragEngine.sharedDecisionScorer() {
            do {
                let budget = config.ragQueryTimeoutSeconds == 0
                    ? Double.greatestFiniteMagnitude : max(30, config.ragQueryTimeoutSeconds * 3)
                let passages = fused20.map { $0.text.isEmpty ? "（该分区无正文）" : $0.text }
                let scores = try await RAGEngine.withTimeout(budget) {
                    try await scorer.score(query: query, passages: passages,
                                           coreSentenceCount: self.config.decisionCoreSentenceCount,
                                           useCoreSentences: self.config.decisionUseCoreSentences)
                }
                // topK（工具契约）与 decisionTopN 取较小，尊重调用方的气泡数量。
                let n = max(1, min(topK, config.decisionTopN, fused20.count))
                let ranked = zip(fused20.indices, scores)
                    .sorted { $0.1 > $1.1 ? true : ($0.1 == $1.1 && $0.0 < $1.0) }
                    .prefix(n)
                citations = ranked.enumerated().map { i, pair in
                    let c = fused20[pair.0]
                    return RAGCitation(id: "\(c.zoneID)|\(c.localID)", rank: i + 1,
                                source: c.source.isEmpty ? c.zoneName : (c.source as NSString).lastPathComponent,
                                heading: c.heading, text: passages[pair.0],
                                score: pair.1, isPrimary: i == 0)
                }
                usedDecision = true
                AgentLog.info("[multi-zone] 判定模式汇总：\(citations.count) 条，"
                    + "耗时 \(Int(Date().timeIntervalSince(started) * 1000))ms")
            } catch {
                AgentLog.warning("[multi-zone] 判定汇总失败，降级到原精排：\(error.localizedDescription)")
            }
        }
        if !usedDecision {
            citations = isColbert
                ? try await colbertFusedRerank(fused20, query: query, topK: topK)
                : try await vectorFusedRerank(fused20, query: query, topK: topK)
        }
        return RAGRetrievalOutcome(context: MultiZoneRetrieval.primaryContext(citations),
                                   citations: citations)
    }

    /// 向量模式融合后精排：reranker 共享无状态，对全部候选文本一次性打分取 topK。
    private func vectorFusedRerank(_ fused: [MultiZoneRetrieval.ZoneCandidate],
                                    query: String, topK: Int) async throws -> [RAGCitation] {
        let hits = fused.map { c in
            SearchHit(rowID: Int64(c.localID) ?? 0, text: c.text, source: c.source,
                      chunkIndex: 0, heading: c.heading, score: c.baseScore,
                      vectorScore: nil, ftsScore: nil, rrfScore: c.rrfScore, binaryScore: nil)
        }
        let reranked = await ragEngine.rerankHits(query: query, hits: hits, topK: topK)
        return reranked.prefix(max(1, min(topK, config.decisionTopN, 10))).enumerated().map { i, item in
            // 用精排保留的原始索引映射回跨库候选，避免按 text 反查导致重复正文串台。
            let c = fused[item.originalIndex]
            let h = item.hit
            return RAGCitation(id: "\(c.zoneID)|\(c.localID)", rank: i + 1,
                               source: h.source.isEmpty ? c.zoneName : (h.source as NSString).lastPathComponent,
                               heading: h.heading, text: h.text, score: h.score, isPrimary: i == 0)
        }
    }

    /// 极速模式融合后精排：按 zoneID 把候选路由回各库做 MaxSim，合并取 topK。
    private func colbertFusedRerank(_ fused: [MultiZoneRetrieval.ZoneCandidate],
                                     query: String, topK: Int) async throws -> [RAGCitation] {
        struct Scored { let c: MultiZoneRetrieval.ZoneCandidate; let score: Double; let text: String }
        var scored: [Scored] = []
        let byZone = Dictionary(grouping: fused, by: \.zoneID)
        for (zoneID, group) in byZone {
            guard let engine = zoneEngines[zoneID] else {
                scored += group.map { Scored(c: $0, score: $0.baseScore, text: $0.text) }
                continue
            }
            do {
                let hits = await engine.colbertRerank(query: query, docIDs: group.map(\.localID))
                guard !hits.isEmpty else {
                    AgentLog.warning("[multi-zone] 极速精排空结果 zone=\(zoneID)，用粗筛分兜底")
                    scored += group.map { Scored(c: $0, score: $0.baseScore, text: $0.text) }
                    continue
                }
                let scoreByID = Dictionary(uniqueKeysWithValues: hits.map { ($0.docID, $0.score) })
                let textByID = Dictionary(uniqueKeysWithValues: hits.map { ($0.docID, $0.content) })
                for c in group {
                    // 未回填残差/精排失败会给 0 分，此时回退粗筛分避免排位丢失。
                    let s = scoreByID[c.localID] ?? 0
                    let finalScore = s > 0 ? s : c.baseScore
                    scored.append(Scored(c: c, score: finalScore,
                                         text: textByID[c.localID] ?? c.text))
                }
            } catch {
                AgentLog.warning("[multi-zone] 极速精排失败 zone=\(zoneID)：\(error.localizedDescription)")
                scored += group.map { Scored(c: $0, score: $0.baseScore, text: $0.text) }
            }
        }
        let top = scored.sorted { $0.score > $1.score }.prefix(max(1, min(topK, config.decisionTopN, 10)))
        return top.enumerated().map { i, s in
            RAGCitation(id: "\(s.c.zoneID)|\(s.c.localID)", rank: i + 1,
                        source: s.c.source.isEmpty ? s.c.zoneName : (s.c.source as NSString).lastPathComponent,
                        heading: s.c.heading, text: s.text, score: s.score, isPrimary: i == 0)
        }
    }

    /// 决策模式多分区检索：两次判定。
    /// ① 依据对话状态+问句判定从哪些知识分区检索；② 各分区 FTS 粗筛 50 →
    /// 按判定分加权 RRF 融合取前 20 → 判定模型按核心句评判 → 取 Top3。
    public func decisionMultiZoneSearch(query: String, topK: Int,
                                        source: String? = nil) async throws -> RAGRetrievalOutcome {
        let zones = config.mountedZones
        guard zones.count > 1, zoneEngines.count > 1 else {
            return try await ragEngine.searchWithCitations(query: query, topK: topK, source: source)
        }
        let started = Date()
        AgentLog.info("[decision-multi] 两次判定联合检索开始 query=\(query.prefix(60)) "
            + "挂载分区=\(zones.map(\.name).joined(separator: "、"))")
        guard let scorer = try? ragEngine.sharedDecisionScorer() else {
            // 判定模型不可用：降级为普通多分区联合检索。
            AgentLog.warning("[decision-multi] 判定模型不可用，降级为普通多库联合检索")
            return try await multiZoneSearch(query: query, topK: topK, source: source)
        }
        // 对话状态：最近一条用户输入 + 当前问句。
        let stateText = "用户问句：\(query)"
        // ① 第一次判定：为每个分区构造「领域名 + 代表文档名」描述，模型打分选库。
        let zoneDescriptions = zones.map { zone -> String in
            let docs = (try? zoneEngines[zone.id]?.listDocuments(limit: 8).names) ?? []
            return "知识分区【\(zone.name)】：代表文档 \(docs.joined(separator: "、"))"
        }
        if Task.isCancelled { throw CancellationError() }
        // 与单库 decisionSearch 一致的墙钟预算，避免模型 forward 卡住拖垮整轮。
        let timeout = config.ragQueryTimeoutSeconds
        let budget = timeout == 0 ? Double.greatestFiniteMagnitude : max(30, timeout * 3)
        let zoneScores = try await RAGEngine.withTimeout(budget) {
            try await scorer.score(query: stateText, passages: zoneDescriptions,
                                   coreSentenceCount: 3, useCoreSentences: false)
        }
        AgentLog.info("[decision-multi] ①选库判定完成：\(zip(zones, zoneScores).map { "\($0.0.name)=\(String(format: "%.3f", $0.1))" }.joined(separator: ", "))")
        var selected: [(zone: KnowledgeZone, weight: Double)] = []
        for (i, zone) in zones.enumerated() {
            let s = i < zoneScores.count ? zoneScores[i] : 0
            if s >= 0.4 { selected.append((zone, s)) }
        }
        // 只选中 0/1 个分区时保底扩展到得分最高的 2 个：多分区场景应体现
        // 「联合检索」，避免判定阈值把范围收窄成单库（看起来像没启用多库）。
        let rankedZones = zones.enumerated().sorted { i, j in
            let si = i.offset < zoneScores.count ? zoneScores[i.offset] : 0
            let sj = j.offset < zoneScores.count ? zoneScores[j.offset] : 0
            return si > sj
        }
        if selected.count < 2, rankedZones.count > 1 {
            var seen = Set(selected.map { $0.zone.id })
            for candidate in rankedZones where seen.count < 2 {
                if seen.contains(candidate.element.id) { continue }
                let own = candidate.offset < zoneScores.count ? zoneScores[candidate.offset] : 0
                selected.append((candidate.element, max(0.01, own)))
                seen.insert(candidate.element.id)
            }
        }
        if selected.isEmpty {
            // 全不达标时保守选全部（避免漏检），权重用各分区自己的判定分+0.01。
            selected = zones.enumerated().map { i, zone in
                let own = i < zoneScores.count ? zoneScores[i] : 0
                return (zone, own + 0.01)
            }
        }
        AgentLog.info("[decision-multi] 选库 \(selected.map { "\($0.zone.name):\($0.weight)" }.joined(separator: ","))")

        // ② 各选中分区 FTS 粗筛 50。
        let perZone = config.decisionCoarseCandidates
        let coarseResults: [(zone: KnowledgeZone, weight: Double, hits: [SearchHit])] =
            selected.compactMap { item in
                guard let engine = zoneEngines[item.zone.id] else { return nil }
                let hits = (try? engine.decisionCoarseFTS(query: query, topK: perZone,
                                                          source: source)) ?? []
                return hits.isEmpty ? nil : (item.zone, item.weight, hits)
            }
        AgentLog.info("[decision-multi] ②粗筛：\(coarseResults.map { "\($0.zone.name)=\($0.hits.count)" }.joined(separator: ", "))")
        guard !coarseResults.isEmpty else {
            return RAGRetrievalOutcome(context: "（知识库中未找到相关内容）", citations: [])
        }
        // 加权 RRF 融合取前 20。
        let zoneResults = coarseResults.map { item in
            MultiZoneRetrieval.ZoneResult(
                zoneID: item.zone.id,
                zoneName: item.zone.name,
                citations: item.hits.prefix(perZone).map { hit in
                    RAGCitation(id: "\(item.zone.id):\(hit.rowID)", rank: 0,
                                source: (hit.source as NSString).lastPathComponent,
                                heading: hit.heading, text: hit.text,
                                score: hit.score, isPrimary: false)
                },
                weight: item.weight)
        }
        let fused20 = Array(MultiZoneRetrieval.fuse(zoneResults,
                                                     topK: max(20, topK),
                                                     rrfK: config.ragRRFK).prefix(20))
        guard !fused20.isEmpty else {
            return RAGRetrievalOutcome(context: "（知识库中未找到相关内容）", citations: [])
        }
        AgentLog.info("[decision-multi] ③加权 RRF 融合：\(fused20.count) 条候选进入判定")
        // ③ 第二次判定：按核心句评判 20 个候选，取 Top3（或 decisionTopN）。
        if Task.isCancelled { throw CancellationError() }
        let scoreStarted = Date()
        let passages = fused20.map(\.text)
        let scores = try await RAGEngine.withTimeout(budget) {
            try await scorer.score(query: query, passages: passages,
                                   coreSentenceCount: self.config.decisionCoreSentenceCount,
                                   useCoreSentences: self.config.decisionUseCoreSentences)
        }
        AgentLog.info("[decision-multi] ④核心句判定完成 \(scores.count) 条，"
            + "耗时 \(Int(Date().timeIntervalSince(scoreStarted) * 1000))ms")
        let n = max(1, min(config.decisionTopN, fused20.count))
        let ranked = zip(fused20, scores).sorted { $0.1 > $1.1 }.prefix(n)
        let citations = ranked.enumerated().map { index, pair in
            var c = pair.0
            c.rank = index + 1
            c.score = pair.1
            c.isPrimary = index == 0
            return c
        }
        AgentLog.info("[decision-multi] 完成：返回 Top\(citations.count) "
            + "\(citations.map { "\($0.source):\(String(format: "%.3f", $0.score))" }.joined(separator: ", "))，"
            + "总耗时 \(Int(Date().timeIntervalSince(started) * 1000))ms")
        return RAGRetrievalOutcome(context: MultiZoneRetrieval.primaryContext(citations),
                                   citations: citations)
    }

    public func close() {
        takeBackgroundTask()?.cancel()
        mcpTaskLock.lock()
        mcpClosed = true
        mcpTaskGeneration += 1
        mcpConnectTask?.cancel()
        mcpConnectTask = nil
        mcpTaskLock.unlock()
        tools.close()
        for client in mcpClients { client.shutdown() }
        mcpClients.removeAll()
        skillRetriever.close()
        ragEngine.close()
        // 额外分区 engine 也要关闭；跳过 ragEngine 本身避免重复 close。
        for engine in zoneEngines.values where engine !== ragEngine { engine.close() }
        GraphMemoryStore.flushShared()
        GraphMemoryStore.releaseShared(workspace: config.workspace)
        (llm as? ClosableLLMService)?.close()
        if let auxLLM, auxLLM !== llm {
            (auxLLM as? ClosableLLMService)?.close()
        }
    }

    /// 退出应用前的有序收尾：等待后台任务与 GPU 工作真正结束，再释放模型。
    ///
    /// `close()` 只是发起卸载；若进程紧跟着退出，MLX 仍在执行的 asyncEval
    /// 会与进程退出时销毁 command encoder 并发，触发 Metal 断言 / SIGSEGV。
    public func shutdownForTermination() async {
        if let background = takeBackgroundTask() {
            background.cancel()
            // 退出流程也不能被不响应取消的辅助模型无限拖住。
            let exited = await Self.waitForTaskExit(background,
                                                   timeoutNanos: 10_000_000_000)
            if !exited {
                AgentLog.warning("[background] 退出时后台任务 10s 未结束，继续卸载模型")
            }
        }
        mcpTaskLock.lock()
        mcpClosed = true
        mcpTaskGeneration += 1
        mcpConnectTask?.cancel()
        mcpConnectTask = nil
        mcpTaskLock.unlock()
        tools.close()
        for client in mcpClients { client.shutdown() }
        mcpClients.removeAll()
        skillRetriever.close()
        ragEngine.close()
        for engine in zoneEngines.values where engine !== ragEngine { engine.close() }
        if let service = llm as? AsyncUnloadableLLMService {
            await service.unloadAndWait()
        } else {
            (llm as? ClosableLLMService)?.close()
        }
        if let auxLLM, auxLLM !== llm {
            if let service = auxLLM as? AsyncUnloadableLLMService {
                await service.unloadAndWait()
            } else {
                (auxLLM as? ClosableLLMService)?.close()
            }
        }
        GraphMemoryStore.flushShared()
        GraphMemoryStore.releaseShared(workspace: config.workspace)
    }

    /// 卸载辅助模型，保留主模型与当前会话。
    public func unloadAuxLLM() {
        (auxLLM as? ClosableLLMService)?.close()
        auxLLM = nil
    }

    public func switchLLM(_ newLLM: LLMService) {
        if llm !== newLLM {
            (llm as? ClosableLLMService)?.close()
        }
        llm = newLLM
        cachedDynamicContext = nil
        invalidateKVCache()
    }

    public func invalidateKVCache() {
        // 本地后端持有跨轮 KV cache；历史被折叠/回滚或换模型时丢弃。
        (llm as? KVCacheInvalidating)?.invalidateKVCache()
    }

    public func reloadRAG() {
        ragEngine.invalidateStatusCache()
        ragEngine.reload()
    }

    public func history() -> [ChatMessage] {
        session.buildMessages(systemPrompt: systemPrompt())
    }

    // MARK: 主入口

    @discardableResult
    public func run(_ userInput: String, images: [Data] = []) async -> String {
        // 底层组件（MLX/GGUF/RAG/解析）的 AgentLog 输出按本次请求路由到自己的
        // callbacks，而不是进程级全局 sink：并发会话/测试之间不会互相串日志。
        // detached 后台任务不继承 taskSink，仍由全局 sink 兜底。
        await AgentLog.withTaskSink({ [weak self] line in
            self?.callbacks.onLog(line)
        }) {
            await self.runInternal(userInput, images: images)
        }
    }

    /// 以「独立任务」方式运行一次（供定时任务使用）。
    ///
    /// 定时任务在后台无人值守执行，不能续跑用户此前遗留的计划：运行前把
    /// 计划（内存 + `plan.json`）暂存并清空，结束后恢复。否则 9:00 的提醒
    /// 可能被模型理解成「继续上次未完成的步骤」而执行错误动作；运行结束时
    /// 也会丢弃定时任务自己建立的计划，避免污染用户接下来的对话。
    @discardableResult
    public func runIsolated(_ userInput: String) async -> String {
        let savedPlan = PlanStore.load(config) ?? session.getPlan()
        let savedBoundary = session.working["__plan_boundary__"]
        let epoch = planEpoch
        if savedPlan != nil {
            session.clearPlan()
            PlanStore.clear(config)
            session.working.removeValue(forKey: "__plan_boundary__")
            callbacks.onPlanUpdated(nil)
        }
        defer {
            // 执行期间用户手动清空过计划：尊重用户操作，不要恢复旧快照。
            if planEpoch == epoch {
                if let savedPlan {
                    try? PlanStore.save(savedPlan, config: config)
                    session.setPlan(savedPlan)
                    if let savedBoundary {
                        session.working["__plan_boundary__"] = savedBoundary
                    }
                    callbacks.onPlanUpdated(savedPlan)
                } else {
                    session.clearPlan()
                    PlanStore.clear(config)
                    session.working.removeValue(forKey: "__plan_boundary__")
                    callbacks.onPlanUpdated(nil)
                }
            }
        }
        return await run(userInput)
    }

    private func runInternal(_ userInput: String, images: [Data]) async -> String {
        // 前台请求优先：上一轮的后台摘要/事实抽取还在跑时，先取消并等它退出，
        // 否则 2B 辅助模型会和主模型抢 GPU（实测主模型会从 ~15 tps 掉到 ~5 tps）。
        await cancelBackgroundWork()
        turnIndex += 1
        cachedDynamicContext = nil
        // 新用户请求：动态上下文重新注入一次（内容由本行之后的缓存重建）。
        kernel.beginUserRequest()
        clearFinishedTurnSnapshot()
        freezeRequestSystemPrompt()  // 段内冻结 system 前缀；落盘也写这一份
        lastUserText = userInput
        log("USER: \(userInput)")

        // 确保首轮就能看到 MCP 工具（配置了服务时）。
        await connectMCP()

        // 只有非寒暄/简单问询才进入技能固化统计；是否真正提示还要看本轮
        // 是否用过计划（见本次 run 末尾），避免简单查询被提示保存技能。
        let skillCandidate: Skill? = SkillManager.isTrivialRequest(userInput)
            || SkillManager.isCapabilityQuery(userInput)
            ? nil
            : skills.recordRequest(userInput)

        // 跨会话恢复：只在 Session 尚无 key_info 时恢复持久化检查点。
        if session.getKeyInfo().isEmpty {
            let checkpoint = CheckpointStore.readPersisted(config)
            if !checkpoint.isEmpty {
                session.setKeyInfo(checkpoint)
            }
        }

        // 任务边界：已收尾的旧计划清空；未收尾的注入一次性判断提示。
        // 寒暄轮/能力问询轮既不触碰计划，也不提供任何工具——直接文本回复。
        let isGreeting = SkillManager.isGreeting(userInput)
        let lightweightTurn = isGreeting || SkillManager.isCapabilityQuery(userInput)
        if session.getPlan() == nil, let stored = PlanStore.load(config) {
            session.setPlan(stored)
        }
        noteSkillLearningSignal(userInput, lightweight: lightweightTurn)
        if session.getPlan() != nil, !lightweightTurn {
            if session.planOpenSteps().isEmpty {
                session.clearPlan()
                PlanStore.clear(config)
                log("[plan] 上一任务已收尾，清空计划以便重新规划")
            } else {
                let open = session.planOpenSteps().prefix(5)
                    .map { "\($0.id).\($0.title)" }.joined(separator: "; ")
                session.working["__plan_boundary__"] = .string(
                    "上一个任务的计划还有未完成步骤：\(open)。请先判断本条消息的意图："
                    + "若是一个**新任务**（与旧计划的目标无关），立即用 update_plan"
                    + "(op=\"create\", goal=..., steps=[...]) 建立新计划替换旧计划，"
                    + "不要沿用无关步骤；若是要求**继续**上一个任务，则从剩余步骤接着执行。")
                log("[plan] 上一任务计划未收尾，已注入新任务/继续任务的判断提示")
            }
        }
        callbacks.onPlanUpdated(session.getPlan())

        // 本轮是否真正创建/更新过计划：寒暄、简单问询、单次查询不会变，
        // 因此不触发技能固化建议。
        let planRevisionBefore = session.planRevision
        kernel.maxTurns = lightweightTurn ? 2 : maxIterations
        var result = await kernel.run(userInput, resume: false,
                                      emitFinished: false, toolsEnabled: !lightweightTurn,
                                      images: images)
        // distillRequested 是「本段内调用过 start_long_term_update」；
        // 自动续跑会重置它，因此必须在门面层跨段累计。
        var distillRequested = result.distilled
        var carry = kernel.failureSnapshot()

        // ---- 自动续跑：消灭「必须人工输入继续」 ----
        var segments = 0
        while segments < autoContinueMax,
              autoContinue,
              (result.reason == .maxTurns ||
               (result.reason == .done && !session.planOpenSteps().isEmpty)),
              !callbacks.isCancelled() {
            segments += 1
            kernel.seedFailures(carry)
            let open = session.planOpenSteps().prefix(5)
                .map { "\($0.id).\($0.title)" }.joined(separator: "; ")
            var failText = ""
            if let lastFail = carry.lastFail {
                failText = "\n上一段最后一次失败：`\(lastFail.name)` → "
                    + "\(lastFail.error.prefix(200))。请先读完整错误定位原因，再换一种"
                    + "**没试过**的方法（换工具 / 换参数 / 用 code_run 写最小诊断脚本验证假设）。"
                    + "不要原样重试，也不要只回答文字。"
            }

            if result.reason == .maxTurns {
                log("[agent] 达到轮次上限，自动续跑 \(segments)/\(autoContinueMax)")
                // 最后一段续跑必须强制收尾：2B 模型拿到结果后容易一直调工具
                // 不总结，把 3 段额度全部耗光，用户等 8 分钟仍拿不到答复。
                if segments == autoContinueMax {
                    session.working["__truncated__"] = .string(
                        "[系统提示] 这是**最后一次**自动续跑。**不要再调用任何工具**："
                        + "基于已获得的工具结果直接输出最终答复"
                        + "（<summary>…</summary> 加正文）。确实无法完成的步骤，"
                        + "在答复里说明原因即可。")
                } else {
                    session.working["__truncated__"] = .string(
                        "[系统提示] 上一段在达到轮次上限（\(maxIterations)）后被自动续跑。"
                        + "请先用一句话总结当前进展，然后**立即继续执行**计划中的剩余步骤；"
                        + "若全部步骤已完成，用 update_plan 收尾并给出最终答复。\(failText)")
                }
                result = await kernel.run(
                    "[System] 任务在上一段达到轮次上限被截断，已自动续跑。"
                    + (segments == autoContinueMax
                       ? "这是最后一次续跑：不要再调用工具，立即基于已有结果给出最终答复。"
                       : "请继续执行计划中的剩余步骤；不要询问用户是否继续。"),
                    resume: true, emitFinished: false,
                    toolsEnabled: !lightweightTurn)
            } else {
                log("[agent] 模型提前收尾但计划未完成（\(open)），自动续跑 "
                    + "\(segments)/\(autoContinueMax)")
                session.working["__plan_boundary__"] = .string(
                    "你上一条回复被判为提前收尾，但计划中仍有未完成步骤：\(open)。\(failText)\n"
                    + "若这些步骤的工作**实际已经完成**，立即用 update_plan"
                    + "(op=\"patch\", step_id=..., status=\"done\", note=\"结论\") 逐一收尾，"
                    + "然后给出最终答复；若尚未完成，继续执行；"
                    + "只有至少 3 种不同方法都确认不可行，才标为 blocked/skipped "
                    + "并写明原因后再给出最终答复。不要再次只给出文字总结就停止。")
                result = await kernel.run(
                    "[System] 上一条回复被判定为提前收尾，已自动续跑。"
                    + "请按上面的指示处理计划中的剩余步骤；不要询问用户是否继续。",
                    resume: true, emitFinished: false,
                    toolsEnabled: !lightweightTurn)
            }
            distillRequested = distillRequested || result.distilled
            carry = kernel.failureSnapshot()
        }

        if result.reason == .maxTurns {
            if segments > 0 {
                log("[agent] 轮次上限且自动续跑额度用尽，停车等待用户指示。")
                session.working["__truncated__"] = .string(
                    "[系统提示] 任务已连续 \(segments + 1) 段达到最大轮次上限"
                    + "（每段 \(maxIterations) 轮），已自动停止。"
                    + "请先总结已完成的进展与剩余工作，等待用户指示后再继续。")
            } else {
                session.working["__truncated__"] = .string(
                    "[系统提示] 上一轮在达到最大轮次上限（\(maxIterations)）后被中断，"
                    + "任务可能尚未完成。请先总结已完成的进展，再继续执行剩余步骤。")
            }
        }

        var text = result.text
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            // 按终止原因给出准确文案，避免把取消/报错都误报成「连续空响应」。
            switch result.reason {
            case .cancelled:
                text = "（已取消本次生成。）"
            case .error:
                text = "（生成失败，未收到模型回复。请检查网络/API 配置后重试，"
                    + "或查看日志中的错误详情。）"
            case .done:
                text = "（模型多次重试后仍未返回可见正文，已停止。"
                    + "请换个说法再试，或检查模型的输出设置。）"
            case .maxTurns, .interrupt:
                text = "（已达到本轮最大轮次上限，本轮自动停止，尚未取得可见结果。）"
            }
            log("[agent] 空回复（reason=\(result.reason.rawValue)）：已向用户返回可见提示")
        }
        // 自动续跑额度用尽后，旧实现只把状态写进 session.working（用户
        // 看不见），前台依旧像「卡住了没结果」。把停车原因追加到可见答复。
        if result.reason == .maxTurns && segments > 0 {
            text += "\n\n---\n（已连续 \(segments + 1) 段达到轮次上限，系统已自动停止；"
                + "以上为本轮已取得的结果，剩余步骤需要你再发一句继续。）"
        }
        // 技能固化只针对「建过多步骤计划且已收尾」或「显式触发长期蒸馏」的任务。
        // 寒暄、简单查询、未完成的任务即使重复出现，也不会再提示保存技能。
        let planTouched = session.planRevision != planRevisionBefore
        let plan = session.getPlan()
        let planDone = (plan?.steps.count ?? 0) >= 2
            && (plan?.steps.allSatisfy { $0.status == "done" } ?? false)
        let substantive = distillRequested || (planTouched && planDone)
        // 只有「计划至少 3 步 + 用户中途纠正/补充 + 全部完成」才建议固化；
        // 一轮内直接做完的任务、定时任务都不进入学习。
        if !tools.isUnattended,
           session.isSkillLearningEligible(),
           let skillCandidate, skills.match(userInput) == nil {
            let signal = session.skillLearningSignal()
            AgentLog.info("[skill] 建议固化 name=\(skillCandidate.name) "
                + "steps=\(plan?.steps.count ?? 0) interventions=\(signal.guidanceTurns) "
                + "corrected=\(signal.corrected) supplemented=\(signal.supplemented)")
            callbacks.onSkillSuggested(SkillSuggestion(
                name: skillCandidate.name,
                description: skillCandidate.description,
                score: Double(skillCandidate.keywords.count)))
        }
        // 改进1/2：本用户请求的完整轨迹落盘（无条件、不依赖辅助模型）。
        // 用 `finishedTurnSnapshot`（早于 CONSOLIDATE 收集）而非 session.turns，
        // 否则被折掉的轮全文会永久丢失。
        // `requestSystemPrompt` 是本折叠段冻结的那一份（本轮模型实际看到的前缀），
        // 不能用请求结束后的 systemPrompt()：本轮若 patch 过 L1，签名已变，写进去
        // 的会是"未来"的提示，历史与实发不一致。
        // 不再在请求末尾释放：段内跨请求复用同一前缀，直到折叠边界才刷新。
        rawSessionStore.appendSession(
            systemText: takeRequestSystemPrompt() ?? systemPrompt(),
            userText: lastUserText.isEmpty ? userInput : lastUserText,
            turns: takeFinishedTurnSnapshot(),
            finalText: text)

        // 改进7 前半：过期记录预处理/打包（纯文件，单次最多 1 天）。
        // 归档只处理「今天之前」的文件，与上面的当天追加写不冲突，因此
        // 放到后台串行链：它会读最多 2000 万字符并 zip 压缩，旧实现同步
        // 卡在 run 返回前，直接拉长「本轮结束 → 用户可再发送」的间隔。
        scheduleArchive()

        // 寒暄/能力问询跳过事实抽取与蒸馏，但**记忆维护仍要跑**：它消费的是
        // 持久化队列，否则这些轮归档的 session 会一直挂在队列里等不到提炼。
        fireFinished(text: text, distillRequested: distillRequested,
                     substantive: substantive, isGreeting: lightweightTurn)
        callbacks.onFinished()
        return text
    }

    // MARK: 系统提示

    private func clearFinishedTurnSnapshot() {
        rawSessionSnapshotLock.lock()
        finishedTurnSnapshot.removeAll()
        rawSessionSnapshotLock.unlock()
    }

    /// 段内（两次折叠之间）system 前缀逐字节冻结，只有折叠边界才重新读
    /// L0/L1：否则 L1 在段中被 patch 会让头部变化 → KV 全量重建。
    private func freezeRequestSystemPrompt(force: Bool = false) {
        requestSystemLock.lock()
        let needsRebuild = force
            || frozenSystemEpoch != systemFoldEpoch
            || requestSystemPrompt == nil
        requestSystemLock.unlock()
        guard needsRebuild else { return }
        let frozen = buildStaticSystemPrompt()
        requestSystemLock.lock()
        requestSystemPrompt = frozen
        frozenSystemEpoch = systemFoldEpoch
        requestSystemLock.unlock()
    }

    /// 折叠边界（周期性 / 90% / 提前折叠 / 异常重试）：
    /// ① L0/L1 变更到此才生效，段内 system 前缀保持恒定（C）；
    /// ② 用新 system 预热 KV，让折叠后的首轮只 prefill 增量而非全量（D）。
    private func advanceSystemFoldEpoch() {
        requestSystemLock.lock()
        systemFoldEpoch += 1
        requestSystemLock.unlock()
    }

    func onFoldBoundary(reason: String, tools: [JSONValue]) async {
        // 事实抽取已改为按 4h 增量读落盘的 raw_session 文件，不依赖折叠边界；
        // 被清掉的旧轮全文早由 RawSessionStore 逐轮落盘，不会丢。
        advanceSystemFoldEpoch()
        freezeRequestSystemPrompt(force: true)
        guard let warmup = llm as? WarmupLLMService else { return }
        let current = takeRequestSystemPrompt() ?? buildStaticSystemPrompt()
        do {
            // 只预热永不变化的 system + tools 前缀。earlier_context 每次折叠
            // 都会变，交给 chatStream 从快照复制前缀、只 prefill 增量。
            try await warmup.warmup(systemPrompt: current, tools: tools)
            log("[context] 折叠后已预热 system 前缀（\(reason)）")
        } catch {
            AgentLog.warning("[context] 折叠后预热失败（\(reason)）："
                + error.localizedDescription)
        }
    }

    private func takeRequestSystemPrompt() -> String? {
        requestSystemLock.lock(); defer { requestSystemLock.unlock() }
        return requestSystemPrompt
    }

    private func markMemoryTidyDone(turn: Int) {
        tidyStateLock.lock()
        lastMemoryTidyTurn = turn
        tidyStateLock.unlock()
    }

    private func readLastMemoryTidyTurn() -> Int {
        tidyStateLock.lock(); defer { tidyStateLock.unlock() }
        return lastMemoryTidyTurn
    }

    private func takeFinishedTurnSnapshot() -> [TurnRecord] {
        rawSessionSnapshotLock.lock()
        defer { rawSessionSnapshotLock.unlock() }
        let snapshot = finishedTurnSnapshot
        finishedTurnSnapshot.removeAll()
        return snapshot
    }

    /// 稳定的系统提示 = BASE 协议 + L0 META-SOP + L1 全局索引。
    ///
    /// 改进3/4：L0 与 L1 **常驻** system prompt，L1 跟在 L0 之后。
    /// 段内冻结：两次折叠之间的所有 THINK 轮（含跨用户请求）都拿到逐字节
    /// 相同的 system 字符串，即使中途 patch 了 L1 也不会让 KV 前缀抖动。
    /// L0/L1 变更延迟到折叠边界，由 `onFoldBoundary` 按文件签名重建并预热；
    /// 文件变 → 前缀变 → 该轮 KV 失效一次（低频、可预期，且本来就要重建）。
    public func systemPrompt() -> String {
        if let frozen = takeRequestSystemPrompt() { return frozen }
        return buildStaticSystemPrompt()
    }

    /// 按 L0/L1 文件内容签名重建 system 前缀。签名计算与缓存更新放在同一临界区，
    /// 避免并发调用缓存出「旧签名 + 新内容」的不一致组合。
    private func buildStaticSystemPrompt() -> String {
        staticSystemLock.lock()
        defer { staticSystemLock.unlock() }
        let signature = memoryDirectory.signature()
        if signature == cachedStaticSignature, let cachedStaticSystem {
            return cachedStaticSystem
        }
        let sop = memoryDirectory.loadSOP(maxChars: SystemPromptLimits.maxSOPChars)
        let l1 = memoryDirectory.loadL1(maxChars: SystemPromptLimits.maxL1Chars)
        var text = baseSystemPrompt
        if !sop.isEmpty {
            text += "\n\n# [L0 META-SOP: memory_management_sop]\n" + sop
        }
        if !l1.isEmpty {
            text += "\n\n# [L1 Global Memory Insight]\n" + l1
        }
        cachedStaticSystem = text
        cachedStaticSignature = signature
        return text
    }

    /// 动态上下文（工作记忆、当天事实、检查点、命中技能、知识库状态）。
    ///
    /// 由 `KernelLoop` 追加到当前轮 user 消息尾部，而不是放进 system 消息：
    /// 这样动态内容只影响提示词尾部，前面的大段历史仍能命中 KV cache。
    /// 同一次用户请求内缓存一次，避免每轮工具循环都改写前缀。
    public func dynamicContextPrompt() -> String {
        if let cachedDynamicContext { return cachedDynamicContext }
        var sections: [String] = []

        // 检查点常驻注入：优先 Session（本轮工具实时更新），其次持久化文件。
        let checkpoint = session.getKeyInfo().isEmpty
            ? CheckpointStore.readPersisted(config)
            : session.getKeyInfo()
        if !checkpoint.isEmpty {
            sections.append("# Working checkpoint\n\(checkpoint.prefix(4_000))")
        }

        // 命中技能：截断并提示用 skill_tools(op="load") 读全文。
        // 至少 3 分才注入：名称命中(3) 或关键词/tags 命中(2)+描述，
        // 单个常见词或两个描述词命中不足以代表任务与技能相关，
        // 误注入错误技能会污染上下文（正确技能可由 semantic_search 召回）。
        if !lastUserText.isEmpty, let matched = skills.matchScored(lastUserText),
           matched.score >= 3 {
            let skill = matched.skill
            var prompt = skill.body
            if prompt.count > SystemPromptLimits.maxSkillPromptChars {
                prompt = String(prompt.prefix(SystemPromptLimits.maxSkillPromptChars))
                    + "\n…（已截断，用 skill_tools(op=\"load\") 读取完整技能内容）"
            }
            sections.append("# Active skill: \(skill.name)\n\(prompt)")
            AgentLog.info("[skill] 自动注入 name=\(skill.name) score=\(matched.score) "
                + "chars=\(prompt.count) dir=\(skill.directory)")
        }

        // 知识库状态只读各 RAGEngine 的内存快照：绝不在这里同步打开/扫描向量库。
        // 千万级切片的库上 COUNT(*) 需要数分钟，会把首轮 LLM 调用卡住。
        // 多分区时汇总所有启用分区，明确引导模型做跨库联合检索、不要按单一
        // 文档名限定 source（否则只有默认库命中，其它库被过滤空）。
        let mounted = config.mountedZones
        if mounted.count > 1 {
            var lines: [String] = ["# Knowledge base（\(mounted.count) 个已挂载知识分区，检索默认覆盖全部分区）"]
            var totalChunks = 0
            for zone in mounted {
                guard let engine = zoneEngines[zone.id],
                      let st = engine.cachedStatus(maxAge: .infinity) else { continue }
                let chunks = Int(st["chunks_stored"]?.intValue ?? 0)
                let sources = st["sources"]?.arrayValue?.compactMap(\.stringValue) ?? []
                let cnt = Int(st["source_count"]?.intValue ?? sources.count)
                totalChunks += chunks
                let names = sources.prefix(5).map { ($0 as NSString).lastPathComponent }
                    .joined(separator: ", ")
                lines.append("- 分区【\(zone.name)】(id=\(zone.id))：\(chunks) 切片 / \(cnt) 文档"
                    + (names.isEmpty ? "" : "（\(names)）"))
            }
            if totalChunks == 0 {
                lines.append("（各分区尚无已索引内容，请先同步）")
            }
            lines.append(KNOWLEDGE_BASE_HINT)
            lines.append("多分区默认跨全部分区联合检索，不要传 source 限定到单个文档名。")
            sections.append(lines.joined(separator: "\n"))
        } else if let status = ragStatus() {
            let chunks = Int(status["chunks_stored"]?.intValue ?? 0)
            let sources = status["sources"]?.arrayValue?.compactMap(\.stringValue) ?? []
            let sourceCount = status["source_count"]?.intValue ?? sources.count
            if chunks > 0 {
                let shown = min(8, sources.count)
                var names = sources.prefix(shown).map { ($0 as NSString).lastPathComponent }
                    .joined(separator: ", ")
                if sourceCount > shown { names += " (and \(sourceCount - shown) more)" }
                sections.append("# Knowledge base\n\(chunks) chunks in \(sourceCount) file(s): "
                    + "\(names). " + KNOWLEDGE_BASE_HINT)
            } else {
                sections.append("# Knowledge base\nConfigured but empty. Use `rag_tools(op=\"ingest\")` to index.")
            }
        } else if FileManager.default.fileExists(atPath: ragEngine.indexDBPath) {
            sections.append("# Knowledge base\nIndexed locally. " + KNOWLEDGE_BASE_HINT)
        }

        let text = sections.joined(separator: "\n\n")
        cachedDynamicContext = text
        return text
    }

    private func ragStatus() -> [String: JSONValue]? {
        // 快照由 AppModel 启动后/同步后刷新，写操作会主动失效；轮次路径只读
        // 缓存且不设过期时间，避免长对话中途丢统计后反复走「无统计」提示。
        ragEngine.cachedStatus(maxAge: .infinity)
    }

    // MARK: 上下文 >90% 自动收缩（带回水线的摘要回注路径）

    /// 触发线 90%（`Budget.needsAutoShrink`），收缩目标 = 可用预算的 60%
    ///（触发线下 30 个百分点回水）：`foldToRewatermark` 从最老轮逐轮折叠、
    /// 每轮摘要行回注 `<earlier_context>`，降到位即停——收缩一次到位，
    /// 避免贴着触发线反复收缩、反复付 KV 重建。全程不调 LLM（逐轮摘要
    /// 早在响应时存入 `TurnRecord.summary`）。
    /// 周期性折叠走 `discardHistory`，同样回注摘要行，两条路径都不调模型。
    func shrinkSession(reason: String) async -> Bool {
        guard session.turns.count > 1 else { return false }
        let foldedLines = session.foldToRewatermark(targetRatio: 0.60, keepingRecent: 1)
        guard !foldedLines.isEmpty else { return false }
        // memory.md / UI 展示用的折叠总览（与回注行同源，只读不改上下文）。
        let final = l3Fold(foldedLines, maxLines: session.maxDigestLines)
        appendMemorySummary(final, reason: reason)
        // 回注区有字数上限：保头保尾，避免只留最老几轮、丢掉最近的结论。
        let brief = final.count <= 400
            ? final
            : String(final.prefix(200)) + "\n…\n" + String(final.suffix(180))
        let digestLine = "[摘要 \(Self.timeString())] \(brief)"
        // 折叠摘要追加进当天 digest 文件，保证"当天对话摘要历史"不缺项。
        rawSessionStore.appendFoldedDigest([digestLine])
        log("[context] 自动收缩：折叠 \(foldedLines.count) 轮（回水目标 60%），"
            + "摘要行已回注 earlier_context（reason: \(reason)）")
        callbacks.onContextShrunk(final, reason)
        return true
    }

    private func appendMemorySummary(_ summary: String, reason: String) {
        let path = (config.workspace as NSString).appendingPathComponent("memory.md")
        let existing = (try? String(contentsOfFile: path, encoding: .utf8)) ?? ""
        let entry = "\n## 上下文摘要 - \(Self.timestamp())\n- 触发原因: \(reason)\n\n\(summary)\n"
        try? AtomicFile.write(existing + entry, to: path)
    }

    // MARK: update_working_checkpoint 超长压缩

    /// 便签超过 `CheckpointStore.maxKeyInfo` 字符时的一次有损压缩（工具经
    /// ToolFactory 注入此入口）。与其它后台任务同策略：辅助模型优先、主模型
    /// isolated 兜底；失败/不可用返回 nil，由工具侧走保头保尾兜底。
    /// 工具执行点在主模型两轮生成之间，GPU 空闲，可直接用 `collectChat`；
    /// `beginBackgroundIfIdle` 顺带挡住与后台记忆任务并发抢辅助模型。
    func compressCheckpointText(_ text: String) async -> String? {
        guard beginBackgroundIfIdle() else { return nil }
        defer { endBackground() }
        let system = "你负责压缩 Agent 的工作便签。只输出压缩后的便签正文，不要任何解释或前后缀。"
        let user = """
        把下面的工作便签压缩到 \(CheckpointStore.maxKeyInfo) 字符以内（硬上限）。
        保留全部具体事实：已验证的路径/ID/端口/命令、决策及其理由、已排除的方案、
        下一步计划。删除重复表述、流水账和与当前任务无关的内容。只输出压缩后的正文。
        --- 便签 ---
        \(text)
        """
        let output = await collectChat(
            [ChatMessage(role: .system, content: system),
             ChatMessage(role: .user, content: user)],
            temperature: 0.1)
        let compressed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !compressed.isEmpty else { return nil }
        return compressed
    }

    // MARK: 结束后：后台事实抽取 + L4 蒸馏

    /// 只有计划仍在进行时用户追加的消息才算「引导」；计划已完成则重置，
    /// 避免下一轮寒暄或新任务继承旧信号。
    private func noteSkillLearningSignal(_ input: String, lightweight: Bool) {
        guard !lightweight else { return }
        guard let plan = session.getPlan(), !session.planOpenSteps().isEmpty else {
            session.resetSkillLearningSignal()
            return
        }
        let corrective = SkillManager.isCorrectiveFeedback(input)
        let supplemental = !corrective && SkillManager.isSupplementalFeedback(input)
        session.noteSkillGuidance(planGoal: plan.goal,
                                  corrective: corrective,
                                  supplemental: supplemental)
        let signal = session.skillLearningSignal()
        AgentLog.info("[skill] 记录介入 plan=\(String(plan.goal.prefix(60))) "
            + "turns=\(signal.guidanceTurns) corrected=\(signal.corrected) "
            + "supplemented=\(signal.supplemented)")
    }

    /// 事实抽取是**文件驱动的后台任务**：每 4 小时最多一次，增量读取当天
    /// `raw_session/<date>.txt` 里新增的完整 SESSION 块，只用辅助模型、只写
    /// memory graph。既不逐轮跑、也绝不回退主模型，因此不会与前台抢 GPU。
    /// 其它后台工作（历史提炼 / 记忆整理 / L4 蒸馏）仍按原策略执行。
    func fireFinished(text: String, distillRequested: Bool? = nil,
                      substantive: Bool = true,
                      isGreeting: Bool = false) {
        let user = lastUserText.trimmingCharacters(in: .whitespacesAndNewlines)
        // 寒暄/能力问询不做蒸馏；此处只判断本轮是否值得做 L4 经验蒸馏。
        let substantiveTurn = !isGreeting && user.count >= 6 && text.count >= 8

        // 改进7：待提炼队列是**持久化**的，寒暄轮、辅助模型暂不可用、后台任务
        // 被下一次 run 取消等情况都不会丢，本轮或下一轮重新拾起即可。
        let hasPending = !memoryDirectory.pendingInsightIds().isEmpty
        // 改进8：至少 8 轮做一次整理建议（与折叠周期对齐），
        // 后台用辅助引擎产出建议、不改记忆文件。
        let tidyEvery = max(8, config.kernelConsolidateEvery)
        let shouldTidy = !isGreeting && session.currentTurn >= readLastMemoryTidyTurn() + tidyEvery
        let shouldDistill = substantiveTurn && substantive
            && (distillRequested ?? kernel.distillRequested)
        // 事实抽取必须满足：有辅助模型（唯一允许的模型）+ 距上次成功 ≥4h
        // + 当天文件有新增的完整 session。
        let factDue = auxLLM != nil && factExtractionWorker.isDue()
        let hasLLMWork = factDue || hasPending || shouldTidy || shouldDistill
        guard hasLLMWork else { return }

        // 其余后台 LLM 工作影响整体功能：辅助优先、主模型 isolated 兜底；
        // 主模型连 isolated 都不支持时跳过（持久化队列保留）。事实抽取是
        // 唯一例外，上面已要求 auxLLM 存在。
        let service = auxLLM ?? llm
        if service === llm,
           Self.backgroundRequestOptions(for: service, isMain: true) == nil {
            log("[memory] 主模型不支持独立会话，跳过后台 LLM；待提炼队列已保留")
            return
        }

        let transcript = session.allDigestLines().suffix(25).joined(separator: "\n")
        backgroundTaskLock.lock()
        // 已有后台任务在跑：不 cancel 重建，避免前台每轮多过一次看门狗。
        if backgroundTask != nil {
            backgroundTaskLock.unlock()
            return
        }
        backgroundGeneration += 1
        let generation = backgroundGeneration
        // 同步记下这次调度：任务可能被下一次 run 立刻取消，不能在 4h 内
        // 反复重试（那会重新变成「每轮拉起后台模型」）。游标未推进，
        // 下一个窗口会把积压增量一次性读完。
        if factDue { factExtractionWorker.markAttempt() }
        let task = Task.detached(priority: .utility) { [weak self] in
            guard let self else { return }
            defer { self.finishBackgroundTask(generation) }
            if factDue { await self.runFactExtraction() }
            guard !Task.isCancelled else { return }
            if shouldDistill { await self.distillExperience(transcript: transcript) }
            guard !Task.isCancelled else { return }
            // 改进7：从持久化队列消费（不依赖本次归档结果），成功才出队。
            // 改进8：辅助引擎产出整理建议，不改写 L1/L2/L3。
            if hasPending || shouldTidy {
                let maintenance = MemoryMaintenance(
                    directory: self.memoryDirectory,
                    log: { [weak self] line in self?.log(line) },
                    collect: { [weak self] messages, temperature in
                        guard let self, self.beginBackgroundIfIdle() else { return "" }
                        defer { self.endBackground() }
                        return await self.collectChat(messages, temperature: temperature)
                    })
                if hasPending { await maintenance.mineHistoryInsights() }
                guard !Task.isCancelled else { return }
                if shouldTidy, await maintenance.tidyMemory() {
                    // 仅在整理**成功产出建议**后才推进节流标记；被取消 / 校验失败
                    // 会保留待办，下一轮重试，不会"标记已消费却没干活"。
                    self.markMemoryTidyDone(turn: self.session.currentTurn)
                }
            }
        }
        backgroundTask = task
        backgroundTaskLock.unlock()
    }

    /// 一次性 continuation 保护：`task.value` 与超时任务赛跑，先到者 resume。
    private final class ExitWaiter: @unchecked Sendable {
        private let lock = NSLock()
        private var resumed = false
        private let continuation: CheckedContinuation<Bool, Never>
        init(_ continuation: CheckedContinuation<Bool, Never>) {
            self.continuation = continuation
        }
        func finish(_ value: Bool) {
            lock.lock()
            guard !resumed else { lock.unlock(); return }
            resumed = true
            lock.unlock()
            continuation.resume(returning: value)
        }
    }

    /// 等待后台任务退出，带看门狗。辅助模型的一次生成可能正卡在 GPU 上、
    /// 迟迟不响应取消；无限 `await task.value` 会把新请求（甚至退出流程）卡死。
    /// 用两个非结构化任务赛跑，超时即可返回，返回 false 表示任务仍在跑。
    static func waitForTaskExit(_ task: Task<Void, Never>,
                                timeoutNanos: UInt64) async -> Bool {
        await withCheckedContinuation { cont in
            let waiter = ExitWaiter(cont)
            Task.detached {
                _ = await task.value
                waiter.finish(true)
            }
            Task.detached {
                try? await Task.sleep(nanoseconds: timeoutNanos)
                waiter.finish(false)
            }
        }
    }

    /// 取消上一轮的后台辅助任务，并等它让出 GPU（最多 3s）。
    private func cancelBackgroundWork() async {
        guard let task = takeBackgroundTask() else { return }
        let started = Date()
        task.cancel()
        let exited = await Self.waitForTaskExit(task, timeoutNanos: 3_000_000_000)
        let waited = Int(Date().timeIntervalSince(started) * 1000)
        AgentLog.info("[background] 取消上轮后台任务 waited=\(waited)ms exited=\(exited)")
        if !exited {
            AgentLog.warning("[background] 后台任务取消 3s 未退出，主请求不再等待")
        }
    }

    /// 取走并清空后台任务引用（同步方法，避免在 async 上下文直接操作 NSLock）。
    private func takeBackgroundTask() -> Task<Void, Never>? {
        backgroundTaskLock.lock()
        defer { backgroundTaskLock.unlock() }
        let task = backgroundTask
        backgroundTask = nil
        return task
    }

    /// 后台任务自然结束；只有仍是最新任务时才清空引用。
    private func finishBackgroundTask(_ generation: Int) {
        backgroundTaskLock.lock()
        if backgroundGeneration == generation { backgroundTask = nil }
        backgroundTaskLock.unlock()
    }

    /// 后台辅助任务是否已完全空闲（AppModel 空闲回收前检查）。
    public func isBackgroundIdle() -> Bool {
        backgroundTaskLock.lock()
        let hasTask = backgroundTask != nil
        backgroundTaskLock.unlock()
        if hasTask { return false }
        if backgroundLock.try() {
            backgroundLock.unlock()
            return true
        }
        return false
    }

    /// 在同步 helper 里调用 `try()`，避免 async 上下文直接使用 NSLock 的
    /// 非 async-safe 接口（Swift 6 会升级为错误）。
    private func beginBackgroundIfIdle() -> Bool {
        backgroundLock.try()
    }

    private func endBackground() {
        backgroundLock.unlock()
    }

    /// 一次后台增量抽取：文件游标取批次 → 辅助模型 → 写 graph → 成功才提交。
    /// 失败 / 被取消不推进游标，下次重跑（图谱写入本身幂等）。
    private func runFactExtraction() async {
        guard let batch = factExtractionWorker.takePending() else { return }
        let text = batch.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty {
            // 整段只有 system/tool、没有 USER/AGENT 正文：也要推进游标，
            // 否则这些空块会被之后的每一轮反复重读。
            factExtractionWorker.commit(batch)
            return
        }
        let done = await extractFacts(conversation: text)
        guard done, !Task.isCancelled else { return }
        factExtractionWorker.commit(batch)
        AgentLog.info("[memory] 事实抽取完成 sessions=\(batch.sessionCount) "
            + "offset=\(batch.newOffset)（4h 节流）")
    }

    /// 返回 true 表示这批对话已抽取过（模型有响应，无论是否解析出事实），
    /// 调用方据此推进文件游标；false = 被取消 / 模型无响应，保留待重试。
    private func extractFacts(conversation: String) async -> Bool {
        // 事实抽取只允许辅助模型：辅助不可用时直接放弃，绝不回退主模型，
        // 否则主模型 isolated 副本会全量 prefill 并与前台会话抢 GPU。
        guard auxLLM != nil else { return false }
        guard beginBackgroundIfIdle() else { return false }
        defer { endBackground() }
        let text = String(conversation.prefix(FactExtractionWorker.maxTranscriptChars))
        // collectChat 内部取 `auxLLM ?? llm`；上面的 guard 保证必走辅助模型。
        let raw = await collectChat([
            ChatMessage(role: .system, content: FACT_EXTRACT_SYSTEM),
            ChatMessage(role: .user, content: factExtractPrompt(conversation: text)),
        ], temperature: 0.1)
        guard !Task.isCancelled, !raw.isEmpty else { return false }
        let json = Self.stripJSONFence(raw)
        guard let value = JSONValue.parse(json), let data = value.objectValue else { return true }
        let draft = FactExtractionDraft(
            entities: (data["entities"]?.arrayValue ?? []).compactMap { item -> FactEntityDraft? in
                guard let object = item.objectValue else { return nil }
                return FactEntityDraft(
                    name: object["name"]?.stringValue ?? "",
                    type: object["type"]?.stringValue ?? "",
                    observations: (object["observations"]?.arrayValue ?? [])
                        .compactMap(\.stringValue))
            },
            relations: (data["relations"]?.arrayValue ?? []).compactMap { item -> FactRelationDraft? in
                guard let object = item.objectValue else { return nil }
                return FactRelationDraft(
                    source: object["source"]?.stringValue ?? "",
                    target: object["target"]?.stringValue ?? "",
                    label: object["label"]?.stringValue ?? "")
            })
        // 小模型会概率性犯规：空观察实体、泛化实体名、孤立关系一旦入库就是
        // 永久噪声（createRelation 还会自动补建空实体），必须在写入前过滤。
        let cleaned = FactExtractionValidator.sanitize(draft)
        guard !cleaned.draft.entities.isEmpty else {
            log("[memory] 本批对话无可持久化事实"
                + (cleaned.report.totalDropped > 0
                   ? "（护栏丢弃 \(cleaned.report.totalDropped) 项）" : ""))
            return true
        }
        if cleaned.report.totalDropped > 0 {
            log("[memory] 事实抽取护栏丢弃：实体 \(cleaned.report.droppedEntities)、"
                + "观察 \(cleaned.report.droppedObservations)、关系 \(cleaned.report.droppedRelations)")
        }

        let graph = GraphMemoryStore.shared(
            workspace: config.workspace,
            embedding: LocalModelFactory.shared.makeEmbedding(config: config))
        // 消歧 + 合并统一在 store 层完成：同一实体不重复入库，只追加观察与关系。
        let merged = await graph.mergeFactDraft(cleaned.draft)
        log("[memory] 写入 memory graph：\(cleaned.draft.entities.count) 个实体候选（已消歧合并）、"
            + "\(merged.observations) 条 observation + \(merged.relations) 条新增 relation")
        return true
    }

    private func distillExperience(transcript: String) async {
        guard !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        let raw = await collectChat([
            ChatMessage(role: .system, content: DISTILL_SYSTEM),
            ChatMessage(role: .user, content: makeDistillPrompt(transcript: String(transcript.prefix(4_000)))),
        ], temperature: 0.2)
        let data = l4Normalize(l4ExtractJSON(raw))
        let observations = data.facts + data.lessons
        if !observations.isEmpty {
            let graph = GraphMemoryStore.shared(
                workspace: config.workspace,
                embedding: LocalModelFactory.shared.makeEmbedding(config: config))
            graph.upsertEntity("Experience", type: "distilled", observations: observations)
            log("[distill] 写入 \(observations.count) 条经验到知识图谱")
        }
        if !data.skill.isEmpty {
            let skill = data.skill
            do {
                _ = try skills.save(name: skill.name, description: skill.summary,
                                    whenToUse: skill.whenToUse, steps: skill.steps,
                                    pitfalls: data.lessons.joined(separator: "; "))
                log("[distill] 固化技能 \(skill.name)")
            } catch {
                log("[distill] 技能保存失败: \(error.localizedDescription)")
            }
        }
    }

    // MARK: 模型辅助

    /// 用辅助模型整理 RAG 检索结果：先写一段简短的全局摘要，再原样保留
    /// Top 片段的原文。只压缩重复，不重写原文，避免主模型拿到的细节失真。
    /// 输出为空或失败由调用方回退原文。
    private func condenseRAGResult(_ text: String,
                                   stage: RAGCondenseStage) async -> String? {
        guard beginBackgroundIfIdle() else { return nil }
        defer { endBackground() }
        let system: String
        switch stage {
        case .chapter:
            system = "你是检索结果摘要器。输入是单个章节或文档片段的检索结果。"
                + "只输出要点摘要：提取关键事实、数值、名称、结论与因果，并标注来源。"
                + "总长不超过 300 字；若有极关键的原文句（不超过 80 字）可原样附 1 条，"
                + "其余一律概括。禁止复制整段原文、禁止逐句改写、禁止输出「原文片段」列表。"
                + "不要编造或添加原文没有的信息，不要发表评论。只输出摘要正文。"
        case .merge:
            system = "你是检索结果整合器。输入是同一问题下多个章节/文档的要点摘要。"
                + "请合并去重，输出一段总摘要：关键事实、数值、名称、结论与因果，"
                + "并保留来源标注；随后用 `原文片段:` 开头、以 `---` 分隔，"
                + "摘录最有价值的原文片段（各不超过 120 字），不要改写、不要编造、不要评论。"
                + "总输出必须严格控制在 1800 字以内：优先保证摘要完整，"
                + "原文片段按价值取舍，宁少勿超（超出部分会被硬截断）。"
                + "输出格式：先 `摘要:` 一段，再 `原文片段:` 若干段。"
        }
        // 输出侧断流上限：目标长度 + 余量。断流立即取消底层生成，
        // 是「模型不守约」时的硬止损；截断到目标长度由调用方负责。
        let maxOutputChars: Int?
        switch stage {
        case .chapter: maxOutputChars = 700      // 目标 300 字 + 1 条原句 + 余量
        case .merge: maxOutputChars = 2_000      // 目标 1800 字 + 余量
        }
        let output = await collectChat([
            ChatMessage(role: .system, content: system),
            ChatMessage(role: .user, content: text),
        ], temperature: 0.2, maxOutputChars: maxOutputChars)
        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        log("[rag] 辅助模型整理（\(stage == .chapter ? "章节" : "整合")）："
            + "\(text.count) 字 -> \(trimmed.count) 字")
        return trimmed
    }

    private static func stripJSONFence(_ raw: String) -> String {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard text.hasPrefix("```") else { return text }
        var lines = text.components(separatedBy: "\n")
        if !lines.isEmpty { lines.removeFirst() }
        if let last = lines.last, last.hasPrefix("```") { lines.removeLast() }
        text = lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        return text
    }

    /// 后台辅助请求的覆盖项。
    ///
    /// - 主模型 + 支持独立会话（MLX / ONNX / 远端 HTTP）：走 isolated 副本并关闭
    ///   thinking。后台 prompt 与主会话完全不同，共用 KV 必然全量重建并覆盖主会话
    ///   的 `rawCache/sessionTranscript`，下一次用户消息只能重新 prefill。
    /// - 主模型 + 不支持独立会话（llama.cpp 的 isolated 会清共享缓存）：返回 nil，
    ///   调用方直接跳过后台任务。
    /// - 辅助模型：与主会话无关，按默认选项调用即可。
    static func backgroundRequestOptions(for service: LLMService,
                                         isMain: Bool) -> LLMRequestOptions? {
        // 支持独立会话的后端（MLX/GGUF 的 isolated 副本池）一律走 isolated：
        // 辅助模型同样能复用稳定前缀，避免每次摘要都全量 prefill。
        if service.supportsConcurrentIsolatedRequests {
            return LLMRequestOptions(enableThinking: false, isolated: true)
        }
        // 主模型必须支持独立会话才允许后台调用；辅助模型可用普通会话兜底。
        return isMain ? nil : .default
    }

    /// 后台辅助任务的统一 LLM 入口（事实抽取 / 经验蒸馏 / 记忆整理 /
    /// RAG 检索浓缩）。折叠摘要已改为逐轮自带 summary 的确定性折叠；
    /// 事实抽取额外要求 `auxLLM` 存在（见 `extractFacts`），不会回退主模型。
    ///
    /// 这些任务影响整体功能（记忆/检索质量），因此**辅助模型优先、主模型
    /// 兜底**：辅助不可用时走主模型 isolated 副本（关闭 thinking、不碰主
    /// 会话 KV）；主模型连 isolated 都不支持时任务跳过。章节目录摘要不属
    /// 于此类——它是锦上添花，只用辅助模型（见 BookSummaryCoordinator）。
    private func collectChat(_ messages: [ChatMessage],
                             temperature: Double,
                             maxOutputChars: Int? = nil) async -> String {
        let service = auxLLM ?? llm
        guard let options = Self.backgroundRequestOptions(for: service,
                                                          isMain: service === llm) else {
            log("[llm] 主模型不支持独立会话，跳过后台调用")
            return ""
        }
        var output = ""
        do {
            for try await event in service.chatStream(messages: messages, tools: [],
                                                      temperature: temperature,
                                                      options: options) {
                // AsyncThrowingStream 的迭代不会自动感知 Task 取消；主动退出后
                // 流销毁会触发 onTermination，进而取消底层 MLX 生成。
                if Task.isCancelled { break }
                if event.kind == .content {
                    output += event.content
                    // 2B 模型偶尔无视 prompt 的长度约束把输入整段复述回来；
                    // 摘要类任务靠输出侧提前断流止损（断流即取消底层生成），
                    // 而不是等它把几千字生成完再截断。
                    if let maxOutputChars, output.count >= maxOutputChars {
                        log("[llm] 后台输出超过 \(maxOutputChars) 字符，提前终止生成")
                        break
                    }
                }
                if event.kind == .done { break }
            }
        } catch is CancellationError {
            // 前台请求抢占，后台任务主动让出 GPU。
        } catch AgentError.cancelled {
            // 同上：MLX 服务会把取消映射成 AgentError.cancelled。
        } catch {
            log("[llm] 后台调用失败: \(error.localizedDescription)")
        }
        // 辅助模型默认（`aux_llm_kv_reuse=false`）在调用结束后丢弃 KV：后台摘要 /
        // 事实抽取的内容每次都不同、跨调用命中率低，长 KV 常驻不划算；
        // 开启复用后保留缓存，相同前缀（system prompt）可跨调用复用。
        // 主模型（兜底路径）的常驻会话缓存始终不能在这里清。
        if service !== llm, !config.auxKvReuse {
            (service as? KVCacheInvalidating)?.invalidateKVCache()
        }
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: 日志

    func log(_ line: String) {
        callbacks.onLog(line)
    }

    func logToolResult(_ name: String, _ result: ToolResult) {
        let line = "[tool] \(name) -> success=\(result.success) out_len=\(result.output.count) "
            + "err=\(String(result.error.prefix(120)))"
        log(line)
        // 工具执行结果同时落日志文件：UI 日志面板不持久化，事后无法定位
        // 「模型说调用了工具但没执行」这类问题。
        AgentLog.info(line)
    }

    private static func dayString() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.string(from: Date())
    }

    private static func timestamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return formatter.string(from: Date())
    }

    private static func timeString() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm"
        return formatter.string(from: Date())
    }
}
