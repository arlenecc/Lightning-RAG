import Foundation

// MARK: - 工具分发

/// 把内核的 ToolCall 列表变成 ToolResult 列表（顺序一致）。
///
/// 确认、取消、并发这些「如何执行」的细节属于能力层，不放进内核。
public final class AgentToolDispatcher {
    private weak var agent: Agent?
    private var schemaCache: [JSONValue]?
    private let lock = NSLock()

    private let parallelSafe: Set<String> = [
        "web_scan", "rag_search", "rag_outline",
        "skill_search", "skill_semantic_search", "memory_search",
    ]
    private let maxParallel = 8

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

            if parallelSafe.contains(call.name) {
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
            let slice = Array(jobs[offset..<min(offset + workerCount, jobs.count)])
            for (_, call) in slice {
                agent.callbacks.onToolStart(call.name, call.arguments)
            }
            let collected = await withTaskGroup(of: (Int, String, ToolResult).self,
                                                returning: [(Int, String, ToolResult)].self) { group in
                for (index, call) in slice {
                    let tool = agent.tools.get(call.name)
                    group.addTask {
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

    /// 只在「上下文占用 >90%」的自动收缩时生成并回注旧上下文摘要。
    /// 每 8 轮的周期性折叠走丢弃式路径（保留最近 2 轮原文），不经过这里、不回注任何内容。
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
    public let skills: SkillManager
    public let skillIndex: SkillIndex
    public let skillRetriever: SkillRetriever
    public let ragEngine: RAGEngine
    public let workMemory: WorkMemory
    public let session: Session
    /// 多级记忆目录（L0–L4）。与 `.agent/` 下既有 WorkMemory/GraphMemory 并存。
    public let memoryDirectory: MemoryDirectory
    public let rawSessionStore: RawSessionStore
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
    private let autoContinueMax = 4
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
                skills: SkillManager? = nil) {
        self.config = config
        config.ensureWorkspace()
        self.llm = llm
        self.callbacks = callbacks

        let agentDir = (config.workspace as NSString).appendingPathComponent(".agent")
        let skillsDirectory = (agentDir as NSString).appendingPathComponent("skills")
        let builtin = (config.workspace as NSString).appendingPathComponent("skills")
        self.skills = skills ?? SkillManager(skillsDirectory: skillsDirectory,
                                             builtinDirectories: [builtin])
        self.skillIndex = SkillIndex(skillsDirectory: skillsDirectory)
        let embedding = LocalModelFactory.shared.makeEmbedding(config: config)
        self.skillRetriever = SkillRetriever(manager: self.skills, embedding: embedding)
        self.ragEngine = RAGEngine(config: config, embedding: embedding)
        self.workMemory = WorkMemory(workspace: config.workspace)
        let memoryDir = MemoryDirectory(workspace: config.workspace)
        memoryDir.ensureLayout()
        self.memoryDirectory = memoryDir
        self.rawSessionStore = RawSessionStore(directory: memoryDir)
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
                          config.kernelRawTurns == 0 ? 10 : config.kernelRawTurns,
                          (config.kernelConsolidateEvery == 0 ? 8 : config.kernelConsolidateEvery) + 2),
            maxDigestLines: config.kernelMaxDigestLines == 0 ? 40 : config.kernelMaxDigestLines
        )

        if let toolRegistry {
            self.tools = toolRegistry
        } else {
            self.tools = AgentToolFactory.build(
                config: config, callbacks: callbacks, session: self.session,
                ragEngine: self.ragEngine, skillManager: self.skills,
                skillIndex: self.skillIndex, skillRetriever: self.skillRetriever,
                workMemory: self.workMemory, sopTracker: sharedSOPTracker)
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
        // 运行中替换辅助模型），阈值判断在 RAGEngine.condenseIfNeeded 内。
        self.ragEngine.auxCondenser = { [weak self] text in
            guard let self, self.auxLLM != nil else { return nil }
            return await self.condenseRAGResult(text)
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

    /// 丢弃计划（Session 内 + 落盘文件），不触碰对话历史。
    /// 用于启动时清理上次未完成的计划，以及计划面板的「清空计划」。
    @discardableResult
    public func discardPersistedPlan() -> Bool {
        let hadPersisted = PlanStore.load(config) != nil
        session.clearPlan()
        PlanStore.clear(config)
        return hadPersisted
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
        GraphMemoryStore.flushShared()
        GraphMemoryStore.releaseShared(workspace: config.workspace)
        workMemory.flush()
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
        workMemory.flush()
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

    private func runInternal(_ userInput: String, images: [Data]) async -> String {
        // 前台请求优先：上一轮的后台摘要/事实抽取还在跑时，先取消并等它退出，
        // 否则 2B 辅助模型会和主模型抢 GPU（实测主模型会从 ~15 tps 掉到 ~5 tps）。
        await cancelBackgroundWork()
        turnIndex += 1
        cachedDynamicContext = nil
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
                session.working["__truncated__"] = .string(
                    "[系统提示] 上一段在达到轮次上限（\(maxIterations)）后被自动续跑。"
                    + "请先用一句话总结当前进展，然后**立即继续执行**计划中的剩余步骤；"
                    + "若全部步骤已完成，用 update_plan 收尾并给出最终答复。\(failText)")
                result = await kernel.run(
                    "[System] 任务在上一段达到轮次上限被截断，已自动续跑。"
                    + "请继续执行计划中的剩余步骤；不要询问用户是否继续。",
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
                text = "（已达到本轮最大轮次上限，任务可能未完成。"
                    + "请先总结已完成的进展，再询问用户是否继续。）"
            }
            log("[agent] 空回复（reason=\(result.reason.rawValue)）：已向用户返回可见提示")
        }
        // 技能固化只针对「建过多步骤计划且已收尾」或「显式触发长期蒸馏」的任务。
        // 寒暄、简单查询、未完成的任务即使重复出现，也不会再提示保存技能。
        let planTouched = session.planRevision != planRevisionBefore
        let plan = session.getPlan()
        let planComplete = (plan?.steps.count ?? 0) >= 2
            && (plan?.steps.allSatisfy { $0.status == "done" } ?? false)
        let substantive = distillRequested || (planTouched && planComplete)
        if substantive, let skillCandidate, skills.match(userInput) == nil {
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

        // 改进7 前半：过期记录预处理/打包（纯文件，单次最多 1 天；内部在删原文前
        // 把 session 写入持久化队列，因此后续任何异常都不会让提炼任务丢失）。
        _ = rawSessionStore.archivePastSessions()

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
        advanceSystemFoldEpoch()
        freezeRequestSystemPrompt(force: true)
        guard let warmup = llm as? WarmupLLMService else { return }
        let current = takeRequestSystemPrompt() ?? buildStaticSystemPrompt()
        do {
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
        var text = BASE_SYSTEM_PROMPT
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

        // 工作记忆 + 当天自动抽取事实。
        let work = tools.workMemory ?? workMemory
        var compact: [String: String] = [:]
        var todayFacts: [String] = []
        var workMemoryChars = 0
        for (key, value) in work.list().sorted(by: { $0.key < $1.key }) {
            if key == "__auto_facts__" {
                todayFacts = Self.parseDailyFacts(value).1
                continue
            }
            guard compact.count < SystemPromptLimits.maxWorkMemoryKeys,
                  workMemoryChars < SystemPromptLimits.maxWorkMemoryChars else { continue }
            let capped = value.count > 200 ? String(value.prefix(200)) + "…" : value
            compact[key] = capped
            workMemoryChars += capped.count
        }
        if !todayFacts.isEmpty || !compact.isEmpty {
            var workParts: [String] = []
            if !todayFacts.isEmpty {
                var capped: [String] = []
                var used = 0
                for fact in todayFacts {
                    if capped.count >= SystemPromptLimits.maxInjectedFacts ||
                        used >= SystemPromptLimits.maxFactsChars { break }
                    let item = fact.count <= SystemPromptLimits.maxFactChars
                        ? fact : String(fact.prefix(SystemPromptLimits.maxFactChars)) + "…"
                    capped.append(item)
                    used += item.count
                }
                var block = capped.map { "- \($0)" }.joined(separator: "\n")
                let dropped = todayFacts.count - capped.count
                if dropped > 0 { block += "\n- (…\(dropped) more omitted)" }
                workParts.append("# Today's facts (auto-extracted)\n" + block)
            }
            if !compact.isEmpty {
                workParts.append("# Work memory (scratchpad)\n"
                    + JSONValue.object(compact.mapValues { .string($0) }).jsonString())
            }
            sections.append(workParts.joined(separator: "\n\n"))
        }

        // 检查点常驻注入：优先 Session（本轮工具实时更新），其次持久化文件。
        let checkpoint = session.getKeyInfo().isEmpty
            ? CheckpointStore.readPersisted(config)
            : session.getKeyInfo()
        if !checkpoint.isEmpty {
            sections.append("# Working checkpoint\n\(checkpoint.prefix(4_000))")
        }

        // 命中技能：截断并提示用 skill_load 读全文。
        if !lastUserText.isEmpty, let matched = skills.match(lastUserText) {
            var prompt = matched.body
            if prompt.count > SystemPromptLimits.maxSkillPromptChars {
                prompt = String(prompt.prefix(SystemPromptLimits.maxSkillPromptChars))
                    + "\n…（已截断，用 skill_load 读取完整技能内容）"
            }
            sections.append("# Active skill: \(matched.name)\n\(prompt)")
        }

        // 知识库状态只读 RAGEngine 的快照：绝不在这里同步打开/扫描向量库。
        // 千万级切片的库上 COUNT(*) 需要数分钟，会把首轮 LLM 调用卡住。
        if let status = ragStatus() {
            let chunks = Int(status["chunks_stored"]?.intValue ?? 0)
            let sources = status["sources"]?.arrayValue?.compactMap(\.stringValue) ?? []
            let sourceCount = status["source_count"]?.intValue ?? sources.count
            if chunks > 0 {
                let shown = min(8, sources.count)
                var names = sources.prefix(shown).map { ($0 as NSString).lastPathComponent }
                    .joined(separator: ", ")
                if sourceCount > shown { names += " (and \(sourceCount - shown) more)" }
                sections.append("# Knowledge base\n\(chunks) chunks in \(sourceCount) file(s): "
                    + "\(names). Follow the knowledge-base strategy in the system prompt.")
            } else {
                sections.append("# Knowledge base\nConfigured but empty. Use `rag_ingest` to index.")
            }
        } else if FileManager.default.fileExists(atPath: ragEngine.indexDBPath) {
            // 启动后状态快照还没生成（由 UI 在后台刷新）：先给一个无统计的轻提示，
            // 仍可正常检索，不阻塞本轮。
            sections.append("# Knowledge base\nIndexed locally. "
                + "Follow the knowledge-base strategy in the system prompt to retrieve passages.")
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

    // MARK: 上下文 >90% 自动收缩（唯一会回注旧摘要的路径）

    /// 用辅助模型把旧对话压成一条要点摘要，回注到 `<earlier_context>`。
    /// 旧轮本身直接丢弃（不生成逐轮摘要行），避免摘要越堆越占上下文。
    /// 周期性折叠不调这里 —— 那条路径直接清空历史、从 system 重新开始。
    func shrinkSession(reason: String) async -> Bool {
        guard session.turns.count > 1 else { return false }
        let old = Array(session.turns.dropLast())
        let transcript = old.map { turn in
            let results = turn.toolResults.map { String($0.content.prefix(400)) }.joined()
            return "\(turn.digestLine())\n\(turn.assistantText.prefix(600))\(results)"
        }.joined(separator: "\n")

        let prompt = """
        把下面的对话压缩成要点摘要，只保留：
        1. 用户的核心需求与当前目标 2. 已做的决定及原因
        3. 已完成的步骤与结果（含工具关键输出） 4. 未完成的子任务与下一步
        5. 具体路径、数值、标识符、配置值、偏好、URL 6. 错误/警告等需注意的事项

        要求：只输出要点列表，不要前言或解释；不要复述对话；不要编造。

        对话历史:
        \(transcript)
        """
        let summary = await collectChat([
            ChatMessage(role: .system,
                        content: "你是精确的对话摘要器，用中文输出要点列表，不加前言和解释。"),
            ChatMessage(role: .user, content: prompt),
        ], temperature: 0.2)
        let final = summary.isEmpty
            ? old.map { $0.digestLine() }.joined(separator: "; ")
            : summary
        appendMemorySummary(final, reason: reason)
        let folded = session.discardTurns(old.count)
        let digestLine = "[摘要 \(Self.timeString())] \(final.prefix(400))"
        session.appendDigest(digestLine)
        // 改进2：自动收缩生成的上下文摘要也追加进当天 digest 文件，
        // 保证"当天对话摘要历史"不因只统计逐轮快照而缺项。
        rawSessionStore.appendFoldedDigest([digestLine])
        log("[context] 自动收缩：丢弃 \(folded) 轮，回注 1 条摘要（reason: \(reason)）")
        callbacks.onContextShrunk(final, reason)
        return true
    }

    private func appendMemorySummary(_ summary: String, reason: String) {
        let path = (config.workspace as NSString).appendingPathComponent("memory.md")
        let existing = (try? String(contentsOfFile: path, encoding: .utf8)) ?? ""
        let entry = "\n## 上下文摘要 - \(Self.timestamp())\n- 触发原因: \(reason)\n\n\(summary)\n"
        try? AtomicFile.write(existing + entry, to: path)
    }

    // MARK: 结束后：事实抽取 + L4 蒸馏

    func fireFinished(text: String, distillRequested: Bool? = nil,
                      substantive: Bool = true,
                      isGreeting: Bool = false) {
        // 与参考实现一致：取最近 6 条 User/Assistant 消息（交替），而不是每轮只取一条。
        var recentLines: [String] = []
        for turn in session.turns.reversed() {
            let prompt = turn.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
            if !prompt.isEmpty {
                recentLines.append("User: \(prompt)")
                if recentLines.count >= 6 { break }
            }
            let assistant = turn.assistantText.trimmingCharacters(in: .whitespacesAndNewlines)
            if !assistant.isEmpty {
                recentLines.append("Assistant: \(assistant)")
                if recentLines.count >= 6 { break }
            }
        }
        let recent = recentLines.reversed().joined(separator: "\n\n")
        let user = lastUserText.trimmingCharacters(in: .whitespacesAndNewlines)
        // 简单请求往往一句话（「查下 xx 文件多少行」+「120 行」），
        // 阈值放宽到 6/8 字。寒暄不做事实抽取/蒸馏，但仍要消费归档队列。
        let canExtract = !isGreeting && user.count >= 6 && text.count >= 8 && !recent.isEmpty

        // 改进7：待提炼队列是**持久化**的，寒暄轮、辅助模型暂不可用、后台任务
        // 被下一次 run 取消等情况都不会丢，本轮或下一轮重新拾起即可。
        let hasPending = !memoryDirectory.pendingInsightIds().isEmpty
        // 改进8：至少 8 轮做一次整理建议（与折叠周期对齐），
        // 后台用辅助引擎产出建议、不改记忆文件。
        let tidyEvery = max(8, config.kernelConsolidateEvery)
        let shouldTidy = !isGreeting && session.currentTurn >= readLastMemoryTidyTurn() + tidyEvery
        let hasLLMWork = canExtract || hasPending || shouldTidy
        guard hasLLMWork else { return }

        // 后台 LLM 工作（事实抽取/经验蒸馏/记忆整理）影响整体功能：
        // **辅助模型优先，主模型兜底**——兜底必须走 isolated 副本（关闭
        // thinking、不碰主会话 KV，见 backgroundRequestOptions）；主模型连
        // isolated 都不支持时才跳过（队列保留待下次消费）。
        // 注意：章节目录摘要不在此列——它不影响功能，只用辅助模型
        // （见 BookSummaryCoordinator，辅助不可用即停做）。
        let service = auxLLM ?? llm
        guard Self.backgroundRequestOptions(for: service, isMain: service === llm) != nil else {
            log("[memory] 主模型不支持独立会话，跳过后台 LLM；待提炼队列已保留")
            return
        }

        // 没有计划的简单对话不做技能/经验蒸馏；事实抽取仍然保留。
        let shouldDistill = canExtract && substantive
            && (distillRequested ?? kernel.distillRequested)
        let transcript = session.allDigestLines().suffix(25).joined(separator: "\n")
        backgroundTaskLock.lock()
        backgroundTask?.cancel()
        backgroundGeneration += 1
        let generation = backgroundGeneration
        let task = Task.detached(priority: .utility) { [weak self] in
            guard let self else { return }
            defer { self.finishBackgroundTask(generation) }
            if canExtract { await self.extractFacts(conversation: recent) }
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
        task.cancel()
        let exited = await Self.waitForTaskExit(task, timeoutNanos: 3_000_000_000)
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

    private func extractFacts(conversation: String) async {
        guard beginBackgroundIfIdle() else { return }
        defer { endBackground() }
        let text = String(conversation.prefix(4_000))
        let raw = await collectChat([
            ChatMessage(role: .system, content: FACT_EXTRACT_SYSTEM),
            ChatMessage(role: .user, content: factExtractPrompt(conversation: text)),
        ], temperature: 0.1)
        let json = Self.stripJSONFence(raw)
        guard let value = JSONValue.parse(json), let data = value.objectValue else { return }
        let entities = data["entities"]?.arrayValue ?? []
        let relations = data["relations"]?.arrayValue ?? []
        guard !entities.isEmpty || !relations.isEmpty else {
            log("[memory] 本轮无可持久化事实")
            return
        }

        let graph = GraphMemoryStore.shared(
            workspace: config.workspace,
            embedding: LocalModelFactory.shared.makeEmbedding(config: config))
        var observations: [String] = []
        for item in entities {
            guard let entity = item.objectValue,
                  let name = entity["name"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !name.isEmpty else { continue }
            let obs = (entity["observations"]?.arrayValue ?? [])
                .compactMap(\.stringValue)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            if !obs.isEmpty {
                graph.addObservations(name, obs)
                observations.append(contentsOf: obs)
            } else if let type = entity["type"]?.stringValue, !type.isEmpty {
                graph.upsertEntity(name, type: type)
            }
        }
        for item in relations {
            guard let rel = item.objectValue,
                  let source = rel["source"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines),
                  let target = rel["target"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines),
                  let label = rel["label"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !source.isEmpty, !target.isEmpty, !label.isEmpty else { continue }
            graph.createRelation(source: source, target: target, label: label)
        }
        if !observations.isEmpty { appendDailyFacts(observations) }
        log("[memory] 抽取 \(observations.count) 条 observation + \(relations.count) 条 relation")
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

    private func appendDailyFacts(_ facts: [String]) {
        // 必须复用共享实例：新建实例会带着旧快照全量回写，覆盖工具刚写入的键。
        let work = tools.workMemory ?? workMemory
        let today = Self.dayString()
        var (bucketDate, bucketFacts) = Self.parseDailyFacts(work.get("__auto_facts__") ?? "")
        if bucketDate != today { bucketFacts = [] }
        var seen = Set(bucketFacts)
        for fact in facts where !fact.isEmpty && !seen.contains(fact) {
            bucketFacts.append(fact)
            seen.insert(fact)
        }
        if bucketFacts.count > SystemPromptLimits.maxDailyFacts {
            bucketFacts = Array(bucketFacts.suffix(SystemPromptLimits.maxDailyFacts))
        }
        let payload = JSONValue.object([
            "date": .string(today),
            "facts": .array(bucketFacts.map { .string($0) }),
        ])
        work.set("__auto_facts__", payload.jsonString())
    }

    static func parseDailyFacts(_ raw: String) -> (String, [String]) {
        guard let value = JSONValue.parse(raw),
              let data = value.objectValue else { return ("", []) }
        let date = data["date"]?.stringValue ?? ""
        let facts = (data["facts"]?.arrayValue ?? []).compactMap(\.stringValue)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        return (date, facts)
    }

    // MARK: 模型辅助

    /// 用辅助模型整理 RAG 检索结果：先写一段简短的全局摘要，再原样保留
    /// Top 片段的原文。只压缩重复，不重写原文，避免主模型拿到的细节失真。
    /// 输出为空或失败由调用方回退原文。
    private func condenseRAGResult(_ text: String) async -> String? {
        guard beginBackgroundIfIdle() else { return nil }
        defer { endBackground() }
        let output = await collectChat([
            ChatMessage(role: .system,
                        content: "你是检索结果整理器。输入是若干条带来源标注的检索片段。"
                            + "请先写一段简短摘要（关键事实、数值、名称、结论与因果，"
                            + "并标注来源），然后**原样保留最重要的前 3 条片段全文**，"
                            + "用 `原文片段:` 开头、以 `---` 分隔各片段，不要改写、不要省略、"
                            + "不要合并句子。不要编造或添加原文没有的信息，不要发表评论。"
                            + "输出格式：先 `摘要:` 一段，再 `原文片段:` 若干段。"),
            ChatMessage(role: .user, content: text),
        ], temperature: 0.2)
        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        log("[rag] 辅助模型整理：\(text.count) 字 -> \(trimmed.count) 字")
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
        guard isMain else { return .default }
        guard service.supportsConcurrentIsolatedRequests else { return nil }
        return LLMRequestOptions(enableThinking: false, isolated: true)
    }

    /// 后台辅助任务的统一 LLM 入口（折叠摘要 / 事实抽取 / 经验蒸馏 /
    /// 记忆整理 / RAG 检索浓缩）。
    ///
    /// 这些任务影响整体功能（记忆/检索质量），因此**辅助模型优先、主模型
    /// 兜底**：辅助不可用时走主模型 isolated 副本（关闭 thinking、不碰主
    /// 会话 KV）；主模型连 isolated 都不支持时任务跳过。章节目录摘要不属
    /// 于此类——它是锦上添花，只用辅助模型（见 BookSummaryCoordinator）。
    private func collectChat(_ messages: [ChatMessage], temperature: Double) async -> String {
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
                if event.kind == .content { output += event.content }
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
        log("[tool] \(name) -> success=\(result.success) out_len=\(result.output.count) "
            + "err=\(String(result.error.prefix(120)))")
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
