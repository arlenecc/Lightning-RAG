import Foundation

// MARK: - 内核注入的能力

/// 内核只做控制流：LLM、工具、取消与日志全部通过闭包注入。
///
/// 这样内核不依赖 `Agent` 类，也不持有任何工具实现。
public struct KernelRuntime {
    public typealias StreamFunction = ([ChatMessage], [JSONValue], Double)
        -> AsyncThrowingStream<LLMStreamEvent, Error>
    public typealias ExecuteFunction = ([ToolCall]) async throws -> [ToolResult]

    /// 发起一次流式 LLM 调用：(messages, tools, temperature)。
    public var stream: StreamFunction
    /// 请求级覆盖版 stream（如空正文重试时关闭 thinking 让模型直接作答）。
    /// nil 时回退 `stream`（兼容测试 mock 与未迁移的构造方）。
    public var streamWithOptions: (([ChatMessage], [JSONValue], Double, LLMRequestOptions)
        -> AsyncThrowingStream<LLMStreamEvent, Error>)?
    /// 分发一批工具调用，返回与调用顺序一致的结果。
    public var execute: ExecuteFunction
    /// 工具 schema（对应 Python dispatcher.schemas()）。
    public var schemas: () throws -> [JSONValue]
    /// 协作式取消轮询。
    public var shouldCancel: () -> Bool
    /// 历史被改写时丢弃本地后端的跨轮 KV 缓存（HTTP 后端为空）。
    public var invalidateKV: (() -> Void)?
    /// 内核日志。
    public var log: (String) -> Void

    public init(stream: @escaping StreamFunction,
                streamWithOptions: (([ChatMessage], [JSONValue], Double, LLMRequestOptions)
                    -> AsyncThrowingStream<LLMStreamEvent, Error>)? = nil,
                execute: @escaping ExecuteFunction,
                schemas: @escaping () throws -> [JSONValue] = { [] },
                shouldCancel: @escaping () -> Bool = { false },
                invalidateKV: (() -> Void)? = nil,
                log: @escaping (String) -> Void = { _ in }) {
        self.stream = stream
        self.streamWithOptions = streamWithOptions
        self.execute = execute
        self.schemas = schemas
        self.shouldCancel = shouldCancel
        self.invalidateKV = invalidateKV
        self.log = log
    }
}

/// 最近一次工具失败（工具名 + 错误摘要）。
public struct ToolFailure: Sendable, Equatable {
    public var name: String
    public var error: String
    public init(name: String, error: String) {
        self.name = name
        self.error = error
    }
}

/// 失败上下文快照，供上层在「自动续跑」的段之间传递。
public struct FailureSnapshot: Sendable {
    public var failedTotal: Int
    public var lastFail: ToolFailure?
    public var failSigs: [String]
    public var failStreak: Int
    public var finishNudge: Int
    public var planFinishNudge: Int
    public var blockNudge: Int
    public var emptyRetry: Int
    public var reaskGuard: Int
    public var planNudged: Int
    public var didWork: Bool
    /// 自动续跑跨段保留：步骤内失败数与重复调用检测。
    public var stepFails: Int
    public var repeatStreak: Int
    public var lastCallSigs: [String]
    /// 跨段保留：意图空转与上下文超长重试的已用额度。旧实现每段都重新
    /// 发放完整配额，等于把上限放大到段数倍。
    public var intentStall: Int
    public var shrinkGuard: Int

    public init(failedTotal: Int = 0, lastFail: ToolFailure? = nil,
                failSigs: [String] = [], failStreak: Int = 0,
                finishNudge: Int = 0, planFinishNudge: Int = 0,
                blockNudge: Int = 0, emptyRetry: Int = 0,
                reaskGuard: Int = 0, planNudged: Int = 0,
                didWork: Bool = false, stepFails: Int = 0,
                repeatStreak: Int = 0, lastCallSigs: [String] = [],
                intentStall: Int = 0, shrinkGuard: Int = 0) {
        self.failedTotal = failedTotal
        self.lastFail = lastFail
        self.failSigs = failSigs
        self.failStreak = failStreak
        self.finishNudge = finishNudge
        self.planFinishNudge = planFinishNudge
        self.blockNudge = blockNudge
        self.emptyRetry = emptyRetry
        self.reaskGuard = reaskGuard
        self.planNudged = planNudged
        self.didWork = didWork
        self.stepFails = stepFails
        self.repeatStreak = repeatStreak
        self.lastCallSigs = lastCallSigs
        self.intentStall = intentStall
        self.shrinkGuard = shrinkGuard
    }
}

// MARK: - 常量（对齐 Python loop.py）

private let maxEmptyRetry = 3
private let llmRetryBaseDelay = 1.5
private let maxLLMRetries = 2
private let l0HardCap = 20_000

private let truncationMarks = ["[!!! 流异常中断", "!!!Error:", "max_tokens !!!]"]
private let codeBlockPattern = "```[a-zA-Z0-9_]*\\n[\\s\\S]{50,}?```"
private let tagStripPattern = "<(thinking|think|summary)>[\\s\\S]*?</\\1>"

private let planNudgeMax = 3
private let multistepHintPattern =
    "(步骤|计划|规划|先做|先要|然后|接着|依次|分别|逐个|多个|多项|"
    + "第[一二三四五六七八九十\\d]+步|^\\s*\\d+[\\.、\\)]|一、|二、|首先|其次|最后|"
    + "以下任务|分别做|依次做|分步)"
private let multistepVerbPattern =
    "(重构|重写|批量|全部|所有|每个|一系列|整套|完整|"
    + "部署|迁移|搭建|改造|升级|替换|整合|拆分|合并|"
    + "整理|归档|备份|还原|恢复|排查|定位并|分析并|"
    + "获取|查询|搜索|读取|抓取|下载|"
    + "制定|实现|开发|配置|初始化|生成报告|写一份|做一份)"

private let planFinishHint = """
[System] 你既没有调用 update_plan 把计划写入执行计划（Session.working['plan'] 为空），也没有发出任何实际工具调用——任务等于什么都没做，哪怕你在文本里给出了结论也只是猜测。
请立即：1) 需要外部信息时直接发出第一步的实际工具调用（如 web_scan 获取页面、rag_tools(op=\"search\") 查询知识库）并持续推进；2) 或用 update_plan(op='create', goal=..., steps=[...]) 先登记计划，紧接着发出第一步调用。
禁止只在文本里描述计划或给出来自猜测的结论——每一步都必须真正调用工具，拿到真实结果后再总结。
"""

private let finishNudgeMax = 2
/// 反思门上限。实测一次多步任务可被放大到 29 次 LLM 调用 / 108s，
/// 每次提醒都会让模型再多生成一轮并把提示累积进上下文；收紧到能
/// 纠偏即可，失败场景也别无限重试。
private let finishNudgeFailMax = 4
private let finishNudgePlanMax = 3
/// 「只声明要做什么、却一次工具都不发」最多容忍几轮。小模型会把
/// `update_plan` 协议复述成文字，内核再提示「先建计划」就会形成正反馈，
/// 计划与第一步永远不推进；两次之后必须收敛，不能无限提示。
private let maxIntentStall = 2
private let maxFailSigs = 8
private let minAttemptsBeforeBlock = 3
private let blockNudgeMax = 2
private let repeatCallWarn = 1
private let repeatCallLimit = 3
/// 成果核验反思的最多打回次数：以资深专家身份审视最终结果，质量不达标
/// 最多打回完善 2 次，之后无论是否完美都收下，避免无限自我批评死循环。
private let reviewMax = 2
/// 单次大纲（rag_tools(op="outline")）之后允许的 rag_tools(op="search") 次数硬上限。
/// 超过后内核直接把多余检索预置为失败，逼模型基于已有结果 + 大纲摘要作答，
/// 杜绝「同一问题反复换关键词 scatter 检索」的无效搜索。
private let maxRagSearchesPerOutline = 3

private let askUserPattern =
    "(请(提供|告诉|告知|说明|补充|指定|确认)|需要(你|您)(提供|告诉|确认|补充)|"
    + "能否(提供|告诉|说明)|麻烦(提供|告诉)|"
    + "(哪个|哪些)(文件|目录|路径|项目|模块|分支|环境)|"
    + "想?要我(先)?(做|处理|修改|改)|希望我(做|先做)|你希望|你想要)"

/// 只在一段内核循环的**首轮**注入一次。旧实现每个 THINK 轮都 `pending +=`
/// 一遍（约 200 字），而 pending 会随 userText 写进 `rec.prompt` 永久留在
/// 历史里，于是每重试一轮上下文就多一段重复训诫，prefill 单调变长。
private let planGateHint = """

[System] 非平凡任务先 update_plan(op="create", goal=..., steps=[...]) 登记 2~7 步；计划与首个工具调用可同轮发出。登记后立即执行，每步完成用 update_plan(op="patch", status="done") 收尾，不要只登记计划就停下。
"""

// MARK: - KernelLoop

/// 三态自动机主循环：THINK → ACT →（CONSOLIDATE）→ THINK。
///
/// 它不拥有工具实现、不拥有记忆、不拥有 RAG —— 那些都是注入的能力。
public final class KernelLoop {
    public private(set) var state: KernelState = .think
    public let session: Session
    public let budget: ContextBudget
    public let consolidateEvery: Int
    public let maxToolChars: Int
    /// RAG 工具结果进历史的收紧上限（字符）。0 表示不额外收紧，沿用 L1 的
    /// `maxToolChars`。见 `AgentConfig.kernelRagHistoryChars`。
    public let ragHistoryChars: Int
    public var maxTurns: Int
    public private(set) var distillRequested = false

    private let runtime: KernelRuntime
    private let callbacks: AgentCallbacks
    private let config: AgentConfig
    private let systemPrompt: () -> String
    /// 动态上下文（命中技能/知识库状态），追加到当前轮 user 消息尾部。
    private let dynamicContext: () async -> String
    private let hooks: KernelHooks

    private var schemasCache: [JSONValue]?
    private var schemaTokens = 0
    private var emptyRetry = 0
    /// 最近一次 THINK 的输出被截断（工具调用未生成完整）。
    private var truncated = false
    private var shrinkGuard = 0
    private var reaskGuard = 0
    private var turnsUsed = 0
    private var lastConsolidatedTurn = -1
    /// 同一轮只做一次「提前折叠」，避免单轮内容本身就超顶时死循环。
    private var lastEarlyFoldTurn = -1
    private var totalTokens = 0
    private var finishNudge = 0

    // ---- 失败恢复记账（每次 run() 重置）----
    private var failStreak = 0
    private var failedTotal = 0
    private var lastFail: ToolFailure?
    private var failSigs: [String] = []
    private var stepFails = 0
    private var blockNudge = 0
    private var didWork = false
    /// 本段是否至少有一次「真正成功」的工具结果。preset 预置失败、调度异常
    /// 都不算；成果核验反思只能建立在真实数据之上，不能因为「发过工具调用」
    /// 就把幻觉结论核验通过。
    private var hasToolSuccess = false
    /// 意图空转耗尽后由内核代发过一次 web_scan（一段只兜底一次）。
    private var didFallbackFetch = false
    /// 意图空转耗尽后由内核代发过一次 web_search（一段只兜底一次）。
    private var didFallbackSearch = false
    /// 意图空转耗尽后由内核代发过一次 long_memory(op="search")。
    private var didFallbackMemorySearch = false
    /// 问「知识库里有什么」时内核代发 rag_tools(status) 的兜底（只发一次）。
    private var didFallbackKBStatus = false
    /// 兜底抓取拿到的真实内容：模型若仍不给正文，失败说明里也要交付它。
    private var fallbackToolText = ""
    private var planNudged = 0
    private var planFinishNudge = 0
    /// 连续「只描述计划 / 只声明要开工」却零工具调用的轮数。
    private var intentStall = 0
    /// 规划提示是否已在本段注入过（一段只注入一次）。
    private var planGateHintInjected = false
    /// 动态上下文是否已在本段注入过。`dynamicContext()` 按「一次用户请求」
    /// 冻结内容，同一段内逐轮注入只是把同一份 ~7.5k 字符重复写进历史。
    private var dynamicContextInjected = false
    private var repeatStreak = 0
    private var lastCallSigs: [String] = []
    /// 成果核验反思：已进行的核验轮数（0=尚未核验）与当前候选答复。
    private var reviewRounds = 0
    private var reviewCandidate = ""
    // ---- 内核决策门（complete 收尾拦截 + plan_step 计划评审）----
    /// 决策判定器（MeteorKernelJudge）。Agent 按配置装配；nil = 功能关闭，
    /// 内核行为与未接入前完全一致。判定失败/超时由实现返回 nil，静默放行。
    public var decisionJudge: KernelDecisionJudge?
    /// 收尾拦截已触发次数（预算 = config.kernelDecisionGateMaxFinishBlocks）。
    private var decisionFinishBlocks = 0
    /// 生成前判定已放行：本轮任务的证据已确认足够，收尾时不再重复判定。
    private var decisionPrechecked = false
    /// 本轮任务中 rag_search 曾返回最高相似度 > 0.5 的切片：检索本身已找到
    /// 匹配材料，视为完成度达成，生成前门与收尾门都不再做完成度判定。
    private var ragHighConfidence = false
    /// 计划评审提示已触发次数（预算 = config.kernelDecisionGateMaxPlanReviews）。
    private var decisionPlanReviews = 0
    /// 已评审过的计划标识（goal+stepCount），同一计划只评审一次。
    private var decisionReviewedPlanKey = ""
    /// 成功返回过结果的调用签名：完全相同参数再发一次，结果不会变。
    /// 模型常「拿到 web_scan 结果却原样重发刷屏」，用它在 preset 阶段直接
    /// 拦截，强令基于已有结果作答或换参数，而不是等 4 次重复才终止。
    private var successSigs: Set<String> = []
    private var toolCallCounter = 0
    /// 当前「大纲后检索预算」已用次数；每次 rag_tools(op="outline") 清零，每次 run() 也清零。
    private var ragSearchCount = 0
    private var resume = false
    private var pendingSeed: FailureSnapshot?

    public init(runtime: KernelRuntime,
                config: AgentConfig,
                callbacks: AgentCallbacks? = nil,
                systemPrompt: @escaping () -> String,
                dynamicContext: @escaping () async -> String = { "" },
                session: Session? = nil,
                hooks: KernelHooks? = nil,
                maxTurns: Int = 0) {
        self.runtime = runtime
        self.config = config
        self.callbacks = callbacks ?? NoopCallbacks()
        self.systemPrompt = systemPrompt
        self.dynamicContext = dynamicContext
        self.hooks = hooks ?? NoopKernelHooks()

        let resolved = session ?? Session(
            budget: ContextBudget(total: config.kernelMaxContextTokens),
            rawTurns: config.kernelRawTurns == 0 ? 20 : config.kernelRawTurns,
            maxDigestLines: config.kernelMaxDigestLines == 0 ? 20 : config.kernelMaxDigestLines
        )
        self.session = resolved
        // 必须与 Session 共用同一个预算对象，否则内核记的账和 Session 折叠时
        // 读的余额是两本账。
        self.budget = resolved.budget

        // 折叠会丢弃旧历史并强制一次 KV 重建；周期至少 2 轮，避免每个用户回合
        // 都重建。默认 8 轮：折叠后保留最近 2 轮原文，新前缀 = system + 2 轮，
        // prefill 仍远短于整段历史，也避免了折叠处上下文断裂。
        // 周期不能超过 raw 窗口：超出时窗口每轮都滑动，等于每轮都重建。
        let configured = max(2,
            config.kernelConsolidateEvery == 0 ? 18 : config.kernelConsolidateEvery)
        self.consolidateEvery = max(1, min(configured, max(1, resolved.rawTurns)))
        self.maxToolChars = config.kernelMaxToolChars == 0 ? 12_000 : config.kernelMaxToolChars
        self.ragHistoryChars = config.kernelRagHistoryChars
        let fallbackTurns = config.maxIterations == 0 ? 15 : config.maxIterations
        self.maxTurns = max(1, maxTurns == 0 ? fallbackTurns : maxTurns)
    }

    // MARK: 工具 schema

    public func setSchemas(_ schemas: [JSONValue]) {
        schemasCache = schemas
        schemaTokens = schemas.isEmpty
            ? 0
            : TokenEstimator.estimate(JSONValue.array(schemas).jsonString())
        budget.setUsed("tools", tokens: schemaTokens)
    }

    public func schemas() -> [JSONValue] {
        if schemasCache == nil {
            do {
                setSchemas(try runtime.schemas())
            } catch {
                // 静默降级成「没有任何工具」会让模型空跑，至少留一条日志。
                log("[kernel] 工具 schema 获取失败，本轮将无工具可用: \(describe(error))")
                setSchemas([])
            }
        }
        budget.setUsed("tools", tokens: schemaTokens)
        return schemasCache ?? []
    }

    /// 工具集动态变化（例如 MCP 异步注册完成）后强制下一轮重建 schema。
    public func invalidateSchemas() {
        schemasCache = nil
        schemaTokens = 0
        budget.setUsed("tools", tokens: 0)
    }

    // MARK: 主循环

    // MARK: - 主循环

    /// 执行一段内核循环。
    ///
    /// `emitFinished=false` 供「自动续跑」使用：续跑段是同一个用户轮次的一部分，
    /// 不能重复触发 `hooks.finished`（它会启动 L4 蒸馏等请求收尾任务）。
    /// 事实抽取已改为按 4 小时文件游标在请求收尾调度，不在这里按轮触发。
    /// `toolsEnabled=false` 只用于「不提供任何工具 schema」的纯文本调用。
    public func run(_ userInput: String,
                    resume: Bool = false,
                    emitFinished: Bool = true,
                    toolsEnabled: Bool = true,
                    images: [Data] = []) async -> KernelResult {
        let session = self.session
        state = .think
        emptyRetry = 0
        shrinkGuard = 0
        reaskGuard = 0
        finishNudge = 0
        failStreak = 0
        failedTotal = 0
        lastFail = nil
        failSigs = []
        stepFails = 0
        blockNudge = 0
        didWork = false
        hasToolSuccess = false
        didFallbackFetch = false
        didFallbackSearch = false
        didFallbackMemorySearch = false
        didFallbackKBStatus = false
        fallbackToolText = ""
        planNudged = 0
        planFinishNudge = 0
        intentStall = 0
        planGateHintInjected = false
        dynamicContextInjected = false
        repeatStreak = 0
        lastCallSigs = []
        successSigs.removeAll()
        reviewRounds = 0
        reviewCandidate = ""
        ragSearchCount = 0
        self.resume = resume
        if let seed = pendingSeed {
            failedTotal = seed.failedTotal
            lastFail = seed.lastFail
            failSigs = Array(seed.failSigs.suffix(maxFailSigs))
            failStreak = seed.failStreak
            finishNudge = seed.finishNudge
            planFinishNudge = seed.planFinishNudge
            blockNudge = seed.blockNudge
            emptyRetry = seed.emptyRetry
            reaskGuard = seed.reaskGuard
            planNudged = seed.planNudged
            didWork = seed.didWork
            stepFails = seed.stepFails
            repeatStreak = seed.repeatStreak
            lastCallSigs = Array(seed.lastCallSigs.suffix(8))
            intentStall = seed.intentStall
            shrinkGuard = seed.shrinkGuard
        }
        pendingSeed = nil
        turnsUsed = 0
        distillRequested = false

        var pending = userInput
        /// 出现过「只有思考没有正文」的空回复后，本轮剩余调用关闭 thinking，
        /// 让模型直接给出 <summary> + 正文（2026-11-08 cnn.com 事故）。
        var forceNoThinking = false
        var finalText = ""
        var exitReason: ExitReason = .maxTurns
        /// 触发 `.done` 退出的最后一轮是否为「无工具纯文字回答」。文字步骤
        /// （说明 / 总结 / 边界）本就靠文字收尾，不应被当成「步骤未完成」训诫。
        var finalTurnNoTool = false
        // 多模态图像只挂到「本次 run 的第一条用户消息」（即触发本轮的用户输入），
        // 不重复挂到后续的工具有/续跑轮次。resume（自动续跑）由 Agent 显式传空。
        var attachImages: [Data]? = images.isEmpty ? nil : images

        while turnsUsed < maxTurns {
            if cancelled() {
                log("[kernel] 收到取消请求，终止")
                exitReason = .cancelled
                break
            }

            // ---- CONSOLIDATE ----
            // 折叠只在启用了工具的正式轮进行：折叠会 invalidateKV 并触发
            // didFold → warmup，MLX 丢掉刚累积的会话缓存，GGUF 要重跑整段
            // 前缀 prefill。
            if toolsEnabled, needsConsolidate() {
                state = .consolidate
                lastConsolidatedTurn = session.currentTurn
                let foldReason: String
                // needsAutoShrink = 已用超过 90% 最大上下文：经上层
                // shrinkSession 折叠到 60% 回水线（逐轮自带 summary 的
                // 确定性折叠，不调 LLM），被折叠轮的摘要行回注 <earlier_context>。
                // 注意不用 totalOver（79% 输出预留线），那是预算分配量。
                if budget.needsAutoShrink, shrinkGuard < 3 {
                    foldReason = "context-90"
                    shrinkGuard += 1
                    let shrunk = await hooks.consolidated(session, reason: "context-90")
                    log("[kernel] CONSOLIDATE (上下文 >90%) shrunk=\(shrunk) "
                        + session.statsDescription())
                    if shrunk {
                        runtime.invalidateKV?()
                    } else {
                        // 上层没有接住（如未绑定回调）：退化成确定性折叠，
                        // 摘要行同样回注 earlier_context，保证不死循环。
                        let dropped = session.discardHistory(keepingRecent: 2)
                        if dropped > 0 { runtime.invalidateKV?() }
                    }
                } else {
                    foldReason = "periodic"
                    // 常规主动折叠：折叠掉的历史轮摘要行回注 earlier_context，
                    // 并保留最近 2 轮原文，避免折叠处上下文断裂。
                    let dropped = session.discardHistory(keepingRecent: 2)
                    if dropped > 0 { runtime.invalidateKV?() }
                    log("[kernel] CONSOLIDATE (每 \(consolidateEvery) 轮) "
                        + "dropped=\(dropped) 保留最近 2 轮 \(session.statsDescription())")
                }
                // 边界通知：刷新 system 前缀（L0/L1 变更延迟到这里生效）+ 预热 KV。
                await hooks.didFold(session, reason: foldReason, tools: schemas())
                _ = transition("done")
            }

            // ---- THINK ----
            state = .think
            // 规划门：还没干过实际活、且还没有计划时，持续提醒先规划再动手。
            // 规划提示一段只注入一次：旧实现每个 THINK 轮都 `pending +=`
            // 一遍，而 pending 会写进 rec.prompt 永久留在历史里，越重试
            // 上下文越长（实测 3614→5689 字符）。意图空转期间也不注入，
            // 避免与「直接调用工具」的指令互相打架。
            // 内核代建计划后的「立即执行」强指令：专治「反复描述计划不调工具」
            // 的空转。isMultistep 任务已由代建分支接管，这里不再注入 planGateHint，
            // 避免「先登记」提示诱发规划焦虑。
            if toolsEnabled && session.getPlan() == nil && !didWork
                && intentStall == 0 && !planGateHintInjected
                && !isSimpleInput(userInput) && !isMultistep(userInput) {
                pending += planGateHint
                planGateHintInjected = true
            }
            var userText = session.anchorPrompt(extra: pending)
            // 动态上下文只加在 user 消息尾部：system 与历史前缀保持不变，
            // 本地 KV cache 才能跨轮复用。
            // 动态上下文（命中技能/KB 说明）按「一次用户请求」
            // 冻结，同一段内逐轮注入只是把同一份 ~7.5k 字符重复写进历史，
            // 快速吃满 raw 窗口并触发提前折叠。只在首个 THINK 轮注入：后续轮
            // 模型在历史里能看到首轮那份；已注入的轮原样保留（KV 对齐铁律），
            // 由 beginUserRequest()/resetRuntimeState() 在新请求时重置。
            let dynamic = await dynamicContext()
            if !dynamic.isEmpty && !dynamicContextInjected {
                userText += "\n\n" + dynamic
                dynamicContextInjected = true
            }
            // KV 对齐铁律：写进 rec.prompt 的就是本轮实际发给模型的完整
            // user 文本，**事后不得裁剪、替换或压缩历史里的任何一轮**。
            // MLX 的 token 级前缀匹配 / 消息级 delta、GGUF 的 delta-append
            // 都依赖「Session 历史 == 缓存里的 token 序列」；一旦把训诫或
            // 被拒正文从历史里抹掉，下一轮渲染会在很早的位置分叉，而
            // Qwen3.5 的 MambaCache 不可裁剪，只能整段重建（7~20s/轮）。
            // 所以减少上下文膨胀只能靠「少注入新增量」：一次性提示、精简
            // 文案、以及第 1/2 项减少被拒轮数，绝不能改写已落盘的历史。
            let rec = session.startTurn(prompt: userText, images: attachImages)
            attachImages = nil
            turnsUsed += 1
            let sys = systemPrompt()
            let toolSchemas = schemas()   // 先记 tools 账，raw 硬顶才算得准
            // A：段内禁止滑窗。若把全部已完成轮发出去会越过 raw 硬顶，先一次性
            // 提前折叠（保留最近 2 轮 + 当前轮），把「每轮滑窗 → KV 反复全量
            // prefill」收敛成「每 N 轮一次可预期的重建」。
            if session.rawWouldExceedCeiling(systemTokens: TokenEstimator.estimate(sys)),
               lastEarlyFoldTurn != session.currentTurn {
                lastEarlyFoldTurn = session.currentTurn
                let dropped = session.discardHistory(keepingRecent: 3)
                if dropped > 0 {
                    runtime.invalidateKV?()
                    log("[kernel] 提前折叠（raw 窗口将越顶）dropped=\(dropped) "
                        + "保留最近 2 轮 \(session.statsDescription())")
                    await hooks.didFold(session, reason: "early-raw-ceiling",
                                        tools: toolSchemas)
                }
            }
            let messages = session.buildMessages(systemPrompt: sys, userText: userText)

            var respOptional: LLMResponse? = nil
            do {
                respOptional = try await ask(
                    messages, toolsEnabled: toolsEnabled,
                    options: LLMRequestOptions(
                        enableThinking: forceNoThinking ? false : nil))
            } catch {
                let msg = describe(error)
                // 上下文超长：丢弃旧历史后重试同一轮，不生成摘要回注。
                // 只有「占用 >90%」的自动收缩才允许把旧摘要写回上下文。
                if isContextError(msg) && shrinkGuard < 3 {
                    shrinkGuard += 1
                    session.rollbackTurn()
                    runtime.invalidateKV?()
                    let dropped = session.resetForFreshContext()
                    log("[kernel] 上下文超长，丢弃 \(dropped) 轮旧历史后重试 (\(shrinkGuard))")
                    await hooks.didFold(session, reason: "context-limit", tools: schemas())
                    continue
                }
                log("[kernel] LLM error: \(msg)")
                hooks.failed(session, error: msg)
                exitReason = .error
                break
            }

            guard let resp = respOptional else {
                // 流式期间被取消：这一轮只有 user 没有 assistant，回退掉。
                session.rollbackTurn()
                exitReason = .cancelled
                break
            }

            if !resp.hasCalls {
                // 空回复/截断必须先于一切进度门处理：连正文都没有的轮次
                // 根本谈不上「没写计划」「步骤未完成」，旧顺序让空正文先被
                // planFinish 门白耗 3 次重答，才轮到真正的对症重试。
                let event = noToolEvent(resp)
                // 核验轮自己没吐出可见内容时，绝不启动「空正文 3 次重试」链：
                // 候选答复已经在手，重试只会把「没结果」放大成整轮失败。
                if event == "retry" && reviewRounds > 0 {
                    session.finishTurn(rec, assistantText: resp.content,
                                       summary: l2Summary(resp.content),
                                       rawContent: resp.rawContent)
                    finalText = reviewCandidate
                    finalTurnNoTool = true
                    exitReason = .done
                    log("[kernel] 成果核验轮无可见输出，收下上一轮候选")
                    break
                }
                if event == "retry" {
                    // 提交本轮（可能为空回复）：回滚会让下一轮消息前缀与
                    // KV cache 错位（unexpected-assistant 全量重建）。空回复
                    // 提交后 turnMessages 只含 user 消息，与 cache 追加对齐。
                    session.finishTurn(rec, assistantText: resp.content,
                                       summary: l2Summary(resp.content),
                                       rawContent: resp.rawContent)
                    if resp.content
                        .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        // 只有思考没有正文：关闭 thinking 重试，否则同样的
                        // 长思考会原样再来一遍（3 次全空后只能空手退出）。
                        forceNoThinking = true
                    }
                    pending = resp.truncated
                        ? "[ERROR] 上一轮的工具调用被截断：参数还没输出完生成就结束了，工具没有执行。请重新生成，并把较大的写入内容拆成多次较小的工具调用，不要把长文件塞进一次调用。"
                        : """
                        [ERROR] 上一轮回复为空或流被截断（很可能只有思考、没有正文）。请直接输出 <summary>…</summary> 和简短正文作答，不要长篇思考、不要重复分析过程；如需操作必须显式调用工具。系统已在本轮关闭思考模式。
                        """
                    continue
                }
                if event == "reask" {
                    reaskGuard += 1
                    if reaskGuard <= 1 {
                        // 同反思门：提交本轮，保持 KV 前缀与历史一致。
                        session.finishTurn(rec, assistantText: resp.content,
                                           summary: l2Summary(resp.content),
                                           rawContent: resp.rawContent)
                        pending = """
                        [System] 检测到你上轮主要内容是大段代码却未调用工具。若这些代码需要执行/写入/分析，请显式调用 code_run / file_tools(op=\"write\") / file_tools(op=\"patch\")；若只是展示给用户的示例，请在回复中补充自然语言说明。
                        """
                        continue
                    }
                    // 第二次不再纠缠：把这轮的正文当作最终答复收下。
                    finalText = resp.content
                    session.finishTurn(rec, assistantText: resp.content,
                                       summary: l2Summary(resp.content),
                                       rawContent: resp.rawContent)
                    exitReason = .done
                    break
                }
                // 重试耗尽仍无可见正文：不再把空串交给上层兜底，而是把失败
                // 原因、已有工具结果与计划状态一次讲清楚（知道怎么失败，
                // 而不是让计划静默卡死或"假装完成"）。
                let visibleNow = stripSummaryTags(resp.content)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if visibleNow.isEmpty && event != "intent" {
                    markPlanBlocked(reason: "生成连续为空/被截断，未能推进")
                    finalText = "（模型连续 \(maxEmptyRetry) 轮未输出可见正文——"
                        + "通常只有思考且被截断、或思考后未给出正文。本轮已停止，"
                        + "系统已自动关闭思考模式重试仍未解决。"
                        + summarizeToolResults()
                        + "\n建议：换用工具调用能力更强的模型，或把任务拆成单步指令后重试。）"
                    session.finishTurn(rec, assistantText: resp.content,
                                       summary: l2Summary(resp.content),
                                       rawContent: resp.rawContent)
                    exitReason = .done
                    log("[kernel] 空正文重试耗尽：已标记计划受阻并交付失败说明")
                    break
                }

                // 反思门：模型想收尾，但计划里还有未完成步骤 —— 不能就这么放它走。
                // （含「复述一句用户请求就 EOS」的过短回复：isMultistep 现已
                // 识别「获取/查询/搜索」类任务，统一走本门的 3 次推进预算。）
                let openSteps = session.planOpenSteps()

                // 计划状态被 update_plan 全部标成 done/skipped，但本段从未调用
                // 任何实际工具：这不是完成，而是绕过执行。必须逼它先拿到真实
                // 工具结果；提醒预算耗尽后，结尾会把虚假 done 降级为 skipped。
                let planClosedWithoutWork = toolsEnabled && !didWork
                    && isMultistep(userInput)
                    && session.getPlan() != nil && openSteps.isEmpty
                if planClosedWithoutWork, finishNudge < finishNudgePlanMax {
                    finishNudge += 1
                    session.finishTurn(rec, assistantText: resp.content,
                                       summary: l2Summary(resp.content),
                                       rawContent: resp.rawContent)
                    pending = """
                    [System] 计划里的步骤已经被 update_plan 标成完成/跳过，但本段从未调用任何实际工具——update_plan 只改计划状态，不能替代文件读取 / 检索 / 执行，也不能用 create 时直接写 status="done" 绕过执行。
                    若任务需要外部或文件信息，现在必须发出一个真实工具调用（如 web_search / web_scan / rag_tools(op="search") / file_tools / shell_run），拿到结果后再收尾。
                    若确实完全不需要工具，请在正文中说明依据；内核还会再提醒 \(finishNudgePlanMax - finishNudge) 次。
                    """
                    log("[kernel] 计划已闭合但零实际工具调用，提醒 \(finishNudge)/\(finishNudgePlanMax)")
                    continue
                }

                // 「我要先创建计划 / 现在开始执行第一步」这类纯意图声明：
                // 既没有工具调用也没有任何实质结果，绝不能当最终答复收下。
                // 旧路径会由 noToolEvent 判 done → reason=done、plan 仍为空
                // 又不触发自动续跑，用户反复点「继续」只得到同样的自我复述
                // （2026-11-08 cnn.com 事故）。这里给一条真正走得通的路：
                // 跳过计划门、直接点名真实工具、只要求输出工具调用。
                // resume（自动续跑）段同样保留意图检测：旧实现用 !resume
                // 把所有门控一并关掉，续跑段第一句元陈述就被当最终答复。
                // 可见正文里的元陈述（event=done）也归入同一门；长期记忆兜底
                // 同样依据这个判定，web_scan 兜底仍只在正文为空的 event=intent
                // 时触发（避免改变既有 cnn.com 事故的收敛行为）。
                let intentOnly = event == "intent"
                    || (!didWork && Self.looksLikeIntentOnly(resp.content))
                if toolsEnabled, intentOnly {
                    intentStall += 1
                    // 只有思考没有正文的意图空转（实测 MiniCPM5：工具结果后的
                    // 收尾轮在 <think> 里把答案想完就 EOS，正文为空），下一轮必须
                    // 关闭 thinking，否则同样的长思考原样再来一遍形成 3 次空转。
                    // 与 retry 路径（726 行）的 forceNoThinking 处理保持一致。
                    if resp.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        forceNoThinking = true
                    }
                    // 空转场景也要保证计划面板有内容：多步任务模型零工具调用、
                    // 只描述计划，面板会一直空着，用户只看到「直接失败」。这里
                    // 补建最小计划（不拦截执行），让卡点可见、可恢复。
                    if session.getPlan() == nil, isMultistep(userInput) {
                        let plan = Plan(goal: String(userInput.prefix(80)), steps: [
                            PlanStep(id: "1",
                                     title: "立即调用工具获取真实数据（如 web_scan / rag_tools(op=\"search\")）",
                                     status: "doing"),
                            PlanStep(id: "2", title: "基于工具结果整理并给出最终答复",
                                     status: "pending"),
                        ])
                        session.setPlan(plan)
                        do {
                            try PlanStore.save(plan, config: config)
                        } catch {
                            AgentLog.warning("[kernel] 计划落盘失败: \(error.localizedDescription)")
                        }
                        callbacks.onPlanUpdated(plan)
                        log("[kernel] 意图空转场景内核补建最小计划（不拦截工具执行）")
                    }
                    session.finishTurn(rec, assistantText: resp.content,
                                       summary: l2Summary(resp.content),
                                       rawContent: resp.rawContent)
                    if intentStall <= maxIntentStall {
                        let names = toolSchemas.compactMap {
                            $0["function"]?["name"]?.stringValue
                        }
                        let toolList = names.isEmpty
                            ? "web_scan / shell_run / file_tools(op=\"read\") / code_run"
                            : names.joined(separator: " / ")
                        // 用户明确给了网址时，给出可直接照抄的调用参数：小模型
                        // 看到精确示例比抽象指令更容易真的发出 tool_call。
                        let urlExample = Self.firstWebURL(in: userInput).map { url in
                            "\n可直接照抄这一条（不要改写 URL）："
                                + "web_scan({\"url\": \"\(url)\"})。"
                        } ?? ""
                        // 问「知识库里有什么」时同样给出可照抄的调用：
                        // 实测 4B 模型会连轮只输出 <summary> 而不发 tool_call。
                        let kbExample = Self.looksLikeKnowledgeBaseQuestion(userInput)
                            ? "\n可直接照抄：rag_tools({\"op\": \"status\"})；"
                                + "查内容用 rag_tools({\"op\": \"search\", "
                                + "\"query\": \"\(Self.jsonExampleString(userInput))\"})。"
                            : ""
                        // 问技能库时给语义检索示例；问本地文件时给读取示例。
                        let skillExample = Self.looksLikeSkillQuestion(userInput)
                            ? "\n可直接照抄：skill_tools({\"op\": \"semantic_search\", "
                                + "\"query\": \"\(Self.jsonExampleString(userInput))\"})。"
                            : ""
                        let fileExample = Self.firstLocalPath(in: userInput).map { path in
                            "\n可直接照抄：file_tools({\"op\": \"read\", "
                                + "\"path\": \"\(Self.jsonExampleString(path))\"})。"
                        } ?? ""
                        // 问「你还记得…」时同样给出可照抄的调用：小模型会反复
                        // 说「需要调用 long_memory」但不知道复合工具的 op 参数。
                        let memoryExample = Self.looksLikeMemoryQuestion(userInput)
                            ? "\n可直接照抄这一条：long_memory({\"op\": \"search\", "
                                + "\"query\": \"\(Self.jsonExampleString(userInput))\"})。"
                            : ""
                        // 已有 RAG 成功结果时，把「立即调用工具/检索」前置步骤标 done，
                        // 否则模型一直看着 ▶ doing 的「先取数据」，反复声称拿到了数据却不作答。
                        if didWork, hasToolSuccess, hasSuccessfulRagResult() {
                            completeStalledRagPlan()
                        }
                        pending = """
                        [System] 上一条回复只是描述了你打算做什么，没有发出任何工具调用，任务没有任何进展。
                        历史里此前「我要创建计划」「现在开始执行第一步」之类的文字都是无效回复，不要复述它们。
                        注意：<summary>…</summary> 只是内部摘要，既不是给用户的答复，也不能代替工具调用。
                        本轮**禁止输出 `<summary>` 标签**，也不要输出任何正文。
                        现在**跳过 update_plan**，不要再创建计划、不要复述任务、不要输出「我需要」「让我」「现在开始」这类文字。
                        \(hasSuccessfulRagResult() ? "[System] 知识库检索已完成，前置数据已就绪：现在直接基于上方 RAG 结果给出最终答复，不要再调用任何工具。" : "直接发出一个真实工具调用去获取所需信息，然后基于工具返回结果作答。")\(urlExample)\(kbExample)\(skillExample)\(fileExample)\(memoryExample)
                        可用工具：\(toolList)。
                        本轮只输出工具调用本身，不要输出其它解释文字。
                        """
                        log("[kernel] 意图空转 \(intentStall)/\(maxIntentStall)："
                            + "跳过计划门，强制直接调用工具")
                        continue
                    }
                    // 三次强制之后模型仍不发工具调用，而用户在问「知识库里有什么」：
                    // 内核代发一次 rag_tools(op="status")，把真实统计交给模型总结。
                    // 弱模型常把锚点提示里的 <PLAN> 整段复述成答复而非发 tool_call，
                    // 兜底保证用户拿到真实数据且不会看到内部协议。
                    let isDocContentQuestion = Self.looksLikeDocumentContentQuestion(userInput)
                        && hasSuccessfulRagResult()
                    if !didFallbackKBStatus, intentOnly,
                       Self.looksLikeKnowledgeBaseQuestion(userInput) || isDocContentQuestion {
                        didFallbackKBStatus = true
                        // 清单问句→status；"X 主要讲什么"→outline/search（用用户原话检索）。
                        let fallbackArgs: [String: JSONValue] = isDocContentQuestion
                            ? ["op": .string("search"), "query": .string(userInput)]
                            : ["op": .string("status")]
                        let call = ToolCall(id: "", name: "rag_tools",
                                            arguments: fallbackArgs)
                        var stats: [ToolResult] = []
                        do {
                            stats = try await runtime.execute([call])
                        } catch {
                            log("[kernel] 知识库状态兜底查询失败：\(describe(error))")
                        }
                        if let first = stats.first, first.success {
                            didWork = true
                            fallbackToolText = first.output
                            let head = String(first.output.prefix(8_000))
                            let opLabel = isDocContentQuestion ? "op=search" : "op=status"
                            let answerHint = isDocContentQuestion
                                ? "基于检索到的内容总结该书讲了什么" : "文档数、切片数、检索策略等"
                            pending = """
                            [System] 内核已代你执行 rag_tools \(opLabel)，结果如下。
                            **不要再调用任何工具，也不要复述 <PLAN> / <WORKING MEMORY> 等
                            内部提示，也不要只输出 <summary> 元标签**：直接基于下面的
                            真实结果用中文回答用户问题（\(answerHint)），
                            没有数据就明确说明，不要编造。

                            \(head)
                            """
                            log("[kernel] 意图空转耗尽：内核代发 rag_tools(status)"
                                + "（\(first.output.count) 字符）")
                            continue
                        }
                    }
                    // 三次强制之后模型仍不发工具调用，而用户在问长期记忆：
                    // 内核代发一次 long_memory(op="search")，把真实记忆交给
                    // 模型总结（只读操作；结果只进下一轮 user 提示，不伪造
                    // tool_calls 写入 Session）。
                    if !didFallbackMemorySearch, intentOnly,
                       Self.looksLikeMemoryQuestion(userInput) {
                        didFallbackMemorySearch = true
                        let call = ToolCall(id: "", name: "long_memory",
                                            arguments: ["op": .string("search"),
                                                        "query": .string(userInput)])
                        var recalled: [ToolResult] = []
                        do {
                            recalled = try await runtime.execute([call])
                        } catch {
                            log("[kernel] 长期记忆兜底检索失败：\(describe(error))")
                        }
                        if let first = recalled.first, first.success {
                            didWork = true
                            fallbackToolText = first.output
                            let head = String(first.output.prefix(8_000))
                            pending = """
                            [System] 内核已代你执行 long_memory(op="search")，结果如下。
                            **不要再调用任何工具，也不要只输出 <summary> 元标签**：
                            直接基于下面的长期记忆结果用中文回答；没有命中就明确说
                            「没有找到相关记忆」，不要编造。

                            \(head)
                            """
                            log("[kernel] 意图空转耗尽：内核代发 long_memory 检索"
                                + "（\(first.output.count) 字符）")
                            continue
                        }
                    }
                    // 三次强制之后模型仍不发工具调用，而用户明确给了网址：
                    // 内核代发一次 web_scan，把真实数据交给模型总结。工具结果
                    // 只进下一轮 user 提示，不伪造 tool_calls 写进 Session，
                    // 因此「Session 历史 == KV 缓存 token 序列」不被破坏。
                    if !didFallbackFetch, event == "intent",
                       let url = Self.firstWebURL(in: userInput) {
                        didFallbackFetch = true
                        let call = ToolCall(id: "", name: "web_scan",
                                            arguments: ["url": .string(url)])
                        var fetched: [ToolResult] = []
                        do {
                            fetched = try await runtime.execute([call])
                        } catch {
                            log("[kernel] 兜底抓取失败：\(describe(error))")
                        }
                        if let first = fetched.first, first.success,
                           !first.output.isEmpty {
                            fallbackToolText = first.output
                            didWork = true
                            let head = String(first.output.prefix(8_000))
                            pending = """
                            [System] 内核已代你执行 web_scan 取得真实数据（见下方）。
                            **不要再调用任何工具，也不要只输出 <summary> 元标签**：
                            直接基于下面的真实内容，用中文写一份分点的最新新闻总结作为最终答复。
                            只写真实出现的内容，不要编造标题或数字。

                            \(head)
                            """
                            log("[kernel] 意图空转耗尽：内核代发 web_scan 兜底抓取"
                                + "（\(first.output.count) 字符）")
                            continue
                        }
                    }
                    // 三次强制之后模型仍不发调用，且任务不是给网址 / 本地文件 /
                    // 知识库 / 技能 / 长期记忆：代发一次 web_search，把真实搜索结果
                    // 交给模型收尾。搜索类任务不再因小模型只会描述计划而空转失败。
                    // 文档内容问句（"X 主要讲什么"）即使不含"知识库"字样，只要历史里
                    // 已有 RAG 命中，就属于知识库问答，绝不能代发 web_search 跑偏。
                    let shouldNotWebSearch = Self.looksLikeKnowledgeBaseQuestion(userInput)
                        || (Self.looksLikeDocumentContentQuestion(userInput)
                            && hasSuccessfulRagResult())
                    if !didFallbackSearch, !resume, intentOnly,
                       Self.firstWebURL(in: userInput) == nil,
                       Self.firstLocalPath(in: userInput) == nil,
                       !shouldNotWebSearch,
                       !Self.looksLikeSkillQuestion(userInput),
                       !Self.looksLikeMemoryQuestion(userInput) {
                        didFallbackSearch = true
                        let call = ToolCall(
                            id: "", name: "web_search",
                            arguments: ["query": .string(userInput),
                                        "count": .int(8)])
                        var searched: [ToolResult] = []
                        do {
                            searched = try await runtime.execute([call])
                        } catch {
                            log("[kernel] 兜底搜索失败：\(describe(error))")
                        }
                        if let first = searched.first, first.success,
                           !first.output.isEmpty {
                            fallbackToolText = first.output
                            didWork = true
                            let head = String(first.output.prefix(8_000))
                            pending = """
                            [System] 内核已代你执行 web_search，搜索结果如下。
                            **不要再调用任何工具，也不要只输出 <summary> 元标签**：
                            直接基于下面的搜索结果用中文回答；需要正文时从结果中取真实 URL 再 web_scan。
                            没有命中就明确说「没有找到相关信息」，不要编造。

                            \(head)
                            """
                            log("[kernel] 意图空转耗尽：内核代发 web_search"
                                + "（\(first.output.count) 字符）")
                            continue
                        }
                    }
                    finalText = "（模型连续 \(maxIntentStall + 1) 轮只描述了计划、"
                        + "没有发出任何工具调用，任务未能执行。请换用工具调用能力更强的模型，"
                        + "或把请求写成明确的单步指令后重试。）"
                    if !fallbackToolText.isEmpty {
                        finalText += "\n\n内核已代抓取的页面内容（供参考）：\n"
                            + String(fallbackToolText.prefix(2_000))
                    }
                    exitReason = .done
                    log("[kernel] 意图空转已达上限，返回明确失败说明而不是元陈述")
                    break
                }

                // 任务尚未起步，且模型是在向用户追问必要信息 → 收下提问。
                if session.getPlan() == nil && !didWork && looksLikeAskingUser(resp.content) {
                    finalText = resp.content
                    session.finishTurn(rec, assistantText: resp.content,
                                       summary: l2Summary(resp.content),
                                       rawContent: resp.rawContent)
                    log("[kernel] 任务未起步且模型在向用户追问必要信息 → 收下提问，交给用户回答")
                    exitReason = .done
                    break
                }

                // 多步任务却连一次工具都没执行就想收尾：不放行。这里不再用
                // looksLikeResult 豁免——v1.2.27 放行后用户实测出现「模型一句
                // 『已完成：xxx』根本没调工具就被当最终答复」，2 步任务的计划
                // 面板是空的、任务也没执行。没有 didWork 的「结果」只能是
                // 幻觉或对任务的复述；有 didWork 的纯文字收尾走
                // turnedTextAnswer 分支，不受此门影响。
                if session.getPlan() == nil
                    && (isMultistep(userInput))
                    && !didWork
                    && planFinishNudge < finishNudgePlanMax
                    && !resume {
                    planFinishNudge += 1
                    // 提交（而不是回滚）本轮：本地 LLM 的 KV cache 已包含这轮
                    // 生成内容，回滚会让 Session 历史与 cache 前缀错位，下一轮
                    // 只能全量 prefill（每轮卡顿的根因之一）。提交后提示作为
                    // 新一轮 pending，模型仍能在历史里看到自己被拒的回复。
                    session.finishTurn(rec, assistantText: resp.content,
                                       summary: l2Summary(resp.content),
                                       rawContent: resp.rawContent)
                    pending = planFinishHint
                    log("[kernel] 反思门：多步任务未写计划也未动手，强制要求先 create 并开工")
                    continue
                }

                let nudgeBudget: Int
                if failedTotal > 0 {
                    nudgeBudget = finishNudgeFailMax
                } else if !didWork {
                    nudgeBudget = finishNudgePlanMax
                } else {
                    nudgeBudget = finishNudgeMax
                }
                // 已完成实质工作、且本轮是「纯文字回答」（无工具调用）时，不再逼模型
                // 调工具——说明 / 总结 / 能力边界这类文字步骤本就不需要工具。否则会陷入
                // 「步骤 X 已完成，无需调用工具」与「你必须调用工具」的死循环，并可能把
                // 这句元陈述当成最终答复返回。直接放行到无工具即结束分支收下答案。
                let turnedTextAnswer = resp.toolCalls.isEmpty && didWork
                if !openSteps.isEmpty && finishNudge < nudgeBudget && !turnedTextAnswer {
                    finishNudge += 1
                    let todo = openSteps.prefix(5)
                        .map { "\($0.id).\($0.title)" }.joined(separator: "; ")
                    // 同上：提交本轮保持 KV cache 与 Session 历史前缀一致。
                    session.finishTurn(rec, assistantText: resp.content,
                                       summary: l2Summary(resp.content),
                                       rawContent: resp.rawContent)
                    if failedTotal > 0, let failure = lastFail {
                        // 失败感知的提醒：把最近一次失败摆到模型眼前，并要求
                        //「依据失败信息自动解决」而不是停下来等用户发话。
                        let tried = triedApproaches()
                        let triedText = tried.isEmpty
                            ? ""
                            : "\n已试过且失败的方案（不要再走）：\(tried)。"
                        pending = """
                        [System] 计划中还有 \(openSteps.count) 个未完成步骤：\(todo)。此前有工具调用失败（共 \(failedTotal) 次，最近：\(failure.name) → \(String(failure.error.prefix(200)))。\(triedText)
                        这是第 \(finishNudge) 次提醒。禁止停下来、禁止问用户「是否继续」、禁止只输出文字总结——你必须**调用工具**往下推进：
                        1) 读完整错误，判定属于参数错误 / 路径不存在 / 权限 / 依赖缺失 / 语法错误 / 网络不可达 中的哪一类；
                        2) 针对该原因换一种**没试过**的方法（换工具、换参数、或用 code_run 写最小诊断脚本先验证假设）；
                        3) 至少尝试 3 种不同方法都确认不可行，才把该步标记 blocked 并在 note 写明原因，然后继续后续步骤。
                        现在就调用工具。
                        """
                    } else if !didWork {
                        let nowStep = openSteps.first(where: { $0.status == "doing" })
                            ?? openSteps[0]
                        let phrase = resume ? "在本段没有调用任何工具" : "还没有调用**实际工具**去推进"
                        pending = """
                        [System] 计划已建立，但你还\(phrase)当前步骤——只调 `update_plan` 不算推进，它只更新计划状态、不能替代真实的文件读取 / 检索 / 执行。这是第 \(finishNudge) 次提醒。
                        现在立即开始执行当前步骤（Now）：\(nowStep.id). \(nowStep.title)。
                        要求：下一轮必须包含**实际工具调用**（读文件 / 执行命令 / 检索 / 写代码等），不要只描述你打算做什么，不要停下来问用户是否开始或是否继续。
                        """
                    } else {
                        pending = """
                        [System] 计划中还有 \(openSteps.count) 个未完成步骤：\(todo)。请继续执行下一步（调用相应工具）。注意：若某步的工作**实际已经完成**只是忘了更新状态，用 update_plan(op="patch", step_id=..., status="done", note="结论") 收尾即可，不要为响应提醒而重复已完成的工作；若某步确实无法完成，把它标为 blocked/skipped 并写明原因。
                        """
                    }
                    log("[kernel] 反思门：还有 \(openSteps.count) 步未完成，提醒继续")
                    continue
                }
                // ---- 内核决策门（complete）：收尾拦截 ----
                // 模拟实验显示决策模型对「已收集信息是否足以完整回答」判定
                // 6/6 全对：模型认为可以收尾时再问一次决策模型，不足则打回
                // 继续调用工具补齐。判定失败/超时返回 nil → 静默放行；
                // 成果核验轮（reviewRounds > 0）与预算耗尽后不再拦截。
                // 注意：不要求 openSteps.isEmpty——内核会为任何工具任务代建
                // 最小计划，模型收尾时常未 patch 状态，若要求计划闭合此门
                // 几乎永不触发；开放步骤的「继续执行」提醒由反思门负责，
                // 纯文字收尾（turnedTextAnswer）的质量由本门把关。
                if toolsEnabled, didWork, hasToolSuccess,
                   !decisionPrechecked,
                   reviewRounds == 0,
                   let judge = decisionJudge,
                   decisionFinishBlocks < config.kernelDecisionGateMaxFinishBlocks {
                    let sufficient = await judge.informationSufficient(
                        task: userInput,
                        evidence: decisionEvidence(),
                        draft: stripSummaryTags(resp.content))
                    if sufficient == false {
                        decisionFinishBlocks += 1
                        // 提交本轮保持 KV cache 与 Session 历史前缀一致（与
                        // 反思门同款），拦截提示作为下一轮 pending 注入。
                        session.finishTurn(rec, assistantText: resp.content,
                                           summary: l2Summary(resp.content),
                                           rawContent: resp.rawContent)
                        pending = """
                        [System] 决策模型判定当前收集的信息可能不足以完整回答任务（第 \(decisionFinishBlocks)/\(config.kernelDecisionGateMaxFinishBlocks) 次拦截）。
                        请继续调用工具补齐缺失信息（换关键词检索、读取其它来源、检查遗漏的必要数据点）；
                        若确认无法获取，请在答复中明确说明缺失的部分与原因后再收尾，不要编造。
                        """
                        log("[kernel] 决策门：拦截收尾 "
                            + "\(decisionFinishBlocks)/\(config.kernelDecisionGateMaxFinishBlocks)（信息不足）")
                        continue
                    }
                    if sufficient == true {
                        log("[kernel] 决策门：信息足够，放行收尾")
                    }
                }

                // 成果核验反思：以从业 10 年的资深专家身份审视最终结果，质量
                // 不达标最多打回 2 次完善。只在「多步任务 + 有实质工作」时开启
                // ——单步工具操作、简单问答不核验，避免每次收尾白多一轮。
                if toolsEnabled && didWork && hasToolSuccess
                    && isMultistep(userInput) {
                    if reviewRounds == 0 {
                        // 第一轮核验：把当前答复记为候选，注入专家核验提示。
                        // 保持默认请求通道：请求级关闭 thinking 会新建
                        // isolated 副本、丢掉 KV 前缀（真机每轮全量 prefill
                        // 约 80s）。核验轮无输出/只吐元标记都有收候选保护。
                        reviewCandidate = resp.content
                        reviewRounds = 1
                        session.finishTurn(rec, assistantText: resp.content,
                                           summary: l2Summary(resp.content),
                                           rawContent: resp.rawContent)
                        pending = Self.expertReviewHint(round: 1, max: reviewMax)
                        log("[kernel] 成果核验反思 1/\(reviewMax)：专家视角审视结果")
                        continue
                    }
                    if Self.isReviewPass(resp.content)
                        || Self.isReviewMetaOnly(resp.content) {
                        // pass 或只吐 `<review>fix</review>` 这类元标记：绝不
                        // 能把元标记当最终答复（2026-11-09 cnn.com 实测把整份
                        // 新闻总结覆盖成一句 fix），收下候选。
                        finalText = reviewCandidate
                        if !Self.isReviewPass(resp.content) {
                            log("[kernel] 成果核验：模型只输出元标记，收下候选")
                        }
                    } else if Self.isNoImprovement(reviewCandidate, resp.content) {
                        // 反思没带来实质改进（只是把候选重复堆叠），收下干净的
                        // 单次候选，不再为「完善」而反复刷屏。
                        finalText = reviewCandidate
                        log("[kernel] 成果核验：反思未带来实质改进，收下候选")
                    } else if reviewRounds < reviewMax {
                        // 专家给出完善后的答复，再核验一轮。
                        reviewCandidate = resp.content
                        reviewRounds += 1
                        session.finishTurn(rec, assistantText: resp.content,
                                           summary: l2Summary(resp.content),
                                           rawContent: resp.rawContent)
                        pending = Self.expertReviewHint(round: reviewRounds,
                                                        max: reviewMax)
                        log("[kernel] 成果核验反思 \(reviewRounds)/\(reviewMax)：继续完善")
                        continue
                    } else {
                        // 达到上限：收下本轮答复。
                        finalText = resp.content
                    }
                } else {
                    finalText = resp.content
                }
                session.finishTurn(rec, assistantText: resp.content,
                                   summary: l2Summary(resp.content),
                                   rawContent: resp.rawContent)
                finalTurnNoTool = true
                exitReason = .done
                break
            }

            // ---- ACT ----
            _ = transition("has_calls")
            let toolCalls = resp.toolCalls

            // ---- 规划前置门：先于执行 ----
            let hasActual = toolCalls.contains { $0.name != "update_plan" }
            let hasPlanOp = toolCalls.contains { $0.name == "update_plan" }
            // 计划面板保障：多步任务（isMultistep 命中，或已实际发出工具调用）
            // 而模型尚未自觉建计划时，内核代建一条最小计划并同步面板/落盘。
            // 旧实现只在「已发工具调用」后才代建，导致「先规划」提示诱发
            // 的空转场景（模型反复描述计划却零工具调用）永远建不起计划、面板
            // 始终为空。代建**不拦截执行**，只保证面板有内容、进度可见、反思
            // 门与收尾判定有据可依；模型随后仍可 update_plan 覆盖或细化。
            if toolsEnabled, session.getPlan() == nil,
               (hasActual || isMultistep(userInput)), !hasPlanOp {
                let plan = Plan(goal: String(userInput.prefix(80)), steps: [
                    PlanStep(id: "1",
                             title: "立即调用工具获取真实数据（如 web_scan / rag_tools(op=\"search\")）",
                             status: "doing"),
                    PlanStep(id: "2", title: "基于工具结果整理并给出最终答复",
                             status: "pending"),
                ])
                session.setPlan(plan)
                do {
                    try PlanStore.save(plan, config: config)
                } catch {
                    AgentLog.warning("[kernel] 计划落盘失败: \(error.localizedDescription)")
                }
                callbacks.onPlanUpdated(plan)
                log("[kernel] 多步任务模型未建计划，内核代建最小计划（不拦截工具执行）")
            }

            // ---- 大纲后检索次数硬上限（默认 3）----
            // 每次 `rag_tools(op="outline")` 开启新一轮「检索预算」并清零；超出上限的 `rag_tools(op="search")`
            // 直接预置失败，逼模型基于已有结果与大纲摘要综合作答，杜绝无效重复检索。
            var executeCalls: [ToolCall] = []
            var preset: [Int: ToolResult] = [:]
            var liveCount = ragSearchCount
            for (i, tc) in toolCalls.enumerated() {
                let sig = callSig(tc)
                let tcOp = tc.arguments["op"]?.stringValue ?? ""
                let isRagOutline = tc.name == "rag_tools" && tcOp == "outline"
                // 已经以完全相同参数失败过的调用，再执行必然还是失败。旧版只在
                // 事后给软提示，模型仍把同一个 file_tools(read) / shell_run cat 原样
                // 重发三次（知识库问答里读臆造的 .md 路径），整轮被耗空。
                if failSigs.contains(sig) {
                    preset[i] = .fail(
                        "该调用此前已用完全相同的参数失败过，已跳过（原样重试不会改变结果）。"
                        + "请换参数/换工具，或直接依据已获得的信息作答；"
                        + "已索引文档的内容用 rag_tools 获取，不要重复读不存在的路径。"
                        + "外部网站与本地文件任务请改用 web_scan / file_tools / shell_run 等对应工具。")
                    continue
                }
                // 成功过的调用原样重发：结果已在上一轮返回、不会再变。模型常
                // 「拿到 web_scan 结果却反复重发相同 URL 刷屏」，旧逻辑只给软
                // 提示，模型无视后连刷 4~5 次才被终止。这里直接拦截，逼它基于
                // 已有结果作答，或换参数/换工具获取其它内容。
                // 排除控制类工具：rag_tools(outline) 重复调用会重置检索预算、update_plan
                // 会更新计划状态，重复执行是必要的，不能被「成功过」拦掉。
                if successSigs.contains(sig),
                   !isRagOutline, tc.name != "update_plan" {
                    preset[i] = .fail(
                        "该调用此前已成功返回过结果（内容见上一轮历史）。完全相同参数"
                        + "再调一次不会带来任何新信息——请直接基于已获得的结果作答、"
                        + "给出最终总结；若确需其它内容，请换参数 / 换工具（如翻页、"
                        + "换 URL、用 rag_tools(op=\"search\") / ask_user 等），不要原样重发。")
                    continue
                }
                // 知识库内容问句：历史已有 RAG 命中时禁止过早转 web_search。
                // 只有 RAG 检索 ≥2 次仍判信息不足才允许联网补充（用户要实时信息除外）。
                if tc.name == "web_search",
                   !Self.looksLikeKnowledgeBaseQuestion(userInput),
                   Self.looksLikeDocumentContentQuestion(userInput),
                   hasSuccessfulRagResult(), ragSearchCount < 2 {
                    let hint = "[System] 已检索到该书/文档的切片（见上方工具结果）。"
                        + "优先基于 RAG 结果直接作答，不要转 web_search；"
                        + "仅当已检索 \(ragSearchCount) 次仍确认信息不足时，才可联网补充。"
                    preset[i] = .fail("知识库已有相关内容，暂不转联网。\(hint)")
                    log("[kernel] 拦截过早 web_search：RAG 已命中，检索 \(ragSearchCount) 次")
                    pending = (pending.isEmpty ? "" : pending + "\n\n") + hint
                    continue
                }
                switch (tc.name, tcOp) {
                case ("rag_tools", "outline"):
                    // 不再重置检索计数：outline 反复调用会把「每次 outline 后
                    // ≤3 次 search」的预算无限刷新，实测同一任务检索 5 次仍不够，
                    // 每轮多付 6~12s prefill。计数只在 beginRequest/reset 时归零。
                    executeCalls.append(tc)
                case ("rag_tools", "search"):
                    if liveCount >= maxRagSearchesPerOutline {
                        preset[i] = .fail(
                            "已达单次大纲后的检索上限（\(maxRagSearchesPerOutline) 次）。"
                            + "请基于已检索内容与大纲摘要直接综合作答，不要再调用 "
                            + "rag_tools(op=\"search\")。")
                        continue
                    }
                    liveCount += 1
                    executeCalls.append(tc)
                default:
                    executeCalls.append(tc)
                }
            }
            if liveCount != ragSearchCount {
                log("[kernel] 大纲后检索计数 \(ragSearchCount) → \(liveCount)"
                    + "（上限 \(maxRagSearchesPerOutline)）")
            }
            ragSearchCount = liveCount

            let rawResults: [ToolResult]
            do {
                let executed = try await runtime.execute(executeCalls)
                var it = executed.makeIterator()
                rawResults = toolCalls.enumerated().map { (i, _) in
                    if let pre = preset[i] { return pre }
                    // 并行/串行分支返回数量不足时不能强解包崩掉整个内核。
                    return it.next() ?? .fail("工具调度返回的结果数量不足")
                }
            } catch {
                log("[kernel] dispatcher error: \(describe(error))")
                rawResults = toolCalls.enumerated().map { (i, _) in
                    preset[i] ?? .fail("工具未返回结果")
                }
            }

            // 失败记账必须在拼装 pending 之前：步骤内失败数要在 update_plan
            // 把它重置之前，被「未充分尝试就 blocked」的守卫读走。
            let stepFailsBefore = stepFails
            let batch = trackFailures(toolCalls, rawResults)
            // 记录成功签名：成功返回的调用，下次原样重发时在 preset 阶段直接拦截。
            for (i, tc) in toolCalls.enumerated() {
                if i < rawResults.count, rawResults[i].success {
                    successSigs.insert(callSig(tc))
                }
            }
            // rag_search 已确认找到匹配材料时视为达成完成度，生成前门与收尾门
            // 都不再判定（即使本批工具混有 update_plan 等非检索调用）：
            // - 向量检索：召回最高相似度 > 0.5，或判定确认存在相关切片（任一即可）；
            // - 极速检索：问句相关度判定确实执行且至少一条切片相关。
            // 不校验 op：rag_tools 省略 op 且带 query 时默认就是 search。
            for (i, result) in rawResults.enumerated()
            where i < toolCalls.count && toolCalls[i].name == "rag_tools" {
                let maxScore = result.metrics["rag_max_score"] ?? 0
                let confirmed = (result.metrics["rag_relevance_confirmed"] ?? 0) > 0.5
                guard maxScore > 0.5 || confirmed else { continue }
                if !ragHighConfidence {
                    log(confirmed
                        ? "[kernel] rag_search 已判定存在与问句相关的切片，跳过完成度判定"
                        : "[kernel] rag_search 最高相似度 > 0.5，跳过完成度判定")
                }
                ragHighConfidence = true
            }
            // 只有真正发出非 update_plan 的工具调用才算开工。
            // `update_plan` 只改计划状态，绝不能算作「已经做过事」，否则
            // 模型可以 create 时直接写 status="done"、再输出一段文字就收尾，
            // 计划面板显示完成但实际一个工具都没执行（真实空转漏洞）。
            if hasActual { didWork = true }
            // 只有真正非 update_plan 的工具成功返回过内容才让成果核验门开启：
            // preset 预置失败 / dispatcher 异常都会让 hasActual 为真，
            // update_plan 的成功也不能当作「有工具结果」。
            for (index, call) in toolCalls.enumerated()
            where call.name != "update_plan" {
                guard index < rawResults.count,
                      rawResults[index].success,
                      !rawResults[index].output.isEmpty else { continue }
                hasToolSuccess = true
            }

            // ---- 内核决策门（plan_step）：计划评审 ----
            // 计划创建成功后评审每个未完成步骤是否必要；判「不必要」的步骤
            // 注入一次建议性提示（不阻断计划创建与执行）。同一计划只评审
            // 一次；判定失败/超时静默跳过。
            var planReviewWarning: String?
            let createdPlanThisTurn = toolCalls.enumerated().contains { idx, tc in
                tc.name == "update_plan"
                    && tc.arguments["op"]?.stringValue == "create"
                    && idx < rawResults.count && rawResults[idx].success
            }
            if let judge = decisionJudge, createdPlanThisTurn,
               let plan = session.getPlan(), !plan.steps.isEmpty {
                let key = "\(plan.goal)|\(plan.steps.count)|"
                    + plan.steps.map(\.title).joined(separator: "|")
                if decisionPlanReviews < config.kernelDecisionGateMaxPlanReviews,
                   key != decisionReviewedPlanKey {
                    decisionReviewedPlanKey = key
                    var unnecessary: [String] = []
                    for step in plan.steps
                    where step.status != "done" && step.status != "skipped" {
                        if await judge.stepNecessary(goal: plan.goal,
                                                     step: step.title) == false {
                            unnecessary.append(step.title)
                        }
                    }
                    if !unnecessary.isEmpty {
                        decisionPlanReviews += 1
                        planReviewWarning = """
                        [System] 决策模型认为计划中以下步骤可能对目标并非必要：\(unnecessary.joined(separator: "；"))。
                        请检查计划：删除纯复述性 / 无信息增量的步骤，补充缺失的关键步骤（信息来源、验证环节），
                        用 update_plan(op="patch") 或重新 create 更新后继续执行。若确认该步骤必要，忽略本提示。
                        """
                        log("[kernel] 决策门：计划评审发现 \(unnecessary.count) 个可能不必要的步骤")
                    } else {
                        log("[kernel] 决策门：计划评审通过（\(plan.steps.count) 步均必要）")
                    }
                }
            }

            // ---- 重复调用检测（成功调用也要管）----
            let sigs = toolCalls.map(callSig)
            if !sigs.isEmpty && sigs == lastCallSigs {
                repeatStreak += 1
            } else {
                repeatStreak = 0
            }
            lastCallSigs = sigs
            var repeatWarning: String?
            if repeatStreak >= repeatCallWarn {
                log("[kernel] 连续 \(repeatStreak + 1) 次相同的工具调用（\(toolCalls.first?.name ?? "?")）→ 判定卡住")
                if repeatStreak >= repeatCallLimit {
                    log("[kernel] 重复调用达到上限，结束本轮")
                    // 不能把最后一轮正文丢掉：模型可能已经给出了阶段性说明。
                    // 同时把计划里未完成步骤标 blocked：失败要可见、可恢复，
                    // 不能让面板永远停在 doing（2026-11-08 cnn.com 事故）。
                    markPlanBlocked(reason: "连续 \(repeatStreak + 1) 次完全相同的工具调用"
                        + "（\(toolCalls.first?.name ?? "?")）未能推进")
                    finalText = resp.content.trimmingCharacters(
                        in: .whitespacesAndNewlines).isEmpty
                        ? "（检测到连续 \(repeatStreak + 1) 次完全相同的工具调用，"
                            + "内核已停止本轮以避免空转。已把计划受阻原因记入计划面板。"
                            + summarizeToolResults()
                            + "\n建议：换用工具调用能力更强的模型，或把任务拆成单步指令后重试。）"
                        : resp.content
                    exitReason = .done
                    break
                }
                repeatWarning = """
                [System] 你已经连续 \(repeatStreak + 1) 次调用**完全相同**的工具与参数（\(toolCalls.first?.name ?? "?")）——这不会让任务前进哪怕一步。上一轮的工具结果已经返回，所需资料大概率已经拿到。立刻换一个动作：
                1) **最优先**：直接基于已获得的工具结果用文字作答（写 <summary>…</summary> 加正文），不要再调用工具；
                2) 若该工具确实拿不到数据，换其它工具 / URL / 参数（如 rag_tools(op=\"search\")、ask_user），或用 update_plan 把该步标 blocked 并写明原因后继续；
                3) 若还没有计划，先 update_plan(create) 登记再推进。
                不要再次发出与上一轮相同的调用。
                """
            }

            let outcomes = toOutcomes(toolCalls, rawResults)
            let toolResults = zip(toolCalls, outcomes).map { tc, outcome in
                var rec = ToolResultRecord(id: tc.id, name: tc.name, content: outcome.data)
                // RAG 结果 bulky：进历史 / KV 前缀前按 `ragHistoryChars` 单独收紧，
                // 只保留最相关头部，避免几千字符的大返回把历史撑大、拖慢后续每轮生成。
                // 不影响 `outcomes` —— nextPrompt / shouldExit 已基于完整数据算过。
                let trimmed: String
                if rec.name == "rag_tools" {
                    trimmed = KernelLoop.trimRagHistory(rec.name, rec.content,
                                                       limit: ragHistoryChars)
                } else {
                    // web_search/web_scan 大返回（>2000 字）先提核心句压缩，
                    // 避免大文本进上下文拖慢生成或触发暴力裁剪丢信息。
                    trimmed = KernelLoop.trimWebHistory(rec.name, rec.content)
                }
                if trimmed.count < rec.content.count {
                    let tag = rec.name == "rag_tools" ? "[rag-trim]" : "[web-trim]"
                    log("\(tag) \(rec.name) 原始 \(rec.content.count) → \(trimmed.count) 字符")
                }
                rec.content = trimmed
                return rec
            }
            let summary = resp.summary.isEmpty
                ? l2Summary(resp.content, toolNames: toolCalls.map(\.name))
                : resp.summary
            session.finishTurn(rec, assistantText: resp.content,
                               toolCalls: toolCalls, toolResults: toolResults,
                               summary: summary, rawContent: resp.rawContent)
            resetRetryGuards()

            if outcomes.contains(where: { $0.shouldExit }) {
                finalText = resp.content
                exitReason = .interrupt
                break
            }

            let nextPrompts = Set(outcomes.compactMap { $0.nextPrompt }.filter { !$0.isEmpty })
            if nextPrompts.isEmpty {
                finalText = resp.content
                exitReason = .done
                break
            }

            var parts = nextPrompts.sorted()
            // 失败恢复提示：计划步骤出错时不许停下等用户说「继续」。
            parts.append(contentsOf: recoveryHints(toolCalls, rawResults,
                                                   batch: batch,
                                                   stepFailsBefore: stepFailsBefore))
            // 补建提示：已开始执行但 plan 仍空 → 提醒补建计划（兜底）。
            if didWork && session.getPlan() == nil && planNudged < planNudgeMax {
                planNudged += 1
                parts.append("""
                [System] 你已经开始执行但 Session 中还没有执行计划（plan 为空）——后续多步任务将没有进度可跟踪。请立即用 update_plan(op='create', goal=..., steps=[...]) 把目标和已完成/待做步骤登记为计划；之后每完成一步用 update_plan(op='patch', step_id=..., status='done', note='结论') 更新进度。计划每轮回注，是跟踪推进的依据。
                """)
            }
            // 重复调用/计划评审提醒必须在 pending 赋值前并入，否则会被整体覆盖。
            if let repeatWarning { parts.append(repeatWarning) }
            if let planReviewWarning { parts.append(planReviewWarning) }

            // 决策门（生成前）：检索类工具刚返回，先判定证据是否足够，再让模型
            // 写最终答案。避免「信息不足却先花 400 token 写草稿、随后被打回」
            // 的浪费；放行后收尾门不再重复判定（decisionPrechecked）。
            let retrievalOnly = !toolCalls.isEmpty && toolCalls.allSatisfy {
                ["rag_tools", "web_scan", "web_search"].contains($0.name)
            }
            // status（清单）与 outline（大纲）都是知识库的结构性结果：
            // 「知识库里有什么」「这本书讲什么」这类问句，拿到清单/大纲即可作答。
            // 实测 0.8B 判定模型仍会给低分（0.326/0.266 < 0.40）并打回，模型随即
            // 被推去反复检索、越查越偏。这里直接放行，并标记 prechecked 让收尾门
            // 也不再重复判定。
            let kbStructureAnswered = toolCalls.contains {
                guard $0.name == "rag_tools" else { return false }
                let op = $0.arguments["op"]?.stringValue
                return op == "status" || op == "outline"
            }
            // 向量检索的最高相似度 > 0.5，或任一模式判定确认存在相关切片：
            // 检索本身已找到匹配材料，视为达成完成度，不再让 0.8B 判定模型
            // 对整段证据做二次猜测。该标记在结果返回时即写入
            // （ragHighConfidence），收尾门同样受其约束。
            if toolsEnabled, hasToolSuccess, !decisionPrechecked, reviewRounds == 0 {
                if ragHighConfidence {
                    decisionPrechecked = true
                    log("[kernel] 决策门（生成前）：rag_search 已确认检索到匹配材料，直接放行")
                } else if retrievalOnly {
                    if kbStructureAnswered {
                        decisionPrechecked = true
                        log("[kernel] 决策门（生成前）：rag_tools(op=status/outline) 清单/大纲结果直接放行")
                    } else if decisionFinishBlocks < config.kernelDecisionGateMaxFinishBlocks,
                              let judge = decisionJudge {
                        let sufficient = await judge.informationSufficient(
                            task: userInput,
                            evidence: decisionEvidence())
                        if sufficient == false {
                            decisionFinishBlocks += 1
                            parts.append("""
                            [System] 决策模型判定当前收集的信息可能不足以完整回答任务（第 \(decisionFinishBlocks)/\(config.kernelDecisionGateMaxFinishBlocks) 次拦截）。
                            本轮不要直接写最终答案；请继续调用工具补齐缺失信息（换关键词/换来源、检查遗漏的必要数据点），不要重复上一轮相同的检索。
                            若确认无法获取，请在答复中明确说明缺失的部分与原因后再收尾，不要编造。
                            """)
                            log("[kernel] 决策门（生成前）：拦截 "
                                + "\(decisionFinishBlocks)/\(config.kernelDecisionGateMaxFinishBlocks)（信息不足）")
                        } else if sufficient == true {
                            decisionPrechecked = true
                            log("[kernel] 决策门（生成前）：信息足够，放行生成")
                        } else {
                            // nil = 判定不可用（模型缺失/超时）：静默放行，本轮任务
                            // 不再重复判定，避免每次收尾都白等一次超时。
                            decisionPrechecked = true
                            log("[kernel] 决策门（生成前）：判定不可用，跳过后续判定")
                        }
                    }
                }
            }
            pending = parts.joined(separator: "\n")
            _ = transition("continue")
        }

        if exitReason == .maxTurns && finalText.isEmpty {
            // 旧实现只回一句「任务可能未完成」：实测工具已执行 4 次、用户
            // 等了 6 轮，却拿不到任何结果。把已取得的工具结果一并汇总，
            // 让这次等待至少交付点东西；同时把 doing 步骤标 blocked，
            // 计划面板要能看出「停在哪」。
            markPlanBlocked(reason: "轮次上限（\(maxTurns)）耗尽，等待用户继续",
                            includePending: false)
            let highlights = summarizeToolResults()
            finalText = "（已达到本轮最大轮次上限 \(maxTurns)，本轮自动停止。）"
            finalText += highlights.isEmpty ? "\n本轮尚未取得任何工具结果。" : highlights
        }
        // 提醒次数用尽后模型仍要收尾：尊重决定，但必须把「其实没做完」告诉用户。
        // 例外：最后一轮是无工具的纯文字回答——文字步骤（说明 / 总结 / 边界）本就
        // 靠文字收尾，不应被误报为「步骤未完成」。
        if exitReason == .done && finishNudge > 0 && !finalTurnNoTool {
            let left = session.planOpenSteps()
            if !left.isEmpty {
                let todo = left.prefix(8).map { "\($0.id).\($0.title)" }.joined(separator: "; ")
                finalText += "\n\n---\n⚠ 计划中仍有 \(left.count) 个未完成步骤：\(todo)"
                    + "（内核已多次要求继续执行仍未完成。若某步其实已做完，用 update_plan 补标 done；若确实不可行，标 blocked 并写明原因。）"
            }
        }

        // 纯文字收尾且计划仍有未完成步骤：自动标记完成并落盘，避免下次用户输入时
        // 内核带着未完成计划从上一步重新执行（模型用文字总结代替 update_plan 收尾
        // 时极易漏标 done）。仅在本轮确实执行过工具（didWork）时才兜底，避免
        // 「从未干活就标完成」把真实计划清掉。
        if exitReason == .done && finalTurnNoTool && didWork {
            if let plan = session.getPlan(),
               plan.steps.contains(where: { $0.status == "pending" || $0.status == "doing" }) {
                var completed = plan
                var autoDone = 0
                for i in completed.steps.indices {
                    if completed.steps[i].status == "pending"
                        || completed.steps[i].status == "doing" {
                        completed.steps[i].status = "done"
                        if completed.steps[i].note.isEmpty {
                            completed.steps[i].note = "自动标记完成（无工具收尾）"
                        }
                        autoDone += 1
                    }
                }
                session.setPlan(completed)
                do {
                    try PlanStore.save(completed, config: config)
                } catch {
                    AgentLog.warning("[kernel] 计划落盘失败: \(error.localizedDescription)")
                }
                callbacks.onPlanUpdated(completed)
                if autoDone > 0 {
                    finalText += "\n\n---\n（计划已自动标记 \(autoDone) 个步骤为完成）"
                }
            }
        }

        // 计划被模型用 update_plan 标完成，但整段没有任何实际工具调用：
        // 不能把虚假 done 留在计划面板里，也不能让最终答复看起来已验证完成。
        if exitReason == .done && !didWork, isMultistep(userInput),
           let plan = session.getPlan() {
            var adjusted = plan
            var downgraded = 0
            for index in adjusted.steps.indices
            where adjusted.steps[index].status == "done" {
                adjusted.steps[index].status = "skipped"
                if adjusted.steps[index].note.isEmpty {
                    adjusted.steps[index].note = "无实际工具验证（文字收尾），已降级为 skipped"
                }
                downgraded += 1
            }
            if downgraded > 0 {
                session.setPlan(adjusted)
                do {
                    try PlanStore.save(adjusted, config: config)
                } catch {
                    AgentLog.warning("[kernel] 计划降级落盘失败: \(error.localizedDescription)")
                }
                callbacks.onPlanUpdated(adjusted)
                finalText += "\n\n---\n⚠ 计划中 \(downgraded) 个“完成”步骤没有实际工具验证，"
                    + "已降级为 skipped，不能视为已完成。"
                log("[kernel] 零实际工具调用，计划中 \(downgraded) 个 done 降级为 skipped")
            }
        }

        if emitFinished {
            hooks.finished(session, text: finalText)
        }
        return KernelResult(text: finalText, reason: exitReason,
                            turns: turnsUsed, distilled: distillRequested)
    }

    // MARK: 转移判定

    @discardableResult
    private func transition(_ event: String) -> KernelState? {
        if case .state(let next) = nextState(state, event) {
            state = next
            return next
        }
        return nil
    }

    private func needsConsolidate() -> Bool {
        let turn = session.currentTurn
        // needsAutoShrink = 已用超过 90% 最大上下文（独立于 21% 输出预留）：
        // 触发唯一一次「摘要回注式」自动收缩。totalOver 只是预算诊断量，
        // 不再触发折叠；raw 软配额是建议值，用 rawOver 判紧急会变成
        // 「每轮折 1 轮 + 每轮丢 KV + 每轮全量 prefill」。
        if budget.needsAutoShrink { return true }
        if turn > 0 && turn % consolidateEvery == 0 && turn != lastConsolidatedTurn {
            return true
        }
        return false
    }

    // MARK: THINK：一次 LLM 调用

    /// 给 LLM 流加取消看门狗。
    ///
    /// HTTP 流在服务端静默时会长时间没有新事件，而内核只在收到事件时才能
    /// 轮询取消标志。这里用独立任务定期检查：一旦用户取消就结束消费侧流，
    /// 消费循环退出 → onTermination 取消 producer → 底层任务被取消。
    private func cancellableStream(
        _ source: AsyncThrowingStream<LLMStreamEvent, Error>
    ) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        AsyncThrowingStream { outer in
            let producer = Task {
                do {
                    for try await event in source {
                        outer.yield(event)
                    }
                    outer.finish()
                } catch {
                    outer.finish(throwing: error)
                }
            }
            let watchdog = Task { [weak self] in
                while !Task.isCancelled {
                    guard let self else { return }
                    if self.cancelled() {
                        outer.finish()
                        return
                    }
                    do {
                        try await Task.sleep(nanoseconds: 200_000_000)
                    } catch {
                        return
                    }
                }
            }
            outer.onTermination = { _ in
                producer.cancel()
                watchdog.cancel()
            }
        }
    }

    private func ask(_ messages: [ChatMessage],
                     toolsEnabled: Bool = true,
                     options: LLMRequestOptions = .default) async throws -> LLMResponse? {
        var contentBuf: [String] = []
        var reasoningBuf: [String] = []
        var calls: [ToolCall] = []
        var usage: [String: Int] = [:]
        var rawBuf = ""
        var ttfb: Date?
        var lastEmit: Date?
        var liveEst = 0
        var attempt = 0
        let t0 = Date()
        var cancelled = false
        var seenDone = false
        let toolSchemas = toolsEnabled ? schemas() : []

        while true {
            contentBuf = []
            reasoningBuf = []
            calls = []
            usage = [:]
            rawBuf = ""
            ttfb = nil
            lastEmit = nil
            liveEst = 0
            seenDone = false
            truncated = false
            var gotAny = false
            do {
                // 只在有实际覆盖（关闭 thinking / isolated）时走 options 通道：
                // 默认调用保持原 3 参路径，后端重载选择与 KV 复用行为不变。
                let baseStream: AsyncThrowingStream<LLMStreamEvent, Error>
                if options != .default, let streamWithOptions = runtime.streamWithOptions {
                    baseStream = streamWithOptions(messages, toolSchemas,
                                                   config.temperature, options)
                } else {
                    baseStream = runtime.stream(messages, toolSchemas, config.temperature)
                }
                let stream = cancellableStream(baseStream)
                // 模型若把注入的锚点提示（### [WORKING MEMORY] / ### [PLAN]）原样
                // 复述，从标记出现起把后续内容转进思考区，不让它进对话气泡。
                var inAnchorEcho = false
                var anchorTail = ""
                for try await event in stream {
                    gotAny = true
                    if self.cancelled() {
                        cancelled = true
                        break
                    }
                    let now = Date()
                    if ttfb == nil {
                        ttfb = now
                        // 首字节延迟是「每句回复很慢」最直接的量化指标：
                        // 包含发送前的后台等待之外的全部 LLM 侧耗时。
                        AgentLog.info("[ttfb] 首字节 \(Int(now.timeIntervalSince(t0) * 1000))ms "
                            + "kind=\(event.kind) attempt=\(attempt) "
                            + "messages=\(messages.count) tools=\(toolSchemas.count)")
                    }
                    switch event.kind {
                    case .content:
                        // contentBuf 累积完整响应文本（判定用），回调则按锚点复述分流：
                        // 模型复述注入的 ### [WORKING MEMORY] / ### [PLAN] 时，从标记起
                        // 转进思考区，不让内部协议进对话气泡。
                        contentBuf.append(event.content)
                        let probe = anchorTail + event.content
                        if inAnchorEcho {
                            callbacks.onReasoning(event.content)
                        } else {
                            let markers = ["### [WORKING MEMORY]", "###[WORKING MEMORY]",
                                           "### [PLAN]", "###[PLAN]"]
                            if let hit = markers.first(where: { probe.contains($0) }),
                               let idx = probe.range(of: hit)?.lowerBound {
                                inAnchorEcho = true
                                anchorTail = ""
                                let before = String(probe[..<idx])
                                if !before.isEmpty { callbacks.onContent(before) }
                                let echo = String(probe[idx...])
                                if !echo.isEmpty { callbacks.onReasoning(echo) }
                            } else {
                                // 保留可能构成标记前缀的尾部（如 "### [WORK..."），
                                // 避免标记被拆到两个 chunk 时漏判。
                                let keep = Self.anchorPrefixSuffixLength(probe)
                                let emit = String(probe.dropLast(keep))
                                anchorTail = String(probe.suffix(keep))
                                if !emit.isEmpty { callbacks.onContent(emit) }
                            }
                        }
                        liveEst += TokenEstimator.estimate(event.content)
                    case .reasoning:
                        reasoningBuf.append(event.content)
                        callbacks.onReasoning(event.content)
                        liveEst += TokenEstimator.estimate(event.content)
                    case .done:
                        seenDone = true
                        calls = event.toolCalls
                        usage = event.usage
                        rawBuf = event.rawContent
                        truncated = event.truncated
                        continue
                    }
                    if lastEmit == nil {
                        lastEmit = now
                    } else if liveEst > 0, let last = lastEmit,
                              now.timeIntervalSince(last) >= 0.3 {
                        let elapsed = max(now.timeIntervalSince(ttfb ?? t0), 1e-6)
                        callbacks.onTokenSpeed(totalTokens + liveEst,
                                               Double(liveEst) / elapsed)
                        lastEmit = now
                    }
                }
                // 流结束：暂存的标记前缀若未构成完整锚点，还原为正文显示。
                if !anchorTail.isEmpty && !inAnchorEcho {
                    callbacks.onContent(anchorTail)
                }
                break
            } catch {
                // 只有「一个 chunk 都没收到」才重试：已吐过内容的流重试会拼接两段输出。
                if self.cancelled() || Task.isCancelled {
                    return nil
                }
                if gotAny || attempt >= maxLLMRetries { throw error }
                let msg = describe(error)
                if isContextError(msg) || KernelRegex.search("HTTP 4\\d\\d", msg) {
                    throw error
                }
                attempt += 1
                let delay = llmRetryBaseDelay * pow(2.0, Double(attempt - 1))
                log("[kernel] LLM 首字节前失败，\(String(format: "%.1f", delay))s 后重试 (\(attempt)): \(msg)")
                if self.cancelled() { return nil }
                // 分段休眠并轮询取消，避免用户点了取消还要干等。
                var slept = 0.0
                while slept < delay {
                    if self.cancelled() || Task.isCancelled { return nil }
                    do {
                        try await Task.sleep(
                            nanoseconds: UInt64(min(0.1, delay - slept) * 1_000_000_000)
                        )
                    } catch {
                        return nil
                    }
                    slept += 0.1
                }
            }
        }

        // 看门狗可能已因取消而结束流（此时循环正常退出），统一在这里判定。
        if cancelled || self.cancelled() || Task.isCancelled { return nil }
        // 模型可能跳过 <think> 起始标签、直接输出思考再以 </think> 收尾：
        // 这段内容不能进正文，否则会写入 transcript 并污染下一轮上下文。
        let stray = splitStrayThinking(contentBuf.joined())
        let text = stray.content
        let reasoningText = reasoningBuf.joined() + stray.reasoning
        if !seenDone && (!text.isEmpty || !reasoningText.isEmpty) {
            log("[agent] warning: LLM stream ended without 'done' event")
        }

        if !usage.isEmpty {
            callbacks.onUsage(usage)
            let apiReasoning = usage["reasoning_tokens"] ?? 0
            let estReasoning = TokenEstimator.estimate(reasoningText)
            let apiTotal = usage["total_tokens"] ?? 0
            // API 的 total 可信（含 reasoning）时才直接用；否则补上估算值。
            if apiReasoning > 0 || reasoningText.isEmpty {
                totalTokens += apiTotal
            } else {
                totalTokens += apiTotal + estReasoning
            }
        }
        let completion = usage["completion_tokens"] ?? 0
        let generated = max(Date().timeIntervalSince(ttfb ?? t0), 1e-6)
        callbacks.onTokenSpeed(totalTokens,
                               completion > 0 ? Double(completion) / generated : 0.0)

        let toolCalls = calls.map(toolCallWithID)
        return LLMResponse(
            content: text,
            reasoning: reasoningText,
            toolCalls: toolCalls,
            usage: usage,
            summary: text.isEmpty ? "" : l2Summary(text, toolNames: toolCalls.map(\.name)),
            rawContent: rawBuf,
            truncated: truncated
        )
    }

    /// 部分本地模型不发 id；没有 id 就无法把 tool 结果关联回去。
    /// 前缀必须区别于模型常见的 "call_1"，否则同一轮内可能撞 id。
    private func toolCallWithID(_ raw: ToolCall) -> ToolCall {
        guard raw.id.isEmpty else { return raw }
        toolCallCounter += 1
        var copy = raw
        copy.id = "kcall_\(toolCallCounter)"
        return copy
    }

    // MARK: ACT：RAG 工具结果进历史前的收紧

    /// RAG 工具结果进历史 / KV 前缀前的纯函数收紧：保留最相关头部、丢弃尾部，
    /// 避免 bulky 大返回把历史撑大、拖慢后续每轮生成。仅 `rag_tools`（合并后的
    /// 知识库工具）结果收紧；`limit <= 0`、或内容未超限时原样返回。
    static func trimRagHistory(_ name: String, _ content: String, limit: Int) -> String {
        guard limit > 0, name == "rag_tools", content.count > limit else {
            return content
        }
        // RAG 结果回灌给模型前，不再 prefix 截断（只覆盖前面几个切片），
        // 而是对「每个切片」提取核心观点句后全部保留：覆盖所有返回切片，
        // 且无论原文多长，核心句集合长度恒定，不会让模型输出变慢。
        var sections: [(header: String, text: String)] = []
        var currentHeader = ""
        var currentLines: [String] = []
        func flush() {
            if currentHeader.isEmpty && currentLines.isEmpty { return }
            sections.append((currentHeader, currentLines.joined(separator: "\n")))
            currentHeader = ""
            currentLines = []
        }
        for line in content.split(separator: "\n", omittingEmptySubsequences: false) {
            let s = String(line)
            if s.hasPrefix("### [") {
                flush()
                currentHeader = s
            } else {
                currentLines.append(s)
            }
        }
        flush()

        // 无句末标点的超长单句（如 5000 个相同字符）：核心句提取无从压缩，
        // 退回原 prefix 截断语义，避免核心句=原文让 limit 保护失效。
        if CoreOpinionExtractor.splitSentences(content).count <= 1 {
            return String(content.prefix(limit))
                + "\n…[RAG 结果已截断至 \(limit) 字符，原始 \(content.count) 字符]"
        }

        // 自适应降级：先每路 top10；仍超预算就把句子数减半（每路 top5 → top3）再试。
        // 用「含 footer 的完整文本」判断，避免估算偏差导致该降级时没降级。
        var usedTopN = CoreOpinionExtractor.perMethodTopN
        var compressed = ""
        var out: [String] = []
        func footerText(_ topN: Int) -> String {
            "\n[RAG 全部切片已提取核心句（共 \(sections.count) 段，每路 top\(topN)），"
                + "原始 \(content.count) 字符]"
        }
        while true {
            out = []
            if sections.isEmpty {
                out.append(CoreOpinionExtractor.compressed(content, topN: usedTopN))
            } else {
                for section in sections {
                    let core = CoreOpinionExtractor.compressed(section.text, topN: usedTopN)
                    out.append(section.header)
                    out.append(core.isEmpty ? section.text : core)
                }
            }
            compressed = out.joined(separator: "\n")
            if compressed.count + footerText(usedTopN).count <= limit || usedTopN <= 3 { break }
            usedTopN = max(3, usedTopN / 2)  // 10 → 5 → 3 下限
        }
        // 含 footer 后仍超预算 → 走下面的按切片均分兜底。
        guard compressed.count + footerText(usedTopN).count <= limit else {
            // 极端情况：核心句仍超限 → 按切片均分预算，保证每个切片的标题和
            // 至少一条核心句都保留（不能因长度预算又退化成"只覆盖前几个切片"）。
            let marker = "\n[RAG 全部切片已提取核心句"
            let body = compressed.components(separatedBy: marker).first ?? compressed
            let blocks = body.components(separatedBy: "### [").dropFirst().map { "### [" + $0 }
            let footerBudget = 120
            // 标题约 30~40 字；预算紧时每片正文可低至 1 句（标题必须保留）。
            let perBlock = max(0, (limit - footerBudget) / max(1, blocks.count) - 45)
            var kept: [String] = []
            for block in blocks {
                let lines = block.split(separator: "\n").map(String.init)
                guard let header = lines.first else { continue }
                var text = ""
                for line in lines.dropFirst() {
                    if text.count + line.count + 1 <= perBlock {
                        text += (text.isEmpty ? "" : "\n") + line
                    } else { break }
                }
                kept.append(header + (text.isEmpty ? "\n（内容超预算，详见引用）" : "\n" + text))
            }
            return kept.joined(separator: "\n")
                + "\n[RAG 全部 \(blocks.count) 切片已提取核心句（每路 top\(usedTopN)，按预算每片限 \(perBlock) 字符）；原始 \(content.count) 字符]"
        }
        return compressed
            + "\n[RAG 全部切片已提取核心句（共 \(sections.count) 段，每路 top\(usedTopN)），"
            + "原始 \(content.count) 字符]"
    }

    /// 网络工具结果进上下文前的核心句压缩：>2000 字才触发。
    /// - web_search：按「N. 标题 / URL / 摘要」分段，标题与 URL 原样保留（模型要用 URL
    ///   调 web_scan），只对摘要提取核心句。
    /// - web_scan：网页正文整段提取核心句（标题/元信息行原样保留）。
    static func trimWebHistory(_ name: String, _ content: String,
                               threshold: Int = 2_000) -> String {
        // 用户约定：超过 threshold（默认 2000 字）就压缩，与预算 limit 无关。
        guard (name == "web_search" || name == "web_scan"),
              content.count > threshold else { return content }

        if name == "web_search" {
            var out: [String] = []
            var current: [String] = []
            func flush() {
                guard !current.isEmpty else { return }
                // 首行「N. 标题」、URL 行原样；其余（摘要等）合并提取核心句。
                let head = current.prefix(2)
                let rest = current.dropFirst(2).joined(separator: " ")
                var body = rest
                if rest.count > threshold / 4 {
                    let core = CoreOpinionExtractor.compressed(rest, topN: 5)
                    if !core.isEmpty { body = core }
                }
                out.append((head + [body]).joined(separator: "\n"))
                current = []
            }
            for line in content.split(separator: "\n", omittingEmptySubsequences: false) {
                let s = String(line)
                let numbered = s.range(of: #"^\d+\. "#, options: .regularExpression) != nil
                if numbered { flush() }
                current.append(s)
            }
            flush()
            let result = out.joined(separator: "\n")
            return result.count < content.count
                ? result + "\n[web_search 摘要已提取核心句，原始 \(content.count) 字符]"
                : content
        }

        // web_scan：保留开头连续的非空元信息行（标题/URL/状态），遇空行后是正文。
        let lines = content.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        let metaCount = min(6, lines.firstIndex(where: { $0.isEmpty }) ?? lines.count)
        let meta = Array(lines.prefix(metaCount))
        let body = Array(lines.dropFirst(metaCount)).joined(separator: "\n")
        guard body.count > threshold else { return content }
        let core = CoreOpinionExtractor.compressed(body, topN: CoreOpinionExtractor.perMethodTopN)
        guard !core.isEmpty else { return content }
        let result = (meta + [core]).joined(separator: "\n")
        return result.count < content.count
            ? result + "\n[web_scan 正文已提取核心句，原始 \(content.count) 字符]"
            : content
    }

    // MARK: ACT：工具结果 → StepOutcome（L1 裁剪在这里发生）

    private func toOutcomes(_ toolCalls: [ToolCall],
                            _ rawResults: [ToolResult]) -> [StepOutcome] {
        let n = max(1, toolCalls.count)
        // B：按本折叠段内剩余轮数均摊硬余量，预算前移到工具结果产出时，
        // 让段内累计 raw 不越顶 → 不滑窗 / 不提前折叠 → KV 前缀只追加。
        let turn = max(1, session.currentTurn)
        let period = max(1, consolidateEvery)
        let segmentLeft = max(1, period - ((turn - 1) % period))
        let maxChars = budget.remainingForTool(n, maxChars: maxToolChars,
                                               reserveTurns: segmentLeft)
        var out: [StepOutcome] = []
        for (tc, res) in zip(toolCalls, rawResults) {
            // 只有门控**成功**才允许触发后续经验蒸馏。门控失败（如当前轮数 <5）
            // 返回 .fail，若仍置位就等于绕过「短任务不许碰长期记忆」的约束。
            if tc.name == "start_long_term_update", res.success {
                distillRequested = true
            }
            // L0：先兜住病态长度；L1：再按 RAW 区余额裁到真正要注入的长度。
            let captured = l0Capture(res.toMessage(), hardCap: l0HardCap)
            out.append(Self.outcome(data: l1Trim(captured, maxChars: maxChars),
                                    result: res))
        }
        return out
    }

    /// 工具结果 → 自动机下一步。`"\n"` 是「本轮继续」的哨兵；工具若显式要求
    /// 中断（`ToolResult.shouldExit`）必须透传，否则 `outcomes.contains {
    /// $0.shouldExit }` 恒为 false，中断信号被吞掉、自动机会继续空转。
    static func outcome(data: String, result: ToolResult) -> StepOutcome {
        StepOutcome(data: data,
                    nextPrompt: result.shouldExit ? nil : "\n",
                    shouldExit: result.shouldExit)
    }

    // MARK: ACT：失败记账与自动恢复提示

    private struct FailureBatch {
        var failed: [(name: String, error: String)] = []
        var repeats: [(name: String, error: String)] = []
        var planMutated = false
    }

    private func resultOK(_ res: ToolResult) -> (Bool, String) {
        if res.success { return (true, "") }
        let err = res.error.isEmpty ? "未知错误" : String(res.error.prefix(300))
        return (false, err)
    }

    private func callSig(_ tc: ToolCall) -> String {
        let args = JSONValue.object(tc.arguments).jsonString()
        return "\(tc.name)::\(String(args.prefix(500)))"
    }

    private func trackFailures(_ toolCalls: [ToolCall],
                               _ rawResults: [ToolResult]) -> FailureBatch {
        var batch = FailureBatch()
        for (tc, res) in zip(toolCalls, rawResults) {
            let (ok, err) = resultOK(res)
            if tc.name == "update_plan" {
                if ok { batch.planMutated = true }
                continue
            }
            if ok { continue }
            batch.failed.append((tc.name, err))
            failStreak += 1
            failedTotal += 1
            lastFail = ToolFailure(name: tc.name, error: err)
            stepFails += 1
            let sig = callSig(tc)
            if failSigs.contains(sig) {
                batch.repeats.append((tc.name, err))
            }
            failSigs.append(sig)
            if failSigs.count > maxFailSigs {
                failSigs.removeFirst(failSigs.count - maxFailSigs)
            }
        }
        if batch.failed.isEmpty {
            failStreak = 0
        }
        if batch.planMutated {
            stepFails = 0
        }
        return batch
    }

    private func recoveryHints(_ toolCalls: [ToolCall], _ rawResults: [ToolResult],
                               batch: FailureBatch,
                               stepFailsBefore: Int) -> [String] {
        var hints: [String] = []
        if let repeatFailure = batch.repeats.first {
            hints.append("""
            [System] 检测到与之前完全相同的失败调用：`\(repeatFailure.name)`（错误：\(String(repeatFailure.error.prefix(160)))）。原样重试不会改变结果，禁止重复。请换参数/换工具/换思路；或用 code_run 写诊断脚本定位根因；确认此路不通后，把对应计划步骤标记 blocked 并继续后续步骤。
            """)
        } else if failStreak >= 2 {
            let failure = lastFail ?? ToolFailure(name: "", error: "")
            let tried = triedApproaches()
            hints.append("""
            [System] 已连续失败 \(failStreak) 次（最近：\(failure.name) → \(String(failure.error.prefix(200)))。已试过且失败的方案：\(tried.isEmpty ? "（无记录）" : tried)。
            不要停下来问用户是否继续——请依据错误信息自动修复：先读完整错误定位原因，再换一种**没试过**的方法重试（不同工具/不同参数/先用 code_run 写最小诊断脚本验证假设）。同一思路只允许试一次。
            """)
        } else if let failure = batch.failed.first {
            // 首次失败也必须是「换方案」而不是「原样重发 / 停下报告」。
            hints.append("""
            [System] 工具 `\(failure.name)` 调用失败：\(String(failure.error.prefix(200)))。请**先完整读完错误**，判定是参数错误 / 路径不存在 / 权限 / 依赖缺失 / 语法错误 / 网络不可达 中的哪一类，然后换一种方法达成同一步骤的目标（换工具、换参数，或先用 code_run 写最小诊断脚本验证假设）。不要原样重发同一个调用，不要停下来向用户报告失败——失败是给你定位用的信号，不是停车的理由。
            """)
        }
        if prematureBlock(toolCalls, rawResults, stepFailsBefore: stepFailsBefore) {
            hints.append("""
            [System] 步骤刚被标记 blocked，但本步骤至今只失败了 \(stepFailsBefore) 次——尝试不够。请先用 update_plan 把它恢复 doing，再依据上面的错误信息换**没试过**的方案重试（至少 3 种不同方法）；只有多种方法都确认不可行后，才标记 blocked 并写明原因，然后继续后续步骤。不要向用户确认「是否继续」。
            """)
        }
        return hints
    }

    private func prematureBlock(_ toolCalls: [ToolCall], _ rawResults: [ToolResult],
                                stepFailsBefore: Int) -> Bool {
        if blockNudge >= blockNudgeMax { return false }
        if stepFailsBefore >= minAttemptsBeforeBlock { return false }
        for (tc, res) in zip(toolCalls, rawResults) {
            if tc.name != "update_plan" { continue }
            if (tc.arguments["status"]?.stringValue ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased() != "blocked" { continue }
            if resultOK(res).0 {
                blockNudge += 1
                return true
            }
        }
        return false
    }

    private func triedApproaches(limit: Int = 5) -> String {
        var seen: [String] = []
        for sig in failSigs {
            let parts = sig.components(separatedBy: "::")
            let name = parts.first ?? ""
            if name.isEmpty { continue }
            let args = String(parts.dropFirst().joined(separator: "::").prefix(60))
                .replacingOccurrences(of: "\n", with: " ")
            let item = args.isEmpty ? name : "\(name)(\(args))"
            if !seen.contains(item) { seen.append(item) }
        }
        return seen.suffix(limit).joined(separator: "； ")
    }

    // MARK: 无工具调用时的判定

    private func isSimpleInput(_ userInput: String) -> Bool {
        let text = userInput.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty { return true }
        if text.count > 60 { return false }
        return !isMultistep(text)
    }

    /// 用户是否在问「知识库里有什么」：这类问题应走 `rag_tools(op="status")`。
    static func looksLikeKnowledgeBaseQuestion(_ userInput: String) -> Bool {
        let text = userInput.lowercased()
        let markers = ["知识库", "knowledge base", "文档库", "向量库", "知识图谱", "rag_tools"]
        return markers.contains { text.contains($0) }
    }

    /// 问「某本书/文档讲什么」：问句不含「知识库」字样，但本质是从已索引文档
    /// 提取内容，不属于需要联网的实时信息。这类问句禁止在 RAG 有结果时转 web_search。
    static func looksLikeDocumentContentQuestion(_ text: String) -> Bool {
        let lower = text.lowercased()
        let contentMarkers = ["讲什么", "主要内容", "主要讲", "概述", "简介", "写了什么",
                              "说的是什么", "讲的内容", "什么内容", "总结", "介绍"]
        guard contentMarkers.contains(where: { lower.contains($0) }) else { return false }
        // 带明确联网词时仍允许联网（用户主动要实时/外部信息）。
        let webMarkers = ["网址", "链接", "最新", "实时", "今天", "网页", "http"]
        return !webMarkers.contains(where: { lower.contains($0) })
    }

    /// 本次请求里是否已有成功的 RAG 检索结果（历史 turns 的 toolResults）。
    private func hasSuccessfulRagResult() -> Bool {
        session.turns.contains { turn in
            turn.toolResults.contains { result in
                result.name == "rag_tools"
                    && !result.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    && !result.content.hasPrefix("ERROR")
                    && !result.content.contains("success=false")
            }
        }
    }

    /// 意图空转且已有 RAG 结果时，把「立即调用工具」这类前置步骤标 done，
    /// 避免模型被锚在 ▶ doing 反复说"先取数据"却不作答。
    private func completeStalledRagPlan() {
        guard var plan = session.getPlan() else { return }
        var changed = false
        for i in plan.steps.indices where plan.steps[i].status != "done" {
            let note = plan.steps[i].note
            if note.contains("调用工具") || note.contains("获取真实数据")
                || note.contains("检索") || note.contains("web_scan") {
                plan.steps[i].status = "done"
                changed = true
            }
        }
        if changed { session.setPlan(plan) }
    }

    /// 用户是否在问长期记忆（「你还记得…」「我之前说过…」「我的偏好」）。
    /// 弱工具调用模型会反复说「需要调用 long_memory」却发不出调用；
    /// 意图空转门据此给出可照抄参数，并在耗尽时由内核代查一次。
    static func looksLikeMemoryQuestion(_ userInput: String) -> Bool {
        let text = userInput.lowercased()
        let markers = [
            "长期记忆", "记忆", "记得", "还记得", "之前说", "以前说", "说过",
            "我的偏好", "我喜欢", "我的习惯", "我常用", "我们项目",
            "memory", "remember", "what did i", "my preference",
        ]
        return markers.contains { text.contains($0) }
    }

    /// 用户是否在问技能库（「有没有处理 PDF 的技能」「skill 里有什么」）。
    static func looksLikeSkillQuestion(_ userInput: String) -> Bool {
        let text = userInput.lowercased()
        return text.contains("技能") || text.contains("skill")
    }

    /// 把用户原话转成可安全嵌入示例 JSON 的字符串（转义反斜杠与引号）。
    static func jsonExampleString(_ value: String, limit: Int = 200) -> String {
        String(value.prefix(limit))
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }

    /// 回复是否只是「声明要做什么」的元陈述，而不是任务结果。
    ///
    /// 典型事故（2026-11-08「获取 cnn.com 今日新闻并总结」）：模型把
    /// system 里的 `update_plan` 协议复述成「我需要先创建任务计划……现在
    /// 开始执行第一步」，一次工具都不发；内核却把这段当成正常正文收下，
    /// `plan` 仍为空又不触发自动续跑，用户反复点「继续」只得到同样的
    /// 自我复述。只在「无工具调用 + 未做过实际工作」时调用本判定，
    /// 避免误伤正常的文字总结。
    static func looksLikeIntentOnly(_ content: String) -> Bool {
        let text = stripSummaryTags(content)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, text.count <= 800 else { return false }
        let markers = [
            "我需要先", "我先", "让我", "接下来我", "现在我开始", "现在开始执行",
            "创建任务计划", "创建计划", "建立计划", "建立任务", "重新执行", "同步执行",
            "同步发出", "将任务分解", "制定并执行", "开始执行第一步", "立即执行第一步",
            "紧接着发出", "步骤：", "目标：", "我需要：", "请立即：",
            "执行计划", "可验证的步骤", "创建 2~7", "开始执行计划",
            "获取", "并总结", "抓取", "并总结", "查询", "并总结",
            "我将", "我打算", "计划如下", "同步执行", "重新执行工具",
            // 只说明「需要调用某工具获取」但仍不发 tool_call 的元陈述。
            "需要调用", "工具获取", "调用工具获取",
            // 虚假进度声明：模型没调工具却声称「已拿到 / 本步完成 / 可继续」，
            // 这类空转若不拦，会被当正常正文收下、反复刷屏到轮次耗尽。
            "已拿到", "已获取", "已读取", "本步完成", "证实", "推翻",
            "可继续提取", "可继续下一步", "下一步是", "我将获取", "我将抓取",
        ]
        // 通用兜底：明确点名了工具、又用了「调用/查询/读取…」这类动作词，
        // 但整段没有任何 tool_call —— 无论具体措辞如何都应算意图空转。
        let toolNames = [
            "file_tools", "web_scan", "rag_tools", "skill_tools", "long_memory",
            "memory", "shell_run", "code_run", "update_plan", "ask_user", "mcp",
        ]
        let actionVerbs = [
            "调用", "使用", "执行", "查询", "检索", "读取", "获取", "搜索",
            "扫描", "抓取", "运行",
        ]
        let mentionsTool = toolNames.contains { text.contains($0) }
        let mentionsAction = actionVerbs.contains { text.contains($0) }
        if mentionsTool && mentionsAction { return true }

        // 没有点名工具、但用「我需要先查看/让我检查」这类第一人称意图句
        // 描述下一步，同样不应作为最终答复。
        let firstPersonIntent = [
            "我需要", "我先", "让我", "我将", "接下来我", "待我", "稍后我", "正在",
        ]
        let inspectVerbs = [
            "查看", "检查", "看一下", "先看", "搜索", "检索", "查询", "读取",
            "获取", "确认", "分析", "扫描",
        ]
        if firstPersonIntent.contains(where: { text.contains($0) }),
           inspectVerbs.contains(where: { text.contains($0) }) {
            return true
        }
        return markers.filter { text.contains($0) }.count >= 2
    }

    /// 回复是否已经是一份可交付的结果（而非意图声明 / 追问 / 空话）。
    /// 仅用于「多步任务、plan 为空、尚未动手」时避免 planFinish 门把
    /// 模型第 1 轮就答对的正文作废、逼它重答 3 次（实测 4 倍墙钟）。
    /// 保守判定：必须通过元陈述排除、末尾无问号，且带明确结论标记或
    /// 结构化编号列表。
    static func looksLikeResult(_ content: String) -> Bool {
        let text = stripSummaryTags(content)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard text.count >= 12, !looksLikeIntentOnly(text) else { return false }
        let tail = String(text.suffix(120))
        guard !tail.contains("？"), !tail.contains("?") else { return false }
        let markers = ["已完成", "完成了", "结果是", "答案是", "结论", "共计",
                       "共有", "总共", "分别为", "分别是", "总计", "得到",
                       "查到", "因此", "综上"]
        if markers.contains(where: { text.contains($0) }) { return true }
        return KernelRegex.search("(^|\\n)\\s*[1-9][\\.、\\)]", text)
            && text.contains("。")
    }

    /// 成果核验反思提示：以从业 10 年的领域资深专家身份，严格审视即将交付
    /// 的结果，要求完整、有据、准确、清晰。发现不足时直接输出完善后的完整
    /// 答复（而非只列问题），充分时输出 <review>pass</review> 结束核验。
    static func expertReviewHint(round: Int, max: Int) -> String {
        """
        [System] 成果核验（第 \(round)/\(max) 轮）——请以从业 10 年的领域资深专家身份，严格审视你上一轮给出的最终答复，重点检查：
        1) 完整性：是否覆盖了用户要求的全部要点，有无遗漏；
        2) 真实性：结论是否全部来自工具返回的真实数据，而非猜测或常识脑补；
        3) 准确性：数字、路径、名称、步骤等细节是否准确，有无前后矛盾或夸大；
        4) 表达：结构是否清晰、语言是否精炼，有无冗余或答非所问。
        **直接输出结论，不要进入思考模式、不要长篇分析过程。**若发现任何不足：直接输出完善后的完整最终答复（不要只列问题清单、不要复述「作为专家我认为」这类套话、不要再调用工具）；若已经足够好、无需任何改动：只输出 <review>pass</review>。
        """
    }

    /// 从用户输入里抽出第一个可抓取的 URL（含裸域名），用于意图空转提示
    /// 给出可照抄的调用参数，以及耗尽时内核代发 `web_scan` 兜底。
    static func firstWebURL(in text: String) -> String? {
        let patterns = [
            "https?://[^\\s，。；、\"'）)]+",
            "(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\\.)+[A-Za-z]{2,}"
                + "(?::[0-9]{1,5})?(?:/[^\\s，。；、\"'）)]*)?",
        ]
        // 文件名/扩展名不是网址（「读一下 config.js」「参考 report.pdf」）。
        let nonWebTLDs: Set<String> = [
            "js", "ts", "py", "md", "json", "txt", "pdf", "swift", "cpp",
            "java", "go", "rs", "csv", "xml", "html", "css", "sh", "yml",
            "yaml", "log", "zip", "png", "jpg",
        ]
        for pattern in patterns {
            guard let match = KernelRegex.matches(
                pattern, text, options: [.caseInsensitive]).first,
                  let range = Range(match.range, in: text) else { continue }
            let candidate = String(text[range])
            let hostPart = candidate
                .replacingOccurrences(of: "https://", with: "")
                .replacingOccurrences(of: "http://", with: "")
                .split(separator: "/").first ?? ""
            let tld = String(hostPart.split(separator: ".").last ?? "").lowercased()
            if nonWebTLDs.contains(tld) { continue }
            if let url = WebURLPolicy.normalizedURLString(candidate) {
                return url
            }
        }
        return nil
    }

    /// 从用户输入里抽出第一个本地路径（绝对路径或常见扩展名文件），
    /// 用于意图空转提示给出可照抄的 `file_tools(op="read")` 参数。
    static func firstLocalPath(in text: String) -> String? {
        let extensions: Set<String> = [
            "md", "txt", "pdf", "csv", "tsv", "json", "jsonl", "yaml", "yml",
            "toml", "ini", "conf", "env", "xml", "log", "sqlite", "db",
            "swift", "py", "js", "ts", "html", "css", "sh", "sql",
            "xlsx", "xls", "docx", "pptx", "png", "jpg", "jpeg",
        ]
        let separators = CharacterSet(charactersIn:
            " \n\t\r，。；、,;:\"'“”‘’（）()[]{}<>")
        let cleaned = KernelRegex.replace("https?://[^\\s]+", in: text, with: " ",
                                          options: [.caseInsensitive])
        for raw in cleaned.components(separatedBy: separators) {
            let token = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !token.isEmpty, !token.contains("://"),
                  !token.hasPrefix("//") else { continue }
            if token.hasPrefix("/"), token.count > 1 { return token }
            guard let dot = token.lastIndex(of: ".") else { continue }
            let ext = String(token[token.index(after: dot)...]).lowercased()
            if extensions.contains(ext) { return token }
        }
        return nil
    }

    /// 核验轮是否只输出了元标记（`<review>fix</review>`、`pass`、空串），
    /// 而不是一份可交付的答复。这类输出绝不能覆盖上一轮候选。
    static func isReviewMetaOnly(_ content: String) -> Bool {
        let text = stripSummaryTags(content)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return true }
        let withoutReview = KernelRegex.replace(
            "<review>[\\s\\S]*?</review>", in: text, with: "",
            options: [.caseInsensitive])
        let residual = KernelRegex.replace(
            tagStripPattern, in: withoutReview, with: " ",
            options: [.caseInsensitive])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if residual.isEmpty { return true }
        let lower = residual.lowercased()
            .trimmingCharacters(in: CharacterSet(charactersIn: "。，.!！:："))
        return ["fix", "pass", "revise", "ok", "需要修改", "无需修改"]
            .contains(lower)
    }

    /// 核验轮模型是否判定「通过」（无需再改）。
    static func isReviewPass(_ content: String) -> Bool {
        let text = stripSummaryTags(content)
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        if text.contains("<review>pass</review>") { return true }
        if text.contains("<review>pass") { return true }
        // 极短的纯 pass 声明也视为通过，避免漏判。
        return text.count < 40 && text.contains("pass")
    }

    /// 核验轮输出是否相对候选「没有实质改进」——小模型被要求「完善」时
    /// 常把同一句结论原样堆叠 N 遍（真机实测 2B 把一句失败说明重复 12 次），
    /// 而非真正改进。这类重复不能算完善，应直接收下干净的单次候选。
    static func isNoImprovement(_ candidate: String, _ new: String) -> Bool {
        let c = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
        let n = new.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !c.isEmpty, !n.isEmpty else { return false }
        if c == n { return true }
        // 新内容只是候选的重复堆叠（长度成倍、且整体包含候选）。
        return n.count >= c.count * 2 && n.contains(c)
    }

    private func isMultistep(_ text: String) -> Bool {
        KernelRegex.search(multistepHintPattern, text,
                           options: [.caseInsensitive, .anchorsMatchLines])
            || KernelRegex.search(multistepVerbPattern, text, options: [.caseInsensitive])
    }

    private func looksLikeAskingUser(_ content: String) -> Bool {
        let text = content.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty { return false }
        let tail = String(text.suffix(200))
        if tail.contains("？") || tail.contains("?") { return true }
        return KernelRegex.search(askUserPattern, tail, options: [.caseInsensitive])
    }

    /// 最近几条工具结果的紧凑汇总，用于失败/耗尽时让用户看到已取得的进展。
    /// HTML 开头的结果先转可见文本，避免汇总里全是 `<!DOCTYPE>`/字体清单。
    /// 决策门判定不再使用它（改用 `decisionEvidence()` 的生成原文），
    /// 这里保持短摘要，避免用户侧看到整页工具原文。
    private func summarizeToolResults(limit: Int = 5) -> String {
        func preview(_ content: String) -> String {
            let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.hasPrefix("<") else { return String(trimmed.prefix(600)) }
            let text = ToolSupport.htmlToText(trimmed, maxChars: 6_000)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return String((text.isEmpty ? trimmed : text).prefix(600))
        }
        let highlights = session.turns
            .flatMap { $0.toolResults }
            .filter { !$0.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .suffix(limit)
            .map { "\n• \($0.name)：" + preview($0.content) }
            .joined()
        return highlights.isEmpty ? "" : "\n\n已取得的进展：" + highlights
    }

    /// 决策门材料包：与最终交给生成模型的资料严格同源，不做二次裁剪。
    /// - 工具结果直接取 Session 中即将随历史发出的原文：RAG 的 top-1 切片
    ///   已按 `ragHistoryChars` 在入库时收紧，这里不再截断/去重/限条数，
    ///   否则判定依据会少于生成依据，结论自然不准；
    /// - 会话状态取 `anchorPrompt()`，即下一轮 user 消息里回注给模型的
    ///   WORKING MEMORY / key_info / PLAN / facts 同一份。
    private func decisionEvidence() -> String {
        let records = session.windows().raw
            .flatMap { $0.toolResults }
            .filter { !$0.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        let state = session.anchorPrompt()
            .trimmingCharacters(in: .whitespacesAndNewlines)

        var blocks: [String] = []
        if !state.isEmpty {
            blocks.append("\n\n会话状态（与下一轮回注给模型的同一份）：\n" + state)
        }
        if !records.isEmpty {
            let results = records
                .map { "\n• \($0.name)：\n\($0.content)" }
                .joined()
            blocks.append("\n\n工具结果（与生成时上下文的原文一致）：" + results)
        }
        return blocks.joined()
    }

    /// 生成连续失败/轮次耗尽时把未完成步骤标 blocked 并落盘：让失败可见、
    /// 可恢复，而不是让计划静默卡死，或被「纯文字收尾自动标 done」假装完成。
    /// `includePending=false` 用于「轮次耗尽待续」：只收 doing（当前步停在哪），
    /// pending（尚未开始）留给下一次用户「继续」。
    private func markPlanBlocked(reason: String, includePending: Bool = true) {
        guard var plan = session.getPlan(),
              plan.steps.contains(where: {
                  $0.status == "doing" || (includePending && $0.status == "pending")
              })
        else { return }
        for index in plan.steps.indices {
            let status = plan.steps[index].status
            guard status == "doing" || (includePending && status == "pending") else { continue }
            plan.steps[index].status = "blocked"
            if plan.steps[index].note.isEmpty { plan.steps[index].note = reason }
        }
        session.setPlan(plan)
        do {
            try PlanStore.save(plan, config: config)
        } catch {
            AgentLog.warning("[kernel] 计划落盘失败: \(error.localizedDescription)")
        }
        callbacks.onPlanUpdated(plan)
        log("[kernel] 计划未完成步骤已标记 blocked（\(reason)）")
    }

    /// probe 尾部有多少字符可能是某个锚点标记的前缀（用于跨 chunk 暂存）。
    private static func anchorPrefixSuffixLength(_ text: String) -> Int {
        let markers = ["### [WORKING MEMORY]", "###[WORKING MEMORY]",
                       "### [PLAN]", "###[PLAN]"]
        var best = 0
        for marker in markers {
            let limit = min(marker.count, text.count)
            for length in stride(from: limit, through: 1, by: -1) {
                if marker.hasPrefix(String(text.suffix(length))) {
                    best = max(best, length)
                    break
                }
            }
        }
        return best
    }

    private func noToolEvent(_ resp: LLMResponse) -> String {
        let content = resp.content.trimmingCharacters(in: .whitespacesAndNewlines)
        // 模型把每轮注入的锚点提示（### [WORKING MEMORY] / ### [PLAN] + dynamic
        // context）原样复述成回复——这是空转，不是答复，绝不能当 finalText 收下。
        // 即便剥协议后残留几行 KB 说明，也按意图空转交给 intentStall 门收敛。
        let anchorEchoMarkers = [
            "### [WORKING MEMORY]", "###[WORKING MEMORY]",
            "### [PLAN]", "###[PLAN]",
        ]
        // 正常答复绝不会包含这些内部锚点标记；一旦出现就是复述，直接判空转。
        if anchorEchoMarkers.contains(where: { content.contains($0) }) {
            return "intent"
        }
        // 工具调用被截断：可见正文可能只是半截前言，不能当最终答复收下。
        if resp.truncated {
            emptyRetry += 1
            return emptyRetry <= maxEmptyRetry ? "retry" : "done"
        }
        // 只有 reasoning、没有可见正文时不能当作最终答复：UI 与用户都
        // 收不到任何内容，必须重试；重试耗尽后由上层给出明确的空回复提示。
        if content.isEmpty {
            emptyRetry += 1
            return emptyRetry <= maxEmptyRetry ? "retry" : "done"
        }
        let tail = String(content.suffix(120))
        if truncationMarks.contains(where: { tail.contains($0) }) {
            emptyRetry += 1
            return emptyRetry <= maxEmptyRetry ? "retry" : "done"
        }
        // 已给出正文（哪怕很短，如「红色方块。」）就直接放行：缺 `<summary>`
        // 不是重试理由，为补标签重跑一轮只会制造重复的元陈述。
        // 只有「整段回复剥掉 <summary> 后完全没有正文」才说明用户收不到内容。
        let visible = stripSummaryTags(content)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if visible.isEmpty {
            // 只有 `<summary>…</summary>` / `<think>…</think>` 等元标签：模型
            // 描述了「我要调用 web_scan」却既没发工具调用也没写正文。
            // 这不是「空正文」，按空正文重试会注入「请输出简短正文」并关闭
            // thinking，反而诱导小模型继续复述元陈述（2026-11-09 cnn.com
            // 实测连报 3 次失败后空手结束）。交给意图空转门：强制发出真实
            // 工具调用；URL 任务还会在耗尽时由内核代发 web_scan 兜底。
            if !content.isEmpty { return "intent" }
            emptyRetry += 1
            return emptyRetry <= maxEmptyRetry ? "retry" : "done"
        }
        if looksLikeUnexecutedCode(visible) {
            return "reask"
        }
        return "done"
    }

    /// 恰好一个大代码块 + 结尾 + 正文几乎没有自然语言 → 疑似该执行却没执行。
    private func looksLikeUnexecutedCode(_ content: String) -> Bool {
        let matches = KernelRegex.matches(codeBlockPattern, content)
        if matches.count != 1 { return false }
        guard let match = matches.first,
              let range = Range(match.range, in: content) else { return false }
        let endOffset = content.distance(from: content.startIndex, to: range.upperBound)
        if endOffset < content.count - 2 { return false }
        var residual = content.replacingCharacters(in: range, with: "")
        residual = KernelRegex.replace(tagStripPattern, in: residual, with: " ",
                                       options: [.caseInsensitive])
        residual = KernelRegex.replace("\\s+", in: residual, with: "")
        return residual.count <= 30
    }

    // MARK: 辅助

    private func resetRetryGuards() {
        // 不清 emptyRetry：旧实现每次 ACT 成功就把它归零，一次请求里
        // 「空回复 3 次 → 工具成功 → 又 3 次」可以反复续命。改成整段
        // 共享 3 次空回复预算，由 run()/seedFailures 统一重置或恢复。
        reaskGuard = 0
    }

    private func cancelled() -> Bool {
        // 注入的 shouldCancel 是主通道；同时兼容直接把回调交给内核的用法。
        runtime.shouldCancel() || callbacks.isCancelled()
    }

    private func log(_ message: String) {
        runtime.log(message)
    }

    private func describe(_ error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? String(describing: error)
    }

    private func isContextError(_ message: String) -> Bool {
        let lower = message.lowercased()
        return ["context_length", "context length", "context window",
                "maximum context", "prompt is too long", "prompt too long",
                "too many tokens", "token limit"].contains { lower.contains($0) }
    }

    // MARK: 失败上下文的跨段传递

    /// 导出失败上下文，供上层在「自动续跑」的段之间传递。
    public func failureSnapshot() -> FailureSnapshot {
        FailureSnapshot(
            failedTotal: failedTotal,
            lastFail: lastFail,
            failSigs: Array(failSigs.suffix(maxFailSigs)),
            failStreak: failStreak,
            finishNudge: finishNudge,
            planFinishNudge: planFinishNudge,
            blockNudge: blockNudge,
            emptyRetry: emptyRetry,
            reaskGuard: reaskGuard,
            planNudged: planNudged,
            didWork: didWork,
            stepFails: stepFails,
            repeatStreak: repeatStreak,
            lastCallSigs: Array(lastCallSigs.suffix(8)),
            intentStall: intentStall,
            shrinkGuard: shrinkGuard
        )
    }

    /// 开始一个新的用户请求：重置与「单次请求」绑定的注入标志。
    ///
    /// 动态上下文按一次请求冻结，同一段 autoContinue 的所有 THINK 轮都
    /// 不该重复注入；只有 Agent 开启新用户输入时才允许重新注入。
    public func beginUserRequest() {
        dynamicContextInjected = false
        // 决策门记账必须按「用户请求」重置：这些变量是 KernelLoop 实例级状态，
        // 若只在「清空对话」时重置，一次判定放行（decisionPrechecked=true）
        // 会让后续所有无关请求永久跳过生成前门/收尾门/计划评审。
        resetDecisionGateState()
    }

    private func resetDecisionGateState() {
        decisionFinishBlocks = 0
        decisionPrechecked = false
        ragHighConfidence = false
        decisionPlanReviews = 0
        decisionReviewedPlanKey = ""
    }

    /// 把上一段的失败上下文带进下一段 run()（自动续跑用）。
    public func seedFailures(_ snapshot: FailureSnapshot?) {
        pendingSeed = snapshot
    }

    /// 清空跨轮运行时记账（UI「清空对话」）。
    public func resetRuntimeState() {
        distillRequested = false
        totalTokens = 0
        lastConsolidatedTurn = -1
        turnsUsed = 0
        emptyRetry = 0
        shrinkGuard = 0
        reaskGuard = 0
        finishNudge = 0
        failStreak = 0
        failedTotal = 0
        lastFail = nil
        failSigs.removeAll()
        stepFails = 0
        blockNudge = 0
        didWork = false
        hasToolSuccess = false
        didFallbackFetch = false
        didFallbackSearch = false
        didFallbackMemorySearch = false
        didFallbackKBStatus = false
        fallbackToolText = ""
        planNudged = 0
        planFinishNudge = 0
        intentStall = 0
        planGateHintInjected = false
        dynamicContextInjected = false
        repeatStreak = 0
        lastCallSigs.removeAll()
        resetDecisionGateState()
        pendingSeed = nil
        resume = false
        state = .think
    }
}
