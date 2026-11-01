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
                execute: @escaping ExecuteFunction,
                schemas: @escaping () throws -> [JSONValue] = { [] },
                shouldCancel: @escaping () -> Bool = { false },
                invalidateKV: (() -> Void)? = nil,
                log: @escaping (String) -> Void = { _ in }) {
        self.stream = stream
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

    public init(failedTotal: Int = 0, lastFail: ToolFailure? = nil,
                failSigs: [String] = [], failStreak: Int = 0,
                finishNudge: Int = 0, planFinishNudge: Int = 0,
                blockNudge: Int = 0, emptyRetry: Int = 0,
                reaskGuard: Int = 0, planNudged: Int = 0,
                didWork: Bool = false, stepFails: Int = 0,
                repeatStreak: Int = 0, lastCallSigs: [String] = []) {
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
    + "制定|实现|开发|配置|初始化|生成报告|写一份|做一份)"

private let planFinishHint = """
[System] 你只是在回复文本里写出了计划、并声称「开始执行第一个任务」，但既没有调用 update_plan 把计划写入执行计划（Session.working['plan'] 为空），也没有发出任何实际工具调用——这等于什么都没做。
请立即：1) 用 update_plan(op='create', goal=..., steps=[...]) 把计划写入执行计划；2) 紧接着发出第一步的实际工具调用并持续推进，直到计划完成。
禁止只在文本里描述计划——计划必须写进执行计划，每一步都必须真正调用工具。
"""

private let finishNudgeMax = 2
private let finishNudgeFailMax = 8
private let finishNudgePlanMax = 6
private let maxFailSigs = 8
private let minAttemptsBeforeBlock = 3
private let blockNudgeMax = 2
private let repeatCallWarn = 2
private let repeatCallLimit = 4

private let askUserPattern =
    "(请(提供|告诉|告知|说明|补充|指定|确认)|需要(你|您)(提供|告诉|确认|补充)|"
    + "能否(提供|告诉|说明)|麻烦(提供|告诉)|"
    + "(哪个|哪些)(文件|目录|路径|项目|模块|分支|环境)|"
    + "想?要我(先)?(做|处理|修改|改)|希望我(做|先做)|你希望|你想要)"

private let planGateHint = """

[System] 开始执行前，先用 update_plan(op="create", goal=..., steps=[...]) 把目标和路径规划成 2~7 个可验证的步骤，并写入执行计划（Session.working['plan']）；**计划建好之前不要先发实际工具调用**——先把目标与步骤登记为执行计划，再**紧接着在同一轮或下一轮开始执行第一步**（计划建好不等于任务开始，不要停下来问用户是否开始/是否继续）；然后从第一步开始，每完成一步就用 update_plan(op="patch", step_id=..., status="done", note="结论") 更新进度。
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
    public var maxTurns: Int
    public private(set) var distillRequested = false

    private let runtime: KernelRuntime
    private let callbacks: AgentCallbacks
    private let config: AgentConfig
    private let systemPrompt: () -> String
    /// 动态上下文（工作记忆/检查点/技能/知识库状态），追加到当前轮 user 消息尾部。
    private let dynamicContext: () -> String
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
    private var planNudged = 0
    private var planFinishNudge = 0
    private var repeatStreak = 0
    private var lastCallSigs: [String] = []
    private var toolCallCounter = 0
    private var resume = false
    private var pendingSeed: FailureSnapshot?

    public init(runtime: KernelRuntime,
                config: AgentConfig,
                callbacks: AgentCallbacks? = nil,
                systemPrompt: @escaping () -> String,
                dynamicContext: @escaping () -> String = { "" },
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
            rawTurns: config.kernelRawTurns == 0 ? 2 : config.kernelRawTurns,
            maxDigestLines: config.kernelMaxDigestLines == 0 ? 40 : config.kernelMaxDigestLines
        )
        self.session = resolved
        // 必须与 Session 共用同一个预算对象，否则内核记的账和 Session 折叠时
        // 读的余额是两本账。
        self.budget = resolved.budget

        // 折叠会改写消息前缀并强制一次 KV 重建；至少 8 轮再折，
        // 避免每个用户回合都触发一次慢 prefill。
        // 但周期不能超过 raw 窗口：超出时窗口每轮都滑动，等于每轮都重建。
        let configured = max(8,
            config.kernelConsolidateEvery == 0 ? 8 : config.kernelConsolidateEvery)
        self.consolidateEvery = max(1, min(configured, max(1, resolved.rawTurns)))
        self.maxToolChars = config.kernelMaxToolChars == 0 ? 8_000 : config.kernelMaxToolChars
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

    /// 执行一段内核循环。
    ///
    /// `emitFinished=false` 供「自动续跑」使用：续跑段是同一个用户轮次的一部分，
    /// 不能触发 `hooks.finished`（它会做事实抽取与 L4 蒸馏）。
    /// `toolsEnabled=false` 供寒暄轮使用：不提供任何工具 schema，
    /// 模型只能直接给出文本回复。
    public func run(_ userInput: String,
                    resume: Bool = false,
                    emitFinished: Bool = true,
                    toolsEnabled: Bool = true) async -> KernelResult {
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
        planNudged = 0
        planFinishNudge = 0
        repeatStreak = 0
        lastCallSigs = []
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
        }
        pendingSeed = nil
        turnsUsed = 0
        distillRequested = false

        var pending = userInput
        var finalText = ""
        var exitReason: ExitReason = .maxTurns

        while turnsUsed < maxTurns {
            if cancelled() {
                log("[kernel] 收到取消请求，终止")
                exitReason = .cancelled
                break
            }

            // ---- CONSOLIDATE ----
            if needsConsolidate() {
                state = .consolidate
                lastConsolidatedTurn = session.currentTurn
                let over = budget.totalOver
                let reason = over ? "预算告罄" : "每 \(consolidateEvery) 轮"
                // 一次折到只剩最近一轮：留满下一个周期的 raw 窗口。
                // 只折「超出部分」时，折叠后窗口仍贴着上限，2~3 轮后又开始
                // 每轮滑窗（`<earlier_context>` 变化 → KV 前缀失效 → 全量 prefill）。
                let folded = session.foldForConsolidation()
                if folded > 0 {
                    runtime.invalidateKV?()
                }
                log("[kernel] CONSOLIDATE (\(reason)) folded=\(folded) \(session.statsDescription())")
                _ = await hooks.consolidated(session, reason: reason)
                if over && folded == 0 && shrinkGuard < 3 {
                    shrinkGuard += 1
                    _ = await hooks.consolidated(session, reason: "fold-exhausted")
                }
                _ = transition("done")
            }

            // ---- THINK ----
            state = .think
            // 规划门：还没干过实际活、且还没有计划时，持续提醒先规划再动手。
            if session.getPlan() == nil && !didWork && !isSimpleInput(userInput) {
                pending += planGateHint
            }
            var userText = session.anchorPrompt(extra: pending)
            // 动态上下文只加在 user 消息尾部：system 与历史前缀保持不变，
            // 本地 KV cache 才能跨轮复用。
            let dynamic = dynamicContext()
            if !dynamic.isEmpty {
                userText += "\n\n" + dynamic
            }
            let rec = session.startTurn(prompt: userText)
            turnsUsed += 1
            let messages = session.buildMessages(systemPrompt: systemPrompt(), userText: userText)
            _ = schemas()

            var respOptional: LLMResponse? = nil
            do {
                respOptional = try await ask(messages, toolsEnabled: toolsEnabled)
            } catch {
                let msg = describe(error)
                // 上下文超长是可恢复的：交给上层兜底收缩后重试同一轮。
                if isContextError(msg) && shrinkGuard < 3 {
                    shrinkGuard += 1
                    session.rollbackTurn()
                    runtime.invalidateKV?()
                    let shrunk = await hooks.consolidated(session, reason: "fold-exhausted")
                    if shrunk {
                        log("[kernel] 上下文超长，兜底收缩后重试 (\(shrinkGuard))")
                        continue
                    }
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
                // 反思门：模型想收尾，但计划里还有未完成步骤 —— 不能就这么放它走。
                let openSteps = session.planOpenSteps()

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

                // 多步任务却连执行计划都没写、也没真正动手就想收尾。
                if session.getPlan() == nil
                    && (isMultistep(userInput))
                    && !didWork
                    && planFinishNudge < finishNudgePlanMax
                    && !resume {
                    planFinishNudge += 1
                    session.rollbackTurn()
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
                if !openSteps.isEmpty && finishNudge < nudgeBudget {
                    finishNudge += 1
                    let todo = openSteps.prefix(5)
                        .map { "\($0.id).\($0.title)" }.joined(separator: "; ")
                    session.rollbackTurn()
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
                        let phrase = resume ? "在本段没有调用任何工具" : "**一次工具都没调用**"
                        pending = """
                        [System] 计划已建立，但你还\(phrase)——只把计划写出来不算开始执行。这是第 \(finishNudge) 次提醒。
                        现在立即开始执行当前步骤（Now）：\(nowStep.id). \(nowStep.title)。
                        要求：下一轮必须包含**工具调用**（读文件 / 执行命令 / 检索 / 写代码等实际动作），不要只描述你打算做什么，不要停下来问用户是否开始或是否继续。
                        """
                    } else {
                        pending = """
                        [System] 计划中还有 \(openSteps.count) 个未完成步骤：\(todo)。请继续执行下一步（调用相应工具）。注意：若某步的工作**实际已经完成**只是忘了更新状态，用 update_plan(op="patch", step_id=..., status="done", note="结论") 收尾即可，不要为响应提醒而重复已完成的工作；若某步确实无法完成，把它标为 blocked/skipped 并写明原因。
                        """
                    }
                    log("[kernel] 反思门：还有 \(openSteps.count) 步未完成，提醒继续")
                    continue
                }

                let event = noToolEvent(resp)
                if event == "retry" {
                    session.rollbackTurn()
                    pending = resp.truncated
                        ? "[ERROR] 上一轮的工具调用被截断：参数还没输出完生成就结束了，工具没有执行。请重新生成，并把较大的写入内容拆成多次较小的工具调用，不要把长文件塞进一次调用。"
                        : "[ERROR] 上一轮回复为空或流被截断，请重新生成；如需操作必须显式调用工具。"
                    continue
                }
                if event == "reask" {
                    reaskGuard += 1
                    if reaskGuard <= 1 {
                        session.rollbackTurn()
                        pending = """
                        [System] 检测到你上轮主要内容是大段代码却未调用工具。若这些代码需要执行/写入/分析，请显式调用 code_run / file_write / file_patch；若只是展示给用户的示例，请在回复中补充自然语言说明。
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
                finalText = resp.content
                session.finishTurn(rec, assistantText: resp.content,
                                   summary: l2Summary(resp.content),
                                   rawContent: resp.rawContent)
                exitReason = .done
                break
            }

            // ---- ACT ----
            _ = transition("has_calls")
            let toolCalls = resp.toolCalls

            // ---- 规划前置门：先于执行 ----
            let hasActual = toolCalls.contains { $0.name != "update_plan" }
            let hasPlanOp = toolCalls.contains { $0.name == "update_plan" }
            if session.getPlan() == nil
                && !didWork
                && hasActual && !hasPlanOp
                && !resume {
                let simple = turnsUsed <= 1
                    && userInput.count <= 60
                    && !isMultistep(userInput)
                if !simple {
                    session.rollbackTurn()
                    pending = """
                    [System] 你准备执行实际工具，但还没有执行计划（Session 中 plan 为空）。请先用 update_plan(op='create', goal='一句话目标', steps=[...]) 把目标与 2~7 个可验证步骤规划好；创建后工具结果会指示『立即开始第一步』，不要停下来问用户是否开始。计划每轮回注，是后续多步任务跟踪的依据，必须先于执行写入。
                    """
                    log("[kernel] 规划前置门：拦截首次执行，要求先 update_plan(create)")
                    continue
                }
            }

            let rawResults: [ToolResult]
            do {
                rawResults = try await runtime.execute(toolCalls)
            } catch {
                log("[kernel] dispatcher error: \(describe(error))")
                rawResults = Array(repeating: ToolResult.fail("工具未返回结果"),
                                   count: toolCalls.count)
            }

            // 失败记账必须在拼装 pending 之前：步骤内失败数要在 update_plan
            // 把它重置之前，被「未充分尝试就 blocked」的守卫读走。
            let stepFailsBefore = stepFails
            let batch = trackFailures(toolCalls, rawResults)
            // 建计划本身不算「开工」：只调 update_plan 就收尾 = 只规划不执行。
            if hasActual { didWork = true }

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
                    finalText = resp.content.trimmingCharacters(
                        in: .whitespacesAndNewlines).isEmpty
                        ? "（检测到连续 \(repeatStreak + 1) 次完全相同的工具调用，"
                            + "内核已停止本轮以避免空转。请调整指令后重试。）"
                        : resp.content
                    exitReason = .done
                    break
                }
                repeatWarning = """
                [System] 你已经连续 \(repeatStreak + 1) 次调用**完全相同**的工具与参数（\(toolCalls.first?.name ?? "?")）——这不会让任务前进哪怕一步。立刻换一个动作：
                1) 若这一步其实做不了/不需要工具，就用 update_plan(patch) 把它标为 done（写明结论）或 skipped，然后进入下一步；
                2) 若任务本身已经可以回答（例如只是问候/闲聊/缺少信息），直接用文字回复用户结束本轮；
                3) 若确实需要用户补充信息，用 ask_user 工具提问。
                不要再次发出与上一轮相同的调用。
                """
            }

            let outcomes = toOutcomes(toolCalls, rawResults)
            let toolResults = zip(toolCalls, outcomes).map { tc, outcome in
                ToolResultRecord(id: tc.id, name: tc.name, content: outcome.data)
            }
            let summary = resp.summary.isEmpty
                ? l2Summary(resp.content, toolNames: toolCalls.map(\.name))
                : resp.summary
            session.finishTurn(rec, assistantText: resp.content,
                               toolCalls: toolCalls, toolResults: toolResults,
                               summary: summary, rawContent: resp.rawContent)
            resetRetryGuards()
            let hasSummary = hasSummaryTag(resp.content)

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
            if !hasSummary {
                parts.append("[TIPS] 回复正文必须包含 <summary>…</summary>（≤30 字物理快照）。")
            }
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
            // 重复调用提醒必须在 pending 赋值前并入，否则会被整体覆盖。
            if let repeatWarning { parts.append(repeatWarning) }
            pending = parts.joined(separator: "\n")
            _ = transition("continue")
        }

        if exitReason == .maxTurns && finalText.isEmpty {
            finalText = "（已达到本轮最大轮次上限 \(maxTurns)，任务可能未完成。请先总结已完成的进展，再询问用户是否继续。）"
        }
        // 提醒次数用尽后模型仍要收尾：尊重决定，但必须把「其实没做完」告诉用户。
        if exitReason == .done && finishNudge > 0 {
            let left = session.planOpenSteps()
            if !left.isEmpty {
                let todo = left.prefix(8).map { "\($0.id).\($0.title)" }.joined(separator: "; ")
                finalText += "\n\n---\n⚠ 计划中仍有 \(left.count) 个未完成步骤：\(todo)"
                    + "（内核已多次要求继续执行仍未完成。若某步其实已做完，用 update_plan 补标 done；若确实不可行，标 blocked 并写明原因。）"
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
        // 紧急折叠只看硬顶（是否侵入输出预留）。raw 软配额是建议值，
        // 用 rawOver 判紧急会变成「每轮折 1 轮 + 每轮丢 KV + 每轮全量 prefill」。
        if budget.totalOver { return true }
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
                     toolsEnabled: Bool = true) async throws -> LLMResponse? {
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
                let stream = cancellableStream(
                    runtime.stream(messages, toolSchemas, config.temperature))
                for try await event in stream {
                    gotAny = true
                    if self.cancelled() {
                        cancelled = true
                        break
                    }
                    let now = Date()
                    if ttfb == nil { ttfb = now }
                    switch event.kind {
                    case .content:
                        contentBuf.append(event.content)
                        callbacks.onContent(event.content)
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
        let reasoningText = reasoningBuf.joined()
        if !seenDone && (!contentBuf.joined().isEmpty || !reasoningText.isEmpty) {
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

        let text = contentBuf.joined()
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

    // MARK: ACT：工具结果 → StepOutcome（L1 裁剪在这里发生）

    private func toOutcomes(_ toolCalls: [ToolCall],
                            _ rawResults: [ToolResult]) -> [StepOutcome] {
        let n = max(1, toolCalls.count)
        let maxChars = budget.remainingForTool(n, maxChars: maxToolChars)
        var out: [StepOutcome] = []
        for (tc, res) in zip(toolCalls, rawResults) {
            if tc.name == "start_long_term_update" {
                distillRequested = true
            }
            // L0：先兜住病态长度；L1：再按 RAW 区余额裁到真正要注入的长度。
            let captured = l0Capture(res.toMessage(), hardCap: l0HardCap)
            out.append(StepOutcome(data: l1Trim(captured, maxChars: maxChars),
                                   nextPrompt: "\n"))
        }
        return out
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

    private func noToolEvent(_ resp: LLMResponse) -> String {
        let content = resp.content.trimmingCharacters(in: .whitespacesAndNewlines)
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
        if truncationMarks.contains(where: { tail.contains($0) })
            || (content.hasSuffix("</summary>") && content.count < 100) {
            emptyRetry += 1
            return emptyRetry <= maxEmptyRetry ? "retry" : "done"
        }
        if looksLikeUnexecutedCode(content) {
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
        emptyRetry = 0
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
            lastCallSigs: Array(lastCallSigs.suffix(8))
        )
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
        planNudged = 0
        planFinishNudge = 0
        repeatStreak = 0
        lastCallSigs.removeAll()
        pendingSeed = nil
        resume = false
        state = .think
    }
}
