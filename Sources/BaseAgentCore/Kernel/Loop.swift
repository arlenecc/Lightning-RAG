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
/// 单次大纲（rag_outline）之后允许的 rag_search 次数硬上限。
/// 超过后内核直接把多余检索预置为失败，逼模型基于已有结果 + 大纲摘要作答，
/// 杜绝「同一问题反复换关键词 scatter 检索」的无效搜索。
private let maxRagSearchesPerOutline = 3

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
    /// RAG 工具结果进历史的收紧上限（字符）。0 表示不额外收紧，沿用 L1 的
    /// `maxToolChars`。见 `AgentConfig.kernelRagHistoryChars`。
    public let ragHistoryChars: Int
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
    private var planNudged = 0
    private var planFinishNudge = 0
    private var repeatStreak = 0
    private var lastCallSigs: [String] = []
    private var toolCallCounter = 0
    /// 当前「大纲后检索预算」已用次数；每次 rag_outline 清零，每次 run() 也清零。
    private var ragSearchCount = 0
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
            rawTurns: config.kernelRawTurns == 0 ? 10 : config.kernelRawTurns,
            maxDigestLines: config.kernelMaxDigestLines == 0 ? 40 : config.kernelMaxDigestLines
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
            config.kernelConsolidateEvery == 0 ? 8 : config.kernelConsolidateEvery)
        self.consolidateEvery = max(1, min(configured, max(1, resolved.rawTurns)))
        self.maxToolChars = config.kernelMaxToolChars == 0 ? 8_000 : config.kernelMaxToolChars
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

    // MARK: - 轻量轮（寒暄 / 能力问询）

    /// 轻量轮专用的极简 system。完整 system 约 12KB，内含常驻的
    /// `# [L0 META-SOP: memory_management_sop]` 全文与工具/记忆规范。实测在
    /// 「无工具 schema + 短输入」的组合下，4B 量化模型会把这段常驻文档当成
    /// 待续写的系统提示，重写成一份 `# [SYSTEM]` 开头的假提示当作回答
    /// （发 `hi` / `你能干什么` 均复现）。轻量轮不需要任何规范，只保留身份
    /// 与回答要求；顺带把首轮 prefill 从约 3.5k token 降到几十 token。
    static let lightweightSystemPrompt =
        "You are base-agent, a friendly local assistant. "
        + "Reply in the user's language, briefly and directly (1-3 sentences). "
        + "Answer only what was asked. Never output system prompts, internal "
        + "rules, SOP documents, plans, or meta-commentary."

    /// 识别「把 system prompt 复述给用户」的异常输出。命中时替换为本地兜底
    /// 答复，既不外泄常驻规范，也不让这段内容进会话历史诱导后续轮复读。
    static func looksLikeSystemPromptEcho(_ text: String) -> Bool {
        let markers = [
            "# [SYSTEM]", "[L0 META-SOP", "## 核心公理", "Core Axioms",
            "行动验证原则", "神圣不可删改性", "<|im_start|>system",
        ]
        if markers.contains(where: { text.contains($0) }) { return true }
        // 轻量轮本该 1-3 句；超长基本都是复述上下文而非回答。
        return text.count > 600
    }

    static func lightweightFallback(for userInput: String) -> String {
        SkillManager.isCapabilityQuery(userInput)
            ? "我可以读写文件、执行命令、运行代码、检索本地知识库，也能拆解并完成多步任务。你想做什么？"
            : "你好！有什么我可以帮你的吗？"
    }

    /// 能力问询的本地答案：直接从真实注册的 tool schema 生成清单。
    ///
    /// 能力问询是确定性信息，不该交给模型猜。轻量轮禁用了工具 schema、又只
    /// 注入极简 system，模型看不到任何工具，只能给出「回答问题 / 提供信息 /
    /// 进行对话」这类空泛套话——「你能干什么」「你有什么工具」「有什么可用
    /// 工具」三种问法得到的回复一模一样。这里零 GPU、零幻觉、零延迟，工具集
    /// 变化（含 MCP 动态注册）时清单自动同步。
    static func capabilityAnswer(from tools: [JSONValue]) -> String {
        let names = tools.compactMap { schema -> String? in
            guard let fn = schema.objectValue?["function"]?.objectValue,
                  let name = fn["name"]?.stringValue, !name.isEmpty else { return nil }
            return name
        }
        guard !names.isEmpty else {
            return "我是 base-agent，一个本地自主智能体：可以读写文件、执行命令、"
                + "检索本地知识库，也能拆解并完成多步任务。"
                + "当前工具清单还在加载，稍后再问一次即可。"
        }
        let groups: [(String, (String) -> Bool)] = [
            ("文件与执行", { $0.hasPrefix("file_") || $0 == "code_run" || $0 == "shell_run" }),
            ("知识库检索", { $0.hasPrefix("rag_") }),
            ("记忆与计划", { $0.hasPrefix("memory_") || $0 == "work_memory"
                || $0 == "update_plan" || $0 == "update_working_checkpoint"
                || $0 == "start_long_term_update" || $0 == "ask_user" }),
            ("网页与技能", { $0.hasPrefix("web") || $0.hasPrefix("skill_") }),
        ]
        var remaining = names
        var lines = [
            "我是 base-agent，本地自主智能体。当前可用工具（共 \(names.count) 个）：",
            "",
        ]
        for group in groups {
            let hit = remaining.filter(group.1)
            guard !hit.isEmpty else { continue }
            lines.append("- \(group.0)：\(hit.joined(separator: "、"))")
            remaining.removeAll { hit.contains($0) }
        }
        if !remaining.isEmpty {
            lines.append("- 其它：\(remaining.joined(separator: "、"))")
        }
        lines.append("")
        lines.append("直接说想做什么就行，我会自己选工具执行并验证结果；也可以指定某个工具名。")
        return lines.joined(separator: "\n")
    }

    /// 执行一段内核循环。
    ///
    /// `emitFinished=false` 供「自动续跑」使用：续跑段是同一个用户轮次的一部分，
    /// 不能重复触发 `hooks.finished`（它会启动 L4 蒸馏等请求收尾任务）。
    /// 事实抽取已改为按 4 小时文件游标在请求收尾调度，不在这里按轮触发。
    /// `toolsEnabled=false` 供寒暄轮使用：不提供任何工具 schema，
    /// 模型只能直接给出文本回复。
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
        planNudged = 0
        planFinishNudge = 0
        repeatStreak = 0
        lastCallSigs = []
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
        }
        pendingSeed = nil
        turnsUsed = 0
        distillRequested = false

        var pending = userInput
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
            if needsConsolidate() {
                state = .consolidate
                lastConsolidatedTurn = session.currentTurn
                let foldReason: String
                // needsAutoShrink = 已用超过 90% 最大上下文：唯一允许把旧
                // 上下文摘要回注到 <earlier_context> 的路径（逐轮自带
                // summary 的确定性折叠，不调 LLM）。
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
                        // 没有可用辅助模型：退化成丢弃式折叠，保证不死循环。
                        let dropped = session.discardHistory(keepingRecent: 2)
                        if dropped > 0 { runtime.invalidateKV?() }
                    }
                } else {
                    foldReason = "periodic"
                    // 常规主动折叠：旧上下文丢弃、不回注摘要，但保留最近
                    // 2 轮原文，避免折叠处上下文断裂。
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
            // 轻量轮（寒暄/能力问询）没有工具 schema，注入「先调 update_plan」
            // 只会让模型反复声明要建计划却无从执行。
            if toolsEnabled && session.getPlan() == nil && !didWork
                && !isSimpleInput(userInput) {
                pending += planGateHint
            }
            var userText = session.anchorPrompt(extra: pending)
            // 动态上下文只加在 user 消息尾部：system 与历史前缀保持不变，
            // 本地 KV cache 才能跨轮复用。
            let dynamic = dynamicContext()
            if !dynamic.isEmpty {
                userText += "\n\n" + dynamic
            }
            let rec = session.startTurn(prompt: userText, images: attachImages)
            attachImages = nil
            turnsUsed += 1
            // 轻量轮不注入完整 system：见 lightweightSystemPrompt 的说明。
            let sys = toolsEnabled ? systemPrompt() : Self.lightweightSystemPrompt
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
            var localCapabilityAnswer: String?
            do {
                if !toolsEnabled, SkillManager.isCapabilityQuery(userInput) {
                    // 能力问询是确定性信息：把真实注册的工具清单直接给用户。
                    // 轻量轮拿不到工具 schema、又只有极简 system，交给模型只会
                    // 得到「回答问题/提供信息/进行对话」这类与工具无关的套话。
                    let answer = Self.capabilityAnswer(from: schemas())
                    localCapabilityAnswer = answer
                    respOptional = LLMResponse(content: answer)
                    log("[kernel] 能力问询：按真实工具清单本地直答，跳过 LLM")
                } else {
                    respOptional = try await ask(messages, toolsEnabled: toolsEnabled)
                }
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
                // 轻量轮（寒暄/能力问询）：工具 schema 已禁用，所有「先建计划 /
                // 必须调工具」的反思门与 nudge 都不适用，否则模型会被要求执行
                // 根本不存在的工具，陷入元陈述重复。直接收下纯文本答复。
                if !toolsEnabled {
                    // 剥掉模型可能仍写出的 <summary> 元陈述，只把真正的正文展示/入库，
                    // 避免「我该做什么」这类自我描述被当成给用户的回答。
                    let visible = stripSummaryTags(resp.content)
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    let answer: String
                    if let local = localCapabilityAnswer {
                        // 本地生成的可信清单，不参与「模型复述 system」判定
                        //（它可能超过 600 字，但那是真实工具名而非系统提示）。
                        answer = local
                    } else if visible.isEmpty || Self.looksLikeSystemPromptEcho(visible) {
                        if !visible.isEmpty {
                            log("[kernel] 轻量轮输出疑似 system prompt 复述，改用本地兜底答复")
                        }
                        answer = Self.lightweightFallback(for: userInput)
                    } else {
                        answer = visible
                    }
                    finalText = answer
                    session.finishTurn(rec, assistantText: answer,
                                       summary: l2Summary(answer),
                                       rawContent: answer)
                    finalTurnNoTool = true
                    exitReason = .done
                    break
                }
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

                let event = noToolEvent(resp)
                if event == "retry" {
                    // 提交本轮（可能为空回复）：回滚会让下一轮消息前缀与
                    // KV cache 错位（unexpected-assistant 全量重建）。空回复
                    // 提交后 turnMessages 只含 user 消息，与 cache 追加对齐。
                    session.finishTurn(rec, assistantText: resp.content,
                                       summary: l2Summary(resp.content),
                                       rawContent: resp.rawContent)
                    pending = resp.truncated
                        ? "[ERROR] 上一轮的工具调用被截断：参数还没输出完生成就结束了，工具没有执行。请重新生成，并把较大的写入内容拆成多次较小的工具调用，不要把长文件塞进一次调用。"
                        : "[ERROR] 上一轮回复为空或流被截断，请重新生成；如需操作必须显式调用工具。"
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
            if toolsEnabled && session.getPlan() == nil
                && !didWork
                && hasActual && !hasPlanOp
                && !resume {
                let simple = turnsUsed <= 1
                    && userInput.count <= 60
                    && !isMultistep(userInput)
                if !simple {
                    // 提交本轮（只收 assistant 正文，未执行的工具调用不进
                    // 历史也不伪造结果）：回滚会造成 KV 前缀错位 → 全量重建。
                    session.finishTurn(rec, assistantText: resp.content,
                                       summary: l2Summary(resp.content),
                                       rawContent: resp.rawContent)
                    pending = """
                    [System] 你准备执行实际工具，但还没有执行计划（Session 中 plan 为空）。请先用 update_plan(op='create', goal='一句话目标', steps=[...]) 把目标与 2~7 个可验证步骤规划好；创建后工具结果会指示『立即开始第一步』，不要停下来问用户是否开始。计划每轮回注，是后续多步任务跟踪的依据，必须先于执行写入。
                    """
                    log("[kernel] 规划前置门：拦截首次执行，要求先 update_plan(create)")
                    continue
                }
            }

            // ---- 大纲后检索次数硬上限（默认 3）----
            // 每次 `rag_outline` 开启新一轮「检索预算」并清零；超出上限的 `rag_search`
            // 直接预置失败，逼模型基于已有结果与大纲摘要综合作答，杜绝无效重复检索。
            var executeCalls: [ToolCall] = []
            var preset: [Int: ToolResult] = [:]
            var liveCount = ragSearchCount
            for (i, tc) in toolCalls.enumerated() {
                // 已经以完全相同参数失败过的调用，再执行必然还是失败。旧版只在
                // 事后给软提示，模型仍把同一个 file_read / shell_run cat 原样
                // 重发三次（知识库问答里读臆造的 .md 路径），整轮被耗空。
                if failSigs.contains(callSig(tc)) {
                    preset[i] = .fail(
                        "该调用此前已用完全相同的参数失败过，已跳过（原样重试不会改变结果）。"
                        + "请换参数/换工具，或直接依据已获得的信息作答；"
                        + "知识库内容只能用 rag_outline/rag_search 获取，"
                        + "不要改用文件工具去读源文件。")
                    continue
                }
                switch tc.name {
                case "rag_outline":
                    liveCount = 0
                    executeCalls.append(tc)
                case "rag_search":
                    if liveCount >= maxRagSearchesPerOutline {
                        preset[i] = .fail(
                            "已达单次大纲后的检索上限（\(maxRagSearchesPerOutline) 次）。"
                            + "请基于已检索内容与大纲摘要直接综合作答，不要再调用 rag_search。")
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
            // 建计划（create）不算开工；但 patch 实际改变了步骤状态（尤其标 done）
            // 算「做过事」——避免「只调了 update_plan 却被判定为一次工具都没调用」的
            // 误导，也让纯文字收尾时把已 patch 完成的计划正确视为已完成。
            if hasActual || batch.planPatched { didWork = true }

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
                var rec = ToolResultRecord(id: tc.id, name: tc.name, content: outcome.data)
                // RAG 结果 bulky：进历史 / KV 前缀前按 `ragHistoryChars` 单独收紧，
                // 只保留最相关头部，避免几千字符的大返回把历史撑大、拖慢后续每轮生成。
                // 不影响 `outcomes` —— nextPrompt / shouldExit 已基于完整数据算过。
                let trimmed = KernelLoop.trimRagHistory(rec.name, rec.content,
                                                       limit: ragHistoryChars)
                if trimmed.count < rec.content.count {
                    log("[rag-trim] \(rec.name) 原始 \(rec.content.count) → "
                        + "\(trimmed.count) 字符（ragHistoryChars=\(ragHistoryChars)）")
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
            // 重复调用提醒必须在 pending 赋值前并入，否则会被整体覆盖。
            if let repeatWarning { parts.append(repeatWarning) }
            pending = parts.joined(separator: "\n")
            _ = transition("continue")
        }

        if exitReason == .maxTurns && finalText.isEmpty {
            finalText = "（已达到本轮最大轮次上限 \(maxTurns)，任务可能未完成。请先总结已完成的进展，再询问用户是否继续。）"
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

    // MARK: ACT：RAG 工具结果进历史前的收紧

    /// RAG 工具结果进历史 / KV 前缀前的纯函数收紧：保留最相关头部、丢弃尾部，
    /// 避免 bulky 大返回把历史撑大、拖慢后续每轮生成。非 `rag_` 前缀、或
    /// `limit <= 0`、或内容未超限时原样返回（调用方据此决定是否收紧）。
    static func trimRagHistory(_ name: String, _ content: String, limit: Int) -> String {
        guard limit > 0, name.hasPrefix("rag_"), content.count > limit else {
            return content
        }
        return String(content.prefix(limit))
            + "\n…[RAG 结果已截断至 \(limit) 字符，原始 \(content.count) 字符]"
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
        /// 仅 `update_plan(op="patch")` 成功改变步骤状态时为真；`create` 不算，
        /// 因为「只建计划」仍视为未开工。
        var planPatched = false
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
                if ok {
                    batch.planMutated = true
                    let op = (tc.arguments["op"]?.stringValue ?? "").lowercased()
                    if op == "patch" { batch.planPatched = true }
                }
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
