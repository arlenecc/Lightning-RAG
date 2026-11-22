import SwiftUI
import AppKit
import UniformTypeIdentifiers
import BaseAgentCore

struct SettingsView: View {
    @ObservedObject var model: AppViewModel
    @Environment(\.dismissWindow) private var dismissWindow
    @Environment(\.openWindow) private var openWindow

    // MARK: 主模型

    @State private var mainMode = "local"
    @State private var mainBackend = "mlx-lm"
    @State private var mainPath = ""
    @State private var mainContext = 32768
    @State private var mainKVCache = 32_768
    @State private var kvInt8Enabled = true
    @State private var flashAttentionEnabled = true
    @State private var mainTokenizer = ""
    @State private var auxTokenizer = ""
    @State private var mainTemperature = 0.7
    @State private var mainTopP = 0.95
    @State private var mainMinP = 0.05
    @State private var mainTopK = 20
    @State private var mainRepetitionPenalty = 1.0
    @State private var mainTimeout = 120.0
    /// 单次任务允许的最大工具调用轮数。
    @State private var maxIterations = 15
    /// 是否把 L0 META-SOP / L1 索引常驻 system prompt（默认关闭，节省上下文）。
    @State private var memoryHierarchyInSystemPrompt = false
    @State private var mainPromptBatch = 512
    @State private var mainThinking = true
    @State private var mainKVReuse = true
    /// 应用启动时是否预加载主模型（auto_preload_main）。
    @State private var autoPreloadMain = true

    // MARK: API 模式

    @State private var baseURL = ""
    @State private var apiKey = ""
    @State private var apiModel = ""
    @State private var apiModels: [String] = []
    @State private var fetchingModels = false

    // MARK: 辅助模型

    @State private var auxEnabled = false
    @State private var auxBackend = "mlx-lm"
    @State private var auxPath = ""
    @State private var auxContext = 8192
    @State private var auxKVCache = 8192
    @State private var auxTemperature = 0.3
    @State private var auxTopP = 0.95
    @State private var auxMinP = 0.05
    @State private var auxTopK = 20
    @State private var auxRepetitionPenalty = 1.1
    @State private var auxTimeout = 120.0
    @State private var auxPromptBatch = 512
    @State private var auxThinking = false
    @State private var auxKVReuse = false
    @State private var auxKVInt8Enabled = true
    /// 主模型 GGUF 的多模态投影文件（mmproj），可选；需要多模态能力时必须设置。
    @State private var mainMmprojPath = ""
    /// 辅助模型 GGUF 的多模态投影文件（mmproj），可选；需要多模态能力时必须设置。
    @State private var auxMmprojPath = ""

    // MARK: 知识同步

    @State private var workspace = ""
    @State private var knowledgeBase = ""
    @State private var embeddingBackend = "mlx"
    @State private var embeddingPath = "models/jina-embeddings-v5-text-nano-mlx-int4"
    @State private var rerankerBackend = "onnx"
    @State private var rerankerPath = ""
    /// 当前 Embedding 后端下「切片实际生效值」的 token 上限（nil = 无精确换算）。
    @State private var embeddingContentLimit: Int?

    // MARK: 高级设置

    @State private var pdfEngine: PDFEngine = .vision
    @State private var epubOCREngine: PDFEngine = .vision
    @State private var ocrModelPath = ""
    @State private var ocrMMProjPath = ""
    /// 自定义 OCR 选 ONNX 时必填的 tokenizer 文件（或含 tokenizer.json 的目录）。
    @State private var ocrTokenizerPath = ""
    @State private var vlmBackend = "mlx"
    @State private var ocrPreprocess = true
    @State private var ocrBinarize = true
    @State private var chunkSize = 800
    @State private var chunkOverlap = 80
    /// 向量落盘元素类型与 Matryoshka 截断策略。
    @State private var vectorElementType: VectorElementType = .int8
    @State private var matryoshkaMode: MatryoshkaMode = .auto
    @State private var matryoshkaDim = 128
    /// ANN 检索开关与同步后自动构建。
    @State private var annEnabled = true
    @State private var annAutoBuild = true
    /// 二值粗筛开关与第一阶段候选数。
    @State private var binaryCoarseEnabled = true
    @State private var binaryCoarseCandidates = 200
    @State private var rerankEnabled = true
    @State private var rerankTopN = 3
    /// reranker 候选池放大倍数（默认 3）：候选池 = max(top_k, TopN) × 它。
    @State private var rerankCandidateMultiplier = 3
    /// 重排分数过滤阈值。
    @State private var rerankThreshold = 0.45
    /// 知识库只读查询超时（0 = 关闭）。
    @State private var queryTimeoutSeconds = 5.0
    @State private var semanticChunking = true
    @State private var autoIngest = false
    /// 同步并发、Vision OCR 并发与 MLX 内存上限。
    @State private var ingestConcurrency = 10
    @State private var visionConcurrency = 4
    @State private var mlxCacheLimitMB = 1_024
    @State private var mlxMemoryLimitMB = 0
    @State private var embeddingMinBatch = 2
    @State private var embeddingMaxBatch = 32
    @State private var embeddingBatchStep = 2
    @State private var embeddingInitialBatch = 16
    @State private var embeddingMemoryLimitMB = 6_144
    @State private var embeddingThroughputTolerance = 0.03
    @State private var embeddingProbeBatches = 2
    @State private var embeddingProbeSeconds = 2.0
    @State private var embeddingTokenBudgetMin = 4_096
    @State private var embeddingTokenBudgetMax = 8_192
    @State private var embeddingTokenBudgetInitial = 6_144
    @State private var embeddingTokenBudgetStep = 512
    /// 默认信任 mtime + 大小；开启后为「保留时间戳的同尺寸改动」再校验内容指纹。
    @State private var verifyFingerprint = false
    /// 知识图谱融合策略（配置字段仍为 ragRetrievalStrategy）。
    @State private var retrievalStrategy: RAGRetrievalStrategy = .naive
    @State private var showAdvanced = false
    /// RAG MCP 服务（Streamable HTTP）：监听范围 / 端口（运行中时锁定）。
    @State private var ragMCPScope = false
    @State private var ragMCPPortText = "8080"
    /// 检索阅读模式（ColBERT 大分片多向量检索）。
    @State private var colbertReaderEnabled = false
    @State private var colbertQuantBits = 4
    @State private var colbertCoarseCandidates = 50
    @State private var colbertTopN = 3
    @State private var colbertUseANN = true
    @State private var colbertLengthNorm = false
    @State private var colbertQueryPrune = true
    /// Weighted RRF 向量路权重（BM25 = 1 - 该值）；向量/极速模式共用。
    @State private var rrfVectorWeight: Double = 0.7
    /// 多库联合检索是否用判定模型（向量/极速模式共用开关）。
    @State private var multiZoneUseDecisionJudge = false
    @State private var kernelDecisionGateEnabled = true
    @State private var kernelFinishGateThreshold = 0.5
    @State private var kernelPlanReviewThreshold = 0.5
    @State private var skillDecisionGateThreshold = 0.5

    /// 多知识分区配置行（UI 可增删）。
    private struct ZoneRow: Identifiable {
        let id: String
        var name: String
        var source: String
        var enabled: Bool
        var isDefault: Bool
    }
    @State private var zoneRows: [ZoneRow] = []

    // MARK: RAG 模式三选一 + 决策判定
    @State private var ragMode = "vector"
    @State private var decisionModelBackend = "mlx-vlm"
    @State private var decisionModelKind = "meteor"
    @State private var decisionModelPath = "models/PKU-KCL-Meteor-0.8B-mlx-int4"
    @State private var decisionCoarseCandidates = 50
    @State private var decisionUseCoreSentences = true
    @State private var decisionCoreSentenceCount = 3
    @State private var decisionTopN = 5

    @State private var statusMessage = ""

    private let llmTypes: [(id: String, label: String)] = [
        ("mlx-lm", "MLX-LM"),
        ("mlx-vlm", "MLX-VLM"),
        ("onnx", "ONNX"),
        ("gguf", "GGUF"),
    ]
    private let embeddingTypes: [(id: String, label: String)] = [
        ("fastembed", "FastEmbed"),
        ("mlx", "MLX"),
        ("onnx", "ONNX"),
        ("gguf", "GGUF"),
    ]

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    agentSection
                    directorySection
                    knowledgeSection
                }
                .padding(16)
            }

            Divider()

            HStack(spacing: 8) {
                if model.isModelLoading {
                    ProgressView().controlSize(.small)
                    Text("模型加载中…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Text(statusMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                Spacer()
                Button("应用配置") { applyAllConfig() }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.isSyncing || model.isDeduplicating || model.isBuildingANN
                        || model.isModelLoading || model.isRunning || model.isWarmingUp)
                Button("关闭") { dismissWindow() }
                    .keyboardShortcut(.cancelAction)
            }
            .padding(12)
        }
        .frame(minWidth: 880, minHeight: 760)
        .onAppear(perform: load)
        .onChange(of: embeddingBackend) { refreshEmbeddingLimit() }
        .onChange(of: embeddingPath) { refreshEmbeddingLimit() }
    }

    // MARK: - Agent 模型配置区

    private var agentSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                mainModelBlock
                Divider()
                auxModelBlock
                Divider()
                runParametersRow
                Divider()
                kernelDecisionGateRow
            }
            .padding(8)
        } label: {
            Text("Agent 模型配置").font(.headline)
        }
    }

    /// 决策阈值档位：0.30 ~ 0.95，每 0.05 一档（0.50 为默认）。
    private static let decisionThresholdSteps: [Double] = (6...19).map { Double($0) * 0.05 }

    /// 决策阈值 Picker。用数组下标绑定，避免 Double 浮点相等导致选中项丢失。
    private func decisionThresholdPicker(_ title: String, selection: Binding<Double>,
                                         help: String) -> some View {
        let steps = Self.decisionThresholdSteps
        let closestIndex = steps.indices.min {
            abs(steps[$0] - selection.wrappedValue) < abs(steps[$1] - selection.wrappedValue)
        } ?? 4
        return Picker(title, selection: Binding(
            get: { closestIndex },
            set: { index in
                guard steps.indices.contains(index) else { return }
                selection.wrappedValue = steps[index]
            })) {
            ForEach(steps.indices, id: \.self) { index in
                Text(String(format: "%.2f", steps[index])).tag(index)
            }
        }
        .font(.caption2)
        .frame(width: 110)
        .help(help)
    }

    /// 决策门一行：启用开关 + 计划完整性/执行完成度/技能符合度三个阈值。
    private var kernelDecisionGateRow: some View {
        HStack(spacing: 10) {
            Text("决策门")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Toggle("启用", isOn: $kernelDecisionGateEnabled)
                .font(.caption2)
                .toggleStyle(.checkbox)
                .help("收尾时决策模型判定「信息是否足以完整回答」，计划创建后评审「每步是否必要合理」；"
                    + "技能检索/加载/摘要注入前也由决策模型判定是否合适。判定失败/超时静默放行。"
                    + "复用「高级 RAG / 检索模式」中的决策模型，无需额外加载")
            if kernelDecisionGateEnabled {
                decisionThresholdPicker("计划完整性", selection: $kernelPlanReviewThreshold,
                    help: "计划步骤评审阈值：步骤不必要概率高于该值才提示修订；越高越保守（越少提示）")
                decisionThresholdPicker("执行完成度", selection: $kernelFinishGateThreshold,
                    help: "收尾拦截阈值：信息不足概率高于该值才拦截补工具；越高越保守（越少拦截）")
                decisionThresholdPicker("技能符合度", selection: $skillDecisionGateThreshold,
                    help: "技能加载/摘要注入判定阈值：不合适概率高于该值才拦截；越高越保守（越少拦截）")
            }
            Spacer(minLength: 4)
        }
    }

    private var mainModelBlock: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Text("主模型").font(.subheadline).bold()
                if mainMode == "local" {
                    Toggle("启动时预加载", isOn: $autoPreloadMain)
                        .toggleStyle(.checkbox)
                        .help("默认开启：应用启动时在后台加载主模型；关闭后需手动点「加载模型」。修改后需点「应用配置」保存；同步结束后的模型恢复不受影响")
                }
                Spacer()
                Picker("", selection: $mainMode) {
                    Text("本地模式").tag("local")
                    Text("OpenAI 兼容接口").tag("api")
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 260)
            }

            if mainMode == "local" {
                localModelRows(
                    backend: $mainBackend,
                    path: $mainPath,
                    pathHint: "默认：models/qwen3.5-4b-agentic-coder-v4-4bit-mlx",
                    tokenizer: $mainTokenizer,
                    mmproj: $mainMmprojPath,
                    context: $mainContext,
                    kvCap: $mainKVCache,
                    temperature: $mainTemperature,
                    topP: $mainTopP,
                    minP: $mainMinP,
                    topK: $mainTopK,
                    repetitionPenalty: $mainRepetitionPenalty,
                    promptBatch: $mainPromptBatch,
                    thinking: $mainThinking,
                    kvReuse: $mainKVReuse,
                    kvInt8: $kvInt8Enabled,
                    flashAttention: $flashAttentionEnabled
                ) {
                    HStack(spacing: 8) {
                        if model.mainModelLoaded {
                            Button("重新加载") { loadMainModel() }
                                .buttonStyle(.borderedProminent)
                                .tint(.green)
                                .disabled(model.isModelLoading || model.isRunning
                                    || model.isWarmingUp || model.isSyncing
                                    || model.isBuildingANN)
                        } else {
                            Button("加载模型") { loadMainModel() }
                                .disabled(model.isModelLoading || model.isRunning
                                    || model.isWarmingUp || model.isSyncing
                                    || model.isBuildingANN)
                        }
                        Button("卸载") { model.unloadMainModel() }
                            .disabled(model.isModelLoading || model.isRunning
                                || model.isWarmingUp || model.isSyncing)
                    }
                }
            } else {
                apiModelRows
            }
        }
    }

    private var apiModelRows: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text("Base URL").font(.caption)
                TextField("https://api.openai.com/v1", text: $baseURL)
                    .textFieldStyle(.roundedBorder)
                    .frame(minWidth: 180)
                Text("API Key").font(.caption)
                SecureField("sk-…", text: $apiKey)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 140)
                Text("模型").font(.caption)
                if apiModels.isEmpty {
                    TextField("模型名", text: $apiModel)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 170)
                } else {
                    Picker("", selection: $apiModel) {
                        ForEach(apiModelChoices, id: \.self) { name in
                            Text(name).tag(name)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 170)
                }
                Button(fetchingModels ? "获取中…" : "获取模型") { fetchModels() }
                    .disabled(fetchingModels)
            }

            HStack(alignment: .center, spacing: 6) {
                parameterFields(
                    context: $mainContext,
                    temperature: $mainTemperature,
                    topP: $mainTopP,
                    minP: $mainMinP,
                    topK: $mainTopK,
                    repetitionPenalty: $mainRepetitionPenalty
                )
                Spacer(minLength: 4)
                Button("应用配置") { applyAPIConfig() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(model.isSyncing || model.isDeduplicating || model.isBuildingANN)
            }
        }
    }

    private var auxModelBlock: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Text("辅助模型").font(.subheadline).bold()
                Toggle("启用", isOn: $auxEnabled)
                    .toggleStyle(.checkbox)
                    .help("默认关闭：启动和知识同步前后都不加载辅助模型；关闭时 RAG 精简跳过，后台记忆/总结回退主模型。修改后需点击「应用配置」或「同步知识」保存")
                Spacer()
            }
            localModelRows(
                backend: $auxBackend,
                path: $auxPath,
                pathHint: "默认：models/Qwen3.5-2B-MLX-4bit",
                tokenizer: $auxTokenizer,
                mmproj: $auxMmprojPath,
                context: $auxContext,
                kvCap: $auxKVCache,
                temperature: $auxTemperature,
                topP: $auxTopP,
                minP: $auxMinP,
                topK: $auxTopK,
                repetitionPenalty: $auxRepetitionPenalty,
                promptBatch: $auxPromptBatch,
                thinking: $auxThinking,
                kvReuse: $auxKVReuse,
                kvInt8: $auxKVInt8Enabled,
                flashAttention: $flashAttentionEnabled
            ) {
                HStack(spacing: 8) {
                    if model.auxModelLoaded {
                        Button("重新加载") { loadAuxModel() }
                            .buttonStyle(.borderedProminent)
                            .tint(.green)
                            .disabled(!auxEnabled || model.isModelLoading || model.isRunning
                                || model.isWarmingUp || model.isSyncing
                                || model.isBuildingANN)
                    } else {
                        Button("加载模型") { loadAuxModel() }
                            .disabled(!auxEnabled || model.isModelLoading || model.isRunning
                                || model.isWarmingUp || model.isSyncing
                                || model.isBuildingANN)
                    }
                    Button("卸载") { model.unloadAuxModel() }
                        .disabled(model.isModelLoading || model.isRunning
                            || model.isWarmingUp || model.isSyncing)
                }
            }
        }
    }

    /// Agent 运行参数：迭代轮数 / 查询超时 / MLX 缓存与内存上限。
    private var runParametersRow: some View {
        HStack(spacing: 14) {
            Text("运行参数")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Stepper("最大迭代：\(maxIterations) 轮",
                    value: $maxIterations, in: 1...50)
                .help("单次任务允许的最大工具调用轮数，默认 15；"
                    + "调大能处理更长的任务，也会增加耗时与 API 费用")
            Stepper("查询超时：\(String(format: "%.0f", queryTimeoutSeconds))s",
                    value: $queryTimeoutSeconds, in: 0...60, step: 1)
                .help("知识库只读查询（打开库 / 状态 / 检索）的超时秒数，默认 5s，0 = 关闭。"
                    + "大库或上次同步中断后 SQLite 恢复 WAL 较慢时，可适当调大")
            Stepper("MLX 缓存：\(mlxCacheLimitMB)MB",
                    value: $mlxCacheLimitMB,
                    in: 256...2_048, step: 128)
                .help("MLX 可复用 buffer 的上限，默认 1024MB；"
                    + "运行时夹在 256...2048，只影响空闲缓存，不含模型权重与 KV。"
                    + "实测 1GB 让 decode 快 5–8% 且接近 embedding 工作集，2GB 基本没有额外提速")
            Stepper("MLX 上限：\(mlxMemoryLimitText)",
                    value: $mlxMemoryLimitMB,
                    in: 0...32_768, step: 512)
                .help("MLX 全局内存硬上限（含权重/KV/buffer）；"
                    + "0 = 自动取 min(物理内存 65%, 9.6GB)。误配可能导致 OOM")
            Toggle("多级记忆管理", isOn: $memoryHierarchyInSystemPrompt)
                .toggleStyle(.checkbox)
                .help("勾选后 L0 META-SOP 与 L1 索引常驻 system prompt；"
                    + "默认关闭，每轮省约 2.6k 字符上下文。关闭不影响 memory 文件与记忆工具")
            Spacer(minLength: 8)
            Button("应用配置") { applyAPIConfig() }
                .disabled(model.isSyncing || model.isDeduplicating || model.isBuildingANN
                    || model.isModelLoading || model.isRunning || model.isWarmingUp)
        }
        .font(.caption)
    }

    /// 本地模型：类型 + 路径 + 参数 + 操作按钮（两行紧贴，参数标签与文本框同行）。
    private func localModelRows(
        backend: Binding<String>,
        path: Binding<String>,
        pathHint: String,
        tokenizer: Binding<String>? = nil,
        mmproj: Binding<String>? = nil,
        context: Binding<Int>,
        kvCap: Binding<Int>,
        temperature: Binding<Double>,
        topP: Binding<Double>,
        minP: Binding<Double>,
        topK: Binding<Int>,
        repetitionPenalty: Binding<Double>,
        promptBatch: Binding<Int>,
        thinking: Binding<Bool>,
        kvReuse: Binding<Bool>,
        kvInt8: Binding<Bool>,
        flashAttention: Binding<Bool>,
        @ViewBuilder actions: () -> some View
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("模型类型").font(.caption)
                Picker("", selection: backend) {
                    ForEach(llmTypes, id: \.id) { type in
                        Text(type.label).tag(type.id)
                    }
                }
                .labelsHidden()
                .frame(width: 140)

                Text("本地路径").font(.caption)
                TextField(pathHint, text: path)
                    .textFieldStyle(.roundedBorder)
                Button("选择…") {
                    chooseModelPath(path, backend: backend.wrappedValue,
                                    dirTitle: "选择模型目录",
                                    fileTitle: "选择 GGUF 模型文件")
                }
            }
            if let tokenizer, backend.wrappedValue == "onnx" {
                HStack(spacing: 8) {
                    Text("tokenizer路径").font(.caption)
                    TextField("可选：含 tokenizer.json 的目录，留空从模型目录读取",
                              text: tokenizer)
                        .textFieldStyle(.roundedBorder)
                    Button("选择…") {
                        chooseDirectory(tokenizer, title: "选择 tokenizer 目录")
                    }
                }
            }

            // GGUF 多模态投影文件（mmproj）：可选；需要多模态能力时必须设置。
            if let mmproj, backend.wrappedValue == "gguf" {
                HStack(spacing: 8) {
                    Text("mmproj路径").font(.caption)
                    TextField("可选：多模态投影文件，留空自动查找模型同目录的 mmproj*.gguf",
                              text: mmproj)
                        .textFieldStyle(.roundedBorder)
                    Button("选择…") { chooseFile(mmproj, title: "选择 mmproj 文件") }
                }
            }

            // 参数行（含「思考 / KV复用 / KV量化」开关）尽量与操作按钮同行；
            // 窗口较窄放不下时按钮自动落到下一行右端，参数行不会被挤压变形。
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .center, spacing: 6) {
                    parameterFields(
                        context: context,
                        kvCap: kvCap,
                        temperature: temperature,
                        topP: topP,
                        minP: minP,
                        topK: topK,
                        repetitionPenalty: repetitionPenalty,
                        promptBatch: backend.wrappedValue == "mlx-lm" ? promptBatch : nil,
                        thinking: thinking,
                        // ONNX 自动复用 past_key_values、MLX-VLM 每轮重建 prompt，
                        // 都不消费这两个开关，故仅在支持的后端上显示。
                        kvReuse: supportsKVToggles(backend.wrappedValue) ? kvReuse : nil,
                        kvInt8: supportsKVToggles(backend.wrappedValue) ? kvInt8 : nil,
                        flashAttention: supportsKVToggles(backend.wrappedValue)
                            ? flashAttention : nil
                    )
                    Spacer(minLength: 4)
                    actions()
                }
                VStack(alignment: .leading, spacing: 6) {
                    parameterFields(
                        context: context,
                        kvCap: kvCap,
                        temperature: temperature,
                        topP: topP,
                        minP: minP,
                        topK: topK,
                        repetitionPenalty: repetitionPenalty,
                        promptBatch: backend.wrappedValue == "mlx-lm" ? promptBatch : nil,
                        thinking: thinking,
                        kvReuse: supportsKVToggles(backend.wrappedValue) ? kvReuse : nil,
                        kvInt8: supportsKVToggles(backend.wrappedValue) ? kvInt8 : nil,
                        flashAttention: supportsKVToggles(backend.wrappedValue)
                            ? flashAttention : nil
                    )
                    HStack(spacing: 6) {
                        Spacer(minLength: 4)
                        actions()
                    }
                }
            }
        }
    }

    // MARK: - 知识同步配置区

    private var knowledgeSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                // RAG 模式三选一：切换即切换检索召回路径。
                HStack(spacing: 8) {
                    Text("RAG模式").font(.caption).bold()
                    Picker("", selection: $ragMode) {
                        Text("向量检索模式").tag("vector")
                        Text("极速检索模式").tag("colbert")
                        Text("决策判定模式").tag("decision")
                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 320)
                    Spacer(minLength: 8)
                }

                if ragMode == "vector" {
                // Embedding 行：模型类型 / 本地路径之后紧接切片设置（语义切片 +
                // 切片大小 + overlap）。优先单行；窗口极窄时把切片设置换到下一行，
                // 避免控件被裁剪。
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 8) {
                        embeddingPathControls
                        chunkControls()
                        Spacer(minLength: 8)
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 8) {
                            embeddingPathControls
                            Spacer(minLength: 8)
                        }
                        HStack(spacing: 8) {
                            chunkControls()
                            Spacer(minLength: 8)
                        }
                    }
                }

                // Reranker 行：模型类型 / 本地路径之后是「启用重排序 + TopN +
                // 候选池 + 阈值」，同样按可用宽度自动折行。二值粗筛已移到向量库行。
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 8) {
                        rerankerPathControls
                        rerankControls()
                        Spacer(minLength: 8)
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 8) {
                            rerankerPathControls
                            Spacer(minLength: 8)
                        }
                        HStack(spacing: 8) {
                            rerankControls()
                            Spacer(minLength: 8)
                        }
                    }
                }

                // 向量库行：元素类型 / Matryoshka / ANN 检索与自动构建 / 显式重建。
                // 索引状态与维护按钮并入下一行「二值 ANN」之后；
                // 窗口较窄时各自整组换行，仍归属同一组设置。
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 8) {
                        vectorStorageControls
                        annToggles()
                        binaryCoarseControls()
                        Spacer(minLength: 8)
                        vectorRebuildButtons()
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        vectorStorageControls
                        HStack(spacing: 8) {
                            annToggles()
                            binaryCoarseControls()
                            Spacer(minLength: 8)
                            vectorRebuildButtons()
                        }
                    }
                }
                // ANN 设置：二值 ANN（stage 1 自研 bann）与主 ANN 索引的
                // 状态与维护按钮（构建/续建、停止、删除）。
                Text("ANN设置").font(.caption).foregroundStyle(.secondary)

                // 二值 ANN：stage 1 优先走自研 bann，未构建时回退 bin 表暴力汉明。
                // 后接主 ANN 索引摘要与维护按钮（构建/续建、停止、删除）。
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 8) {
                        binaryANNStatusLabel
                        annStatusLabel
                        Spacer(minLength: 8)
                        annButtons()
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        binaryANNStatusLabel
                        HStack(spacing: 8) {
                            annStatusLabel
                            Spacer(minLength: 8)
                            annButtons()
                        }
                    }
                }
                // 闲时章节摘要：ANN 状态下方单独一行；文案已含「章节摘要：」前缀。
                Text(model.bookSummaryStatusText)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .help("无前台活动 3 分钟后自动逐章生成摘要：辅助模型已启用且加载时"
                        + "每 15 分钟一章；未启用/未加载时临时加载，每 5 分钟一章，"
                        + "空闲会话结束即卸载。")
                if model.vectorRebuildRequired {
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill")
                        Text("向量库需重建：\(model.vectorRebuildReason)。"
                            + "旧切片仍保留，全文检索可用；重建会按当前配置清空并全量重新同步。")
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .font(.caption2)
                    .foregroundStyle(.red)
                }
                // Weighted RRF 混合权重 + 多库判定开关 + 向量库完备状态
                rrfWeightRow
                multiZoneJudgeRow
                vectorCompletenessRow
                } // end if ragMode == vector

                if ragMode == "colbert" {
                    colbertConfigBlock
                }

                if ragMode == "decision" {
                    decisionConfigBlock
                }

                DisclosureGroup("高级设置", isExpanded: $showAdvanced) {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack(spacing: 8) {
                            Text("知识图谱融合策略").font(.caption)
                            Picker("", selection: Binding(
                                get: { model.knowledgeGraphBuilt ? retrievalStrategy : .naive },
                                set: { retrievalStrategy = $0 }
                            )) {
                                ForEach(RAGRetrievalStrategy.allCases, id: \.self) { s in
                                    Text(s.displayName).tag(s)
                                }
                            }
                            .labelsHidden()
                            .frame(width: 190)
                            .disabled(!model.knowledgeGraphBuilt)
                            Button(model.isBuildingKG ? "图谱构建中…" : "构建知识图谱") {
                                model.requestKnowledgeGraphBuild()
                            }
                            .disabled(model.isSyncing || model.isDeduplicating
                                || model.isBuildingANN || model.isBuildingKG)
                            .help("用辅助模型在已完成切片上抽取实体/关系并建立图谱；"
                                + "建成后才能选择使用 local/global/hybrid/mix 四种图谱检索策略")
                            if model.isBuildingKG {
                                Button("停止") { model.stopKnowledgeGraphBuild() }
                            }
                            Button(model.isResolvingEntities ? "实体消歧中…" : "实体消歧") {
                                model.requestEntityResolution()
                            }
                            .disabled(model.isSyncing || model.isDeduplicating
                                || model.isBuildingANN || model.isBuildingKG
                                || model.isResolvingEntities || !model.knowledgeGraphBuilt)
                            .help("用向量语义相似度对图谱内同指实体做进一步合并（全量扫描，耗时与实体数相关）；"
                                + "抽取/构建阶段只用字符全匹配，快速且不依赖向量")
                            if model.knowledgeGraphBuilt {
                                Text("已建 \(model.knowledgeGraphEntityCount) 实体 / \(model.knowledgeGraphRelationCount) 关系")
                                    .font(.caption2).foregroundStyle(.secondary)
                            } else {
                                Text("未构建（当前用 naive）").font(.caption2).foregroundStyle(.secondary)
                            }
                            Spacer(minLength: 8)
                        }
                        HStack(spacing: 8) {
                            Text("RAG MCP服务").font(.caption)
                            Picker("", selection: $ragMCPScope) {
                                Text("仅限本机").tag(false)
                                Text("局域网可用").tag(true)
                            }
                            .labelsHidden()
                            .frame(width: 104)
                            .disabled(model.ragMCPRunning)
                            Text("端口").font(.caption2).foregroundStyle(.secondary)
                            TextField("8080", text: $ragMCPPortText)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 64)
                                .disabled(model.ragMCPRunning)
                            Circle()
                                .fill(model.ragMCPRunning ? Color.green : Color.secondary.opacity(0.5))
                                .frame(width: 8, height: 8)
                            Text(model.ragMCPRunning ? "运行中" : "未启动")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .frame(width: 44, alignment: .leading)
                            Button(model.ragMCPRunning ? "停止服务" : "启动服务") {
                                toggleRAGMCPService()
                            }
                            .disabled(ragMCPStartDisabled)
                            Spacer(minLength: 8)
                        }
                        .help("把知识库 RAG 能力以 MCP 服务（Streamable HTTP）提供给其它大模型应用；"
                              + "修改监听范围或端口后需停止再启动才生效。默认仅本机 127.0.0.1:8080")
                        Divider()
                        HStack(spacing: 8) {
                            Text("自定义OCR模型").font(.caption)
                            Picker("", selection: $vlmBackend) {
                                Text("MLX").tag("mlx")
                                Text("GGUF").tag("gguf")
                                Text("ONNX").tag("onnx")
                            }
                            .labelsHidden()
                            .frame(width: 80)
                            TextField(
                                vlmBackend == "mlx"
                                    ? "模型路径（MLX 填目录）"
                                    : vlmBackend == "onnx"
                                        ? "ONNX 模型文件（或含权重的目录）"
                                        : "GGUF 模型文件路径",
                                text: $ocrModelPath
                            )
                            .textFieldStyle(.roundedBorder)
                            .frame(minWidth: 160)
                            Button("选择…") {
                                switch vlmBackend {
                                case "mlx":
                                    chooseDirectory($ocrModelPath, title: "选择 OCR 模型目录")
                                case "onnx":
                                    chooseONNX($ocrModelPath,
                                               title: "选择 ONNX 模型文件或目录")
                                default:
                                    chooseFile($ocrModelPath, title: "选择 OCR 模型文件")
                                }
                            }
                            Button("保存设置") { saveAdvancedSettings() }
                                .disabled(model.isSyncing || model.isDeduplicating
                                    || model.isBuildingANN)
                                .help("保存并应用高级设置")
                            Button("识别测试") { openOCRLab() }
                                .disabled(model.isSyncing || model.isDeduplicating
                                    || model.isBuildingANN)
                                .help("打开 OCR 识别测试窗口，用指定模型识别图片或 PDF 并对比结果")
                        }
                        if vlmBackend == "gguf" {
                            HStack(spacing: 8) {
                                Text("mmproj路径").font(.caption)
                                TextField("可选：多模态投影文件，留空自动查找模型同目录的 mmproj*.gguf",
                                          text: $ocrMMProjPath)
                                    .textFieldStyle(.roundedBorder)
                                Button("选择…") { chooseFile($ocrMMProjPath, title: "选择 mmproj 文件") }
                                Spacer(minLength: 8)
                            }
                        }
                        if vlmBackend == "onnx" {
                            HStack(spacing: 8) {
                                Text("tokenizer路径").font(.caption)
                                TextField("必填：tokenizer.json 文件，或含它的目录",
                                          text: $ocrTokenizerPath)
                                    .textFieldStyle(.roundedBorder)
                                Button("选择…") { chooseFile($ocrTokenizerPath, title: "选择 tokenizer 文件") }
                                Spacer(minLength: 8)
                            }
                        }
                        HStack(spacing: 10) {
                            Text("PDF OCR").font(.caption)
                            Picker("", selection: $pdfEngine) {
                                Text("OvisOCR3").tag(PDFEngine.ovis)
                                Text("Vision OCR").tag(PDFEngine.vision)
                                Text("混合（Vision + Ovis 表格）").tag(PDFEngine.hybrid)
                                if customOCRAvailable {
                                    Text("自定义：\(customOCRName)").tag(PDFEngine.custom)
                                }
                            }
                            .labelsHidden()
                            .frame(width: 240)
                            Text("EPUB OCR").font(.caption)
                            Picker("", selection: $epubOCREngine) {
                                Text("Apple Vision").tag(PDFEngine.vision)
                                Text("OvisOCR3").tag(PDFEngine.ovis)
                                if customOCRAvailable {
                                    Text("自定义：\(customOCRName)").tag(PDFEngine.custom)
                                }
                            }
                            .labelsHidden()
                            .frame(width: 170)
                            Toggle("图像归一化", isOn: $ocrPreprocess)
                            Toggle("Otsu 二值化", isOn: $ocrBinarize)
                            Button("转换工具") { openConverter() }
                                .help("打开转换工具窗口：把 PDF/EPUB/Office/HTML/图片/文本等批量转换为 Markdown，"
                                    + "沿用这里的 OCR 与切片前的解析设置")
                        }
                        ViewThatFits(in: .horizontal) {
                            HStack(spacing: 14) {
                                syncSettingsLabel
                                Toggle("启动时自动同步", isOn: $autoIngest)
                                    .help("启动后自动执行一次知识增量同步")
                                syncResourceControls()
                                    .font(.caption)
                                Spacer(minLength: 8)
                            }
                            VStack(alignment: .leading, spacing: 6) {
                                HStack(spacing: 14) {
                                    syncSettingsLabel
                                    Toggle("启动时自动同步", isOn: $autoIngest)
                                        .help("启动后自动执行一次知识增量同步")
                                    Spacer(minLength: 8)
                                }
                                syncResourceControls()
                                    .font(.caption)
                            }
                        }
                        HStack(spacing: 8) {
                            Text("自适应子批参数")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                            Stepper("下限：\(embeddingMinBatch)",
                                    value: $embeddingMinBatch, in: 2...64)
                                .help("自适应下调的最小值，默认 2；超过上限时按上限收口")
                            Stepper("上限：\(embeddingMaxBatch)",
                                    value: $embeddingMaxBatch, in: 2...256)
                                .help("embedding 自适应子批的最大值。实测 477 token 长块在 batch 4-16 已到吞吐平台，默认 32 兼顾短块；64 以上只增加内存与单批延迟")
                            Stepper("步进：\(embeddingBatchStep)",
                                    value: $embeddingBatchStep, in: 1...16)
                                .help("每次上探/下探的调整幅度，默认 2")
                            Stepper("启动预设：\(embeddingInitialBatch)",
                                    value: $embeddingInitialBatch, in: 2...256)
                                .help("同步开始时的初始子批，默认 16；会被夹在下限与上限之间")
                            Stepper("探测批次：\(embeddingProbeBatches)",
                                    value: $embeddingProbeBatches, in: 1...16)
                                .help("每档至少观察多少个 embedding 批后评估一次，默认 2")
                            Stepper("探测窗口：\(String(format: "%.1f", embeddingProbeSeconds))s",
                                    value: $embeddingProbeSeconds, in: 0.5...30, step: 0.5)
                                .help("每档至少观察多少秒后评估一次，默认 2s；与探测批次数满足其一即评估")
                            Stepper("内存阈值：\(embeddingMemoryLimitMB)MB",
                                    value: $embeddingMemoryLimitMB,
                                    in: 1_024...32_768, step: 512)
                                .help("进程 footprint 超过该值立即下调子批；模型单独运行峰值可达数 GB，默认 6144MB")
                            Stepper("吞吐容差：\(Int((embeddingThroughputTolerance * 100).rounded()))%",
                                    value: $embeddingThroughputTolerance,
                                    in: 0.01...0.50, step: 0.01)
                                .help("吞吐变化小于该比例视为测量噪声，不调整子批；默认 3%，实测噪声较大时可调到 8-10%")
                        }
                        .font(.caption)
                        HStack(spacing: 8) {
                            Text("embedding token 预算")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                            Stepper("下限：\(embeddingTokenBudgetMin)",
                                    value: $embeddingTokenBudgetMin,
                                    in: 512...65_536, step: 512)
                                .help("单批 token 预算的下限，默认 4096；"
                                    + "长切片由预算决定批大小，短切片仍受条数上限约束")
                            Stepper("上限：\(embeddingTokenBudgetMax)",
                                    value: $embeddingTokenBudgetMax,
                                    in: 512...65_536, step: 512)
                                .help("单批 token 预算的上限，默认 8192；"
                                    + "超过上限时按内存阈值下调，避免 padding/峰值内存过高")
                            Stepper("启动：\(embeddingTokenBudgetInitial)",
                                    value: $embeddingTokenBudgetInitial,
                                    in: 512...65_536, step: 512)
                                .help("每次同步开始时的 token 预算，默认 6144；会被夹在下限与上限之间")
                            Stepper("步进：\(embeddingTokenBudgetStep)",
                                    value: $embeddingTokenBudgetStep,
                                    in: 128...8_192, step: 128)
                                .help("每轮上探/下探的 token 预算幅度，默认 512")
                        }
                        .font(.caption)
                    }
                    .padding(.top, 8)
                }
            }
            .padding(8)
        } label: {
            Text("RAG模式配置").font(.headline)
        }
    }

    // MARK: - 工作目录配置区（多知识分区）
    private var directorySection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    Text("工作目录").font(.caption)
                    TextField("~/base-agent-workspace", text: $workspace)
                        .textFieldStyle(.roundedBorder)
                    Button("选择…") { chooseWorkspaceDirectory() }
                }
                ForEach($zoneRows) { $zone in
                    VStack(alignment: .leading, spacing: 4) {
                        // 第一行：启用 + 设为默认 + 领域名 + 源目录 + 选择
                        HStack(spacing: 6) {
                            Toggle("启用", isOn: $zone.enabled).toggleStyle(.checkbox)
                                .font(.caption2)
                            Button(zone.isDefault ? "默认" : "设为默认") {
                                setDefaultZone(zone.id)
                            }
                            .buttonStyle(.borderless).font(.caption2)
                            .disabled(zone.isDefault)
                            TextField("领域名（如 法律）", text: $zone.name)
                                .textFieldStyle(.roundedBorder)
                                .frame(maxWidth: 120)
                            TextField("源目录（留空=workspace/knowledge_base）", text: $zone.source)
                                .textFieldStyle(.roundedBorder)
                                .frame(maxWidth: .infinity)
                            Button("选择…") { chooseZoneDirectory(zone.id) }
                                .font(.caption2)
                            if !zone.isDefault {
                                Button("删除") { removeZone(zone.id) }
                                    .buttonStyle(.borderless).font(.caption2)
                            }
                        }
                        // 第二行：分区操作按钮 + 同步进度
                        // 应用配置由窗口底部统一入口负责；同步知识在配置变化时
                        // 会自行先应用，因此每行不再重复放「应用配置」。
                        HStack(spacing: 6) {
                            Button(model.zoneSyncEngineID == zone.id ? "同步中…" : "同步知识") {
                                applyAndSyncZone(zone.id)
                            }
                            .disabled(model.isSyncing)
                            Button("停止同步") { model.stopSync() }
                                .disabled(model.zoneSyncEngineID != zone.id)
                            Button("向量去重") { model.dedupeZone(source: zone.source) }
                                .disabled(model.isSyncing || ragMode == "colbert")
                                .help(ragMode == "colbert" ? "极速检索模式无向量库，无需向量去重" : "清理本知识分区内的重复向量切片")
                            if model.zoneSyncEngineID == zone.id && !model.zoneSyncStatusText.isEmpty {
                                ProgressView().controlSize(.mini)
                                Text(model.zoneSyncStatusText)
                                    .font(.caption2).foregroundStyle(.secondary)
                                    .lineLimit(1).truncationMode(.middle)
                            }
                            Spacer(minLength: 0)
                        }
                        .buttonStyle(.borderless)
                        .font(.caption2)
                    }
                    .padding(6)
                    .background(zone.isDefault ? Color.accentColor.opacity(0.08) : Color.clear)
                    .cornerRadius(4)
                }
                HStack {
                    Button("添加新知识库") { addZone() }
                        .buttonStyle(.borderless)
                    Spacer()
                }
            }
            .padding(8)
        } label: {
            Text("工作目录配置").font(.headline)
        }
    }

    /// 为指定分区选择源目录。用 id 定位行写回，避免 ForEach 临时 Binding 问题。
    private func chooseZoneDirectory(_ id: String) {
        let title = (zoneRows.first { $0.id == id }?.name ?? "知识库") + " 源目录"
        let current = zoneRows.first { $0.id == id }?.source
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if let url = runOpenPanel({ panel in
            panel.title = title
            panel.canChooseFiles = false
            panel.canChooseDirectories = true
            panel.allowsMultipleSelection = false
            if !current.isEmpty {
                panel.directoryURL = URL(fileURLWithPath: (current as NSString).expandingTildeInPath)
            }
        }), let idx = zoneRows.firstIndex(where: { $0.id == id }) {
            zoneRows[idx].source = url.path
        }
    }

    private func addZone() {
        let id = String(UUID().uuidString.prefix(8))
        zoneRows.append(ZoneRow(id: id, name: "新知识库",
                                source: "", enabled: true, isDefault: zoneRows.isEmpty))
    }

    private func setDefaultZone(_ id: String) {
        for i in zoneRows.indices { zoneRows[i].isDefault = (zoneRows[i].id == id) }
    }

    private func removeZone(_ id: String) {
        zoneRows.removeAll { $0.id == id }
        // 独立存储目录（rag_data/<hash>）保留，不误删用户数据；点「应用配置」后
        // 该分区不再挂载，需要彻底删除可手动清理对应 rag_data 子目录。
        statusMessage = "已从配置移除知识分区（索引数据保留在 rag_data，可手动清理）"
    }

    // MARK: - 极速检索模式配置（原在高级设置）
    private var colbertConfigBlock: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text("量化残差").font(.caption2).foregroundStyle(.secondary)
                Picker("", selection: $colbertQuantBits) {
                    Text("2bit").tag(2); Text("4bit").tag(4)
                }
                .labelsHidden().frame(width: 70)
                Text("粗筛候选").font(.caption2).foregroundStyle(.secondary)
                Picker("", selection: $colbertCoarseCandidates) {
                    Text("10").tag(10); Text("20").tag(20)
                    Text("50").tag(50); Text("100").tag(100)
                }
                .labelsHidden().frame(width: 64)
                Text("精选topN").font(.caption2).foregroundStyle(.secondary)
                Picker("", selection: $colbertTopN) {
                    Text("1").tag(1); Text("3").tag(3); Text("5").tag(5)
                }
                .labelsHidden().frame(width: 56)
                Toggle("使用ANN", isOn: $colbertUseANN).toggleStyle(.checkbox)
                Toggle("长度归一化", isOn: $colbertLengthNorm).toggleStyle(.checkbox)
                Text("查询侧剪枝").font(.caption2).foregroundStyle(.secondary)
                Picker("", selection: $colbertQueryPrune) {
                    Text("轻量").tag(true); Text("不剪枝").tag(false)
                }
                .labelsHidden().frame(width: 70)
                Spacer(minLength: 8)
            }
            HStack(spacing: 8) {
                // ColBERT 编码完备状态 + 手动补做，与构建按钮同一行（状态在前）。
                if !model.vectorCompletenessText.isEmpty {
                    Text(model.vectorCompletenessText)
                        .font(.caption2)
                        .foregroundStyle(model.vectorIsComplete ? Color.secondary : Color.orange)
                        .lineLimit(1)
                    if !model.vectorIsComplete && !model.vectorCompletenessText.contains("不可用") {
                        Button(model.isBackfillingVectors ? "补做中…" : "补做向量") {
                            model.requestVectorBackfill()
                        }
                        .disabled(model.isBackfillingVectors || model.isSyncing)
                        .help("把尚未编码的切片入队，后台异步补齐；补齐前仍可被关键词检索命中")
                    }
                }
                Button(model.isBuildingColbertANN ? "索引构建中…" : "构建/续建向量索引") {
                    applyAndBuildColbertANN()
                }
                .disabled(model.isBuildingColbertANN || model.isSyncing)
                if model.isBuildingColbertANN {
                    Button("停止") { model.stopColbertANNBuild() }
                }
                Text("模型 arlenecc/jina-colbert-v2-64-onnx-int8")
                    .font(.caption2).foregroundStyle(.secondary)
                if !model.colbertIndexSummary.isEmpty {
                    Text(model.colbertIndexSummary).font(.caption2).foregroundStyle(.secondary)
                }
                Spacer(minLength: 8)
            }
            // Weighted RRF 混合权重（向量粗筛 vs BM25 粗筛）+ 多库判定开关
            rrfWeightRow
            multiZoneJudgeRow
        }
    }

    // MARK: - 决策判定模式配置
    private var decisionConfigBlock: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Text("决策模型").font(.caption)
                Picker("", selection: $decisionModelKind) {
                    Text("Meteor (readout)").tag("meteor")
                    Text("Valen (MixerHead)").tag("valen")
                }
                .labelsHidden().frame(width: 150)
                .help("决策模型后端种类，影响内核决策门、RAG 决策检索与技能判定；"
                    + "Valen 路径默认指向外置 SSD 目录")
                Picker("", selection: $decisionModelBackend) {
                    Text("MLX-LM").tag("mlx-lm")
                    Text("MLX-VLM").tag("mlx-vlm")
                    Text("GGUF").tag("gguf")
                }
                .labelsHidden().frame(width: 110)
                TextField("模型路径", text: $decisionModelPath)
                    .textFieldStyle(.roundedBorder)
                Button("选择…") { chooseDirectory($decisionModelPath, title: "选择决策模型目录") }
            }
            HStack(spacing: 8) {
                Text("粗筛数量").font(.caption)
                Picker("", selection: $decisionCoarseCandidates) {
                    Text("20").tag(20); Text("50").tag(50); Text("100").tag(100)
                }
                .labelsHidden().frame(width: 80)
                Toggle("使用核心句判定", isOn: $decisionUseCoreSentences)
                    .toggleStyle(.checkbox)
                Text("核心句数").font(.caption2).foregroundStyle(.secondary)
                Picker("", selection: $decisionCoreSentenceCount) {
                    Text("3").tag(3); Text("5").tag(5)
                }
                .labelsHidden().frame(width: 56)
                .disabled(!decisionUseCoreSentences)
                Text("判定返回TopN").font(.caption2).foregroundStyle(.secondary)
                Picker("", selection: $decisionTopN) {
                    Text("3").tag(3); Text("5").tag(5); Text("10").tag(10)
                }
                .labelsHidden().frame(width: 64)
                Spacer(minLength: 8)
            }
            Text("先用字符级 BM25 从原文切片抽取核心句，再由决策模型对粗筛切片逐一判定相关性，返回得分最高的 TopN。")
                .font(.caption2).foregroundStyle(.secondary)
            if decisionModelBackend != "mlx-vlm" {
                Text("提示：PKU-KCL-Meteor 目前仅支持 MLX-VLM 后端；当前所选后端暂未实现，将回退到 MLX-VLM 加载。")
                    .font(.caption2).foregroundStyle(.orange)
            }
        }
    }

    /// Weighted RRF 权重行：向量/极速模式共用。BM25 权重 = 1 - 向量权重。
    private var rrfWeightRow: some View {
        HStack(spacing: 8) {
            Text("混合权重").font(.caption2).foregroundStyle(.secondary)
            Text("向量 \(Int((rrfVectorWeight * 100).rounded()))%")
                .font(.caption2).monospacedDigit()
            Slider(value: $rrfVectorWeight, in: 0.1...0.9, step: 0.1)
                .frame(maxWidth: 200)
            Text("BM25 \(Int(((1 - rrfVectorWeight) * 100).rounded()))%")
                .font(.caption2).monospacedDigit().foregroundStyle(.secondary)
            Spacer(minLength: 8)
        }
        .help("Weighted RRF：向量路与 BM25 路按权重融合名次（默认 7:3）")
    }

    /// 多库联合检索判定开关：向量/极速模式共用。
    private var multiZoneJudgeRow: some View {
        Toggle("多库联合检索使用判定模式", isOn: $multiZoneUseDecisionJudge)
            .font(.caption2)
            .help("启用后：选库时判定模型决定查哪些库，跨库 RRF 融合后再判定一次取 TopN；"
                + "关闭则用 FTS 探查选库 + 各库原精排（reranker/MaxSim）")
    }

    /// 向量库完备状态行：向量/极速模式共用。不完备时显示「补做向量」按钮。
    @ViewBuilder
    private var vectorCompletenessRow: some View {
        if !model.vectorCompletenessText.isEmpty {
            HStack(spacing: 8) {
                Text(model.vectorCompletenessText)
                    .font(.caption2)
                    .foregroundStyle(model.vectorIsComplete ? Color.secondary : Color.orange)
                    .lineLimit(2)
                // 库不可用（total<0）时按钮无效，不展示——避免反复点击浪费 3 秒并给出误导成功提示。
                if !model.vectorIsComplete && !model.vectorCompletenessText.contains("不可用") {
                    Button(model.isBackfillingVectors ? "补做中…" : "补做向量") {
                        model.requestVectorBackfill()
                    }
                    .disabled(model.isBackfillingVectors || model.isSyncing)
                    .help("把尚未计算向量的切片入队，后台异步补齐；补齐前这些切片仍可被全文检索命中")
                }
                Spacer(minLength: 8)
            }
        }
    }

    // MARK: - 知识同步配置的分段控件

    /// Embedding 的「模型类型 + 本地路径 + 选择…」。
    @ViewBuilder
    private var embeddingPathControls: some View {
        modelPathControls(title: "Embedding",
                          backend: $embeddingBackend,
                          path: $embeddingPath,
                          chooseTitle: "选择 Embedding 目录")
    }

    /// Reranker 的「模型类型 + 本地路径 + 选择…」。
    @ViewBuilder
    private var rerankerPathControls: some View {
        modelPathControls(title: "Reranker",
                          backend: $rerankerBackend,
                          path: $rerankerPath,
                          chooseTitle: "选择 Reranker 目录")
    }

    @ViewBuilder
    private func modelPathControls(title: String,
                                   backend: Binding<String>,
                                   path: Binding<String>,
                                   chooseTitle: String) -> some View {
        Text(title).font(.caption)
        Picker("", selection: backend) {
            ForEach(embeddingTypes, id: \.id) { type in
                Text(type.label).tag(type.id)
            }
        }
        .labelsHidden()
        .frame(width: 130)
        Text("本地路径").font(.caption)
        TextField("可选", text: path)
            .textFieldStyle(.roundedBorder)
            .frame(minWidth: 140)
            .help("留空时使用应用内置的默认模型；仅在需要自定义模型时填写路径")
        Button("选择…") {
            chooseModelPath(path, backend: backend.wrappedValue,
                            dirTitle: chooseTitle,
                            fileTitle: "选择 \(title) GGUF 文件")
        }
    }

    /// 切片设置：语义切片开关 + 切片大小（含模型窗口收口提示）+ overlap。
    @ViewBuilder
    private func chunkControls() -> some View {
        Toggle("语义切片", isOn: $semanticChunking)
            .help("开启后大段落在句间相似度低谷处断开，切片阶段需嵌入全部候选句，明显更慢；"
                + "关闭则按结构+token 边界递归切片")
        VStack(alignment: .leading, spacing: 1) {
            Stepper("切片大小：\(chunkSize) token", value: $chunkSize,
                    in: 100...4000, step: 50)
                .help("含 overlap 的总预算，全链路（切分、入库、检索呈现）统一按它计算；"
                    + "超过 embedding 推理输入上限（1024 token，模型窗口更小时取更小值）"
                    + "会自动压到该上限。纯中文约 1 字 ≈ 1 token，中英混排约 1.8 字符/token")
            if let limit = embeddingContentLimit, chunkSize > limit {
                Text("实际生效：\(limit) token（受 embedding 推理输入上限限制）")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
        }
        Stepper("overlap：\(chunkOverlap) token", value: $chunkOverlap,
                in: 0...1000, step: 10)
            .help("占切片大小的 10%，计入总预算：正文按 chunk_size − overlap 切片")
    }

    /// 重排序开关、TopN 与候选池倍数：开启后最终返回分数最高的 TopN 条。
    @ViewBuilder
    private func rerankControls() -> some View {
        Toggle("启用重排序", isOn: $rerankEnabled)
            .help("开启后先用混合检索取候选，再由 reranker 重新打分排序，"
                + "按下面的 TopN 返回最终结果；关闭则直接用 RRF 排序结果。"
                + "路径保持默认或留空时使用 App 内置 jina reranker")
        Stepper("TopN：\(rerankTopN)", value: $rerankTopN, in: 1...20)
            .help("重排序之后保留的结果数量，默认 3；候选池会按它放大后再打乱重排。"
                + "关闭重排序时按检索调用方的 top_k 返回")
        Stepper("候选池：\(rerankCandidateMultiplier)×", value: $rerankCandidateMultiplier, in: 1...5)
            .help("候选池 = max(调用方 top_k, TopN) × 倍数，默认 3；"
                + "调小可减少送进 reranker 的切片条数、降低延迟，"
                + "调大召回更稳但重排更慢")
        Stepper("阈值：\(String(format: "%.2f", rerankThreshold))",
                value: $rerankThreshold, in: 0...1, step: 0.05)
            .help("重排分数过滤阈值（0-1），默认 0.45 适配 jina；低于阈值的片段会被过滤，"
                + "全部低于阈值时回退原排序。换 bge 等 reranker 时通常需要上调")
    }

    @ViewBuilder
    private var syncSettingsLabel: some View {
        Text("同步参数设置")
            .font(.caption.weight(.semibold))
            .foregroundStyle(.secondary)
    }

    /// 同步并发 / OCR 并发：与「同步参数设置」同组，
    /// 优先与标题、自动同步开关排在同一行。MLX 内存参数已移至「Agent 模型配置」区。
    @ViewBuilder
    private func syncResourceControls() -> some View {
        Stepper("同步并发：\(ingestConcurrency)",
                value: $ingestConcurrency, in: 1...10)
            .help("同时处理的文件数，默认 10，运行时上限 6；"
                + "纯文本文件多时调高可填满等待空档")
        Stepper("OCR 并发：\(visionConcurrency)",
                value: $visionConcurrency, in: 1...8)
            .help("Apple Vision OCR 的全局并发闸门，默认 4；"
                + "实测 4 路已接近吞吐拐点，再多不一定更快")
        Toggle("校验内容指纹", isOn: $verifyFingerprint)
            .help("默认信任文件 mtime + 大小，未变化文件不再读盘采样内容指纹。"
                + "开启后每次增量同步都为同尺寸文件重新计算指纹，"
                + "可发现「保留时间戳的同尺寸改动」，代价是额外磁盘读取")
    }

    @ViewBuilder
    private func knowledgeButtons() -> some View {
        Button("应用配置") { applyKnowledgeConfig() }
            .disabled(model.isSyncing || model.isDeduplicating || model.isBuildingANN
                || model.isModelLoading || model.isRunning || model.isWarmingUp)
        Button("同步知识") { applyAndSync() }
            .help("同步期间会自动等待模型加载/预热结束")
            .disabled(model.isSyncing || model.isDeduplicating || model.isBuildingANN
                || model.isRunning)
    }

    @ViewBuilder
    private func stopAndDedupeButtons() -> some View {
        Button("停止同步") { model.stopSync() }
            .disabled(!model.isSyncing)
        Button(model.isDeduplicating ? "去重中…" : "向量去重") {
            model.requestVectorDedupe()
        }
        .help("排查并清理向量库内重复的向量，保留入库时间最早的一份")
        .disabled(model.isDeduplicating || model.isBuildingANN)
    }

    /// 向量 ANN 索引状态：分块 IVF 摘要（未构建 / 已索引切片数 / 尾块 / 需重建）。
    /// 显示名用「向量ANN」与同行的「二值 ANN」区分。
    @ViewBuilder
    private var annStatusLabel: some View {
        HStack(spacing: 6) {
            Text("向量ANN").font(.caption)
            Text(model.annStatusText)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .help("分块 IVF 索引：块内 k-means 聚类 + int8 量化按簇落盘，检索只读命中的簇，"
                    + "避免在大库上全表扫描；索引水位之后的新切片按 rowid 顺序精确补齐。"
                    + "库内切片较少时精确扫描更快，可先不构建")
        }
    }

    /// 二值 ANN（bann）状态：stage 1 的自研分块 IVF 摘要。
    @ViewBuilder
    private var binaryANNStatusLabel: some View {
        HStack(spacing: 6) {
            Text("二值 ANN").font(.caption)
            Text(model.binaryANNStatusText)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .help("二值签名的分块 IVF 索引：stage 1 优先走 bann，"
                    + "未构建或不可用时回退 bin 表暴力汉明")
        }
    }

    @ViewBuilder
    private func annButtons() -> some View {
        Button(model.isBuildingANN ? "索引维护中…" : "构建/续建向量索引") {
            model.requestANNIndexBuild()
        }
        .help("二值粗筛开启且签名覆盖完整时构建二值 ANN，否则构建主 ANN；"
            + "构建可中断，已写完的块保留")
        .disabled(model.isBuildingANN || model.isSyncing || model.isDeduplicating
            || model.isModelLoading || model.isWarmingUp || model.isRunning)
        Button("停止构建") { model.stopANNIndexBuild() }
            .disabled(!model.isBuildingANN)
        Button("删除索引") { model.requestDropANNIndex() }
            .help("删除主 ANN 与二值 ANN 索引，检索回退到精确扫描"
                + "（2100 万切片级别的大库上会明显变慢）")
            .disabled(model.isBuildingANN || model.isSyncing || model.isDeduplicating
                || model.isRunning)
    }

    // MARK: - 向量存储 / Matryoshka

    /// 当前 Embedding 路径是否指向 jina-v5（向量维度与元素类型可配置）。
    /// 空路径 + MLX 会用内置默认 jina-v5，因此同样按 V5 处理。
    private var isJinaV5Embedding: Bool {
        let lower = embeddingPath.lowercased()
        if lower.isEmpty {
            return normalizedEmbeddingType(embeddingBackend) == "mlx"
        }
        return lower.contains("jina-embeddings-v5")
            || lower.contains("jina_v5")
            || lower.contains("jina-v5")
    }

    /// 截断维选项：jina-v5 暴露模型声明的六档，其它模型保持原有三档。
    private var matryoshkaDimensionOptions: [Int] {
        isJinaV5Embedding
            ? MatryoshkaDimensions.jinaV5
            : MatryoshkaDimensions.standard
    }

    /// 向量落盘元素类型 + Matryoshka 模式/截断维。
    @ViewBuilder
    private var vectorStorageControls: some View {
        Text("向量库").font(.caption)
        Picker("", selection: $vectorElementType) {
            ForEach(VectorElementType.allCases, id: \.rawValue) { type in
                Text(vectorElementLabel(type)).tag(type)
            }
        }
        .labelsHidden()
        .frame(width: 120)
        .help("向量落盘元素类型：int8 体积是 fp32 的 1/4（默认）；"
            + "fp16 每维 2 字节（sqlite-vec 无该列类型，走精确扫描）；"
            + "bin 体积最小但只有汉明排序。")
        Text("Matryoshka").font(.caption)
        Picker("", selection: $matryoshkaMode) {
            ForEach(MatryoshkaMode.allCases, id: \.rawValue) { mode in
                Text(mode.displayName).tag(mode)
            }
        }
        .labelsHidden()
        .frame(width: 135)
        .help("截断向量后强制重新归一化，入库与查询使用同一套变换。"
            + "自动模式对 nomic / jina-v3 / jina-v5 生效；关闭时 jina-v5 输出原生 768 维。")
        Picker("", selection: $matryoshkaDim) {
            ForEach(matryoshkaDimensionOptions, id: \.self) { dimension in
                Text("\(dimension) 维").tag(dimension)
            }
        }
        .labelsHidden()
        .frame(width: 90)
        .disabled(matryoshkaMode == .off)
        .help(isJinaV5Embedding
            ? "jina-v5 仅提供模型训练过的 32 / 64 / 128 / 256 / 512 / 768 维；"
                + "默认 128，向量库最小且召回稳定。关闭 Matryoshka 时输出原生 768 维。"
            : "Matryoshka 截断维：越小向量库越小、召回精度越低；默认 128。"
                + "nomic / jina-v3 在自动模式下都按此维度截断。")
    }

    @ViewBuilder
    private func annToggles() -> some View {
        Toggle("ANN 检索", isOn: $annEnabled)
            .help("开启后检索走分块 IVF 索引；索引未构建、维度不符或尾块超预算时"
                + "自动回退精确检索。关闭后始终精确检索，大库上会明显变慢")
        Toggle("自动构建", isOn: $annAutoBuild)
            .help("同步完成后切片数达到阈值（默认 2 万）时自动构建/续建向量索引；"
                + "关闭后只能手动点「构建/续建向量索引」")
    }

    /// 二值粗筛开关与第一阶段候选数（与向量库配置同一行，重建向量库按钮之前）。
    @ViewBuilder
    private func binaryCoarseControls() -> some View {
        Toggle("二值粗筛", isOn: $binaryCoarseEnabled)
            .font(.caption2)
            .help("开启后先用二值向量召回候选，再按主向量余弦精排，"
                + "最后与全文检索 RRF 融合；可用时取代 ANN，不可用时"
                + "自动回退原流程。主存储为 BIN 时仅支持单阶段检索；"
                + "已有库需要重新同步后才会生效。")
        Stepper(value: $binaryCoarseCandidates, in: 50...1000, step: 50) {
            Text("候选 \(binaryCoarseCandidates)").font(.caption2)
        }
        .disabled(!binaryCoarseEnabled)
        .help("第一阶段二值粗筛候选数：越大召回越稳、精排读取越多；"
            + "默认 200，允许 50...1000。")
    }

    @ViewBuilder
    private func vectorRebuildButtons() -> some View {
        Button("重建向量库") { model.requestVectorRebuild() }
            .help("按当前元素类型/Matryoshka 配置重建向量表：清空旧切片、全文索引、"
                + "ANN 与二值 ANN 索引后全量重新同步；可中断，知识库原始文件不受影响")
            .disabled(model.isSyncing || model.isDeduplicating || model.isBuildingANN
                || model.isModelLoading || model.isWarmingUp || model.isRunning)
    }

    private func vectorElementLabel(_ type: VectorElementType) -> String {
        switch type {
        case .fp32: return "FP32"
        case .int8: return "INT8（推荐）"
        case .fp16: return "FP16（2 字节/维）"
        case .bin: return "BIN（最小）"
        }
    }

    /// MLX 全局内存上限的展示文本：0 表示自动。
    private var mlxMemoryLimitText: String {
        mlxMemoryLimitMB <= 0 ? "自动" : "\(mlxMemoryLimitMB)MB"
    }

    // MARK: - 参数控件

    /// 参数区：优先把数字字段与开关放在同一行；窗口较窄时拆成「数字字段」
    /// +「思考 / KV复用 / KV量化」两行，开关始终紧挨在一起且不会被裁剪。
    @ViewBuilder
    private func parameterFields(
        context: Binding<Int>,
        kvCap: Binding<Int>? = nil,
        temperature: Binding<Double>,
        topP: Binding<Double>,
        minP: Binding<Double>,
        topK: Binding<Int>,
        repetitionPenalty: Binding<Double>,
        promptBatch: Binding<Int>? = nil,
        thinking: Binding<Bool>? = nil,
        kvReuse: Binding<Bool>? = nil,
        kvInt8: Binding<Bool>? = nil,
        flashAttention: Binding<Bool>? = nil
    ) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .center, spacing: 6) {
                numberFields(
                    context: context, kvCap: kvCap, temperature: temperature,
                    topP: topP, minP: minP, topK: topK,
                    repetitionPenalty: repetitionPenalty, promptBatch: promptBatch
                )
                parameterToggles(thinking: thinking, kvReuse: kvReuse, kvInt8: kvInt8,
                                 flashAttention: flashAttention)
            }
            VStack(alignment: .leading, spacing: 3) {
                numberFields(
                    context: context, kvCap: kvCap, temperature: temperature,
                    topP: topP, minP: minP, topK: topK,
                    repetitionPenalty: repetitionPenalty, promptBatch: promptBatch
                )
                parameterToggles(thinking: thinking, kvReuse: kvReuse, kvInt8: kvInt8,
                                 flashAttention: flashAttention)
            }
        }
    }

    @ViewBuilder
    private func numberFields(
        context: Binding<Int>,
        kvCap: Binding<Int>?,
        temperature: Binding<Double>,
        topP: Binding<Double>,
        minP: Binding<Double>,
        topK: Binding<Int>,
        repetitionPenalty: Binding<Double>,
        promptBatch: Binding<Int>?
    ) -> some View {
        HStack(alignment: .center, spacing: 3) {
            numberField("Context", value: context, width: 60)
            if let kvCap {
                numberField("KV 上限", value: kvCap, width: 66)
            }
            numberField("Temp", value: temperature, width: 48)
            numberField("top_p", value: topP, width: 48)
            numberField("min_p", value: minP, width: 48)
            numberField("top_k", value: topK, width: 48)
            numberField("repeat", value: repetitionPenalty, width: 52)
            if let promptBatch {
                numberField("batch", value: promptBatch, width: 52)
            }
        }
    }

    /// 「思考 / KV复用 / KV量化 / Flash Attention」开关组；不支持的组合（nil）不渲染。
    @ViewBuilder
    private func parameterToggles(
        thinking: Binding<Bool>?,
        kvReuse: Binding<Bool>?,
        kvInt8: Binding<Bool>?,
        flashAttention: Binding<Bool>? = nil
    ) -> some View {
        if thinking != nil || kvReuse != nil || kvInt8 != nil || flashAttention != nil {
            HStack(alignment: .center, spacing: 10) {
                if let flashAttention {
                    Toggle("Flash Attention", isOn: flashAttention)
                        .toggleStyle(.checkbox)
                        .font(.caption2)
                        .fixedSize()
                        .help("默认开启：MLX 使用未量化 KV 走 Metal fused attention，"
                            + "GGUF 强制启用 Flash Attention；M3+/M4 或纯注意力模型收益更明显。"
                            + "GGUF 可与 KV量化 叠加（fa_q8_0 kernel 省显存）；"
                            + "MLX 上游限制二者只能取其一，开启时 KV量化 自动停用")
                }
                if let thinking {
                    Toggle("思考", isOn: thinking)
                        .toggleStyle(.checkbox)
                        .font(.caption2)
                        .fixedSize()
                        .help("开启后模型会先生成思考内容，显著增加延迟与 token；"
                            + "主模型默认开启，辅助模型默认关闭；"
                            + "后台事实抽取/蒸馏始终强制关闭")
                }
                if let kvReuse {
                    Toggle("KV复用", isOn: kvReuse)
                        .toggleStyle(.checkbox)
                        .font(.caption2)
                        .fixedSize()
                        .help("关闭后每轮全量重建 KV（等效禁用前缀缓存）：主模型与辅助模型各自独立")
                }
                if let kvInt8 {
                    Toggle("KV量化", isOn: kvInt8)
                        .toggleStyle(.checkbox)
                        .font(.caption2)
                        .fixedSize()
                        .help("KV cache int8 量化：KV 内存约减半，质量几乎无损；主模型与辅助模型各自独立")
                }
            }
        }
    }

    private func numberField(_ title: String, value: Binding<Int>,
                             width: CGFloat) -> some View {
        HStack(spacing: 3) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
            TextField("", value: value, format: .number)
                .textFieldStyle(.roundedBorder)
                .frame(width: width)
        }
    }

    private func numberField(_ title: String, value: Binding<Double>,
                             width: CGFloat) -> some View {
        HStack(spacing: 3) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
            TextField("", value: value, format: .number)
                .textFieldStyle(.roundedBorder)
                .frame(width: width)
        }
    }

    // MARK: - RAG MCP 服务

    /// 端口输入是否合法（非法时禁用启动按钮，避免保存坏配置）。
    private var ragMCPPortValue: Int? {
        Int(ragMCPPortText.trimmingCharacters(in: .whitespaces))
    }
    private var ragMCPStartDisabled: Bool {
        if model.ragMCPRunning { return model.isSyncing }
        return (ragMCPPortValue.flatMap { (1...65535).contains($0) } != true)
            || model.isSyncing
    }

    private func toggleRAGMCPService() {
        if model.ragMCPRunning {
            model.stopRAGMCPService()
            return
        }
        guard let port = ragMCPPortValue, (1...65535).contains(port) else { return }
        model.updateRAGMCPSettings(listenLAN: ragMCPScope, port: port)
        model.startRAGMCPService()
    }

    // MARK: - 加载 / 应用

    private func load() {
        let config = model.config
        mainMode = config.llmProvider.lowercased() == "api" ? "api" : "local"
        mainBackend = normalizedLLMType(config.localLLMBackend)
        mainPath = config.localLLMModel
        mainTokenizer = config.llmTokenizerPath
        mainContext = config.maxContextTokens
        mainKVCache = config.kvCacheMaxTokens > 0
            ? config.kvCacheMaxTokens : config.maxContextTokens
        kvInt8Enabled = config.kvCacheInt8
        flashAttentionEnabled = config.flashAttention
        mainTemperature = config.temperature
        mainTopP = config.topP
        mainMinP = config.minP
        mainTopK = config.topK
        mainRepetitionPenalty = config.repetitionPenalty
        mainTimeout = config.requestTimeout
        mainMmprojPath = config.llmMmprojPath
        maxIterations = config.maxIterations
        memoryHierarchyInSystemPrompt = config.memoryHierarchyInSystemPrompt
        mainPromptBatch = config.promptBatchSize
        mainThinking = config.localLLMEnableThinking
        mainKVReuse = config.llmKVReuse
        autoPreloadMain = config.autoPreloadMain

        baseURL = config.baseURL
        apiKey = config.apiKey
        apiModel = config.model

        auxEnabled = config.auxEnabled
        auxBackend = normalizedLLMType(config.auxLocalLLMBackend)
        auxPath = config.auxLocalLLMModel
        auxTokenizer = config.auxTokenizerPath
        auxContext = config.auxMaxContextTokens
        auxKVCache = config.auxKvCacheMaxTokens > 0
            ? config.auxKvCacheMaxTokens : config.auxMaxContextTokens
        auxTemperature = config.auxTemperature
        auxTopP = config.auxTopP
        auxMinP = config.auxMinP
        auxTopK = config.auxTopK
        auxRepetitionPenalty = config.auxRepetitionPenalty
        auxTimeout = config.auxRequestTimeout
        auxPromptBatch = config.auxPromptBatchSize
        auxThinking = config.auxLocalLLMEnableThinking
        auxKVReuse = config.auxKvReuse
        auxKVInt8Enabled = config.auxKvCacheInt8
        auxMmprojPath = config.auxMmprojPath

        workspace = config.workspace
        knowledgeBase = config.knowledgeBase
        embeddingBackend = normalizedEmbeddingType(config.localEmbeddingBackend)
        embeddingPath = config.localEmbeddingModelPath
        rerankerBackend = normalizedEmbeddingType(config.localRerankerBackend)
        rerankerPath = config.localRerankerModelPath

        ocrModelPath = config.ragOCRModelPath
        let hasCustomOCR = !config.ragOCRModelPath
            .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        pdfEngine = (!hasCustomOCR && config.ragPDFEngine == .custom)
            ? .vision : config.ragPDFEngine
        epubOCREngine = (!hasCustomOCR && config.ragEPUBOCREngine == .custom)
            ? .vision : config.ragEPUBOCREngine
        ocrMMProjPath = config.ragOCRMMProjPath
        ocrTokenizerPath = config.ragOCRTokenizerPath
        vlmBackend = config.localVLMBackend.lowercased() == "gguf" ? "gguf"
            : config.localVLMBackend.lowercased() == "onnx" ? "onnx" : "mlx"
        ocrPreprocess = config.ragOCRPreprocess
        ocrBinarize = config.ragOCRBinarize
        chunkSize = config.ragChunkSize
        chunkOverlap = config.ragChunkOverlap
        vectorElementType = config.ragVectorElementType
        matryoshkaMode = config.ragMatryoshka
        matryoshkaDim = config.ragMatryoshkaDim
        rerankEnabled = config.ragRerankEnabled
        rerankTopN = config.ragRerankTopN
        rerankCandidateMultiplier = config.ragRerankCandidateMultiplier
        rerankThreshold = config.ragRerankThreshold
        queryTimeoutSeconds = config.ragQueryTimeoutSeconds
        semanticChunking = config.ragSemanticChunking
        autoIngest = config.ragAutoIngest
        ingestConcurrency = config.ragIngestConcurrency
        visionConcurrency = config.ragVisionConcurrency
        ragMCPScope = config.ragMCPListenLan
        ragMCPPortText = String(config.ragMCPPort)
        mlxCacheLimitMB = config.mlxCacheLimitMB
        mlxMemoryLimitMB = config.mlxMemoryLimitMB
        annEnabled = config.ragANNEnabled
        annAutoBuild = config.ragANNAutoBuild
        binaryCoarseEnabled = config.ragBinaryCoarseEnabled
        binaryCoarseCandidates = config.ragBinaryCoarseCandidates
        embeddingMinBatch = config.ragEmbeddingMinBatch
        embeddingMaxBatch = config.ragEmbeddingMaxBatch
        embeddingBatchStep = config.ragEmbeddingBatchStep
        embeddingInitialBatch = config.ragEmbeddingInitialBatch
        embeddingMemoryLimitMB = config.ragEmbeddingMemoryLimitMB
        embeddingThroughputTolerance = config.ragEmbeddingThroughputTolerance
        embeddingProbeBatches = config.ragEmbeddingProbeBatches
        embeddingProbeSeconds = config.ragEmbeddingProbeSeconds
        embeddingTokenBudgetMin = config.ragEmbeddingTokenBudgetMin
        embeddingTokenBudgetMax = config.ragEmbeddingTokenBudgetMax
        embeddingTokenBudgetInitial = config.ragEmbeddingTokenBudgetInitial
        embeddingTokenBudgetStep = config.ragEmbeddingTokenBudgetStep
        verifyFingerprint = config.ragVerifyFingerprint
        colbertReaderEnabled = config.colbertReaderEnabled
        colbertQuantBits = config.colbertQuantBits
        colbertCoarseCandidates = config.colbertCoarseCandidates
        colbertTopN = config.colbertTopN
        colbertUseANN = config.colbertUseANN
        colbertLengthNorm = config.colbertLengthNorm
        colbertQueryPrune = config.colbertQueryPrune
        rrfVectorWeight = config.ragRRFVectorWeight
        multiZoneUseDecisionJudge = config.multiZoneUseDecisionJudge
        kernelDecisionGateEnabled = config.kernelDecisionGateEnabled
        kernelFinishGateThreshold = config.kernelFinishGateThreshold
        kernelPlanReviewThreshold = config.kernelPlanReviewThreshold
        skillDecisionGateThreshold = config.skillDecisionGateThreshold
        zoneRows = config.knowledgeZones.map {
            ZoneRow(id: $0.id, name: $0.name, source: $0.source,
                    enabled: $0.enabled, isDefault: $0.isDefault)
        }
        ragMode = config.ragMode
        decisionModelBackend = config.decisionModelBackend
        decisionModelKind = config.decisionModelKind
        decisionModelPath = config.decisionModelPath
        decisionCoarseCandidates = config.decisionCoarseCandidates
        decisionUseCoreSentences = config.decisionUseCoreSentences
        decisionCoreSentenceCount = config.decisionCoreSentenceCount
        decisionTopN = config.decisionTopN
        retrievalStrategy = config.ragRetrievalStrategy
        refreshEmbeddingLimit()
    }

    /// 刷新「切片实际生效值」提示：embedding 模型输入窗口小于配置的切片大小时，
    /// ingest 会把整块收口到窗口内（同一份收口逻辑在 RAGEngine 与 provider 里共用）。
    /// 这里只做一次「面板选择 → 配置」的换算，不改变当前状态提示。
    private func refreshEmbeddingLimit() {
        let saved = statusMessage
        defer { statusMessage = saved }
        // 路径还没填完整时不要去解析模型目录：既算不出上限，又会刷无效路径告警。
        let path = embeddingPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if !path.isEmpty {
            let isGGUF = AgentConfig.normalizeEmbeddingBackend(embeddingBackend) == "gguf"
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: (path as NSString).expandingTildeInPath,
                                                isDirectory: &isDirectory),
                  isDirectory.boolValue != isGGUF else {
                embeddingContentLimit = nil
                return
            }
        }
        guard let config = buildConfig() else {
            embeddingContentLimit = nil
            return
        }
        embeddingContentLimit = LocalModelFactory.shared
            .embeddingContentTokenLimit(config: config)
    }

    private func loadMainModel() {
        guard !model.isSyncing, !model.isDeduplicating, !model.isBuildingANN else {
            statusMessage = "知识同步/索引构建/向量去重进行中，暂不能加载模型"
            return
        }
        guard let config = buildConfig() else { return }
        statusMessage = "正在加载主模型…"
        model.reloadMainModel(config: config)
    }

    private func loadAuxModel() {
        guard !model.isSyncing, !model.isDeduplicating, !model.isBuildingANN else {
            statusMessage = "知识同步/索引构建/向量去重进行中，暂不能加载模型"
            return
        }
        guard let config = buildConfig() else { return }
        statusMessage = "正在加载辅助模型…"
        model.reloadAuxModel(config: config)
    }

    private func applyAPIConfig() {
        guard let config = buildConfig() else { return }
        statusMessage = "配置已应用"
        model.applyConfig(config)
    }

    /// 窗口底部统一的「应用配置」：保存窗口内全部设置并重建 Agent。
    /// 校验失败时 `buildConfig()` 已在 `statusMessage` 写入原因，这里无需额外处理。
    private func applyAllConfig() {
        guard let config = buildConfig() else { return }
        statusMessage = "全部设置已保存并应用"
        model.applyConfig(config)
    }

    private func applyKnowledgeConfig() {
        guard !model.isSyncing, !model.isDeduplicating, !model.isBuildingANN else {
            statusMessage = "知识同步/索引构建/向量去重进行中，暂不能应用配置"
            return
        }
        guard let config = buildConfig() else { return }
        statusMessage = "知识同步配置已应用并保存"
        model.applyConfig(config)
    }

    private func applyAndSync() {
        guard let config = buildConfig() else { return }
        // 知识库目录切换后旧索引不迁移：直接对新目录强制全量重建，
        // 避免增量同步对着已有 manifest 判定「无变化」而秒完成。
        let kbChanged = model.config.resolvedKnowledgeBase != config.resolvedKnowledgeBase
        statusMessage = kbChanged
            ? "知识库目录已切换为 \(config.resolvedKnowledgeBase)，将强制全量同步…"
            : "配置已应用，正在同步知识库…"
        model.applyConfig(config) {
            model.syncKnowledgeBase(force: kbChanged)
        }
    }

    /// 配置签名：用于判断窗口内设置是否与已生效配置一致，避免「点同步」时
    /// 无谓地重建 Agent（卸载/重载模型 + 取消在途同步）。
    private func configSignature(_ config: AgentConfig) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: config.export(),
                                                      options: [.sortedKeys]) else { return "" }
        return String(decoding: data, as: UTF8.self)
    }

    /// 同步指定知识分区（分区有独立 engine 与存储目录）。
    /// 只有窗口配置确实变化时才 applyConfig；否则直接同步当前 Agent 的目标分区，
    /// 避免重建 Agent 打断其它分区正在进行的同步。
    private func applyAndSyncZone(_ zoneID: String) {
        // 有分区正在同步时拒绝新的同步请求：applyConfig 会重建 Agent 并取消
        // 在途同步，导致「已同步一半的分区被中断、结果作废」。
        if model.isSyncing {
            statusMessage = "已有知识分区正在同步，请先点「停止同步」再同步其它分区"
            return
        }
        guard let config = buildConfig() else { return }
        if configSignature(config) == configSignature(model.config) {
            statusMessage = "正在同步知识分区…"
            model.syncZone(zoneID: zoneID)
            return
        }
        statusMessage = "配置已应用，正在同步知识分区…"
        model.applyConfig(config) {
            model.syncZone(zoneID: zoneID)
        }
    }

    /// 极速检索模式「构建/续建向量索引」：先把窗口内选中的 RAG 模式应用到引擎，
    /// 再构建 ColBERT ANN。否则刚切到极速模式就点构建时，引擎仍是旧模式，
    /// buildColbertANN 会因 colbertReaderEnabled=false 报「检索阅读模式未启用」。
    private func applyAndBuildColbertANN() {
        guard let config = buildConfig() else { return }
        guard config.colbertReaderEnabled else {
            statusMessage = "请先切换到极速检索模式，再构建向量索引"
            return
        }
        statusMessage = "配置已应用，正在构建向量索引…"
        model.applyConfig(config) {
            model.requestColbertANNBuild()
        }
    }

    /// 自定义 OCR 模型的显示名：MLX 取模型目录名，GGUF 取文件名去掉扩展名。
    private var customOCRName: String {
        let path = ocrModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty else { return "" }
        let url = URL(fileURLWithPath: path)
        if vlmBackend == "mlx" {
            return url.lastPathComponent
        }
        let name = url.deletingPathExtension().lastPathComponent
        return name.isEmpty ? url.lastPathComponent : name
    }

    private var customOCRAvailable: Bool {
        !customOCRName.isEmpty
    }

    /// KV 复用 / 量化只对 MLX-LM 与 GGUF 生效：
    /// ONNX 自动复用 past_key_values，MLX-VLM 每轮重建 prompt，均不消费这两个开关。
    private func supportsKVToggles(_ backend: String) -> Bool {
        backend == "mlx-lm" || backend == "gguf"
    }

    /// 保存并应用高级设置（包含自定义 OCR 模型、PDF/EPUB OCR 引擎等）。
    private func saveAdvancedSettings() {
        guard !model.isSyncing, !model.isDeduplicating, !model.isBuildingANN else {
            statusMessage = "知识同步/索引构建/向量去重进行中，暂不能保存设置"
            return
        }
        if !customOCRAvailable {
            if pdfEngine == .custom { pdfEngine = .vision }
            if epubOCREngine == .custom { epubOCREngine = .vision }
        }
        guard let config = buildConfig() else { return }
        statusMessage = "高级设置已保存并应用"
        model.applyConfig(config)
    }

    /// 打开识别测试窗口：把当前面板上的 OCR 设置（含尚未保存的修改）带过去。
    private func openOCRLab() {
        var config = model.config
        let path = ocrModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if !path.isEmpty {
            config.ragOCRModelName = customOCRName
        }
        config.ragOCRModelPath = path
        config.ragOCRMMProjPath = ocrMMProjPath.trimmingCharacters(in: .whitespacesAndNewlines)
        config.ragOCRTokenizerPath = ocrTokenizerPath.trimmingCharacters(in: .whitespacesAndNewlines)
        config.localVLMBackend = vlmBackend
        config.ragOCRPreprocess = ocrPreprocess
        config.ragOCRBinarize = ocrBinarize
        config.ragPDFEngine = pdfEngine
        config.ragEPUBOCREngine = epubOCREngine
        model.ocrLabConfig = config
        openWindow(id: "ocr-lab")
    }

    /// 打开转换工具窗口：使用当前面板上的解析/OCR 设置快照（含尚未保存的修改），
    /// 只影响这一窗口内的转换参数，不写回 config.json、不重建 Agent。
    private func openConverter() {
        var config = model.config
        let path = ocrModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if !path.isEmpty {
            config.ragOCRModelName = customOCRName
        }
        config.ragOCRModelPath = path
        config.ragOCRMMProjPath = ocrMMProjPath.trimmingCharacters(in: .whitespacesAndNewlines)
        config.ragOCRTokenizerPath = ocrTokenizerPath.trimmingCharacters(in: .whitespacesAndNewlines)
        config.localVLMBackend = vlmBackend
        config.ragOCRPreprocess = ocrPreprocess
        config.ragOCRBinarize = ocrBinarize
        config.ragPDFEngine = pdfEngine
        config.ragEPUBOCREngine = epubOCREngine
        // 快照里若指定了自定义 OCR 却没有模型路径，回退 Apple Vision，
        // 避免转换窗口因空模型路径直接失败。
        if !customOCRAvailable {
            if config.ragPDFEngine == .custom { config.ragPDFEngine = .vision }
            if config.ragEPUBOCREngine == .custom { config.ragEPUBOCREngine = .vision }
        } else {
            config.ragOCRModelName = customOCRName
        }
        model.converterConfig = config
        openWindow(id: "markdown-converter")
    }

    private func buildConfig() -> AgentConfig? {
        let mb = AgentConfig.normalizeLLMBackend(mainBackend)
        let ab = AgentConfig.normalizeLLMBackend(auxBackend)
        let supportedLLM = ["mlx-lm", "mlx-vlm", "gguf", "onnx"]
        if mainMode == "local", !supportedLLM.contains(mb) {
            statusMessage = "主模型类型 \(mainBackend) 尚未接入，请选择 MLX-LM / MLX-VLM / GGUF / ONNX"
            return nil
        }
        if !supportedLLM.contains(ab) {
            statusMessage = "辅助模型类型 \(auxBackend) 尚未接入，请选择 MLX-LM / MLX-VLM / GGUF / ONNX"
            return nil
        }
        if mainMode == "api" {
            guard !baseURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                statusMessage = "API 模式需要填写 Base URL"
                return nil
            }
            guard !apiModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                statusMessage = "API 模式需要选择或填写模型"
                return nil
            }
        }
        let eb = AgentConfig.normalizeEmbeddingBackend(embeddingBackend)
        guard eb == "onnx" || eb == "fastembed" || eb == "mlx" || eb == "gguf" else {
            statusMessage = "Embedding 类型 \(embeddingBackend) 当前版本尚未接入，请选择 ONNX / FastEmbed / MLX / GGUF"
            return nil
        }
        // jina-v5 与其它模型一样尊重用户选择的元素类型（默认 int8）。
        // fp16 走普通表 + 精确扫描，四种类型都可以直接应用。
        let effectiveElementType = vectorElementType
        if let reason = effectiveElementType.unsupportedReason {
            statusMessage = reason
            return nil
        }
        let rb = AgentConfig.normalizeEmbeddingBackend(rerankerBackend)
        if rerankEnabled, rb != "onnx", rb != "fastembed", rb != "mlx", rb != "gguf" {
            statusMessage = "Reranker 类型 \(rerankerBackend) 当前版本尚未接入，请选择 ONNX / FastEmbed / MLX / GGUF"
            return nil
        }

        var config = model.config
        config.llmProvider = mainMode
        config.localLLMBackend = AgentConfig.normalizeLLMBackend(mainBackend)
        config.localLLMModel = mainPath.trimmingCharacters(in: .whitespacesAndNewlines)
        config.llmTokenizerPath = mainTokenizer.trimmingCharacters(in: .whitespacesAndNewlines)
        config.llmMmprojPath = mainMmprojPath.trimmingCharacters(in: .whitespacesAndNewlines)
        config.maxContextTokens = max(8192, mainContext)
        config.kvCacheMaxTokens = max(0, mainKVCache)
        config.kvCacheInt8 = kvInt8Enabled
        config.flashAttention = flashAttentionEnabled
        config.temperature = clamp(mainTemperature, 0, 2)
        config.topP = clamp(mainTopP, 0, 1)
        config.minP = clamp(mainMinP, 0, 1)
        config.topK = max(0, mainTopK)
        config.repetitionPenalty = max(0, mainRepetitionPenalty)
        config.requestTimeout = max(10, mainTimeout)
        config.maxIterations = min(max(1, maxIterations), 50)
        config.memoryHierarchyInSystemPrompt = memoryHierarchyInSystemPrompt
        config.mlxCacheLimitMB = min(max(256, mlxCacheLimitMB), 2_048)
        config.mlxMemoryLimitMB = max(0, mlxMemoryLimitMB)
        config.promptBatchSize = max(0, mainPromptBatch)
        config.localLLMEnableThinking = mainThinking
        config.llmKVReuse = mainKVReuse
        config.autoPreloadMain = autoPreloadMain

        if mainMode == "api" {
            config.baseURL = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
            config.apiKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
            config.model = apiModel.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        config.auxEnabled = auxEnabled
        config.auxLLMProvider = "local"
        config.auxLocalLLMBackend = AgentConfig.normalizeLLMBackend(auxBackend)
        config.auxLocalLLMModel = auxPath.trimmingCharacters(in: .whitespacesAndNewlines)
        config.auxTokenizerPath = auxTokenizer.trimmingCharacters(in: .whitespacesAndNewlines)
        config.auxMmprojPath = auxMmprojPath.trimmingCharacters(in: .whitespacesAndNewlines)
        config.auxMaxContextTokens = max(8192, auxContext)
        config.auxKvCacheMaxTokens = max(0, auxKVCache)
        config.auxTemperature = clamp(auxTemperature, 0, 2)
        config.auxTopP = clamp(auxTopP, 0, 1)
        config.auxMinP = clamp(auxMinP, 0, 1)
        config.auxTopK = max(0, auxTopK)
        config.auxRepetitionPenalty = max(0, auxRepetitionPenalty)
        config.auxRequestTimeout = max(10, auxTimeout)
        config.auxPromptBatchSize = max(0, auxPromptBatch)
        config.auxLocalLLMEnableThinking = auxThinking
        config.auxKvReuse = auxKVReuse
        config.auxKvCacheInt8 = auxKVInt8Enabled

        config.workspace = (workspace.trimmingCharacters(in: .whitespacesAndNewlines) as NSString)
            .expandingTildeInPath
        config.knowledgeBase = knowledgeBase.trimmingCharacters(in: .whitespacesAndNewlines)

        config.embeddingProvider = "local"
        config.localEmbeddingBackend = eb
        config.localEmbeddingModelPath = embeddingPath.trimmingCharacters(in: .whitespacesAndNewlines)
        // 模型名不再由用户编辑：填了本地路径时从目录名派生；
        // 未填路径则保留配置文件里的既有名字（默认 jina-v5 int4）。
        if !config.localEmbeddingModelPath.isEmpty {
            let derived = URL(fileURLWithPath: config.localEmbeddingModelPath).lastPathComponent
            config.localEmbeddingModel = derived
            config.ragEmbeddingModel = derived
        }

        config.rerankerProvider = "local"
        config.localRerankerBackend = rb
        config.localRerankerModelPath = rerankerPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if !config.localRerankerModelPath.isEmpty {
            let derived = URL(fileURLWithPath: config.localRerankerModelPath).lastPathComponent
            config.localRerankerModel = derived
            config.ragRerankModel = derived
        }

        let ocrPath = ocrModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        config.ragPDFEngine = (pdfEngine == .custom && !customOCRAvailable)
            ? .vision : pdfEngine
        config.ragEPUBOCREngine = (epubOCREngine == .custom && !customOCRAvailable)
            ? .vision : epubOCREngine
        if !ocrPath.isEmpty {
            // 模型名不再由用户填写：MLX 取目录名，GGUF 取文件名去掉扩展名。
            config.ragOCRModelName = customOCRName
        }
        config.ragOCRModelPath = ocrPath
        config.ragOCRMMProjPath = ocrMMProjPath.trimmingCharacters(in: .whitespacesAndNewlines)
        config.ragOCRTokenizerPath = ocrTokenizerPath.trimmingCharacters(in: .whitespacesAndNewlines)
        config.localVLMBackend = vlmBackend
        config.ragOCRPreprocess = ocrPreprocess
        config.ragOCRBinarize = ocrBinarize
        config.ragChunkSize = max(100, chunkSize)
        config.ragChunkOverlap = min(max(0, chunkOverlap), config.ragChunkSize / 2)
        config.ragRerankEnabled = rerankEnabled
        config.ragRerankTopN = min(max(1, rerankTopN), 20)
        config.ragRerankCandidateMultiplier = min(max(1, rerankCandidateMultiplier), 5)
        config.ragRerankThreshold = min(max(0, rerankThreshold), 1)
        config.ragQueryTimeoutSeconds = min(max(0, queryTimeoutSeconds), 300)
        config.ragSemanticChunking = semanticChunking
        config.ragAutoIngest = autoIngest
        config.ragIngestConcurrency = min(max(1, ingestConcurrency), 10)
        config.ragVisionConcurrency = min(max(1, visionConcurrency), 8)
        config.ragVectorElementType = effectiveElementType
        config.ragMatryoshka = matryoshkaMode
        config.ragMatryoshkaDim = matryoshkaDimensionOptions.contains(matryoshkaDim)
            ? matryoshkaDim
            : MatryoshkaDimensions.defaultValue
        config.ragANNEnabled = annEnabled
        config.ragANNAutoBuild = annAutoBuild
        config.ragBinaryCoarseEnabled = binaryCoarseEnabled
        config.ragBinaryCoarseCandidates = min(max(50, binaryCoarseCandidates), 1000)
        config.ragEmbeddingMaxBatch = min(max(2, embeddingMaxBatch), 256)
        config.ragEmbeddingMinBatch = min(max(2, embeddingMinBatch), 64)
        config.ragEmbeddingMinBatch = min(config.ragEmbeddingMinBatch,
                                          config.ragEmbeddingMaxBatch)
        config.ragEmbeddingInitialBatch = min(max(config.ragEmbeddingMinBatch,
                                                  embeddingInitialBatch),
                                              config.ragEmbeddingMaxBatch)
        config.ragEmbeddingBatchStep = min(max(1, embeddingBatchStep), 16)
        config.ragEmbeddingMemoryLimitMB = min(max(1_024, embeddingMemoryLimitMB), 32_768)
        config.ragEmbeddingThroughputTolerance = min(max(0.01, embeddingThroughputTolerance),
                                                      0.5)
        config.ragEmbeddingProbeBatches = min(max(1, embeddingProbeBatches), 16)
        config.ragEmbeddingProbeSeconds = min(max(0.5, embeddingProbeSeconds), 30)
        config.ragEmbeddingTokenBudgetMin = min(max(512, embeddingTokenBudgetMin), 65_536)
        config.ragEmbeddingTokenBudgetMax = min(max(config.ragEmbeddingTokenBudgetMin,
                                                    embeddingTokenBudgetMax), 65_536)
        config.ragEmbeddingTokenBudgetInitial = min(max(config.ragEmbeddingTokenBudgetMin,
                                                        embeddingTokenBudgetInitial),
                                                    config.ragEmbeddingTokenBudgetMax)
        config.ragEmbeddingTokenBudgetStep = min(max(128, embeddingTokenBudgetStep), 8_192)
        config.ragVerifyFingerprint = verifyFingerprint
        config.ragMode = ["vector", "colbert", "decision"].contains(ragMode) ? ragMode : "vector"
        config.colbertReaderEnabled = (config.ragMode == "colbert")
        config.decisionModelBackend = ["mlx-lm", "mlx-vlm", "gguf"].contains(decisionModelBackend)
            ? decisionModelBackend : "mlx-vlm"
        config.decisionModelKind = (decisionModelKind == "valen") ? "valen" : "meteor"
        config.decisionModelPath = decisionModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        config.decisionCoarseCandidates = [20, 50, 100].contains(decisionCoarseCandidates)
            ? decisionCoarseCandidates : 50
        config.decisionUseCoreSentences = decisionUseCoreSentences
        config.decisionCoreSentenceCount = [3, 5].contains(decisionCoreSentenceCount)
            ? decisionCoreSentenceCount : 3
        config.decisionTopN = [3, 5, 10].contains(decisionTopN) ? decisionTopN : 5
        config.colbertQuantBits = colbertQuantBits == 2 ? 2 : 4
        config.colbertCoarseCandidates = [10, 20, 50, 100].contains(colbertCoarseCandidates)
            ? colbertCoarseCandidates : 50
        config.colbertTopN = [1, 3, 5].contains(colbertTopN) ? colbertTopN : 3
        config.colbertUseANN = colbertUseANN
        config.colbertLengthNorm = colbertLengthNorm
        config.colbertQueryPrune = colbertQueryPrune
        config.ragRRFVectorWeight = min(max(rrfVectorWeight, 0.1), 0.9)
        config.multiZoneUseDecisionJudge = multiZoneUseDecisionJudge
        config.kernelDecisionGateEnabled = kernelDecisionGateEnabled
        config.kernelFinishGateThreshold = kernelFinishGateThreshold
        config.kernelPlanReviewThreshold = kernelPlanReviewThreshold
        config.skillDecisionGateThreshold = skillDecisionGateThreshold
        config.knowledgeZones = zoneRows.map {
            KnowledgeZone(id: $0.id, name: $0.name, source: $0.source,
                          enabled: $0.enabled, isDefault: $0.isDefault)
        }
        // 默认分区的 source 同步到 knowledgeBase（兼容单库字段）。
        if let def = config.knowledgeZones.first(where: { $0.isDefault }) {
            config.knowledgeBase = def.source
        }
        // 图谱未建成（或已失效）时只允许 naive，避免保存一个不可用的策略。
        config.ragRetrievalStrategy = model.knowledgeGraphBuilt ? retrievalStrategy : .naive

        if mainMode == "local", !config.localLLMModel.isEmpty,
           config.localLLMModel != model.config.localLLMModel {
            statusMessage = "已应用本地模型路径"
        }
        return config
    }

    private var apiModelChoices: [String] {
        var choices = apiModels
        let current = apiModel.trimmingCharacters(in: .whitespacesAndNewlines)
        if !current.isEmpty, !choices.contains(current) {
            choices.insert(current, at: 0)
        }
        return choices
    }

    private func fetchModels() {
        let base = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !base.isEmpty else {
            statusMessage = "请先填写 Base URL"
            return
        }
        fetchingModels = true
        statusMessage = "正在获取模型列表…"
        let client = OpenAILLMClient(baseURL: base,
                                     apiKey: apiKey.trimmingCharacters(in: .whitespacesAndNewlines),
                                     model: apiModel)
        Task {
            do {
                let ids = try await client.listModels()
                client.close()
                await MainActor.run {
                    fetchingModels = false
                    apiModels = ids
                    if apiModel.isEmpty, let first = ids.first { apiModel = first }
                    statusMessage = ids.isEmpty
                        ? "接口没有返回模型"
                        : "已获取 \(ids.count) 个模型"
                }
            } catch {
                client.close()
                await MainActor.run {
                    fetchingModels = false
                    statusMessage = "获取模型失败：\(error.localizedDescription)"
                }
            }
        }
    }

    private func chooseWorkspaceDirectory() {
        let current = workspace.trimmingCharacters(in: .whitespacesAndNewlines)
        if let url = runOpenPanel({ panel in
            panel.title = "选择工作目录"
            panel.canChooseFiles = false
            panel.canChooseDirectories = true
            panel.allowsMultipleSelection = false
            if !current.isEmpty {
                panel.directoryURL = URL(fileURLWithPath: (current as NSString).expandingTildeInPath)
            }
        }) { workspace = url.path }
    }

    /// 统一弹出 NSOpenPanel：先激活 App，否则设置窗口非前台时 modal 面板
    /// 会被挡在后面/不出现在最前，表现为「点击选择没反应」。
    @discardableResult
    private func runOpenPanel(_ configure: (NSOpenPanel) -> Void) -> URL? {
        let panel = NSOpenPanel()
        configure(panel)
        NSApp.activate(ignoringOtherApps: true)
        return panel.runModal() == .OK ? panel.url : nil
    }

    private func chooseDirectory(_ binding: Binding<String>, title: String) {
        let current = binding.wrappedValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if let url = runOpenPanel({ panel in
            panel.title = title
            panel.canChooseFiles = false
            panel.canChooseDirectories = true
            panel.allowsMultipleSelection = false
            if !current.isEmpty {
                panel.directoryURL = URL(fileURLWithPath: (current as NSString).expandingTildeInPath)
            }
        }) { binding.wrappedValue = url.path }
    }

    private func chooseFile(_ binding: Binding<String>, title: String) {
        let current = binding.wrappedValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if let url = runOpenPanel({ panel in
            panel.title = title
            panel.canChooseFiles = true
            panel.canChooseDirectories = false
            panel.allowsMultipleSelection = false
            if !current.isEmpty {
                panel.directoryURL = URL(fileURLWithPath: (current as NSString).expandingTildeInPath)
                    .deletingLastPathComponent()
            }
        }) { binding.wrappedValue = url.path }
    }

    /// GGUF 模型是单个文件：限制只能选 .gguf 文件。
    private func chooseGGUF(_ binding: Binding<String>, title: String) {
        let current = binding.wrappedValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if let url = runOpenPanel({ panel in
            panel.title = title
            panel.canChooseFiles = true
            panel.canChooseDirectories = false
            panel.allowsMultipleSelection = false
            if #available(macOS 11.0, *) {
                panel.allowedContentTypes = [UTType(filenameExtension: "gguf")]
                    .compactMap { $0 }
            }
            if !current.isEmpty {
                panel.directoryURL = URL(fileURLWithPath: (current as NSString).expandingTildeInPath)
                    .deletingLastPathComponent()
            }
        }) { binding.wrappedValue = url.path }
    }

    /// ONNX 模型既可以是单个 `.onnx` 文件，也可以是含 onnx 权重的目录：
    /// 允许用户二选一（优先选文件，选到目录则交给运行时按约定搜索）。
    private func chooseONNX(_ binding: Binding<String>, title: String) {
        let current = binding.wrappedValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if let url = runOpenPanel({ panel in
            panel.title = title
            panel.canChooseFiles = true
            panel.canChooseDirectories = true
            panel.allowsMultipleSelection = false
            if #available(macOS 11.0, *) {
                panel.allowedContentTypes = [UTType(filenameExtension: "onnx")]
                    .compactMap { $0 }
            }
            if !current.isEmpty {
                panel.directoryURL = URL(fileURLWithPath: (current as NSString).expandingTildeInPath)
                    .deletingLastPathComponent()
            }
        }) { binding.wrappedValue = url.path }
    }

    /// 按 backend 选择模型路径：
    /// - GGUF 是单个文件；
    /// - ONNX 既可是文件也可是目录（优先选文件，选目录则运行时按约定搜索）；
    /// - 其它（MLX / MLX-VLM / FastEmbed）是目录。
    private func chooseModelPath(_ binding: Binding<String>,
                                 backend: String,
                                 dirTitle: String,
                                 fileTitle: String) {
        switch backend {
        case "gguf":
            chooseGGUF(binding, title: fileTitle)
        case "onnx":
            chooseONNX(binding, title: "选择 ONNX 模型文件或目录")
        default:
            chooseDirectory(binding, title: dirTitle)
        }
    }

    private func clamp(_ value: Double, _ lower: Double, _ upper: Double) -> Double {
        min(max(value, lower), upper)
    }

    /// 旧配置可能包含 openai / llama_cpp 等值；Picker 只能选中已声明的选项。
    private func normalizedLLMType(_ raw: String) -> String {
        let value = AgentConfig.normalizeLLMBackend(raw)
        return llmTypes.contains { $0.id == value } ? value : "mlx-lm"
    }

    private func normalizedEmbeddingType(_ raw: String) -> String {
        let value = AgentConfig.normalizeEmbeddingBackend(raw)
        return embeddingTypes.contains { $0.id == value } ? value : "onnx"
    }
}
