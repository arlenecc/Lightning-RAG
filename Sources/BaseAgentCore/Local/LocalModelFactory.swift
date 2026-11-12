import Foundation
import CoreGraphics
import MLX

/// 本地模型工厂。
///
/// - `provider == "local"`：创建 ONNX / MLX 实现，并按模型路径缓存单例；
/// - `provider == "api"`：Embedding / Reranker / VLM 返回 nil，LLM 返回 OpenAI 兼容客户端；
/// - 本地模型缺失时返回「明确抛错」的实现，而不是静默降级。
public final class LocalModelFactory: @unchecked Sendable {
    public static let shared = LocalModelFactory()

    private let embeddings = ModelCache<EmbeddingProvider>()
    private let rerankers = ModelCache<RerankProvider>()
    private let llms = ModelCache<LLMService>()
    private let vlms = ModelCache<VLMService>()
    /// 同步 / 转换 / OCR 测试共用的 OCR 任务计数；最后一个任务结束时释放权重。
    /// 闭包只在真正结束时才访问单例，避免初始化期间递归访问 `shared`。
    private let ocrLifecycle = OCRModelLifecycle {
        await LocalModelFactory.shared.unloadOCRWeights()
    }

    private init() {
        Self.configureMetallibPath()
        // MLX 默认缓存可达统一内存 75%，长文 OCR/对话容易把 footprint 顶到很高。
        // 16GB 机器上实测 1024MB 的 buffer cache 对 decode 提速约 5-8%
        //（512MB 时 prefill 大 buffer 会把 decode 复用的小 buffer 挤出，导致
        // 每 token 频繁重新分配 Metal buffer），超出部分仍会回收给系统。
        setCacheLimitMB(1024)
        setMemoryLimitMB(0)
    }

    /// 打包成 .app 后，MLX 只会在 `<bundleURL>/mlx-swift_Cmlx.bundle/…` 与已加载的
    /// bundle 里找 `default.metallib`，而 SwiftPM 资源实际位于 Contents/Resources。
    /// 找不到金属库会让首次求值崩溃，因此这里显式指定路径。
    private static func configureMetallibPath() {
        let relative = "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
        let candidates: [String] = [
            Bundle.main.resourceURL?.appendingPathComponent(relative).path,
            Bundle.main.bundleURL.appendingPathComponent(relative).path,
            Bundle.main.bundleURL.deletingLastPathComponent()
                .appendingPathComponent("Resources/\(relative)").path,
        ].compactMap { $0 }

        for path in candidates where FileManager.default.fileExists(atPath: path) {
            MLX.GPU.metallib = URL(fileURLWithPath: path)
            AgentLog.info("[metal] metallib=\(path)")
            return
        }
        // 裸可执行文件（未打包）依赖 MLX 的同目录查找，无需显式设置。
    }

    /// 设置 MLX buffer cache 上限（MB）。只限制空闲 buffer，不限制权重与 KV。
    /// 允许试到 2GB；实测 embedding 的复用 buffer 工作集约 1.2GB，1GB 已接近
    /// 上限，2GB 基本没有额外提速，只是多留空闲 buffer。
    public func setCacheLimitMB(_ megabytes: Int) {
        let clamped = max(256, min(2_048, megabytes))
        MLX.Memory.cacheLimit = clamped * 1_024 * 1_024
    }

    /// 设置 MLX 全局内存硬上限（MB）。这是权重 + KV + buffer 的总闸门，
    /// 用于兜底防止泄漏把系统内存吃光；0 表示按 min(物理内存 65%, 9.6GB)
    /// 自动取值，避免大内存机器把上限放得过高。
    public func setMemoryLimitMB(_ megabytes: Int) {
        let physicalMB = Int(ProcessInfo.processInfo.physicalMemory / 1_024 / 1_024)
        let automatic = max(1_024, min(physicalMB * 65 / 100, 9_830))
        let requested = megabytes > 0 ? megabytes : automatic
        let clamped = max(1_024, min(requested, max(1_024, physicalMB - 2_048)))
        MLX.Memory.memoryLimit = clamped * 1_024 * 1_024
        if clamped != configuredMemoryLimitMB {
            configuredMemoryLimitMB = clamped
            AgentLog.info("[memory] MLX 全局内存上限 \(clamped)MB（物理 \(physicalMB)MB）")
        }
    }

    private var configuredMemoryLimitMB = -1

    /// 配置变更 / 模型切换时释放缓存，避免旧 MLX/ONNX 实例常驻内存。
    public func clearCaches() {
        embeddings.clear()
        rerankers.clear()
        llms.clear()
        vlms.clear()
    }

    /// 只释放 LLM 缓存；Embedding/Reranker 保留。
    ///
    /// 不在卸载路径同步调用 `MLX.Memory.clearCache()`：可能仍有 asyncEval
    /// 在 GPU 线程执行，与 command encoder 释放并发会触发 SIGSEGV。
    public func unloadLLMCaches() {
        llms.clear()
    }

    /// 等待 MLX 默认 stream 上的 GPU 工作全部完成。
    ///
    /// 退出流程与释放 MLX 全局状态（cache/memory limit、模型容器）前必须调用，
    /// 否则未完成的 asyncEval 会与 command buffer 的 commit/销毁并发。
    public func synchronizeGPU() {
        MLX.Stream.gpu.synchronize()
    }

    /// 退出流程专用：等待所有按需模型的权重真正卸载，再清空实例缓存。
    public func unloadAllForTermination() async {
        for provider in embeddings.allValues() { await provider.unload() }
        for reranker in rerankers.allValues() { await reranker.unload() }
        for service in vlms.allValues() { await service.unload() }
        embeddings.clear()
        rerankers.clear()
        vlms.clear()
        synchronizeGPU()
    }

    /// 释放全部 OCR VLM 权重。provider 对象保留，下次识别时按需重载。
    ///
    /// 主模型、辅助模型、Embedding 与 Reranker 在启动预加载后常驻，不在这里
    /// 卸载：Embedding 每次重建约 2.4s、Reranker ONNX session 约 10~15s，
    /// 回收省下的内存会被下一次首问的加载延迟抵消。
    public func unloadOCRWeights() async {
        let vlmValues = vlms.allValues()
        for service in vlmValues {
            await service.unload()
        }
        let activeMB = MLX.Memory.activeMemory / 1_024 / 1_024
        let cacheMB = MLX.Memory.cacheMemory / 1_024 / 1_024
        AgentLog.info("[memory] 释放 OCR \(vlmValues.count) 个模型，"
            + "MLX active=\(activeMB)MB cache=\(cacheMB)MB")
    }

    /// 空闲回收兜底：没有在途 OCR 任务时才释放，避免与同步/转换并发。
    public func unloadIdleOCRWeights() async {
        await ocrLifecycle.unloadIfIdle()
    }

    /// 标记一个 OCR 任务开始（同步 / Markdown 转换 / OCR 测试）。
    public func beginOCRJob() async {
        await ocrLifecycle.begin()
    }

    /// 标记任务结束；最后一个任务结束时立即释放两个 Ovis 实例的权重。
    public func endOCRJob() async {
        await ocrLifecycle.end()
    }

    // MARK: - Embedding

    /// 本地 Embedding provider；API 模式返回 nil。
    public func makeEmbedding(config: AgentConfig) -> EmbeddingProvider? {
        let provider = config.embeddingProvider.lowercased()
        guard provider == "local" else {
            return nil
        }

        let backend = AgentConfig.normalizeEmbeddingBackend(config.localEmbeddingBackend)
        guard backend == "onnx" || backend == "fastembed" || backend == "mlx" || backend == "gguf" else {
            let value = config.localEmbeddingModelPath.isEmpty
                ? config.localEmbeddingModel : config.localEmbeddingModelPath
            return FailingEmbeddingProvider(
                identifier: "local:\(backend):\(value)",
                error: LocalModelError.invalidConfiguration(
                    "当前版本 Embedding 支持 ONNX / FastEmbed / MLX / GGUF，尚未接入 \(backend)")
            )
        }

        let name = config.localEmbeddingModel.isEmpty
            ? config.ragEmbeddingModel
            : config.localEmbeddingModel
        // MLX 后端默认用随 App 分发的 jina-v5 int4（768→128 维、8192 窗口、
        // 前缀即任务类型），与 ONNX 量化名区分；显式配置了
        // local_embedding_model / _path 时以用户配置为准。
        let effectiveName = (backend == "mlx"
                             && config.localEmbeddingModel.isEmpty
                             && config.localEmbeddingModelPath.isEmpty)
            ? "jina-embeddings-v5-text-nano-mlx-int4"
            : name
        let value = config.localEmbeddingModelPath.isEmpty
            ? effectiveName
            : config.localEmbeddingModelPath
        let baseKey = "embedding:\(backend):\(value)"

        // 先解析目录：失败不进缓存——目录可能尚未就绪或依赖运行中才设置的
        // 环境变量（如测试的 BASE_AGENT_MODELS_DIR），缓存失败实例会让
        // 后续有效配置永远拿到 Failing。
        let directory: URL
        do {
            if backend == "gguf" {
                let file = try GGUFFileResolver.resolve(
                    kind: .embedding, value: value, config: config)
                return embeddings.value(for: baseKey) {
                    LlamaCppEmbeddingProvider(modelPath: file.path, modelName: value)
                }
            }
            directory = try LocalModelRegistry.resolveONNXDirectory(
                kind: .embedding,
                value: value,
                config: config,
                // 非 MLX 后端不得命中 MLX 权重别名（目录里没有 ONNX 权重）。
                aliasFilter: backend == "mlx"
                    ? nil : { !$0.hasSuffix("-mlx") }
            )
        } catch {
            AgentLog.warning("本地 Embedding 不可用：\(error.localizedDescription)")
            return FailingEmbeddingProvider(
                identifier: "local:\(backend):\(value)",
                error: error
            )
        }

        // jina-v5 的 Matryoshka 维度由设置决定：缓存键必须纳入维度，
        // 否则切换维度后会复用旧维度的 provider。其它模型的 key 保持不变。
        let isV5 = JinaV5Profile.isJinaV5(directory: directory)
        let v5Dimension = JinaV5Profile.outputDimension(
            mode: config.ragMatryoshka, configured: config.ragMatryoshkaDim)
        let key = isV5 ? "\(baseKey)#v5dim=\(v5Dimension)" : baseKey

        if backend == "mlx" {
            // 实测多实例只是共享同一块 GPU，吞吐无收益且内存翻倍，
            // 因此固定单实例；并发由 AdaptiveEmbeddingScheduler 串行。
            return embeddings.value(for: key) {
                // jina-v5 / jina-v3 都不在 MLXEmbedders 的工厂里，走自定义架构。
                if isV5 {
                    return JinaV5MLXEmbeddingProvider(
                        directory: directory, modelName: value,
                        dimension: v5Dimension)
                }
                if JinaV3Profile.isJinaV3(directory: directory) {
                    return JinaV3MLXEmbeddingProvider(
                        directory: directory, modelName: value)
                }
                return MLXEmbeddingProvider(directory: directory, modelName: value)
            }
        }
        return embeddings.value(for: key) {
            ONNXEmbeddingProvider(
                directory: directory, modelName: value,
                v5Dimension: v5Dimension)
        }
    }

    /// 当前配置下 embedding 的「单条内容 token 上限」；不支持精确计数的后端返回 nil。
    ///
    /// 与 ingest 的切片收口同源（同一个 provider 缓存实例），设置界面用它显示
    /// 「切片大小实际生效值」，避免面板显示长切片、实际按模型窗口短切。
    public func embeddingContentTokenLimit(config: AgentConfig) -> Int? {
        (makeEmbedding(config: config) as? EmbeddingTokenCounting)?.maxContentTokens
    }

    // MARK: - Reranker

    /// 本地 Reranker provider；API 模式返回 nil。
    public func makeReranker(config: AgentConfig) -> RerankProvider? {
        let provider = config.rerankerProvider.lowercased()
        guard provider == "local" else {
            return nil
        }

        let backend = AgentConfig.normalizeEmbeddingBackend(config.localRerankerBackend)
        guard backend == "onnx" || backend == "fastembed" || backend == "mlx" || backend == "gguf" else {
            let value = config.localRerankerModelPath.isEmpty
                ? config.localRerankerModel : config.localRerankerModelPath
            return FailingRerankProvider(
                identifier: "local:\(backend):\(value)",
                error: LocalModelError.invalidConfiguration(
                    "当前版本 Reranker 支持 ONNX / FastEmbed / MLX / GGUF，尚未接入 \(backend)")
            )
        }

        let name = config.localRerankerModel.isEmpty
            ? config.ragRerankModel
            : config.localRerankerModel
        let value = config.localRerankerModelPath.isEmpty
            ? name
            : config.localRerankerModelPath
        let key = "reranker:\(backend):\(value)"

        // 与 Embedding 一致：目录解析失败不进缓存。模型可能正在下载或刚补到
        // 新位置，旧实现把 Failing 实例缓存进 LRU，文件就位后仍会一直失败
        //（要重启或改配置才恢复）；这里失败后下次调用重新解析即可自愈。
        if backend == "gguf" {
            do {
                let file = try GGUFFileResolver.resolve(
                    kind: .reranker, value: value, config: config)
                return rerankers.value(for: key) {
                    LlamaCppReranker(modelPath: file.path)
                }
            } catch {
                AgentLog.warning("本地 Reranker 不可用：\(error.localizedDescription)")
                return FailingRerankProvider(
                    identifier: "local:\(backend):\(value)",
                    error: error
                )
            }
        }
        let directory: URL
        do {
            directory = try LocalModelRegistry.resolveONNXDirectory(
                kind: .reranker,
                value: value,
                config: config
            )
        } catch {
            AgentLog.warning("本地 Reranker 不可用：\(error.localizedDescription)")
            return FailingRerankProvider(
                identifier: "local:\(backend):\(value)",
                error: error
            )
        }
        return rerankers.value(for: key) {
            if backend == "mlx" {
                return MLXReranker(directory: directory, modelName: value)
            }
            return ONNXReranker(
                directory: directory,
                modelName: value)
        }
    }

    // MARK: - LLM

    /// 主 LLM。API 模式复用 `OpenAILLMClient`，本地模式使用 MLX。
    public func makeLLM(config: AgentConfig) -> LLMService {
        switch config.llmProvider.lowercased() {
        case "api":
            return OpenAILLMClient(config: config)
        case "local", "":
            break
        default:
            return FailingLLMService(
                identifier: "local:mlx:unsupported",
                error: LocalModelError.invalidConfiguration(
                    "不支持的 llm_provider：\(config.llmProvider)（仅支持 api / local）"
                )
            )
        }

        let backend = AgentConfig.normalizeLLMBackend(config.localLLMBackend)
        guard backend == "mlx-lm" || backend == "mlx-vlm"
                || backend == "gguf" || backend == "onnx" else {
            return FailingLLMService(
                identifier: "local:\(backend):unsupported",
                error: LocalModelError.invalidConfiguration(
                    "主模型支持 MLX-LM / MLX-VLM / GGUF / ONNX，尚未接入 \(backend)"
                )
            )
        }

        let factor = config.localLLMModel.isEmpty
            ? (LocalModelRegistry.defaultName(for: .llm) ?? "")
            : config.localLLMModel
        if backend == "onnx" {
            let tokenizerPath = config.llmTokenizerPath
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let tokenizerDirectory = tokenizerPath.isEmpty
                ? nil : URL(fileURLWithPath: tokenizerPath)
            let key = "llm-onnx:\(factor):\(config.effectiveMainContextTokens):"
                + "\(config.maxTokens):\(tokenizerPath)"
            return llms.value(for: key) {
                do {
                    let directory = try LocalModelRegistry.resolveONNXDirectory(
                        kind: .llm, value: factor, config: config)
                    return ONNXLLMService(
                        directory: directory,
                        tokenizerDirectory: tokenizerDirectory,
                        modelName: factor,
                        contextWindow: config.effectiveMainContextTokens,
                        maxTokens: config.maxTokens,
                        topP: config.topP,
                        minP: config.minP,
                        topK: config.topK,
                        repetitionPenalty: config.repetitionPenalty,
                        enableThinking: config.localLLMEnableThinking
                    )
                } catch {
                    AgentLog.warning("本地 ONNX LLM 不可用：\(error.localizedDescription)")
                    return FailingLLMService(
                        identifier: "local:onnx:\(factor)",
                        error: error
                    )
                }
            }
        }
        if backend == "mlx-vlm" {
            let key = "llm-vlm:\(factor):\(config.effectiveMainContextTokens):\(config.maxTokens)"
            return llms.value(for: key) {
                do {
                    let directory = try LocalModelRegistry.resolveDirectory(
                        kind: .llm, value: factor, config: config)
                    return MLXVLMLLMService(
                        directory: directory,
                        modelName: factor,
                        maxTokens: config.maxTokens,
                        topP: config.topP,
                        minP: config.minP,
                        topK: config.topK,
                        repetitionPenalty: config.repetitionPenalty,
                        enableThinking: config.localLLMEnableThinking,
                        contextWindow: config.effectiveMainContextTokens
                    )
                } catch {
                    AgentLog.warning("本地 MLX-VLM LLM 不可用：\(error.localizedDescription)")
                    return FailingLLMService(
                        identifier: "local:mlx-vlm:\(factor)",
                        error: error
                    )
                }
            }
        }
        if backend == "gguf" {
            // GGUF 的 FA 可与 q8_0 KV 叠加，故用原始 kvCacheInt8（不套 effective）。
            let key = "llm-gguf:\(factor):\(config.effectiveMainContextTokens):\(config.maxTokens):\(config.llmKVReuse):\(config.kvCacheInt8):\(config.flashAttention)"
            return llms.value(for: key) {
                do {
                    let file = try GGUFFileResolver.resolve(
                        kind: .llm, value: factor, config: config)
                    return LlamaCppLLMService(
                        modelPath: file.path,
                        contextWindow: config.effectiveMainContextTokens,
                        maxTokens: config.maxTokens,
                        topP: config.topP,
                        minP: config.minP,
                        topK: config.topK,
                        repetitionPenalty: config.repetitionPenalty,
                        enableThinking: config.localLLMEnableThinking,
                        gpuLayers: config.llamaGpuLayers,
                        mmprojPath: GGUFFileResolver.resolveMMProj(
                            modelPath: file.path,
                            explicit: config.llmMmprojPath.isEmpty
                                ? nil : config.llmMmprojPath),
                        kvReuse: config.llmKVReuse,
                        kvInt8: config.kvCacheInt8,
                        flashAttention: config.flashAttention
                    )
                } catch {
                    AgentLog.warning("本地 GGUF LLM 不可用：\(error.localizedDescription)")
                    return FailingLLMService(
                        identifier: "local:gguf:\(factor)",
                        error: error
                    )
                }
            }
        }
        let key = llmKey(
            prefix: "llm",
            value: factor,
            maxTokens: config.maxTokens,
            topP: config.topP,
            minP: config.minP,
            topK: config.topK,
            repetitionPenalty: config.repetitionPenalty,
            enableThinking: config.localLLMEnableThinking,
            contextWindow: config.effectiveMainContextTokens,
            promptBatchSize: config.promptBatchSize,
            kvReuse: config.llmKVReuse,
            kvInt8: config.effectiveMainKVInt8
        )

        return llms.value(for: key) {
            do {
                let directory = try LocalModelRegistry.resolveDirectory(
                    kind: .llm,
                    value: factor,
                    config: config
                )
                return MLXLLMService(
                    directory: directory,
                    modelName: factor,
                    maxTokens: config.maxTokens,
                    topP: config.topP,
                    minP: config.minP,
                    topK: config.topK,
                    repetitionPenalty: config.repetitionPenalty,
                    enableThinking: config.localLLMEnableThinking,
                    contextWindow: config.effectiveMainContextTokens,
                    promptBatchSize: config.promptBatchSize,
                    kvReuse: config.llmKVReuse,
                    kvInt8: config.effectiveMainKVInt8
                )
            } catch {
                AgentLog.warning("本地 LLM 不可用：\(error.localizedDescription)")
                return FailingLLMService(
                    identifier: "local:mlx:\(factor)",
                    error: error
                )
            }
        }
    }

    /// 辅助 LLM。参数使用 aux_* 配置，模型加载逻辑与主 LLM 一致。
    public func makeAuxLLM(config: AgentConfig) -> LLMService {
        makeAuxLLM(config: config, sharedCache: true)
    }

    /// 临时辅助 LLM：不写入共享缓存，空闲会话结束后可独立卸载。
    public func makeTransientAuxLLM(config: AgentConfig) -> LLMService {
        makeAuxLLM(config: config, sharedCache: false)
    }

    private func makeAuxLLM(config: AgentConfig, sharedCache: Bool) -> LLMService {
        switch config.auxLLMProvider.lowercased() {
        case "api":
            return OpenAILLMClient(
                baseURL: config.baseURL,
                apiKey: config.apiKey,
                model: config.model,
                timeout: config.auxRequestTimeout,
                maxTokens: config.auxMaxTokens,
                topP: config.auxTopP,
                minP: config.auxMinP,
                topK: config.auxTopK,
                repetitionPenalty: config.auxRepetitionPenalty,
                contextWindow: config.effectiveAuxContextTokens
            )
        case "local", "":
            break
        default:
            return FailingLLMService(
                identifier: "local:mlx:unsupported",
                error: LocalModelError.invalidConfiguration(
                    "不支持的 aux_llm_provider：\(config.auxLLMProvider)（仅支持 api / local）"
                )
            )
        }

        let backend = AgentConfig.normalizeLLMBackend(config.auxLocalLLMBackend)
        guard backend == "mlx-lm" || backend == "mlx-vlm"
                || backend == "gguf" || backend == "onnx" else {
            return FailingLLMService(
                identifier: "local:\(backend):unsupported",
                error: LocalModelError.invalidConfiguration(
                    "辅助模型支持 MLX-LM / MLX-VLM / GGUF / ONNX，尚未接入 \(backend)"
                )
            )
        }

        let value = config.auxLocalLLMModel.isEmpty
            ? (LocalModelRegistry.defaultName(for: .aux) ?? "")
            : config.auxLocalLLMModel
        if backend == "onnx" {
            let tokenizerPath = config.auxTokenizerPath
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let tokenizerDirectory = tokenizerPath.isEmpty
                ? nil : URL(fileURLWithPath: tokenizerPath)
            let key = "aux-onnx:\(value):\(config.effectiveAuxContextTokens):"
                + "\(config.auxMaxTokens):\(tokenizerPath)"
            return auxValue(key: key, sharedCache: sharedCache) {
                do {
                    let directory = try LocalModelRegistry.resolveONNXDirectory(
                        kind: .aux, value: value, config: config)
                    return ONNXLLMService(
                        directory: directory,
                        tokenizerDirectory: tokenizerDirectory,
                        modelName: value,
                        contextWindow: config.effectiveAuxContextTokens,
                        maxTokens: config.auxMaxTokens,
                        topP: config.auxTopP,
                        minP: config.auxMinP,
                        topK: config.auxTopK,
                        repetitionPenalty: config.auxRepetitionPenalty,
                        enableThinking: config.auxLocalLLMEnableThinking
                    )
                } catch {
                    AgentLog.warning("本地 ONNX 辅助模型不可用：\(error.localizedDescription)")
                    return FailingLLMService(
                        identifier: "local:onnx:\(value)",
                        error: error
                    )
                }
            }
        }
        if backend == "mlx-vlm" {
            let key = "aux-vlm:\(value):\(config.effectiveAuxContextTokens):\(config.auxMaxTokens)"
            return auxValue(key: key, sharedCache: sharedCache) {
                do {
                    let directory = try LocalModelRegistry.resolveDirectory(
                        kind: .aux, value: value, config: config)
                    return MLXVLMLLMService(
                        directory: directory,
                        modelName: value,
                        maxTokens: config.auxMaxTokens,
                        topP: config.auxTopP,
                        minP: config.auxMinP,
                        topK: config.auxTopK,
                        repetitionPenalty: config.auxRepetitionPenalty,
                        enableThinking: config.auxLocalLLMEnableThinking,
                        contextWindow: config.effectiveAuxContextTokens
                    )
                } catch {
                    AgentLog.warning("本地 MLX-VLM 辅助模型不可用：\(error.localizedDescription)")
                    return FailingLLMService(
                        identifier: "local:mlx-vlm:\(value)",
                        error: error
                    )
                }
            }
        }
        if backend == "gguf" {
            let key = "aux-gguf:\(value):\(config.effectiveAuxContextTokens):\(config.auxMaxTokens):\(config.auxKvReuse):\(config.auxKvCacheInt8):\(config.flashAttention)"
            return auxValue(key: key, sharedCache: sharedCache) {
                do {
                    let file = try GGUFFileResolver.resolve(
                        kind: .aux, value: value, config: config)
                    return LlamaCppLLMService(
                        modelPath: file.path,
                        contextWindow: config.effectiveAuxContextTokens,
                        maxTokens: config.auxMaxTokens,
                        topP: config.auxTopP,
                        minP: config.auxMinP,
                        topK: config.auxTopK,
                        repetitionPenalty: config.auxRepetitionPenalty,
                        enableThinking: config.auxLocalLLMEnableThinking,
                        gpuLayers: config.llamaGpuLayers,
                        mmprojPath: GGUFFileResolver.resolveMMProj(
                            modelPath: file.path,
                            explicit: config.auxMmprojPath.isEmpty
                                ? nil : config.auxMmprojPath),
                        kvReuse: config.auxKvReuse,
                        kvInt8: config.auxKvCacheInt8,
                        flashAttention: config.flashAttention
                    )
                } catch {
                    AgentLog.warning("本地 GGUF 辅助模型不可用：\(error.localizedDescription)")
                    return FailingLLMService(
                        identifier: "local:gguf:\(value)",
                        error: error
                    )
                }
            }
        }
        let key = llmKey(
            prefix: "aux",
            value: value,
            maxTokens: config.auxMaxTokens,
            topP: config.auxTopP,
            minP: config.auxMinP,
            topK: config.auxTopK,
            repetitionPenalty: config.auxRepetitionPenalty,
            enableThinking: config.auxLocalLLMEnableThinking,
            contextWindow: config.effectiveAuxContextTokens,
            promptBatchSize: config.auxPromptBatchSize,
            kvReuse: config.auxKvReuse,
            kvInt8: config.effectiveAuxKVInt8
        )

        return auxValue(key: key, sharedCache: sharedCache) {
            do {
                let directory = try LocalModelRegistry.resolveDirectory(
                    kind: .aux,
                    value: value,
                    config: config
                )
                return MLXLLMService(
                    directory: directory,
                    modelName: value,
                    maxTokens: config.auxMaxTokens,
                    topP: config.auxTopP,
                    minP: config.auxMinP,
                    topK: config.auxTopK,
                    repetitionPenalty: config.auxRepetitionPenalty,
                    enableThinking: config.auxLocalLLMEnableThinking,
                    contextWindow: config.effectiveAuxContextTokens,
                    promptBatchSize: config.auxPromptBatchSize,
                    kvReuse: config.auxKvReuse,
                    kvInt8: config.effectiveAuxKVInt8
                )
            } catch {
                AgentLog.warning("本地辅助 LLM 不可用：\(error.localizedDescription)")
                return FailingLLMService(
                    identifier: "local:mlx:\(value)",
                    error: error
                )
            }
        }
    }

    /// 常驻辅助模型走共享缓存；临时辅助模型每次独立构建，避免占用缓存实例。
    private func auxValue(key: String, sharedCache: Bool,
                          build: () -> LLMService) -> LLMService {
        sharedCache ? llms.value(for: key, build: build) : build()
    }

    // MARK: - VLM

    /// 本地 OCR VLM；API 模式或未启用时返回 nil。
    public func makeVLM(config: AgentConfig) -> VLMService? {
        guard config.vlmProvider.lowercased() == "local" else {
            return nil
        }
        return makeOCRModel(config: config)
    }

    /// PDF Ovis 引擎专用：无论 `vlm_provider` 如何配置，都按
    /// `rag_ocr_model_path` / `rag_ocr_model_name` 加载 OvisOCR3。
    /// `instance` 用于多实例池：不同序号返回独立加载的实例。
    public func makeOCRModel(config: AgentConfig, instance: Int = 0) -> VLMService {
        let backend = config.localVLMBackend.lowercased()
        guard backend == "mlx" || backend == "gguf" || backend == "onnx" || backend.isEmpty else {
            return FailingVLMService(
                identifier: "local:\(backend)-vlm:unsupported",
                error: LocalModelError.invalidConfiguration(
                    "本地 VLM 支持 mlx / gguf / onnx 后端，当前配置为 \(config.localVLMBackend)"
                )
            )
        }

        let value: String
        if !config.ragOCRModelPath.isEmpty {
            value = config.ragOCRModelPath
        } else if !config.localVLMModel.isEmpty {
            value = config.localVLMModel
        } else {
            value = config.ragOCRModelName
        }
        guard !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return FailingVLMService(
                identifier: "local:\(backend)-vlm:unconfigured",
                error: LocalModelError.invalidConfiguration(
                    "未配置 rag_ocr_model_path / rag_ocr_model_name"
                )
            )
        }

        let expanded = (value as NSString).expandingTildeInPath
        let looksGGUF = backend == "gguf"
            || value.lowercased().hasSuffix(".gguf")
            || GGUFFileResolver.ggufFile(atPath: expanded) != nil
        let directURL = URL(fileURLWithPath: expanded)
        let looksONNX = backend == "onnx"
            || ((try? LocalModelRegistry.resolveONNXModel(
                in: directURL, kind: .vlm)) != nil)
        let key = "vlm:\(backend):\(value)#\(instance)"

        return vlms.value(for: key) {
            do {
                if looksGGUF {
                    let file = try GGUFFileResolver.resolve(
                        kind: .vlm, value: value, config: config)
                    let mmproj = GGUFFileResolver.resolveMMProj(
                        modelPath: file.path,
                        explicit: config.ragOCRMMProjPath.isEmpty
                            ? nil : config.ragOCRMMProjPath
                    )
                    return LlamaCppVLMService(modelPath: file.path,
                                             mmprojPath: mmproj,
                                             gpuLayers: config.llamaGpuLayers)
                }
                if looksONNX {
                    let directory = try LocalModelRegistry.resolveONNXDirectory(
                        kind: .vlm, value: value, config: config)
                    // ONNX OCR 主力是 rapidocr 式（检测+识别）；单一合并 VLM 图作为罕见后备。
                    if ONNXOCRService.isONNXOCRDirectory(directory) {
                        return ONNXOCRService(modelDirectory: directory, modelName: value)
                    }
                    let tokenizerPath = config.ragOCRTokenizerPath
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    let tokenizerDirectory = tokenizerPath.isEmpty
                        ? nil : URL(fileURLWithPath: tokenizerPath)
                    return try ONNXVLMService(
                        modelLocation: directory,
                        tokenizerDirectory: tokenizerDirectory,
                        modelName: value,
                        contextWindow: config.effectiveMainContextTokens,
                        defaultMaxTokens: 2_048)
                }
                let directory = try LocalModelRegistry.resolveDirectory(
                    kind: .vlm,
                    value: value,
                    config: config
                )
                return OvisVLMService(directory: directory, modelName: value)
            } catch {
                AgentLog.warning("本地 VLM 不可用：\(error.localizedDescription)")
                return FailingVLMService(
                    identifier: "local:\(backend)-vlm:\(value)",
                    error: error
                )
            }
        }
    }

    /// RAG OCR 专用：按 `rag_ovis_instances` 返回单实例或多实例池。
    /// 池内实例独立懒加载、各自串行，池级并行提高扫描页吞吐；
    /// 第二实例仅在真正并发时才占内存。
    public func makeRAGOCRVLM(config: AgentConfig) -> VLMService {
        let instances = max(1, min(2, config.ragOvisInstances))
        guard instances > 1 else { return makeOCRModel(config: config) }
        return VLMServicePool(size: instances) { index in
            LocalModelFactory.shared.makeOCRModel(config: config, instance: index)
        }
    }

    // MARK: - 缓存与错误实现

    private func llmKey(
        prefix: String,
        value: String,
        maxTokens: Int,
        topP: Double,
        minP: Double,
        topK: Int,
        repetitionPenalty: Double,
        enableThinking: Bool,
        contextWindow: Int,
        promptBatchSize: Int,
        kvReuse: Bool,
        kvInt8: Bool
    ) -> String {
        "\(prefix):\(value):\(maxTokens):\(topP):\(minP):\(topK):\(repetitionPenalty):\(enableThinking):\(contextWindow):\(promptBatchSize):\(kvReuse):\(kvInt8)"
    }
}

// MARK: - 缓存

/// OvisOCR3 多实例池：实例独立懒加载、各自串行，池级借还并行。
/// 实现 VLMService，对 PDF/EPUB 解析器透明——解析器按批提交任务，
/// 实际并发由池内实例数限制。
public final class VLMServicePool: VLMService, @unchecked Sendable {
    public let identifier: String
    /// 全部实例可用才视为可用；配置错误（Failing 实例）时调用方统一跳过。
    public var isAvailable: Bool { services.allSatisfy { $0.isAvailable } }

    private let services: [VLMService]
    private let gate: AsyncSemaphore
    private let lock = NSLock()
    private var idle: [VLMService]

    init(size: Int, factory: (Int) -> VLMService) {
        let built = (0..<max(1, size)).map(factory)
        self.services = built
        self.identifier = "vlm-pool(\(built.count)):\(built.first?.identifier ?? "empty")"
        self.gate = AsyncSemaphore(value: built.count)
        self.idle = built
    }

    public func recognize(image: CGImage, prompt: String, maxTokens: Int) async throws -> String {
        try await gate.withPermit {
            lock.lock()
            let service = idle.removeLast()
            lock.unlock()
            defer {
                lock.lock()
                idle.append(service)
                lock.unlock()
            }
            return try await service.recognize(image: image,
                                               prompt: prompt,
                                               maxTokens: maxTokens)
        }
    }

    /// 卸载池内每个实例的权重；保留池本身，下次识别时按需重载。
    ///
    /// 关键修复：默认的 `VLMService.unload()` 是空实现，若不复写，`LocalModelFactory
    /// .unloadOCRWeights` 在 `endOCRJob` 时遍历缓存实例调用 `unload()` 对池无效，
    /// 池内 Ovis 实例权重会一直常驻（约 8GB），跨多次转换/同步持续叠加。
    public func unload() async {
        for service in services {
            await service.unload()
        }
    }
}

/// 线程安全的模型实例缓存。构建过程在锁外执行，避免加载大模型时阻塞其他查询。
private final class ModelCache<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String: Value] = [:]

    func value(for key: String, build: () -> Value) -> Value {
        lock.lock()
        if let existing = storage[key] {
            lock.unlock()
            return existing
        }
        lock.unlock()

        let built = build()

        lock.lock()
        if let existing = storage[key] {
            lock.unlock()
            return existing
        }
        storage[key] = built
        lock.unlock()
        return built
    }

    func clear() {
        lock.lock()
        storage.removeAll()
        lock.unlock()
    }

    /// 当前缓存的全部实例（用于空闲卸载权重，不删除实例本身）。
    func allValues() -> [Value] {
        lock.lock()
        defer { lock.unlock() }
        return Array(storage.values)
    }
}

// MARK: - 明确抛错的占位实现

/// 本地 Embedding 不可用时的占位实现。错误会延迟到调用时抛出，便于上层统一处理。
private final class FailingEmbeddingProvider: EmbeddingProvider, @unchecked Sendable {
    let identifier: String
    let error: Error
    var dimension: Int { 768 }

    init(identifier: String, error: Error) {
        self.identifier = identifier
        self.error = error
    }

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        throw error
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        throw error
    }
}

private final class FailingRerankProvider: RerankProvider, @unchecked Sendable {
    let identifier: String
    let error: Error

    init(identifier: String, error: Error) {
        self.identifier = identifier
        self.error = error
    }

    func rerank(query: String, documents: [String]) async throws -> [Double] {
        throw error
    }
}

private final class FailingLLMService: LLMService, PreloadableLLMService,
    @unchecked Sendable {
    let identifier: String
    let error: Error

    init(identifier: String, error: Error) {
        self.identifier = identifier
        self.error = error
    }

    func preload() async throws {
        throw error
    }

    func chatStream(
        messages: [ChatMessage],
        tools: [JSONValue],
        temperature: Double
    ) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: error)
        }
    }
}

private final class FailingVLMService: VLMService, @unchecked Sendable {
    let identifier: String
    let error: Error
    var isAvailable: Bool { false }

    init(identifier: String, error: Error) {
        self.identifier = identifier
        self.error = error
    }

    func recognize(image: CGImage, prompt: String, maxTokens: Int) async throws -> String {
        throw error
    }
}
