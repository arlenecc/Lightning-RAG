import Foundation
import CryptoKit
import Darwin

// MARK: - MCP

public struct MCPServerConfig: Codable, Sendable, Equatable {
    public var name: String
    public var command: String
    public var args: [String]
    public var env: [String: String]
    /// 非空时走 Streamable HTTP 直连该端点（忽略 command/args/env）。
    public var url: String?

    public init(name: String, command: String, args: [String] = [],
                env: [String: String] = [:], url: String? = nil) {
        self.name = name
        self.command = command
        self.args = args
        self.env = env
        self.url = url
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        // HTTP 直连形态可以没有 command；旧配置一律有。
        command = try container.decodeIfPresent(String.self, forKey: .command) ?? ""
        args = try container.decodeIfPresent([String].self, forKey: .args) ?? []
        env = try container.decodeIfPresent([String: String].self, forKey: .env) ?? [:]
        url = try container.decodeIfPresent(String.self, forKey: .url)
    }
}

// MARK: - PDF 解析引擎

/// PDF 文档解析引擎。docling / marker 已移除：
/// - ovis：OvisOCR3 视觉语言模型逐页转 Markdown
/// - vision：Apple Vision 逐页 OCR（默认，最快，无表格结构）
/// - hybrid：Vision 为主，仅表格页交给 OvisOCR3（扫描书籍最快）
/// - custom：使用「高级 RAG / OCR 设置」中自定义的 OCR 模型逐页转 Markdown
public enum PDFEngine: String, Codable, Sendable, CaseIterable {
    case ovis
    case vision
    case hybrid
    case custom

    public init(stored: String) {
        let v = stored.lowercased()
        if v == "vision" || v == "pdfkit_vision" || v == "apple_vision" {
            self = .vision
        } else if v == "ovis" || v == "ovisocr3" {
            self = .ovis
        } else if v == "hybrid" || v == "mixed" || v == "auto" {
            self = .hybrid
        } else if v == "custom" || v == "custom_vlm" {
            self = .custom
        } else {
            // 未显式配置 ovis / hybrid / custom 时一律先走 Apple Vision，
            // 避免旧值（如 docling）把 OCR 默默带回纯 Ovis 路径。
            self = .vision
        }
    }
}

// MARK: - RAG 检索策略

/// 知识图谱加持的 RAG 检索策略（默认 naive = 图谱能力加入前的原始实现）。
///
/// - `naive`：不用图谱，只做原文切片召回（历史默认）。
/// - `local`：从命中实体拉邻居边与关联实体，适合问具体细节。
/// - `global`：从命中关系反查两边实体及度数，适合跨文档宏观主题/总结。
/// - `hybrid`：实体 + 关系两个子图轮询交织合并，适合综合问题。
/// - `mix`：实体 + 关系 + 原文切片全做，图谱语境之上保留原文兜底。
public enum RAGRetrievalStrategy: String, Codable, Sendable, CaseIterable {
    case local
    case global
    case hybrid
    case mix
    case naive

    public init(stored: String) {
        switch stored.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "local": self = .local
        case "global": self = .global
        case "hybrid": self = .hybrid
        case "mix": self = .mix
        default: self = .naive
        }
    }

    /// 是否依赖知识图谱（只有图谱构建完成后这四种才可选）。
    public var usesKnowledgeGraph: Bool { self != .naive }

    public var displayName: String {
        switch self {
        case .local: return "local（实体/邻居）"
        case .global: return "global（关系/宏观）"
        case .hybrid: return "hybrid（实体+关系）"
        case .mix: return "mix（实体+关系+原文）"
        case .naive: return "naive（原文召回）"
        }
    }
}

// MARK: - AgentConfig

/// 运行时配置，持久化到 `~/.base-agent/config.json`。
/// 字段命名对齐原 Python `AgentConfig`（JSON 中为 snake_case）。
public struct AgentConfig: Sendable {
    public var baseURL: String = "https://api.openai.com/v1"
    public var apiKey: String = ""
    public var model: String = "gpt-4o-mini"
    public var temperature: Double = 0.7
    public var maxTokens: Int = 32768
    public var topP: Double = 0.95
    public var minP: Double = 0.05
    public var topK: Int = 20
    public var repetitionPenalty: Double = 1.0
    public var workspace: String = expandPath("~/base-agent-workspace")
    public var knowledgeBase: String = ""
    public var requestTimeout: Double = 120.0
    public var maxIterations: Int = 15
    /// 预留（当前未接线）：历史窗口实际由 kernelRawTurns / kernelConsolidateEvery
    /// / kernelMaxContextTokens 决定，本字段只做配置兼容，不参与裁剪决策。
    public var maxHistory: Int = 50
    public var maxContextTokens: Int = 32768
    /// prompt prefill 每次前向的 token 上限（0 = 使用 MLX 默认值）。
    public var promptBatchSize: Int = 1024
    /// MLX 空闲 buffer cache 上限（MB）。只限制可复用 buffer，不含模型权重与 KV；
    /// 运行时夹在 256...2048，默认 1024：实测 1GB 让 decode 快 5–8%，
    /// 且仍未超过 embedding 的复用工作集。
    public var mlxCacheLimitMB: Int = 1_024
    /// MLX 全局内存硬上限（MB，含权重/KV/buffer）；0 = 自动取
    /// min(物理内存 65%, 9.6GB)。
    public var mlxMemoryLimitMB: Int = 0
    /// 主模型 KV cache 的 token 硬上限；0 表示跟随 maxContextTokens。
    /// 32k 约对应 1GB KV（Qwen3.5 只有 1/4 层是全注意力）。
    public var kvCacheMaxTokens: Int = 32_768
    /// 主 LLM 跨轮 KV/前缀复用开关。开启时 system+tools 等不变前缀的 KV
    /// 跨轮保留，每轮只 prefill 新增消息；关闭时每轮全量重建。
    /// 同时作用于 MLX 与 GGUF (llama.cpp) 后端；辅助模型用 `auxKvReuse`。
    public var llmKVReuse: Bool = true
    /// 辅助模型 KV cache 的 token 硬上限；0 表示跟随 auxMaxContextTokens。
    /// 默认与上下文同为 8k：辅助调用多为短摘要/精简，降低 KV 常驻和 prefill。
    public var auxKvCacheMaxTokens: Int = 8_192
    /// 辅助模型跨轮 KV/前缀复用开关（`aux_llm_kv_reuse`，默认关闭）：
    /// 辅助调用多为一次性长文本、跨调用命中率低，关闭时每轮全量重建并在
    /// 调用结束后丢弃 KV，避免长 KV 常驻；开启后保留缓存供相同前缀复用。
    public var auxKvReuse: Bool = false
    /// KV cache int8 量化（MLX 用 kvBits=8 的 QuantizedKVCache，GGUF 用
    /// ggml q8_0 KV）。q8_0 对生成质量几乎无损，KV 内存约减半；
    /// 作用于 MLX 与 GGUF 后端的主模型，辅助模型用 `auxKvCacheInt8`。
    public var kvCacheInt8: Bool = true
    /// 辅助模型的 KV cache int8 量化开关（缺省跟随 `kv_cache_int8`）。
    public var auxKvCacheInt8: Bool = true
    /// Flash Attention（默认开启，作用于 MLX 与 GGUF 的主/辅模型）：
    /// - MLX：使用未量化 KV 缓存，让 attention 走 `MLXFast.scaledDotProductAttention`
    ///   （Metal fused kernel）。上游 `QuantizedKVCache` 走手写反量化 attention，
    ///   不经过 fused kernel，故 MLX 上 flash 与 int8 KV 二者只能取其一（flash 优先）。
    /// - GGUF：`flash_attn_type = ENABLED`；llama.cpp 提供 `fa_q8_0` /
    ///   `fa_vec_q8_0` 专用 kernel，**可与 q8_0 KV 量化叠加**（实测组合正常生成）。
    /// 注意：本机 M1 + Qwen3.5（head_dim 256、无 NAX）实测提升在噪声范围内，
    /// 但 M3+/M4 或纯 attention 模型可受益，故默认开启。
    public var flashAttention: Bool = true

    /// 仅 MLX 使用：flash 开启时必须用未量化 KV，否则 flash 路径不会生效。
    /// GGUF 直接用 `kvCacheInt8` / `auxKvCacheInt8`（可与 flash 叠加）。
    public var effectiveMainKVInt8: Bool { kvCacheInt8 && !flashAttention }
    public var effectiveAuxKVInt8: Bool { auxKvCacheInt8 && !flashAttention }
    /// 预留（当前未接线）：上下文收缩由 L3 CONSOLIDATE 与 context-limit 重建处理，
    /// 不由本比例触发；保留字段以兼容旧配置文件。
    public var contextShrinkRatio: Double = 0.9

    /// 实际生效的主模型上下文：min(配置上下文, KV 硬上限)。
    public var effectiveMainContextTokens: Int {
        let cap = kvCacheMaxTokens > 0 ? kvCacheMaxTokens : maxContextTokens
        // 下限 8k：system + 工具 schema 已约 4.7k，再低会每轮都撑爆 prompt。
        return max(8_192, min(maxContextTokens, cap))
    }

    /// 实际生效的辅助模型上下文：min(配置上下文, KV 硬上限)。
    public var effectiveAuxContextTokens: Int {
        let cap = auxKvCacheMaxTokens > 0 ? auxKvCacheMaxTokens : auxMaxContextTokens
        return max(4_096, min(auxMaxContextTokens, cap))
    }

    // 内核：三态自动机 + 五级压缩
    public var kernelMaxContextTokens: Int = 30_000
    /// 稳定 raw 窗口：段内（两次 CONSOLIDATE 之间）保留最近 `kernelRawTurns` 个
    /// 已完成轮作为原文；溢出的旧轮**直接丢弃**，不再折成摘要回注上下文。
    /// 默认 10 = 折叠周期 8 + 折叠时保留的 2 轮：段内最多正好 10 轮，
    /// 不会中途滑窗挤掉最老轮（那会让 KV 前缀每轮变化、反复全量 prefill）。
    /// 它同时也是 CONSOLIDATE 周期的上钳位（见 `Loop.swift` 构造）。
    public var kernelRawTurns: Int = 10
    /// 周期性主动折叠间隔（轮），默认 8。触发时丢弃旧历史，但**保留最近 2 轮
    /// 原文**以防折叠处上下文断裂；不回注任何摘要、不给上下文加负担，KV 只
    /// 重建一次（旧历史已由 `onTurnFinished` 逐轮落盘，交给 L2 摘要 / L3 折叠
    /// 历史 / L4 长期记忆维护）。
    /// 实际生效周期 = `min(max(1, 本值), kernelRawTurns)`；配大于 rawTurns 会被钳住。
    /// **唯一例外**：上下文占用超过最大上下文 90%（`ContextBudget.needsAutoShrink`）时
    /// 触发自动收缩，才会用 LLM 生成旧上下文摘要并回注 `<earlier_context>`。
    public var kernelConsolidateEvery: Int = 8
    /// L1 工具结果字符上限。与 RAG 输出总上限（ragSearchMaxTotalChars=12000）
    /// 取较小值生效：默认 8000 让 RAG 结果不会被内核二次静默截断。
    public var kernelMaxToolChars: Int = 8_000
    /// RAG 工具结果「进历史 / KV 前缀」的额外收紧上限（字符）。0 表示不额外收紧，
    /// 沿用 L1 的 `kernelMaxToolChars`。RAG 结果通常 bulky，单独收紧只保留最相关
    /// 头部、丢弃尾部，避免几千字符的结果把历史 / KV 前缀撑大、拖慢后续每轮生成。
    public var kernelRagHistoryChars: Int = 2_000
    public var kernelMaxDigestLines: Int = 40

    // RAG
    /// 切片大小（token，含 overlap）。默认 800：jina-embeddings-v3 声明
    /// 窗口 8192，但 embedding 推理前统一压到 1024 token；800 token 在保证
    /// 语义完整的同时把切片数压到 bge 512 窗口方案的三分之一左右（向量库
    /// 体积与检索次数同步下降）。
    public var ragChunkSize: Int = 800
    /// 默认 chunk_size 的 10%（token）。计入 chunk_size 总预算，
    /// 正文按 `chunk_size - overlap` 切片，拼好后仍不超过 800 token。
    public var ragChunkOverlap: Int = 80
    /// 语义切片开关：开启时大段落在句间相似度低谷处断开（切片阶段需把
    /// 每个候选句嵌入一遍，显著慢于纯结构切片）；关闭时按标题结构 +
    /// 字符边界递归切片，速度快一个数量级，检索质量损失有限。
    public var ragSemanticChunking: Bool = true
    /// 默认 jina-embeddings-v5-text-nano（MLX int4）：768 原生维，
    /// 业务固定取前 128 维并重新归一化；查询 / 切片分别使用
    /// `Query: ` / `Document: ` 前缀，8192 上下文。
    public var ragEmbeddingModel: String = "jina-embeddings-v5-text-nano-mlx-int4"
    /// 默认 ONNX int8 版 jina reranker：CPU 推理吞吐约为 MLX mxfp8 的 2 倍，
    /// 1024 token 上下文。logit 统一经 sigmoid 后，800 字中文目标切片实测
    /// 约 0.53-0.91，无关切片约 0.03-0.27，默认阈值 0.45 有稳定间隔。
    /// 追求排序质量可改回 bge-reranker-v2-m3-mlx-mxfp8（MLX 后端）并上调阈值。
    public var ragRerankModel: String = "jina-reranker-v2-base-multilingual-onnx"
    /// 重排分数过滤阈值（0-1）。全部低于阈值时回退原排序，不会空结果。
    public var ragRerankThreshold: Double = 0.45
    public var ragRerankEnabled: Bool = true
    /// 重排序后保留的结果数量（1...20，默认 3）。开启重排序时以此为准：
    /// 候选池按它放大后交给 reranker，最终只取分数最高的 N 条；
    /// 关闭重排序时仍按调用方 `top_k` 返回。
    public var ragRerankTopN: Int = 3
    /// 候选池放大倍数（1...5，默认 3）。候选池 =
    /// max(调用方 top_k, TopN) × 倍数；越大召回越稳，但送进 reranker
    /// 的条数也越多。配置面板放在 TopN 之后，可随时调小降低 rerank 延迟。
    public var ragRerankCandidateMultiplier: Int = 3
    /// rerank 的墙钟超时（秒）。超时立即回退 RRF 排序，不再等模型跑完；
    /// 0 表示不限制（仅建议调试时使用）。
    public var ragRerankTimeoutSeconds: Double = 5
    /// 检索结果格式化时单条片段的字符上限（超出截断加省略号）。
    public var ragSearchMaxCharsPerResult: Int = 1_500
    /// 检索结果总字符上限（超出截断）。
    public var ragSearchMaxTotalChars: Int = 12_000
    /// 知识库只读查询（打开库 / 状态 / 检索）的超时秒数。上次同步被中断、
    /// 应用被强杀时，SQLite 需要先恢复遗留 WAL，可能耗时数分钟；启动与对话
    /// 路径都必须设上限：超出即跳过本次查询，不再阻塞。0 = 关闭超时。
    public var ragQueryTimeoutSeconds: Double = 5
    /// 辅助模型精简触发下限：检索结果总字数低于该值时原样返回，
    /// 不调用辅助模型（内容不多时总结只会丢细节、平白增加延迟）。
    /// 默认 8000：5 条 × 1500 字上限的常见结果不会触发，只有多文档/
    /// 超长结果才交给辅助模型。
    public var ragAuxSummarizeMinChars: Int = 8_000
    /// 辅助模型精简触发上限：超过该值的结果不做精简
    ///（内容过大时一次性摘要不可靠）。
    public var ragAuxSummarizeMaxChars: Int = 12_000
    /// 启动时自动同步知识库。默认关闭：首次全量同步会长时间占用
    /// embedding 模型与磁盘 IO，容易与对话争抢资源；需要时手动触发。
    public var ragAutoIngest: Bool = false
    public var ragPDFEngine: PDFEngine = .vision
    /// 扫描版 EPUB 的 OCR 后端：vision（Apple Vision）/ ovis（OvisOCR3）。
    /// 两条路径都会先做统一图像归一化。
    public var ragEPUBOCREngine: PDFEngine = .vision
    public var ragOCREngine: String = "mlx-vlm"
    public var ragOCRModelName: String = "OvisOCR3-MLX-4bit-mixed46"
    public var ragOCRModelPath: String = "models/OvisOCR3-MLX-4bit-mixed46"
    /// GGUF 视觉模型的多模态投影文件；留空时自动查找模型同目录的 mmproj*.gguf。
    public var ragOCRMMProjPath: String = ""
    /// OCR ONNX 模型的 tokenizer 目录（自定义 OCR 选 ONNX 后端时必填）。
    public var ragOCRTokenizerPath: String = ""
    /// 预留：主模型 GGUF 的 mmproj 路径（当前主/辅 LLM 为纯文本，未接入视觉输入）。
    public var llmMmprojPath: String = ""
    /// 主模型 ONNX 的 tokenizer 目录；留空时从模型目录读取。
    public var llmTokenizerPath: String = ""
    /// 预留：辅助模型 GGUF 的 mmproj 路径（当前主/辅 LLM 为纯文本，未接入视觉输入）。
    public var auxMmprojPath: String = ""
    /// 辅助模型 ONNX 的 tokenizer 目录；留空时从模型目录读取。
    public var auxTokenizerPath: String = ""
    /// 预留（当前未接线）：OCR 后端由 ragOCREngine / ragOCRModelPath 选择，
    /// 暂无 CoreML 分支；保留字段以兼容旧配置文件。
    public var ragOCRCoreML: Bool = false
    public var ragOCRPreprocess: Bool = true
    public var ragOCRBinarize: Bool = true
    /// llama.cpp 的 GPU 层数（-ngl）；内存紧张时可设为 0 改用 CPU。
    public var llamaGpuLayers: Int = 99
    /// 同时处理的文件数（1...10）。解析/OCR 与 embedding 流水线并行；
    /// 纯文本文件不经过 OCR 闸门，多 worker 可把等待空档填满。
    public var ragIngestConcurrency: Int = 10
    /// Apple Vision OCR 的全局并发闸门（跨文件共享）。
    /// 实测 4 路已接近吞吐拐点，再多不一定快甚至变慢；0 也按 4 处理，
    /// 保留覆盖入口便于实验。
    public var ragVisionConcurrency: Int = 4
    /// OvisOCR3 并行实例数（1...2）。4bit 权重约 0.7GB，单实例运行约 1GB；
    /// 默认 2：同步期间双实例并行识别扫描页。权重仍按需懒加载，同步/转换/
    /// OCR 测试结束后由 `OCRModelLifecycle` 立即释放两个实例，不常驻内存。
    public var ragOvisInstances: Int = 2

    // Embedding 子批自适应（AdaptiveEmbeddingScheduler）
    /// 子批下限（2...64），默认 2；运行时还会夹到不超过上限。
    public var ragEmbeddingMinBatch: Int = 2
    /// 子批上限（2...256）。默认 32：实测 477 token 长块在 4-16 已达吞吐平台，
    /// 32 兼顾短块；64 以上只增加内存与单批延迟。
    public var ragEmbeddingMaxBatch: Int = 32
    /// 每次同步的起始子批（2...256），默认 16；运行时夹到 [下限, 上限]。
    public var ragEmbeddingInitialBatch: Int = 16
    /// 每轮上探/下探的步进（1...16），默认 2。
    public var ragEmbeddingBatchStep: Int = 2
    /// 子批调参的进程 footprint 上限（MB，1024...32768），默认 6144。
    /// 超过该值立即下调子批；bge 单独运行时峰值约 1.6GB。
    public var ragEmbeddingMemoryLimitMB: Int = 6_144
    /// 吞吐变化容差（0.01...0.5），默认 0.03。变化小于该比例视为测量噪声，
    /// 不调整子批；实测同 batch 重复测量的噪声可达 ±20%，可调到 0.08-0.10。
    public var ragEmbeddingThroughputTolerance: Double = 0.03
    /// 每档至少观察的批次数（1...16，默认 2），与 probeSeconds 满足其一即评估。
    public var ragEmbeddingProbeBatches: Int = 2
    /// 每档至少观察的秒数（0.5...30，默认 2）。
    public var ragEmbeddingProbeSeconds: Double = 2.0
    /// 单批 token 预算下限（默认 4096）。
    public var ragEmbeddingTokenBudgetMin: Int = 4_096
    /// 单批 token 预算上限（默认 8192）。
    public var ragEmbeddingTokenBudgetMax: Int = 8_192
    /// 同步起始的 token 预算（默认 6144）。
    public var ragEmbeddingTokenBudgetInitial: Int = 6_144
    /// 自适应调整 token 预算的步进（默认 512）。
    public var ragEmbeddingTokenBudgetStep: Int = 512
    /// 增量同步是否校验文件内容指纹。默认关闭：信任 mtime + size，
    /// 只在文件变化或 force 时重新采样，避免每次同步都读一遍文件头尾。
    /// 开启后即使 mtime/size 相同也会比对采样指纹（能发现保留时间戳的改动）。
    public var ragVerifyFingerprint: Bool = false

    // SQLite + sqlite-vec 存储
    /// 预留（当前固定 sqlite）：migrate() 会把非 "sqlite" 的配置强制归一，
    /// 生产检索只由 ragSQLiteVecEnabled / elementType 决定存储形态。
    public var ragStorageBackend: String = "sqlite"
    public var ragSQLiteVecEnabled: Bool = true
    /// 混合检索 RRF 常数（原项目 RRF k=60）。
    public var ragRRFK: Int = 60
    /// 向量召回 over-fetch 倍数。混合检索先融合 rowid/分数再 hydrate，
    /// 内层只需比最终条数多 1-2 倍；定向来源（sourceContains）按需自动放大。
    public var ragVectorOverfetch: Int = 2
    /// FTS5 召回 over-fetch 倍数。
    public var ragFTSOverfetch: Int = 4

    // RAG MCP 服务（MCP 2025-06-18 Streamable HTTP，供其它大模型应用对接）
    /// 监听范围：false = 仅本机（绑定 127.0.0.1，默认）；true = 局域网（绑定 0.0.0.0）。
    public var ragMCPListenLan: Bool = false
    /// 服务监听端口（默认 8080；migrate 会钳制到 1~65535）。
    public var ragMCPPort: Int = 8080
    /// 服务开关（默认关闭）。App 启动后按此恢复服务状态；修改监听范围/端口
    /// 后需要停止再启动才生效。
    public var ragMCPEnabled: Bool = false

    // 向量落盘编码
    /// 向量元素类型：int8（默认；默认模型 128 维时每行 128B）/ fp32 / fp16 / bin。
    /// fp16 每维 2 字节，因 sqlite-vec 0.1.9 没有该列类型，走普通表 + 精确扫描。
    /// 改变类型会让已有向量失效，需要重建向量库。
    public var ragVectorElementType: VectorElementType = .defaultValue
    /// Matryoshka 截断策略（auto = nomic / jina-v3 由外层截断；
    /// jina-v5 由 provider 内部按配置维输出）。
    public var ragMatryoshka: MatryoshkaMode = .auto
    /// Matryoshka 截断维：默认 128。jina-v5 支持
    /// 32 / 64 / 128 / 256 / 512 / 768；其它模型沿用 128 / 256 / 512。
    public var ragMatryoshkaDim: Int = MatryoshkaDimensions.defaultValue

    // 知识图谱检索
    /// RAG 检索策略（默认 naive = 图谱能力加入前的原始实现）。
    /// 其余四种策略需先「构建知识图谱」；图谱未就绪时自动回退 naive。
    public var ragRetrievalStrategy: RAGRetrievalStrategy = .naive
    /// 实体/关系关键词召回时的相似度下限（低于该余弦值的候选直接丢弃）。
    public var ragKGSimilarityFloor: Double = 0.2
    /// 实体召回的 top_k（默认 40）。
    public var ragKGEntityTopK: Int = 40
    /// 关系召回的 top_k（默认 40）。
    public var ragKGRelationTopK: Int = 40
    /// 实体/关系反查原文时每条最多带回的相关切片数（默认 5）。
    public var ragKGMaxChunksPerHit: Int = 5
    /// 实体上下文 token 预算上限（默认 6000）。
    public var ragKGEntityTokenBudget: Int = 6_000
    /// 关系上下文 token 预算上限（默认 8000）。
    public var ragKGRelationTokenBudget: Int = 8_000
    /// KG + 原文 + system + query 的总 token 预算（默认 30000）。
    public var ragKGTotalTokenBudget: Int = 30_000
    /// 单个切片最多保留的实体数（默认 20）。
    public var ragKGMaxEntitiesPerChunk: Int = 20
    /// 单个切片最多保留的关系数（默认 50）。
    public var ragKGMaxRelationsPerChunk: Int = 50
    /// 实体/关系描述的最大字符数（默认 100）。
    public var ragKGMaxDescriptionChars: Int = 100

    // 二值粗筛（两段式检索）
    /// 检索是否先走二值向量粗筛；关闭只影响检索，入库仍维护签名。
    public var ragBinaryCoarseEnabled: Bool = true
    /// 第一阶段候选数：默认 200，加载时收口到 50...1000。
    public var ragBinaryCoarseCandidates: Int = 200

    // ANN 向量索引（分块 IVF，见 VectorStoreANN.swift）
    /// 检索是否走 ANN 索引。索引不可用（未构建/维度不符/尾块超预算）时
    /// 自动回退精确检索；关闭则始终精确检索（大库上会超时被跳过）。
    public var ragANNEnabled: Bool = true
    /// 每块簇数。块越大召回越稳、内存峰值越高（块 = 簇数 × 每簇行数）。
    public var ragANNClusters: Int = 256
    /// 每簇行数。簇越小越省读放大、召回越依赖 nprobe。
    public var ragANNClusterSize: Int = 512
    /// 每块探测的簇数（召回随探测数上升，读放大同步增加）。
    /// 离线评测（VectorStoreANNQualityTests：recall@20/nDCG@20）中
    /// nprobe 1→2→4 的召回为 26%→45%→74%，取 4；真实库可用
    /// `RAGEngine.evaluateANNQuality` 再校准。
    public var ragANNProbe: Int = 4
    /// 用精确向量重打分的候选数（0 = 关闭）。只需覆盖最终 topK 并留余量，
    /// 评测中 64 与 128 的召回/排序一致，取 64 控制点查成本。
    public var ragANNRescore: Int = 64
    /// 自动构建的切片数下限：小库上精确扫描足够快，不值得建索引。
    public var ragANNMinChunks: Int = 20_000
    /// 同步结束后自动构建/续建 ANN 索引（索引已就绪且尾块不多时跳过）。
    public var ragANNAutoBuild: Bool = true

    // 本地模型 provider
    // 模型类型统一使用 mlx-lm / mlx-vlm / onnx / gguf；旧值 mlx 迁移为 mlx-lm。
    public var llmProvider: String = "local"
    public var localLLMModel: String = "models/qwen3.5-4b-agentic-coder-v4-4bit-mlx"
    public var localLLMBackend: String = "mlx-lm"
    /// 主模型 thinking 默认开启：关闭会破坏主引擎的推理与工具决策质量。
    /// 不能为了工具路由延迟而牺牲主模型默认值；低延迟轮次应在请求级关闭，
    /// 或交给辅助模型。后台任务始终请求级强制关闭（见 Agent）。
    public var localLLMEnableThinking: Bool = true
    /// 启动时是否预加载主模型。默认开启；关闭后启动只创建 Agent，不加载权重，
    /// 需在配置窗口手动「加载模型」。不影响配置应用/同步结束后的模型恢复。
    public var autoPreloadMain: Bool = true
    /// 默认模型缺失时是否允许自动从 hf-mirror 下载补全。默认开启；
    /// 用户在首启提示里选择「自行下载指定」后置为 false——不再自动下载，
    /// 由用户自行获取模型并在配置窗口指定，配置就绪后自动解除使用限制。
    public var autoDownloadMissingModels: Bool = true

    /// 是否启用辅助模型。默认关闭：启动和知识同步前后都不加载辅助模型；
    /// RAG 精简直接跳过，后台记忆/总结回退主模型（若主模型可用）。
    public var auxEnabled: Bool = false
    public var auxLLMProvider: String = "local"
    public var auxLocalLLMModel: String = "models/Qwen3.5-2B-MLX-4bit"
    public var auxLocalLLMBackend: String = "mlx-lm"
    /// 预留（当前未接线）：各辅助任务（摘要/事实抽取/蒸馏/RAG 浓缩）按任务
    /// 固定低温度 0.1~0.2，避免发散；设置面板保留该字段以兼容旧配置，
    /// 修改它不会改变辅助调用温度。其它采样参数（auxTopP/auxTopK 等）正常生效。
    public var auxTemperature: Double = 0.3
    public var auxMaxTokens: Int = 32768
    /// 辅助调用默认 8k 上下文：缩短 prefill、降低 KV 内存，加快响应。
    public var auxMaxContextTokens: Int = 8_192
    public var auxPromptBatchSize: Int = 512
    public var auxTopP: Double = 0.95
    public var auxMinP: Double = 0.05
    public var auxTopK: Int = 20
    public var auxRepetitionPenalty: Double = 1.1
    public var auxRequestTimeout: Double = 120.0
    public var auxLocalLLMEnableThinking: Bool = false

    public var embeddingProvider: String = "local"
    public var localEmbeddingModel: String = "jina-embeddings-v5-text-nano-mlx-int4"
    /// 显式本地目录；非空时优先于模型名。
    /// 使用相对模型根目录的路径，由 LocalModelRegistry 统一解析。
    public var localEmbeddingModelPath: String = "models/jina-embeddings-v5-text-nano-mlx-int4"
    /// onnx / fastembed / mlx / gguf
    /// Embedding 后端：mlx（Metal GPU，默认，使用本地 int4/int8 混合量化权重）
    /// / onnx（CPU int8）/ fastembed / gguf。
    public var localEmbeddingBackend: String = "mlx"

    public var rerankerProvider: String = "local"
    public var localRerankerModel: String = "jina-reranker-v2-base-multilingual-onnx"
    public var localRerankerModelPath: String = "models/jina-reranker-v2-base-multilingual-onnx"
    /// onnx / fastembed / mlx / gguf
    /// 默认 ONNX：与默认模型 jina-reranker-v2-base-multilingual-onnx 的权重格式一致。
    public var localRerankerBackend: String = "onnx"

    public var vlmProvider: String = "api"
    public var localVLMModel: String = ""
    public var localVLMBackend: String = "mlx"

    public var browserEndpoint: String = ""
    public var alwaysConfirmCode: Bool = false
    public var mcpServers: [MCPServerConfig] = []

    public init() {}

    // MARK: Paths

    public var resolvedKnowledgeBase: String {
        knowledgeBase.isEmpty
            ? (workspace as NSString).appendingPathComponent("knowledge_base")
            : expandPath(knowledgeBase)
    }

    /// RAG 索引与产物目录：按知识库目录隔离（`rag_data/<目录名-路径哈希>/`）。
    /// 历史版本所有知识库共用 `rag_data` 根目录，切换知识库目录后互相把对方
    /// 的条目当「已删除文件」清掉，导致增量同步失效；按知识库隔离后各自
    /// 保留 manifest 与索引。知识库目录是只读的原始文件目录，绝不向其写入。
    public var ragIndexDirectory: String {
        Self.ragIndexDirectory(forKB: resolvedKnowledgeBase, workspace: workspace)
    }

    /// 按知识库目录推导索引子目录：真实路径规范化 + 目录名 + 绝对路径哈希，
    /// 保证同名不同位置的目录各自隔离，同一路径多次推导结果稳定。
    public static func ragIndexDirectory(forKB kb: String, workspace: String) -> String {
        let base = (workspace as NSString).appendingPathComponent("rag_data")
        let canonical = canonicalizePath(kb)
        let name = (canonical as NSString).lastPathComponent
        let digest = SHA256.hash(data: Data(canonical.utf8))
        let short = digest.prefix(4).map { String(format: "%02x", $0) }.joined()
        return (base as NSString).appendingPathComponent("\(name.isEmpty ? "kb" : name)-\(short)")
    }

    /// `realpath` 规范化：解析 `/var` → `/private/var` 与其它符号链接。
    public static func canonicalizePath(_ path: String) -> String {
        let expanded = (path as NSString).expandingTildeInPath
        guard let resolved = realpath(expanded, nil) else {
            return URL(fileURLWithPath: expanded).standardizedFileURL.path
        }
        defer { free(resolved) }
        return String(cString: resolved)
    }

    public var dbPath: String {
        (ragIndexDirectory as NSString).appendingPathComponent("index.sqlite")
    }

    public var modelsDirectory: String {
        (workspace as NSString).appendingPathComponent("models")
    }

    @discardableResult
    public func ensureWorkspace() -> String {
        try? FileManager.default.createDirectory(atPath: workspace, withIntermediateDirectories: true)
        return workspace
    }

    // MARK: Load / Save

    public static func load(_ path: String) -> AgentConfig {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: path) else { return AgentConfig() }
        guard let data = fileManager.contents(atPath: path),
              let raw = try? JSONDecoder().decode(JSONValue.self, from: data),
              let dict = raw.objectValue else {
            // 配置文件损坏时备份后使用默认值启动。
            let backup = path + ".corrupt"
            try? fileManager.removeItem(atPath: backup)
            try? fileManager.moveItem(atPath: path, toPath: backup)
            AgentLog.warning("config 无法解析，已备份到 \(backup)，使用默认配置")
            return AgentConfig()
        }

        var cfg = AgentConfig()
        cfg.apply(dict)
        cfg.migrate()
        return cfg
    }

    private mutating func apply(_ d: [String: JSONValue]) {
        func string(_ key: String) -> String? {
            guard let v = d[key] else { return nil }
            return v.stringValue
        }
        func int(_ key: String) -> Int? {
            guard let v = d[key] else { return nil }
            return v.intValue
        }
        func double(_ key: String) -> Double? {
            guard let v = d[key] else { return nil }
            return v.doubleValue
        }
        func bool(_ key: String) -> Bool? {
            guard let v = d[key] else { return nil }
            return v.boolValue
        }

        if let v = string("base_url") { baseURL = v }
        if let v = string("api_key") { apiKey = v }
        if let v = string("model") { model = v }
        if let v = double("temperature") { temperature = v }
        if let v = int("max_tokens") { maxTokens = v }
        if let v = double("top_p") { topP = v }
        if let v = double("min_p") { minP = v }
        if let v = int("top_k") { topK = v }
        if let v = double("repetition_penalty") { repetitionPenalty = v }
        if let v = string("workspace"), !v.isEmpty { workspace = expandPath(v) }
        if let v = string("knowledge_base") { knowledgeBase = v }
        if let v = double("request_timeout") { requestTimeout = v }
        if let v = int("max_iterations") { maxIterations = v }
        if let v = int("max_history") { maxHistory = v }
        if let v = int("max_context_tokens") { maxContextTokens = v }
        if let v = double("context_shrink_ratio") { contextShrinkRatio = v }

        if let v = int("prompt_batch_size") { promptBatchSize = v }
        if let v = int("mlx_cache_limit_mb") { mlxCacheLimitMB = v }
        if let v = int("mlx_memory_limit_mb") { mlxMemoryLimitMB = v }
        if let v = int("kv_cache_max_tokens") { kvCacheMaxTokens = v }
        if let v = bool("llm_kv_reuse") { llmKVReuse = v }
        if let v = bool("kv_cache_int8") { kvCacheInt8 = v }
        if let v = bool("flash_attention") { flashAttention = v }
        // 旧配置没有辅助模型的 KV 键：复用默认关闭（与改造前「用后即清」一致），
        // 量化跟随主模型。
        auxKvReuse = bool("aux_llm_kv_reuse") ?? false
        auxKvCacheInt8 = bool("aux_kv_cache_int8") ?? kvCacheInt8
        if let v = int("aux_kv_cache_max_tokens") { auxKvCacheMaxTokens = v }
        if let v = int("kernel_max_context_tokens") { kernelMaxContextTokens = v }
        if let v = int("kernel_raw_turns") { kernelRawTurns = v }
        if let v = int("kernel_consolidate_every") { kernelConsolidateEvery = v }
        if let v = int("kernel_max_tool_chars") { kernelMaxToolChars = v }
        if let v = int("kernel_rag_history_chars") { kernelRagHistoryChars = v }
        if let v = int("kernel_max_digest_lines") { kernelMaxDigestLines = v }

        if let v = int("rag_chunk_size") { ragChunkSize = v }
        if let v = int("rag_chunk_overlap") { ragChunkOverlap = v }
        if let v = bool("rag_semantic_chunking") { ragSemanticChunking = v }
        if let v = int("rag_ingest_concurrency") {
            ragIngestConcurrency = min(max(1, v), 10)
        }
        if let v = bool("rag_verify_fingerprint") { ragVerifyFingerprint = v }
        if let v = int("rag_vision_concurrency") { ragVisionConcurrency = max(0, v) }
        if let v = int("rag_ovis_instances") { ragOvisInstances = min(max(1, v), 2) }
        if let v = int("rag_embedding_min_batch") { ragEmbeddingMinBatch = v }
        if let v = int("rag_embedding_max_batch") { ragEmbeddingMaxBatch = v }
        if let v = int("rag_embedding_initial_batch") { ragEmbeddingInitialBatch = v }
        if let v = int("rag_embedding_batch_step") { ragEmbeddingBatchStep = v }
        if let v = int("rag_embedding_memory_limit_mb") { ragEmbeddingMemoryLimitMB = v }
        if let v = double("rag_embedding_throughput_tolerance") {
            ragEmbeddingThroughputTolerance = v
        }
        if let v = int("rag_embedding_probe_batches") { ragEmbeddingProbeBatches = v }
        if let v = double("rag_embedding_probe_seconds") { ragEmbeddingProbeSeconds = v }
        if let v = int("rag_embedding_token_budget_min") { ragEmbeddingTokenBudgetMin = v }
        if let v = int("rag_embedding_token_budget_max") { ragEmbeddingTokenBudgetMax = v }
        if let v = int("rag_embedding_token_budget_initial") { ragEmbeddingTokenBudgetInitial = v }
        if let v = int("rag_embedding_token_budget_step") { ragEmbeddingTokenBudgetStep = v }
        if let v = string("rag_embedding_model") { ragEmbeddingModel = v }
        if let v = string("rag_rerank_model") { ragRerankModel = v }
        if let v = bool("rag_rerank_enabled") { ragRerankEnabled = v }
        if let v = int("rag_rerank_top_n") { ragRerankTopN = v }
        if let v = int("rag_rerank_candidate_multiplier") { ragRerankCandidateMultiplier = v }
        if let v = double("rag_rerank_timeout_seconds"), v >= 0 { ragRerankTimeoutSeconds = v }
        if let v = double("rag_query_timeout_seconds"), v >= 0 { ragQueryTimeoutSeconds = v }
        if let v = bool("rag_ann_enabled") { ragANNEnabled = v }
        if let v = int("rag_ann_clusters") { ragANNClusters = v }
        if let v = int("rag_ann_cluster_size") { ragANNClusterSize = v }
        if let v = int("rag_ann_nprobe") { ragANNProbe = v }
        if let v = int("rag_ann_rescore") { ragANNRescore = v }
        if let v = int("rag_ann_min_chunks") { ragANNMinChunks = v }
        if let v = bool("rag_ann_auto_build") { ragANNAutoBuild = v }
        if let v = bool("rag_auto_ingest") { ragAutoIngest = v }
        if let v = string("rag_pdf_engine") { ragPDFEngine = PDFEngine(stored: v) }
        if let v = string("rag_epub_ocr_engine") { ragEPUBOCREngine = PDFEngine(stored: v) }
        if let v = string("rag_ocr_engine") { ragOCREngine = v }
        if let v = string("rag_ocr_model_name") { ragOCRModelName = v }
        if let v = string("rag_ocr_model_path") { ragOCRModelPath = v }
        if let v = string("rag_ocr_mmproj_path") { ragOCRMMProjPath = v }
        if let v = string("rag_ocr_tokenizer_path") { ragOCRTokenizerPath = v }
        if let v = string("llm_mmproj_path") { llmMmprojPath = v }
        if let v = string("llm_tokenizer_path") { llmTokenizerPath = v }
        if let v = string("aux_mmproj_path") { auxMmprojPath = v }
        if let v = string("aux_tokenizer_path") { auxTokenizerPath = v }
        if let v = bool("rag_ocr_coreml") { ragOCRCoreML = v }
        if let v = bool("rag_ocr_preprocess") { ragOCRPreprocess = v }
        if let v = bool("rag_ocr_binarize") { ragOCRBinarize = v }
        if let v = int("llama_gpu_layers") { llamaGpuLayers = max(0, v) }

        if let v = string("rag_storage_backend") { ragStorageBackend = v }
        if let v = bool("rag_sqlite_vec_enabled") { ragSQLiteVecEnabled = v }
        if let v = int("rag_rrf_k") { ragRRFK = v }
        if let v = double("rag_rerank_threshold") { ragRerankThreshold = v }
        if let v = int("rag_search_max_chars_per_result") { ragSearchMaxCharsPerResult = v }
        if let v = int("rag_search_max_total_chars") { ragSearchMaxTotalChars = v }
        if let v = int("rag_aux_summarize_min_chars") { ragAuxSummarizeMinChars = v }
        if let v = int("rag_aux_summarize_max_chars") { ragAuxSummarizeMaxChars = v }
        if let v = int("rag_vector_overfetch") { ragVectorOverfetch = v }
        if let v = int("rag_fts_overfetch") { ragFTSOverfetch = v }
        if let v = bool("rag_mcp_listen_lan") { ragMCPListenLan = v }
        if let v = int("rag_mcp_port") { ragMCPPort = v }
        if let v = bool("rag_mcp_enabled") { ragMCPEnabled = v }
        if let v = string("rag_vector_element_type"),
           let parsed = VectorElementType(rawValue: v.lowercased()) {
            ragVectorElementType = parsed
        }
        if let v = string("rag_matryoshka"),
           let parsed = MatryoshkaMode(rawValue: v.lowercased()) {
            ragMatryoshka = parsed
        }
        if let v = int("rag_matryoshka_dim") { ragMatryoshkaDim = v }
        if let v = string("rag_retrieval_strategy") {
            ragRetrievalStrategy = RAGRetrievalStrategy(stored: v)
        }
        if let v = double("rag_kg_similarity_floor") { ragKGSimilarityFloor = v }
        if let v = int("rag_kg_entity_top_k") { ragKGEntityTopK = v }
        if let v = int("rag_kg_relation_top_k") { ragKGRelationTopK = v }
        if let v = int("rag_kg_max_chunks_per_hit") { ragKGMaxChunksPerHit = v }
        if let v = int("rag_kg_entity_token_budget") { ragKGEntityTokenBudget = v }
        if let v = int("rag_kg_relation_token_budget") { ragKGRelationTokenBudget = v }
        if let v = int("rag_kg_total_token_budget") { ragKGTotalTokenBudget = v }
        if let v = int("rag_kg_max_entities_per_chunk") { ragKGMaxEntitiesPerChunk = v }
        if let v = int("rag_kg_max_relations_per_chunk") { ragKGMaxRelationsPerChunk = v }
        if let v = int("rag_kg_max_description_chars") { ragKGMaxDescriptionChars = v }
        if let v = bool("rag_binary_coarse_enabled") { ragBinaryCoarseEnabled = v }
        if let v = int("rag_binary_coarse_candidates") { ragBinaryCoarseCandidates = v }

        if let v = string("llm_provider") { llmProvider = v }
        if let v = string("local_llm_model") { localLLMModel = v }
        if let v = string("local_llm_backend") { localLLMBackend = v }
        if let v = bool("local_llm_enable_thinking") { localLLMEnableThinking = v }
        if let v = bool("auto_preload_main") { autoPreloadMain = v }
        if let v = bool("auto_download_missing_models") { autoDownloadMissingModels = v }

        if let v = bool("aux_enabled") { auxEnabled = v }
        if let v = string("aux_llm_provider") { auxLLMProvider = v }
        if let v = string("aux_local_llm_model") { auxLocalLLMModel = v }
        if let v = string("aux_local_llm_backend") { auxLocalLLMBackend = v }
        if let v = double("aux_temperature") { auxTemperature = v }
        if let v = int("aux_max_tokens") { auxMaxTokens = v }
        if let v = double("aux_top_p") { auxTopP = v }
        if let v = double("aux_min_p") { auxMinP = v }
        if let v = int("aux_top_k") { auxTopK = v }
        if let v = double("aux_repetition_penalty") { auxRepetitionPenalty = v }
        if let v = double("aux_request_timeout") { auxRequestTimeout = v }
        if let v = bool("aux_local_llm_enable_thinking") { auxLocalLLMEnableThinking = v }
        if let v = int("aux_max_context_tokens") { auxMaxContextTokens = v }
        if let v = int("aux_prompt_batch_size") { auxPromptBatchSize = v }

        if let v = string("embedding_provider") { embeddingProvider = v }
        if let v = string("local_embedding_model") { localEmbeddingModel = v }
        if let v = string("local_embedding_model_path") { localEmbeddingModelPath = v }
        if let v = string("local_embedding_backend") { localEmbeddingBackend = v }
        if let v = string("reranker_provider") { rerankerProvider = v }
        if let v = string("local_reranker_model") { localRerankerModel = v }
        if let v = string("local_reranker_model_path") { localRerankerModelPath = v }
        if let v = string("local_reranker_backend") { localRerankerBackend = v }
        if let v = string("vlm_provider") { vlmProvider = v }
        if let v = string("local_vlm_model") { localVLMModel = v }
        if let v = string("local_vlm_backend") { localVLMBackend = v }

        // 兼容其他版本/参考实现的字段命名（仅当本版本规范键缺失时生效）。
        if d["llm_provider"] == nil, let v = string("llm_mode") {
            llmProvider = v
        }
        if d["local_llm_backend"] == nil, let v = string("llm_backend") {
            localLLMBackend = v
        }
        if d["local_llm_model"] == nil, let v = string("llm_model_path") {
            localLLMModel = v
        }
        if d["local_llm_enable_thinking"] == nil,
           let v = bool("llm_enable_thinking") {
            localLLMEnableThinking = v
        }
        if d["aux_local_llm_backend"] == nil, let v = string("aux_llm_backend") {
            auxLocalLLMBackend = v
        }
        if d["aux_local_llm_model"] == nil, let v = string("aux_llm_model_path") {
            auxLocalLLMModel = v
        }
        if d["aux_local_llm_enable_thinking"] == nil,
           let v = bool("aux_llm_enable_thinking") {
            auxLocalLLMEnableThinking = v
        }
        if d["local_embedding_backend"] == nil,
           let v = string("rag_embedding_backend") {
            localEmbeddingBackend = v
        }
        if d["local_embedding_model_path"] == nil,
           let v = string("rag_embedding_model_path") {
            localEmbeddingModelPath = v
        }
        if d["local_reranker_backend"] == nil,
           let v = string("rag_rerank_backend") {
            localRerankerBackend = v
        }
        // 旧配置为 MLX / ONNX 各保存一份路径；按当前类型选择对应路径，避免 MLX
        // 后端误用 ONNX 文件（反之亦然）。
        if d["local_reranker_model_path"] == nil {
            if Self.normalizeEmbeddingBackend(localRerankerBackend) == "mlx",
               let v = string("rag_rerank_mlx_model_path"), !v.isEmpty {
                localRerankerModelPath = v
            } else if let v = string("rag_rerank_model_path") {
                localRerankerModelPath = v
            }
        }

        if let v = string("browser_endpoint") { browserEndpoint = v }
        if let v = bool("always_confirm_code") { alwaysConfirmCode = v }
        if let servers = d["mcp_servers"]?.arrayValue {
            mcpServers = servers.compactMap { item in
                guard let o = item.objectValue,
                      let name = o["name"]?.stringValue,
                      let command = o["command"]?.stringValue else { return nil }
                let args = o["args"]?.arrayValue?.compactMap(\.stringValue) ?? []
                var env: [String: String] = [:]
                if let e = o["env"]?.objectValue {
                    env = e.compactMapValues(\.stringValue)
                }
                return MCPServerConfig(name: name, command: command, args: args, env: env)
            }
        }
    }

    /// 旧版本配置迁移（docling/marker → ovis、旧 embedding/reranker 名）。
    public mutating func migrate() {
        // rerank 候选池倍数：手改配置可能越界，统一收口到 1...5，
        // 避免一次把上百条切片送进 ONNX。
        ragRerankCandidateMultiplier = max(1, min(5, ragRerankCandidateMultiplier))
        // 二值粗筛候选数：手改配置可能越界，统一收口到 50...1000。
        ragBinaryCoarseCandidates = max(50, min(1_000, ragBinaryCoarseCandidates))
        // RAG MCP 服务端口：钳制到合法 TCP 端口范围，非法值回落默认 8080。
        if !(1...65535).contains(ragMCPPort) { ragMCPPort = 8080 }
        // docling/marker 已移除：旧配置值统一由 PDFEngine(stored:) 归一化为 ovis。
        // 默认 embedding 迁移到 jina-embeddings-v5-text-nano（MLX int4）：
        // 上一版默认是 jina-v3，更早是 nomic MLX int8 / bge。
        // 只迁移恰好等于旧默认的 model + path 组合，用户显式选择的其他
        // 模型/目录保持不变。
        let defaultEmbeddingName = "jina-embeddings-v5-text-nano-mlx-int4"
        let defaultEmbeddingPath = "models/\(defaultEmbeddingName)"
        let legacyEmbeddingNames: Set<String> = [
            "all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-large-zh-v1.5",
            "",
            "nomic-ai/nomic-embed-text-v1.5",
            "nomic-ai/nomic-embed-text-v1.5-Q",
            "bge-small-zh-v1.5-mlx",
            "BAAI/bge-small-zh-v1.5",
            "nomic-ai/nomic-embed-text-v1.5-mlx",
            "nomic-ai/nomic-embed-text-v1.5-mlx-8bit",
            "nomic-embed-text-v1.5-mlx-8bit",
            "nomic-embed-text-v1.5-mlx-int8",
            "jina-embeddings-v3-mlx-int4",
        ]
        let legacyEmbeddingPaths: Set<String> = [
            "bge-small-zh-v1.5-mlx",
            "models/bge-small-zh-v1.5-mlx",
            "nomic-embed-text-v1.5-mlx",
            "models/nomic-embed-text-v1.5-mlx",
            "nomic-embed-text-v1.5-mlx-8bit",
            "models/nomic-embed-text-v1.5-mlx-8bit",
            "nomic-embed-text-v1.5-mlx-int8",
            "models/nomic-embed-text-v1.5-mlx-int8",
            "jina-embeddings-v3-mlx-int4",
            "models/jina-embeddings-v3-mlx-int4",
        ]
        let legacyPathSuffixes = [
            "/bge-small-zh-v1.5-mlx",
            "/nomic-embed-text-v1.5-mlx",
            "/nomic-embed-text-v1.5-mlx-8bit",
            "/nomic-embed-text-v1.5-mlx-int8",
            "/jina-embeddings-v3-mlx-int4",
        ]
        let pathIsLegacy = legacyEmbeddingPaths.contains(localEmbeddingModelPath)
            || legacyPathSuffixes.contains { localEmbeddingModelPath.hasSuffix($0) }
        // 旧默认都是 MLX 目录；非 MLX 后端不参与这次默认模型切换。
        if Self.normalizeEmbeddingBackend(localEmbeddingBackend) == "mlx" {
            let ragIsLegacy = legacyEmbeddingNames.contains(ragEmbeddingModel)
            let localModelIsLegacy = legacyEmbeddingNames.contains(localEmbeddingModel)
            // 完整旧默认：model 与 path 同时等于旧默认才一起迁移。
            // 旧版只写 rag_embedding_model_path 的配置（model 已是当前默认）
            // 只修路径，避免 model 指向 V5、path 仍指向 v3。
            let migrateLocalConfig = (localModelIsLegacy && pathIsLegacy)
                || (localEmbeddingModel == defaultEmbeddingName && pathIsLegacy)
            if ragIsLegacy { ragEmbeddingModel = defaultEmbeddingName }
            if migrateLocalConfig {
                localEmbeddingModel = defaultEmbeddingName
                localEmbeddingModelPath = defaultEmbeddingPath
            }
            // 旧默认 512 / 256 维随默认模型一起收口到 V5 默认的 128；
            // 用户显式配置的其它模型/维度不动。
            if ragIsLegacy || migrateLocalConfig,
               ragMatryoshkaDim == 512 || ragMatryoshkaDim == 256 {
                ragMatryoshkaDim = 128
            }
        }
        // 旧 reranker 默认（HF 名）迁移到本地 MLX mxfp8 目录名。
        if ragRerankModel == "BAAI/bge-reranker-v2-m3"
            || ragRerankModel == "BAAI/bge-reranker-v2-m3-int8" {
            ragRerankModel = "bge-reranker-v2-m3-mlx-mxfp8"
        }
        // 上一版默认（bge mxfp8 + MLX）迁移到 jina ONNX：仅当模型与后端
        // 都恰好等于旧默认时才迁移，用户显式选择的其他组合保持不变。
        let legacyRerankerName = "bge-reranker-v2-m3-mlx-mxfp8"
        if localRerankerBackend == "mlx",
           localRerankerModel == legacyRerankerName,
           localRerankerModelPath.isEmpty
            || localRerankerModelPath == "models/\(legacyRerankerName)"
            || localRerankerModelPath == legacyRerankerName
            || localRerankerModelPath.hasSuffix("/\(legacyRerankerName)") {
            localRerankerBackend = "onnx"
            localRerankerModel = "jina-reranker-v2-base-multilingual-onnx"
            localRerankerModelPath = "models/jina-reranker-v2-base-multilingual-onnx"
            if ragRerankModel == legacyRerankerName {
                ragRerankModel = "jina-reranker-v2-base-multilingual-onnx"
            }
        }
        if maxTokens == 4096 { maxTokens = 32768 }
        if maxContextTokens < 8192 { maxContextTokens = maxTokens }
        // 旧默认 500/50（bge 512 窗口）迁移到 800/80（nomic 8192 窗口）。
        // 只迁移「恰好等于旧默认值」的组合，用户自定义值保持不变。
        if ragChunkSize == 500, ragChunkOverlap == 50 {
            ragChunkSize = 800
            ragChunkOverlap = 80
        }
        if ragChunkOverlap >= ragChunkSize {
            ragChunkOverlap = max(0, ragChunkSize / 10)
        }
        if ragStorageBackend != "sqlite" { ragStorageBackend = "sqlite" }

        // 旧配置使用 openai / openai_compatible 表示 API 模式。
        let provider = llmProvider.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if provider == "openai" || provider == "openai_compatible" || provider == "openai-compatible" {
            llmProvider = "api"
        }
        let auxProvider = auxLLMProvider.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if auxProvider == "openai" || auxProvider == "openai_compatible" || auxProvider == "openai-compatible" {
            auxLLMProvider = "api"
        }

        localLLMBackend = Self.normalizeLLMBackend(localLLMBackend)
        auxLocalLLMBackend = Self.normalizeLLMBackend(auxLocalLLMBackend)
        localEmbeddingBackend = Self.normalizeEmbeddingBackend(localEmbeddingBackend)
        localRerankerBackend = Self.normalizeEmbeddingBackend(localRerankerBackend)
        if auxMaxContextTokens < 8192 { auxMaxContextTokens = max(8192, auxMaxTokens) }
        if auxMaxContextTokens <= 0 { auxMaxContextTokens = 32768 }
        // MLX 空闲 buffer cache 运行时上限 2GB：旧配置可能残留 4096 等超大值，
        // 加载时一并压回，避免配置文件里的值与实际生效值不一致。
        mlxCacheLimitMB = max(256, min(2_048, mlxCacheLimitMB))
        // Embedding 子批自适应参数：手改 config.json 或旧配置可能越界。
        ragEmbeddingMaxBatch = max(2, min(256, ragEmbeddingMaxBatch))
        ragEmbeddingMinBatch = max(2, min(64, ragEmbeddingMinBatch))
        ragEmbeddingMinBatch = min(ragEmbeddingMinBatch, ragEmbeddingMaxBatch)
        ragEmbeddingInitialBatch = max(ragEmbeddingMinBatch,
                                       min(ragEmbeddingInitialBatch, ragEmbeddingMaxBatch))
        ragEmbeddingBatchStep = max(1, min(16, ragEmbeddingBatchStep))
        ragEmbeddingMemoryLimitMB = max(1_024, min(32_768, ragEmbeddingMemoryLimitMB))
        ragEmbeddingThroughputTolerance = max(0.01,
                                              min(0.5, ragEmbeddingThroughputTolerance))
        ragEmbeddingProbeBatches = max(1, min(16, ragEmbeddingProbeBatches))
        ragEmbeddingProbeSeconds = max(0.5, min(30, ragEmbeddingProbeSeconds))
        ragEmbeddingTokenBudgetMin = max(512, min(65_536, ragEmbeddingTokenBudgetMin))
        ragEmbeddingTokenBudgetMax = max(ragEmbeddingTokenBudgetMin,
                                          min(65_536, ragEmbeddingTokenBudgetMax))
        ragEmbeddingTokenBudgetInitial = max(ragEmbeddingTokenBudgetMin,
                                             min(ragEmbeddingTokenBudgetMax,
                                                 ragEmbeddingTokenBudgetInitial))
        ragEmbeddingTokenBudgetStep = max(128, min(8_192, ragEmbeddingTokenBudgetStep))
        // Matryoshka 截断维只允许模型声明的六档，其它值回落到默认 128。
        // 元素类型不做收口：int8 / fp32 / fp16 / bin 全部由用户选择。
        if !MatryoshkaDimensions.jinaV5.contains(ragMatryoshkaDim) {
            ragMatryoshkaDim = MatryoshkaDimensions.defaultValue
        }
        // 知识图谱检索参数收口：手改 config.json 可能越界。
        ragKGSimilarityFloor = max(0, min(0.9, ragKGSimilarityFloor))
        ragKGEntityTopK = max(1, min(200, ragKGEntityTopK))
        ragKGRelationTopK = max(1, min(200, ragKGRelationTopK))
        ragKGMaxChunksPerHit = max(1, min(20, ragKGMaxChunksPerHit))
        ragKGEntityTokenBudget = max(500, min(20_000, ragKGEntityTokenBudget))
        ragKGRelationTokenBudget = max(500, min(20_000, ragKGRelationTokenBudget))
        ragKGTotalTokenBudget = max(4_096, min(60_000, ragKGTotalTokenBudget))
        // 抽取限额收口：实体/关系按行数上限，描述按字符数上限。
        ragKGMaxEntitiesPerChunk = max(1, min(200, ragKGMaxEntitiesPerChunk))
        ragKGMaxRelationsPerChunk = max(1, min(500, ragKGMaxRelationsPerChunk))
        ragKGMaxDescriptionChars = max(10, min(2_000, ragKGMaxDescriptionChars))
    }

    /// 统一主/辅模型类型：mlx → mlx-lm；llama.cpp / llama_cpp → gguf。
    public static func normalizeLLMBackend(_ raw: String) -> String {
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "", "mlx", "mlx-lm", "mlx_lm":
            return "mlx-lm"
        case "mlx-vlm", "mlx_vlm":
            return "mlx-vlm"
        case "onnx":
            return "onnx"
        case "gguf", "llama_cpp", "llama.cpp", "llama-cpp":
            return "gguf"
        default:
            return raw.lowercased()
        }
    }

    /// 统一 Embedding / Reranker 类型：onnx / fastembed / mlx / gguf。
    public static func normalizeEmbeddingBackend(_ raw: String) -> String {
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "", "onnx":
            return "onnx"
        case "fastembed":
            return "fastembed"
        case "mlx", "mlx_lm", "mlx-lm":
            return "mlx"
        case "gguf", "llama_cpp", "llama.cpp", "llama-cpp":
            return "gguf"
        default:
            return raw.lowercased()
        }
    }

    /// 导出全部配置为字典（save 落盘与测试断言共用）。
    public func export() -> [String: Any] {
        return [
            "base_url": baseURL, "api_key": apiKey, "model": model,
            "temperature": temperature, "max_tokens": maxTokens,
            "top_p": topP, "min_p": minP, "top_k": topK,
            "repetition_penalty": repetitionPenalty,
            "workspace": workspace, "knowledge_base": knowledgeBase,
            "request_timeout": requestTimeout, "max_iterations": maxIterations,
            "max_history": maxHistory, "max_context_tokens": maxContextTokens,
            "context_shrink_ratio": contextShrinkRatio,
            "prompt_batch_size": promptBatchSize,
            "mlx_cache_limit_mb": mlxCacheLimitMB,
            "mlx_memory_limit_mb": mlxMemoryLimitMB,
            "kv_cache_max_tokens": kvCacheMaxTokens,
            "llm_kv_reuse": llmKVReuse,
            "kv_cache_int8": kvCacheInt8,
            "flash_attention": flashAttention,
            "aux_kv_cache_max_tokens": auxKvCacheMaxTokens,
            "aux_llm_kv_reuse": auxKvReuse,
            "aux_kv_cache_int8": auxKvCacheInt8,
            "kernel_max_context_tokens": kernelMaxContextTokens,
            "kernel_raw_turns": kernelRawTurns,
            "kernel_consolidate_every": kernelConsolidateEvery,
            "kernel_max_tool_chars": kernelMaxToolChars,
            "kernel_rag_history_chars": kernelRagHistoryChars,
            "kernel_max_digest_lines": kernelMaxDigestLines,
            "rag_chunk_size": ragChunkSize, "rag_chunk_overlap": ragChunkOverlap,
            "rag_semantic_chunking": ragSemanticChunking,
            "rag_ingest_concurrency": ragIngestConcurrency,
            "rag_verify_fingerprint": ragVerifyFingerprint,
            "rag_vision_concurrency": ragVisionConcurrency,
            "rag_ovis_instances": ragOvisInstances,
            "rag_embedding_min_batch": ragEmbeddingMinBatch,
            "rag_embedding_max_batch": ragEmbeddingMaxBatch,
            "rag_embedding_initial_batch": ragEmbeddingInitialBatch,
            "rag_embedding_batch_step": ragEmbeddingBatchStep,
            "rag_embedding_memory_limit_mb": ragEmbeddingMemoryLimitMB,
            "rag_embedding_throughput_tolerance": ragEmbeddingThroughputTolerance,
            "rag_embedding_probe_batches": ragEmbeddingProbeBatches,
            "rag_embedding_probe_seconds": ragEmbeddingProbeSeconds,
            "rag_embedding_token_budget_min": ragEmbeddingTokenBudgetMin,
            "rag_embedding_token_budget_max": ragEmbeddingTokenBudgetMax,
            "rag_embedding_token_budget_initial": ragEmbeddingTokenBudgetInitial,
            "rag_embedding_token_budget_step": ragEmbeddingTokenBudgetStep,
            "rag_embedding_model": ragEmbeddingModel,
            "rag_rerank_model": ragRerankModel,
            "rag_rerank_enabled": ragRerankEnabled,
            "rag_rerank_top_n": ragRerankTopN,
            "rag_rerank_candidate_multiplier": ragRerankCandidateMultiplier,
            "rag_rerank_timeout_seconds": ragRerankTimeoutSeconds,
            "rag_query_timeout_seconds": ragQueryTimeoutSeconds,
            "rag_ann_enabled": ragANNEnabled,
            "rag_ann_clusters": ragANNClusters,
            "rag_ann_cluster_size": ragANNClusterSize,
            "rag_ann_nprobe": ragANNProbe,
            "rag_ann_rescore": ragANNRescore,
            "rag_ann_min_chunks": ragANNMinChunks,
            "rag_ann_auto_build": ragANNAutoBuild,
            "rag_auto_ingest": ragAutoIngest,
            "rag_pdf_engine": ragPDFEngine.rawValue,
            "rag_epub_ocr_engine": ragEPUBOCREngine.rawValue,
            "rag_ocr_engine": ragOCREngine,
            "rag_ocr_model_name": ragOCRModelName,
            "rag_ocr_model_path": ragOCRModelPath,
            "rag_ocr_mmproj_path": ragOCRMMProjPath,
            "rag_ocr_tokenizer_path": ragOCRTokenizerPath,
            "llm_mmproj_path": llmMmprojPath,
            "llm_tokenizer_path": llmTokenizerPath,
            "aux_mmproj_path": auxMmprojPath,
            "aux_tokenizer_path": auxTokenizerPath,
            "rag_ocr_coreml": ragOCRCoreML,
            "rag_ocr_preprocess": ragOCRPreprocess,
            "rag_ocr_binarize": ragOCRBinarize,
            "llama_gpu_layers": llamaGpuLayers,
            "rag_storage_backend": ragStorageBackend,
            "rag_sqlite_vec_enabled": ragSQLiteVecEnabled,
            "rag_rrf_k": ragRRFK,
            "rag_rerank_threshold": ragRerankThreshold,
            "rag_search_max_chars_per_result": ragSearchMaxCharsPerResult,
            "rag_search_max_total_chars": ragSearchMaxTotalChars,
            "rag_aux_summarize_min_chars": ragAuxSummarizeMinChars,
            "rag_aux_summarize_max_chars": ragAuxSummarizeMaxChars,
            "rag_vector_overfetch": ragVectorOverfetch,
            "rag_fts_overfetch": ragFTSOverfetch,
            "rag_mcp_listen_lan": ragMCPListenLan,
            "rag_mcp_port": ragMCPPort,
            "rag_mcp_enabled": ragMCPEnabled,
            "rag_vector_element_type": ragVectorElementType.rawValue,
            "rag_matryoshka": ragMatryoshka.rawValue,
            "rag_matryoshka_dim": ragMatryoshkaDim,
            "rag_retrieval_strategy": ragRetrievalStrategy.rawValue,
            "rag_kg_similarity_floor": ragKGSimilarityFloor,
            "rag_kg_entity_top_k": ragKGEntityTopK,
            "rag_kg_relation_top_k": ragKGRelationTopK,
            "rag_kg_max_chunks_per_hit": ragKGMaxChunksPerHit,
            "rag_kg_entity_token_budget": ragKGEntityTokenBudget,
            "rag_kg_relation_token_budget": ragKGRelationTokenBudget,
            "rag_kg_total_token_budget": ragKGTotalTokenBudget,
            "rag_kg_max_entities_per_chunk": ragKGMaxEntitiesPerChunk,
            "rag_kg_max_relations_per_chunk": ragKGMaxRelationsPerChunk,
            "rag_kg_max_description_chars": ragKGMaxDescriptionChars,
            "rag_binary_coarse_enabled": ragBinaryCoarseEnabled,
            "rag_binary_coarse_candidates": ragBinaryCoarseCandidates,
            "llm_provider": llmProvider,
            "local_llm_model": localLLMModel,
            "local_llm_backend": localLLMBackend,
            "local_llm_enable_thinking": localLLMEnableThinking,
            "auto_preload_main": autoPreloadMain,
            "auto_download_missing_models": autoDownloadMissingModels,
            "aux_enabled": auxEnabled,
            "aux_llm_provider": auxLLMProvider,
            "aux_local_llm_model": auxLocalLLMModel,
            "aux_local_llm_backend": auxLocalLLMBackend,
            "aux_temperature": auxTemperature,
            "aux_max_tokens": auxMaxTokens,
            "aux_top_p": auxTopP,
            "aux_min_p": auxMinP,
            "aux_top_k": auxTopK,
            "aux_repetition_penalty": auxRepetitionPenalty,
            "aux_request_timeout": auxRequestTimeout,
            "aux_local_llm_enable_thinking": auxLocalLLMEnableThinking,
            "aux_max_context_tokens": auxMaxContextTokens,
            "aux_prompt_batch_size": auxPromptBatchSize,
            "embedding_provider": embeddingProvider,
            "local_embedding_model": localEmbeddingModel,
            "local_embedding_model_path": localEmbeddingModelPath,
            "local_embedding_backend": localEmbeddingBackend,
            "reranker_provider": rerankerProvider,
            "local_reranker_model": localRerankerModel,
            "local_reranker_model_path": localRerankerModelPath,
            "local_reranker_backend": localRerankerBackend,
            "vlm_provider": vlmProvider,
            "local_vlm_model": localVLMModel,
            "local_vlm_backend": localVLMBackend,
            "browser_endpoint": browserEndpoint,
            "always_confirm_code": alwaysConfirmCode,
            "mcp_servers": mcpServers.map {
                ["name": $0.name, "command": $0.command, "args": $0.args, "env": $0.env]
            },
        ]
    }

    public func save(_ path: String) throws {
        let data = try JSONSerialization.data(withJSONObject: export(),
                                              options: [.prettyPrinted, .sortedKeys])
        try AtomicFile.write(data, to: path)
    }

    // MARK: - 配置差异（决定是否需要重建 Agent）

    /// 只影响主 LLM 实例的配置键：这些键变化可以走「只替换主模型」的轻量路径。
    /// 其余键（切片、知识库目录、embedding / reranker 后端与模型、OCR、MCP、
    /// 工具等）都会被 Agent / RAGEngine 在构建时快照，必须整体重建，
    /// 否则会出现「面板设了长切片、运行时仍按旧切片处理」这类不同步。
    public static let mainLLMOnlyKeys: Set<String> = [
        "llm_provider", "base_url", "api_key", "model",
        "temperature", "max_tokens", "top_p", "min_p", "top_k",
        "repetition_penalty", "request_timeout",
        "local_llm_model", "local_llm_backend", "local_llm_enable_thinking",
        "llm_tokenizer_path", "llm_mmproj_path",
        "max_context_tokens", "context_shrink_ratio", "prompt_batch_size",
        "kv_cache_max_tokens", "llm_kv_reuse", "kv_cache_int8", "flash_attention",
        "auto_preload_main",
    ]

    /// 只影响辅助 LLM 槽位的配置键（辅助模型本身按需加载/卸载，槽位开关可原地生效）。
    public static let auxLLMOnlyKeys: Set<String> = [
        "aux_enabled", "aux_llm_provider", "aux_local_llm_model",
        "aux_local_llm_backend", "aux_local_llm_enable_thinking",
        "aux_temperature", "aux_max_tokens", "aux_top_p", "aux_min_p", "aux_top_k",
        "aux_repetition_penalty", "aux_request_timeout",
        "aux_max_context_tokens", "aux_prompt_batch_size", "aux_kv_cache_max_tokens",
        "aux_llm_kv_reuse", "aux_kv_cache_int8", "flash_attention",
        "aux_tokenizer_path", "aux_mmproj_path",
    ]

    /// 与另一份配置逐键比较（基于 `export()` 全量导出），返回值不同的键。
    public func changedKeys(comparedTo other: AgentConfig) -> Set<String> {
        let lhs = export()
        let rhs = other.export()
        var changed: Set<String> = []
        for key in Set(lhs.keys).union(rhs.keys) {
            guard let left = lhs[key], let right = rhs[key] else {
                changed.insert(key)
                continue
            }
            if !Self.configValueEqual(left, right) {
                changed.insert(key)
            }
        }
        return changed
    }

    /// 需要重建 Agent 的差异键：`allowed`（本槽位模型参数）之外的任何差异都要重建。
    /// 返回排序后的键名，便于日志与测试断言。
    public func agentRebuildKeys(comparedTo other: AgentConfig,
                                 allowing allowed: Set<String>) -> [String] {
        changedKeys(comparedTo: other).subtracting(allowed).sorted()
    }

    /// `export()` 只产出 String / Bool / Int / Double / 数组 / 字典，桥接后深比较；
    /// 不用 `String(describing:)` 是因为字典描述顺序不稳定，会把相同配置判成不同。
    private static func configValueEqual(_ lhs: Any, _ rhs: Any) -> Bool {
        if let left = lhs as? NSObject, let right = rhs as? NSObject {
            return left.isEqual(right)
        }
        return String(describing: lhs) == String(describing: rhs)
    }
}
