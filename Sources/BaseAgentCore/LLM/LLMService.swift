import Foundation

/// 流式事件：与 Python 侧 `StreamEvent` 对齐。
public struct LLMStreamEvent: Sendable {
    public enum Kind: Sendable {
        case content
        case reasoning
        case done
    }

    public var kind: Kind
    public var content: String
    /// 仅在 done 事件携带。
    public var toolCalls: [ToolCall]
    public var usage: [String: Int]
    /// 本地 MLX 后端用于重建逐 token 历史；远端 HTTP 后端为空。
    public var rawContent: String
    /// 本轮输出被截断（工具调用未生成完整）：需要重新生成而不是当作最终答复。
    public var truncated: Bool

    public init(kind: Kind, content: String = "", toolCalls: [ToolCall] = [],
                usage: [String: Int] = [:], rawContent: String = "",
                truncated: Bool = false) {
        self.kind = kind
        self.content = content
        self.toolCalls = toolCalls
        self.usage = usage
        self.rawContent = rawContent
        self.truncated = truncated
    }
}

/// 单次请求覆盖项；不改变服务实例的默认配置。
public struct LLMRequestOptions: Sendable, Equatable {
    /// 覆盖服务构造时的 thinking 开关；nil 表示沿用服务默认。
    public var enableThinking: Bool?
    /// 独立会话：不读取也不写入跨轮 KV/历史，供并行的单次请求使用。
    public var isolated: Bool

    public init(enableThinking: Bool? = nil, isolated: Bool = false) {
        self.enableThinking = enableThinking
        self.isolated = isolated
    }

    public static let `default` = LLMRequestOptions()
}

/// LLM 后端统一接口：远端 OpenAI 兼容服务与本地 MLX 模型都实现它。
public protocol LLMService: AnyObject {
    var identifier: String { get }
    var supportsTools: Bool { get }
    var contextWindow: Int { get }
    /// 同一实例是否允许并发执行多个 `isolated` 一次性请求。
    ///
    /// 辅助模型的长文本任务（章节/全书摘要）会把分片并行提交；只有后端能为
    /// 每个请求提供独立 KV/context 时才为 true。串行后端返回 false，批处理会
    /// 自动退化为逐条执行，避免在内部互斥锁前堆积无意义的并发任务。
    var supportsConcurrentIsolatedRequests: Bool { get }
    func chatStream(messages: [ChatMessage],
                    tools: [JSONValue],
                    temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error>
    /// 带请求覆盖项的流式调用。未实现的远端后端按默认选项处理。
    func chatStream(messages: [ChatMessage],
                    tools: [JSONValue],
                    temperature: Double,
                    options: LLMRequestOptions) -> AsyncThrowingStream<LLMStreamEvent, Error>
    /// 批量摘要生成（可选能力）。支持时把整批请求合成一次模型调用生成；
    /// 返回 nil 表示不支持，调用方回退到逐条路径。结果按请求顺序对齐。
    func generateSummariesBatch(
        requests: [(system: String, prompt: String)],
        temperature: Double,
        maxTokensPerRow: Int
    ) async throws -> [String]?
}

public extension LLMService {
    var supportsTools: Bool { true }
    var contextWindow: Int { 32768 }
    var supportsConcurrentIsolatedRequests: Bool { false }

    func chatStream(messages: [ChatMessage],
                    tools: [JSONValue],
                    temperature: Double,
                    options: LLMRequestOptions) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        chatStream(messages: messages, tools: tools, temperature: temperature)
    }

    func generateSummariesBatch(
        requests: [(system: String, prompt: String)],
        temperature: Double,
        maxTokensPerRow: Int
    ) async throws -> [String]? {
        nil
    }
}

/// 持有网络连接池的 LLM 后端在卸载时需要显式释放。
public protocol ClosableLLMService: AnyObject {
    func close()
}

/// 异步卸载的本地 LLM：`close()` 只是发起释放，退出流程必须等它真正完成，
/// 否则 MLX 的 GPU 工作会与进程退出（销毁 command encoder）并发而崩溃。
public protocol AsyncUnloadableLLMService: AnyObject {
    func unloadAndWait() async
}

/// 支持主动加载权重到内存的本地 LLM。
public protocol PreloadableLLMService: AnyObject {
    func preload() async throws
}

/// 支持用稳定 system prompt 预填充 KV cache 的本地 LLM。
///
/// 预热只做 prefill，不生成 token；正式请求可以直接在这段前缀后追加。
public protocol WarmupLLMService: AnyObject {
    func warmup(systemPrompt: String, tools: [JSONValue]) async throws
}

/// 持有跨轮 KV cache 的本地 LLM：历史被折叠/回滚时丢弃缓存。
public protocol KVCacheInvalidating: AnyObject {
    func invalidateKVCache()
}

/// 尚未加载任何模型时的占位实现：调用会得到可诊断的错误。
public final class UnloadedLLMService: LLMService, @unchecked Sendable {
    public let identifier = "unloaded"
    public let supportsTools = false
    public let contextWindow = 0

    public init() {}

    public func chatStream(messages: [ChatMessage],
                           tools: [JSONValue],
                           temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: AgentError.llm(
                "当前没有已加载的主模型，请打开右上角「配置」窗口加载模型"))
        }
    }
}
