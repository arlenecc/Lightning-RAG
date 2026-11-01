import Foundation
import MLX
import MLXVLM
import MLXLMCommon
import CoreImage
import ImageIO

/// 用 MLX-VLM 容器承载主/辅 LLM。
///
/// 与 OCR 的 `OvisVLMService` 共用加载器，但走完整的 chat 流程：
/// 支持工具定义、thinking 标签拆分与流式输出。每个请求重建完整 prompt，
/// 不做跨轮 KV 复用（VLM 容器的多模态 prepare 与纯文本缓存不兼容）。
public final class MLXVLMLLMService: LLMService, PreloadableLLMService,
                                     ClosableLLMService, AsyncUnloadableLLMService,
                                     @unchecked Sendable {
    public let directory: URL
    public let modelName: String
    public let contextWindow: Int
    public let maxTokens: Int
    public let topP: Double
    public let minP: Double
    public let topK: Int
    public let repetitionPenalty: Double
    public let enableThinking: Bool

    private let state = VLMLoadState()
    private let gate = AsyncGate()

    public var identifier: String { "local:mlx-vlm:\(modelName)" }
    public var supportsTools: Bool { true }

    public init(directory: URL,
                modelName: String,
                maxTokens: Int,
                topP: Double,
                minP: Double,
                topK: Int,
                repetitionPenalty: Double,
                enableThinking: Bool,
                contextWindow: Int) {
        self.directory = directory
        self.modelName = modelName
        self.maxTokens = max(1, maxTokens)
        self.topP = topP
        self.minP = minP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.enableThinking = enableThinking
        self.contextWindow = max(1_024, contextWindow)
    }

    public func preload() async throws {
        let type = LocalModelRegistry.modelType(in: directory)
        guard let type else {
            throw LocalModelError.invalidConfiguration(
                "无法读取 VLM 模型类型：\(directory.path)/config.json")
        }
        guard await VLMTypeRegistry.shared.contains(type) else {
            throw LocalModelError.unsupportedArchitecture(
                "model_type=\(type) 未在 mlx-swift-lm 的 VLMTypeRegistry 中注册")
        }
        _ = try await state.container(directory: directory)
    }

    public func close() {
        Task { await state.unload() }
    }

    /// 等待 GPU 队列排空与容器释放；退出与临时摘要会话结束都依赖它。
    public func unloadAndWait() async {
        MLX.Stream.gpu.synchronize()
        await state.unload()
        MLX.Stream.gpu.synchronize()
    }

    public func chatStream(messages: [ChatMessage],
                           tools: [JSONValue],
                           temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        chatStream(messages: messages, tools: tools, temperature: temperature,
                   options: .default)
    }

    public func chatStream(messages: [ChatMessage],
                           tools: [JSONValue],
                           temperature: Double,
                           options: LLMRequestOptions) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        makeStream(messages: messages, tools: tools, temperature: temperature,
                   enableThinking: options.enableThinking ?? enableThinking)
    }

    private func makeStream(messages: [ChatMessage],
                            tools: [JSONValue],
                            temperature: Double,
                            enableThinking: Bool) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await gate.withLock {
                        let container = try await state.container(directory: directory)
                        let mlxMessages = makeVLMMessages(messages)
                        let toolSpecs = MLXLLMService.makeToolSpecs(tools)
                        let context: [String: any Sendable] = [
                            "enable_thinking": enableThinking
                        ]
                        let imageCount = messages.reduce(0) {
                            $0 + (($1.images?.count) ?? 0)
                        }
                        if imageCount > 0 {
                            AgentLog.info("[mlx-vlm] 多模态请求：\(imageCount) 张图像")
                        }
                        let input = try await container.prepare(input: UserInput(
                            chat: mlxMessages,
                            tools: toolSpecs,
                            additionalContext: context
                        ))
                        let parameters = GenerateParameters(
                            maxTokens: maxTokens,
                            temperature: Float(temperature),
                            topP: Float(topP),
                            topK: topK,
                            minP: Float(minP),
                            repetitionPenalty: Float(repetitionPenalty)
                        )
                        let stream = try await container.generate(input: input,
                                                                  parameters: parameters)
                        var splitter = ThinkSplitter(startInThink: false)
                        var calls: [ToolCall] = []
                        var usage: [String: Int] = [:]
                        var raw = ""

                        for await generation in stream {
                            try Task.checkCancellation()
                            switch generation {
                            case .chunk(let chunk):
                                raw += chunk
                                for piece in splitter.feed(chunk) where !piece.text.isEmpty {
                                    continuation.yield(LLMStreamEvent(
                                        kind: piece.reasoning ? .reasoning : .content,
                                        content: piece.text
                                    ))
                                }
                            case .info(let info):
                                usage = [
                                    "prompt_tokens": info.promptTokenCount,
                                    "completion_tokens": info.generationTokenCount,
                                    "total_tokens": info.promptTokenCount + info.generationTokenCount,
                                    "tokens_per_second": Int(info.tokensPerSecond),
                                ]
                            case .toolCall(let call):
                                calls.append(MLXLLMService.toToolCall(call))
                            case .rejectedToolCall(_):
                                break
                            }
                        }

                        for piece in splitter.flush() where !piece.text.isEmpty {
                            continuation.yield(LLMStreamEvent(
                                kind: piece.reasoning ? .reasoning : .content,
                                content: piece.text
                            ))
                        }
                        continuation.yield(LLMStreamEvent(
                            kind: .done,
                            content: "",
                            toolCalls: calls,
                            usage: usage,
                            rawContent: raw
                        ))
                        continuation.finish()
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - 多模态消息转换

    /// 把聊天消息映射为 MLXVLM 的 `Chat.Message`，并在带图的用户消息上挂载图像。
    /// 结构与 `MLXLLMService.makeChatMessages` 完全一致（直接复用），仅把携带图像
    /// 的用户消息替换为 `.user(content, images:)`：图像按出现顺序与对话模板里的
    /// `<image>` 占位符一一对应（交由 MLXVLM 的 prepare 处理）。
    private func makeVLMMessages(_ messages: [ChatMessage]) -> [Chat.Message] {
        let base = MLXLLMService.makeChatMessages(messages)
        return zip(messages, base).map { message, mlx in
            guard message.role == .user,
                  let images = message.images, !images.isEmpty,
                  let vlmImages = vlmImages(from: images) else {
                return mlx
            }
            return .user(message.content, images: vlmImages)
        }
    }

    /// 把 JPEG/PNG 字节转为 MLXVLM 可消费的图像；解码失败的字节被跳过。
    /// 全部失败（或没有可解码的）时返回 nil，调用方回落为纯文本消息。
    private func vlmImages(from dataList: [Data]) -> [UserInput.Image]? {
        var result: [UserInput.Image] = []
        for data in dataList {
            guard let cgImage = cgImage(from: data) else { continue }
            result.append(.ciImage(CIImage(cgImage: cgImage)))
        }
        return result.isEmpty ? nil : result
    }

    private func cgImage(from data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }
}
