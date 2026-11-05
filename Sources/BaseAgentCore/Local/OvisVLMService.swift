import Foundation
import CoreGraphics
import CoreImage
import MLXVLM
import MLXLMCommon

/// OvisOCR3 的 VLM 服务。
///
/// OvisOCR3 的 `config.json` 使用 `model_type=qwen3_5` /
/// `Qwen3_5ForConditionalGeneration`（Qwen3.5 后训练），因此可以直接走
/// `mlx-swift-lm` 已注册的 Qwen3.5 VLM 实现。
///
/// 仅当模型确实是未注册的 Ovis2 架构（`model_type` 含 `ovis`）时，
/// `isAvailable` 才返回 false 并在 `recognize` 抛出明确错误，绝不伪造结果。
public final class OvisVLMService: VLMService, @unchecked Sendable {
    public let identifier: String
    public let modelName: String
    public let directory: URL

    private let modelType: String?
    private let preflightError: Error?
    private let state = VLMLoadState()
    /// 多文件并发 ingest 时串行进入同一个 MLX VLM 实例。
    private let gate = AsyncGate()

    public var isAvailable: Bool {
        preflightError == nil
    }

    public init(directory: URL, modelName: String) {
        self.directory = directory
        self.modelName = modelName
        self.identifier = "local:mlx-vlm:\(modelName)"

        let type = LocalModelRegistry.modelType(in: directory)
        self.modelType = type

        if LocalModelRegistry.isOvisArchitecture(directory: directory) {
            preflightError = LocalModelError.unsupportedArchitecture(
                "检测到未注册的 Ovis2 架构（model_type=\(type ?? "unknown")，模型：\(modelName)）。"
                    + "mlx-swift-lm 未提供该架构实现，无法加载 \(directory.path)。"
                    + "OvisOCR3 请使用 model_type=qwen3_5 的 MLX 转换版本。"
            )
        } else if type == nil {
            preflightError = LocalModelError.invalidConfiguration(
                "无法从 \(directory.path)/config.json 读取 model_type 或 architectures"
            )
        } else {
            preflightError = nil
        }
    }

    // MARK: - VLMService

    public func recognize(image: CGImage, prompt: String, maxTokens: Int) async throws -> String {
        try await gate.withLock {
            try await recognizeLocked(image: image, prompt: prompt, maxTokens: maxTokens)
        }
    }

    private func recognizeLocked(image: CGImage, prompt: String,
                                 maxTokens: Int) async throws -> String {
        if let preflightError { throw preflightError }
        guard let modelType else {
            throw LocalModelError.invalidConfiguration("VLM 模型缺少 model_type：\(directory.path)")
        }

        let supported = await VLMTypeRegistry.shared.contains(modelType)
        guard supported else {
            throw LocalModelError.unsupportedArchitecture(
                "model_type=\(modelType) 未在 mlx-swift-lm 的 VLMTypeRegistry 中注册"
                    + "（模型：\(modelName)，目录：\(directory.path)）"
            )
        }

        let container: ModelContainer
        do {
            container = try await state.container(directory: directory)
        } catch let error as LocalModelError {
            throw error
        } catch {
            throw LocalModelError.runtime(
                "VLMModelFactory 加载 \(modelName) 失败：\(error.localizedDescription)"
            )
        }

        // 与 Vision 路径对称：把图像输入构建限制在 autoreleasepool 内，
        // 避免每页 OCR 在 Swift 并发工作线程上累积 autoreleased Core Image 对象
        // （长批量 OCR / 识别失败导致更多页走 OCR 时内存会持续上涨直到崩溃）。
        let userInput: UserInput = autoreleasepool {
            let ciImage = CIImage(cgImage: image)
            return UserInput(chat: [
                .user(prompt, images: [.ciImage(ciImage)])
            ])
        }

        do {
            let input = try await container.prepare(input: userInput)
            let parameters = GenerateParameters(
                maxTokens: max(1, maxTokens),
                temperature: 0,
                topP: 1.0,
                topK: 0,
                minP: 0.0,
                repetitionPenalty: nil
            )
            let stream = try await container.generate(input: input, parameters: parameters)

            var text = ""
            for await generation in stream {
                try Task.checkCancellation()
                switch generation {
                case .chunk(let chunk):
                    text += chunk
                case .info(_), .toolCall(_), .rejectedToolCall(_):
                    // OCR 场景不消费思考/工具信息，只保留最终正文。
                    break
                }
            }
            // `input` 为局部变量，函数返回时 ARC 自然释放其持有的 MLX 张量，
            // 无需手动置 nil（LMInput 非可选类型不能赋 nil，且它不会跨函数调用滞留）。
            return Self.stripThinking(text).trimmingCharacters(in: .whitespacesAndNewlines)
        } catch is CancellationError {
            throw AgentError.cancelled
        } catch let error as LocalModelError {
            throw error
        } catch {
            throw LocalModelError.runtime(
                "\(modelName) OCR 推理失败：\(error.localizedDescription)"
            )
        }
    }

    /// 释放 OCR 模型权重；provider 保留，下次识别时按需重载。
    public func unload() async {
        await state.unload()
    }

    // MARK: - 内部工具

    /// 去掉 Qwen/Ovis 可能输出的 think 片段，只保留最终 Markdown 正文。
    private static func stripThinking(_ text: String) -> String {
        var result = text
        let pattern = "<think>[\\s\\S]*?</think>"
        // NSRegularExpression 是 autoreleased 的 Foundation 对象，长批量推理下
        // 需在 autoreleasepool 内创建，否则随并发工作线程累积。
        let stripped = autoreleasepool {
            if let regex = try? NSRegularExpression(pattern: pattern, options: []) {
                let range = NSRange(result.startIndex..<result.endIndex, in: result)
                return regex.stringByReplacingMatches(in: result, options: [], range: range, withTemplate: "")
            }
            return result
        }
        result = stripped
            .replacingOccurrences(of: "<think>", with: "")
            .replacingOccurrences(of: "</think>", with: "")
        return result
    }
}

/// VLM 容器只加载一次；失败后缓存错误，避免重复加载数 GB 权重。
actor VLMLoadState {
    private var container: ModelContainer?
    private var failure: Error?

    func container(directory: URL) async throws -> ModelContainer {
        if let container { return container }
        if let failure { throw failure }

        do {
            let container = try await VLMModelFactory.shared.loadContainer(
                from: directory,
                using: LocalTokenizerLoader()
            )
            self.container = container
            return container
        } catch {
            self.failure = error
            throw error
        }
    }

    func unload() {
        container = nil
        failure = nil
    }
}
