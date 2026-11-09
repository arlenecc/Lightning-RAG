import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import MLXHuggingFace
import Tokenizers

/// 模型加载完成后报告的信息
struct LoadedModelInfo: Equatable {
    /// 目录名
    var name: String
    /// 目录绝对路径
    var directory: String
    /// config.json 中的 model_type（多模态模型取 text_config）
    var modelType: String
    /// config.json 中的 architectures
    var architecture: String
    /// 量化信息，如 "4 bit"
    var quantization: String
    /// 权重文件总大小（字节）
    var weightsBytes: Int64
    /// 加载耗时（秒）
    var loadDuration: TimeInterval
}

enum LLMEngineError: LocalizedError {
    case notLoaded
    case invalidDirectory(String)

    var errorDescription: String? {
        switch self {
        case .notLoaded:
            return "模型尚未加载，请先选择模型目录并点击「开始加载」。"
        case .invalidDirectory(let reason):
            return "模型目录无效：\(reason)"
        }
    }
}

/// 生成过程中的事件，通过 AsyncStream 推送给界面
enum LLMEvent: Sendable {
    /// 一段流式文本
    case chunk(String)
    /// 生成完成，携带 mlx-swift-lm 的性能统计
    case completed(GenerateCompletionInfo)
    /// 生成失败或被停止
    case failed(String)
}

/// 把非 Sendable 的值（ChatSession）传入 Task 的轻量封装。
/// 安全性由 LLMEngine actor 保证：同一时刻只有一个生成任务访问该 session。
struct UncheckedSendableBox<Value>: @unchecked Sendable {
    let value: Value
    init(_ value: Value) {
        self.value = value
    }
}

/// 封装 mlx-swift-lm 的本地模型加载与流式生成。
///
/// - 加载：`loadModelContainer(from: directory, using:)` 直接读取本地目录中的
///   config.json、chat_template 与 *.safetensors，无需联网。
/// - 生成：通过 `ChatSession.streamDetails(to:)` 流式获取文本与性能统计。
/// - actor 隔离保证同一时刻只有一个生成任务访问 ChatSession。
actor LLMEngine {

    private var session: ChatSession?
    private var container: ModelContainer?
    private var generationTask: Task<Void, Never>?

    private(set) var loadedInfo: LoadedModelInfo?

    // MARK: - 加载

    /// 从本地目录加载模型并创建对话会话
    func load(directory: URL, generateParameters: GenerateParameters) async throws -> LoadedModelInfo {
        let fm = FileManager.default

        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: directory.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw LLMEngineError.invalidDirectory("目录不存在：\(directory.path)")
        }
        guard fm.fileExists(atPath: directory.appendingPathComponent("config.json").path) else {
            throw LLMEngineError.invalidDirectory("目录中缺少 config.json")
        }
        let weightURLs = (try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        let safetensorsURLs = weightURLs.filter { $0.pathExtension == "safetensors" }
        guard !safetensorsURLs.isEmpty else {
            throw LLMEngineError.invalidDirectory("目录中缺少 *.safetensors 权重文件")
        }

        let start = Date()

        // 注册自定义架构（如 spark2_5），使 loadModelContainer 能识别并构建
        await SparkModel.register()

        // 通过 mlx-swift-lm 直接从本地目录加载模型与分词器
        let container = try await loadModelContainer(
            from: directory,
            using: #huggingFaceTokenizerLoader()
        )

        let duration = Date().timeIntervalSince(start)

        // ChatSession 会自动应用模型的 chat template，维护多轮对话与 KV cache
        let session = ChatSession(container, generateParameters: generateParameters)
        self.session = session
        self.container = container
        self.generationTask?.cancel()
        self.generationTask = nil

        let weightsBytes = safetensorsURLs.reduce(into: Int64(0)) { total, url in
            total += Int64((try? fm.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0)
        }
        let info = Self.describeModel(
            directory: directory,
            weightsBytes: weightsBytes,
            loadDuration: duration
        )
        self.loadedInfo = info
        return info
    }

    /// 读取 config.json，提取模型类型、架构与量化信息
    private static func describeModel(
        directory: URL,
        weightsBytes: Int64,
        loadDuration: TimeInterval
    ) -> LoadedModelInfo {
        var modelType = "未知"
        var architecture = ""
        var quantization = "未量化"

        if let data = FileManager.default.contents(atPath: directory.appendingPathComponent("config.json").path),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            // 多模态模型（如 Qwen3.5）的文本配置嵌套在 text_config 中
            let textConfig = json["text_config"] as? [String: Any] ?? json
            modelType = (textConfig["model_type"] as? String) ?? (json["model_type"] as? String) ?? "未知"
            let architectures = (json["architectures"] as? [String])
                ?? (textConfig["architectures"] as? [String]) ?? []
            architecture = architectures.joined(separator: ", ")
            if let bits = (json["quantization"] as? [String: Any])?["bits"] as? Int {
                quantization = "\(bits) bit"
            }
        }

        return LoadedModelInfo(
            name: directory.lastPathComponent,
            directory: directory.path,
            modelType: modelType,
            architecture: architecture,
            quantization: quantization,
            weightsBytes: weightsBytes,
            loadDuration: loadDuration
        )
    }

    /// 卸载模型，释放内存
    func unload() {
        generationTask?.cancel()
        generationTask = nil
        session = nil
        container = nil
        loadedInfo = nil
    }

    // MARK: - 生成

    /// 更新生成参数（温度、最大 token 数等），下一次生成生效
    func update(generateParameters: GenerateParameters) {
        session?.generateParameters = generateParameters
    }

    /// 开始一次流式生成。事件通过 continuation 推送：
    /// `.chunk` 文本片段、`.completed` 性能统计、`.failed` 错误。
    func generate(prompt: String, continuation: AsyncStream<LLMEvent>.Continuation) async {
        generationTask?.cancel()

        guard let session else {
            continuation.yield(.failed(LLMEngineError.notLoaded.localizedDescription))
            continuation.finish()
            return
        }

        let box = UncheckedSendableBox(session)
        generationTask = Task { [box] in
            defer { continuation.finish() }
            var sawCompletion = false
            do {
                for try await event in box.value.streamDetails(to: prompt) {
                    if Task.isCancelled { break }
                    switch event {
                    case .chunk(let text):
                        continuation.yield(.chunk(text))
                    case .info(let info):
                        sawCompletion = true
                        continuation.yield(.completed(info))
                    case .toolCall, .rejectedToolCall:
                        break
                    }
                }
                if !sawCompletion && !Task.isCancelled {
                    continuation.yield(.failed("模型未返回完整结果"))
                }
            } catch {
                if !Task.isCancelled {
                    continuation.yield(.failed(error.localizedDescription))
                }
            }
        }
    }

    /// 停止当前生成
    func stop() {
        generationTask?.cancel()
        generationTask = nil
    }

    /// 清空对话历史与 KV cache（保留模型）
    func clearConversation() async {
        generationTask?.cancel()
        generationTask = nil
        if let session {
            // ChatSession 非 Sendable，通过 UncheckedSendableBox 传入 @concurrent 方法；
            // actor 隔离保证同一时刻只有一个调用方访问该 session
            let box = UncheckedSendableBox(session)
            await box.value.clear()
        }
    }
}
