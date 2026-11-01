import Foundation
import CoreGraphics
import OnnxRuntimeBindings

/// 本地 ONNX 视觉语言模型（VLM）服务，用于自定义 OCR 的「ONNX」后端。
///
/// 约定（与设置窗口里选择的「单一合并 ONNX 图」一致）：
/// - 单个 `.onnx` 文件，encoder + decoder 合一；
/// - 输入：`input_ids` / `attention_mask`（可选）/ `position_ids`（可选）/
///   `pixel_values`（图像）/ `past`（可选，单一合并 KV 张量）；
/// - 输出：`logits` + `present`（与 `past` 对应的合并 KV 张量，可选）。
///
/// 两种 KV 模式自动识别：
/// 1. 图含**单一** `past`/`present` → 启用 KV 复用（增量解码），更快；
///    此时需要 `config.json` 提供 `num_hidden_layers` / `num_attention_heads` /
///    `num_key_value_heads` / `head_dim`（或 `hidden_size`），用于推导合并 KV 维度。
/// 2. 图**不含** `past` → 每步重跑整段序列（通用、无需猜形状，但更慢）。
///
/// 多分层 `past_key_values.*`（optimum 风格）暂不在此支持，请改用合并 KV 的导出。
///
/// 图像预处理与文本布局可由模型目录的 `config.json` 覆盖：
/// `image_size`、`mean`、`std`、`image_token`、`num_image_tokens`、
/// `prompt_prefix`、`prompt_suffix`、`add_bos`。
public final class ONNXVLMService: VLMService, @unchecked Sendable {
    public let identifier: String
    public var isAvailable: Bool {
        stateLock.lock()
        let failure = loadFailure
        stateLock.unlock()
        return failure == nil
    }

    private let modelLocation: URL
    private let tokenizerDirectory: URL?
    private let modelName: String
    private let contextWindow: Int
    private let defaultMaxTokens: Int

    private let stateLock = NSLock()
    private var loaded: ONNXVLMLoaded?
    private var loadFailure: Error?

    public init(
        modelLocation: URL,
        tokenizerDirectory: URL? = nil,
        modelName: String,
        contextWindow: Int = 4_096,
        defaultMaxTokens: Int = 2_048
    ) {
        self.modelLocation = modelLocation
        self.tokenizerDirectory = tokenizerDirectory
        self.modelName = modelName
        self.contextWindow = max(256, contextWindow)
        self.defaultMaxTokens = max(64, defaultMaxTokens)
        self.identifier = "local:onnx-vlm:\(modelName)"
    }

    public func recognize(image: CGImage, prompt: String, maxTokens: Int) async throws -> String {
        let loaded = try await ensureLoaded()
        let budget = max(64, min(max(64, maxTokens), defaultMaxTokens))
        return try await Task.detached(priority: .userInitiated) {
            try loaded.recognize(image: image, prompt: prompt, maxTokens: budget,
                                 contextWindow: self.contextWindow)
        }.value
    }

    public func unload() async {
        stateLock.lock()
        loaded = nil
        loadFailure = nil
        stateLock.unlock()
    }

    private func ensureLoaded() async throws -> ONNXVLMLoaded {
        stateLock.lock()
        if let loaded { let value = loaded; stateLock.unlock(); return value }
        if let failure = loadFailure { stateLock.unlock(); throw failure }
        stateLock.unlock()
        do {
            let value = try await ONNXVLMLoaded.load(
                modelLocation: modelLocation,
                tokenizerDirectory: tokenizerDirectory,
                modelName: modelName)
            stateLock.lock()
            loaded = value
            stateLock.unlock()
            return value
        } catch {
            stateLock.lock()
            loadFailure = error
            stateLock.unlock()
            throw error
        }
    }
}

// MARK: - 加载

private final class ONNXVLMLoaded: @unchecked Sendable {
    let session: OrtSession
    let tokenizer: LocalTokenizer
    let graph: VLMDecoderGraph
    let eosTokens: Set<Int>
    let vlm: VLMGraphConfig

    private init(session: OrtSession, tokenizer: LocalTokenizer, graph: VLMDecoderGraph,
                 eosTokens: Set<Int>, vlm: VLMGraphConfig) {
        self.session = session
        self.tokenizer = tokenizer
        self.graph = graph
        self.eosTokens = eosTokens
        self.vlm = vlm
    }

    static func load(modelLocation: URL, tokenizerDirectory: URL?, modelName: String) async throws -> ONNXVLMLoaded {
        let modelURL = try LocalModelRegistry.resolveONNXModel(in: modelLocation, kind: .vlm)
        let modelDir = modelURL.pathExtension.lowercased() == "onnx"
            ? modelURL.deletingLastPathComponent() : modelLocation
        let tokenizer = try await LocalTokenizer.load(directory: tokenizerDirectory ?? modelDir)

        let vlm = VLMGraphConfig.load(directory: modelDir)

        guard let imageTokenId = tokenizer.base.convertTokenToId(vlm.imageToken),
              imageTokenId >= 0 else {
            throw LocalModelError.invalidConfiguration(
                "tokenizer 未注册图像占位符 \(vlm.imageToken)（OCR ONNX 模型：\(modelName)）；"
                    + "请在模型目录 config.json 设置正确的 image_token")
        }

        let session = try OrtSession(modelPath: modelURL.path)
        let graph = try VLMDecoderGraph(session: session, directory: modelDir,
                                        vlm: vlm, imageTokenId: imageTokenId)

        // 约定要求 pixel_values；缺失则明确报错而不是偷偷降级。
        guard graph.pixelValuesInput != nil else {
            throw LocalModelError.invalidConfiguration(
                "ONNX VLM 图缺少 pixel_values 输入（\(modelURL.path)），无法接入图像；"
                    + "请使用 encoder+decoder 合一、含 pixel_values 的导出")
        }

        var eos = Set<Int>()
        if let value = tokenizer.base.eosTokenId, value >= 0 { eos.insert(value) }
        if let data = try? Data(contentsOf: modelDir.appendingPathComponent("generation_config.json")),
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let value = object["eos_token_id"] as? Int { eos.insert(value) }
            else if let values = object["eos_token_id"] as? [Int] { eos.formUnion(values) }
        }

        AgentLog.info("[onnx-vlm] 已加载 \(modelName)：\(modelURL.lastPathComponent) "
            + "KV=\(graph.mode) size=\(vlm.imageSize)")

        return ONNXVLMLoaded(session: session, tokenizer: tokenizer, graph: graph,
                             eosTokens: eos, vlm: vlm)
    }

    // MARK: - 推理

    func recognize(image: CGImage, prompt: String, maxTokens: Int,
                  contextWindow: Int) throws -> String {
        // 1) 图像预处理 → pixel_values
        let size = vlm.imageSize
        let pixels = try VLMImagePreprocessor.pixelValues(
            from: image, size: CGSize(width: size, height: size),
            mean: vlm.mean, std: vlm.std)

        // 2) 构造输入序列：可选 BOS + N 个图像占位符 + 指令（前缀/后缀可选）
        var ids: [Int] = []
        if vlm.addBOS, let bos = tokenizer.base.bosTokenId, bos >= 0 { ids.append(bos) }
        ids.append(contentsOf: Array(repeating: graph.imageTokenId, count: vlm.numImageTokens))
        if !vlm.promptPrefix.isEmpty {
            ids.append(contentsOf: tokenizer.encode(vlm.promptPrefix, addSpecialTokens: false))
        }
        ids.append(contentsOf: tokenizer.encode(prompt, addSpecialTokens: false))
        if !vlm.promptSuffix.isEmpty {
            ids.append(contentsOf: tokenizer.encode(vlm.promptSuffix, addSpecialTokens: false))
        }
        guard !ids.isEmpty else {
            throw LocalModelError.invalidConfiguration("OCR ONNX 输入序列为空（图像占位符或指令缺失）")
        }
        guard ids.count <= contextWindow - 8 else {
            throw LocalModelError.runtime(
                "OCR ONNX 输入 \(ids.count) tokens 超过上下文上限 \(contextWindow)")
        }

        // 更新 pixel_values 张量内容（图像每次不同）
        let pixelValue = try OrtValue(float32: pixels, shape: [1, 3, size, size])

        // 3) 自回归解码
        let generated = try graph.generate(
            promptIDs: ids,
            pixelValues: pixelValue,
            eosTokens: eosTokens,
            maxTokens: maxTokens,
            contextWindow: contextWindow)

        let raw = tokenizer.decode(generated, skipSpecialTokens: true)
        return Self.stripThinking(raw).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// 去掉可能的 think 片段，只保留最终 Markdown 正文。
    private static func stripThinking(_ text: String) -> String {
        var result = text
        if let regex = try? NSRegularExpression(pattern: "<think>[\\s\\S]*?</think>", options: []) {
            let range = NSRange(result.startIndex..<result.endIndex, in: result)
            result = regex.stringByReplacingMatches(in: result, options: [], range: range, withTemplate: "")
        }
        return result
            .replacingOccurrences(of: "<think>", with: "")
            .replacingOccurrences(of: "</think>", with: "")
    }
}

// MARK: - 图适配

/// ONNX VLM 解码图：封装 KV 复用 / 重跑两种模式，对外暴露 `generate`。
private final class VLMDecoderGraph: @unchecked Sendable {
    let session: OrtSession
    let inputIDs: String
    let attentionMask: String?
    let positionIDs: String?
    let pixelValuesInput: String?
    let logitsOutput: String
    let imageTokenId: Int

    enum Mode: Sendable {
        /// 单一合并 `past`/`present`，增量解码。
        case merged(past: String, present: String, lastDim: Int)
        /// 无 `past`，每步重跑整段序列。
        case rerun
    }

    let mode: Mode

    init(session: OrtSession, directory: URL, vlm: VLMGraphConfig, imageTokenId: Int) throws {
        self.session = session
        self.imageTokenId = imageTokenId
        guard let inputIDs = session.inputNames.first(where: {
            $0 == "input_ids" || $0.hasSuffix("input_ids")
        }) else {
            throw LocalModelError.invalidConfiguration(
                "ONNX VLM 图没有 input_ids 输入（\(session.modelPath)）")
        }
        guard let logits = session.outputNames.first(where: {
            $0 == "logits" || $0.contains("logits")
        }) else {
            throw LocalModelError.invalidConfiguration(
                "ONNX VLM 图没有 logits 输出（\(session.modelPath)）")
        }
        self.inputIDs = inputIDs
        self.logitsOutput = logits
        self.attentionMask = session.inputNames.first {
            $0 == "attention_mask" || $0.hasSuffix("attention_mask")
        }
        self.positionIDs = session.inputNames.first {
            $0 == "position_ids" || $0.hasSuffix("position_ids")
        }
        self.pixelValuesInput = session.inputNames.first {
            $0 == "pixel_values" || $0.hasSuffix("pixel_values")
        }

        let pastInputs = session.inputNames.filter { $0.lowercased().contains("past") }
        let presentOutputs = session.outputNames.filter { $0.lowercased().contains("present") }

        if pastInputs.count == 1, let past = pastInputs.first,
           let present = presentOutputs.first {
            // 合并 KV：推导 lastDim = 2 * layers * kvHeads * headDim
            let arch = VLMArchitecture(directory: directory)
            guard let arch,
                  let lastDim = arch.mergedKVLastDim else {
                throw LocalModelError.invalidConfiguration(
                    "ONNX VLM 使用合并 KV，但缺少 config.json 的 num_hidden_layers / "
                    + "num_attention_heads / num_key_value_heads / head_dim（或 hidden_size），"
                    + "无法推导 KV 维度（模型：\(session.modelPath)）")
            }
            self.mode = .merged(past: past, present: present, lastDim: lastDim)
        } else if pastInputs.isEmpty {
            self.mode = .rerun
        } else {
            throw LocalModelError.invalidConfiguration(
                "ONNX VLM 仅支持单一合并 past/present；该图含 \(pastInputs.count) 个分层 "
                + "past_key_values 输入（\(pastInputs.first ?? "")…）。请改用合并 KV 的导出")
        }
    }

    /// 贪心解码：返回生成的 token 序列（不含 prompt 与 EOS）。
    func generate(promptIDs: [Int], pixelValues: OrtValue, eosTokens: Set<Int>,
                  maxTokens: Int, contextWindow: Int) throws -> [Int] {
        switch mode {
        case let .merged(past, present, lastDim):
            return try generateMerged(promptIDs: promptIDs, pixelValues: pixelValues,
                                      pastInput: past, presentOutput: present,
                                      lastDim: lastDim, eosTokens: eosTokens,
                                      maxTokens: maxTokens, contextWindow: contextWindow)
        case .rerun:
            return try generateRerun(promptIDs: promptIDs, pixelValues: pixelValues,
                                     eosTokens: eosTokens,
                                     maxTokens: maxTokens, contextWindow: contextWindow)
        }
    }

    // MARK: 合并 KV 增量解码

    private func generateMerged(promptIDs: [Int], pixelValues: OrtValue,
                                pastInput: String, presentOutput: String, lastDim: Int,
                                eosTokens: Set<Int>, maxTokens: Int,
                                contextWindow: Int) throws -> [Int] {
        var past = try OrtValue(float32: [], shape: [1, 0, lastDim])
        var pastLength = 0
        let chunkSize = 256

        // Prefill：分块喂入 prompt，累积 KV。
        var index = 0
        var lastLogits: [Float] = []
        while index < promptIDs.count {
            let end = min(index + chunkSize, promptIDs.count)
            let chunk = Array(promptIDs[index..<end])
            lastLogits = try runMerged(chunk: chunk, past: &past, pastInput: pastInput,
                                       presentOutput: presentOutput, pastLength: pastLength,
                                       pixelValues: pixelValues)
            pastLength += chunk.count
            index = end
        }

        var generated: [Int] = []
        let budget = min(maxTokens, contextWindow - promptIDs.count - 8)
        while generated.count < budget {
            let token = argmax(lastLogits)
            if eosTokens.contains(token) { break }
            generated.append(token)
            lastLogits = try runMerged(chunk: [token], past: &past, pastInput: pastInput,
                                       presentOutput: presentOutput, pastLength: pastLength,
                                       pixelValues: pixelValues)
            pastLength += 1
        }
        return generated
    }

    private func runMerged(chunk: [Int], past: inout OrtValue, pastInput: String,
                           presentOutput: String, pastLength: Int,
                           pixelValues: OrtValue) throws -> [Float] {
        let length = pastLength + chunk.count
        var inputs: [String: OrtValue] = [:]
        inputs[inputIDs] = try OrtValue(int64: chunk.map(Int64.init), shape: [1, chunk.count])
        if let attentionMask {
            inputs[attentionMask] = try OrtValue(
                int64: Array(repeating: 1, count: length), shape: [1, length])
        }
        if let positionIDs {
            inputs[positionIDs] = try OrtValue(
                int64: (pastLength..<length).map(Int64.init), shape: [1, chunk.count])
        }
        inputs[pastInput] = past
        inputs[pixelValuesInput!] = pixelValues

        let outputs = try session.run(inputs: inputs, outputNames: session.outputNames)
        guard let logitsValue = outputs[logitsOutput],
              let present = outputs[presentOutput] else {
            throw LocalModelError.runtime("ONNX VLM 输出缺少 logits 或 present")
        }
        past = present
        return try Self.lastLogits(logitsValue)
    }

    // MARK: 无 KV 重跑整段

    private func generateRerun(promptIDs: [Int], pixelValues: OrtValue,
                               eosTokens: Set<Int>, maxTokens: Int,
                               contextWindow: Int) throws -> [Int] {
        var ids = promptIDs
        var generated: [Int] = []
        let budget = min(maxTokens, contextWindow - promptIDs.count - 8)
        while generated.count < budget {
            let logits = try runRerun(ids: ids, pixelValues: pixelValues)
            let token = argmax(logits)
            if eosTokens.contains(token) { break }
            generated.append(token)
            ids.append(token)
        }
        return generated
    }

    private func runRerun(ids: [Int], pixelValues: OrtValue) throws -> [Float] {
        var inputs: [String: OrtValue] = [:]
        inputs[inputIDs] = try OrtValue(int64: ids.map(Int64.init), shape: [1, ids.count])
        if let attentionMask {
            inputs[attentionMask] = try OrtValue(
                int64: Array(repeating: 1, count: ids.count), shape: [1, ids.count])
        }
        if let positionIDs {
            inputs[positionIDs] = try OrtValue(
                int64: (0..<ids.count).map(Int64.init), shape: [1, ids.count])
        }
        inputs[pixelValuesInput!] = pixelValues
        let outputs = try session.run(inputs: inputs, outputNames: session.outputNames)
        guard let logitsValue = outputs[logitsOutput] else {
            throw LocalModelError.runtime("ONNX VLM 输出缺少 logits")
        }
        return try Self.lastLogits(logitsValue)
    }

    // MARK: 工具

    private static func lastLogits(_ value: OrtValue) throws -> [Float] {
        let tensor = try value.floatTensor()
        guard let vocab = tensor.shape.last, vocab > 0 else {
            throw LocalModelError.runtime("ONNX VLM logits 形状异常：\(tensor.shape)")
        }
        let rows = tensor.values.count / vocab
        guard rows > 0 else { throw LocalModelError.runtime("ONNX VLM logits 为空") }
        let offset = (rows - 1) * vocab
        return Array(tensor.values[offset..<(offset + vocab)])
    }

    private func argmax(_ logits: [Float]) -> Int {
        guard !logits.isEmpty else { return 0 }
        var best = 0
        var bestValue = logits[0]
        for index in 1..<logits.count where logits[index] > bestValue {
            bestValue = logits[index]
            best = index
        }
        return best
    }
}

// MARK: - 架构信息（推导合并 KV 维度）

private struct VLMArchitecture {
    var layerCount: Int
    var kvHeads: Int
    var headDim: Int

    var mergedKVLastDim: Int? {
        guard layerCount > 0, kvHeads > 0, headDim > 0 else { return nil }
        return 2 * layerCount * kvHeads * headDim
    }

    init?(directory: URL) {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        let layers = (object["num_hidden_layers"] as? Int)
            ?? (object["num_layers"] as? Int)
            ?? (object["n_layer"] as? Int)
        let heads = (object["num_attention_heads"] as? Int)
            ?? (object["n_head"] as? Int)
        guard let layers, layers > 0, let heads, heads > 0 else { return nil }
        let kvHeads = (object["num_key_value_heads"] as? Int)
            ?? (object["n_head_kv"] as? Int)
            ?? heads
        let hidden = (object["hidden_size"] as? Int) ?? (object["n_embd"] as? Int) ?? 0
        let headDim = (object["head_dim"] as? Int)
            ?? (hidden > 0 ? hidden / heads : 0)
        guard kvHeads > 0, headDim > 0 else { return nil }
        self.layerCount = layers
        self.kvHeads = kvHeads
        self.headDim = headDim
    }
}

// MARK: - 图参数（可由模型目录 config.json 覆盖）

private struct VLMGraphConfig: Sendable {
    var imageSize: Int = 448
    var mean: [Float] = [0.485, 0.456, 0.406]
    var std: [Float] = [0.229, 0.224, 0.225]
    var imageToken: String = "<image>"
    var numImageTokens: Int = 256
    var promptPrefix: String = ""
    var promptSuffix: String = ""
    var addBOS: Bool = true

    /// 从模型目录 config.json 读取覆盖项；缺失时保留默认值。
    static func load(directory: URL) -> VLMGraphConfig {
        var config = VLMGraphConfig()
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return config
        }
        if let value = object["image_size"] as? Int, value > 0 { config.imageSize = value }
        if let arr = object["mean"] as? [Double], arr.count == 3 {
            config.mean = arr.map(Float.init)
        } else if let arr = object["mean"] as? [Float], arr.count == 3 {
            config.mean = arr
        }
        if let arr = object["std"] as? [Double], arr.count == 3 {
            config.std = arr.map(Float.init)
        } else if let arr = object["std"] as? [Float], arr.count == 3 {
            config.std = arr
        }
        if let value = object["image_token"] as? String, !value.isEmpty {
            config.imageToken = value
        }
        if let value = object["num_image_tokens"] as? Int, value > 0 {
            config.numImageTokens = value
        }
        if let value = object["prompt_prefix"] as? String { config.promptPrefix = value }
        if let value = object["prompt_suffix"] as? String { config.promptSuffix = value }
        if let value = object["add_bos"] as? Bool { config.addBOS = value }
        return config
    }
}

// MARK: - 图像预处理

/// 把 CGImage 转成 ONNX VLM 需要的 `pixel_values`：resize → RGB → [0,1] →
/// ImageNet 归一化 → NCHW float32。
enum VLMImagePreprocessor {
    /// 真实图像预处理。
    static func pixelValues(from image: CGImage, size: CGSize,
                            mean: [Float], std: [Float]) throws -> [Float] {
        let w = Int(size.width), h = Int(size.height)
        guard w > 0, h > 0 else {
            throw LocalModelError.invalidConfiguration("ONNX VLM 图像目标尺寸非法：\(size)")
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: w * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw LocalModelError.runtime("ONNX VLM 无法创建图像上下文")
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let data = context.data else {
            throw LocalModelError.runtime("ONNX VLM 图像上下文无数据")
        }
        let bytes = data.bindMemory(to: UInt8.self, capacity: w * h * 4)
        var out = [Float](repeating: 0, count: 3 * w * h)
        for y in 0..<h {
            for x in 0..<w {
                let i = (y * w + x) * 4
                let r = Float(bytes[i]) / 255
                let g = Float(bytes[i + 1]) / 255
                let b = Float(bytes[i + 2]) / 255
                let o = y * w + x
                out[o] = (r - mean[0]) / std[0]
                out[w * h + o] = (g - mean[1]) / std[1]
                out[2 * w * h + o] = (b - mean[2]) / std[2]
            }
        }
        return out
    }
}
