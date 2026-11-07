import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Jinja
import llama

/// 进程内 llama.cpp 运行时错误。
public enum LlamaCppError: LocalizedError, Sendable {
    case modelLoad(String)
    case context(String)
    case template(String)
    case tokenize(String)
    case decode(String)
    case mmproj(String)
    case unsupported(String)

    public var errorDescription: String? {
        switch self {
        case .modelLoad(let message): return "GGUF 模型加载失败：\(message)"
        case .context(let message): return "llama.cpp 上下文创建失败：\(message)"
        case .template(let message): return "GGUF chat template 渲染失败：\(message)"
        case .tokenize(let message): return "GGUF 分词失败：\(message)"
        case .decode(let message): return "GGUF 推理失败：\(message)"
        case .mmproj(let message): return "GGUF 多模态投影加载失败：\(message)"
        case .unsupported(let message): return "GGUF 模型不支持该能力：\(message)"
        }
    }
}

/// llama.cpp 全局后端只初始化一次，并把 warning/error 转到应用日志。
enum LlamaCppBackend {
    nonisolated(unsafe) private static var initialized = false
    private static let lock = NSLock()

    static func ensureInitialized() {
        lock.lock()
        defer { lock.unlock() }
        guard !initialized else { return }
        initialized = true
        llama_backend_init()
        llama_log_set({ level, text, _ in
            // CONT（5）是上一行的续行，不是独立告警；只转发 WARN/ERROR。
            guard level.rawValue == GGML_LOG_LEVEL_WARN.rawValue
                    || level.rawValue == GGML_LOG_LEVEL_ERROR.rawValue,
                  let text else { return }
            let message = String(cString: text).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !message.isEmpty, message != "." else { return }
            // 获取 embedding 时 llama.cpp 会提示覆盖输出标记，这是预期行为。
            guard !message.contains("embeddings required but some input tokens were not marked as outputs"),
                  !message.contains("non-consecutive token position")
            else { return }
            AgentLog.warning("[llama.cpp] \(message)")
        }, nil)
        AgentLog.info("[llama.cpp] 进程内后端 \(String(cString: llama_version()))")
    }
}

/// GGUF 模型句柄。权重常驻，context 由各 provider 单独创建。
final class LlamaCppModel: @unchecked Sendable {
    let raw: OpaquePointer
    let vocab: OpaquePointer
    let path: String
    let contextTrain: Int
    let embeddingSize: Int
    let chatTemplate: String?

    init(path: String, gpuLayers: Int) throws {
        LlamaCppBackend.ensureInitialized()
        var params = llama_model_default_params()
        let layers = gpuLayers < 0 ? -1 : max(0, Int32(gpuLayers))
        params.n_gpu_layers = llama_supports_gpu_offload() ? layers : 0
        let loaded = path.withCString { llama_model_load_from_file($0, params) }
        guard let loaded else {
            throw LlamaCppError.modelLoad("无法读取 \(path)")
        }
        guard let vocab = llama_model_get_vocab(loaded) else {
            llama_model_free(loaded)
            throw LlamaCppError.modelLoad("模型没有可用词表：\(path)")
        }
        self.raw = loaded
        self.vocab = vocab
        self.path = path
        self.contextTrain = Int(llama_model_n_ctx_train(loaded))
        self.embeddingSize = Int(llama_model_n_embd(loaded))
        self.chatTemplate = Self.loadChatTemplate(loaded)
    }

    deinit {
        llama_model_free(raw)
    }

    var addBos: Bool { llama_vocab_get_add_bos(vocab) }
    var addEos: Bool { llama_vocab_get_add_eos(vocab) }
    var addSep: Bool { llama_vocab_get_add_sep(vocab) }
    var eosToken: llama_token { llama_vocab_eos(vocab) }
    var sepToken: llama_token { llama_vocab_sep(vocab) }
    var bosToken: llama_token { llama_vocab_bos(vocab) }
    var vocabSize: Int { Int(llama_vocab_n_tokens(vocab)) }

    func isEndOfGeneration(_ token: llama_token) -> Bool {
        llama_vocab_is_eog(vocab, token)
    }

    // MARK: - 分词 / 反分词

    func tokenize(_ text: String, addSpecial: Bool) throws -> [llama_token] {
        let byteCount = Int32(text.utf8.count)
        var count = llama_tokenize(vocab, text, byteCount, nil, 0, addSpecial, true)
        if count == Int32.min {
            throw LlamaCppError.tokenize("文本过长，分词结果溢出")
        }
        if count < 0 { count = -count }
        guard count > 0 else { return [] }

        var tokens = [llama_token](repeating: 0, count: Int(count))
        let written = tokens.withUnsafeMutableBufferPointer { buffer in
            llama_tokenize(vocab, text, byteCount, buffer.baseAddress,
                           Int32(buffer.count), addSpecial, true)
        }
        guard written >= 0 else {
            throw LlamaCppError.tokenize("llama_tokenize 返回 \(written)")
        }
        return Array(tokens.prefix(Int(written)))
    }

    /// 单个 token 的 UTF-8 字节。控制 token 不渲染（special=false）。
    func pieceBytes(_ token: llama_token) -> [UInt8] {
        var buffer = [CChar](repeating: 0, count: 128)
        var written = llama_token_to_piece(vocab, token, &buffer, Int32(buffer.count), 0, false)
        if written < 0 {
            let needed = Int(-written)
            buffer = [CChar](repeating: 0, count: needed)
            written = llama_token_to_piece(vocab, token, &buffer, Int32(buffer.count), 0, false)
        }
        guard written > 0 else { return [] }
        return buffer.prefix(Int(written)).map { UInt8(bitPattern: $0) }
    }

    /// 特殊 token 的文本形式（如 `<|im_start|>`），用于模板变量。
    func specialTokenText(_ token: llama_token) -> String? {
        guard token != LLAMA_TOKEN_NULL else { return nil }
        var buffer = [CChar](repeating: 0, count: 128)
        var written = llama_token_to_piece(vocab, token, &buffer, Int32(buffer.count), 0, true)
        if written < 0 {
            let needed = Int(-written)
            buffer = [CChar](repeating: 0, count: needed)
            written = llama_token_to_piece(vocab, token, &buffer, Int32(buffer.count), 0, true)
        }
        guard written > 0 else { return nil }
        return String(decoding: buffer.prefix(Int(written)).map { UInt8(bitPattern: $0) },
                      as: UTF8.self)
    }

    // MARK: - GGUF 元数据

    static func metadataString(_ model: OpaquePointer, key: String) -> String? {
        let length = key.withCString { llama_model_meta_val_str(model, $0, nil, 0) }
        guard length > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: Int(length) + 1)
        let written = key.withCString {
            llama_model_meta_val_str(model, $0, &buffer, buffer.count)
        }
        guard written >= 0 else { return nil }
        return String(cString: buffer)
    }

    private static func metadataKeys(_ model: OpaquePointer) -> [String] {
        let count = llama_model_meta_count(model)
        guard count > 0 else { return [] }
        return (0..<Int(count)).compactMap { index in
            let length = llama_model_meta_key_by_index(model, Int32(index), nil, 0)
            guard length > 0 else { return nil }
            var buffer = [CChar](repeating: 0, count: Int(length) + 1)
            let written = llama_model_meta_key_by_index(
                model, Int32(index), &buffer, buffer.count)
            guard written >= 0 else { return nil }
            return String(cString: buffer)
        }
    }

    private static func loadChatTemplate(_ model: OpaquePointer) -> String? {
        if let direct = metadataString(model, key: "tokenizer.chat_template"),
           !direct.isEmpty {
            return direct
        }
        let keys = metadataKeys(model).filter {
            $0.hasPrefix("tokenizer.chat_template")
        }
        if let preferred = keys.first(where: { $0.lowercased().contains("default") }),
           let value = metadataString(model, key: preferred), !value.isEmpty {
            return value
        }
        for key in keys {
            if let value = metadataString(model, key: key), !value.isEmpty {
                return value
            }
        }
        if let raw = llama_model_chat_template(model, nil) {
            let value = String(cString: raw)
            return value.isEmpty ? nil : value
        }
        return nil
    }
}

// MARK: - Prompt 渲染

enum LlamaCppPromptRenderer {
    /// 用 GGUF 自带的 Jinja 模板渲染对话。
    ///
    /// llama.cpp 的 C 模板接口不支持 tools 参数，这里直接用 swift-jinja
    /// 渲染元数据里的原始模板，保证工具定义、enable_thinking 与模型预期一致。
    static func render(
        template: String?,
        model: LlamaCppModel,
        messages: [ChatMessage],
        tools: [JSONValue],
        enableThinking: Bool,
        addGenerationPrompt: Bool
    ) throws -> String {
        guard let template, !template.isEmpty else {
            throw LlamaCppError.template(
                "模型没有 tokenizer.chat_template，无法按对话格式生成（\(model.path)）")
        }

        var context: [String: Jinja.Value] = [
            "messages": .array(messages.map(messageValue)),
            "add_generation_prompt": .boolean(addGenerationPrompt),
            "enable_thinking": .boolean(enableThinking),
        ]
        if !tools.isEmpty {
            context["tools"] = .array(tools.map(jinjaValue))
        }
        if let bos = model.specialTokenText(model.bosToken) {
            context["bos_token"] = .string(bos)
        }
        if let eos = model.specialTokenText(model.eosToken) {
            context["eos_token"] = .string(eos)
        }

        do {
            return try Jinja.Template(template).render(context)
        } catch {
            throw LlamaCppError.template(
                "\(error.localizedDescription)（\(String(describing: error))）")
        }
    }

    /// `add_generation_prompt` 追加的文本里，assistant 角色标记之后的部分。
    ///
    /// 这段文本属于模板而不是模型输出，但下一轮把它交给模板时会原样出现在
    /// assistant 消息里；把它并入 rawContent 才能让跨轮 token 前缀完全对齐。
    static func generationPrefix(full: String, base: String) -> String {
        guard full.hasPrefix(base) else { return "" }
        let suffix = String(full.dropFirst(base.count))
        let markers = [
            "<|im_start|>assistant",
            "<|start_header_id|>assistant<|end_header_id|>",
            "<|assistant|>",
            "<start_of_turn>model",
            "[/INST]",
        ]
        var end: String.Index?
        for marker in markers {
            if let range = suffix.range(of: marker) {
                if end == nil || range.upperBound > end! {
                    end = range.upperBound
                }
            }
        }
        guard let end else { return "" }
        return String(suffix[end...])
    }

    private static func messageValue(_ message: ChatMessage) -> Jinja.Value {
        var dictionary: [String: Jinja.Value] = [
            "role": .string(message.role.rawValue),
            "content": .string(message.content),
        ]
        if let name = message.name, !name.isEmpty {
            dictionary["name"] = .string(name)
        }
        if let toolCallID = message.toolCallID, !toolCallID.isEmpty {
            dictionary["tool_call_id"] = .string(toolCallID)
        }
        if let toolCalls = message.toolCalls, !toolCalls.isEmpty {
            dictionary["tool_calls"] = .array(toolCalls.map { call in
                let arguments = call.rawArguments.isEmpty
                    ? JSONValue.object(call.arguments).jsonString()
                    : call.rawArguments
                return .object([
                    "id": .string(call.id),
                    "type": .string("function"),
                    "function": .object([
                        "name": .string(call.name),
                        "arguments": .string(arguments),
                    ]),
                ])
            })
        }
        return .object(dictionary)
    }

    private static func jinjaValue(_ value: JSONValue) -> Jinja.Value {
        switch value {
        case .null:
            return .null
        case .bool(let value):
            return .boolean(value)
        case .int(let value):
            return .int(value)
        case .double(let value):
            return .double(value)
        case .string(let value):
            return .string(value)
        case .array(let values):
            return .array(values.map(jinjaValue))
        case .object(let object):
            var result: [String: Jinja.Value] = [:]
            for (key, value) in object {
                result[key] = jinjaValue(value)
            }
            return .object(result)
        }
    }
}

// MARK: - 采样

struct LlamaCppSampling {
    var temperature: Float
    var topK: Int
    var topP: Float
    var minP: Float
    var repetitionPenalty: Float

    init(temperature: Double, topK: Int, topP: Double, minP: Double,
         repetitionPenalty: Double) {
        self.temperature = Float(max(0, temperature))
        self.topK = max(0, topK)
        self.topP = Float(min(max(topP, 0), 1))
        self.minP = Float(min(max(minP, 0), 1))
        self.repetitionPenalty = Float(max(1, repetitionPenalty))
    }
}

/// llama.cpp 采样链；每个请求一条，避免把上一轮 token 纳入惩罚。
final class LlamaCppSampler {
    private let chain: UnsafeMutablePointer<llama_sampler>

    init(vocabSize: Int, config: LlamaCppSampling) {
        var params = llama_sampler_chain_default_params()
        params.no_perf = true
        guard let chain = llama_sampler_chain_init(params) else {
            fatalError("llama_sampler_chain_init failed")
        }
        self.chain = chain

        if config.repetitionPenalty > 1.0 {
            llama_sampler_chain_add(chain, llama_sampler_init_penalties(
                Int32(vocabSize), 64, config.repetitionPenalty, 0, 0))
        }
        if config.topK > 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(Int32(config.topK)))
        }
        if config.topP < 1.0 {
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(config.topP, 1))
        }
        if config.minP > 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_min_p(config.minP, 1))
        }
        if config.temperature <= 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy())
        } else {
            llama_sampler_chain_add(chain, llama_sampler_init_temp(config.temperature))
            llama_sampler_chain_add(chain, llama_sampler_init_dist(
                UInt32.random(in: 0...UInt32.max)))
        }
    }

    deinit {
        llama_sampler_free(chain)
    }

    func sample(context: OpaquePointer) -> llama_token {
        llama_sampler_sample(chain, context, -1)
    }

    func accept(_ token: llama_token) {
        llama_sampler_accept(chain, token)
    }
}

// MARK: - 上下文辅助

enum LlamaCppContext {
    static func make(
        model: LlamaCppModel,
        contextWindow: Int,
        batchSize: Int,
        embeddings: Bool,
        kvInt8: Bool = false,
        /// nil = 保持 llama.cpp 默认 AUTO（embedding/VLM 路径）；
        /// true = 强制 ENABLED，false = 显式 DISABLED（LLM 路径由配置驱动）。
        flashAttention: Bool? = nil
    ) throws -> OpaquePointer {
        var params = llama_context_default_params()
        if let flashAttention {
            params.flash_attn_type = flashAttention
                ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                : LLAMA_FLASH_ATTN_TYPE_DISABLED
        }
        let window = max(512, contextWindow)
        let batch = max(64, min(batchSize, window))
        params.n_ctx = UInt32(window)
        params.n_batch = UInt32(batch)
        params.n_ubatch = UInt32(min(batch, 512))
        params.n_threads = Int32(max(4, min(8, ProcessInfo.processInfo.activeProcessorCount - 2)))
        params.n_threads_batch = params.n_threads
        params.embeddings = embeddings
        // KV cache int8 量化（q8_0）：KV 内存约减半，质量几乎无损。
        // 仅 LLM（主/辅）开启；embedding/VLM 上下文保持 f16。
        params.type_k = kvInt8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16
        params.type_v = kvInt8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16
        params.offload_kqv = true
        params.no_perf = true
        // 滑动窗口注意力的模型默认只保留窗口内 KV，无法按前缀裁剪；
        // 这里保留完整 KV，跨轮才能用 seq_rm 复用 system 前缀。
        params.swa_full = true
        guard let context = llama_init_from_model(model.raw, params) else {
            throw LlamaCppError.context("llama_init_from_model 返回空（\(model.path)）")
        }
        return context
    }

    static func decode(tokens: [llama_token], context: OpaquePointer,
                       chunkSize: Int) throws {
        guard !tokens.isEmpty else { return }
        let size = max(1, chunkSize)
        var index = 0
        while index < tokens.count {
            let end = min(index + size, tokens.count)
            var chunk = Array(tokens[index..<end])
            let result = chunk.withUnsafeMutableBufferPointer { buffer in
                llama_decode(context, llama_batch_get_one(
                    buffer.baseAddress, Int32(buffer.count)))
            }
            if result != 0 {
                throw LlamaCppError.decode("llama_decode 返回 \(result)")
            }
            index = end
        }
    }

    static func decode(token: llama_token, context: OpaquePointer) throws {
        var value = token
        let result = llama_decode(context, llama_batch_get_one(&value, 1))
        if result != 0 {
            throw LlamaCppError.decode("llama_decode 返回 \(result)")
        }
    }

    static func commonPrefix(_ lhs: [llama_token], _ rhs: [llama_token]) -> Int {
        let limit = min(lhs.count, rhs.count)
        var index = 0
        while index < limit, lhs[index] == rhs[index] {
            index += 1
        }
        return index
    }
}

// MARK: - 流式 UTF-8 解码

/// token 边界可能切开多字节 UTF-8，必须按字节缓冲后再交给文本切分器。
struct LlamaCppUTF8StreamDecoder {
    private var pending: [UInt8] = []

    mutating func feed(_ bytes: [UInt8]) -> String {
        pending.append(contentsOf: bytes)
        let incomplete = Self.incompleteSuffixCount(pending)
        let usable = incomplete > 0 ? pending.count - incomplete : pending.count
        guard usable > 0 else { return "" }
        let prefix = Array(pending.prefix(usable))
        pending.removeFirst(usable)
        return String(decoding: prefix, as: UTF8.self)
    }

    mutating func flush() -> String {
        let remaining = pending
        pending = []
        guard !remaining.isEmpty else { return "" }
        return String(decoding: remaining, as: UTF8.self)
    }

    private static func incompleteSuffixCount(_ bytes: [UInt8]) -> Int {
        guard !bytes.isEmpty else { return 0 }
        var index = bytes.count - 1
        var continuationCount = 0
        while index >= 0, bytes[index] & 0xC0 == 0x80 {
            continuationCount += 1
            index -= 1
            if continuationCount > 3 { return 0 }
        }
        guard index >= 0 else { return 0 }
        let lead = bytes[index]
        let expected: Int
        switch lead {
        case 0x00..<0x80: expected = 1
        case 0xC0..<0xE0: expected = 2
        case 0xE0..<0xF0: expected = 3
        case 0xF0..<0xF8: expected = 4
        default:
            return 0
        }
        let available = bytes.count - index
        return available < expected ? available : 0
    }
}

// MARK: - 图片编码

enum LlamaCppImage {
    static func jpegData(_ image: CGImage, quality: Double = 0.92) -> Data? {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data, UTType.jpeg.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(destination, image, [
            kCGImageDestinationLossyCompressionQuality: quality,
        ] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return data as Data
    }
}
