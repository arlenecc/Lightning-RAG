import Foundation
import CoreGraphics
import llama

/// 进程内 GGUF 视觉模型（主模型 + mmproj，通过 mtmd 直接推理）。
public final class LlamaCppVLMService: VLMService, @unchecked Sendable {
    public let identifier: String
    public let modelPath: String
    public let isAvailable = true

    private let mmprojPath: String?
    private let contextWindow: Int
    private let gpuLayers: Int
    private let gate = AsyncGate()
    private let stateLock = NSLock()
    private var runtime: LlamaCppVisionRuntime?
    private var loadFailure: Error?

    init(modelPath: String, mmprojPath: String?, contextWindow: Int = 8_192,
         gpuLayers: Int = 99) {
        self.modelPath = modelPath
        self.mmprojPath = mmprojPath
        self.contextWindow = contextWindow
        self.gpuLayers = gpuLayers
        self.identifier = "llama.cpp-vlm:\((modelPath as NSString).lastPathComponent)"
    }

    public func recognize(image: CGImage, prompt: String, maxTokens: Int) async throws -> String {
        try await gate.withLock {
            let runtime = try self.ensureRuntime()
            return try runtime.recognize(image: image, prompt: prompt,
                                         maxTokens: max(1, maxTokens))
        }
    }

    public func unload() async {
        await gate.acquire()
        stateLock.lock()
        let current = runtime
        runtime = nil
        loadFailure = nil
        stateLock.unlock()
        current?.close()
        gate.release()
    }

    private func ensureRuntime() throws -> LlamaCppVisionRuntime {
        stateLock.lock()
        defer { stateLock.unlock() }
        if let runtime { return runtime }
        if let loadFailure { throw loadFailure }
        guard let mmprojPath else {
            let error = LlamaCppError.mmproj(
                "GGUF VLM 需要 mmproj 文件（\(modelPath) 同目录未找到 mmproj-*.gguf）")
            loadFailure = error
            throw error
        }
        do {
            let value = try LlamaCppVisionRuntime(
                modelPath: modelPath,
                mmprojPath: mmprojPath,
                contextWindow: contextWindow,
                gpuLayers: gpuLayers)
            runtime = value
            return value
        } catch {
            loadFailure = error
            throw error
        }
    }
}

final class LlamaCppVisionRuntime: @unchecked Sendable {
    private let model: LlamaCppModel
    private let context: OpaquePointer
    private let vision: OpaquePointer
    private let contextWindow: Int
    private let batchSize: Int
    private var closed = false

    init(modelPath: String, mmprojPath: String, contextWindow: Int, gpuLayers: Int) throws {
        let window = max(1_024, contextWindow)
        let batch = min(2_048, window)
        let loadedModel = try LlamaCppModel(path: modelPath, gpuLayers: gpuLayers)
        let createdContext = try LlamaCppContext.make(
            model: loadedModel,
            contextWindow: window,
            batchSize: batch,
            embeddings: false)

        var params = mtmd_context_params_default()
        params.use_gpu = LlamaCppBackend.supportsGPUOffload
        params.print_timings = false
        params.warmup = false
        params.n_threads = Int32(max(4, min(8, ProcessInfo.processInfo.activeProcessorCount - 2)))
        guard let createdVision = mmprojPath.withCString({ path in
            mtmd_init_from_file(path, loadedModel.raw, params)
        }) else {
            llama_free(createdContext)
            throw LlamaCppError.mmproj("mtmd_init_from_file 返回空：\(mmprojPath)")
        }
        self.contextWindow = window
        self.batchSize = batch
        self.model = loadedModel
        self.context = createdContext
        self.vision = createdVision
    }

    func close() {
        guard !closed else { return }
        closed = true
        mtmd_free(vision)
        llama_free(context)
    }

    deinit {
        close()
    }

    func recognize(image: CGImage, prompt: String, maxTokens: Int) throws -> String {
        guard let jpeg = LlamaCppImage.jpegData(image) else {
            throw LlamaCppError.decode("无法编码页面图片")
        }
        let opt = mtmd_helper_init_opt_default()
        let wrapper = jpeg.withUnsafeBytes { buffer -> mtmd_helper_bitmap_wrapper in
            mtmd_helper_bitmap_init_from_buf(
                vision,
                buffer.bindMemory(to: UInt8.self).baseAddress,
                buffer.count,
                false,
                opt)
        }
        guard let bitmap = wrapper.bitmap else {
            throw LlamaCppError.decode("图片解码失败（mmproj 不支持该格式）")
        }
        defer { mtmd_bitmap_free(bitmap) }

        let marker = String(cString: mtmd_default_marker())
        let user = ChatMessage(role: .user, content: marker + "\n" + prompt)
        let formatted = try LlamaCppPromptRenderer.render(
            template: model.chatTemplate,
            model: model,
            messages: [user],
            tools: [],
            enableThinking: false,
            addGenerationPrompt: true)
        guard formatted.contains(marker) else {
            throw LlamaCppError.template("chat template 丢失了媒体占位符 \(marker)")
        }

        llama_memory_clear(llama_get_memory(context), true)
        guard let chunks = mtmd_input_chunks_init() else {
            throw LlamaCppError.decode("mtmd_input_chunks_init 失败")
        }
        defer { mtmd_input_chunks_free(chunks) }

        var text = mtmd_input_text(text: nil, text_len: 0,
                                   add_special: true, parse_special: true)
        var bitmapPointer: OpaquePointer? = bitmap
        let tokenizeResult: Int32 = formatted.withCString { pointer in
            text.text = pointer
            text.text_len = formatted.utf8.count
            return withUnsafePointer(to: &bitmapPointer) { bitmaps in
                mtmd_tokenize(vision, chunks, &text, bitmaps, 1)
            }
        }
        guard tokenizeResult == 0 else {
            throw LlamaCppError.decode("mtmd_tokenize 返回 \(tokenizeResult)")
        }

        var nPast: llama_pos = 0
        let evalResult = mtmd_helper_eval_chunks(
            vision, context, chunks, 0, 0, Int32(batchSize), true, &nPast)
        guard evalResult == 0 else {
            throw LlamaCppError.decode("mtmd 图像编码失败（\(evalResult)）")
        }

        let sampling = LlamaCppSampling(
            temperature: 0, topK: 0, topP: 1, minP: 0, repetitionPenalty: 1)
        let sampler = LlamaCppSampler(vocabSize: model.vocabSize, config: sampling)
        var decoder = LlamaCppUTF8StreamDecoder()
        var output = ""
        var generated = 0
        while generated < maxTokens, Int(nPast) < contextWindow {
            try Task.checkCancellation()
            let token = sampler.sample(context: context)
            sampler.accept(token)
            if model.isEndOfGeneration(token) { break }
            try LlamaCppContext.decode(token: token, context: context)
            nPast += 1
            generated += 1
            let text = decoder.feed(model.pieceBytes(token))
            if !text.isEmpty { output += text }
        }
        output += decoder.flush()
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
