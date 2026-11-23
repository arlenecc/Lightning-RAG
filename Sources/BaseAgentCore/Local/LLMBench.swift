import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import llama

/// LLMBench：推理基线测量（分析计划阶段 0）。
/// 以 512-token 分段计时 prefill（观察分段耗时随偏移的增长曲线，判断线性/超线性），
/// 再测 decode 逐 token 吞吐。由 RAGBenchmark --llm-bench 调用，只打印结果行。
public enum LLMBench {

    struct Options {
        var backend = "mlx"
        var modelPath = ""
        var sizes: [Int] = [256, 512, 1024, 2048, 4096, 7671]
        var chunk = 512
        var decodeTokens = 64
        var contextWindow = 32768
        var gpuLayers = 99
        var repeatLong = 1
        /// 走 TokenIterator（生产路径：模型内部分块 prefill，只对末块求 logits）。
        var tokenIter = false
        /// 采样参数（仅 token-iter 路径）：temp=0 走 argmax，>0 走生产同款
        /// topP/topK/minP 采样器。用于量化采样器本身每 token 的开销。
        var temperature = 0.0
        var topP = 0.95
        var topK = 20
        var minP = 0.05
        /// 逐 token detokenize 微基准的 token 数（0 = 关闭）。
        var tokBench = 0
    }

    /// 真实感填充文本（中英混合，模拟 agent system+tools 前缀的密度）。
    static let filler =
        "你是一个多工具智能体，可调用 search_file、read_page、write_note、run_shell、web_fetch 等工具。" +
        "收到任务后先规划步骤，再逐步调用工具并汇总结果。Filler sentence for benchmark; " +
        "the quick brown fox jumps over the lazy dog while reading documentation about " +
        "retrieval augmented generation and local inference on Apple Silicon GPUs. "

    // MARK: - GDN 路径探针（qwen3.5 线性注意力层：fused kernel vs ops 回退）

    static func gdnProbe() {
        // 4B 实际形状：24 GDN 层（32 层 × interval 4），Hv=32 Hk=16 Dk=128 Dv=128
        let (B, T, Hk, Dk, Hv, Dv) = (1, 512, Int32(16), Int32(128), Int32(32), Int32(128))
        func rand(_ shape: [Int], dtype: MLX.DType) -> MLXArray {
            MLXRandom.normal(shape).asType(dtype) * 0.1
        }
        let q = rand([B, T, Int(Hk), Int(Dk)], dtype: .bfloat16)
        let k = rand([B, T, Int(Hk), Int(Dk)], dtype: .bfloat16)
        let v = rand([B, T, Int(Hv), Int(Dv)], dtype: .bfloat16)
        let a = rand([B, T, Int(Hv)], dtype: .bfloat16)
        let b = rand([B, T, Int(Hv)], dtype: .bfloat16)
        let aLog = rand([Int(Hv)], dtype: .bfloat16)
        let dtBias = rand([Int(Hv)], dtype: .float32)

        for useKernel in [true, false] {
            // 预热（kernel 编译/图编译）
            _ = gatedDeltaUpdate(q: q, k: k, v: v, a: a, b: b, aLog: aLog,
                                 dtBias: dtBias, useKernel: useKernel)
            eval(q)
            let c0 = Date()
            let (y, state) = gatedDeltaUpdate(q: q, k: k, v: v, a: a, b: b,
                                              aLog: aLog, dtBias: dtBias,
                                              useKernel: useKernel)
            eval(y, state)
            let ms = Date().timeIntervalSince(c0) * 1000
            // 4B 有 24 个 GDN 层，全模型 512-token chunk = 24 × 单层耗时
            print("GDN-PROBE path=\(useKernel ? "kernel" : "ops") T=\(T) "
                + "ms=\(String(format: "%.0f", ms)) tok_s=\(String(format: "%.0f", Double(T) / ms * 1000)) "
                + "x24layers_ms=\(String(format: "%.0f", ms * 24))")
        }
        // decode 形态（T=1）
        let q1 = q[0..., 0..<1, 0..., 0...]
        let k1 = k[0..., 0..<1, 0..., 0...]
        let v1 = v[0..., 0..<1, 0..., 0...]
        let a1 = a[0..., 0..<1, 0...]
        let b1 = b[0..., 0..<1, 0...]
        for useKernel in [true, false] {
            _ = gatedDeltaUpdate(q: q1, k: k1, v: v1, a: a1, b: b1, aLog: aLog,
                                 dtBias: dtBias, useKernel: useKernel)
            eval(q1)
            let c0 = Date()
            for _ in 0..<64 {
                let (y, state) = gatedDeltaUpdate(q: q1, k: k1, v: v1, a: a1, b: b1,
                                                  aLog: aLog, dtBias: dtBias,
                                                  useKernel: useKernel)
                eval(y, state)
            }
            let perToken = Date().timeIntervalSince(c0) * 1000 / 64
            print("GDN-PROBE path=\(useKernel ? "kernel" : "ops") T=1 "
                + "ms_per_tok=\(String(format: "%.2f", perToken))")
        }
    }

    public static func run(_ argv: [String]) async throws {
        setbuf(stdout, nil)  // 崩溃时保证已打印的 STEP 不丢失
        var o = Options()
        var i = 0
        while i < argv.count {
            switch argv[i] {
            case "--backend": i += 1; o.backend = argv[i]
            case "--model": i += 1; o.modelPath = argv[i]
            case "--sizes":
                i += 1
                o.sizes = argv[i].split(separator: ",").compactMap { Int($0) }
            case "--chunk": i += 1; o.chunk = Int(argv[i]) ?? 512
            case "--decode-tokens": i += 1; o.decodeTokens = Int(argv[i]) ?? 64
            case "--ctx": i += 1; o.contextWindow = Int(argv[i]) ?? 32768
            case "--gpu-layers": i += 1; o.gpuLayers = Int(argv[i]) ?? 99
            case "--repeat-long": i += 1; o.repeatLong = Int(argv[i]) ?? 1
            case "--token-iter": o.tokenIter = true
            case "--temp": i += 1; o.temperature = Double(argv[i]) ?? 0
            case "--top-p": i += 1; o.topP = Double(argv[i]) ?? 0.95
            case "--top-k": i += 1; o.topK = Int(argv[i]) ?? 20
            case "--min-p": i += 1; o.minP = Double(argv[i]) ?? 0.05
            case "--tok-bench": i += 1; o.tokBench = Int(argv[i]) ?? 512
            default: break
            }
            i += 1
        }
        guard !o.modelPath.isEmpty else {
            if argv.contains("--gdn-probe") {
                gdnProbe()
                return
            }
            fputs("LLMBench: 需要 --model\n", stderr)
            exit(2)
        }
        if o.backend == "gguf" {
            try runGGUF(o)
        } else if o.tokBench > 0 {
            try await tokenizerBench(o)
        } else {
            try await runMLX(o, useTokenIterator: o.tokenIter)
        }
    }

    /// 逐 token detokenize 微基准：复刻 NaiveStreamingDetokenizer 的两种形态。
    /// - grow：段内不换行地一直增长，每 token 重解码整段（中文长段落的最坏情况）
    /// - reset：每个 token 单独解码（有换行/分段的乐观情况）
    static func tokenizerBench(_ o: Options) async throws {
        let dir = URL(fileURLWithPath: o.modelPath)
        let tokenizer = try await LocalTokenizer.load(directory: dir)
        let tokens = buildTokens(tokenizer, target: o.tokBench)
        for resetEveryToken in [false, true] {
            var seg: [Int] = []
            let t0 = Date()
            for t in tokens {
                seg.append(t)
                _ = tokenizer.base.decode(tokens: seg)
                if resetEveryToken { seg.removeAll(keepingCapacity: true) }
            }
            let ms = Date().timeIntervalSince(t0) * 1000
            print("TOK-BENCH mode=\(resetEveryToken ? "reset" : "grow") n=\(tokens.count) "
                + "total_ms=\(String(format: "%.1f", ms)) "
                + "per_token_ms=\(String(format: "%.3f", ms / Double(tokens.count)))")
        }
    }

    // MARK: - MLX

    static func buildTokens(_ tokenizer: LocalTokenizer, target: Int) -> [Int] {
        let fill = tokenizer.encode(filler, addSpecialTokens: false)
        var tokens = tokenizer.encode("You are a helpful agent. ", addSpecialTokens: false)
        while tokens.count < target { tokens += fill }
        return Array(tokens.prefix(target))
    }

    static func runMLX(_ o: Options, useTokenIterator: Bool = false) async throws {
        // 与应用一致的全局环境：default.metallib 注册（缺它首次求值即崩）+ 内存上限
        _ = LocalModelFactory.shared
        let dir = URL(fileURLWithPath: o.modelPath)
        let tokenizer = try await LocalTokenizer.load(directory: dir)
        let loadStart = Date()
        let container = try await LLMModelFactory.shared.loadContainer(
            from: dir, using: LocalTokenizerLoader())
        let loadSeconds = Date().timeIntervalSince(loadStart)
        print("BENCH-META backend=mlx load_seconds=\(String(format: "%.2f", loadSeconds))")

        let maxSize = o.sizes.max() ?? 7671
        for size in o.sizes {
            let tokens = buildTokens(tokenizer, target: size)
            // 最大 size 额外重复 pass，用于对比冷/热（首次含 kernel 编译）。
            let passes = (size == maxSize) ? 1 + max(0, o.repeatLong) : 1
            for pass in 0..<passes {
                var chunkMS: [Int] = []
                var decodeMS: Double = 0
                var decodeCount = 0
                let passStart = Date()
                try await container.perform { ctx in
                    // --token-iter：用 TokenIterator 跑完整生成，隔离库路径问题
                    if useTokenIterator {
                        var prefill = PrefillParameters()
                        if o.chunk > 0 { prefill.stepSize = o.chunk }
                        let parameters = GenerateParameters(
                            maxTokens: o.decodeTokens, temperature: Float(o.temperature),
                            topP: Float(o.topP), topK: o.topK, minP: Float(o.minP),
                            prefill: prefill)
                        let t0 = Date()
                        var iterator = try TokenIterator(
                            input: .init(text: .init(tokens: MLXArray(tokens))),
                            model: ctx.model, parameters: parameters)
                        // init 内完成 prefill：单独记账，避免与 decode 混算。
                        let prefillSeconds = iterator.promptPrefillTime
                        var n = 0
                        while let _ = iterator.next() { n += 1 }
                        let totalSeconds = Date().timeIntervalSince(t0)
                        decodeMS = max(0, totalSeconds - prefillSeconds) * 1000
                        decodeCount = n
                        print("STEP token-iter done n=\(n) "
                            + "prefill_s=\(String(format: "%.2f", prefillSeconds))")
                        print("BENCH-PREFILL backend=mlx size=\(size) pass=\(pass) "
                            + "prefill_ms=\(String(format: "%.0f", prefillSeconds * 1000)) "
                            + "prefill_tok_s=\(String(format: "%.0f", Double(tokens.count) / prefillSeconds))")
                        return
                    }
                    let caches = try ctx.model.newCache(parameters: nil)
                    print("STEP newCache ok layers=\(caches.count)")
                    var state = LMOutput.State()
                    var lastLogits: MLXArray?
                    var offset = 0
                    // chunk<=0：全序列单次 forward（测分块自身的开销）
                    let chunkLen = o.chunk > 0 ? o.chunk : max(tokens.count, 1)
                    while offset < tokens.count {
                        let end = min(offset + chunkLen, tokens.count)
                        let slice = Array(tokens[offset..<end])
                        let input = LMInput.Text(
                            tokens: MLXArray(slice).reshaped([1, slice.count]))
                        print("STEP forward offset=\(offset) n=\(slice.count)")
                        let c0 = Date()
                        let output = ctx.model(input, cache: caches, state: state)
                        print("STEP forward returned")
                        state = output.state ?? LMOutput.State()
                        lastLogits = output.logits
                        // 只对末块求 logits：与生产路径（TokenIterator）一致。
                        // 中间块求 lm_head 是纯浪费——7671 段实测占 prefill 约 30%。
                        // 中间块 eval 缓存即可同步住 body，得到分块真实耗时。
                        if end == tokens.count {
                            output.logits.eval()
                        } else {
                            MLX.eval(caches)
                        }
                        print("STEP eval done")
                        chunkMS.append(Int(Date().timeIntervalSince(c0) * 1000))
                        offset = end
                    }
                    // decode：从最后一个 prefill logits 采样起步，逐 token 推进
                    if let logits = lastLogits, o.decodeTokens > 0 {
                        var next = Int(
                            logits[0, logits.dim(1) - 1, 0...].argMax().item(Int32.self))
                        let vocab = logits.dim(2)
                        for _ in 0..<o.decodeTokens {
                            let c0 = Date()
                            let input = LMInput.Text(
                                tokens: MLXArray([Int32(next)]).reshaped([1, 1]))
                            let out = ctx.model(input, cache: caches, state: state)
                            state = out.state ?? LMOutput.State()
                            let row = out.logits.dim(1) > 1
                                ? out.logits[0, out.logits.dim(1) - 1, 0...]
                                : out.logits[0, 0, 0...]
                            // 与 TokenIterator 一致：token 与缓存一起异步求值，
                            // 每步只同步一次。先单独 eval logits 再 argMax 会多一次
                            // 阻塞（实测 decode 掉到一半）。
                            let token = row.argMax()
                            MLX.asyncEval([token] + caches.flatMap { $0.state })
                            next = Int(token.item(Int32.self))
                            decodeMS += Date().timeIntervalSince(c0) * 1000
                            decodeCount += 1
                        }
                    }
                }
                let prefillMS = chunkMS.reduce(0, +)
                let prefillTPS = Double(tokens.count) / Double(prefillMS) * 1000
                let decodeTPS = decodeCount > 0
                    ? Double(decodeCount) / decodeMS * 1000 : 0
                print("BENCH-PASS backend=mlx size=\(size) pass=\(pass) "
                    + "chunk_ms=[\(chunkMS.map(String.init).joined(separator: ","))] "
                    + "prefill_ms=\(prefillMS) prefill_tok_s=\(String(format: "%.1f", prefillTPS)) "
                    + "decode_ms_per_tok=\(decodeCount > 0 ? String(format: "%.1f", decodeMS / Double(decodeCount)) : "-") "
                    + "decode_tok_s=\(String(format: "%.1f", decodeTPS)) "
                    + "pass_seconds=\(String(format: "%.2f", Date().timeIntervalSince(passStart)))")
            }
        }
    }

    // MARK: - GGUF

    static func runGGUF(_ o: Options) {
        let loadStart = Date()
        let model = try! LlamaCppModel(path: o.modelPath, gpuLayers: o.gpuLayers)
        print("BENCH-META backend=gguf load_seconds=\(String(format: "%.2f", Date().timeIntervalSince(loadStart))) "
            + "ctx_train=\(model.contextTrain) vocab=\(model.vocabSize)")

        let ctx = try! LlamaCppContext.make(
            model: model, contextWindow: o.contextWindow, batchSize: 2048,
            embeddings: false, kvInt8: false, flashAttention: true)
        defer { llama_free(ctx) }

        let sampler = LlamaCppSampler(
            vocab: model.vocab,
            vocabSize: model.vocabSize,
            config: LlamaCppSampling(temperature: 0, topK: 0, topP: 1, minP: 0,
                                     repetitionPenalty: 1))
        let fill = (try! model.tokenize(filler, addSpecial: false))
        let bos = model.bosToken

        let maxSize = o.sizes.max() ?? 7671
        for size in o.sizes {
            var tokens: [llama_token] = model.addBos ? [bos] : []
            while tokens.count < size { tokens += fill }
            tokens = Array(tokens.prefix(size))

            let passes = (size == maxSize) ? 1 + max(0, o.repeatLong) : 1
            for pass in 0..<passes {
                var chunkMS: [Int] = []
                var decodeMS: Double = 0
                var decodeCount = 0
                let passStart = Date()
                var offset = 0
                // chunk<=0：按 n_batch 上限 2048 整块解码
                let chunkLen = o.chunk > 0 ? o.chunk : 2048
                while offset < tokens.count {
                    let end = min(offset + chunkLen, tokens.count)
                    let slice = Array(tokens[offset..<end])
                    let c0 = Date()
                    try! LlamaCppContext.decode(tokens: slice, context: ctx,
                                                chunkSize: slice.count)
                    chunkMS.append(Int(Date().timeIntervalSince(c0) * 1000))
                    offset = end
                }
                var next = sampler.sample(context: ctx)
                sampler.accept(next)
                for _ in 0..<o.decodeTokens {
                    let c0 = Date()
                    try! LlamaCppContext.decode(token: next, context: ctx)
                    next = sampler.sample(context: ctx)
                    sampler.accept(next)
                    decodeMS += Date().timeIntervalSince(c0) * 1000
                    decodeCount += 1
                }
                let prefillMS = chunkMS.reduce(0, +)
                let prefillTPS = Double(tokens.count) / Double(prefillMS) * 1000
                let decodeTPS = decodeCount > 0
                    ? Double(decodeCount) / decodeMS * 1000 : 0
                print("BENCH-PASS backend=gguf size=\(size) pass=\(pass) "
                    + "chunk_ms=[\(chunkMS.map(String.init).joined(separator: ","))] "
                    + "prefill_ms=\(prefillMS) prefill_tok_s=\(String(format: "%.1f", prefillTPS)) "
                    + "decode_ms_per_tok=\(decodeCount > 0 ? String(format: "%.1f", decodeMS / Double(decodeCount)) : "-") "
                    + "decode_tok_s=\(String(format: "%.1f", decodeTPS)) "
                    + "pass_seconds=\(String(format: "%.2f", Date().timeIntervalSince(passStart)))")
            }
        }
    }
}
