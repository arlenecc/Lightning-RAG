import XCTest
@testable import BaseAgentCore
import MLX
import MLXLLM
import MLXLMCommon

/// P1：单线程 batched prefill + batched decode 的单元测试。
///
/// 崩溃根因是同模型多路并发 generation（每路独立 trace，撞上共享权重
/// 临时替换的 tracer 占位数组）。批量生成器把 map 阶段的多路请求合成
/// 单个 batch，单线程跑 prefill/decode。
final class BatchedMLXGenerationTests: XCTestCase {

    // MARK: - BatchGenerationPlan.leftPad

    func testLeftPadPlanPadsToCommonLength() throws {
        let plan = try XCTUnwrap(
            BatchGenerationPlan.leftPad(rows: [[1, 2, 3], [4]], padToken: 0))
        XCTAssertEqual(plan.tokens, [[1, 2, 3], [0, 0, 4]])
        XCTAssertEqual(plan.leftPadding, [0, 2])
        XCTAssertEqual(plan.lengths, [3, 1])
    }

    func testLeftPadPlanEmptyInputIsNil() {
        XCTAssertNil(BatchGenerationPlan.leftPad(rows: [], padToken: 0))
    }

    func testLeftPadPlanSingleRowKeepsOrder() throws {
        let plan = try XCTUnwrap(
            BatchGenerationPlan.leftPad(rows: [[9, 8, 7]], padToken: 3))
        XCTAssertEqual(plan.tokens, [[9, 8, 7]])
        XCTAssertEqual(plan.leftPadding, [0])
        XCTAssertEqual(plan.lengths, [3])
    }

    func testLeftPadPlanKeepsRowOrder() throws {
        // 行序必须保持请求顺序，生成结果按行索引写回。
        let rows = [[1], [2, 2], [3, 3, 3]]
        let plan = try XCTUnwrap(BatchGenerationPlan.leftPad(rows: rows, padToken: 0))
        XCTAssertEqual(plan.lengths, [1, 2, 3])
        XCTAssertEqual(plan.leftPadding, [2, 1, 0])
        XCTAssertEqual(plan.tokens, [[0, 0, 1], [0, 2, 2], [3, 3, 3]])
    }

    // MARK: - BatchedAttentionKVCache

    /// 构造 [B, 1, columns, 2] 的键值张量（每列 2 个元素便于核对）。
    private func makeCache(
        pads: [Int]?, rows: [Float], columns: Int
    ) -> (BatchedAttentionKVCache, MLXArray) {
        let cache = BatchedAttentionKVCache(leftPadding: pads)
        let data = MLXArray(rows).reshaped([max(1, pads?.count ?? 1), 1, columns, 2])
        return (cache, data)
    }

    func testAttentionCacheZeroesPadColumnsInFirstChunk() {
        // row0 pad 2 列（[1,2] [3,4] 与 [5,6] 中的前两列），row1 不 pad。
        let (cache, kv) = makeCache(
            pads: [2, 0],
            rows: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            columns: 3)
        let (k, _) = cache.update(keys: kv, values: kv)
        let data = k.asType(.float32).asArray(Float.self)
        // row0 cols 0、1 → 0
        XCTAssertEqual(Array(data[0..<4]), [0, 0, 0, 0])
        // row0 col2 原样 [5, 6]
        XCTAssertEqual(Array(data[4..<6]), [5, 6])
        // row1 原样
        XCTAssertEqual(Array(data[6..<12]), [7, 8, 9, 10, 11, 12])
    }

    func testAttentionCacheAdvancesOffset() {
        let (cache, kv) = makeCache(
            pads: [2, 0], rows: Array(repeating: 1, count: 12), columns: 3)
        _ = cache.update(keys: kv, values: kv)
        XCTAssertEqual(cache.offset, 3)
        _ = cache.update(keys: kv, values: kv)
        XCTAssertEqual(cache.offset, 6)
    }

    func testAttentionCacheBatchRopeOffsetAfterUpdate() {
        let (cache, kv) = makeCache(
            pads: [2, 0], rows: Array(repeating: 1, count: 12), columns: 3)
        _ = cache.update(keys: kv, values: kv)
        guard case .batch(let offset) = cache.ropeOffset else {
            return XCTFail("expected batch rope offset")
        }
        // offset(3) - pads([2, 0]) = [1, 3]
        XCTAssertEqual(offset.asArray(Int.self), [1, 3])
    }

    func testAttentionCacheScalarRopeOffsetWithoutPadding() {
        // pads 为 nil 时批大小为 1，行数据只需 1*3*2 个元素。
        let (cache, kv) = makeCache(pads: nil, rows: Array(repeating: 1, count: 6), columns: 3)
        _ = cache.update(keys: kv, values: kv)
        guard case .scalar(let value) = cache.ropeOffset else {
            return XCTFail("expected scalar rope offset")
        }
        XCTAssertEqual(value, 3)
    }

    func testAttentionCacheKeepsPaddingAfterFinalize() {
        // finalize 只清 SSM 侧瞬态掩码；注意力缓存的 pads 必须保留，
        // decode 阶段 ropeOffset 仍要按行偏移。
        let (cache, kv) = makeCache(
            pads: [1, 0], rows: Array(repeating: 1, count: 12), columns: 3)
        _ = cache.update(keys: kv, values: kv)
        cache.finalize()
        guard case .batch(let offset) = cache.ropeOffset else {
            return XCTFail("expected batch rope offset after finalize")
        }
        XCTAssertEqual(offset.asArray(Int.self), [2, 3])
    }

    func testAttentionCacheSecondChunkDoesNotZero() {
        // 第 2 个 chunk 从 offset=3 开始；pad 条件 cols < pads - offset 全为负，
        // 此时不应再清零任何列。
        let (cache, kv) = makeCache(
            pads: [2, 0],
            rows: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            columns: 3)
        _ = cache.update(keys: kv, values: kv)
        let chunk2 = MLXArray((13...24).map(Float.init)).reshaped([2, 1, 3, 2])
        let (k2, _) = cache.update(keys: chunk2, values: chunk2)
        let data = k2.asType(.float32).asArray(Float.self)
        // 返回张量按行分块：row0 = data[0..<12]、row1 = data[12..<24]。
        // row0 前两列（chunk1 的 pad 列）保持清零；两行的 chunk2 列都原样写入。
        XCTAssertEqual(Array(data[0..<4]), [0, 0, 0, 0])
        XCTAssertEqual(Array(data[6..<12]), (13...18).map(Float.init))
        XCTAssertEqual(Array(data[18..<24]), (19...24).map(Float.init))
    }

    func testAttentionCacheCopyKeepsPaddingAndState() {
        let (cache, kv) = makeCache(
            pads: [2, 0], rows: Array(repeating: 1, count: 12), columns: 3)
        _ = cache.update(keys: kv, values: kv)
        let copied = cache.copy() as! BatchedAttentionKVCache
        guard case .batch(let offset) = copied.ropeOffset else {
            return XCTFail("copied cache lost batch rope offset")
        }
        XCTAssertEqual(offset.asArray(Int.self), [1, 3])
        XCTAssertEqual(copied.offset, cache.offset)
        XCTAssertEqual(copied.state.count, cache.state.count)
    }

    // MARK: - 左 padding 注意力掩码（makeMask）

    func testAttentionCacheMakeMaskBlocksPadColumnsInFirstChunk() {
        // offset 尚未推进（prefill 第一个 chunk 前向），n=3：
        // row0 pad 2 列 ⇒ j<2 全部不可注意；row1 无 pad，纯因果。
        let cache = BatchedAttentionKVCache(leftPadding: [2, 0])
        guard case .array(let mask) = cache.makeMask(
            n: 3, windowSize: nil, returnArray: false)
        else { return XCTFail("expected array mask") }
        XCTAssertEqual(mask.shape, [2, 1, 3, 3])
        let values = mask.asArray(Bool.self)
        XCTAssertEqual(
            values,
            [false, false, false,
             false, false, false,
             false, false, true,
             true, false, false,
             true, true, false,
             true, true, true])
    }

    func testAttentionCacheMakeMaskCausalOnLaterChunk() {
        // 第二 chunk：offset=3，n=2。pad 列（j<2）依然不可注意；
        // row0 的 j=3/4 是自己的 token（位置 3、4 ≥ pad 2），因果可用。
        let cache = BatchedAttentionKVCache(leftPadding: [2, 0])
        let kv = MLXArray(Array(repeating: Float(1), count: 12)).reshaped([2, 1, 3, 2])
        _ = cache.update(keys: kv, values: kv)
        guard case .array(let mask) = cache.makeMask(
            n: 2, windowSize: nil, returnArray: false)
        else { return XCTFail("expected array mask") }
        XCTAssertEqual(mask.shape, [2, 1, 2, 5])
        let values = mask.asArray(Bool.self)
        XCTAssertEqual(
            values,
            [false, false, true, true, false,
             false, false, true, true, true,
             true, true, true, true, false,
             true, true, true, true, true])
    }

    func testAttentionCacheMakeMaskDecodeBlocksPads() {
        // decode 步（n=1）也必须屏蔽本行 pad 列，否则零 K 键会以 0 分
        // 稀释 softmax。offset=3（kv 长度 4）。
        let cache = BatchedAttentionKVCache(leftPadding: [2, 0])
        let kv = MLXArray(Array(repeating: Float(1), count: 12)).reshaped([2, 1, 3, 2])
        _ = cache.update(keys: kv, values: kv)
        guard case .array(let mask) = cache.makeMask(
            n: 1, windowSize: nil, returnArray: false)
        else { return XCTFail("expected array mask") }
        XCTAssertEqual(mask.shape, [2, 1, 1, 4])
        let values = mask.asArray(Bool.self)
        XCTAssertEqual(
            values,
            [false, false, true, true,
             true, true, true, true])
    }

    func testAttentionCacheMakeMaskBackToSymbolicWithoutPadding() {
        let cache = BatchedAttentionKVCache(leftPadding: nil)
        switch cache.makeMask(n: 3, windowSize: nil, returnArray: false) {
        case .causal: break
        default:
            XCTFail("无 padding 时应退回符号因果掩码")
        }
    }

    // MARK: - SSM 掩码（框架能力回归，prefill 依赖它）

    func testMambaCacheLeftPaddingMaskSemantics() throws {
        let cache = MambaCache(leftPadding: [2, 0])
        let input = MLXArray(Array(repeating: Float(0), count: 6)).reshaped([2, 3])
        let mask = createSSMMask(h: input, cache: cache)
        let values = try XCTUnwrap(mask).asArray(Bool.self)
        // row0: pad 2 列 → [F, F, T]；row1: [T, T, T]
        XCTAssertEqual(values, [false, false, true, true, true, true])
    }

    // MARK: - 端到端一致性（env-gated，加载真实模型）

    /// 环境变量指定的本地模型目录。
    private func e2eModelDirectory() throws -> URL {
        guard
            let dir = ProcessInfo.processInfo.environment["BATCH_E2E_MODEL_DIR"],
            !dir.isEmpty
        else {
            throw XCTSkip("需要 BATCH_E2E_MODEL_DIR 指向本地模型目录")
        }
        return URL(fileURLWithPath: dir)
    }

    /// 加载环境变量指定的本地模型。
    private func loadE2EContainer() async throws -> ModelContainer {
        try await LLMModelFactory.shared.loadContainer(
            from: try e2eModelDirectory(),
            using: LocalTokenizerLoader())
    }

    /// 渲染 E2E 用的三条提示（长度不同，保证存在左 padding 行）。
    private func prepareE2ERows(_ container: ModelContainer) async throws -> [[Int]] {
        let prompts = [
            "用一句话介绍苹果公司。",
            "1+1 等于几？直接回答。",
            "写一句关于秋天的话。",
        ]
        let context: [String: any Sendable] = ["enable_thinking": false]
        var rows: [[Int]] = []
        rows.reserveCapacity(prompts.count)
        for prompt in prompts {
            let input = try await container.prepare(input: UserInput(
                chat: [.system("你是简洁的助手。"), .user(prompt)],
                tools: nil,
                additionalContext: context))
            rows.append(input.text.tokens.asArray(Int.self))
        }
        return rows
    }

    /// 与生产循环一致的停止集合：model config eos + tokenizer eos + extraEOSTokens。
    private func stopTokenIds(
        tokenizer: any Tokenizer, configuration: ModelConfiguration
    ) -> Set<Int> {
        var ids = configuration.eosTokenIds
        if let eos = tokenizer.eosTokenId {
            ids.insert(eos)
        }
        for token in configuration.extraEOSTokens {
            if let id = tokenizer.convertTokenToId(token) {
                ids.insert(id)
            }
        }
        return ids
    }

    func testBatchGreedyMatchesSequentialGreedy() async throws {
        // 贪心采样下，验证批量路径与逐条路径的贪心前缀一致。
        // 运行方式：BATCH_E2E_MODEL_DIR=/path/to/model swift test --filter BatchGreedy
        let container = try await loadE2EContainer()
        let model = await container.perform { UncheckedSendableBox($0.model) }.value
        let tokenizer = await container.tokenizer
        let configuration = await container.configuration
        guard let padToken = tokenizer.eosTokenId else {
            throw XCTSkip("tokenizer 没有 eos token")
        }
        let stops = stopTokenIds(tokenizer: tokenizer, configuration: configuration)
        let rows = try await prepareE2ERows(container)

        // 顺序贪心基线：每条提示独立走 TokenIterator。
        // TokenIterator 自身只按 maxTokens 停止；生产循环（runSynchronousGenerationLoop）
        // 会额外在停止 token/unknown token 处终止，基线必须复刻这一语义。
        let parameters = GenerateParameters(maxTokens: 100, temperature: 0)
        var sequential: [[Int]] = []
        let sequentialStart = Date()
        for row in rows {
            let cache = try model.newCache(parameters: parameters)
            var iterator = try TokenIterator(
                input: LMInput(text: .init(tokens: MLXArray(row))),
                model: model, cache: cache,
                parameters: parameters)
            var generated: [Int] = []
            while let token = iterator.next() {
                if stops.contains(token) || token == tokenizer.unknownTokenId { break }
                generated.append(token)
            }
            sequential.append(generated)
        }
        let sequentialElapsed = Date().timeIntervalSince(sequentialStart)

        let plan = try XCTUnwrap(
            BatchGenerationPlan.leftPad(rows: rows, padToken: padToken))
        let batchStart = Date()
        let batched = try BatchedMLXGenerator.generate(
            tokens: plan,
            model: model,
            stopTokenIds: stops,
            padToken: padToken,
            maxTokensPerRow: 100,
            temperature: 0)
        let batchElapsed = Date().timeIntervalSince(batchStart)
        // 性能基准：同一批提示，串行逐条 vs 单线程批量（仅作参照，不做断言，
        // 避免不同机器的绝对耗时让测试不稳定）。
        print("[batch-e2e] 行数=\(rows.count) 串行=\(sequentialElapsed)s 批量=\(batchElapsed)s")

        XCTAssertEqual(batched.count, sequential.count)
        for row in sequential.indices {
            // 批量 decode 使用带显式掩码的通用 kernel，单流 decode 走 compiled
            // 快路径，浮点累加顺序不同；长序列出现近邻 argmax 翻转属预期，
            // 因此这里校验前缀（prefill 的 pad 掩码/RoPE 数学正确性）。
            let prefix = min(8, sequential[row].count, batched[row].count)
            XCTAssertEqual(
                Array(batched[row].prefix(prefix)), Array(sequential[row].prefix(prefix)),
                "第 \(row) 行 batch 与顺序贪心前缀不一致")
        }
    }

    func testGenerateSummariesBatchReturnsSummaries() async throws {
        // 服务层集成：真实模型 + chat 模板 + 批量生成 + decode 全链路。
        // 需要 Qwen3.5 系（SSM 混合架构）模型目录。
        let service = MLXLLMService(
            directory: try e2eModelDirectory(),
            modelName: "batch-e2e",
            maxTokens: 128,
            enableThinking: false,
            contextWindow: 8192,
            kvReuse: false,
            kvInt8: false)

        let requests = [
            (system: "你是简洁的摘要助手。", prompt: "用一句话总结：苹果公司是一家设计消费电子产品的公司。"),
            (system: "你是简洁的摘要助手。", prompt: "用一句话总结：秋天是收获的季节。"),
            (system: "你是简洁的摘要助手。", prompt: "直接回答：一加一等于几？"),
        ]
        let summaries = try await service.generateSummariesBatch(
            requests: requests, temperature: 0.2, maxTokensPerRow: 64)
        let result = try XCTUnwrap(summaries, "批量摘要应返回结果（模型需为 Qwen3.5 混合架构）")
        XCTAssertEqual(result.count, requests.count)
        XCTAssertTrue(result.allSatisfy { !$0.isEmpty }, "每行摘要都不应为空：\(result)")
    }

    func testBatchPrefillLogitsMatchUnpaddedPrefill() async throws {
        // 左 padding 的 prefill 数学必须与单条未 padding 的 prefill 等价：
        // 比对最后位置 logits 的最大绝对误差与贪心 token。
        let container = try await loadE2EContainer()
        let model = await container.perform { UncheckedSendableBox($0.model) }.value
        let tokenizer = await container.tokenizer
        guard let padToken = tokenizer.eosTokenId else {
            throw XCTSkip("tokenizer 没有 eos token")
        }
        let rows = try await prepareE2ERows(container)
        let plan = try XCTUnwrap(
            BatchGenerationPlan.leftPad(rows: rows, padToken: padToken))
        let paddedLength = plan.tokens[0].count

        let caches = try BatchedMLXGenerator.makeCaches(
            model: model, leftPadding: plan.leftPadding)
        let batched = model(
            LMInput.Text(tokens: MLXArray(plan.tokens)), cache: caches, state: nil)
        let batchLogits = try XCTUnwrap(batched.logits)

        for (row, tokens) in rows.enumerated() {
            let cache = try model.newCache(parameters: nil)
            let single = model(
                LMInput.Text(tokens: MLXArray([tokens])), cache: cache, state: nil)
            let singleLogits = try XCTUnwrap(single.logits)
            let paddedLast = batchLogits[row, paddedLength - 1]
            let singleLast = singleLogits[0, tokens.count - 1]
            let maxDiff = (paddedLast - singleLast).abs().max().item(Float.self)
            // 实测：即使完全没有 padding，仅 B=1→B=3 的 bf16 kernel 差异就有
            // 约 0.33 的 logits 偏差；这里用 0.6 作为「没有实质性错位」的上界，
            // 并以 top-1 完全一致作为正确性判据。
            XCTAssertLessThan(
                maxDiff, 0.6, "第 \(row) 行左 padding 导致 logits 偏差过大：\(maxDiff)")
            XCTAssertEqual(
                argMax(paddedLast, axis: -1).item(Int.self),
                argMax(singleLast, axis: -1).item(Int.self),
                "第 \(row) 行贪心 token 不一致")
        }
    }
}