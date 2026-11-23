import XCTest
@testable import BaseAgentCore

/// 声明了输入窗口的假 embedding：切片收口逻辑的确定性测试替身。
private final class WindowedFakeEmbedding: EmbeddingProvider, EmbeddingTokenCounting {
    let identifier = "windowed-fake"
    let dimension = 4
    let window = 512

    var maxContentTokens: Int { window - 2 }

    func makeTokenCounter() async -> (@Sendable (String) -> Int)? { nil }

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map { _ in [Float](repeating: 1, count: dimension) }
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        [Float](repeating: 1, count: dimension)
    }
}

/// 未声明窗口的 embedding（如 API/第三方后端）：不得被收口。
private final class PlainFakeEmbedding: EmbeddingProvider {
    let identifier = "plain-fake"
    let dimension = 4

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map { _ in [Float](repeating: 1, count: dimension) }
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        [Float](repeating: 1, count: dimension)
    }
}

/// 切片参数一致性：设置面板的切片设置必须贯通所有分片环节
///（切分 → 入库 → 检索呈现），任何环节都不许再按固定短值处理。
final class ChunkParameterConsistencyTests: XCTestCase {

    func testChunkSizeIsClampedToEmbeddingWindowWithOverlapScaled() {
        var config = AgentConfig()
        config.ragChunkSize = 1_200
        config.ragChunkOverlap = 120

        let resolved = RAGEngine.resolvedChunkingOptions(
            config: config, embedding: WindowedFakeEmbedding())
        XCTAssertEqual(resolved.clampedFrom, 1_200, "收口前的用户设置值必须回传，用于日志")
        XCTAssertEqual(resolved.options.chunkSizeTokens, 510, "不得超模型窗口（512 − 2 特殊 token）")
        XCTAssertEqual(resolved.options.overlapTokens, 51, "overlap 计入模型输入，按 10% 同步收口")
    }

    func testChunkSizeBelowWindowIsKeptVerbatim() {
        var config = AgentConfig()
        config.ragChunkSize = 400
        config.ragChunkOverlap = 40

        let resolved = RAGEngine.resolvedChunkingOptions(
            config: config, embedding: WindowedFakeEmbedding())
        XCTAssertNil(resolved.clampedFrom, "未超窗口时不得改动用户设置")
        XCTAssertEqual(resolved.options.chunkSizeTokens, 400)
        XCTAssertEqual(resolved.options.overlapTokens, 40)
    }

    func testChunkSizeIsNotClampedWithoutWindowDeclaration() {
        var config = AgentConfig()
        config.ragChunkSize = 1_800
        config.ragChunkOverlap = 180

        let resolved = RAGEngine.resolvedChunkingOptions(
            config: config, embedding: PlainFakeEmbedding())
        XCTAssertNil(resolved.clampedFrom)
        XCTAssertEqual(resolved.options.chunkSizeTokens, 1_800,
                       "后端未声明窗口时保持用户设置，不能私自切短")
    }

    func testResultCharLimitCoversLongEnglishChunks() {
        // 英文安全换算按 4 字符/token：500 token 切片至少 2064 字符，
        // 默认 1500 字符不能把完整切片截短。
        XCTAssertEqual(RAGEngine.resultCharLimit(configured: 1_500, chunkTokens: 500), 2_064)
        XCTAssertEqual(RAGEngine.resultCharLimit(configured: 1_500, chunkTokens: 800), 3_264)
        XCTAssertEqual(RAGEngine.resultCharLimit(configured: 1_500, chunkTokens: 1_200), 4_864)
        // 用户显式配置更大的呈现上限时不得被压缩。
        XCTAssertEqual(RAGEngine.resultCharLimit(configured: 8_000, chunkTokens: 500), 8_000)
    }

    func testOutputCharLimitAlignsWithKernelToolBudget() {
        var config = AgentConfig()
        config.ragSearchMaxTotalChars = 12_000

        config.kernelMaxToolChars = 6_000
        XCTAssertEqual(RAGEngine.outputCharLimit(config: config, maxTotalChars: nil), 6_000,
                       "RAG 输出不能超过内核工具结果上限，否则会被二次静默截断")

        config.kernelMaxToolChars = 8_000
        XCTAssertEqual(RAGEngine.outputCharLimit(config: config, maxTotalChars: nil), 8_000)

        // RAG 自身更小的上限优先。
        config.ragSearchMaxTotalChars = 5_000
        XCTAssertEqual(RAGEngine.outputCharLimit(config: config, maxTotalChars: nil), 5_000)

        // 内核未配置时使用与默认值一致的 12000 兜底。
        config.kernelMaxToolChars = 0
        config.ragSearchMaxTotalChars = 20_000
        XCTAssertEqual(RAGEngine.outputCharLimit(config: config, maxTotalChars: nil), 12_000)

        // 调用方显式指定时优先；但不得低于 1000 字符的最低可用上下文。
        XCTAssertEqual(RAGEngine.outputCharLimit(config: config, maxTotalChars: 3_000), 3_000)
        config.ragSearchMaxTotalChars = 500
        config.kernelMaxToolChars = 8_000
        XCTAssertEqual(RAGEngine.outputCharLimit(config: config, maxTotalChars: nil), 1_000)
    }

    func testGGUFEmbeddingDeclaresItsContentWindow() {
        // GGUF 后端原先在 embed() 里静默 prefix(2046)：切片环节不知情，
        // 长切片的尾部会被模型丢掉。现在必须显式声明窗口。
        XCTAssertEqual(LlamaCppEmbeddingProvider.contentTokenLimit,
                       EmbeddingInputLimits.maxTokens - 2,
                       "2048 上下文在推理前压到 1024 token，再扣 BOS/EOS")
        XCTAssertGreaterThan(LlamaCppEmbeddingProvider.contentTokenLimit, 1_000,
                             "压到 1024 后仍应保留足够的长切片预算")

        // 建实例不加载权重：maxContentTokens 必须与静态换算一致，
        // 并扣除 nomic 的 document 前缀预留。
        let provider = LlamaCppEmbeddingProvider(modelPath: "/tmp/not-loaded.gguf",
                                                 modelName: "nomic-embed-text-v1.5")
        let reserve = EmbeddingPrefixStrategy(modelName: "nomic-embed-text-v1.5")
            .documentPrefixTokenReserve
        XCTAssertEqual(reserve, 8)
        XCTAssertEqual(provider.maxContentTokens,
                       LlamaCppEmbeddingProvider.contentTokenLimit - reserve)
    }

    func testAgentRebuildKeysSeparateLLMOnlyChanges() {
        var base = AgentConfig()
        base.ragChunkSize = 500
        base.temperature = 0.7

        var onlyTemperature = base
        onlyTemperature.temperature = 0.9
        XCTAssertTrue(onlyTemperature
            .agentRebuildKeys(comparedTo: base, allowing: AgentConfig.mainLLMOnlyKeys)
            .isEmpty, "仅调温度不必重建 Agent")

        var resized = base
        resized.ragChunkSize = 1_200
        resized.ragSemanticChunking = false
        XCTAssertEqual(
            resized.agentRebuildKeys(comparedTo: base, allowing: AgentConfig.mainLLMOnlyKeys),
            ["rag_chunk_size", "rag_semantic_chunking"],
            "切片参数变化必须触发重建，否则 RAGEngine 仍按旧快照切片")
        XCTAssertEqual(
            resized.agentRebuildKeys(comparedTo: base, allowing: AgentConfig.auxLLMOnlyKeys),
            ["rag_chunk_size", "rag_semantic_chunking"],
            "辅助槽位同样不能吞掉切片参数变化")

        var toggledAux = base
        toggledAux.auxEnabled = !base.auxEnabled
        XCTAssertTrue(toggledAux
            .agentRebuildKeys(comparedTo: base, allowing: AgentConfig.auxLLMOnlyKeys)
            .isEmpty, "辅助模型开关属于辅助槽位自身参数")
        XCTAssertEqual(
            toggledAux.agentRebuildKeys(comparedTo: base, allowing: AgentConfig.mainLLMOnlyKeys),
            ["aux_enabled"], "主槽位不得吞掉辅助模型开关")

        // 辅助模型的 KV 复用 / 量化是独立参数：重载辅助模型即可生效，
        // 但换主模型不能把它的变化一起吞掉。
        var toggledAuxKV = base
        toggledAuxKV.auxKvReuse.toggle()
        toggledAuxKV.auxKvCacheInt8.toggle()
        XCTAssertTrue(toggledAuxKV
            .agentRebuildKeys(comparedTo: base, allowing: AgentConfig.auxLLMOnlyKeys)
            .isEmpty, "辅助 KV 开关属于辅助槽位自身参数")
        XCTAssertEqual(
            toggledAuxKV.agentRebuildKeys(comparedTo: base,
                                          allowing: AgentConfig.mainLLMOnlyKeys),
            ["aux_kv_cache_int8", "aux_llm_kv_reuse"], "主槽位不得吞掉辅助 KV 设置")

        var toggledMainKV = base
        toggledMainKV.llmKVReuse.toggle()
        toggledMainKV.kvCacheInt8.toggle()
        XCTAssertTrue(toggledMainKV
            .agentRebuildKeys(comparedTo: base, allowing: AgentConfig.mainLLMOnlyKeys)
            .isEmpty, "主 KV 开关属于主槽位自身参数")
        XCTAssertEqual(
            toggledMainKV.agentRebuildKeys(comparedTo: base,
                                           allowing: AgentConfig.auxLLMOnlyKeys),
            ["kv_cache_int8", "llm_kv_reuse"], "辅助槽位不得吞掉主模型 KV 设置")

        XCTAssertTrue(base
            .agentRebuildKeys(comparedTo: base, allowing: AgentConfig.mainLLMOnlyKeys)
            .isEmpty, "配置未变时不应触发任何重建")
    }
}