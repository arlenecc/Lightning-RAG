import Foundation
import XCTest
@testable import BaseAgentCore

/// 真机模型抽取质量验证（默认跳过）。
/// 运行：`KG_REAL_MODEL_TEST=1 swift test --filter KnowledgeGraphRealModelTests`
/// 可选：`KG_REAL_ONLY=2B|0.8B` 只跑一个模型，`KG_REAL_MODEL_DIR` 指定权重目录
///（默认 `models/Qwen3.5-0.8B-MLX-4bit`），`KG_REAL_FORMAT=delimited` 切回分隔符输出
///（默认 JSON），`KG_REAL_PENALTY` 覆盖重复惩罚。
final class KnowledgeGraphRealModelTests: XCTestCase {

    private let sample = """
    # 项目与团队

    Alice Chen 是 Acme Corp 的软件工程师，负责领导 Project Aurora。
    Project Aurora 是一个知识图谱项目，使用 Swift 和 MLX 开发。
    Acme Corp 总部位于 Shanghai，成立于 2010 年。
    Bob Li 与 Alice Chen 在 Project Aurora 上协作，Bob 负责数据管道。
    """

    /// 核心实体（人/组织/项目）——抽取质量必须命中。
    private let coreEntities = ["Alice Chen", "Acme Corp", "Project Aurora", "Bob Li"]
    /// 次要实体（技术/地点）——2B 不保证，作为质量参考不计入断言。
    private let extraEntities = ["Swift", "MLX", "Shanghai"]

    func testAuxModelExtractionQuality() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["KG_REAL_MODEL_TEST"] == "1",
                          "设置 KG_REAL_MODEL_TEST=1 才运行真实模型测试")
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let only = ProcessInfo.processInfo.environment["KG_REAL_ONLY"]?.uppercased()
        // 2B 是当前选用的辅助模型：硬断言抽取质量。
        if only == nil || only == "2B" {
            let primary = root.appendingPathComponent("models/Qwen3.5-2B-MLX-4bit")
            if FileManager.default.fileExists(atPath: primary.path) {
                try await measure(name: "Qwen3.5-2B-MLX-4bit", directory: primary,
                                  maxTokens: 1_024, assertsQuality: true)
            } else {
                print("REAL Qwen3.5-2B-MLX-4bit: 模型不存在，跳过")
            }
        }
        // 4B 主模型也可用作辅助模型：用于对照「更慢但更稳」的取舍。
        if only == nil || only == "4B" {
            let main = root.appendingPathComponent("models/qwen3.5-4b-agentic-coder-v4-4bit-mlx")
            if FileManager.default.fileExists(atPath: main.path) {
                try await measure(name: "qwen3.5-4b-agentic-coder-v4", directory: main,
                                  maxTokens: 1_024, assertsQuality: true)
            } else {
                print("REAL qwen3.5-4b-agentic-coder-v4: 模型不存在，跳过")
            }
        }
        // 0.8B 仅作基线对照，不判定：小模型输出会缺字段/循环，不代表当前选型。
        if only == nil || only == "0.8B" {
            let baseline = ProcessInfo.processInfo.environment["KG_REAL_MODEL_DIR"]
                .map { URL(fileURLWithPath: $0) }
                ?? root.appendingPathComponent("models/Qwen3.5-0.8B-MLX-4bit")
            if FileManager.default.fileExists(atPath: baseline.path) {
                try await measure(name: "Qwen3.5-0.8B-MLX-4bit", directory: baseline,
                                  maxTokens: 1_024, assertsQuality: false)
            } else {
                print("REAL Qwen3.5-0.8B-MLX-4bit: 模型不存在，跳过")
            }
        }
    }

    private func measure(name: String, directory: URL,
                         maxTokens: Int, assertsQuality: Bool) async throws {
        var config = AgentConfig()
        config.workspace = FileManager.default.currentDirectoryPath
        config.auxLLMProvider = "local"
        config.auxLocalLLMBackend = "mlx-lm"
        config.auxLocalLLMModel = directory.path
        config.auxMaxContextTokens = 8_192
        config.auxLocalLLMEnableThinking = false
        let env = ProcessInfo.processInfo.environment
        let format = KGExtractionFormat(stored: env["KG_REAL_FORMAT"] ?? "json")
        if let value = env["KG_REAL_PENALTY"].flatMap(Double.init) {
            config.auxRepetitionPenalty = value
        }
        // JSON 模式输出比分隔符行更冗长，给足 token 预算，避免测试自身截断。
        config.auxMaxTokens = format == .json ? max(maxTokens, 2_048) : maxTokens

        let llm = LocalModelFactory.shared.makeAuxLLM(config: config)
        if let preloadable = llm as? PreloadableLLMService {
            try await preloadable.preload()
        }

        // 用真实 builder 的抽取路径（含完成标记/重复行提前收流），
        // 复现生产上的实际生成行为，而不是另写一份裸流式调用。
        let directoryTemp = FileManager.default.temporaryDirectory
            .appendingPathComponent("kg-real-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directoryTemp,
                                                withIntermediateDirectories: true)
        let store = try KnowledgeGraphStore(
            dbPath: directoryTemp.appendingPathComponent("kg.sqlite").path,
            embedding: StubEmbedding(), enableSQLiteVec: false)
        let graph = KnowledgeGraph(
            snapshotPath: directoryTemp.appendingPathComponent("snapshot.json").path)
        let builder = KnowledgeGraphBuilder(store: store, graph: graph, llm: llm,
                                            format: format)

        let started = Date()
        let raw = try await builder.extractRaw(
            chunk: sample, sectionContext: "H1 → 项目与团队")
        let seconds = Date().timeIntervalSince(started)
        // 收流保护会取消底层生成任务；测试进程紧接着退出，先等 GPU 收尾。
        LocalModelFactory.shared.synchronizeGPU()
        let parsed = format == .json
            ? KnowledgeGraphBuilder.parseExtractionJSON(raw)
            : KnowledgeGraphBuilder.parseExtraction(raw)

        let entityNames = Set(parsed.entities.map { $0.name.lowercased() })
        func matched(_ list: [String]) -> [String] {
            list.filter { expected in
                let needle = expected.lowercased()
                return entityNames.contains { $0.contains(needle) || needle.contains($0) }
            }
        }
        let matchedCore = matched(coreEntities)
        let matchedExtra = matched(extraEntities)
        let validRelations = parsed.relations.filter { relation in
            entityNames.contains(relation.source.lowercased())
                && entityNames.contains(relation.target.lowercased())
        }
        // 图谱按无向实体对合并；缺失端点会以 UNKNOWN 占位落库，
        // 因此这里统计所有解析出的实体对，而不是只看端点都在实体表内的关系。
        let uniquePairs = Set(parsed.relations.map { relation in
            [relation.source.lowercased(), relation.target.lowercased()]
                .sorted().joined(separator: "|")
        })
        let completed = format == .json
            ? (raw.contains("{") && raw.contains("}"))
            : raw.contains("<|COMPLETE|>")
        print("""
        REAL \(name) [\(format.rawValue)]: \(String(format: "%.1f", seconds))s chars=\(raw.count)
          entities=\(parsed.entities.count) relations=\(parsed.relations.count)
          matchedCore=\(matchedCore.count)/\(coreEntities.count) \(matchedCore)
          matchedExtra=\(matchedExtra.count)/\(extraEntities.count) \(matchedExtra)
          validRelations=\(validRelations.count)/\(parsed.relations.count) \
        uniquePairs=\(uniquePairs.count)
          completed=\(completed)
          entityTypes=\(parsed.entities.map { "\($0.name):\($0.type)" })
          triples=\(uniquePairs.sorted())
        RAW:
        \(raw)
        """)
        // 进程退出时 MLX 释放可能崩溃导致 stdout 缓冲丢失，这里立即刷出报告。
        fflush(stdout)
        // 收流保护会取消底层生成任务；等取消的 MLX 任务真正收尾，再卸载权重，
        // 否则进程退出时销毁 command encoder 会与在途工作并发而崩溃。
        try await Task.sleep(nanoseconds: 2_000_000_000)
        if let service = llm as? AsyncUnloadableLLMService {
            await service.unloadAndWait()
        }
        LocalModelFactory.shared.unloadLLMCaches()
        LocalModelFactory.shared.synchronizeGPU()

        guard assertsQuality else { return }
        XCTAssertFalse(parsed.entities.isEmpty, "\(name) 没有抽出实体")
        XCTAssertGreaterThanOrEqual(matchedCore.count, 3,
                                    "\(name) 核心实体命中不足：\(matchedCore)")
        XCTAssertGreaterThanOrEqual(validRelations.count, 3,
                                    "\(name) 有效关系不足")
        XCTAssertGreaterThanOrEqual(uniquePairs.count, 4,
                                    "\(name) 去重后的实体对不足")
        // 收流 + 一次补抽应让输出保持在有限长度，不会无限复读到 token 上限。
        XCTAssertLessThan(raw.count, 8_000, "\(name) 输出过长，疑似未触发收流保护")
    }
}

/// 抽取阶段不访问向量库，构造 store 只需满足接口。
private final class StubEmbedding: EmbeddingProvider {
    let identifier = "kg-real-stub"
    let dimension = 8

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map { _ in [Float](repeating: 0, count: dimension) }
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        [Float](repeating: 0, count: dimension)
    }
}
