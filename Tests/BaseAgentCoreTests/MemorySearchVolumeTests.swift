import Foundation
import XCTest
@testable import BaseAgentCore

/// 模拟测量：一次 `long_memory(op="search")` 收紧到 1000 字符后能带回多少信息。
///
/// 覆盖冷启动 / 典型 / 大图 / hub / 超长观察，以及语义通道与关键词回退通道，
/// 量化字符数、估算 token 数与各段条数，并标注是否触发了一跳扩展。
final class MemorySearchVolumeTests: XCTestCase {

    private var dirs: [URL] = []

    override func tearDown() {
        for dir in dirs { try? FileManager.default.removeItem(at: dir) }
        dirs.removeAll()
        super.tearDown()
    }

    /// 可控的假 embedding：含 "PostgreSQL" 的观察余弦 0.75（≥0.5 命中），其余为 0。
    private final class SimulatedEmbedding: EmbeddingProvider, @unchecked Sendable {
        let identifier = "simulated-embedding"
        let dimension = 4
        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map(Self.vector)
        }
        func embedQuery(_ text: String) async throws -> [Float] { [1, 0, 0, 0] }
        private static func vector(_ text: String) -> [Float] {
            text.lowercased().contains("postgresql")
                ? [0.75, 0.6614, 0, 0]
                : [0, 1, 0, 0]
        }
    }

    // MARK: - 夹具

    private func makeFixture(entityCount: Int, observationsPerEntity: Int,
                             relationCount: Int, hub: Bool,
                             longObservations: Bool = false,
                             embedding: EmbeddingProvider?)
        -> (GraphMemoryStore, MemorySearchTool) {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("memory-volume-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        dirs.append(dir)
        let graph = GraphMemoryStore(
            path: dir.appendingPathComponent("long_memory.graph.json").path,
            embedding: embedding)

        func name(_ index: Int) -> String { index == 0 ? "PostgreSQL" : "实体\(index)" }

        for index in 0..<entityCount {
            var observations: [String] = []
            for j in 0..<observationsPerEntity {
                let relevant = (index + j) % 3 == 0
                let topic = relevant ? "PostgreSQL" : "组件\(index)"
                let base = "\(name(index)) 在\(topic)场景下承担第\(j)项稳定职责，"
                    + "涉及配置项、版本约束与长期适用边界，需要跨会话保留"
                observations.append(longObservations
                    ? base + String(repeating: "补充说明与上下文细节", count: 30)
                    : base)
            }
            graph.upsertEntity(name(index), type: index == 0 ? "tool" : "concept",
                               observations: observations)
        }
        for k in 0..<relationCount {
            let src = hub ? 0 : k % entityCount
            let dst = hub ? (k + 1) % entityCount : (k * 7 + 1) % entityCount
            if src != dst {
                graph.createRelation(source: name(src), target: name(dst), label: "依赖")
            }
        }
        var config = AgentConfig()
        config.workspace = dir.path
        return (graph, MemorySearchTool(config: config, graphMemory: graph))
    }

    // MARK: - 单次测量

    private func measure(scenario: String, channel: String, topK: Int,
                         graph: GraphMemoryStore, tool: MemorySearchTool,
                         query: String = "PostgreSQL") async throws -> String {
        let result = try await tool.run([
            "query": .string(query),
            "top_k": .int(topK),
        ])
        XCTAssertTrue(result.success, result.error)
        let object = try XCTUnwrap(JSONValue.parse(result.output)?.objectValue)
        XCTAssertLessThanOrEqual(result.output.count,
                                 MemorySearchTool.maxOutputChars)

        let hits = object["hits"]?.arrayValue?.count ?? 0
        let entities = object["entities"]?.arrayValue?.count ?? 0
        let relations = object["relations"]?.arrayValue?.count ?? 0
        let facts = object["related_facts"]?.arrayValue?.count ?? 0
        let chars = result.output.count
        let tokens = TokenEstimator.estimate(result.output)
        let hasNeighbor = ((object["entities"]?.arrayValue ?? []).compactMap(\.objectValue))
            .contains { $0["role"]?.stringValue == "neighbor" }
        let expanded = hasNeighbor || facts > 0

        // 同一输入、放宽到 6000 字符时的参考量级，用于对比收紧效果。
        let recalled = await graph.search(query: query, topK: topK)
        let looseChars = MemorySearchTool
            .buildPayload(hits: recalled, graph: graph, budget: 6_000)
            .jsonString().count

        return "\(scenario) | \(channel) | top_k=\(topK) | hits=\(hits) "
            + "| entities=\(entities) | relations=\(relations) | facts=\(facts) "
            + "| chars=\(chars) | tokens=\(tokens) | loose=\(looseChars) "
            + "| expanded=\(expanded ? "是" : "否")"
    }

    // MARK: - 模拟

    func testMeasureRecallVolume() async throws {
        var lines: [String] = [
            "场景 | 通道 | top_k | 命中 | 实体 | 关系 | 相关事实 | 字符 | 估算token | 放宽到6000 | 是否扩展",
            "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
        ]

        // 1) 冷启动：刚清理过图谱，只有少量事实。
        for (channel, embedding) in channels() {
            let (graph, tool) = makeFixture(entityCount: 3, observationsPerEntity: 2,
                                            relationCount: 2, hub: false,
                                            embedding: embedding)
            lines.append(try await measure(scenario: "冷启动 3实体/6观察", channel: channel,
                                           topK: 5, graph: graph, tool: tool))
        }

        // 2) 典型积累：30 实体 / 90 观察 / 45 关系。
        for (channel, embedding) in channels() {
            let (graph, tool) = makeFixture(entityCount: 30, observationsPerEntity: 3,
                                            relationCount: 45, hub: false,
                                            embedding: embedding)
            lines.append(try await measure(scenario: "典型 30实体/90观察", channel: channel,
                                           topK: 5, graph: graph, tool: tool))
        }

        // 3) 大图：100 实体 / 400 观察 / 200 关系。
        let (bigGraph, bigTool) = makeFixture(entityCount: 100, observationsPerEntity: 4,
                                              relationCount: 200, hub: false,
                                              embedding: SimulatedEmbedding())
        lines.append(try await measure(scenario: "大图 100实体/400观察", channel: "语义",
                                       topK: 5, graph: bigGraph, tool: bigTool))

        // 4) hub：单中心连 59 个邻居。
        let (hubGraph, hubTool) = makeFixture(entityCount: 60, observationsPerEntity: 3,
                                              relationCount: 59, hub: true,
                                              embedding: SimulatedEmbedding())
        lines.append(try await measure(scenario: "hub 60实体/59关系", channel: "语义",
                                       topK: 5, graph: hubGraph, tool: hubTool))

        // 5) top_k 扫描（典型图、语义通道）。
        let (scanGraph, scanTool) = makeFixture(entityCount: 30, observationsPerEntity: 3,
                                                relationCount: 45, hub: false,
                                                embedding: SimulatedEmbedding())
        for topK in [1, 5, 10, 20] {
            lines.append(try await measure(scenario: "top_k扫描", channel: "语义",
                                           topK: topK, graph: scanGraph, tool: scanTool))
        }

        // 6) 超长观察：预算耗尽，扩展应被放弃。
        let (longGraph, longTool) = makeFixture(entityCount: 20, observationsPerEntity: 2,
                                                relationCount: 30, hub: false,
                                                longObservations: true,
                                                embedding: SimulatedEmbedding())
        lines.append(try await measure(scenario: "超长观察×40", channel: "语义",
                                       topK: 20, graph: longGraph, tool: longTool))

        print("[memory-volume]\n" + lines.joined(separator: "\n"))
    }

    private func channels() -> [(String, EmbeddingProvider?)] {
        [("关键词回退", nil), ("语义", SimulatedEmbedding())]
    }
}
