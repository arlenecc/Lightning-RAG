import Foundation
import XCTest
@testable import BaseAgentCore

/// 长期记忆检索的收紧契约：≤1000 字符、按时间取最近 5 条观察、
/// 先保直接关联实体/关系，仅在有余额时做一跳扩展；以及观察时间戳的持久化兼容。
final class MemorySearchNeighborhoodTests: XCTestCase {

    private var workspace: URL!
    private var graph: GraphMemoryStore!
    private var tool: MemorySearchTool!

    override func setUp() {
        super.setUp()
        workspace = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("memory-search-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        // 不注入 embedding：search() 走关键词回退路径，结果确定、无需真实模型。
        graph = GraphMemoryStore(path: workspace
            .appendingPathComponent("long_memory.graph.json").path)
        var config = AgentConfig()
        config.workspace = workspace.path
        tool = MemorySearchTool(config: config, graphMemory: graph)
        let dir = workspace
        addTeardownBlock { [weak self, dir] in
            self?.graph.flush()
            if let dir { try? FileManager.default.removeItem(at: dir) }
        }
    }

    // MARK: - 夹具与辅助

    private func seedGraph() {
        graph.upsertEntity("PostgreSQL", type: "tool", observations: [
            "PostgreSQL 承担星图项目的主数据库角色",
            "PostgreSQL 禁止在生产环境启用 ORM 自动建表",
        ])
        graph.upsertEntity("星图", type: "project", observations: [
            "星图项目使用 PostgreSQL 15 作为主数据库",
        ])
        graph.upsertEntity("用户", type: "person", observations: [
            "用户偏好在 SQL 访问中不使用 ORM",
        ])
        graph.createRelation(source: "星图", target: "PostgreSQL", label: "使用")
        graph.createRelation(source: "用户", target: "星图", label: "负责")
    }

    private func search(_ query: String, topK: Int = 5) async throws -> [String: JSONValue] {
        let result = try await tool.run([
            "query": .string(query),
            "top_k": .int(topK),
        ])
        XCTAssertTrue(result.success, result.error)
        guard let object = JSONValue.parse(result.output)?.objectValue else {
            XCTFail("返回不是 JSON 对象: \(result.output)")
            return [:]
        }
        return object
    }

    private static func objects(_ value: JSONValue?) -> [[String: JSONValue]] {
        (value?.arrayValue ?? []).compactMap(\.objectValue)
    }

    // MARK: - 1000 字符预算

    func testReturnsCoreWithinBudgetAndCompleteJSON() async throws {
        seedGraph()
        let result = try await tool.run(["query": .string("PostgreSQL")])
        XCTAssertTrue(result.success, result.error)
        XCTAssertLessThanOrEqual(result.output.count,
                                 MemorySearchTool.maxOutputChars)
        let object = try XCTUnwrap(JSONValue.parse(result.output)?.objectValue)
        XCTAssertNotNil(object["hits"])
        XCTAssertNotNil(object["entities"])
        XCTAssertNotNil(object["relations"])
        XCTAssertNotNil(object["related_facts"])
        XCTAssertFalse(object["hits"]?.arrayValue?.isEmpty ?? true)
    }

    func testBudgetNeverExceededAcrossShapes() {
        for index in 0..<12 {
            graph.upsertEntity("实体\(index)", type: "concept", observations: [
                String(repeating: "观察内容", count: 20 + index * 5),
            ])
            graph.createRelation(source: "实体0", target: "实体\(index)", label: "关联")
        }
        let hits = (0..<12).map {
            GraphObservationHit(entity: "实体\($0)",
                                text: String(repeating: "观察内容", count: 20 + $0 * 5),
                                score: 0.9, timestamp: Double($0))
        }
        for budget in [1_000, 800, 600, 400] {
            let payload = MemorySearchTool.buildPayload(hits: hits, graph: graph,
                                                        budget: budget)
            XCTAssertLessThanOrEqual(payload.jsonString().count, budget,
                                     "预算 \(budget) 被突破")
            XCTAssertNotNil(payload.objectValue, "裁剪后仍是完整 JSON")
        }
    }

    func testNoHitReturnsEmptySections() async throws {
        seedGraph()
        let object = try await search("完全不存在的检索词")
        XCTAssertEqual(object["hits"]?.arrayValue?.count ?? 0, 0)
        XCTAssertEqual(object["entities"]?.arrayValue?.count ?? 0, 0)
        XCTAssertEqual(object["relations"]?.arrayValue?.count ?? 0, 0)
        XCTAssertEqual(object["related_facts"]?.arrayValue?.count ?? 0, 0)
    }

    func testEmptyQueryFails() async throws {
        let result = try await tool.run(["query": .string("  ")])
        XCTAssertFalse(result.success)
    }

    // MARK: - 最近 5 条观察

    func testKeepsFiveMostRecentObservations() {
        for index in 1...8 {
            graph.upsertEntity("实体\(index)", type: "concept",
                               observations: ["事实\(index)"])
        }
        let hits = (1...8).map {
            GraphObservationHit(entity: "实体\($0)", text: "事实\($0)",
                                score: 0.5, timestamp: Double($0))
        }
        let payload = MemorySearchTool.buildPayload(hits: hits, graph: graph)
        guard let object = payload.objectValue else {
            return XCTFail("返回不是 JSON 对象")
        }
        let texts = Self.objects(object["hits"]).compactMap { $0["text"]?.stringValue }
        XCTAssertEqual(texts, ["事实8", "事实7", "事实6", "事实5", "事实4"],
                       "应按写入时间倒序保留最近 5 条")
    }

    func testOnlyRecentObservationsDriveEntitiesAndRelations() {
        for index in 1...8 {
            graph.upsertEntity("实体\(index)", type: "concept",
                               observations: ["事实\(index)"])
        }
        // 旧实体的边不应进入结果，最近实体的边才应进入。
        graph.createRelation(source: "实体1", target: "实体2", label: "旧关系")
        graph.createRelation(source: "实体8", target: "实体7", label: "新关系")

        let hits = (1...8).map {
            GraphObservationHit(entity: "实体\($0)", text: "事实\($0)",
                                score: 0.5, timestamp: Double($0))
        }
        let payload = MemorySearchTool.buildPayload(hits: hits, graph: graph)
        guard let object = payload.objectValue else {
            return XCTFail("返回不是 JSON 对象")
        }
        let entityNames = Set(Self.objects(object["entities"])
            .compactMap { $0["name"]?.stringValue })
        XCTAssertFalse(entityNames.contains("实体1"), "旧观察的实体不应占核心位置")
        let labels = Self.objects(object["relations"])
            .compactMap { $0["label"]?.stringValue }
        XCTAssertTrue(labels.contains("新关系"))
        XCTAssertFalse(labels.contains("旧关系"),
                       "关系必须只来自最近 5 条观察直接关联的实体")
    }

    // MARK: - 扩展取舍

    /// 短观察 → 核心很小、余额充足 → 应发生一跳扩展。
    func testExpansionHappensWhenBudgetRemains() {
        graph.upsertEntity("A", type: "concept", observations: ["A 短事实"])
        graph.upsertEntity("B", type: "concept", observations: ["B 短事实"])
        graph.createRelation(source: "A", target: "B", label: "关联")
        let hits = [GraphObservationHit(entity: "A", text: "A 短事实",
                                        score: 0.9, timestamp: 10)]
        let payload = MemorySearchTool.buildPayload(hits: hits, graph: graph)
        guard let object = payload.objectValue else {
            return XCTFail("返回不是 JSON 对象")
        }
        let roles = Self.objects(object["entities"]).compactMap { $0["role"]?.stringValue }
        let facts = Self.objects(object["related_facts"])
        XCTAssertTrue(roles.contains("neighbor") || !facts.isEmpty,
                      "有余额时应做一跳邻居扩展")
        XCTAssertLessThanOrEqual(payload.jsonString().count,
                                 MemorySearchTool.maxOutputChars)
    }

    /// 长观察把预算吃满 → 扩展被放弃（不出现邻居实体 / 相关事实）。
    func testExpansionDroppedWhenBudgetExhausted() {
        let longText = String(repeating: "历史观察内容", count: 30)
        for index in 0..<6 {
            graph.upsertEntity("长实体\(index)", type: "concept",
                               observations: [longText])
            if index > 0 {
                graph.createRelation(source: "长实体0", target: "长实体\(index)",
                                     label: "较长的关联标签说明")
            }
        }
        let hits = (0..<5).map {
            GraphObservationHit(entity: "长实体\($0)", text: longText,
                                score: 0.9, timestamp: Double($0))
        }
        let payload = MemorySearchTool.buildPayload(hits: hits, graph: graph)
        guard let object = payload.objectValue else {
            return XCTFail("返回不是 JSON 对象")
        }
        XCTAssertLessThanOrEqual(payload.jsonString().count,
                                 MemorySearchTool.maxOutputChars)
        XCTAssertFalse(Self.objects(object["entities"]).contains {
            $0["role"]?.stringValue == "neighbor"
        }, "预算耗尽时应放弃邻居扩展")
        XCTAssertTrue(Self.objects(object["related_facts"]).isEmpty,
                      "预算耗尽时应放弃相关事实扩展")
        XCTAssertGreaterThan(object["hits"]?.arrayValue?.count ?? 0, 0,
                             "观察本身必须保留")
    }

    // MARK: - 观察时间戳持久化 / 旧库兼容

    func testObservationDatesSurviveReload() throws {
        graph.upsertEntity("A", type: "concept", observations: ["A 事实一"])
        graph.addObservations("A", ["A 事实二"])
        graph.flush()

        let reloaded = GraphMemoryStore(path: workspace
            .appendingPathComponent("long_memory.graph.json").path)
        let entity = try XCTUnwrap(reloaded.getEntity("A"))
        let dates = entity.observationDates
        XCTAssertEqual(dates.count, 2)
        XCTAssertTrue(dates.allSatisfy { $0 > 0 }, "新写入的观察应有时间戳")
        XCTAssertLessThan(dates[0], dates[1], "后追加的观察时间更晚")
    }

    func testEntityToolOutputDoesNotExposeTimestamps() throws {
        let entity = GraphEntity(name: "A", type: "concept",
                                 observations: ["事实一", "事实二"],
                                 createdAt: "2026-01-01T00:00:00",
                                 observationDates: [100, 200])
        let data = try JSONEncoder().encode(entity)
        let text = try XCTUnwrap(String(data: data, encoding: .utf8))
        XCTAssertFalse(text.contains("observation_dates"),
                       "工具返回不应带内部时间戳数组")
        XCTAssertTrue(text.contains("observations"))
    }

    func testLegacyGraphWithoutObservationDatesIsCompatible() throws {
        let legacy = """
        {"entities":{"a":{"name":"A","type":"concept","observations":["旧事实"],"created_at":"2026-01-01T00:00:00"}},"relations":[]}
        """
        let path = workspace.appendingPathComponent("legacy.graph.json").path
        try legacy.write(toFile: path, atomically: true, encoding: .utf8)
        let store = GraphMemoryStore(path: path)
        let entity = try XCTUnwrap(store.getEntity("A"))
        XCTAssertEqual(entity.observations, ["旧事实"])
        XCTAssertEqual(entity.observationDates, [0],
                       "旧库缺失时间戳应回填 0，视为最旧")
    }
}
