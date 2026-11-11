import Foundation
import XCTest
@testable import BaseAgentCore

/// KnowledgeGraph 内存图的图算法：子图展开、寻路（DFS/BFS/Dijkstra）、删除实体及关联边。
final class KnowledgeGraphAlgorithmsTests: XCTestCase {

    private var tempDir: String?

    override func setUp() {
        super.setUp()
        let dir = NSTemporaryDirectory().appending("kg-algo-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        tempDir = dir
    }

    override func tearDown() {
        if let dir = tempDir { try? FileManager.default.removeItem(atPath: dir) }
        tempDir = nil
        super.tearDown()
    }

    /// 用临时快照路径构造图（避免 upsert 防抖保存污染用户目录）。
    private func makeGraph() -> KnowledgeGraph {
        KnowledgeGraph(snapshotPath: (tempDir! as NSString).appendingPathComponent("g.snapshot.json"))
    }

    private func entity(_ name: String) -> KGEntity {
        KGEntity(id: name, name: name, type: "concept", description: "",
                 properties: [:], sourceIDs: [], createdAt: 0, updatedAt: 0)
    }
    private func rel(_ id: String, _ src: String, _ dst: String,
                     weight: Double = 1) -> KGRelationship {
        KGRelationship(id: id, srcEntity: src, dstEntity: dst, relationType: "rel",
                       description: "", weight: weight, properties: [:],
                       sourceIDs: [], createdAt: 0, updatedAt: 0)
    }

    /// 线图 A-B-C-D（无捷径）。
    private func makeChainGraph() -> KnowledgeGraph {
        let g = makeGraph()
        for n in ["A", "B", "C", "D"] { g.upsertEntity(entity(n)) }
        g.upsertRelationship(rel("r1", "A", "B"))
        g.upsertRelationship(rel("r2", "B", "C"))
        g.upsertRelationship(rel("r3", "C", "D"))
        return g
    }

    func testBFSReturnsShortestHopPath() {
        let g = makeChainGraph()
        let p = g.shortestPath(from: "A", to: "D", algorithm: .bfs)
        XCTAssertEqual(p.nodes, ["A", "B", "C", "D"], "BFS 应返回跳数最短路径")
    }

    func testDijkstraPrefersHighWeightEdge() {
        // 独立图：A-B-C-D 线性 + A-D 高权重捷径（权重 5）。
        let g = makeGraph()
        for n in ["A", "B", "C", "D"] { g.upsertEntity(entity(n)) }
        g.upsertRelationship(rel("r1", "A", "B"))
        g.upsertRelationship(rel("r2", "B", "C"))
        g.upsertRelationship(rel("r3", "C", "D"))
        g.upsertRelationship(rel("r4", "A", "D", weight: 5))
        let p = g.shortestPath(from: "A", to: "D", algorithm: .dijkstra)
        // A-D 捷径权重 5 → cost 最低，应直接走 A→D
        XCTAssertEqual(p.nodes, ["A", "D"], "Dijkstra 应按权重优先走高权重捷径")
        XCTAssertEqual(p.relationships.map { $0.id }, ["r4"])
    }

    func testDFSReachesDestination() {
        let g = makeChainGraph()
        let p = g.shortestPath(from: "A", to: "D", algorithm: .dfs)
        XCTAssertEqual(p.nodes.first, "A")
        XCTAssertEqual(p.nodes.last, "D")
        XCTAssertFalse(p.nodes.isEmpty)
    }

    func testPathIsEmptyWhenDisconnected() {
        let g = makeGraph()
        g.upsertEntity(entity("X"))
        g.upsertEntity(entity("Y"))
        let p = g.shortestPath(from: "X", to: "Y", algorithm: .bfs)
        XCTAssertTrue(p.nodes.isEmpty, "不连通时应返回空路径")
    }

    func testSubgraphBFSExpandsToDepth() {
        let g = makeChainGraph()
        let sub = g.subgraph(root: "B", depth: 1, edgeTypes: nil)
        // B 的 1 跳：A、B、C → 3 实体；关系 A-B、B-C → 2 条
        XCTAssertEqual(Set(sub.entities.map { $0.name }), ["A", "B", "C"])
        XCTAssertEqual(sub.relationships.count, 2)
    }

    func testSubgraphRespectsEdgeTypeFilter() {
        let g = makeChainGraph()
        let sub = g.subgraph(root: "B", depth: 1, edgeTypes: ["nonexistent"])
        XCTAssertEqual(sub.relationships.count, 0, "edgeTypes 白名单应过滤掉全部边")
        // 实体仍按 BFS 展开到邻居（边过滤只影响关系集合，不影响节点集合）
        XCTAssertEqual(Set(sub.entities.map { $0.name }), ["A", "B", "C"])
    }

    func testDeleteEntityRemovesNodeAndIncidentEdges() {
        let g = makeChainGraph()
        g.deleteEntity("B")
        XCTAssertNil(g.entity(named: "B"), "被删实体不应存在")
        // 线性图里 A 只连 B：删除 B 后 A 不再有邻居
        XCTAssertTrue(g.neighbors(of: "A").isEmpty, "A 的 B 邻居边应被删除")
        XCTAssertTrue(g.incidentRelationships(of: "A").isEmpty)
        // C 仍连 D
        XCTAssertTrue(g.neighbors(of: "C").contains("D"))
        // 其余实体仍在
        XCTAssertNotNil(g.entity(named: "A"))
        XCTAssertNotNil(g.entity(named: "D"))
    }
}
