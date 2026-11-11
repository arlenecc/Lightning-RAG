import Foundation
import XCTest
@testable import BaseAgentCore

/// 实体消歧与合并：事实抽取入库时，同名 / 语义同指的实体要合并到已有记录，
/// 不重复入库；新内容以追加观察、更新关系的方式并入。
///
/// 注意：`resolveEntity` / `mergeFactDraft` 的**语义**路径需显式 `semantic: true`，
/// 因为插入阶段（事实抽取）默认 `semantic: false` 只做字符全匹配
/// —— 避免每次入库都重算全库实体名向量（大图下开销不可接受）。
final class EntityResolutionTests: XCTestCase {

    private var dirs: [URL] = []

    override func tearDown() {
        for dir in dirs { try? FileManager.default.removeItem(at: dir) }
        dirs.removeAll()
        super.tearDown()
    }

    /// 可控假 embedding：名字里的语义由「词元」决定，便于精确验证余弦阈值与 type 冲突。
    private final class SimulatedEmbedding: EmbeddingProvider, @unchecked Sendable {
        let identifier = "entity-resolution-embedding"
        let dimension = 3
        // 「特朗普」与「川普」都归到第一维；「苹果公司」归第二维；「苹果水果」归第三维。
        func embedQuery(_ text: String) async throws -> [Float] { Self.vector(text) }
        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map(Self.vector)
        }
        private static func vector(_ text: String) -> [Float] {
            let lower = text.lowercased()
            if lower.contains("特朗普") || lower.contains("川普") { return [1, 0, 0] }
            if lower.contains("苹果") && lower.contains("公司") { return [0, 1, 0] }
            if lower.contains("苹果") { return [0, 0, 1] }
            return [0, 0, 0]
        }
    }

    private func makeStore(_ embedding: EmbeddingProvider?) -> GraphMemoryStore {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("entity-resolution-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        dirs.append(dir)
        return GraphMemoryStore(
            path: dir.appendingPathComponent("long_memory.graph.json").path,
            embedding: embedding)
    }

    /// 可计数的假 embedding：验证实体名向量只计算一次并缓存复用
    /// （大图下避免每消歧一个实体就重复 embed 全库名字）。
    private final class CountingEmbedding: EmbeddingProvider, @unchecked Sendable {
        let identifier = "counting-embedding"
        let dimension = 3
        private let counter = Counter()
        final class Counter: @unchecked Sendable {
            private let lock = NSLock()
            private(set) var documents = 0
            func add(_ n: Int) { lock.lock(); documents += n; lock.unlock() }
        }
        var documentCount: Int { counter.documents }
        func embedQuery(_ text: String) async throws -> [Float] { [1, 0, 0] }
        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            counter.add(texts.count)
            return texts.map { _ in [1, 0, 0] }
        }
    }

    // MARK: - 精确匹配（不依赖 embedding）

    func testExactNameCaseInsensitiveMerges() async {
        let store = makeStore(nil)  // 无 embedding，验证精确路径
        store.upsertEntity("特朗普", type: "person", observations: ["特朗普是美国前总统"])

        let canonical = await store.resolveEntity(name: "特朗普")
        XCTAssertEqual(canonical, "特朗普")
        let canonicalUpper = await store.resolveEntity(name: "  特朗普  ")
        XCTAssertEqual(canonicalUpper, "特朗普", "首尾空白应被归一化")

        XCTAssertEqual(store.listEntities().count, 1, "不应新建重复实体")
    }

    // MARK: - 语义匹配（需显式 semantic: true）

    func testSemanticMatchMergesIntoExistingEntity() async {
        let store = makeStore(SimulatedEmbedding())
        store.upsertEntity("特朗普", type: "person", observations: ["特朗普是美国前总统"])

        // 库内只有「特朗普」，消歧「川普」应合并到它。
        let canonical = await store.resolveEntity(name: "川普", type: "person", semantic: true)
        XCTAssertEqual(canonical, "特朗普", "语义同指实体应合并到已有记录")

        // 调用方随后 upsert(canonical)：验证合并后观察追加、不新建实体。
        store.upsertEntity(canonical, type: "person", observations: ["川普曾就任第45任总统"])
        XCTAssertEqual(store.listEntities().count, 1)
        let entity = store.listEntities().first!
        XCTAssertEqual(entity.observations.count, 2, "合并后应保留两条观察")
    }

    func testSemanticMismatchKeepsSeparateEntities() async {
        let store = makeStore(SimulatedEmbedding())
        store.upsertEntity("苹果公司", type: "organization", observations: ["苹果公司是科技公司"])

        // 「苹果水果」语义向量不同 → 不合并，返回原名（新实体）。
        let canonical = await store.resolveEntity(name: "苹果水果", type: "other", semantic: true)
        XCTAssertEqual(canonical, "苹果水果")
        store.upsertEntity(canonical, type: "other", observations: ["苹果水果是水果"])
        XCTAssertEqual(store.listEntities().count, 2, "语义不同的实体不应被合并")
    }

    // MARK: - type 冲突保护

    func testTypeConflictRejectsSemanticMerge() async {
        let store = makeStore(SimulatedEmbedding())
        store.upsertEntity("特朗普", type: "person",
                           observations: ["特朗普是美国前总统"])

        // 「川普」与「特朗普」在假 embedding 里向量完全相同（余弦 1.0 ≥ 阈值），
        // 但 type=project 与已有 person 冲突 → 必须拒绝语义合并，避免误合并。
        let rejected = await store.resolveEntity(name: "川普", type: "project", semantic: true)
        XCTAssertEqual(rejected, "川普", "type 冲突时不得语义合并")

        // 对照：type 一致（person）时正常合并，证明拒绝确实来自 type 而非向量。
        let accepted = await store.resolveEntity(name: "川普", type: "person", semantic: true)
        XCTAssertEqual(accepted, "特朗普")

        XCTAssertEqual(store.listEntities().count, 1,
                       "resolveEntity 本身不写库，实体数不变")
    }

    // MARK: - embedding 不可用回退

    func testEmbeddingUnavailableFallsBackToExact() async {
        let store = makeStore(nil)
        store.upsertEntity("特朗普", type: "person", observations: ["特朗普是美国前总统"])

        // 无 embedding：即便请求语义消歧，也应回退为不合并（宁漏勿错），不崩溃。
        let canonical = await store.resolveEntity(name: "川普", type: "person", semantic: true)
        XCTAssertEqual(canonical, "川普", "无 embedding 时应回退为不合并（宁漏勿错）")
    }

    // MARK: - 阈值保守

    func testThresholdIsConfigurable() async {
        let store = makeStore(SimulatedEmbedding())
        store.upsertEntity("特朗普", type: "person", observations: ["特朗普是美国前总统"])
        // SimulatedEmbedding 中「川普」与「特朗普」余弦 = 1.0，默认阈值 0.90 会合并。
        let merged = await store.resolveEntity(name: "川普", type: "person", semantic: true)
        XCTAssertEqual(merged, "特朗普")

        // 阈值提到 1.01：余弦 1.0 < 1.01，不再合并。
        store.entityResolutionThreshold = 1.01
        let kept = await store.resolveEntity(name: "川普", type: "person", semantic: true)
        XCTAssertEqual(kept, "川普", "阈值上调后应拒绝合并")
    }

    // MARK: - 关系端点消歧

    func testRelationEndpointsResolveToCanonicalNames() async {
        let store = makeStore(SimulatedEmbedding())
        store.upsertEntity("特朗普", type: "person", observations: ["特朗普是美国前总统"])

        // 新关系端点「川普」应消歧到已有「特朗普」，而非现场补建空实体。
        let source = await store.resolveEntity(name: "川普", semantic: true)
        let target = await store.resolveEntity(name: "总统职位", semantic: true)
        store.upsertEntity(source, type: "person", observations: ["川普曾就任总统"])
        store.upsertEntity(target, type: "concept", observations: ["总统职位是最高行政职位"])
        store.createRelation(source: source, target: target, label: "担任")

        // 验证关系端点指向合并后的规范名，且没有「川普」这个重复实体。
        XCTAssertTrue(store.listRelations().contains {
            $0.source == "特朗普" && $0.target == "总统职位"
        })
        XCTAssertNil(store.listEntities().first { $0.name == "川普" },
                     "关系端点消歧后不应残留重复实体")
    }

    // MARK: - 向量缓存

    func testNameVectorsAreCachedAcrossResolutions() async {
        let embedding = CountingEmbedding()
        let store = makeStore(embedding)
        store.upsertEntity("实体一", type: "other", observations: ["实体一承担职责"])
        store.upsertEntity("实体二", type: "other", observations: ["实体二承担职责"])
        store.upsertEntity("实体三", type: "other", observations: ["实体三承担职责"])

        // 第 1 次：候选名自身 1 次 + 全库 3 个名字 1 次 = 4。
        _ = await store.resolveEntity(name: "候选甲", semantic: true)
        let afterFirst = embedding.documentCount
        XCTAssertEqual(afterFirst, 4, "首次需 embed 候选名与全库实体名")

        // 后续两次：全库名字向量已缓存，每次只 embed 候选名自身。
        _ = await store.resolveEntity(name: "候选乙", semantic: true)
        _ = await store.resolveEntity(name: "候选丙", semantic: true)
        XCTAssertEqual(embedding.documentCount, afterFirst + 2,
                       "不得重复 embed 已有实体名")
    }

    // MARK: - 端到端：一批三元组入库

    /// 用户核心诉求：两批抽取里「同指但不同名」的实体不重复入库，
    /// 只追加观察与关系；实体总数不增长（需显式开启语义消歧）。
    func testMergeFactDraftMergesAcrossBatches() async {
        let store = makeStore(SimulatedEmbedding())

        let batch1 = FactExtractionDraft(
            entities: [FactEntityDraft(name: "特朗普", type: "person",
                                       observations: ["特朗普是美国前总统"])],
            relations: [])
        _ = await store.mergeFactDraft(batch1)
        XCTAssertEqual(store.listEntities().count, 1)

        let batch2 = FactExtractionDraft(
            entities: [
                FactEntityDraft(name: "川普", type: "person",
                                observations: ["川普曾就任第45任总统"]),
                FactEntityDraft(name: "总统职位", type: "concept",
                                observations: ["总统职位是最高行政职位"]),
            ],
            relations: [FactRelationDraft(source: "川普", target: "总统职位", label: "担任")])
        let merged = await store.mergeFactDraft(batch2, semantic: true)

        // 「川普」合并进已有「特朗普」；只新增「总统职位」这一个真正的新实体。
        XCTAssertEqual(store.listEntities().count, 2,
                       "同指实体不得重复入库")
        XCTAssertNil(store.listEntities().first { $0.name == "川普" })
        let trump = store.listEntities().first { $0.name == "特朗普" }!
        XCTAssertEqual(trump.observations.count, 2, "合并后应追加观察")
        XCTAssertEqual(trump.observationDates.count, 2,
                       "观察时间戳须与观察数组保持对齐")
        XCTAssertTrue(store.listRelations().contains {
            $0.source == "特朗普" && $0.target == "总统职位" && $0.label == "担任"
        }, "关系端点必须指向消歧后的规范名")
        XCTAssertEqual(merged.relations, 1)
    }

    /// 插入阶段默认 `semantic: false` 只做字符全匹配：字符不同的同指实体
    /// 不会被合并，但也不会误合并（宁漏勿错，且避免每次入库重算向量）。
    func testMergeFactDraftDefaultIsExactMatchOnly() async {
        let store = makeStore(SimulatedEmbedding())
        _ = await store.mergeFactDraft(FactExtractionDraft(
            entities: [FactEntityDraft(name: "特朗普", type: "person",
                                       observations: ["特朗普是美国前总统"])],
            relations: []))
        // 第二批「川普」字符不同于「特朗普」 → 默认不合并。
        _ = await store.mergeFactDraft(FactExtractionDraft(
            entities: [FactEntityDraft(name: "川普", type: "person",
                                       observations: ["川普曾就任第45任总统"])],
            relations: []))
        XCTAssertEqual(store.listEntities().count, 2,
                       "插入阶段默认只做字符全匹配，不合并语义同指实体")
        XCTAssertNotNil(store.listEntities().first { $0.name == "川普" })
    }

    /// embedding 不可用时（或默认关闭），关系端点仍必须指向本批实体，
    /// 不得触发 `createRelation` 补建空实体（宁漏勿错，也不能制造噪声）。
    func testMergeFactDraftWithoutEmbeddingKeepsRelationsConsistent() async {
        let store = makeStore(nil)
        let draft = FactExtractionDraft(
            entities: [
                FactEntityDraft(name: "特朗普", type: "person",
                                observations: ["特朗普是美国前总统"]),
                FactEntityDraft(name: "总统职位", type: "concept",
                                observations: ["总统职位是最高行政职位"]),
            ],
            relations: [FactRelationDraft(source: "特朗普", target: "总统职位", label: "担任")])
        _ = await store.mergeFactDraft(draft)

        XCTAssertEqual(store.listEntities().count, 2, "不得补建空实体")
        XCTAssertTrue(store.listRelations().contains {
            $0.source == "特朗普" && $0.target == "总统职位"
        })
    }

    /// 两端被消歧到同一实体时，关系退化为自环，应被跳过。
    func testMergeFactDraftSkipsSelfLoopAfterResolution() async {
        let store = makeStore(SimulatedEmbedding())
        // 「特朗普」与「川普」都会被消歧到同一规范名（第二个合并到第一个）。
        let draft = FactExtractionDraft(
            entities: [
                FactEntityDraft(name: "特朗普", type: "person",
                                observations: ["特朗普是美国前总统"]),
                FactEntityDraft(name: "川普", type: "person",
                                observations: ["川普曾就任第45任总统"]),
            ],
            relations: [FactRelationDraft(source: "特朗普", target: "川普", label: "别名")])
        let merged = await store.mergeFactDraft(draft, semantic: true)

        XCTAssertEqual(store.listEntities().count, 1)
        XCTAssertEqual(merged.relations, 0, "自环关系应跳过")
        XCTAssertTrue(store.listRelations().isEmpty)
    }
}
