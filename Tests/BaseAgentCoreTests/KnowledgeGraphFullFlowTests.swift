import XCTest
import Foundation
@testable import BaseAgentCore

/// 知识图谱全流程审查回归：外部切片反查、缓存内容校验、关键词缓存、增量跳过外部源。
final class KnowledgeGraphFullFlowTests: XCTestCase {
    private final class Emb: EmbeddingProvider {
        let identifier = "kg-flow-emb"
        let dimension = 4
        func embedDocuments(_ texts: [String]) async throws -> [[Float]] { texts.map { _ in [1,0,0,0] } }
        func embedQuery(_ text: String) async throws -> [Float] { [1,0,0,0] }
    }

    private func makeStore(_ root: URL) throws -> KnowledgeGraphStore {
        try KnowledgeGraphStore(dbPath: root.appendingPathComponent("index.sqlite").path,
                                embedding: Emb())
    }

    /// P0-1: 带 zone 前缀的外部切片写入后，source_ids 反查必须能拿回原文。
    func testExternalChunkReverseLookup() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("kg-flow-ext-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try makeStore(root)
        let external = [
            KnowledgeGraphStore.ChunkListItem(id: "nature|x#0001", text: "寄生虫改变宿主行为。",
                                              source: "自然/寄生虫星球.md", heading: "第一章"),
            KnowledgeGraphStore.ChunkListItem(id: "history|y#0002", text: "病毒驱动基因转移。",
                                              source: "历史/从黎明到衰落.md", heading: "第二章"),
        ]
        try store.replaceAll(entities: [], relationships: [], externalChunks: external)
        XCTAssertEqual(store.listExternalChunkRefs().count, 2)
        let refs = try store.chunkRefs(ids: ["nature|x#0001", "missing|z"])
        XCTAssertEqual(refs.count, 1)
        XCTAssertEqual(refs.first?.text, "寄生虫改变宿主行为。")
        XCTAssertEqual(refs.first?.heading, "第一章")
    }

    /// P0-2: 外部切片参与增量比对所需的哈希清单（通过 listExternalChunkRefs 可取得）。
    func testExternalChunkHashesAvailable() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("kg-flow-hash-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try makeStore(root)
        try store.replaceAll(entities: [], relationships: [], externalChunks: [
            .init(id: "社会|a", text: "甲正文", source: "社会/a.md", heading: "章"),
        ])
        let hashes = store.listExternalChunkRefs().map {
            (id: $0.id, hash: KnowledgeGraphBuilder.contentHash($0.text))
        }
        XCTAssertEqual(hashes.count, 1)
        XCTAssertEqual(hashes.first?.id, "社会|a")
        XCTAssertFalse(hashes.first?.hash.isEmpty ?? true)
    }

    /// P1-1: 同一 chunk id 正文改写后，缓存不得命中旧抽取结果。
    func testExtractionCacheRejectsChangedContent() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("kg-flow-cache-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try makeStore(root)
        try store.cacheExtraction(chunkID: "c1", contentHash: "hash-old", output: "OLD")
        XCTAssertEqual(store.cachedExtraction(chunkID: "c1", contentHash: "hash-old"), "OLD")
        XCTAssertNil(store.cachedExtraction(chunkID: "c1", contentHash: "hash-new"),
                     "正文改写后不得命中旧缓存")
    }

    /// P1-4: 关键词缓存命中后无需再次 LLM 抽取（直接读缓存值）。
    func testKeywordCache() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("kg-flow-kw-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try makeStore(root)
        XCTAssertNil(store.cachedKeywords(for: "寄生虫"))
        store.cacheKeywords(query: "寄生虫", high: ["寄生虫"], low: ["宿主", "生态"])
        let cached = store.cachedKeywords(for: "寄生虫")
        XCTAssertEqual(cached?.high, ["寄生虫"])
        XCTAssertEqual(cached?.low, ["宿主", "生态"])
    }
}
