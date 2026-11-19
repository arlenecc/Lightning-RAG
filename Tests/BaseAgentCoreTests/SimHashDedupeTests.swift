import XCTest
@testable import BaseAgentCore

final class SimHashDedupeTests: XCTestCase {
    /// MurmurHash3 x64_128 低 64 位权威向量（与 tools/SimHashDedupe.swift 自检一致）。
    func testMurmurHashVectors() {
        XCTAssertEqual(SimHashDedupe.hash64(""), 0)
        XCTAssertEqual(SimHashDedupe.hash64("foo"), UInt64("e271865701f54561", radix: 16)!)
        XCTAssertEqual(SimHashDedupe.hash64("hello"), UInt64("cbd8a7b341bd9b02", radix: 16)!)
        XCTAssertEqual(SimHashDedupe.hash64("量子力学"), UInt64("dc380f4d0ae2fc34", radix: 16)!)
        XCTAssertEqual(SimHashDedupe.hash64("abcdefghijklmnopq"), UInt64("7564747f88bda657", radix: 16)!)
        // 同 seed 稳定、不同 seed 不同
        XCTAssertEqual(SimHashDedupe.hash64("世界", seed: 0),
                       SimHashDedupe.hash64("世界", seed: 0))
        XCTAssertNotEqual(SimHashDedupe.hash64("世界", seed: 0),
                          SimHashDedupe.hash64("世界", seed: 1))
    }

    func testFingerprintStableAndSensitive() {
        let text = "当事人一方不履行合同义务，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。"
        let fp = SimHashDedupe.fingerprint(text)
        XCTAssertFalse(fp.isEmpty)
        XCTAssertEqual(fp, SimHashDedupe.fingerprint(text), "同文本指纹必须稳定")
        // 轻微改写：改动少量字符，汉明距离应较小（LSH 能召回）
        let tweak = "当事人一方不履行合同义务，应当承担继续履行、采取补救措施或者赔偿损失等违约责任呢。"
        let d = SimHashDedupe.hamming(fp, SimHashDedupe.fingerprint(tweak))
        XCTAssertNotNil(d)
        XCTAssertLessThanOrEqual(d!, 6, "轻微改写汉明距离应较小，实际 \(d!)")
        // 完全不同领域文本：汉明距离应较大
        let other = "今天的天气很好，阳光明媚，微风和煦，非常适合外出散步和旅游，公园里有很多人在享受时光。"
        let d2 = SimHashDedupe.hamming(fp, SimHashDedupe.fingerprint(other))!
        XCTAssertGreaterThan(d2, d!, "不同文本距离应大于轻微改写")
    }

    func testLSHIndexRecallsNearDuplicate() {
        var index = SimHashDedupe.Index()
        let intro = "合同违约责任条款与赔偿责任的详细规定。当事人一方不履行合同义务的，应当承担继续履行或者赔偿损失等违约责任。"
        let base = String(repeating: intro, count: 4) + "约定的违约金低于损失的可请求调整。"
        index.insert(key: "base.md", fingerprintHex: SimHashDedupe.fingerprint(base))
        // 近似副本：正文几乎一致，仅结尾一句轻微改写（真实"近似副本/轻微编辑"场景）。
        let near = String(repeating: intro, count: 4) + "约定的违约金低于损失时可以请求予以适当调整。"
        let fpNear = SimHashDedupe.fingerprint(near)
        let d = SimHashDedupe.hamming(SimHashDedupe.fingerprint(base), fpNear)!
        XCTAssertLessThanOrEqual(d, 3, "轻微结尾改写汉明距离应 ≤3，实际 \(d)")
        let hit = index.nearest(fpNear, threshold: 3)
        XCTAssertNotNil(hit, "近似副本应被 LSH 召回")
        XCTAssertEqual(hit?.key, "base.md")
        let unrelated = String(repeating: "股票市场今日收盘上涨，成交量放大，投资者情绪乐观。", count: 4)
        XCTAssertNil(index.nearest(SimHashDedupe.fingerprint(unrelated), threshold: 3),
                     "无关文本不应误判")
    }

    /// 端到端：解析→markdown 后 SimHash 近似排重，近似副本不再切片入库。
    func testIngestSkipsNearDuplicateBeforeChunking() async throws {
        final class FakeEmbedding: EmbeddingProvider {
            let identifier = "sh-fake"; let dimension = 8
            func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
                texts.map { t in
                    var v = [Float](repeating: 0.01, count: 8)
                    for s in t.unicodeScalars { v[Int(s.value) % 8] += 1 }
                    return VectorCodec.normalize(v)
                }
            }
            func embedQuery(_ t: String) async throws -> [Float] { try await embedDocuments([t])[0] }
        }
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("sh-ingest-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        let intro = "合同违约责任条款与赔偿责任的详细规定。当事人一方不履行合同义务的，应当承担继续履行或者赔偿损失等违约责任。"
        let original = String(repeating: intro, count: 4) + "约定的违约金低于损失的可请求调整。"
        let nearDup = String(repeating: intro, count: 4) + "约定的违约金低于损失时可以请求予以适当调整。"
        let unrelated = String(repeating: "股票市场今日收盘上涨，成交量放大，投资者情绪乐观，后市展望积极。", count: 4)
        // 第一次同步：原文 a + 无关文档 c（此时 b 还不存在）。
        try original.write(to: kb.appendingPathComponent("a.md"), atomically: true, encoding: .utf8)
        try unrelated.write(to: kb.appendingPathComponent("c.md"), atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragSemanticChunking = false
        config.ragChunkSize = 200
        config.ragChunkOverlap = 0
        let engine = RAGEngine(config: config, embedding: FakeEmbedding(), reranker: nil)
        let first = await engine.ingest()
        XCTAssertEqual(first.processed, 2, "首次应入库原文+无关文档")
        XCTAssertEqual(first.errors.count, 0, first.errors.joined(separator: "; "))
        let chunksAfterFirst = engine.status()["chunks_stored"]?.intValue ?? -1

        // 第二次同步：加入近似副本 b —— 已入库过的近似文件应被跳过切片入库。
        try nearDup.write(to: kb.appendingPathComponent("b.md"), atomically: true, encoding: .utf8)
        let second = await engine.ingest()
        XCTAssertEqual(second.errors.count, 0, second.errors.joined(separator: "; "))
        XCTAssertGreaterThanOrEqual(second.deduplicated, 1, "近似副本应被标记为去重跳过")
        XCTAssertEqual(second.processed, 0, "近似副本不应切片入库")
        XCTAssertEqual(engine.status()["chunks_stored"]?.intValue ?? -1,
                       chunksAfterFirst, "切片总数不应增长")
        engine.close()
    }

    func testJaccardVerify() {
        let a = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥"
        XCTAssertEqual(SimHashDedupe.jaccard(a, a), 1.0, accuracy: 1e-9)
        XCTAssertEqual(SimHashDedupe.jaccard(a, " completely differentcontent "), 0.0, accuracy: 1e-9)
    }
}
