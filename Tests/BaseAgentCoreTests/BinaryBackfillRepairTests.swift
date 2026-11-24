import XCTest
import Foundation
@testable import BaseAgentCore

/// 一次性修复工具（环境变量驱动，默认跳过）：
///
/// 针对历史构建窗口写入的陈旧二值通道（bin 表 bit[128] + 96 字节签名无法写入、
/// meta 计数虚高），按当前 provider 位宽重建 bin 表并用真实模型补齐全部签名。
/// 用法：
/// ```
/// RAG_REPAIR_DB=/path/to/index.sqlite swift test --filter BinaryBackfillRepairTests
/// ```
/// 前提：占用该库的应用已退出。
final class BinaryBackfillRepairTests: XCTestCase {
    private static let modelName = "jina-embeddings-v5-text-nano-mlx-int4"
    /// 与库里存的 binary_source 完全一致（含 `models/` 相对路径前缀），
    /// 避免修复后应用再打开时因来源不一致又触发一次重建。
    private static let qualifiedModelName = "models/\(modelName)"

    func testRebuildBinaryTableAndBackfillSignatures() async throws {
        guard let dbPath = ProcessInfo.processInfo.environment["RAG_REPAIR_DB"],
              !dbPath.trimmingCharacters(in: .whitespaces).isEmpty else {
            throw XCTSkip("未设置 RAG_REPAIR_DB，跳过一次性修复")
        }
        EmbeddingBenchmarkTests.configureMetallib()
        let directory = EmbeddingBenchmarkTests.resolveModelsRoot()
            .appendingPathComponent(Self.modelName)
        try XCTSkipUnless(
            JinaV5Profile.isJinaV5(directory: directory)
                && JinaV5Profile.weightsURL(in: directory) != nil,
            "模型缺失：\(directory.path)")

        let provider = JinaV5MLXEmbeddingProvider(
            directory: directory, modelName: Self.qualifiedModelName)
        addTeardownBlock { await provider.unload() }
        XCTAssertEqual(provider.identifier,
                       "local:mlx-embedding:\(Self.qualifiedModelName)")
        XCTAssertEqual(provider.dimension, 128, "库存主向量为 128 维截断输出")
        XCTAssertEqual(provider.binaryDimension, 768, "签名位宽应为原生 768")

        // 指纹传空：跳过指纹校验（维度/元素类型与库一致时主向量通道不受影响），
        // 仅由 setupBinaryIndexLocked 把 bit[128] bin 表重建为 bit[768]。
        let store = try VectorStore(dbPath: dbPath, dimension: 128,
                                    enableSQLiteVec: true, elementType: .int8,
                                    binaryDimension: 768,
                                    binarySource: provider.identifier,
                                    indexFingerprint: "")
        addTeardownBlock { store.close() }
        XCTAssertNil(store.rebuildReason, "主向量通道必须保持可用")
        XCTAssertTrue(store.vectorEnabled)
        XCTAssertTrue(store.binaryStorageEnabled, "打开时应已重建 bit[768] bin 表")
        XCTAssertEqual(store.binaryRows, 0, "重建后 bin 行数必须归零")

        let enqueued = store.enqueueMissingVectors()
        print("[repair] 重入队缺失二值签名的切片 \(enqueued) 条"
              + "（已在 pending 表中的不重复计入）")

        let started = Date()
        var total = 0
        while true {
            let pending = store.nextPendingTexts(limit: 32)
            guard !pending.isEmpty else { break }
            let output = try await provider.embedDocumentsWithSignatures(pending.map(\.text))
            guard output.vectors.count == pending.count,
                  output.signatures.count == pending.count,
                  output.signatures.allSatisfy({ $0.count == 768 / 8 }) else {
                XCTFail("embedding 输出数量/签名位宽不符：vectors=\(output.vectors.count)，"
                        + "pending=\(pending.count)，signatures=\(output.signatures.count)")
                return
            }
            for (i, item) in pending.enumerated() {
                try store.fillPendingVector(rowid: item.rowid,
                                            vector: output.vectors[i],
                                            binarySignature: output.signatures[i])
                total += 1
            }
            if total % 256 == 0 {
                print("[repair] 已补齐 \(total)/\(enqueued)，"
                      + String(format: "耗时 %.1fs", Date().timeIntervalSince(started)))
            }
        }

        XCTAssertEqual(store.pendingVectorCount(), 0, "所有切片都应完成补齐")
        XCTAssertGreaterThan(store.binaryRows, 0, "bin 行数必须与切片总数一致")
        print("[repair] 校验通过：pending=0，binary_rows=\(store.binaryRows)，"
              + "本次补齐 \(total) 条")
        store.close()
        print("[repair] 完成：共补齐 \(total) 条二值签名，"
              + String(format: "总耗时 %.1fs", Date().timeIntervalSince(started)))
    }
}
