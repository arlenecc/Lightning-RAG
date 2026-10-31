# 二值 ANN 粗筛与 Stage 2 批读 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为二值粗筛增加自研分块 IVF ANN（`bann`），并用 `chunk_vectors_rerank` 镜像表把 stage 2 的 vec0 随机点查改为 rowid 升序、每 256 条一批的顺序读取。

**Architecture:** rerank 表与主向量同事务写入、`exactVectorsLocked` 优先读镜像并逐行回退主表；bann 复用现有 ANN 的 bits 块布局、k-means、打包、簇心缓存与 TopK 收集器，独立使用 `bann_*` 表。检索回退链为 bann → `chunk_vectors_bin` vec0 KNN → 现有 ANN/精确路径；构建入口按二值签名覆盖情况在建 bann 或主 ANN 之间自动选择。

**Tech Stack:** Swift 6、SwiftPM、XCTest、SQLite、sqlite-vec 0.1.9、Accelerate/vDSP。

---

## 文件结构

| 文件 | 职责 |
| --- | --- |
| `Sources/BaseAgentCore/RAG/Storage/VectorStoreRerank.swift`（新建） | rerank 表 schema/meta/写入/删除/清空/状态 |
| `Sources/BaseAgentCore/RAG/Storage/VectorStoreBinaryANN.swift`（新建） | bann schema/meta/构建/查询/签名流式读取/生命周期 |
| `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift`（修改） | 新属性、init 接线、写入/删除/清空/重建、`exactVectorsLocked` 批读 |
| `Sources/BaseAgentCore/RAG/Storage/VectorStoreBinary.swift`（修改） | stage 1 优先 bann、回退 vec0、stage 2 保留 binaryScore |
| `Sources/BaseAgentCore/RAG/RAGEngine.swift`（修改） | 构建目标选择、bann 状态、手动/自动构建、删除、status |
| `Sources/BaseAgentCore/Tools/RAGTool.swift`（修改） | `rag_status` 增加二值 ANN / backend / stage2 行 |
| `Sources/BaseAgentApp/AppModel.swift`（修改） | bann 状态文本、按目标确认构建 |
| `Sources/BaseAgentApp/SettingsView.swift`（修改） | 显示二值 ANN 状态与构建文案 |
| `Tests/BaseAgentCoreTests/VectorStoreRerankTests.swift`（新建） | rerank 存储与 stage 2 读取测试 |
| `Tests/BaseAgentCoreTests/VectorStoreBinaryANNTests.swift`（新建） | bann 构建/查询/回退/生命周期测试 |
| `Tests/BaseAgentCoreTests/RAGEngineBinaryANNTests.swift`（新建） | 引擎目标选择/状态/检索集成 |
| `Tests/BaseAgentCoreTests/RAGToolBinaryStatusTests.swift`（修改） | 工具输出断言 |

---

### Task 1: rerank 镜像表与生命周期

**Files:**
- Create: `Sources/BaseAgentCore/RAG/Storage/VectorStoreRerank.swift`
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift`
- Test: `Tests/BaseAgentCoreTests/VectorStoreRerankTests.swift`

- [ ] **Step 1: Write the failing tests**

```swift
import XCTest
@testable import BaseAgentCore

final class VectorStoreRerankTests: XCTestCase {
    private var directory: URL!
    private let dimension = 16

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rerank-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func makeStore(_ name: String, type: VectorElementType = .int8) throws -> VectorStore {
        try VectorStore(dbPath: directory.appendingPathComponent("\(name).sqlite").path,
                        dimension: dimension, elementType: type,
                        binaryDimension: dimension, binarySource: "sig",
                        indexFingerprint: "fp-\(name)")
    }

    private func vector(_ index: Int) -> [Float] {
        var value = [Float](repeating: 0, count: dimension)
        value[index % dimension] = 1
        return value
    }

    private func seed(_ store: VectorStore, count: Int) throws -> [Int64] {
        var chunks: [Chunk] = []
        var vectors: [[Float]] = []
        var signatures: [Data] = []
        for index in 0..<count {
            chunks.append(Chunk(text: "chunk \(index)", source: "s\(index).md", chunkIndex: 0))
            vectors.append(vector(index))
            signatures.append(Data(repeating: UInt8(index % 255), count: dimension / 8))
        }
        _ = try store.add(chunks: chunks, vectors: vectors, binarySignatures: signatures)
        return (try store.rowIDs(source: "s0.md")) + (1...max(0, count - 1)).map { Int64($0) }
    }

    func testAddMirrorsEncodedBytesIntoRerankTable() throws {
        let store = try makeStore("mirror")
        _ = try store.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0),
                     Chunk(text: "beta", source: "b.md", chunkIndex: 0)],
            vectors: [vector(0), vector(1)],
            binarySignatures: [Data([0xFF, 0x00]), Data([0x00, 0xFF])])
        XCTAssertTrue(store.rerankStorageEnabled)
        XCTAssertEqual(store.rerankRows, 2)
        let count = try store.db.scalar("SELECT COUNT(*) FROM chunk_vectors_rerank").intValue
        XCTAssertEqual(count, 2)
        store.close()
    }

    func testDeleteAndClearResetRerankCounter() throws {
        let store = try makeStore("delete")
        _ = try store.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0)],
            vectors: [vector(0)],
            binarySignatures: [Data([0xFF, 0x00])])
        let rowID = try XCTUnwrap(try store.rowIDs(source: "a.md").first)
        _ = try store.deleteRowIDs([rowID])
        XCTAssertEqual(store.rerankRows, 0)
        try store.clear()
        XCTAssertEqual(try store.db.scalar("SELECT COUNT(*) FROM chunk_vectors_rerank").intValue, 0)
        store.close()
    }

    func testReopenKeepsRerankAndRebuildDropsIt() throws {
        let path = directory.appendingPathComponent("reopen.sqlite").path
        var store: VectorStore? = try VectorStore(
            dbPath: path, dimension: dimension, elementType: .int8,
            binaryDimension: dimension, binarySource: "sig", indexFingerprint: "fp")
        _ = try store!.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0)],
            vectors: [vector(0)], binarySignatures: [Data([0xFF, 0x00])])
        XCTAssertEqual(store!.rerankRows, 1)
        store!.close()

        store = try VectorStore(
            dbPath: path, dimension: dimension, elementType: .int8,
            binaryDimension: dimension, binarySource: "sig", indexFingerprint: "fp")
        XCTAssertEqual(store!.rerankRows, 1)
        _ = try store!.rebuildForCurrentConfiguration()
        XCTAssertEqual(store!.rerankRows, 0)
        store!.close()
    }

    func testBinMainTypeDisablesRerankMirror() throws {
        let store = try makeStore("bin-main", type: .bin)
        XCTAssertFalse(store.rerankStorageEnabled)
        XCTAssertEqual(store.rerankRows, 0)
        store.close()
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter VectorStoreRerankTests`
Expected: compile failure, `VectorStore` has no `rerankStorageEnabled` / `rerankRows` / rerank table.

- [ ] **Step 3: Add rerank storage file**

```swift
import Foundation

extension VectorStore {
    static let rerankRowsKey = "rerank_rows"
    static let rerankDimensionKey = "rerank_dimension"
    static let rerankElementTypeKey = "rerank_element_type"

    /// 按当前主向量配置对齐镜像表。派生数据不一致时直接丢弃重建。
    func setupRerankStorageLocked() throws {
        rerankStorageEnabled = false
        rerankRows = 0
        rerankStorageReason = nil
        guard elementType != .bin else {
            rerankStorageReason = "bin 主存储不需要 rerank 镜像"
            try? db.transaction {
                try db.execute("DROP TABLE IF EXISTS \(Self.rerankVectorTable)")
                try deleteRerankMetaLocked()
            }
            return
        }

        let existingSQL = try db.scalar(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            [.text(Self.rerankVectorTable)]).stringValue
        let storedDimension = try db.scalar(
            "SELECT value FROM meta WHERE key = ?", [.text(Self.rerankDimensionKey)]).stringValue
        let storedType = try db.scalar(
            "SELECT value FROM meta WHERE key = ?", [.text(Self.rerankElementTypeKey)]).stringValue
        let storedRows = Self.readStoredCount(db, key: Self.rerankRowsKey)
        let schemaOK = existingSQL?.contains("embedding BLOB") == true
        if schemaOK, storedDimension == String(dimension),
           storedType == elementType.rawValue, let storedRows {
            rerankRows = max(0, storedRows)
            rerankStorageEnabled = true
            return
        }

        try db.transaction {
            try db.execute("DROP TABLE IF EXISTS \(Self.rerankVectorTable)")
            try deleteRerankMetaLocked()
            try db.execute("""
            CREATE TABLE \(Self.rerankVectorTable) (
                rowid INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL
            )
            """)
            rerankRows = 0
            try writeRerankMetaLocked()
        }
        rerankStorageEnabled = true
    }

    func insertRerankVector(rowID: Int64, blob: Data) throws {
        guard rerankStorageEnabled else { return }
        try db.execute(
            "INSERT OR REPLACE INTO \(Self.rerankVectorTable)(rowid, embedding) VALUES(?, ?)",
            [.int(rowID), .blob(blob)])
    }

    @discardableResult
    func deleteRerankRowsLocked(_ rowIDs: [Int64]) throws -> Int {
        guard rerankStorageEnabled, !rowIDs.isEmpty else { return 0 }
        var removed = 0
        var index = 0
        while index < rowIDs.count {
            let batch = Array(rowIDs[index..<min(index + deleteBatchSize, rowIDs.count)])
            let placeholders = Array(repeating: "?", count: batch.count).joined(separator: ",")
            removed += try db.execute(
                "DELETE FROM \(Self.rerankVectorTable) WHERE rowid IN (\(placeholders))",
                batch.map { SQLValue.int($0) })
            index += batch.count
        }
        rerankRows = max(0, rerankRows - removed)
        return removed
    }

    func resetRerankStorageLocked(recreate: Bool = true) throws {
        rerankRows = 0
        try db.execute("DELETE FROM \(Self.rerankVectorTable)")
        try persistRerankCounterLocked()
        if recreate, elementType != .bin {
            try db.execute("""
            CREATE TABLE IF NOT EXISTS \(Self.rerankVectorTable) (
                rowid INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL
            )
            """)
            try writeRerankMetaLocked()
        }
    }

    func dropRerankStorageLocked() throws {
        rerankRows = 0
        try db.execute("DROP TABLE IF EXISTS \(Self.rerankVectorTable)")
        try deleteRerankMetaLocked()
    }

    func persistRerankCounterLocked() throws {
        guard rerankStorageEnabled else { return }
        try db.execute("""
        INSERT INTO meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, [.text(Self.rerankRowsKey), .text(String(rerankRows))])
    }

    func writeRerankMetaLocked() throws {
        let values: [(String, String)] = [
            (Self.rerankRowsKey, String(rerankRows)),
            (Self.rerankDimensionKey, String(dimension)),
            (Self.rerankElementTypeKey, elementType.rawValue),
        ]
        for (key, value) in values {
            try db.execute("""
            INSERT INTO meta(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, [.text(key), .text(value)])
        }
    }

    func deleteRerankMetaLocked() throws {
        try db.execute("DELETE FROM meta WHERE key IN (?, ?, ?)", [
            .text(Self.rerankRowsKey), .text(Self.rerankDimensionKey),
            .text(Self.rerankElementTypeKey),
        ])
    }
}
```

- [ ] **Step 4: Wire rerank into `VectorStore`**

Add to `VectorStore.swift`:

```swift
public static let rerankVectorTable = "chunk_vectors_rerank"
var rerankStorageEnabled = false
var rerankRows = 0
var rerankStorageReason: String?
```

In `init`, after the vector table block and before `setupBinaryIndexLocked()`:

```swift
do {
    try setupRerankStorageLocked()
} catch {
    AgentLog.warning("rerank 镜像初始化失败，stage 2 回退 vec0: \(error.localizedDescription)")
    rerankStorageEnabled = false
    rerankStorageReason = "初始化失败：\(error.localizedDescription)"
}
```

In `insertVector`, after the main-table insert and before returning:

```swift
try insertRerankVector(rowID: rowID, blob: blob)
```

In `deleteRowIDsLocked`, inside the batch loop after deleting the vector table:

```swift
if rerankStorageEnabled {
    try deleteRerankRowsLocked(batch)
}
```

In `clear()`, inside the transaction after the vector-table delete:

```swift
try resetRerankStorageLocked()
```

In `rebuildForCurrentConfiguration()` and `rebuildDimensionLocked()`, after dropping/recreating the vector table:

```swift
try resetRerankStorageLocked()
```

In `adoptCurrentConfigurationLocked()`, inside the transaction after dropping the vector table:

```swift
try dropRerankStorageLocked()
```

In `recreateVectorTableLocked()`, after dropping the vector table:

```swift
try resetRerankStorageLocked()
```

In `withChunkCounterRollback`, snapshot/restore:

```swift
let rerankBefore = rerankRows
...
rerankRows = rerankBefore
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `swift test --filter VectorStoreRerankTests`
Expected: PASS, 4 tests.

- [ ] **Step 6: Commit**

```bash
git add Sources/BaseAgentCore/RAG/Storage/VectorStoreRerank.swift \
        Sources/BaseAgentCore/RAG/Storage/VectorStore.swift \
        Tests/BaseAgentCoreTests/VectorStoreRerankTests.swift
git commit -m "feat(rag): mirror main vectors for stage 2 rerank"
```

---

### Task 2: stage 2 优先读 rerank 镜像并逐行回退

**Files:**
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift:1132-1152`
- Modify: `Tests/BaseAgentCoreTests/VectorStoreRerankTests.swift`

- [ ] **Step 1: Add failing tests**

```swift
func testExactVectorsPrefersRerankWhenMainRowIsDeleted() throws {
    let store = try makeStore("prefer-rerank")
    _ = try store.add(
        chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0)],
        vectors: [vector(0)], binarySignatures: [Data([0xFF, 0x00])])
    let rowID = try XCTUnwrap(try store.rowIDs(source: "a.md").first)
    try store.db.execute("DELETE FROM \(VectorStore.vectorTable) WHERE rowid = ?",
                         [.int(rowID)])
    let decoded = try store.exactVectorsLocked(rowIDs: [rowID])
    XCTAssertEqual(decoded[rowID]?.count, dimension)
    XCTAssertEqual(decoded[rowID]?[0], 1, accuracy: 0.02)
    store.close()
}

func testExactVectorsFallsBackForMissingRerankRows() throws {
    let store = try makeStore("fallback-rerank")
    _ = try store.add(
        chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0),
                 Chunk(text: "beta", source: "b.md", chunkIndex: 0)],
        vectors: [vector(0), vector(1)],
        binarySignatures: [Data([0xFF, 0x00]), Data([0x00, 0xFF])])
    let ids = try store.rowIDs(source: "a.md") + store.rowIDs(source: "b.md")
    try store.db.execute("DELETE FROM \(VectorStore.rerankVectorTable) WHERE rowid = ?",
                         [.int(ids[0])])
    let decoded = try store.exactVectorsLocked(rowIDs: ids)
    XCTAssertEqual(decoded.count, 2)
    XCTAssertEqual(decoded[ids[0]]?.count, dimension)
    store.close()
}

func testExactVectorsReadsMoreThanOneBatch() throws {
    let store = try makeStore("batches")
    var chunks: [Chunk] = []
    var vectors: [[Float]] = []
    var signatures: [Data] = []
    for index in 0..<300 {
        chunks.append(Chunk(text: "chunk \(index)", source: "s\(index).md", chunkIndex: 0))
        vectors.append(vector(index))
        signatures.append(Data(repeating: UInt8(index % 255), count: dimension / 8))
    }
    _ = try store.add(chunks: chunks, vectors: vectors, binarySignatures: signatures)
    let ids = try store.db.query("SELECT rowid FROM chunks").compactMap { $0["rowid"]?.intValue }
    let decoded = try store.exactVectorsLocked(rowIDs: Array(ids.reversed()))
    XCTAssertEqual(decoded.count, 300)
    store.close()
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter VectorStoreRerankTests`
Expected: `testExactVectorsPrefersRerankWhenMainRowIsDeleted` fails because exactVectors reads the deleted main row.

- [ ] **Step 3: Rewrite `exactVectorsLocked`**

```swift
/// 供 stage 2 与 ANN 重打分使用：优先读 rerank 镜像，缺失行回退主向量表。
/// rowid 升序 + 256 条一批，降低 B-tree 随机跳转与 prepare/step 次数。
func exactVectorsLocked(rowIDs: [Int64]) throws -> [Int64: [Float]] {
    guard !rowIDs.isEmpty else { return [:] }
    let ordered = rowIDs.sorted()
    let batchSize = 256
    var output: [Int64: [Float]] = [:]
    output.reserveCapacity(ordered.count)
    var index = 0
    while index < ordered.count {
        let batch = Array(ordered[index..<min(index + batchSize, ordered.count)])
        var missing = batch
        if rerankStorageEnabled, rerankRows > 0 {
            let placeholders = Array(repeating: "?", count: batch.count).joined(separator: ",")
            let rows = try db.query(
                "SELECT rowid, embedding FROM \(Self.rerankVectorTable) "
                + "WHERE rowid IN (\(placeholders))",
                batch.map { SQLValue.int($0) })
            for row in rows {
                guard let rowID = row["rowid"]?.intValue,
                      let data = row["embedding"]?.dataValue else { continue }
                output[rowID] = VectorEncoder.decode(data, as: elementType, dimension: dimension)
            }
            let found = Set(output.keys)
            missing = batch.filter { !found.contains($0) }
        }
        if !missing.isEmpty {
            let placeholders = Array(repeating: "?", count: missing.count).joined(separator: ",")
            let rows = try db.query(
                "SELECT rowid, embedding FROM \(Self.vectorTable) "
                + "WHERE rowid IN (\(placeholders))",
                missing.map { SQLValue.int($0) })
            for row in rows {
                guard let rowID = row["rowid"]?.intValue,
                      let data = row["embedding"]?.dataValue else { continue }
                output[rowID] = VectorEncoder.decode(data, as: elementType, dimension: dimension)
            }
        }
        index += batch.count
    }
    return output
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter VectorStoreRerankTests`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add Sources/BaseAgentCore/RAG/Storage/VectorStore.swift \
        Tests/BaseAgentCoreTests/VectorStoreRerankTests.swift
git commit -m "feat(rag): read stage 2 vectors from rerank mirror"
```

---

### Task 3: bann schema、meta、状态与构建

**Files:**
- Create: `Sources/BaseAgentCore/RAG/Storage/VectorStoreBinaryANN.swift`
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift`（新增 `bannCentroidCache`）
- Test: `Tests/BaseAgentCoreTests/VectorStoreBinaryANNTests.swift`

- [ ] **Step 1: Write the failing tests**

```swift
import XCTest
@testable import BaseAgentCore

final class VectorStoreBinaryANNTests: XCTestCase {
    private var directory: URL!
    private let dimension = 16
    private let binaryDimension = 16

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("bann-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: directory) }

    private func dbPath(_ name: String) -> String {
        directory.appendingPathComponent("\(name).sqlite").path
    }

    private func makeStore(_ name: String) throws -> VectorStore {
        try VectorStore(dbPath: dbPath(name), dimension: dimension, elementType: .int8,
                        binaryDimension: binaryDimension, binarySource: "sig",
                        indexFingerprint: "fp-\(name)")
    }

    private func signature(bit: Int) -> Data {
        var bytes = Data(repeating: 0, count: binaryDimension / 8)
        bytes[bit / 8] |= UInt8(1 << (bit % 8))
        return bytes
    }

    private func seed(_ store: VectorStore, count: Int) throws {
        var chunks: [Chunk] = []
        var vectors: [[Float]] = []
        var signatures: [Data] = []
        for index in 0..<count {
            var vector = [Float](repeating: 0, count: dimension)
            vector[index % dimension] = 1
            chunks.append(Chunk(text: "chunk \(index)", source: "s\(index).md", chunkIndex: 0))
            vectors.append(vector)
            signatures.append(signature(bit: index % binaryDimension))
        }
        _ = try store.add(chunks: chunks, vectors: vectors, binarySignatures: signatures)
    }

    func testBuildProducesReadyStatus() throws {
        let store = try makeStore("ready")
        try seed(store, count: 40)
        let result = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
        XCTAssertFalse(result.cancelled)
        let status = store.binaryANNIndexStatus()
        XCTAssertTrue(status.isReady, status.summary)
        XCTAssertEqual(status.blocks, 1)
        XCTAssertEqual(status.indexedRows, 40)
        XCTAssertEqual(status.dimension, binaryDimension)
        XCTAssertEqual(status.coverage, 1.0, accuracy: 0.0001)
        store.close()
    }

    func testBuildRejectsIncompleteSignatures() throws {
        let store = try makeStore("incomplete")
        _ = try store.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0)],
            vectors: [[1] + [Float](repeating: 0, count: dimension - 1)])
        XCTAssertThrowsError(
            try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4))
        XCTAssertFalse(store.binaryANNIndexStatus().isReady)
        store.close()
    }

    func testBuildIsCancellableAndResumable() throws {
        let store = try makeStore("resume")
        try seed(store, count: 64)
        var calls = 0
        let partial = try store.buildBinaryANNIndex(
            clusters: 4, clusterSize: 16, iterations: 4,
            progress: { _ in },
            isCancelled: { calls += 1; return calls > 1 })
        XCTAssertTrue(partial.cancelled)
        XCTAssertEqual(store.binaryANNIndexStatus().blocks, 1)
        _ = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
        let status = store.binaryANNIndexStatus()
        XCTAssertTrue(status.isReady, status.summary)
        XCTAssertEqual(status.indexedRows, 64)
        store.close()
    }

    func testClearAndDropRemoveIndex() throws {
        let store = try makeStore("clear")
        try seed(store, count: 40)
        _ = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
        XCTAssertTrue(store.binaryANNIndexStatus().isReady)
        try store.dropBinaryANNIndex()
        XCTAssertFalse(store.binaryANNIndexStatus().isReady)
        _ = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
        try store.clear()
        XCTAssertEqual(store.binaryANNIndexStatus().blocks, 0)
        store.close()
    }

    func testDeleteAccumulatesStaleRows() throws {
        let store = try makeStore("stale")
        try seed(store, count: 40)
        _ = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
        let ids = try store.rowIDs(source: "s0.md")
        _ = try store.deleteRowIDs(ids)
        let status = store.binaryANNIndexStatus()
        XCTAssertEqual(status.staleRows, 1)
        XCTAssertEqual(status.effectiveCoverage, 39.0 / 40.0, accuracy: 0.0001)
        store.close()
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter VectorStoreBinaryANNTests`
Expected: compile failure, `buildBinaryANNIndex` / `binaryANNIndexStatus` / `dropBinaryANNIndex` 不存在。

- [ ] **Step 3: Add `VectorStoreBinaryANN.swift` storage half**

```swift
import Foundation

struct BANNStoredMeta {
    var version = 0
    var clusters = 0
    var blockRows = 0
    var dim = 0
    var source = ""
    var blocks = 0
    var rows = 0
    var staleRows = 0
    var lastRowID: Int64 = 0
    var builtAt: Double = 0
}

extension VectorStore {
    static let bannFormatVersion = 2
    static let bannCellsTable = "bann_cells"
    static let bannBlocksTable = "bann_blocks"
    private var bannMetaPrefix: String { "bann_" }

    func ensureBANNSchemaLocked() throws {
        try db.execScript("""
        CREATE TABLE IF NOT EXISTS \(Self.bannBlocksTable) (
            block_id INTEGER PRIMARY KEY,
            clusters INTEGER NOT NULL,
            dim INTEGER NOT NULL,
            payload BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS \(Self.bannCellsTable) (
            block_id INTEGER NOT NULL,
            cluster_id INTEGER NOT NULL,
            seq INTEGER NOT NULL,
            count INTEGER NOT NULL,
            payload BLOB NOT NULL,
            PRIMARY KEY (block_id, cluster_id, seq)
        ) WITHOUT ROWID;
        """)
    }

    func bannStoredMetaLocked() throws -> BANNStoredMeta {
        var meta = BANNStoredMeta()
        let rows = try db.query("SELECT key, value FROM meta WHERE key GLOB '\(bannMetaPrefix)*'")
        var values: [String: String] = [:]
        for row in rows {
            if let key = row["key"]?.stringValue, let value = row["value"]?.stringValue {
                values[key] = value
            }
        }
        meta.version = Int(values["bann_ver"] ?? "") ?? 0
        meta.clusters = Int(values["bann_clusters"] ?? "") ?? 0
        meta.blockRows = Int(values["bann_block_rows"] ?? "") ?? 0
        meta.dim = Int(values["bann_dim"] ?? "") ?? 0
        meta.source = values["bann_source"] ?? ""
        meta.blocks = Int(values["bann_blocks"] ?? "") ?? 0
        meta.rows = Int(values["bann_rows"] ?? "") ?? 0
        meta.staleRows = Int(values["bann_stale_rows"] ?? "") ?? 0
        meta.lastRowID = Int64(values["bann_last_rowid"] ?? "") ?? 0
        meta.builtAt = Double(values["bann_built_at"] ?? "") ?? 0
        return meta
    }

    func writeBANNMetaLocked(_ meta: BANNStoredMeta) throws {
        let stored = try bannStoredMetaLocked()
        let values: [(String, String)] = [
            ("bann_ver", String(meta.version)),
            ("bann_clusters", String(meta.clusters)),
            ("bann_block_rows", String(meta.blockRows)),
            ("bann_dim", String(meta.dim)),
            ("bann_source", meta.source),
            ("bann_blocks", String(meta.blocks)),
            ("bann_rows", String(stored.rows + meta.rows)),
            ("bann_stale_rows", String(stored.staleRows)),
            ("bann_last_rowid", String(meta.lastRowID)),
            ("bann_built_at", String(meta.builtAt)),
        ]
        for (key, value) in values {
            try db.execute("""
            INSERT INTO meta(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, [.text(key), .text(value)])
        }
    }

    func resetBANNIndexLocked() throws {
        try db.execute("DELETE FROM \(Self.bannCellsTable)")
        try db.execute("DELETE FROM \(Self.bannBlocksTable)")
        try db.execute("DELETE FROM meta WHERE key GLOB '\(bannMetaPrefix)*'")
        bannCentroidCache = nil
    }

    func dropBinaryANNIndexLocked() throws {
        try db.execute("DROP TABLE IF EXISTS \(Self.bannCellsTable)")
        try db.execute("DROP TABLE IF EXISTS \(Self.bannBlocksTable)")
        try db.execute("DELETE FROM meta WHERE key GLOB '\(bannMetaPrefix)*'")
        bannCentroidCache = nil
    }

    public func dropBinaryANNIndex() throws {
        lock.lock(); defer { lock.unlock() }
        try db.transaction { try dropBinaryANNIndexLocked() }
    }

    func recordBANNStaleRowsLocked(_ rowIDs: [Int64]) throws {
        guard !rowIDs.isEmpty,
              let meta = try? bannStoredMetaLocked(), meta.blocks > 0,
              meta.lastRowID > 0 else { return }
        let stale = rowIDs.reduce(0) { $0 + ($1 <= meta.lastRowID ? 1 : 0) }
        guard stale > 0 else { return }
        let updated = min(meta.rows, max(0, meta.staleRows) + stale)
        guard updated != meta.staleRows else { return }
        try db.execute("""
        INSERT INTO meta(key, value) VALUES('bann_stale_rows', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, [.text(String(updated))])
    }
}
```

- [ ] **Step 4: Add bann build, status and signature streaming**

Append to `VectorStoreBinaryANN.swift`:

```swift
extension VectorStore {
    func bannTailScanRowCap() -> Int {
        guard binaryDimension > 0 else { return 0 }
        let rowBytes = VectorElementType.bin.blobSize(dimension: binaryDimension)
        guard rowBytes > 0 else { return 0 }
        return max(1024, Self.annTailScanByteBudget / rowBytes)
    }

    func bannTailRowCountLocked(after rowID: Int64, indexedRows: Int) -> Int {
        let byCount = max(0, binaryRows - indexedRows)
        let cap = bannTailScanRowCap()
        let counted = (try? db.scalar("""
        SELECT COUNT(*) FROM (
            SELECT 1 FROM \(Self.binaryVectorTable) WHERE rowid > ? LIMIT ?
        )
        """, [.int(rowID), .int(Int64(cap + 1))]).intValue).flatMap { Int($0) } ?? 0
        return max(byCount, counted)
    }

    public func binaryANNIndexStatus() -> ANNIndexStatus {
        lock.lock(); defer { lock.unlock() }
        return binaryANNIndexStatusLocked()
    }

    func binaryANNIndexStatusLocked() -> ANNIndexStatus {
        var status = ANNIndexStatus()
        guard let meta = try? bannStoredMetaLocked(), meta.blocks > 0 else {
            status.summary = "二值 ANN 索引未构建"
            return status
        }
        let total = count
        let tail = bannTailRowCountLocked(after: meta.lastRowID, indexedRows: meta.rows)
        let cap = bannTailScanRowCap()
        let unindexed = max(0, tail - cap)
        let dimOK = meta.dim > 0 && meta.dim == binaryDimension
        let sourceOK = !binarySource.isEmpty && meta.source == binarySource
        let versionOK = meta.version == Self.bannFormatVersion
        let stale = min(meta.rows, max(0, meta.staleRows))
        status.clusters = meta.clusters
        status.blockRows = meta.blockRows
        status.blocks = meta.blocks
        status.indexedRows = meta.rows
        status.tailRows = tail
        status.unindexedRows = unindexed
        status.staleRows = stale
        status.dimension = meta.dim
        status.builtAt = meta.builtAt
        status.isReady = dimOK && sourceOK && versionOK && tail == 0
        status.isUsable = dimOK && sourceOK && versionOK && unindexed == 0
        status.coverage = total > 0 ? min(1, Double(meta.rows) / Double(total)) : 0
        status.effectiveCoverage = total > 0
            ? min(1, Double(max(0, meta.rows - stale)) / Double(total)) : 0
        status.needsRebuild = status.isUsable
            && (tail > annRebuildTailThreshold(dim: meta.dim)
                || Double(stale) / Double(max(1, meta.rows)) >= Self.annStaleRebuildRatio)
        if !versionOK { status.summary = "二值 ANN 索引格式版本 \(meta.version) 与当前 \(Self.bannFormatVersion) 不符，需重建"; return status }
        if !dimOK { status.summary = "二值 ANN 索引位宽 \(meta.dim) 与当前 \(binaryDimension) 不符，需重建"; return status }
        if !sourceOK { status.summary = "二值 ANN 索引签名来源与当前配置不符，需重建"; return status }
        var parts = [
            "二值 ANN 索引：\(status.clusters) 簇/块 × \(status.blockRows / max(1, status.clusters)) 行/簇",
            "\(status.blocks) 块", "已索引 \(status.indexedRows) 切片",
            "\(status.dimension) bit",
        ]
        if status.tailRows > 0 { parts.append("尾块 \(status.tailRows) 切片（精确补齐）") }
        if status.unindexedRows > 0 { parts.append("尾块超出补齐上限，需重建索引") }
        if status.staleRows > 0 { parts.append("陈旧 \(status.staleRows) 切片") }
        parts.append(String(format: "覆盖 %.1f%%", status.coverage * 100))
        if status.needsRebuild { parts.append("建议重建") }
        status.summary = parts.joined(separator: "，")
        return status
    }

    func streamBinarySignatures(from rowID: Int64, dim: Int,
                                _ body: (Int64, UnsafeRawPointer) throws -> Bool) throws {
        let rowBytes = VectorElementType.bin.blobSize(dimension: dim)
        guard rowBytes > 0 else { return }
        try db.stream(
            "SELECT rowid, embedding FROM \(Self.binaryVectorTable) WHERE rowid > ?",
            [.int(rowID)]) { row in
                let hitRowID = row.int(0)
                guard let blob = row.blob(1), blob.count >= rowBytes else { return true }
                return try body(hitRowID, blob.pointer)
            }
    }

    func gatherBANNBlockLocked(after rowID: Int64, maxRows: Int,
                               into buffer: inout ANNBlockBuffer) throws {
        buffer.reset(maxRows: maxRows, dim: binaryDimension, elementType: .bin)
        try streamBinarySignatures(from: rowID, dim: binaryDimension) { hitRowID, pointer in
            buffer.append(rowID: hitRowID, pointer: pointer, dim: binaryDimension,
                          elementType: .bin, scratch: &[])
            return buffer.count < maxRows
        }
        if buffer.needsSort { buffer.sortByRowID() }
    }

    func writeBANNBlockLocked(blockID: Int, clusters: Int, blockRows: Int,
                              iterations: Int, buffer: ANNBlockBuffer) throws {
        let rows = buffer.count
        guard rows > 0 else { return }
        let activeClusters = min(clusters, rows)
        var centroids = [Float](repeating: 0, count: clusters * binaryDimension)
        var counts = [Int32](repeating: 0, count: clusters)
        if let seeds = annSeedCentroids(buffer: buffer, dim: binaryDimension,
                                        clusters: activeClusters),
           let trained = ANNClustering.train(
            buffer: buffer, rows: rows, dim: binaryDimension,
            clusters: activeClusters, iterations: iterations, seeds: seeds) {
            for i in 0..<(activeClusters * binaryDimension) { centroids[i] = trained.centroids[i] }
            for cluster in 0..<activeClusters { counts[cluster] = trained.counts[cluster] }
        }
        let assignments = ANNClustering.assign(
            buffer: buffer, rows: rows, dim: binaryDimension,
            clusters: activeClusters, centroids: centroids)
        var offsets = [Int](repeating: 0, count: activeClusters + 1)
        for assignment in assignments { offsets[Int(assignment) + 1] += 1 }
        for cluster in 0..<activeClusters { offsets[cluster + 1] += offsets[cluster] }
        var order = [Int32](repeating: 0, count: rows)
        var cursor = offsets
        for (index, assignment) in assignments.enumerated() {
            let cluster = Int(assignment)
            order[cursor[cluster]] = Int32(index)
            cursor[cluster] += 1
        }

        try db.transaction {
            for cluster in 0..<activeClusters {
                let start = offsets[cluster]
                let end = offsets[cluster + 1]
                guard end > start else { continue }
                let payload = ANNPacking.packCell(buffer: buffer, order: order,
                                                  range: start..<end, dim: binaryDimension)
                try db.execute("""
                INSERT OR REPLACE INTO \(Self.bannCellsTable)
                    (block_id, cluster_id, seq, count, payload)
                VALUES(?, ?, 0, ?, ?)
                """, [.int(Int64(blockID)), .int(Int64(cluster)),
                      .int(Int64(end - start)), .blob(payload)])
            }
            let payload = ANNPacking.packCentroids(
                centroids: centroids, counts: counts,
                clusters: activeClusters, dim: binaryDimension)
            try db.execute("""
            INSERT OR REPLACE INTO \(Self.bannBlocksTable)(block_id, clusters, dim, payload)
            VALUES(?, ?, ?, ?)
            """, [.int(Int64(blockID)), .int(Int64(activeClusters)),
                  .int(Int64(binaryDimension)), .blob(payload)])
            try writeBANNMetaLocked(BANNStoredMeta(
                version: Self.bannFormatVersion, clusters: clusters, blockRows: blockRows,
                dim: binaryDimension, source: binarySource, blocks: blockID + 1,
                rows: rows, lastRowID: buffer.lastRowID,
                builtAt: Date().timeIntervalSince1970))
        }
        bannCentroidCache = nil
    }

    @discardableResult
    public func buildBinaryANNIndex(
        clusters: Int = 256, clusterSize: Int = 512, iterations: Int = 8,
        progress: ((ANNBuildProgress) -> Void)? = nil,
        isCancelled: (() -> Bool)? = nil
    ) throws -> ANNBuildResult {
        let started = Date()
        guard binaryStorageEnabled, binaryDimension > 0 else {
            throw AgentError.storage("bin 表不可用，无法构建二值 ANN")
        }
        guard binaryRows == count, binaryRows > 0 else {
            throw AgentError.storage("二值签名覆盖不完整，请重新同步后再构建二值 ANN")
        }
        let clusterCount = min(max(2, clusters), 4096)
        let rowsPerCluster = min(max(1, clusterSize), 8192)
        let blockRows = clusterCount * rowsPerCluster
        func locked<T>(_ body: () throws -> T) rethrows -> T {
            lock.lock(); defer { lock.unlock() }
            return try body()
        }
        var meta = BANNStoredMeta()
        var stopRowID: Int64 = 0
        try locked {
            try ensureBANNSchemaLocked()
            meta = try bannStoredMetaLocked()
            let paramsChanged = meta.version != Self.bannFormatVersion
                || meta.dim != binaryDimension || meta.source != binarySource
                || meta.clusters != clusterCount || meta.blockRows != blockRows
            let staleTooHigh = meta.blocks > 0 && meta.rows > 0
                && Double(meta.staleRows) / Double(meta.rows) >= Self.annStaleRebuildRatio
            if paramsChanged || staleTooHigh {
                if meta.blocks > 0 { AgentLog.info("二值 ANN: 参数/陈旧比例触发整体重建") }
                try resetBANNIndexLocked()
                meta = BANNStoredMeta()
            }
            stopRowID = (try? db.scalar(
                "SELECT COALESCE(MAX(rowid), 0) FROM \(Self.binaryVectorTable)").intValue)
                .flatMap { Int64($0) } ?? 0
        }
        var result = ANNBuildResult()
        result.clusters = clusterCount
        result.blockRows = blockRows
        var blockID = meta.blocks
        var indexedRows = meta.rows
        var watermark = meta.lastRowID
        let total = binaryRows
        let remaining = max(0, total - indexedRows)
        let estimatedBlocks = max(1, (remaining + blockRows - 1) / blockRows)
        progress?(ANNBuildProgress(
            stage: "start", blocks: blockID, totalBlocks: estimatedBlocks,
            rows: indexedRows, totalRows: max(total, indexedRows),
            message: "开始构建二值 ANN 索引（\(clusterCount) 簇/块 × \(rowsPerCluster) 行/簇）"))

        while true {
            let step = try locked { () -> (finished: Bool, cancelled: Bool,
                                           update: ANNBuildProgress?) in
                if let isCancelled, isCancelled() { return (true, true, nil) }
                if watermark >= stopRowID { return (true, false, nil) }
                var buffer = ANNBlockBuffer()
                try gatherBANNBlockLocked(after: watermark, maxRows: blockRows, into: &buffer)
                guard buffer.count > 0 else { return (true, false, nil) }
                try writeBANNBlockLocked(blockID: blockID, clusters: clusterCount,
                                         blockRows: blockRows, iterations: iterations,
                                         buffer: buffer)
                blockID += 1
                indexedRows += buffer.count
                watermark = buffer.lastRowID
                return (false, false, ANNBuildProgress(
                    stage: "blocks", blocks: blockID,
                    totalBlocks: max(estimatedBlocks, blockID),
                    rows: indexedRows, totalRows: max(total, indexedRows),
                    message: "已写入 \(blockID) 块二值 ANN（\(indexedRows) 切片）"))
            }
            if let update = step.update { progress?(update) }
            if step.cancelled { result.cancelled = true; break }
            if step.finished { break }
        }

        locked {
            result.blocks = blockID
            result.indexedRows = indexedRows
            result.tailRows = max(0, count - indexedRows)
        }
        result.elapsed = Date().timeIntervalSince(started)
        var summary = "二值 ANN 索引：\(blockID) 块 / \(indexedRows) 切片，耗时 "
            + String(format: "%.1fs", result.elapsed)
        if result.tailRows > 0 { summary += "，尾块 \(result.tailRows) 切片" }
        if result.cancelled { summary += "（已取消，下次可续建）" }
        result.summary = summary
        AgentLog.info(summary)
        return result
    }
}
```

- [ ] **Step 5: Add cache property and lifecycle wiring in `VectorStore.swift`**

Add property:

```swift
var bannCentroidCache: ANNCentroidCache?
```

In `deleteRowIDsLocked`, after ANN stale accounting:

```swift
try recordBANNStaleRowsLocked(rowIDs)
```

In `clear()` after `resetANNIndexLocked()`:

```swift
try? resetBANNIndexLocked()
```

In `adoptCurrentConfigurationLocked` / `rebuildForCurrentConfiguration` / `rebuildDimensionLocked`, extend the existing ANN table/meta cleanup with:

```swift
try db.execute("DROP TABLE IF EXISTS \(Self.bannCellsTable)")
try db.execute("DROP TABLE IF EXISTS \(Self.bannBlocksTable)")
```

and after the transaction:

```swift
bannCentroidCache = nil
```

In `setupBinaryIndexLocked()`, inside the transaction that drops/recreates `chunk_vectors_bin`:

```swift
try dropBinaryANNIndexLocked()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `swift test --filter VectorStoreBinaryANNTests`
Expected: PASS, 5 tests.

- [ ] **Step 7: Commit**

```bash
git add Sources/BaseAgentCore/RAG/Storage/VectorStoreBinaryANN.swift \
        Sources/BaseAgentCore/RAG/Storage/VectorStore.swift \
        Tests/BaseAgentCoreTests/VectorStoreBinaryANNTests.swift
git commit -m "feat(rag): build binary ANN index over signatures"
```

---

### Task 4: bann 查询、簇心缓存、尾块补齐与 stage 1 回退链

**Files:**
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStoreBinaryANN.swift`
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStoreBinary.swift`
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift`（计数器）
- Modify: `Tests/BaseAgentCoreTests/VectorStoreBinaryANNTests.swift`

- [ ] **Step 1: Add failing tests**

```swift
func testBannQueryMatchesBruteForceWithFullProbe() throws {
    let store = try makeStore("query")
    try seed(store, count: 40)
    _ = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
    let query = signature(bit: 7)
    let bann = try XCTUnwrap(
        try store.bannCandidatesLocked(signature: query, topK: 5, probeOverride: 4))
    let vec = try store.db.query(
        "SELECT rowid, distance FROM \(VectorStore.binaryVectorTable) "
        + "WHERE embedding MATCH vec_bit(?) AND k = 5", [.blob(query)])
    XCTAssertEqual(bann.map(\.0), vec.compactMap { $0["rowid"]?.intValue })
    store.close()
}

func testSearchUsesBannAndSkipsANN() throws {
    let store = try makeStore("use-bann")
    try seed(store, count: 40)
    _ = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
    let query = signature(bit: 3)
    let hits = try store.hybridSearch(
        vector: [1] + [Float](repeating: 0, count: dimension - 1),
        query: "zzz", topK: 3,
        binary: BinaryCoarseOptions(signature: query, candidateLimit: 20),
        ann: ANNQueryOptions(nprobe: 4, rescore: 0))
    XCTAssertFalse(hits.isEmpty)
    XCTAssertGreaterThan(store.bannQueryCount, 0)
    XCTAssertEqual(store.annQueryCount, 0)
    XCTAssertEqual(store.binaryVecQueryCount, 0)
    store.close()
}

func testSearchFallsBackToVec0WhenBannNotBuilt() throws {
    let store = try makeStore("vec-fallback")
    try seed(store, count: 40)
    let query = signature(bit: 1)
    _ = try store.hybridSearch(
        vector: [1] + [Float](repeating: 0, count: dimension - 1),
        query: "zzz", topK: 3,
        binary: BinaryCoarseOptions(signature: query, candidateLimit: 20),
        ann: ANNQueryOptions(nprobe: 4, rescore: 0))
    XCTAssertEqual(store.bannQueryCount, 0)
    XCTAssertGreaterThan(store.binaryVecQueryCount, 0)
    XCTAssertEqual(store.annQueryCount, 0)
    store.close()
}

func testTailIsSearchableAfterBuild() throws {
    let store = try makeStore("tail")
    try seed(store, count: 32)
    _ = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
    _ = try store.add(
        chunks: [Chunk(text: "tail", source: "tail.md", chunkIndex: 0)],
        vectors: [[Float](repeating: 0, count: dimension)],
        binarySignatures: [Data(repeating: 0xFF, count: binaryDimension / 8)])
    let hits = try store.hybridSearch(
        vector: [1] + [Float](repeating: 0, count: dimension - 1),
        query: "tail", topK: 5, binary: BinaryCoarseOptions(
            signature: Data(repeating: 0xFF, count: binaryDimension / 8),
            candidateLimit: 20))
    XCTAssertTrue(hits.contains { $0.text == "tail" })
    store.close()
}

func testDeletedRowsAreExcludedFromBannResults() throws {
    let store = try makeStore("deleted")
    try seed(store, count: 20)
    _ = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
    _ = try store.deleteBySource("s3.md")
    let hits = try store.hybridSearch(
        vector: [1] + [Float](repeating: 0, count: dimension - 1),
        query: "zzz", topK: 5,
        binary: BinaryCoarseOptions(signature: signature(bit: 3), candidateLimit: 20))
    XCTAssertFalse(hits.contains { $0.source == "s3.md" })
    store.close()
}

func testStageTwoUsesRerankMirror() throws {
    let store = try makeStore("rerank-stage2")
    try seed(store, count: 20)
    _ = try store.buildBinaryANNIndex(clusters: 4, clusterSize: 16, iterations: 4)
    XCTAssertEqual(store.binaryIndexStatus().stage2Backend, "rerank")
    store.close()
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter VectorStoreBinaryANNTests`
Expected: compile failure, `bannCandidatesLocked` / counters / `stage2Backend` 不存在。

- [ ] **Step 3: Add query implementation**

Append to `VectorStoreBinaryANN.swift`:

```swift
extension VectorStore {
    private func bannCentroidCacheLocked(blocks: Int, clusters: Int,
                                         dim: Int) throws -> ANNCentroidCache {
        if let cache = bannCentroidCache, cache.blocks == blocks,
           cache.clusters == clusters, cache.dim == dim { return cache }
        var codes = [Int8](repeating: 0, count: blocks * clusters * dim)
        var scales = [Float](repeating: 0, count: blocks * clusters)
        for block in 0..<blocks {
            guard let payload = try db.scalar(
                "SELECT payload FROM \(Self.bannBlocksTable) WHERE block_id = ?",
                [.int(Int64(block))]).dataValue else {
                throw AgentError.storage("二值 ANN 索引块 \(block) 缺失，需要重建")
            }
            let activeClusters = Int((try? db.scalar(
                "SELECT clusters FROM \(Self.bannBlocksTable) WHERE block_id = ?",
                [.int(Int64(block))]).intValue) ?? Int64(clusters))
            try decodeBANNCentroidPayload(
                payload, activeClusters: min(clusters, max(0, activeClusters)),
                clusters: clusters, dim: dim, block: block,
                codes: &codes, scales: &scales)
        }
        let cache = ANNCentroidCache(blocks: blocks, clusters: clusters,
                                     dim: dim, codes: codes, scales: scales)
        bannCentroidCache = cache
        return cache
    }

    private func decodeBANNCentroidPayload(
        _ payload: Data, activeClusters: Int, clusters: Int, dim: Int, block: Int,
        codes: inout [Int8], scales: inout [Float]
    ) throws {
        let headerBytes = 12
        let entryBytes = 8 + dim
        guard activeClusters > 0, dim > 0,
              payload.count >= headerBytes + activeClusters * entryBytes else {
            throw AgentError.storage("二值 ANN 块 \(block) 数据损坏（长度 \(payload.count)）")
        }
        payload.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            var offset = headerBytes
            for cluster in 0..<activeClusters {
                let scale = raw.loadUnaligned(fromByteOffset: offset + 4, as: Float.self)
                let codeBase = base.advanced(by: offset + 8).assumingMemoryBound(to: Int8.self)
                let destination = block * clusters * dim + cluster * dim
                for i in 0..<dim { codes[destination + i] = codeBase[i] }
                scales[block * clusters + cluster] = scale
                offset += entryBytes
            }
        }
    }

    private func scoreBANNCellsLocked(block: Int, clusters: [Int], dim: Int,
                                      queryBits: [UInt8],
                                      pool: inout ANNTopKCollector) throws {
        guard !clusters.isEmpty else { return }
        let placeholders = Array(repeating: "?", count: clusters.count).joined(separator: ",")
        var bindings: [SQLValue] = [.int(Int64(block))]
        bindings.append(contentsOf: clusters.map { .int(Int64($0)) })
        let rows = try db.query("""
        SELECT payload FROM \(Self.bannCellsTable)
        WHERE block_id = ? AND cluster_id IN (\(placeholders))
        ORDER BY cluster_id, seq
        """, bindings)
        let bitBytes = VectorElementType.bin.blobSize(dimension: dim)
        let rowBytes = 8 + bitBytes
        for row in rows {
            guard let payload = row["payload"]?.dataValue else { continue }
            let count = payload.count / rowBytes
            guard count > 0 else { continue }
            payload.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                for index in 0..<count {
                    let offset = index * rowBytes
                    let rowID = raw.loadUnaligned(fromByteOffset: offset, as: Int64.self)
                    let hamming = VectorEncoder.hamming(
                        pointer: UnsafeRawPointer(base.advanced(by: offset + 8)),
                        bytes: queryBits, count: bitBytes)
                    pool.offer(rowid: rowID, score: 1 - Double(hamming) / Double(dim))
                }
            }
        }
    }

    private func scanBANTailLocked(signature: Data, after rowID: Int64, dim: Int,
                                   rowLimit: Int, pool: inout ANNTopKCollector) throws {
        let queryBits = [UInt8](signature)
        var scanned = 0
        try streamBinarySignatures(from: rowID, dim: dim) { hitRowID, pointer in
            let hamming = VectorEncoder.hamming(pointer: pointer, bytes: queryBits,
                                                count: dim / 8)
            pool.offer(rowid: hitRowID, score: 1 - Double(hamming) / Double(dim))
            scanned += 1
            return scanned < rowLimit
        }
    }

    /// 二值 ANN 候选；返回 nil 表示不可用，调用方回退 vec0 bin KNN。
    func bannCandidatesLocked(signature: Data, topK: Int,
                              probeOverride: Int? = nil) throws -> [(Int64, Double)]? {
        guard topK > 0, binaryStorageEnabled, binaryDimension > 0,
              signature.count == binaryDimension / 8,
              binaryRows == count, count > 0 else { return nil }
        guard let meta = try? bannStoredMetaLocked(), meta.blocks > 0,
              meta.version == Self.bannFormatVersion,
              meta.dim == binaryDimension, meta.source == binarySource else { return nil }
        let cap = bannTailScanRowCap()
        let tailRows = bannTailRowCountLocked(after: meta.lastRowID, indexedRows: meta.rows)
        guard tailRows <= cap else { return nil }

        let rowsPerCluster = max(1, meta.blockRows / max(1, meta.clusters))
        let baseProbe = 2 + (topK + rowsPerCluster - 1) / rowsPerCluster
        let probe = probeOverride ?? min(16, max(4, baseProbe) + (meta.blocks <= 2 ? 4 : 0))
        let queryVector = VectorCodec.normalize(
            VectorEncoder.dequantizeBits(signature, dimension: meta.dim))
        let queryBits = [UInt8](signature)
        let capacity = min(4096, max(topK * 4, 512))
        var pool = ANNTopKCollector(capacity: capacity)
        let centroids = try bannCentroidCacheLocked(blocks: meta.blocks,
                                                    clusters: meta.clusters, dim: meta.dim)
        var scratch: [Float] = []
        queryVector.withUnsafeBufferPointer { pointer in
            guard let base = pointer.baseAddress else { return }
            for block in 0..<meta.blocks {
                let probes = centroids.nearest(query: base, block: block,
                                               nprobe: probe, scratch: &scratch)
                try? scoreBANNCellsLocked(block: block, clusters: probes, dim: meta.dim,
                                          queryBits: queryBits, pool: &pool)
            }
        }
        if tailRows > 0 {
            try scanBANTailLocked(signature: signature, after: meta.lastRowID,
                                  dim: meta.dim, rowLimit: tailRows, pool: &pool)
        }
        let candidates = pool.sortedItems()
        guard !candidates.isEmpty else { return nil }
        bannQueryCount += 1
        return Array(candidates.prefix(topK))
    }
}
```

- [ ] **Step 4: Rewrite `binaryCoarseCandidatesLocked` stage 1**

Replace the vec0 KNN block in `VectorStoreBinary.swift` with:

```swift
let k = min(maxVectorK, max(candidateLimit, candidateCount))
func vec0Hits() throws -> [(Int64, Double)] {
    binaryVecQueryCount += 1
    let rows = try db.query(
        "SELECT rowid, distance FROM \(Self.binaryVectorTable) "
        + "WHERE embedding MATCH vec_bit(?) AND k = ?",
        [.blob(signature), .int(Int64(k))])
    return rows.compactMap { row in
        guard let rowID = row["rowid"]?.intValue else { return nil }
        let distance = row["distance"]?.doubleValue ?? Double(binaryDimension)
        return (rowID, max(0, 1 - distance / Double(binaryDimension)))
    }
}

let bannHits = try? bannCandidatesLocked(signature: signature, topK: k)
let hits = bannHits ?? (try vec0Hits())
guard !hits.isEmpty else { return .skipped("二值粗筛无候选") }

func rerank(_ hits: [(Int64, Double)]) throws -> [(Int64, Double, Double)] {
    let stored = try exactVectorsLocked(rowIDs: hits.map(\.0))
    var output: [(Int64, Double, Double)] = []
    output.reserveCapacity(hits.count)
    for (rowID, binaryScore) in hits {
        guard let stored = stored[rowID], stored.count == vector.count else { continue }
        output.append((rowID, VectorCodec.cosine(vector, stored), binaryScore))
    }
    return output
}

var reranked = try rerank(hits)
if reranked.isEmpty, bannHits != nil {
    reranked = try rerank(try vec0Hits())
}
reranked.sort { $0.1 > $1.1 }
let selected = Array(reranked.prefix(candidateCount))
return .used(
    candidates: selected.map { ($0.0, $0.1) },
    scores: Dictionary(uniqueKeysWithValues: selected.map { ($0.0, $0.2) }))
```

- [ ] **Step 5: Add counters and `stage2Backend`**

In `VectorStore.swift`:

```swift
var bannQueryCount = 0
var binaryVecQueryCount = 0
```

In `VectorStoreBinary.swift` `BinaryIndexStatus` add:

```swift
public var rerankRows: Int
public var stage2Backend: String
```

with init defaults and fill in `binaryIndexStatusLocked`:

```swift
let backend: String
if chunkRows > 0, rerankRows == chunkRows { backend = "rerank" }
else if rerankRows > 0 { backend = "partial" }
else { backend = "vec0" }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `swift test --filter VectorStoreBinaryANNTests`
Expected: PASS, 11 tests. Also run `swift test --filter VectorStoreBinarySearchTests` to ensure unchanged vec0 fallback behavior.

- [ ] **Step 7: Commit**

```bash
git add Sources/BaseAgentCore/RAG/Storage/VectorStoreBinaryANN.swift \
        Sources/BaseAgentCore/RAG/Storage/VectorStoreBinary.swift \
        Sources/BaseAgentCore/RAG/Storage/VectorStore.swift \
        Tests/BaseAgentCoreTests/VectorStoreBinaryANNTests.swift
git commit -m "feat(rag): query binary ANN with vec0 fallback"
```

---

### Task 5: 引擎构建目标、状态与工具输出

**Files:**
- Modify: `Sources/BaseAgentCore/RAG/RAGEngine.swift`
- Modify: `Sources/BaseAgentCore/Tools/RAGTool.swift`
- Create: `Tests/BaseAgentCoreTests/RAGEngineBinaryANNTests.swift`
- Modify: `Tests/BaseAgentCoreTests/RAGToolBinaryStatusTests.swift`

- [ ] **Step 1: Write failing engine tests**

```swift
import XCTest
@testable import BaseAgentCore

final class RAGEngineBinaryANNTests: XCTestCase {
    private var root: URL!
    private var kb: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("binary-ann-engine-\(UUID().uuidString)")
        kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    private final class SigningEmbedding: EmbeddingProvider, EmbeddingBinarySigning {
        let identifier = "binary-ann-fake"
        let dimension = 8
        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map { text in
                var value = [Float](repeating: 0, count: 8)
                for (index, scalar) in text.unicodeScalars.enumerated() {
                    value[(Int(scalar.value) + index) % 8] += 1
                }
                return VectorCodec.normalize(value)
            }
        }
        func embedQuery(_ text: String) async throws -> [Float] {
            try await embedDocuments([text])[0]
        }
    }

    private func writeDocs(_ count: Int) throws {
        for index in 1...count {
            let text = String(repeating: "apple banana turbine engine \(index) ", count: 20)
                + "\nKnowledge retrieval pipeline stores vectors for semantic search."
            try text.write(to: kb.appendingPathComponent("doc\(index).md"),
                           atomically: true, encoding: .utf8)
        }
    }

    private func makeEngine(_ adjust: (inout AgentConfig) -> Void = { _ in }) -> RAGEngine {
        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 5
        config.ragVectorElementType = .int8
        config.ragANNMinChunks = 1
        config.ragANNClusters = 4
        config.ragANNClusterSize = 8
        adjust(&config)
        return RAGEngine(config: config, knowledgeBase: kb.path,
                         embedding: SigningEmbedding(), reranker: nil)
    }

    func testBuildTargetsBinaryWhenCoverageComplete() async throws {
        try writeDocs(2)
        let engine = makeEngine()
        _ = await engine.ingest()
        XCTAssertEqual(engine.resolvedANNIndexTarget(), .binary)
        let result = try XCTUnwrap(engine.buildANNIndex())
        XCTAssertTrue(result.summary.contains("二值 ANN"), result.summary)
        XCTAssertTrue(try XCTUnwrap(engine.binaryANNIndexStatus()).isReady)
        XCTAssertFalse(try XCTUnwrap(engine.annIndexStatus()).isReady)
        engine.close()
    }

    func testBuildFallsBackToDenseWhenBinaryDisabled() async throws {
        try writeDocs(2)
        let engine = makeEngine { $0.ragBinaryCoarseEnabled = false }
        _ = await engine.ingest()
        XCTAssertEqual(engine.resolvedANNIndexTarget(), .dense)
        _ = try XCTUnwrap(engine.buildANNIndex())
        XCTAssertTrue(try XCTUnwrap(engine.annIndexStatus()).isReady)
        XCTAssertFalse(try XCTUnwrap(engine.binaryANNIndexStatus()).isReady)
        engine.close()
    }

    func testStatusExposesBinaryANNFields() async throws {
        try writeDocs(2)
        let engine = makeEngine()
        _ = await engine.ingest()
        _ = try XCTUnwrap(engine.buildANNIndex())
        let status = engine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(status["binary_ann"]?.stringValue, "ready")
        XCTAssertEqual(status["binary_backend"]?.stringValue, "bann")
        XCTAssertEqual(status["stage2_backend"]?.stringValue, "rerank")
        XCTAssertEqual(status["binary_ann_building"]?.boolValue, false)
        engine.close()
    }

    func testDropRemovesBothIndexes() async throws {
        try writeDocs(2)
        let engine = makeEngine()
        _ = await engine.ingest()
        _ = try XCTUnwrap(engine.buildANNIndex())
        try engine.dropANNIndex()
        XCTAssertFalse(try XCTUnwrap(engine.binaryANNIndexStatus()).isReady)
        XCTAssertFalse(try XCTUnwrap(engine.annIndexStatus()).isReady)
        engine.close()
    }

    func testSearchUsesBinaryANN() async throws {
        try writeDocs(2)
        let engine = makeEngine()
        _ = await engine.ingest()
        _ = try XCTUnwrap(engine.buildANNIndex())
        let hits = try await engine.search(query: "apple turbine", topK: 3)
        XCTAssertFalse(hits.isEmpty)
        XCTAssertTrue(hits.contains { $0.binaryScore != nil })
        engine.close()
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter RAGEngineBinaryANNTests`
Expected: compile failure, `resolvedANNIndexTarget` / `binaryANNIndexStatus` 不存在。

- [ ] **Step 3: Add target selection and build dispatch**

In `RAGEngine.swift`:

```swift
public enum ANNIndexTarget: String, Sendable {
    case dense = "dense"
    case binary = "binary"
}
```

Add state:

```swift
private var annBuildTarget: ANNIndexTarget = .dense
```

Add method:

```swift
public func resolvedANNIndexTarget(timeout: Double? = nil) -> ANNIndexTarget {
    guard config.ragBinaryCoarseEnabled,
          effectiveVectorElementType != .bin else { return .dense }
    guard let binary = binaryIndexStatus(timeout: timeout),
          binary.isComplete, binary.dimension > 0 else { return .dense }
    return .binary
}

public func binaryANNIndexStatus(timeout: Double? = nil) -> ANNIndexStatus? {
    try? withStoreBounded(timeout: resolvedTimeout(timeout)) {
        $0.binaryANNIndexStatus()
    }
}
```

In `performANNBuild`, replace `store.buildANNIndex(...)` with:

```swift
let target = resolvedANNIndexTarget()
annBuildLock.lock()
annBuildTarget = target
annBuildLock.unlock()
let result: ANNBuildResult
switch target {
case .binary:
    result = try store.buildBinaryANNIndex(
        clusters: clusters, clusterSize: clusterSize,
        progress: { [weak self] update in
            self?.recordANNBuildProgress(update)
            progress?(update)
            guard Date().timeIntervalSince(lastLog) >= 10,
                  update.stage == "blocks" else { return }
            lastLog = Date()
            logFn("RAG: \(update.message)")
        },
        isCancelled: { [weak self] in
            Task.isCancelled || (self?.isCancelled ?? true)
        })
case .dense:
    result = try store.buildANNIndex(
        clusters: clusters, clusterSize: clusterSize,
        progress: { [weak self] update in
            self?.recordANNBuildProgress(update)
            progress?(update)
            guard Date().timeIntervalSince(lastLog) >= 10,
                  update.stage == "blocks" else { return }
            lastLog = Date()
            logFn("RAG: \(update.message)")
        },
        isCancelled: { [weak self] in
            Task.isCancelled || (self?.isCancelled ?? true)
        })
}
logFn("RAG: \(result.summary)")
invalidateStatusCache()
return result
```

Update `dropANNIndex()`:

```swift
try store.dropANNIndex()
try store.dropBinaryANNIndex()
AgentLog.info("RAG: ANN 与二值 ANN 索引已删除")
```

Update auto-build guard:

```swift
guard config.ragANNAutoBuild,
      config.ragANNEnabled || config.ragBinaryCoarseEnabled else { return }
```

In `runBackgroundANNAutoBuild`, replace status overview with target-aware status:

```swift
let store = try ensureStore()
let chunks = store.count
let target = resolvedANNIndexTarget()
let status = target == .binary
    ? store.binaryANNIndexStatus()
    : store.annIndexStatus()
```

and call `performANNBuild` unchanged (it recomputes target; config is stable).

- [ ] **Step 4: Extend status and build-state merge**

In `status()` add to the returned dictionary:

```swift
"binary_ann": .string(binaryANNState),
"binary_ann_index": .string(binaryANN.summary),
"binary_ann_note": .string(binaryANNNote),
"binary_ann_building": .bool(buildingBinaryANN),
"ann_build_target": .string(annBuildTargetRaw),
"binary_backend": .string(binaryBackend),
"rerank_rows": .int(binary.rerankRows),
"stage2_backend": .string(binary.stage2Backend),
```

Compute:

```swift
let binaryANN = store.binaryANNIndexStatus()
let binaryANNState: String
let binaryANNNote: String
if !config.ragBinaryCoarseEnabled {
    binaryANNState = "off"; binaryANNNote = "已关闭"
} else if store.elementType == .bin {
    binaryANNState = "bypassed_bin"; binaryANNNote = "bin 主存储仅支持单阶段检索"
} else if binarySignatureDimension == 0 || !binary.storageEnabled || !binary.isComplete {
    binaryANNState = "incomplete"; binaryANNNote = "二值签名覆盖不完整，请重新同步"
} else if buildingBinaryANN {
    binaryANNState = "building"; binaryANNNote = "正在构建二值 ANN"
} else if binaryANN.blocks == 0 {
    binaryANNState = "not_built"; binaryANNNote = "本次回退 bin 表暴力汉明"
} else if !binaryANN.isUsable {
    binaryANNState = "unusable"; binaryANNNote = binaryANN.summary
} else if binaryANN.needsRebuild {
    binaryANNState = "needs_rebuild"; binaryANNNote = binaryANN.summary
} else if binaryANN.tailRows > 0 {
    binaryANNState = "tail"; binaryANNNote = binaryANN.summary
} else {
    binaryANNState = "ready"; binaryANNNote = ""
}
let binaryBackend: String
if binaryANNState == "ready" || binaryANNState == "tail" { binaryBackend = "bann" }
else if binary.storageEnabled { binaryBackend = "sqlite-vec" }
else { binaryBackend = "none" }
```

In `mergingANNBuildState`, read `annBuildTarget` and add:

```swift
let target = annBuildTarget
output["ann_build_target"] = .string(running ? target.rawValue : "")
output["binary_ann_building"] = .bool(running && target == .binary)
if running && target == .binary { output["binary_ann"] = .string("building") }
```

- [ ] **Step 5: Extend `RAGTool` output**

In `RAGTool.swift`, parse:

```swift
let binaryANN = status["binary_ann"]?.stringValue ?? ""
let binaryANNIndex = status["binary_ann_index"]?.stringValue ?? ""
let binaryANNNote = status["binary_ann_note"]?.stringValue ?? ""
let binaryBackend = status["binary_backend"]?.stringValue ?? ""
```

Update the binary coarse line to include backend, and append a new line:

```swift
if !binaryANN.isEmpty {
    switch binaryANN {
    case "ready", "tail":
        lines.append("  二值 ANN: \(binaryANNIndex)")
    case "not_built":
        lines.append("  二值 ANN: 未构建（本次回退 bin 表暴力汉明）")
    case "building":
        lines.append("  二值 ANN: 构建中")
    case "needs_rebuild":
        lines.append("  二值 ANN: 建议重建（\(binaryANNNote)）")
    case "incomplete", "unusable":
        lines.append("  二值 ANN: 不可用（\(binaryANNNote.isEmpty ? "原因未知" : binaryANNNote)）")
    default:
        break
    }
}
```

Update the `ready` binary coarse line to `二值粗筛: 可用（\(binaryBackend)，...`.

- [ ] **Step 6: Update `RAGToolBinaryStatusTests`**

Add assertions for a ready snapshot:

```swift
status["binary_ann"] = .string("ready")
status["binary_ann_index"] = .string("二值 ANN 索引：4 块 / 40 切片")
status["binary_backend"] = .string("bann")
...
XCTAssertTrue(output.contains("二值 ANN: 二值 ANN 索引"))
XCTAssertTrue(output.contains("bann"))
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `swift test --filter RAGEngineBinaryANNTests`
Run: `swift test --filter RAGToolBinaryStatusTests`
Run: `swift test --filter RAGEngineANNTests`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add Sources/BaseAgentCore/RAG/RAGEngine.swift \
        Sources/BaseAgentCore/Tools/RAGTool.swift \
        Tests/BaseAgentCoreTests/RAGEngineBinaryANNTests.swift \
        Tests/BaseAgentCoreTests/RAGToolBinaryStatusTests.swift
git commit -m "feat(rag): select and expose binary ANN build target"
```

---

### Task 6: AppModel / SettingsView 接线

**Files:**
- Modify: `Sources/BaseAgentApp/AppModel.swift`
- Modify: `Sources/BaseAgentApp/SettingsView.swift`

- [ ] **Step 1: Add bann status text**

In `AppModel.swift` add:

```swift
@Published var binaryANNStatusText = "二值 ANN 索引未构建"
```

In `refreshRAGStatus`, read:

```swift
let binaryANN = status["binary_ann_index"]?.stringValue ?? ""
```

and assign:

```swift
if !binaryANN.isEmpty { self.binaryANNStatusText = binaryANN }
```

- [ ] **Step 2: Confirm and build against the resolved target**

In `requestANNIndexBuild`, replace:

```swift
let status = agent.ragEngine.annIndexStatus()
```

with:

```swift
let target = agent.ragEngine.resolvedANNIndexTarget()
let status = target == .binary
    ? agent.ragEngine.binaryANNIndexStatus()
    : agent.ragEngine.annIndexStatus()
```

Pass target to `performANNIndexBuild(agent:target:indexTarget:)`, use its summary in the confirm text and logs, and start status text with:

```swift
statusText = indexTarget == .binary ? "正在构建二值 ANN 索引…" : "正在构建 ANN 索引…"
```

- [ ] **Step 3: Show both index statuses in Settings**

In `SettingsView.swift`, after `annStatusLabel`, add:

```swift
@ViewBuilder
private var binaryANNStatusLabel: some View {
    HStack(spacing: 6) {
        Text("二值 ANN").font(.caption)
        Text(model.binaryANNStatusText)
            .font(.caption2)
            .foregroundStyle(.secondary)
            .lineLimit(2)
            .help("二值签名的分块 IVF 索引：stage 1 优先走 bann，未构建时回退 bin 表暴力汉明")
    }
}
```

Place it next to `annStatusLabel` in the vector store section.

- [ ] **Step 4: Update button copy**

Change `"构建 ANN 索引"` to `"构建/续建向量索引"` and help to:

```swift
.help("二值粗筛开启且签名覆盖完整时构建二值 ANN，否则构建主 ANN；"
    + "构建可中断，已写完的块保留")
```

Delete button help:

```swift
.help("删除主 ANN 与二值 ANN 索引，检索回退到精确扫描")
```

- [ ] **Step 5: Build and run app-related tests**

Run: `swift build`
Expected: BUILD SUCCEEDED.
Run: `swift test --filter RAGEngineBinaryANNTests`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Sources/BaseAgentApp/AppModel.swift Sources/BaseAgentApp/SettingsView.swift
git commit -m "feat(app): expose binary ANN status and target-aware build"
```

---

### Task 7: 回归、基准与文档

**Files:**
- Modify: `README.md`（更新测试用例数，若文件存在）
- Test: `Tests/BaseAgentCoreTests/BinaryRetrievalScaleBenchmarkTests.swift`（未跟踪，不提交）
- Test: `Tests/BaseAgentCoreTests/RealCorpusBinaryBenchmarkTests.swift`（未跟踪，不提交）

- [ ] **Step 1: Run targeted regression**

```bash
swift test --filter VectorStoreRerankTests
swift test --filter VectorStoreBinaryANNTests
swift test --filter VectorStoreBinarySearchTests
swift test --filter VectorStoreANNTests
swift test --filter RAGEngineBinaryANNTests
swift test --filter RAGEngineANNTests
swift test --filter RAGToolBinaryStatusTests
```

Expected: all PASS.

- [ ] **Step 2: Run full suite**

Run: `swift test`
Expected: all PASS. Record the total test count.

- [ ] **Step 3: Run synthetic benchmark**

```bash
BINARY_BENCH_ROWS=50000 BINARY_BENCH_MAIN_DIM=128 BINARY_BENCH_SIG_BITS=768 \
BINARY_BENCH_CLUSTERS=96 swift test -c release --filter BinaryRetrievalScaleBenchmarkTests
```

Then repeat with `BINARY_BENCH_ROWS=200000` and `BINARY_BENCH_ROWS=1000000`.
Expected:
- 1M stage 2 ≤ 10ms（200 候选）；
- 1M 二值总耗时 ≤ exact 的 50%；
- 50k/200k 二值总耗时 ≤ exact，stage 1 候选覆盖不低于 vec0 二值路径的 90%。

- [ ] **Step 4: Run real-corpus benchmark**

```bash
REAL_BENCH_KEEP=1 REAL_BENCH_CANDIDATES=200 \
REAL_BENCH_OUT=/tmp/real-binary-ann.txt \
swift test -c release --filter RealCorpusBinaryBenchmarkTests
cat /tmp/real-binary-ann.txt
```

Expected:
- coverage 1.0，`ann_bypassed=true`；
- bann 总耗时 ≤ 现有 vec0 二值路径（约 5.5ms）；
- true recall@10 下降 ≤ 3pp，recall@50 下降 ≤ 5pp。

- [ ] **Step 5: Update README test count**

If `README.md` contains a test count, update it to the new total from Step 2. Do not commit the untracked benchmark files or `benchmarks/`.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: update rag test baseline"
```

---

## Self-Review

**Spec coverage：**

- rerank 表 schema/meta/生命周期/回退 → Task 1、Task 2。
- bann schema/meta/构建/状态/尾块/陈旧/生命周期 → Task 3。
- bann 查询、簇心缓存、nprobe 派生、回退链、stage 2 保留 binaryScore → Task 4。
- 引擎构建目标、自动构建、删除、status、RAGTool → Task 5。
- AppModel/SettingsView 状态与文案 → Task 6。
- 回归与基准验收 → Task 7。

**类型一致性：**

- `VectorStore.rerankVectorTable` / `rerankStorageEnabled` / `rerankRows` 在 Task 1 定义，Task 2/4/5 使用一致。
- `buildBinaryANNIndex`、`binaryANNIndexStatus`、`dropBinaryANNIndex`、`bannCandidatesLocked` 在 Task 3/4 定义，Task 5 使用一致。
- `bannQueryCount` / `binaryVecQueryCount` / `stage2Backend` 在 Task 4 定义，Task 4/5 测试使用一致。
- `ANNIndexTarget` 在 Task 5 定义，Task 6 的 AppModel 使用 `.binary` / `.dense`。

**Placeholder scan：** 无 TBD/TODO；所有测试、命令、提交信息已给出。
