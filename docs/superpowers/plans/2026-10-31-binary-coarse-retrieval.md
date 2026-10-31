# 二值粗筛两段式检索 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 为 RAG 增加独立 bin 签名表与两阶段检索：先用 XOR + Popcount 汉明粗筛 top N，再用主向量余弦精排，并与 FTS5 BM25 做 RRF 融合。

**Architecture:** 在 embedding provider 层新增 `EmbeddingBinarySigning`，从原生输出直接生成固定位宽签名；`VectorStore` 新增独立 vec0 `bit[N]` 表 `chunk_vectors_bin`，rowid 与 `chunks` 一致，并用 meta 计数器维护覆盖率；`hybridSearch` 在二值可用时用 stage 1 汉明 KNN + stage 2 精确余弦取代 ANN 通道，不可用时自动回退原路径。开关只影响检索，入库始终维护签名（主类型 bin / provider 不支持 / sqlite-vec 不可用时除外）。

**Tech Stack:** Swift 6、SwiftPM、XCTest、SQLite、sqlite-vec 0.1.9（vec0 `bit[N]`）、FTS5、MLX / ONNX / llama.cpp embedding 后端。

---

## 文件结构

### 新增文件

| 文件 | 职责 |
| --- | --- |
| `Sources/BaseAgentCore/RAG/Storage/VectorStoreBinary.swift` | `BinaryIndexStatus`、`BinaryCoarseOptions`、bin 表建表/meta 对齐/状态/写入/删除计数/二值 KNN 候选/回退日志节流 |
| `Tests/BaseAgentCoreTests/BinarySignatureTests.swift` | 固定位宽签名编码、LSB-first、补零、汉明距离 |
| `Tests/BaseAgentCoreTests/EmbeddingBinarySigningTests.swift` | 协议默认实现、`SearchHit.binaryScore` |
| `Tests/BaseAgentCoreTests/ProviderBinarySigningTests.swift` | 非 v5 provider 的 conformance 与默认位宽 |
| `Tests/BaseAgentCoreTests/AdaptiveEmbeddingSigningTests.swift` | 调度器签名路径、原始下标回写、不支持时返回 nil |
| `Tests/BaseAgentCoreTests/VectorStoreBinaryTests.swift` | bin 表生命周期、写入/删除/重建、覆盖率、meta 对齐 |
| `Tests/BaseAgentCoreTests/VectorStoreBinarySearchTests.swift` | 两阶段检索、ANN 旁路、回退、定向检索忽略二值 |
| `Tests/BaseAgentCoreTests/BinaryCoarseConfigTests.swift` | 配置默认值、读写往返、50...1000 收口 |
| `Tests/BaseAgentCoreTests/RAGEngineBinaryTests.swift` | 引擎入库/查询/状态接线、回退分支 |
| `Tests/BaseAgentCoreTests/RAGToolBinaryStatusTests.swift` | `rag_status` 的二值粗筛输出 |

### 修改文件

| 文件 | 修改内容 |
| --- | --- |
| `Sources/BaseAgentCore/RAG/Storage/VectorElementType.swift` | `VectorEncoder.encodeSignature(_:bitDimension:)` |
| `Sources/BaseAgentCore/RAG/Models.swift` | `BinaryEmbeddingOutput`、`EmbeddingBinarySigning`、`SearchHit.binaryScore` |
| `Sources/BaseAgentCore/RAG/MatryoshkaEmbedding.swift` | `EmbeddingBinarySigning` 透传；`binarySourceIdentifier` 去掉 `#mrl` |
| `Sources/BaseAgentCore/RAG/AdaptiveEmbeddingScheduler.swift` | `binaryDimension`、`embedDocumentsWithSignatures`、单一 `runEmbedding` |
| `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift` | bin 表存储属性、init 参数、`add(binarySignatures:)`、删除/清空/重建、`hybridSearch` 两阶段、ANN 旁路、`binaryScore` |
| `Sources/BaseAgentCore/Local/JinaV5MLXEmbeddingProvider.swift` | 原生 768 维签名 |
| `Sources/BaseAgentCore/Local/JinaV5MLXModel.swift` | `encode` 同时返回截断向量与原生向量 |
| `Sources/BaseAgentCore/Local/ONNXEmbeddingProvider.swift` | `pool` 同时返回 vectors/nativeVectors；原生签名 |
| `Sources/BaseAgentCore/Local/MLXEmbeddingProvider.swift` | 声明 `EmbeddingBinarySigning` |
| `Sources/BaseAgentCore/Local/JinaV3MLXEmbeddingProvider.swift` | 声明 `EmbeddingBinarySigning` |
| `Sources/BaseAgentCore/Local/LlamaCppEmbeddingProvider.swift` | 声明 `EmbeddingBinarySigning` |
| `Sources/BaseAgentCore/Core/Config.swift` | `ragBinaryCoarseEnabled` / `ragBinaryCoarseCandidates` + apply/export/migrate |
| `Sources/BaseAgentCore/RAG/RAGEngine.swift` | 签名位宽/来源、入库签名、查询签名、status 字段 |
| `Sources/BaseAgentCore/Tools/RAGTool.swift` | `rag_status` 输出二值粗筛 |
| `Sources/BaseAgentApp/SettingsView.swift` | 二值粗筛开关与候选数 UI |
| `Tests/BaseAgentCoreTests/MatryoshkaEmbeddingTests.swift` | 签名透传、截断前签名、`#mrl` 不入签名来源 |
| `Tests/BaseAgentCoreTests/JinaV5MLXEmbeddingE2ETests.swift` | MLX 原生签名 E2E（模型缺失时跳过） |
| `Tests/BaseAgentCoreTests/JinaV5ONNXEmbeddingE2ETests.swift` | ONNX 原生签名 E2E（模型缺失时跳过） |
| `README.md` | 配置示例、配置窗口说明、测试计数 |

### 关键访问控制决定

`VectorStoreBinary.swift` 作为扩展文件时不能新增存储属性，也不能访问 `private` 成员。因此：

- `VectorStore.swift` 中新增的 `binaryDimension` / `binaryStorageEnabled` / `binarySource` / `binaryRows` / `lastBinaryFallbackReason` / `lastBinaryFallbackLog` / `annQueryCount` 必须声明在主类里；
- 上述属性与 `sqliteVecAvailable` 使用 **internal**（或 `public internal(set)`）访问级别；
- 二进制方法放在 `VectorStoreBinary.swift`，避免 `VectorStore.swift` 继续膨胀。

---

## 实施约定

- 每个 Task 都按 RED → GREEN → REFACTOR 顺序执行。
- 测试命令统一使用 `swift test --disable-sandbox --filter <TestClass>`；全量使用 `swift test --disable-sandbox`。
- 每个 Task 最后单独提交；提交信息使用 `feat(rag): ...` / `test(rag): ...` / `docs(rag): ...`。
- 不在计划中引入除 sqlite-vec 之外的新依赖。
- 旧库不回填 bin 表；覆盖率不足时只提示重新同步，不自动重建主库。

## Task 1: 固定位宽二值签名编码

**Files:**
- Create: `Tests/BaseAgentCoreTests/BinarySignatureTests.swift`
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorElementType.swift`

**设计：** 签名规则与现有 `encodeBits` 一致（`x > 0 → 1`，LSB-first），但位宽由参数固定：不足补零、超出截断、`bitDimension == 0` 返回空 `Data`。这保证查询侧与字段侧使用同一规则，且不会因为末字节多余位导致汉明距离抖动。

- [x] **Step 1: 写失败测试**

```swift
import XCTest
@testable import BaseAgentCore

final class BinarySignatureTests: XCTestCase {
    func testSignBitUsesStrictlyPositiveValues() {
        let vector: [Float] = [0.1, 0, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
        let data = VectorEncoder.encodeSignature(vector, bitDimension: 8)
        XCTAssertEqual(Array(data), [0b10101001])
    }

    func testLSBFirstMatchesExistingBitEncoder() {
        let vector: [Float] = (0..<16).map { index in
            index.isMultiple(of: 3) ? -Float(index) : Float(index + 1)
        }
        XCTAssertEqual(
            VectorEncoder.encodeSignature(vector, bitDimension: 16),
            VectorEncoder.encodeBits(vector))
    }

    func testNonMultipleOfEightPadsLastByte() {
        let data = VectorEncoder.encodeSignature(
            [0.1, -0.2, 0.3, -0.4, 0.5], bitDimension: 5)
        XCTAssertEqual(Array(data), [0b00010101])
    }

    func testPadsZeroBitsWhenBitDimensionExceedsVector() {
        let data = VectorEncoder.encodeSignature([0.1, -0.2, 0.3], bitDimension: 16)
        XCTAssertEqual(data.count, 2)
        XCTAssertEqual(Array(data), [0b00000101, 0])
    }

    func testHammingDistanceCountsDifferingSignBits() {
        let first = VectorEncoder.encodeSignature([0.1, -0.2, 0.3, -0.4], bitDimension: 4)
        let second = VectorEncoder.encodeSignature([-0.1, 0.2, -0.3, 0.4], bitDimension: 4)
        XCTAssertEqual(VectorEncoder.hamming(first, second), 4)
    }

    func testZeroBitDimensionProducesEmptySignature() {
        XCTAssertEqual(VectorEncoder.encodeSignature([1, 2], bitDimension: 0), Data())
    }
}
```

- [x] **Step 2: 运行测试确认 RED**

```bash
swift test --disable-sandbox --filter BinarySignatureTests
```

预期：编译失败，`VectorEncoder` 没有 `encodeSignature(_:bitDimension:)`。

- [x] **Step 3: 最小实现**

在 `VectorElementType.swift` 的 `VectorEncoder` 中，紧挨现有 `encodeBits` 之前加入：

```swift
    /// 固定位宽签名：x > 0 置 1，LSB-first；不足补零、超出截断。
    /// 查询签名与字段签名必须使用同一 `bitDimension`，否则末字节多余位
    /// 会进入汉明距离，造成与真实余弦排序不一致的抖动。
    public static func encodeSignature(_ vector: [Float], bitDimension: Int) -> Data {
        guard bitDimension > 0 else { return Data() }
        let byteCount = (bitDimension + 7) / 8
        var bytes = [UInt8](repeating: 0, count: byteCount)
        for (index, value) in vector.enumerated()
            where index < bitDimension && value > 0 {
            bytes[index / 8] |= UInt8(1 << (index % 8))
        }
        return Data(bytes)
    }
```

- [x] **Step 4: 运行测试确认 GREEN**

```bash
swift test --disable-sandbox --filter BinarySignatureTests
```

预期：6 个测试全部通过。

- [x] **Step 5: 提交**

```bash
git add Tests/BaseAgentCoreTests/BinarySignatureTests.swift \
        Sources/BaseAgentCore/RAG/Storage/VectorElementType.swift
git commit -m "feat(rag): add fixed-width binary signature encoder"
```

---

## Task 2: 模型层契约与 `SearchHit.binaryScore`

**Files:**
- Create: `Tests/BaseAgentCoreTests/EmbeddingBinarySigningTests.swift`
- Modify: `Sources/BaseAgentCore/RAG/Models.swift`

**设计：** 新增 `BinaryEmbeddingOutput` 与 `EmbeddingBinarySigning`。默认实现直接对 `EmbeddingProvider` 的向量做固定位宽签名，位宽向上取整到 8 的倍数。`SearchHit` 增加 `binaryScore`，默认 nil，保证现有调用点不受影响。

- [x] **Step 1: 写失败测试**

```swift
import XCTest
@testable import BaseAgentCore

private final class DefaultSigningEmbedding: EmbeddingProvider,
                                            EmbeddingBinarySigning,
                                            @unchecked Sendable {
    let identifier = "default-signing"
    let dimension = 10

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.enumerated().map { index, _ in
            (0..<dimension).map { offset in
                Float((index + offset) % 3 == 0 ? -1 : 1)
            }
        }
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        try await embedDocuments([text])[0]
    }
}

final class EmbeddingBinarySigningTests: XCTestCase {
    func testDefaultBinaryDimensionRoundsUpToByteBoundary() {
        let embedding = DefaultSigningEmbedding()
        XCTAssertEqual(embedding.binaryDimension, 16)
        XCTAssertEqual(embedding.binarySourceIdentifier, embedding.identifier)
    }

    func testDefaultDocumentSignaturesMatchFixedWidthEncoder() async throws {
        let embedding = DefaultSigningEmbedding()
        let output = try await embedding.embedDocumentsWithSignatures(["a", "b"])
        XCTAssertEqual(output.vectors.count, 2)
        XCTAssertEqual(output.signatures.count, 2)
        for (vector, signature) in zip(output.vectors, output.signatures) {
            XCTAssertEqual(signature.count, 2)
            XCTAssertEqual(signature,
                           VectorEncoder.encodeSignature(vector, bitDimension: 16))
        }
    }

    func testDefaultQuerySignatureMatchesDocumentEncoding() async throws {
        let embedding = DefaultSigningEmbedding()
        let query = try await embedding.embedQueryWithSignature("same text")
        XCTAssertEqual(query.vector.count, 10)
        XCTAssertEqual(query.signature,
                       VectorEncoder.encodeSignature(query.vector, bitDimension: 16))
    }

    func testSearchHitCarriesBinaryScoreWithoutBreakingDefaults() {
        let hit = SearchHit(rowID: 1, text: "x", source: "s", chunkIndex: 0,
                            binaryScore: 0.75)
        XCTAssertEqual(hit.binaryScore, 0.75)
        let legacy = SearchHit(rowID: 2, text: "y", source: "s", chunkIndex: 0)
        XCTAssertNil(legacy.binaryScore)
    }
}
```

- [x] **Step 2: 运行测试确认 RED**

```bash
swift test --disable-sandbox --filter EmbeddingBinarySigningTests
```

预期：编译失败，`EmbeddingBinarySigning` / `binaryScore` 未定义。

- [x] **Step 3: 实现**

在 `Models.swift` 的 `SearchHit` 中，`rrfScore` 之后加入：

```swift
    /// stage 1 二值粗筛的汉明相似度；未走二值粗筛时为 nil。
    public var binaryScore: Double?
```

初始化器参数末尾加入 `binaryScore: Double? = nil`，并在函数体中赋值：

```swift
                rrfScore: Double? = nil, binaryScore: Double? = nil) {
        ...
        self.rrfScore = rrfScore
        self.binaryScore = binaryScore
    }
```

在 `EmbeddingProvider` 之后加入：

```swift
/// 一次同时返回主向量与二值签名。签名必须来自原生输出（截断/量化之前）。
public struct BinaryEmbeddingOutput: Sendable {
    public var vectors: [[Float]]
    public var signatures: [Data]

    public init(vectors: [[Float]], signatures: [Data]) {
        self.vectors = vectors
        self.signatures = signatures
    }
}

/// 能提供二值签名的 embedding。
public protocol EmbeddingBinarySigning: AnyObject, Sendable {
    /// 签名位宽，必须是 8 的倍数。
    var binaryDimension: Int { get }
    /// 签名来源标识，用于 bin 表 meta 一致性校验。
    var binarySourceIdentifier: String { get }
    func embedDocumentsWithSignatures(_ texts: [String]) async throws -> BinaryEmbeddingOutput
    func embedQueryWithSignature(_ text: String) async throws
        -> (vector: [Float], signature: Data)
}

public extension EmbeddingBinarySigning where Self: EmbeddingProvider {
    var binaryDimension: Int { ((dimension + 7) / 8) * 8 }
    var binarySourceIdentifier: String { identifier }

    func embedDocumentsWithSignatures(_ texts: [String]) async throws -> BinaryEmbeddingOutput {
        let vectors = try await embedDocuments(texts)
        let bitDimension = binaryDimension
        return BinaryEmbeddingOutput(
            vectors: vectors,
            signatures: vectors.map {
                VectorEncoder.encodeSignature($0, bitDimension: bitDimension)
            })
    }

    func embedQueryWithSignature(_ text: String) async throws
        -> (vector: [Float], signature: Data) {
        let vector = try await embedQuery(text)
        return (vector,
                VectorEncoder.encodeSignature(vector, bitDimension: binaryDimension))
    }
}
```

- [x] **Step 4: 运行测试确认 GREEN**

```bash
swift test --disable-sandbox --filter EmbeddingBinarySigningTests
```

预期：4 个测试全部通过。

- [x] **Step 5: 提交**

```bash
git add Sources/BaseAgentCore/RAG/Models.swift \
        Tests/BaseAgentCoreTests/EmbeddingBinarySigningTests.swift
git commit -m "feat(rag): add EmbeddingBinarySigning contract and binary score"
```

## Task 3: 非 v5 provider 声明二值签名能力

**Files:**
- Create: `Tests/BaseAgentCoreTests/ProviderBinarySigningTests.swift`
- Modify: `Sources/BaseAgentCore/Local/MLXEmbeddingProvider.swift:10`
- Modify: `Sources/BaseAgentCore/Local/JinaV3MLXEmbeddingProvider.swift:7-8`
- Modify: `Sources/BaseAgentCore/Local/LlamaCppEmbeddingProvider.swift:5-7`
- Modify: `Sources/BaseAgentCore/Local/ONNXEmbeddingProvider.swift:8-9`

**设计：** 这四个 provider 的输出已经是原生维、归一化向量，直接使用 `EmbeddingBinarySigning` 的默认实现。空目录构造不会加载权重，因此可以在单元测试里验证 conformance 与默认位宽。

- [x] **Step 1: 写失败测试**

```swift
import XCTest
@testable import BaseAgentCore

final class ProviderBinarySigningTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("binary-provider-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testNonV5ProvidersAdvertiseDefaultBinaryDimension() {
        let mlx = MLXEmbeddingProvider(directory: directory, modelName: "mlx-stub")
        let jinaV3 = JinaV3MLXEmbeddingProvider(directory: directory,
                                                modelName: "jina-v3-stub")
        let onnx = ONNXEmbeddingProvider(directory: directory, modelName: "onnx-stub")
        let llama = LlamaCppEmbeddingProvider(modelPath: directory.path,
                                              modelName: "llama-stub")

        let providers: [(String, EmbeddingProvider, Int)] = [
            ("mlx", mlx, 768),
            ("jina-v3", jinaV3, 1_024),
            ("onnx", onnx, 768),
            ("llama", llama, 768),
        ]
        for (name, provider, expected) in providers {
            let signing = provider as? EmbeddingBinarySigning
            XCTAssertNotNil(signing, "\(name) 必须声明 EmbeddingBinarySigning")
            XCTAssertEqual(signing?.binaryDimension, expected, "\(name)")
            XCTAssertEqual(signing?.binarySourceIdentifier, provider.identifier, "\(name)")
            XCTAssertEqual(signing?.binaryDimension % 8, 0, "\(name)")
        }
    }
}
```

- [x] **Step 2: 运行测试确认 RED**

```bash
swift test --disable-sandbox --filter ProviderBinarySigningTests
```

预期：编译失败，四个 provider 还没有声明 `EmbeddingBinarySigning`。

- [x] **Step 3: 声明 conformance**

只修改类声明，不新增方法（默认实现已由 Task 2 提供）：

```swift
// MLXEmbeddingProvider.swift
public final class MLXEmbeddingProvider: EmbeddingProvider, EmbeddingTokenCounting,
                                        EmbeddingBinarySigning, @unchecked Sendable {

// JinaV3MLXEmbeddingProvider.swift
final class JinaV3MLXEmbeddingProvider: EmbeddingProvider,
                                       EmbeddingTokenCounting,
                                       EmbeddingBinarySigning,
                                       @unchecked Sendable {

// LlamaCppEmbeddingProvider.swift
public final class LlamaCppEmbeddingProvider: EmbeddingProvider,
                                             EmbeddingTokenCounting,
                                             EmbeddingBinarySigning,
                                             @unchecked Sendable {

// ONNXEmbeddingProvider.swift
public final class ONNXEmbeddingProvider: EmbeddingProvider, EmbeddingTokenCounting,
                                          EmbeddingBinarySigning,
                                          @unchecked Sendable {
```

`FailingEmbeddingProvider` / `ModelUnavailableEmbeddingProvider` 不声明 conformance。

- [x] **Step 4: 运行测试确认 GREEN**

```bash
swift test --disable-sandbox --filter ProviderBinarySigningTests
```

预期：1 个测试通过。

- [x] **Step 5: 提交**

```bash
git add Tests/BaseAgentCoreTests/ProviderBinarySigningTests.swift \
        Sources/BaseAgentCore/Local/MLXEmbeddingProvider.swift \
        Sources/BaseAgentCore/Local/JinaV3MLXEmbeddingProvider.swift \
        Sources/BaseAgentCore/Local/LlamaCppEmbeddingProvider.swift \
        Sources/BaseAgentCore/Local/ONNXEmbeddingProvider.swift
git commit -m "feat(rag): declare binary signing for native embedding providers"
```

---

## Task 4: jina-v5 从原生 768 维生成签名

**Files:**
- Modify: `Sources/BaseAgentCore/Local/JinaV5MLXModel.swift:255-303`
- Modify: `Sources/BaseAgentCore/Local/JinaV5MLXEmbeddingProvider.swift:8-58`
- Modify: `Sources/BaseAgentCore/Local/ONNXEmbeddingProvider.swift:140-287`
- Modify: `Tests/BaseAgentCoreTests/JinaV5MLXEmbeddingE2ETests.swift`
- Modify: `Tests/BaseAgentCoreTests/JinaV5ONNXEmbeddingE2ETests.swift`

**设计：** provider 内部先截断再归一化，主向量已经丢失截断前信息。必须让一次前向同时返回 `vectors`（截断后）与 `nativeVectors`（完整 768 维），签名只从 `nativeVectors` 生成，禁止第二次推理。

- [x] **Step 1: 追加失败 E2E 测试（MLX）**

在 `JinaV5MLXEmbeddingE2ETests` 中追加：

```swift
    func testSignaturesUseNative768DimensionNotTruncatedOutput() async throws {
        let directory = EmbeddingBenchmarkTests.resolveModelsRoot()
            .appendingPathComponent(Self.modelName)
        try XCTSkipUnless(
            JinaV5Profile.isJinaV5(directory: directory)
                && JinaV5Profile.weightsURL(in: directory) != nil,
            "模型缺失：\(directory.path)")

        let narrow = JinaV5MLXEmbeddingProvider(
            directory: directory, modelName: Self.modelName, dimension: 128)
        let wide = JinaV5MLXEmbeddingProvider(
            directory: directory, modelName: Self.modelName, dimension: 256)
        addTeardownBlock {
            await narrow.unload()
            await wide.unload()
        }

        XCTAssertEqual(narrow.binaryDimension, 768)
        XCTAssertEqual(narrow.binarySourceIdentifier, narrow.identifier)

        let texts = ["苹果公司发布了新款智能手机。", "机器学习通过数据训练模型。"]
        let narrowOutput = try await narrow.embedDocumentsWithSignatures(texts)
        let wideOutput = try await wide.embedDocumentsWithSignatures(texts)

        XCTAssertEqual(narrowOutput.vectors.map(\.count), [128, 128])
        XCTAssertEqual(wideOutput.vectors.map(\.count), [256, 256])
        XCTAssertEqual(narrowOutput.signatures.count, texts.count)
        XCTAssertEqual(narrowOutput.signatures.map(\.count), [96, 96],
                       "768 bit = 96 字节")
        XCTAssertEqual(narrowOutput.signatures, wideOutput.signatures,
                       "签名必须来自截断前的同一份原生输出")

        let query = try await narrow.embedQueryWithSignature("苹果发布了新手机")
        XCTAssertEqual(query.vector.count, 128)
        XCTAssertEqual(query.signature.count, 96)
    }
```

- [x] **Step 2: 追加失败 E2E 测试（ONNX）**

在 `JinaV5ONNXEmbeddingE2ETests` 中追加：

```swift
    func testOnnxSignaturesUseNative768Dimension() async throws {
        let directory = try modelDirectory()
        let narrow = ONNXEmbeddingProvider(
            directory: directory, modelName: Self.modelName, v5Dimension: 128)
        let wide = ONNXEmbeddingProvider(
            directory: directory, modelName: Self.modelName, v5Dimension: 256)
        addTeardownBlock {
            await narrow.unload()
            await wide.unload()
        }

        XCTAssertEqual(narrow.binaryDimension, 768)
        XCTAssertEqual(narrow.binarySourceIdentifier, narrow.identifier)

        let texts = ["苹果公司发布了新款智能手机。", "机器学习通过数据训练模型。"]
        let narrowOutput = try await narrow.embedDocumentsWithSignatures(texts)
        let wideOutput = try await wide.embedDocumentsWithSignatures(texts)
        XCTAssertEqual(narrowOutput.vectors.map(\.count), [128, 128])
        XCTAssertEqual(wideOutput.vectors.map(\.count), [256, 256])
        XCTAssertEqual(narrowOutput.signatures.map(\.count), [96, 96])
        XCTAssertEqual(narrowOutput.signatures, wideOutput.signatures,
                       "ONNX v5 也必须在截断前取签名")

        let query = try await narrow.embedQueryWithSignature("苹果发布了新手机")
        XCTAssertEqual(query.vector.count, 128)
        XCTAssertEqual(query.signature.count, 96)
    }
```

- [x] **Step 3: 运行测试确认 RED**

```bash
swift test --disable-sandbox --filter JinaV5MLXEmbeddingE2ETests
swift test --disable-sandbox --filter JinaV5ONNXEmbeddingE2ETests
```

预期：编译失败，provider 没有 `embedDocumentsWithSignatures` / `binaryDimension`；模型缺失时新用例跳过，但旧用例仍应可编译。

- [x] **Step 4: 改造 MLX 模型与前向**

在 `JinaV5MLXModel.swift` 增加结果类型：

```swift
struct JinaV5MLXEncodeResult: Sendable {
    let vectors: [[Float]]
    let nativeVectors: [[Float]]
}
```

把 `encode` 返回类型改为 `JinaV5MLXEncodeResult`，在截断前保存原生 pooled 行：

```swift
        var pooled = hidden[batchIndices, lengthIndices]
        // 二值签名只能从截断前的原生 768 维生成；归一化不改变符号。
        let nativeVectors = pooled.map { $0.asArray(Float.self) }

        let target = max(1, min(dimension, pooled.dim(-1)))
        if pooled.dim(-1) > target {
            pooled = pooled[.ellipsis, 0 ..< target]
        }
        let normalized = pooled.l2Normalized(axis: -1, eps: 1e-12)
        normalized.eval()
        return JinaV5MLXEncodeResult(
            vectors: normalized.map { $0.asArray(Float.self) },
            nativeVectors: nativeVectors)
```

- [x] **Step 5: 改造 MLX provider**

声明 conformance 并用一次前向同时返回两份结果：

```swift
final class JinaV5MLXEmbeddingProvider: EmbeddingProvider,
                                       EmbeddingTokenCounting,
                                       EmbeddingBinarySigning,
                                       @unchecked Sendable {
    var binaryDimension: Int { JinaV5Profile.nativeDimension }
    var binarySourceIdentifier: String { identifier }

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        try await embedDocumentsWithSignatures(texts).vectors
    }

    func embedDocumentsWithSignatures(_ texts: [String]) async throws
        -> BinaryEmbeddingOutput {
        guard !texts.isEmpty else {
            return BinaryEmbeddingOutput(vectors: [], signatures: [])
        }
        let prefixed = texts.map {
            prefixStrategy.apply($0, task: .retrievalDocument)
        }
        return try await embedWithSignatures(prefixed)
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        try await embedQueryWithSignature(text).vector
    }

    func embedQueryWithSignature(_ text: String) async throws
        -> (vector: [Float], signature: Data) {
        let output = try await embedWithSignatures([
            prefixStrategy.apply(text, task: .retrievalQuery),
        ])
        return (output.vectors.first ?? [], output.signatures.first ?? Data())
    }

    private func embedWithSignatures(_ texts: [String]) async throws
        -> BinaryEmbeddingOutput {
        let (model, tokenizer) = try await state.handles(directory: directory)
        let batch = tokenizer.encodeBatch(texts, maxLength: maxLength)
        let result = model.encode(inputIDs: batch.inputIDs,
                                  attentionMask: batch.attentionMask,
                                  dimension: dimension)
        return BinaryEmbeddingOutput(
            vectors: result.vectors,
            signatures: result.nativeVectors.map {
                VectorEncoder.encodeSignature($0, bitDimension: binaryDimension)
            })
    }
```

删除原有的 `private func embed(_:) -> [[Float]]`，避免保留第二次前向路径。

- [x] **Step 6: 改造 ONNX pooling 与 provider**

在 `ONNXEmbeddingProvider` 增加内部结果类型：

```swift
private struct PooledEmbeddings {
    let vectors: [[Float]]
    let nativeVectors: [[Float]]
}
```

`embed(_:task:)` 返回 `PooledEmbeddings`；`embedDocuments` / `embedQuery` 只取 `.vectors`。`pool` 的 v5 分支在截断前保存原行：

```swift
// [B,H] 分支
let rawRows = (0..<batch).map { row in
    Array(tensor.values[(row * hidden)..<((row + 1) * hidden)])
}
return PooledEmbeddings(
    vectors: rawRows.map {
        JinaV5Profile.truncateAndNormalize($0, dimension: v5OutputDimension)
    },
    nativeVectors: rawRows)

// [B,T,H] 分支
let nativeRow = Array(tensor.values[offset..<(offset + hidden)])
vectors.append(JinaV5Profile.truncateAndNormalize(
    nativeRow, dimension: v5OutputDimension))
nativeVectors.append(nativeRow)
```

非 v5 两个分支返回 `nativeVectors: vectors`。然后增加：

```swift
public var binaryDimension: Int {
    isJinaV5 ? JinaV5Profile.nativeDimension : ((dimension + 7) / 8) * 8
}
public var binarySourceIdentifier: String { identifier }

public func embedDocumentsWithSignatures(_ texts: [String]) async throws
    -> BinaryEmbeddingOutput {
    guard !texts.isEmpty else {
        return BinaryEmbeddingOutput(vectors: [], signatures: [])
    }
    let prefixed = texts.map { prefixStrategy.apply($0, task: .retrievalDocument) }
    let pooled = try await embed(prefixed, task: .retrievalDocument)
    return BinaryEmbeddingOutput(
        vectors: pooled.vectors,
        signatures: pooled.nativeVectors.map {
            VectorEncoder.encodeSignature($0, bitDimension: binaryDimension)
        })
}

public func embedQueryWithSignature(_ text: String) async throws
    -> (vector: [Float], signature: Data) {
    let pooled = try await embed([
        prefixStrategy.apply(text, task: .retrievalQuery)
    ], task: .retrievalQuery)
    return (pooled.vectors.first ?? [], pooled.nativeVectors.first.map {
        VectorEncoder.encodeSignature($0, bitDimension: binaryDimension)
    } ?? Data())
}
```

- [x] **Step 7: 运行测试确认 GREEN**

```bash
swift test --disable-sandbox --filter JinaV5MLXEmbeddingE2ETests
swift test --disable-sandbox --filter JinaV5ONNXEmbeddingE2ETests
swift test --disable-sandbox --filter ProviderBinarySigningTests
```

预期：有两份真实模型时新增签名用例通过；缺少模型时相应用例标记 skipped，编译与既有用例通过。

- [x] **Step 8: 提交**

```bash
git add Sources/BaseAgentCore/Local/JinaV5MLXModel.swift \
        Sources/BaseAgentCore/Local/JinaV5MLXEmbeddingProvider.swift \
        Sources/BaseAgentCore/Local/ONNXEmbeddingProvider.swift \
        Tests/BaseAgentCoreTests/JinaV5MLXEmbeddingE2ETests.swift \
        Tests/BaseAgentCoreTests/JinaV5ONNXEmbeddingE2ETests.swift
git commit -m "feat(rag): derive jina-v5 signatures from native 768d output"
```

---

## Task 5: Matryoshka 透传原生签名

**Files:**
- Modify: `Sources/BaseAgentCore/RAG/MatryoshkaEmbedding.swift:60-62`（新增协议扩展）
- Modify: `Tests/BaseAgentCoreTests/MatryoshkaEmbeddingTests.swift:4-24`（新增签名 stub）
- Test: `Tests/BaseAgentCoreTests/MatryoshkaEmbeddingTests.swift:94`（新增两个测试）

**设计：** 截断只影响主向量，签名必须来自底层原生输出。底层声明了
`EmbeddingBinarySigning` 时原样透传签名，只对 vectors 做截断 + 归一化；
底层不支持时，先取底层完整向量生成签名，再截断向量。`binarySourceIdentifier`
永远不带 `#mrl`，否则只改截断维就会让 bin 表 meta 被误判为不一致。

- [x] **Step 1: 写失败测试**

在 `MatryoshkaStubEmbedding` 之后加入一个显式声明签名的 stub：

```swift
private final class MatryoshkaSigningStub: EmbeddingProvider,
                                            EmbeddingBinarySigning,
                                            @unchecked Sendable {
    let identifier: String
    let dimension: Int
    let binaryDimension: Int
    let vector: [Float]
    let signature: Data

    init(identifier: String, dimension: Int, binaryDimension: Int,
         vector: [Float], signature: Data) {
        self.identifier = identifier
        self.dimension = dimension
        self.binaryDimension = binaryDimension
        self.vector = vector
        self.signature = signature
    }

    var binarySourceIdentifier: String { identifier }

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map { _ in vector }
    }

    func embedQuery(_ text: String) async throws -> [Float] { vector }

    func embedDocumentsWithSignatures(_ texts: [String]) async throws -> BinaryEmbeddingOutput {
        BinaryEmbeddingOutput(vectors: texts.map { _ in vector },
                              signatures: texts.map { _ in signature })
    }

    func embedQueryWithSignature(_ text: String) async throws
        -> (vector: [Float], signature: Data) {
        (vector, signature)
    }
}
```

在 `MatryoshkaEmbeddingTests` 末尾加入：

```swift
    /// 截断只作用于主向量：底层签名必须原样透传，不能从截断后的向量重算。
    func testSignaturesPassThroughWhileVectorsTruncate() async throws {
        let base = MatryoshkaSigningStub(
            identifier: "native-signing", dimension: 8, binaryDimension: 8,
            vector: [3, 4, 0, 0, -1, 1, -1, 1],
            signature: Data([0b0000_1111]))
        let wrapped = MatryoshkaEmbedding(base: base, truncateDimension: 2)

        let documents = try await wrapped.embedDocumentsWithSignatures(["a", "b"])
        XCTAssertEqual(documents.vectors.count, 2)
        XCTAssertEqual(documents.vectors[0], [0.6, 0.8])
        XCTAssertEqual(documents.signatures,
                       [Data([0b0000_1111]), Data([0b0000_1111])],
                       "签名应来自底层原生输出，而不是 [3, 4] 截断后的向量")
        XCTAssertEqual(wrapped.binaryDimension, 8)
        XCTAssertEqual(wrapped.binarySourceIdentifier, "native-signing")
        XCTAssertFalse(wrapped.binarySourceIdentifier.contains("#mrl"),
                       "签名来源不能带截断维后缀")

        let query = try await wrapped.embedQueryWithSignature("q")
        XCTAssertEqual(query.vector, [0.6, 0.8])
        XCTAssertEqual(query.signature, Data([0b0000_1111]))
    }

    /// 底层不会签名时，包装器必须从截断前的完整向量生成签名。
    func testWrapperSignsFullVectorWhenBaseCannotSign() async throws {
        let base = MatryoshkaStubEmbedding(
            identifier: "nomic-test", dimension: 8,
            source: [1, -1, 1, -1, 1, -1, 1, -1])
        let wrapped = MatryoshkaEmbedding(base: base, truncateDimension: 2)

        let documents = try await wrapped.embedDocumentsWithSignatures(["a", "b"])
        XCTAssertEqual(documents.vectors[0], [1, 0])
        XCTAssertEqual(documents.signatures,
                       [Data([0b01010101]), Data([0b01010101])],
                       "签名必须是完整 8 维的 0x55；从 [1, 0] 重算只会得到 0x01")
        XCTAssertEqual(wrapped.binaryDimension, 8)
        XCTAssertEqual(wrapped.binarySourceIdentifier, "nomic-test")

        let query = try await wrapped.embedQueryWithSignature("q")
        XCTAssertEqual(query.vector, [1, 0])
        XCTAssertEqual(query.signature, Data([0b01010101]))
    }
```

- [x] **Step 2: 运行测试确认 RED**

```bash
swift test --disable-sandbox --filter MatryoshkaEmbeddingTests
```

预期：编译失败，`MatryoshkaEmbedding` 还没有 `embedDocumentsWithSignatures` /
`binaryDimension`。

- [x] **Step 3: 最小实现**

在 `MatryoshkaEmbedding.swift` 末尾追加（`base` 是 private，扩展与类在
同一文件，因此可以直接访问）：

```swift
// MARK: - EmbeddingBinarySigning

extension MatryoshkaEmbedding: EmbeddingBinarySigning {
    /// 签名位宽与截断维无关。
    public var binaryDimension: Int {
        if let signing = base as? EmbeddingBinarySigning {
            return signing.binaryDimension
        }
        return ((base.dimension + 7) / 8) * 8
    }

    /// 签名来源不能带 `#mrl`：改截断维时 bin 表仍然有效。
    public var binarySourceIdentifier: String {
        (base as? EmbeddingBinarySigning)?.binarySourceIdentifier ?? base.identifier
    }

    public func embedDocumentsWithSignatures(_ texts: [String]) async throws
        -> BinaryEmbeddingOutput {
        if let signing = base as? EmbeddingBinarySigning {
            let output = try await signing.embedDocumentsWithSignatures(texts)
            return BinaryEmbeddingOutput(
                vectors: output.vectors.map {
                    Self.truncateAndNormalize($0, dimension: truncateDimension)
                },
                signatures: output.signatures)
        }

        let vectors = try await base.embedDocuments(texts)
        let bitDimension = binaryDimension
        return BinaryEmbeddingOutput(
            vectors: vectors.map {
                Self.truncateAndNormalize($0, dimension: truncateDimension)
            },
            signatures: vectors.map {
                VectorEncoder.encodeSignature($0, bitDimension: bitDimension)
            })
    }

    public func embedQueryWithSignature(_ text: String) async throws
        -> (vector: [Float], signature: Data) {
        if let signing = base as? EmbeddingBinarySigning {
            let output = try await signing.embedQueryWithSignature(text)
            return (Self.truncateAndNormalize(output.vector, dimension: truncateDimension),
                    output.signature)
        }

        let vector = try await base.embedQuery(text)
        return (Self.truncateAndNormalize(vector, dimension: truncateDimension),
                VectorEncoder.encodeSignature(vector, bitDimension: binaryDimension))
    }
}
```

- [x] **Step 4: 运行测试确认 GREEN**

```bash
swift test --disable-sandbox --filter MatryoshkaEmbeddingTests
```

预期：既有用例与新增 2 个用例全部通过。

- [x] **Step 5: 提交**

```bash
git add Sources/BaseAgentCore/RAG/MatryoshkaEmbedding.swift \
        Tests/BaseAgentCoreTests/MatryoshkaEmbeddingTests.swift
git commit -m "feat(rag): preserve native signatures through matryoshka wrapper"
```

---

## Task 6: 调度器签名调度

**Files:**
- Create: `Tests/BaseAgentCoreTests/AdaptiveEmbeddingSigningTests.swift`
- Modify: `Sources/BaseAgentCore/RAG/AdaptiveEmbeddingScheduler.swift:282-283`（新增 `binaryDimension`）
- Modify: `Sources/BaseAgentCore/RAG/AdaptiveEmbeddingScheduler.swift:329-393`（拆分 `embedDocuments` / 新增 `runEmbedding`）
- Modify: `Sources/BaseAgentCore/RAG/AdaptiveEmbeddingScheduler.swift:404-461`（`embedWindow` 返回 vectors + signatures）

**设计：** 不复制第二套调度算法。把现有 `embedDocuments` 主体提取为
`runEmbedding(includeSignatures:)`，签名与向量共用同一套 token 计数、全局
排序、窗口切分、自适应组批与原始下标回写。`embedDocuments` 仍走普通
`base.embedDocuments`（保持原行为）；只有 `embedDocumentsWithSignatures`
在底层支持二值时走签名接口，不支持时返回 nil。

- [x] **Step 1: 写失败测试**

创建 `Tests/BaseAgentCoreTests/AdaptiveEmbeddingSigningTests.swift`：

```swift
import XCTest
@testable import BaseAgentCore

private final class SigningSchedulerStub: EmbeddingProvider,
                                          EmbeddingTokenCounting,
                                          EmbeddingBinarySigning,
                                          @unchecked Sendable {
    let identifier = "signing-scheduler"
    let dimension = 8
    let maxContentTokens = 4_096
    var binaryDimension: Int { 8 }
    var binarySourceIdentifier: String { identifier }

    private let lock = NSLock()
    private(set) var plainBatches: [[String]] = []
    private(set) var signingBatches: [[String]] = []
    var breakSignatures = false

    func makeTokenCounter() async -> (@Sendable (String) -> Int)? {
        { $0.count }
    }

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        lock.lock()
        plainBatches.append(texts)
        lock.unlock()
        return texts.map(Self.vector(for:))
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        Self.vector(for: text)
    }

    func embedDocumentsWithSignatures(_ texts: [String]) async throws
        -> BinaryEmbeddingOutput {
        lock.lock()
        signingBatches.append(texts)
        lock.unlock()
        let signatures = breakSignatures
            ? []
            : texts.map { Data([UInt8(truncatingIfNeeded: $0.count)]) }
        return BinaryEmbeddingOutput(
            vectors: texts.map(Self.vector(for:)),
            signatures: signatures)
    }

    func embedQueryWithSignature(_ text: String) async throws
        -> (vector: [Float], signature: Data) {
        (Self.vector(for: text), Data([UInt8(truncatingIfNeeded: text.count)]))
    }

    /// 首维 = 文本长度，便于按原始下标反查。
    private static func vector(for text: String) -> [Float] {
        [Float(text.count)] + [Float](repeating: 0, count: 7)
    }
}

private final class PlainSchedulerStub: EmbeddingProvider, @unchecked Sendable {
    let identifier = "plain-scheduler"
    let dimension = 8

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map { _ in [Float](repeating: 0, count: 8) }
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        [Float](repeating: 0, count: 8)
    }
}

final class AdaptiveEmbeddingSigningTests: XCTestCase {
    private func makeScheduler(_ base: EmbeddingProvider) -> AdaptiveEmbeddingScheduler {
        AdaptiveEmbeddingScheduler(
            base: base,
            options: .init(minBatchSize: 1, maxBatchSize: 2,
                           initialBatchSize: 2, step: 1,
                           memoryLimitMB: 32_768, throughputTolerance: 0.5,
                           probeBatches: 16, probeSeconds: 30,
                           tokenBudgetMin: 4_096, tokenBudgetMax: 4_096,
                           initialTokenBudget: 4_096, tokenBudgetStep: 128,
                           windowTexts: 2),
            log: { _ in })
    }

    /// 长度降序后跨多个窗口，签名与向量必须按同一套原始下标回写。
    func testSignaturePathRestoresOriginalIndicesAcrossWindows() async throws {
        let base = SigningSchedulerStub()
        let scheduler = makeScheduler(base)
        let lengths = [100, 1, 90, 2, 80, 3]
        let texts = lengths.map { String(repeating: "a", count: $0) }
        var progress: [(Int, Int)] = []

        let output = try await scheduler.embedDocumentsWithSignatures(texts) { done, total in
            progress.append((done, total))
        }
        let result = try XCTUnwrap(output)
        XCTAssertEqual(result.vectors.compactMap { $0.first.map(Int.init) }, lengths,
                       "向量必须按原始下标写回")
        XCTAssertEqual(result.signatures.map { Int($0[0]) }, lengths,
                       "签名必须和向量使用同一套窗口/组批下标映射")
        XCTAssertEqual(progress.map(\.0), [2, 4, 6])
        XCTAssertTrue(progress.allSatisfy { $0.1 == texts.count })
        XCTAssertEqual(base.signingBatches.flatMap { $0 }.count, texts.count)
        XCTAssertTrue(base.plainBatches.isEmpty, "签名路径不应再调用普通 embedDocuments")
        XCTAssertEqual(scheduler.binaryDimension, 8)
    }

    func testPlainEmbeddingDoesNotRequestSignatures() async throws {
        let base = SigningSchedulerStub()
        let scheduler = makeScheduler(base)
        let vectors = try await scheduler.embedDocuments(["a", "bb", "ccc"])

        XCTAssertEqual(vectors.compactMap { $0.first.map(Int.init) }, [1, 2, 3])
        XCTAssertEqual(base.plainBatches.flatMap { $0 }.count, 3)
        XCTAssertTrue(base.signingBatches.isEmpty,
                      "普通入库路径不应额外触发签名，保持原行为")
    }

    func testUnsupportedBaseReturnsNilAndForwardsNilDimension() async throws {
        let scheduler = makeScheduler(PlainSchedulerStub())
        XCTAssertNil(scheduler.binaryDimension)
        let output = try await scheduler.embedDocumentsWithSignatures(["a", "b"])
        XCTAssertNil(output, "底层不支持时返回 nil，由调用方回退原流程")
    }

    func testSignatureCountMismatchThrowsInsteadOfWritingPartialData() async throws {
        let base = SigningSchedulerStub()
        base.breakSignatures = true
        let scheduler = makeScheduler(base)

        do {
            _ = try await scheduler.embedDocumentsWithSignatures(["a", "bb"])
            XCTFail("签名数量不符必须抛错")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("签名数量异常"),
                          error.localizedDescription)
        }
    }
}
```

- [x] **Step 2: 运行测试确认 RED**

```bash
swift test --disable-sandbox --filter AdaptiveEmbeddingSigningTests
```

预期：编译失败，`AdaptiveEmbeddingScheduler` 还没有
`embedDocumentsWithSignatures` / `binaryDimension`。

- [x] **Step 3: 新增 `binaryDimension`**

在 `public var dimension` 之后加入：

```swift
    /// 底层 provider 的二值签名位宽；nil 表示不支持。
    public var binaryDimension: Int? {
        guard let signing = base as? EmbeddingBinarySigning,
              signing.binaryDimension > 0 else { return nil }
        return signing.binaryDimension
    }
```

- [x] **Step 4: 拆分 `embedDocuments` 并新增 `runEmbedding`**

把现有 `public func embedDocuments(_ texts:progress:)`（从该函数声明到
`return result` 的结尾）整体替换为以下内容；前面转发用的
`embedDocuments(_:)` 保持不变：

```swift
    public func embedDocuments(
        _ texts: [String],
        progress: ((Int, Int) -> Void)?
    ) async throws -> [[Float]] {
        try await runEmbedding(texts, progress: progress,
                               includeSignatures: false).vectors
    }

    /// 入库用：底层支持二值时一次返回主向量与签名；不支持时返回 nil。
    public func embedDocumentsWithSignatures(
        _ texts: [String],
        progress: ((Int, Int) -> Void)? = nil
    ) async throws -> BinaryEmbeddingOutput? {
        guard binaryDimension != nil else { return nil }
        return try await runEmbedding(texts,
                                      progress: progress,
                                      includeSignatures: true)
    }

    /// 向量与签名的唯一调度路径。
    ///
    /// `includeSignatures == false` 时只走底层 `embedDocuments`，保持原有
    /// 行为；为 true 时走底层签名接口。两条路径共用同一套 token 计数、
    /// 全局排序、窗口切分与自适应组批，避免维护第二套调度算法。
    private func runEmbedding(
        _ texts: [String],
        progress: ((Int, Int) -> Void)?,
        includeSignatures: Bool
    ) async throws -> BinaryEmbeddingOutput {
        guard !texts.isEmpty else {
            return BinaryEmbeddingOutput(vectors: [], signatures: [])
        }

        // 全篇只创建一次计数器、只计数一次。分词器初始化开销大，按窗口
        // 重复创建会反复初始化；计数期间持有信号量，避免与 embedding
        // 调用并发使用同一个分词器。
        let tokenCounts = try await semaphore.withPermit { () -> [Int] in
            if Task.isCancelled { throw AgentError.cancelled }
            let counter: @Sendable (String) -> Int
            if let provider = base as? EmbeddingTokenCounting,
               let exact = await provider.makeTokenCounter() {
                counter = exact
            } else {
                counter = { EmbeddingTokenEstimator.count($0) }
            }
            return texts.map { max(1, counter($0)) }
        }
        // 大块优先：单条超预算时独占一批，小块随后紧凑填满。
        let order = texts.indices.sorted { lhs, rhs in
            tokenCounts[lhs] == tokenCounts[rhs]
                ? lhs < rhs
                : tokenCounts[lhs] > tokenCounts[rhs]
        }

        let window = max(1, options.windowTexts)
        var result = BinaryEmbeddingOutput(
            vectors: [[Float]](repeating: [], count: texts.count),
            signatures: includeSignatures
                ? [Data](repeating: Data(), count: texts.count)
                : [])
        var processed = 0
        var start = 0
        while start < order.count {
            if Task.isCancelled { throw AgentError.cancelled }
            let end = min(start + window, order.count)
            let slice = Array(order[start..<end])
            let completed = processed
            let output = try await embedWindow(
                slice,
                texts: texts,
                tokenCounts: tokenCounts,
                progress: { done, _ in progress?(completed + done, texts.count) },
                total: texts.count,
                includeSignatures: includeSignatures)
            for (offset, index) in slice.enumerated() {
                result.vectors[index] = output.vectors[offset]
                if includeSignatures {
                    result.signatures[index] = output.signatures[offset]
                }
            }
            processed += slice.count
            start = end
            await Task.yield()
        }
        return result
    }
```

- [x] **Step 5: 修改 `embedWindow` 同时返回签名**

把现有 `private func embedWindow(...)` 整体替换为：

```swift
    /// 单个窗口内的组批与执行。调用方负责按窗口切分并传入全篇 token
    /// 计数，这里只在窗口期间持有信号量，因此其他文件可以在窗口之间
    /// 获得模型时间。
    ///
    /// 窗口内的下标已按长度降序排列，贪心装入「条数上限 + token 预算」
    /// 双约束的子批；当前批装不下的切片跳过并留给下一批，继续用更短的
    /// 切片填满预算，减少 padding 浪费。向量与签名按窗口内位置写回，
    /// 调用方再映射回原始下标。
    private func embedWindow(
        _ indices: [Int],
        texts: [String],
        tokenCounts: [Int],
        progress: ((Int, Int) -> Void)?,
        total: Int,
        includeSignatures: Bool
    ) async throws -> BinaryEmbeddingOutput {
        try await semaphore.withPermit {
            var result = BinaryEmbeddingOutput(
                vectors: [[Float]](repeating: [], count: indices.count),
                signatures: includeSignatures
                    ? [Data](repeating: Data(), count: indices.count)
                    : [])
            // pending 使用窗口内位置，避免全局下标与局部下标的映射开销。
            var pending = Array(indices.indices)
            var processed = 0
            while !pending.isEmpty {
                if Task.isCancelled { throw AgentError.cancelled }
                stateLock.lock()
                let sizeCap = max(1, batchSize)
                let budget = max(1, tokenBudget)
                stateLock.unlock()

                var selected: [Int] = []
                selected.reserveCapacity(min(sizeCap, pending.count))
                var remaining: [Int] = []
                remaining.reserveCapacity(pending.count)
                var tokens = 0
                for position in pending {
                    let cost = tokenCounts[indices[position]]
                    guard selected.count < sizeCap else {
                        remaining.append(position)
                        continue
                    }
                    // 装不下就先跳过，继续用后面更短的切片填满预算；
                    // 单条超过预算时仍允许独占一批，否则会永远装不进去。
                    if selected.isEmpty || tokens + cost <= budget {
                        selected.append(position)
                        tokens += cost
                    } else {
                        remaining.append(position)
                    }
                }
                pending = remaining
                processed += selected.count
                let batch = selected.map { texts[indices[$0]] }
                let batchStarted = Date()
                let output: BinaryEmbeddingOutput
                if includeSignatures {
                    guard let signing = base as? EmbeddingBinarySigning else {
                        throw AgentError.storage("底层 embedding 不支持二值签名")
                    }
                    output = try await signing.embedDocumentsWithSignatures(batch)
                    guard output.signatures.count == batch.count else {
                        throw AgentError.storage(
                            "embedding 子批二值签名数量异常: "
                            + "\(output.signatures.count) != \(batch.count)")
                    }
                } else {
                    output = BinaryEmbeddingOutput(
                        vectors: try await base.embedDocuments(batch),
                        signatures: [])
                }
                let seconds = Date().timeIntervalSince(batchStarted)
                guard output.vectors.count == batch.count else {
                    throw AgentError.storage(
                        "embedding 子批返回数量异常: \(output.vectors.count) != \(batch.count)")
                }
                for (offset, position) in selected.enumerated() {
                    result.vectors[position] = output.vectors[offset]
                    if includeSignatures {
                        result.signatures[position] = output.signatures[offset]
                    }
                }
                record(texts: batch.count, tokens: tokens, seconds: seconds)
                progress?(processed, total)
            }
            return result
        }
    }
```

- [x] **Step 6: 运行测试确认 GREEN**

```bash
swift test --disable-sandbox --filter AdaptiveEmbeddingSigningTests
swift test --disable-sandbox --filter EmbeddingSchedulerTests
swift test --disable-sandbox --filter MatryoshkaEmbeddingTests
```

预期：新增 4 个用例通过；既有调度器用例与 Matryoshka 用例保持通过（普通
`embedDocuments` 行为未变）。

- [x] **Step 7: 提交**

```bash
git add Sources/BaseAgentCore/RAG/AdaptiveEmbeddingScheduler.swift \
        Tests/BaseAgentCoreTests/AdaptiveEmbeddingSigningTests.swift
git commit -m "feat(rag): schedule binary signatures alongside embedding vectors"
```

---

## Task 7: bin 表 schema、meta 与状态

**Files:**
- Create: `Sources/BaseAgentCore/RAG/Storage/VectorStoreBinary.swift`
- Create: `Tests/BaseAgentCoreTests/VectorStoreBinaryTests.swift`
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift:52`（`sqliteVecAvailable` 改为 internal）
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift:67-73`（新增 meta 键与存储属性）
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift:75-111`（init 参数与 setup 调用）
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift:823`（`readStoredCount` 改为 internal）

**设计：** bin 表是派生数据，打开库时按 meta 对齐：维度 / 来源 / 量化版本
任一不符、meta 缺失或表结构不符时直接 DROP 并重建空表，`binary_rows` 归零；
`binaryDimension == 0`、主类型 `.bin` 或 sqlite-vec 不可用时不创建也不主动
删除 bin 表。状态全部来自 O(1) 计数器，不执行 `COUNT(*)`。

- [x] **Step 1: 写失败测试**

创建 `Tests/BaseAgentCoreTests/VectorStoreBinaryTests.swift`：

```swift
import XCTest
@testable import BaseAgentCore

final class VectorStoreBinaryTests: XCTestCase {
    private var directory: URL!
    private let dimension = 16
    private let binaryDimension = 16

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-vector-bin-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func dbPath(_ name: String) -> String {
        directory.appendingPathComponent("\(name).sqlite").path
    }

    private func oneHot(_ index: Int) -> [Float] {
        var vector = [Float](repeating: 0, count: dimension)
        vector[index] = 1
        return vector
    }

    private func chunks(_ count: Int) -> [Chunk] {
        (0..<count).map { Chunk(text: "doc \($0)", source: "s\($0).md", chunkIndex: 0) }
    }

    /// 16 bit 签名固定 2 字节。
    private func signature(_ value: UInt8) -> Data {
        Data([value, value &+ 1])
    }

    private func binaryTableSQL(_ store: VectorStore) throws -> String? {
        try store.db.scalar(
            "SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
            [.text(VectorStore.binaryVectorTable)]).stringValue
    }

    private func meta(_ store: VectorStore, _ key: String) throws -> String? {
        try store.db.scalar("SELECT value FROM meta WHERE key = ?", [.text(key)]).stringValue
    }

    func testCreatesBinaryTableForSupportedConfiguration() throws {
        let store = try VectorStore(
            dbPath: dbPath("supported"), dimension: dimension,
            elementType: .int8, binaryDimension: binaryDimension,
            binarySource: "sig-native", indexFingerprint: "fp")
        let status = store.binaryIndexStatus()
        XCTAssertTrue(status.storageEnabled)
        XCTAssertEqual(status.dimension, binaryDimension)
        XCTAssertEqual(status.rows, 0)
        XCTAssertEqual(status.chunkRows, 0)
        XCTAssertEqual(status.coverage, 0)
        XCTAssertFalse(status.isComplete)

        let sql = try XCTUnwrap(binaryTableSQL(store))
        XCTAssertTrue(sql.contains("USING vec0"))
        XCTAssertTrue(sql.contains("bit[16]"))
        XCTAssertEqual(try meta(store, VectorStore.binaryDimensionKey), "16")
        XCTAssertEqual(try meta(store, VectorStore.binarySourceKey), "sig-native")
        XCTAssertEqual(try meta(store, VectorStore.binaryQuantizerVersionKey), "1")
        XCTAssertEqual(try meta(store, VectorStore.binaryRowsKey), "0")
        store.close()
    }

    func testDisablesBinaryStorageWithoutProviderOrSQLiteVec() throws {
        let noProvider = try VectorStore(
            dbPath: dbPath("no-provider"), dimension: dimension,
            elementType: .int8, binaryDimension: 0, binarySource: "",
            indexFingerprint: "no-provider")
        let noProviderStatus = noProvider.binaryIndexStatus()
        XCTAssertFalse(noProviderStatus.storageEnabled)
        XCTAssertTrue(noProviderStatus.reason?.contains("不支持") == true,
                      noProviderStatus.reason ?? "")
        XCTAssertNil(try binaryTableSQL(noProvider), "不支持时不应创建 bin 表")
        noProvider.close()

        let noVec = try VectorStore(
            dbPath: dbPath("no-vec"), dimension: dimension,
            enableSQLiteVec: false, elementType: .int8,
            binaryDimension: binaryDimension, binarySource: "sig",
            indexFingerprint: "no-vec")
        let noVecStatus = noVec.binaryIndexStatus()
        XCTAssertFalse(noVecStatus.storageEnabled)
        XCTAssertTrue(noVecStatus.reason?.contains("sqlite-vec") == true,
                      noVecStatus.reason ?? "")
        XCTAssertNil(try binaryTableSQL(noVec))
        noVec.close()

        let binPrimary = try VectorStore(
            dbPath: dbPath("bin-primary"), dimension: dimension,
            elementType: .bin, binaryDimension: binaryDimension,
            binarySource: "sig", indexFingerprint: "bin-primary")
        let binStatus = binPrimary.binaryIndexStatus()
        XCTAssertFalse(binStatus.storageEnabled)
        XCTAssertTrue(binStatus.reason?.contains("bin 主存储") == true,
                      binStatus.reason ?? "")
        XCTAssertNil(try binaryTableSQL(binPrimary))
        binPrimary.close()
    }

    /// 维度或来源变化时，bin 表必须 DROP 并重建为空表，不能复用旧签名。
    func testMetaMismatchDropsBinaryRowsAndRecreatesEmptyTable() throws {
        let path = dbPath("meta-mismatch")
        do {
            let store = try VectorStore(
                dbPath: path, dimension: dimension, elementType: .int8,
                binaryDimension: binaryDimension, binarySource: "source-a",
                indexFingerprint: "fp")
            _ = try store.add(chunks: chunks(2), vectors: [oneHot(0), oneHot(1)])
            try store.db.execute(
                "INSERT INTO \(VectorStore.binaryVectorTable)(rowid, embedding) "
                + "VALUES(1, vec_bit(?))", [.blob(signature(0x0F))])
            try store.db.execute(
                "UPDATE meta SET value = '1' WHERE key = ?",
                [.text(VectorStore.binaryRowsKey)])
            store.close()
        }

        let reopened = try VectorStore(
            dbPath: path, dimension: dimension, elementType: .int8,
            binaryDimension: 24, binarySource: "source-b",
            indexFingerprint: "fp")
        let status = reopened.binaryIndexStatus()
        XCTAssertTrue(status.storageEnabled)
        XCTAssertEqual(status.rows, 0)
        XCTAssertEqual(status.chunkRows, 2)
        XCTAssertEqual(status.coverage, 0)
        XCTAssertFalse(status.isComplete)
        let sql = try XCTUnwrap(binaryTableSQL(reopened))
        XCTAssertTrue(sql.contains("bit[24]"), sql)
        XCTAssertEqual(
            try reopened.db.scalar(
                "SELECT COUNT(*) FROM \(VectorStore.binaryVectorTable)").intValue, 0)
        XCTAssertEqual(try meta(reopened, VectorStore.binarySourceKey), "source-b")
        XCTAssertEqual(try meta(reopened, VectorStore.binaryDimensionKey), "24")
        reopened.close()
    }

    /// meta 一致但表被删掉时，重开应补建空表而不是报错。
    func testMissingBinaryTableIsRecreatedWhenMetaMatches() throws {
        let path = dbPath("missing-table")
        do {
            let store = try VectorStore(
                dbPath: path, dimension: dimension, elementType: .int8,
                binaryDimension: binaryDimension, binarySource: "sig",
                indexFingerprint: "fp")
            try store.db.execute("DROP TABLE \(VectorStore.binaryVectorTable)")
            store.close()
        }

        let reopened = try VectorStore(
            dbPath: path, dimension: dimension, elementType: .int8,
            binaryDimension: binaryDimension, binarySource: "sig",
            indexFingerprint: "fp")
        let status = reopened.binaryIndexStatus()
        XCTAssertTrue(status.storageEnabled)
        XCTAssertEqual(status.rows, 0)
        XCTAssertNotNil(try binaryTableSQL(reopened))
        reopened.close()
    }

    /// 覆盖率只读计数器：rows == chunkRows 才算完整。这里用 meta + 手写行
    /// 模拟 Task 8 写入后的状态，验证重开能恢复计数器。
    func testStatusUsesCountersAndBecomesCompleteWhenRowsMatchChunks() throws {
        let path = dbPath("coverage")
        do {
            let store = try VectorStore(
                dbPath: path, dimension: dimension, elementType: .int8,
                binaryDimension: binaryDimension, binarySource: "sig",
                indexFingerprint: "fp")
            _ = try store.add(chunks: chunks(2), vectors: [oneHot(0), oneHot(1)])
            let partial = store.binaryIndexStatus()
            XCTAssertEqual(partial.rows, 0)
            XCTAssertEqual(partial.chunkRows, 2)
            XCTAssertEqual(partial.coverage, 0)
            XCTAssertFalse(partial.isComplete)

            for rowID in [Int64(1), Int64(2)] {
                try store.db.execute(
                    "INSERT INTO \(VectorStore.binaryVectorTable)(rowid, embedding) "
                    + "VALUES(?, vec_bit(?))",
                    [.int(rowID), .blob(signature(UInt8(rowID)))])
            }
            try store.db.execute(
                "UPDATE meta SET value = '2' WHERE key = ?",
                [.text(VectorStore.binaryRowsKey)])
            store.close()
        }

        let reopened = try VectorStore(
            dbPath: path, dimension: dimension, elementType: .int8,
            binaryDimension: binaryDimension, binarySource: "sig",
            indexFingerprint: "fp")
        let complete = reopened.binaryIndexStatus()
        XCTAssertEqual(complete.rows, 2)
        XCTAssertEqual(complete.chunkRows, 2)
        XCTAssertEqual(complete.coverage, 1)
        XCTAssertTrue(complete.isComplete)
        reopened.close()
    }
}
```

- [x] **Step 2: 运行测试确认 RED**

```bash
swift test --disable-sandbox --filter VectorStoreBinaryTests
```

预期：编译失败，`VectorStore` 没有新的 init 参数 / `binaryIndexStatus()`，
`VectorStoreBinary` 类型不存在。

- [x] **Step 3: 扩展 `VectorStore.swift` 的存储属性与 init**

在静态表名之后加入：

```swift
    public static let binaryVectorTable = "chunk_vectors_bin"
```

在现有 meta key 定义之后加入：

```swift
    static let binaryRowsKey = "binary_rows"
    static let binaryDimensionKey = "binary_dimension"
    static let binarySourceKey = "binary_source"
    static let binaryQuantizerVersionKey = "binary_quantizer_version"
    static let binaryQuantizerVersion = 1
```

把 `private let sqliteVecAvailable: Bool` 改为：

```swift
    let sqliteVecAvailable: Bool
```

在 `annCentroidCache` 之后加入（扩展文件不能新增存储属性，必须声明在主类）：

```swift
    /// 二值签名位宽；0 表示 provider 不支持或主类型为 bin。
    let binaryDimension: Int
    /// 签名来源标识，用于 bin 表 meta 一致性校验。
    let binarySource: String
    /// bin 表是否可读写（sqlite-vec + 表存在 + 维度/来源有效）。
    var binaryStorageEnabled: Bool = false
    /// bin 表当前行数，O(1) 覆盖率计数器。
    var binaryRows: Int = 0
    /// 不可用原因（供 status 展示）。
    var binaryStorageReason: String?
    /// 二值查询实际走 stage 1 的次数；供测试验证 ANN 旁路。
    var annQueryCount: Int = 0
    /// 最近一次二值回退原因与日志时间（60 秒节流）。
    var lastBinaryFallbackReason: String?
    var lastBinaryFallbackLog: Date = .distantPast
```

把 init 签名改为：

```swift
    public init(dbPath: String, dimension: Int, enableSQLiteVec: Bool = true,
                elementType: VectorElementType = .fp32,
                binaryDimension: Int = 0, binarySource: String = "",
                indexFingerprint: String = "") throws {
```

在 `try elementType.validate(dimension: resolvedDimension)` 之后加入：

```swift
        let resolvedBinaryDimension = max(0, binaryDimension)
        guard resolvedBinaryDimension == 0 || resolvedBinaryDimension % 8 == 0 else {
            throw AgentError.storage(
                "二值签名维度 \(resolvedBinaryDimension) 必须是 8 的倍数")
        }
```

在 init 开头给新属性赋值：

```swift
        self.binaryDimension = resolvedBinaryDimension
        self.binarySource = binarySource
```

在 init 末尾（主向量表创建完成之后）加入：

```swift
        do {
            try setupBinaryIndexLocked()
        } catch {
            AgentLog.warning("二值索引初始化失败，已停用: \(error.localizedDescription)")
            binaryStorageEnabled = false
            binaryStorageReason = "初始化失败：\(error.localizedDescription)"
        }
```

把 `private static func readStoredCount` 改为 `static func readStoredCount`（扩展
文件需要复用它读取 meta 计数器）。

- [x] **Step 4: 创建 `VectorStoreBinary.swift`**

```swift
import Foundation

/// bin 表对外的只读状态快照。不触发任何查询，全部来自 O(1) 计数器。
public struct BinaryIndexStatus: Sendable {
    /// sqlite-vec + bin 表存在 + N > 0。
    public var storageEnabled: Bool
    public var dimension: Int
    public var rows: Int
    public var chunkRows: Int
    public var coverage: Double
    public var isComplete: Bool
    public var reason: String?

    public init(storageEnabled: Bool = false,
                dimension: Int = 0,
                rows: Int = 0,
                chunkRows: Int = 0,
                coverage: Double = 0,
                isComplete: Bool = false,
                reason: String? = nil) {
        self.storageEnabled = storageEnabled
        self.dimension = dimension
        self.rows = rows
        self.chunkRows = chunkRows
        self.coverage = coverage
        self.isComplete = isComplete
        self.reason = reason
    }
}

/// 二值粗筛的查询参数；存储可用性与覆盖率由 `VectorStore` 判定。
public struct BinaryCoarseOptions: Sendable {
    public var signature: Data
    public var candidateLimit: Int

    public init(signature: Data, candidateLimit: Int) {
        self.signature = signature
        self.candidateLimit = max(1, candidateLimit)
    }
}

extension VectorStore {
    // MARK: - 打开库时对齐

    /// 按当前参数对齐 bin 表与 meta。派生数据不一致时直接丢弃重建。
    /// 调用方：`VectorStore.init`（未持锁，单线程）。
    func setupBinaryIndexLocked() throws {
        binaryRows = 0
        binaryStorageEnabled = false
        binaryStorageReason = nil

        guard binaryDimension > 0 else {
            binaryStorageReason = "embedding 不支持二值签名"
            return
        }
        guard elementType != .bin else {
            binaryStorageReason = "bin 主存储仅支持单阶段检索"
            return
        }
        guard sqliteVecAvailable else {
            binaryStorageReason = "sqlite-vec 不可用"
            return
        }
        guard binaryDimension % 8 == 0 else {
            binaryStorageReason = "二值签名维度必须是 8 的倍数"
            return
        }
        guard !binarySource.isEmpty else {
            binaryStorageReason = "缺少二值签名来源"
            return
        }

        let existingSQL = try db.scalar(
            "SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
            [.text(Self.binaryVectorTable)]).stringValue
        let storedDimension = try db.scalar(
            "SELECT value FROM meta WHERE key = ?",
            [.text(Self.binaryDimensionKey)]).stringValue
        let storedSource = try db.scalar(
            "SELECT value FROM meta WHERE key = ?",
            [.text(Self.binarySourceKey)]).stringValue
        let storedVersion = try db.scalar(
            "SELECT value FROM meta WHERE key = ?",
            [.text(Self.binaryQuantizerVersionKey)]).stringValue
        let storedRows = Self.readStoredCount(db, key: Self.binaryRowsKey)

        let schemaMatches = existingSQL?.contains("USING vec0") == true
            && existingSQL?.contains("bit[\(binaryDimension)]") == true
        if existingSQL != nil, schemaMatches,
           storedDimension == String(binaryDimension),
           storedSource == binarySource,
           storedVersion == String(Self.binaryQuantizerVersion),
           let storedRows {
            binaryRows = max(0, storedRows)
            binaryStorageEnabled = true
            return
        }

        // 表缺失、表结构不符或 meta 任一不符：直接丢弃重建空表。
        try db.transaction {
            try db.execute("DROP TABLE IF EXISTS \(Self.binaryVectorTable)")
            try deleteBinaryMetaLocked()
            try db.execute("""
            CREATE VIRTUAL TABLE \(Self.binaryVectorTable)
            USING vec0(embedding bit[\(binaryDimension)])
            """)
            binaryRows = 0
            try writeBinaryMetaLocked()
        }
        binaryStorageEnabled = true
    }

    // MARK: - Meta

    func writeBinaryMetaLocked() throws {
        let values: [(String, String)] = [
            (Self.binaryDimensionKey, String(binaryDimension)),
            (Self.binarySourceKey, binarySource),
            (Self.binaryQuantizerVersionKey, String(Self.binaryQuantizerVersion)),
            (Self.binaryRowsKey, String(binaryRows)),
        ]
        for (key, value) in values {
            try db.execute("""
            INSERT INTO meta(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, [.text(key), .text(value)])
        }
    }

    func deleteBinaryMetaLocked() throws {
        try db.execute("DELETE FROM meta WHERE key IN (?, ?, ?, ?)", [
            .text(Self.binaryRowsKey),
            .text(Self.binaryDimensionKey),
            .text(Self.binarySourceKey),
            .text(Self.binaryQuantizerVersionKey),
        ])
    }

    // MARK: - 状态

    /// 只读快照；`rows` / `chunkRows` 均来自增量计数器。
    public func binaryIndexStatus() -> BinaryIndexStatus {
        lock.lock(); defer { lock.unlock() }
        return binaryIndexStatusLocked()
    }

    func binaryIndexStatusLocked() -> BinaryIndexStatus {
        let chunkRows = count
        let coverage = chunkRows > 0
            ? min(1.0, Double(binaryRows) / Double(chunkRows))
            : 0
        return BinaryIndexStatus(
            storageEnabled: binaryStorageEnabled,
            dimension: binaryDimension,
            rows: binaryRows,
            chunkRows: chunkRows,
            coverage: coverage,
            isComplete: binaryStorageEnabled && chunkRows > 0 && binaryRows == chunkRows,
            reason: binaryStorageReason)
    }
}
```

- [x] **Step 5: 运行测试确认 GREEN**

```bash
swift test --disable-sandbox --filter VectorStoreBinaryTests
swift test --disable-sandbox --filter VectorStoreElementTypeTests
```

预期：新增 5 个用例通过；既有元素类型用例不受影响。

- [x] **Step 6: 提交**

```bash
git add Sources/BaseAgentCore/RAG/Storage/VectorStore.swift \
        Sources/BaseAgentCore/RAG/Storage/VectorStoreBinary.swift \
        Tests/BaseAgentCoreTests/VectorStoreBinaryTests.swift
git commit -m "feat(rag): add binary vector table schema and status"
```

---

## Task 8: 写入、删除、清空与重建同步计数器

**Files:**
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift:324-349`（`adoptCurrentConfigurationLocked`）
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift:405-487`（`add` 拆分为带签名重载）
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift:490-508`（`rebuildDimensionLocked`）
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift:551-573`（删除 bin 行）
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift:690-772`（`clear` / `rebuild`）
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift:804-812`（回滚双计数器）
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStoreBinary.swift`（插入/持久化 bin 计数器）
- Test: `Tests/BaseAgentCoreTests/VectorStoreBinaryTests.swift`

**设计：** 主向量与 bin 行在同一事务内写入、删除、清空与重建；`binary_rows`
只按实际写入/删除行数增量变化。签名缺失时允许只写主向量，覆盖率自然下降；
签名数量或字节数不符时整个事务回滚。事务失败时 `cachedChunkCount` 与
`binaryRows` 一起恢复。

- [x] **Step 1: 写失败测试**

在 `VectorStoreBinaryTests` 末尾加入：

```swift
    func testAddWritesBinaryRowsWithSameRowIDsAndPersistsCoverage() throws {
        let path = dbPath("write")
        let store = try VectorStore(
            dbPath: path, dimension: dimension, elementType: .int8,
            binaryDimension: binaryDimension, binarySource: "sig",
            indexFingerprint: "fp")
        let input = chunks(3)
        let signatures = [UInt8(1), 2, 3].map(signature)
        XCTAssertEqual(
            try store.add(chunks: input,
                          vectors: [oneHot(0), oneHot(1), oneHot(2)],
                          binarySignatures: signatures), 3)

        let status = store.binaryIndexStatus()
        XCTAssertEqual(status.rows, 3)
        XCTAssertEqual(status.chunkRows, 3)
        XCTAssertEqual(status.coverage, 1)
        XCTAssertTrue(status.isComplete)

        let chunkIDs = try store.db.query("SELECT rowid FROM chunks ORDER BY rowid")
            .compactMap { $0["rowid"]?.intValue }
        let binIDs = try store.db.query(
            "SELECT rowid FROM \(VectorStore.binaryVectorTable) ORDER BY rowid")
            .compactMap { $0["rowid"]?.intValue }
        XCTAssertEqual(binIDs, chunkIDs, "bin 行 rowid 必须与 chunks 严格一致")
        XCTAssertEqual(try meta(store, VectorStore.binaryRowsKey), "3")
        store.close()

        let reopened = try VectorStore(
            dbPath: path, dimension: dimension, elementType: .int8,
            binaryDimension: binaryDimension, binarySource: "sig",
            indexFingerprint: "fp")
        XCTAssertTrue(reopened.binaryIndexStatus().isComplete,
                      "覆盖率计数器必须跨重开保持")
        reopened.close()
    }

    /// 没有签名时主向量照常写入，但覆盖率下降；补齐签名后同一 source
    /// 会被 upsert 覆盖，覆盖率恢复完整。
    func testAddWithoutSignaturesKeepsMainVectorsAndCoverageFalls() throws {
        let store = try VectorStore(
            dbPath: dbPath("partial"), dimension: dimension, elementType: .int8,
            binaryDimension: binaryDimension, binarySource: "sig",
            indexFingerprint: "fp")
        XCTAssertEqual(
            try store.add(chunks: chunks(2), vectors: [oneHot(0), oneHot(1)]), 2)
        let partial = store.binaryIndexStatus()
        XCTAssertEqual(partial.rows, 0)
        XCTAssertEqual(partial.chunkRows, 2)
        XCTAssertEqual(partial.coverage, 0)
        XCTAssertFalse(partial.isComplete)
        XCTAssertFalse(try store.search(vector: oneHot(0), topK: 1).isEmpty,
                       "无签名不应影响主向量写入与检索")

        XCTAssertEqual(
            try store.add(chunks: chunks(2), vectors: [oneHot(0), oneHot(1)],
                          binarySignatures: [signature(1), signature(2)]), 2)
        XCTAssertTrue(store.binaryIndexStatus().isComplete, "重同步后覆盖率应恢复")
        store.close()
    }

    func testSignatureCountMismatchRollsBackMainAndBinaryRows() throws {
        let store = try VectorStore(
            dbPath: dbPath("count-mismatch"), dimension: dimension, elementType: .int8,
            binaryDimension: binaryDimension, binarySource: "sig",
            indexFingerprint: "fp")
        _ = try store.add(
            chunks: [Chunk(text: "keep", source: "keep.md", chunkIndex: 0)],
            vectors: [oneHot(0)], binarySignatures: [signature(1)])
        XCTAssertEqual(store.count, 1)
        XCTAssertEqual(store.binaryIndexStatus().rows, 1)

        XCTAssertThrowsError(
            try store.add(chunks: chunks(2), vectors: [oneHot(1), oneHot(2)],
                          binarySignatures: [signature(2)])
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("签名数量"),
                          error.localizedDescription)
        }
        XCTAssertEqual(store.count, 1, "失败必须回滚主向量")
        XCTAssertEqual(store.binaryIndexStatus().rows, 1, "失败必须回滚 bin 计数")
        XCTAssertEqual(try store.db.query("SELECT text FROM chunks").count, 1)
        store.close()
    }

    func testSignatureLengthMismatchRollsBackWholeTransaction() throws {
        let store = try VectorStore(
            dbPath: dbPath("length-mismatch"), dimension: dimension, elementType: .int8,
            binaryDimension: binaryDimension, binarySource: "sig",
            indexFingerprint: "fp")
        XCTAssertThrowsError(
            try store.add(chunks: chunks(2), vectors: [oneHot(0), oneHot(1)],
                          binarySignatures: [Data([0x01]), signature(2)])
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("字节数"),
                          error.localizedDescription)
        }
        XCTAssertEqual(store.count, 0)
        XCTAssertEqual(store.binaryIndexStatus().rows, 0)
        XCTAssertEqual(
            try store.db.scalar(
                "SELECT COUNT(*) FROM \(VectorStore.binaryVectorTable)").intValue, 0)
        store.close()
    }

    func testDeleteClearAndRebuildKeepBinaryCounterInSync() throws {
        let store = try VectorStore(
            dbPath: dbPath("lifecycle"), dimension: dimension, elementType: .int8,
            binaryDimension: binaryDimension, binarySource: "sig",
            indexFingerprint: "fp")
        _ = try store.add(chunks: chunks(2), vectors: [oneHot(0), oneHot(1)],
                          binarySignatures: [signature(1), signature(2)])
        let ids = try store.db.query("SELECT rowid FROM chunks ORDER BY rowid")
            .compactMap { $0["rowid"]?.intValue }
        XCTAssertEqual(try store.deleteRowIDs([ids[0]]), 1)

        let afterDelete = store.binaryIndexStatus()
        XCTAssertEqual(afterDelete.rows, 1)
        XCTAssertEqual(afterDelete.chunkRows, 1)
        XCTAssertTrue(afterDelete.isComplete)
        XCTAssertEqual(
            try store.db.scalar(
                "SELECT COUNT(*) FROM \(VectorStore.binaryVectorTable)").intValue, 1)

        _ = try store.add(
            chunks: [Chunk(text: "new", source: "new.md", chunkIndex: 0)],
            vectors: [oneHot(2)], binarySignatures: [signature(3)])
        XCTAssertEqual(store.binaryIndexStatus().rows, 2)

        try store.clear()
        let cleared = store.binaryIndexStatus()
        XCTAssertEqual(cleared.rows, 0)
        XCTAssertEqual(cleared.chunkRows, 0)
        XCTAssertEqual(cleared.coverage, 0)
        XCTAssertEqual(
            try store.db.scalar(
                "SELECT COUNT(*) FROM \(VectorStore.binaryVectorTable)").intValue, 0)

        _ = try store.add(chunks: chunks(1), vectors: [oneHot(0)],
                          binarySignatures: [signature(1)])
        XCTAssertEqual(store.binaryIndexStatus().rows, 1)
        _ = try store.rebuildForCurrentConfiguration()
        let rebuilt = store.binaryIndexStatus()
        XCTAssertEqual(rebuilt.rows, 0)
        XCTAssertEqual(rebuilt.chunkRows, 0)
        XCTAssertEqual(
            try store.db.scalar(
                "SELECT COUNT(*) FROM \(VectorStore.binaryVectorTable)").intValue, 0)
        store.close()
    }

    /// 空库按真实维度重建主表时，残留的 bin 行也必须一并清掉。
    func testDimensionRebuildClearsStaleBinaryRows() throws {
        let store = try VectorStore(
            dbPath: dbPath("dimension-rebuild"), dimension: dimension,
            elementType: .int8, binaryDimension: binaryDimension,
            binarySource: "sig", indexFingerprint: "fp")
        try store.db.execute(
            "INSERT INTO \(VectorStore.binaryVectorTable)(rowid, embedding) "
            + "VALUES(1, vec_bit(?))", [.blob(signature(1))])
        try store.db.execute(
            "UPDATE meta SET value = '1' WHERE key = ?",
            [.text(VectorStore.binaryRowsKey)])

        let eightDimensions = [Float](repeating: 0, count: 8)
        _ = try store.add(
            chunks: [Chunk(text: "new", source: "new.md", chunkIndex: 0)],
            vectors: [eightDimensions])

        XCTAssertEqual(store.dimension, 8)
        XCTAssertEqual(store.binaryIndexStatus().rows, 0)
        XCTAssertEqual(
            try store.db.scalar(
                "SELECT COUNT(*) FROM \(VectorStore.binaryVectorTable)").intValue, 0)
        store.close()
    }
```

- [x] **Step 2: 运行测试确认 RED**

```bash
swift test --disable-sandbox --filter VectorStoreBinaryTests
```

预期：编译失败，`add(chunks:vectors:binarySignatures:)` 不存在；其余新增用例
在编译通过后也会因计数未同步而失败。

- [x] **Step 3: 在 `VectorStoreBinary.swift` 增加写入/计数辅助方法**

在 `writeBinaryMetaLocked` 之前加入：

```swift
    /// 为指定 rowid 写入一条已校验长度的二值签名。
    func insertBinarySignature(rowID: Int64, signature: Data) throws {
        try db.execute(
            "INSERT INTO \(Self.binaryVectorTable)(rowid, embedding) "
            + "VALUES(?, vec_bit(?))",
            [.int(rowID), .blob(signature)])
    }

    /// 增量持久化 bin 行数；必须与所在事务一起提交/回滚。
    func persistBinaryCounterLocked() throws {
        guard binaryStorageEnabled else { return }
        try db.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            + "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [.text(Self.binaryRowsKey), .text(String(binaryRows))])
    }
```

- [x] **Step 4: 改造 `add` 写入 bin 行**

把现有 `add(chunks:vectors:)` 整体替换为：

```swift
    @discardableResult
    public func add(chunks: [Chunk], vectors: [[Float]]) throws -> Int {
        try add(chunks: chunks, vectors: vectors, binarySignatures: nil)
    }

    /// 批量写入（按 source upsert）。签名非 nil 时，主向量与 bin 行在同一
    /// 事务内写入，rowid 严格一致；签名缺失时只写主向量，覆盖率随之下降。
    @discardableResult
    public func add(chunks: [Chunk], vectors: [[Float]],
                    binarySignatures: [Data]?) throws -> Int {
        guard !chunks.isEmpty else { return 0 }
        guard chunks.count == vectors.count else {
            throw AgentError.storage(
                "chunks 与 vectors 数量不一致: \(chunks.count) vs \(vectors.count)")
        }
        if let binarySignatures, binarySignatures.count != chunks.count {
            throw AgentError.storage(
                "二值签名数量与切片数量不一致: "
                + "\(binarySignatures.count) vs \(chunks.count)")
        }
        // FTS 预分词是纯 CPU 工作（实测约占单切片写入九成），放在写锁外批量并行完成：
        // 写事务只做插入，检索取向量候选时不会被整批分词阻塞。
        let ftsTexts = TextAnalyzer.ftsIndexTexts(chunks.map(\.text))
        lock.lock(); defer { lock.unlock() }
        if let rebuildReason {
            throw AgentError.storage(
                "向量库需要重建（\(rebuildReason)）；请先在设置中执行「重建向量库」")
        }
        // 先确定计数基线：旧库没有 meta 计数器时用 MAX(rowid) 兜底。
        ensureChunkCounterLocked()

        // 模型未声明维度时，首次推理才发现真实值：空库允许直接重建；
        // 已有数据的库必须报错，避免静默清空用户的知识库。
        if let first = vectors.first, first.count != dimension {
            guard !hasChunks else {
                throw AgentError.storage(
                    "向量维度 \(first.count) != \(dimension)；请重建向量库后再同步")
            }
            try elementType.validate(dimension: first.count)
            AgentLog.warning("VectorStore: 按实际输出维度 \(first.count) 重建空库（原 \(dimension)）")
            try rebuildDimensionLocked(first.count)
        }

        if let binarySignatures, binaryStorageEnabled {
            let expected = binaryDimension / 8
            for signature in binarySignatures where signature.count != expected {
                throw AgentError.storage(
                    "二值签名字节数 \(signature.count) != \(expected)")
            }
        }

        let uniqueSources = Set(chunks.map(\.source))
        var inserted = 0
        var insertedBinary = 0
        try withChunkCounterRollback {
            try db.transaction {
                // 1) 删除这些 source 的旧数据（分批，事务内）。
                for source in uniqueSources {
                    _ = try deleteBySourceLocked(source)
                }
                // 2) 分批插入正文 + FTS + 向量（以及可选 bin 行）。
                var offset = 0
                while offset < chunks.count {
                    let end = min(offset + insertBatchSize, chunks.count)
                    for i in offset..<end {
                        let chunk = chunks[i]
                        let vector = VectorCodec.normalize(vectors[i])
                        guard vector.count == dimension else {
                            throw AgentError.storage("向量维度 \(vector.count) != \(dimension)")
                        }
                        let metadataJSON = JSONValue.object(chunk.metadata.mapValues { .string($0) }).jsonString()
                        let returned = try db.query("""
                        INSERT INTO chunks(id, source, source_hash, chunk_index, text, heading, metadata, created_at)
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?) RETURNING rowid
                        """, [
                            .text(chunk.id), .text(chunk.source),
                            .text(Self.hashSource(chunk.source)),
                            .int(Int64(chunk.chunkIndex)), .text(chunk.text),
                            .text(chunk.heading), .text(metadataJSON),
                            .double(Date().timeIntervalSince1970),
                        ])
                        guard let rowID = returned.first?["rowid"]?.intValue else {
                            throw AgentError.storage("写入 chunks 后未返回 rowid: \(chunk.id)")
                        }
                        try db.execute(
                            "INSERT INTO chunks_fts(rowid, tokens, source_hash, chunk_index) VALUES(?, ?, ?, ?)",
                            [.int(rowID), .text(ftsTexts[i]),
                             .text(Self.hashSource(chunk.source)), .int(Int64(chunk.chunkIndex))]
                        )
                        try insertVector(rowID: rowID, vector: vector)
                        if binaryStorageEnabled, let binarySignatures {
                            try insertBinarySignature(rowID: rowID,
                                                      signature: binarySignatures[i])
                            insertedBinary += 1
                        }
                        inserted += 1
                    }
                    offset = end
                }
                // 删除阶段已扣减，这里只补新增，随事务一起提交。
                if let cachedChunkCount {
                    self.cachedChunkCount = cachedChunkCount + inserted
                    try persistChunkCounterLocked()
                }
                if binaryStorageEnabled {
                    binaryRows += insertedBinary
                    try persistBinaryCounterLocked()
                }
            }
        }
        AgentLog.info("VectorStore: 写入 \(inserted) 个切片"
            + (insertedBinary > 0 ? "、\(insertedBinary) 条二值签名" : "")
            + "（\(uniqueSources.count) 个来源，单事务）")
        return inserted
    }
```

- [x] **Step 5: 删除与回滚同步 `binaryRows`**

在 `deleteRowIDsLocked` 的 `DELETE FROM chunk_vectors` 之后加入：

```swift
            if binaryStorageEnabled {
                let removedBinary = try db.execute(
                    "DELETE FROM \(Self.binaryVectorTable) WHERE rowid IN (\(placeholders))",
                    args)
                binaryRows = max(0, binaryRows - removedBinary)
            }
```

在该函数末尾（`cachedChunkCount` 持久化之后）加入：

```swift
        try persistBinaryCounterLocked()
```

把 `withChunkCounterRollback` 改为：

```swift
    private func withChunkCounterRollback<T>(_ body: () throws -> T) rethrows -> T {
        let chunksBefore = cachedChunkCount
        let binaryBefore = binaryRows
        do {
            return try body()
        } catch {
            cachedChunkCount = chunksBefore
            binaryRows = binaryBefore
            throw error
        }
    }
```

- [x] **Step 6: 清空、显式重建与空库采纳**

在 `adoptCurrentConfigurationLocked` 的 `try? resetANNIndexLocked()` 之前加入：

```swift
            if binaryStorageEnabled {
                try db.execute("DROP TABLE IF EXISTS \(Self.binaryVectorTable)")
                try deleteBinaryMetaLocked()
                binaryRows = 0
            }
            binaryStorageEnabled = false
```

在 `clear()` 的 `try? resetANNIndexLocked()` 之前加入：

```swift
                if binaryStorageEnabled {
                    try db.execute("DELETE FROM \(Self.binaryVectorTable)")
                    binaryRows = 0
                    try persistBinaryCounterLocked()
                }
```

在 `rebuildForCurrentConfiguration()` 的 `try? resetANNIndexLocked()` 之前加入：

```swift
                if binaryStorageEnabled {
                    try db.execute("DELETE FROM \(Self.binaryVectorTable)")
                    binaryRows = 0
                    try persistBinaryCounterLocked()
                }
```

在 `rebuildDimensionLocked` 的 `try ensureVectorTable(...)` 之后加入：

```swift
            if binaryStorageEnabled {
                try db.execute("DELETE FROM \(Self.binaryVectorTable)")
                binaryRows = 0
                try persistBinaryCounterLocked()
            }
```

- [x] **Step 7: 运行测试确认 GREEN**

```bash
swift test --disable-sandbox --filter VectorStoreBinaryTests
swift test --disable-sandbox --filter VectorStoreTests
swift test --disable-sandbox --filter VectorStoreElementTypeTests
```

预期：新增 6 个用例通过；既有 VectorStore 用例保持通过。

- [x] **Step 8: 提交**

```bash
git add Sources/BaseAgentCore/RAG/Storage/VectorStore.swift \
        Sources/BaseAgentCore/RAG/Storage/VectorStoreBinary.swift \
        Tests/BaseAgentCoreTests/VectorStoreBinaryTests.swift
git commit -m "feat(rag): maintain binary rows in vector store lifecycle"
```

## Task 9: `hybridSearch` 两阶段检索与 ANN 旁路

**Files:**
- Create: `Tests/BaseAgentCoreTests/VectorStoreBinarySearchTests.swift`
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift:59`（`maxVectorK` 改为 internal）
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift:1313-1415`（`hybridSearch`）
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStoreANN.swift:757-758`（`annQueryCount` 探针）
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStoreBinary.swift`（二值两阶段候选 + 回退日志节流）

**设计：** 二值可用时，`hybridSearch` 先在写入锁内执行 stage 1（vec0 `bit[N]`
汉明 KNN，取 `k = min(4096, max(candidateLimit, candidateCount))`），再用
`exactVectorsLocked` 读取主向量做 stage 2 余弦精排，最终向量候选交给原有
RRF/FTS/hydrate 流程；整个过程不调用 ANN。任一条不可用（关闭、覆盖率不足、
维度/长度不符、定向检索、待重建、主类型 bin）都回退现有
`vectorCandidatesLocked`，并按原因 60 秒节流记录一次日志。FTS 仍按
`candidateCount` 全局召回，因此 FTS-only 命中可以没有 `binaryScore`。

- [x] **Step 1: 写失败测试**

创建 `Tests/BaseAgentCoreTests/VectorStoreBinarySearchTests.swift`：

```swift
import XCTest
@testable import BaseAgentCore

/// 两阶段二值检索：stage 1 汉明粗筛、stage 2 余弦精排、ANN 旁路与回退。
final class VectorStoreBinarySearchTests: XCTestCase {
    private var directory: URL!
    private let dimension = 16
    private let binaryDimension = 16

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-binary-search-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func dbPath(_ name: String) -> String {
        directory.appendingPathComponent("\(name).sqlite").path
    }

    private func vector(_ first: Float, _ second: Float) -> [Float] {
        Array(repeating: first, count: 8) + Array(repeating: second, count: 8)
    }

    private func makeStore(_ name: String) throws -> VectorStore {
        try VectorStore(dbPath: dbPath(name), dimension: dimension,
                        elementType: .int8, binaryDimension: binaryDimension,
                        binarySource: "sig", indexFingerprint: "fp-\(name)")
    }

    /// 查询 [1×8, -1×8]；A 与查询同签名但余弦低，B 翻转 1 bit 但余弦最高，
    /// C 与查询完全相反。用于验证 stage 2 不是只按汉明距离排序。
    private func fixture() -> (query: [Float], a: [Float], b: [Float], c: [Float],
                               querySignature: Data, signatures: [Data]) {
        let query = vector(1, -1)
        let a = Array(repeating: Float(0.01), count: 8)
            + Array(repeating: Float(-1), count: 8)
        let b = Array(repeating: Float(1), count: 8)
            + Array(repeating: Float(-1), count: 7) + [Float(1)]
        let c = vector(-1, 1)
        return (query, a, b, c,
                Data([0xFF, 0x00]),
                [Data([0xFF, 0x00]), Data([0xFF, 0x80]), Data([0x00, 0xFF])])
    }

    func testTwoStageSearchRanksByCosineAndSetsBinaryScores() throws {
        let store = try makeStore("two-stage")
        let fixture = fixture()
        _ = try store.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0),
                     Chunk(text: "beta", source: "b.md", chunkIndex: 0),
                     Chunk(text: "gamma", source: "c.md", chunkIndex: 0)],
            vectors: [fixture.a, fixture.b, fixture.c],
            binarySignatures: fixture.signatures)
        let hits = try store.hybridSearch(
            vector: fixture.query, query: "zzz", topK: 3,
            binary: BinaryCoarseOptions(signature: fixture.querySignature,
                                        candidateLimit: 200))
        XCTAssertEqual(hits.map(\.text), ["beta", "alpha", "gamma"])
        XCTAssertEqual(hits[0].vectorScore ?? 0, 0.875, accuracy: 0.001)
        XCTAssertEqual(hits[0].binaryScore ?? 0, 0.9375, accuracy: 0.001)
        XCTAssertEqual(hits[1].vectorScore ?? 0, 0.714, accuracy: 0.01)
        XCTAssertEqual(hits[1].binaryScore ?? 0, 1.0, accuracy: 0.0001)
        XCTAssertNil(hits[0].ftsScore)
        XCTAssertEqual(store.annQueryCount, 0, "二值可用时不应查询 ANN")
        store.close()
    }

    /// stage 1 只保留 20 个汉明最近的候选；FTS-only 命中不在候选池内，
    /// 仍必须通过 BM25/RRF 进入最终结果，且 `binaryScore` 为 nil。
    func testStageOneCandidatesLeaveFtsOnlyHitWithoutBinaryScore() throws {
        let store = try makeStore("fts-fusion")
        let query = vector(1, -1)
        let querySignature = Data([0xFF, 0x00])
        var chunks: [Chunk] = []
        var vectors: [[Float]] = []
        var signatures: [Data] = []
        for index in 0..<22 {
            chunks.append(Chunk(text: "filler \(index)", source: "f\(index).md",
                                chunkIndex: 0))
            vectors.append(query)
            signatures.append(querySignature)
        }
        chunks.append(Chunk(text: "needle", source: "needle.md", chunkIndex: 0))
        vectors.append(vector(-1, 1))
        signatures.append(Data([0x00, 0xFF]))
        for index in 22..<25 {
            chunks.append(Chunk(text: "tail \(index)", source: "t\(index).md",
                                chunkIndex: 0))
            vectors.append(vector(-1, 1))
            signatures.append(Data([0x00, 0xFF]))
        }
        _ = try store.add(chunks: chunks, vectors: vectors,
                          binarySignatures: signatures)

        let hits = try store.hybridSearch(
            vector: query, query: "needle", topK: 10,
            binary: BinaryCoarseOptions(signature: querySignature,
                                        candidateLimit: 20))
        let needle = try XCTUnwrap(hits.first { $0.text == "needle" })
        XCTAssertNil(needle.binaryScore)
        XCTAssertNil(needle.vectorScore)
        XCTAssertNotNil(needle.ftsScore)
        store.close()
    }

    func testBinaryFallsBackWhenCoverageIncomplete() throws {
        let store = try makeStore("coverage-fallback")
        let query = vector(1, -1)
        _ = try store.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0),
                     Chunk(text: "beta", source: "b.md", chunkIndex: 0)],
            vectors: [query, vector(-1, 1)])
        XCTAssertEqual(store.binaryIndexStatus().coverage, 0)

        let hits = try store.hybridSearch(
            vector: query, query: "zzz", topK: 2,
            binary: BinaryCoarseOptions(signature: Data([0xFF, 0x00]),
                                        candidateLimit: 200),
            ann: ANNQueryOptions(nprobe: 2, rescore: 0))
        XCTAssertEqual(hits.count, 2)
        XCTAssertTrue(hits.allSatisfy { $0.binaryScore == nil })
        XCTAssertGreaterThan(store.annQueryCount, 0, "覆盖不足时应回退 ANN/精确路径")
        store.close()
    }

    func testBinaryFallsBackOnQuerySignatureLengthMismatch() throws {
        let store = try makeStore("length-fallback")
        let fixture = fixture()
        _ = try store.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0),
                     Chunk(text: "beta", source: "b.md", chunkIndex: 0)],
            vectors: [fixture.a, fixture.b],
            binarySignatures: [fixture.signatures[0], fixture.signatures[1]])

        let hits = try store.hybridSearch(
            vector: fixture.query, query: "zzz", topK: 2,
            binary: BinaryCoarseOptions(signature: Data([0xFF]), candidateLimit: 200),
            ann: ANNQueryOptions(nprobe: 2, rescore: 0))
        XCTAssertEqual(hits.count, 2)
        XCTAssertTrue(hits.allSatisfy { $0.binaryScore == nil })
        XCTAssertGreaterThan(store.annQueryCount, 0)
        store.close()
    }

    func testTargetedSearchIgnoresBinaryOptions() throws {
        let store = try makeStore("targeted")
        let fixture = fixture()
        _ = try store.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0),
                     Chunk(text: "beta", source: "b.md", chunkIndex: 0)],
            vectors: [fixture.a, fixture.b],
            binarySignatures: [fixture.signatures[0], fixture.signatures[1]])
        let options = BinaryCoarseOptions(signature: fixture.querySignature,
                                          candidateLimit: 200)

        let filtered = try store.hybridSearch(
            vector: fixture.query, query: "zzz", topK: 2,
            sourceFilter: "a.md", binary: options)
        XCTAssertEqual(filtered.map(\.text), ["alpha"])
        XCTAssertNil(filtered.first?.binaryScore)

        let contains = try store.hybridSearch(
            vector: fixture.query, query: "zzz", topK: 2,
            sourceContains: "a", binary: options)
        XCTAssertEqual(contains.map(\.text), ["alpha"])
        XCTAssertNil(contains.first?.binaryScore)
        store.close()
    }

    func testBinaryAvailableDoesNotQueryANN() throws {
        let store = try makeStore("ann-bypass")
        let fixture = fixture()
        _ = try store.add(
            chunks: [Chunk(text: "alpha", source: "a.md", chunkIndex: 0),
                     Chunk(text: "beta", source: "b.md", chunkIndex: 0)],
            vectors: [fixture.a, fixture.b],
            binarySignatures: [fixture.signatures[0], fixture.signatures[1]])
        let ann = ANNQueryOptions(nprobe: 2, rescore: 0)

        let binaryHits = try store.hybridSearch(
            vector: fixture.query, query: "zzz", topK: 2,
            binary: BinaryCoarseOptions(signature: fixture.querySignature,
                                        candidateLimit: 200),
            ann: ann)
        XCTAssertFalse(binaryHits.isEmpty)
        XCTAssertEqual(store.annQueryCount, 0)

        _ = try store.hybridSearch(
            vector: fixture.query, query: "zzz", topK: 2,
            ann: ann)
        XCTAssertEqual(store.annQueryCount, 1, "不带二值时仍应走原 ANN 通道")
        store.close()
    }
}
```

- [x] **Step 2: 运行测试确认 RED**

```bash
swift test --disable-sandbox --filter VectorStoreBinarySearchTests
```

预期：编译失败，`hybridSearch` 没有 `binary:` 参数、`annQueryCount` 不存在；
`BinaryCoarseOptions` 已在 Task 7 定义，其余测试依赖 Stage 1/2 尚未实现。

- [x] **Step 3: 让 `annQueryCount` 可被测试观察，并放开 `maxVectorK`**

在 `VectorStore.swift` 中把：

```swift
    private let maxVectorK = 4096
```

改为：

```swift
    let maxVectorK = 4096
```

在 `VectorStoreANN.swift` 的 `annCandidatesLocked` 第一行加入：

```swift
        annQueryCount += 1
```

- [x] **Step 4: 改造 `hybridSearch`**

把签名改为：

```swift
    public func hybridSearch(vector: [Float], query: String, topK: Int,
                             sourceFilter: String? = nil,
                             sourceContains: String? = nil,
                             rrfK: Int = 60, overfetch: Int = 2,
                             binary: BinaryCoarseOptions? = nil,
                             ann: ANNQueryOptions? = nil,
                             deadline: Date? = nil) throws -> [SearchHit] {
```

把现有这一段：

```swift
        let vectorCandidates: [(Int64, Double)] = vectorUsable
            ? try vectorCandidatesLocked(vector: vector, topK: candidateCount,
                                         sourceFilter: sourceFilter,
                                         ann: ann, targeted: targeted)
            : []
```

替换为：

```swift
        var binaryScores: [Int64: Double] = [:]
        var vectorCandidates: [(Int64, Double)] = []
        if vectorUsable {
            if let binary, sourceFilter == nil, !targeted {
                switch try binaryCoarseCandidatesLocked(
                    vector: vector, signature: binary.signature,
                    candidateLimit: binary.candidateLimit,
                    candidateCount: candidateCount, targeted: false) {
                case .used(let candidates, let scores):
                    vectorCandidates = candidates
                    binaryScores = scores
                case .skipped(let reason):
                    noteBinaryFallbackLocked(reason)
                    vectorCandidates = try vectorCandidatesLocked(
                        vector: vector, topK: candidateCount,
                        sourceFilter: sourceFilter, ann: ann, targeted: targeted)
                }
            } else {
                vectorCandidates = try vectorCandidatesLocked(
                    vector: vector, topK: candidateCount,
                    sourceFilter: sourceFilter, ann: ann, targeted: targeted)
            }
        }
```

在 hydrate 映射中补上 `binaryScore`：

```swift
                    copy.rrfScore = hit.score
                    copy.vectorScore = vectorScores[hit.rowID]
                    copy.ftsScore = ftsScores[hit.rowID]
```

注意：上面只是定位用的现有代码，实际修改为：

```swift
                    copy.rrfScore = hit.score
                    copy.vectorScore = vectorScores[hit.rowID]
                    copy.ftsScore = ftsScores[hit.rowID]
                    copy.binaryScore = binaryScores[hit.rowID]
```

- [x] **Step 5: 在 `VectorStoreBinary.swift` 增加两阶段查询与日志节流**

在 `BinaryCoarseOptions` 之后、`extension VectorStore {` 内部加入：

```swift
    enum BinaryCoarseQueryOutcome {
        case used(candidates: [(Int64, Double)], scores: [Int64: Double])
        case skipped(String)
    }

    /// stage 1 二值 KNN + stage 2 主向量余弦精排。不可用时返回跳过原因，
    /// 由 `hybridSearch` 回退原向量通道。调用方需已持有 `lock`。
    func binaryCoarseCandidatesLocked(
        vector: [Float], signature: Data, candidateLimit: Int,
        candidateCount: Int, targeted: Bool
    ) throws -> BinaryCoarseQueryOutcome {
        guard !targeted else { return .skipped("定向检索不参与二值粗筛") }
        guard rebuildReason == nil else { return .skipped("向量库待重建") }
        guard elementType != .bin else {
            return .skipped("bin 主存储仅支持单阶段检索")
        }
        guard binaryStorageEnabled else {
            return .skipped(binaryStorageReason ?? "bin 表不可用")
        }
        let expectedBytes = binaryDimension / 8
        guard signature.count == expectedBytes else {
            return .skipped("查询签名长度 \(signature.count) != \(expectedBytes)")
        }
        let chunkRows = count
        guard chunkRows > 0 else { return .skipped("知识库为空") }
        guard binaryRows == chunkRows else {
            return .skipped("二值签名覆盖 \(binaryRows)/\(chunkRows)，请重新同步")
        }

        let k = min(maxVectorK, max(candidateLimit, candidateCount))
        let rows = try db.query(
            "SELECT rowid, distance FROM \(Self.binaryVectorTable) "
            + "WHERE embedding MATCH vec_bit(?) AND k = ?",
            [.blob(signature), .int(Int64(k))])
        let hits = rows.compactMap { row -> (Int64, Double)? in
            guard let rowID = row["rowid"]?.intValue else { return nil }
            return (rowID, row["distance"]?.doubleValue ?? Double(binaryDimension))
        }
        guard !hits.isEmpty else { return .used(candidates: [], scores: [:]) }

        let storedVectors = try exactVectorsLocked(rowIDs: hits.map(\.0))
        var reranked: [(rowID: Int64, cosine: Double, binary: Double)] = []
        reranked.reserveCapacity(hits.count)
        for (rowID, distance) in hits {
            guard let stored = storedVectors[rowID], stored.count == vector.count else {
                continue
            }
            let cosine = VectorCodec.cosine(vector, stored)
            let binaryScore = max(0, 1 - distance / Double(binaryDimension))
            reranked.append((rowID, cosine, binaryScore))
        }
        reranked.sort { $0.cosine > $1.cosine }
        let selected = Array(reranked.prefix(candidateCount))
        return .used(
            candidates: selected.map { ($0.rowID, $0.cosine) },
            scores: Dictionary(uniqueKeysWithValues: selected.map { ($0.rowID, $0.binary) }))
    }
```

在 `binaryIndexStatusLocked()` 之后加入：

```swift
    /// 同一原因 60 秒内只记一条，避免每次查询刷日志。
    func noteBinaryFallbackLocked(_ reason: String) {
        let now = Date()
        if reason == lastBinaryFallbackReason,
           now.timeIntervalSince(lastBinaryFallbackLog) < 60 {
            return
        }
        lastBinaryFallbackReason = reason
        lastBinaryFallbackLog = now
        AgentLog.info("二值粗筛回退：\(reason)")
    }
```

- [x] **Step 6: 运行测试确认 GREEN**

```bash
swift test --disable-sandbox --filter VectorStoreBinarySearchTests
swift test --disable-sandbox --filter VectorStoreBinaryTests
```

预期：新增 6 个用例通过；既有 bin 表生命周期用例保持通过。

- [x] **Step 7: 提交**

```bash
git add Sources/BaseAgentCore/RAG/Storage/VectorStore.swift \
        Sources/BaseAgentCore/RAG/Storage/VectorStoreANN.swift \
        Sources/BaseAgentCore/RAG/Storage/VectorStoreBinary.swift \
        Tests/BaseAgentCoreTests/VectorStoreBinarySearchTests.swift
git commit -m "feat(rag): add two-stage binary coarse retrieval"
```

## Task 10: 二值粗筛配置项

**Files:**
- Modify: `Sources/BaseAgentCore/Core/Config.swift:251-261`（新增字段）
- Modify: `Sources/BaseAgentCore/Core/Config.swift:480-535`（`apply`）
- Modify: `Sources/BaseAgentCore/Core/Config.swift:758-776`（`migrate`）
- Modify: `Sources/BaseAgentCore/Core/Config.swift:848-894`（`export`）
- Test: `Tests/BaseAgentCoreTests/BinaryCoarseConfigTests.swift`

**设计：** 新增 `ragBinaryCoarseEnabled`（默认 `true`）与
`ragBinaryCoarseCandidates`（默认 200）。持久化键为
`rag_binary_coarse_enabled` / `rag_binary_coarse_candidates`；加载后
`migrate()` 把候选数统一收口到 `50...1000`。开关只控制检索，不影响入库签名。

- [x] **Step 1: 写失败测试**

创建 `Tests/BaseAgentCoreTests/BinaryCoarseConfigTests.swift`：

```swift
import XCTest
@testable import BaseAgentCore

final class BinaryCoarseConfigTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("binary-config-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func writeConfig(_ json: String) throws -> String {
        let path = directory.appendingPathComponent("config-\(UUID().uuidString).json").path
        try json.write(toFile: path, atomically: true, encoding: .utf8)
        return path
    }

    func testDefaultsEnableBinaryCoarseWith200Candidates() {
        let config = AgentConfig()
        XCTAssertTrue(config.ragBinaryCoarseEnabled)
        XCTAssertEqual(config.ragBinaryCoarseCandidates, 200)
    }

    func testLoadReadsAndClampsBinaryCoarseOptions() throws {
        let path = try writeConfig("""
        {
          "rag_binary_coarse_enabled": false,
          "rag_binary_coarse_candidates": 5000
        }
        """)
        let config = AgentConfig.load(path)
        XCTAssertFalse(config.ragBinaryCoarseEnabled)
        XCTAssertEqual(config.ragBinaryCoarseCandidates, 1000)
    }

    func testMigrateClampsCandidatesToSupportedRange() {
        var config = AgentConfig()
        config.ragBinaryCoarseCandidates = 49
        config.migrate()
        XCTAssertEqual(config.ragBinaryCoarseCandidates, 50)

        config.ragBinaryCoarseCandidates = 500
        config.migrate()
        XCTAssertEqual(config.ragBinaryCoarseCandidates, 500)
    }

    func testExportWritesBinaryCoarseKeys() {
        var config = AgentConfig()
        config.ragBinaryCoarseEnabled = false
        config.ragBinaryCoarseCandidates = 120
        let exported = config.export()
        XCTAssertEqual(exported["rag_binary_coarse_enabled"] as? Bool, false)
        XCTAssertEqual(exported["rag_binary_coarse_candidates"] as? Int, 120)
    }
}
```

- [x] **Step 2: 运行测试确认 RED**

```bash
swift test --disable-sandbox --filter BinaryCoarseConfigTests
```

预期：编译失败，`ragBinaryCoarseEnabled` / `ragBinaryCoarseCandidates` 不存在。

- [x] **Step 3: 新增字段**

在 `AgentConfig` 的 `ragMatryoshkaDim` 之后、ANN 配置之前加入：

```swift
    // 二值粗筛（两段式检索）
    /// 检索是否先走二值向量粗筛；关闭只影响检索，入库仍维护签名。
    public var ragBinaryCoarseEnabled: Bool = true
    /// 第一阶段候选数：默认 200，加载时收口到 50...1000。
    public var ragBinaryCoarseCandidates: Int = 200
```

- [x] **Step 4: 接入 `apply` / `migrate` / `export`**

在 `apply` 的 `rag_matryoshka_dim` 之后加入：

```swift
        if let v = bool("rag_binary_coarse_enabled") { ragBinaryCoarseEnabled = v }
        if let v = int("rag_binary_coarse_candidates") { ragBinaryCoarseCandidates = v }
```

在 `migrate()` 的 Matryoshka 注释之前加入：

```swift
        // 二值粗筛候选数：手改配置可能越界，统一收口到 50...1000。
        ragBinaryCoarseCandidates = max(50, min(1_000, ragBinaryCoarseCandidates))
```

在 `export()` 的 `"rag_matryoshka_dim": ragMatryoshkaDim,` 之后加入：

```swift
            "rag_binary_coarse_enabled": ragBinaryCoarseEnabled,
            "rag_binary_coarse_candidates": ragBinaryCoarseCandidates,
```

- [x] **Step 5: 运行测试确认 GREEN**

```bash
swift test --disable-sandbox --filter BinaryCoarseConfigTests
swift test --disable-sandbox --filter RAGEngineVectorConfigTests
```

预期：4 个新用例通过；既有向量配置用例保持通过。

- [x] **Step 6: 提交**

```bash
git add Sources/BaseAgentCore/Core/Config.swift \
        Tests/BaseAgentCoreTests/BinaryCoarseConfigTests.swift
git commit -m "feat(rag): add binary coarse retrieval config"
```

## Task 11: RAGEngine 入库、查询与状态接线

**Files:**
- Create: `Tests/BaseAgentCoreTests/RAGEngineBinaryTests.swift`
- Modify: `Sources/BaseAgentCore/RAG/RAGEngine.swift:680-710`（签名位宽/来源 + `ensureStore`）
- Modify: `Sources/BaseAgentCore/RAG/RAGEngine.swift:1075-1165`（入库开关透传）
- Modify: `Sources/BaseAgentCore/RAG/RAGEngine.swift:1320-1336`（`process` 参数）
- Modify: `Sources/BaseAgentCore/RAG/RAGEngine.swift:1557-1576`（签名入库）
- Modify: `Sources/BaseAgentCore/RAG/RAGEngine.swift:1870-1884`（`binaryIndexStatus`）
- Modify: `Sources/BaseAgentCore/RAG/RAGEngine.swift:2125-2157`（查询签名）
- Modify: `Sources/BaseAgentCore/RAG/RAGEngine.swift:2434-2461`（状态字段）

**设计：** 入库时只要主元素类型不是 `.bin` 且调度器声明了签名位宽，就调用
`embedDocumentsWithSignatures`，并在一次写库中同时落主向量与签名；查询开关
开启时用 `embedQueryWithSignature` 一次前向得到主向量与签名。`status()` 把
存储状态与配置/provider 能力映射为 `binary_coarse` 五态，并输出
`binary_dimension` / `binary_coverage` / `binary_rows` / `binary_backend` /
`binary_note` / `ann_bypassed`。库打开时把签名位宽与来源传给 `VectorStore`。

- [x] **Step 1: 写失败测试**

创建 `Tests/BaseAgentCoreTests/RAGEngineBinaryTests.swift`：

```swift
import XCTest
@testable import BaseAgentCore

private func fakeVector(_ text: String) -> [Float] {
    var vector = [Float](repeating: 0, count: 8)
    for (index, scalar) in text.unicodeScalars.enumerated() {
        vector[(Int(scalar.value) + index) % 8] += 1
    }
    return VectorCodec.normalize(vector)
}

/// 引擎与二值粗筛的集成：入库签名、查询签名、状态五态与 bin 主存储旁路。
final class RAGEngineBinaryTests: XCTestCase {
    private var root: URL!
    private var kb: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("binary-engine-\(UUID().uuidString)")
        kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private final class SigningEmbedding: EmbeddingProvider, EmbeddingBinarySigning {
        let identifier = "binary-engine-fake"
        let dimension = 8

        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map(fakeVector)
        }

        func embedQuery(_ text: String) async throws -> [Float] {
            fakeVector(text)
        }
    }

    private final class PlainEmbedding: EmbeddingProvider {
        let identifier = "plain-engine-fake"
        let dimension = 8

        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map(fakeVector)
        }

        func embedQuery(_ text: String) async throws -> [Float] {
            fakeVector(text)
        }
    }

    private func makeConfig(_ adjust: (inout AgentConfig) -> Void = { _ in }) -> AgentConfig {
        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 5
        config.ragVectorElementType = .int8
        adjust(&config)
        return config
    }

    private func writeDocs(_ count: Int) throws {
        for index in 1...count {
            let text = """
            # 文档 \(index)

            Apple banana orange fruit \(index). 苹果 香蕉 橙子 水果 \(index)。
            涡轮 发动机 机械 设备 编号 \(index)。turbine engine machine \(index)。
            """
            try text.write(to: kb.appendingPathComponent("doc\(index).md"),
                           atomically: true, encoding: .utf8)
        }
    }

    private func makeEngine(_ config: AgentConfig,
                            embedding: EmbeddingProvider) -> RAGEngine {
        RAGEngine(config: config, knowledgeBase: kb.path,
                  embedding: embedding, reranker: nil)
    }

    func testIngestWritesBinarySignaturesAndStatusReady() async throws {
        try writeDocs(2)
        let engine = makeEngine(makeConfig(), embedding: SigningEmbedding())
        let stats = await engine.ingest()
        XCTAssertGreaterThan(stats.chunks, 0)

        let status = engine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(status["binary_coarse"]?.stringValue, "ready")
        XCTAssertEqual(status["binary_dimension"]?.intValue, 8)
        XCTAssertEqual(status["binary_rows"]?.intValue,
                       status["chunks_stored"]?.intValue)
        XCTAssertEqual(status["binary_coverage"]?.doubleValue ?? 0, 1.0,
                       accuracy: 0.001)
        XCTAssertEqual(status["binary_backend"]?.stringValue, "sqlite-vec")
        XCTAssertEqual(status["ann_bypassed"]?.boolValue, true)
        XCTAssertEqual(status["binary_note"]?.stringValue, "")

        let binary = try XCTUnwrap(engine.binaryIndexStatus())
        XCTAssertTrue(binary.storageEnabled)
        XCTAssertTrue(binary.isComplete)
        XCTAssertEqual(binary.dimension, 8)
        engine.close()
    }

    func testSearchUsesBinaryQueryAndPopulatesBinaryScore() async throws {
        try writeDocs(2)
        let engine = makeEngine(makeConfig(), embedding: SigningEmbedding())
        _ = await engine.ingest()
        let hits = try await engine.search(query: "apple fruit turbine", topK: 3)
        XCTAssertFalse(hits.isEmpty)
        XCTAssertTrue(hits.contains { $0.binaryScore != nil },
                      "开启二值粗筛时向量候选应带汉明相似度")
        engine.close()
    }

    /// 开关关闭只影响检索：入库仍维护签名，查询不再产生 binaryScore。
    func testDisabledBinaryCoarseStillWritesSignatures() async throws {
        try writeDocs(2)
        let config = makeConfig { $0.ragBinaryCoarseEnabled = false }
        let engine = makeEngine(config, embedding: SigningEmbedding())
        _ = await engine.ingest()

        let status = engine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(status["binary_coarse"]?.stringValue, "off")
        XCTAssertEqual(status["binary_rows"]?.intValue,
                       status["chunks_stored"]?.intValue)
        XCTAssertEqual(status["ann_bypassed"]?.boolValue, false)

        let hits = try await engine.search(query: "apple fruit turbine", topK: 3)
        XCTAssertFalse(hits.isEmpty)
        XCTAssertTrue(hits.allSatisfy { $0.binaryScore == nil })
        engine.close()
    }

    func testPlainProviderReportsUnsupportedWithoutBinaryRows() async throws {
        try writeDocs(1)
        let engine = makeEngine(makeConfig(), embedding: PlainEmbedding())
        _ = await engine.ingest()

        let status = engine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(status["binary_coarse"]?.stringValue, "unsupported")
        XCTAssertEqual(status["binary_rows"]?.intValue, 0)
        XCTAssertTrue(status["binary_note"]?.stringValue?.contains("不支持") == true,
                      status["binary_note"]?.stringValue ?? "")
        XCTAssertEqual(status["ann_bypassed"]?.boolValue, false)

        let binary = try XCTUnwrap(engine.binaryIndexStatus())
        XCTAssertFalse(binary.storageEnabled)
        engine.close()
    }

    func testBinPrimaryStorageBypassesBinaryCoarse() async throws {
        try writeDocs(1)
        let config = makeConfig { $0.ragVectorElementType = .bin }
        let engine = makeEngine(config, embedding: SigningEmbedding())
        _ = await engine.ingest()

        let status = engine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(status["binary_coarse"]?.stringValue, "bypassed_bin")
        XCTAssertTrue(status["binary_note"]?.stringValue?.contains("bin 主存储") == true,
                      status["binary_note"]?.stringValue ?? "")
        XCTAssertEqual(status["binary_rows"]?.intValue, 0)
        XCTAssertEqual(status["ann_bypassed"]?.boolValue, false)
        engine.close()
    }
}
```

- [x] **Step 2: 运行测试确认 RED**

```bash
swift test --disable-sandbox --filter RAGEngineBinaryTests
```

预期：编译失败，`RAGEngine` 没有 `binaryIndexStatus()` / 状态字段；
`store.add(binarySignatures:)` 尚未接线；测试在编译通过后会因状态缺失失败。

- [x] **Step 3: 新增签名位宽/来源，并在打开库时传入**

在 `RAGEngine` 的 `effectiveVectorElementType` 之后加入：

```swift
    /// 二值签名位宽；主存储为 bin 时不启用（bin 只有单阶段检索）。
    public var binarySignatureDimension: Int {
        guard effectiveVectorElementType != .bin else { return 0 }
        return (embedding as? EmbeddingBinarySigning)?.binaryDimension ?? 0
    }

    /// 二值签名来源标识。Matryoshka 不改变签名来源，避免截断维变化误清空 bin 表。
    public var binarySignatureSource: String {
        guard binarySignatureDimension > 0, let embedding else { return "" }
        return (embedding as? EmbeddingBinarySigning)?.binarySourceIdentifier
            ?? embedding.identifier
    }
```

把 `ensureStore()` 中的 `VectorStore` 初始化改为：

```swift
        let newStore = try VectorStore(dbPath: self.indexDBPath,
                                       dimension: embedding.dimension,
                                       enableSQLiteVec: config.ragSQLiteVecEnabled,
                                       elementType: effectiveVectorElementType,
                                       binaryDimension: binarySignatureDimension,
                                       binarySource: binarySignatureSource,
                                       indexFingerprint: vectorIndexFingerprint)
```

- [x] **Step 4: 入库时按能力请求签名**

在 `ingest` 创建 `scheduler` 之后加入：

```swift
        let binarySignaturesEnabled = effectiveVectorElementType != .bin
            && scheduler.binaryDimension != nil
```

在 `process` 参数列表的 `config: AgentConfig,` 之后加入：

```swift
        binarySignaturesEnabled: Bool,
```

在 `Self.process(...)` 调用中 `config: configRef,` 之后加入：

```swift
                                binarySignaturesEnabled: binarySignaturesEnabled,
```

把 `process` 内从 `let embedStarted = Date()` 到 `_ = try store.add(...)` 的整段替换为：

```swift
            let embedStarted = Date()
            var lastEmbedLog = Date()
            let onEmbedProgress: (Int, Int) -> Void = { done, count in
                let now = Date()
                guard now.timeIntervalSince(lastEmbedLog) >= 60 else { return }
                lastEmbedLog = now
                log("RAG: \(job.label) 向量化中 \(done)/\(count)"
                    + "（已用 \(Self.elapsedText(embedStarted))）")
            }
            let texts = chunks.map(\.text)
            let signed: BinaryEmbeddingOutput?
            let vectors: [[Float]]
            if binarySignaturesEnabled {
                signed = try await embedding.embedDocumentsWithSignatures(
                    texts, progress: onEmbedProgress)
                vectors = signed?.vectors
                    ?? (try await embedding.embedDocuments(texts, progress: onEmbedProgress))
            } else {
                signed = nil
                vectors = try await embedding.embedDocuments(
                    texts, progress: onEmbedProgress)
            }
            guard vectors.count == chunks.count else {
                throw AgentError.storage("embedding 数量与切片不一致")
            }
            if let signed, signed.signatures.count != chunks.count {
                throw AgentError.storage("embedding 签名数量与切片不一致")
            }
            let embedSeconds = Date().timeIntervalSince(embedStarted)
            if isCancelled() || Task.isCancelled { throw AgentError.cancelled }

            let writeStarted = Date()
            _ = try store.add(chunks: chunks, vectors: vectors,
                              binarySignatures: signed?.signatures)
```

- [x] **Step 5: 查询时一次前向得到主向量与签名**

把 `search` 中的：

```swift
        let queryVector = try await embedding.embedQuery(query)
```

替换为：

```swift
        let queryVector: [Float]
        let binaryOptions: BinaryCoarseOptions?
        if config.ragBinaryCoarseEnabled,
           effectiveVectorElementType != .bin,
           let signing = embedding as? EmbeddingBinarySigning,
           signing.binaryDimension > 0 {
            let signed = try await signing.embedQueryWithSignature(query)
            queryVector = signed.vector
            binaryOptions = BinaryCoarseOptions(
                signature: signed.signature,
                candidateLimit: config.ragBinaryCoarseCandidates)
        } else {
            queryVector = try await embedding.embedQuery(query)
            binaryOptions = nil
        }
```

在 `store.hybridSearch` 调用中加入 `binary`（放在 `ann` 之前）：

```swift
                    rrfK: config.ragRRFK,
                    overfetch: config.ragVectorOverfetch,
                    binary: binaryOptions,
                    ann: annQueryOptions,
                    deadline: deadline
```

- [x] **Step 6: 状态接口与五态映射**

在 `annIndexStatus(timeout:)` 之后加入：

```swift
    /// 二值粗筛状态。与 ANN 一样走带超时的只读入口。
    public func binaryIndexStatus(timeout: Double? = nil) -> BinaryIndexStatus? {
        try? withStoreBounded(timeout: resolvedTimeout(timeout)) { $0.binaryIndexStatus() }
    }
```

在 `status()` 的 `withStoreBounded` 闭包内，`let ann = store.annIndexStatus()` 之后加入：

```swift
            let binary = store.binaryIndexStatus()
            let binaryState: String
            let binaryNote: String
            if !config.ragBinaryCoarseEnabled {
                binaryState = "off"
                binaryNote = "已关闭"
            } else if store.elementType == .bin {
                binaryState = "bypassed_bin"
                binaryNote = "bin 主存储仅支持单阶段检索"
            } else if binarySignatureDimension == 0 || !binary.storageEnabled {
                binaryState = "unsupported"
                binaryNote = binary.reason ?? "embedding 不支持二值签名"
            } else if !binary.isComplete {
                binaryState = "incomplete"
                let percent = Int((binary.coverage * 100).rounded())
                binaryNote = "覆盖 \(percent)%，请重新同步"
            } else {
                binaryState = "ready"
                binaryNote = ""
            }
```

在返回字典的 `"ann_effective_coverage"` 之后加入：

```swift
                // 二值粗筛：off → bypassed_bin → unsupported → incomplete → ready。
                // ANN 字段保留，ann_bypassed 表示本次全局检索由二值取代 ANN。
                "binary_coarse": .string(binaryState),
                "binary_dimension": .int(binary.dimension),
                "binary_coverage": .double(binary.coverage),
                "binary_rows": .int(binary.rows),
                "binary_backend": .string(binary.storageEnabled ? "sqlite-vec" : "none"),
                "binary_note": .string(binaryNote),
                "ann_bypassed": .bool(binaryState == "ready"),
```

- [x] **Step 7: 运行测试确认 GREEN**

```bash
swift test --disable-sandbox --filter RAGEngineBinaryTests
swift test --disable-sandbox --filter RAGEngineANNTests
swift test --disable-sandbox --filter RAGEngineVectorConfigTests
```

预期：5 个新用例通过；既有引擎/ANN/配置用例保持通过。

- [x] **Step 8: 提交**

```bash
git add Sources/BaseAgentCore/RAG/RAGEngine.swift \
        Tests/BaseAgentCoreTests/RAGEngineBinaryTests.swift
git commit -m "feat(rag): wire binary coarse retrieval into RAG engine"
```

## Task 12: 设置界面与 `rag_status` 输出

**Files:**
- Modify: `Sources/BaseAgentApp/SettingsView.swift:86-88`（状态）
- Modify: `Sources/BaseAgentApp/SettingsView.swift:527-552`（新增一行）
- Modify: `Sources/BaseAgentApp/SettingsView.swift:1189-1203`（`loadConfig`）
- Modify: `Sources/BaseAgentApp/SettingsView.swift:1511-1517`（`saveConfig`）
- Modify: `Sources/BaseAgentCore/Tools/RAGTool.swift:255-314`（`rag_status`）
- Test: `Tests/BaseAgentCoreTests/RAGToolBinaryStatusTests.swift`

**设计：** 设置面板在现有「向量库 / ANN」合并行之后新增独立的「二值粗筛 +
候选数」一行，开关关闭时候选数 Stepper 禁用；保存时把候选数收口到
`50...1000`。`rag_status` 读取引擎新增字段，按五态输出一行中文说明，并在
`ann_bypassed=true` 时给 ANN 行加「本次由二值粗筛取代」。

- [x] **Step 1: 写失败测试**

创建 `Tests/BaseAgentCoreTests/RAGToolBinaryStatusTests.swift`：

```swift
import XCTest
@testable import BaseAgentCore

private func toolFakeVector(_ text: String) -> [Float] {
    var vector = [Float](repeating: 0, count: 8)
    for (index, scalar) in text.unicodeScalars.enumerated() {
        vector[(Int(scalar.value) + index) % 8] += 1
    }
    return VectorCodec.normalize(vector)
}

/// `rag_status` 对二值粗筛五态的输出。
final class RAGToolBinaryStatusTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("binary-tool-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private final class SigningEmbedding: EmbeddingProvider, EmbeddingBinarySigning {
        let identifier = "binary-tool-fake"
        let dimension = 8

        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map(toolFakeVector)
        }

        func embedQuery(_ text: String) async throws -> [Float] {
            toolFakeVector(text)
        }
    }

    private final class PlainEmbedding: EmbeddingProvider {
        let identifier = "plain-tool-fake"
        let dimension = 8

        func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
            texts.map(toolFakeVector)
        }

        func embedQuery(_ text: String) async throws -> [Float] {
            toolFakeVector(text)
        }
    }

    private func writeDoc(to kb: URL) throws {
        let text = """
        # 文档

        Apple banana orange fruit. 苹果 香蕉 橙子 水果。
        """
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        try text.write(to: kb.appendingPathComponent("doc.md"),
                       atomically: true, encoding: .utf8)
    }

    private func statusOutput(
        name: String,
        adjust: (inout AgentConfig) -> Void = { _ in },
        embedding: EmbeddingProvider
    ) async throws -> String {
        let workspace = root.appendingPathComponent(name)
        let kb = workspace.appendingPathComponent("kb")
        try writeDoc(to: kb)

        var config = AgentConfig()
        config.workspace = workspace.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 5
        config.ragVectorElementType = .int8
        adjust(&config)

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: embedding, reranker: nil)
        _ = await engine.ingest()
        _ = engine.status(sourceLimit: 0, maxAge: 0)
        let tool = RagStatusTool(config: engine.config, engine: engine)
        let result = try await tool.run([:])
        engine.close()
        XCTAssertTrue(result.success, result.error)
        return result.output
    }

    func testReadyOutputExplainsBinaryReplacesANN() async throws {
        let output = try await statusOutput(name: "ready",
                                            embedding: SigningEmbedding())
        XCTAssertTrue(
            output.contains("二值粗筛: 可用（8 bit，覆盖 100%，本次取代 ANN）"),
            output)
        XCTAssertTrue(output.contains("（本次由二值粗筛取代）"), output)
    }

    func testOffUnsupportedAndBypassedOutputs() async throws {
        let off = try await statusOutput(
            name: "off",
            adjust: { $0.ragBinaryCoarseEnabled = false },
            embedding: SigningEmbedding())
        XCTAssertTrue(off.contains("二值粗筛: 关闭"), off)

        let unsupported = try await statusOutput(
            name: "unsupported", embedding: PlainEmbedding())
        XCTAssertTrue(
            unsupported.contains("二值粗筛: 不可用（embedding 不支持二值签名）"),
            unsupported)

        let bypassed = try await statusOutput(
            name: "bypassed",
            adjust: { $0.ragVectorElementType = .bin },
            embedding: SigningEmbedding())
        XCTAssertTrue(
            bypassed.contains("二值粗筛: 停用（bin 主存储仅支持单阶段检索）"),
            bypassed)
    }
}
```

- [x] **Step 2: 运行测试确认 RED**

```bash
swift test --disable-sandbox --filter RAGToolBinaryStatusTests
```

预期：编译失败，`RAGTool` 尚未读取二值字段；编译通过后输出缺少
「二值粗筛」行。

- [x] **Step 3: 新增设置状态与控件**

在 `SettingsView` 的 `annAutoBuild` 之后加入：

```swift
    /// 二值粗筛开关与第一阶段候选数。
    @State private var binaryCoarseEnabled = true
    @State private var binaryCoarseCandidates = 200
```

在 `ViewThatFits` 之后、章节摘要状态之前加入：

```swift
                HStack(spacing: 8) {
                    Toggle("二值粗筛", isOn: $binaryCoarseEnabled)
                        .help("开启后先用二值向量召回候选，再按主向量余弦精排，"
                            + "最后与全文检索 RRF 融合；可用时取代 ANN，不可用时"
                            + "自动回退原流程。主存储为 BIN 时仅支持单阶段检索；"
                            + "已有库需要重新同步后才会生效。")
                    Stepper(value: $binaryCoarseCandidates, in: 50...1000, step: 50) {
                        Text("候选 \(binaryCoarseCandidates)")
                    }
                    .disabled(!binaryCoarseEnabled)
                    .help("第一阶段二值粗筛候选数：越大召回越稳、精排读取越多；"
                        + "默认 200，允许 50...1000。")
                    Spacer(minLength: 8)
                }
```

- [x] **Step 4: 读写配置**

在 `loadConfig` 的 `annAutoBuild = config.ragANNAutoBuild` 之后加入：

```swift
        binaryCoarseEnabled = config.ragBinaryCoarseEnabled
        binaryCoarseCandidates = config.ragBinaryCoarseCandidates
```

在 `saveConfig` 的 `config.ragANNAutoBuild = annAutoBuild` 之后加入：

```swift
        config.ragBinaryCoarseEnabled = binaryCoarseEnabled
        config.ragBinaryCoarseCandidates = min(max(50, binaryCoarseCandidates), 1000)
```

- [x] **Step 5: `rag_status` 输出**

在 `RagStatusTool.run` 读取字段区加入：

```swift
        let binaryCoarse = status["binary_coarse"]?.stringValue ?? ""
        let binaryDimension = status["binary_dimension"]?.intValue ?? 0
        let binaryCoverage = status["binary_coverage"]?.doubleValue ?? 0
        let binaryNote = status["binary_note"]?.stringValue ?? ""
        let annBypassed = status["ann_bypassed"]?.boolValue ?? false
```

在「向量库待重建」提示之后、ANN 输出之前加入：

```swift
        if !binaryCoarse.isEmpty {
            switch binaryCoarse {
            case "ready":
                let percent = Int((min(1, max(0, binaryCoverage)) * 100).rounded())
                lines.append("  二值粗筛: 可用（\(binaryDimension) bit，覆盖 \(percent)%"
                    + (annBypassed ? "，本次取代 ANN" : "") + "）")
            case "incomplete":
                let note = binaryNote.isEmpty
                    ? "覆盖 \(Int((min(1, max(0, binaryCoverage)) * 100).rounded())%，请重新同步"
                    : binaryNote
                lines.append("  二值粗筛: 不可用（\(note)）")
            case "bypassed_bin":
                lines.append("  二值粗筛: 停用（bin 主存储仅支持单阶段检索）")
            case "off":
                lines.append("  二值粗筛: 关闭")
            default:
                lines.append("  二值粗筛: 不可用（"
                    + "\(binaryNote.isEmpty ? "原因未知" : binaryNote)）")
            }
        }
```

把 ANN 索引行改为：

```swift
        if !annIndex.isEmpty {
            lines.append("  ANN 索引: \(annIndex)"
                + (annBypassed ? "（本次由二值粗筛取代）" : ""))
        }
```

- [x] **Step 6: 运行测试确认 GREEN**

```bash
swift test --disable-sandbox --filter RAGToolBinaryStatusTests
swift build --disable-sandbox
```

预期：2 个新用例通过，App 模块编译通过。

- [x] **Step 7: 提交**

```bash
git add Sources/BaseAgentApp/SettingsView.swift \
        Sources/BaseAgentCore/Tools/RAGTool.swift \
        Tests/BaseAgentCoreTests/RAGToolBinaryStatusTests.swift
git commit -m "feat(rag): expose binary coarse settings and status"
```

## Task 13: README 更新、全量验证与计划自检

**Files:**
- Modify: `README.md:50`（测试覆盖计数与新增覆盖项）
- Modify: `README.md:67`（知识同步配置窗口说明）
- Modify: `README.md:160-164`（存储设计 bullet 与 meta 说明）
- Modify: `README.md:181`（jina-v5 原生签名说明）
- Modify: `README.md:228-232`（配置示例）
- Modify: `docs/superpowers/plans/2026-10-31-binary-coarse-retrieval.md`（自检发现的问题就地修正）

**设计：** 先跑定向矩阵确认每个 Task 的用例都通过，再跑全量测试拿到真实
`Executed N tests`，用这个数字回填 README，而不是凭估算写死。README 只描述
已经实现并能被测试证明的行为，未实现的能力不写进文档。

- [x] **Step 1: 跑签名与模型层定向测试**

```bash
swift test --disable-sandbox --filter BinarySignatureTests
swift test --disable-sandbox --filter EmbeddingBinarySigningTests
swift test --disable-sandbox --filter ProviderBinarySigningTests
swift test --disable-sandbox --filter AdaptiveEmbeddingSigningTests
swift test --disable-sandbox --filter MatryoshkaEmbeddingTests
swift test --disable-sandbox --filter JinaV5MLXEmbeddingE2ETests
swift test --disable-sandbox --filter JinaV5ONNXEmbeddingE2ETests
```

预期：

- `BinarySignatureTests` 6 个、`EmbeddingBinarySigningTests` 4 个、
  `ProviderBinarySigningTests` 1 个、`AdaptiveEmbeddingSigningTests` 4 个全部通过；
- `MatryoshkaEmbeddingTests` 既有用例 + 新增 2 个通过；
- 两套 jina-v5 E2E 在本地权重存在时通过；权重缺失时输出 `skipped`，
  但编译与其余用例必须正常。

- [x] **Step 2: 跑存储与检索定向测试**

```bash
swift test --disable-sandbox --filter VectorStoreBinaryTests
swift test --disable-sandbox --filter VectorStoreBinarySearchTests
swift test --disable-sandbox --filter VectorStoreTests
```

预期：`VectorStoreBinaryTests` 5 个、`VectorStoreBinarySearchTests` 6 个全部
通过；既有 `VectorStoreTests` 全部通过（证明 `add(chunks:vectors:)` 兼容）。

- [x] **Step 3: 跑配置、引擎、工具与 ANN 回归**

```bash
swift test --disable-sandbox --filter BinaryCoarseConfigTests
swift test --disable-sandbox --filter RAGEngineBinaryTests
swift test --disable-sandbox --filter RAGEngineANNTests
swift test --disable-sandbox --filter RAGToolBinaryStatusTests
swift build --disable-sandbox
```

预期：`BinaryCoarseConfigTests` 4 个、`RAGEngineBinaryTests` 5 个、
`RAGToolBinaryStatusTests` 2 个全部通过；`RAGEngineANNTests` 全部通过
（证明二值可用时只是查询旁路，ANN 构建/状态逻辑未被破坏）；App 模块编译通过。

- [x] **Step 4: 跑全量测试并记录真实数量**

```bash
swift test --disable-sandbox 2>&1 | tee /tmp/binary-coarse-tests.log
grep -E "Executed [0-9]+ tests" /tmp/binary-coarse-tests.log | tail -1
```

预期：`Executed 577 tests, with 0 failures`（基线 530 + 本次新增 47 个用例）。
若实际数字不同，以 `Executed N tests` 的输出为准，下一步回填 N；若出现失败，
先修复失败再更新 README，不允许带着失败提交。

- [x] **Step 5: 更新 README 测试覆盖与 jina-v5 说明**

第 50 行做三处替换：

1. 计数：

```text
旧：测试覆盖（当前 530 个 XCTest；
新：测试覆盖（当前 577 个 XCTest；
```

2. 在测试项列表中加入二值粗筛覆盖：

```text
旧：sqlite-vec 事务写入/回滚、两段式 KNN、FTS5 中文分词、RRF 混合检索、
新：sqlite-vec 事务写入/回滚、两段式 KNN、二值签名编码与 LSB-first 位打包、
    bin 表同 rowid 写入与覆盖率计数器、两阶段粗筛精排与 ANN 旁路、
    FTS5 中文分词、RRF 混合检索、
```

3. 在真实模型链路描述中补签名：

```text
旧：→ 用户选择的元素类型入库（默认 int8，128 维时 128B/行）→ 查询 top1 与 `rag_search` 全链路；
新：→ 用户选择的元素类型入库（默认 int8，128 维时 128B/行），同时从截断前原生
    768 维生成二值签名写入 `chunk_vectors_bin` → 查询 top1 与 `rag_search` 全链路；
```

第 181 行在句末加入：

```text
旧：provider 内部完成截断，Matryoshka 不再包装；关闭 Matryoshka 时输出原生 768 维。
新：provider 内部完成截断，Matryoshka 不再包装；关闭 Matryoshka 时输出原生 768 维。
    二值签名始终取截断前的原生 768 维，与主向量共用同一次前向。
```

- [x] **Step 6: 更新 README 配置窗口说明**

在第 67 行 `「ANN 检索」对应 rag_ann_enabled，「自动构建」对应 rag_ann_auto_build。`
之后插入：

```text
紧跟在 ANN 开关之后是「二值粗筛」开关与候选数 Stepper：开关对应
`rag_binary_coarse_enabled`（默认开启），候选数对应
`rag_binary_coarse_candidates`（默认 200，50...1000，步进 50，关闭时禁用）。
二值可用时全局检索由它取代 ANN，不可用或定向检索时自动回退原流程；主存储为
BIN 时该功能停用，旧库需重新同步后才会生效。
```

- [x] **Step 7: 更新 README 存储设计**

在第 160 行「向量元素类型」bullet 与第 161 行「Matryoshka」bullet 之间插入：

```text
- **二值粗筛**：独立 `chunk_vectors_bin`（vec0 `bit[N]`，固定汉明距离，不能写
  `distance_metric`），rowid 与 `chunks` 严格一致；签名取 `x > 0` 的符号位、
  LSB-first 打包，位宽按模型原生维补齐到 8 的倍数，jina-v5 从截断前原生 768 维
  生成。查询先用汉明 KNN 取 top N（`rag_binary_coarse_candidates`，默认 200，
  50...1000，实际不低于 RRF 内层候选数且不超过 4096），再按 rowid 读主向量做余弦
  精排，最后与 FTS5 BM25 做 RRF 融合。二值可用时取代 ANN 召回；开关关闭、覆盖率
  不足、签名长度不符、provider 不支持、sqlite-vec 不可用、主类型 bin 或定向检索时
  完整回退原流程。bin 表是派生数据，维度/来源/量化版本变化时直接 DROP 重建；覆盖率
  用 `binary_rows` / `chunks_count` 计数器 O(1) 计算，不执行 `COUNT(*)`，相关 meta
  不进入主索引指纹，开关或截断维变化不会触发主库重建。
```

- [x] **Step 8: 更新 README 配置示例**

在第 232 行 `"rag_matryoshka_dim": 128,` 之后加入两行：

```json
  "rag_binary_coarse_enabled": true,
  "rag_binary_coarse_candidates": 200,
```

- [x] **Step 9: 计划自检**

先扫描占位符与任务结构：

```bash
grep -nE "TBD|TODO|implement later|fill in details|Similar to Task|适当|酌情|待补" \
  docs/superpowers/plans/2026-10-31-binary-coarse-retrieval.md | grep -v "grep -nE"
grep -c "^## Task" docs/superpowers/plans/2026-10-31-binary-coarse-retrieval.md
grep -c "^git commit -m" docs/superpowers/plans/2026-10-31-binary-coarse-retrieval.md
grep -c "^    func test" docs/superpowers/plans/2026-10-31-binary-coarse-retrieval.md
```

预期：第一条命令无输出（没有占位符；`| grep -v "grep -nE"` 排除命令自身的
匹配行）；任务数 `13`；提交次数 `13`；测试方法数 `47`。若数字不符，按实际差异
修正计划，不修改预期值蒙混过去。

再对照 spec 做覆盖检查，逐行确认没有遗漏：

| Spec 章节 | 覆盖任务 |
| --- | --- |
| 已确认决策 1–8 | Task 7–9（存储/ANN 旁路/回退）、Task 10（默认与范围）、Task 11（bin 旁路） |
| 非目标 | Task 2/5（签名只来自原生输出）、Task 9（只加查询探针，不改 ANN 构建）、Task 11（定向检索不加速） |
| 存储层：表结构 / meta / 覆盖率 / 生命周期 / 存储状态 | Task 7、Task 8 |
| 模型层：新协议 / jina-v5 原生维 / Matryoshka / 调度器 / 入库 | Task 2、3、4、5、6、11 |
| 查询流程：`RAGEngine.search` / `hybridSearch` / `SearchHit` / 候选数与降级 | Task 9、Task 11 |
| 配置 / 设置界面 / 状态与工具输出 / 错误处理 | Task 10、11、12 |
| 测试策略 1–6 | Task 1–12；回归与 README 基线由 Task 13 收口 |
| 实施顺序 1–7 | Task 1–13 一一对应 |

说明：spec 测试策略中写的 `MatryoshkaBinaryTests` 落在既有
`MatryoshkaEmbeddingTests.swift` 内（新增 2 个用例），不另建同名文件；spec 的
「`binary_score` 与精排序号」由 `VectorStoreBinarySearchTests` 覆盖。

最后核对类型与符号一致性，逐项确认：

- `BinaryEmbeddingOutput(vectors:signatures:)` 在 Task 2 定义，Task 4/6/11 使用；
- `EmbeddingBinarySigning` 的 `binaryDimension` / `binarySourceIdentifier` /
  `embedDocumentsWithSignatures` / `embedQueryWithSignature` 在 Task 2 定义，
  Task 3/4/5 实现或透传，Task 6/11 调用；
- `BinaryIndexStatus` 的 `storageEnabled` / `dimension` / `rows` / `chunkRows` /
  `coverage` / `isComplete` / `reason` 在 Task 7 定义，Task 11 读取；
- `BinaryCoarseOptions(signature:candidateLimit:)` 在 Task 7 定义，Task 11 构造，
  Task 9 消费；
- `VectorStore.add(chunks:vectors:binarySignatures:)` 在 Task 8 定义，Task 11 调用；
- `VectorStore.binaryIndexStatus()` 在 Task 7 定义，Task 11 转发；
- `VectorStore.hybridSearch(..., binary:ann:...)` 在 Task 9 定义，Task 11 调用；
- `SearchHit.binaryScore` 在 Task 2 定义，Task 9 赋值，Task 12 的 `rag_status`
  不直接读取该字段（按需在检索结果里展示）；
- 配置字段 `ragBinaryCoarseEnabled` / `ragBinaryCoarseCandidates` 与键
  `rag_binary_coarse_enabled` / `rag_binary_coarse_candidates` 在 Task 10 定义，
  Task 11/12 使用；
- `binaryVectorTable` / `binaryDimension` / `binaryRows` / `binaryStorageEnabled` /
  `binarySource` / `binaryStorageReason` / `annQueryCount` 均为 internal，Task 8/9
  的扩展文件可以直接访问。

- [x] **Step 10: 提交**

```bash
git add README.md \
        docs/superpowers/plans/2026-10-31-binary-coarse-retrieval.md
git commit -m "docs(rag): document binary coarse retrieval"
```

## 完成定义

- 13 个 Task 全部完成，且每个 Task 都有独立提交。
- 定向测试矩阵与全量 `swift test` 通过；README 的测试计数与全量输出一致。
- `rag_status` 能区分 `off` / `ready` / `incomplete` / `unsupported` /
  `bypassed_bin`，`ready` 时明确标注本次由二值粗筛取代 ANN。
- 旧库（无 bin 表）打开、检索、同步行为不变；重新同步后覆盖率升至 100%，
  两段式检索生效。
- 二值 `bit[N]` 表不参与 ANN 构建，`VectorStoreANN.swift` 只增加查询探针，
  构建、存储与维护逻辑保持不变。
