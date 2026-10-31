# 二值粗筛两段式检索设计

**日期：** 2026-10-31

**目标：** 在 RAG 检索中引入「二值向量粗筛 + 主向量精排」的两段式流程。入库与查询在模型原生输出上额外产出一份符号位二值向量（Binary Quantization），位打包后写入独立的 `chunk_vectors_bin`；检索先用汉明距离极速召回 top N（默认 200），再读取主向量做余弦精排，随后与 FTS5 全局 BM25 做 RRF 融合，最后交给 reranker。二值粗筛可配置开关，默认开启；开启且可用时取代 ANN 成为向量召回通道。

## 已确认决策

1. **存储方案 A：** 使用独立 vec0 表 `chunk_vectors_bin`（`embedding bit[N]`），rowid 与 `chunks` 一一对应，不与主向量表 JOIN。
2. **取代 ANN：** 全局检索中，二值粗筛可用时完全不调用 ANN；不可用时完整回退现有 ANN / 精确扫描流程。定向检索（`sourceFilter` / `sourceContains`）仍走原路径。
3. **保留 FTS：** 全局 BM25 通道不变，向量精排结果与 FTS 候选做 RRF 融合。
4. **签名质量：** 二值签名必须来自模型**原生维输出**，不能从已截断或已量化的主向量派生；查询侧使用同一来源。
5. **不回填旧库：** 不对已有 int8 向量回填二值签名；检测到 bin 覆盖不完整时提示重新同步，两段式暂不生效。
6. **默认开启：** `ragBinaryCoarseEnabled` 默认 `true`，不考虑旧用户行为迁移。
7. **候选数可配：** 默认 200，允许 50...1000；实际取值不低于 RRF 内层候选数、不超过 sqlite-vec 的 k 上限 4096。
8. **主存储为 bin：** `ragVectorElementType == .bin` 时自动旁路两段式，状态提示「bin 主存储仅支持单阶段检索」。

## 非目标

- 不实现学习型量化、均值阈值或按维度加权的二值量化；首版只用符号位 `x > 0`。
- 不把二值粗筛用于 `sourceFilter` / `sourceContains` 定向检索。
- 不改变 ANN 索引的构建、存储与维护逻辑；ANN 仅作为二值不可用时的降级通道。
- 不把二值签名纳入主索引指纹；切换开关不触发主向量库重建。
- 不改动 reranker、LLM、OCR、Memory、SkillRetriever 等其它 `EmbeddingProvider` 消费者的行为。
- 不为旧库提供离线回填工具。

## 总体架构

```
入库：模型原生输出 ─┬─> 截断 + 归一化 ─> int8 / fp16 / fp32 / bin 主向量表
                    └─> 符号位量化 + 位打包 ─> chunk_vectors_bin (vec0 bit[N])

查询：查询原生输出 ─┬─> 主查询向量
                    └─> 查询二值签名
                         ↓
       stage 1：bin KNN（汉明距离）→ top N（默认 200）
                         ↓
       stage 2：按 rowid 读主向量 → 余弦精排 → candidateCount
                         ↓
       FTS5 全局 BM25 ──> RRF 融合 ──> hydrate ──> reranker ──> topN
```

设计要点：

- **符号位不受归一化影响。** L2 归一化只除以正数，不改变任何分量的符号；因此“原生输出先归一化再取符号”与“原生输出直接取符号”等价。唯一必须单独处理的是 provider 内部已经做了 Matryoshka 截断的 jina-v5（需要保留截断前的原生 768 维）。
- **bin 表是派生数据。** 可随时丢弃重建，不承载用户原始信息；维度、来源或量化版本变化时直接 DROP，不要求主库重建。
- **入库始终维护二值签名（主类型非 bin 且 provider 支持时）。** 开关只控制检索；这样用户之后开启开关无需重新同步。每切片额外成本约 `N/8` 字节（jina-v5 768 bit = 96B）。
- **定向检索不参与粗筛。** 单来源/来源子串检索仍使用原有 ANN 或精确扫描，保证定向召回语义不被二值候选池截断。

## 存储层

### 表结构

新增常量与虚拟表：

```swift
public static let binaryVectorTable = "chunk_vectors_bin"

CREATE VIRTUAL TABLE chunk_vectors_bin USING vec0(embedding bit[N])
```

- `N = binaryDimension`，为模型原生维补齐到 8 的倍数；jina-v5 为 768，nomic / jina-v3 为 768 / 1024。
- `rowid` 与 `chunks.rowid` 严格一致；主向量表与 bin 表分别在同一个事务内写入。
- 只有 `sqlite-vec` 可用、`N > 0` 且主元素类型不是 `.bin` 时才创建/维护 bin 表。

### 新增 meta 键

| key | 含义 |
| --- | --- |
| `binary_rows` | bin 表当前行数，用于 O(1) 计算覆盖率 |
| `binary_dimension` | 当前 bin 表位宽 |
| `binary_source` | 签名来源标识（`binarySourceIdentifier`） |
| `binary_quantizer_version` | 量化规则版本，首版为 `1` |

打开库时：

- `binary_dimension` / `binary_source` / `binary_quantizer_version` 任一与当前参数不符 → `DROP TABLE chunk_vectors_bin`，`binary_rows = 0`，状态变为「需重新同步」。
- bin 表存在但 meta 缺失 → 视为异常，DROP 并归零；不做 `COUNT(*)`。
- meta 与当前参数一致但表缺失 → 重新建空表并归零。
- `binaryDimension == 0`（provider 不支持或主类型为 bin）→ 不读写 bin 表，也不主动删除已有表；主类型切换本身会触发主库重建。

### 覆盖率

```
binary_coverage = chunks_count > 0
    ? min(1.0, Double(binary_rows) / Double(chunks_count))
    : 0
complete ⇔ chunks_count > 0 && binary_rows == chunks_count
```

`chunks_count` 与 `binary_rows` 都是增量维护的计数器，不在状态/检索热路径执行 `COUNT(*)`。

### 生命周期

- **写入：** `add(chunks:vectors:binarySignatures:)` 在现有单事务内，为每个切片用同一 rowid 插入主向量与 bin 行；`binary_rows += 实际插入的 bin 行数`。
- **删除：** `deleteRowIDsLocked` 在删除主向量时同步删除 bin 行，用 `sqlite3_changes()` 返回的实际删除行数扣减 `binary_rows`。
- **清空：** `clear()` 删除 bin 表数据，归零并写入 meta；可按当前配置重建空表。
- **重建：** `rebuildForCurrentConfiguration()`、`adoptCurrentConfigurationLocked()`、`rebuildDimensionLocked()` 丢弃 bin 表及相关 meta，再按当前参数重建。
- **回滚：** 现有 `withChunkCounterRollback` 扩展为同时快照 `cachedChunkCount` 与 `binaryRows`；事务失败时两个计数器一起恢复。
- **签名缺失：** 某次 `add` 没有收到签名（provider 不支持或中途降级）时，照常写主向量但不写 bin 行；覆盖率下降，检索自动回退原流程。

### 存储状态

`VectorStore` 暴露只读快照，不触发查询：

```swift
public struct BinaryIndexStatus: Sendable {
    public var storageEnabled: Bool   // sqlite-vec + bin 表存在 + N > 0
    public var dimension: Int
    public var rows: Int
    public var chunkRows: Int
    public var coverage: Double
    public var isComplete: Bool
    public var reason: String?
}
```

`RAGEngine` 结合配置与 provider 能力映射成最终状态（见「状态展示」）。

## 模型层

### 新协议

在 `RAG/Models.swift` 新增：

```swift
public struct BinaryEmbeddingOutput: Sendable {
    public var vectors: [[Float]]
    public var signatures: [Data]
}

public protocol EmbeddingBinarySigning: AnyObject, Sendable {
    /// 签名位宽：原生维补齐到 8 的倍数。
    var binaryDimension: Int { get }
    /// 签名来源标识；包装器必须透传底层值，避免 Matryoshka 维度变化误判。
    var binarySourceIdentifier: String { get }
    func embedDocumentsWithSignatures(_ texts: [String]) async throws -> BinaryEmbeddingOutput
    func embedQueryWithSignature(_ text: String) async throws -> (vector: [Float], signature: Data)
}
```

默认实现（`extension EmbeddingBinarySigning where Self: EmbeddingProvider`）：

- `binaryDimension = ceil(dimension / 8) * 8`
- `binarySourceIdentifier = identifier`
- 调用现有 `embedDocuments` / `embedQuery`，把返回向量视为原生向量，用 `VectorEncoder.encodeSignature(_:bitDimension:)` 打包。
- 适用于输出未经截断的 MLX 通用、JinaV3 MLX、llama.cpp，以及非 v5 的 ONNX。

`encodeSignature` 规则：

- 逐维 `bit = x > 0`，低位在前（LSB-first）。
- 输出长度固定为 `ceil(bitDimension / 8)` 字节；`bitDimension` 大于向量维时补零。
- 入库与查询共用同一函数，保证汉明距离口径一致。

### jina-v5 原生维

jina-v5 的 provider 内部会在截断到 128 / 256 / ... 后再归一化，必须改造为一次前向同时保留两份结果：

- `JinaV5MLXModel.encode` 返回 `(final: [[Float]], native: [[Float]])`：`native` 是 last-token pooling 后的完整 768 维（归一化不影响符号），`final` 是现有截断 + 重归一化结果。
- `JinaV5MLXEmbeddingProvider.embedDocumentsWithSignatures` 使用同一次 `encode` 结果生成签名；`embedDocuments` 仍只返回 `final`。
- `ONNXEmbeddingProvider.pool` 的 v5 分支在调用 `JinaV5Profile.truncateAndNormalize` 之前，把完整 hidden 行保存为签名来源；`[B, 768]` 输出直接使用该行。
- `binaryDimension = 768`，`binarySourceIdentifier = identifier`。

### Matryoshka 包装器

`MatryoshkaEmbedding` 实现 `EmbeddingBinarySigning`：

- `binaryDimension` / `binarySourceIdentifier` 优先透传底层 `EmbeddingBinarySigning`；底层不支持时用 `base.dimension` 补齐并返回 `base.identifier`。
- `embedDocumentsWithSignatures`：底层支持签名时，直接取 `(vectors, signatures)`，只对 vectors 做截断 + 重归一化；底层不支持时直接调用 `base.embedDocuments`（此时拿到的是截断前的完整向量），再用它生成签名，最后截断 vectors。
- `embedQueryWithSignature` 同理。
- 这样 nomic / jina-v3 的二值签名同样是原生维，不受 `ragMatryoshkaDim` 影响。

### 调度器透传

`AdaptiveEmbeddingScheduler` 新增：

```swift
public func embedDocumentsWithSignatures(
    _ texts: [String],
    progress: ((Int, Int) -> Void)?
) async throws -> BinaryEmbeddingOutput?
```

实现要求：

- 与现有 `embedDocuments(_:progress:)` 共用同一套「token 计数 → 大块优先排序 → 窗口切分 → 条数/token 双约束组批 → 原始下标回写」逻辑；只增加一个 `includeSignatures` 分支，禁止复制第二套调度算法。
- 底层是 `EmbeddingBinarySigning` 时逐批调用其二进制方法，并把 signatures 与 vectors 共用同一套两级下标映射（窗口内位置 → 全篇原始下标）回写；否则返回 `nil`。
- 同时暴露 `public var binaryDimension: Int?`，值为 `(base as? EmbeddingBinarySigning)?.binaryDimension`；调用方据此决定是否需要签名，避免“先失败再重跑一次向量化”。
- 现有 `embedDocuments` 行为、进度回调、自适应批大小与指标完全不变。

### 入库流程（RAGEngine）

1. 按现有逻辑创建 `AdaptiveEmbeddingScheduler`。
2. 当 `effectiveVectorElementType != .bin` 且 `scheduler.binaryDimension != nil` 时，调用 `scheduler.embedDocumentsWithSignatures(_:progress:)`；否则沿用 `scheduler.embedDocuments(_:progress:)`。
3. 拿到 `BinaryEmbeddingOutput` 后校验 `signatures.count == vectors.count`，再调用 `store.add(chunks:vectors:binarySignatures:)`。
4. 若调用方已确认支持但调度器返回 `nil`（理论上的竞态），回退一次普通 `embedDocuments`，且本次不写 bin 行；状态覆盖率随之下降。
5. 主类型为 `.bin`、provider 不支持或 `sqlite-vec` 不可用时，不生成也不写二值签名。

进度日志、取消检查、写库后的 manifest 更新保持现有顺序不变。

## 查询流程

### RAGEngine.search

1. 取 `signing = embedding as? EmbeddingBinarySigning`。
2. 当 `config.ragBinaryCoarseEnabled`、`signing != nil` 且 `effectiveVectorElementType != .bin` 时：
   - 调用 `signing.embedQueryWithSignature(query)`，一次前向同时得到查询主向量与签名；
   - 构造 `BinaryCoarseOptions(signature: signature, candidateLimit: config.ragBinaryCoarseCandidates)`。
3. 否则沿用 `embedding.embedQuery(query)`，`binary = nil`。
4. 后续 `hybridSearch` 增加参数：

```swift
public func hybridSearch(
    vector: [Float], query: String, topK: Int,
    sourceFilter: String? = nil,
    sourceContains: String? = nil,
    rrfK: Int = 60, overfetch: Int = 2,
    binary: BinaryCoarseOptions? = nil,
    ann: ANNQueryOptions? = nil,
    deadline: Date? = nil
) throws -> [SearchHit]
```

`BinaryCoarseOptions` 只携带查询签名与候选上限；存储可用性、覆盖率、维度与版本由 store 判定。

### VectorStore.hybridSearch

在写入锁内、建立读快照之前完成 stage 1 + stage 2：

1. **可用性判定**（全部满足才启用）：
   - `rebuildReason == nil`；
   - `elementType != .bin`；
   - `BinaryIndexStatus.storageEnabled && isComplete`；
   - `signature.count == ceil(binaryDimension / 8)`；
   - 本次不是 `sourceFilter` / `sourceContains` 定向检索。
2. **stage 1 粗筛：**
   - `k = min(4096, max(candidateLimit, candidateCount))`；
   - `SELECT rowid, distance FROM chunk_vectors_bin WHERE embedding MATCH vec_bit(?) AND k = ?`；
   - 汉明相似度 = `max(0, 1 - distance / N)`，写入 `SearchHit.binaryScore` 供调试。
3. **stage 2 精排：**
   - 用 `exactVectorsLocked(rowIDs:)` 按批读取候选主向量（int8 ÷127、fp16 解码、fp32 原样、bin 旁路）；
   - 与归一化查询向量算余弦，降序排序，取前 `candidateCount` 作为向量池。
4. **ANN 旁路：** 二值可用时完全不调用 `annCandidatesLocked`；二值不可用时保持现有 `vectorCandidatesLocked(..., ann:)`。
5. **FTS 与 RRF：** `ftsCandidates` 仍按 `candidateCount` 取全局 BM25，RRF 融合、hydrate、`vectorScore` / `ftsScore` / `rrfScore` 赋值规则不变。
6. **定向检索：** `sourceFilter` / `sourceContains` 直接走现有路径；即使传入了 `binary` 也忽略。

### SearchHit

`SearchHit` 新增可选字段：

```swift
/// stage 1 二值粗筛的汉明相似度；未走二值粗筛时为 nil。
public var binaryScore: Double?
```

`score`、`vectorScore`、`ftsScore`、`rrfScore` 语义不变。

### 候选数与降级

- 配置候选数只决定 stage 1 的召回深度；stage 2 始终收敛到 RRF 内层 `candidateCount`。
- 配置值小于 `candidateCount` 时按 `candidateCount` 生效，避免向量池小于融合所需。
- 覆盖不完整、签名长度不符、bin 表缺失、provider 不支持、sqlite-vec 不可用、主类型 bin：`binary` 参数被忽略，完整回退原流程。
- 回退原因记录日志并节流（同一 store 实例同一原因最多每 60 秒一次），避免高频检索刷屏。
- 二值粗筛本身不设“失败再精确扫描”的双重保险：stage 1 的召回取舍由候选数配置控制，避免一次查询承担两条全量路径。

## 配置

`Config.swift` 新增（放在 ANN 配置之前）：

```swift
/// 二值粗筛（两段式检索）开关，默认开启。
public var ragBinaryCoarseEnabled: Bool = true

/// 第一阶段二值粗筛候选数，默认 200，允许 50...1000。
public var ragBinaryCoarseCandidates: Int = 200
```

- 持久化键：`rag_binary_coarse_enabled`、`rag_binary_coarse_candidates`。
- 加载、保存与手改配置规范化时统一收口到 `50...1000`。
- 二值签名写入不受该开关影响。

## 设置界面

`SettingsView.swift` 在向量库 / ANN 区域新增：

- `Toggle("二值粗筛", isOn: $binaryCoarseEnabled)`；
- 候选数输入或 Stepper，范围 50...1000，步进 50，开关关闭时禁用；
- 帮助文案：
  - 开启后先用二值向量召回候选，再按主向量余弦精排，最后与全文检索 RRF 融合；
  - 可用时取代 ANN，不可用时自动回退原流程；
  - 主存储为 BIN 时仅支持单阶段检索；
  - 已有库需要重新同步后才会生效；
- `loadConfig` / `saveConfig` 读写新增两个字段，保存时收口候选数范围。

## 状态与工具输出

`RAGEngine.status()` 新增字段：

| 字段 | 取值 |
| --- | --- |
| `binary_coarse` | `off` / `ready` / `incomplete` / `unsupported` / `bypassed_bin` |
| `binary_dimension` | 当前生效签名位宽；provider 支持时取协议值，否则取已存元数据值或 0 |
| `binary_coverage` | 0...1 |
| `binary_rows` | 当前 bin 行数 |
| `binary_backend` | `sqlite-vec` / `none` |
| `binary_note` | 人类可读原因，如「覆盖 42%，请重新同步」 |
| `ann_bypassed` | 配置开启且存储 `ready` 时为 true（表示全局检索将由二值粗筛取代 ANN） |

状态判定优先级：

1. 开关关闭 → `off`；
2. 主元素类型 `.bin` → `bypassed_bin`（备注「bin 主存储仅支持单阶段检索」）；
3. provider 不支持签名或 sqlite-vec 不可用 → `unsupported`；
4. 覆盖率不完整 / 维度或来源不符 → `incomplete`（备注「请重新同步」）；
5. 以上都通过 → `ready`。

`bypassed_targeted` 不是全局状态，只用于单次查询的日志与内部判定（`sourceFilter` / `sourceContains` 请求即使携带 `binary` 也会被 store 忽略），不出现在 `status()` 输出中。

`RAGTool` 的 `rag_status` 输出对应行：

- `ready`：`二值粗筛: 可用（768 bit，覆盖 100%，本次取代 ANN）`
- `incomplete`：`二值粗筛: 不可用（覆盖 42%，请重新同步）`
- `bypassed_bin`：`二值粗筛: 停用（bin 主存储仅支持单阶段检索）`
- `off` / `unsupported`：一行原因说明。

ANN 原有字段保留，`ann_bypassed = true` 时明确说明本次由二值粗筛取代。

## 错误处理

| 场景 | 行为 |
| --- | --- |
| 签名字节数不等于 `ceil(N/8)` | `add` 抛存储错误，整个事务回滚，不产生半截 bin 数据 |
| 签名数量不等于切片数量 | 同上 |
| bin 表写入失败 | 同上，主向量与 chunks 一并回滚 |
| bin 表维度 / 来源 / 量化版本不符 | 打开库时 DROP bin 表并归零，状态 `incomplete` |
| 查询签名不可用或长度不符 | 忽略 `binary`，回退原流程，记录警告 |
| 覆盖不完整 | 不做部分检索，回退原流程；状态提示重新同步 |
| 定向检索 | 绕过二值粗筛，走原定向路径 |
| 主类型 bin / sqlite-vec 不可用 | 不创建/不维护 bin 表，状态明确提示 |

不静默删除主库数据；bin 表属于派生数据，可在不一致时直接丢弃。

## 测试策略

按 TDD 分层，先写失败测试再实现：

### 1. 签名编码

`BinarySignatureTests`

- 符号位判定 `x > 0`，0 与负数均为 0；
- LSB-first 位序与 `vec_quantize_binary` 一致；
- 非 8 倍数维度补齐字节；
- `bitDimension` 大于向量维时补零；
- 文档/查询签名对称，汉明距离可预期。

### 2. 存储层

`VectorStoreBinaryTests`

- 仅 `sqlite-vec` 可用、N > 0、主类型非 bin 时创建 bin 表；
- `add` 同事务写入主向量与 bin 行，rowid 一致；
- 签名字节数/数量异常时整体回滚；
- 删除切片同步删除 bin 行并扣减计数；
- `clear` / `rebuild` / 维度变化后 bin 表归零；
- 覆盖率计数器在写入/删除/重开后正确；
- meta 维度/来源/版本不符时 DROP 并置 `incomplete`；
- 旧库无 bin 表时 `storageEnabled == false`，原检索流程不受影响；
- 主类型 bin 时 `bypassed_bin`；
- `binary_score` 与精排序号符合预期。

### 3. 模型层

- `MatryoshkaBinaryTests`：截断维变化不改变签名，也不触发 bin 表重建（`binarySourceIdentifier` 不含 `#mrl` 后缀）；签名来自底层完整向量；
- provider 默认实现：未截断输出生成的签名与 `VectorEncoder.encodeSignature` 一致；
- jina-v5：一次 `encode` 同时返回 final 与 native；`native.count == 768`，签名从 native 生成；
- ONNX v5：截断前取签名；非 v5 走默认实现；
- `AdaptiveEmbeddingScheduler`：签名按原始下标回写，向量与签名数量一致；底层不支持时返回 nil 且 `embedDocuments` 行为不变。

### 4. 检索与引擎

`RAGEngineBinaryTests`

- 构造 300+ 切片与可控签名，验证 stage 1 只取配置候选数、stage 2 按主向量余弦重排；
- 二值可用时 ANN 不被调用（用 spy / 可观测的 ANN 状态）；
- FTS 候选仍参与 RRF；
- 覆盖不足、provider 不支持、主类型 bin、定向检索时完整回退；
- `binaryScore` / `vectorScore` / `rrfScore` 赋值正确；
- rerank 与最终 topN 行为不变。

### 5. 配置与界面

- 默认值（开启、200）；
- 加载/保存序列化往返；
- 50...1000 收口（含手改越界值）；
- 设置界面开关关闭时候选数禁用（由现有 Settings 测试或手动验证覆盖）。

### 6. 回归

- `swift build`；
- 定向测试；
- 全量测试（沿用现有 530 项基线，新增用例后更新 README 数字）；
- 旧库（无 bin 表）打开、检索、同步的行为不变。

## 实施顺序

1. `BinarySignatureTests` + `VectorEncoder.encodeSignature` + `BinaryEmbeddingOutput` / `EmbeddingBinarySigning` 协议与默认实现。
2. 各 provider 实现（MLX / JinaV3 MLX / llama.cpp / 非 v5 ONNX 默认；jina-v5 MLX 与 ONNX v5 原生维）。
3. `MatryoshkaEmbedding` 透传与 `AdaptiveEmbeddingScheduler` 签名调度。
4. `VectorStore` bin 表 schema、meta、写入/删除/清空/重建、覆盖率与状态。
5. `VectorStore.hybridSearch` 两阶段查询、ANN 旁路、`SearchHit.binaryScore`。
6. `RAGEngine` 查询/入库接线、Config、SettingsView、状态与 `RAGTool`。
7. README、测试基线更新，定向 + 全量验证。

## 风险与取舍

- **召回损失：** 二值粗筛是有损召回；top N 过小会漏掉相关切片。通过默认 200、可配 50...1000 以及 reranker 前的 RRF 缓解；用户可关闭开关回到原流程。
- **额外存储：** 每切片约 `N/8` 字节（jina-v5 96B）。相对主向量（int8 128B）约增加 75%；换来粗筛阶段的位运算级读放大。
- **jina-v5 改造风险：** 需要确保 `encode` 只做一次前向；签名不得触发第二次推理。
- **旧库体验：** 默认开启但覆盖率不足，会先回退原流程并提示重新同步；不会静默降级为低召回检索。
- **定向检索：** 单来源检索仍走 ANN/精确路径，不享受二值加速；保持原有召回语义优先。
- **ANN 仍然维护：** 覆盖不完整时 ANN 仍是后备；本设计不改变其构建策略，避免同步行为出现两套分支。
