# 二值 ANN 粗筛与 stage 2 批读设计

**日期：** 2026-10-31

**目标：** 在已落地的「二值粗筛 + 主向量精排」两段式链路上解决两个瓶颈：

1. stage 1 的 `chunk_vectors_bin` KNN 仍是 sqlite-vec 暴力扫描，大库上没有真正加速；
2. stage 2 对 vec0 主表按 rowid 点查，在大库上近似线性扫描（1M 行 200 候选约 264ms）。

方案是新增一份**主向量镜像表** `chunk_vectors_rerank` 供 stage 2 顺序批读，并在二值签名上构建**自研分块 IVF ANN**（`bann_blocks` / `bann_cells`）。二值粗筛就绪时优先走 `bann`，未构建或不完整时回退现有 `chunk_vectors_bin` 暴力汉明，仍不可用时回退现有 ANN / 精确路径。

## 已确认决策

1. **Stage 2 用普通表镜像主向量。** 新增 `chunk_vectors_rerank(rowid INTEGER PRIMARY KEY, embedding BLOB)`，写入内容与主向量表完全相同的编码字节，同事务维护；`exactVectorsLocked` 优先读它，缺失 rowid 按 rowid 回退主向量表。
2. **二值 ANN 独立命名空间。** 新增 `bann_cells` / `bann_blocks` / `bann_*` meta，不复用 `ann_*` 表；复用现有 `ANNBlockBuffer`、`ANNClustering`、`ANNPacking`、`ANNCentroidCache`、`ANNTopKCollector` 的 bits 布局与算法。
3. **不改写主 ANN。** 现有 dense ANN 的构建、查询、维护逻辑保持不变；新增文件只做 bann 编排与签名流式读取，避免回归。
4. **回退链。** `bann` 可检索 → `bann`；`bann` 未构建/不可用 → 现有 `chunk_vectors_bin` vec0 KNN；两者都不可用 → 现有 ANN / 精确路径。
5. **构建入口复用。** 手动构建与后台自动构建仍走 `RAGEngine.buildANNIndex`；二值粗筛开启且签名覆盖完整时构建 `bann`，否则构建主 ANN。
6. **不新增配置项。** 复用 `ragANNClusters` / `ragANNClusterSize` / `ragANNProbe` / `ragANNRescore` / `ragANNMinChunks` / `ragANNAutoBuild`；bann 的 nprobe 按候选数与块规模内部派生。
7. **旧库不自动回填。** 打开旧库时创建空的 rerank 表，缺失部分回退 vec0；重新同步后 rerank 自动补齐。bann 可以从已有完整 bin 表直接构建，不要求重新同步。
8. **接受存储成本。** rerank 表约等于一份主向量（128 维 int8 为 128B/切片）；bann 约 8B + N/8 字节/切片（768 bit 为 104B/切片）。两者都是派生数据，可随时删除重建。

## 非目标

- 不改进 1-bit 签名的排序质量，不引入学习型量化、多 bit 签名或维度加权。
- 不自动回填旧库的 rerank 表，也不为旧库提供离线迁移工具。
- 不改变 `chunk_vectors_bin` 的 schema、签名编码规则、覆盖率语义和两段式检索的候选数配置。
- 不把 bann 用于 `sourceFilter` / `sourceContains` 定向检索。
- 不新增 bann 的簇数/块大小/nprobe 设置项，也不调整现有 ANN 的默认参数。
- 不并发构建多个索引，不做多线程查询。
- 不修改 reranker、LLM、OCR、Memory 等其它消费者。

## 总体架构

```
入库：主向量编码字节 ─┬─> chunk_vectors (vec0)
                      ├─> chunk_vectors_rerank (普通表，镜像字节)
                      └─> (可选) bann_blocks / bann_cells（从签名构建）

      二值签名 ──> chunk_vectors_bin (vec0 bit[N])

查询：查询签名
        │
        ├─ bann 就绪 ──> 簇心选簇 → 读 cell → 汉明 top 池（尾块签名补齐）
        ├─ bann 不可用 ─> chunk_vectors_bin vec0 KNN
        └─ 二者不可用 ─> 现有 ANN / 精确路径
        │
        └─ stage 2：候选 rowid 升序 → 每 256 条读 rerank 表
                     → 缺失回退主表 → 余弦精排 → RRF → reranker
```

设计要点：

- **rerank 表只解决读放大，不改变排序。** 字节与主表一致，`VectorEncoder.decode` 口径不变；即使 rerank 表为空，结果也与旧路径完全一致，只是慢。
- **bann 只影响 stage 1 候选集合。** stage 2 仍用主向量余弦重排，bann 的近似性不会被带入最终分数；`binaryScore` 保留 bann / vec0 的汉明相似度。
- **bann 是派生数据。** bin 表维度、来源、量化版本变化，或清空/重建向量库时，bann 一起丢弃；不做跨配置复用。
- **bann 构建依赖完整签名。** 只有 `binaryRows == chunks_count` 才允许构建；覆盖不完整时构建主 ANN，保证仍有可用的向量索引。

## 存储层：rerank 镜像表

### 表结构

```swift
public static let rerankVectorTable = "chunk_vectors_rerank"

CREATE TABLE IF NOT EXISTS chunk_vectors_rerank (
    rowid INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL
)
```

- 普通表，不包 `vec_int8()` / `vec_bit()`；直接存 `VectorEncoder.encode` 的原始字节。
- `rowid` 与 `chunks` / `chunk_vectors` 严格一致。
- 主元素类型为 `.bin` 时不启用（避免再复制一份位图）；已有表在切换到 `.bin` 时删除。

### 新增 meta 键

| key | 含义 |
| --- | --- |
| `rerank_rows` | 当前镜像行数，增量维护 |
| `rerank_dimension` | 镜像时的向量维度 |
| `rerank_element_type` | 镜像时的落盘元素类型 |

打开库时：

- 表缺失、列结构不符、meta 维度/类型不符、或 meta 缺失但表非空 → `DROP TABLE` 后按当前配置重建空表，`rerankRows = 0`。
- 当前元素类型为 `.bin` → 删除 rerank 表与相关 meta，`rerankStorageEnabled = false`。
- 旧库没有 rerank 表 → 创建空表，`rerankStorageEnabled = true`、`rerankRows = 0`；不执行回填。
- `rebuildReason != nil` 时仍可建空表，但检索侧不会读取（向量通道已停用）。

### 生命周期

- **写入：** `insertVector` 在主表写入成功后，用同一个 `blob` 写 rerank 表；两表在同一事务内提交。主表为 vec0 时仍用普通 `?` 绑定 rerank。
- **删除：** `deleteRowIDsLocked` 按批删除 rerank 行，用 `execute` 返回的变更数扣减 `rerankRows`。
- **清空：** `clear()` 删除 rerank 全部行并归零；需要重建向量表时连表一起重建。
- **重建 / 维度变化 / 空库采纳新配置：** 丢弃 rerank 表并按当前配置重建空表。
- **回滚：** `withChunkCounterRollback` 同时快照/恢复 `cachedChunkCount`、`binaryRows`、`rerankRows`。

### stage 2 读取

`exactVectorsLocked(rowIDs:)` 改为：

1. `rowIDs` 升序排序（提高 rowid B-tree 局部性，避免跨批来回跳）。
2. 每 256 条一批。
3. 若 `rerankStorageEnabled && rerankRows > 0`：先查 rerank 表，收集本批已找到的 rowid。
4. 本批缺失的 rowid 再按 256 条一批查主向量表（vec0 或普通表）。
5. 统一用 `VectorEncoder.decode(data, as: elementType, dimension: dimension)` 解码。

约束：

- rerank 表里的字节与主表一致，因此不需要额外版本字段；meta 维度/类型不符时整表丢弃。
- 缺失行回退主表保证旧库、部分同步、删除竞态下结果仍完整；绝不因为 rerank 缺行而少返回候选。
- 读取顺序变化不改变余弦分数与排序。

### 状态

`BinaryIndexStatus` / `RAGEngine.status()` 补充：

| 字段 | 含义 |
| --- | --- |
| `rerank_rows` | rerank 表当前行数 |
| `stage2_backend` | `rerank`（完整）/ `partial`（部分旧库）/ `vec0`（未启用或为空） |

判定：

```
complete ⇔ chunks_count > 0 && rerank_rows == chunks_count
partial  ⇔ rerank_rows > 0 && rerank_rows < chunks_count
vec0     ⇔ rerank_storage_enabled == false || rerank_rows == 0
```

## 存储层：二值 ANN（bann）

### 表结构

```swift
static let bannCellsTable = "bann_cells"
static let bannBlocksTable = "bann_blocks"
private var bannMetaPrefix: String { "bann_" }
static let bannFormatVersion = 2
```

```sql
CREATE TABLE IF NOT EXISTS bann_blocks (
    block_id INTEGER PRIMARY KEY,
    clusters INTEGER NOT NULL,
    dim INTEGER NOT NULL,
    payload BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS bann_cells (
    block_id INTEGER NOT NULL,
    cluster_id INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    count INTEGER NOT NULL,
    payload BLOB NOT NULL,
    PRIMARY KEY (block_id, cluster_id, seq)
) WITHOUT ROWID;
```

- 表结构、payload 布局与现有 `ann_*` 完全一致；`dim` 为签名位宽（如 768）。
- `bann_cells.payload` 每行 `[rowid Int64][位图 N/8]`，可直接复用 `ANNPacking.packCell` 的 bits 分支与 `VectorEncoder.hamming`。
- `bann_blocks.payload` 复用 `ANNPacking.packCentroids`（int8 簇心 + 尺度）。

### meta

| key | 含义 |
| --- | --- |
| `bann_ver` | 格式版本，当前 `2` |
| `bann_clusters` | 每块簇数 |
| `bann_block_rows` | 每块行数 |
| `bann_dim` | 签名位宽 |
| `bann_source` | `binarySource` |
| `bann_blocks` | 已写入块数 |
| `bann_rows` | 已索引签名行数 |
| `bann_stale_rows` | 水位之前被删除/替换的旧 rowid 数 |
| `bann_last_rowid` | 构建水位 |
| `bann_built_at` | 构建时间 |

新增内部结构 `BANNStoredMeta`，字段对应上表；`bannStoredMetaLocked()` 一次读取全部 `bann_*`。

### 构建

`buildBinaryANNIndex(clusters:clusterSize:iterations:progress:isCancelled:)`：

1. **前置校验：** `binaryStorageEnabled == true`、`binaryRows == count`、`binaryRows > 0`、`binaryDimension > 0`；否则抛 `AgentError.storage("二值签名覆盖不完整，请重新同步后再构建二值 ANN")`。
2. **参数对齐：** 版本、`binaryDimension`、`binarySource`、簇数、块大小任一与 meta 不符 → 重置索引。
3. **陈旧回收：** `bann_stale_rows / bann_rows >= 15%` → 重置后全量重建（阈值复用 `annStaleRebuildRatio`）。
4. **进度与断点：** 与 ANN 相同，`bann_blocks` / `bann_last_rowid` / `bann_rows` 记录断点；取消后再次调用从水位续建。
5. **取数：** `streamBinarySignatures(from:dim:)` 顺序读取 `chunk_vectors_bin` 的 `rowid, embedding`，按 rowid 攒块；乱序时复用 `ANNBlockBuffer.sortByRowID()` 兜底。
6. **块内训练：** `ANNBlockBuffer.reset(maxRows:dim:elementType:.bin)`；`ANNClustering.train/assign` 用 dequantize 后的 ±1/√N 向量；簇心归一化后由 `ANNPacking.packCentroids` 量化。
7. **落盘：** 单事务写 `bann_cells`（按 `cluster_id, seq` 连续）与 `bann_blocks`，再累加写 meta；失败整块回滚、水位不前进。
8. **锁粒度：** 准备与每块各自持锁，块间释放；progress 回调在锁外执行（与 ANN 相同）。
9. **结果：** 复用 `ANNBuildResult`，summary 前缀「二值 ANN 索引：」。

构建参数直接复用 `ragANNClusters`（默认 256）与 `ragANNClusterSize`（默认 512）；1M 切片约 8 块。

### 查询

`bannCandidatesLocked(signature:topK:probeOverride:)`：

1. **可用性：** bann meta 存在、`blocks > 0`、版本/位宽/来源匹配当前配置、bin 覆盖完整；否则返回 `nil`。
2. **尾块：** 水位之后的签名行数 `tail = max(binaryRows - indexedRows, COUNT(bin WHERE rowid > watermark LIMIT cap+1))`。读取预算沿用 128MB / `N/8`；`tail > cap` → 返回 `nil`，回退 vec0。
3. **簇选择：** 查询签名 `VectorEncoder.dequantizeBits` 还原为 ±1/√N 并归一化；`ANNCentroidCache.nearest` 每块选 nprobe 个簇。
4. **nprobe 派生：**
   ```
   rowsPerCluster = max(1, blockRows / clusters)
   base           = 2 + ceil(candidateLimit / rowsPerCluster)
   probe          = min(16, max(4, base) + (blocks <= 2 ? 4 : 0))
   ```
   默认 200 候选、512 行/簇：1-2 块探 8 簇，多块探 4 簇。测试可通过 `probeOverride` 强制全簇探测。
5. **cell 打分：** 每块一条 SQL 读取所有命中簇，按 `[rowid][bitmap]` 逐行 `hamming`，分数 `1 - hamming / N`，写入 `ANNTopKCollector`。
6. **候选池：** `capacity = min(4096, max(topK * 4, 512))`，为陈旧 rowid 留余量。
7. **尾块补齐：** 从 `bann_last_rowid` 之后逐行读签名并汉明打分，最多 `tail` 行。
8. **返回：** 降序取 `topK`；空池或异常 → `nil`（调用方回退 vec0 / ANN / 精确路径）。bann 内部不做余弦重打分。
9. **末块簇数：** 尾块行数少于配置簇数时，`bann_blocks` 记录实际簇数；加载簇心时按每块实际簇数解码，空缺簇不参与探测。

### 生命周期与一致性

- **删除：** `deleteRowIDsLocked` 在删除 bin 行的同时调用 `recordBANNStaleRowsLocked(rowIDs)`，只统计 `<= bann_last_rowid` 的行；随同一事务提交。
- **清空：** `clear()` 调用 `resetBANNIndexLocked()`（删数据、删 meta、清缓存），表不存在时用 `try?`。
- **重建 / 维度或元素类型变化：** `rebuildDimensionLocked`、`rebuildForCurrentConfiguration`、`adoptCurrentConfigurationLocked` 一并重置/删除 bann。
- **bin 配置变化：** `setupBinaryIndexLocked` 丢弃 bin 表时，同事务 `dropBinaryANNIndexLocked()`；避免 bann 继续引用旧签名。
- **显式删除：** `dropBinaryANNIndex()` 删除两张表与 meta，清 `bannCentroidCache`。
- **缓存：** 新增 `bannCentroidCache` 存储属性；构建一块、重置、删除后置空。
- **不删除主库数据：** bann 全部是派生数据，任何不一致都直接丢弃重建。

### 状态

`binaryANNIndexStatus()` 返回复用 `ANNIndexStatus`（不再新增一套结构），字段口径：

- `indexedRows` = `bann_rows`，`tailRows` 按签名水位计算，`unindexedRows` 超出补齐预算部分；
- `isReady` = 有块 + 版本/位宽/来源匹配 + tail == 0；
- `isUsable` = 有块 + 元数据匹配 + unindexed == 0；
- `needsRebuild` = 尾块超过四分之一预算或陈旧比例 >= 15%；
- `summary` 形如「二值 ANN 索引：96 块 / 7999 切片 / 768 bit，覆盖 100%」。

`RAGEngine.binaryANNIndexStatus(timeout:)` 走与 `annIndexStatus` 相同的超时只读入口。

## 检索流程

### `VectorStore.binaryCoarseCandidatesLocked`

在现有 precheck（非定向、库待重建检查、主类型非 bin、bin 存储可用、签名长度正确、`binaryRows == chunkRows`）之后：

1. **stage 1：**
   - 调 `bannCandidatesLocked(signature:topK:candidateLimit...)`；
   - 成功且非空 → 使用 bann 的 `(rowid, binaryScore)`；
   - 返回 nil → 走现有 `chunk_vectors_bin` KNN，把 `distance` 转成 `1 - distance / N`；
   - vec0 也没有候选 → `.skipped("二值粗筛无候选")`，由 `hybridSearch` 回退现有向量通道。
   - 若 bann 候选在 stage 2 全部失效（已删除），再回退一次 vec0 bin KNN，避免陈旧行把结果挤空。
2. **stage 2：** 对 stage 1 rowid 调 `exactVectorsLocked`（自动走 rerank 表），余弦降序取 `candidateCount`；同时保留每个 rowid 的 `binaryScore`。
3. **计数：** 新增内部计数器 `bannQueryCount`、`binaryVecQueryCount`，供测试确认走了哪条 stage 1；`annQueryCount` 在二值路径中保持 0。
4. **回退链不变：** 二值不可用 → `vectorCandidatesLocked(..., ann:)`；定向检索仍忽略 `binary`。

### 分数与输出

- `binaryScore` = stage 1 的汉明相似度（bann 与 vec0 同口径）；stage 2 不覆盖。
- `vectorScore` = 主向量余弦；`ftsScore` / `rrfScore` 语义不变。
- bann 的近似性只影响 stage 1 候选集合，不影响候选内排序。

## 构建策略与引擎接线

### 目标选择

新增公开类型：

```swift
public enum ANNIndexTarget: String, Sendable {
    case dense = "dense"
    case binary = "binary"
}
```

`RAGEngine.resolvedANNIndexTarget(timeout:)`：

- `ragBinaryCoarseEnabled == true`
- 且 `effectiveVectorElementType != .bin`
- 且 `binaryIndexStatus().isComplete == true`
- 且 `binaryIndexStatus().dimension > 0`

全部满足 → `.binary`；否则 `.dense`。

### 手动构建

`RAGEngine.buildANNIndex`：

- 保持现有签名与返回值类型；内部按 `resolvedANNIndexTarget` 调 `store.buildBinaryANNIndex` 或 `store.buildANNIndex`。
- 构建前记录 `annBuildTarget`，进度/日志/结果摘要明确写出「二值 ANN」或「ANN」。
- 不删除另一种索引：主 ANN 与 bann 可以共存；二值就绪时检索旁路主 ANN。

### 自动构建

`scheduleANNAutoBuild` / `runBackgroundANNAutoBuild`：

- 触发条件改为 `config.ragANNAutoBuild && (ragANNEnabled || ragBinaryCoarseEnabled)`；
- 切片数仍要求 `>= ragANNMinChunks`；
- 目标按 `resolvedANNIndexTarget`；
- 判断是否需要构建时使用对应目标的状态（bann 未建/尾块/陈旧）；
- 二值覆盖不完整时目标自动落到 `.dense`，不阻塞主 ANN 维护。

### 删除

- `VectorStore.dropBinaryANNIndex()` 删除 bann。
- `RAGEngine.dropANNIndex()` 同时删除主 ANN 与 bann，日志说明「ANN 与二值 ANN 索引已删除」；旧测试只断言主 ANN 消失，仍然成立。

## 状态、设置界面与工具输出

### `RAGEngine.status()` 新增字段

| 字段 | 取值 |
| --- | --- |
| `binary_ann` | `off` / `bypassed_bin` / `incomplete` / `not_built` / `building` / `ready` / `tail` / `needs_rebuild` / `unusable` |
| `binary_ann_index` | bann 状态摘要（人类可读） |
| `binary_ann_note` | 不可用/需重建原因 |
| `binary_ann_building` | 当前是否正在构建 bann |
| `ann_build_target` | 当前构建目标：`dense` / `binary` / `""` |
| `binary_backend` | `bann` / `sqlite-vec` / `none` |
| `rerank_rows` | rerank 表行数 |
| `stage2_backend` | `rerank` / `partial` / `vec0` |

`binary_ann` 判定优先级：

1. 二值粗筛关闭 → `off`；
2. 主元素类型 `.bin` → `bypassed_bin`；
3. 签名不支持/覆盖不完整 → `incomplete`（note 提示重新同步）；
4. 正在构建 → `building`；
5. bann 无块 → `not_built`；
6. `isUsable == false` 或元数据不符 → `unusable`；
7. `needsRebuild` → `needs_rebuild`；
8. `tailRows > 0` → `tail`（仍可用，尾块精确补齐）；
9. 其余 → `ready`。

`binary_backend`：

```
bann        ⇔ binary_ann ∈ {ready, tail}
sqlite-vec  ⇔ 二值粗筛 ready/incomplete 且 bin 表可用
none        ⇔ 其它
```

`ann_bypassed` 语义保持不变：二值粗筛 `ready` 时为 true，无论 stage 1 走 bann 还是 vec0。

### `RAGTool` 输出

在「二值粗筛」后新增一行：

- `ready`：`二值 ANN: 就绪（96 块 / 7999 切片，768 bit，覆盖 100%）`
- `tail`：`二值 ANN: 可用（尾块 120 切片，已精确补齐）`
- `not_built`：`二值 ANN: 未构建（本次回退 bin 表暴力汉明）`
- `building`：`二值 ANN: 构建中 42%`
- `needs_rebuild`：`二值 ANN: 建议重建（陈旧 18%）`
- `incomplete` / `unusable`：一行原因。
- 二值粗筛行补充 backend：`二值粗筛: 可用（bann，768 bit，覆盖 100%，本次取代 ANN）`。

### 设置界面

- `AppModel` 新增 `binaryANNStatusText`，`refreshRAGStatus` 读取 `binary_ann_index`；`requestANNIndexBuild` 先取 `resolvedANNIndexTarget()`，用对应状态做重建确认与文案。
- `SettingsView` 的索引状态区新增第二行「二值 ANN」，显示 bann 摘要；主 ANN 摘要保留。
- 构建按钮文案与帮助改为「构建/续建向量索引」：二值开启且覆盖完整时构建二值 ANN，否则构建主 ANN；构建进行中显示目标。
- 删除索引按钮说明同时删除两种索引。

## 错误处理

| 场景 | 行为 |
| --- | --- |
| rerank 表不存在/损坏/配置不符 | 打开库时 DROP 重建空表，`rerankRows = 0`，检索回退主表 |
| rerank 缺行（旧库/部分同步） | 该批回退主表读取，结果不缺失 |
| rerank 写入失败 | 与主向量/chunks 同事务回滚，计数器恢复 |
| 签名覆盖不完整时构建 bann | 抛 `AgentError.storage`，不写入任何块 |
| bin 表在 bann 构建前被重置 | `setupBinaryIndexLocked` 同事务删除 bann；构建下一轮重新开始 |
| bann 元数据版本/位宽/来源不符 | 视为不可用，回退 vec0；状态 `unusable` 并提示重建 |
| 尾块超出补齐预算 | bann 返回 nil，回退 vec0；状态 `unusable` |
| bann 查询异常 | 捕获并记录 warning，回退 vec0；不抛出到检索主流程 |
| bann 构建块失败 | 该块事务回滚，水位不前进；下次从断点续建 |
| 陈旧行比例 >= 15% | 构建时整体重置重建；状态 `needs_rebuild` |
| 定向检索 | 忽略二值，走原定向路径 |

所有 bann / rerank 数据都是派生数据，不一致时直接丢弃，不触碰 `chunks` 正文与用户文档。

## 测试策略（TDD）

先写失败测试，再实现；每步保持 `swift build` 通过。

### 1. rerank 镜像表

`VectorStoreRerankTests`

- 非 bin 类型打开库时创建表，meta 维度/类型正确；
- `add` 后 rerank 行数等于切片数，字节与主表一致；
- 删除主表行后 `exactVectorsLocked` 仍能从 rerank 读取（证明优先路径）；
- 删除 rerank 行后 `exactVectorsLocked` 回退主表并返回完整结果；
- 超过 256 个 rowid 分批读取，结果完整且与逐个读取一致；
- `deleteRowIDs` / `clear` / 重建 / 维度变化后行数归零；
- 元素类型切到 `.bin` 时表被删除；
- 事务失败时 `rerankRows` 随其它计数器回滚。

### 2. bann 存储与查询

`VectorStoreBinaryANNTests`

- 完整签名下构建成功，`binaryANNIndexStatus().isReady == true`，块数/行数/覆盖率正确；
- 签名覆盖不完整时 `buildBinaryANNIndex` 抛错且不产生块；
- 受控签名下 bann 查询返回汉明最近的 rowid，`binaryScore` 正确；
- `probeOverride` 全簇探测时与暴力汉明 topK 一致；
- bann 未构建时 `hybridSearch` 走 vec0，`binaryVecQueryCount > 0`、`bannQueryCount == 0`；
- bann 可用时 `bannQueryCount > 0`、`binaryVecQueryCount == 0`、`annQueryCount == 0`；
- 构建后新增切片进尾块，检索能看到；
- 删除来源后不再返回陈旧 rowid，陈旧计数正确；
- 陈旧比例 >= 15% 时重建回收；
- `clear` / `dropBinaryANNIndex` 后状态回到未构建；
- 重新打开库时 bin 来源/位宽变化，bann 被丢弃。
- bann 可用时 stage 2 使用 rerank 表（通过 `stage2_backend == "rerank"` 与结果验证）。

### 3. 引擎接线

`RAGEngineBinaryANNTests`

- 二值开启且覆盖完整时 `resolvedANNIndexTarget() == .binary`，手动构建产生 bann、不产生主 ANN 块；
- 二值关闭/覆盖不完整时目标 `.dense`，构建主 ANN；
- 自动构建按同一目标选择，`ragANNMinChunks` 生效；
- `status()` 输出 `binary_ann` / `binary_backend` / `stage2_backend` / `rerank_rows`；
- `dropANNIndex()` 同时删除主 ANN 与 bann；
- 检索走 bann 时 `binaryScore` 非空、`annQueryCount == 0`；
- `RAGToolBinaryStatusTests` 增加 bann 行与 backend 文案断言。

### 4. 回归

- `swift build`；
- 既有 ANN / 二值 / 引擎测试全量通过；
- 旧库（rerank 空表、bann 未建）打开、检索、同步行为不劣于原路径；
- 全量 XCTest 基线更新 README 中的用例数。

### 5. 性能与质量基准

使用现有未跟踪基准脚本（不提交）：

- `BinaryRetrievalScaleBenchmarkTests`：
  - 1M / 128 维主向量 / 768 bit：stage 2 ≤ 10ms（200 候选），二值总耗时 ≤ 精确检索的 50%；
  - 50k / 200k：bann + rerank 总耗时 ≤ 精确检索，且 stage 1 候选覆盖不低于 vec0 二值路径的 90%；
  - 记录 bann stage 1、stage 2、exact、vec0 二值四条延迟。
- `RealCorpusBinaryBenchmarkTests`（15 本、7999 切片）：
  - bann 路径总耗时 ≤ 现有 vec0 二值路径（约 5.5ms）；
  - 真实 recall@10 相比 vec0 二值路径下降不超过 3 个百分点，recall@50 下降不超过 5 个百分点；
  - coverage 仍为 1.0，`ann_bypassed == true`。

## 实施顺序

1. `VectorStoreRerankTests` + rerank 表 schema/meta/写入/删除/清空/重建/回退 + `exactVectorsLocked` 批读。
2. `VectorStoreBinaryANNTests`（存储/构建）+ `BANNStoredMeta`、schema、签名流式读取、块构建、状态、生命周期。
3. 同一测试文件补查询用例 + `bannCandidatesLocked`、簇心缓存、尾块补齐、stage 1 回退链与计数器。
4. `RAGEngineBinaryANNTests` + 构建目标选择、手动/自动构建、删除、状态字段、RAGTool 输出。
5. `AppModel` / `SettingsView` 状态与构建文案接线，编译验证。
6. 跑回归与真实/合成基准，记录数据，更新 README 用例数。
7. 全量测试通过后提交；未跟踪基准脚本不提交。

## 风险与取舍

- **额外存储：** rerank 使主向量占用约翻倍（128 维 int8 为 128B/切片）；bann 再增加约 104B/切片（768 bit）。这是用空间换 stage 2 的确定性提速；删除任一派生表都不影响正文。
- **旧库 stage 2 仍慢：** 未回填的旧库 `stage2_backend = partial/vec0`，需要重新同步或重建才能享受全部收益；bann 仍可先加速 stage 1。状态字段会明确提示，不静默。
- **bann 召回：** 二值 ANN 是有损近似；nprobe 派生以默认 200 候选为基准，极端密集近邻场景可能低于全量二值扫描。通过候选池 4 倍余量、stage 2 余弦重排、可配置候选数缓解；真实语料以 3/5 个百分点为验收线。
- **构建成本：** bann 的 k-means 成本与主 ANN 同量级，大库构建仍需数分钟；复用断点、陈旧阈值与后台构建，避免阻塞检索。
- **代码重复：** bann 不复用 `ann_*` 表名与全部查询函数，编排层会有一定重复；换取主 ANN 零回归和清晰的独立生命周期。共享的量化/聚类/打包/TopK 全部复用。
- **参数耦合：** bann 复用 ANN 的簇数/块大小配置；默认对 128 维 int8 友好，但极小库（几百切片）建 bann 不一定比 vec0 快，状态会如实展示，用户可不构建。
