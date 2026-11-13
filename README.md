Lightning-RAG

如果你有大量资料，希望有个轻量本地知识库，方便查询，并且还能脱离云端LLM服务器，永久本地离线免费使用，那这个软件就是你想要的，抛掉一大堆各种依赖，底层实现直接解析提取各类文件，整个软件体积只有127M，dmg只有39M，软件大小只有同类软件的1/20，向量存储磁盘空间占用可以压缩20倍，性能基本无损，打破了SQLite的功能限制，实现真正的ANN支持，可以存储不限量的文件，数据量大时，检索不会明显变慢，如果你喜欢尝鲜使用各种新的模型，可以在配置里随意切换各环节的模型，享受DIY的乐趣，随时体验最新模型的能力。

Lightning-RAG 一个使用swift实现的极简超轻量RAG，可以纯本地使用（2B以上模型即可，所有主流电脑都可以无压力运行，如果要使用agent能力，建议主模型4B以上，如果只是当知识库使用，可以直接使用2B模型，辅助模型建议2B（知识图谱抽取实测 0.8B 无法稳定输出完整关系行，只能抽出个别实体，不建议用于图谱构建；0.8B 对长文本摘要效果也偏弱，虽然已经使用了map-reduce方式处理长文本，但主要影响后台自动生成的全书摘要和章节摘要的质量，优点是速度约为 2B 的两倍、资源占用更少），并指定使用各环节使用的模型，swift直接实现针对pdf，epub，word，html等各类文件的解析、识别、抽取、清洗、格式化，也提供可单独使用的信息抽取工具，速度极快，可作为本地知识库使用，或阅读辅助工具使用，内置极简Agent，交互过程中进行多层信息压缩，把上下文控制在30k以内，可以从skill库学习和检索，并自动蒸馏使用过程中的经验形成新的skill，在使用中形成个性化的各项能力，具备成长性

macOS 原生自主 Agent。控制流是**三态自动机 + 五级上下文压缩**，能力层是工具集，经验层是事实抽取与 L4 蒸馏，知识层是 **SQLite + sqlite-vec + FTS5** 的本地 RAG + 本地知识图谱，支持五种检索策略（local / global / hybrid / mix / naive）。图谱构建支持逐切片实体/关系抽取、同名聚合与描述归纳、无向关系合并、UNKNOWN 占位、SwiftGraph 内存图与快照、增量更新；检索支持高低层关键词三路召回、实体/关系反查 chunk、轮询去重、度数/weight 合成排序与 token 预算；知识图谱未构建完成时自动回退 naive。

详细模块设计与事务/检索流程见 [软件设计文档](docs/软件设计文档.md)。

## 环境要求

- macOS 14+
- Swift 6.3+ 工具链（Xcode 26+）。依赖 `mlx-swift 0.32.3` 的 `Package.swift` 声明 `// swift-tools-version: 6.3`，SwiftPM 依赖解析会强制要求工具链 ≥ 6.3.0；旧工具链构建时报 `Dependencies could not be resolved because 'mlx-swift' >= 0.31.5 contains incompatible tools version (6.3.0)`（仅影响从源码构建，运行打包好的 DMG 不受影响）。本项目在 Swift 6.4 工具链验证
- 本地模型（可选）：
  - LLM：MLX-LM / MLX-VLM 模型目录、GGUF 文件、ONNX 解码器目录（`input_ids` + `logits`，可选 `past_key_values`）
  - OCR VLM：MLX 模型目录、GGUF + mmproj；ONNX VLM 暂无通用图适配
  - Embedding / Reranker：ONNX 模型目录（含 `tokenizer.json`）、MLX 模型目录或 GGUF 文件
  - 默认 Embedding 为 lite 版内置（clean 版首次启动自动下载）的 `jina-embeddings-v5-text-nano-mlx-int4`（EuroBERT-210M，int4），未配置时开箱可用；`jina-embeddings-v5-text-nano-onnx-int8` 为可选后端
  - 默认 Reranker 为 lite 版内置（clean 版首次启动自动下载）的 `jina-reranker-v2-base-multilingual-onnx`（XLM-R base，ONNX int8，`model_quantized.onnx` 约 267MB），配置留空时开箱可用；RAG 全链路（切片 / 向量 / 召回 / 重排）无需配置任何本地模型，只需配置主模型与辅助模型
  - 默认主模型 `models/qwen3.5-4b-agentic-coder-v4-4bit-mlx`；辅助模型默认 `models/Qwen3.5-2B-MLX-4bit`，但默认**不启用**，需要在配置窗口显式勾选「启用」
- 运行时已随包内置：
  - GGUF：llama.cpp 官方 XCFramework，进程内直接加载模型，不再启动 `llama-server`
  - ONNX：onnxruntime-swift，不再依赖 Python / `onnxruntime-genai`

## 构建与运行

```bash
swift build --disable-sandbox
swift run Lightning-RAG
```

> 如果出现 `swift-plugin-server` / 宏插件 `malformed response` 错误，是 SwiftPM 沙箱与 MLX 宏插件在部分工具链上的兼容问题；使用 `--disable-sandbox` 可绕过。正常环境可直接 `swift build`。
>
> 如果依赖解析时报 `contains incompatible tools version (6.3.0)`，说明本机 Swift 工具链低于 6.3（`swift --version` 确认），升级 Xcode 26+ 后重试。

运行测试：

```bash
swift test --disable-sandbox
```

## 打包

```bash
swift build -c release --disable-sandbox

# 两种版本（VARIANT 默认 clean）
VARIANT=clean scripts/package-app.sh release && VARIANT=clean scripts/package-dmg.sh
VARIANT=lite  scripts/package-app.sh release && VARIANT=lite  scripts/package-dmg.sh
```

| VARIANT | 内置模型 | DMG |
|---|---|---|
| `clean`（默认） | 无；potion 语义切片 / jina embedding / jina reranker 均在首次启动时从 hf-mirror 自动下载 | `dist/Lightning-RAG-1.3.9-clean.dmg` |
| `lite` | potion-multilingual-128M int4（语义切片）+ jina-embeddings-v5-text-nano-mlx-int4 + jina-reranker-v2-base-multilingual-onnx | `dist/Lightning-RAG-1.3.9-lite.dmg` |

两个版本共用 `dist/Lightning-RAG.app`（应用本体与显示名不变，只有内置模型不同），
因此需按「打包 App → 打包 DMG」的顺序逐个构建。两个版本都不含主模型、辅助模型
与 OCR，需要在配置中指定本地模型路径。`clean` 把体积压到最小（DMG 约 39MB），
适合网络良好的环境；`lite` 内置三个基础 RAG 模型（DMG 约 449MB），首次启动即可
离线使用语义切片、Embedding 与重排。版本号可通过 `VERSION` 环境变量或
`package-dmg.sh` 的参数覆盖。

测试覆盖（当前 1000+ 个 XCTest；真实模型 E2E 在本地权重存在时随每次测试自动运行，缺少权重或需要外网/GGUF 等环境的用例会跳过）：sqlite-vec 事务写入/回滚、两段式 KNN、二值签名编码与 LSB-first 位打包、
    bin 表同 rowid 写入与覆盖率计数器、两阶段粗筛精排与 ANN 旁路、
    FTS5 中文分词、RRF 混合检索、RAG 增量/删除清理、重排序 TopN 决定最终条数（关闭时回到调用方 top_k）、内置 reranker 的打包清单与默认配置解析（`localRerankerModel` / `localRerankerModelPath` 留空时仍回落到 App 内置 jina reranker）、EPUB/DOCX/HTML 解析、扫描版 EPUB 归一化 + OCR、PDF 文本层、Ovis 文本页快路径、NaturalLanguage 切片、切片参数一致性（模型窗口收口、检索呈现上限随切片放大、切片参数变化触发重建）、配置迁移与旧键兼容（含 `auto_preload_main` / `aux_enabled` 默认值、主/辅助 KV 复用与量化键的缺省跟随与读写往返）、Agent 工具循环、ask_user 续跑、checkpoint 同步、code_run 写护栏、L3 去重、大纲围栏，以及摘要溢出保留、代码围栏/中文换行/正文数字保留、技能索引入口回退、空库维度自愈、`models/` 相对路径解析，以及 KV 复用增量与完整 prompt 的逐 token 模板对齐、预热前缀对齐、清空对话时失效 KV cache；GGUF 模型路径/mmproj 解析、进程内 llama.cpp 框架加载、ONNX 缺少权重时的明确报错，合成 ONNX decoder 图的 past/present 管线，简单 HTML 表格转 Markdown、复杂表格保留 HTML、Markdown 语法/缩进保留、纯文本 HTML/实体/噪声清理、空行与空格密度压缩、跨目录副本与 content_hash 去重、手动向量去重保留最早入库切片并写副本 tombstone，以及切片缺失/旧版 manifest/失败页文件的增量重建。真实模型链路覆盖：jina-v5 的 `Query: `/`Document: ` 前缀 → 可配置 Matryoshka 维度（默认 128，可选 32/64/128/256/512/768）截断 → 重新 L2 归一化 → 用户选择的元素类型入库（默认 int8，128 维时 128B/行），同时从截断前原生
    768 维生成二值签名写入 `chunk_vectors_bin` → 查询 top1 与 `rag_tools(op="search")` 全链路；jina-v3 的切片 → embedding → 256 维 Matryoshka 截断 → 重新归一化 → int8 入库（256B/行）→ 查询 top1；nomic-int8 的 int8 + Matryoshka 512 召回与 fp32 全维（768）基线对比；以及内置 jina reranker 经 `LocalModelFactory` 加载 ONNX 后对相关段落打分（达到默认 0.45 阈值且高于强无关基线）；`RAGPipelineVerificationTests` 使用仓库 `models/`（可用 `EMBED_BENCH_MODELS_DIR` 指向 App 包模型）跑通 6 篇文档入库 → 22 切片 int8 → 混合检索 + RRF → 重排 → ANN → `rag_tools` 的 search / outline / status 全链路。

定时任务有独立测试文件 `ScheduledTaskTests`：每日/间隔规则与跨月滚动、`schedules.json` 原子往返与缺字段容错、变更通知、`DispatchSourceTimer` 到期只触发一次、停用不触发、`run_now` 立即执行并更新运行时间、`schedule_task` 的 CRUD 与非法参数拒绝、工具 schema 与确认策略、Agent 注册工具并在关闭时停止调度器，以及 `runIsolated` 的计划隔离与恢复。

HTML 与 Office 解析另有专项覆盖：HTML 保存页的本地 iframe / frameset 接力（取正文最多的一份）、big5 等老编码嗅探、依赖 JavaScript 的页面给出明确失败原因、含 `h1` 的 `<header>` 保留、裸文本节点与 `<li>` 内块级元素、结构化清洗档保留 URL/IP 与段落空行；DOCX 编号列表与 `w:br`/`xml:space`、脚注编号与末尾列表、`styles.xml` 样式名解析标题；XLSX 日期与百分比格式、注音跳过、多行单元格不拆表；PPTX 演讲者备注与段内换行。旧版 Office 另有专项覆盖：OLE2 复合文档扇区/FAT/miniFAT 读取与损坏文件拒绝、`.doc` 字号标题/粗斜体/中文、`.xls` BIFF8 共享字符串/日期/百分比/公式与多 sheet、`.ppt` 幻灯片顺序、标题正文、演讲者备注与 Windows-1252 文本，以及 PPT95/加密/空文档/记录超限的明确报错。独立 Markdown 转换另覆盖单文件输出、目录相对结构、同名后缀、单文件失败不中断整批、输出目录跳过与不支持格式拒绝。

知识图谱有独立测试文件（与长期记忆 memory graph 测试互不混用）：`KnowledgeGraphBuildTests`（实体/关系合并、快照、限额与收流保护）、`KnowledgeGraphVec0Tests`（vec0 与精确扫描一致、单表分区、旧库迁移）、`KnowledgeGraphIncrementalTests`（覆盖式更新、Section Context、三元组去重）、`KnowledgeGraphEndToEndTests`（多文档全链路 + 五种策略 + 新增文档增量）、`KnowledgeGraphExtractionBenchmarkTests`（缓存命中与并发抽取）、`KnowledgeGraphRealModelTests`（`KG_REAL_MODEL_TEST=1` 时用真实 2B/0.8B 验证抽取质量）。

## RAG MCP 服务（对外接入）

协议实现基于官方 [MCP Swift SDK](https://github.com/modelcontextprotocol/swift-sdk)
（vendor 在 `Vendor/mcp-swift-sdk/`，2025-11-25 规范：Streamable HTTP 传输、
协议版本协商、会话管理、SSE 流、Origin 防回绑校验均由官方实现承担）。
本机知识库通过 7 个 MCP 工具对外暴露：`ingest_document`、`delete_document`
（别名 `forget_context`）、`list_document`、`extract_metadata`、
`hybrid_search_with_rerank`、`get_document`——入库/检索/删除与 App 内功能
共享同一个知识库引擎，不影响原流程。

### Streamable HTTP（默认关闭）

配置窗口「高级设置 → RAG MCP 服务」：选择仅限本机 / 局域网 + 端口，启动后：

- 端点：`http://127.0.0.1:<port>/mcp`（局域网模式为 `http://0.0.0.0:<port>/mcp`）；
- 多会话：每个客户端 `initialize` 各自获得独立 `Mcp-Session-Id`，`DELETE` 显式终止；
- 安全（规范 MUST）：本机模式校验 Host/Origin 防 DNS rebinding，非本机来源 403；
  不支持的 `MCP-Protocol-Version` 返回 400；非 initialize 请求缺会话头 400、未知会话 404。

客户端（Claude Desktop / Claude Code 等支持 Streamable HTTP 的应用）接入示例：

```json
{
  "mcpServers": {
    "lightning-rag": {
      "url": "http://127.0.0.1:8080/mcp"
    }
  }
}
```

## 界面

- **主窗口**：左侧约 3/4 为对话区（流式正文、50ms 合并增量、自动滚动、工具调用气泡），底部居中 tokens 计数与输入框；右侧为任务计划（目标/步骤/当前进行中标记）、定时任务（规则/下次执行/上次结果，启停、立即执行、删除）与思考过程三段，下方状态栏显示模型、同步进度与 RAG 统计。模型协议里的 `<summary>…</summary>` 不进入对话正文，其内容转入右侧思考过程。
- 输入框 Enter 发送、Shift+Enter 换行；中文输入法正在组合候选词时 Enter 放行给 IME 确认候选，不会误发送或清空已输入内容。
- 底部状态栏按顺序显示「模型加载中… → 模型加载成功 → 模型预热中… → 预热结束 → 就绪」，加载/预热期间带进度指示并禁用发送；失败时显示「模型加载失败」或「预热失败」。
- **右上角按钮**：「配置」和「日志」各自打开独立窗口；日志窗口含日志/工具调用两个页签。
- **配置窗口（Agent 模型配置）**：
  - 主模型：本地模式 / OpenAI 兼容接口二选一，默认本地模式。
  - 本地模式：模型类型（MLX-LM / MLX-VLM / ONNX / GGUF）、本地路径与目录选择、`context`/`KV 上限`/`temp`/`top_p`/`min_p`/`top_k`/`repeat`/`batch`/超时/「思考」「KV复用」「KV量化」「Flash Attention」开关，以及「启动时预加载」复选框、「加载模型」「卸载」。`KV 上限` 对应 `kv_cache_max_tokens` / `aux_kv_cache_max_tokens`，设为 0 表示跟随 Context；`batch` 对应 `prompt_batch_size`（辅助模型为 `aux_prompt_batch_size`），控制 prefill 每次前向的 token 上限，`0` 表示使用 MLX 默认值。「思考」「KV复用」「KV量化」紧跟在参数后面同一行（窗口较窄时整组开关换到参数行下一行，仍同行显示），仅在消费它们的后端（MLX-LM / GGUF）出现：`KV复用` = `llm_kv_reuse`（辅助模型 `aux_llm_kv_reuse`），`KV量化` = `kv_cache_int8`（辅助模型 `aux_kv_cache_int8`）。`Flash Attention` = `flash_attention`（默认开启，主/辅共用）：MLX 使用未量化 KV 走 Metal fused attention、`GGUF` 强制 `flash_attn_type=ENABLED`；`GGUF` 可与 `KV量化` 叠加（llama.cpp 提供 `fa_q8_0` kernel，实测组合正常），`MLX` 因上游量化缓存不走 fused kernel，二者只能取其一（开启时 KV 量化自动停用）。M3+/M4 或纯注意力模型收益更明显。主/辅助两套开关各自独立：主模型默认思考开、复用开、量化开；辅助模型默认思考关、复用关（后台摘要/事实抽取用后即清，避免长 KV 常驻）、量化跟随主模型。主模型思考模式是主引擎推理与工具决策的前提，不能为了降低工具路由延迟而默认关闭；后台摘要/事实抽取始终请求级强制关闭。
  - API 模式：Base URL、API Key、模型下拉与「获取模型」，同一组参数与「应用配置」。
  - 辅助模型：标题后带「启用」复选框（对应 `aux_enabled`，默认关闭）。本地模式同主模型布局（含独立的「KV复用」「KV量化」开关），默认 `models/Qwen3.5-2B-MLX-4bit`；未启用时「加载模型 / 重新加载」按钮禁用。
  - 启动时按 `auto_preload_main`（默认开启）预加载主模型；辅助模型仅在 `aux_enabled=true` 时加载。关闭主模型启动预加载后，需在配置窗口手动点「加载模型」；同步结束后的模型恢复不受影响。未指定模型时使用默认主模型 `models/qwen3.5-4b-agentic-coder-v4-4bit-mlx`。加载会真正把参数应用到新创建的模型实例。
  - Agent 区底部「运行参数」行：「最大迭代」（`max_iterations`，默认 15，1...50），修改后点该行「应用配置」保存。
- **配置窗口（知识同步）**：`工作目录` 单独一行；`源目录`（原「知识库目录」）行内紧跟四个操作按钮：应用配置 / 同步知识 / 停止同步 / 向量去重（排查并清理向量库内重复的切片，保留入库时间最早的一份），按钮与目录选择在同一行排列（窗口较窄时不拆分）。同一区域提供 RAG 检索策略下拉（local / global / hybrid / mix / naive）与「构建知识图谱 / 停止构建 / 实体消歧」；图谱未构建完成时四种图谱策略不可选。Embedding 行 = 类型（FastEmbed / ONNX / MLX / GGUF）+ 本地路径 + 选择… +「语义切片」开关 /「切片大小」/「overlap」；Reranker 行 = 同布局 +「启用重排序」开关 /「TopN」/「候选池」/「阈值」。「启用重排序」对应 `rag_rerank_enabled`（默认开启），「TopN」对应 `rag_rerank_top_n`（默认 3，1...20），「候选池」对应 `rag_rerank_candidate_multiplier`（默认 3，1...5：候选池 = `max(调用方 top_k, TopN) × 倍数`），「阈值」对应 `rag_rerank_threshold`（默认 0.45，0...1），表示 reranker 之后保留的结果数量、候选池放大倍数与分数过滤下限；`rag_rerank_timeout_seconds`（默认 5s，0 = 不限制）控制 rerank 墙钟超时，超时回退 RRF 并只返回前 3 条。Reranker 行之后依次是「二值粗筛」开关与候选数 Stepper、`ANN设置` 分组标签、向量库与 ANN 索引合并行：向量元素类型（默认 int8；可选 int8 / fp32 / fp16 / bin）/ Matryoshka 模式与截断维（jina-v5 六档 32 / 64 / 128 / 256 / 512 / 768，其它模型保留 128 / 256 / 512） /「ANN 检索」与「自动构建」开关 / ANN 索引摘要与「构建 ANN 索引 / 停止构建 / 删除索引」，「重建向量库」按钮固定在该组行末。「ANN 检索」对应 `rag_ann_enabled`，「自动构建」对应 `rag_ann_auto_build`。「二值粗筛」开关对应 `rag_binary_coarse_enabled`（默认开启），候选数对应 `rag_binary_coarse_candidates`（默认 200，50...1000，步进 50，关闭时禁用）。二值可用时全局检索由它取代 ANN，不可用或定向检索时自动回退原流程；主存储为 BIN 时该功能停用，旧库需重新同步后才会生效。高级区保留 PDF/EPUB OCR 引擎、OCR 模型（MLX / GGUF / ONNX）、mmproj、归一化/二值化，以及「RAG 参数设置」中的「查询超时」（`rag_query_timeout_seconds`，默认 5s，0 = 关闭）和「同步参数设置」中的「启动时自动同步」（`rag_auto_ingest`，默认关闭；开启后每次启动会自动增量扫描知识库，首次全量同步可能长时间占用 embedding 模型与磁盘 IO）、「同步并发」（`rag_ingest_concurrency`）、「OCR 并发」（`rag_vision_concurrency`）、「MLX 缓存」（`mlx_cache_limit_mb`）与「MLX 上限」（`mlx_memory_limit_mb`，0 = 自动）。向量库待重建时，红色提示显示在该合并行下方。
  - 高级区的「PDF OCR 设置」行末提供「转换工具」按钮，打开独立窗口，把单个文件或整个目录批量转换为 Markdown。复用知识同步的解析器（PDF / EPUB / HTML / TXT / 图片，Office 支持现代格式 `.docx` / `.xlsx` / `.pptx` 与旧版 `.doc` / `.xls` / `.ppt`），沿用打开窗口时面板上的 OCR / PDF / 归一化设置快照；不写回 `config.json`，也不重建 Agent。目录递归时保留相对子目录结构，输出目录位于输入目录内会自动跳过；同批同名（如 `a.txt` 与 `a.html`）依次生成 `a-2.md`、`a-3.md`；单个文件失败不会中断整批。进度区显示当前文件、当前环节（解析 / 图片归一化 / 图片识别 / 清洗与 Markdown 转换）与总体进度，关闭窗口会取消仍在进行的转换。转换按文件体积分批并发：达到 32MB 的文件单独成批串行，其余文件按同步并发（最多 6）并行；PDF 解析按体积收紧页级 worker / 在途页并分批产图，内嵌大图先按目标尺寸降采样，超过 200MP 回退有界栅格化。转换只到写出 Markdown，不做切片、embedding 与入库。
- 确认弹窗默认按钮为「拒绝」；`ask_user` 的回答作为工具结果继续同一轮推理。
- RAG 状态在后台线程查询，不在主线程建库/COUNT；状态只读取来源总数与前 256 个来源名，大知识库不会把全部路径读入内存。连续应用配置只生效最后一次，旧同步完成后只重载它自己的引擎，不会影响新 Agent。

配置保存在 `~/.base-agent/config.json`，可在配置窗口修改；工作区默认 `~/base-agent-workspace`。

**自定义 system prompt**：首次启动会在 `<workspace>/assets/system_prompt.txt` 写出内置的 `BASE_SYSTEM_PROMPT`；之后每次启动读取该文件并替换内置提示词，可直接用文本编辑器调整模型行为规则，**重启（或重新应用配置）后生效**（会话期间保持不变，以免 system 前缀抖动使 KV cache 反复重建）。文件内容空白或不可读时本次回退内置提示词，且不会覆盖文件。它替换的是协议主体，L0 META-SOP 与 L1 全局索引仍按多级记忆机制追加在其后；轻量轮（纯寒暄/能力问询）使用内核的极简提示，不受该文件影响。内置提示词中的 `<summary>` 标签、`rag_tools(op="outline")` 起手式、工具状态词等是内核回注与测试的硬依赖，建议在原文基础上调整，不要整段删除。

**v1.3.9**：知识图谱抽取默认切到 JSON 格式并加固提示词。**格式可切**：`rag_kg_extraction_format` 默认 `json`（`entities` / `relationships` 数组），可改 `delimited` 兼容 0.8B 等小模型；两种格式缓存键隔离，配置文件中带 `_comment_*` 说明。**提示词加固**：正文代码围栏隔离、Section Context 标注不可信元数据、新增 relation/entity 与分隔符的 Wrong/Correct 反例、纯数字/年份不成实体、端点缺失提示补抽。**解析兜底**：4 段关系用 keywords 补描述，JSON 支持代码围栏与数组 keywords。**二次补抽**：首轮为空或末行残缺时追加一次只补漏请求。**崩溃修复**：图谱抽取并发默认改为 1（`rag_kg_max_concurrency`），修复 2B 四路并发导致 MLX 活跃内存 11GB、`mlx_async_eval` 触发 `EXC_BREAKPOINT` 的构建崩溃。**实测**：2B JSON 模式 19.5 秒抽到 7/7 实体、5 条有效关系；0.8B 仍不建议用于图谱构建。

**v1.3.8**：知识图谱抽取质量与性能收口。**限额可配**：单切片实体 20 / 关系 50 / 描述 100 字（`rag_kg_max_entities_per_chunk` / `rag_kg_max_relations_per_chunk` / `rag_kg_max_description_chars`），提示词、解析与落库三处同时生效；关系描述在归纳后再次截断。**收流保护**：模型输出 `<|COMPLETE|>` 后立即停止消费；同一实体对重复关系行累计达 12 条时提前取消生成（实测 2B 从复读到 6k+ 字符/123 秒缩到约 2.5k 字符/50 秒）。**抽取提速**：按切片内容哈希跨重建复用抽取结果、LLM 归纳结果缓存、支持独立会话的后端最多 4 路并发，阶段日志输出抽取/归纳/向量化耗时。**真机验证**：2B 抽取 4/4 核心实体与 5~6 条无向关系，0.8B 关系行字段残缺、有效关系为 0，不建议用于图谱构建。长期记忆侧同步补强事实抽取校验与邻域检索。

**v1.3.7**：新增 RAG 知识图谱能力与五种检索策略（local / global / hybrid / mix / naive）。图谱构建支持逐切片实体/关系抽取、同名聚合与描述归纳、无向关系合并、UNKNOWN 占位、SwiftGraph 内存图与快照、增量更新；实体/关系向量按 jina-v5 128 维重新归一化后 int8 写入 `kg_vectors` 与 vec0 单表 `kg_vectors_vec`；检索支持高低层关键词三路召回、实体/关系反查 chunk、轮询去重、度数/weight 合成排序与 token 预算；配置面板提供构建/停止/消歧与策略选择，未构建完成时自动回退 naive。

## 核心设计

### 1. 三态内核

`KernelLoop` 只有三个状态：`THINK → ACT → CONSOLIDATE`。LLM、工具分发、取消、日志全部通过 `KernelRuntime` 闭包注入，内核不依赖 Agent 与具体工具实现。

- **L0**：工具返回即捕获，硬上限 20k 字符。
- **L1**：注入前按预算裁剪工具结果（head + tail，HTML 标签白名单去噪，保留源码尖括号）。
- **L2**：旧轮次折叠为 `<summary>` / 工具名摘要。
- **L3**：超预算时折叠到 `raw_turns + 1` 轮，每 `kernelConsolidateEvery` 轮主动 consolidate（默认 8，构造时以 `kernelRawTurns` 为上钳位）；raw 窗口至少覆盖一个 consolidate 周期，避免每个用户回合都因窗口滑动而重建 KV cache；连续重复摘要按正文合并。
- **L4**：`start_long_term_update` 或长任务结束时用 LLM 蒸馏事实、经验与技能；自动续跑会跨段累计蒸馏请求，不会因分段而丢失。

### 2. 工具与能力层

工具由 `AgentToolFactory` 统一装配，按「合并后的组合工具 + 少量独立工具」组织（模型侧只看到下表的工具名，子操作通过 `op` 参数选择）：

| 工具 | 形式 | 子操作 |
| --- | --- | --- |
| `file_tools` | 组合 | `read` / `write` / `modify`（别名 `patch`）/ `list` / `search` |
| `web_scan` | 组合 | `scan`（抓取并提取正文，JS 页面自动渲染）/ `render`（WebKit 渲染、等待选择器、执行 JS、截图） |
| `long_memory` | 组合 | `graph`（实体 / 关系 / 观察）/ `search`（语义召回） |
| `skill_tools` | 组合 | `search` / `semantic_search` / `load` / `save` / `list` |
| `rag_tools` | 组合 | `search` / `outline` / `status` / `ingest` |
| `code_run` | 独立 | 受控进程内运行 Python |
| `shell_run` | 独立 | 受控 shell 命令 |
| `ask_user` | 独立 | 中断本轮并向用户提问 |
| `update_plan` / `update_working_checkpoint` / `start_long_term_update` | 独立 | 计划、工作便签、长期记忆蒸馏 |
| `schedule_task` | 独立 | `create` / `list` / `delete` / `enable` / `disable` / `run_now` |
| MCP 工具 | 动态 | `mcp_<server>_<tool>`（随服务端工具清单变化热更新） |

**RAG 检索策略（少而精）**：用户问书/文档内容时，先 `rag_tools(op="outline")` 获取全书/章节摘要与结构树；若大纲摘要已覆盖问题则直接引用、**不再检索**；整体性问题（全书大意/主题）只搜前言/引言/导论，针对性问题只搜目标章节（检索词带上章节标题，`source=书名`）。内核在每次 `rag_tools(op="outline")`（或每次 `run()`）之后对 `rag_tools(op="search")` 设**硬上限（默认 3 次）**：超出部分不再真正调用，直接预置失败并提示「基于已检索内容与大纲摘要直接综合作答」，彻底杜绝同一问题反复换关键词 scatter 检索。

**检索即展示（Retrieval-as-Presentation）**：`rag_tools(op="search")` 的全部召回切片不再全量进模型上下文——只有相关度最高的 1 条切片进上下文供模型总结作答（图谱策略下实体/关系仍全量进上下文，原文片段同样只留 1 条），其余切片通过 `AgentCallbacks.onRetrievedChunks` 附加到回复气泡后供用户点开查看（含模型参考的 top-1，星标高亮）。点击气泡在独立「引用切片」窗口显示切片原文，`⌘[` / `⌘]` 或左右按钮切换同一回复内的相邻切片，避免切片截断造成的阅读断裂。无 UI 消费方（RAG MCP 服务）仍沿用 `searchFormatted` 的全量切片行为。

#### 知识图谱（RAG Knowledge Graph）

RAG 知识图谱与长期记忆的 **memory graph 是两套完全独立的实现**：前者面向知识库文档、按切片抽取实体/关系并参与检索；后者是 Agent 的长期记忆，二者数据、存储与测试互不混用。

- **五种检索策略**（`rag_retrieval_strategy`）：`naive`（原文召回，图谱能力加入前的实现，默认）、`local`（命中实体拉一跳邻居与关联边）、`global`（命中关系反查两端实体与度数）、`hybrid`（实体 + 关系轮询交织）、`mix`（实体 + 关系 + 原文切片）。只有图谱构建状态为 `complete` 时其余四种才可在配置面板选择；内容变化或图谱未完成时自动回退 `naive`。
- **构建**：在已完成切片上由辅助模型逐片抽取 `entity<|#|>name<|#|>type<|#|>desc` 与 `relation<|#|>src<|#|>dst<|#|>keywords<|#|>desc`，注入所在章节层级作为 Section Context。单切片限额可配：实体 20、关系 50、描述 100 字（`rag_kg_max_entities_per_chunk` / `rag_kg_max_relations_per_chunk` / `rag_kg_max_description_chars`）；同名实体聚合、类型 `/` 合并、文件来源合并，描述在至少 2 条时由 LLM 归纳；无向关系按实体对合并，`weight = max(累加权重, 去重来源数)`；关系中缺失的端点落 UNKNOWN 占位实体。
- **向量**：实体文本 `name\ndescription`，关系文本 `keywords\tsrc\ndst\ndescription`，关系额外写入正/反序两条 id（`#rev`）；文本先按 1000 token 截断，jina-v5 取 128 维后重新 L2 归一化并量化 int8；查询侧同样加 `Query: ` 前缀并做相同变换。向量写入 `kg_vectors`（普通表 int8 BLOB）与单张 `kg_vectors_vec` vec0 表（`target_type` 分区键区分实体/关系），vec0 不可用时回退精确余弦扫描。
- **图谱与快照**：SwiftGraph 无向图承载内存图，节点/边保留来源 chunk 用于归因反查；快照 JSON 带 `schemaVersion` 与 DB `revision` 校验，1 秒防抖落盘；`kg_build_state` 记录 building / complete / stale。
- **增量更新**：只抽取 `kg_processed_chunks` 中不存在或内容哈希变化的切片，返回实体/关系变更集，内存图做局部 upsert，不整图重建；文档删除/内容变化会把图谱标记 stale。
- **检索**：查询先由 LLM 抽低层实体词与高层主题词，低层词向量查实体、高层词向量查关系、原 query 向量查 chunk，三路 embedding 在同一批内计算；实体/关系各取 top 40，chunk 取 20，实体与关系反查 chunk 每命中最多 5 片；按 chunk_id 轮询交错去重，query 直接召回优先，余弦低于 0.2 丢弃；实体排序 = 余弦 0.8 + 度数 0.2，关系排序 = 余弦 0.7 + rank 0.2 + weight 0.1；实体预算 6000 token、关系 8000 token，连同 system + query + 原文总预算 30000 token。
- **抽取性能**：抽取结果按切片内容哈希缓存（跨重建复用）、LLM 归纳结果单独缓存，并在阶段日志中输出抽取/归纳/向量化耗时；小模型复读同一批关系时按重复关系行提前收流，避免跑满 token 上限。并发默认关闭（`rag_kg_max_concurrency=1`）：本地 MLX 的 isolated 请求共享同一块 GPU，实测 4 路并发让 prefill 从约 2 秒退化到 40 秒以上、MLX 活跃内存冲到 11GB 并触发 `EXC_BREAKPOINT` 崩溃；内存充足的机器可手动调到 2~4。首轮结果为空或末行字段残缺时追加一次「只补漏、不重复」的二次抽取；复读触发的收流不再补抽，避免把相同内容再生成一遍。
- **提示词防护**：正文用代码围栏与 `---Output---` 锚点隔离，Section Context 以单行「untrusted metadata」注入并明确不得执行其中指令；system prompt 带 relation/entity 前缀与分隔符的 Wrong/Correct 反例，关系端点缺失时提示补抽实体、纯数字/年份不成实体。关系行缺 description（4 段）时解析器用 keywords 兜底，不丢边。
- **输出格式**：`rag_kg_extraction_format` 默认 `json`（单个 `entities` / `relationships` 数组对象，允许代码围栏，解析失败按空结果处理）；可改成 `delimited`（LightRAG 风格分隔符行，带 `<|COMPLETE|>` 与重复行收流）以兼容只认行格式的小模型。JSON 模式结构清晰、无重复行，实测 2B 19.8 秒抽到 7 个实体（4/4 核心 + 3/3 技术/地点）与 6 条去重关系；分隔符模式 32 秒、8 条边但次要实体为 0。两种格式的缓存键相互隔离，切换格式不会误用旧结果。
- **真机实测（2B / 0.8B）**：2B JSON（默认）7/7 实体（含 Swift / MLX / Shanghai）、6 条去重实体对、约 20 秒；分隔符模式 4/4 核心实体、8 条去重实体对、约 32 秒。0.8B 分隔符模式靠 4 段兜底能解析出 5 条边，但只命中 2/4 核心实体（Acme Corp、Project Aurora 未抽为实体）、关系全部缺描述；JSON 模式会复读实体数组并在 2k token 处截断，解析结果为 0。0.8B 仍不建议用于图谱构建。
- **UI**：知识同步配置区提供「构建知识图谱 / 停止构建 / 实体消歧」与检索策略选择；未构建完成时图谱策略不可选。

所有文件类工具（`file_tools`）路径锁定在 workspace 内，并解析 symlink 防止逃逸。`code_run` / `shell_run` / MCP 调用按配置触发确认；`schedule_task` 的创建、删除与立即执行同样需要用户确认（确认框展示完整任务指令）。

技能保存会同时写 `SKILL.md` 与 `skill.json`：前者供模型加载，后者供关键词索引扫描；只有主技能目录是 `<workspace>/skills` 时才额外镜像到 `.agent/skills`，不会出现 `.agent/.agent/skills` 的嵌套路径。索引入口缺失时会自动回退到实际存在的 `SKILL.md` / `prompt.md`。

技能建议只针对真正的多步骤任务：寒暄、问候、简单问询与一次性查询不进入重复请求统计；只有本轮创建过 ≥2 步的计划且所有步骤都已 `done`，或模型显式调用 `start_long_term_update`，才会提示固化。没有计划的对话也不会注入 `skill_tools(op="save")` 的周期性 TIPS，系统提示同样禁止对这类请求询问是否保存技能。

### 2.1 安全与资源

- Python `code_run` 通过 `sitecustomize` 注入写护栏：`builtins.open` / `io.open` / `os.open` 的写模式只能落在 workspace 内，同时放行 `subprocess` 的 int fd 管道。
- 子进程超时/取消时递归终止进程树（`pgrep -P`），避免 `npx`/shell 留下孤儿；失败时把 stdout/stderr 尾部并入错误，模型能看到真实报错。
- 风险判定覆盖裸 `~`、`/`、`..`、`$HOME`/`${HOME}` 等目标，且不会把 `rm -rf ./` 误判为工作区外操作。
- `web_scan` 首请求与每次重定向都拒绝 loopback/私网/link-local/云元数据地址（SSRF）；响应体上限 8MB、重定向 5 次，任务取消时立即中止下载。
- MCP stdio 客户端（官方 SDK `Client` + `StdioTransport`）：启动串行化；`shutdown()` 后不会被 `callTool` 复活；进程退出立即断开并使挂起请求失败；服务端 `tools/list_changed` 会同步更新工具注册表与 schema 缓存。
- `AtomicFile` 使用同目录临时文件 + 原子替换，不再「先删旧再 move」。
- 所有 `URLSession`（OpenAI/Web/MCP 之外的 HTTP 工具）在 Agent 关闭或配置切换时 `invalidateAndCancel`。
- MLX 空闲 buffer cache 上限夹在 256MB...2GB（`mlx_cache_limit_mb`，默认 1024MB）；实测 1GB 让 decode 快 5–8%，且仍接近 jina-v3 embedding 的复用 buffer 工作集（约 1.27GB），2GB 不会带来额外提速，只是多留空闲缓存。配置切换时清理模型实例缓存，避免旧权重常驻。卸载/切换路径不再同步调用 `MLX.Memory.clearCache()`，防止与仍在 GPU 线程执行的 `asyncEval` 并发释放 command encoder 导致 SIGSEGV。
- SQLite 事务只在 COMMIT/RELEASE 成功后递减嵌套深度，提交失败也保持深度非负，避免后续事务被误判为 savepoint。
- 流式回复每 50ms 合并一次；内容更新时滚动不再逐帧动画，日志窗口复用同一个 `DateFormatter`，长会话下重绘更稳定。
- LLM 流带 200ms 取消看门狗：服务端静默时用户点「终止」也能立即退出，不必等到请求超时。
- 知识图谱写盘防抖；裁剪只在真正超限时执行，向量缓存按文本去重并分批嵌入；切换 workspace 时释放旧共享实例。

### 2.2 定时任务（Scheduled Tasks）

用户可以直接在对话里说「每天早上 9 点帮我整理昨天的日志」「每 30 分钟检查一次服务状态」，模型会调用 `schedule_task(op="create", prompt=..., time="09:00" 或 every_minutes=30)` 建立任务：

- **规则**：`every_minutes`（1...10080 分钟）或 `time`（每天 `HH:mm`，本地时区）；`prompt` 必须自包含，到点后在没有当前对话的情况下交给 Agent 执行，可调用任意工具（`shell_run` / `code_run` / MCP 外部工具 / `file_tools` / `rag_tools` 等），因此支持「调用外部自定义工具或代码」。
- **持久化**：任务清单写入 `<workspace>/.agent/schedules.json`（`AtomicFile` 原子写，字段容错解码），启动时自动恢复；任务 id、标题、规则、启停状态、上次运行时间与结果摘要一并保存。
- **调度**：`ScheduleRunner` 用单个 `DispatchSourceTimer` 每秒扫描一次（串行队列），到期任务先落盘 `lastRunAt` 再回调，保证同一时间点不会被重复触发；`run_now` 也会先更新运行时间。
- **执行与展示**：App 在空闲时执行到期任务，忙时排队（用户消息优先）；执行前在对话内插入「⏰ 定时任务开始执行」系统气泡，结果以助手气泡流式输出。定时任务使用 `Agent.runIsolated`，运行前暂存并清空用户当前计划，结束后恢复，避免无人值守时误续跑用户遗留的步骤。
- **无人值守策略**：执行期间没有人工确认通道——需要确认的高风险工具按任务创建时用户批准的指令自动放行，每次放行都写入日志；`ask_user` 自动跳过。定时任务运行期间禁止再创建/修改/删除定时任务，防止自我调度循环。
- **管理**：右侧「定时任务」面板显示规则、下次执行时间与上次结果，支持启用/停用、立即执行与删除；也可在对话里用 `schedule_task(op="list"/"delete"/"enable"/"disable"/"run_now")` 管理。

已知行为：系统睡眠期间 timer 不触发，唤醒后补跑一次；规则仅支持「每 N 分钟」与「每天 HH:mm」两种。

### 3. RAG 文档解析

| 格式 | 实现 |
| --- | --- |
| PDF（默认 `ovis`） | 先走 PDFKit 文本层：有效字符 ≥ 50 的页面直接提取并清洗为 Markdown；文字页中检测到表格时只裁剪表格区域（300 DPI）→ 归一化 → OvisOCR2 转 Markdown → 拼回原位置；文本层不足的扫描页才整页渲染（200 DPI，最长边 1408）→ OvisOCR2。纯文字且无表格的 PDF 不加载 Ovis。 |
| PDF（可选 `vision`） | 同样先走 PDFKit 文本层（≥ 50）；扫描页 → 统一归一化（缩放/灰度/对比度/锐化/Otsu 二值化）→ Apple Vision `VNRecognizeTextRequest` |
| EPUB（文本） | ZipFoundation 解包 + AEXML 解析 container/OPF/NCX + SwiftSoup 清理 XHTML → Markdown（标题/列表/表格/代码块），保留 TOC 大纲 |
| EPUB（扫描版） | 按 spine/图片顺序提取整页图片 → 统一归一化（缩放/灰度/对比度/锐化/可选 Otsu）→ Apple Vision 或 OvisOCR2 OCR → 按页标记拼接，与 PDF 流程一致 |
| DOCX / XLSX / PPTX | ZipFoundation + AEXML 解析 OOXML，输出 Markdown 标题与表格 |
| HTML | SwiftSoup 提取标题/列表/表格并转 Markdown |
| TXT / MD / CSV / JSON | 原生解析并清洗；CSV 转 Markdown 表格；JSON/JSONL 包成 ```` ```json ```` 代码围栏；支持 UTF-8/UTF-16 BOM 与 GB18030 |

PDF 文本层对所有引擎共用一套有界提取（`PDFTextExtractor`）：图片 XObject 达到 16 的页面不再调用 `PDFPage.string`，改用 `CGPDFScanner` 按字体 + ToUnicode + 文本状态逐操作符提取，避开 CoreGraphics 文本行布局在图文混排页的病态空白区域分析（实测 383 页、319MB 的 PDF 中，第 18 页 392 张图会让单页内存涨到 42GB 且长时间不返回）；扫描器输出为空时自然进入 OCR，绝不回退 PDFKit。低图片页继续走 PDFKit，保留成熟的阅读顺序与分段行为；简单字体固定按单字节码解码，只有 Type0 使用 ToUnicode 的 codespace 宽度。此外针对带特殊 APP13/Exif 段、CoreGraphics 拒绝解码的扫描版 PDF（整页渲染可能全白），`CPDFShim` 在渲染结果被判定为近空白且页面含内嵌图片 XObject 时，自动回退到内嵌 JPEG 流级提取，避免全书被误判为空白页。

统一清洗（`DocumentCleaner`）会先暂存围栏代码块与行内代码（占位符使用私用区字符，不会被控制字符归一化破坏），再处理 Ovis OCR 输出的 HTML `<table>`：简单表格转成 GFM Markdown（兼容缺失闭合标签、`<br>`），而含 `colspan`/`rowspan` 合并单元格、嵌套表格、`caption`、多行表头或单元格内列表/图片/公式等无法无损表达的复杂表格会保留原始 HTML `<table>`，避免信息丢失。之后才清理标签/实体/噪声。HTML 解码会把 `&amp;`、`&nbsp;`、`&hellip;` 等还原为实际字符，而不是统一替换成空格；块级标签保留段落换行，内联标签直接删除；脚本/样式/SVG/iframe 等容器整块移除，属性与 CSS 残片单独清理。自动识别 Markdown 后，标题、列表、引用、链接、表格、分隔线、代码块与缩进都会被保留，纯文本则把链接压成文字并去掉 URL/邮箱。Unicode 层会删除 NBSP/零宽字符/软连字符/BOM/替换字符（U+FFFD）/变体选择符/私用区/非字符与孤立组合符；合并汉字之间的 OCR 空格（「新 中 新」→「新中新」）；把 `......` 归一为 `……`、压缩重复中文标点；连续装饰符号与符号密度过高的噪声行会丢弃，连续重复行只保留一条。中文硬换行在两侧都是 CJK 时不补空格；纯数字行只在页首/页尾 3 行内视为页码，正文中的年份/编号不会被误删。最后做密度压缩：去掉行尾空格，页标记、连续列表项、表格行、引用行之间不保留空行；短 OCR 行直接合并，长段落之间保留一个空行；标题、代码块与保留的 HTML 表格前后保留结构分隔。实测一本 2101 行的 OCR Markdown 从 1050 个空行压到 47 个。清洗规则变化会使 `pipeline_version` 递增，下一次同步会重建旧文件的索引。

`rag_pdf_engine` 与 `rag_epub_ocr_engine` 接受 `ovis` / `vision` / `hybrid`；旧配置中的 `docling` / `marker` 等未知值统一归一化为 `vision`（默认先走 Apple Vision，避免旧值把 OCR 默默带回纯 Ovis 路径）。`hybrid` 是 PDF 的推荐模式：扫描页按 8 页一批渲染，批内非表格页并发走 Apple Vision OCR（跨文件总量受全局信号量限制到 CPU 核心数），只有检测到表格线框或 Vision 文本过少的页面才串行交给 OvisOCR2 保留表格结构；纯文本页仍走 PDFKit 文本层，文本页中的表格继续交给 Ovis。文件级由 `rag_ingest_concurrency` 控制（默认 6、上限 6）；纯 `ovis` 模式默认创建两个 Ovis 实例，实际 OCR 并发为 2；`hybrid` 的 Vision 页可随文件 worker 并行。扫描版 EPUB 默认使用 `vision`（无需额外模型），选择 `ovis` 时需要可用的 OvisOCR2 权重；两条路径都会先做同一套图像归一化，`rag_ocr_preprocess` / `rag_ocr_binarize` 可分别关闭归一化与二值化。

OvisOCR2 的 MLX 权重使用 `model_type=qwen3_5`（Qwen3.5-0.8B 后训练），可直接复用 mlx-swift-lm 的 Qwen3.5 VLM 实现；真正未注册的 Ovis2 架构会明确报错。

### 4. 切片

`TextChunker` 使用 Apple **NaturalLanguage**（`NLTokenizer`）完成句子与词级切分：

1. 结构切片：按 Markdown / 中文章节 / 编号标题切分章节；
2. 语义切片：内置 `potion-multilingual-128M`（int4 ONNX，纯 CPU，无 512 token 位置上限）计算相邻句余弦，在相似度低谷处断开；模型不可用时退化为词重叠；
3. 递归回退：段落 → 换行 → 句号 → 空格 → token 窗口，严格遵守 token 预算；
4. 块间按 token 数添加 overlap（计入总预算）。

默认 `chunk_size=800 token`、`overlap=80 token`（10%，含在 800 内）。同步时优先用模型自带 `tokenizer.json` 做精确计数（不含特殊 token），分词器不可用时退回标定过的保守估算器（`EmbeddingTokenEstimator`）：纯中文约 1 字 ≈ 1 token，中英混排约 1.8 字符/token，英文约 3.5 字符/token；因此 800 token 大致对应 800 字中文或 1000–1400 字混排。默认 jina-embeddings-v5（与 jina-v3 一样）声明窗口为 8192 token，但 embedding 推理前会统一压到 1024 token，默认 800 token 的切片完整计算、不会被截断；若把切片大小调到超过推理输入上限（1024 token，或改用 bge 等更小的 512 窗口模型），同步时会自动压到该上限，overlap 同步压到上限的 10%，设置面板会显示「实际生效：N token」。所有切分路径都在句子结束处断开；旧配置里的默认 500/50 会在加载时自动迁移到 800/80，自定义组合保持不变。

切片参数（大小 / overlap / 语义切片开关）只有设置面板一个来源，控件就排在「知识同步配置」的 Embedding 行里（模型类型 / 本地路径之后）：切片环节统一读 `RAGEngine.resolvedChunkingOptions`，检索呈现的单条上限取 `max(配置值, 切片 token × 1.5 + 64)`（不再固定 1500 字符截短），而切片参数、embedding 后端与目录都进 manifest 的 ingest 签名，改任一项都会逐文件重建；「加载主/辅助模型」按钮在白名单（仅 LLM 参数）之外的配置有变化时会走完整 `applyConfig` 重建 Agent 与 RAGEngine，不会出现面板改了切片、运行时仍按旧快照切片。中途停止后已完成的文件保留，下次从剩余文件继续。

重排序开关、TopN、候选池与阈值都在 Reranker 行（模型类型 / 本地路径之后）：`rag_rerank_enabled` 默认开启，`rag_rerank_top_n` 默认 3（1...20）决定重排序后保留的结果数量，`rag_rerank_threshold` 默认 0.45（0...1，面板「阈值」）控制分数过滤下限——候选池按 `max(调用方 top_k, TopN) × rag_rerank_candidate_multiplier` 取（倍数默认 3，1...5，面板「候选池」），reranker 重新打分、按 `rag_rerank_threshold` 过滤后只返回分数最高的 TopN 条；`rag_rerank_timeout_seconds`（默认 5s，0 = 不限制）是 rerank 的墙钟超时，超时会取消模型调用并明确回退 RRF 排序，降级路径最多返回 RRF 前 3 条；关闭重排序时仍按调用方的 `top_k` 返回。

多文件同步默认使用 6 个 worker（`rag_ingest_concurrency`，1...6）：解析/OCR 与其它文件的切片、embedding、入库流水线重叠。embedding 仍由共享调度器串行进入模型，子批从 `rag_embedding_initial_batch` 开始（默认 16，运行时夹到 [下限, 上限]），按 `rag_embedding_batch_step` 步进（默认 2，1...16），范围由 `rag_embedding_min_batch`（默认 2，2...64）到 `rag_embedding_max_batch`（默认 32，2...256）；每 `rag_embedding_probe_batches` 批（默认 2）或 `rag_embedding_probe_seconds` 秒（默认 2）根据吞吐和进程内存自动上探/下探，进程 footprint 超过 `rag_embedding_memory_limit_mb`（默认 6144MB）立即下调；吞吐变化小于 `rag_embedding_throughput_tolerance`（默认 0.03 = ±3%）视为测量噪声，不调整。实测 bge 477 token 长块在 batch 4–16 已到吞吐平台，默认 32 兼顾短块；同 batch 重复测量的噪声可达 ±20%，需要更稳时可把容差调到 0.08–0.10。这 8 个子批参数都可在设置窗口「知识同步配置」的高级区同一行调整（顺序：下限、上限、步进、启动预设、探测批次、探测窗口、内存阈值、吞吐容差）。日志只保留每本书的开始、每分钟进度和完成汇总（含各阶段耗时与最终子批大小）。

### 5. SQLite + sqlite-vec 存储

`VectorStore` 的设计约束：

- **主表与向量表分离**：`chunks` 存正文/元数据，`chunk_vectors`（vec0）只存向量；
- **单事务写入**：FTS 预分词在写锁外按批并行完成，`chunks` + `chunks_fts` + `chunk_vectors` 在同一 `BEGIN IMMEDIATE` 事务内写入，失败整体回滚；
- **单事务删除**：先取 rowid，再按批删除三张表，仍在同一事务内；
- **两段式 KNN**：先查 `chunk_vectors` 拿 rowid，再回 `chunks` 按 rowid 取正文，绝不 JOIN 虚拟表；
- **过滤**：不依赖虚拟表过滤，统一 over-fetch 后在第二阶段过滤；定向检索自动放大召回；
- **混合检索**：FTS5（预分词，含 CJK bigram） + 向量，RRF 融合；
- **向量元素类型**：`rag_vector_element_type` 支持 int8（默认）/ fp32 / fp16 / bin；典型体积为 128 维 int8 128B/行（jina-v5 默认）、128 维 fp16 256B/行、256 维 bin 32B/行（汉明距离）。int8 用对称量化 `round(x×127)`，入库与查询都包 `vec_int8(?)`；fp16 是真正的 IEEE 754 half（每维 2 字节），因 sqlite-vec 0.1.9 没有 fp16 列类型，改用普通表存储 + 精确余弦扫描（`vector_backend=sqlite-exact`），ANN 索引仍可用（构建时解码为 Float 再量化）；bin 用符号位 + LSB-first 位图，查询包 `vec_bit(?)`。jina-v5 与其它模型一样使用用户配置的元素类型，不再强制 int8；`RAGEngine.effectiveVectorElementType` 仅在 provider 通过 `EmbeddingVectorElementConstraining` 声明硬约束时才覆盖（当前内置模型均无约束）；
- **二值粗筛**：独立 `chunk_vectors_bin`（vec0 `bit[N]`，固定汉明距离，不能写 `distance_metric`），rowid 与 `chunks` 严格一致；签名取 `x > 0` 的符号位、LSB-first 打包，位宽按模型原生维补齐到 8 的倍数，jina-v5 从截断前原生 768 维生成。查询先用汉明 KNN 取 top N（`rag_binary_coarse_candidates`，默认 200，50...1000，实际不低于 RRF 内层候选数且不超过 4096），再按 rowid 读主向量做余弦精排，最后与 FTS5 BM25 做 RRF 融合。二值可用时取代 ANN 召回；开关关闭、覆盖率不足、签名长度不符、provider 不支持、sqlite-vec 不可用、主类型 bin 或定向检索时完整回退原流程。bin 表是派生数据，维度/来源/量化版本变化时直接 DROP 重建；覆盖率用 `binary_rows` / `chunks_count` 计数器 O(1) 计算，不执行 `COUNT(*)`，相关 meta 不进入主索引指纹，开关或截断维变化不会触发主库重建。
- **Matryoshka**：默认 `rag_matryoshka=auto`（nomic / jina-v3 / jina-v5 启用），截断维由 `rag_matryoshka_dim` 指定（默认 128）。jina-v5 只接受模型声明的 32 / 64 / 128 / 256 / 512 / 768，由 provider 内部按配置维截断并重新归一化，`MatryoshkaEmbedding.wrap` 直接放行，不会二次截断；nomic / jina-v3 先截断，**强制重新归一化**，再按生效元素类型量化入库；查询侧执行完全相同的变换。关闭 Matryoshka 时 jina-v5 输出原生 768 维，其它模型不截断；
- sqlite-vec 静态编入进程（`SQLITE_CORE` 模式），无需安装扩展；若初始化失败则显式退化为精确余弦扫描，并在 `rag_tools(op="status")` 中标注。

向量维度、元素类型与索引指纹（`embedding.identifier | dimension | matryoshkaDim | elementType`）写入 `meta`。配置变化时不再自动清空旧索引：非空库标记待重建，向量检索暂停、同步拒绝写入、全文检索仍可用，用户在设置面板执行「重建向量库」后才清空并按当前配置全量重新同步；空库直接采纳新配置。若向量库被删除但 manifest 残留，同步会自动识别并全量重建。空库遇到首次推理才发现的真实维度时会按实际输出维度重建。

检索走只读 WAL 快照：向量候选在写入锁内取，FTS 与主表 hydrate 在同一读事务内完成，写入/删除在快照外可并发提交，不再整段持锁；FTS 与向量两路各自 over-fetch。解析引擎、OCR 模型、切片参数或 embedding 模型变化会按文件写入 ingest 签名：下一次同步只重建签名不一致的文件，已完成的文件不会重复解析。每个文件入库后立即落盘 manifest，用户中途「停止同步」或进程退出后，下次同步直接从剩余未完成文件继续。单个文件变更后解析失败会清掉它的旧索引并在下次重试。

manifest 条目同时记录 `chunkCount` 与 `complete`：增量跳过前会校验库内该来源的实际切片数、Markdown 缓存是否存在；数量不一致、旧版条目缺少完成标记、或有失败页的文件都会重新处理，不再出现「实际没完成却被跳过」的情况。有 OCR 失败页的文件仍保留已成功内容，但标记为未完整完成，下次同步自动重试。

同步开始会先按「文件大小 + 采样指纹」分组，同一本书放在不同目录时只解析一个代表文件；只有大小相同的文件才计算采样指纹，避免对绝大多数唯一文件做无谓读取。同步完成后按清洗正文的 `content_hash` 再做一次全局去重，保留一个来源，删除其余来源的切片、摘要与 Markdown 缓存。被清理的正文副本会写 manifest tombstone（`duplicateOf`），后续增量直接跳过，不会每轮重复解析。知识库根目录统一用 `realpath` 规范化，避免 `/var` 与 `/private/var` 等 firmlink 导致相对路径和 manifest 键不一致。

知识同步配置区的「向量去重」可随时手动排查向量库：按切片正文 SHA256 分组，每组只保留入库时间最早的一份，单事务删除重复的 `chunks` / `chunks_fts` / `chunk_vectors`；整篇被清空的来源写 `duplicateOf` tombstone 并删除 documents/Markdown 缓存，后续增量同步直接跳过，避免重复向量继续制造检索噪声。去重期间与同步、模型加载/卸载、对话发送互斥，结束后自动刷新 RAG 统计。

### 6. 本地模型

- LLM / VLM：**mlx-swift-lm**（`MLXLLM` / `MLXLMCommon` / `MLXVLM`）、**llama.cpp C API**（GGUF）与 **onnxruntime-swift**（ONNX）。
  - MLX-LM / MLX-VLM：原生内存内推理；
  - GGUF：`LlamaCppLLMService` / `LlamaCppVLMService` 在进程内直接加载权重（Metal GPU offload、Jinja chat template、工具调用、mtmd 多模态），不启动 `llama-server`；Embedding 使用 pooled embedding，Reranker 要求 RANK pooling；
  - ONNX LLM：`ONNXLLMService` 用 onnxruntime-swift 直接运行 decoder 图（`input_ids` / `attention_mask` / `position_ids` + `past_key_values`），支持流式生成与工具调用；ONNX VLM 因缺少通用图适配暂未接入，会明确报错而不是回退 Python。
  - **Spark-X2.6（自定义架构 `model_type=spark2_5`）**：mlx-swift-lm 内置表没有该架构，应用内 `Sources/BaseAgentCore/Local/SparkModel.swift` 手写配置解码与计算图（headwise sigmoid 输出门、逐层双 RoPE、sliding(512)/full 混合注意力、GEGLU MLP、tied 量化 embedding），并在 `LLMLoadState.container` 加载前注册 `LLMTypeRegistry.shared`。配置窗口把主模型类型选 **MLX-LM**、路径直接填模型目录（如 `/Volumes/SSD/Spark-X2.6-4B-MLX-4bit`）即可。chat template 的 `tojson(separators=(...))` 是 HF Jinja 语法、`swift-jinja` 不支持，`ChatTemplateCompatibility` 会在加载 tokenizer 时自动去掉该关键字参数（仅影响 JSON 空白）；工具调用由 `tokenizer_config.json` 的 `tool_parser_type=glm47` 映射到 `.glm4` 解析，应用侧文本兜底额外兼容零参数的 `<tool_call>name</tool_call>`（上游 `GLM4ToolCallParser` 要求至少一个 `<arg_key>`）。4bit 权重真机验证：加载约 8–11s、thinking 模式工具调用参数正确、跨轮 `token-reuse` 生效。
  - **Spark-X2.6 GGUF**：`/Volumes/SSD/gguf/Spark-X2.6-4B.i1-Q4_K_M.gguf` 这类单文件由 `GGUFFileResolver` 直接解析，llama.cpp b11815 原生识别 `general.architecture=spark2_5`（sliding/full 混合注意力），配置窗口把主模型类型选 **GGUF**、路径填该 `.gguf` 即可。GGUF 元数据内的 `tokenizer.chat_template` 同样含 `tojson(separators=...)`，`LlamaCppModel` 加载时复用 `ChatTemplateCompatibility.strippingSeparators`；工具调用走 GGUF / ONNX / MLX 共用的 `LocalToolCallParser.glm4Payload`（GLM47 `<arg_key>/<arg_value>`，含零参数形式）。真机验证：加载生成、thinking 流式切分、工具调用参数、跨轮 `seq_rm` trim、FA + q8_0 KV、warmup 全流程均通过。
- Embedding / Reranker：**onnxruntime-swift**（`OnnxRuntimeBindings`）+ `onnxruntime-swift-package-manager`，或 **mlx-swift-lm** 的 `MLXEmbedders` / `MLXRerankers`，或 GGUF（llama.cpp pooled embedding / RANK pooling）。类型支持 ONNX / FastEmbed / MLX / GGUF。
  - jina-embeddings-v5-text-nano（默认）：MLX int4 与可选 ONNX int8 都不在通用工厂中。`JinaV5MLXModel` 直接读取 `model.safetensors`（兼容 `weights.safetensors`）推理 EuroBERT-210M（双向 attention + last-token pooling）；ONNX 图内完成 last-token + L2。两类后端按 `rag_matryoshka_dim`（默认 128，六档 32 / 64 / 128 / 256 / 512 / 768）截断、**重新 L2 归一化**，元素类型由用户配置（默认 int8，可选 fp32 / fp16 / bin）；文档切片前缀 `Document: `，检索查询前缀 `Query: `（provider 内部只加一次）；provider 内部完成截断，Matryoshka 不再包装；关闭 Matryoshka 时输出原生 768 维。
    二值签名始终取截断前的原生 768 维，与主向量共用同一次前向。
  - jina-embeddings-v3：MLX unified 权重（RoPE + task LoRA）不在 `MLXEmbedders` 默认工厂中，由 `JinaV3MLXModel` 直接读取 `weights.safetensors` 推理；ONNX 单文件用标量 `task_id` 选择任务。两类后端统一加 `passage: ` / `query: ` 前缀，文档切片用 `retrieval.passage`（task_id=1），检索用 `retrieval.query`（task_id=0）。
  - **输入上限 1024 token**：模型声明窗口（jina-v5 / jina-v3 8192、GGUF 2048）远大于默认 800 token 的切片，所有本地 embedding 后端在推理前都统一压到 1024 token（含特殊 token；模型窗口更小时按更小值），切片收口也读同一上限，避免为用不到的窗口容量做无效计算。
- Tokenizer：**swift-transformers** 的 `Tokenizers` 模块，读取模型目录内 `tokenizer.json`。
- OpenAI 兼容 SSE 按事件边界聚合多行 `data:`（服务端把 JSON 拆成多行时不会丢增量），并忽略注释/心跳行。
- 本地 MLX 的 `maxTokens` 会按「contextWindow - prompt 估算」动态收紧；文本兜底工具调用只接受本次请求声明过的工具名。
- **跨轮 KV cache 复用**：首轮完整 prefill 后保留 raw `[KVCache]` 与 `LMOutput.State`；后续轮次只把新增的 user/tool 消息追加进缓存（已生成的 assistant 文本本就在缓存里），不会每轮重算上万 token 的历史。tool 结果会改写成 `<tool_response>` 包裹的 user 消息再渲染，保证只渲染增量时与完整 prompt 逐 token 对齐（否则会漏掉 `<|im_start|>user` 头，上下文错位导致输出异常）。Qwen3.5 的 Mamba/GDN 缓存不可裁剪，因此采用 append-only；当累计 token 接近上下文上限时回退为一次完整重建。历史折叠/回滚、换模型、清空对话（`Agent.reset`）、调用失败或取消都会清空缓存，下一轮自动重建。
- **加载后预热**：主模型加载成功后，立即用稳定的 `BASE_SYSTEM_PROMPT` + 当前工具清单渲染出不带 generation prompt 的 system 前缀（当前约 4700 tokens），只做 prefill 并保留 cache/state。首次正式请求只需追加 user 增量（实测 `cached=4736 delta=216`），不再等整段 prompt 重算。预热前会先等待 MCP 连接完成；工具清单变化时按 `tools` 指纹自动重建，不会带错前缀。GGUF 后端同样会预热 system + tools + user 头前缀；实测 `cached=328/341`。Qwen3.5 的混合 SSM 层不支持按位置裁剪 KV，因此仅当前缀完全一致时续写，否则整段重建，不会输出错位上下文。
- **重建前缀复用**：历史折叠、raw 窗口滑动、rollback 导致消息前缀变化时，从预热时保留的 system+tools 快照复制 KV cache，只 prefill 动态历史；system 前缀不再重复计算。实测 5471-token 重建从约 34s 降到 5.3s（805 token），另一次 626-token 重建约 4.1s。
- **prefill 批量**：`prompt_batch_size` / `aux_prompt_batch_size`（默认 512，最小 64）映射到 `PrefillParameters.stepSize`，用于控制长 prompt prefill 每次前向的 token 数；调大通常能提高吞吐，但会占用更多内存。配置窗口的 `batch` 字段与 JSON 键一一对应。
- **投机解码**：当前主/辅模型均为 Qwen3.5 hybrid（GDN/Mamba 层），`MambaCache.isTrimmable == false`，mlx-swift-lm 的 `SpeculativeTokenIterator` 会在初始化时直接拒绝；两个模型的实际权重中也没有 MTP 头（config 虽写 `mtp_num_hidden_layers=1`，safetensors 中 `mtp` key 为 0），因此无法复用辅助模型起草。若后续换成全注意力、缓存可裁剪且共享 tokenizer 的主/辅模型，可再接入该路径。
- **内存上限**：`mlx_memory_limit_mb`（默认 0 = min(物理内存 65%, 9.6GB)）是 MLX 权重+KV+buffer 的总闸门；`mlx_cache_limit_mb`（默认 1024，运行时夹在 256...2048）只限制 MLX 空闲 buffer；1GB 让 decode 快 5–8%，实测 8×784-token 的 jina-v3 embedding 批在 1GB 与 2GB 下耗时基本一致（cache 占用 995MB / 1266MB），512MB 则明显偏小。主模型 `kv_cache_max_tokens`（默认 32768，约 1GB KV）和辅助模型 `aux_kv_cache_max_tokens`（默认 8192，约 100MB）是 KV 硬上限，实际上下文取 `min(配置上下文, 上限)`，内核预算同步收紧。完整 prompt 超过预算时明确报错，不会继续分配 KV。辅助模型默认（`aux_llm_kv_reuse=false`）每次后台摘要/事实抽取后立即失效 KV，开启复用后改为保留前缀缓存跨调用复用；主模型 `llm_kv_reuse` 与辅助模型的复用/量化开关各自独立，量化默认都开（KV 内存约减半）。预热前缀快照只在 ≤8192 token 时保留。RAG 检索 `topK` 封顶 20，避免 reranker 批量被模型参数放大。
- **模型常驻策略**：主模型与 Reranker 加载后常驻不自动卸载；启动时（以及切换配置后）同时预加载本地 Embedding 与 Reranker 并保持常驻，首问不再现场加载 ONNX session（实测 10~15s）；辅助模型仅在 `aux_enabled=true` 时常驻，关闭时 Agent 的 `auxLLM` 保持 `nil`（RAG 精简跳过，后台记忆/总结回退主模型）；OCR 只在解析/转换时使用（默认 2 个 Ovis 实例，同步期间并行识别）；同步、Markdown 转换或 OCR 测试结束时由 `OCRModelLifecycle` 立即卸载全部 Ovis 权重（两个实例一起释放），下次需要时再按需加载；`LocalModelFactory.unloadIdleOCRWeights()` 仅作为无在途 OCR 任务时的空闲兜底。闲时章节摘要使用更长的 10 分钟空闲窗口，避免与前台抢 GPU。provider 对象保留，不影响任何工具调用。
- 为提升缓存命中，system prompt 只保留稳定前缀（默认 `BASE_SYSTEM_PROMPT`，可由 `<workspace>/assets/system_prompt.txt` 替换），工作记忆、当天事实、checkpoint、技能与知识库状态等动态内容拼在 user 消息尾部；同一轮内的动态上下文做缓存，避免工具循环中反复变化。运行日志会输出 `[mlx-kv] reuse=… cached=… delta=…` 与 `[mlx] prompt=… cached=… generated=…`，可直接确认本轮复用了多少 token。
- Embedding 输入前缀按模型族统一处理（`EmbeddingPrefixStrategy`，ONNX / MLX / GGUF 行为一致）：nomic 使用 `search_document: ` / `search_query: `；bge-zh 只在 query 侧加 `为这个句子生成表示以用于检索相关文章：`，document 不加，bge-en / bge-m3 不加；jina 使用 `passage: ` / `query: `，并保留 `clustering: ` / `classification: ` 任务前缀；E5 使用 `passage: ` / `query: `；其他模型原样。前缀在 provider 内幂等添加，调用方不要重复拼；切片预算会按 document 前缀预留 token。mean pooling + L2 归一化；MLX embedding 从 `config.json` 读取隐藏维度。
- 模型查找顺序：配置显式路径 → `<workspace>/models/<name>` → `~/base-agent/models/<name>` → HuggingFace / FastEmbed 缓存。旧配置的 `models/<name>` 相对写法会自动去掉 `models/` 前缀后在这些根目录下查找。
- 主/辅模型支持**独立预加载与卸载**：启动时按 `auto_preload_main` 与 `llm_provider` 决定是否预加载主模型，辅助模型仅在 `aux_enabled=true` 时预加载；「加载模型」只替换对应槽位的模型实例并读入权重，另一个模型保持常驻；「卸载」只作用于当前模型。加载参数（context / temp / top_p / min_p / top_k / repeat / batch / thinking）进入模型实例缓存键，修改后会创建新实例。
- 生成、加载、预热、知识同步期间禁用模型加载/卸载/同步按钮与消息发送；配置切换会等待旧请求结束后丢弃其流式缓冲，迟到的 50ms flush 也不会写进新气泡。
- **知识同步期间释放模型**：同步开始前卸载已加载的主/辅模型并清理 LLM 缓存，同步结束（成功/失败/取消）后按同步前状态自动重新加载，主模型会重新预热。开始与恢复两处都会检查 `aux_enabled`：关闭时日志写「主模型（辅助模型未启用）」，不会卸载、创建或重新加载辅助模型。日志依次输出「已卸载主模型 / 已卸载辅助模型 / 重新加载 / 主模型预热结束」；模型未加载时输出「无需卸载」，API 模式的主模型客户端保持原样。同步期间配置窗口的加载按钮会拒绝操作。`applyConfig { 同步 }` 路径不再并行预加载模型，避免同步期间模型又常驻内存。
- 「停止同步」会立即把状态栏改为「正在停止同步…」并禁用停止按钮，同时设置 RAG 取消标志并取消同步 Task；RAG 在文件边界、embedding 批次边界以及解析完成后都会检查取消，不再等整个文件/长 OCR 完成才响应。

PDF Ovis 引擎按 `rag_ocr_model_path` / `rag_ocr_model_name` 加载，不受 `vlm_provider` 影响。

## 配置示例

```json
{
  "workspace": "~/base-agent-workspace",
  "knowledge_base": "",
  "llm_provider": "local",
  "local_llm_backend": "mlx-lm",
  "local_llm_model": "models/qwen3.5-4b-agentic-coder-v4-4bit-mlx",
  "local_llm_enable_thinking": true,
  "auto_preload_main": true,
  "aux_enabled": false,
  "aux_llm_provider": "local",
  "aux_local_llm_backend": "mlx-lm",
  "aux_local_llm_model": "models/Qwen3.5-2B-MLX-4bit",
  "aux_max_context_tokens": 8192,
  "aux_kv_cache_max_tokens": 8192,
  "aux_temperature": 0.3,
  "aux_repetition_penalty": 1.1,
  "aux_local_llm_enable_thinking": false,
  "embedding_provider": "local",
  "local_embedding_backend": "mlx",
  "local_embedding_model": "jina-embeddings-v5-text-nano-mlx-int4",
  "local_embedding_model_path": "models/jina-embeddings-v5-text-nano-mlx-int4",
  "rag_chunk_size": 800,
  "rag_chunk_overlap": 80,
  "rag_vector_element_type": "int8",
  "rag_matryoshka": "auto",
  "rag_matryoshka_dim": 128,
  "rag_binary_coarse_enabled": true,
  "rag_binary_coarse_candidates": 200,
  "reranker_provider": "local",
  "local_reranker_backend": "onnx",
  "local_reranker_model": "jina-reranker-v2-base-multilingual-onnx",
  "local_reranker_model_path": "models/jina-reranker-v2-base-multilingual-onnx",
  "rag_rerank_enabled": true,
  "rag_rerank_top_n": 3,
  "rag_rerank_candidate_multiplier": 3,
  "rag_rerank_timeout_seconds": 5,
  "rag_rerank_threshold": 0.45,
  "rag_pdf_engine": "ovis",
  "rag_epub_ocr_engine": "vision",
  "rag_ocr_preprocess": true,
  "rag_ocr_binarize": true,
  "rag_ocr_model_name": "OvisOCR2-0.8B-MLX-4bit",
  "rag_storage_backend": "sqlite",
  "rag_sqlite_vec_enabled": true,
  "rag_rrf_k": 60,
  "rag_vector_overfetch": 4,
  "rag_fts_overfetch": 4,
  "rag_retrieval_strategy": "naive",
  "rag_kg_similarity_floor": 0.2,
  "rag_kg_entity_top_k": 40,
  "rag_kg_relation_top_k": 40,
  "rag_kg_max_chunks_per_hit": 5,
  "rag_kg_entity_token_budget": 400,
  "rag_kg_relation_token_budget": 600,
  "rag_kg_chunk_token_budget": 2000,
  "rag_kg_total_token_budget": 3000,
  "rag_kg_max_entities_per_chunk": 20,
  "rag_kg_max_relations_per_chunk": 50,
  "rag_kg_max_description_chars": 100,
  "rag_kg_extraction_format": "json",
  "rag_kg_max_concurrency": 1
}
```

旧版  配置键（`llm_mode`、`llm_backend`、`llm_model_path`、`aux_llm_*`、`rag_embedding_backend`、`rag_embedding_model_path`、`rag_rerank_backend`、`rag_rerank_model_path`、`rag_rerank_mlx_model_path`）在 canonical 键缺失时自动兼容；`mlx` / `mlx_lm` 归一化为 `mlx-lm`，`llama.cpp` / `llama_cpp` 归一化为 `gguf`，`openai` 归一化为 `api`。

使用纯 API 模式时设置：

```json
{
  "llm_provider": "api",
  "base_url": "https://api.openai.com/v1",
  "api_key": "sk-...",
  "model": "gpt-4o-mini"
}
```

## 项目结构

```
Package.swift
Sources/
  CSQLiteVec/                  # sqlite-vec v0.1.9 amalgamation（静态编入）
  BaseAgentCore/
    Core/                      # JSONValue / ChatMessage / Tool 协议 / Config
    Kernel/                    # 三态自动机、Session、预算、L0-L4 压缩
    LLM/                       # OpenAI 兼容 SSE 客户端 + LLMService 协议
    Local/                     # MLX / ONNX / Tokenizer / 模型注册表
    RAG/
      Parsers/                 # PDFKit / Ovis / Vision / EPUB / Office / HTML
      Chunking/                # NaturalLanguage 切片与大纲
      Storage/                 # SQLite、sqlite-vec、FTS5、VectorStore
      KnowledgeGraph/          # 实体/关系抽取、vec0 向量、SwiftGraph、五策略检索
      RAGEngine.swift          # 同步、Manifest、检索、rerank
    Tools/                     # 组合工具（file_tools/web_scan/long_memory/skill_tools/rag_tools）
                               # + 独立工具（code_run/shell_run/ask_user/plan/schedule_task/MCP）
    Scheduling/                # 定时任务模型、schedules.json 存储、DispatchSourceTimer 调度器
    Memory/                    # 工作记忆、长期记忆、memory graph（与 RAG 知识图谱独立）
    Skills/                    # 技能发现、索引、语义检索
    Agent/                     # Agent 门面、工具工厂、系统提示
  BaseAgentApp/                # SwiftUI 界面
Vendor/mcp-swift-sdk/          # 官方 MCP Swift SDK（本地 vendor）
Tests/BaseAgentCoreTests/      # XCTest
docs/软件设计文档.md
```

## 已知限制

- 知识图谱依赖辅助模型的抽取质量：2B 能稳定抽到核心实体与主干关系，但次要实体（技术/地点）和个别关系类型可能漏抽或误标；0.8B 实测无法稳定输出完整关系行，不建议用于图谱构建。图谱规模远小于切片全库，vec0 不可用时回退精确余弦扫描。
- PDFKit 不暴露内嵌图片与表格区域坐标：Ovis 引擎用「文本列对齐」启发式判断表格页并整页交给 VLM，未实现参考项目的表格裁剪回填与内嵌原图优先。
- EPUB 以 spine 文件为章节单位，未按 TOC 锚点二次切章。扫描版 EPUB 仅在「章节文字层低于阈值且含图片」时对该章图片逐页 OCR；纯 CSS 背景图、SVG 矢量图（非 `<image>` 引用）和加密字体渲染的文字不在识别范围内。
- 子进程未使用 `setsid`，超时/取消通过递归进程树终止实现；极端的双重 fork 守护进程仍可能逃逸。
- 本地 MLX/ONNX/GGUF 权重未随仓库提供；首次使用需自行下载到 `<workspace>/models/<name>` 或 `~/base-agent/models/<name>`。真实模型推理需在目标机器验证。
- 定时任务由 App 进程内的 `DispatchSourceTimer` 驱动：应用退出后不会继续执行，系统睡眠期间不触发（唤醒后补跑一次）；规则仅支持「每 N 分钟」与「每天 HH:mm」，不支持 cron / 每周 / 每月表达式。无人值守执行时高风险工具按用户创建任务时批准的指令自动放行，请只创建可信任务。
- 主/辅 LLM 支持 MLX-LM / MLX-VLM / GGUF / ONNX；OCR VLM 支持 MLX / GGUF+mmproj（ONNX VLM 暂无通用图适配）；Embedding / Reranker 支持 ONNX / FastEmbed / MLX / GGUF。FastEmbed 复用 ONNX 运行时及其缓存目录。GGUF 与 ONNX 均在进程内运行，不需要外部 server 或 Python。
- ONNX LLM 的空 KV 按 float32 + `config.json` 的层数/KV 头数/head_dim 初始化；fp16 past 或非标准 KV 布局的导出会给出明确错误，需要针对性适配。
- 本机已实测：GGUF LLM（Qwen3.5-9B，CPU 与 Metal GPU，工具调用正常）、GGUF VLM（Qwythos-9B + mmproj，OCR 输出正确）、GGUF Embedding（nomic-embed-text-v1.5，768 维）、GGUF system 前缀预热复用、MLX-VLM 主模型（Qwen3.5-2B）。ONNX LLM、ONNX VLM 与 GGUF Reranker 的代码路径已接入，但当前机器没有对应模型，未做推理实测；缺少模型或运行时会给出明确错误。
- 配置窗口覆盖主要参数；MCP 服务器列表等高级配置仍可直接编辑 `~/.base-agent/config.json`。


## License

GPLv3（见 [LICENSE](LICENSE)）。
