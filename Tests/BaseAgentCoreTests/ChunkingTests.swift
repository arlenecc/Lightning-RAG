import XCTest
@testable import BaseAgentCore

final class ChunkingTests: XCTestCase {
    func testOutlineAndStructuredChunks() async {
        let text = """
        # 第一章 概述

        这是第一章的第一段。人工智能正在快速发展，很多应用已经落地。第二句话继续补充内容。

        ## 1.1 背景

        背景部分包含许多细节。我们需要理解历史、现状和未来趋势。这里再补充一些内容，让文本长度超过切片阈值。

        # 第二章 方法

        第二章介绍方法。这里包含实验设计、数据采集和评估指标。最后给出结论。
        """
        let nodes = DocumentOutline.extract(text)
        XCTAssertEqual(nodes.count, 3)
        XCTAssertEqual(nodes[0].title, "第一章 概述")
        XCTAssertEqual(nodes[2].level, 1)

        let options = ChunkingOptions(chunkSizeTokens: 12, overlapTokens: 3,
                                      minChunkTokens: 10,
                                      enableSemantic: false,
                                      tokenCounter: { $0.count })
        let document = ParsedDocument(text: text, source: "doc.md")
        let chunks = await TextChunker(options: options).chunk(document: document, variant: .hybrid)
        XCTAssertGreaterThan(chunks.count, 1)
        XCTAssertEqual(chunks.map(\.chunkIndex), Array(0..<chunks.count))
        XCTAssertTrue(chunks.contains { $0.heading.contains("第一章") })
        XCTAssertTrue(chunks.allSatisfy { !$0.text.isEmpty })
        XCTAssertTrue(chunks.allSatisfy { options.tokenCounter($0.text) <= options.chunkSizeTokens },
                      "chunk 含 overlap 不应突破 token 上限")
    }

    /// Word/PPT 正文里的编号条目是列表内容，不能作为切片章节标题。
    func testStructuredChunksIgnoreNumberedBodyAsHeadings() async {
        let text = """
        ## Slide 1：年度计划

        1. 背景与目标

        2. 范围与限制

        ## Slide 2：执行方案

        一、组织分工

        二、交付计划
        """
        let options = ChunkingOptions(chunkSizeTokens: 12, overlapTokens: 2,
                                      minChunkTokens: 4,
                                      enableSemantic: false,
                                      tokenCounter: { $0.count })
        let document = ParsedDocument(text: text, source: "deck.ppt")
        let chunks = await TextChunker(options: options).chunk(document: document)

        XCTAssertFalse(chunks.isEmpty)
        XCTAssertFalse(chunks.contains { $0.heading.contains("1. 背景") })
        XCTAssertFalse(chunks.contains { $0.heading.contains("一、组织分工") })
        XCTAssertTrue(chunks.contains { $0.heading.contains("Slide 1") })
    }

    /// 分页标记与页码不进入切片，溯源只保留文件名（source）。
    func testChunksStripPageMarkersWithoutPagesMetadata() async {
        let text = """
        <!-- page 3 -->

        这是第一页的正文内容，讲述基本概念与背景信息，长度足够形成切片。

        # 第一章 概述

        <!-- page 4 -->

        这是第二页的正文内容，继续补充细节与示例说明，长度同样足够。

        <!-- page 8 -->

        这是第八页的正文内容，与前文不连续，页码应压缩为独立区间。
        """
        let options = ChunkingOptions(chunkSizeTokens: 200, overlapTokens: 0,
                                      minChunkTokens: 10,
                                      enableSemantic: false,
                                      tokenCounter: { $0.count })
        let document = ParsedDocument(text: text, source: "book.pdf")
        let chunks = await TextChunker(options: options).chunk(document: document)
        XCTAssertFalse(chunks.isEmpty)
        XCTAssertTrue(chunks.allSatisfy { !$0.text.contains("<!-- page") },
                      "切片文本不应包含分页标记")
        XCTAssertTrue(chunks.allSatisfy { $0.metadata.isEmpty },
                      "切片不再记录页码元数据")
        XCTAssertTrue(chunks.allSatisfy { $0.source == "book.pdf" },
                      "溯源只保留文件名")
        // 正文完整保留。
        XCTAssertTrue(chunks.contains { $0.text.contains("第一页的正文") })
    }

    /// 页码/年份正文不得被识别成章节标题（曾导致整本书退化为按页切片）。
    func testOutlineRejectsPageHeadersAndDateProse() {
        let text = """
        # 2 / 164

        # 164

        1898 年拜耶林克发现烟草花叶病毒

        # 1898 年，烟草花叶病毒的发现过程

        2 / 164

        第 3 页

        Page 4 of 10

        # 第一章 绪论
        """
        let nodes = DocumentOutline.extract(text)
        XCTAssertEqual(nodes.map(\.title), ["第一章 绪论"])
    }

    /// 语义主题边界在两侧片段都足够大时保留；短片段则由 chunk 级合并补足。
    func testSemanticBoundariesSurviveMinChunkChars() async {
        let text = String(repeating: "苹果是一种常见的水果，富含维生素和膳食纤维。", count: 3)
            + "香蕉口感香甜。"
            + String(repeating: "今天天气晴朗，阳光明媚，适合外出散步。", count: 3)
            + "天空没有云朵，微风轻拂。"
        let options = ChunkingOptions(chunkSizeTokens: 80, overlapTokens: 0,
                                      minChunkTokens: 80,
                                      semanticThreshold: 0.55, enableSemantic: true,
                                      tokenCounter: { $0.count })
        let document = ParsedDocument(text: text, source: "fruit.md")
        // 前四句水果、后四句天气：主题之间相似度为 0，主题内部为 1。
        let similarity: SemanticChunker.AdjacentSimilarity = { sentences in
            (0..<(sentences.count - 1)).map { index in
                let a = sentences[index]
                let b = sentences[index + 1]
                let fruitA = a.contains("苹果") || a.contains("香蕉")
                let fruitB = b.contains("苹果") || b.contains("香蕉")
                return fruitA == fruitB ? 1.0 : 0.0
            }
        }
        let chunks = await TextChunker(options: options, similarity: similarity)
            .chunk(document: document, variant: .hybrid)
        XCTAssertEqual(chunks.count, 2, chunks.map(\.text).joined(separator: " | "))
        XCTAssertTrue(chunks[0].text.contains("苹果"))
        XCTAssertTrue(chunks[1].text.contains("天空"))
    }

    /// 同一章节内的短小节应合并到接近 chunk_size，不能各自成为小片。
    func testShortSubsectionsCoalesceWithinChapter() async {
        let text = """
        # 第一章 总述

        ## 背景

        短内容一，只有一句话。

        ## 现状

        短内容二，也只有一句话。

        # 第二章 方法

        ## 步骤

        第二章的短内容，不应与第一章合并。
        """
        let options = ChunkingOptions(chunkSizeTokens: 200, overlapTokens: 0,
                                      minChunkTokens: 10,
                                      enableSemantic: false,
                                      tokenCounter: { $0.count })
        let document = ParsedDocument(text: text, source: "report.docx")
        let chunks = await TextChunker(options: options).chunk(document: document)
        XCTAssertEqual(chunks.count, 2, chunks.map { $0.heading + ":" + $0.text }.joined(separator: " | "))
        XCTAssertTrue(chunks[0].text.contains("短内容一"))
        XCTAssertTrue(chunks[0].text.contains("短内容二"), "同章短小节应合并")
        XCTAssertFalse(chunks[0].text.contains("第二章"), "不应跨章合并")
        XCTAssertTrue(chunks[1].text.contains("第二章"))
    }

    /// 不超过切片大小设置的短章节整章作为一个切片，不做二次切分。
    func testShortSectionKeptWhole() async {
        let text = """
        苹果是一种常见的水果。香蕉也是一种常见的水果。今天天气晴朗。天空没有云朵。
        """
        let options = ChunkingOptions(chunkSizeTokens: 200, overlapTokens: 40,
                                      minChunkTokens: 1,
                                      enableSemantic: true,
                                      tokenCounter: { $0.count })
        let document = ParsedDocument(text: text, source: "short.md")
        let chunks = await TextChunker(options: options)
            .chunk(document: document, variant: .hybrid)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0].text, text.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// 页码标记与各种页码形态在清洗阶段全部删除，任意位置都不保留。
    func testCleanerRemovesPageMarkersAndPageNumbersAnywhere() {
        let input = """
        <!-- page 1 -->
        正文第一段。

        2 / 164

        # 2 / 164

        第 3 页

        Page 4 of 10

        <!-- page 2 -->
        正文第二段。
        """
        let cleaned = DocumentCleaner.clean(input)
        XCTAssertFalse(cleaned.contains("<!-- page"), cleaned)
        XCTAssertFalse(cleaned.contains("2 / 164"), cleaned)
        XCTAssertFalse(cleaned.contains("第 3 页"), cleaned)
        XCTAssertFalse(cleaned.contains("Page 4"), cleaned)
        XCTAssertTrue(cleaned.contains("正文第一段"), cleaned)
        XCTAssertTrue(cleaned.contains("正文第二段"), cleaned)
    }

    func testSemanticChunkerSplitsAtLowSimilarity() async {
        let text = """
        苹果是一种水果。香蕉也是一种水果。今天天气晴朗。天空没有云朵。
        """
        let options = ChunkingOptions(chunkSizeTokens: 8, overlapTokens: 2,
                                      minChunkTokens: 1,
                                      semanticThreshold: 0.9, enableSemantic: true,
                                      tokenCounter: { $0.count })
        let document = ParsedDocument(text: text, source: "fruit.md")
        let similarity: SemanticChunker.AdjacentSimilarity = { sentences in
            Array(repeating: 0.0, count: max(0, sentences.count - 1))
        }
        let chunks = await TextChunker(options: options, similarity: similarity)
            .chunk(document: document, variant: .semantic)
        XCTAssertFalse(chunks.isEmpty)
        XCTAssertTrue(chunks.allSatisfy { !$0.text.isEmpty })
    }

    func testDocumentCleaner() {
        let dirty = "第 1 页\n正文内容第一行。\n正文内容第二行。\n\n\n\n第 2 页\n更多内容。\n第 3 页"
        let cleaned = DocumentCleaner.clean(dirty, keepHeadersFooters: false)
        XCTAssertFalse(cleaned.contains("第 1 页"))
        XCTAssertFalse(cleaned.contains("第 3 页"))
        XCTAssertTrue(cleaned.contains("正文内容第一行"))
        XCTAssertFalse(cleaned.contains("\n\n\n\n"))
    }

    func testOutlineEnglishBracketAndCommaPatterns() {
        let text = """
        Chapter 1 Introduction

        第一章 引言，研究背景

        （一）数据准备

        Part II Methods

        Section 3.1 Sampling

        第二章开始，我们讨论了实验设计、数据采集和评估指标的完整流程，并给出了分析结果与改进方向
        """
        let nodes = DocumentOutline.extract(text)
        let titles = nodes.map(\.title)
        XCTAssertTrue(titles.contains("Chapter 1 Introduction"), "英文 Chapter 应识别")
        XCTAssertTrue(titles.contains { $0.contains("引言") && $0.contains("第一章") },
                      "含逗号的短中文标题应保留")
        XCTAssertTrue(titles.contains { $0.contains("数据准备") }, "括号序号（一）应识别")
        XCTAssertTrue(titles.contains("Part II Methods"), "英文 Part + 罗马数字应识别")
        XCTAssertTrue(titles.contains { $0.contains("Sampling") }, "英文 Section 应识别为二级")
        XCTAssertFalse(titles.contains {
            $0.contains("实验设计") && $0.contains("分析结果")
        }, "长正文句子不应被误判为标题")
        let section = nodes.first { $0.title.contains("Sampling") }
        XCTAssertEqual(section?.level, 2)
    }

    // MARK: token 预算

    /// 估算器必须偏保守（估算 ≥ 实际 token），保证硬切回退也不会把块推出上下文窗口。
    /// 实际 token 数由模型自带 tokenizer.json 实测。
    func testEmbeddingTokenEstimatorIsConservativeUpperBound() {
        let chinese = "云计算技术近年来在数据存储与弹性伸缩方面发展迅速，企业纷纷将业务迁移上云。"
        XCTAssertGreaterThanOrEqual(EmbeddingTokenEstimator.count(chinese), 37, "实测 37 token")
        XCTAssertLessThanOrEqual(EmbeddingTokenEstimator.count(chinese), 42,
                                 "纯中文不应被过度放大（1 字 ≈ 1 token）")

        let english = "Retrieval-augmented generation combines a vector store with a language model."
        XCTAssertGreaterThanOrEqual(EmbeddingTokenEstimator.count(english), 20, "实测 20 token")

        let hash = "a3f5c9e1b7d26840f1e2d3c4b5a697887766554433221100ffeeddccbbaa9988"
        XCTAssertGreaterThanOrEqual(EmbeddingTokenEstimator.count(hash), 40,
                                    "实测 40 token（哈希/编号是最坏情况）")

        XCTAssertEqual(EmbeddingTokenEstimator.count(""), 0)
    }

    /// overlap 计入总预算：任何切片（含重叠）都不超过 chunk_size token，
    /// 且中文正文仍在句号处断开。
    func testChunkTokenBudgetIncludesOverlap() async {
        let text = String(repeating: "苹果是一种常见的水果，富含维生素和膳食纤维。", count: 40)
        let options = ChunkingOptions(chunkSizeTokens: 100, overlapTokens: 10,
                                      minChunkTokens: 1,
                                      enableSemantic: false)
        let document = ParsedDocument(text: text, source: "budget.md")
        let chunks = await TextChunker(options: options)
            .chunk(document: document, variant: .recursive)
        XCTAssertGreaterThan(chunks.count, 1)
        XCTAssertTrue(chunks.allSatisfy { EmbeddingTokenEstimator.count($0.text) <= options.chunkSizeTokens },
                      "含 overlap 的切片不应超过 100 token：\(chunks.map { EmbeddingTokenEstimator.count($0.text) })")
        XCTAssertTrue(chunks.allSatisfy { $0.text.hasSuffix("。") },
                      "切片应在句子结束处断开：\(chunks.map { "\($0.text.suffix(6))[\(EmbeddingTokenEstimator.count($0.text))]" })")
    }

    /// 中文句子之间不能因为语义切片而硬插空格，切片必须保持原文连续。
    func testSemanticChunkerJoinsCJKSentencesWithoutSpaces() async {
        let text = "苹果是一种水果。香蕉也是一种水果。今天天气晴朗。天空没有云朵。"
        let options = ChunkingOptions(chunkSizeTokens: 1_000, overlapTokens: 0,
                                      minChunkTokens: 1,
                                      semanticThreshold: 0.55, enableSemantic: true,
                                      tokenCounter: { $0.count })
        let similarity: SemanticChunker.AdjacentSimilarity = { sentences in
            Array(repeating: 1.0, count: max(0, sentences.count - 1))
        }
        let chunks = await SemanticChunker.split(text, options: options, similarity: similarity)
        XCTAssertEqual(chunks?.count, 1)
        XCTAssertEqual(chunks?.first, text, "中文句子拼接不得插入空格")
    }

    /// 短语义片在弱边界应合并到同一片，避免一本书被切成大量单句碎片。
    func testSemanticChunkerMergesShortChunksAtWeakBoundary() async {
        let text = "甲乙丙丁戊。己庚辛壬癸。子丑寅卯辰。巳午未申酉。"
        let options = ChunkingOptions(chunkSizeTokens: 100, overlapTokens: 0,
                                      minChunkTokens: 60,
                                      semanticThreshold: 0.55, enableSemantic: true,
                                      tokenCounter: { $0.count })
        let similarity: SemanticChunker.AdjacentSimilarity = { sentences in
            Array(repeating: 0.5, count: max(0, sentences.count - 1))
        }
        let chunks = await SemanticChunker.split(text, options: options, similarity: similarity)
        XCTAssertEqual(chunks?.count, 1, "短语义片应在弱边界合并")
        XCTAssertFalse(chunks?.first?.contains("。 ") == true, "合并后中文仍不应插入空格")
    }

    /// 强主题边界不能被长度合并抹掉：即使两侧都很短，也应保持独立语义单元。
    func testSemanticChunkerKeepsShortChunksAcrossStrongBoundary() async {
        let text = "甲乙丙丁戊。己庚辛壬癸。子丑寅卯辰。巳午未申酉。"
        let options = ChunkingOptions(chunkSizeTokens: 100, overlapTokens: 0,
                                      minChunkTokens: 60,
                                      semanticThreshold: 0.55, enableSemantic: true,
                                      tokenCounter: { $0.count })
        let similarity: SemanticChunker.AdjacentSimilarity = { sentences in
            Array(repeating: 0.0, count: max(0, sentences.count - 1))
        }
        let chunks = await SemanticChunker.split(text, options: options, similarity: similarity)
        XCTAssertEqual(chunks?.count, 4, "强边界不得被短块合并越过")
    }

    /// 真实内置模型集成：三主题各三句，应在两个主题切换处切成 3 段。
    func testSemanticChunkerUsesBundledModelToSplitTopics() async throws {
        let text = "公司今天发布了第三季度财报。财报显示营收同比增长百分之二十。财报还显示净利润创下历史新高。"
            + "研究人员在深海发现了一种新的鱼类。这种新鱼类生活在四千米深的海底。新鱼类的身体几乎完全透明。"
            + "球队在主场赢得了比赛。球迷们在看台为球队高声欢呼。教练称赞了球员们比赛中的表现。"
        let options = ChunkingOptions(
            chunkSizeTokens: 500,
            overlapTokens: 0,
            minChunkTokens: 0,
            semanticThreshold: 0.55,
            enableSemantic: true
        )
        let result = await SemanticChunker.split(text, options: options)
        let chunks = try XCTUnwrap(result)
        XCTAssertEqual(chunks.count, 3, "\(chunks)")
    }
}
