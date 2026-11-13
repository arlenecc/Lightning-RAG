import XCTest
import PDFKit
import CoreText
import AppKit
import ZIPFoundation
@testable import BaseAgentCore

final class ParserTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-parser-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    // MARK: 图片解析路由

    /// 图片文件应路由到 ImageParser（Apple Vision OCR），且扩展名白名单覆盖。
    func testImageFilesRouteToImageParser() throws {
        var config = AgentConfig()
        config.workspace = directory.path
        let registry = DocumentParserRegistry.makeDefault(config: config)
        for ext in ["png", "jpg", "jpeg", "heic", "webp", "tiff"] {
            let parser = registry.freshParser(
                for: directory.appendingPathComponent("photo.\(ext)"),
                options: ParserOptions())
            XCTAssertTrue(parser is ImageParser, "\(ext) 应路由到 ImageParser，实际 \(String(describing: parser))")
        }
        XCTAssertTrue(RAGEngine.supportedExtensions.isSuperset(of: ["png", "jpg", "heic"]),
                      "同步扫描的白名单应包含图片扩展名")
    }

    // MARK: 纯文本结构还原

    func testPlainTextStructureRestoresHeadingsAndMarkdown() throws {
        let longPara1 = String(repeating: "云计算技术近年来在数据存储与弹性伸缩方面发展迅速，企业纷纷将业务迁移上云。", count: 2)
        let longPara2 = String(repeating: "容器化部署让应用的发布流程更加标准化，同时也带来了编排与调度的新挑战。", count: 2)
        let toc = """
        第一章 云计算概述 .......... 1
          1.1 基本概念 .......... 2
          1.2 发展历程 .......... 8
        第二章 核心技术 .......... 15
        """
        let body = """
        \(toc)

        \(longPara1)

        部署实践要点

        \(longPara2)
        """
        let url = directory.appendingPathComponent("plain.txt")
        try body.write(to: url, atomically: true, encoding: .utf8)

        let parsed = try awaitSync {
            try await PlainTextParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        // 目录块转为 Markdown 列表（点线与页码剔除），正文标题转成 Markdown。
        // 注：清洗阶段会移除汉字间空格（OCR 噪声处理），故「第一章」后无空格。
        XCTAssertFalse(parsed.text.contains(".........."))
        XCTAssertTrue(parsed.text.contains("- 第一章"))
        XCTAssertTrue(parsed.text.contains("- 1.1 基本概念"))
        XCTAssertTrue(parsed.text.contains("- 第二章"))
        // 独立短行夹在长段落之间 → 弱标题还原。
        XCTAssertTrue(parsed.text.contains("## 部署实践要点"))
        // 大纲包含目录与正文标题（outline 在清洗前生成，保留原空格）。
        XCTAssertTrue(parsed.outline.contains("云计算概述"))
        XCTAssertTrue(parsed.outline.contains("部署实践要点"))
        XCTAssertEqual(parsed.metadata["structure_restored"], "true")
    }

    func testPlainTextStructureSkipsPoemLikeShortLines() throws {
        let longLine = String(repeating: "这是一个足够长的正文段落，用来提供格式对比的锚点。", count: 3)
        let body = """
        \(longLine)

        床前明月光

        疑是地上霜

        举头望明月

        \(longLine)
        """
        XCTAssertNil(PlainTextStructure.restore(body),
                     "连续短行堆（诗歌）不应被还原成标题结构")
    }

    func testPlainTextMarkdownLikeIsSkipped() throws {
        XCTAssertTrue(PlainTextStructure.isMarkdownLike("# 标题\n\n## 小节\n正文"))
        XCTAssertFalse(PlainTextStructure.isMarkdownLike("纯文本没有标记\n第二行正文"))
    }

    /// 文字版 PDF：文本层提取的纯文本无 markdown 结构时，
    /// 也应经过启发式还原转成 Markdown。
    func testTextLayerPDFRestoresStructure() throws {
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let url = directory.appendingPathComponent("text-layer.pdf")
        let ctx = CGContext(url as CFURL, mediaBox: &box, nil)!
        // 与真实排版一致：按行绘制（每行不超过页宽）。
        func draw(_ lines: [(String, CGFloat)]) {
            ctx.beginPDFPage(nil)
            var y: CGFloat = 740
            for (text, size) in lines {
                let attr = NSAttributedString(string: text, attributes: [
                    .font: NSFont.systemFont(ofSize: size)])
                let line = CTLineCreateWithAttributedString(attr)
                ctx.textPosition = CGPoint(x: 54, y: y)
                CTLineDraw(line, ctx)
                y -= size + 14
            }
            ctx.endPDFPage()
        }
        let p1 = "云计算技术近年来在数据存储与弹性伸缩方面发展迅速，企业业务加速上云，运维模式也随之演进。"
        let p2 = "容器化部署让应用的发布流程更加标准化，同时带来了编排与调度方面的新挑战与容量规划要求。"
        let p3 = "混合云方案在合规与成本之间取得了平衡，关键业务保留在本地集群，弹性负载交给公有云。"
        let p4 = "监控体系需要覆盖指标、日志与链路三个维度，通过统一的采样策略控制存储与排障成本。"
        let p5 = "容量评估应当基于历史峰值与增长趋势建模，预留合理的安全水位并定期回收闲置资源。"
        draw([
            ("第一章 概述", 18),
            (p1, 11), (p2, 11),
            ("部署模式对比", 14),
            (p3, 11),
        ])
        draw([
            (p4, 11),
            ("容量规划实践", 14),
            (p5, 11),
        ])
        ctx.closePDF()

        let parsed = try awaitSync {
            try await OvisPDFParser(vlm: nil).parse(
                url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertEqual(parsed.metadata["structure_restored"], "true")
        // 强模式标题与弱标题都转成了 Markdown 标题。
        XCTAssertTrue(parsed.text.contains("# 第一章概述"))
        XCTAssertTrue(parsed.text.contains("## 部署模式对比"))
        XCTAssertTrue(parsed.text.contains("## 容量规划实践"))
        XCTAssertTrue(parsed.outline.contains("部署模式对比"))
    }

    // MARK: PDF 文本层

    func testPDFKitTextLayerPath() throws {
        let url = directory.appendingPathComponent("text.pdf")
        try makeTextPDF(at: url, text: String(repeating: "Hello PDFKit text layer. ", count: 8))

        let parser = VisionPDFParser()
        let parsed = try awaitSync {
            try await parser.parse(url: url, options: ParserOptions(pdfEngine: .vision), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("Hello PDFKit"))
        XCTAssertEqual(parsed.metadata["engine"], "vision")
    }

    /// 扫描书的文本层常只有「每页相同的水印 + 网址」：原始字数可能过
    /// 文本层阈值，但清洗时会被当作跨页页眉页脚整段删除。必须识别为
    /// 不可用文本层并改走 OCR，否则整份文档清洗为空、转换失败。
    func testWatermarkOnlyTextLayerFallsBackToOCR() throws {
        let url = directory.appendingPathComponent("watermark-scan.pdf")
        let watermark = "www.example.com scanned book watermark"
        try makeWatermarkedImagePDF(at: url, pageCount: 3, watermark: watermark)

        let pages = Array(repeating: watermark + "\n" + watermark, count: 3)
        let marginKeys = DocumentCleaner.repeatedMarginKeys(in: pages)
        XCTAssertFalse(marginKeys.isEmpty, "跨页重复水印应被识别为页眉页脚")
        XCTAssertLessThan(
            DocumentCleaner.effectiveOCRTextCharacterCount(
                pages[0], excludingMarginKeys: marginKeys),
            PDFPageRenderer.textLayerMinCharacters)

        let parser = VisionPDFParser()
        let parsed = try awaitSync {
            try await parser.parse(url: url,
                                   options: ParserOptions(pdfEngine: .vision),
                                   progress: nil)
        }
        XCTAssertEqual(parsed.metadata["ocr_pages"], "3", "只有水印的文本层必须全部走 OCR")
        XCTAssertTrue(parsed.text.uppercased().contains("SCANNED"), parsed.text)
    }

    /// 同一水印问题在 Ovis 路径也要回退到视觉模型。
    func testWatermarkOnlyTextLayerFallsBackToVLMInOvis() throws {
        let url = directory.appendingPathComponent("watermark-ovis.pdf")
        let watermark = "www.example.com scanned book watermark"
        try makeWatermarkedImagePDF(at: url, pageCount: 3, watermark: watermark)

        let fake = FakeVLM()
        let parser = OvisPDFParser(vlm: fake)
        let parsed = try awaitSync {
            try await parser.parse(url: url,
                                   options: ParserOptions(pdfEngine: .ovis),
                                   progress: nil)
        }
        XCTAssertEqual(fake.calls, 3, "只有水印的文本页应交给 OvisOCR3")
        XCTAssertEqual(parsed.metadata["text_pages"], "0")
        XCTAssertEqual(parsed.metadata["ocr_pages"], "3")
    }

    // MARK: Ovis 路径

    func testOvisParserUsesVLM() throws {
        let url = directory.appendingPathComponent("image.pdf")
        try makeImagePDF(at: url, text: "Ovis page content")
        let fake = FakeVLM()
        let parser = OvisPDFParser(vlm: fake)
        let parsed = try awaitSync {
            try await parser.parse(url: url, options: ParserOptions(pdfEngine: .ovis), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("Ovis page content"))
        XCTAssertEqual(fake.calls, 1)
        XCTAssertEqual(parsed.metadata["engine"], "ovis")
    }

    func testOvisParserTextPDFDoesNotRequireVLM() throws {
        let url = directory.appendingPathComponent("text-ovis.pdf")
        try makeTextPDF(at: url, text: String(repeating: "Hello PDFKit text layer. ", count: 8))

        let parser = OvisPDFParser(vlm: nil)
        let parsed = try awaitSync {
            try await parser.parse(url: url, options: ParserOptions(pdfEngine: .ovis), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("Hello PDFKit"), parsed.text)
        XCTAssertEqual(parsed.metadata["engine"], "ovis")
        XCTAssertEqual(parsed.metadata["text_pages"], "1")
        XCTAssertEqual(parsed.metadata["ocr_pages"], "0")
    }

    func testHybridParserTextPDFDoesNotRequireVLM() throws {
        let url = directory.appendingPathComponent("text-hybrid.pdf")
        try makeTextPDF(at: url, text: String(repeating: "Hello PDFKit text layer. ", count: 8))

        let parser = OvisPDFParser(vlm: nil)
        let parsed = try awaitSync {
            try await parser.parse(url: url, options: ParserOptions(pdfEngine: .hybrid), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("Hello PDFKit"), parsed.text)
        XCTAssertEqual(parsed.metadata["engine"], "hybrid")
        XCTAssertEqual(parsed.metadata["text_pages"], "1")
        XCTAssertEqual(parsed.metadata["vision_pages"], "0")
        XCTAssertEqual(parsed.metadata["ocr_pages"], "0")
    }

    func testCleanerRemovesUnicodeAndOCRNoise() throws {
        let input = """
        新 中 新

        原 卷 印 书 集

        A\u{00A0}B\u{200B}C\u{FFFD}D\u{00AD}E

        草地......林木......鳥......

        •••中楽玉 •中雟蔥玉

        d。.-。.51

        # 新 草 草
        # 新 草 草

        Apple 苹果
        """
        let output = DocumentCleaner.clean(input)
        XCTAssertTrue(output.contains("新中新"), output)
        XCTAssertTrue(output.contains("原卷印书集"), output)
        XCTAssertFalse(output.unicodeScalars.contains { $0.value == 0x00A0 })
        XCTAssertFalse(output.unicodeScalars.contains { $0.value == 0x200B })
        XCTAssertFalse(output.unicodeScalars.contains { $0.value == 0xFFFD })
        XCTAssertFalse(output.unicodeScalars.contains { $0.value == 0x00AD })
        XCTAssertTrue(output.contains("草地……林木……鳥……"), output)
        XCTAssertFalse(output.contains("中楽玉"), output)
        XCTAssertFalse(output.contains("d。."), output)
        XCTAssertEqual(output.components(separatedBy: "# 新草草").count - 1, 1, output)
        XCTAssertTrue(output.contains("Apple 苹果"), output)
    }

    /// 真实文件暴露的清洗缺口：册页脚注、散落数据行；以及日期/标题不误删。
    func testCleanerRemovesVolumeMarkersAndScatteredDataRows() throws {
        let input = """
        第一章 概述

        这是正文段落，讲述某年的产量变化与趋势，用来提供上下文。

        （3）上

        （12）

        23 大土地所有者 385 000' 19 250 000

        24 小土地所有者 1050 000 21 000 000

        1949 年 10 月 1 日是重要日期，涉及 385 000 名工人。

        （一）问题的提出

        这是第二段正文，保持完整不被误删，信息密度应当保留。
        """
        let output = DocumentCleaner.clean(input)
        XCTAssertFalse(output.contains("（3）上"), output)
        XCTAssertFalse(output.contains("（12）"), output)
        XCTAssertFalse(output.contains("385 000' 19 250 000"), output)
        XCTAssertFalse(output.contains("1050 000 21 000 000"), output)
        // 日期与真标题、正文段落必须保留。
        XCTAssertTrue(output.contains("1949 年 10 月 1 日"), output)
        XCTAssertTrue(output.contains("（一）问题的提出"), output)
        XCTAssertTrue(output.contains("这是第二段正文"), output)
    }

    func testCleanerConvertsHTMLTablesToMarkdown() throws {
        let input = """
        # 参数表

        <table>
        <thead><tr><th>名称</th><th>数值</th></tr></thead>
        <tbody>
        <tr><td>内存<br>占用</td><td>1&nbsp;GB</td></tr>
        <tr><td>A &amp; B</td><td>2</td></tr>
        </tbody>
        </table>

        <p>结论：<b>正常</b></p>
        """
        let output = DocumentCleaner.clean(input)
        XCTAssertTrue(output.contains("| 名称 | 数值 |"), output)
        XCTAssertTrue(output.contains("| --- | --- |"), output)
        XCTAssertTrue(output.contains("| 内存占用 | 1 GB |"), output)
        XCTAssertTrue(output.contains("| A & B | 2 |"), output)
        XCTAssertTrue(output.contains("结论：正常"), output)
        XCTAssertFalse(output.contains("<table"), output)
        XCTAssertFalse(output.contains("<td"), output)
        XCTAssertFalse(output.contains("<p>"), output)
        XCTAssertFalse(output.contains("&nbsp;"), output)
        XCTAssertFalse(output.contains("&amp;"), output)
    }

    func testCleanerCompressesBlankLinesAndKeepsStructure() throws {
        let long1 = String(repeating: "这是一个较长的真实段落。", count: 10)
        let long2 = String(repeating: "另一个较长的真实段落。", count: 10)
        let input = """
        短行一

        短行二

        \(long1)

        \(long2)

        # 标题

        - 项一

        - 项二

        | a | b |
        | --- | --- |
        | c | d |

        <!-- page 2 -->

        正文  内容  content
        """
        let output = DocumentCleaner.clean(input)
        XCTAssertTrue(output.contains("短行一\n短行二"), output)
        XCTAssertTrue(output.contains("\(long1)\n\n\(long2)"), output)
        XCTAssertTrue(output.contains("# 标题\n\n- 项一\n- 项二"), output)
        XCTAssertTrue(output.contains("| a | b |\n| --- | --- |\n| c | d |"), output)
        XCTAssertFalse(output.contains("<!-- page"), "页码标记不进入最终文本：\(output)")
        XCTAssertTrue(output.contains("正文内容 content"), output)
        XCTAssertFalse(output.contains("\n\n\n"), output)
    }

    func testCleanerKeepsComplexHTMLTable() throws {
        let input = """
        # 复杂表格

        <table>
        <tr><th rowspan="2">项目</th><th colspan="2">数值</th></tr>
        <tr><td>内存</td><td>1&nbsp;GB</td></tr>
        </table>

        <p>表格后的正文。</p>
        """
        let output = DocumentCleaner.clean(input)
        // 合并单元格无法用 GFM 表达，必须保留 table 结构。
        XCTAssertTrue(output.contains("<table"), output)
        XCTAssertTrue(output.contains("rowspan=\"2\""), output)
        XCTAssertTrue(output.contains("colspan=\"2\""), output)
        XCTAssertTrue(output.contains("内存"), output)
        XCTAssertTrue(output.contains("1 GB"), output)
        XCTAssertTrue(output.contains("表格后的正文"), output)
        XCTAssertFalse(output.contains("| 项目 |"), output)
    }

    func testCleanerKeepsNestedAndBlockContentTables() throws {
        let nested = """
        <table><tr><td>外层</td><td>
        <table><tr><td>内层 A</td><td>内层 B</td></tr></table>
        </td></tr></table>
        """
        let nestedOutput = DocumentCleaner.clean(nested)
        XCTAssertTrue(nestedOutput.contains("外层"), nestedOutput)
        XCTAssertTrue(nestedOutput.contains("内层 A"), nestedOutput)
        XCTAssertEqual(nestedOutput.components(separatedBy: "<table").count - 1, 2, nestedOutput)

        let blockContent = """
        <table><tr><td><ul><li>甲</li><li>乙</li></ul></td><td><img src="x.png"></td></tr></table>
        """
        let blockOutput = DocumentCleaner.clean(blockContent)
        XCTAssertTrue(blockOutput.contains("<ul>"), blockOutput)
        XCTAssertTrue(blockOutput.contains("<img"), blockOutput)
        XCTAssertTrue(blockOutput.contains("甲"), blockOutput)
    }

    func testOvisCleanVLMOutputConvertsHTMLTable() throws {
        let raw = "<table><tr><th>列 A</th><th>列 B</th></tr>\n"
            + "<tr><td>1<br>行</td><td>2</td></tr></table>"
        let cleaned = OvisPDFParser.cleanVLMOutput(raw)
        XCTAssertTrue(cleaned.contains("| 列 A | 列 B |"), cleaned)
        XCTAssertTrue(cleaned.contains("| 1 行 | 2 |"), cleaned)
        XCTAssertFalse(cleaned.contains("<table"), cleaned)
        XCTAssertFalse(cleaned.contains("<td"), cleaned)
    }

    func testOvisCleanVLMOutputKeepsComplexHTMLTable() throws {
        let raw = "<table><tr><td colspan=\"2\">标题</td></tr>"
            + "<tr><td>A</td><td>B</td></tr></table>"
        let cleaned = OvisPDFParser.cleanVLMOutput(raw)
        XCTAssertTrue(cleaned.contains("<table"), cleaned)
        XCTAssertTrue(cleaned.contains("colspan=\"2\""), cleaned)
        XCTAssertTrue(cleaned.contains("标题"), cleaned)
        XCTAssertTrue(cleaned.contains("A") && cleaned.contains("B"), cleaned)
    }

    func testCleanerConvertsMalformedHTMLTable() throws {
        let input = "<table><tr><td>A<td>B<tr><td>1<td>2</table>"
        let output = DocumentCleaner.clean(input)
        XCTAssertTrue(output.contains("| A | B |"), output)
        XCTAssertTrue(output.contains("| --- | --- |"), output)
        XCTAssertTrue(output.contains("| 1 | 2 |"), output)
        XCTAssertFalse(output.contains("<td"), output)
    }

    func testCleanerPreservesMarkdownSyntaxAndRemovesHTML() throws {
        let input = """
        # 标题

        - 列表项
        - [链接](https://example.com/a?b=1&c=2)
        - **加粗** 文本

        > 引用

        ---

        正文\u{200B}含零宽\u{FFFD}字符，实体 &hellip; &lt;tag&gt; <span>HTML</span>。
        行内代码 `<span>` 必须保留。

        ```swift
        let x = "<p>代码</p>"
        ```
        """
        let output = DocumentCleaner.clean(input)
        XCTAssertTrue(output.contains("# 标题"), output)
        XCTAssertTrue(output.contains("- 列表项"), output)
        XCTAssertTrue(output.contains("[链接](https://example.com/a?b=1&c=2)"), output)
        XCTAssertTrue(output.contains("**加粗**"), output)
        XCTAssertTrue(output.contains("> 引用"), output)
        XCTAssertTrue(output.contains("\n---\n"), output)
        XCTAssertTrue(output.contains("…"), output)
        XCTAssertTrue(output.contains("`<span>`"), output)
        XCTAssertTrue(output.contains("let x = \"<p>代码</p>\""), output)
        XCTAssertTrue(output.contains("HTML"), output)
        XCTAssertFalse(output.contains("<span>HTML</span>"), output)
        XCTAssertFalse(output.contains("<tag>"), output)
        XCTAssertFalse(output.unicodeScalars.contains { $0.value == 0x200B })
        XCTAssertFalse(output.unicodeScalars.contains { $0.value == 0xFFFD })
    }

    func testCleanerPreservesMarkdownIndentation() throws {
        let input = """
        # 标题

        1. 父项
            - 子项
                代码块行
        """
        let output = DocumentCleaner.clean(input)
        XCTAssertTrue(output.contains("1. 父项"), output)
        XCTAssertTrue(output.contains("    - 子项"), output)
        XCTAssertTrue(output.contains("        代码块行"), output)
    }

    func testCleanerStripsPlainTextHTMLAndNoise() throws {
        let input = """
        <p>Hello <b>world</b></p><div>Next&nbsp;line &amp; more</div>
        这是一段足够长的正文内容••••应当保留，访问 https://example.com/x 获取更多。
        """
        let output = DocumentCleaner.clean(input)
        XCTAssertTrue(output.contains("Hello world"), output)
        XCTAssertTrue(output.contains("Next line & more"), output)
        XCTAssertTrue(output.contains("这是一段足够长的正文内容 应当保留"), output)
        XCTAssertFalse(output.contains("<p>"), output)
        XCTAssertFalse(output.contains("<b>"), output)
        XCTAssertFalse(output.contains("••••"), output)
        XCTAssertFalse(output.contains("https://example.com"), output)
        XCTAssertFalse(output.contains("&nbsp;"), output)
        XCTAssertFalse(output.contains("&amp;"), output)
    }

    func testTableLayoutDetectorFindsGridAndRejectsLines() throws {
        let grid = try makeLineImage(width: 800, height: 600,
                                     rows: [120, 300, 480],
                                     columns: [120, 400, 680])
        XCTAssertTrue(PDFPageRenderer.looksLikeTable(grid))

        let horizontalOnly = try makeLineImage(width: 800, height: 600,
                                               rows: [120, 300, 480], columns: [])
        XCTAssertFalse(PDFPageRenderer.looksLikeTable(horizontalOnly))

        let blank = try makeLineImage(width: 800, height: 600, rows: [], columns: [])
        XCTAssertFalse(PDFPageRenderer.looksLikeTable(blank))
    }

    func testOvisParserReplacesTextTableWithOvisMarkdown() throws {
        let url = directory.appendingPathComponent("text-table.pdf")
        let text = """
        Intro before table.

        | Name | Value |
        | --- | --- |
        | A | 1 |
        | B | 2 |

        After table.
        """
        try makeTextPDF(at: url, text: text)

        let fake = TableFakeVLM()
        let parser = OvisPDFParser(vlm: fake)
        let parsed = try awaitSync {
            try await parser.parse(url: url, options: ParserOptions(pdfEngine: .ovis), progress: nil)
        }
        XCTAssertEqual(fake.calls, 1)
        XCTAssertTrue(parsed.text.contains("Intro before table"), parsed.text)
        XCTAssertTrue(parsed.text.contains("After table"), parsed.text)
        XCTAssertTrue(parsed.text.contains("| X | 99 |"), parsed.text)
        XCTAssertFalse(parsed.text.contains("| Name | Value |"), parsed.text)
        XCTAssertEqual(parsed.metadata["table_regions"], "1")
    }

    func testHybridParserFallsBackToOvisForLowConfidencePage() throws {
        let url = directory.appendingPathComponent("low-text-hybrid.pdf")
        // 只有 3 个字符：Vision 识别结果低于有效字符阈值，应回退 OvisOCR3。
        try makeImagePDF(at: url, text: "abc")
        let fake = FakeVLM()
        let parser = OvisPDFParser(vlm: fake)
        let parsed = try awaitSync {
            try await parser.parse(url: url, options: ParserOptions(pdfEngine: .hybrid),
                                   progress: nil)
        }
        XCTAssertEqual(fake.calls, 1, "Vision 文本过少时必须回退 OvisOCR3")
        XCTAssertTrue(parsed.text.contains("Ovis page content"), parsed.text)
        XCTAssertEqual(parsed.metadata["engine"], "hybrid")
        XCTAssertEqual(parsed.metadata["vision_pages"], "0")
        XCTAssertEqual(parsed.metadata["ocr_pages"], "1")
    }

    // MARK: EPUB

    func testEPUBParsingWithZipFoundationAndSwiftSoup() throws {
        let url = directory.appendingPathComponent("book.epub")
        try makeEPUB(at: url)
        let parser = EPUBParser()
        let parsed = try awaitSync {
            try await parser.parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("Chapter One"))
        XCTAssertTrue(parsed.text.contains("Hello EPUB"))
        XCTAssertTrue(parsed.outline.contains("Chapter One"))
        XCTAssertEqual(parsed.metadata["engine"], "epub")
    }

    func testScannedEPUBUsesNormalizedImageOCRViaOvis() throws {
        let url = directory.appendingPathComponent("scanned.epub")
        let image = try makePNGData(width: 1600, height: 2400)
        try makeScannedEPUB(at: url, imageData: image)

        let fake = FakeVLM()
        let parser = EPUBParser(vlm: fake)
        let parsed = try awaitSync {
            try await parser.parse(url: url,
                                   options: ParserOptions(epubOCREngine: .ovis),
                                   progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("Ovis page content"), parsed.text)
        XCTAssertEqual(fake.calls, 1)
        // 归一化把长边 2400 压到 1500（1600×0.625=1000）。
        XCTAssertEqual(fake.lastImageWidth, 1000)
        XCTAssertEqual(fake.lastImageHeight, 1500)
        XCTAssertEqual(parsed.metadata["ocr_engine"], "ovis")
        XCTAssertEqual(parsed.metadata["ocr_pages"], "1")
        XCTAssertEqual(parsed.metadata["scanned_chapters"], "1")
    }

    /// Epubor 等转制书：整本书挤在一个 spine 条目里（约 9000 字目录文本 +
    /// 300 张扫描页图，模拟真实扫描书的文字层总量）。总量远超逐章阈值，
    /// 只能靠整书预采样（平均每图文字）判定。
    func testMonolithicScannedEPUBTriggersOCRBeyondTextThreshold() throws {
        let url = directory.appendingPathComponent("monolithic.epub")
        let image = try makePNGData(width: 320, height: 480)
        try makeMonolithicEPUB(at: url, imageData: image, imageCount: 300)

        let fake = NumberedFakeVLM()
        let parser = EPUBParser(vlm: fake)
        let parsed = try awaitSync {
            try await parser.parse(url: url,
                                   options: ParserOptions(epubOCREngine: .ovis),
                                   progress: nil)
        }
        XCTAssertEqual(fake.calls, 300, "预采样判定扫描书后，含图条目必须整书 OCR")
        XCTAssertEqual(parsed.metadata["ocr_pages"], "300")
        XCTAssertTrue(parsed.text.contains("第 3 页 OCR 正文"), parsed.text)
        XCTAssertTrue(parsed.text.contains("第 300 页 OCR 正文"), parsed.text)
        XCTAssertEqual(parsed.metadata["scanned_chapters"], "1")
    }

    /// 短章组合洞：预采样不命中（前 20 条目纯文字无图），尾部一条目只有
    /// 25 张图 + 约 3000 字文字层（图 ≤30 且字 <5000，旧双硬边界漏判），
    /// 但平均每图仅 ~120 字 < 150 → 必须经逐章比值判定触发 OCR。
    func testShortScannedChapterTriggersOCRViaRatioFallback() throws {
        let url = directory.appendingPathComponent("mixed.epub")
        let image = try makePNGData(width: 320, height: 480)
        try makeMixedEPUB(at: url, textChapterCount: 20,
                          imageData: image, imageCount: 25,
                          scannedFillerLines: 88)

        let fake = NumberedFakeVLM()
        let parser = EPUBParser(vlm: fake)
        let parsed = try awaitSync {
            try await parser.parse(url: url,
                                   options: ParserOptions(epubOCREngine: .ovis),
                                   progress: nil)
        }
        XCTAssertEqual(fake.calls, 25, "每图文字不足 150 的条目必须走 OCR，无论图数是否超过 30")
        XCTAssertEqual(parsed.metadata["ocr_pages"], "25")
        XCTAssertTrue(parsed.text.contains("第 1 页 OCR 正文"), parsed.text)
        XCTAssertTrue(parsed.text.contains("第 25 页 OCR 正文"), parsed.text)
        XCTAssertEqual(parsed.metadata["scanned_chapters"], "1")
    }

    /// 多章扫描书：各章 OCR 必须跨章并发，不能按 spine 串行等待。
    /// 页号仍按 spine 顺序预分配，输出页标记不能错乱。
    func testScannedEPUBOCRunsChaptersConcurrently() throws {
        let url = directory.appendingPathComponent("multi-scanned.epub")
        let image = try makePNGData(width: 320, height: 480)
        try makeMultiChapterScannedEPUB(at: url, chapterCount: 3, imageData: image)

        let fake = ConcurrencyTrackingVLM()
        let parser = EPUBParser(vlm: fake)
        let parsed = try awaitSync {
            try await parser.parse(url: url,
                                   options: ParserOptions(epubOCREngine: .ovis),
                                   progress: nil)
        }
        XCTAssertEqual(fake.calls, 3)
        XCTAssertGreaterThanOrEqual(fake.maxConcurrent, 2, "章节 OCR 必须跨章并发")
        XCTAssertEqual(parsed.metadata["ocr_pages"], "3")
        XCTAssertEqual(parsed.metadata["scanned_chapters"], "3")
        for page in 1...3 {
            XCTAssertTrue(parsed.text.contains("第 \(page) 页"), "页标记必须按 spine 顺序保留：\(parsed.text)")
        }
    }

    /// 真实失败形态：spine 只有封面和目录页，正文文件都在 manifest 里但没被
    /// spine 引用。旧实现只在 spine 完全为空时回退，结果只抽出书名/目录。
    func testEPUBIncompleteSpineRecoversManifestChapters() throws {
        let url = directory.appendingPathComponent("partial-spine.epub")
        try writeEPUB(at: url, entries: [
            ("mimetype", Data("application/epub+zip".utf8)),
            ("META-INF/container.xml", Data(Self.containerXML.utf8)),
            ("OEBPS/content.opf", Data("""
            <?xml version="1.0"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
              <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:title>只有书名和目录</dc:title><dc:identifier id="id">partial</dc:identifier>
              </metadata>
              <manifest>
                <item id="cover" href="cover.xhtml" media-type="application/xhtml+xml"/>
                <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
                <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
                <item id="ch2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
              </manifest>
              <spine><itemref idref="cover"/><itemref idref="nav"/></spine>
            </package>
            """.utf8)),
            ("OEBPS/cover.xhtml", Data("""
            <html xmlns="http://www.w3.org/1999/xhtml"><body><h1>只有书名和目录</h1></body></html>
            """.utf8)),
            ("OEBPS/nav.xhtml", Data("""
            <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><body>
            <nav epub:type="toc"><ol>
              <li><a href="ch1.xhtml">第一章 起点</a></li>
              <li><a href="ch2.xhtml">第二章 转折</a></li>
            </ol></nav></body></html>
            """.utf8)),
            ("OEBPS/ch1.xhtml", Data("""
            <html xmlns="http://www.w3.org/1999/xhtml"><body><h1>第一章 起点</h1>
            <p>他推开那扇门，真正的第一章正文从这里开始。</p></body></html>
            """.utf8)),
            ("OEBPS/ch2.xhtml", Data("""
            <html xmlns="http://www.w3.org/1999/xhtml"><body><h1>第二章 转折</h1>
            <p>第二章正文继续讲述后面的故事。</p></body></html>
            """.utf8)),
        ])

        let parser = EPUBParser()
        let parsed = try awaitSync {
            try await parser.parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("真正的第一章正文从这里开始"), parsed.text)
        XCTAssertTrue(parsed.text.contains("第二章正文继续讲述后面的故事"), parsed.text)
        XCTAssertEqual(parsed.metadata["recovered_chapters"], "2")
    }

    /// 部分转制工具在 OPF 里写 Windows 反斜杠路径，ZIP 内实际是 `/`。
    func testEPUBBackslashHrefResolvesToArchiveEntry() throws {
        let url = directory.appendingPathComponent("backslash.epub")
        try writeEPUB(at: url, entries: [
            ("mimetype", Data("application/epub+zip".utf8)),
            ("META-INF/container.xml", Data(Self.containerXML.utf8)),
            ("OEBPS/content.opf", Data("""
            <?xml version="1.0"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
              <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:title>反斜杠路径</dc:title><dc:identifier id="id">slash</dc:identifier>
              </metadata>
              <manifest>
                <item id="ch1" href="Text\\Chapter1.xhtml" media-type="application/xhtml+xml"/>
              </manifest>
              <spine><itemref idref="ch1"/></spine>
            </package>
            """.utf8)),
            ("OEBPS/Text/Chapter1.xhtml", Data("""
            <html xmlns="http://www.w3.org/1999/xhtml"><body>
            <p>反斜杠路径章节正文。</p></body></html>
            """.utf8)),
        ])

        let parsed = try awaitSync {
            try await EPUBParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("反斜杠路径章节正文"), parsed.text)
    }

    /// OPF href 与 ZIP 条目大小写不一致（Windows 工具常见），需按大小写
    /// 不敏感回退查找，否则整章被静默跳过。
    func testEPUBHrefCaseMismatchResolvesToArchiveEntry() throws {
        let url = directory.appendingPathComponent("case-mismatch.epub")
        try writeEPUB(at: url, entries: [
            ("mimetype", Data("application/epub+zip".utf8)),
            ("META-INF/container.xml", Data(Self.containerXML.utf8)),
            ("OEBPS/content.opf", Data("""
            <?xml version="1.0"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
              <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:title>大小写不一致</dc:title><dc:identifier id="id">case</dc:identifier>
              </metadata>
              <manifest>
                <item id="ch1" href="text/chapter1.xhtml" media-type="application/xhtml+xml"/>
              </manifest>
              <spine><itemref idref="ch1"/></spine>
            </package>
            """.utf8)),
            ("OEBPS/Text/Chapter1.xhtml", Data("""
            <html xmlns="http://www.w3.org/1999/xhtml"><body>
            <p>大小写不一致的章节正文仍然可读。</p></body></html>
            """.utf8)),
        ])

        let parsed = try awaitSync {
            try await EPUBParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("大小写不一致的章节正文仍然可读"), parsed.text)
    }

    /// 固定版式 EPUB 常把正文画在 SVG `<text>` 里；旧实现整块跳过 svg，
    /// 章节只剩目录/页码文字。
    func testEPUBSVGTextIsExtracted() throws {
        let url = directory.appendingPathComponent("svg-text.epub")
        try writeEPUB(at: url, entries: [
            ("mimetype", Data("application/epub+zip".utf8)),
            ("META-INF/container.xml", Data(Self.containerXML.utf8)),
            ("OEBPS/content.opf", Data("""
            <?xml version="1.0"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
              <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:title>固定版式</dc:title><dc:identifier id="id">svg</dc:identifier>
              </metadata>
              <manifest>
                <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
                <item id="page" href="page1.xhtml" media-type="application/xhtml+xml"/>
              </manifest>
              <spine><itemref idref="nav"/><itemref idref="page"/></spine>
            </package>
            """.utf8)),
            ("OEBPS/nav.xhtml", Data("""
            <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><body>
            <nav epub:type="toc"><ol><li><a href="page1.xhtml">第一页</a></li></ol></nav></body></html>
            """.utf8)),
            ("OEBPS/page1.xhtml", Data("""
            <html xmlns="http://www.w3.org/1999/xhtml"><body>
            <div class="page"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 800">
              <text x="40" y="80" font-size="24">固定版式 SVG 正文第一行</text>
              <text x="40" y="120" font-size="24">第二行也在 SVG 里</text>
            </svg></div></body></html>
            """.utf8)),
        ])

        let parsed = try awaitSync {
            try await EPUBParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("固定版式 SVG 正文第一行"), parsed.text)
        XCTAssertTrue(parsed.text.contains("第二行也在 SVG 里"), parsed.text)
    }

    /// 部分书 spine 只有封面 + 扉页（titlepage），正文全在 manifest 里。
    /// 扉页不属于正文章节，否则会被误判为「spine 完整」。
    func testEPUBSpineWithOnlyFrontMatterRecoversBody() throws {
        let url = directory.appendingPathComponent("front-matter-spine.epub")
        try writeEPUB(at: url, entries: [
            ("mimetype", Data("application/epub+zip".utf8)),
            ("META-INF/container.xml", Data(Self.containerXML.utf8)),
            ("OEBPS/content.opf", Data("""
            <?xml version="1.0"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
              <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:title>扉页之后</dc:title><dc:identifier id="id">front</dc:identifier>
              </metadata>
              <manifest>
                <item id="cover" href="cover.xhtml" media-type="application/xhtml+xml"/>
                <item id="titlepage" href="titlepage.xhtml" media-type="application/xhtml+xml"/>
                <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
              </manifest>
              <spine><itemref idref="cover"/><itemref idref="titlepage"/></spine>
            </package>
            """.utf8)),
            ("OEBPS/cover.xhtml", Data("""
            <html xmlns="http://www.w3.org/1999/xhtml"><body><h1>扉页之后</h1></body></html>
            """.utf8)),
            ("OEBPS/titlepage.xhtml", Data("""
            <html xmlns="http://www.w3.org/1999/xhtml"><body><p>扉页</p></body></html>
            """.utf8)),
            ("OEBPS/ch1.xhtml", Data("""
            <html xmlns="http://www.w3.org/1999/xhtml"><body><h1>第一章</h1>
            <p>扉页之后的正文终于出现在这里。</p></body></html>
            """.utf8)),
        ])

        let parsed = try awaitSync {
            try await EPUBParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("扉页之后的正文终于出现在这里"), parsed.text)
        XCTAssertEqual(parsed.metadata["recovered_chapters"], "1")
    }

    /// 非 UTF-8 的 XHTML（常见于繁体 EPUB）必须按 `<meta charset>` 解码，
    /// 否则整章变成乱码或空文本。
    func testEPUBDeclaredBig5EncodingIsDecoded() throws {
        let phrase = "繁體中文正文測試，這一頁的內容應該完整可讀。"
        let big5 = String.Encoding(rawValue: CFStringConvertEncodingToNSStringEncoding(
            CFStringEncoding(CFStringEncodings.big5.rawValue)))
        let html = """
        <html><head><meta charset="big5"/></head><body><p>\(phrase)</p></body></html>
        """
        let data = try XCTUnwrap(html.data(using: big5), "Big5 编码失败")

        let url = directory.appendingPathComponent("big5.epub")
        try writeEPUB(at: url, entries: [
            ("mimetype", Data("application/epub+zip".utf8)),
            ("META-INF/container.xml", Data(Self.containerXML.utf8)),
            ("OEBPS/content.opf", Data("""
            <?xml version="1.0"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
              <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:title>Big5 書</dc:title><dc:identifier id="id">big5</dc:identifier>
              </metadata>
              <manifest>
                <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
              </manifest>
              <spine><itemref idref="ch1"/></spine>
            </package>
            """.utf8)),
            ("OEBPS/ch1.xhtml", data),
        ])

        let parsed = try awaitSync {
            try await EPUBParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains(phrase), parsed.text)
    }

    /// 图片页不一定用 `<img src>`：背景图、srcset、`<object data>` 都要能
    /// 被扫描页判定捕获，否则整章被当成空文本跳过。
    func testEPUBNonImgSrcImagesTriggerOCR() throws {
        let url = directory.appendingPathComponent("image-refs.epub")
        let image = try makePNGData(width: 320, height: 480)
        try writeEPUB(at: url, entries: [
            ("mimetype", Data("application/epub+zip".utf8)),
            ("META-INF/container.xml", Data(Self.containerXML.utf8)),
            ("OEBPS/content.opf", Data("""
            <?xml version="1.0"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
              <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:title>图片引用</dc:title><dc:identifier id="id">refs</dc:identifier>
              </metadata>
              <manifest>
                <item id="p1" href="page1.xhtml" media-type="application/xhtml+xml"/>
                <item id="p2" href="page2.xhtml" media-type="application/xhtml+xml"/>
                <item id="p3" href="page3.xhtml" media-type="application/xhtml+xml"/>
                <item id="i1" href="images/page1.png" media-type="image/png"/>
                <item id="i2" href="images/page2.png" media-type="image/png"/>
                <item id="i3" href="images/page3.png" media-type="image/png"/>
              </manifest>
              <spine><itemref idref="p1"/><itemref idref="p2"/><itemref idref="p3"/></spine>
            </package>
            """.utf8)),
            ("OEBPS/page1.xhtml", Data("""
            <html xmlns="http://www.w3.org/1999/xhtml"><body>
            <div style="background-image:url('images/page1.png');width:100%;height:100%"></div>
            </body></html>
            """.utf8)),
            ("OEBPS/page2.xhtml", Data("""
            <html xmlns="http://www.w3.org/1999/xhtml"><body>
            <img srcset="images/page2.png 1x" alt=""/>
            </body></html>
            """.utf8)),
            ("OEBPS/page3.xhtml", Data("""
            <html xmlns="http://www.w3.org/1999/xhtml"><body>
            <object data="images/page3.png" type="image/png"></object>
            </body></html>
            """.utf8)),
            ("OEBPS/images/page1.png", image),
            ("OEBPS/images/page2.png", image),
            ("OEBPS/images/page3.png", image),
        ])

        let fake = FakeVLM()
        let parsed = try awaitSync {
            try await EPUBParser(vlm: fake).parse(
                url: url, options: ParserOptions(epubOCREngine: .ovis), progress: nil)
        }
        XCTAssertEqual(fake.calls, 3, "三种非 <img src> 的图片引用都应触发 OCR")
        XCTAssertEqual(parsed.metadata["scanned_chapters"], "3")
        XCTAssertEqual(parsed.metadata["ocr_pages"], "3")
    }

    // MARK: 配置迁移

    func testConfigMigrationRemovesDoclingAndMarker() throws {
        let path = directory.appendingPathComponent("config.json")
        let json = """
        {"rag_pdf_engine":"docling","rag_embedding_model":"all-MiniLM-L6-v2",
         "workspace":"\(directory.path)","rag_chunk_size":500,"rag_chunk_overlap":900}
        """
        try json.write(to: path, atomically: true, encoding: .utf8)
        let config = AgentConfig.load(path.path)
        // docling 已移除：未识别的旧值统一回退到 Apple Vision，避免默默走 Ovis。
        XCTAssertEqual(config.ragPDFEngine, .vision)
        XCTAssertEqual(config.ragEmbeddingModel, "jina-embeddings-v5-text-nano-mlx-int4")
        // overlap 越界时按默认比例（chunk_size 的 10%）重置：500 / 10 = 50。
        XCTAssertEqual(config.ragChunkOverlap, 50)
    }

    func testConfigMigratesOldChunkDefaults() throws {
        let path = directory.appendingPathComponent("config-chunks.json")
        try #"{"rag_chunk_size":500,"rag_chunk_overlap":50}"#
            .write(to: path, atomically: true, encoding: .utf8)
        let config = AgentConfig.load(path.path)
        // 旧默认 500/50（bge 512 窗口）迁移到 800/80（nomic 8192 窗口）。
        XCTAssertEqual(config.ragChunkSize, 800)
        XCTAssertEqual(config.ragChunkOverlap, 80)
    }

    /// 新默认值：jina-v5 MLX int4 + 固定 128 维 + int8、800/80 切片等。
    func testConfigNewRAGDefaults() throws {
        let config = AgentConfig()
        XCTAssertEqual(config.ragEmbeddingModel, "jina-embeddings-v5-text-nano-mlx-int4")
        XCTAssertEqual(config.localEmbeddingModel, "jina-embeddings-v5-text-nano-mlx-int4")
        XCTAssertEqual(config.localEmbeddingModelPath,
                       "models/jina-embeddings-v5-text-nano-mlx-int4")
        XCTAssertEqual(config.localEmbeddingBackend, "mlx")
        XCTAssertEqual(config.ragMatryoshka, .auto)
        XCTAssertEqual(config.ragMatryoshkaDim, 128)
        XCTAssertEqual(config.ragVectorElementType, .int8)
        XCTAssertEqual(config.ragChunkSize, 800)
        XCTAssertEqual(config.ragChunkOverlap, 80)
        XCTAssertEqual(config.ragSearchMaxTotalChars, 12_000)
        XCTAssertEqual(config.ragAuxSummarizeMinChars, 5_000)
        XCTAssertEqual(config.ragAuxSummarizeMaxChars, 20_000)
        XCTAssertEqual(config.ragVectorOverfetch, 2,
                       "混合检索先融合 rowid/分数，内层只需 1-2 倍召回")
        let options = ChunkingOptions(config: config)
        XCTAssertEqual(options.chunkSizeTokens, 800, "切片预算按 token 计，含 overlap")
        XCTAssertEqual(options.overlapTokens, 80, "overlap 占 10%，计入 800 token 总预算")

        // 阈值可持久化并回读（用户后续可直接改配置文件调参）。
        let path = directory.appendingPathComponent("config-rag-thresholds.json")
        let json = #"{"rag_aux_summarize_min_chars":3000,"rag_aux_summarize_max_chars":9000,"#
            + #""rag_search_max_total_chars":10000,"rag_rerank_threshold":0.55}"#
        try json.write(to: path, atomically: true, encoding: .utf8)
        let loaded = AgentConfig.load(path.path)
        XCTAssertEqual(loaded.ragAuxSummarizeMinChars, 3_000)
        XCTAssertEqual(loaded.ragAuxSummarizeMaxChars, 9_000)
        XCTAssertEqual(loaded.ragSearchMaxTotalChars, 10_000)
        XCTAssertEqual(loaded.ragRerankThreshold, 0.55)
        XCTAssertEqual(loaded.export()["rag_aux_summarize_min_chars"] as? Int, 3_000)
    }

    /// jina-v5 开放的 Matryoshka 维度：六档合法，其它值回落 128。
    func testConfigAcceptsJinaV5MatryoshkaDimensions() throws {
        for dimension in [32, 64, 128, 256, 512, 768] {
            let path = directory.appendingPathComponent("config-mrl-\(dimension).json")
            try #"{"rag_matryoshka_dim":\#(dimension)}"#
                .write(to: path, atomically: true, encoding: .utf8)
            XCTAssertEqual(AgentConfig.load(path.path).ragMatryoshkaDim, dimension,
                           "jina-v5 声明的维度必须可持久化并回读")
        }

        let invalid = directory.appendingPathComponent("config-mrl-invalid.json")
        try #"{"rag_matryoshka_dim":100}"#
            .write(to: invalid, atomically: true, encoding: .utf8)
        XCTAssertEqual(AgentConfig.load(invalid.path).ragMatryoshkaDim, 128,
                       "未训练维度必须回落到默认 128")
    }

    /// embedding token 预算与指纹校验开关：默认值、持久化与越界收口。
    func testConfigEmbeddingTokenBudgetAndFingerprintVerify() throws {
        let defaults = AgentConfig()
        XCTAssertEqual(defaults.ragEmbeddingTokenBudgetMin, 4_096)
        XCTAssertEqual(defaults.ragEmbeddingTokenBudgetMax, 8_192)
        XCTAssertEqual(defaults.ragEmbeddingTokenBudgetInitial, 6_144)
        XCTAssertEqual(defaults.ragEmbeddingTokenBudgetStep, 512)
        XCTAssertFalse(defaults.ragVerifyFingerprint, "默认信任 mtime + size")
        XCTAssertEqual(defaults.ragANNProbe, 4, "离线评测后默认探 4 个簇")
        XCTAssertEqual(defaults.ragANNRescore, 64)

        let path = directory.appendingPathComponent("config-token-budget.json")
        let json = #"{"rag_embedding_token_budget_min":8192,"#
            + #""rag_embedding_token_budget_max":2048,"#
            + #""rag_embedding_token_budget_initial":1000,"#
            + #""rag_embedding_token_budget_step":64,"#
            + #""rag_verify_fingerprint":true}"#
        try json.write(to: path, atomically: true, encoding: .utf8)
        let loaded = AgentConfig.load(path.path)
        XCTAssertEqual(loaded.ragEmbeddingTokenBudgetMin, 8_192)
        XCTAssertEqual(loaded.ragEmbeddingTokenBudgetMax, 8_192, "上限不得低于下限")
        XCTAssertEqual(loaded.ragEmbeddingTokenBudgetInitial, 8_192, "初始预算夹在区间内")
        XCTAssertEqual(loaded.ragEmbeddingTokenBudgetStep, 128, "步进至少 128")
        XCTAssertTrue(loaded.ragVerifyFingerprint)
        XCTAssertEqual(loaded.export()["rag_verify_fingerprint"] as? Bool, true)
        XCTAssertEqual(loaded.export()["rag_embedding_token_budget_min"] as? Int, 8_192)
    }

    /// 旧默认 nomic MLX int8 / jina-v3 MLX int4 应迁移到 jina-v5 MLX int4，
    /// 并把旧默认 512 / 256 维收口到 128。
    func testConfigMigratesLegacyEmbeddingDefaultsToJinaV5() throws {
        let path = directory.appendingPathComponent("config-embedding-migration.json")
        let json = """
        {"local_embedding_backend":"mlx",
         "local_embedding_model":"nomic-embed-text-v1.5-mlx-int8",
         "local_embedding_model_path":"models/nomic-embed-text-v1.5-mlx-int8",
         "rag_embedding_model":"nomic-embed-text-v1.5-mlx-int8",
         "rag_matryoshka_dim":512}
        """
        try json.write(to: path, atomically: true, encoding: .utf8)
        let config = AgentConfig.load(path.path)

        XCTAssertEqual(config.ragEmbeddingModel, "jina-embeddings-v5-text-nano-mlx-int4")
        XCTAssertEqual(config.localEmbeddingModel, "jina-embeddings-v5-text-nano-mlx-int4")
        XCTAssertEqual(config.localEmbeddingModelPath,
                       "models/jina-embeddings-v5-text-nano-mlx-int4")
        XCTAssertEqual(config.ragMatryoshkaDim, 128)
        XCTAssertEqual(config.ragVectorElementType, .int8)
    }

    /// 上一版默认 jina-v3 也要迁移到 V5；用户自定义模型保持不变。
    func testConfigMigratesJinaV3DefaultButKeepsCustomModel() throws {
        let path = directory.appendingPathComponent("config-embedding-v3-migration.json")
        let json = """
        {"local_embedding_backend":"mlx",
         "local_embedding_model":"jina-embeddings-v3-mlx-int4",
         "local_embedding_model_path":"models/jina-embeddings-v3-mlx-int4",
         "rag_embedding_model":"jina-embeddings-v3-mlx-int4",
         "rag_matryoshka_dim":256}
        """
        try json.write(to: path, atomically: true, encoding: .utf8)
        let migrated = AgentConfig.load(path.path)
        XCTAssertEqual(migrated.ragEmbeddingModel, "jina-embeddings-v5-text-nano-mlx-int4")
        XCTAssertEqual(migrated.localEmbeddingModel,
                       "jina-embeddings-v5-text-nano-mlx-int4")
        XCTAssertEqual(migrated.localEmbeddingModelPath,
                       "models/jina-embeddings-v5-text-nano-mlx-int4")
        XCTAssertEqual(migrated.ragMatryoshkaDim, 128)

        let customPath = directory.appendingPathComponent("config-embedding-custom.json")
        let custom = """
        {"local_embedding_backend":"mlx",
         "local_embedding_model":"jina-embeddings-v3-mlx-int4",
         "local_embedding_model_path":"models/my-embedding",
         "rag_embedding_model":"my-embedding",
         "rag_matryoshka_dim":512}
        """
        try custom.write(to: customPath, atomically: true, encoding: .utf8)
        let kept = AgentConfig.load(customPath.path)
        XCTAssertEqual(kept.ragEmbeddingModel, "my-embedding")
        XCTAssertEqual(kept.localEmbeddingModel, "jina-embeddings-v3-mlx-int4")
        XCTAssertEqual(kept.localEmbeddingModelPath, "models/my-embedding")
        XCTAssertEqual(kept.ragMatryoshkaDim, 512,
                       "用户显式配置的 Matryoshka 维不变")
    }

    /// 跨轮 KV/前缀复用开关：默认开启、可关闭，且写入导出字典。
    func testConfigKVReuseToggle() throws {
        XCTAssertTrue(AgentConfig().llmKVReuse,
                      "默认应开启跨轮 KV 复用")
        let path = directory.appendingPathComponent("config-kv-reuse.json")
        try #"{"llm_kv_reuse":false}"#
            .write(to: path, atomically: true, encoding: .utf8)
        let config = AgentConfig.load(path.path)
        XCTAssertFalse(config.llmKVReuse)
        XCTAssertFalse(config.export()["llm_kv_reuse"] as? Bool ?? true)
    }

    /// 语义切片开关：默认开启、可关闭，并映射到切片选项。
    func testConfigSemanticChunkingToggle() throws {
        XCTAssertTrue(AgentConfig().ragSemanticChunking)
        let path = directory.appendingPathComponent("config-semantic.json")
        try #"{"rag_semantic_chunking":false}"#
            .write(to: path, atomically: true, encoding: .utf8)
        let config = AgentConfig.load(path.path)
        XCTAssertFalse(config.ragSemanticChunking)
        XCTAssertFalse(config.export()["rag_semantic_chunking"] as? Bool ?? true)
        XCTAssertFalse(ChunkingOptions(config: config).enableSemantic)
    }

    /// MLX buffer cache 上限：默认 1GB，允许试到 2GB，超大值加载时压回 2GB。
    func testConfigClampsMLXCacheLimit() throws {
        XCTAssertEqual(AgentConfig().mlxCacheLimitMB, 1_024)

        let path = directory.appendingPathComponent("config-mlx-cache.json")
        try #"{"mlx_cache_limit_mb":2048,"mlx_memory_limit_mb":0}"#
            .write(to: path, atomically: true, encoding: .utf8)
        var config = AgentConfig.load(path.path)
        XCTAssertEqual(config.mlxCacheLimitMB, 2_048)
        XCTAssertEqual(config.export()["mlx_cache_limit_mb"] as? Int, 2_048)

        try #"{"mlx_cache_limit_mb":4096}"#
            .write(to: path, atomically: true, encoding: .utf8)
        config = AgentConfig.load(path.path)
        XCTAssertEqual(config.mlxCacheLimitMB, 2_048)

        try #"{"mlx_cache_limit_mb":128}"#
            .write(to: path, atomically: true, encoding: .utf8)
        config = AgentConfig.load(path.path)
        XCTAssertEqual(config.mlxCacheLimitMB, 256)
    }

    /// Embedding 子批自适应参数：默认 2 / 32 / 16 / 2 / 6144 / 3% / 2 批 / 2s，
    /// 并夹取越界值与下限高于上限的冲突。
    func testConfigEmbeddingSchedulerDefaultsAndClamps() throws {
        let defaults = AgentConfig()
        XCTAssertEqual(defaults.ragEmbeddingMinBatch, 2)
        XCTAssertEqual(defaults.ragEmbeddingMaxBatch, 32)
        XCTAssertEqual(defaults.ragEmbeddingInitialBatch, 16)
        XCTAssertEqual(defaults.ragEmbeddingBatchStep, 2)
        XCTAssertEqual(defaults.ragEmbeddingMemoryLimitMB, 6_144)
        XCTAssertEqual(defaults.ragEmbeddingThroughputTolerance, 0.03, accuracy: 1e-9)
        XCTAssertEqual(defaults.ragEmbeddingProbeBatches, 2)
        XCTAssertEqual(defaults.ragEmbeddingProbeSeconds, 2.0, accuracy: 1e-9)

        let path = directory.appendingPathComponent("config-embedding-scheduler.json")
        let high = """
        {"rag_embedding_min_batch":999,
         "rag_embedding_max_batch":999,
         "rag_embedding_initial_batch":999,
         "rag_embedding_batch_step":999,
         "rag_embedding_memory_limit_mb":999999,
         "rag_embedding_throughput_tolerance":9,
         "rag_embedding_probe_batches":99,
         "rag_embedding_probe_seconds":999}
        """
        try high.write(to: path, atomically: true, encoding: .utf8)
        var config = AgentConfig.load(path.path)
        XCTAssertEqual(config.ragEmbeddingMinBatch, 64)
        XCTAssertEqual(config.ragEmbeddingMaxBatch, 256)
        XCTAssertEqual(config.ragEmbeddingInitialBatch, 256)
        XCTAssertEqual(config.ragEmbeddingBatchStep, 16)
        XCTAssertEqual(config.ragEmbeddingMemoryLimitMB, 32_768)
        XCTAssertEqual(config.ragEmbeddingThroughputTolerance, 0.5, accuracy: 1e-9)
        XCTAssertEqual(config.ragEmbeddingProbeBatches, 16)
        XCTAssertEqual(config.ragEmbeddingProbeSeconds, 30, accuracy: 1e-9)
        let exported = config.export()
        XCTAssertEqual(exported["rag_embedding_min_batch"] as? Int, 64)
        XCTAssertEqual(exported["rag_embedding_max_batch"] as? Int, 256)
        XCTAssertEqual(exported["rag_embedding_initial_batch"] as? Int, 256)
        XCTAssertEqual(exported["rag_embedding_batch_step"] as? Int, 16)
        XCTAssertEqual(exported["rag_embedding_memory_limit_mb"] as? Int, 32_768)
        XCTAssertEqual(exported["rag_embedding_throughput_tolerance"] as? Double, 0.5)
        XCTAssertEqual(exported["rag_embedding_probe_batches"] as? Int, 16)
        XCTAssertEqual(exported["rag_embedding_probe_seconds"] as? Double, 30)

        let low = """
        {"rag_embedding_min_batch":0,
         "rag_embedding_max_batch":1,
         "rag_embedding_initial_batch":0,
         "rag_embedding_batch_step":0,
         "rag_embedding_memory_limit_mb":128,
         "rag_embedding_throughput_tolerance":0.0001,
         "rag_embedding_probe_batches":0,
         "rag_embedding_probe_seconds":0}
        """
        try low.write(to: path, atomically: true, encoding: .utf8)
        config = AgentConfig.load(path.path)
        XCTAssertEqual(config.ragEmbeddingMinBatch, 2)
        XCTAssertEqual(config.ragEmbeddingMaxBatch, 2)
        XCTAssertEqual(config.ragEmbeddingInitialBatch, 2)
        XCTAssertEqual(config.ragEmbeddingBatchStep, 1)
        XCTAssertEqual(config.ragEmbeddingMemoryLimitMB, 1_024)
        XCTAssertEqual(config.ragEmbeddingThroughputTolerance, 0.01, accuracy: 1e-9)
        XCTAssertEqual(config.ragEmbeddingProbeBatches, 1)
        XCTAssertEqual(config.ragEmbeddingProbeSeconds, 0.5, accuracy: 1e-9)

        // 下限高于上限、起始值超出区间时，按 max 收口。
        let conflict = """
        {"rag_embedding_min_batch":20,
         "rag_embedding_max_batch":8,
         "rag_embedding_initial_batch":99}
        """
        try conflict.write(to: path, atomically: true, encoding: .utf8)
        config = AgentConfig.load(path.path)
        XCTAssertEqual(config.ragEmbeddingMaxBatch, 8)
        XCTAssertEqual(config.ragEmbeddingMinBatch, 8)
        XCTAssertEqual(config.ragEmbeddingInitialBatch, 8)
    }

    /// 启动预加载主模型默认开启；`auto_preload_main` 可显式关闭/打开。
    func testConfigAutoPreloadMainDefaultsOn() throws {
        XCTAssertTrue(AgentConfig().autoPreloadMain)
        XCTAssertEqual(AgentConfig().export()["auto_preload_main"] as? Bool, true)

        let path = directory.appendingPathComponent("config-auto-preload-main.json")
        try #"{"auto_preload_main":false}"#
            .write(to: path, atomically: true, encoding: .utf8)
        var config = AgentConfig.load(path.path)
        XCTAssertFalse(config.autoPreloadMain)
        XCTAssertEqual(config.export()["auto_preload_main"] as? Bool, false)

        try #"{"auto_preload_main":true}"#
            .write(to: path, atomically: true, encoding: .utf8)
        config = AgentConfig.load(path.path)
        XCTAssertTrue(config.autoPreloadMain)
        XCTAssertEqual(config.export()["auto_preload_main"] as? Bool, true)
    }

    /// 辅助模型默认关闭：缺省即为 false，且 `aux_enabled` 可显式打开/关闭。
    func testConfigAuxEnabledDefaultsOff() throws {
        XCTAssertFalse(AgentConfig().auxEnabled)
        XCTAssertEqual(AgentConfig().export()["aux_enabled"] as? Bool, false)

        let path = directory.appendingPathComponent("config-aux-enabled.json")
        try #"{"aux_enabled":true}"#
            .write(to: path, atomically: true, encoding: .utf8)
        var config = AgentConfig.load(path.path)
        XCTAssertTrue(config.auxEnabled)
        XCTAssertEqual(config.export()["aux_enabled"] as? Bool, true)

        try #"{"aux_enabled":false}"#
            .write(to: path, atomically: true, encoding: .utf8)
        config = AgentConfig.load(path.path)
        XCTAssertFalse(config.auxEnabled)
        XCTAssertEqual(config.export()["aux_enabled"] as? Bool, false)
    }

    /// 辅助模型默认为低延迟/忠实原文调参：8k 上下文与 KV、低温、轻重复惩罚、
    /// 思考关闭；这些值仍应可被用户配置覆盖。
    func testConfigAuxModelLatencyDefaults() throws {
        let defaults = AgentConfig()
        XCTAssertEqual(defaults.auxMaxContextTokens, 8_192)
        XCTAssertEqual(defaults.auxKvCacheMaxTokens, 8_192)
        XCTAssertEqual(defaults.effectiveAuxContextTokens, 8_192)
        XCTAssertEqual(defaults.auxTemperature, 0.3)
        XCTAssertEqual(defaults.auxRepetitionPenalty, 1.1)
        XCTAssertFalse(defaults.auxLocalLLMEnableThinking)
        XCTAssertEqual(defaults.export()["aux_max_context_tokens"] as? Int, 8_192)
        XCTAssertEqual(defaults.export()["aux_kv_cache_max_tokens"] as? Int, 8_192)
        XCTAssertEqual(defaults.export()["aux_temperature"] as? Double, 0.3)
        XCTAssertEqual(defaults.export()["aux_repetition_penalty"] as? Double, 1.1)
        XCTAssertEqual(defaults.export()["aux_local_llm_enable_thinking"] as? Bool, false)
    }

    /// 新的低延迟默认值只作缺省，用户显式配置仍原样生效，不被强制覆盖。
    func testConfigAuxModelLatencyDefaultsRemainOverridable() throws {
        let path = directory.appendingPathComponent("config-aux-latency.json")
        try """
        {"aux_max_context_tokens":16384,"aux_kv_cache_max_tokens":4096,
         "aux_temperature":0.8,"aux_repetition_penalty":1.0,
         "aux_local_llm_enable_thinking":true}
        """.write(to: path, atomically: true, encoding: .utf8)
        let config = AgentConfig.load(path.path)
        XCTAssertEqual(config.auxMaxContextTokens, 16_384)
        XCTAssertEqual(config.auxKvCacheMaxTokens, 4_096)
        XCTAssertEqual(config.effectiveAuxContextTokens, 4_096)
        XCTAssertEqual(config.auxTemperature, 0.8)
        XCTAssertEqual(config.auxRepetitionPenalty, 1.0)
        XCTAssertTrue(config.auxLocalLLMEnableThinking)
        XCTAssertEqual(config.export()["aux_max_context_tokens"] as? Int, 16_384)
        XCTAssertEqual(config.export()["aux_kv_cache_max_tokens"] as? Int, 4_096)
        XCTAssertEqual(config.export()["aux_temperature"] as? Double, 0.8)
        XCTAssertEqual(config.export()["aux_repetition_penalty"] as? Double, 1.0)
        XCTAssertEqual(config.export()["aux_local_llm_enable_thinking"] as? Bool, true)
    }

    /// 辅助模型的 KV 复用 / 量化独立于主模型：旧配置缺失这两个键时，
    /// 复用保持默认关闭（与「用后即清」的既有行为一致）、量化跟随主模型；
    /// 写入后各自独立生效（面板上主/辅助各有一组开关）。
    func testConfigAuxKVSettingsFollowMainThenIndependent() throws {
        let path = directory.appendingPathComponent("config-aux-kv.json")

        // 旧配置只写了主模型的键：复用默认关闭，量化跟随主模型。
        try """
        {"llm_kv_reuse":true,"kv_cache_int8":false}
        """.write(to: path, atomically: true, encoding: .utf8)
        var config = AgentConfig.load(path.path)
        XCTAssertTrue(config.llmKVReuse)
        XCTAssertFalse(config.kvCacheInt8)
        XCTAssertFalse(config.auxKvReuse, "辅助模型默认不复用 KV")
        XCTAssertFalse(config.auxKvCacheInt8, "缺省应跟随主模型的 KV 量化")

        // 显式写入辅助键后与主模型解耦，两个方向互不影响。
        try """
        {"llm_kv_reuse":false,"kv_cache_int8":true,
         "aux_llm_kv_reuse":true,"aux_kv_cache_int8":false}
        """.write(to: path, atomically: true, encoding: .utf8)
        config = AgentConfig.load(path.path)
        XCTAssertFalse(config.llmKVReuse)
        XCTAssertTrue(config.kvCacheInt8)
        XCTAssertTrue(config.auxKvReuse)
        XCTAssertFalse(config.auxKvCacheInt8)
        XCTAssertEqual(config.export()["aux_llm_kv_reuse"] as? Bool, true)
        XCTAssertEqual(config.export()["aux_kv_cache_int8"] as? Bool, false)

        // 默认值：复用关闭、量化开启（KV 内存约减半）。
        XCTAssertFalse(AgentConfig().auxKvReuse)
        XCTAssertTrue(AgentConfig().auxKvCacheInt8)
    }

    /// MLX embedding 默认后端：别名能把 Q 名解析到本地目录。
    func testConfigMLXEmbeddingDefaults() throws {
        XCTAssertEqual(AgentConfig().localEmbeddingBackend, "mlx",
                       "默认应使用 MLX embedding")
    }

    func testConfigLoadsLegacyModelKeys() throws {
        let path = directory.appendingPathComponent("legacy-config.json")
        let json = """
        {
          "llm_mode": "local",
          "llm_backend": "mlx_lm",
          "llm_model_path": "models/qwen3.5-4b-agentic-coder-v4-4bit-mlx",
          "aux_llm_backend": "mlx_lm",
          "aux_llm_model_path": "models/Qwen3.5-2B-MLX-4bit",
          "rag_embedding_backend": "mlx",
          "rag_embedding_model_path": "models/nomic-embed-text-v1.5-mlx",
          "rag_rerank_backend": "mlx",
          "rag_rerank_model_path": "models/bge-reranker-v2-m3-mlx-mxfp8"
        }
        """
        try json.write(to: path, atomically: true, encoding: .utf8)
        let config = AgentConfig.load(path.path)
        XCTAssertEqual(config.llmProvider, "local")
        XCTAssertEqual(config.localLLMBackend, "mlx-lm")
        XCTAssertEqual(config.localLLMModel,
                       "models/qwen3.5-4b-agentic-coder-v4-4bit-mlx")
        XCTAssertEqual(config.auxLocalLLMBackend, "mlx-lm")
        XCTAssertEqual(config.auxLocalLLMModel, "models/Qwen3.5-2B-MLX-4bit")
        XCTAssertEqual(config.localEmbeddingBackend, "mlx")
        // 旧 fp16 MLX 路径统一迁移到新的默认 jina-v5 int4 模型。
        XCTAssertEqual(config.localEmbeddingModelPath,
                       "models/jina-embeddings-v5-text-nano-mlx-int4")
        XCTAssertEqual(config.localRerankerBackend, "mlx")
        XCTAssertEqual(config.localRerankerModelPath,
                       "models/bge-reranker-v2-m3-mlx-mxfp8")
    }

    func testConfigMapsOpenAIModeAndPrefersMLXRerankerPath() throws {
        let path = directory.appendingPathComponent("legacy-config-openai.json")
        let json = """
        {
          "llm_mode": "openai",
          "rag_rerank_backend": "mlx",
          "rag_rerank_model_path": "models/bge-reranker-v2-m3-ONNX/model_quantized.onnx",
          "rag_rerank_mlx_model_path": "models/bge-reranker-v2-m3-mlx-mxfp8"
        }
        """
        try json.write(to: path, atomically: true, encoding: .utf8)
        let config = AgentConfig.load(path.path)
        XCTAssertEqual(config.llmProvider, "api")
        XCTAssertEqual(config.localRerankerBackend, "mlx")
        XCTAssertEqual(config.localRerankerModelPath,
                       "models/bge-reranker-v2-m3-mlx-mxfp8")
    }

    func testModelRegistryResolvesLegacyModelsPrefix() throws {
        let workspace = directory.appendingPathComponent("workspace")
        let modelDir = workspace.appendingPathComponent("models/demo-model")
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3"}"#
            .write(to: modelDir.appendingPathComponent("config.json"),
                   atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = workspace.path
        // `models/...` 现在固定锚定应用所在目录的 models，测试用环境变量覆盖该根目录。
        setenv("BASE_AGENT_MODELS_DIR", workspace.appendingPathComponent("models").path, 1)
        defer { unsetenv("BASE_AGENT_MODELS_DIR") }
        let resolved = try LocalModelRegistry.resolveDirectory(
            kind: .llm,
            value: "models/demo-model",
            config: config
        )
        XCTAssertEqual(resolved.standardizedFileURL.path,
                       modelDir.standardizedFileURL.path)
    }

    func testConfigVisionEngine() throws {
        let path = directory.appendingPathComponent("config2.json")
        try #"{"rag_pdf_engine":"vision","rag_epub_ocr_engine":"ovis"}"#
            .write(to: path, atomically: true, encoding: .utf8)
        let config = AgentConfig.load(path.path)
        XCTAssertEqual(config.ragPDFEngine, .vision)
        XCTAssertEqual(config.ragEPUBOCREngine, .ovis)
    }

    func testConfigHybridEngine() throws {
        let path = directory.appendingPathComponent("config-hybrid.json")
        try #"{"rag_pdf_engine":"hybrid"}"#
            .write(to: path, atomically: true, encoding: .utf8)
        let config = AgentConfig.load(path.path)
        XCTAssertEqual(config.ragPDFEngine, .hybrid)
        XCTAssertEqual(PDFEngine(stored: "mixed"), .hybrid)
    }

    // MARK: helpers

    private func awaitSync<T>(_ body: @escaping () async throws -> T) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<T, Error>!
        Task {
            do { result = .success(try await body()) }
            catch { result = .failure(error) }
            semaphore.signal()
        }
        _ = semaphore.wait(timeout: .now() + 60)
        return try result.get()
    }

    private func makeTextPDF(at url: URL, text: String) throws {
        var mediaBox = CGRect(x: 0, y: 0, width: 612, height: 792)
        guard let consumer = CGDataConsumer(url: url as CFURL),
              let context = CGContext(consumer: consumer, mediaBox: &mediaBox, nil) else {
            throw AgentError.parse("无法创建 PDF context")
        }
        context.beginPDFPage(nil)
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 24),
            .foregroundColor: NSColor.black,
        ]
        let attributed = NSAttributedString(string: text, attributes: attributes)
        let framesetter = CTFramesetterCreateWithAttributedString(attributed)
        let path = CGPath(rect: CGRect(x: 40, y: 80, width: 532, height: 630), transform: nil)
        let frame = CTFramesetterCreateFrame(framesetter, CFRange(location: 0, length: 0), path, nil)
        CTFrameDraw(frame, context)
        context.endPDFPage()
        context.closePDF()
    }

    /// 构造「每页一张扫描图 + 只有水印文本层」的 PDF：文本层每页重复
    /// 同一句网址水印，正文只存在于图片里。每页图片文字带页码，避免
    /// 清洗器把跨页相同的 OCR 结果当作页眉页脚删除。
    private func makeWatermarkedImagePDF(at url: URL,
                                         pageCount: Int,
                                         watermark: String) throws {
        var mediaBox = CGRect(x: 0, y: 0, width: 612, height: 792)
        guard let consumer = CGDataConsumer(url: url as CFURL),
              let context = CGContext(consumer: consumer, mediaBox: &mediaBox, nil) else {
            throw AgentError.parse("无法创建 PDF context")
        }
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 10),
            .foregroundColor: NSColor.black,
        ]
        let line = CTLineCreateWithAttributedString(
            NSAttributedString(string: watermark, attributes: attributes))
        for page in 0..<pageCount {
            let image = try makeScannedPageImage(
                text: "SCANNED PAGE \(page + 1)")
            context.beginPDFPage(nil)
            context.draw(image, in: mediaBox)
            // 页首、页尾各画一次水印，模拟扫描书常见的双向水印。
            context.textPosition = CGPoint(x: 40, y: 760)
            CTLineDraw(line, context)
            context.textPosition = CGPoint(x: 40, y: 20)
            CTLineDraw(line, context)
            context.endPDFPage()
        }
        context.closePDF()
    }

    private func makeScannedPageImage(text: String) throws -> CGImage {
        let size = NSSize(width: 612, height: 792)
        let image = NSImage(size: size)
        image.lockFocus()
        NSColor.white.setFill()
        NSRect(origin: .zero, size: size).fill()
        text.draw(at: NSPoint(x: 60, y: 500),
                  withAttributes: [.font: NSFont.systemFont(ofSize: 36),
                                   .foregroundColor: NSColor.black])
        image.unlockFocus()
        var rect = CGRect(origin: .zero, size: size)
        return try XCTUnwrap(image.cgImage(forProposedRect: &rect, context: nil, hints: nil))
    }

    private func makeImagePDF(at url: URL, text: String) throws {
        let size = NSSize(width: 612, height: 792)
        let image = NSImage(size: size)
        image.lockFocus()
        NSColor.white.setFill()
        NSRect(origin: .zero, size: size).fill()
        text.draw(at: NSPoint(x: 60, y: 600),
                  withAttributes: [.font: NSFont.systemFont(ofSize: 36),
                                   .foregroundColor: NSColor.black])
        image.unlockFocus()
        guard let page = PDFPage(image: image) else {
            throw AgentError.parse("无法创建 PDFPage")
        }
        let document = PDFDocument()
        document.insert(page, at: 0)
        guard document.write(to: url) else {
            throw AgentError.parse("无法写入 PDF")
        }
    }

    private func makeLineImage(width: Int, height: Int,
                               rows: [Int], columns: [Int]) throws -> CGImage {
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: nil,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: 0,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw AgentError.parse("无法创建测试位图")
        }
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.setStrokeColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        context.setLineWidth(3)
        for y in rows {
            context.move(to: CGPoint(x: 40, y: y))
            context.addLine(to: CGPoint(x: width - 40, y: y))
        }
        for x in columns {
            context.move(to: CGPoint(x: x, y: 40))
            context.addLine(to: CGPoint(x: x, y: height - 40))
        }
        context.strokePath()
        guard let image = context.makeImage() else {
            throw AgentError.parse("无法生成测试位图")
        }
        return image
    }

    private func makeEPUB(at url: URL) throws {
        guard let archive = try Archive(url: url, accessMode: .create) else {
            throw AgentError.parse("无法创建 EPUB 归档")
        }
        let files: [(String, String)] = [
            ("mimetype", "application/epub+zip"),
            ("META-INF/container.xml", """
            <?xml version="1.0"?>
            <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
              <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
            </container>
            """),
            ("OEBPS/content.opf", """
            <?xml version="1.0"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
              <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:title>Test Book</dc:title><dc:identifier id="id">test</dc:identifier>
              </metadata>
              <manifest>
                <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
                <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
              </manifest>
              <spine toc="ncx"><itemref idref="ch1"/></spine>
            </package>
            """),
            ("OEBPS/toc.ncx", """
            <?xml version="1.0"?>
            <ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
              <navMap><navPoint id="n1" playOrder="1">
                <navLabel><text>Chapter One</text></navLabel>
                <content src="ch1.xhtml"/>
              </navPoint></navMap>
            </ncx>
            """),
            ("OEBPS/ch1.xhtml", """
            <html xmlns="http://www.w3.org/1999/xhtml"><head><title>Chapter One</title></head>
            <body><h1>Chapter One</h1><p>Hello EPUB world. This is the first chapter.</p>
            <ul><li>Point A</li><li>Point B</li></ul></body></html>
            """),
        ]
        for (path, content) in files {
            let data = Data(content.utf8)
            try archive.addEntry(with: path, type: .file,
                                 uncompressedSize: Int64(data.count),
                                 compressionMethod: path == "mimetype" ? .none : .deflate,
                                 provider: { position, size in
                data.subdata(in: Int(position)..<Int(position) + size)
            })
        }
    }

    private static let containerXML = """
    <?xml version="1.0"?>
    <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
      <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
    </container>
    """

    /// 按给定条目直接写 EPUB；用于构造路径/编码/图片引用等边缘形态。
    private func writeEPUB(at url: URL, entries: [(String, Data)]) throws {
        guard let archive = try Archive(url: url, accessMode: .create) else {
            throw AgentError.parse("无法创建 EPUB 归档")
        }
        for (path, data) in entries {
            try archive.addEntry(with: path, type: .file,
                                 uncompressedSize: Int64(data.count),
                                 compressionMethod: path == "mimetype" ? .none : .deflate,
                                 provider: { position, size in
                data.subdata(in: Int(position)..<Int(position) + size)
            })
        }
    }

    /// 扫描版 EPUB：每章只有一个整页图片，没有可读文字。
    private func makeScannedEPUB(at url: URL, imageData: Data) throws {
        guard let archive = try Archive(url: url, accessMode: .create) else {
            throw AgentError.parse("无法创建 EPUB 归档")
        }
        let files: [(String, String)] = [
            ("mimetype", "application/epub+zip"),
            ("META-INF/container.xml", """
            <?xml version="1.0"?>
            <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
              <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
            </container>
            """),
            ("OEBPS/content.opf", """
            <?xml version="1.0"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
              <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:title>Scanned Book</dc:title><dc:identifier id="id">scan</dc:identifier>
              </metadata>
              <manifest>
                <item id="p1" href="page1.xhtml" media-type="application/xhtml+xml"/>
                <item id="img1" href="images/page1.png" media-type="image/png"/>
                <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
              </manifest>
              <spine toc="ncx"><itemref idref="p1"/></spine>
            </package>
            """),
            ("OEBPS/toc.ncx", """
            <?xml version="1.0"?>
            <ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
              <navMap><navPoint id="n1" playOrder="1">
                <navLabel><text>Page One</text></navLabel>
                <content src="page1.xhtml"/>
              </navPoint></navMap>
            </ncx>
            """),
            ("OEBPS/page1.xhtml", """
            <html xmlns="http://www.w3.org/1999/xhtml"><head><title>Page One</title></head>
            <body><div><img src="images/page1.png" alt=""/></div></body></html>
            """),
        ]
        for (path, content) in files {
            let data = Data(content.utf8)
            try archive.addEntry(with: path, type: .file,
                                 uncompressedSize: Int64(data.count),
                                 compressionMethod: path == "mimetype" ? .none : .deflate,
                                 provider: { position, size in
                data.subdata(in: Int(position)..<Int(position) + size)
            })
        }
        try archive.addEntry(with: "OEBPS/images/page1.png", type: .file,
                             uncompressedSize: Int64(imageData.count),
                             compressionMethod: .deflate,
                             provider: { position, size in
            imageData.subdata(in: Int(position)..<Int(position) + size)
        })
    }

    /// 多章扫描书：N 个 spine 条目，各 1 张整页图、无文字层。
    /// 用于验证跨章 OCR 并发与页号顺序。
    private func makeMultiChapterScannedEPUB(at url: URL, chapterCount: Int,
                                             imageData: Data) throws {
        guard let archive = try Archive(url: url, accessMode: .create) else {
            throw AgentError.parse("无法创建 EPUB 归档")
        }
        var manifestItems: [String] = []
        var spineRefs: [String] = []
        var files: [(String, String)] = [
            ("mimetype", "application/epub+zip"),
            ("META-INF/container.xml", """
            <?xml version="1.0"?>
            <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
              <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
            </container>
            """),
        ]
        for i in 0..<chapterCount {
            manifestItems.append("<item id=\"c\(i)\" href=\"chapter\(i).html\" media-type=\"application/xhtml+xml\"/>")
            spineRefs.append("<itemref idref=\"c\(i)\"/>")
            files.append(("OEBPS/chapter\(i).html", """
            <html xmlns="http://www.w3.org/1999/xhtml"><body><p><img src="images/page\(i).png"/></p></body></html>
            """))
        }
        files.append(("OEBPS/content.opf", """
        <?xml version="1.0"?>
        <package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id">
          <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
            <dc:title>Multi Scanned</dc:title><dc:identifier id="id">multi</dc:identifier>
          </metadata>
          <manifest>\(manifestItems.joined(separator: "\n"))</manifest>
          <spine>\(spineRefs.joined(separator: "\n"))</spine>
        </package>
        """))
        for (path, content) in files {
            let data = Data(content.utf8)
            try archive.addEntry(with: path, type: .file,
                                 uncompressedSize: Int64(data.count),
                                 compressionMethod: path == "mimetype" ? .none : .deflate,
                                 provider: { position, size in
                data.subdata(in: Int(position)..<Int(position) + size)
            })
        }
        for i in 0..<chapterCount {
            try archive.addEntry(with: "OEBPS/images/page\(i).png", type: .file,
                                 uncompressedSize: Int64(imageData.count),
                                 compressionMethod: .deflate,
                                 provider: { position, size in
                imageData.subdata(in: Int(position)..<Int(position) + size)
            })
        }
    }

    /// 转制书形态：单个 spine 条目 = 目录文本 + N 张扫描页图。
    /// 目录文本量超过旧兜底阈值（5000），只能靠整书预采样判定。
    private func makeMonolithicEPUB(at url: URL, imageData: Data,
                                    imageCount: Int) throws {
        guard let archive = try Archive(url: url, accessMode: .create) else {
            throw AgentError.parse("无法创建 EPUB 归档")
        }
        let tocLines = (1...300).map { "第\($0)章 测试目录条目\($0) 及其配套小节说明与相关延伸阅读\($0)" }
        let tocHTML = tocLines.map { "<p>\($0)</p>" }.joined(separator: "\n")
        let imgs = (0..<imageCount).map { i in
            "<p><img src=\"images/page_\(i).png\" alt=\"\"/></p>"
        }.joined(separator: "\n")
        let files: [(String, String)] = [
            ("mimetype", "application/epub+zip"),
            ("META-INF/container.xml", """
            <?xml version="1.0"?>
            <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
              <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
            </container>
            """),
            ("OEBPS/content.opf", """
            <?xml version="1.0"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id">
              <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:title>Monolithic Book</dc:title><dc:identifier id="id">mono</dc:identifier>
              </metadata>
              <manifest>
                <item id="t" href="text00000.html" media-type="application/xhtml+xml"/>
                <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
                \(Array((0..<imageCount).map { i in
                    "<item id=\"i\(i)\" href=\"images/page_\(i).png\" media-type=\"image/png\"/>"
                }).joined(separator: "\n"))
              </manifest>
              <spine toc="ncx"><itemref idref="t"/></spine>
            </package>
            """),
            ("OEBPS/toc.ncx", """
            <?xml version="1.0"?>
            <ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
              <navMap><navPoint id="n1" playOrder="1">
                <navLabel><text>目录</text></navLabel>
                <content src="text00000.html"/>
              </navPoint></navMap>
            </ncx>
            """),
            ("OEBPS/text00000.html", """
            <html xmlns="http://www.w3.org/1999/xhtml"><head><title>目录</title></head>
            <body>\(tocHTML)\(imgs)</body></html>
            """),
        ]
        for (path, content) in files {
            let data = Data(content.utf8)
            try archive.addEntry(with: path, type: .file,
                                 uncompressedSize: Int64(data.count),
                                 compressionMethod: path == "mimetype" ? .none : .deflate,
                                 provider: { position, size in
                data.subdata(in: Int(position)..<Int(position) + size)
            })
        }
        for i in 0..<imageCount {
            try archive.addEntry(with: "OEBPS/images/page_\(i).png", type: .file,
                                 uncompressedSize: Int64(imageData.count),
                                 compressionMethod: .deflate,
                                 provider: { position, size in
                imageData.subdata(in: Int(position)..<Int(position) + size)
            })
        }
    }

    /// 混合书：前 textChapterCount 条目为纯文字章（无图 → 预采样不命中），
    /// 尾部一个扫描章 = scannedFillerLines 行文字层（默认约 6100 字）+ N 张扫描页图。
    private func makeMixedEPUB(at url: URL, textChapterCount: Int,
                               imageData: Data, imageCount: Int,
                               scannedFillerLines: Int = 180) throws {
        guard let archive = try Archive(url: url, accessMode: .create) else {
            throw AgentError.parse("无法创建 EPUB 归档")
        }
        let fillerLine = "这一章的文字层是转制软件内置识别的低质量输出，字符总量恰好超过五千。"
        let fillerHTML = (0..<scannedFillerLines).map { _ in "<p>\(fillerLine)</p>" }.joined(separator: "\n")
        let imgs = (0..<imageCount).map { i in
            "<p><img src=\"images/page_\(i).png\" alt=\"\"/></p>"
        }.joined(separator: "\n")

        var files: [(String, String)] = [
            ("mimetype", "application/epub+zip"),
            ("META-INF/container.xml", """
            <?xml version="1.0"?>
            <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
              <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
            </container>
            """),
            ("OEBPS/toc.ncx", """
            <?xml version="1.0"?>
            <ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
              <navMap><navPoint id="n1" playOrder="1">
                <navLabel><text>目录</text></navLabel>
                <content src="text0.html"/>
              </navPoint></navMap>
            </ncx>
            """),
        ]

        var manifestItems: [String] = []
        var spineRefs: [String] = []
        for i in 0..<textChapterCount {
            manifestItems.append(
                "<item id=\"t\(i)\" href=\"text\(i).html\" media-type=\"application/xhtml+xml\"/>")
            spineRefs.append("<itemref idref=\"t\(i)\"/>")
            files.append(("OEBPS/text\(i).html", """
            <html xmlns="http://www.w3.org/1999/xhtml"><head><title>第\(i)章</title></head>
            <body><p>第\(i)章 普通文字章节的正文内容，不包含任何图片。</p></body></html>
            """))
        }
        manifestItems.append(
            "<item id=\"scanned\" href=\"scanned.html\" media-type=\"application/xhtml+xml\"/>")
        spineRefs.append("<itemref idref=\"scanned\"/>")
        manifestItems.append(
            "<item id=\"ncx\" href=\"toc.ncx\" media-type=\"application/x-dtbncx+xml\"/>")
        for i in 0..<imageCount {
            manifestItems.append(
                "<item id=\"i\(i)\" href=\"images/page_\(i).png\" media-type=\"image/png\"/>")
        }
        files.append(("OEBPS/content.opf", """
        <?xml version="1.0"?>
        <package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id">
          <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
            <dc:title>Mixed Book</dc:title><dc:identifier id="id">mixed</dc:identifier>
          </metadata>
          <manifest>
            \(manifestItems.joined(separator: "\n"))
          </manifest>
          <spine toc="ncx">\(spineRefs.joined(separator: "\n"))</spine>
        </package>
        """))
        files.append(("OEBPS/scanned.html", """
        <html xmlns="http://www.w3.org/1999/xhtml"><head><title>扫描章</title></head>
        <body>\(fillerHTML)\(imgs)</body></html>
        """))

        for (path, content) in files {
            let data = Data(content.utf8)
            try archive.addEntry(with: path, type: .file,
                                 uncompressedSize: Int64(data.count),
                                 compressionMethod: path == "mimetype" ? .none : .deflate,
                                 provider: { position, size in
                data.subdata(in: Int(position)..<Int(position) + size)
            })
        }
        for i in 0..<imageCount {
            try archive.addEntry(with: "OEBPS/images/page_\(i).png", type: .file,
                                 uncompressedSize: Int64(imageData.count),
                                 compressionMethod: .deflate,
                                 provider: { position, size in
                imageData.subdata(in: Int(position)..<Int(position) + size)
            })
        }
    }

    private func makePNGData(width: Int, height: Int) throws -> Data {
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
            isPlanar: false, colorSpaceName: .deviceRGB,
            bytesPerRow: 0, bitsPerPixel: 0
        ) else {
            throw AgentError.parse("无法创建位图")
        }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        NSColor.white.setFill()
        NSRect(x: 0, y: 0, width: width, height: height).fill()
        NSColor.black.setFill()
        NSRect(x: 120, y: 300, width: width - 240, height: 180).fill()
        NSGraphicsContext.restoreGraphicsState()

        guard let data = rep.representation(using: .png, properties: [:]) else {
            throw AgentError.parse("无法编码 PNG")
        }
        return data
    }
}

private final class TableFakeVLM: VLMService {
    let identifier = "table-vlm"
    let isAvailable = true
    var calls = 0

    func recognize(image: CGImage, prompt: String, maxTokens: Int) async throws -> String {
        calls += 1
        return """
        | ColumnA | ColumnB |
        | --- | --- |
        | X | 99 |
        """
    }
}

private final class FakeVLM: VLMService {
    let identifier = "fake-vlm"
    let isAvailable = true
    // EPUB 同一批内会并发识别多页，计数必须加锁；否则自增丢失会让
    // 「N 页都应触发 OCR」的断言偶发少一（旧实现的真实 flake）。
    private let lock = NSLock()
    private var _calls = 0
    private var _lastImageWidth = 0
    private var _lastImageHeight = 0

    var calls: Int { lock.lock(); defer { lock.unlock() }; return _calls }
    var lastImageWidth: Int { lock.lock(); defer { lock.unlock() }; return _lastImageWidth }
    var lastImageHeight: Int { lock.lock(); defer { lock.unlock() }; return _lastImageHeight }

    func recognize(image: CGImage, prompt: String, maxTokens: Int) async throws -> String {
        lock.lock()
        _calls += 1
        _lastImageWidth = image.width
        _lastImageHeight = image.height
        lock.unlock()
        return "# Ovis page content\n\nFake VLM output."
    }
}

/// 记录跨章节 OCR 的最大并发；每页延迟 30ms 让其他章节有机会进入。
private final class ConcurrencyTrackingVLM: VLMService, @unchecked Sendable {
    let identifier = "concurrency-tracking"
    let isAvailable = true
    private let lock = NSLock()
    private var active = 0
    private var _calls = 0
    private var _maxConcurrent = 0

    var calls: Int { lock.lock(); defer { lock.unlock() }; return _calls }
    var maxConcurrent: Int { lock.lock(); defer { lock.unlock() }; return _maxConcurrent }

    func recognize(image: CGImage, prompt: String, maxTokens: Int) async throws -> String {
        lock.lock()
        _calls += 1
        let n = _calls
        active += 1
        _maxConcurrent = max(_maxConcurrent, active)
        lock.unlock()
        try? await Task.sleep(nanoseconds: 30_000_000)
        lock.lock(); active -= 1; lock.unlock()
        return "第 \(n) 页 OCR 正文\n\n这是用于并发测试的模拟识别内容，字符量足够通过清洗阈值。"
    }
}

/// 每页返回带序号的差异化内容，避免跨页重复清洗把 OCR 正文当页眉页脚去除。
/// 双实例池并发调用，计数必须加锁，否则丢计数且页号重复。
private final class NumberedFakeVLM: VLMService, @unchecked Sendable {
    let identifier = "fake-vlm-numbered"
    let isAvailable = true
    private let lock = NSLock()
    private var _calls = 0
    var calls: Int {
        lock.lock(); defer { lock.unlock() }
        return _calls
    }

    func recognize(image: CGImage, prompt: String, maxTokens: Int) async throws -> String {
        lock.lock()
        _calls += 1
        let n = _calls
        lock.unlock()
        return "## 第 \(n) 页 OCR 正文\n\n这是第 \(n) 页的模拟识别内容，包含足够的文字量以通过清洗阈值。"
    }
}
