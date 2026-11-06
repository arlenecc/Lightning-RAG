import XCTest
import ZIPFoundation
@testable import BaseAgentCore

final class OfficeAndHTMLTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-office-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testDocxParagraphsHeadingsAndTables() throws {
        let url = directory.appendingPathComponent("sample.docx")
        let documentXML = """
        <?xml version="1.0"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body>
            <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>项目概述</w:t></w:r></w:p>
            <w:p><w:r><w:t>这是一个 DOCX 段落。</w:t></w:r><w:r><w:t>第二段文字。</w:t></w:r></w:p>
            <w:tbl>
              <w:tr><w:tc><w:p><w:r><w:t>列A</w:t></w:r></w:p></w:tc>
                    <w:tc><w:p><w:r><w:t>列B</w:t></w:r></w:p></w:tc></w:tr>
              <w:tr><w:tc><w:p><w:r><w:t>1</w:t></w:r></w:p></w:tc>
                    <w:tc><w:p><w:r><w:t>2</w:t></w:r></w:p></w:tc></w:tr>
            </w:tbl>
          </w:body>
        </w:document>
        """
        try makeArchive(at: url, files: [
            ("word/document.xml", documentXML),
            ("[Content_Types].xml", "<?xml version=\"1.0\"?><Types/>"),
        ])
        let parsed = try awaitSync {
            try await OfficeParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("# 项目概述"))
        XCTAssertTrue(parsed.text.contains("这是一个 DOCX 段落"))
        XCTAssertTrue(parsed.text.contains("| 列A | 列B |"))
        XCTAssertEqual(parsed.metadata["engine"], "docx")
    }

    func testHTMLToMarkdown() throws {
        let url = directory.appendingPathComponent("page.html")
        let html = """
        <html><head><title>Demo</title><script>alert(1)</script></head>
        <body><h1>标题一</h1><p>正文 <b>加粗</b>。</p>
        <ul><li>条目一</li><li>条目二</li></ul>
        <table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>
        </body></html>
        """
        try html.write(to: url, atomically: true, encoding: .utf8)
        let parsed = try awaitSync {
            try await HTMLParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("# 标题一"))
        XCTAssertTrue(parsed.text.contains("加粗"))
        XCTAssertTrue(parsed.text.contains("条目一"))
        XCTAssertTrue(parsed.text.contains("| A | B |"))
        XCTAssertFalse(parsed.text.contains("alert(1)"))
    }

    func testBoldOnlyParagraphBecomesHeading() throws {
        let url = directory.appendingPathComponent("calibre.html")
        let html = """
        <html><head><title>Demo</title></head><body>
        <p class="c1"><b class="calibre1">这是金钱的使用价值</b></p>
        <p>正文第一段，讲述金钱的使用价值与收藏价值。</p>
        <p><b>强调</b>开头的正文段落，后面还有更多内容需要继续阅读。</p>
        <div><b>独立加粗的短句</b></div>
        </body></html>
        """
        try html.write(to: url, atomically: true, encoding: .utf8)
        let parsed = try awaitSync {
            try await HTMLParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("### 这是金钱的使用价值"),
                      "纯 <b> 段落应转为标题")
        XCTAssertTrue(parsed.outline.contains("这是金钱的使用价值"),
                      "纯 <b> 段落应进入大纲")
        XCTAssertFalse(parsed.outline.contains("强调"),
                       "段首强调不是标题")
        XCTAssertTrue(parsed.text.contains("**强调**开头的正文段落"),
                      "段首强调应保留为正文")
        XCTAssertTrue(parsed.outline.contains("独立加粗的短句"),
                      "纯 <b> 的 div 也应进入大纲")
    }

    // MARK: HTML：框架接力 / 编码 / 空正文诊断

    /// frameset 页面没有 body：`<title>` 不能当成正文，框架指向网络地址时
    /// 应明确报出「正文在框架里」而不是产出一行标题就算成功。
    func testHTMLFramesetWithoutLocalFrameReportsReason() throws {
        let url = directory.appendingPathComponent("remote-frames.html")
        try """
        <html><head><title>只应出现在标题里</title>
        <style>body{margin:0}</style></head>
        <frameset cols="20%,80%">
          <frame src="https://example.com/toc.html">
          <frame src="https://example.com/body.html">
        </frameset>
        </html>
        """.write(to: url, atomically: true, encoding: .utf8)

        XCTAssertThrowsError(try awaitSync {
            try await HTMLParser().parse(url: url, options: ParserOptions(), progress: nil)
        }) { error in
            let message = (error as? AgentError)?.errorDescription ?? "\(error)"
            // `<title>` 不算正文：必须报出正文在框架里，而不是把标题当正文。
            XCTAssertTrue(message.contains("框架"), message)
            XCTAssertFalse(message.contains("只应出现在标题里"), message)
        }
    }

    func testHTMLSavedPagePullsContentFromLocalFrame() throws {
        let framesDirectory = directory.appendingPathComponent("page_files")
        try FileManager.default.createDirectory(at: framesDirectory,
                                                withIntermediateDirectories: true)
        let frame = """
        <html><body><h1>框架内的真实标题</h1>
        <p>这是被保存网页放在 iframe 里的正文，浏览器另存为会把正文单独存成框架文件。</p>
        </body></html>
        """
        try frame.write(to: framesDirectory.appendingPathComponent("10.html"),
                        atomically: true, encoding: .utf8)

        let url = directory.appendingPathComponent("page.html")
        let shell = """
        <html><head><title>外层标题</title></head>
        <body><div id="wrapper"></div>
        <iframe src="page_files/10.html" width="100%"></iframe>
        </body></html>
        """
        try shell.write(to: url, atomically: true, encoding: .utf8)

        let parsed = try awaitSync {
            try await HTMLParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("框架内的真实标题"), parsed.text)
        XCTAssertTrue(parsed.text.contains("浏览器另存为"), parsed.text)
        XCTAssertEqual(parsed.metadata["embedded_source"], "10.html")
    }

    /// 外部 HTML 不得用 file:// 或 ../ 把知识库目录之外的本地文件内容带进索引。
    func testHTMLFrameRejectsOutOfRootFileURLs() throws {
        let outside = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-outside-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: outside) }
        try "绝密：不应进入知识库的本地文件内容"
            .write(to: outside.appendingPathComponent("secret.html"),
                   atomically: true, encoding: .utf8)

        let cases = [
            outside.appendingPathComponent("secret.html").absoluteString,
            "../\(outside.lastPathComponent)/secret.html",
            "\(outside.path)/secret.html",
        ]
        for (index, reference) in cases.enumerated() {
            let url = directory.appendingPathComponent("escape-\(index).html")
            try """
            <html><head><title>外壳</title></head><body>
            <iframe src="\(reference)"></iframe>
            </body></html>
            """.write(to: url, atomically: true, encoding: .utf8)
            XCTAssertThrowsError(try awaitSync {
                try await HTMLParser().parse(url: url, options: ParserOptions(), progress: nil)
            }, "引用 \(reference) 应被拒绝")
            let parsed = try? awaitSync {
                try await HTMLParser().parse(url: url, options: ParserOptions(), progress: nil)
            }
            XCTAssertFalse(parsed?.text.contains("绝密") == true)
        }
    }

    /// 合法的同目录 / 子目录框架接力仍需正常工作。
    func testHTMLFrameAllowsInRootFileURL() throws {
        let frames = directory.appendingPathComponent("page_files", isDirectory: true)
        try FileManager.default.createDirectory(at: frames, withIntermediateDirectories: true)
        let target = frames.appendingPathComponent("body.html")
        try """
        <html><body><p>来自入口目录内 file:// 框架的正文内容，足够长以被选中。</p></body></html>
        """.write(to: target, atomically: true, encoding: .utf8)
        let url = directory.appendingPathComponent("page.html")
        try """
        <html><head><title>外壳</title></head><body>
        <iframe src="\(target.absoluteString)"></iframe>
        </body></html>
        """.write(to: url, atomically: true, encoding: .utf8)

        let parsed = try awaitSync {
            try await HTMLParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("入口目录内"), parsed.text)
        XCTAssertEqual(parsed.metadata["embedded_source"], "body.html")
    }

    /// zip bomb：声明大小或实际解压量超限时必须返回 nil，且不能把整条数据读进内存。
    func testArchiveReaderRejectsOversizedEntry() throws {
        let archiveURL = directory.appendingPathComponent("bomb.zip")
        let archive = try Archive(url: archiveURL, accessMode: .create)
        // 高压缩比内容：1MB 的 'A' 压缩后只有几 KB，但解压会分配 1MB。
        let payload = Data(repeating: 0x41, count: 1 << 20)
        try archive.addEntry(with: "big.txt", type: .file,
                             uncompressedSize: Int64(payload.count),
                             compressionMethod: .deflate,
                             provider: { position, size in
                                 payload.subdata(in: Int(position)..<Int(position) + size)
                             })
        let read = try Archive(url: archiveURL, accessMode: .read)
        XCTAssertNotNil(RAGArchiveReader.data(read, path: "big.txt"))
        XCTAssertNil(RAGArchiveReader.data(read, path: "big.txt", maxBytes: 1024))
    }

    /// 默认上限放宽到 128 MiB：介于旧上限（64 MiB）与新上限之间的条目
    /// 必须能通过预检，避免极端扫描版 EPUB 的整章正文被静默跳过。
    func testArchiveReaderDefaultLimitIs128MiB() throws {
        XCTAssertEqual(RAGArchiveReader.maxEntryBytes, 128 * 1024 * 1024)
        let archiveURL = directory.appendingPathComponent("large-entry.zip")
        let archive = try Archive(url: archiveURL, accessMode: .create)
        let payload = Data(repeating: 0x41, count: 1 << 20)
        // 声明 100 MiB（> 旧 64 MiB、< 新 128 MiB），实际只写 1 MiB：
        // 只验证目录预检阈值，不真的分配上百 MB 内存。
        try archive.addEntry(with: "big.txt", type: .file,
                             uncompressedSize: Int64(100 * 1024 * 1024),
                             compressionMethod: .deflate,
                             provider: { position, size in
                                 let start = Int(position)
                                 guard start < payload.count else { return Data() }
                                 return payload.subdata(
                                    in: start..<min(start + size, payload.count))
                             })
        let read = try Archive(url: archiveURL, accessMode: .read)
        XCTAssertEqual(read["big.txt"]?.uncompressedSize, UInt64(100 * 1024 * 1024))
        XCTAssertNotNil(RAGArchiveReader.data(read, path: "big.txt",
                                              maxBytes: RAGArchiveReader.maxEntryBytes),
                        "100 MiB 条目在新默认上限内，不应被预检拒绝")
        XCTAssertNil(RAGArchiveReader.data(read, path: "big.txt",
                                          maxBytes: 64 * 1024 * 1024),
                     "旧上限 64 MiB 会拒绝它（对照）")
    }

    func testHTMLFramesetPicksLocalFrameWithMostContent() throws {
        try """
        <html><body><p>目录</p></body></html>
        """.write(to: directory.appendingPathComponent("toc.html"),
                  atomically: true, encoding: .utf8)
        let body = "<html><body><p>" + String(repeating: "正文内容。", count: 40) + "</p></body></html>"
        try body.write(to: directory.appendingPathComponent("body.html"),
                       atomically: true, encoding: .utf8)

        let url = directory.appendingPathComponent("frameset.html")
        try """
        <html><head><title>框架页</title></head>
        <frameset cols="20%,80%">
          <frame src="toc.html">
          <frame src="body.html" name="main">
        </frameset>
        </html>
        """.write(to: url, atomically: true, encoding: .utf8)

        let parsed = try awaitSync {
            try await HTMLParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("正文内容。正文内容。"), parsed.text)
        XCTAssertEqual(parsed.metadata["embedded_source"], "body.html")
    }

    func testHTMLLegacyCharsetIsDecoded() throws {
        let html = """
        <html><head><meta http-equiv="Content-Type" content="text/html; charset=big5"></head>
        <body><p>繁體中文測試內容，用來驗證 big5 解碼。</p></body></html>
        """
        let big5 = String.Encoding(
            rawValue: CFStringConvertEncodingToNSStringEncoding(
                CFStringEncoding(CFStringEncodings.big5.rawValue)))
        let data = try XCTUnwrap(html.data(using: big5))
        let url = directory.appendingPathComponent("legacy.html")
        try data.write(to: url)

        let parsed = try awaitSync {
            try await HTMLParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("繁體中文測試內容"), parsed.text)
    }

    func testHTMLJavaScriptOnlyPageReportsReason() throws {
        let url = directory.appendingPathComponent("app.html")
        try """
        <html><head><title>前端应用</title>
        <script src="app.js"></script></head>
        <body><div id="root"></div></body></html>
        """.write(to: url, atomically: true, encoding: .utf8)

        XCTAssertThrowsError(try awaitSync {
            try await HTMLParser().parse(url: url, options: ParserOptions(), progress: nil)
        }) { error in
            let message = (error as? AgentError)?.errorDescription ?? "\(error)"
            XCTAssertTrue(message.contains("JavaScript"), message)
        }
    }

    func testHTMLHeaderWithHeadingIsKept() throws {
        let url = directory.appendingPathComponent("header.html")
        try """
        <html><body>
        <header><h1>文章主标题</h1><p>作者：某人</p></header>
        <header class="site"><span>站点 Logo</span></header>
        <p>正文第一段，说明文章的主题与结论，内容长度足够被保留下来。</p>
        </body></html>
        """.write(to: url, atomically: true, encoding: .utf8)

        let parsed = try awaitSync {
            try await HTMLParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("# 文章主标题"), parsed.text)
        XCTAssertTrue(parsed.text.contains("作者：某人"), parsed.text)
        XCTAssertFalse(parsed.text.contains("站点 Logo"), parsed.text)
    }

    func testHTMLStructuredProfileKeepsURLAndIPLines() throws {
        let url = directory.appendingPathComponent("address.html")
        try """
        <html><body>
        <p>http://172.16.5.50/nlp-engine/shark</p>
        <p>112.226.240.93部署目录：</p>
        <p>部署地址：101.132.187.81 /root/cs</p>
        </body></html>
        """.write(to: url, atomically: true, encoding: .utf8)

        let parsed = try awaitSync {
            try await HTMLParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("http://172.16.5.50/nlp-engine/shark"), parsed.text)
        XCTAssertTrue(parsed.text.contains("112.226.240.93部署目录："), parsed.text)
        XCTAssertTrue(parsed.text.contains("101.132.187.81 /root/cs"), parsed.text)
    }

    func testStructuredProfileKeepsURLsAndDataRows() throws {
        let input = """
        http://172.16.5.50/nlp-engine/shark

        112.226.240.93部署目录：

        部署地址：101.132.187.81 /root/cs

        23 大土地所有者 385 000' 19 250 000
        """
        let ocr = DocumentCleaner.clean(input)
        XCTAssertFalse(ocr.contains("172.16.5.50"), ocr)
        XCTAssertFalse(ocr.contains("112.226.240.93"), ocr)

        let structured = DocumentCleaner.clean(input, profile: .structured)
        XCTAssertTrue(structured.contains("http://172.16.5.50/nlp-engine/shark"), structured)
        XCTAssertTrue(structured.contains("112.226.240.93部署目录："), structured)
        XCTAssertTrue(structured.contains("部署地址：101.132.187.81 /root/cs"), structured)
    }

    func testStructuredProfileKeepsShortParagraphBoundaries() throws {
        let text = """
        交接文档

        运维负责人：张三
        """
        let structured = DocumentCleaner.clean(text, markdown: false, profile: .structured)
        XCTAssertTrue(structured.contains("交接文档\n\n运维负责人：张三"), structured)
    }

    /// 汉字之间的空格在结构化文档里是作者排版（标题「第一章 概述」），
    /// 只有 OCR 档才按识别噪声合并，否则标题和正文会被粘连。
    func testStructuredProfileKeepsCJKSpacing() throws {
        let text = "第一章 旧版 Word 概述\n\n正文介绍兼容策略。"

        let structured = DocumentCleaner.clean(text, markdown: false, profile: .structured)
        XCTAssertTrue(structured.contains("第一章 旧版 Word 概述"), structured)

        let ocr = DocumentCleaner.clean(text, markdown: false)
        XCTAssertTrue(ocr.contains("第一章旧版 Word 概述"), ocr)
    }

    /// 手写/导出的 HTML 常把正文直接放在 `<body>` 或 `<div>` 下，
    /// 没有 `<p>` 包裹；只遍历元素会把这些裸文本整段丢掉。
    func testHTMLBareTextNodesAreKept() throws {
        let url = directory.appendingPathComponent("bare.html")
        try """
        <html><body>
        第一行裸文本
        <div>div 内的裸文本<span>与行内标签</span></div>
        <p>段落文本</p>
        </body></html>
        """.write(to: url, atomically: true, encoding: .utf8)

        let parsed = try awaitSync {
            try await HTMLParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("第一行裸文本"), parsed.text)
        XCTAssertTrue(parsed.text.contains("div 内的裸文本与行内标签"), parsed.text)
        XCTAssertTrue(parsed.text.contains("段落文本"), parsed.text)
    }

    /// `<li>` 里包 `<p>`/`<div>` 时不能把列表项拆成多行，否则 Markdown 列表断掉。
    func testHTMLListItemWithBlockChildrenStaysOneLine() throws {
        let url = directory.appendingPathComponent("list.html")
        try """
        <html><body><ul>
        <li><p>第一项</p><p>补充说明</p></li>
        <li>第二项</li>
        </ul></body></html>
        """.write(to: url, atomically: true, encoding: .utf8)

        let parsed = try awaitSync {
            try await HTMLParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("- 第一项"), parsed.text)
        XCTAssertTrue(parsed.text.contains("补充说明"), parsed.text)
        XCTAssertTrue(parsed.text.contains("- 第二项"), parsed.text)
        // 列表项内部的块级子元素不得把列表拆成多行。
        let listLines = parsed.text.components(separatedBy: "\n")
            .filter { $0.hasPrefix("- ") }
        XCTAssertEqual(listLines.count, 2, listLines.joined(separator: " / "))
    }

    // MARK: DOCX：列表 / 换行 / 脚注

    func testDocxListNumberingAndLineBreaks() throws {
        let url = directory.appendingPathComponent("list.docx")
        let documentXML = """
        <?xml version="1.0"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body>
            <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>
              <w:r><w:t>第一项</w:t></w:r></w:p>
            <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>
              <w:r><w:t>第二项</w:t></w:r></w:p>
            <w:p><w:pPr><w:numPr><w:ilvl w:val="1"/><w:numId w:val="2"/></w:numPr></w:pPr>
              <w:r><w:t>圆点项</w:t></w:r></w:p>
            <w:p><w:r><w:t xml:space="preserve">Hello </w:t></w:r><w:r><w:t>world</w:t></w:r>
              <w:br/><w:t>换行后的第二行</w:t></w:p>
          </w:body>
        </w:document>
        """
        let numberingXML = """
        <?xml version="1.0"?>
        <w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:abstractNum w:abstractNumId="0">
            <w:lvl w:ilvl="0"><w:numFmt w:val="decimal"/></w:lvl>
          </w:abstractNum>
          <w:abstractNum w:abstractNumId="1">
            <w:lvl w:ilvl="1"><w:numFmt w:val="bullet"/></w:lvl>
          </w:abstractNum>
          <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
          <w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num>
        </w:numbering>
        """
        try makeArchive(at: url, files: [
            ("word/document.xml", documentXML),
            ("word/numbering.xml", numberingXML),
            ("[Content_Types].xml", "<?xml version=\"1.0\"?><Types/>"),
        ])

        let parsed = try awaitSync {
            try await OfficeParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("1. 第一项"), parsed.text)
        XCTAssertTrue(parsed.text.contains("2. 第二项"), parsed.text)
        XCTAssertTrue(parsed.text.contains("- 圆点项"), parsed.text)
        XCTAssertTrue(parsed.text.contains("Hello world"), parsed.text)
        XCTAssertTrue(parsed.text.contains("Hello world\n换行后的第二行"), parsed.text)
    }

    func testDocxFootnotesAreExtracted() throws {
        let url = directory.appendingPathComponent("notes.docx")
        let documentXML = """
        <?xml version="1.0"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body>
            <w:p><w:r><w:t>正文引用</w:t></w:r>
              <w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr>
              <w:footnoteReference w:id="2"/></w:r></w:p>
          </w:body>
        </w:document>
        """
        let footnotesXML = """
        <?xml version="1.0"?>
        <w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:footnote w:type="separator" w:id="0"><w:p><w:r><w:t>分隔线</w:t></w:r></w:p></w:footnote>
          <w:footnote w:id="2"><w:p><w:r><w:t>这是脚注内容。</w:t></w:r></w:p></w:footnote>
        </w:footnotes>
        """
        try makeArchive(at: url, files: [
            ("word/document.xml", documentXML),
            ("word/footnotes.xml", footnotesXML),
            ("[Content_Types].xml", "<?xml version=\"1.0\"?><Types/>"),
        ])

        let parsed = try awaitSync {
            try await OfficeParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("正文引用[^1]"), parsed.text)
        XCTAssertTrue(parsed.text.contains("## 脚注"), parsed.text)
        XCTAssertTrue(parsed.text.contains("1. 这是脚注内容。"), parsed.text)
        XCTAssertFalse(parsed.text.contains("分隔线"), parsed.text)
        XCTAssertEqual(parsed.metadata["footnotes"], "1")
    }

    /// WPS/Word 常把样式 ID 与显示名解耦：`pStyle=1` 的名称在 styles.xml 里
    /// 才写作「heading 1」，只看 pStyle 的值会漏掉全部标题。
    func testDocxHeadingResolvedFromStylesXml() throws {
        let url = directory.appendingPathComponent("styled.docx")
        let documentXML = """
        <?xml version="1.0"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body>
            <w:p><w:pPr><w:pStyle w:val="1"/></w:pPr><w:r><w:t>样式表标题</w:t></w:r></w:p>
            <w:p><w:pPr><w:pStyle w:val="2"/></w:pPr><w:r><w:t>继承来的二级标题</w:t></w:r></w:p>
            <w:p><w:pPr><w:pStyle w:val="afc"/></w:pPr>
              <w:r><w:t>这是正文段落，样式表里没有大纲级别。</w:t></w:r></w:p>
          </w:body>
        </w:document>
        """
        let stylesXML = """
        <?xml version="1.0"?>
        <w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:style w:type="paragraph" w:styleId="1">
            <w:name w:val="heading 1"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr>
          </w:style>
          <w:style w:type="paragraph" w:styleId="2"><w:name w:val="标题 2"/></w:style>
          <w:style w:type="paragraph" w:styleId="afc"><w:name w:val="正文"/></w:style>
        </w:styles>
        """
        try makeArchive(at: url, files: [
            ("word/document.xml", documentXML),
            ("word/styles.xml", stylesXML),
            ("[Content_Types].xml", "<?xml version=\"1.0\"?><Types/>"),
        ])

        let parsed = try awaitSync {
            try await OfficeParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("# 样式表标题"), parsed.text)
        XCTAssertTrue(parsed.text.contains("## 继承来的二级标题"), parsed.text)
        XCTAssertTrue(parsed.text.contains("这是正文段落，样式表里没有大纲级别。"), parsed.text)
        XCTAssertEqual(parsed.outline.components(separatedBy: "\n").count, 2, parsed.outline)
    }

    // MARK: XLSX：日期 / 百分比 / 注音

    func testXlsxDateAndPercentFormats() throws {
        let url = directory.appendingPathComponent("formats.xlsx")
        let workbookXML = """
        <?xml version="1.0"?>
        <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
          <sheets><sheet name="数据" sheetId="1" r:id="rId1"/></sheets>
        </workbook>
        """
        let relsXML = """
        <?xml version="1.0"?>
        <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
          <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
        </Relationships>
        """
        let stylesXML = """
        <?xml version="1.0"?>
        <styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
          <numFmts count="1">
            <numFmt numFmtId="164" formatCode="yyyy&quot;年&quot;m&quot;月&quot;"/>
          </numFmts>
          <cellXfs count="4">
            <xf numFmtId="0"/><xf numFmtId="14"/><xf numFmtId="164"/><xf numFmtId="10"/>
          </cellXfs>
        </styleSheet>
        """
        let sheetXML = """
        <?xml version="1.0"?>
        <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
          <sheetData>
            <row r="1">
              <c r="A1" s="1"><v>45000</v></c>
              <c r="B1" s="2"><v>45000</v></c>
              <c r="C1" s="3"><v>0.5</v></c>
              <c r="D1"><v>3.14</v></c>
            </row>
          </sheetData>
        </worksheet>
        """
        try makeArchive(at: url, files: [
            ("xl/workbook.xml", workbookXML),
            ("xl/_rels/workbook.xml.rels", relsXML),
            ("xl/styles.xml", stylesXML),
            ("xl/worksheets/sheet1.xml", sheetXML),
            ("[Content_Types].xml", "<?xml version=\"1.0\"?><Types/>"),
        ])

        let parsed = try awaitSync {
            try await OfficeParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("2023-03-15"), parsed.text)
        XCTAssertTrue(parsed.text.contains("50%"), parsed.text)
        XCTAssertTrue(parsed.text.contains("3.14"), parsed.text)
        XCTAssertFalse(parsed.text.contains("45000"), parsed.text)
    }

    func testXlsxPhoneticRunsAreSkipped() throws {
        let url = directory.appendingPathComponent("phonetic.xlsx")
        let workbookXML = """
        <?xml version="1.0"?>
        <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
          <sheets><sheet name="表" sheetId="1" r:id="rId1"/></sheets>
        </workbook>
        """
        let relsXML = """
        <?xml version="1.0"?>
        <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
          <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
        </Relationships>
        """
        let sharedXML = """
        <?xml version="1.0"?>
        <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="1" uniqueCount="1">
          <si><t>日本語</t><rPh sb="0" eb="3"><t>にほんご</t></rPh><phoneticPr fontId="1"/></si>
        </sst>
        """
        let sheetXML = """
        <?xml version="1.0"?>
        <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
          <sheetData><row r="1"><c r="A1" t="s"><v>0</v></c></row></sheetData>
        </worksheet>
        """
        try makeArchive(at: url, files: [
            ("xl/workbook.xml", workbookXML),
            ("xl/_rels/workbook.xml.rels", relsXML),
            ("xl/sharedStrings.xml", sharedXML),
            ("xl/worksheets/sheet1.xml", sheetXML),
            ("[Content_Types].xml", "<?xml version=\"1.0\"?><Types/>"),
        ])

        let parsed = try awaitSync {
            try await OfficeParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("日本語"), parsed.text)
        XCTAssertFalse(parsed.text.contains("にほんご"), parsed.text)
    }

    /// 单元格内的换行/竖线会截断 Markdown 表格，必须转义成单行。
    func testXlsxMultilineCellKeepsTableRow() throws {
        let url = directory.appendingPathComponent("multiline.xlsx")
        let workbookXML = """
        <?xml version="1.0"?>
        <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
          <sheets><sheet name="表" sheetId="1" r:id="rId1"/></sheets>
        </workbook>
        """
        let relsXML = """
        <?xml version="1.0"?>
        <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
          <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
        </Relationships>
        """
        let sharedXML = """
        <?xml version="1.0"?>
        <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="2" uniqueCount="2">
          <si><t>问题</t></si>
          <si><t>第一行&#10;第二行|带竖线</t></si>
        </sst>
        """
        let sheetXML = """
        <?xml version="1.0"?>
        <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
          <sheetData><row r="1">
            <c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c>
          </row></sheetData>
        </worksheet>
        """
        try makeArchive(at: url, files: [
            ("xl/workbook.xml", workbookXML),
            ("xl/_rels/workbook.xml.rels", relsXML),
            ("xl/sharedStrings.xml", sharedXML),
            ("xl/worksheets/sheet1.xml", sheetXML),
            ("[Content_Types].xml", "<?xml version=\"1.0\"?><Types/>"),
        ])

        let parsed = try awaitSync {
            try await OfficeParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        // 单元格内的换行被压平为一行，竖线转义；表格结构不被破坏。
        XCTAssertFalse(parsed.text.contains("第一行\n第二行"), parsed.text)
        XCTAssertTrue(parsed.text.contains("第一行"), parsed.text)
        XCTAssertTrue(parsed.text.contains("第二行\\|带竖线"), parsed.text)
        let tableLines = parsed.text.components(separatedBy: "\n")
            .filter { $0.hasPrefix("|") }
        // 表头 + 分隔行；单元格内换行不得多拆出一行。
        XCTAssertEqual(tableLines.count, 2, tableLines.joined(separator: " / "))
    }

    // MARK: PPTX：演讲者备注 / 段内换行

    func testPptxSpeakerNotesAndLineBreaks() throws {
        let url = directory.appendingPathComponent("deck.pptx")
        let presentationXML = """
        <?xml version="1.0"?>
        <p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                        xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
          <p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst>
        </p:presentation>
        """
        let presentationRelsXML = """
        <?xml version="1.0"?>
        <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
          <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
        </Relationships>
        """
        let slideXML = """
        <?xml version="1.0"?>
        <p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
               xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
          <p:cSld><p:spTree>
            <p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>
              <p:txBody><a:p><a:r><a:t>产品路线图</a:t></a:r></a:p></p:txBody></p:sp>
            <p:sp><p:nvSpPr><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr>
              <p:txBody><a:p><a:r><a:t>第一行</a:t></a:r><a:br/><a:r><a:t>第二行</a:t></a:r></a:p></p:txBody></p:sp>
          </p:spTree></p:cSld>
        </p:sld>
        """
        let slideRelsXML = """
        <?xml version="1.0"?>
        <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
          <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide1.xml"/>
        </Relationships>
        """
        let notesXML = """
        <?xml version="1.0"?>
        <p:notes xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
          <p:cSld><p:spTree>
            <p:sp><p:nvSpPr><p:nvPr><p:ph type="sldNum"/></p:nvPr></p:nvSpPr>
              <p:txBody><a:p><a:r><a:t>99 页码</a:t></a:r></a:p></p:txBody></p:sp>
            <p:sp><p:nvSpPr><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr>
              <p:txBody><a:p><a:r><a:t>备注：本页讲清路线图的三个里程碑。</a:t></a:r></a:p></p:txBody></p:sp>
          </p:spTree></p:cSld>
        </p:notes>
        """
        try makeArchive(at: url, files: [
            ("ppt/presentation.xml", presentationXML),
            ("ppt/_rels/presentation.xml.rels", presentationRelsXML),
            ("ppt/slides/slide1.xml", slideXML),
            ("ppt/slides/_rels/slide1.xml.rels", slideRelsXML),
            ("ppt/notesSlides/notesSlide1.xml", notesXML),
            ("[Content_Types].xml", "<?xml version=\"1.0\"?><Types/>"),
        ])

        let parsed = try awaitSync {
            try await OfficeParser().parse(url: url, options: ParserOptions(), progress: nil)
        }
        XCTAssertTrue(parsed.text.contains("# Slide 1：产品路线图"), parsed.text)
        XCTAssertTrue(parsed.text.contains("第一行\n第二行"), parsed.text)
        XCTAssertTrue(parsed.text.contains("**演讲者备注**"), parsed.text)
        XCTAssertTrue(parsed.text.contains("三个里程碑"), parsed.text)
        XCTAssertFalse(parsed.text.contains("99 页码"), parsed.text)
        XCTAssertEqual(parsed.metadata["notes_slides"], "1")
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

    private func makeArchive(at url: URL, files: [(String, String)]) throws {
        guard let archive = try Archive(url: url, accessMode: .create) else {
            throw AgentError.parse("无法创建归档")
        }
        for (path, content) in files {
            let data = Data(content.utf8)
            try archive.addEntry(with: path, type: .file,
                                 uncompressedSize: Int64(data.count),
                                 compressionMethod: .deflate,
                                 provider: { position, size in
                data.subdata(in: Int(position)..<Int(position) + size)
            })
        }
    }
}
