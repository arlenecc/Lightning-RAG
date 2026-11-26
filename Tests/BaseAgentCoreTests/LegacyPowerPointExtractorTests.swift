import XCTest
@testable import BaseAgentCore

final class LegacyPowerPointExtractorTests: XCTestCase {

    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-legacy-ppt-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testExtractsSlidesInSlideListOrderWithNotesAndBytes() throws {
        let builder = PowerPointTestBuilder(
            slides: [
                .init(slideId: 1, blocks: [
                    .init(headerType: 0, text: "季度报告"),
                    .init(headerType: 1, text: "第一段\r第二段"),
                    .init(headerType: 1, text: "Café résumé", useBytesAtom: true),
                ]),
                .init(slideId: 2, blocks: [
                    .init(headerType: 0, text: "第二页标题"),
                    .init(headerType: 1, text: "中文正文"),
                ]),
                .init(slideId: 3, blocks: []),
            ],
            notes: [
                .init(slideIdRef: 1, blocks: [
                    .init(headerType: 2, text: "讲稿第一页\r备注第二段"),
                ]),
            ]
        )

        let output = try LegacyPowerPointExtractor().extract(url: try write(builder.build()))

        XCTAssertTrue(output.markdown.contains("## Slide 1：季度报告"), output.markdown)
        XCTAssertTrue(output.markdown.contains("## Slide 2：第二页标题"), output.markdown)
        XCTAssertTrue(output.markdown.contains("## Slide 3"), output.markdown)
        XCTAssertTrue(output.markdown.contains("第一段\n\n第二段"), output.markdown)
        XCTAssertTrue(output.markdown.contains("Café résumé"), output.markdown)
        XCTAssertTrue(output.markdown.contains("中文正文"), output.markdown)
        XCTAssertTrue(output.markdown.contains(
            "**演讲者备注**\n\n讲稿第一页\n\n备注第二段"), output.markdown)
        XCTAssertEqual(output.markdown.components(separatedBy: "季度报告").count - 1, 1)

        let first = try XCTUnwrap(output.markdown.range(of: "## Slide 1"))
        let second = try XCTUnwrap(output.markdown.range(of: "## Slide 2"))
        let third = try XCTUnwrap(output.markdown.range(of: "## Slide 3"))
        XCTAssertLessThan(first.lowerBound, second.lowerBound)
        XCTAssertLessThan(second.lowerBound, third.lowerBound)

        XCTAssertEqual(output.metadata["engine"], "ppt")
        XCTAssertEqual(output.metadata["legacy_format"], "ppt-97")
        XCTAssertEqual(output.metadata["slides"], "3")
        XCTAssertEqual(output.metadata["notes_slides"], "1")
        XCTAssertEqual(output.metadata["text_atoms"], "6")
        XCTAssertEqual(output.outline.map(\.title), [
            "Slide 1：季度报告",
            "Slide 2：第二页标题",
            "Slide 3",
        ])
    }

    func testBytesAtomUsesWindows1252() throws {
        let builder = PowerPointTestBuilder(slides: [
            .init(slideId: 1, blocks: [
                .init(headerType: 0, text: "Résumé 2024", useBytesAtom: true),
                .init(headerType: 1, text: "naïve café", useBytesAtom: true),
            ]),
        ])

        let output = try LegacyPowerPointExtractor().extract(url: try write(builder.build()))
        XCTAssertTrue(output.markdown.contains("## Slide 1：Résumé 2024"), output.markdown)
        XCTAssertTrue(output.markdown.contains("naïve café"), output.markdown)
    }

    func testBytesAtomDecodesGBKChinese() throws {
        // GBK 的「季度报告」字节含非法 UTF-8 起始字节，UTF-8 解码会失败。
        let builder = PowerPointTestBuilder(slides: [
            .init(slideId: 1, blocks: [
                .init(headerType: 0, text: "季度报告", useBytesAtom: true,
                      byteEncoding: PowerPointTestBuilder.gb18030),
                .init(headerType: 1, text: "中文正文", useBytesAtom: true,
                      byteEncoding: PowerPointTestBuilder.gb18030),
            ]),
        ])

        let output = try LegacyPowerPointExtractor().extract(url: try write(builder.build()))
        XCTAssertTrue(output.markdown.contains("## Slide 1：季度报告"), output.markdown)
        XCTAssertTrue(output.markdown.contains("中文正文"), output.markdown)
        XCTAssertEqual(output.outline.map(\.title), ["Slide 1：季度报告"])
    }

    func testBytesAtomDecodesGBKChineseThatLooksLikeValidUTF8() throws {
        // 「中文」的 GBK 字节 D6 D0 CE C4 恰好是合法 UTF-8，会被解成
        // 希伯来/希腊字母；解码器必须识别这种乱码并改选 GB18030。
        let builder = PowerPointTestBuilder(slides: [
            .init(slideId: 1, blocks: [
                .init(headerType: 0, text: "中文", useBytesAtom: true,
                      byteEncoding: PowerPointTestBuilder.gb18030),
            ]),
        ])

        let output = try LegacyPowerPointExtractor().extract(url: try write(builder.build()))
        XCTAssertTrue(output.markdown.contains("## Slide 1：中文"), output.markdown)
    }

    func testBytesAtomDecodesUTF8Chinese() throws {
        let builder = PowerPointTestBuilder(slides: [
            .init(slideId: 1, blocks: [
                .init(headerType: 0, text: "中文标题", useBytesAtom: true,
                      byteEncoding: .utf8),
            ]),
        ])

        let output = try LegacyPowerPointExtractor().extract(url: try write(builder.build()))
        XCTAssertTrue(output.markdown.contains("## Slide 1：中文标题"), output.markdown)
    }

    func testBytesAtomDecodesUTF8WesternTextWithoutSwitchingToGBK() throws {
        // 合法 UTF-8 西文不应因为 GB18030 也能解码而被误判成中文。
        let builder = PowerPointTestBuilder(slides: [
            .init(slideId: 1, blocks: [
                .init(headerType: 0, text: "Résumé 2024", useBytesAtom: true,
                      byteEncoding: .utf8),
            ]),
        ])

        let output = try LegacyPowerPointExtractor().extract(url: try write(builder.build()))
        XCTAssertTrue(output.markdown.contains("## Slide 1：Résumé 2024"), output.markdown)
    }

    func testBytesAtomKeepsUTF8CyrillicText() throws {
        // 合法 UTF-8 的西里尔文同样能被 GB18030 解成汉字，不能只凭「能解出
        // 中文」就覆盖 UTF-8；只有多种文字混杂或 C1 控制字符才是 GBK 乱码。
        let builder = PowerPointTestBuilder(slides: [
            .init(slideId: 1, blocks: [
                .init(headerType: 0, text: "Привет 2024", useBytesAtom: true,
                      byteEncoding: .utf8),
                .init(headerType: 1, text: "Ελληνικά", useBytesAtom: true,
                      byteEncoding: .utf8),
            ]),
        ])

        let output = try LegacyPowerPointExtractor().extract(url: try write(builder.build()))
        XCTAssertTrue(output.markdown.contains("## Slide 1：Привет 2024"), output.markdown)
        XCTAssertTrue(output.markdown.contains("Ελληνικά"), output.markdown)
    }

    func testSlideContainersAreUsedWhenSlideListIsMissing() throws {
        let builder = PowerPointTestBuilder(
            slides: [
                .init(slideId: 2, blocks: [
                    .init(headerType: 0, text: "文件顺序一"),
                    .init(headerType: 1, text: "正文一"),
                ]),
                .init(slideId: 1, blocks: [
                    .init(headerType: 0, text: "文件顺序二"),
                    .init(headerType: 1, text: "正文二"),
                ]),
            ],
            includeSlideList: false
        )

        let output = try LegacyPowerPointExtractor().extract(url: try write(builder.build()))
        XCTAssertTrue(output.markdown.contains("## Slide 1：文件顺序一"), output.markdown)
        XCTAssertTrue(output.markdown.contains("## Slide 2：文件顺序二"), output.markdown)
        XCTAssertEqual(output.metadata["slides"], "2")
    }

    func testUnlistedSlidesAreRestoredInFileOrder() throws {
        let builder = PowerPointTestBuilder(slides: [
            .init(slideId: 1, blocks: [
                .init(headerType: 0, text: "第一页"),
                .init(headerType: 1, text: "正文一"),
            ]),
            .init(slideId: 2, blocks: [
                .init(headerType: 0, text: "缺失页"),
                .init(headerType: 1, text: "正文二"),
            ], inSlideList: false),
            .init(slideId: 3, blocks: [
                .init(headerType: 0, text: "第三页"),
                .init(headerType: 1, text: "正文三"),
            ]),
        ])

        let output = try LegacyPowerPointExtractor().extract(url: try write(builder.build()))
        XCTAssertTrue(output.markdown.contains("## Slide 2：缺失页"), output.markdown)
        XCTAssertTrue(output.markdown.contains("正文二"), output.markdown)
        XCTAssertEqual(output.metadata["slides"], "3")
        XCTAssertEqual(output.outline.map(\.title), [
            "Slide 1：第一页",
            "Slide 2：缺失页",
            "Slide 3：第三页",
        ])

        let first = try XCTUnwrap(output.markdown.range(of: "## Slide 1"))
        let second = try XCTUnwrap(output.markdown.range(of: "## Slide 2"))
        let third = try XCTUnwrap(output.markdown.range(of: "## Slide 3"))
        XCTAssertLessThan(first.lowerBound, second.lowerBound)
        XCTAssertLessThan(second.lowerBound, third.lowerBound)
    }

    func testSlideContainerBodyIsKeptWhenListTextIsPartial() throws {
        let builder = PowerPointTestBuilder(slides: [
            .init(slideId: 1,
                  blocks: [
                      .init(headerType: 0, text: "标题"),
                      .init(headerType: 1, text: "容器正文"),
                  ],
                  listBlocks: [
                      .init(headerType: 0, text: "标题"),
                  ]),
        ])

        let output = try LegacyPowerPointExtractor().extract(url: try write(builder.build()))
        XCTAssertTrue(output.markdown.contains("## Slide 1：标题"), output.markdown)
        XCTAssertTrue(output.markdown.contains("容器正文"), output.markdown)
        XCTAssertEqual(output.markdown.components(separatedBy: "容器正文").count - 1, 1)
    }

    func testMismatchedSlideIdsFallBackToSlideContainers() throws {
        // 真实文件里 SlideListWithText 的 ID 与 Slide 容器 SlideAtom 的
        // 字段并不保证一致；列表没有文本时必须按位置回退到容器。
        let builder = PowerPointTestBuilder(slides: [
            .init(slideId: 1,
                  blocks: [
                      .init(headerType: 0, text: "容器标题"),
                      .init(headerType: 1, text: "容器正文"),
                  ],
                  listSlideId: 900,
                  listBlocks: []),
        ])

        let output = try LegacyPowerPointExtractor().extract(url: try write(builder.build()))
        XCTAssertTrue(output.markdown.contains("## Slide 1：容器标题"), output.markdown)
        XCTAssertTrue(output.markdown.contains("容器正文"), output.markdown)
        XCTAssertEqual(output.metadata["slides"], "1")
    }

    func testMissingPowerPointStreamIsParseError() throws {
        let builder = PowerPointTestBuilder(
            slides: [
                .init(slideId: 1, blocks: [
                    .init(headerType: 0, text: "标题"),
                ]),
            ],
            streamName: "Wrong"
        )
        let url = try write(builder.build())

        XCTAssertThrowsError(try LegacyPowerPointExtractor().extract(url: url)) { error in
            guard case AgentError.parse(let message) = error else {
                XCTFail("期望解析错误，实际：\(error)")
                return
            }
            XCTAssertTrue(message.contains("PowerPoint"), message)
            XCTAssertTrue(message.contains(url.lastPathComponent), message)
        }
    }

    func testPowerPoint95IsUnsupported() throws {
        let builder = PowerPointTestBuilder(firstRecordType: 0x0000)
        let url = try write(builder.build())

        XCTAssertThrowsError(try LegacyPowerPointExtractor().extract(url: url)) { error in
            guard case AgentError.unsupported(let message) = error else {
                XCTFail("期望不支持错误，实际：\(error)")
                return
            }
            XCTAssertTrue(message.contains("PowerPoint 97"), message)
            XCTAssertTrue(message.contains("PowerPoint 95"), message)
        }
    }

    func testEncryptedPresentationIsUnsupported() throws {
        let builder = PowerPointTestBuilder(
            slides: [
                .init(slideId: 1, blocks: [
                    .init(headerType: 0, text: "加密"),
                ]),
            ],
            encrypted: true
        )
        let url = try write(builder.build())

        XCTAssertThrowsError(try LegacyPowerPointExtractor().extract(url: url)) { error in
            guard case AgentError.unsupported(let message) = error else {
                XCTFail("期望不支持错误，实际：\(error)")
                return
            }
            XCTAssertTrue(message.contains("加密") || message.contains("密码"), message)
        }
    }

    func testRecordDepthLimitIsReported() throws {
        let builder = PowerPointTestBuilder(
            slides: [
                .init(slideId: 1, blocks: [
                    .init(headerType: 0, text: "深度"),
                ]),
            ],
            extraNesting: LegacyOfficeLimits.maxRecordDepth + 2
        )
        let url = try write(builder.build())

        XCTAssertThrowsError(try LegacyPowerPointExtractor().extract(url: url)) { error in
            guard case AgentError.parse(let message) = error else {
                XCTFail("期望解析错误，实际：\(error)")
                return
            }
            XCTAssertTrue(message.contains("深度"), message)
        }
    }

    func testRecordCountLimitIsReported() throws {
        let builder = PowerPointTestBuilder(
            slides: [
                .init(slideId: 1, blocks: [
                    .init(headerType: 0, text: "数量"),
                ]),
            ],
            trailingRecords: LegacyOfficeLimits.maxRecordCount + 1,
            sectorSize: 4096
        )
        let url = try write(builder.build())

        XCTAssertThrowsError(try LegacyPowerPointExtractor().extract(url: url)) { error in
            guard case AgentError.parse(let message) = error else {
                XCTFail("期望解析错误，实际：\(error)")
                return
            }
            XCTAssertTrue(message.contains("记录数量"), message)
        }
    }

    func testEmptyPresentationThrowsParseError() throws {
        let builder = PowerPointTestBuilder()
        let url = try write(builder.build())

        XCTAssertThrowsError(try LegacyPowerPointExtractor().extract(url: url)) { error in
            guard case AgentError.parse(let message) = error else {
                XCTFail("期望解析错误，实际：\(error)")
                return
            }
            XCTAssertTrue(message.contains("未解析出正文"), message)
        }
    }

    func testInvalidFileThrowsParseError() throws {
        let url = directory.appendingPathComponent("broken.ppt")
        try Data("not an OLE2 file".utf8).write(to: url)

        XCTAssertThrowsError(try LegacyPowerPointExtractor().extract(url: url)) { error in
            guard case AgentError.parse(let message) = error else {
                XCTFail("期望解析错误，实际：\(error)")
                return
            }
            XCTAssertTrue(message.contains("broken.ppt"), message)
        }
    }

    func testUnmatchedNotesAreStillAssignedToSlides() throws {
        // 备注 ID 只匹配上一部分幻灯片时，剩余备注不得静默丢失。
        let builder = PowerPointTestBuilder(slides: [
            .init(slideId: 1, blocks: [
                .init(headerType: 0, text: "第一页"),
                .init(headerType: 1, text: "正文一"),
            ]),
            .init(slideId: 2, blocks: [
                .init(headerType: 0, text: "第二页"),
                .init(headerType: 1, text: "正文二"),
            ]),
        ], notes: [
            .init(slideIdRef: 999, blocks: [.init(headerType: 2, text: "孤立备注")]),
        ])
        let doc = try LegacyPowerPointExtractor().extract(url: try write(builder.build()))
        XCTAssertTrue(doc.text.contains("孤立备注"), "未匹配备注应兜底到幻灯片")
    }

    // MARK: - helpers

    private func write(_ file: OLE2TestFile) throws -> URL {
        let url = directory.appendingPathComponent("sample-\(UUID().uuidString).ppt")
        try file.data.write(to: url)
        return url
    }
}
