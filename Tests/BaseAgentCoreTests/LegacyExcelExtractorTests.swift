import XCTest
@testable import BaseAgentCore

final class LegacyExcelExtractorTests: XCTestCase {

    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-legacy-excel-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testWorkbookExtractsSheetsValuesAndFormulas() throws {
        let sumTokens = BIFF8TestBuilder.areaToken(firstRow: 1, lastRow: 4,
                                                   firstColumn: 0, lastColumn: 0)
            + BIFF8TestBuilder.funcVarToken(function: 4, arguments: 1)
        let builder = BIFF8TestBuilder(
            sheets: [
                .init(name: "数据", cells: [
                    .init(row: 0, column: 0, value: .labelSST(0)),
                    .init(row: 0, column: 1, value: .label("数值")),
                    .init(row: 1, column: 0, value: .label("日期")),
                    .init(row: 1, column: 1, value: .number(45000, xf: 1)),
                    .init(row: 2, column: 0, value: .label("百分比")),
                    .init(row: 2, column: 1, value: .number(0.12, xf: 2)),
                    .init(row: 3, column: 0, value: .label("RK 整数")),
                    .init(row: 3, column: 1,
                          value: .rk(BIFF8TestBuilder.rkInteger(1234), xf: 0)),
                    .init(row: 4, column: 0, value: .label("MULRK")),
                    .init(row: 4, column: 1,
                          value: .mulrk(values: [BIFF8TestBuilder.rkInteger(7),
                                                 BIFF8TestBuilder.rkInteger(8)], xf: 0)),
                    .init(row: 5, column: 0, value: .label("布尔")),
                    .init(row: 5, column: 1, value: .boolean(true, xf: 0)),
                    .init(row: 6, column: 0, value: .label("错误")),
                    .init(row: 6, column: 1, value: .error(0x07, xf: 0)),
                    .init(row: 7, column: 0, value: .label("公式")),
                    .init(row: 7, column: 1,
                          value: .formula(tokens: sumTokens,
                                          result: BIFF8TestBuilder.numberResult(10),
                                          string: nil, xf: 0)),
                ]),
                .init(name: "汇总", cells: [
                    .init(row: 0, column: 0, value: .label("备注")),
                    .init(row: 0, column: 1, value: .labelSST(2)),
                    .init(row: 1, column: 0, value: .label("缓存")),
                    .init(row: 1, column: 1,
                          value: .formula(tokens: [0xFF],
                                          result: BIFF8TestBuilder.numberResult(42),
                                          string: nil, xf: 0)),
                    .init(row: 2, column: 0, value: .label("文本")),
                    .init(row: 2, column: 1,
                          value: .formula(tokens: BIFF8TestBuilder.stringToken("hello"),
                                          result: BIFF8TestBuilder.stringResult(),
                                          string: "hello", xf: 0)),
                    .init(row: 3, column: 0, value: .label("算术")),
                    .init(row: 3, column: 1,
                          value: .formula(tokens: BIFF8TestBuilder.refToken(row: 1, column: 1)
                                          + BIFF8TestBuilder.intToken(2) + [0x05],
                                          result: BIFF8TestBuilder.numberResult(90000),
                                          string: nil, xf: 0)),
                ]),
            ],
            sharedStrings: ["项目", "日期", "中文标签"],
            formats: [(id: 164, value: "yyyy-mm-dd")],
            xfFormatIDs: [0, 164, 9]
        )

        let url = try write(builder.build())
        let output = try LegacyExcelExtractor().extract(url: url)

        XCTAssertTrue(output.markdown.contains("## Sheet: 数据"), output.markdown)
        XCTAssertTrue(output.markdown.contains("## Sheet: 汇总"), output.markdown)
        XCTAssertTrue(output.markdown.contains("2023-03-15"), output.markdown)
        XCTAssertTrue(output.markdown.contains("12%"), output.markdown)
        XCTAssertTrue(output.markdown.contains("=SUM(A2:A5)"), output.markdown)
        XCTAssertTrue(output.markdown.contains("| MULRK | 7 | 8 |"), output.markdown)
        XCTAssertTrue(output.markdown.contains("TRUE"), output.markdown)
        XCTAssertTrue(output.markdown.contains("#DIV/0!"), output.markdown)
        XCTAssertTrue(output.markdown.contains("中文标签"), output.markdown)
        XCTAssertTrue(output.markdown.contains("42"), output.markdown)
        XCTAssertTrue(output.markdown.contains("hello"), output.markdown)
        XCTAssertTrue(output.markdown.contains("=B2*2"), output.markdown)

        XCTAssertEqual(output.metadata["engine"], "xls")
        XCTAssertEqual(output.metadata["legacy_format"], "biff8")
        XCTAssertEqual(output.metadata["sheets"], "2")
        XCTAssertEqual(output.metadata["formula_unparsed"], "1")
        XCTAssertEqual(output.outline.map(\.title), ["Sheet: 数据", "Sheet: 汇总"])
    }

    func testSharedStringsWithContinuationAtStringBoundary() throws {
        let builder = BIFF8TestBuilder(
            sheets: [
                .init(name: "SST", cells: [
                    .init(row: 0, column: 0, value: .labelSST(0)),
                    .init(row: 0, column: 1, value: .labelSST(1)),
                    .init(row: 1, column: 0, value: .labelSST(2)),
                    .init(row: 1, column: 1, value: .labelSST(3)),
                ]),
            ],
            sharedStrings: ["第一段", "second", "第三段", "fourth"],
            sstContinueAfterString: 2
        )

        let output = try LegacyExcelExtractor().extract(url: try write(builder.build()))
        XCTAssertTrue(output.markdown.contains("第一段"), output.markdown)
        XCTAssertTrue(output.markdown.contains("second"), output.markdown)
        XCTAssertTrue(output.markdown.contains("第三段"), output.markdown)
        XCTAssertTrue(output.markdown.contains("fourth"), output.markdown)
    }

    func testSharedStringsSplitInsideCharacters() throws {
        let builder = BIFF8TestBuilder(
            sheets: [
                .init(name: "SST", cells: [
                    .init(row: 0, column: 0, value: .labelSST(0)),
                    .init(row: 0, column: 1, value: .labelSST(1)),
                ]),
            ],
            sharedStrings: ["旧版表格共享字符串跨记录测试", "后续中文标签"],
            sstSplitInsideString: (index: 0, characters: 4)
        )

        let output = try LegacyExcelExtractor().extract(url: try write(builder.build()))
        XCTAssertTrue(output.markdown.contains("旧版表格共享字符串跨记录测试"), output.markdown)
        XCTAssertTrue(output.markdown.contains("后续中文标签"), output.markdown)
    }

    func testBIFF5BookStreamIsUnsupported() throws {
        let builder = BIFF8TestBuilder(
            sheets: [.init(name: "Book", cells: [
                .init(row: 0, column: 0, value: .label("旧版")),
            ])],
            streamName: "Book"
        )
        let url = try write(builder.build())

        XCTAssertThrowsError(try LegacyExcelExtractor().extract(url: url)) { error in
            guard case AgentError.unsupported(let message) = error else {
                XCTFail("期望不支持错误，实际：\(error)")
                return
            }
            XCTAssertTrue(message.contains("BIFF5") || message.contains("Excel 97"), message)
        }
    }

    func testEncryptedWorkbookIsUnsupported() throws {
        let builder = BIFF8TestBuilder(
            sheets: [.init(name: "加密", cells: [
                .init(row: 0, column: 0, value: .label("内容")),
            ])],
            filePass: true
        )
        let url = try write(builder.build())

        XCTAssertThrowsError(try LegacyExcelExtractor().extract(url: url)) { error in
            guard case AgentError.unsupported(let message) = error else {
                XCTFail("期望不支持错误，实际：\(error)")
                return
            }
            XCTAssertTrue(message.contains("加密") || message.contains("密码"), message)
        }
    }

    func testRowAndColumnLimitsAreReported() throws {
        let builder = BIFF8TestBuilder(
            sheets: [.init(name: "限制", cells: [
                .init(row: 0, column: 0, value: .label("保留")),
                .init(row: LegacyOfficeLimits.maxExcelRows, column: 0,
                      value: .label("超出行")),
                .init(row: 0, column: LegacyOfficeLimits.maxExcelColumns,
                      value: .label("超出列")),
            ])]
        )

        let output = try LegacyExcelExtractor().extract(url: try write(builder.build()))
        XCTAssertEqual(output.metadata["row_limit_hit"], "true")
        XCTAssertEqual(output.metadata["column_limit_hit"], "true")
        XCTAssertTrue(output.markdown.contains("保留"), output.markdown)
        XCTAssertFalse(output.markdown.contains("超出行"), output.markdown)
        XCTAssertFalse(output.markdown.contains("超出列"), output.markdown)
    }

    /// 工作簿解析可能持续很久；任务取消后必须立即停止，而不是继续读完整个文件。
    func testExtractionStopsWhenTaskIsCancelled() async throws {
        let builder = BIFF8TestBuilder(
            sheets: [
                .init(name: "数据", cells: [
                    .init(row: 0, column: 0, value: .label("项目")),
                    .init(row: 1, column: 0, value: .label("内容")),
                ]),
            ]
        )
        let url = try write(builder.build())

        let task = Task<Void, Error> {
            withUnsafeCurrentTask { $0?.cancel() }
            _ = try LegacyExcelExtractor().extract(url: url)
        }
        do {
            try await task.value
            XCTFail("取消后不应继续解析 Excel")
        } catch AgentError.cancelled {
            // 预期：取消信号被解析器观察到。
        } catch {
            XCTFail("应抛出 AgentError.cancelled，实际：\(error)")
        }
    }

    /// 损坏的工作表仍可跳过并保留其他表，但必须上报失败，供入库层标记未完成。
    func testUnreadableSheetIsReportedAsFailed() throws {
        let builder = BIFF8TestBuilder(
            sheets: [
                .init(name: "有效数据", cells: [
                    .init(row: 0, column: 0, value: .label("项目")),
                    .init(row: 0, column: 1, value: .label("说明")),
                    .init(row: 1, column: 0, value: .label("保险")),
                    .init(row: 1, column: 1, value: .label("保留可解析的工作表内容")),
                ]),
                .init(name: "损坏表", cells: [], rawPayload: Data()),
            ]
        )

        let output = try LegacyExcelExtractor().extract(url: try write(builder.build()))
        XCTAssertTrue(output.markdown.contains("## Sheet: 有效数据"), output.markdown)
        XCTAssertFalse(output.markdown.contains("损坏表"), output.markdown)
        XCTAssertEqual(output.metadata["skipped_sheets"], "1")
        XCTAssertEqual(output.metadata["failed_pages"], "1")
    }

    func testEmptySheetIsSkipped() throws {
        let builder = BIFF8TestBuilder(
            sheets: [
                .init(name: "空表", cells: []),
                .init(name: "有数据", cells: [
                    .init(row: 0, column: 0, value: .label("内容")),
                ]),
            ]
        )

        let output = try LegacyExcelExtractor().extract(url: try write(builder.build()))
        XCTAssertFalse(output.markdown.contains("## Sheet: 空表"), output.markdown)
        XCTAssertTrue(output.markdown.contains("## Sheet: 有数据"), output.markdown)
        XCTAssertEqual(output.metadata["sheets"], "1")
    }

    func testInvalidFileThrowsParseError() throws {
        let url = directory.appendingPathComponent("broken.xls")
        try Data("not an OLE2 file".utf8).write(to: url)

        XCTAssertThrowsError(try LegacyExcelExtractor().extract(url: url)) { error in
            guard case AgentError.parse(let message) = error else {
                XCTFail("期望解析错误，实际：\(error)")
                return
            }
            XCTAssertTrue(message.contains("broken.xls"), message)
        }
    }

    func testCurrencyThousandsAndFictionalLeapDay() throws {
        let builder = BIFF8TestBuilder(
            sheets: [
                .init(name: "格式", cells: [
                    .init(row: 0, column: 0, value: .label("货币")),
                    .init(row: 0, column: 1, value: .number(1234567.5, xf: 1)),
                    .init(row: 1, column: 0, value: .label("千分位")),
                    .init(row: 1, column: 1, value: .number(1234567, xf: 2)),
                    .init(row: 2, column: 0, value: .label("虚构闰日")),
                    .init(row: 2, column: 1, value: .number(60, xf: 3)),
                ]),
            ],
            formats: [
                (1, "[$¥-804]#,##0.00"),
                (2, "#,##0"),
                (3, "yyyy-mm-dd"),
            ],
            xfFormatIDs: [0, 1, 2, 3])
        let doc = try LegacyExcelExtractor().extract(url: try write(builder.build()))
        XCTAssertTrue(doc.text.contains("¥"), "货币符号应保留")
        XCTAssertTrue(doc.text.contains("1,234,567"), "千分位分隔符应保留")
        XCTAssertTrue(doc.text.contains("1900-02-29"), "序列号 60 是 Excel 虚构闰日")
        XCTAssertNotNil(doc.metadata["skipped_cells"])
    }

    // MARK: - helpers

    private func write(_ file: OLE2TestFile) throws -> URL {
        let url = directory.appendingPathComponent("sample-\(UUID().uuidString).xls")
        try file.data.write(to: url)
        return url
    }
}
