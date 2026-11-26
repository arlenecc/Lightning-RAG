import AppKit
import XCTest
@testable import BaseAgentCore

private final class LegacyImageFakeVLM: VLMService {
    let identifier = "legacy-image-fake"
    let isAvailable = true
    private let lock = NSLock()
    private var _calls = 0
    var calls: Int { lock.lock(); defer { lock.unlock() }; return _calls }

    func recognize(image: CGImage, prompt: String, maxTokens: Int) async throws -> String {
        lock.lock(); _calls += 1; lock.unlock()
        return "| 图内数据 | 99 |"
    }
    func unload() async {}
}

final class LegacyOfficeIntegrationTests: XCTestCase {

    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-legacy-office-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testRegistryRoutesLegacyOfficeExtensions() throws {
        var config = AgentConfig()
        config.workspace = directory.path
        let registry = DocumentParserRegistry.makeDefault(config: config)
        let options = ParserOptions()

        for ext in ["doc", "xls", "ppt"] {
            let url = directory.appendingPathComponent("sample.\(ext)")
            let parser = registry.parser(for: url, options: options)
            XCTAssertTrue(parser is LegacyOfficeParser,
                          "\(ext) 应路由到 LegacyOfficeParser，实际 \(String(describing: parser))")
            let fresh = registry.freshParser(for: url, options: options)
            XCTAssertTrue(fresh is LegacyOfficeParser)
        }
    }

    func testSupportedExtensionsAndCleanProfileIncludeLegacyFormats() {
        XCTAssertTrue(RAGEngine.supportedExtensions.isSuperset(of: ["doc", "xls", "ppt"]))
        for ext in ["doc", "xls", "ppt"] {
            let url = URL(fileURLWithPath: "/tmp/sample.\(ext)")
            XCTAssertEqual(DocumentProfile.defaultProfile(for: url), .structured)
        }
    }

    func testLegacyParserParsesExcelWorkbook() async throws {
        let builder = BIFF8TestBuilder(
            sheets: [
                .init(name: "集成", cells: [
                    .init(row: 0, column: 0, value: .label("内容")),
                    .init(row: 1, column: 0, value: .label("中文")),
                ]),
            ]
        )
        let url = try write(builder.build(), ext: "xls")

        let parsed = try await LegacyOfficeParser().parse(
            url: url, options: ParserOptions(), progress: nil)
        XCTAssertTrue(parsed.text.contains("## Sheet: 集成"), parsed.text)
        XCTAssertEqual(parsed.metadata["engine"], "xls")
        XCTAssertEqual(parsed.metadata["clean_profile"], "structured")
        XCTAssertEqual(parsed.metadata["cleaned"], "1")
        XCTAssertTrue(parsed.outline.contains("Sheet: 集成"), parsed.outline)
    }

    func testLegacyParserParsesPowerPointPresentation() async throws {
        let builder = PowerPointTestBuilder(
            slides: [
                .init(slideId: 1, blocks: [
                    .init(headerType: 0, text: "集成测试"),
                    .init(headerType: 1, text: "正文内容"),
                ]),
            ],
            notes: [
                .init(slideIdRef: 1, blocks: [
                    .init(headerType: 2, text: "演讲备注"),
                ]),
            ]
        )
        let url = try write(builder.build(), ext: "ppt")

        let parsed = try await LegacyOfficeParser().parse(
            url: url, options: ParserOptions(), progress: nil)
        XCTAssertTrue(parsed.text.contains("## Slide 1：集成测试"), parsed.text)
        XCTAssertTrue(parsed.text.contains("**演讲者备注**"), parsed.text)
        XCTAssertEqual(parsed.metadata["engine"], "ppt")
        XCTAssertEqual(parsed.metadata["clean_profile"], "structured")
        XCTAssertTrue(parsed.outline.contains("Slide 1：集成测试"), parsed.outline)
    }

    func testLegacyParserParsesWordDocument() async throws {
        let url = try makeDocument()

        let parsed = try await LegacyOfficeParser().parse(
            url: url, options: ParserOptions(), progress: nil)
        XCTAssertTrue(parsed.text.contains("集成标题"), parsed.text)
        XCTAssertTrue(parsed.text.contains("中文正文内容。"), parsed.text)
        XCTAssertEqual(parsed.metadata["engine"], "doc")
        XCTAssertEqual(parsed.metadata["clean_profile"], "structured")
    }

    func testLegacyImageExtractorScansOLE2StyleStream() throws {
        let png = Data(base64Encoded:
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")!
        let stream = Data([UInt8](repeating: 0xAB, count: 32)) + png + Data([0x01, 0x02])
        let images = LegacyOfficeImageExtractor.images(in: stream, label: "Data")
        XCTAssertEqual(images.count, 1)
        XCTAssertEqual(images.first, png)
    }

    func testLegacyImageOCRIsDeduplicated() async throws {
        let png = Data(base64Encoded:
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")!
        let fake = LegacyImageFakeVLM()
        let texts = try await LegacyOfficeImageExtractor.recognize(
            [png, png], vlm: fake,
            options: ParserOptions(epubOCREngine: .ovis), progress: nil)
        XCTAssertEqual(texts.count, 1)
        XCTAssertEqual(fake.calls, 1, "重复图片只应 OCR 一次")
        XCTAssertTrue(texts.first?.contains("图内数据") ?? false)
    }

    // MARK: - helpers

    private func write(_ file: OLE2TestFile, ext: String) throws -> URL {
        let url = directory.appendingPathComponent("sample-\(UUID().uuidString).\(ext)")
        try file.data.write(to: url)
        return url
    }

    private func makeDocument() throws -> URL {
        let attributed = NSMutableAttributedString()
        attributed.append(NSAttributedString(
            string: "集成标题",
            attributes: [.font: NSFont.boldSystemFont(ofSize: 18)]))
        attributed.append(NSAttributedString(
            string: "\n",
            attributes: [.font: NSFont.systemFont(ofSize: 12)]))
        attributed.append(NSAttributedString(
            string: "中文正文内容。",
            attributes: [.font: NSFont.systemFont(ofSize: 12)]))

        let fullRange = NSRange(location: 0, length: attributed.length)
        let data = try attributed.data(from: fullRange, documentAttributes: [
            .documentType: NSAttributedString.DocumentType.docFormat,
        ])
        let url = directory.appendingPathComponent("sample-\(UUID().uuidString).doc")
        try data.write(to: url)
        return url
    }
}
