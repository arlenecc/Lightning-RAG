import XCTest
import Compression
@testable import BaseAgentCore

/// MOBI/AZW3 解析链路的离线回归：PDB 容器、大端序 Header/EXTH、
/// PalmDOC/HuffDic 解压、扫描版判定、ZIP 资源块与拼图引擎。
final class MobiParserTests: XCTestCase {

    // MARK: - 样本构造（全部大端序）

    private func be16(_ value: UInt16) -> [UInt8] {
        [UInt8(truncatingIfNeeded: value >> 8), UInt8(truncatingIfNeeded: value)]
    }

    private func be32(_ value: UInt32) -> [UInt8] {
        [UInt8(truncatingIfNeeded: value >> 24), UInt8(truncatingIfNeeded: value >> 16),
         UInt8(truncatingIfNeeded: value >> 8), UInt8(truncatingIfNeeded: value)]
    }

    private func pdb(records: [Data], type: UInt32 = 0x424F4F4B,
                     creator: UInt32 = 0x4D4F4249) -> Data {
        var out = Data()
        out.append(contentsOf: Array(repeating: 0, count: 32))       // name
        out.append(contentsOf: be16(0))                              // attributes
        out.append(contentsOf: be16(0))                              // version
        out.append(contentsOf: Array(repeating: 0, count: 24))       // 日期 + mod/app/sort
        out.append(contentsOf: be32(type))
        out.append(contentsOf: be32(creator))
        out.append(contentsOf: Array(repeating: 0, count: 8))        // uniqueID + next list
        out.append(contentsOf: be16(UInt16(records.count)))
        var offset = 78 + records.count * 8 + 2  // 记录表后有 2 字节 gap
        for record in records {
            out.append(contentsOf: be32(UInt32(offset)))
            out.append(0)                                            // attributes
            out.append(contentsOf: [0, 0, 0])                        // uniqueID
            offset += record.count
        }
        out.append(contentsOf: [0, 0])                               // gap
        for record in records { out.append(record) }
        return out
    }

    /// Record 0：PalmDOC header(16) + MOBI header(0xE4) + EXTH。
    private func recordZero(textLength: UInt32, compression: UInt16 = 1,
                            firstImage: UInt32 = 0, firstNonBook: UInt32 = 1,
                            mobiType: UInt32 = 2, exth: Bool = true) -> Data {
        var out = Data()
        out.append(contentsOf: be16(compression))
        out.append(contentsOf: be16(0))
        out.append(contentsOf: be32(textLength))
        out.append(contentsOf: be16(1))
        out.append(contentsOf: be16(4096))
        out.append(contentsOf: be16(0))
        out.append(contentsOf: be16(0))

        var mobi = [UInt8](repeating: 0, count: 0xE4)
        func put(_ bytes: [UInt8], at offset: Int) {
            for (index, byte) in bytes.enumerated() where offset + index < mobi.count {
                mobi[offset + index] = byte
            }
        }
        put(Array("MOBI".utf8), at: 0)
        put(be32(0xE4), at: 4)
        put(be32(mobiType), at: 8)
        put(be32(65001), at: 0x0C)
        put(be32(7), at: 0x14)                  // version
        put(be32(firstNonBook), at: 0x40)
        put(be32(0), at: 0x44)                  // fullNameOffset
        put(be32(0), at: 0x48)                  // fullNameLength
        put(be32(firstImage), at: 0x5C)
        put(be32(0), at: 0x60)                  // huffmanRecordOffset
        put(be32(0), at: 0x64)                  // huffmanRecordCount
        put(be32(exth ? 0x40 : 0), at: 0x70)
        out.append(contentsOf: mobi)

        if exth {
            let entries: [(UInt32, String)] = [(99, "测试标题"), (100, "测试作者")]
            var recordsData = Data()
            for (type, value) in entries {
                let data = Array(value.utf8)
                recordsData.append(contentsOf: be32(type))
                recordsData.append(contentsOf: be32(UInt32(8 + data.count)))
                recordsData.append(contentsOf: data)
            }
            var full = Data()
            full.append(contentsOf: Array("EXTH".utf8))
            full.append(contentsOf: be32(UInt32(12 + recordsData.count)))
            full.append(contentsOf: be32(UInt32(entries.count)))
            full.append(recordsData)
            out.append(full)
        }
        return out
    }

    // MARK: - 容器 / Header / EXTH

    func testPDBParsesRecordsWithBigEndianOffsets() throws {
        let data = pdb(records: [Data("record-0".utf8), Data("record-1".utf8)])
        let container = try MobiPDB(data: data)
        XCTAssertEqual(container.recordCount, 2)
        XCTAssertEqual(String(decoding: container.record(at: 0)!, as: UTF8.self), "record-0")
        XCTAssertEqual(String(decoding: container.record(at: 1)!, as: UTF8.self), "record-1")
    }

    func testMobiHeaderAndEXTH() throws {
        let zero = recordZero(textLength: 100, firstImage: 2, firstNonBook: 20)
        let header = try MobiHeader(recordZero: zero)
        XCTAssertEqual(header.firstImageIndex, 2)
        XCTAssertEqual(header.firstNonBookIndex, 20)
        XCTAssertEqual(header.mobiType, 2)
        XCTAssertNotNil(header.exthOffset)
        XCTAssertFalse(header.isKF8)
        let exth = try MobiEXTH(zero, start: header.exthOffset!)
        XCTAssertEqual(exth.string(for: 99), "测试标题")
        XCTAssertEqual(exth.string(for: 100), "测试作者")
        XCTAssertEqual(exth.metadata()["title"], "测试标题")
    }

    func testKF8DetectionViaMobiType() throws {
        let header = try MobiHeader(recordZero: recordZero(textLength: 100,
                                                           mobiType: 232, exth: false))
        XCTAssertTrue(header.isKF8)
    }

    // MARK: - 解压

    func testPalmDOCDecompression() throws {
        // 0x03+'abc'；0x0A 是原样换行；0x78 原样 x；0xC1 = 空格+'A'。
        let compressed = Data([0x03, 0x61, 0x62, 0x63, 0x0A, 0x78, 0xC1])
        var output = Data()
        try MobiCompression.decompressPalmDOC(compressed, into: &output, limit: 0)
        XCTAssertEqual(String(decoding: output, as: UTF8.self), "abc\nx A")
    }

    func testPalmDOCSlidingWindow() throws {
        // 0x80,0x18: distance = ((0x8018)>>3)&0x7ff = 3, length = (0x18&7)+3 = 3
        let compressed = Data([0x03, 0x61, 0x62, 0x63, 0x80, 0x18])
        var output = Data()
        try MobiCompression.decompressPalmDOC(compressed, into: &output, limit: 0)
        XCTAssertEqual(String(decoding: output, as: UTF8.self), "abcabc")
    }

    func testHuffDicRejectsMalformedDictionary() {
        // 正常 HuffDic 路径由 MobiLibmobiSampleTests 的真实样本覆盖。
        XCTAssertThrowsError(try MobiCompression.decompress(
            records: [Data([0, 0])], kind: .huffDic, expectedLength: 2,
            huffmanRecords: [Data("NOPE".utf8)]))
    }

    // MARK: - 扫描版判定

    func testScanDetectionForKindleImageOnlyBook() {
        let html = "<html><body>" + String(repeating: "<img recindex=\"1\"/>", count: 40)
            + String(repeating: "<mbp:pagebreak/>", count: 20) + "</body></html>"
        let result = MobiScanDiagnostics.detect(textLength: 4_096, imageCount: 40,
                                                trailingRecords: 300, html: html)
        XCTAssertTrue(result.isScanned)
        XCTAssertFalse(result.hasTextTag)
        XCTAssertGreaterThan(result.pageBreakCount, 0)
        XCTAssertTrue(result.reasons.contains { $0.contains("无 <text>") })
    }

    func testScanDetectionRejectsNormalTextBook() {
        let sentences = (0..<80).map { "<p>This is paragraph number \($0) with readable content.</p>" }
            .joined()
        let html = "<html><body><text>\(sentences)</text><img src=\"cover.jpg\"/></body></html>"
        let result = MobiScanDiagnostics.detect(textLength: 180_000, imageCount: 1,
                                                trailingRecords: 2, html: html)
        XCTAssertFalse(result.isScanned)
        XCTAssertTrue(result.hasTextTag)
    }

    func testScanDetectionRejectsShortTextWithFewImages() {
        let html = "<html><body><p>A short chapter with only one illustration.</p>"
            + "<img src=\"x.jpg\"/></body></html>"
        XCTAssertFalse(MobiScanDiagnostics.detect(textLength: 3_000, imageCount: 1,
                                                  trailingRecords: 3, html: html).isScanned)
    }

    func testScanDetectionLongBookFallsBackToPerImageDensity() {
        let html = "<html><body>" + String(repeating: "<img recindex=\"1\"/>", count: 60)
            + String(repeating: "<mbp:pagebreak/>", count: 60) + "</body></html>"
        // 正文超过 32KB 且尾记录 < 50：主判据不命中，须走 EPUB 式密度兜底。
        let result = MobiScanDiagnostics.detect(textLength: 120_000, imageCount: 60,
                                                trailingRecords: 8, html: html)
        XCTAssertTrue(result.isScanned)
        XCTAssertTrue(result.reasons.contains { $0.contains("密度兜底") })
    }

    func testScanDetectionRejectsMixedTextAndImages() {
        // 图文混排书：图片多、有分页符，但有大量可读正文，不得判为扫描版。
        let html = "<html><body>" + String(repeating: "<img recindex=\"1\"/>", count: 20)
            + String(repeating: "<mbp:pagebreak/>", count: 20)
            + "<p>" + String(repeating: "这是正文内容。", count: 100) + "</p></body></html>"
        let result = MobiScanDiagnostics.detect(textLength: 30_000, imageCount: 20,
                                                trailingRecords: 100, html: html)
        XCTAssertFalse(result.isScanned, "图文混排不得因图片多被误判为扫描版")
    }

    func testScanDetectionDoesNotTreatTextareaAsTextTag() {
        let html = "<html><body><textarea></textarea>"
            + String(repeating: "<img recindex=\"1\"/>", count: 40) + "</body></html>"
        let result = MobiScanDiagnostics.detect(textLength: 4_096, imageCount: 40,
                                                trailingRecords: 300, html: html)
        XCTAssertFalse(result.hasTextTag)
    }

    // MARK: - ZIP / 拼图

    private func zipEntry(name: String, data: Data, compressed: Bool) -> Data {
        let raw = [UInt8](data)
        let method: UInt16 = compressed ? 8 : 0
        var payload: [UInt8] = raw
        if compressed {
            var encoded = [UInt8](repeating: 0, count: raw.count + 64)
            let count = encoded.withUnsafeMutableBytes { dst in
                compression_encode_buffer(dst.baseAddress!.assumingMemoryBound(to: UInt8.self),
                                          dst.count, raw, raw.count, nil, COMPRESSION_ZLIB)
            }
            // 本机 COMPRESSION_ZLIB 收发裸 deflate，输出可直接作为 ZIP 条目数据。
            payload = Array(encoded[0..<count])
        }
        func le16(_ v: UInt16) -> [UInt8] {
            [UInt8(truncatingIfNeeded: v), UInt8(truncatingIfNeeded: v >> 8)]
        }
        func le32(_ v: UInt32) -> [UInt8] {
            [UInt8(truncatingIfNeeded: v), UInt8(truncatingIfNeeded: v >> 8),
             UInt8(truncatingIfNeeded: v >> 16), UInt8(truncatingIfNeeded: v >> 24)]
        }
        let nameBytes = Array(name.utf8)
        var local = Data()
        local.append(contentsOf: le32(0x04034b50))
        local.append(contentsOf: le16(20))
        local.append(contentsOf: le16(0))
        local.append(contentsOf: le16(method))
        local.append(contentsOf: [0, 0, 0, 0, 0, 0, 0, 0])
        local.append(contentsOf: le32(UInt32(payload.count)))
        local.append(contentsOf: le32(UInt32(raw.count)))
        local.append(contentsOf: le16(UInt16(nameBytes.count)))
        local.append(contentsOf: le16(0))
        local.append(contentsOf: nameBytes)
        local.append(contentsOf: payload)

        var central = Data()
        central.append(contentsOf: le32(0x02014b50))
        central.append(contentsOf: le16(20))
        central.append(contentsOf: le16(20))
        central.append(contentsOf: le16(0))
        central.append(contentsOf: le16(method))
        central.append(contentsOf: [0, 0, 0, 0, 0, 0, 0, 0])
        central.append(contentsOf: le32(UInt32(payload.count)))
        central.append(contentsOf: le32(UInt32(raw.count)))
        central.append(contentsOf: le16(UInt16(nameBytes.count)))
        central.append(contentsOf: [0, 0, 0, 0, 0, 0])
        central.append(contentsOf: [0, 0, 0, 0, 0, 0])     // internal(2) + external(4) attrs
        central.append(contentsOf: le32(0))             // local header offset
        central.append(contentsOf: nameBytes)

        var zip = local
        let centralOffset = zip.count
        zip.append(central)
        zip.append(contentsOf: le32(0x06054b50))
        zip.append(contentsOf: [0, 0, 0, 0])
        zip.append(contentsOf: le16(1))
        zip.append(contentsOf: le16(1))
        zip.append(contentsOf: le32(UInt32(central.count)))
        zip.append(contentsOf: le32(UInt32(centralOffset)))
        zip.append(contentsOf: le16(0))
        return zip
    }

    func testZIPReaderExtractsStoreAndDeflateImages() {
        let image = Data([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0xFF, 0xD9])
        let zip = zipEntry(name: "page001.jpg", data: image, compressed: true)
        let entries = MobiZIPReader.images(in: zip)
        XCTAssertEqual(entries.count, 1)
        XCTAssertEqual(entries.first?.name, "page001.jpg")
        
        if let actual = entries.first?.data, actual != image {
            XCTFail("解压内容不符: \(Array(actual)) vs \(Array(image))")
        }
    }

    func testStitchEngineDetectsDenseImages() {
        let html = "<table><tr><td><img src=\"1\"/></td><td><img src=\"2\"/></td></tr></table>"
        XCTAssertTrue(MobiStitchEngine.shouldStitch(html: html))
        XCTAssertFalse(MobiStitchEngine.shouldStitch(html: "<p><img src=\"1\"/></p><p>text</p>"))
    }

    func testStitchEngineHandlesUnicodeCaseFolding() {
        // lowecased() 会改变 İ 等字符的 UTF-16 长度；索引不能跨字符串实例使用。
        let html = "<TABLE><TR><TD>İİİ</TD><TD><IMG src=\"1\"/></TD>"
            + "<TD><IMG src=\"2\"/></TD></TR></TABLE>"
        XCTAssertTrue(MobiStitchEngine.shouldStitch(html: html))
    }

    func testStitchEngineCombinesImages() {
        func image(_ color: CGFloat) -> CGImage? {
            let size = CGSize(width: 20, height: 40)
            guard let context = CGContext(data: nil, width: 20, height: 40,
                                          bitsPerComponent: 8, bytesPerRow: 0,
                                          space: CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return nil }
            context.setFillColor(CGColor(red: color, green: 0, blue: 0, alpha: 1))
            context.fill(CGRect(origin: .zero, size: size))
            return context.makeImage()
        }
        guard let first = image(1), let second = image(0) else {
            return XCTFail("无法构造测试图片")
        }
        let stitched = MobiStitchEngine.stitch([first, second], columns: 2)
        XCTAssertNotNil(stitched)
        XCTAssertGreaterThanOrEqual(stitched?.width ?? 0, 40)
        XCTAssertGreaterThanOrEqual(stitched?.height ?? 0, 40)
    }

    // MARK: - 接入

    func testSupportedExtensionsIncludeMobiFormats() {
        XCTAssertTrue(RAGEngine.supportedExtensions.contains("mobi"))
        XCTAssertTrue(RAGEngine.supportedExtensions.contains("azw3"))
        XCTAssertTrue(RAGEngine.supportedExtensions.contains("prc"))
    }

    func testDefaultRegistryHasMobiParser() {
        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory() + "mobi-parser-\(UUID().uuidString)"
        let registry = DocumentParserRegistry.makeDefault(config: config)
        let options = ParserOptions()
        XCTAssertNotNil(registry.parser(for: URL(fileURLWithPath: "/tmp/book.mobi"),
                                        options: options))
        XCTAssertNotNil(registry.parser(for: URL(fileURLWithPath: "/tmp/book.azw3"),
                                        options: options))
        XCTAssertNotNil(registry.parser(for: URL(fileURLWithPath: "/tmp/book.prc"),
                                        options: options))
    }
}
