import XCTest
@testable import BaseAgentCore

/// 用 tools/libmobi/tests/samples 里的真实书做端到端验证（门控，默认跳过）。
/// 运行方式：MOBI_SAMPLE_DIR=<samples 目录> swift test --filter MobiLibmobiSampleTests
final class MobiLibmobiSampleTests: XCTestCase {
    func testDRMSamples() async throws {
        guard let dir = ProcessInfo.processInfo.environment["MOBI_SAMPLE_DIR"] else {
            throw XCTSkip("设置 MOBI_SAMPLE_DIR 指向 libmobi/tests/samples 后启用")
        }
        let v1URL = URL(fileURLWithPath: dir)
            .appendingPathComponent("sample-drm-v1.mobi")
        let v1 = try await MobiParser(config: nil).parse(
            url: v1URL, options: ParserOptions(), progress: nil)
        XCTAssertEqual(v1.metadata["drm"], "v1")
        XCTAssertGreaterThan(v1.text.count, 2000, "V1 解密后应有正文")

        let v2URL = URL(fileURLWithPath: dir)
            .appendingPathComponent("sample-drm_pidLTKULBB^5V-v2.mobi")
        let v2 = try await MobiParser(config: nil, drmPID: "LTKULBB*5V").parse(
            url: v2URL, options: ParserOptions(), progress: nil)
        XCTAssertEqual(v2.metadata["drm"], "v2")
        XCTAssertGreaterThan(v2.text.count, 2000, "V2 解密后应有正文")
        print("DRM v1=\(v1.text.count) v2=\(v2.text.count)")
    }

    func testLibmobiSamples() async throws {
        guard let dir = ProcessInfo.processInfo.environment["MOBI_SAMPLE_DIR"] else {
            throw XCTSkip("设置 MOBI_SAMPLE_DIR 指向 libmobi/tests/samples 后启用")
        }
        let files: [String]
        if let only = ProcessInfo.processInfo.environment["MOBI_SAMPLE_FILE"] {
            files = [only]
        } else {
            files = try FileManager.default.contentsOfDirectory(atPath: dir)
                .filter { ($0 as NSString).pathExtension.lowercased() == "mobi" }
                .filter { !$0.contains("drm") }    // DRM 书本解析器明确拒绝
                .sorted()
        }
        XCTAssertFalse(files.isEmpty)
        let parser = MobiParser(config: nil)
        var huffVerified = false
        var cp1252Verified = false

        for file in files {
            let url = URL(fileURLWithPath: dir).appendingPathComponent(file)
            do {
                let data = try Data(contentsOf: url)
                let pdb = try MobiPDB(data: data)
                let zero = try XCTUnwrap(pdb.record(at: 0), "\(file): 缺少 Record 0")
                let palm = try MobiPalmHeader(zero)
                let header = (try? MobiHeader(recordZero: zero)) ?? MobiHeader.fallback(palm: palm, recordCount: pdb.recordCount)
                let kind = MobiCompressionKind(rawValue: palm.compressionRaw)
                let doc = try await parser.parse(url: url, options: ParserOptions(),
                                                 progress: nil)
                // obfuscated-fonts 是字体混淆测试书，正文只有空 span 与 CSS，
                // 本身没有可读正文，不纳入「必须有文本」的断言。
                if !file.contains("obfuscated-fonts") {
                    XCTAssertFalse(doc.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                                   "\(file): 解析结果为空")
                }
                print("SAMPLE \(file): records=\(pdb.recordCount) type=\(header.mobiType) "
                    + "v=\(header.version) enc=\(header.textEncoding) "
                    + "compression=\(palm.compressionRaw) huff=\(kind.map { String(describing: $0) } ?? "nil") "
                    + "textLen=\(palm.textLength) out=\(doc.text.count) "
                    + "title=\(header.fullName) kf8=\(header.isKF8)")
                if file.contains("obfuscated-fonts") {
                    let flowCount = Int(doc.metadata["flow_count"] ?? "0") ?? 0
                    XCTAssertGreaterThan(flowCount, 1, "KF8 应切出多个 flow")
                }
                if file.contains("ncx") {
                    let tocCount = Int(doc.metadata["toc_count"] ?? "0") ?? 0
                    XCTAssertGreaterThan(tocCount, 0, "INDX/NCX 应解析出目录条目")
                }
                if file.contains("dict-infl2") {
                    let marks = Int(doc.metadata["dictionary_mark_count"] ?? "0") ?? 0
                    XCTAssertGreaterThan(marks, 0, "词典应解析出 orth/infl 标记")
                }
                if file.contains("huffdic") {
                    XCTAssertGreaterThan(doc.text.count, 200, "HuffDic 书解出的文本过短")
                    huffVerified = true
                }
                if file.contains("cp1252") {
                    XCTAssertFalse(header.fullName.contains("\u{FFFD}"),
                                   "CP1252 书名解码出现替换字符")
                    cp1252Verified = true
                }
            } catch {
                XCTFail("\(file): \(String(describing: error))")
            }
        }
        if ProcessInfo.processInfo.environment["MOBI_SAMPLE_FILE"] == nil {
            XCTAssertTrue(huffVerified, "未验证 HuffDic 样本")
            XCTAssertTrue(cp1252Verified, "未验证 CP1252 样本")
        }
    }
}
