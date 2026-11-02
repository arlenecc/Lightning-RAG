import XCTest
import PDFKit
import CoreGraphics
@testable import BaseAgentCore

/// 回归验证：书商扫描版 PDF（每页一张整页 DCTDecode 灰度图、零文本层，
/// 且 JPEG 带 Photoshop APP13 / 多段 Exif——CoreGraphics 的 JPEG 识别器
/// 拒绝解码导致 `CGContextDrawPDFPage` 渲染全白）必须能产出可 OCR 的
/// 页面图像。覆盖 /Users/arlenecc/Downloads/fail 的失败样本。
final class ScannedPDFRenderingTests: XCTestCase {

    /// 失败样本目录；不存在则跳过（样本含个人书籍，不入库）。
    private func fixtureDirectory() throws -> URL {
        let env = ProcessInfo.processInfo.environment["PDF_FAIL_FIXTURE_DIR"] ?? ""
        let candidates = [
            env.isEmpty ? nil : env,
            "/Users/arlenecc/Downloads/fail",
        ].compactMap { $0 }
        for candidate in candidates where FileManager.default.fileExists(atPath: candidate) {
            return URL(fileURLWithPath: candidate, isDirectory: true)
        }
        throw XCTSkip("缺少失败样本目录（PDF_FAIL_FIXTURE_DIR / ~/Downloads/fail）")
    }

    private func scannedPDFs(in directory: URL) throws -> [URL] {
        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        return names
            .filter { $0.lowercased().hasSuffix(".pdf") }
            .map { directory.appendingPathComponent($0) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// 每个失败样本：页面 20（有效索引）必须产出非空白的页面图像。
    /// 修复前：CoreGraphics 渲染全白 → 全书被当作空白 → 提取失败。
    func testProduceImageRecoversBlankRenderedScans() async throws {
        let directory = try fixtureDirectory()
        let files = try scannedPDFs(in: directory)
        guard !files.isEmpty else { throw XCTSkip("样本目录为空") }

        // 逐文件在独立 autoreleasepool 中处理，控制 14B 级大文件的内存。
        for file in files {
            try autoreleasepool {
                guard let cgDocument = CGPDFDocument(file as CFURL) else {
                    XCTFail("无法打开：\(file.lastPathComponent)")
                    return
                }
                let pageCount = cgDocument.numberOfPages
                // 抽样 3 页（第 20、40、60 页，1-based），验证渲染兜底。
                for pageNumber in [20, 40, 60] where pageNumber <= pageCount {
                    let image = PDFPageRenderer.produceImage(
                        document: cgDocument, pageIndex: pageNumber - 1,
                        targetLongSide: 1408, dpi: 200)
                    guard let image else {
                        XCTFail("\(file.lastPathComponent) p\(pageNumber)：未产出页面图像")
                        continue
                    }
                    XCTAssertFalse(
                        PDFPageRenderer.isNearlyBlank(image),
                        "\(file.lastPathComponent) p\(pageNumber)：页面图像近空白"
                            + "（\(image.width)×\(image.height)）")
                    print("[scan-fix] \(file.lastPathComponent) p\(pageNumber) → "
                        + "\(image.width)×\(image.height) 非空白 ✓")
                }
            }
        }
    }

    /// 直提取路径：`extractEmbeddedJPEG` 对这些文件应直接命中（shim 接受
    /// raw SOI 数据），产出的图像与渲染兜底同样非空白。
    func testEmbeddedJPEGExtractionAcceptsRawSOIData() async throws {
        let directory = try fixtureDirectory()
        let files = try scannedPDFs(in: directory)
        guard !files.isEmpty else { throw XCTSkip("样本目录为空") }

        for file in files {
            try autoreleasepool {
                guard let document = CGPDFDocument(file as CFURL) else { return }
                guard let cgPage = document.page(at: 20) else { return }
                guard PDFTextExtractor.imageObjectCount(for: cgPage) > 0 else { return }
                // 只对「渲染全白」的文件断言直提取成功（其余文件两条路径
                // 本来就都能出图，属于既有覆盖范围）。
                let rendered = PDFPageRenderer.produceImage(
                    document: document, pageIndex: 19,
                    targetLongSide: 1024, dpi: 72)
                guard let rendered, PDFPageRenderer.isNearlyBlank(rendered) else { return }

                let image = PDFPageRenderer.extractLargestEmbeddedJPEG(
                    cgPage: cgPage, targetLongSide: 1408)
                XCTAssertNotNil(image,
                    "\(file.lastPathComponent)：渲染全白且页面有图，流级兜底提取应成功")
                if let image {
                    XCTAssertFalse(PDFPageRenderer.isNearlyBlank(image),
                        "\(file.lastPathComponent)：兜底提取的图像不应为空白")
                }
            }
        }
    }
}
