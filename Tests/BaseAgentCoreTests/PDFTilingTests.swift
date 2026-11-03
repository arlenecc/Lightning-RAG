import XCTest
import Foundation
import CoreGraphics
@testable import BaseAgentCore

/// 验证极端长宽比（整书压成单张超高窄条）PDF 的渲染分块逻辑：
/// 不再把短边压到几十像素导致整页空白，而是切成多条可读方块。
final class PDFTilingTests: XCTestCase {

    func testExtremeAspectPageIsTiledIntoReadableStrips() throws {
        let url = try Self.makePDF(widthPt: 200, heightPt: 4000)
        guard let doc = CGPDFDocument(url as CFURL) else { throw XCTSkip("合成 PDF 无法打开") }
        let images = PDFPageRenderer.produceImages(
            document: doc, pageIndex: 0, targetLongSide: 1408, dpi: 200)
        XCTAssertGreaterThan(images.count, 1, "超高窄条应被切成多条瓦片")
        for img in images {
            let short = min(img.width, img.height)
            XCTAssertGreaterThanOrEqual(short, 300, "瓦片短边须可读，实际 \(short)px")
        }
    }

    func testNormalPageStaysSingleImage() throws {
        let url = try Self.makePDF(widthPt: 600, heightPt: 800)
        guard let doc = CGPDFDocument(url as CFURL) else { throw XCTSkip("合成 PDF 无法打开") }
        let images = PDFPageRenderer.produceImages(
            document: doc, pageIndex: 0, targetLongSide: 1408, dpi: 200)
        XCTAssertEqual(images.count, 1, "常规页面应保持单张图")
    }

    /// 真实回归：本机 Downloads 下的两本「整书压成单张超长图」PDF，
    /// 验证 produceImages 不再产出 60px 宽的细条，而是可读取的多条瓦片。
    func testRealExtremePDFsTiling() throws {
        let names = [
            "成为-第一夫人养成记.pdf",
            "24堂财富课：陈志武和女儿谈商业模式-如何与孩子谈钱？.pdf",
        ]
        var tested = 0
        for name in names {
            let url = URL(fileURLWithPath: NSHomeDirectory())
                .appendingPathComponent("Downloads")
                .appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path),
                  let doc = CGPDFDocument(url as CFURL) else { continue }
            let images = PDFPageRenderer.produceImages(
                document: doc, pageIndex: 0, targetLongSide: 1408, dpi: 200)
            XCTAssertGreaterThan(images.count, 1, "\(name) 应被分块")
            // 至少应有一块完整可读瓦片（短边 ≥1000）；修复前整页会被压成 ~66px 细条、
            // 且 images.count==1，会被上面的断言拦截。尾块偏短属正常。
            let maxShort = images.map { min($0.width, $0.height) }.max() ?? 0
            XCTAssertGreaterThanOrEqual(
                maxShort, 1000, "\(name) 至少应有可读整块瓦片，实际最大短边 \(maxShort)px")
            tested += 1
        }
        if tested == 0 { throw XCTSkip("本机 Downloads 无待测 PDF") }
    }

    private static func makePDF(widthPt: CGFloat, heightPt: CGFloat) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiling-\(Int(widthPt))x\(Int(heightPt)).pdf")
        var mediaBox = CGRect(x: 0, y: 0, width: widthPt, height: heightPt)
        guard let consumer = CGDataConsumer(url: url as CFURL),
              let ctx = CGContext(consumer: consumer, mediaBox: &mediaBox, nil) else {
            throw NSError(domain: "PDFTilingTests", code: 1, userInfo: nil)
        }
        ctx.beginPDFPage(nil)
        // 画横向黑条，保证页面与瓦片都不是空白。
        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        var y: CGFloat = 0
        while y < heightPt {
            ctx.fill(CGRect(x: 0, y: y, width: widthPt, height: 18))
            y += 60
        }
        ctx.endPDFPage()
        return url
    }
}
