import Foundation
import PDFKit
import CoreGraphics
import ImageIO
import CPDFShim

/// 页面图像 + 页序，用于在并发任务之间安全传递。
public struct PDFPageImage: @unchecked Sendable {
    public let pageIndex: Int
    public let image: CGImage

    public init(pageIndex: Int, image: CGImage) {
        self.pageIndex = pageIndex
        self.image = image
    }
}

/// PDF 页面栅格化：PDFKit 负责解析与 /Rotate 旋转，CoreGraphics 负责位图输出。
///
/// 关键行为（对齐原项目）：
/// - 按 DPI 渲染（默认 200），并限制最长边与总像素数，避免扫描件占用过多内存；
/// - 使用 CropBox（缺失时回退 MediaBox），并补偿 CropBox 非零原点；
/// - 输出白底位图，供 OvisOCR3 或 Apple Vision 使用。
public enum PDFPageRenderer {

    /// 页级并发的上限（与解析器一致）。每次渲染 8 页，OCR 阶段统一受
    /// VisionPDFParser 的全局并发闸门约束，避免文件级并发叠加后打满 CPU。
    public static let maxConcurrentPages = 8
    /// 判断一页文本层是否可用：有效字符数达到该阈值就直接走 PDFKit，不做 OCR。
    public static let textLayerMinCharacters = 50
    public static let defaultMaxLongSide = 4096
    public static let defaultMaxPixelCount = 16_000_000
    /// 内嵌 JPEG 的像素数安全上限（200MP）。超过时放弃「直提取」，
    /// 回退到有界的页面栅格化：扫描页最终长边只有 1408/1500，
    /// 没必要为一页图像承担数 GB 的解码内存。
    public static let maxEmbeddedPixelCount = 200_000_000

    /// 扫描页渲染 worker 数：每 worker 独占一个 CGPDFDocument 实例串行产图，
    /// worker 之间并行。文件级并发由 PDFPageScanRunner 的全局闸门统一限流。
    static var requestedRenderWorkers: Int?

    static var renderWorkers: Int {
        requestedRenderWorkers ?? max(2, min(4, ProcessInfo.processInfo.activeProcessorCount))
    }

    /// 渲染单页为 CGImage。失败时返回 nil（由调用方决定跳过该页）。
    ///
    /// 使用 `CGPDFPage.getDrawingTransform` 而不是手工翻转 CGContext：
    /// PDFKit 的 `page.draw` 自身已带坐标变换，再手动 scale(y: -1) 会把
    /// 扫描页整体旋转 180°，Vision OCR 会输出大量乱码。
    public static func render(page: PDFPage,
                              dpi: Double = 200,
                              maxLongSide: Int = defaultMaxLongSide,
                              maxPixelCount: Int = defaultMaxPixelCount) -> CGImage? {
        if let cgPage = page.pageRef {
            return render(cgPage: cgPage, dpi: dpi,
                          maxLongSide: maxLongSide,
                          maxPixelCount: maxPixelCount)
        }
        // pageRef 不可用时回退：CGContext 本身是 y 向上的 PDF 坐标系，
        // 不再额外翻转；仅补偿 CropBox 的非零原点。
        var box = page.bounds(for: .cropBox)
        if box.width < 1 || box.height < 1 {
            box = page.bounds(for: .mediaBox)
        }
        guard box.width >= 1, box.height >= 1 else { return nil }
        let (pixelWidth, pixelHeight, scale) = pixelSize(
            displayWidth: box.width, displayHeight: box.height,
            dpi: dpi, maxLongSide: maxLongSide, maxPixelCount: maxPixelCount)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: nil,
                                      width: pixelWidth,
                                      height: pixelHeight,
                                      bitsPerComponent: 8,
                                      bytesPerRow: 0,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            return nil
        }
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: pixelWidth, height: pixelHeight))
        context.interpolationQuality = .high
        context.scaleBy(x: scale, y: scale)
        context.translateBy(x: -box.minX, y: -box.minY)
        page.draw(with: .cropBox, to: context)
        return context.makeImage()
    }

    private static func render(cgPage: CGPDFPage,
                               dpi: Double,
                               maxLongSide: Int,
                               maxPixelCount: Int) -> CGImage? {
        var box = cgPage.getBoxRect(.cropBox)
        if box.width < 1 || box.height < 1 {
            box = cgPage.getBoxRect(.mediaBox)
        }
        guard box.width >= 1, box.height >= 1 else { return nil }

        let rotation = ((cgPage.rotationAngle % 360) + 360) % 360
        let quarterTurn = rotation == 90 || rotation == 270
        let displayWidth = quarterTurn ? box.height : box.width
        let displayHeight = quarterTurn ? box.width : box.height
        let (pixelWidth, pixelHeight, _) = pixelSize(
            displayWidth: displayWidth, displayHeight: displayHeight,
            dpi: dpi, maxLongSide: maxLongSide, maxPixelCount: maxPixelCount)

        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: nil,
                                      width: pixelWidth,
                                      height: pixelHeight,
                                      bitsPerComponent: 8,
                                      bytesPerRow: 0,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            return nil
        }
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: pixelWidth, height: pixelHeight))
        context.interpolationQuality = .medium

        // getDrawingTransform 只缩小、不放大：目标尺寸大于页面 Box 时它返回
        // 1:1 居中变换，小尺寸页面会以 72dpi 渲染在巨大空白画布中央，
        // 既降低 OCR 清晰度，也让按页面坐标裁剪的表格区域完全错位。
        // 因此先用它把页面装进与显示尺寸等大的矩形（处理旋转/CropBox），
        // 再由外层 CTM 统一缩放到目标像素尺寸。
        let displayRect = CGRect(x: 0, y: 0,
                                 width: displayWidth, height: displayHeight)
        let fit = cgPage.getDrawingTransform(.cropBox, rect: displayRect,
                                             rotate: 0, preserveAspectRatio: true)
        context.scaleBy(x: CGFloat(pixelWidth) / displayWidth,
                        y: CGFloat(pixelHeight) / displayHeight)
        context.concatenate(fit)
        context.drawPDFPage(cgPage)
        return context.makeImage()
    }

    // MARK: - 内嵌图片直接提取（跳过栅格化）

    /// 扫描页图像统一入口：优先直接提取页面内嵌的整页 JPEG（零重采样），
    /// 不满足条件（多图/非 JPEG/带蒙版/旋转/比例不符）时回退 CoreGraphics 栅格化。
    ///
    /// - Parameters:
    ///   - pageIndex: 0-based 页序。
    ///   - targetLongSide: OCR 预处理的目标长边（Vision 1500 / Ovis 1408），
    ///     超过时缩一次到位，避免「放大渲染再缩小」的浪费。
    static func produceImage(document: CGPDFDocument,
                             pageIndex: Int,
                             targetLongSide: Int,
                             dpi: Double) -> CGImage? {
        guard let cgPage = document.page(at: pageIndex + 1) else { return nil }
        if let extracted = extractEmbeddedJPEG(cgPage: cgPage,
                                               targetLongSide: targetLongSide) {
            return extracted
        }
        let rendered = render(cgPage: cgPage, dpi: dpi,
                              maxLongSide: targetLongSide,
                              maxPixelCount: defaultMaxPixelCount)
        // 渲染出白图但页面确实有图片 XObject：不是真空页，而是 CoreGraphics
        // 拒绝解码该图片（书商扫描件的 JPEG 带特殊 APP 段时会发生，实测
        // 「合同法 第6版」「海商法 第5版」全书渲染全白）。流级兜底：把页内
        // 最大的 DCTDecode 图片直接交给 ImageIO 解码。
        if rendered == nil || isNearlyBlank(rendered!) {
            if PDFTextExtractor.imageObjectCount(for: cgPage) > 0,
               let fallback = extractLargestEmbeddedJPEG(cgPage: cgPage,
                                                         targetLongSide: targetLongSide) {
                return fallback
            }
        }
        return rendered
    }

    /// 兜底：提取页面内像素数最大的 DCTDecode 图片流，直接交给 ImageIO 解码。
    /// 与 `extractEmbeddedJPEG` 的差异：不要求整页唯一图片流，且接受
    /// CoreGraphics 报告 raw 但以 SOI 开头的 JPEG 字节（见 CPDFShim.h）。
    static func extractLargestEmbeddedJPEG(cgPage: CGPDFPage,
                                           targetLongSide: Int) -> CGImage? {
        guard let jpeg = CPDFShimCopyLargestEmbeddedJPEG(cgPage) else { return nil }
        let data = jpeg.takeRetainedValue() as Data
        return decodeEmbeddedJPEG(data, targetLongSide: targetLongSide)
    }

    /// 提取页面唯一的大尺寸 JPEG（扫描件整页图）。C 级判断在 CPDFShim 中，
    /// 任何不确定的情况都返回 nil 走渲染兜底，确保不丢内容。
    ///
    /// 关键：解码时直接用 ImageIO 缩略图接口按目标长边降采样，而不是
    /// `CGImageSourceCreateImageAtIndex` 先解出整张原图再 `fitted` 缩小。
    /// 高像素扫描页（例如 20000×20000）完整解码单页就要 1.6GB，
    /// 多个 worker 在途时会瞬间吃掉几十 GB。
    static func extractEmbeddedJPEG(cgPage: CGPDFPage,
                                    targetLongSide: Int) -> CGImage? {
        guard let jpeg = CPDFShimCopyEmbeddedPageJPEG(cgPage) else { return nil }
        let data = jpeg.takeRetainedValue() as Data
        return decodeEmbeddedJPEG(data, targetLongSide: targetLongSide)
    }

    /// 把内嵌 JPEG 数据解码为不超过 `targetLongSide` 的位图。
    ///
    /// - `maxPixelCount` 为解码前的像素数预检：超过上限直接放弃直提取，
    ///   由调用方走有界栅格化，避免解压炸弹式 JPEG。
    /// - 使用 `CGImageSourceCreateThumbnailAtIndex` + `ShouldCacheImmediately`，
    ///   ImageIO 会在解码阶段直接做子采样，峰值内存约为目标尺寸位图，
    ///   与原始 JPEG 的像素数无关。
    static func decodeEmbeddedJPEG(_ data: Data,
                                   targetLongSide: Int,
                                   maxPixelCount: Int = maxEmbeddedPixelCount) -> CGImage? {
        guard targetLongSide > 0, !data.isEmpty,
              let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            return nil
        }

        if maxPixelCount > 0,
           let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
            as? [CFString: Any],
           let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
           let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
           width > 0, height > 0,
           Int64(width) * Int64(height) > Int64(maxPixelCount) {
            return nil
        }

        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceThumbnailMaxPixelSize: max(1, targetLongSide),
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(
            source, 0, options as CFDictionary) else {
            return nil
        }
        return fitted(image, targetLongSide: targetLongSide)
    }

    /// 长边超出目标时缩一次到位；带 alpha 通道时合成白底。
    static func fitted(_ image: CGImage, targetLongSide: Int) -> CGImage {
        let longSide = max(image.width, image.height)
        let hasAlpha: Bool
        switch image.alphaInfo {
        case .none, .noneSkipFirst, .noneSkipLast: hasAlpha = false
        default: hasAlpha = true
        }
        guard longSide > targetLongSide || hasAlpha else { return image }

        let factor = longSide > targetLongSide
            ? Double(targetLongSide) / Double(longSide) : 1.0
        let width = max(1, Int((Double(image.width) * factor).rounded()))
        let height = max(1, Int((Double(image.height) * factor).rounded()))
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: nil,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: 0,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            return image
        }
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.interpolationQuality = .medium
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage() ?? image
    }

    /// 根据 DPI / 最长边 / 总像素上限计算目标位图尺寸与缩放比。
    private static func pixelSize(displayWidth: CGFloat,
                                  displayHeight: CGFloat,
                                  dpi: Double,
                                  maxLongSide: Int,
                                  maxPixelCount: Int) -> (width: Int, height: Int, scale: CGFloat) {
        var scale = max(dpi, 36) / 72.0
        if maxLongSide > 0 {
            let rawLongSide = max(displayWidth, displayHeight) * scale
            if rawLongSide > Double(maxLongSide) {
                scale *= CGFloat(Double(maxLongSide) / Double(rawLongSide))
            }
        }
        var pixelWidth = max(1, Int((displayWidth * scale).rounded()))
        var pixelHeight = max(1, Int((displayHeight * scale).rounded()))
        if maxPixelCount > 0, pixelWidth * pixelHeight > maxPixelCount {
            let factor = sqrt(Double(maxPixelCount) / (Double(pixelWidth) * Double(pixelHeight)))
            pixelWidth = max(1, Int((Double(pixelWidth) * factor).rounded()))
            pixelHeight = max(1, Int((Double(pixelHeight) * factor).rounded()))
        }
        scale = min(CGFloat(pixelWidth) / displayWidth,
                    CGFloat(pixelHeight) / displayHeight)
        return (pixelWidth, pixelHeight, scale)
    }

    /// 从已渲染的整页位图中裁剪指定页面区域（PDF 页面坐标，原点左下）。
    ///
    /// 仅支持未旋转页面；旋转页返回 nil，由调用方保留文本层内容。
    /// 整页先按较高 DPI 渲染一次，再按比例裁剪，避免每张表重复渲染整页。
    public static func crop(image: CGImage, page: PDFPage, rect: CGRect) -> CGImage? {
        let rotation = ((page.rotation % 360) + 360) % 360
        guard rotation == 0 else { return nil }

        var box = page.bounds(for: .cropBox)
        if box.width < 1 || box.height < 1 {
            box = page.bounds(for: .mediaBox)
        }
        guard box.width >= 1, box.height >= 1 else { return nil }

        let target = rect.intersection(box)
        guard target.width >= 8, target.height >= 8 else { return nil }

        let scaleX = CGFloat(image.width) / box.width
        let scaleY = CGFloat(image.height) / box.height
        let pixelRect = CGRect(
            x: (target.minX - box.minX) * scaleX,
            y: (box.maxY - target.maxY) * scaleY,
            width: target.width * scaleX,
            height: target.height * scaleY
        ).integral.intersection(
            CGRect(x: 0, y: 0, width: image.width, height: image.height))
        guard pixelRect.width >= 16, pixelRect.height >= 16 else { return nil }
        return image.cropping(to: pixelRect)
    }

    /// 渲染整页后裁剪指定区域。批量裁剪同一页时请先调用 `render`，
    /// 再复用 `crop(image:page:rect:)`，避免重复渲染。
    public static func crop(page: PDFPage, rect: CGRect,
                            dpi: Double = 300,
                            maxLongSide: Int = 4096) -> CGImage? {
        guard let full = render(page: page, dpi: dpi, maxLongSide: maxLongSide) else {
            return nil
        }
        return crop(image: full, page: page, rect: rect)
    }

    /// 粗略判断扫描页是否包含表格：在内容区域同时检测到足够长的横线与竖线。
    ///
    /// 只用于「混合引擎」决定该页走 Vision 还是 OvisOCR3：
    /// 误判为表格只会让该页改用 Ovis（慢一些但结果仍正确），
    /// 漏判则退化为 Vision 文本（内容保留、表格结构丢失）。
    public static func looksLikeTable(_ image: CGImage) -> Bool {
        let maxSide = 1000
        let scale = min(1.0, Double(maxSide) / Double(max(image.width, image.height)))
        let width = max(1, Int((Double(image.width) * scale).rounded()))
        let height = max(1, Int((Double(image.height) * scale).rounded()))
        var pixels = [UInt8](repeating: 255, count: width * height)
        guard let context = CGContext(data: &pixels,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: width,
                                      space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
            return false
        }
        context.setFillColor(gray: 1, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.interpolationQuality = .low
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        // 忽略扫描边缘/书脊的黑边，只在内容区域统计。
        let marginX = max(2, width / 25)
        let marginY = max(2, height / 25)
        let contentWidth = width - marginX * 2
        let contentHeight = height - marginY * 2
        guard contentWidth > 10, contentHeight > 10 else { return false }

        let dark: UInt8 = 160
        let minHorizontalRun = max(12, Int(Double(contentWidth) * 0.30))
        let minVerticalRun = max(12, Int(Double(contentHeight) * 0.12))

        func lineCount(horizontal: Bool, minRun: Int) -> Int {
            var hits = 0
            var previousHit = false
            if horizontal {
                for y in marginY..<(height - marginY) {
                    var run = 0
                    var best = 0
                    for x in marginX..<(width - marginX) {
                        if pixels[y * width + x] < dark {
                            run += 1
                            best = max(best, run)
                        } else {
                            run = 0
                        }
                    }
                    let hit = best >= minRun
                    if hit && !previousHit { hits += 1 }
                    previousHit = hit
                }
            } else {
                for x in marginX..<(width - marginX) {
                    var run = 0
                    var best = 0
                    for y in marginY..<(height - marginY) {
                        if pixels[y * width + x] < dark {
                            run += 1
                            best = max(best, run)
                        } else {
                            run = 0
                        }
                    }
                    let hit = best >= minRun
                    if hit && !previousHit { hits += 1 }
                    previousHit = hit
                }
            }
            return hits
        }

        let horizontalLines = lineCount(horizontal: true, minRun: minHorizontalRun)
        let verticalLines = lineCount(horizontal: false, minRun: minVerticalRun)
        return horizontalLines >= 2 && verticalLines >= 2
    }

    /// 近空白页检测：先缩到 512 长边缩略图（避免全尺寸重采样），采样 8192 个
    /// 像素，灰度极差 ≤ tolerance 视为空白。对应原项目 `_pixmap_is_blank` 的行为。
    public static func isNearlyBlank(_ image: CGImage,
                                     tolerance: Int = 12,
                                     maxSamples: Int = 8192) -> Bool {
        let longSide = max(image.width, image.height)
        guard longSide > 0 else { return true }
        let scale = min(1.0, Double(512) / Double(longSide))
        let width = max(1, Int((Double(image.width) * scale).rounded()))
        let height = max(1, Int((Double(image.height) * scale).rounded()))
        var pixels = [UInt8](repeating: 255, count: width * height)
        guard let context = CGContext(data: &pixels,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: width,
                                      space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
            return false
        }
        context.setFillColor(gray: 1, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.interpolationQuality = .low
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        let total = width * height
        let step = max(1, total / max(maxSamples, 1))
        var minimum = 255
        var maximum = 0
        var index = 0
        while index < total {
            let value = Int(pixels[index])
            minimum = min(minimum, value)
            maximum = max(maximum, value)
            index += step
        }
        return maximum - minimum <= tolerance
    }
}
