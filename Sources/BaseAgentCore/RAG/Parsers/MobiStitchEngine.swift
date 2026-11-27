import Foundation
import CoreGraphics
import ImageIO

/// Kindle 扫描书常见「一张整页被切成多张小图」的排版：HTML 里表现为
/// 密集 `<table>` 只含图片，或连续 `<img>` 之间没有换行。直接逐张 OCR
/// 会把一句话拦腰截断，所以先在 Swift 层用 CoreGraphics 拼回整页。
public enum MobiStitchEngine {

    /// 单张拼图的边长上限；超过时整体等比缩小，避免 CGContext 分配超大画布。
    public static let maxCanvasEdge = 8192

    /// 从 HTML 判断是否需要拼图。
    /// - 密集表格：`<table>` 内 `<img>` 数 ≥ minImages 且几乎没有文本；
    /// - 连续图片：去掉标签与空白后，连续 ≥ minImages 个 `<img>` 之间无换行/文本。
    public static func shouldStitch(html: String, minImages: Int = 2) -> Bool {
        let ns = html as NSString
        let options: NSRegularExpression.Options = [.caseInsensitive, .dotMatchesLineSeparators]
        guard let tableRegex = try? NSRegularExpression(
                pattern: #"<table\b[^>]*>(.*?)</table\s*>"#, options: options),
              let imgRegex = try? NSRegularExpression(pattern: #"<img\b[^>]*>"#,
                                                      options: .caseInsensitive),
              let tagRegex = try? NSRegularExpression(pattern: #"<[^>]+>"#, options: []) else {
            return false
        }
        let whole = NSRange(location: 0, length: ns.length)
        // 表格拼图：<table> 内图片 ≥ minImages 且去掉标签后几乎没有文字。
        for match in tableRegex.matches(in: html, options: [], range: whole) {
            guard match.numberOfRanges > 1 else { continue }
            let body = ns.substring(with: match.range(at: 1))
            let bodyNS = body as NSString
            let imageCount = imgRegex.numberOfMatches(
                in: body, options: [], range: NSRange(location: 0, length: bodyNS.length))
            guard imageCount >= minImages else { continue }
            let stripped = tagRegex.stringByReplacingMatches(
                in: body, options: [], range: NSRange(location: 0, length: bodyNS.length),
                withTemplate: "")
            if stripped.trimmingCharacters(in: .whitespacesAndNewlines).utf8.count
                < imageCount * 8 { return true }
        }
        // 连续图片：相邻 <img> 标签之间只有空白。
        let tags = imgRegex.matches(in: html, options: [], range: whole)
        var run = 0
        for index in tags.indices {
            if index == 0 { run = 1; continue }
            let gapStart = tags[index - 1].range.upperBound
            let gapEnd = tags[index].range.lowerBound
            guard gapEnd >= gapStart else { run = 1; continue }
            let gap = ns.substring(with: NSRange(location: gapStart, length: gapEnd - gapStart))
            run = gap.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? run + 1 : 1
            if run >= minImages { return true }
        }
        return false
    }

    /// 将一组小图拼成一张大图。布局策略：
    /// - 细长图（宽 < 高 × 0.6）按 2 列排布，模拟扫描页双栏切片；
    /// - 其余按单列纵向堆叠。
    /// 返回 nil 表示没有可拼接的图像。
    public static func stitch(_ images: [CGImage], columns: Int? = nil) -> CGImage? {
        let valid = images.filter { $0.width > 0 && $0.height > 0 }
        guard !valid.isEmpty else { return nil }
        // width*5 / height*3 用普通 Int 乘法会溢出 trap；转为 Int64 比较。
        let narrow = valid.allSatisfy {
            Int64($0.width) * 5 < Int64($0.height) * 3
        }
        // 奇数张竖向切片用双列会把 0/2 排到左列、1 单独在右列，阅读顺序错乱；
        // 只有偶数张时才按双栏排布，其余保持单列纵向顺序。
        // columns 是 public 参数，外部可传 0/负数，必须钳制避免 % 0 trap。
        let requested = columns ?? ((narrow && valid.count % 2 == 0 && valid.count > 1) ? 2 : 1)
        let columnCount = max(1, requested)
        let widths = valid.map { CGFloat($0.width) }
        let heights = valid.map { CGFloat($0.height) }
        let columnWidth = (0..<columnCount).map { column in
            widths.enumerated().filter { $0.offset % columnCount == column }
                .map(\.element).max() ?? 0
        }
        let columnHeights = Array(repeating: CGFloat(0), count: columnCount)
        var heightsCopy = columnHeights
        for (index, height) in heights.enumerated() {
            heightsCopy[index % columnCount] += height
        }
        var canvasWidth = columnWidth.reduce(0, +)
        var canvasHeight = heightsCopy.max() ?? 0
        guard canvasWidth > 0, canvasHeight > 0 else { return nil }
        let scale = min(1, CGFloat(maxCanvasEdge) / canvasWidth,
                        CGFloat(maxCanvasEdge) / canvasHeight)
        canvasWidth = floor(canvasWidth * scale)
        canvasHeight = floor(canvasHeight * scale)
        guard canvasWidth > 0, canvasHeight > 0 else { return nil }

        guard let context = CGContext(data: nil,
                                      width: Int(canvasWidth),
                                      height: Int(canvasHeight),
                                      bitsPerComponent: 8,
                                      bytesPerRow: 0,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        context.setFillColor(CGColor(colorSpace: CGColorSpaceCreateDeviceGray(),
                                     components: [1, 1]) ?? CGColor(red: 1, green: 1,
                                                                    blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: canvasWidth, height: canvasHeight))
        var offsets = Array(repeating: CGFloat(0), count: columnCount)
        for (index, image) in valid.enumerated() {
            let column = index % columnCount
            let width = widths[index] * scale
            let height = heights[index] * scale
            let rect = CGRect(x: (0..<column).reduce(0) { $0 + columnWidth[$1] * scale },
                              y: offsets[column],
                              width: width, height: height)
            context.draw(image, in: rect)
            offsets[column] += height
        }
        return context.makeImage()
    }

    /// 按 indentify 出的图片顺序分组：需要拼接的相邻小图归为一组，其余单图成组。
    public static func groups(imageCount: Int, stitchRanges: [Range<Int>]) -> [Range<Int>] {
        guard !stitchRanges.isEmpty else {
            return (0..<imageCount).map { $0..<($0 + 1) }
        }
        var groups: [Range<Int>] = []
        var cursor = 0
        for range in stitchRanges.sorted(by: { $0.lowerBound < $1.lowerBound }) {
            let lower = max(cursor, range.lowerBound)
            let upper = min(imageCount, range.upperBound)
            if lower > cursor { groups.append(cursor..<lower) }
            if upper > lower { groups.append(lower..<upper) }
            cursor = max(cursor, upper)
        }
        if cursor < imageCount { groups.append(cursor..<imageCount) }
        return groups
    }
}
