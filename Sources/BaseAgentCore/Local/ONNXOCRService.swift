import Foundation
import CoreGraphics
import OnnxRuntimeBindings

/// 通用本地 ONNX OCR 服务（检测 + 识别 + 可选方向分类的可配置流水线）。
///
/// 设计目标：像标准 onnxruntime 服务一样，用户挂载常见 ONNX OCR 模型即可调用，
/// 而不局限于 rapidocr / PP-OCR。行为由 `OCRRecipe` 描述，来源有二：
/// 1. 模型目录里的 `ocr_config.json`（可选，覆盖/替换默认配方）；
/// 2. 缺省配方（PP-OCR / rapidocr 式：DB 检测 + CTC 识别 + 可选方向分类）。
///
/// 因此「常见其它 ONNX OCR 模型」（不同 PP-OCR 导出、DBNet+CRNN、SVTR-CTC、
/// 以及配置为自回归解码的注意力类识别头等）都能通过声明张量名 / 归一化 / 解码方式
/// 接入，无需改代码。未声明张量名时按 `inputNames/outputNames` 自动绑定。
///
/// 约定（与 rapidocr-onnxruntime 资源布局一致，可被 config 覆盖）：
/// - 检测 onnx 文件名含 `det`，识别含 `rec`，方向分类含 `cls`；
/// - 字符表 `ppocr_keys_v1.txt`（或 `keys.txt`），blank 默认视为末位索引。
public final class ONNXOCRService: VLMService, @unchecked Sendable {
    public let identifier: String
    public var isAvailable: Bool {
        stateLock.lock()
        let failure = loadFailure
        stateLock.unlock()
        return failure == nil
    }

    private let modelDirectory: URL
    private let modelName: String

    private let stateLock = NSLock()
    private var loaded: ONNXOCRLoaded?
    private var loadFailure: Error?

    public init(modelDirectory: URL, modelName: String) {
        self.modelDirectory = modelDirectory
        self.modelName = modelName
        self.identifier = "local:onnx-ocr:\(modelName)"
    }

    public func recognize(image: CGImage, prompt: String, maxTokens: Int) async throws -> String {
        _ = prompt
        _ = maxTokens
        let loaded = try await ensureLoaded()
        return try await Task.detached(priority: .userInitiated) {
            loaded.recognize(image: image)
        }.value
    }

    public func unload() async {
        stateLock.lock()
        loaded = nil
        loadFailure = nil
        stateLock.unlock()
    }

    /// 自动判别：目录含字符表 / `ocr_config.json`，或 ≥2 个 `.onnx`（检测+识别+可选分类）
    /// 即视为多阶段 ONNX OCR 包（而非单一合并的 VLM 图）。
    static func isONNXOCRDirectory(_ directory: URL) -> Bool {
        let fm = FileManager.default
        let markers = ["ppocr_keys_v1.txt", "ppocr_keys.txt", "keys.txt", "ocr_config.json"]
        let hasMarker = markers.contains {
            fm.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
        let onnxCount = (try? fm.contentsOfDirectory(atPath: directory.path))?
            .filter { $0.lowercased().hasSuffix(".onnx") }.count ?? 0
        return hasMarker || onnxCount >= 2
    }

    private func ensureLoaded() async throws -> ONNXOCRLoaded {
        stateLock.lock()
        if let loaded { stateLock.unlock(); return loaded }
        if let failure = loadFailure { stateLock.unlock(); throw failure }
        stateLock.unlock()
        do {
            let value = try ONNXOCRLoaded.load(directory: modelDirectory, modelName: modelName)
            stateLock.lock()
            loaded = value
            stateLock.unlock()
            return value
        } catch {
            stateLock.lock()
            loadFailure = error
            stateLock.unlock()
            throw error
        }
    }
}

// MARK: - 配方（Recipe）

/// OCR 配方：描述检测 / 识别 / 分类三阶段的文件、张量名、预处理与解码方式。
/// 可由模型目录的 `ocr_config.json` 完整覆盖；缺省为 PP-OCR / rapidocr 式。
public struct OCRRecipe: Decodable {
    /// 字符表文件名（相对模型目录）。
    public var charsFile: String = "ppocr_keys_v1.txt"
    public var detection: OCRDetectionConfig = .default
    public var recognition: OCRRecognitionConfig = .default
    public var classification: OCRClassificationConfig? = nil

    public static let `default` = OCRRecipe()
}

/// 检测阶段配置（DB 概率图 → 阈值 → 连通域 → 有向框 → unclip）。
public struct OCRDetectionConfig: Decodable {
    /// 文件名关键字（匹配首个含此串的 `.onnx`）。
    public var file: String = "det"
    /// 输入/输出张量名；nil 时自动取 `inputNames/outputNames` 首个。
    public var input: String? = nil
    public var output: String? = nil
    /// 输出张量中作为概率图的通道索引（DB 通常为 0）。
    public var probChannel: Int = 0
    public var mean: [Float] = [0.485, 0.456, 0.406]
    public var std: [Float] = [0.229, 0.224, 0.225]
    /// 长边上限（保持比例并取整到 32 的倍数）。
    public var targetLong: Int = 960
    public var boxThresh: Float = 0.6
    public var unclipRatio: Float = 1.6
    public var minBoxArea: Float = 4
    public var minSide: Float = 2
    /// 阅读顺序排序容忍（同一行 y 差值小于此值视为同行）。
    public var sortYTolerance: Double = 12

    public static let `default` = OCRDetectionConfig()
}

/// 识别阶段配置。
public struct OCRRecognitionConfig: Decodable {
    public var file: String = "rec"
    public var input: String? = nil
    public var output: String? = nil
    /// 自回归解码时图像输入张量名（缺省取非 token 输入）。
    public var imageInput: String? = nil
    /// 自回归解码时 token 序列输入张量名（缺省取第二个输入）。
    public var tokenInput: String? = nil
    /// 固定高度。
    public var height: Int = 48
    /// 动态宽度上限（按宽高比缩放后裁剪）。
    public var maxWidth: Int = 320
    /// 固定宽（非 nil 时忽略宽高比，直接缩放到此宽）。
    public var fixedWidth: Int? = nil
    public var mean: [Float] = [0.5, 0.5, 0.5]
    public var std: [Float] = [0.5, 0.5, 0.5]
    public var decode: OCRDecodeConfig = .default

    public static let `default` = OCRRecognitionConfig()
}

/// 识别解码方式：`ctc`（CRNN/SVTR 等贪心解码）或 `autoregressive`（注意力类模型）。
public struct OCRDecodeConfig: Decodable {
    /// "ctc" 或 "autoregressive"。
    public var mode: String = "ctc"
    /// CTC 的 blank 索引；nil 表示末位（chars.count）。
    public var blankIndex: Int? = nil
    /// 自回归：起始 token（SOS）。
    public var startToken: Int = 0
    /// 自回归：结束 token（EOS）；-1 表示不提前停止。
    public var eosToken: Int = -1
    public var maxSteps: Int = 48

    public static let `default` = OCRDecodeConfig()
}

/// 方向分类阶段配置（可选；目录存在 `cls` onnx 时自动启用默认）。
public struct OCRClassificationConfig: Decodable {
    public var file: String = "cls"
    public var input: String? = nil
    public var output: String? = nil
    public var height: Int = 48
    public var width: Int = 192
    public var mean: [Float] = [0.5, 0.5, 0.5]
    public var std: [Float] = [0.5, 0.5, 0.5]
    /// 输出索引中代表「需旋转 180°」的类别（PP-OCR 为 1）。
    public var rotatedClassIndex: Int = 1

    public static let `default` = OCRClassificationConfig()
}

// MARK: - 加载

private final class ONNXOCRLoaded: @unchecked Sendable {
    let detector: OrtSession
    let recognizer: OrtSession
    let classifier: OrtSession?
    let chars: [String]
    let recipe: OCRRecipe

    private init(detector: OrtSession, recognizer: OrtSession, classifier: OrtSession?,
                 chars: [String], recipe: OCRRecipe) {
        self.detector = detector
        self.recognizer = recognizer
        self.classifier = classifier
        self.chars = chars
        self.recipe = recipe
    }

    static func load(directory: URL, modelName: String) throws -> ONNXOCRLoaded {
        var recipe = OCRRecipe.loadIfPresent(directory: directory) ?? .default

        guard let detURL = Self.resolveComponent(directory: directory, keyword: recipe.detection.file) else {
            throw LocalModelError.invalidConfiguration(
                "ONNX OCR 模型目录缺少检测 onnx（文件名应含 '\(recipe.detection.file)'，目录：\(directory.path)）")
        }
        guard let recURL = Self.resolveComponent(directory: directory, keyword: recipe.recognition.file),
              !recURL.lastPathComponent.lowercased().contains(recipe.detection.file) else {
            throw LocalModelError.invalidConfiguration(
                "ONNX OCR 模型目录缺少识别 onnx（文件名应含 '\(recipe.recognition.file)' 且不含 '\(recipe.detection.file)'，目录：\(directory.path)）")
        }
        // 方向分类：配方显式给出则用；否则目录存在 cls onnx 时自动启用默认。
        var classification = recipe.classification
        if classification == nil,
           Self.resolveComponent(directory: directory, keyword: "cls") != nil {
            classification = .default
        }
        recipe.classification = classification

        let chars = Self.loadChars(directory: directory, fileName: recipe.charsFile)
        guard !chars.isEmpty else {
            throw LocalModelError.invalidConfiguration(
                "ONNX OCR 模型目录缺少字符表（期望 \(recipe.charsFile)，目录：\(directory.path)）")
        }

        let detector = try OrtSession(modelPath: detURL.path)
        let recognizer = try OrtSession(modelPath: recURL.path)
        let classifier = try classification.flatMap { cfg -> OrtSession? in
            guard let clsURL = Self.resolveComponent(directory: directory, keyword: cfg.file) else {
                return nil
            }
            return try OrtSession(modelPath: clsURL.path)
        }

        AgentLog.info("[onnx-ocr] 已加载 \(modelName)：det=\(detURL.lastPathComponent) "
            + "rec=\(recURL.lastPathComponent) cls=\(classifier == nil ? "无" : "有") "
            + "chars=\(chars.count) decode=\(recipe.recognition.decode.mode)")
        return ONNXOCRLoaded(detector: detector, recognizer: recognizer,
                             classifier: classifier, chars: chars, recipe: recipe)
    }

    private static func resolveComponent(directory: URL, keyword: String) -> URL? {
        guard let files = try? FileManager.default.contentsOfDirectory(atPath: directory.path) else {
            return nil
        }
        let matches = files.filter {
            let lower = $0.lowercased()
            return lower.hasSuffix(".onnx") && lower.contains(keyword.lowercased())
        }
        guard let first = matches.first else { return nil }
        return directory.appendingPathComponent(first)
    }

    private static func loadChars(directory: URL, fileName: String) -> [String] {
        let url = directory.appendingPathComponent(fileName)
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
            .map { String($0) }
        return lines.isEmpty ? [] : lines
    }

    // MARK: - 推理

    /// 整图 OCR：返回按阅读顺序（上→下、左→右）拼接的文本，每行一条。
    func recognize(image: CGImage) -> String {
        let boxes = detect(image: image)
        let ordered = boxes.sorted {
            if abs($0.center.y - $1.center.y) > recipe.detection.sortYTolerance {
                return $0.center.y < $1.center.y
            }
            return $0.center.x < $1.center.x
        }
        var lines: [String] = []
        for box in ordered {
            guard let region = cropRegion(image: image, corners: box.corners) else { continue }
            var work: CGImage = region
            if let classifier, let rotated = classify(image: region), rotated,
               let flipped = rotate180(region) {
                work = flipped
            }
            let text = recognizeRegion(image: work)
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { lines.append(trimmed) }
        }
        return lines.joined(separator: "\n")
    }

    // MARK: 检测

    private func detect(image: CGImage) -> [OCRBox] {
        let w0 = image.width, h0 = image.height
        let cfg = recipe.detection
        let target = detTargetSize(width: w0, height: h0, limit: cfg.targetLong)
        guard let pixels = try? VLMImagePreprocessor.pixelValues(
            from: image, size: CGSize(width: target.width, height: target.height),
            mean: cfg.mean, std: cfg.std) else { return [] }

        let inputName = cfg.input ?? detector.inputNames.first ?? "x"
        let outputName = cfg.output ?? detector.outputNames.first ?? "probability"
        let input = try? OrtValue(float32: pixels, shape: [1, 3, target.height, target.width])
        guard let input,
              let output = (try? detector.run(inputs: [inputName: input]))?[outputName],
              let tensor = try? output.floatTensor() else { return [] }

        // 取概率图通道（支持 [1,C,H,W] / [C,H,W] / [1,1,H,W]）。
        guard let prob = Self.extractChannel(values: tensor.values, shape: tensor.shape,
                                            channel: cfg.probChannel) else { return [] }
        let (hOut, wOut) = Self.spatialShape(tensor.shape)
        guard hOut > 0, wOut > 0 else { return [] }

        let map = Self.resizeBilinear(source: prob, sw: wOut, sh: hOut, dw: w0, dh: h0)

        var boxes: [OCRBox] = []
        let components = Self.connectedComponents(
            prob: map, width: w0, height: h0, threshold: cfg.boxThresh)
        for points in components {
            guard let box = Self.orientedBox(points: points, unclipRatio: cfg.unclipRatio,
                                            minArea: cfg.minBoxArea, minSide: cfg.minSide) else {
                continue
            }
            boxes.append(box)
        }
        return boxes
    }

    private func detTargetSize(width: Int, height: Int, limit: Int) -> (width: Int, height: Int) {
        let maxSide = max(width, height)
        let scale = maxSide > limit ? Double(limit) / Double(maxSide) : 1.0
        let tw = max(32, Int((Double(width) * scale).rounded(.down) / 32) * 32)
        let th = max(32, Int((Double(height) * scale).rounded(.down) / 32) * 32)
        return (tw, th)
    }

    /// 从 [..., C, H, W] 张量中取出第 `channel` 个通道的二维图（长度 H*W）。
    private static func extractChannel(values: [Float], shape: [Int], channel: Int) -> [Float]? {
        guard shape.count >= 3 else { return nil }
        let c = shape[shape.count - 3]
        let h = shape[shape.count - 2]
        let w = shape[shape.count - 1]
        guard c > 0, h > 0, w > 0, channel >= 0, channel < c else { return nil }
        var out = [Float](repeating: 0, count: h * w)
        for y in 0..<h {
            for x in 0..<w {
                out[y * w + x] = values[(channel * h + y) * w + x]
            }
        }
        return out
    }

    private static func spatialShape(_ shape: [Int]) -> (h: Int, w: Int) {
        guard shape.count >= 2 else { return (0, 0) }
        let h = shape[shape.count - 2]
        let w = shape[shape.count - 1]
        return (h, w)
    }

    private static func resizeBilinear(source: [Float], sw: Int, sh: Int,
                                       dw: Int, dh: Int) -> [Float] {
        guard sw > 0, sh > 0, dw > 0, dh > 0 else { return [] }
        var out = [Float](repeating: 0, count: dw * dh)
        for y in 0..<dh {
            let sy = (Double(y) + 0.5) * Double(sh) / Double(dh) - 0.5
            let y0 = max(0, min(sh - 1, Int(sy)))
            let y1 = min(sh - 1, y0 + 1)
            let wy1 = sy - Double(y0)
            let wy0 = 1 - wy1
            for x in 0..<dw {
                let sx = (Double(x) + 0.5) * Double(sw) / Double(dw) - 0.5
                let x0 = max(0, min(sw - 1, Int(sx)))
                let x1 = min(sw - 1, x0 + 1)
                let wx1 = sx - Double(x0)
                let wx0 = 1 - wx1
                let v = Float(Double(source[y0 * sw + x0]) * wy0 * wx0
                            + Double(source[y0 * sw + x1]) * wy0 * wx1
                            + Double(source[y1 * sw + x0]) * wy1 * wx0
                            + Double(source[y1 * sw + x1]) * wy1 * wx1)
                out[y * dw + x] = v
            }
        }
        return out
    }

    private static func connectedComponents(prob: [Float], width: Int, height: Int,
                                            threshold: Float) -> [[(x: Int, y: Int)]] {
        guard width > 0, height > 0 else { return [] }
        var visited = [Bool](repeating: false, count: width * height)
        var results: [[(x: Int, y: Int)]] = []
        let neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (1, -1), (-1, 1), (1, 1)]
        for start in 0..<(width * height) where !visited[start] {
            guard prob[start] > threshold else { visited[start] = true; continue }
            var component: [(x: Int, y: Int)] = []
            var stack = [start]
            visited[start] = true
            while !stack.isEmpty {
                let idx = stack.removeLast()
                let x = idx % width
                let y = idx / width
                component.append((x, y))
                for (dx, dy) in neighbors {
                    let nx = x + dx, ny = y + dy
                    guard nx >= 0, ny >= 0, nx < width, ny < height else { continue }
                    let nidx = ny * width + nx
                    if !visited[nidx], prob[nidx] > threshold {
                        visited[nidx] = true
                        stack.append(nidx)
                    }
                }
            }
            if component.count >= 3 { results.append(component) }
        }
        return results
    }

    private static func orientedBox(points: [(x: Int, y: Int)], unclipRatio: Float,
                                    minArea: Float, minSide: Float) -> OCRBox? {
        let n = points.count
        guard n >= 3 else { return nil }
        var cx = 0.0, cy = 0.0
        for p in points { cx += Double(p.x); cy += Double(p.y) }
        cx /= Double(n); cy /= Double(n)
        var sxx = 0.0, sxy = 0.0, syy = 0.0
        for p in points {
            let dx = Double(p.x) - cx, dy = Double(p.y) - cy
            sxx += dx * dx; sxy += dx * dy; syy += dy * dy
        }
        let angle = 0.5 * atan2(2 * sxy, sxx - syy)
        let ca = cos(-angle), sa = sin(-angle)
        var minX = Double.infinity, maxX = -Double.infinity
        var minY = Double.infinity, maxY = -Double.infinity
        for p in points {
            let dx = Double(p.x) - cx, dy = Double(p.y) - cy
            let rx = dx * ca - dy * sa
            let ry = dx * sa + dy * ca
            minX = min(minX, rx); maxX = max(maxX, rx)
            minY = min(minY, ry); maxY = max(maxY, ry)
        }
        let width = maxX - minX
        let height = maxY - minY
        guard width >= Double(minSide), height >= Double(minSide),
              Float(width * height) >= minArea else { return nil }

        let rca = cos(angle), rsa = sin(angle)
        let unclipped: [(Double, Double)] = [
            (minX * Double(unclipRatio), minY * Double(unclipRatio)),
            (maxX * Double(unclipRatio), minY * Double(unclipRatio)),
            (maxX * Double(unclipRatio), maxY * Double(unclipRatio)),
            (minX * Double(unclipRatio), maxY * Double(unclipRatio)),
        ]
        let corners: [CGPoint] = unclipped.map { (rx, ry) in
            let ox = rx * rca - ry * rsa + cx
            let oy = rx * rsa + ry * rca + cy
            return CGPoint(x: ox, y: oy)
        }
        return OCRBox(corners: corners)
    }

    // MARK: 方向分类

    private func classify(image: CGImage) -> Bool? {
        guard let classifier, let cfg = recipe.classification else { return nil }
        let w = image.width, h = image.height
        let scale = CGFloat(cfg.height) / CGFloat(h)
        let cw = max(cfg.height, Int(CGFloat(w) * scale))
        let cwClamped = min(cw, cfg.width)
        guard let pixels = try? VLMImagePreprocessor.pixelValues(
            from: image, size: CGSize(width: cwClamped, height: cfg.height),
            mean: cfg.mean, std: cfg.std) else { return nil }
        let inputName = cfg.input ?? classifier.inputNames.first ?? "x"
        let outputName = cfg.output ?? classifier.outputNames.first ?? "out"
        guard let value = try? OrtValue(
            float32: pixels, shape: [1, 3, cfg.height, cwClamped]) else { return nil }
        guard let output = (try? classifier.run(inputs: [inputName: value]))?[outputName],
              let tensor = try? output.floatTensor() else { return nil }
        guard tensor.values.count >= 2 else { return nil }
        return tensor.values[cfg.rotatedClassIndex] > tensor.values[1 - cfg.rotatedClassIndex]
    }

    // MARK: 识别

    private func recognizeRegion(image: CGImage) -> String {
        let cfg = recipe.recognition
        let w = image.width, h = image.height
        let scale = CGFloat(cfg.height) / CGFloat(h)
        let rw: Int
        if let fixed = cfg.fixedWidth {
            rw = fixed
        } else {
            rw = max(cfg.height, min(Int(CGFloat(w) * scale), cfg.maxWidth))
        }
        guard let pixels = try? VLMImagePreprocessor.pixelValues(
            from: image, size: CGSize(width: rw, height: cfg.height),
            mean: cfg.mean, std: cfg.std) else { return "" }

        let inputName = cfg.input ?? recognizer.inputNames.first ?? "x"
        let outputName = cfg.output ?? recognizer.outputNames.first ?? "out"
        guard let value = try? OrtValue(
            float32: pixels, shape: [1, 3, cfg.height, rw]) else { return "" }

        if cfg.decode.mode == "autoregressive" {
            return autoregressiveDecode(imageValue: value, imageInput: cfg.imageInput ?? inputName,
                                       tokenInput: cfg.tokenInput, outputName: outputName,
                                       decode: cfg.decode)
        }

        guard let output = (try? recognizer.run(inputs: [inputName: value]))?[outputName],
              let tensor = try? output.floatTensor() else { return "" }
        let shape = tensor.shape
        let (timesteps, vocab): (Int, Int)
        if shape.count == 3 { (timesteps, vocab) = (shape[1], shape[2]) }
        else if shape.count == 2 { (timesteps, vocab) = (shape[0], shape[1]) }
        else { return "" }
        guard timesteps > 0, vocab > 0 else { return "" }
        let blank = cfg.decode.blankIndex ?? chars.count
        return Self.ctcDecode(logits: tensor.values, timesteps: timesteps, vocab: vocab,
                              chars: chars, blank: blank)
    }

    /// 自回归（注意力类）识别：图像固定输入，循环喂入 token 序列取下一 token，直到 EOS。
    /// 适用于 TRBA / ABINet / PARSeq 等导出的「image + input_ids → logits」图，
    /// 具体张量名与起止 token 由配方指定。
    private func autoregressiveDecode(imageValue: OrtValue, imageInput: String,
                                      tokenInput: String?, outputName: String,
                                      decode: OCRDecodeConfig) -> String {
        let tokenName = tokenInput
            ?? (recognizer.inputNames.count > 1
                ? recognizer.inputNames.first(where: { $0 != imageInput }) ?? recognizer.inputNames[1]
                : nil)
        guard let tokenName else { return "" }

        var ids: [Int64] = [Int64(decode.startToken)]
        var result = ""
        for _ in 0..<max(1, decode.maxSteps) {
            guard let tokenTensor = try? OrtValue(int64: ids, shape: [1, ids.count]) else { break }
            guard let output = (try? recognizer.run(inputs: [
                imageInput: imageValue, tokenName: tokenTensor
            ]))?[outputName],
                  let tensor = try? output.floatTensor() else { break }

            let shape = tensor.shape
            let (seq, vocab): (Int, Int)
            if shape.count == 3 { (seq, vocab) = (shape[1], shape[2]) }
            else if shape.count == 2 { (seq, vocab) = (shape[0], shape[1]) }
            else { return result }
            guard seq > 0, vocab > 0 else { return result }

            // 取最后一个时间步的 argmax。
            let offset = (seq - 1) * vocab
            var best = 0, bestValue = tensor.values[offset]
            for c in 1..<vocab {
                let v = tensor.values[offset + c]
                if v > bestValue { bestValue = v; best = c }
            }
            if decode.eosToken >= 0, best == decode.eosToken { break }
            if best < chars.count { result += chars[best] }
            ids.append(Int64(best))
        }
        return result
    }

    private static func ctcDecode(logits: [Float], timesteps: Int, vocab: Int,
                                  chars: [String], blank: Int) -> String {
        var result = ""
        var last = -1
        for t in 0..<timesteps {
            let base = t * vocab
            var best = 0, bestValue = logits[base]
            for c in 1..<vocab {
                let v = logits[base + c]
                if v > bestValue { bestValue = v; best = c }
            }
            if best == blank { last = -1; continue }
            if best == last { continue }
            if best < chars.count { result += chars[best] }
            last = best
        }
        return result
    }

    // MARK: 图像裁剪（旋转框 → 正向矩形）

    private func cropRegion(image: CGImage, corners: [CGPoint]) -> CGImage? {
        guard corners.count == 4 else { return nil }
        let c0 = corners[0], c1 = corners[1], c2 = corners[2], c3 = corners[3]
        let w = hypot(c1.x - c0.x, c1.y - c0.y)
        let h = hypot(c2.x - c1.x, c2.y - c1.y)
        let width = max(w, h)
        let height = min(w, h)
        let cx = (c0.x + c1.x + c2.x + c3.x) / 4
        let cy = (c0.y + c1.y + c2.y + c3.y) / 4
        let angle = atan2(c1.y - c0.y, c1.x - c0.x)
        let ctxW = Int(ceil(width)), ctxH = Int(ceil(height))
        guard ctxW > 0, ctxH > 0 else { return nil }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: ctxW, height: ctxH, bitsPerComponent: 8,
            bytesPerRow: ctxW * 4, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        var t = CGAffineTransform.identity
        t = t.translatedBy(x: CGFloat(ctxW) / 2, y: CGFloat(ctxH) / 2)
        t = t.rotated(by: -angle)
        t = t.translatedBy(x: -cx, y: -cy)
        ctx.concatenate(t)
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return ctx.makeImage()
    }

    private func rotate180(_ image: CGImage) -> CGImage? {
        let w = image.width, h = image.height
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: w * 4, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.translateBy(x: CGFloat(w), y: CGFloat(h))
        ctx.rotate(by: .pi)
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()
    }
}

// MARK: - 配方加载（可选 ocr_config.json）

extension OCRRecipe {
    /// 读取模型目录的 `ocr_config.json`；不存在或解析失败则返回 nil。
    static func loadIfPresent(directory: URL) -> OCRRecipe? {
        let url = directory.appendingPathComponent("ocr_config.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(OCRRecipe.self, from: data)
    }
}

// MARK: - 有向框

private struct OCRBox {
    let corners: [CGPoint]
    let center: CGPoint

    init(corners: [CGPoint]) {
        self.corners = corners
        let sx = corners.reduce(0) { $0 + $1.x }
        let sy = corners.reduce(0) { $0 + $1.y }
        let n = Double(corners.count)
        self.center = CGPoint(x: sx / n, y: sy / n)
    }
}
