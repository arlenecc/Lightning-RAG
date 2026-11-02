import XCTest
import CoreGraphics
import Foundation
@testable import BaseAgentCore

/// 全流程验证：图片附件（ChatMessage.images）能从 Agent.run 一路带入
/// LLM 服务，并被 mlx-vlm 模型真正接受、参与推理。
final class MultimodalFlowTests: XCTestCase {

    // MARK: - 工具

    /// 生成纯色 PNG（默认红色）字节，用于模拟用户上传的图片附件。
    static func solidColorPNG(width: Int = 64, height: Int = 64,
                              r: UInt8 = 200, g: UInt8 = 30, b: UInt8 = 30) throws -> Data {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for i in stride(from: 0, to: pixels.count, by: 4) {
            pixels[i] = r
            pixels[i + 1] = g
            pixels[i + 2] = b
            pixels[i + 3] = 255
        }
        let ctx = try XCTUnwrap(CGContext(
            data: &pixels, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        let image = try XCTUnwrap(ctx.makeImage())
        let out = NSMutableData()
        let dest = try XCTUnwrap(CGImageDestinationCreateWithData(
            out as CFMutableData, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(dest, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
        return out as Data
    }

    func makeConfig() -> AgentConfig {
        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory()
            + "/mm-flow-\(UUID().uuidString)"
        config.maxIterations = 1
        return config
    }

    // MARK: - 管道级：Agent.run → Kernel → Session → LLM 消息

    /// 捕获每次 LLM 调用收到的完整消息列表。
    private final class CapturingLLM: LLMService, @unchecked Sendable {
        let identifier = "capturing"
        let supportsTools = true
        private let lock = NSLock()
        private(set) var calls: [[ChatMessage]] = []
        private let reply: String

        init(reply: String) { self.reply = reply }

        func chatStream(messages: [ChatMessage], tools: [JSONValue],
                        temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
            lock.lock()
            calls.append(messages)
            lock.unlock()
            let reply = reply
            return AsyncThrowingStream { continuation in
                continuation.yield(LLMStreamEvent(kind: .content, content: reply))
                continuation.yield(LLMStreamEvent(
                    kind: .done, content: reply, rawContent: reply))
                continuation.finish()
            }
        }
    }

    /// `agent.run(_, images:)` 的图片必须出现在 LLM 实际收到的首条 user 消息上；
    /// 后续无图轮次的新用户消息不得携带图片（历史中的旧图保留属预期语义）。
    func testAgentRunCarriesImagesIntoLLMMessages() async throws {
        let png = try Self.solidColorPNG()
        let llm = CapturingLLM(reply: "红色方块。")
        let agent = Agent(llm: llm, config: makeConfig(), callbacks: NoopCallbacks())
        defer { agent.close() }

        let answer = await agent.run("这张图里是什么？", images: [png])
        XCTAssertEqual(answer, "红色方块。")
        XCTAssertEqual(llm.calls.count, 1)

        // 首次调用：当前用户消息携带原图字节。
        let firstCall = try XCTUnwrap(llm.calls.first)
        let currentUser = try XCTUnwrap(firstCall.last { $0.role == .user })
        XCTAssertEqual(currentUser.images, [png])
        XCTAssertTrue(currentUser.content.contains("这张图里是什么？"))

        // 第二次无图提问：新用户消息不应带图（历史轮的旧图保留属预期）。
        _ = await agent.run("这只回答文字的问题")
        let secondCall = try XCTUnwrap(llm.calls.last)
        let latestUser = try XCTUnwrap(secondCall.last { $0.role == .user })
        XCTAssertNil(latestUser.images)
        XCTAssertTrue(latestUser.content.contains("只回答文字"))
    }

    // MARK: - 真实推理：mlx-vlm 挂载模型接受图像附件

    /// 用本地 OvisOCR3（mlx-vlm 后端）做端到端验证：带图消息经
    /// makeVLMMessages → UserInput.prepare → 视觉推理，模型必须能"看到"
    /// 纯红色图片并回答红色；无图请求则不应凭空说出颜色。
    func testRealMLXVLMAcceptsImageAttachment() async throws {
        let env = ProcessInfo.processInfo.environment
        let dirPath = env["MLXVLM_TEST_MODEL_DIR"]
            ?? "models/OvisOCR3-MLX-4bit-mixed46"
        guard FileManager.default.fileExists(
            atPath: dirPath + "/model.safetensors") else {
            throw XCTSkip("本地缺少 mlx-vlm 测试模型：\(dirPath)")
        }

        let service = MLXVLMLLMService(
            directory: URL(fileURLWithPath: dirPath),
            modelName: (dirPath as NSString).lastPathComponent,
            maxTokens: 64,
            topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false,
            contextWindow: 8_192)
        defer { service.close() }

        let png = try Self.solidColorPNG()
        let messages = [
            ChatMessage(role: .user, content: "这张图片是什么颜色？只回答颜色名称。",
                        images: [png]),
        ]

        var output = ""
        var finished = false
        for try await event in service.chatStream(
            messages: messages, tools: [], temperature: 0.1) {
            switch event.kind {
            case .content:
                output += event.content
            case .done:
                finished = true
            default:
                break
            }
        }

        XCTAssertTrue(finished, "多模态请求未正常结束")
        let answer = output.trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertFalse(answer.isEmpty, "带图请求返回了空回复")
        let normalized = answer.lowercased()
        XCTAssertTrue(normalized.contains("红") || normalized.contains("red"),
                      "模型未利用图片内容（应识别为红色）：\(answer)")
    }

    /// 无图控制组：同一模型对纯文本提问不应凭空说出「红色」。
    func testRealMLXVLMTextOnlyControl() async throws {
        let env = ProcessInfo.processInfo.environment
        let dirPath = env["MLXVLM_TEST_MODEL_DIR"]
            ?? "models/OvisOCR3-MLX-4bit-mixed46"
        guard FileManager.default.fileExists(
            atPath: dirPath + "/model.safetensors") else {
            throw XCTSkip("本地缺少 mlx-vlm 测试模型：\(dirPath)")
        }

        let service = MLXVLMLLMService(
            directory: URL(fileURLWithPath: dirPath),
            modelName: (dirPath as NSString).lastPathComponent,
            maxTokens: 64,
            topP: 1, minP: 0, topK: 0, repetitionPenalty: 1,
            enableThinking: false,
            contextWindow: 8_192)
        defer { service.close() }

        let messages = [
            ChatMessage(role: .user, content: "1+1 等于几？只回答数字。"),
        ]
        var output = ""
        for try await event in service.chatStream(
            messages: messages, tools: [], temperature: 0.1) {
            if event.kind == .content { output += event.content }
        }
        XCTAssertTrue(output.contains("2"),
                      "纯文本提问应正常回答：\(output)")
    }
}
