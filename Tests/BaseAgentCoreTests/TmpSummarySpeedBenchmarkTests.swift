import XCTest
import Foundation
@testable import BaseAgentCore

/// 临时基准（测完删除）：生产摘要提示下，批量生成 vs 并发 1 逐条串行。
/// 本组对应全书摘要 reduce 阶段：输入较短、输出 150-250 字。
final class TmpSummarySpeedBenchmarkTests: XCTestCase {

    func testBatchVsSerialSummarySpeed() async throws {
        let env = ProcessInfo.processInfo.environment
        let modelDir = env["BATCH_BENCH_MODEL_DIR"] ?? "models/Qwen3.5-2B-MLX-4bit"
        let service = MLXLLMService(
            directory: URL(fileURLWithPath: modelDir),
            modelName: "summary-speed-bench",
            maxTokens: 1_200,
            enableThinking: false,
            contextWindow: 8192,
            kvReuse: false,
            kvInt8: false)

        let requests = try makeRequests()
        // 预热（计时外）：加载容器 + 编译解码 kernel。
        _ = try await service.generateSummariesBatch(
            requests: [(system: "你是助手。", prompt: "回答：一加一等于几？")],
            temperature: 0.2, maxTokensPerRow: 8)

        // 串行路径：复刻 BookSummaryEngine 回退路径（isolated + 关闭 thinking）。
        let serialStart = Date()
        var serialOutputs: [String] = []
        for request in requests {
            let messages = [
                ChatMessage(role: .system, content: request.system),
                ChatMessage(role: .user, content: request.prompt),
            ]
            var output = ""
            for try await event in service.chatStream(
                messages: messages, tools: [], temperature: 0.2,
                options: LLMRequestOptions(enableThinking: false, isolated: true)) {
                if event.kind == .content { output += event.content }
            }
            serialOutputs.append(
                output.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        let serialElapsed = Date().timeIntervalSince(serialStart)

        // 批量路径：与 runBatchGeneration 相同参数（maxLength 300 → 1200 token）。
        let batchStart = Date()
        let batchOutputs = try await service.generateSummariesBatch(
            requests: requests, temperature: 0.2, maxTokensPerRow: 1_200)
        let batchElapsed = Date().timeIntervalSince(batchStart)

        XCTAssertEqual(serialOutputs.count, requests.count)
        let batch = try XCTUnwrap(batchOutputs)
        XCTAssertEqual(batch.count, requests.count)
        XCTAssertTrue(serialOutputs.allSatisfy { !$0.isEmpty },
                      "串行输出不应为空：\(serialOutputs)")
        XCTAssertTrue(batch.allSatisfy { !$0.isEmpty }, "批量输出不应为空：\(batch)")

        let speedup = serialElapsed / max(batchElapsed, 0.001)
        print("[summary-bench-reduce] 请求数=\(requests.count) 串行=\(serialElapsed)s "
            + "批量=\(batchElapsed)s 提速=\(speedup)x "
            + "串行字数=\(serialOutputs.map(\.count)) 批量字数=\(batch.map(\.count))")
    }

    /// 用真实章节文本切 5 个 ~800 字符摘要输入，套生产 reduce 提示。
    private func makeRequests() throws -> [(system: String, prompt: String)] {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let files = ["index_split_013.txt", "index_split_026.txt", "index_split_036.txt"]
        var groups: [String] = []
        for file in files {
            let url = root.appendingPathComponent(
                "benchmarks/book_summary/data/\(file)")
            let text = try String(contentsOf: url, encoding: .utf8)
            let pieces = splitByParagraphs(text, size: 800)
            groups.append(contentsOf: pieces.prefix(2))
        }
        guard groups.count >= 5 else { return [] }
        return groups.prefix(5).map { group in
            let prompt = """
            以下是本书部分章节的摘要。请合并为 150-250 字的内容梗概，保留主线、结构与关键结论，只输出正文。

            \(group)
            """
            return (system: "你是专业的书籍摘要助手，只输出正文。", prompt: prompt)
        }
    }

    /// 按空行段落贪心拼接近 `size` 字符的分片。
    private func splitByParagraphs(_ text: String, size: Int) -> [String] {
        var pieces: [String] = []
        var current = ""
        for paragraph in text.components(separatedBy: "\n\n") {
            let trimmed = paragraph.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let candidate = current.isEmpty ? trimmed : current + "\n\n" + trimmed
            if candidate.count > size, !current.isEmpty {
                pieces.append(current)
                current = trimmed
            } else {
                current = candidate
            }
        }
        if !current.isEmpty { pieces.append(current) }
        return pieces
    }
}
