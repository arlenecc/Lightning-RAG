import XCTest
@testable import BaseAgentCore

/// file_tools(op="list") 分批返回：默认第 1 批 100 条，报告总数与批次数，
/// 可指定 batch 逐批获取全量，避免多文件目录一次撑爆上下文。
final class FileListNoCapTests: XCTestCase {
    private final class PlainLLM: LLMService {
        let identifier = "list-batch"
        let supportsTools = false
        let contextWindow = 4096
        func chatStream(messages: [ChatMessage], tools: [JSONValue],
                        temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
            AsyncThrowingStream { c in c.yield(LLMStreamEvent(kind: .done)); c.finish() }
        }
    }

    private func makeTool(count: Int) throws -> (FileListTool, URL) {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("list-batch-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for i in 0..<count {
            try Data("x".utf8).write(to: root.appendingPathComponent(String(format: "f%03d.txt", i)))
        }
        var config = AgentConfig()
        config.workspace = root.path
        return (FileListTool(config: config, callbacks: NoopCallbacks()), root)
    }

    /// 600 个文件：默认返回第 1 批 100 条，报告 600 条 / 6 批。
    func testDefaultReturnsFirstBatchWithStats() async throws {
        let (tool, root) = try makeTool(count: 600)
        defer { try? FileManager.default.removeItem(at: root) }
        let result = try await tool.run(["path": .string(".")])
        XCTAssertTrue(result.success)
        XCTAssertTrue(result.output.contains("共 600 条，分 6 批（每批 100 条）"))
        XCTAssertTrue(result.output.contains("当前第 1/6 批"))
        let lines = result.output.split(separator: "\n").filter { $0.hasPrefix("file ") }
        XCTAssertEqual(lines.count, 100, "默认只返回 100 条")
        XCTAssertFalse(result.output.contains("f100.txt"), "不应包含第 2 批的文件")
    }

    /// 逐批请求可拿全量：6 批共 600 条，文件互不重复。
    func testAllBatchesCoverEveryFile() async throws {
        let (tool, root) = try makeTool(count: 250)
        defer { try? FileManager.default.removeItem(at: root) }
        var collected: Set<String> = []
        for batch in 1...3 {
            let result = try await tool.run([
                "path": .string("."), "batch": .int(batch),
            ])
            XCTAssertTrue(result.output.contains("共 250 条，分 3 批"))
            XCTAssertTrue(result.output.contains("当前第 \(batch)/3 批"))
            let names = result.output.split(separator: "\n")
                .filter { $0.hasPrefix("file ") }
                .compactMap { $0.split(separator: " ").last.map(String.init) }
            collected.formUnion(names)
        }
        XCTAssertEqual(collected.count, 250, "3 批应覆盖全部 250 个文件")
    }

    /// 超出范围的批次返回空内容与明确提示。
    func testOutOfRangeBatch() async throws {
        let (tool, root) = try makeTool(count: 5)
        defer { try? FileManager.default.removeItem(at: root) }
        let result = try await tool.run([
            "path": .string("."), "batch": .int(9),
        ])
        XCTAssertTrue(result.output.contains("当前第 9/1 批（超出范围，本批为空）"))
        XCTAssertFalse(result.output.contains("file f"))
    }

    /// schema 描述明确分批语义（batch 参数存在、max_entries 已移除）。
    func testSchemaDescribesBatching() throws {
        let (tool, root) = try makeTool(count: 1)
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertTrue(tool.description.contains("batches of 100"))
        let props = tool.parameters.objectValue?["properties"]?.objectValue
        XCTAssertNotNil(props?["batch"])
        XCTAssertNil(props?["max_entries"], "max_entries 应已被 batch 取代")
    }
}

private final class NoopCallbacks: AgentCallbacks {}
