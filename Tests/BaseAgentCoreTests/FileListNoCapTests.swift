import XCTest
@testable import BaseAgentCore

/// file_tools(op="list") 默认全量输出，不再有内置条目数上限；
/// 只有调用方显式传 max_entries 才截断。
final class FileListNoCapTests: XCTestCase {
    private final class PlainLLM: LLMService {
        let identifier = "list-nocap"
        let supportsTools = false
        let contextWindow = 4096
        func chatStream(messages: [ChatMessage], tools: [JSONValue],
                        temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
            AsyncThrowingStream { c in c.yield(LLMStreamEvent(kind: .done)); c.finish() }
        }
    }

    private func makeTool() throws -> (FileListTool, URL) {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("list-nocap-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        // 造 600 个文件（超过旧默认上限 500、低于旧硬上限 5000）
        for i in 0..<600 {
            try Data("x".utf8).write(to: root.appendingPathComponent(String(format: "f%03d.txt", i)))
        }
        var config = AgentConfig()
        config.workspace = root.path
        return (FileListTool(config: config, callbacks: NoopCallbacks()), root)
    }

    /// 不传 max_entries：600 条全部返回，无截断提示。
    func testDefaultListsAllEntries() async throws {
        let (tool, root) = try makeTool()
        defer { try? FileManager.default.removeItem(at: root) }
        let result = try await tool.run(["path": .string(".")])
        XCTAssertTrue(result.success)
        let lines = result.output.split(separator: "\n").filter { $0.hasPrefix("file ") }
        XCTAssertEqual(lines.count, 600, "应返回全部 600 条，不再截断到 500")
        XCTAssertFalse(result.output.contains("已截断"))
    }

    /// 显式传 max_entries：按调用方意愿截断并给出提示。
    func testExplicitCapStillWorks() async throws {
        let (tool, root) = try makeTool()
        defer { try? FileManager.default.removeItem(at: root) }
        let result = try await tool.run([
            "path": .string("."), "max_entries": .int(10),
        ])
        XCTAssertTrue(result.success)
        let lines = result.output.split(separator: "\n").filter { $0.hasPrefix("file ") }
        XCTAssertEqual(lines.count, 10)
        XCTAssertTrue(result.output.contains("max_entries=10"), "应说明是调用方设定的上限")
    }

    /// schema 描述明确说明默认全量、上限可选。
    func testSchemaDescribesNoCap() throws {
        let (tool, root) = try makeTool()
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertTrue(tool.description.contains("ALL entries"))
        let maxParam = tool.parameters.objectValue?["properties"]?.objectValue?["max_entries"]
        XCTAssertNotNil(maxParam, "保留参数供调用方主动收窄")
        XCTAssertTrue(maxParam?.objectValue?["description"]?.stringValue?.contains("Omit for the complete list") ?? false)
    }
}

private final class NoopCallbacks: AgentCallbacks {}
