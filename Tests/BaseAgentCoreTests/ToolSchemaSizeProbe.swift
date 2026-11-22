import XCTest
@testable import BaseAgentCore

/// 测量真实工具 schema 的体积构成，供压缩前后对比（只打印，不断言具体值）。
final class ToolSchemaSizeProbe: XCTestCase {
    private final class PlainLLM: LLMService {
        let identifier = "schema-probe"
        let supportsTools = false
        let contextWindow = 4096
        func chatStream(messages: [ChatMessage], tools: [JSONValue],
                        temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
            AsyncThrowingStream { c in c.yield(LLMStreamEvent(kind: .done)); c.finish() }
        }
    }

    func testMeasureSchemaSize() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("schema-probe-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        var config = AgentConfig()
        config.workspace = root.path
        config.llmProvider = "api"
        let agent = Agent(llm: PlainLLM(), config: config, callbacks: NoopCallbacks())
        let schemas = agent.dispatcher.schemas()

        var totalChars = 0
        var totalDescChars = 0
        var lines: [String] = []
        for schema in schemas {
            let json = schema.jsonString()
            totalChars += json.count
            let name = schema.objectValue?["function"]?.objectValue?["name"]?.stringValue ?? "?"
            let desc = schema.objectValue?["function"]?.objectValue?["description"]?.stringValue ?? ""
            totalDescChars += desc.count
            lines.append("\(name): json=\(json.count) desc=\(desc.count)")
        }
        let report = "[schema-probe] tools=\(schemas.count) totalChars=\(totalChars) "
            + "descChars=\(totalDescChars) estimatedTokens=\(TokenEstimator.estimate(schemas.map { $0.jsonString() }.joined(separator: "\n")))\n"
            + lines.joined(separator: "\n") + "\n"
        FileHandle.standardError.write(Data(report.utf8))
        XCTAssertFalse(schemas.isEmpty)
    }
}

private final class NoopCallbacks: AgentCallbacks {}
