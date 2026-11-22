import XCTest
@testable import BaseAgentCore

/// 精确拆解工具 schema 的字符构成：结构开销 / description / 参数名 / 参数描述 / 枚举。
final class ToolSchemaBreakdownProbe: XCTestCase {
    private final class PlainLLM: LLMService {
        let identifier = "schema-breakdown"
        let supportsTools = false
        let contextWindow = 4096
        func chatStream(messages: [ChatMessage], tools: [JSONValue],
                        temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
            AsyncThrowingStream { c in c.yield(LLMStreamEvent(kind: .done)); c.finish() }
        }
    }

    /// 递归统计 JSON 里所有 string 值的字符数，按 key 归类。
    private func walk(_ value: JSONValue, key: String, into counts: inout [String: Int],
                      strings: inout [String]) {
        switch value {
        case .string(let s):
            counts[key, default: 0] += s.count
            if s.count > 40 { strings.append("[\(key)] \(s.prefix(100))") }
        case .array(let a):
            for item in a { walk(item, key: key, into: &counts, strings: &strings) }
        case .object(let o):
            for (k, v) in o { walk(v, key: k, into: &counts, strings: &strings) }
        default: break
        }
    }

    func testBreakdown() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("schema-breakdown-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        var config = AgentConfig()
        config.workspace = root.path
        config.llmProvider = "api"
        let agent = Agent(llm: PlainLLM(), config: config, callbacks: NoopCallbacks())
        let schemas = agent.dispatcher.schemas()

        var totalJSON = 0
        var counts: [String: Int] = [:]
        var longStrings: [String] = []
        var perTool: [String] = []
        for schema in schemas {
            let json = schema.jsonString()
            totalJSON += json.count
            let name = schema.objectValue?["function"]?.objectValue?["name"]?.stringValue ?? "?"
            perTool.append("\(name)=\(json.count)")
            walk(schema, key: "root", into: &counts, strings: &longStrings)
        }
        // 结构开销 = JSON 总长 - 所有字符串值字符数（含 key 名、标点、大括号）
        let allStrings = counts.values.reduce(0, +)
        var report = "[schema-breakdown] tools=\(schemas.count) totalJSON=\(totalJSON) "
            + "stringValues=\(allStrings) structural=\(totalJSON - allStrings)\n"
        report += "byKey: " + counts.sorted { $0.value > $1.value }
            .map { "\($0.key)=\($0.value)" }.joined(separator: ", ") + "\n"
        report += "perTool: " + perTool.joined(separator: ", ") + "\n"
        report += "longStrings(\(longStrings.count)):\n" + longStrings.prefix(25).joined(separator: "\n")
        FileHandle.standardError.write(Data((report + "\n").utf8))
        XCTAssertFalse(schemas.isEmpty)
    }
}

private final class NoopCallbacks: AgentCallbacks {}
