import XCTest
@testable import BaseAgentCore

/// 基础工具增强回归：file_list/file_search（新）、file_read 分页与编码、
/// file_modify 替换次数、shell_run shell/env 参数、RSS feed 解析。
final class ToolEnhancementTests: XCTestCase {
    private var root: String!

    override func setUp() {
        super.setUp()
        root = NSTemporaryDirectory() + "tool-enhance-\(UUID().uuidString.prefix(6))"
        try? FileManager.default.createDirectory(atPath: root, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(atPath: root)
        root = nil
        super.tearDown()
    }

    private var config: AgentConfig {
        var config = AgentConfig()
        config.workspace = root
        return config
    }

    private func write(_ path: String, _ content: String, encoding: String.Encoding = .utf8) throws {
        let full = root + "/" + path
        try FileManager.default.createDirectory(
            atPath: (full as NSString).deletingLastPathComponent,
            withIntermediateDirectories: true)
        try content.data(using: encoding)!.write(to: URL(fileURLWithPath: full))
    }

    // MARK: file_list

    func testFileListBasicAndHiddenSkip() async throws {
        try write("src/main.swift", "let a = 1")
        try write("src/.hidden.txt", "secret")
        try FileManager.default.createDirectory(atPath: root + "/src/build", withIntermediateDirectories: true)

        let result = try await FileListTool(config: config, callbacks: nil)
            .run(["path": .string(".")])
        XCTAssertTrue(result.success, "\(result.error)")
        XCTAssertTrue(result.output.contains("src/"), "应列出目录：\(result.output)")
        // 非递归时不展开子目录内容
        XCTAssertFalse(result.output.contains("main.swift"))
        // 隐藏条目在递归时也跳过
        let recursive = try await FileListTool(config: config, callbacks: nil)
            .run(["path": .string("."), "recursive": .string("true")])
        XCTAssertTrue(recursive.output.contains("main.swift"))
        XCTAssertFalse(recursive.output.contains(".hidden.txt"))
    }

    func testFileListGlobFilter() async throws {
        try write("a/one.swift", "x")
        try write("a/two.md", "y")
        let result = try await FileListTool(config: config, callbacks: nil)
            .run(["path": .string("a"), "glob": .string("*.swift")])
        XCTAssertTrue(result.success, "\(result.error)")
        XCTAssertTrue(result.output.contains("one.swift"))
        XCTAssertFalse(result.output.contains("two.md"))
    }

    func testFileListOutOfRangeRejected() async throws {
        let result = try await FileListTool(config: config, callbacks: nil)
            .run(["path": .string("..")])
        XCTAssertFalse(result.success)
    }

    // MARK: file_search

    func testFileSearchRegexAndLineNumbers() async throws {
        try write("code/demo.py", "def hello():\n    return 'world'\n\nx = 42\n")
        let result = try await FileSearchTool(config: config, callbacks: nil)
            .run(["pattern": .string("def \\w+"), "path": .string(".")])
        XCTAssertTrue(result.success, "\(result.error)")
        XCTAssertTrue(result.output.contains("code/demo.py:1: def hello()"),
                      "应输出 path:line: 格式：\(result.output)")
    }

    func testFileSearchCaseInsensitiveByDefault() async throws {
        try write("doc.txt", "HELLO world\n")
        let result = try await FileSearchTool(config: config, callbacks: nil)
            .run(["pattern": .string("hello")])
        XCTAssertTrue(result.output.contains("doc.txt:1"), "\(result.output)")
    }

    func testFileSearchSkipsBinary() async throws {
        let full = root + "/blob.bin"
        try FileManager.default.createDirectory(atPath: root, withIntermediateDirectories: true)
        try Data([0x00, 0x01, 0x02, 0x48, 0x49, 0x00]).write(to: URL(fileURLWithPath: full))
        try write("text.txt", "needle\n")
        let result = try await FileSearchTool(config: config, callbacks: nil)
            .run(["pattern": .string("needle")])
        XCTAssertTrue(result.success)
        XCTAssertTrue(result.output.contains("text.txt:1"))
    }

    func testFileSearchNoMatch() async throws {
        try write("t.txt", "nothing here\n")
        let result = try await FileSearchTool(config: config, callbacks: nil)
            .run(["pattern": .string("zzz-not-there")])
        XCTAssertTrue(result.success)
        XCTAssertTrue(result.output.contains("no matches"), "\(result.output)")
    }

    // MARK: file_read 分页 / 行号 / 编码

    func testFileReadPaginationWithLineNumbers() async throws {
        let lines = (1...50).map { "line \($0)" }.joined(separator: "\n")
        try write("big.txt", lines)
        let result = try await FileReadTool(config: config, callbacks: nil).run([
            "path": .string("big.txt"),
            "offset": .string("10"), "limit": .string("5"),
            "line_numbers": .string("true"),
        ])
        XCTAssertTrue(result.success, "\(result.error)")
        XCTAssertTrue(result.output.contains("10: line 10"))
        XCTAssertTrue(result.output.contains("14: line 14"))
        XCTAssertFalse(result.output.contains("9: line 9"))
        XCTAssertTrue(result.output.contains("显示第 10–14 行"))
    }

    func testFileReadGB18030Fallback() async throws {
        // GB18030 编码的中文文本：UTF-8 解码会失败，应回退 GB18030 而不是 base64。
        let encoding = String.Encoding(
            rawValue: CFStringConvertEncodingToNSStringEncoding(
                CFStringEncoding(CFStringEncodings.GB_18030_2000.rawValue)))
        let full = root + "/gb.txt"
        try "中文内容测试".data(using: encoding)!.write(to: URL(fileURLWithPath: full))
        let result = try await FileReadTool(config: config, callbacks: nil)
            .run(["path": .string("gb.txt")])
        XCTAssertTrue(result.success, "\(result.error)")
        XCTAssertTrue(result.output.contains("GB18030"), "应标注编码：\(result.output)")
        XCTAssertTrue(result.output.contains("中文内容测试"), "应正确解码：\(result.output)")
    }

    // MARK: file_modify

    func testFileModifyReportsReplacementCount() async throws {
        try write("m.txt", "alpha beta alpha")
        let all = try await FileModifyTool(config: config, callbacks: nil).run([
            "path": .string("m.txt"), "old": .string("alpha"),
            "new": .string("X"), "replace_all": .string("true"),
        ])
        XCTAssertTrue(all.output.contains("替换 2 处"), "\(all.output)")
        let one = try await FileModifyTool(config: config, callbacks: nil).run([
            "path": .string("m.txt"), "old": .string("beta"), "new": .string("Y"),
        ])
        XCTAssertTrue(one.output.contains("替换 1 处"), "\(one.output)")
    }

    func testFileModifyNotFoundGivesGuidance() async throws {
        try write("n.txt", "hello\r\nworld\r\n")
        let result = try await FileModifyTool(config: config, callbacks: nil).run([
            "path": .string("n.txt"), "old": .string("no-such-text"), "new": .string("x"),
        ])
        XCTAssertFalse(result.success)
        XCTAssertTrue(result.error.contains("file_read"), "应给出排查指引：\(result.error)")
    }

    // MARK: shell_run shell/env

    func testShellRunBashAndEnv() async throws {
        // bash 特有语法（[[ ]]）在 sh 下会静默失败，bash 下通过。
        let bash = try await ShellRunTool(config: config, callbacks: nil).run([
            "command": .string("[[ 1 == 1 ]] && echo bash-ok"),
            "shell": .string("bash"),
        ])
        XCTAssertTrue(bash.success, "\(bash.error)")
        XCTAssertTrue(bash.output.contains("bash-ok"))

        let env = try await ShellRunTool(config: config, callbacks: nil).run([
            "command": .string("echo \"FOO=$FOO\""),
            "env": .object(["FOO": .string("bar123")]),
        ])
        XCTAssertTrue(env.success, "\(env.error)")
        XCTAssertTrue(env.output.contains("FOO=bar123"), "\(env.output)")

        let invalid = try await ShellRunTool(config: config, callbacks: nil).run([
            "command": .string("echo hi"),
            "env": .object(["BAD KEY": .string("v")]),
        ])
        XCTAssertFalse(invalid.success, "非法 env key 应被拒绝")
    }

    func testShellRunUnsupportedShellRejected() async throws {
        let result = try await ShellRunTool(config: config, callbacks: nil).run([
            "command": .string("echo hi"), "shell": .string("powershell"),
        ])
        XCTAssertFalse(result.success)
    }

    // MARK: RSS/Atom feed 解析

    func testSummarizeFeedRSS() {
        let xml = """
        <?xml version="1.0"?><rss version="2.0"><channel>
        <title>News</title>
        <item><title>First headline</title><link>https://example.com/1</link>
            <pubDate>Mon, 09 Nov 2026 10:00:00 GMT</pubDate></item>
        <item><title><![CDATA[Second & headline]]></title>
            <link>https://example.com/2</link></item>
        </channel></rss>
        """
        let result = WebScanTool.summarizeFeed(xml)
        XCTAssertNotNil(result)
        XCTAssertTrue(result!.contains("First headline"), "\(result!)")
        XCTAssertTrue(result!.contains("https://example.com/1"))
        XCTAssertTrue(result!.contains("Second & headline"), "CDATA 应被剥开")
        XCTAssertTrue(result!.contains("共 2 条"))
    }

    func testSummarizeFeedAtomHrefLink() {
        let xml = """
        <?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
        <entry><title>Atom item</title>
            <link href="https://atom.example/x" rel="alternate"/></entry>
        </feed>
        """
        let result = WebScanTool.summarizeFeed(xml)
        XCTAssertNotNil(result)
        XCTAssertTrue(result!.contains("Atom item"))
        XCTAssertTrue(result!.contains("https://atom.example/x"), "自闭合 link 应回退 href：\(result!)")
    }

    func testSummarizeFeedRejectsNonFeed() {
        XCTAssertNil(WebScanTool.summarizeFeed("<root><a>plain xml</a></root>"))
    }
}
