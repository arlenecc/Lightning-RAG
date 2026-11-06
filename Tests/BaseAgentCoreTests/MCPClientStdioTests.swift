import XCTest
@testable import BaseAgentCore

/// Agent 内 MCP 客户端（官方 SDK stdio 封装）的真实子进程测试。
///
/// 用 Python 实现一个最小但符合规范的 stdio MCP server（外部 server 正是
/// agent 的真实使用场景），验证 MCPClient 全流程：
/// initialize 握手 / tools/list / tools/call（成功、isError、中文 UTF-8）/
/// 服务端 notifications/tools/list_changed → onToolsChanged 回调 /
/// shutdown 后不可复活。
final class MCPClientStdioTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mcp-client-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func pythonPath() -> String? {
        if let override = ProcessInfo.processInfo.environment["MCP_TEST_PYTHON"],
           FileManager.default.isExecutableFile(atPath: override) {
            return override
        }
        let candidates = ["/usr/bin/python3", "/opt/homebrew/bin/python3",
                          "/usr/local/bin/python3"]
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
    }

    /// 最小规范 stdio MCP server：换行分隔 JSON-RPC；echo 后推送
    /// tools/list_changed，随后 tools/list 返回追加后的清单。
    private func writeServerScript() throws -> String {
        let script = #"""
        import sys, json

        TOOLS_V1 = [{
            "name": "echo",
            "description": "回显输入文本",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }]
        TOOLS_V2 = TOOLS_V1 + [{
            "name": "extra",
            "description": "清单变更后新增的工具",
            "inputSchema": {"type": "object", "properties": {}},
        }]
        refreshed = False

        def send(obj):
            sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            sys.stdout.flush()

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            method = msg.get("method")
            if "id" not in msg:
                continue
            mid = msg["id"]
            if method == "initialize":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {"name": "fake-mcp", "version": "0.1.0"},
                }})
            elif method == "tools/list":
                send({"jsonrpc": "2.0", "id": mid,
                      "result": {"tools": TOOLS_V2 if refreshed else TOOLS_V1}})
            elif method == "tools/call":
                name = msg["params"]["name"]
                args = msg["params"].get("arguments", {})
                if name == "echo":
                    send({"jsonrpc": "2.0", "id": mid, "result": {
                        "content": [{"type": "text",
                                     "text": "echo: " + args.get("text", "")}],
                        "isError": False}})
                    if not refreshed:
                        refreshed = True
                        send({"jsonrpc": "2.0",
                              "method": "notifications/tools/list_changed"})
                elif name == "boom":
                    send({"jsonrpc": "2.0", "id": mid, "result": {
                        "content": [{"type": "text", "text": "模拟工具失败"}],
                        "isError": True}})
                else:
                    send({"jsonrpc": "2.0", "id": mid, "error": {
                        "code": -32602, "message": "未知工具"}})
            elif method == "ping":
                send({"jsonrpc": "2.0", "id": mid, "result": {}})
        """#
        let path = directory.appendingPathComponent("fake_mcp_server.py").path
        try script.write(toFile: path, atomically: true, encoding: .utf8)
        return path
    }

    private func makeClient(script: String, python: String) -> MCPClient {
        MCPClient(name: "fake-mcp", command: python,
                  args: [script], env: ["PYTHONIOENCODING": "utf-8"],
                  timeout: 30)
    }

    @MainActor
    private func waitFor(_ condition: @autoclosure () -> Bool,
                         timeout: TimeInterval = 10) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() {
            if Date() > deadline { return false }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        return true
    }

    func testMCPClientFullWorkflowOverStdio() async throws {
        guard let python = pythonPath() else {
            throw XCTSkip("未找到 python3（可设 MCP_TEST_PYTHON 指定）")
        }
        let script = try writeServerScript()
        let client = makeClient(script: script, python: python)
        defer { client.shutdown() }

        // 1. 启动 + initialize 握手：版本协商与 serverInfo。
        try await client.start()
        XCTAssertTrue(client.isRunning)
        XCTAssertEqual(
            client.serverInfoValue.objectValue?["serverInfo"]?
                .objectValue?["name"]?.stringValue,
            "fake-mcp")
        XCTAssertEqual(
            client.serverInfoValue.objectValue?["protocolVersion"]?.stringValue,
            "2025-06-18")

        // 2. tools/list：schema 完整透传。
        let tools = client.listTools()
        XCTAssertEqual(tools.count, 1)
        XCTAssertEqual(tools.first?.objectValue?["name"]?.stringValue, "echo")
        let required = tools.first?.objectValue?["inputSchema"]?.objectValue
        XCTAssertEqual(required?["required"]?.arrayValue?.compactMap(\.stringValue),
                       ["text"])

        // 3. tools/call 成功 + 中文 UTF-8 往返。
        let echoed = try await client.callTool(
            name: "echo", arguments: ["text": .string("你好，MCP")])
        XCTAssertEqual(echoed, "echo: 你好，MCP")

        // 4. 服务端 tools/list_changed 推送 → onToolsChanged 收到 2 个工具。
        var latestTools: [JSONValue] = []
        client.onToolsChanged = { latestTools = $0 }
        let gotUpdate = await waitFor(latestTools.count == 2)
        XCTAssertTrue(gotUpdate, "应收到 tools/list_changed 推送后的新清单")
        XCTAssertEqual(
            latestTools.last?.objectValue?["name"]?.stringValue, "extra")
        // 本地缓存同步刷新。
        XCTAssertEqual(client.listTools().count, 2)

        // 5. isError 结果：抛错且带服务端文本。
        do {
            _ = try await client.callTool(name: "boom", arguments: [:])
            XCTFail("isError=true 应抛错")
        } catch let error as MCPClientError {
            XCTAssertTrue(error.message.contains("模拟工具失败"),
                          "错误应包含服务端文本：\(error.message)")
        }

        // 6. shutdown 后 callTool 不得复活进程。
        client.shutdown()
        XCTAssertFalse(client.isRunning)
        do {
            _ = try await client.callTool(name: "echo",
                                          arguments: ["text": .string("x")])
            XCTFail("shutdown 后 callTool 应抛错")
        } catch let error as MCPClientError {
            XCTAssertTrue(error.message.contains("已关闭"),
                          "应提示已关闭：\(error.message)")
        }
    }

    /// 服务进程异常退出：挂起请求立即失败、running 置位，再次调用自动重连。
    func testClientSurvivesProcessExitAndReconnects() async throws {
        guard let python = pythonPath() else {
            throw XCTSkip("未找到 python3（可设 MCP_TEST_PYTHON 指定）")
        }
        // 正常脚本 + 启动后用第二个客户端进程模拟崩溃不可行；
        // 直接 kill 拉起的子进程验证退出感知与自动重连。
        let script = try writeServerScript()
        let client = makeClient(script: script, python: python)
        defer { client.shutdown() }

        try await client.start()
        _ = try await client.callTool(name: "echo", arguments: ["text": .string("hi")])

        // 找到子进程并强杀（fake server 是 python3 <script>）。
        let pids = runCapture("/usr/bin/pgrep", ["-f", script])
            .split(separator: "\n").compactMap { Int($0) }
        XCTAssertFalse(pids.isEmpty, "应能找到 fake server 子进程：\(script)")
        for pid in pids { _ = runCapture("/bin/kill", ["-9", "\(pid)"]) }

        // 进程退出感知：running 很快变 false（terminationHandler）。
        let exited = await waitFor(!client.isRunning, timeout: 10)
        XCTAssertTrue(exited, "进程退出后 isRunning 应置 false")

        // 再次调用 → 自动重连一次并成功。
        let echoed = try await client.callTool(
            name: "echo", arguments: ["text": .string("again")])
        XCTAssertEqual(echoed, "echo: again")
        XCTAssertTrue(client.isRunning, "重连后应恢复 running")
    }

    private func runCapture(_ launchPath: String, _ args: [String]) -> String {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: launchPath)
        process.arguments = args
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            return String(data: data, encoding: .utf8) ?? ""
        } catch {
            return ""
        }
    }
}
