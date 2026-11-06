import Darwin
import Foundation
import MCP

// MARK: - 错误

/// MCP 客户端错误（进程管理 / 超时 / 结果渲染）。协议层错误由官方 SDK 的
/// `MCP.MCPError` 承担，这里统一转成本类型向上抛。
public struct MCPClientError: Error, LocalizedError, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

// MARK: - 官方 SDK MCP 客户端封装（stdio 子进程 / Streamable HTTP 双形态）
//
/// 基于官方 MCP Swift SDK 的 MCP 客户端，支持两种接入形态：
/// - **stdio**：`init(name:command:...)` 拉起子进程，`ProcessPipeTransport` 走管道；
/// - **HTTP**：`init(name:url:...)` 直连 Streamable HTTP 端点（SDK
///   `HTTPClientTransport`），无需子进程。
///
/// SDK 负责协议层：initialize 握手与版本协商（2025-11-25 / 2025-06-18 /
/// 2025-03-26 / 2024-11-05）、JSON-RPC 编解码、请求响应配对、通知分发、
/// 服务端反向请求。本类只负责：
/// - 形态一的生命周期：子进程 PATH 查找、环境变量合并、退出感知；
/// - 超时控制：SDK 无内置超时，统一用任务组竞速实现；
/// - 工具清单缓存与 `notifications/tools/list_changed` 回调（`onToolsChanged`）；
/// - 结果渲染：content 数组 → 文本，超长截断。
public final class MCPClient: @unchecked Sendable {
    public let name: String
    public let command: String
    public let args: [String]
    public let env: [String: String]
    /// 非空 = Streamable HTTP 直连形态（无子进程）。
    public let remoteURL: URL?
    public let timeout: TimeInterval
    public let maxOutputChars: Int

    private let lock = NSRecursiveLock()
    private let toolCallTimeout: TimeInterval = 300

    private var process = Process()
    private var stdinPipe = Pipe()
    private var stdoutPipe = Pipe()
    private var stderrPipe = Pipe()
    private var sdkClient: MCP.Client?
    private var running = false
    private var startInProgress = false
    /// shutdown() 之后永久关闭：callTool 不允许再把它「复活」。
    private var closed = false
    private var serverInfo: JSONValue = .null
    private var tools: [JSONValue] = []
    /// 服务端通知 tools/list_changed 后回调最新工具清单。
    public var onToolsChanged: (([JSONValue]) -> Void)?

    /// stdio 形态：拉起 `command` 子进程走管道。
    public init(name: String, command: String, args: [String] = [],
                env: [String: String] = [:], timeout: TimeInterval = 30,
                maxOutputChars: Int = 200_000) {
        self.name = name
        self.command = command
        self.args = args
        self.env = env
        self.remoteURL = nil
        self.timeout = max(1, timeout)
        self.maxOutputChars = max(1, maxOutputChars)
    }

    /// HTTP 形态：直连 Streamable HTTP 端点（如本机 RAG MCP 服务）。
    public init(name: String, url: String, timeout: TimeInterval = 30,
                maxOutputChars: Int = 200_000) {
        self.name = name
        self.command = ""
        self.args = []
        self.env = [:]
        self.remoteURL = URL(string: url)
        self.timeout = max(1, timeout)
        self.maxOutputChars = max(1, maxOutputChars)
    }

    /// 按配置自动选择形态：`url` 非空走 HTTP，否则 stdio。
    public convenience init(config: MCPServerConfig, timeout: TimeInterval = 30) {
        if let url = config.url, !url.isEmpty {
            self.init(name: config.name, url: url, timeout: timeout)
        } else {
            self.init(name: config.name, command: config.command,
                      args: config.args, env: config.env, timeout: timeout)
        }
    }

    deinit {
        stop()
    }

    // MARK: - 生命周期

    public var isRunning: Bool {
        lock.lock(); defer { lock.unlock() }
        return running
    }

    public var serverInfoValue: JSONValue {
        lock.lock(); defer { lock.unlock() }
        return serverInfo
    }

    /// 启动进程并完成 MCP 握手。
    ///
    /// 并发调用会等待同一个启动过程完成；`running` 只在握手期间置位，
    /// 避免第二个调用在 initialize 完成前就发出 tools/call。
    public func start() async throws {
        while true {
            lock.lock()
            if closed {
                lock.unlock()
                throw MCPClientError("MCP 服务 '\(name)' 已关闭")
            }
            if running && !startInProgress {
                lock.unlock()
                return
            }
            if startInProgress {
                lock.unlock()
                do {
                    try await Task.sleep(nanoseconds: 20_000_000)
                } catch {
                    throw CancellationError()
                }
                continue
            }
            startInProgress = true
            lock.unlock()
            break
        }
        defer {
            lock.lock(); startInProgress = false; lock.unlock()
        }
        try await performStart()
    }

    private func performStart() async throws {
        if let endpoint = remoteURL {
            guard endpoint.scheme == "http" || endpoint.scheme == "https" else {
                throw MCPClientError("MCP 服务 '\(name)' 的端点非法：\(endpoint.absoluteString)")
            }
            try await performStartHTTP(endpoint)
        } else {
            try await performStartStdio()
        }
    }

    /// HTTP 形态：官方 `HTTPClientTransport` 直连，无子进程。
    private func performStartHTTP(_ endpoint: URL) async throws {
        let client = MCP.Client(name: "base-agent", version: AppVersion.current)
        self.setSDKClient(client)
        lock.lock(); running = true; lock.unlock()
        do {
            let info = try await withTimeout(timeout, "initialize") {
                let transport = MCP.HTTPClientTransport(endpoint: endpoint, streaming: true)
                return try await client.connect(transport: transport)
            }
            try await finishHandshake(client: client, info: info)
        } catch {
            stop()
            throw error
        }
    }

    /// stdio 形态：拉起子进程，`ProcessPipeTransport` 走管道。
    private func performStartStdio() async throws {
        // 重连时必须使用全新的 Process / Pipe / Client / Transport：
        // Process 不能重复 run，SDK Client 断开后也不应复用。
        let process = Process()
        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        self.process = process
        self.stdinPipe = stdinPipe
        self.stdoutPipe = stdoutPipe
        self.stderrPipe = stderrPipe

        process.environment = ProcessInfo.processInfo.environment.merging(env) { _, new in new }
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        if command.contains("/") {
            process.executableURL = URL(fileURLWithPath: command)
            process.arguments = args
        } else {
            // 与 Python Popen 一致：无路径分隔符时使用 PATH 查找。
            process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            process.arguments = [command] + args
        }

        lock.lock(); running = true; lock.unlock()

        do {
            try process.run()
        } catch {
            lock.lock(); running = false; lock.unlock()
            throw MCPClientError("启动 MCP 服务 '\(name)' 失败: \(error.localizedDescription)")
        }

        // 进程退出感知：立即断开 SDK 客户端，挂起请求马上失败，
        // 避免一直等到超时。
        process.terminationHandler = { [weak self] _ in
            guard let self else { return }
            self.lock.lock()
            let wasRunning = self.running
            self.running = false
            self.lock.unlock()
            guard wasRunning else { return }
            AgentLog.warning("MCP[\(self.name)]: 服务进程已退出")
            if let client = self.currentSDKClient() {
                Task { await client.disconnect() }
            }
        }

        // stderr 只承载日志，绝不能进协议通道。
        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                return
            }
            guard let text = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
                !text.isEmpty else { return }
            AgentLog.warning("MCP[stderr]: \(String(text.prefix(500)))")
        }

        // 事件驱动管道 Transport（不用官方 StdioTransport：其 10ms 轮询抛错
        // 在 XCTest 错误观察钩子下会饿死协作线程池，见 ProcessPipeTransport 注释）。
        let transport = ProcessPipeTransport(
            input: stdoutPipe.fileHandleForReading.fileDescriptor,
            output: stdinPipe.fileHandleForWriting.fileDescriptor)
        let client = MCP.Client(
            name: "base-agent",
            version: AppVersion.current)
        self.setSDKClient(client)

        do {
            let info = try await withTimeout(timeout, "initialize") {
                try await client.connect(transport: transport)
            }
            try await finishHandshake(client: client, info: info)
        } catch {
            stop()
            throw error
        }
    }

    /// 握手公共尾部：记录 serverInfo、注册 tools/list_changed 处理器、
    /// 拉取初始工具清单。
    private func finishHandshake(client: MCP.Client,
                                 info: MCP.Initialize.Result) async throws {
        let infoValue = try JSONValue(sdkValue: .object([
            "protocolVersion": .string(info.protocolVersion),
            "serverInfo": (try? MCP.Value(info.serverInfo)) ?? .null,
        ]))
        lock.lock()
        serverInfo = infoValue
        lock.unlock()

        await client.onNotification(MCP.ToolListChangedNotification.self) {
            [weak self] _ in
            guard let self else { return }
            // SDK 消息循环会内联 await 本处理器；refreshTools 的
            // listTools 响应同样要经该循环分发，内联等待会自等自死锁
            // （通知 → 等清单响应 → 响应又只能由被阻塞的循环处理）。
            // 必须脱开到非结构化 Task，让循环立刻继续消费消息。
            Task { await self.handleToolsChanged() }
        }
        try await refreshTools()
    }

    private func currentSDKClient() -> MCP.Client? {
        lock.lock(); defer { lock.unlock() }
        return sdkClient
    }

    private func setSDKClient(_ client: MCP.Client?) {
        lock.lock(); sdkClient = client; lock.unlock()
    }

    private func handleToolsChanged() async {
        guard isRunning else { return }
        guard (try? await refreshTools()) != nil else { return }
        // 让上层更新 ToolRegistry / schema；否则工具清单变了，
        // 模型侧仍然只能看到旧工具。
        onToolsChanged?(listTools())
    }

    /// 拉取并缓存工具清单。
    public func refreshTools() async throws {
        guard let client = currentSDKClient(), isRunning else {
            throw MCPClientError("MCP 服务 '\(name)' 未启动")
        }
        let (list, _) = try await client.listTools()
        let definitions: [JSONValue] = try list.map { tool in
            let data = try JSONEncoder().encode(tool)
            return try JSONDecoder().decode(JSONValue.self, from: data)
        }
        lock.lock()
        tools = definitions
        lock.unlock()
    }

    public func listTools() -> [JSONValue] {
        lock.lock(); defer { lock.unlock() }
        return tools
    }

    /// 断开连接（stdio 形态同时终止子进程）。幂等。
    public func stop() {
        lock.lock()
        running = false
        let client = sdkClient
        sdkClient = nil
        let pid = process.processIdentifier
        lock.unlock()

        if remoteURL == nil {
            // stdio 形态才有的子进程/管道清理。
            stderrPipe.fileHandleForReading.readabilityHandler = nil

            if process.isRunning {
                // MCP 子进程通常是 npx/node 包装：必须连同子进程树一起终止。
                if pid > 0 {
                    ToolProcessRunner.terminateProcessTree(pid: pid, signal: SIGTERM)
                    ToolProcessRunner.terminateProcessTree(pid: pid, signal: SIGKILL)
                } else {
                    process.terminate()
                }
            }
            try? stdinPipe.fileHandleForWriting.close()
            try? stdoutPipe.fileHandleForReading.close()
            try? stderrPipe.fileHandleForReading.close()
        }

        if let client {
            // 断开会取消全部挂起请求（SDK 行为），无需自己维护 pending 表。
            Task { await client.disconnect() }
        }
    }

    public func shutdown() {
        lock.lock()
        closed = true
        lock.unlock()
        stop()
    }

    // MARK: - 调用

    /// 调用服务端工具，返回文本结果；`isError=true` 时抛错。
    public func callTool(name: String, arguments: [String: JSONValue]) async throws -> String {
        // 服务进程异常退出后自动重连一次，避免工具永久失效；
        // 但已被 shutdown() 的客户端不允许借 callTool 复活。
        if !isRunning {
            lock.lock()
            let rejected = closed
            lock.unlock()
            if rejected {
                throw MCPClientError("MCP 服务 '\(self.name)' 已关闭")
            }
            try await start()
        }
        guard let client = currentSDKClient() else {
            throw MCPClientError("MCP 服务 '\(name)' 未启动")
        }
        let sdkArgs = try arguments.mapValues { try $0.sdkValue }
        let result = try await withTimeout(toolCallTimeout, "tools/call(\(name))") {
            try await client.callTool(name: name, arguments: sdkArgs)
        }
        // 渲染时就限制长度：避免先拼出完整（可能几十 MB）的字符串再截断。
        let text = Self.render(content: result.content, limit: maxOutputChars + 1)
        if result.isError == true {
            throw MCPClientError("MCP 工具 '\(name)' 返回错误: \(text)")
        }
        guard text.count > maxOutputChars else { return text }
        return String(text.prefix(maxOutputChars))
            + "\n… [MCP 工具输出超过 \(maxOutputChars) 字符，已截断]"
    }

    // MARK: - 超时

    /// SDK 没有内置请求超时：用任务组竞速实现，超时抛错。
    /// 底层请求任务会继续跑到完成/进程退出，响应被丢弃（与旧实现一致）。
    private func withTimeout<T: Sendable>(
        _ seconds: TimeInterval, _ label: String,
        _ operation: @escaping @Sendable () async throws -> T) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask { try await operation() }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(max(1, seconds) * 1_000_000_000))
                throw MCPClientError(
                    "MCP 请求超时（\(self.name), \(label), \(Int(max(1, seconds)))s）")
            }
            guard let result = try await group.next() else {
                throw MCPClientError("MCP 请求失败（\(label)）")
            }
            group.cancelAll()
            return result
        }
    }

    // MARK: - 结果渲染

    private static func render(content: [MCP.Tool.Content], limit: Int) -> String {
        var parts: [String] = []
        var total = 0
        for item in content {
            let piece: String
            switch item {
            case .text(let text, _, _):
                piece = text
            default:
                if let data = try? JSONEncoder().encode(item),
                   let object = try? JSONDecoder().decode(JSONValue.self, from: data) {
                    piece = object.jsonString()
                } else {
                    piece = ""
                }
            }
            guard !piece.isEmpty else { continue }
            parts.append(piece)
            total += piece.count + 1
            if total >= limit { break }
        }
        let joined = parts.joined(separator: "\n")
        return String(joined.prefix(limit))
    }
}
