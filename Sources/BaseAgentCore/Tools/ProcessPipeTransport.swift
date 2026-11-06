import Darwin
import Foundation
import Logging
import MCP

// MARK: - 子进程管道 Transport（事件驱动 stdio）
//
/// 官方 `StdioTransport` 的替代实现，用于 `MCPClient` 连接外部 MCP server。
///
/// 不用官方实现的原因：它在后台任务里以 10ms 间隔轮询 `read()`，靠 EAGAIN
/// 异常驱动；而 XCTest 的 Swift throw 观察钩子会把每个抛出的错误桥接成
/// NSError（捕获并符号化调用栈），轮询任务因此耗尽 Swift 协作线程池，
/// 把 `send` 任务饿死——实测表现为握手请求根本没写出去、双方互等死锁
/// （Swift 子进程毫秒级就绪时侥幸通过，Python 等启动慢的子进程必挂）。
/// 本实现用 DispatchSourceRead 事件驱动读取：无轮询、热路径零抛错，
/// 测试与生产行为一致。
///
/// 管道 fd 归 `MCPClient` 所有（创建/关闭都在那边），这里只做读写：
/// - input（子进程 stdout）设为非阻塞，数据到达时 drain 到 EAGAIN，
///   按换行切分后 yield 完整 JSON-RPC 行；
/// - output（子进程 stdin）保持阻塞写（64KB 管道缓冲，正常消息一次写入；
///   子进程不读时 send 阻塞，由上层超时护栏兜底）；
/// - EOF（子进程退出）→ 流正常结束；读错误 → 流以错误结束。
actor ProcessPipeTransport: MCP.Transport {
    nonisolated let logger: Logger

    private let inputFD: Int32
    private let outputFD: Int32

    private var stream: AsyncThrowingStream<Data, Error>?
    private var continuation: AsyncThrowingStream<Data, Error>.Continuation?
    private var pending = Data()
    private var connected = false
    private var readSource: DispatchSourceRead?

    init(input: Int32, output: Int32) {
        self.inputFD = input
        self.outputFD = output
        self.logger = Logger(label: "base-agent.mcp.pipe") { _ in
            SwiftLogNoOpLogHandler()
        }
    }

    deinit {
        readSource?.cancel()
    }

    func connect() async throws {

        guard !connected else { return }
        // input 设为非阻塞：事件触发后的连续 read 不会阻塞协作线程。
        let flags = fcntl(inputFD, F_GETFL)
        if flags >= 0 {
            fcntl(inputFD, F_SETFL, flags | O_NONBLOCK)
        }
        var captured: AsyncThrowingStream<Data, Error>.Continuation?
        let newStream = AsyncThrowingStream<Data, Error> { captured = $0 }
        stream = newStream
        continuation = captured
        connected = true

        let source = DispatchSource.makeReadSource(
            fileDescriptor: inputFD,
            queue: DispatchQueue(label: "base-agent.mcp.pipe.read"))
        source.setEventHandler { [weak self] in
            guard let self else { return }
            Task { await self.drainAvailableData() }
        }
        readSource = source
        source.resume()
        logger.debug("Pipe transport connected")
    }

    /// 事件回调：把当前可读数据全部取走（读到 EAGAIN 为止）。
    private func drainAvailableData() {
        var buffer = [UInt8](repeating: 0, count: 65536)
        while true {
            let n = buffer.withUnsafeMutableBufferPointer { pointer in
                read(inputFD, pointer.baseAddress, pointer.count)
            }
            if n > 0 {

                deliver(Data(buffer[..<n]))
                if n < buffer.count { break }
                continue
            }
            if n == 0 {
                // EOF：子进程已退出，正常结束流。
                finish(nil)
                return
            }
            if errno == EINTR { continue }
            if errno == EAGAIN || errno == EWOULDBLOCK { break }
            finish(MCP.MCPError.transportError(posixFailure(errno)))
            return
        }
    }

    /// 按换行切分并 yield 完整消息行（MCP stdio 规范：换行分隔、行内无换行）。
    private func deliver(_ chunk: Data) {
        pending.append(chunk)
        while let newlineIndex = pending.firstIndex(of: 0x0A) {
            let line = pending[..<newlineIndex]
            pending = Data(pending[pending.index(after: newlineIndex)...])
            if !line.isEmpty {
                continuation?.yield(Data(line))
            }
        }
    }

    func send(_ data: Data) async throws {

        guard connected else {
            throw MCP.MCPError.transportError(posixFailure(ENOTCONN))
        }
        var message = data
        message.append(0x0A)
        let failure = message.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Int32 in
            var offset = 0
            while offset < raw.count {
                let n = write(outputFD,
                              raw.baseAddress!.advanced(by: offset),
                              raw.count - offset)
                if n > 0 {
                    offset += n
                    continue
                }
                if n < 0 && errno == EINTR { continue }
                return errno
            }
            return 0
        }

        if failure != 0 {
            throw MCP.MCPError.transportError(posixFailure(failure))
        }
    }

    func receive() -> AsyncThrowingStream<Data, Error> {
        // SDK 只在 connect 后取一次；connect 之前理论上不会调用。
        stream ?? AsyncThrowingStream { $0.finish() }
    }

    func disconnect() async {
        finish(nil)
    }

    /// errno → Error（避免引入 SystemPackage 依赖，Errno 在那边）。
    private nonisolated func posixFailure(_ code: Int32) -> Error {
        POSIXError(POSIXErrorCode(rawValue: code) ?? .ENODEV)
    }

    private func finish(_ error: Error?) {
        guard connected else { return }
        connected = false
        readSource?.cancel()
        readSource = nil
        pending.removeAll()
        if let error {
            continuation?.finish(throwing: error)
        } else {
            continuation?.finish()
        }
        continuation = nil
    }
}
