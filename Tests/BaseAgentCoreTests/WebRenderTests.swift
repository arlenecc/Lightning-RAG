import XCTest
import Network
@testable import BaseAgentCore

/// web_render：URL 校验 + WKWebView 真实渲染（本地 HTTP server）。
final class WebRenderTests: XCTestCase {

    // MARK: - URL 校验（纯逻辑，无网络）

    func testValidateURLRejectsNonHTTP() {
        XCTAssertNotNil(WebRenderTool.validateURL("file:///etc/passwd"))
        XCTAssertNotNil(WebRenderTool.validateURL("javascript:alert(1)"))
        XCTAssertNotNil(WebRenderTool.validateURL("ftp://example.com/x"))
        XCTAssertNotNil(WebRenderTool.validateURL("not a url"))
        XCTAssertNil(WebRenderTool.validateURL("https://example.com/page"))
    }

    func testValidateURLBlocksPrivateHosts() {
        // 与 setUp 无关：私网拦截在生产模式下必须始终生效（豁免仅限回环）。
        WebRenderer.localhostAllowedForTesting = false
        XCTAssertNotNil(WebRenderTool.validateURL("http://127.0.0.1/x"))
        XCTAssertNotNil(WebRenderTool.validateURL("http://localhost/x"))
        XCTAssertNotNil(WebRenderTool.validateURL("http://192.168.1.10/x"))
        XCTAssertNotNil(WebRenderTool.validateURL("http://169.254.169.254/latest/meta-data"))
        XCTAssertNotNil(WebRenderTool.validateURL("http://10.0.0.5/x"))
        XCTAssertNotNil(WebRenderTool.validateURL("http://[fe80::1]/x"))
        // 测试模式下仍只豁免回环，其余私网照拦。
        WebRenderer.localhostAllowedForTesting = true
        XCTAssertNil(WebRenderTool.validateURL("http://127.0.0.1/x"))
        XCTAssertNotNil(WebRenderTool.validateURL("http://192.168.1.10/x"))
    }

    func testTextGrowthThreshold() {
        XCTAssertFalse(WebRenderer.textGrowthReached(baseline: 1_000, current: 1_100))
        XCTAssertTrue(WebRenderer.textGrowthReached(baseline: 1_000, current: 1_300))
        XCTAssertTrue(WebRenderer.textGrowthReached(baseline: 0, current: 300))
    }

    // MARK: - WKWebView 渲染（本地 server）

    /// 极简本地 HTTP server：固定响应，端口自动分配。
    private final class MiniServer {
        let listener: NWListener
        let body: String
        private let queue = DispatchQueue(label: "mini-http")

        init(body: String) throws {
            self.body = body
            let parameter = NWParameters.tcp
            listener = try NWListener(using: parameter, on: .any)
            let data = body.data(using: .utf8)!
            listener.newConnectionHandler = { [weak self] connection in
                guard let self else { return }
                connection.start(queue: self.queue)
                connection.receive(minimumIncompleteLength: 1,
                                   maximumLength: 16 * 1024) { [weak self] _, _, _, _ in
                    guard let self else { return }
                    let response = ("HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                        + "Content-Length: \(data.count)\r\nConnection: close\r\n\r\n")
                        .data(using: .utf8)! + data
                    connection.send(content: response, completion: .contentProcessed { _ in
                        connection.cancel()
                    })
                }
            }
        }

        var url: URL {
            let port = listener.port?.rawValue ?? 8080
            return URL(string: "http://127.0.0.1:\(port)/")!
        }

        func start() {
            listener.start(queue: queue)
        }

        func stop() {
            listener.cancel()
        }
    }

    private var server: MiniServer?

    override func setUp() {
        super.setUp()
        WebRenderer.localhostAllowedForTesting = true
    }

    override func tearDown() {
        WebRenderer.localhostAllowedForTesting = false
        server?.stop()
        server = nil
        super.tearDown()
    }

    @discardableResult
    private func startServer(body: String) throws -> URL {
        let s = try MiniServer(body: body)
        s.start()
        Thread.sleep(forTimeInterval: 0.2) // listener 就绪
        server = s
        return s.url
    }

    func testRenderTextAndScript() async throws {
        // 曾因 `rules()` 在非隔离 TaskGroup 子任务里调用 @MainActor 的
        // WKContentRuleListStore 而在 xctest / GUI 里直接崩（signal 5），被误判为
        // 「WKWebView 无法在 xctest 宿主中创建」而长期 skip。根因修复后默认启用，
        // 这两个渲染用例同时充当该崩溃的回归保护。
        let url = try startServer(body: """
        <html><body><h1>Static Heading</h1>
        <div id="late"></div>
        <script>setTimeout(function(){
          document.getElementById('late').textContent = 'late-content-42';
        }, 300);</script>
        </body></html>
        """)
        let tool = WebRenderTool(config: testConfig, callbacks: nil)
        let result = try await tool.run([
            "url": .string(url.absoluteString),
            // `#late` 容器一开始就存在，等元素本身会立即返回；要等它填充完成。
            "wait_selector": .string("#late:not(:empty)"),
            "wait_ms": .string("0"),
            "script": .string("document.getElementById('late').textContent"),
        ])
        XCTAssertTrue(result.success, "expected ok, got error: \(result.error)")
        XCTAssertTrue(result.output.contains("Static Heading"), "应包含静态正文")
        XCTAssertTrue(result.output.contains("late-content-42"), "wait_selector 后应包含延迟填充内容")
        XCTAssertTrue(result.output.contains("[脚本返回]"), "脚本返回段应存在")
    }

    func testAlertDoesNotBlockRender() async throws {
        let url = try startServer(body: """
        <html><body><p>alert-page-ok</p>
        <script>alert('blocking?'); document.title='survived';</script>
        </body></html>
        """)
        let tool = WebRenderTool(config: testConfig, callbacks: nil)
        let result = try await tool.run([
            "url": .string(url.absoluteString),
            "wait_ms": .string("300"),
        ])
        XCTAssertTrue(result.success, "alert 不应阻塞渲染, got: \(result.error)")
        XCTAssertTrue(result.output.contains("alert-page-ok"))
    }

    func testInvalidURLFailsFast() async throws {
        let tool = WebRenderTool(config: testConfig, callbacks: nil)
        let result = try await tool.run(["url": .string("file:///etc/passwd")])
        XCTAssertFalse(result.success)
        XCTAssertTrue(result.error.contains("http/https") || result.error.contains("Invalid URL"),
                      "got: \(result.error)")
    }

    private var testConfig: AgentConfig {
        var config = AgentConfig()
        config.browserEndpoint = ""
        return config
    }
}
