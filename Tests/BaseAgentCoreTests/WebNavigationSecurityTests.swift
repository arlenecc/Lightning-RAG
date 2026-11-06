import XCTest
@testable import BaseAgentCore

/// P0-c：网页链路的 SSRF 防线。
/// 覆盖共享 URL 策略（入口 / 页内导航 / 浏览器桥）、WKWebView 网络层规则、
/// `webexec_js` 的端点与导航参数校验。WKWebView 本身不在 xctest 宿主创建。
final class WebNavigationSecurityTests: XCTestCase {

    private func config(endpoint: String = "") -> AgentConfig {
        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory() + "agent-p0c-\(UUID().uuidString)"
        config.browserEndpoint = endpoint
        return config
    }

    // MARK: 共享策略

    func testSharedPolicyBlocksNonHTTPAndPrivateTargets() throws {
        let blocked = [
            "file:///etc/passwd",
            "javascript:alert(1)",
            "ftp://example.com/x",
            "http://127.0.0.1/admin",
            "http://localhost:9222/json",
            "http://169.254.169.254/latest/meta-data",
            "http://10.0.0.5/",
            "http://192.168.1.10/",
            "http://[fe80::1]/",
            "http://metadata.google.internal/",
        ]
        for raw in blocked {
            let url = try XCTUnwrap(URL(string: raw), "\(raw) 应可解析")
            XCTAssertNotNil(WebURLPolicy.rejectionReason(for: url), "应拒绝 \(raw)")
        }
        XCTAssertNil(WebURLPolicy.rejectionReason(
            for: try XCTUnwrap(URL(string: "https://example.com/page"))))
    }

    func testSharedPolicyOnlyExemptsLoopbackInTests() throws {
        let loopback = try XCTUnwrap(URL(string: "http://127.0.0.1:8080/page"))
        XCTAssertNotNil(WebURLPolicy.rejectionReason(for: loopback))
        XCTAssertNil(WebURLPolicy.rejectionReason(
            for: loopback, allowLoopbackForTesting: true))
        let lan = try XCTUnwrap(URL(string: "http://192.168.1.10/"))
        XCTAssertNotNil(WebURLPolicy.rejectionReason(
            for: lan, allowLoopbackForTesting: true), "测试豁免不得放开整个私网")
    }

    func testWebRenderValidateURLUsesSharedPolicy() {
        WebRenderer.localhostAllowedForTesting = false
        XCTAssertNotNil(WebRenderTool.validateURL("http://169.254.169.254/latest/meta-data"))
        WebRenderer.localhostAllowedForTesting = true
        XCTAssertNil(WebRenderTool.validateURL("http://127.0.0.1:8080/"))
        XCTAssertNotNil(WebRenderTool.validateURL("http://10.0.0.5/"))
        WebRenderer.localhostAllowedForTesting = false
    }

    // MARK: WKWebView 网络层规则

    func testBlockRulesAreValidJSONAndCoverSSRFTargets() throws {
        for allowLoopback in [false, true] {
            let json = WebRenderer.blockRulesJSON(allowLoopback: allowLoopback)
            let data = try XCTUnwrap(json.data(using: .utf8))
            let parsed = try XCTUnwrap(
                try JSONSerialization.jsonObject(with: data) as? [[String: Any]],
                "规则集必须是合法 JSON 数组")
            XCTAssertGreaterThan(parsed.count, 20)
            let filters = parsed.compactMap { rule -> String? in
                (rule["trigger"] as? [String: Any])?["url-filter"] as? String
            }
            XCTAssertTrue(filters.contains("^file://"))
            XCTAssertTrue(filters.contains("^https?://169[.]254[.]"),
                          "云元数据必须被网络层拦住")
            XCTAssertTrue(filters.contains("^https?://10[.]"))
            XCTAssertTrue(filters.contains(".*google-analytics[.]com.*"))
            let hasLoopbackRule = filters.contains("^https?://127[.]")
            XCTAssertEqual(hasLoopbackRule, !allowLoopback,
                           "仅测试模式可放行回环")
        }
    }

    /// 导出两套规则供**独立进程**编译验证。
    ///
    /// 语法错误会让整份规则编译失败、网络层防护静默降级，所以必须实测；但
    /// xctest 宿主里 `WKContentRuleListStore` 的回调不会返回（WebKit 服务不可用），
    /// 故这里只导出，编译由普通 CLI 进程 `swift -e` 读取这两个文件完成：
    ///
    ///     ENABLE_WEBKIT_RENDER_TESTS=1 swift test --filter \
    ///       WebNavigationSecurityTests/testExportContentRulesForExternalCompilation
    ///     swift -e '…compileContentRuleList(forIdentifier:…)…'
    func testExportContentRulesForExternalCompilation() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["ENABLE_WEBKIT_RENDER_TESTS"] == "1",
            "仅导出规则，默认跳过")
        for (suffix, allowLoopback) in [("strict", false), ("loopback", true)] {
            let path = "/tmp/base-agent-webrender-rules-\(suffix).json"
            try WebRenderer.blockRulesJSON(allowLoopback: allowLoopback)
                .write(toFile: path, atomically: true, encoding: .utf8)
        }
    }

    // MARK: webexec_js

    func testWebExecJsRejectsUnsafeNavigationURLBeforeAnyRequest() async throws {
        let tool = WebExecJsTool(config: config(endpoint: "http://localhost:9222"),
                                 callbacks: nil)
        defer { tool.shutdown() }
        for raw in ["file:///etc/passwd", "http://169.254.169.254/latest/meta-data",
                    "http://10.0.0.5/", "javascript:alert(1)"] {
            let result = try await tool.run([
                "url": .string(raw), "script": .string("1+1"),
            ])
            XCTAssertFalse(result.success, "\(raw) 不应交给浏览器桥")
            XCTAssertTrue(result.error.contains("Blocked navigation URL"),
                          "got: \(result.error)")
        }
    }

    func testWebExecJsRejectsMalformedEndpoint() async throws {
        let tool = WebExecJsTool(config: config(endpoint: "file:///tmp/bridge"),
                                 callbacks: nil)
        defer { tool.shutdown() }
        let result = try await tool.run(["script": .string("1+1")])
        XCTAssertFalse(result.success)
        XCTAssertTrue(result.error.contains("Invalid browser_endpoint"),
                      "got: \(result.error)")
    }

    func testRemoteBridgeRequiresConfirmation() {
        XCTAssertFalse(WebExecJsTool(
            config: config(endpoint: "http://localhost:9222"), callbacks: nil
        ).shouldConfirm([:]))
        XCTAssertFalse(WebExecJsTool(
            config: config(endpoint: "http://127.0.0.1:9222"), callbacks: nil
        ).shouldConfirm([:]))
        XCTAssertTrue(WebExecJsTool(
            config: config(endpoint: "http://bridge.example.com:9222"), callbacks: nil
        ).shouldConfirm([:]), "远端端点会把脚本与页面内容外发，必须人工确认")
    }

    func testHeadlessDeniesRemoteBridgeByDefault() async throws {
        let cfg = config(endpoint: "http://bridge.example.com:9222")
        // registry.callbacks 是 weak：测试必须自己持有，否则会退化成「无确认通道」。
        let callbacks = NoopCallbacks()
        let registry = ToolRegistry(config: cfg, callbacks: callbacks)
        let tool = WebExecJsTool(config: cfg, callbacks: callbacks)
        registry.register(tool)
        defer { tool.shutdown() }

        let result = try await registry.execute("webexec_js",
                                                ["script": .string("1+1")])
        XCTAssertFalse(result.success)
        XCTAssertTrue(result.error.contains("User denied"), "got: \(result.error)")
        withExtendedLifetime(callbacks) {}
    }
}
