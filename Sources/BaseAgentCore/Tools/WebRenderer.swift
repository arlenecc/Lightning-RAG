import Foundation
import WebKit

/// 内置网页渲染器：WKWebView 离屏加载 + JavaScript 交互，无需外部浏览器桥。
///
/// 设计要点：
/// - WKWebView 必须在主线程创建/使用，全部操作收拢在 MainActor；
/// - 单实例串行复用（创建成本高，且一个实例同时只能加载一个页面）；
/// - 导航、JS 求值、截图每一步都带超时；用户取消时立即 stopLoading，
///   不会让 continuation 悬挂；
/// - 实现 WKUIDelegate 的 alert/confirm/prompt：页面弹窗直接返回默认值，
///   防止 alert 阻塞整个渲染流程。
final class WebRenderer: @unchecked Sendable {
    /// 渲染池：`web_scan` / `web_render` 共享。单实例串行会让 parallelSafe
    /// 的 8 路并发退化成队头阻塞（动态页渲染 45s 起，8 连发最坏 8×45s）；
    /// 固定 2 个槽位按轮询分派，槽内 WKWebView 仍各自串行复用。
    /// ContentRuleList 由静态 RuleCache 共享，不会重复编译。
    private static let pool: [WebRenderer] = [WebRenderer(), WebRenderer()]
    private static let poolLock = NSLock()
    private static var poolCursor = 0
    static var shared: WebRenderer { nextFromPool() }
    private static func nextFromPool() -> WebRenderer {
        poolLock.lock(); defer { poolLock.unlock() }
        let renderer = pool[poolCursor % pool.count]
        poolCursor += 1
        return renderer
    }
    /// 会话结束时整池清理。
    static func resetAll() {
        poolLock.lock(); defer { poolLock.unlock() }
        pool.forEach { $0.reset() }
    }
    /// 测试专用：允许渲染 localhost（SSRF 防护由工具层负责，这里只管执行）。
    static var localhostAllowedForTesting = false

    /// 页面脚本实际发出的数据请求（含运行时拼好的查询参数）。
    /// web_scan 用它重放/复用接口响应，避免从压缩 JS 里猜接口参数。
    struct CapturedRequest: Sendable {
        var url: String
        var status: Int
        var body: String
    }

    struct Outcome: Sendable {
        var text = ""
        var html: String?
        var scriptResult: String?
        var screenshotPath: String?
        var capturedRequests: [CapturedRequest] = []
    }

    /// atDocumentStart 注入：记录页面自己发出的 XHR / fetch 的 URL、状态与响应体。
    /// 很多列表页的接口参数由运行时拼接（压缩 JS 里只有端点路径），只有页面
    /// 自己知道完整 URL；捕获后由工具层用 URLSession 重放即可拿到结构化数据。
    /// 数组与单条响应都封顶，避免大 JSON 拖垮 evaluateJavaScript 的跨进程传输。
    private static let captureScript = """
    (function () {
      window.__webScanRequests = [];
      var MAX = 12, MAXLEN = 180000;
      function push(url, status, body) {
        try {
          window.__webScanRequests.push({url: String(url || ''), status: status || 0,
                                        body: String(body || '').slice(0, MAXLEN)});
          if (window.__webScanRequests.length > MAX) window.__webScanRequests.shift();
        } catch (e) {}
      }
      var open = XMLHttpRequest.prototype.open;
      XMLHttpRequest.prototype.open = function (method, url) {
        this.__webScanURL = url;
        return open.apply(this, arguments);
      };
      var send = XMLHttpRequest.prototype.send;
      XMLHttpRequest.prototype.send = function () {
        var xhr = this;
        var captured = false;
        function capture() {
          if (captured) return;
          captured = true;
          var body = '';
          try { if (!xhr.responseType || xhr.responseType === 'text') body = xhr.responseText; }
          catch (e) {}
          push(xhr.__webScanURL, xhr.status, body);
        }
        // readyState 4 与 onload 同批触发，比 loadend 更早；两个都挂上去，
        // 避免 wait_selector 命中时响应体还没被记录。
        xhr.addEventListener('readystatechange', function () {
          if (xhr.readyState === 4) capture();
        });
        xhr.addEventListener('loadend', capture);
        return send.apply(this, arguments);
      };
      if (window.fetch) {
        var originalFetch = window.fetch;
        window.fetch = function () {
          var input = arguments[0];
          var url = (input && input.url) ? input.url : input;
          return originalFetch.apply(this, arguments).then(function (response) {
            try {
              response.clone().text().then(function (body) {
                push(url, response.status, body);
              });
            } catch (e) {}
            return response;
          });
        };
      }
    })();
    """

    /// 渲染任务串行化（复用同一个 WKWebView；NSLock 不能跨 await 持有）。
    private let serial = AsyncSemaphore(value: 1)

    /// 网络层拦截：广告/tracker + SSRF 目标（私网、回环、云元数据、file://）。
    ///
    /// `decidePolicyForNavigationAction` 只覆盖主文档/iframe 导航，拦不住页面
    /// 脚本主动发起的 XHR/fetch/子资源；那些必须由 ContentRuleList 在网络层拦掉。
    /// 规则统一用 `[.]` 字符类代替转义点号，避免 Swift/JSON/正则三层转义出错。
    /// 测试豁免回环时编译另一套规则（放行 127.0.0.1/localhost/::1）。
    static func blockRulesJSON(allowLoopback: Bool) -> String {
        let loopback = allowLoopback ? "" : """
          {"trigger":{"url-filter":"^https?://127[.]"},"action":{"type":"block"}},
          {"trigger":{"url-filter":"^https?://localhost"},"action":{"type":"block"}},
          {"trigger":{"url-filter":"^https?://0[.]0[.]0[.]0"},"action":{"type":"block"}},
          {"trigger":{"url-filter":"^https?://[[]::1[]]"},"action":{"type":"block"}},
        """
        return """
        [
          {"trigger":{"url-filter":"^file://"},"action":{"type":"block"}},
          {"trigger":{"url-filter":"^https?://10[.]"},"action":{"type":"block"}},
          {"trigger":{"url-filter":"^https?://192[.]168[.]"},"action":{"type":"block"}},
          {"trigger":{"url-filter":"^https?://169[.]254[.]"},"action":{"type":"block"}},
          {"trigger":{"url-filter":"^https?://172[.]1[6-9][.]"},"action":{"type":"block"}},
          {"trigger":{"url-filter":"^https?://172[.]2[0-9][.]"},"action":{"type":"block"}},
          {"trigger":{"url-filter":"^https?://172[.]3[01][.]"},"action":{"type":"block"}},
          {"trigger":{"url-filter":"^https?://100[.]6[4-9][.]"},"action":{"type":"block"}},
          {"trigger":{"url-filter":"^https?://100[.]7[0-9][.]"},"action":{"type":"block"}},
          {"trigger":{"url-filter":"^https?://100[.]1[01][0-9][.]"},"action":{"type":"block"}},
          {"trigger":{"url-filter":"^https?://100[.]12[0-7][.]"},"action":{"type":"block"}},
          {"trigger":{"url-filter":"^https?://[[]fe80"},"action":{"type":"block"}},
          {"trigger":{"url-filter":"^https?://[[]fc"},"action":{"type":"block"}},
          {"trigger":{"url-filter":"^https?://[[]fd"},"action":{"type":"block"}},
          {"trigger":{"url-filter":"^https?://metadata[.]google[.]internal"},"action":{"type":"block"}},
        \(loopback)
          {"trigger":{"url-filter":".*ajax[.]googleapis[.]com.*"},"action":{"type":"block"}},
          {"trigger":{"url-filter":".*fonts[.]googleapis[.]com.*"},"action":{"type":"block"}},
          {"trigger":{"url-filter":".*fonts[.]gstatic[.]com.*"},"action":{"type":"block"}},
          {"trigger":{"url-filter":".*google-analytics[.]com.*"},"action":{"type":"block"}},
          {"trigger":{"url-filter":".*googletagmanager[.]com.*"},"action":{"type":"block"}},
          {"trigger":{"url-filter":".*doubleclick[.]net.*"},"action":{"type":"block"}},
          {"trigger":{"url-filter":".*connect[.]facebook[.]net.*"},"action":{"type":"block"}},
          {"trigger":{"url-filter":".*scorecardresearch[.]com.*"},"action":{"type":"block"}}
        ]
        """
    }

    /// `WKContentRuleList` 不是 Sendable；用 unchecked box 跨 actor 边界，
    /// 对象本身只在 MainActor 内交给 WKWebViewConfiguration。
    private final class RuleBox: @unchecked Sendable {
        let list: WKContentRuleList?
        init(_ list: WKContentRuleList?) { self.list = list }
    }

    private actor RuleCache {
        static let shared = RuleCache()
        private var lists: [Bool: RuleBox] = [:]
        func cached(_ key: Bool) -> RuleBox? { lists[key] }
        func store(_ box: RuleBox, for key: Bool) { lists[key] = box }
    }

    /// 规则编译必须在 MainActor 上发起：`WKContentRuleListStore.default()` 被
    /// 标注为 @MainActor，从 TaskGroup 的非隔离子任务里隐式异步调用会触发
    /// actor isolation 断言 —— 实测 web_scan 自动渲染时整个进程直接崩溃。
    /// 用 continuation 让出主线程，WebKit 的回调仍能在主 runloop 上投递回来。
    @MainActor
    private static func compileRuleList(allowLoopback: Bool) async -> RuleBox {
        await withCheckedContinuation { cont in
            WKContentRuleListStore.default().compileContentRuleList(
                forIdentifier: "base-agent-webrender-"
                    + (allowLoopback ? "loopback" : "strict"),
                encodedContentRuleList: Self.blockRulesJSON(
                    allowLoopback: allowLoopback)) { list, _ in
                cont.resume(returning: RuleBox(list))
            }
        }
    }

    /// 异步编译 + 进程内缓存。
    ///
    /// **绝不能用 `DispatchSemaphore.wait()` 等编译结果**：回调经主 runloop 投递，
    /// 同步阻塞主线程会让回调永远不回来（实测阻塞版恒定超时、规则集恒为 nil）。
    /// `await` 不阻塞主线程，编译成功或失败都会回调，因此不再需要「与超时任务
    /// 赛跑」——那个 TaskGroup 既会隐式等待挂起的子任务，又是本次崩溃的来源。
    private static func rules(allowLoopback: Bool) async -> RuleBox {
        let cache = RuleCache.shared
        if let cached = await cache.cached(allowLoopback) { return cached }
        let box = await compileRuleList(allowLoopback: allowLoopback)
        if box.list != nil {
            await cache.store(box, for: allowLoopback)
        } else {
            AgentLog.warning("WKContentRuleList 编译失败：页内 XHR 网络层防护降级，"
                + "导航策略仍然生效")
        }
        return box
    }

    /// 页面加载软超时：被墙/长轮询资源会让 didFinish 永不到来，
    /// 到点后停止网络等待、提取已解析的 DOM。
    private static let loadSoftTimeoutNanos: UInt64 = 20_000_000_000
    @MainActor private var webView: WKWebView?
    @MainActor private var dialogBox: ScriptDialogBox?
    @MainActor private var navBox: NavigationBox?
    /// 当前 webView 装配的 ContentRuleList 所用的回环豁免标志。
    @MainActor private var configuredAllowLoopback = false

    /// 页面 alert/confirm/prompt 的静默实现，避免弹窗阻塞渲染。
    @MainActor
    private final class ScriptDialogBox: NSObject, WKUIDelegate {
        func webView(_ webView: WKWebView,
                     runJavaScriptAlertPanelWithMessage message: String,
                     initiatedByFrame frame: WKFrameInfo,
                     completionHandler: @escaping () -> Void) { completionHandler() }

        func webView(_ webView: WKWebView,
                     runJavaScriptConfirmPanelWithMessage message: String,
                     initiatedByFrame frame: WKFrameInfo,
                     completionHandler: @escaping (Bool) -> Void) { completionHandler(false) }

        func webView(_ webView: WKWebView,
                     runJavaScriptTextInputPanelWithPrompt prompt: String,
                     defaultText: String?, initiatedByFrame frame: WKFrameInfo,
                     completionHandler: @escaping (String?) -> Void) { completionHandler(nil) }
    }

    /// 导航回调 → continuation。finish 幂等：二次导航/错误不再重复 resume。
    @MainActor
    private final class NavigationBox: NSObject, WKNavigationDelegate {
        var continuation: CheckedContinuation<Void, Error>?
        private var finished = false
        /// 与工具层同一套 URL 策略；仅本地 mini server 测试会豁免回环。
        var allowLoopbackForTesting = false
        /// 页内被拦下的导航次数（诊断用）。
        var blockedNavigations = 0

        /// 复用前复位：上次任务的 finished 标志不得挡住新页面的回调。
        func reset() {
            finished = false
            continuation = nil
        }

        func finish(_ result: Result<Void, Error>) {
            guard !finished, let continuation else { return }
            finished = true
            self.continuation = nil
            continuation.resume(with: result)
        }

        private func convert(_ error: Error) -> Error {
            let ns = error as NSError
            if ns.domain == NSURLErrorDomain, ns.code == NSURLErrorCancelled {
                return AgentError.cancelled
            }
            return error
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            finish(.success(()))
        }

        func webView(_ webView: WKWebView,
                     didFail navigation: WKNavigation!, withError error: Error) {
            finish(.failure(convert(error)))
        }

        func webView(_ webView: WKWebView,
                     didFailProvisionalNavigation navigation: WKNavigation!,
                     withError error: Error) {
            finish(.failure(convert(error)))
        }

        /// 页面内跳转（JS 重定向、iframe、表单提交、服务端重定向）也必须过 SSRF
        /// 校验：入口 URL 安全 ≠ 页面后续请求的 URL 安全。
        func webView(_ webView: WKWebView,
                     decidePolicyFor navigationAction: WKNavigationAction,
                     decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            guard let url = navigationAction.request.url,
                  WebURLPolicy.rejectionReason(
                      for: url,
                      allowLoopbackForTesting: allowLoopbackForTesting) == nil else {
                blockedNavigations += 1
                decisionHandler(.cancel)
                return
            }
            decisionHandler(.allow)
        }
    }

    /// 等待整页渲染完成并提取结果。`timeout` 覆盖加载 + 等待 + 脚本全流程。
    ///
    /// - `waitForTextGrowthFrom`: 传入静态 HTML 的正文长度时，等待页面正文比它
    ///   明显增长（SPA 数据落地），最多 `maxContentWait` 秒；传 0 也会等待，
    ///   用于静态页面是空壳的情况。nil 表示不做正文增长等待。
    func render(url: URL,
                waitMS: Int,
                waitSelector: String?,
                script: String?,
                wantHTML: Bool,
                wantScreenshot: Bool,
                timeout: TimeInterval,
                waitForTextGrowthFrom: Int? = nil,
                maxContentWait: TimeInterval = 8) async throws -> Outcome {
        guard await serial.acquire() else { throw AgentError.cancelled }
        defer { serial.release() }
        // 编译规则时让出主线程（回调依赖主 runloop），完成后再进 renderOnMain。
        let rules = await Self.rules(allowLoopback: Self.localhostAllowedForTesting)
        return try await withTimeout(timeout, "渲染") {
            try await self.renderOnMain(
                url: url, waitMS: waitMS, waitSelector: waitSelector, script: script,
                wantHTML: wantHTML, wantScreenshot: wantScreenshot,
                waitForTextGrowthFrom: waitForTextGrowthFrom,
                maxContentWait: maxContentWait, rules: rules)
        }
    }

    /// 正文增长是否达到「数据已加载」的阈值。
    /// 基线为 0（空壳）时，出现 200 字符即视为有内容；否则要求至少 +200 字符
    /// 或 +10%，避免动态时间戳/广告等小波动导致过早提取。
    static func textGrowthReached(baseline: Int, current: Int) -> Bool {
        guard baseline > 0 else { return current >= 200 }
        return current >= baseline + max(200, baseline / 10)
    }

    /// 会话结束时清理：停止加载并释放 WKWebView（可在任意线程调用）。
    func reset() {
        Task { @MainActor [weak self] in
            self?.webView?.stopLoading()
            self?.webView = nil
            self?.dialogBox = nil
            self?.navBox = nil
        }
    }

    // MARK: - 内部

    @MainActor
    private func renderOnMain(url: URL,
                              waitMS: Int,
                              waitSelector: String?,
                              script: String?,
                              wantHTML: Bool,
                              wantScreenshot: Bool,
                              waitForTextGrowthFrom: Int?,
                              maxContentWait: TimeInterval,
                              rules: RuleBox) async throws -> Outcome {
        let webView = try ensureWebView(rules: rules.list)

        // 1. 加载页面：didFinish 或软超时取先到者。软超时到点 stopLoading
        //    （停止网络等待、保留已解析 DOM）后继续提取；取消时同样 stopLoading
        //    触发 didFail(cancelled)，continuation 不会悬挂。
        try await withThrowingTaskGroup(of: Void.self) { group in
            // 子任务显式 @MainActor：TaskGroup 的 addTask 不继承调用方隔离，
            // 落到线程池上再 assumeIsolated 会触发断言崩溃（EXC_BREAKPOINT）。
            group.addTask { @MainActor in
                try await withTaskCancellationHandler {
                    try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                        self.navBox?.reset()
                        self.navBox?.continuation = cont
                        _ = webView.load(URLRequest(url: url))
                    }
                } onCancel: {
                    Task { @MainActor in webView.stopLoading() }
                }
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: Self.loadSoftTimeoutNanos)
                await MainActor.run { webView.stopLoading() }
                try? await Task.sleep(nanoseconds: 300_000_000) // 留给 didFail resume
            }
            _ = try await group.next()
            group.cancelAll()
        }

        // 2. didFinish 后等待 SPA 异步渲染：固定等待 + 可选 selector 轮询
        //    + 正文增长轮询（比固定 sleep 更能适应慢接口站点）。
        if waitMS > 0 {
            try? await Task.sleep(nanoseconds: UInt64(min(waitMS, 30_000)) * 1_000_000)
        }
        if let selector = waitSelector?.trimmingCharacters(in: .whitespacesAndNewlines),
           !selector.isEmpty {
            try await waitForSelector(selector, webView: webView, maxWait: 10)
        }
        if let baseline = waitForTextGrowthFrom {
            try await waitForTextGrowth(webView: webView,
                                        baseline: baseline,
                                        maxWait: maxContentWait)
        }

        // 3. 提取正文（innerText 为空时回退 textContent，覆盖部分隐藏布局）。
        var outcome = Outcome()
        outcome.text = (try? await withTimeout(15, "正文提取") {
            try await self.evaluate(
                "(document.body ? document.body.innerText : '') "
                    + "|| (document.body ? document.body.textContent : '')",
                webView: webView) as? String
        }) ?? ""

        // 3b. 取回页面自己发出的数据请求（含运行时拼好的参数）。
        let capturedRaw: String? = try? await withTimeout(10, "XHR 捕获") {
            try await self.evaluate("JSON.stringify(window.__webScanRequests || [])",
                                    webView: webView) as? String
        }
        if let raw = capturedRaw,
           let data = raw.data(using: .utf8),
           let rows = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]] {
            outcome.capturedRequests = rows.compactMap { row in
                guard let url = row["url"] as? String, !url.isEmpty else { return nil }
                return CapturedRequest(url: url,
                                       status: row["status"] as? Int ?? 0,
                                       body: row["body"] as? String ?? "")
            }
        }

        // 4. 用户脚本（返回值需为字符串/数字等基础类型）。
        if let script, !script.isEmpty {
            let result = try await withTimeout(30, "脚本执行") {
                try await self.evaluate(script, webView: webView)
            }
            outcome.scriptResult = Self.stringify(result)
        }

        // 5. 渲染后 HTML。
        if wantHTML {
            outcome.html = (try? await withTimeout(15, "HTML 提取") {
                try await self.evaluate(
                    "document.documentElement.outerHTML", webView: webView) as? String
            }) ?? nil
        }

        // 6. 截图（尽力而为，失败不影响其它结果）。
        if wantScreenshot {
            outcome.screenshotPath = try? await withTimeout(15, "截图") {
                try await self.takeSnapshot(webView: webView)
            }
        }
        return outcome
    }

    @MainActor
    private func ensureWebView(rules: WKContentRuleList?) throws -> WKWebView {
        let allowLoopback = Self.localhostAllowedForTesting
        if let webView, configuredAllowLoopback == allowLoopback {
            navBox?.allowLoopbackForTesting = allowLoopback
            return webView
        }
        // 豁免标志变了：规则集属于 configuration，创建后不可改，只能重建。
        webView?.stopLoading()
        self.webView = nil

        let configuration = WKWebViewConfiguration()
        // default()：跨工具调用保留 cookie/登录态（与外部浏览器桥行为一致）。
        configuration.websiteDataStore = .default()
        if let rules {
            configuration.userContentController.add(rules)
        }
        configuration.userContentController.addUserScript(
            WKUserScript(source: Self.captureScript,
                         injectionTime: .atDocumentStart, forMainFrameOnly: true))
        let webView = WKWebView(frame: CGRect(x: 0, y: 0, width: 1440, height: 900),
                                configuration: configuration)
        let nav = NavigationBox()
        nav.allowLoopbackForTesting = allowLoopback
        let dialog = ScriptDialogBox()
        webView.navigationDelegate = nav
        webView.uiDelegate = dialog
        navBox = nav
        dialogBox = dialog
        self.webView = webView
        configuredAllowLoopback = allowLoopback
        return webView
    }

    @MainActor
    private func waitForSelector(_ selector: String, webView: WKWebView,
                                 maxWait: TimeInterval) async throws {
        let literal = Self.jsStringLiteral(selector)
        let deadline = Date().addingTimeInterval(maxWait)
        while true {
            if Task.isCancelled { throw AgentError.cancelled }
            let found = (try? await evaluate("!!document.querySelector(\(literal))",
                                             webView: webView) as? Bool) ?? false
            if found { return }
            if Date() >= deadline { return } // 超时不报错：返回已有内容并继续
            try await Task.sleep(nanoseconds: 150_000_000)
        }
    }

    /// 轮询正文长度，直到相对静态快照明显增长或超时。
    /// 超时不报错：返回当前已渲染内容，交回调用方决定是否够用。
    @MainActor
    private func waitForTextGrowth(webView: WKWebView,
                                   baseline: Int,
                                   maxWait: TimeInterval) async throws {
        let deadline = Date().addingTimeInterval(max(0, maxWait))
        while true {
            if Task.isCancelled { throw AgentError.cancelled }
            let current: Int
            if let value = try? await evaluate(
                "document.body ? document.body.innerText.length : 0",
                webView: webView),
               let number = value as? NSNumber {
                current = number.intValue
            } else {
                current = 0
            }
            if Self.textGrowthReached(baseline: baseline, current: current) { return }
            if Date() >= deadline { return }
            try await Task.sleep(nanoseconds: 250_000_000)
        }
    }

    @MainActor
    private func evaluate(_ script: String, webView: WKWebView) async throws -> Any? {
        try await withCheckedThrowingContinuation { cont in
            webView.evaluateJavaScript(script) { result, error in
                if let error {
                    cont.resume(throwing: error)
                } else {
                    cont.resume(returning: result)
                }
            }
        }
    }

    @MainActor
    private func takeSnapshot(webView: WKWebView) async throws -> String {
        let configuration = WKSnapshotConfiguration()
        configuration.rect = webView.bounds
        let image = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<NSImage, Error>) in
            webView.takeSnapshot(with: configuration) { image, error in
                if let image {
                    cont.resume(returning: image)
                } else {
                    cont.resume(throwing: error ?? AgentError.parse("截图失败"))
                }
            }
        }
        guard let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:]) else {
            throw AgentError.parse("截图编码失败")
        }
        let path = NSTemporaryDirectory() + "base-agent-web-shot-\(UUID().uuidString.prefix(8)).png"
        try png.write(to: URL(fileURLWithPath: path))
        return path
    }

    /// 超时包装：到点即抛错；body 内的继续执行不会悬挂（底层各有超时/取消）。
    private func withTimeout<T: Sendable>(_ seconds: TimeInterval, _ label: String,
                                          _ body: @escaping @Sendable () async throws -> T) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask { try await body() }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                throw AgentError.parse("\(label)超时（\(Int(seconds))s）")
            }
            guard let first = try await group.next() else {
                throw AgentError.parse("\(label)失败")
            }
            group.cancelAll()
            return first
        }
    }

    private static func stringify(_ value: Any?) -> String? {
        guard let value else { return nil }
        switch value {
        case let string as String: return string
        case let number as NSNumber:
            if number.isBoolean { return number.boolValue ? "true" : "false" }
            return number.stringValue
        default:
            if JSONSerialization.isValidJSONObject([value]),
               let data = try? JSONSerialization.data(withJSONObject: [value]),
               let text = String(data: data, encoding: .utf8) {
                return String(text.dropFirst().dropLast()) // 去掉包装数组
            }
            return String(describing: value)
        }
    }

    /// 把任意字符串编码为 JS 字符串字面量（用于把 CSS 选择器安全嵌入脚本）。
    private static func jsStringLiteral(_ value: String) -> String {
        if let data = try? JSONEncoder().encode(value),
               let literal = String(data: data, encoding: .utf8) {
            return literal
        }
        return "\"\(value.replacingOccurrences(of: "\"", with: "\\\""))\""
    }
}

extension NSNumber {
    /// JS 桥里 Bool 也是 NSNumber；借 CFBooleanRef 区分布尔与数值。
    var isBoolean: Bool {
        CFGetTypeID(self) == CFBooleanGetTypeID()
    }
}
