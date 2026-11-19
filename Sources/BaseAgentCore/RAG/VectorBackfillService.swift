import Foundation

/// 后台向量回填服务。
///
/// 知识同步时切片正文先行入库（FTS 立即可检索），向量/ColBERT 编码由本服务在
/// 后台异步补齐。设计要点：
/// - 不持有 store/embedding/colbert 的强引用，而是通过闭包在运行时向 RAGEngine
///   索取当前实例——配置切换/close 后这些实例会失效，闭包返回 nil 即自然暂停。
/// - 单一长期 Task 循环；kick() 幂等，stop() 取消并等待退出。
/// - 进度以「pending 表行数」为准（落库），App 重启后再次 kick 即从剩余项续做。
/// - 单条失败只留 pending 下次重试，不中断整批、不影响同步主流程。
public final class VectorBackfillService: @unchecked Sendable {
    /// 运行时依赖快照。所有闭包都是同步取引用（不做耗时工作）。
    public struct Context: Sendable {
        /// 普通向量库待回填数；nil 表示该模式不可用。
        let vectorPendingCount: () -> Int
        /// 取一批普通切片文本并按向量填充；返回本批处理数。
        let backfillVectorBatch: (_ limit: Int, _ cancel: @escaping () -> Bool) async throws -> Int
        /// ColBERT 待回填数；nil 表示极速模式不可用。
        let colbertPendingCount: () -> Int
        /// 回填一批 ColBERT 编码；返回本批处理数。
        let backfillColbertBatch: (_ limit: Int, _ cancel: @escaping () -> Bool) async throws -> Int
        /// 队列排空后回头补扫「有正文但无向量」的切片入队，返回新入队数。
        /// 解决 enqueueMissingVectors 单次 LIMIT 导致大库补不完的问题。
        let rescanMissing: () -> Int
        /// 回填全部排空、循环正常退出时回调：用于触发 ANN 自动构建，
        /// 避免「ANN 在回填完成前构建、低 rowid 后续回填进不了索引」。
        let onDrained: (_ hadWork: Bool) -> Void
        /// 日志回调。
        let log: (String) -> Void

        public init(
            vectorPendingCount: @escaping () -> Int,
            backfillVectorBatch: @escaping (_ limit: Int, _ cancel: @escaping () -> Bool) async throws -> Int,
            colbertPendingCount: @escaping () -> Int,
            backfillColbertBatch: @escaping (_ limit: Int, _ cancel: @escaping () -> Bool) async throws -> Int,
            rescanMissing: @escaping () -> Int = { 0 },
            onDrained: @escaping (_ hadWork: Bool) -> Void = { _ in },
            log: @escaping (String) -> Void
        ) {
            self.vectorPendingCount = vectorPendingCount
            self.backfillVectorBatch = backfillVectorBatch
            self.colbertPendingCount = colbertPendingCount
            self.backfillColbertBatch = backfillColbertBatch
            self.rescanMissing = rescanMissing
            self.onDrained = onDrained
            self.log = log
        }
    }

    private let context: Context
    private let lock = NSLock()
    private var task: Task<Void, Never>?
    private var _running = false
    private var _processedTotal = 0
    private var _lastError: String?
    private var _cachedPending = 0
    private var generation: UInt64 = 0

    public init(context: Context) { self.context = context }

    deinit {
        lock.lock()
        let current = task
        task = nil
        lock.unlock()
        current?.cancel()
    }

    /// 是否有回填任务在运行。
    public var isRunning: Bool {
        lock.lock(); defer { lock.unlock() }
        return _running
    }

    /// 已回填的切片累计数（本次 App 运行期）。
    public var processedTotal: Int {
        lock.lock(); defer { lock.unlock() }
        return _processedTotal
    }

    /// 最近一次回填错误（供诊断）。
    public var lastError: String? {
        lock.lock(); defer { lock.unlock() }
        return _lastError
    }

    /// 最近一次后台循环统计到的待回填数（纯内存，供 status 读取，绝不触发 DB IO）。
    /// 后台未运行时为上次快照；下次循环/kick 会刷新。
    public var pendingCount: Int {
        lock.lock(); defer { lock.unlock() }
        return _cachedPending
    }

    /// 启动后台回填（幂等）。已在运行时不重复启动。
    public func kick() {
        lock.lock()
        if task != nil && _running {
            lock.unlock()
            return
        }
        generation &+= 1
        let myGeneration = generation
        let task = Task { [weak self] in
            guard let self else { return }
            await self.runLoop(generation: myGeneration)
        }
        self.task = task
        _running = true
        _lastError = nil
        lock.unlock()
        // pendingCount 是上次快照，循环首轮会立即刷新；这里只表示「已启动」。
        context.log("[backfill] 后台向量回填已启动")
    }

    /// 请求停止（同步，不等待退出）。供 close 使用；递增 generation 令在途循环失效。
    /// 不在此处清 task：留 runLoop 的 defer 清，使后续 stop() 仍可 await 到旧任务。
    public func requestStop() {
        lock.lock()
        generation &+= 1
        let current = task
        _running = false
        lock.unlock()
        current?.cancel()
    }

    /// 停止后台回填并等待退出。generation 校验避免 await 期间的新 kick 被误清状态。
    public func stop() async {
        lock.lock()
        let current = task
        generation &+= 1
        let myGeneration = generation
        lock.unlock()
        current?.cancel()
        _ = await current?.value
        lock.lock()
        // 只在没有更新的循环接管时才改状态，避免覆盖 await 期间 kick 的新循环。
        if self.generation == myGeneration {
            _running = false
            task = nil
        }
        lock.unlock()
    }

    private func isCurrent(_ g: UInt64) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return generation == g && !Task.isCancelled
    }

    private func runLoop(generation g: UInt64) async {
        defer {
            lock.lock()
            _running = false
            if self.generation == g { task = nil }
            lock.unlock()
            context.log("[backfill] 后台回填循环退出")
        }
        let cancel: () -> Bool = { [weak self] in
            guard let self else { return true }
            return !self.isCurrent(g)
        }
        var idleStreak = 0
        var errorStreak = 0
        var hadWorkThisRun = false
        let maxErrorStreak = 5 // 连续失败达阈值即退出，避免永久重试死循环
        while isCurrent(g) {
            var didWork = false
            var hadPending = false
            do {
                // 普通向量回填优先（embedding 模式）。
                let vectorPending = context.vectorPendingCount()
                if vectorPending > 0 {
                    hadPending = true
                    let n = try await context.backfillVectorBatch(16, cancel)
                    if n > 0 { didWork = true; errorStreak = 0; addProcessed(n, totalPending: vectorPending) }
                }
                // ColBERT 回填（极速模式）。
                let colbertPending = context.colbertPendingCount()
                setCachedPending(vectorPending + colbertPending)
                if colbertPending > 0 {
                    hadPending = true
                    let n = try await context.backfillColbertBatch(8, cancel)
                    if n > 0 { didWork = true; errorStreak = 0; addProcessed(n, totalPending: colbertPending) }
                }
            } catch {
                if !isCurrent(g) { break }
                errorStreak += 1
                setError(error.localizedDescription)
                if errorStreak >= maxErrorStreak {
                    context.log("[backfill] 连续 \(errorStreak) 次回填失败，暂停；错误：\(error.localizedDescription)")
                    break
                }
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
            if didWork {
                hadWorkThisRun = true
                idleStreak = 0
                await Task.yield()
            } else {
                // 有 pending 但处理 0 条（如整批失败被吞/维度不符）：累计错误，
                // 避免「pending>0 却按空闲退出」导致状态与日志都看不到问题。
                if hadPending {
                    errorStreak += 1
                    if errorStreak >= maxErrorStreak {
                        setError("连续 \(errorStreak) 批未取得进展，暂停回填")
                        context.log("[backfill] \(String(describing: lastError ?? ""))，待下次同步/重启重试")
                        break
                    }
                }
                // 队列空了≠全部补完：enqueueMissingVectors 单次有 LIMIT，
                // 大库需多轮。空闲时回头补扫，若又有新项则继续处理。
                let rescanned = context.rescanMissing()
                if rescanned > 0 {
                    idleStreak = 0
                    errorStreak = 0
                    context.log("[backfill] 补扫入队 \(rescanned) 条缺失向量")
                    continue
                }
                idleStreak += 1
                // 6 轮空闲（~6s）才认定时排空：退避中的失败项默认 2s 后可重试，
                // 窗口必须大于退避时间，否则循环提前退出会让退避项再也无权重试。
                if idleStreak >= 6 {
                    // 正常排空：通知上层可安全构建 ANN（此时所有向量已回填）。
                    context.onDrained(hadWorkThisRun)
                    break
                }
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
    }

    private var lastLogTime: TimeInterval = 0
    private func addProcessed(_ n: Int, totalPending: Int) {
        lock.lock(); _processedTotal += n; lock.unlock()
        // 日志节流：每 2 秒最多一条，避免大库回填刷屏拖慢主线程 UI 刷新。
        let now = Date().timeIntervalSince1970
        if now - lastLogTime >= 2 {
            lastLogTime = now
            context.log("[backfill] 已回填 \(processedTotal) 条，剩余约 \(totalPending) 条")
        }
    }

    private func setCachedPending(_ value: Int) {
        lock.lock(); _cachedPending = value; lock.unlock()
    }

    private func setError(_ message: String) {
        lock.lock(); _lastError = message; lock.unlock()
        context.log("[backfill] 回填出错：\(message)")
    }
}
