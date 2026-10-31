import Foundation
import Darwin

/// 异步互斥锁：让同一时刻只有一个 embedding/OCR 调用进入模型，
/// 避免多个文件并发时争抢同一个 MLX/ONNX 实例。
final class AsyncGate: @unchecked Sendable {
    private let lock = NSLock()
    private var acquired = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        lock.lock()
        if !acquired {
            acquired = true
            lock.unlock()
            return
        }
        await withCheckedContinuation { continuation in
            lock.lock()
            // 快速检查与注册之间存在窗口：若此时正好被 release，acquired 已
            // 归还为 false。必须在这里领取，否则会漏唤醒、永久等待。
            if !acquired {
                acquired = true
                lock.unlock()
                continuation.resume()
                return
            }
            waiters.append(continuation)
            lock.unlock()
        }
    }

    func release() {
        lock.lock()
        if let waiter = waiters.first {
            waiters.removeFirst()
            lock.unlock()
            waiter.resume()
        } else {
            acquired = false
            lock.unlock()
        }
    }

    func withLock<T>(_ body: () async throws -> T) async rethrows -> T {
        await acquire()
        do {
            let value = try await body()
            release()
            return value
        } catch {
            release()
            throw error
        }
    }
}

/// 计数信号量：用于限制跨文件的全局 CPU 密集任务并发
/// （例如 Apple Vision OCR），避免文件级并发叠加后过度争抢核心。
/// 等待许可时响应任务取消，避免取消后的同步仍被挂在这里。
final class AsyncSemaphore: @unchecked Sendable {
    private struct Waiter {
        let id: UInt64
        let continuation: CheckedContinuation<Bool, Never>
    }

    private let lock = NSLock()
    private var available: Int
    private var waiters: [Waiter] = []
    private var nextID: UInt64 = 0

    init(value: Int) {
        self.available = max(1, value)
    }

    /// 返回 false 表示等待期间任务被取消，未获得许可（不应 release）。
    func acquire() async -> Bool {
        lock.lock()
        if Task.isCancelled {
            lock.unlock()
            return false
        }
        if available > 0 {
            available -= 1
            lock.unlock()
            return true
        }
        let id = nextID
        nextID &+= 1
        lock.unlock()

        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                lock.lock()
                if Task.isCancelled {
                    lock.unlock()
                    continuation.resume(returning: false)
                    return
                }
                // 与 AsyncGate 相同的窗口：快速检查后、注册前可能正好释放了
                // 一个许可。此时直接领取，避免许可被归还却无人唤醒。
                if available > 0 {
                    available -= 1
                    lock.unlock()
                    continuation.resume(returning: true)
                    return
                }
                waiters.append(Waiter(id: id, continuation: continuation))
                lock.unlock()
            }
        } onCancel: {
            lock.lock()
            if let index = waiters.firstIndex(where: { $0.id == id }) {
                let waiter = waiters.remove(at: index)
                lock.unlock()
                waiter.continuation.resume(returning: false)
            } else {
                lock.unlock()
            }
        }
    }

    func release() {
        lock.lock()
        if let waiter = waiters.first {
            waiters.removeFirst()
            lock.unlock()
            waiter.continuation.resume(returning: true)
        } else {
            available += 1
            lock.unlock()
        }
    }

    func withPermit<T>(_ body: () async throws -> T) async throws -> T {
        guard await acquire() else { throw AgentError.cancelled }
        // 许可刚被 release 唤醒时任务可能已被取消：直接归还并退出。
        if Task.isCancelled {
            release()
            throw AgentError.cancelled
        }
        defer { release() }
        return try await body()
    }
}


/// 进程当前物理内存占用（包含 MLX 权重、KV/buffer 与 ONNX 张量）。
enum ProcessMemory {
    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / 1_048_576
    }
}

/// 多文件并发 ingest 时共享的 embedding 调度器：
/// - 按底层 provider 的并发能力限流（MLX 串行，ONNX 可 2 路并发）；
/// - 按吞吐与内存自动调整子批大小（默认 2...32，初始 16，步进 2）；
/// - 内存超过上限时立即下调，吞吐下降时反向探测，持续寻找当前内存下的最优值。
public final class AdaptiveEmbeddingScheduler: EmbeddingProvider, EmbeddingTokenCounting, @unchecked Sendable {

    public struct Options: Sendable {
        public var minBatchSize: Int
        public var maxBatchSize: Int
        public var initialBatchSize: Int
        public var step: Int
        /// 子批调参的内存上限（MB），默认 6GB。
        public var memoryLimitMB: Double
        /// 吞吐变化容差（0.03 = ±3%）。变化小于该比例视为测量噪声，不调整子批。
        public var throughputTolerance: Double
        /// 每个尺寸至少观察的批次数或秒数，避免测量噪声导致抖动。
        public var probeBatches: Int
        public var probeSeconds: Double
        /// 单批 token 预算的下限（默认 4096）。批大小同时受条数上限约束，
        /// 实际以先到者为准；长切片由 token 预算决定，短切片由条数决定。
        public var tokenBudgetMin: Int
        /// 单批 token 预算的上限（默认 8192）。
        public var tokenBudgetMax: Int
        /// 每次同步的起始 token 预算（默认 6144）。
        public var initialTokenBudget: Int
        /// 每轮上探/下探的 token 预算步进（默认 512）。
        public var tokenBudgetStep: Int
        /// 单次 embedDocuments 调用内，每隔多少条文本释放一次全局信号量。
        /// 窗口之间的让出让多个文件交替获得模型时间，避免先到的长文件
        /// 独占模型直到全部切片完成。默认 32，最小 1。
        public var windowTexts: Int

        public init(minBatchSize: Int = 2,
                    maxBatchSize: Int = 32,
                    initialBatchSize: Int = 16,
                    step: Int = 2,
                    memoryLimitMB: Double = 6 * 1_024,
                    throughputTolerance: Double = 0.03,
                    probeBatches: Int = 2,
                    probeSeconds: Double = 2.0,
                    tokenBudgetMin: Int = 4_096,
                    tokenBudgetMax: Int = 8_192,
                    initialTokenBudget: Int = 6_144,
                    tokenBudgetStep: Int = 512,
                    windowTexts: Int = 32) {
            let normalizedMin = max(1, minBatchSize)
            let normalizedMax = max(normalizedMin, maxBatchSize)
            self.minBatchSize = normalizedMin
            self.maxBatchSize = normalizedMax
            self.initialBatchSize = min(max(normalizedMin, initialBatchSize), normalizedMax)
            self.step = max(1, step)
            self.memoryLimitMB = max(256, memoryLimitMB)
            self.throughputTolerance = max(0.005, min(0.5, throughputTolerance))
            self.probeBatches = max(1, probeBatches)
            self.probeSeconds = max(0.1, probeSeconds)
            let normalizedBudgetMin = max(512, tokenBudgetMin)
            let normalizedBudgetMax = max(normalizedBudgetMin, tokenBudgetMax)
            self.tokenBudgetMin = normalizedBudgetMin
            self.tokenBudgetMax = normalizedBudgetMax
            self.initialTokenBudget = min(max(normalizedBudgetMin, initialTokenBudget),
                                          normalizedBudgetMax)
            self.tokenBudgetStep = max(128, tokenBudgetStep)
            self.windowTexts = max(1, windowTexts)
        }
    }

    /// embedding 运行指标：tokens/s 与进程 footprint 峰值。
    public struct Metrics: Sendable {
        public var batches: Int = 0
        public var texts: Int = 0
        public var tokens: Int = 0
        public var seconds: Double = 0
        public var peakMemoryMB: Double = 0
        /// 当前生效的 token 预算（采样时值）。
        public var tokenBudget: Int = 0
        /// 单批最多文本条数（采样时值）。
        public var maxBatchTexts: Int = 0
        /// 实际观察到的最大单批条数。
        public var observedMaxBatchTexts: Int = 0

        public init() {}

        public var tokensPerSecond: Double {
            seconds > 0 ? Double(tokens) / seconds : 0
        }
    }

    private let base: EmbeddingProvider
    private let options: Options
    private let log: (String) -> Void
    /// 并发闸门：按底层模型的并发能力限流（ONNX 可 2 路并发 run、
    /// MLX 串行）。信号量在等待时响应取消（AsyncGate 不会）。
    private let semaphore: AsyncSemaphore
    private let stateLock = NSLock()

    // stateLock 保护以下状态
    /// 单批文本条数上限（与 token 预算共同生效）。
    private var batchSize: Int
    /// 单批 token 预算。
    private var tokenBudget: Int
    private var windowTokens = 0
    private var windowSeconds = 0.0
    private var windowBatches = 0
    private var windowStarted = Date()
    /// 吞吐统一用 token/秒，短切片与长切片可比。
    private var lastThroughput = 0.0
    private var direction = 1   // +1 上探，-1 下探
    private var lastAdjustmentLog = Date.distantPast

    // 累计指标（跨文件、跨调用）
    private var totalBatches = 0
    private var totalTexts = 0
    private var totalTokens = 0
    private var totalSeconds = 0.0
    private var peakMemoryMB = 0.0
    private var observedMaxBatchTexts = 0
    private var lastBatchTexts = 0

    public var identifier: String { base.identifier }
    public var dimension: Int { base.dimension }

    /// 底层 provider 的二值签名位宽；nil 表示不支持。
    public var binaryDimension: Int? {
        guard let signing = base as? EmbeddingBinarySigning,
              signing.binaryDimension > 0 else { return nil }
        return signing.binaryDimension
    }

    /// 供完成日志展示最近一次实际子批的文本条数。
    public var currentBatchSize: Int {
        stateLock.lock()
        defer { stateLock.unlock() }
        return lastBatchTexts > 0 ? lastBatchTexts : batchSize
    }

    /// 当前生效的 token 预算。
    public var currentTokenBudget: Int {
        stateLock.lock()
        defer { stateLock.unlock() }
        return tokenBudget
    }

    /// 累计 tokens/s、峰值内存与批大小快照。
    public func metricsSnapshot() -> Metrics {
        stateLock.lock()
        defer { stateLock.unlock() }
        var metrics = Metrics()
        metrics.batches = totalBatches
        metrics.texts = totalTexts
        metrics.tokens = totalTokens
        metrics.seconds = totalSeconds
        metrics.peakMemoryMB = peakMemoryMB
        metrics.tokenBudget = tokenBudget
        metrics.maxBatchTexts = batchSize
        metrics.observedMaxBatchTexts = observedMaxBatchTexts
        return metrics
    }

    public init(base: EmbeddingProvider,
                options: Options = Options(),
                log: @escaping (String) -> Void = { AgentLog.info($0) }) {
        self.base = base
        self.options = options
        self.log = log
        self.semaphore = AsyncSemaphore(value: max(1, base.maxConcurrentRuns))
        self.batchSize = max(options.minBatchSize,
                             min(options.initialBatchSize, options.maxBatchSize))
        self.tokenBudget = options.initialTokenBudget
    }

    // MARK: - EmbeddingProvider

    public func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        try await embedDocuments(texts, progress: nil)
    }

    /// 用于流水线内的长文档：按窗口分批获取模型，窗口之间让出信号量。
    ///
    /// 一次调用可能包含整个文档的切片。若全程持有信号量，先到的长文档会
    /// 独占模型直到自己全部完成，其他文件只能等待，整体完成时间被最长的
    /// 单文件绑定。按 `options.windowTexts` 切窗口后，每个窗口单独申请许可，
    /// 窗口之间 `Task.yield()`，公平信号量会把模型让给其他等待的文件。
    ///
    /// 切窗口之前先按全篇 token 长度降序排序：同一窗口内的切片长度相近，
    /// 可减少 padding 放大，也避免自适应批次因吞吐下降而不断缩小。
    /// 向量仍按原始下标写回，进度回调按整次调用的条数推进。
    public func embedDocuments(
        _ texts: [String],
        progress: ((Int, Int) -> Void)?
    ) async throws -> [[Float]] {
        try await runEmbedding(texts, progress: progress,
                               includeSignatures: false).vectors
    }

    /// 入库用：底层支持二值时一次返回主向量与签名；不支持时返回 nil。
    public func embedDocumentsWithSignatures(
        _ texts: [String],
        progress: ((Int, Int) -> Void)? = nil
    ) async throws -> BinaryEmbeddingOutput? {
        guard binaryDimension != nil else { return nil }
        return try await runEmbedding(texts,
                                      progress: progress,
                                      includeSignatures: true)
    }

    /// 向量与签名的唯一调度路径。
    ///
    /// `includeSignatures == false` 时只走底层 `embedDocuments`，保持原有
    /// 行为；为 true 时走底层签名接口。两条路径共用同一套 token 计数、
    /// 全局排序、窗口切分与自适应组批，避免维护第二套调度算法。
    private func runEmbedding(
        _ texts: [String],
        progress: ((Int, Int) -> Void)?,
        includeSignatures: Bool
    ) async throws -> BinaryEmbeddingOutput {
        guard !texts.isEmpty else {
            return BinaryEmbeddingOutput(vectors: [], signatures: [])
        }

        // 全篇只创建一次计数器、只计数一次。分词器初始化开销大，按窗口
        // 重复创建会反复初始化；计数期间持有信号量，避免与 embedding
        // 调用并发使用同一个分词器。
        let tokenCounts = try await semaphore.withPermit { () -> [Int] in
            if Task.isCancelled { throw AgentError.cancelled }
            let counter: @Sendable (String) -> Int
            if let provider = base as? EmbeddingTokenCounting,
               let exact = await provider.makeTokenCounter() {
                counter = exact
            } else {
                counter = { EmbeddingTokenEstimator.count($0) }
            }
            return texts.map { max(1, counter($0)) }
        }
        // 大块优先：单条超预算时独占一批，小块随后紧凑填满。
        let order = texts.indices.sorted { lhs, rhs in
            tokenCounts[lhs] == tokenCounts[rhs]
                ? lhs < rhs
                : tokenCounts[lhs] > tokenCounts[rhs]
        }

        let window = max(1, options.windowTexts)
        var result = BinaryEmbeddingOutput(
            vectors: [[Float]](repeating: [], count: texts.count),
            signatures: includeSignatures
                ? [Data](repeating: Data(), count: texts.count)
                : [])
        var processed = 0
        var start = 0
        while start < order.count {
            if Task.isCancelled { throw AgentError.cancelled }
            let end = min(start + window, order.count)
            let slice = Array(order[start..<end])
            let completed = processed
            let output = try await embedWindow(
                slice,
                texts: texts,
                tokenCounts: tokenCounts,
                progress: { done, _ in progress?(completed + done, texts.count) },
                total: texts.count,
                includeSignatures: includeSignatures)
            for (offset, index) in slice.enumerated() {
                result.vectors[index] = output.vectors[offset]
                if includeSignatures {
                    result.signatures[index] = output.signatures[offset]
                }
            }
            processed += slice.count
            start = end
            await Task.yield()
        }
        return result
    }

    /// 单个窗口内的组批与执行。调用方负责按窗口切分并传入全篇 token
    /// 计数，这里只在窗口期间持有信号量，因此其他文件可以在窗口之间
    /// 获得模型时间。
    ///
    /// 窗口内的下标已按长度降序排列，贪心装入「条数上限 + token 预算」
    /// 双约束的子批；当前批装不下的切片跳过并留给下一批，继续用更短的
    /// 切片填满预算，减少 padding 浪费。向量按窗口内位置写回，调用方再
    /// 映射回原始下标。长切片由 token 预算决定批大小，短切片由条数上限
    /// 决定。
    private func embedWindow(
        _ indices: [Int],
        texts: [String],
        tokenCounts: [Int],
        progress: ((Int, Int) -> Void)?,
        total: Int,
        includeSignatures: Bool
    ) async throws -> BinaryEmbeddingOutput {
        try await semaphore.withPermit {
            var result = BinaryEmbeddingOutput(
                vectors: [[Float]](repeating: [], count: indices.count),
                signatures: includeSignatures
                    ? [Data](repeating: Data(), count: indices.count)
                    : [])
            // pending 使用窗口内位置，避免全局下标与局部下标的映射开销。
            var pending = Array(indices.indices)
            var processed = 0
            while !pending.isEmpty {
                if Task.isCancelled { throw AgentError.cancelled }
                stateLock.lock()
                let sizeCap = max(1, batchSize)
                let budget = max(1, tokenBudget)
                stateLock.unlock()

                var selected: [Int] = []
                selected.reserveCapacity(min(sizeCap, pending.count))
                var remaining: [Int] = []
                remaining.reserveCapacity(pending.count)
                var tokens = 0
                for position in pending {
                    let cost = tokenCounts[indices[position]]
                    guard selected.count < sizeCap else {
                        remaining.append(position)
                        continue
                    }
                    // 装不下就先跳过，继续用后面更短的切片填满预算；
                    // 单条超过预算时仍允许独占一批，否则会永远装不进去。
                    if selected.isEmpty || tokens + cost <= budget {
                        selected.append(position)
                        tokens += cost
                    } else {
                        remaining.append(position)
                    }
                }
                pending = remaining
                processed += selected.count
                let batch = selected.map { texts[indices[$0]] }
                let batchStarted = Date()
                let output: BinaryEmbeddingOutput
                if includeSignatures {
                    guard let signing = base as? EmbeddingBinarySigning else {
                        throw AgentError.storage("底层 embedding 不支持二值签名")
                    }
                    output = try await signing.embedDocumentsWithSignatures(batch)
                    guard output.signatures.count == batch.count else {
                        throw AgentError.storage(
                            "embedding 子批二值签名数量异常: "
                            + "\(output.signatures.count) != \(batch.count)")
                    }
                } else {
                    output = BinaryEmbeddingOutput(
                        vectors: try await base.embedDocuments(batch),
                        signatures: [])
                }
                let seconds = Date().timeIntervalSince(batchStarted)
                guard output.vectors.count == batch.count else {
                    throw AgentError.storage(
                        "embedding 子批返回数量异常: \(output.vectors.count) != \(batch.count)")
                }
                for (offset, position) in selected.enumerated() {
                    result.vectors[position] = output.vectors[offset]
                    if includeSignatures {
                        result.signatures[position] = output.signatures[offset]
                    }
                }
                record(texts: batch.count, tokens: tokens, seconds: seconds)
                progress?(processed, total)
            }
            return result
        }
    }

    public func embedQuery(_ text: String) async throws -> [Float] {
        try await semaphore.withPermit {
            try await base.embedQuery(text)
        }
    }

    /// 转发到底层 provider 释放权重。
    public func unload() async {
        await base.unload()
    }

    /// 转发底层模型的窗口预算；底层不支持时视为不限（仍用保守估算器控长）。
    public var maxContentTokens: Int {
        (base as? EmbeddingTokenCounting)?.maxContentTokens ?? Int.max
    }

    /// 转发底层模型的精确 token 计数器；底层不支持时回退到估算器。
    public func makeTokenCounter() async -> (@Sendable (String) -> Int)? {
        guard let provider = base as? EmbeddingTokenCounting else { return nil }
        return await provider.makeTokenCounter()
    }

    // MARK: - 自适应子批

    private func record(texts: Int, tokens: Int, seconds: TimeInterval) {
        stateLock.lock()
        let elapsed = max(seconds, 0.0001)
        windowTokens += tokens
        windowSeconds += elapsed
        windowBatches += 1
        totalBatches += 1
        totalTexts += texts
        totalTokens += tokens
        totalSeconds += elapsed
        lastBatchTexts = texts
        observedMaxBatchTexts = max(observedMaxBatchTexts, texts)
        let memoryMB = ProcessMemory.footprintMB()
        peakMemoryMB = max(peakMemoryMB, memoryMB)
        let enoughBatches = windowBatches >= max(1, options.probeBatches)
        let enoughTime = Date().timeIntervalSince(windowStarted) >= options.probeSeconds
        guard enoughBatches || enoughTime else {
            stateLock.unlock()
            return
        }

        let throughput = Double(windowTokens) / max(windowSeconds, 0.0001)
        let oldSize = batchSize
        let oldBudget = tokenBudget
        var newSize = oldSize
        var newBudget = oldBudget
        var reason: String?

        // 两个旋钮一起调：长切片由 token 预算绑定，短切片由条数绑定；
        // 同时上探/下探可覆盖两种语料，实际批大小取先到者。
        func grow() {
            if newSize < options.maxBatchSize {
                newSize = min(options.maxBatchSize, newSize + options.step)
            }
            if newBudget < options.tokenBudgetMax {
                newBudget = min(options.tokenBudgetMax,
                                newBudget + options.tokenBudgetStep)
            }
        }
        func shrink() {
            if newSize > options.minBatchSize {
                newSize = max(options.minBatchSize, newSize - options.step)
            }
            if newBudget > options.tokenBudgetMin {
                newBudget = max(options.tokenBudgetMin,
                                newBudget - options.tokenBudgetStep)
            }
        }

        if memoryMB > options.memoryLimitMB {
            direction = -1
            shrink()
            if newSize != oldSize || newBudget != oldBudget {
                reason = "内存 \(Int(memoryMB))MB 超限"
            }
        } else if lastThroughput <= 0 {
            grow()
            if newSize != oldSize || newBudget != oldBudget { reason = "首次上探" }
        } else {
            let ratio = throughput / lastThroughput
            let tolerance = options.throughputTolerance
            if ratio >= 1 + tolerance {
                if direction > 0 {
                    grow()
                    if newSize != oldSize || newBudget != oldBudget {
                        reason = "吞吐提升，继续上探"
                    }
                } else {
                    shrink()
                    if newSize != oldSize || newBudget != oldBudget {
                        reason = "下调后吞吐提升，继续下探"
                    }
                }
            } else if ratio <= 1 - tolerance {
                direction = -direction
                if direction > 0 {
                    grow()
                    if newSize != oldSize || newBudget != oldBudget { reason = "下探无提升，回升" }
                } else {
                    shrink()
                    if newSize != oldSize || newBudget != oldBudget { reason = "吞吐下降，下调" }
                }
            }
        }

        lastThroughput = throughput
        windowTokens = 0
        windowSeconds = 0
        windowBatches = 0
        windowStarted = Date()
        batchSize = newSize
        tokenBudget = newBudget
        let changed = newSize != oldSize || newBudget != oldBudget
        // 调参日志节流，避免快速上探时刷屏。
        let shouldLog = changed && Date().timeIntervalSince(lastAdjustmentLog) >= 10
        if shouldLog { lastAdjustmentLog = Date() }
        stateLock.unlock()

        guard changed, shouldLog, let reason else { return }
        log("RAG: embedding 自适应 条数 \(oldSize) → \(newSize)"
            + " / token 预算 \(oldBudget) → \(newBudget)"
            + "（\(reason)，\(Int(throughput)) token/秒，"
            + "内存 \(Int(memoryMB))/\(Int(options.memoryLimitMB)) MB）")
    }
}
