import Foundation

/// OCR 模型的使用生命周期。
///
/// 每次同步、Markdown 转换或 OCR 测试开始时 `begin()`，结束时 `end()`。
/// 只有最后一个在途任务结束时才调用 `unload`，避免并发任务互相卸载权重；
/// 没有在途任务时 `end()` 不会重复释放。
public actor OCRModelLifecycle {
    private var activeJobs = 0
    private var isReleasing = false
    private var beginWaiters: [CheckedContinuation<Void, Never>] = []
    private let unload: @Sendable () async -> Void

    public init(unload: @escaping @Sendable () async -> Void) {
        self.unload = unload
    }

    /// 标记一个 OCR 任务开始，期间不释放权重。
    ///
    /// 卸载期间 `begin()` 必须等待，否则新任务可能先加载模型，又被仍在
    /// 进行的旧卸载流程清掉。actor 在 `await unload()` 处会重入，因此这里
    /// 用 `isReleasing` + continuation 显式阻塞。
    public func begin() async {
        while isReleasing {
            await withCheckedContinuation { beginWaiters.append($0) }
        }
        activeJobs += 1
    }

    /// 标记任务结束；最后一个任务结束时立即释放 OCR 权重。
    public func end() async {
        guard activeJobs > 0 else { return }
        activeJobs -= 1
        guard activeJobs == 0 else { return }
        await releaseWeights()
    }

    /// 空闲回收计时器使用：只有没有在途 OCR 任务时才释放，避免卸载
    /// 正在识别的实例。已有释放在进行时不重复释放。
    public func unloadIfIdle() async {
        guard activeJobs == 0, !isReleasing else { return }
        await releaseWeights()
    }

    private func releaseWeights() async {
        guard !isReleasing else { return }
        isReleasing = true
        await unload()
        isReleasing = false
        resumeBeginWaiters()
    }

    private func resumeBeginWaiters() {
        let waiters = beginWaiters
        beginWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }
}
