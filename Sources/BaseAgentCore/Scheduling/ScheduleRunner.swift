import Foundation

/// 基于 `DispatchSourceTimer` 的定时任务调度器。
///
/// 单个 timer 每秒扫描一次任务列表，到期任务先落盘 `lastRunAt` 再通过
/// `onTaskDue` 交给上层执行 —— 即使上层执行失败或进程随后退出，
/// 也不会在下一个 tick 重复触发同一个时间点。
public final class ScheduleRunner: @unchecked Sendable {
    /// 任务到期回调（在内部调度队列触发，调用方自行切主线程）。
    public var onTaskDue: (@Sendable (ScheduledTask) -> Void)?
    /// 调度日志（启动/触发/落盘失败）。
    public var onLog: (@Sendable (String) -> Void)?

    private let store: ScheduleStore
    /// 串行队列：tick 不会重入，存储写入也按时间顺序发生。
    private let queue = DispatchQueue(label: "base-agent.schedule", qos: .utility)
    private let lock = NSLock()
    private var timer: DispatchSourceTimer?
    private var running = false

    /// 扫描周期（秒）。1 秒粒度足以覆盖「每天 HH:mm」的分钟级精度。
    public var tickInterval: TimeInterval = 1

    public init(store: ScheduleStore) {
        self.store = store
    }

    deinit {
        stop()
    }

    public var isRunning: Bool {
        lock.lock(); defer { lock.unlock() }
        return running
    }

    /// 启动调度器；重复调用无副作用。
    public func start() {
        lock.lock()
        if running {
            lock.unlock()
            return
        }
        running = true
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + tickInterval,
                       repeating: tickInterval,
                       leeway: .milliseconds(200))
        timer.setEventHandler { [weak self] in self?.tick() }
        timer.resume()
        self.timer = timer
        lock.unlock()
        onLog?("[schedule] 定时任务调度器已启动")
    }

    /// 停止调度器；可重复调用。
    public func stop() {
        lock.lock()
        let timer = self.timer
        self.timer = nil
        let wasRunning = running
        running = false
        lock.unlock()
        guard let timer else { return }
        timer.setEventHandler {}
        timer.cancel()
        if wasRunning { onLog?("[schedule] 定时任务调度器已停止") }
    }

    /// 立即执行一次（工具 `run_now`）：先记录运行时间再回调，
    /// 避免下一个 tick 又把它当成到期任务再跑一遍。
    public func triggerNow(_ task: ScheduledTask) {
        do {
            try store.markRun(id: task.id, at: Date())
        } catch {
            onLog?("[schedule] 无法更新任务「\(task.title)」的运行时间：\(error.localizedDescription)")
        }
        disableIfOneShot(task)
        onLog?("[schedule] 手动触发任务：\(task.title)")
        onTaskDue?(task)
    }

    // MARK: 扫描

    private func tick() {
        let now = Date()
        for task in store.all() where task.isDue(now: now) {
            do {
                try store.markRun(id: task.id, at: now)
            } catch {
                // 落盘失败就不执行：否则会每秒重复触发同一个任务。
                onLog?("[schedule] 任务「\(task.title)」状态写入失败，跳过本次触发："
                    + error.localizedDescription)
                continue
            }
            // 一次性任务执行后自动停用，避免 lastRunAt 更新后仍被当成到期任务。
            disableIfOneShot(task)
            onLog?("[schedule] 定时任务到期：\(task.title)")
            onTaskDue?(task)
        }
    }

    private func disableIfOneShot(_ task: ScheduledTask) {
        guard task.rule.kind == .once else { return }
        do {
            _ = try store.setEnabled(id: task.id, false)
        } catch {
            onLog?("[schedule] 一次性任务「\(task.title)」执行后停用失败："
                + error.localizedDescription)
        }
    }
}
