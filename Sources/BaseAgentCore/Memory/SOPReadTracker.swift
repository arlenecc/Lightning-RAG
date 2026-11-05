import Foundation

/// 改进5：记录本次会话已从 `memory/` 全文加载过的 SOP。
///
/// 目的：L1 索引只是「存在性指针」，模型按需 `file_read` 一个 SOP 全文进当前轮，
/// 随后把要点提炼进 `key_info`（每轮自动回注），从而**同一 SOP 只读一次**。
///
/// key 必须是**归一化后的绝对路径**：`file_read` 同时接受
/// `memory/x_sop.md` / `./memory/x_sop.md` / 绝对路径，用原始参数去重会漏判。
/// 只做提示级去重，不永久劝阻重读：`key_info` 是单个可被覆盖的字符串，
/// 一旦要点丢失，模型必须能重新读回全文。
public final class SOPReadTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var loaded: Set<String> = []
    /// 进程内上限，避免长会话无限增长。
    private let capacity: Int

    public init(capacity: Int = 64) {
        self.capacity = max(8, capacity)
    }

    /// 返回 true 表示首次读取。
    public func markIfFirst(_ normalizedPath: String) -> Bool {
        lock.lock(); defer { lock.unlock() }
        if loaded.count >= capacity, !loaded.contains(normalizedPath) {
            loaded.removeAll()   // 简单 LRU 退化：满了就整体重置，规模很小
        }
        return loaded.insert(normalizedPath).inserted
    }

    public func reset() {
        lock.lock(); defer { lock.unlock() }
        loaded.removeAll()
    }

    public var loadedCount: Int {
        lock.lock(); defer { lock.unlock() }
        return loaded.count
    }
}
