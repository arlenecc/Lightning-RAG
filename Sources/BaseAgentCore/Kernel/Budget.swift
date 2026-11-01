import Foundation

// MARK: - 配额

/// 各区配额（token）。加起来 = total。
public struct Quota: Sendable, Equatable {
    /// 系统提示 + 记忆索引 + 工作记忆 checkpoint
    public var system: Int
    /// L3 折叠摘要区
    public var earlier: Int
    /// 最近 K 轮原文（含当前轮 L1 裁剪后的工具输出）
    public var raw: Int
    /// 工具 schema
    public var tools: Int
    /// 模型输出 + 估算误差
    public var reserve: Int

    public init(system: Int, earlier: Int, raw: Int, tools: Int, reserve: Int) {
        self.system = system
        self.earlier = earlier
        self.raw = raw
        self.tools = tools
        self.reserve = reserve
    }
}

// MARK: - 上下文预算

/// 把 30k 切成固定配额，并回答「还剩多少可以给工具输出」。
///
/// 内核从不事后问「上下文是不是太大了」，而是先分配、再让每一级压缩在其配额内
/// 开销。配额按比例从 `total` 派生，因此换模型（更大/更小窗口）时无需改代码。
public final class ContextBudget: @unchecked Sendable {
    /// 比例（合计 1.0）。
    private static let ratio: [String: Double] = [
        "system": 0.13,
        "earlier": 0.09,
        "raw": 0.40,
        "tools": 0.17,
        "reserve": 0.21,
    ]

    public let total: Int
    public let quota: Quota
    /// 实测值（每轮刷新），用于判定是否越界。
    private var used: [String: Int] = [
        "system": 0, "earlier": 0, "raw": 0, "tools": 0,
    ]

    public init(total: Int = 30_000) {
        self.total = max(4_000, total)
        let r = Self.ratio
        self.quota = Quota(
            system: Int(Double(self.total) * r["system"]!),
            earlier: Int(Double(self.total) * r["earlier"]!),
            raw: Int(Double(self.total) * r["raw"]!),
            tools: Int(Double(self.total) * r["tools"]!),
            reserve: Int(Double(self.total) * r["reserve"]!)
        )
    }

    // MARK: 记账

    public func setUsed(_ section: String, tokens: Int) {
        guard used[section] != nil else { return }
        used[section] = tokens
    }

    public func used(_ section: String? = nil) -> Int {
        if let section {
            return used[section] ?? 0
        }
        return used.values.reduce(0, +)
    }

    public func over(_ section: String) -> Bool {
        guard let value = used[section] else { return false }
        let limit: Int
        switch section {
        case "system": limit = quota.system
        case "earlier": limit = quota.earlier
        case "raw": limit = quota.raw
        case "tools": limit = quota.tools
        default: limit = 0
        }
        return value > limit
    }

    public var rawOver: Bool {
        remainingForRaw() <= 0
    }

    /// 整体越界（含 reserve 被吃掉）。
    public var totalOver: Bool {
        used() > total - quota.reserve
    }

    /// 除 RAW 外其它区合计超支的 token 数 —— 要从 RAW 里扣回来。
    public func overspent() -> Int {
        ["system", "earlier", "tools"].reduce(0) { acc, key in
            let limit: Int
            switch key {
            case "system": limit = quota.system
            case "earlier": limit = quota.earlier
            default: limit = quota.tools
            }
            return acc + max(0, (used[key] ?? 0) - limit)
        }
    }

    // MARK: 派生

    /// RAW 区剩余额度（可能为负，此时必须折叠）。
    ///
    /// system / earlier / tools 任一区超支都要从这里扣，否则「30k 硬顶」只是
    /// 记账：工具 schema 涨到 8k、系统提示涨到 6k 时 RAW 仍然拿满额。
    public func remainingForRaw() -> Int {
        quota.raw - (used["raw"] ?? 0) - overspent()
    }

    /// L1：本轮每个工具结果可用的**字符**上限。
    ///
    /// 按「硬上下文余量」而不是 raw 软配额分配：软配额只是建议值，
    /// 用它做阈值会让 raw 超过 12k 后所有工具结果一律被压到 600 字符，
    /// 模型读不到完整文件就只能反复重读，反而更慢。
    public func remainingForTool(_ nCalls: Int = 1, maxChars: Int = 8_000) -> Int {
        let n = max(1, nCalls)
        let hardLeft = total - quota.reserve - used()
        if hardLeft <= 0 {
            // 已经吃进输出预留：只给最小可诊断额度，逼内核立即折叠。
            return min(maxChars, 600)
        }
        return max(600, min(maxChars, hardLeft * 3 / n))
    }

    /// token → 字符的粗换算（用于给压缩器设置裁剪长度）。
    public func charsFor(_ tokens: Int) -> Int {
        max(200, tokens * 3)
    }

    public func snapshot() -> String {
        "budget \(used())/\(total) "
            + "(sys \(used("system"))/\(quota.system), "
            + "earlier \(used("earlier"))/\(quota.earlier), "
            + "raw \(used("raw"))/\(quota.raw), "
            + "tools \(used("tools"))/\(quota.tools))"
    }
}
