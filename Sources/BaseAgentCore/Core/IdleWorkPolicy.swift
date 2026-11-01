import Foundation

/// 空闲工作的启动窗口。
///
/// OCR 回收与闲时摘要分开计时：OCR 权重下一次解析可以按需重载，3 分钟即可
/// 释放；闲时摘要会持续借用辅助模型 GPU，必须等更长的空闲窗口，避免用户
/// 只是短暂离开就与下一次前台请求抢资源。
public enum IdleWorkPolicy {
    /// 无前台活动后释放 OCR 权重的延迟（3 分钟）。
    public static let ocrUnloadDelayNanoseconds: UInt64 = 180_000_000_000
    /// 无前台活动后启动闲时章节摘要的延迟（10 分钟）。
    public static let summaryDelayNanoseconds: UInt64 = 600_000_000_000
}
