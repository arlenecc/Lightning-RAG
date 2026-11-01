import XCTest
@testable import BaseAgentCore

/// P2：生成速度与内存。
final class P2PerformanceTests: XCTestCase {

    /// 主模型必须默认开启 thinking：关闭会破坏主引擎的推理与工具决策质量。
    /// 低延迟的工具路由应走请求级关闭或辅助模型，不能牺牲主模型默认值。
    func testMainModelThinkingOnByDefault() {
        XCTAssertTrue(AgentConfig().localLLMEnableThinking)
    }

    /// OCR 默认使用两个 Ovis 实例：同步期间可并行识别扫描页。
    /// 两个实例只在同步/按需 OCR 时加载权重，任务结束立即释放。
    func testOvisDefaultsToTwoInstances() {
        XCTAssertEqual(AgentConfig().ragOvisInstances, 2)
    }

    /// MLX 空闲 buffer 默认提回 1GB：README 实测 decode 有 5–8% 收益，
    /// 且 1GB 仍未超过 embedding 工作集。
    func testMLXCacheLimitDefaultsToOneGB() {
        XCTAssertEqual(AgentConfig().mlxCacheLimitMB, 1_024)
    }

    /// 闲时摘要要等更长的空闲窗口才启动：OCR 3 分钟即可回收，
    /// 但摘要会持续占用辅助模型，必须在用户离开更久后才开始。
    func testIdleSummaryStartsAfterLongerIdleWindow() {
        XCTAssertGreaterThan(IdleWorkPolicy.summaryDelayNanoseconds,
                             IdleWorkPolicy.ocrUnloadDelayNanoseconds)
        XCTAssertEqual(IdleWorkPolicy.summaryDelayNanoseconds,
                       600_000_000_000)
    }
}
