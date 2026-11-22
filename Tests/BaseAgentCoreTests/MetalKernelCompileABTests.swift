import XCTest
import MLX
@testable import BaseAgentCore

/// A/B 探针：量化「Metal kernel 首次编译」在进程内的成本，
/// 用来判断「持久化编译结果」是否值得投入（跨启动复用）。
final class MetalKernelCompileABTests: XCTestCase {
    /// 用与 LLM prefill 同型的量化矩阵乘 + softmax 图，连续 eval 同一 shape。
    /// 第一次含 shader 编译，之后命中进程内 library_map_。
    func testFirstEvalVsWarmEval() {
        func onePass() -> Double {
            let start = CFAbsoluteTimeGetCurrent()
            let a = MLXArray.ones([256, 512], type: Float.self)
            let b = MLXArray.ones([512, 512], type: Float.self)
            let y = matmul(a, b)
            eval(y)
            return CFAbsoluteTimeGetCurrent() - start
        }
        // 先做一次无关 shape，确保 Metal 设备本身已初始化
        eval(MLXArray.ones([4, 4], type: Float.self))
        let cold = onePass()
        var warm: [Double] = []
        for _ in 0..<5 { warm.append(onePass()) }
        let warmMean = warm.reduce(0, +) / Double(warm.count)
        let line = "[kernel-ab] cold=" + String(format: "%.1f", cold * 1000)
            + "ms warmMean=" + String(format: "%.1f", warmMean * 1000)
            + "ms ratio=" + String(format: "%.1f", cold / max(warmMean, 0.0001)) + "x\n"
        FileHandle.standardError.write(Data(line.utf8))
        // 只做观测，不设性能断言（不同机器差异大）
        XCTAssertGreaterThan(cold, 0)
    }

    /// 跨「新 shape」时是否每次都要重新编译（决定缓存命中率）。
    func testNewShapeCompilesAgain() {
        func pass(_ n: Int) -> Double {
            let start = CFAbsoluteTimeGetCurrent()
            let a = MLXArray.ones([n, 512], type: Float.self)
            let b = MLXArray.ones([512, 512], type: Float.self)
            eval(matmul(a, b))
            return CFAbsoluteTimeGetCurrent() - start
        }
        let first = pass(256)
        let same = pass(256)
        let different = pass(257)
        let line = "[kernel-ab] first=" + String(format: "%.1f", first * 1000)
            + "ms sameShape=" + String(format: "%.1f", same * 1000)
            + "ms newShape=" + String(format: "%.1f", different * 1000) + "ms\n"
        FileHandle.standardError.write(Data(line.utf8))
        XCTAssertGreaterThan(first, 0)
    }
}
