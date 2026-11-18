import XCTest
import Foundation
@testable import BaseAgentCore

/// 检索阅读模式的 handler 读写与 colbertService() 访问不死锁。
///
/// 回归：colbertService() 持有 colbertLock 时若通过计算属性
/// colbertDownloadStatusHandler 读取 handler，其 getter 会再次 lock 同一把
/// 非递归 NSLock，自重入死锁会让启动预加载与状态刷新互相挂死。
final class ColbertLockReentrancyTests: XCTestCase {
    func testConcurrentHandlerAccessDoesNotDeadlock() async {
        var config = AgentConfig()
        config.colbertReaderEnabled = true
        let engine = RAGEngine(config: config, knowledgeBase: "colbert-lock-test")

        let deadline = Date().addingTimeInterval(10)
        let iterations = 300
        DispatchQueue.concurrentPerform(iterations: iterations) { i in
            if i % 2 == 0 {
                engine.colbertDownloadStatusHandler = { _ in }
            } else {
                _ = engine.colbertANNStatus()
            }
        }
        XCTAssertLessThan(Date(), deadline, "colbertLock 并发读写超时（疑似死锁）")
        // 收尾：再各调用一次，确保两条路径都真的走通。
        engine.colbertDownloadStatusHandler = nil
        XCTAssertNotNil(engine.colbertANNStatus())
    }
}
