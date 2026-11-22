import XCTest
@testable import BaseAgentCore

/// 预热进度回调：协议默认空实现不破坏其它后端；MLX/GGUF 实现可设置。
final class WarmupProgressTests: XCTestCase {
    /// 只声明 warmup、不实现 setter 的后端，必须仍能编译并工作（默认实现）。
    private final class LegacyWarmup: WarmupLLMService {
        var calls = 0
        func warmup(systemPrompt: String, tools: [JSONValue]) async throws { calls += 1 }
    }

    private final class ProgressWarmup: WarmupLLMService {
        var handler: (@Sendable (String) -> Void)?
        func setWarmupProgressHandler(_ handler: (@Sendable (String) -> Void)?) {
            self.handler = handler
        }
        func warmup(systemPrompt: String, tools: [JSONValue]) async throws {
            handler?("阶段1")
            handler?("阶段2")
        }
    }

    func testLegacyBackendUsesDefaultNoop() async throws {
        let legacy = LegacyWarmup()
        legacy.setWarmupProgressHandler { _ in XCTFail("默认实现不应回调") }
        try await legacy.warmup(systemPrompt: "s", tools: [])
        XCTAssertEqual(legacy.calls, 1)
    }

    func testProgressBackendReportsStages() async throws {
        let service = ProgressWarmup()
        let box = Box()
        service.setWarmupProgressHandler { box.append($0) }
        try await service.warmup(systemPrompt: "s", tools: [])
        XCTAssertEqual(box.lines, ["阶段1", "阶段2"])
        service.setWarmupProgressHandler(nil)
        XCTAssertNil(service.handler, "结束后应能清空回调，避免持有 UI")
    }
}

private final class Box: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String] = []
    var lines: [String] { lock.lock(); defer { lock.unlock() }; return storage }
    func append(_ s: String) { lock.lock(); storage.append(s); lock.unlock() }
}
