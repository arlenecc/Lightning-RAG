import Foundation
import MLX
import XCTest
@testable import BaseAgentCore

/// 验证 MLX 生成期错误兜底：task-local handler 必须被子 Task 继承，
/// 这样 generateTask 内的 iterator.next()/didSample 报错不会落到全局 fatalError。
final class MLXGenerationErrorGuardTests: XCTestCase {
    private func triggerMLXError() {
        let a = MLXArray(0..<10, [2, 5])
        let b = MLXArray(0..<15, [3, 5])
        eval(a + b)
    }

    func testGenerationErrorBoxRecordsAndThrows() throws {
        let box = MLXGenerationErrorBox()
        box.record("first error")
        box.record("second error")
        XCTAssertThrowsError(try box.check()) { error in
            XCTAssertEqual(error.localizedDescription, "MLX Error: first error")
        }
    }

    func testTaskCreatedInsideWithErrorHandlerInheritsHandler() async throws {
        let box = MLXGenerationErrorBox()
        let worker = withErrorHandler({ message in
            box.record(message)
        }) {
            Task {
                triggerMLXError()
            }
        }
        await worker.value

        let recorded = box.firstMessageForTesting
        XCTAssertNotNil(recorded, "子 Task 中的 MLX 错误应被 task-local handler 捕获")
        XCTAssertTrue((recorded ?? "").contains("broadcast"),
                      "错误信息不应为空或泛化：\(String(describing: recorded))")
    }
}
