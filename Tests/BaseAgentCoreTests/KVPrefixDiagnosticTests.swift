import Foundation
import XCTest
@testable import BaseAgentCore

/// KV 前缀回归：折叠是唯一允许 prompt 前缀塌陷到 system 的边界。
///
/// 诊断阶段实测折叠轮 `prefix=1`（只有 system 消息还能复用）；本测试把该行为
/// 固化为断言，同时锁住反方向的不变式：
/// - 折叠轮：`<earlier_context>` 改写消息索引 1，公共前缀只能是 1；
/// - 非折叠轮：上一轮消息必须完整成为下一轮前缀，不允许 raw 窗口逐轮滑动。
final class KVPrefixDiagnosticTests: XCTestCase {
    private let foldEvery = 8
    private let totalTurns = 18

    private func common(_ lhs: [ChatMessage], _ rhs: [ChatMessage]) -> Int {
        var index = 0
        while index < lhs.count, index < rhs.count, lhs[index] == rhs[index] {
            index += 1
        }
        return index
    }

    func testFoldingBreaksPromptPrefix() {
        let session = Session(budget: ContextBudget(total: 30_000), rawTurns: 8)
        var previous: [ChatMessage] = []
        var previousCounts: [Int: Int] = [:]
        var prefixByTurn: [Int: Int] = [:]
        var digestCounts: [Int: Int] = [:]

        for turn in 1...totalTurns {
            if turn > 1 && (turn - 1) % foldEvery == 0 {
                session.foldForConsolidation()
            }
            previousCounts[turn] = previous.count
            let rec = session.startTurn(prompt: String(repeating: "u", count: 3_200))
            let messages = session.buildMessages(systemPrompt: "SYS")
            prefixByTurn[turn] = common(previous, messages)
            digestCounts[turn] = session.digestLines().count
            previous = messages
            session.finishTurn(rec, assistantText: String(repeating: "a", count: 3_200))
        }

        let foldTurns = stride(from: foldEvery + 1, through: totalTurns, by: foldEvery)
            .map { $0 }
        XCTAssertEqual(foldTurns, [9, 17], "折叠节奏应与 rawTurns 一致")

        // 塌陷只允许发生在折叠轮：其他轮次若掉到 1，说明窗口又在逐轮滑动。
        let collapsedTurns = prefixByTurn
            .filter { $0.key > 1 && $0.value <= 1 }
            .keys.sorted()
        XCTAssertEqual(collapsedTurns, foldTurns, "只有折叠轮允许前缀塌陷")

        for turn in foldTurns {
            XCTAssertEqual(prefixByTurn[turn], 1,
                           "第 \(turn) 轮折叠后只应保留 system 前缀")
            XCTAssertGreaterThan(previousCounts[turn] ?? 0, 1,
                                 "第 \(turn) 轮折叠前应存在可复用的消息前缀")
            XCTAssertGreaterThan(digestCounts[turn] ?? 0, 0,
                                 "第 \(turn) 轮折叠后应有摘要行")
        }

        // 非折叠轮必须完整复用上一轮消息前缀（上一轮列表整体是下一轮的前缀）。
        for turn in 1...totalTurns where !foldTurns.contains(turn) {
            guard turn > 1 else {
                XCTAssertEqual(prefixByTurn[turn], 0, "首轮没有可复用的前缀")
                continue
            }
            XCTAssertEqual(prefixByTurn[turn], previousCounts[turn],
                           "第 \(turn) 轮不应改写已发送过的消息前缀")
        }
    }
}
