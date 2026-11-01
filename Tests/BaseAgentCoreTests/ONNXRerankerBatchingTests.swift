import XCTest
@testable import BaseAgentCore

/// 长度分桶的批次规划：规则必须可解释、可复现，供 `ONNXReranker.rerank` 直接复用。
final class ONNXRerankerBatchingTests: XCTestCase {

    func testGroupsByLengthAscending() {
        // 原始顺序刻意打乱长度：规划后应按长度升序成组，组内保留原始索引。
        let batches = ONNXReranker.lengthBucketedBatches(
            lengths: [120, 900, 118, 500, 305, 121], batchSize: 3)
        XCTAssertEqual(batches, [[2, 0, 5], [4, 3, 1]])
    }

    func testEqualLengthsKeepOriginalOrder() {
        let batches = ONNXReranker.lengthBucketedBatches(
            lengths: [50, 50, 50, 50], batchSize: 2)
        XCTAssertEqual(batches, [[0, 1], [2, 3]])
    }

    func testBatchSizeIsClampedToOne() {
        let batches = ONNXReranker.lengthBucketedBatches(
            lengths: [900, 100, 500], batchSize: 0)
        XCTAssertEqual(batches, [[1], [2], [0]])
    }

    func testEmptyInputProducesNoBatch() {
        XCTAssertEqual(ONNXReranker.lengthBucketedBatches(lengths: [], batchSize: 8), [])
    }

    func testSingleDocument() {
        XCTAssertEqual(ONNXReranker.lengthBucketedBatches(lengths: [300], batchSize: 8), [[0]])
    }
}
