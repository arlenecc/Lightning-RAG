import XCTest
@testable import BaseAgentCore

final class BinarySignatureTests: XCTestCase {
    func testSignBitUsesStrictlyPositiveValues() {
        let vector: [Float] = [0.1, 0, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
        let data = VectorEncoder.encodeSignature(vector, bitDimension: 8)
        XCTAssertEqual(Array(data), [0b10101001])
    }

    func testLSBFirstMatchesExistingBitEncoder() {
        let vector: [Float] = (0..<16).map { index in
            index.isMultiple(of: 3) ? -Float(index) : Float(index + 1)
        }
        XCTAssertEqual(
            VectorEncoder.encodeSignature(vector, bitDimension: 16),
            VectorEncoder.encodeBits(vector))
    }

    func testNonMultipleOfEightPadsLastByte() {
        let data = VectorEncoder.encodeSignature(
            [0.1, -0.2, 0.3, -0.4, 0.5], bitDimension: 5)
        XCTAssertEqual(Array(data), [0b00010101])
    }

    func testPadsZeroBitsWhenBitDimensionExceedsVector() {
        let data = VectorEncoder.encodeSignature([0.1, -0.2, 0.3], bitDimension: 16)
        XCTAssertEqual(data.count, 2)
        XCTAssertEqual(Array(data), [0b00000101, 0])
    }

    func testHammingDistanceCountsDifferingSignBits() {
        let first = VectorEncoder.encodeSignature([0.1, -0.2, 0.3, -0.4], bitDimension: 4)
        let second = VectorEncoder.encodeSignature([-0.1, 0.2, -0.3, 0.4], bitDimension: 4)
        XCTAssertEqual(VectorEncoder.hamming(first, second), 4)
    }

    func testZeroBitDimensionProducesEmptySignature() {
        XCTAssertEqual(VectorEncoder.encodeSignature([1, 2], bitDimension: 0), Data())
    }
}
