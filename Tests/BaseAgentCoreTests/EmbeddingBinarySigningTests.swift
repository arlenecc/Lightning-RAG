import XCTest
@testable import BaseAgentCore

private final class DefaultSigningEmbedding: EmbeddingProvider,
                                            EmbeddingBinarySigning,
                                            @unchecked Sendable {
    let identifier = "default-signing"
    let dimension = 10

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.enumerated().map { index, _ in
            (0..<dimension).map { offset in
                Float((index + offset) % 3 == 0 ? -1 : 1)
            }
        }
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        try await embedDocuments([text])[0]
    }
}

final class EmbeddingBinarySigningTests: XCTestCase {
    func testDefaultBinaryDimensionRoundsUpToByteBoundary() {
        let embedding = DefaultSigningEmbedding()
        XCTAssertEqual(embedding.binaryDimension, 16)
        XCTAssertEqual(embedding.binarySourceIdentifier, embedding.identifier)
    }

    func testDefaultDocumentSignaturesMatchFixedWidthEncoder() async throws {
        let embedding = DefaultSigningEmbedding()
        let output = try await embedding.embedDocumentsWithSignatures(["a", "b"])
        XCTAssertEqual(output.vectors.count, 2)
        XCTAssertEqual(output.signatures.count, 2)
        for (vector, signature) in zip(output.vectors, output.signatures) {
            XCTAssertEqual(signature.count, 2)
            XCTAssertEqual(signature,
                           VectorEncoder.encodeSignature(vector, bitDimension: 16))
        }
    }

    func testDefaultQuerySignatureMatchesDocumentEncoding() async throws {
        let embedding = DefaultSigningEmbedding()
        let query = try await embedding.embedQueryWithSignature("same text")
        XCTAssertEqual(query.vector.count, 10)
        XCTAssertEqual(query.signature,
                       VectorEncoder.encodeSignature(query.vector, bitDimension: 16))
    }

    func testSearchHitCarriesBinaryScoreWithoutBreakingDefaults() {
        let hit = SearchHit(rowID: 1, text: "x", source: "s", chunkIndex: 0,
                            binaryScore: 0.75)
        XCTAssertEqual(hit.binaryScore, 0.75)
        let legacy = SearchHit(rowID: 2, text: "y", source: "s", chunkIndex: 0)
        XCTAssertNil(legacy.binaryScore)
    }
}
