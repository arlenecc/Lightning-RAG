import XCTest
@testable import BaseAgentCore

private final class SigningSchedulerStub: EmbeddingProvider,
                                          EmbeddingTokenCounting,
                                          EmbeddingBinarySigning,
                                          @unchecked Sendable {
    let identifier = "signing-scheduler"
    let dimension = 8
    let maxContentTokens = 4_096
    var binaryDimension: Int { 8 }
    var binarySourceIdentifier: String { identifier }

    private let lock = NSLock()
    private(set) var plainBatches: [[String]] = []
    private(set) var signingBatches: [[String]] = []
    var breakSignatures = false

    func makeTokenCounter() async -> (@Sendable (String) -> Int)? {
        { $0.count }
    }

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        lock.lock()
        plainBatches.append(texts)
        lock.unlock()
        return texts.map(Self.vector(for:))
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        Self.vector(for: text)
    }

    func embedDocumentsWithSignatures(_ texts: [String]) async throws
        -> BinaryEmbeddingOutput {
        lock.lock()
        signingBatches.append(texts)
        lock.unlock()
        let signatures = breakSignatures
            ? []
            : texts.map { Data([UInt8(truncatingIfNeeded: $0.count)]) }
        return BinaryEmbeddingOutput(
            vectors: texts.map(Self.vector(for:)),
            signatures: signatures)
    }

    func embedQueryWithSignature(_ text: String) async throws
        -> (vector: [Float], signature: Data) {
        (Self.vector(for: text), Data([UInt8(truncatingIfNeeded: text.count)]))
    }

    /// 首维 = 文本长度，便于按原始下标反查。
    private static func vector(for text: String) -> [Float] {
        [Float(text.count)] + [Float](repeating: 0, count: 7)
    }
}

private final class PlainSchedulerStub: EmbeddingProvider, @unchecked Sendable {
    let identifier = "plain-scheduler"
    let dimension = 8

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map { _ in [Float](repeating: 0, count: 8) }
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        [Float](repeating: 0, count: 8)
    }
}

final class AdaptiveEmbeddingSigningTests: XCTestCase {
    private func makeScheduler(_ base: EmbeddingProvider) -> AdaptiveEmbeddingScheduler {
        AdaptiveEmbeddingScheduler(
            base: base,
            options: .init(minBatchSize: 1, maxBatchSize: 2,
                           initialBatchSize: 2, step: 1,
                           memoryLimitMB: 32_768, throughputTolerance: 0.5,
                           probeBatches: 16, probeSeconds: 30,
                           tokenBudgetMin: 4_096, tokenBudgetMax: 4_096,
                           initialTokenBudget: 4_096, tokenBudgetStep: 128,
                           windowTexts: 2),
            log: { _ in })
    }

    /// 长度降序后跨多个窗口，签名与向量必须按同一套原始下标回写。
    func testSignaturePathRestoresOriginalIndicesAcrossWindows() async throws {
        let base = SigningSchedulerStub()
        let scheduler = makeScheduler(base)
        let lengths = [100, 1, 90, 2, 80, 3]
        let texts = lengths.map { String(repeating: "a", count: $0) }
        var progress: [(Int, Int)] = []

        let output = try await scheduler.embedDocumentsWithSignatures(texts) { done, total in
            progress.append((done, total))
        }
        let result = try XCTUnwrap(output)
        XCTAssertEqual(result.vectors.compactMap { $0.first.map(Int.init) }, lengths,
                       "向量必须按原始下标写回")
        XCTAssertEqual(result.signatures.map { Int($0[0]) }, lengths,
                       "签名必须和向量使用同一套窗口/组批下标映射")
        XCTAssertEqual(progress.map(\.0), [2, 4, 6])
        XCTAssertTrue(progress.allSatisfy { $0.1 == texts.count })
        XCTAssertEqual(base.signingBatches.flatMap { $0 }.count, texts.count)
        XCTAssertTrue(base.plainBatches.isEmpty, "签名路径不应再调用普通 embedDocuments")
        XCTAssertEqual(scheduler.binaryDimension, 8)
    }

    func testPlainEmbeddingDoesNotRequestSignatures() async throws {
        let base = SigningSchedulerStub()
        let scheduler = makeScheduler(base)
        let vectors = try await scheduler.embedDocuments(["a", "bb", "ccc"])

        XCTAssertEqual(vectors.compactMap { $0.first.map(Int.init) }, [1, 2, 3])
        XCTAssertEqual(base.plainBatches.flatMap { $0 }.count, 3)
        XCTAssertTrue(base.signingBatches.isEmpty,
                      "普通入库路径不应额外触发签名，保持原行为")
    }

    func testUnsupportedBaseReturnsNilAndForwardsNilDimension() async throws {
        let scheduler = makeScheduler(PlainSchedulerStub())
        XCTAssertNil(scheduler.binaryDimension)
        let output = try await scheduler.embedDocumentsWithSignatures(["a", "b"])
        XCTAssertNil(output, "底层不支持时返回 nil，由调用方回退原流程")
    }

    func testSignatureCountMismatchThrowsInsteadOfWritingPartialData() async throws {
        let base = SigningSchedulerStub()
        base.breakSignatures = true
        let scheduler = makeScheduler(base)

        do {
            _ = try await scheduler.embedDocumentsWithSignatures(["a", "bb"])
            XCTFail("签名数量不符必须抛错")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("签名数量异常"),
                          error.localizedDescription)
        }
    }
}
