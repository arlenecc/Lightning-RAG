import XCTest
@testable import BaseAgentCore

final class ProviderBinarySigningTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("binary-provider-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testNonV5ProvidersAdvertiseDefaultBinaryDimension() {
        let mlx = MLXEmbeddingProvider(directory: directory, modelName: "mlx-stub")
        let jinaV3 = JinaV3MLXEmbeddingProvider(directory: directory,
                                                modelName: "jina-v3-stub")
        let onnx = ONNXEmbeddingProvider(directory: directory, modelName: "onnx-stub")
        let llama = LlamaCppEmbeddingProvider(modelPath: directory.path,
                                              modelName: "llama-stub")

        let providers: [(String, EmbeddingProvider, Int)] = [
            ("mlx", mlx, 768),
            ("jina-v3", jinaV3, 1_024),
            ("onnx", onnx, 768),
            ("llama", llama, 768),
        ]
        for (name, provider, expected) in providers {
            let signing = provider as? EmbeddingBinarySigning
            XCTAssertNotNil(signing, "\(name) 必须声明 EmbeddingBinarySigning")
            XCTAssertEqual(signing?.binaryDimension, expected, "\(name)")
            XCTAssertEqual(signing?.binarySourceIdentifier, provider.identifier, "\(name)")
            XCTAssertEqual((signing?.binaryDimension ?? -1) % 8, 0, "\(name)")
        }
    }
}
