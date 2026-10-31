import XCTest
@testable import BaseAgentCore

final class BinaryCoarseConfigTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("binary-config-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func writeConfig(_ json: String) throws -> String {
        let path = directory.appendingPathComponent("config-\(UUID().uuidString).json").path
        try json.write(toFile: path, atomically: true, encoding: .utf8)
        return path
    }

    func testDefaultsEnableBinaryCoarseWith200Candidates() {
        let config = AgentConfig()
        XCTAssertTrue(config.ragBinaryCoarseEnabled)
        XCTAssertEqual(config.ragBinaryCoarseCandidates, 200)
    }

    func testLoadReadsAndClampsBinaryCoarseOptions() throws {
        let path = try writeConfig("""
        {
          "rag_binary_coarse_enabled": false,
          "rag_binary_coarse_candidates": 5000
        }
        """)
        let config = AgentConfig.load(path)
        XCTAssertFalse(config.ragBinaryCoarseEnabled)
        XCTAssertEqual(config.ragBinaryCoarseCandidates, 1000)
    }

    func testMigrateClampsCandidatesToSupportedRange() {
        var config = AgentConfig()
        config.ragBinaryCoarseCandidates = 49
        config.migrate()
        XCTAssertEqual(config.ragBinaryCoarseCandidates, 50)

        config.ragBinaryCoarseCandidates = 500
        config.migrate()
        XCTAssertEqual(config.ragBinaryCoarseCandidates, 500)
    }

    func testExportWritesBinaryCoarseKeys() {
        var config = AgentConfig()
        config.ragBinaryCoarseEnabled = false
        config.ragBinaryCoarseCandidates = 120
        let exported = config.export()
        XCTAssertEqual(exported["rag_binary_coarse_enabled"] as? Bool, false)
        XCTAssertEqual(exported["rag_binary_coarse_candidates"] as? Int, 120)
    }
}
