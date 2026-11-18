import XCTest
import Foundation
@testable import ColBERTReader

/// 检索阅读模式切片策略回归：默认 3000/2500、目录优先、小节尽量不切分。
final class ColbertChunkerTests: XCTestCase {

    /// 每个中文句子约 N 字，用于构造已知长度的正文。
    private func sentences(_ count: Int, charsPerSentence: Int = 100) -> String {
        var out = ""
        for i in 0..<count {
            out += "第\(i + 1)句" + String(repeating: "文", count: charsPerSentence) + "。"
        }
        return out
    }

    func testDefaultOptionsAre3000And2500() {
        let o = ChunkOptions()
        XCTAssertEqual(o.maxChars, 3000)
        XCTAssertEqual(o.minChars, 2500)
        XCTAssertLessThan(o.targetChars, o.maxChars)
    }

    /// 小节整体不超过 3000 时应整节成片，不在内部切开。
    func testShortSectionKeptWhole() throws {
        let md = "# 第一章 概述\n\n" + sentences(20, charsPerSentence: 100) // 约 2000+ 字
        let chunks = try Chunker.split(md, sourceName: "book.md")
        XCTAssertFalse(chunks.isEmpty)
        XCTAssertTrue(chunks.allSatisfy { $0.content.count <= 3000 }, "每片不超 3000")
        XCTAssertEqual(chunks.count, 1, "单个不超上限的小节应整节成一片")
        XCTAssertTrue(chunks[0].content.contains("第一章"))
    }

    /// 超长小节才在内部切分，且每片不超 3000。
    func testOversizedSectionSplitWithinLimit() throws {
        let md = "# 第二章 长章\n\n" + sentences(80, charsPerSentence: 100) // 约 8000+ 字
        let chunks = try Chunker.split(md, sourceName: "book.md")
        XCTAssertGreaterThan(chunks.count, 1)
        XCTAssertTrue(chunks.allSatisfy { $0.content.count <= 3000 },
                      "超长小节切开后每片仍不得超过 3000：\(chunks.map(\.content.count))")
        XCTAssertTrue(chunks.allSatisfy { $0.title.contains("第二章") })
    }

    /// 多个小节各自独立：小节标题成为切片标题，小节边界不被跨节搬句子打散。
    func testSectionsKeepTheirOwnTitles() throws {
        let md = """
        # 第一章 甲

        \(sentences(30, charsPerSentence: 100))

        # 第二章 乙

        \(sentences(30, charsPerSentence: 100))
        """
        let chunks = try Chunker.split(md, sourceName: "book.md")
        XCTAssertGreaterThanOrEqual(chunks.count, 2, "两个较长小节应各自成片")
        let titles = Set(chunks.map(\.title))
        XCTAssertTrue(titles.contains { $0.contains("第一章") })
        XCTAssertTrue(titles.contains { $0.contains("第二章") },
                      "每个小节的标题应保留在自己的切片上，不被跨节合并吞掉")
        // 第一章正文只应出现在标题为第一章的片里（未被搬到第二章的片）。
        let firstChunks = chunks.filter { $0.title.contains("第一章") }
        XCTAssertTrue(firstChunks.allSatisfy { !$0.content.contains("第二章 乙") })
    }
}
