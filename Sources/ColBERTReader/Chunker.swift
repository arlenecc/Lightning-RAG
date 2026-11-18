import Foundation

struct ChunkRecord {
    let title: String
    var content: String
}

struct ChunkOptions {
    /// 检索阅读模式：默认切片 3000 字（原 8000），下限 2500 字。
    /// 大分片决定 top-1 进模型上下文的体积，3000 字既能覆盖一个完整小节、
    /// 又不致压占主模型上下文预算。
    var minChars = 2500
    var maxChars = 3000
    var targetChars = 2750
    var overlapRatio = 0.01
    var semanticThreshold = 0.55
    var similarityFloor = 0.15
}

/// 结构优先 → 语义低谷 → 递归回退的长文档切片器。
/// 表格（不超长）与代码块作为原子块保持完整；断点一律落在句子/原子边界。
enum Chunker {
    private enum AtomKind { case prose, table, code, heading }
    private struct Atom { let text: String; let kind: AtomKind }
    private struct Section { let title: String; let atoms: [Atom] }

    static func split(_ text: String, sourceName: String,
                      similarity: (([String]) -> [Double]?)? = nil,
                      options: ChunkOptions = ChunkOptions()) throws -> [ChunkRecord] {
        let sections = parseSections(text, sourceName: sourceName)
        var chunks: [ChunkRecord] = []
        for section in sections {
            let pieces = try split(section, similarity: similarity, options: options)
            for piece in pieces {
                chunks.append(ChunkRecord(title: section.title, content: piece))
            }
        }
        return coalesce(chunks, options: options)
    }

    // MARK: - 结构解析

    private static let headingPatterns: [String] = [
        "^\\s{0,3}#{1,6}\\s+(.+?)\\s*$",
        "^\\s*第[一二三四五六七八九十百千零〇0-9]+[章节篇部卷](?:[\\s：:、.．]+.*)?$",
        "^\\s*(?:chapter|section|part)\\s+[0-9ivxlcdm]+(?:\\.[0-9]+)*(?:[.:：)）].*)?$",
    ]

    private static func isHeading(_ line: String) -> Bool {
        guard !line.trimmingCharacters(in: .whitespaces).isEmpty else { return false }
        return headingPatterns.contains { pattern in
            line.range(of: pattern, options: [.regularExpression, .caseInsensitive]) != nil
        }
    }

    private static func parseSections(_ text: String, sourceName: String) -> [Section] {
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        var sections: [Section] = []
        var title = sourceName
        var atoms: [Atom] = []
        var paragraph: [String] = []
        var inCode = false
        var codeLines: [String] = []
        var fence = ""
        var tableLines: [String] = []

        func flushParagraph() {
            guard !paragraph.isEmpty else { return }
            let raw = paragraph.joined(separator: "\n")
            let sentences = TextAnalysis.sentences(raw)
            if sentences.isEmpty {
                atoms.append(Atom(text: raw + "\n", kind: .prose))
            } else {
                for (i, s) in sentences.enumerated() {
                    atoms.append(Atom(text: s + (i == sentences.count - 1 ? "\n" : " "), kind: .prose))
                }
            }
            paragraph.removeAll(keepingCapacity: true)
        }
        func flushTable() {
            guard !tableLines.isEmpty else { return }
            atoms.append(Atom(text: tableLines.joined(separator: "\n") + "\n", kind: .table))
            tableLines.removeAll(keepingCapacity: true)
        }
        func flushSection() {
            flushParagraph(); flushTable()
            if !atoms.isEmpty { sections.append(Section(title: title, atoms: atoms)) }
            atoms.removeAll(keepingCapacity: true)
        }

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if inCode {
                codeLines.append(line)
                if trimmed.hasPrefix(fence) {
                    atoms.append(Atom(text: codeLines.joined(separator: "\n") + "\n", kind: .code))
                    codeLines.removeAll(); inCode = false; fence = ""
                }
                continue
            }
            if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
                flushParagraph(); flushTable()
                fence = String(trimmed.prefix(3))
                codeLines = [line]
                inCode = true
                continue
            }
            if trimmed.contains("|") {
                flushParagraph()
                tableLines.append(line)
                continue
            }
            flushTable()
            if isHeading(trimmed) {
                flushSection()
                title = trimmed.replacingOccurrences(of: "^\\s*#+\\s*", with: "", options: .regularExpression)
                atoms.append(Atom(text: line + "\n", kind: .heading))
            } else if trimmed.isEmpty {
                flushParagraph()
            } else {
                paragraph.append(line)
            }
        }
        if inCode { atoms.append(Atom(text: codeLines.joined(separator: "\n"), kind: .code)) }
        flushSection()
        return sections.isEmpty ? [Section(title: sourceName, atoms: [])] : sections
    }

    // MARK: - 章节内切分

    private static func split(_ section: Section,
                              similarity: (([String]) -> [Double]?)?,
                              options: ChunkOptions) throws -> [String] {
        guard !section.atoms.isEmpty else { return [] }
        var atoms = section.atoms
        if atoms.allSatisfy({ $0.kind == .prose }), atoms.count == 1,
           let expanded = expandProse(atoms[0]) {
            atoms = expanded
        }
        let whole = atoms.map(\.text).joined()
        if whole.count <= options.maxChars { return [whole] }

        let texts = atoms.map(\.text)
        let semantic = similarity?(texts)
        let similarities = (semantic?.count == texts.count - 1)
            ? semantic!
            : (0..<max(0, texts.count - 1)).map { i in TextAnalysis.lexicalSimilarity(texts[i], texts[i + 1]) }
        let threshold = adaptiveThreshold(similarities, fallback: options.semanticThreshold,
                                          floor: options.similarityFloor)
        let overlap = max(0, Int(Double(options.targetChars) * options.overlapRatio))

        var pieces: [String] = []
        var current: [Atom] = []
        var currentChars = 0

        func flush(at index: Int) {
            guard !current.isEmpty else { return }
            let cut = min(max(1, index), current.count)
            let head = current.prefix(cut).map(\.text).joined()
            pieces.append(head.trimmingCharacters(in: .whitespacesAndNewlines))
            let tail = Array(current.dropFirst(cut))
            if overlap > 0, tail.isEmpty {
                current = overlapTail(of: current, chars: overlap)
            } else {
                current = tail + overlapTail(of: Array(current.prefix(cut)), chars: overlap)
            }
            currentChars = current.map(\.text.count).reduce(0, +)
        }

        for (i, atom) in atoms.enumerated() {
            let count = atom.text.count
            if count > options.maxChars {
                flush(at: current.count)
                pieces.append(contentsOf: splitOversized(atom, options: options))
                current.removeAll(); currentChars = 0
                continue
            }
            if currentChars + count > options.maxChars, !current.isEmpty {
                flush(at: chooseBreak(current, similarities: similarities,
                                      offset: i - current.count, threshold: threshold,
                                      minChars: options.minChars, maxChars: options.maxChars))
            }
            current.append(atom)
            currentChars += count
            let boundary = i < similarities.count ? similarities[i] : Double.infinity
            if currentChars >= options.minChars, boundary <= threshold, i + 1 < atoms.count {
                flush(at: current.count)
            }
        }
        if !current.isEmpty {
            pieces.append(current.map(\.text).joined().trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return mergeShort(pieces, options: options)
    }

    /// 在长度范围内选相似度最低的边界；找不到低谷时退到最后一个原子（必为句末/原子末）。
    private static func chooseBreak(_ atoms: [Atom], similarities: [Double], offset: Int,
                                    threshold: Double, minChars: Int, maxChars: Int) -> Int {
        guard atoms.count > 1 else { return 1 }
        var best = atoms.count - 1
        var bestSim = Double.infinity
        var length = 0
        for j in 0..<atoms.count {
            length += atoms[j].text.count
            if length > maxChars { break }
            guard j >= 1, length >= minChars * 3 / 4 else { continue }
            let simIndex = offset + j
            let sim = simIndex < similarities.count ? similarities[simIndex] : Double.infinity
            if sim <= threshold, sim < bestSim { best = j + 1; bestSim = sim }
        }
        return min(best, atoms.count)
    }

    private static func overlapTail(of atoms: [Atom], chars: Int) -> [Atom] {
        guard chars > 0, atoms.count > 1 else { return [] }
        var picked: [Atom] = []
        var used = 0
        for atom in atoms.reversed() {
            guard used + atom.text.count <= chars, picked.count < atoms.count - 1 else { break }
            picked.insert(atom, at: 0)
            used += atom.text.count
        }
        return picked
    }

    private static func expandProse(_ atom: Atom) -> [Atom]? {
        let sentences = TextAnalysis.sentences(atom.text)
        guard sentences.count > 1 else { return nil }
        return sentences.map { Atom(text: $0 + " ", kind: .prose) }
    }

    private static func splitOversized(_ atom: Atom, options: ChunkOptions) -> [String] {
        let lines = atom.text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        switch atom.kind {
        case .table where lines.count > 2:
            let header = lines.prefix(2).joined(separator: "\n")   // 表头 + 分隔行
            var out: [String] = []
            var current = header
            for row in lines.dropFirst(2) {
                if current.count + row.count + 1 > options.maxChars {
                    out.append(current); current = header
                }
                current += "\n" + row
            }
            out.append(current)
            return out
        case .code:
            let fence = String(lines.first?.trimmingCharacters(in: .whitespaces).prefix(3) ?? "```")
            var out: [String] = []
            var current = fence
            for line in lines {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.hasPrefix(fence) { continue }
                if current.count + line.count + 1 > options.maxChars {
                    out.append(current + "\n" + fence); current = fence
                }
                current += "\n" + line
            }
            out.append(current + "\n" + fence)
            return out
        default:
            let sentences = TextAnalysis.sentences(atom.text)
            var out: [String] = []
            var current = ""
            for sentence in sentences {
                if !current.isEmpty, current.count + sentence.count > options.maxChars {
                    out.append(current); current = ""
                }
                current = TextAnalysis.join(current, sentence)
            }
            if !current.isEmpty { out.append(current) }
            return splitByCharactersIfNeeded(out, max: options.maxChars)
        }
    }

    private static func splitByCharactersIfNeeded(_ pieces: [String], max: Int) -> [String] {
        var out: [String] = []
        for piece in pieces {
            if piece.count <= max { out.append(piece); continue }
            var start = piece.startIndex
            while start < piece.endIndex {
                let end = piece.index(start, offsetBy: max, limitedBy: piece.endIndex) ?? piece.endIndex
                out.append(String(piece[start..<end]))
                start = end
            }
        }
        return out
    }

    // MARK: - 合并小片 / 自适应阈值

    private static func adaptiveThreshold(_ values: [Double], fallback: Double, floor: Double) -> Double {
        guard values.count >= 4 else { return fallback }
        let sorted = values.sorted()
        let index = Int(Double(sorted.count - 1) * 0.25)
        return max(floor, min(fallback, sorted[index]))
    }

    private static func mergeShort(_ pieces: [String], options: ChunkOptions) -> [String] {
        guard pieces.count > 1 else { return pieces }
        var merged: [String] = []
        for piece in pieces {
            if let last = merged.last,
               last.count < options.minChars || piece.count < options.minChars,
               last.count + piece.count + 1 <= options.maxChars {
                merged[merged.count - 1] = last + "\n" + piece
            } else {
                merged.append(piece)
            }
        }
        return merged
    }

    private static func coalesce(_ chunks: [ChunkRecord], options: ChunkOptions) -> [ChunkRecord] {
        guard chunks.count > 1 else { return chunks }
        var out: [ChunkRecord] = []
        for chunk in chunks {
            if let last = out.last,
               last.content.count < options.minChars || chunk.content.count < options.minChars {
                // 跨小节（title 不同）只允许整节合并：不切断任何一个小节。
                // 整节合并后内容含两节全文（各自标题行仍在正文里），不构成「切分小节」。
                if last.content.count + chunk.content.count + 1 <= options.maxChars {
                    out[out.count - 1] = ChunkRecord(title: last.title,
                                                    content: last.content + "\n" + chunk.content)
                    continue
                }
                // 合成会超上限：只在同一小节内部（超长被切成多片）搬句子补齐；
                // 跨小节搬句子会把一节正文拆散到两片，违背「小节尽量不切分」。
                if last.title == chunk.title,
                   let split = rebalance(last, chunk, options: options) {
                    out.removeLast()
                    out.append(contentsOf: split)
                    continue
                }
            }
            out.append(chunk)
        }
        var result = out
        for i in stride(from: result.count - 1, through: 1, by: -1) { ensureMin(i, in: &result, options: options) }
        for i in 0..<max(0, result.count - 1) { ensureMinForward(i, in: &result, options: options) }
        return result
    }

    /// 首片偏短时从下一片头部递归搬句子（先保证下一片自身达标）。
    /// 只在同一小节内部搬运，跨小节保持边界完整。
    private static func ensureMinForward(_ index: Int, in chunks: inout [ChunkRecord],
                                         options: ChunkOptions) {
        guard index + 1 < chunks.count, chunks[index].content.count < options.minChars else { return }
        guard chunks[index].title == chunks[index + 1].title else { return }
        ensureMinForward(index + 1, in: &chunks, options: options)
        while chunks[index].content.count < options.minChars {
            let nextSentences = TextAnalysis.sentences(chunks[index + 1].content)
            guard let head = nextSentences.first, nextSentences.count > 1 else { return }
            let rest = nextSentences.dropFirst().joined(separator: " ")
            let candidate = chunks[index].content + "\n" + head
            guard rest.count >= options.minChars / 2, candidate.count <= options.maxChars else { return }
            chunks[index].content = candidate
            chunks[index + 1].content = rest
        }
    }

    /// 从尾到头递归补齐短片：先补上一片，再从上一片句尾搬句子过来。
    /// 只在同一小节内部搬运，跨小节保持边界完整。
    private static func ensureMin(_ index: Int, in chunks: inout [ChunkRecord], options: ChunkOptions) {
        guard index > 0, chunks[index].content.count < options.minChars else { return }
        guard chunks[index].title == chunks[index - 1].title else { return }
        ensureMin(index - 1, in: &chunks, options: options)
        while chunks[index].content.count < options.minChars {
            let donorSentences = TextAnalysis.sentences(chunks[index - 1].content)
            guard donorSentences.count > 1, let tail = donorSentences.last else { return }
            let remaining = donorSentences.dropLast().joined(separator: " ")
            let candidate = tail + "\n" + chunks[index].content
            guard remaining.count >= options.minChars / 2, candidate.count <= options.maxChars else { return }
            chunks[index - 1].content = remaining
            chunks[index].content = candidate
        }
    }

    private static func rebalance(_ first: ChunkRecord, _ second: ChunkRecord,
                                  options: ChunkOptions) -> [ChunkRecord]? {
        var kept = TextAnalysis.sentences(first.content)
        guard kept.count > 1 else { return nil }
        var moved: [String] = []
        var nextText = second.content
        while nextText.count < options.minChars, kept.count > 1, let tail = kept.last {
            let candidate = tail + "\n" + nextText
            if candidate.count > options.maxChars { break }
            moved.insert(tail, at: 0)
            kept.removeLast()
            nextText = candidate
        }
        let firstText = kept.joined(separator: " ")
        guard !moved.isEmpty, firstText.count >= options.minChars * 4 / 5,
              nextText.count >= options.minChars * 9 / 10,
              firstText.count <= options.maxChars, nextText.count <= options.maxChars else { return nil }
        return [ChunkRecord(title: first.title, content: firstText),
                ChunkRecord(title: second.title, content: nextText)]
    }
}
