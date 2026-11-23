import XCTest
@testable import BaseAgentCore

/// 对照：RAG top3 切片全部用核心句压缩后总结 vs 只给 top1 原文总结。
/// 评价：决策模型对答案的「信息充分性」打分 + 关键事实命中数 + 输出长度/耗时。
final class Top3CoreVsTop1SummaryTests: XCTestCase {
    private let modelDir = "/Users/arlenecc/codebase/agent-app/models/qwen3.5-4b-agentic-coder-v4-4bit-mlx"
    private let decisionDir = "/Users/arlenecc/codebase/agent-app/models/PKU-KCL-Meteor-0.8B-mlx-int4"

    /// 3 段模拟 RAG 切片（同一主题：元素周期表发现史），每段含背景+观点+细节。
    private var passages: [String] {[
        [
            "门捷列夫之梦这本书讲述了元素周期表的发现历程。核心观点认为，科学的突破往往源于直觉和灵感，",
            "而不是纯粹的线性推理。作者强调门捷列夫在1869年提出周期表时，大胆预言了尚未发现元素的性质。",
            "这段还提到当时的化学界对原子量测定存在诸多误差，导致早期分类尝试屡屡失败。",
            "历史上德贝赖纳提出三元律，纽兰兹提出八音律，但都被学界忽视或嘲笑。",
            "本书用大量细节描写了19世纪欧洲化学实验室的场景，以及科学家之间的竞争与合作。",
        ].joined(),
        [
            "纽兰兹的八音律是元素周期律发现的重要一步。他把已知元素按原子量排列，发现每隔八个元素性质重复。",
            "核心观点认为，这个规律虽然过于僵化，却是首次证实元素间存在统一规律。",
            "1865年他向伦敦化学学会报告时遭到嘲笑，有人讽刺他是否试过按字母顺序排列。",
            "直到25年后的1887年，才获得英国皇家学会戴维奖章的认可。",
            "书中还描述了纽兰兹作为业余化学家的坚持，以及他与其他科学家的通信往来。",
            "这段补充了大量时代背景，包括维多利亚时代的科学普及运动和学术界的保守风气。",
        ].joined(),
        [
            "德尚古尔多阿在1864年发现了卤素家族的相似性。他指出氯、溴、碘等元素具有共同的化学性质，",
            "这是首个证实元素间存在统一规律的发现。核心观点强调，他的工作为后来门捷列夫的突破奠定了基础。",
            "这段还提到他如何收集大量实验数据、对比不同元素的反应性，以及当时测量技术的局限。",
            "书中用一整章描述了他的实验室笔记和未发表的手稿，这些材料后来被重新发现。",
            "作者还讨论了科学发现中的偶然性与必然性，以及优先权争议的历史。",
        ].joined(),
    ]}

    private let question = "元素周期律的发现过程中，门捷列夫、纽兰兹、德尚古尔多阿各自的关键贡献是什么？"
    private let keyFacts = ["门捷列夫", "纽兰兹", "八音律", "德尚古尔多阿", "卤素", "1869", "1865", "1864"]

    private func collect(_ service: LLMService, user: String) async throws -> String {
        var out = ""
        for try await event in service.chatStream(messages: [
            ChatMessage(role: .user, content: user),
        ], tools: [], temperature: 0) {
            if case .content = event.kind { out += event.content }
        }
        return out
    }

    func testTop3CompressedVsTop1Summary() async throws {
        guard FileManager.default.fileExists(atPath: modelDir + "/config.json"),
              FileManager.default.fileExists(atPath: decisionDir + "/readout.safetensors") else {
            throw XCTSkip("本机无主模型或决策模型，跳过对照")
        }
        var config = AgentConfig()
        config.workspace = NSTemporaryDirectory()
        config.llmProvider = "local"
        config.localLLMModel = modelDir
        config.auxMaxTokens = 256
        let main = LocalModelFactory.shared.makeLLM(config: config)
        if let preload = main as? PreloadableLLMService { try await preload.preload() }

        // 方案 A：top3 全部核心句压缩（各 top10 去重）后拼成材料
        let compressed = passages.map { CoreOpinionExtractor.compressed($0, topN: 10) }
        let materialA = compressed.enumerated().map { "[切片\($0.offset + 1)]\n\($0.element)" }.joined(separator: "\n")
        let promptA = "请仅依据以下资料回答问题，列出每个人的关键贡献，简洁作答。\n\(materialA)\n\n问题：\(question)"

        // 方案 B：只给 top1 原文
        let promptB = "请仅依据以下资料回答问题，列出每个人的关键贡献，简洁作答。\n[切片1]\n\(passages[0])\n\n问题：\(question)"

        let t0 = Date()
        let answerA = try await collect(main, user: promptA)
        let tA = Date().timeIntervalSince(t0)
        let t1 = Date()
        let answerB = try await collect(main, user: promptB)
        let tB = Date().timeIntervalSince(t1)

        // 决策模型对两个答案打「信息充分性」分
        let judge = MeteorDecisionScorer(modelDir: decisionDir)
        let judgeQ = "Are the key contributions of Mendeleev, Newlands and De Chancourtois fully covered? "
            + "(A: insufficient, B: sufficient)"
        let pA = try await judge.judgeBinary(state: answerA, question: judgeQ, maxStateChars: Int.max)
        let pB = try await judge.judgeBinary(state: answerB, question: judgeQ, maxStateChars: Int.max)

        func facts(_ text: String) -> Int { keyFacts.filter { text.contains($0) }.count }
        let fmt = { (x: Double) in String(format: "%.3f", x) }
        let line = "[top3-compare] A(top3核心句) 材料=\(materialA.count)字 答案=\(answerA.count)字 "
            + "事实命中=\(facts(answerA))/\(keyFacts.count) prob=\(fmt(pA)) 耗时=\(fmt(tA))s\n"
            + "[top3-compare] B(top1原文) 材料=\(passages[0].count)字 答案=\(answerB.count)字 "
            + "事实命中=\(facts(answerB))/\(keyFacts.count) prob=\(fmt(pB)) 耗时=\(fmt(tB))s\n"
            + "[top3-compare] A答案: \(answerA)\n"
            + "[top3-compare] B答案: \(answerB)\n"
        FileHandle.standardError.write(Data(line.utf8))

        XCTAssertFalse(answerA.isEmpty)
        XCTAssertFalse(answerB.isEmpty)
    }
}
