import XCTest
@testable import BaseAgentCore

/// system prompt 精简红线。
///
/// 这些字面量是内核逻辑、既有测试或模型行为的硬依赖，精简措辞时**不得删除**：
/// - `KnowledgeBaseToolPolicyTests` 逐字断言联网授权与「不全局禁用其它工具」；
/// - `KernelLoop.looksLikeSystemPromptEcho` 靠 SOP 公理标题 + 600 字长度识别
///   「模型把 system prompt 复述给用户」；
/// - `MemoryMaintenance.validL1` 依赖 `[RULES]` / 层级路径标记；
/// - `Compressor` / `Loop` 依赖 `<summary>` 与计划状态词。
///
/// 同时锁住 v1.3.5 工具合并成果：提示词里不得回流旧工具名或全局禁令措辞。
final class SystemPromptContractTests: XCTestCase {

    func testBaseKeepsNetworkAuthorization() {
        let p = BASE_SYSTEM_PROMPT
        for s in ["联网抓取", "无法访问网络", "不要用知识库检索代替联网", "web_scan"] {
            XCTAssertTrue(p.contains(s), "BASE 缺少联网授权字面量：\(s)")
        }
    }

    func testBaseKeepsProtocolAndStateWords() {
        let p = BASE_SYSTEM_PROMPT
        for s in ["<summary>", "update_plan", "blocked", "skipped",
                  "code_run", "shell_run", "ask_user"] {
            XCTAssertTrue(p.contains(s), "BASE 缺少协议/状态词：\(s)")
        }
        // 语义检索原话：向量检索质量护栏，不可压缩掉。
        XCTAssertTrue(p.contains("Semantic search queries"))
        XCTAssertTrue(p.contains("user's own natural-language phrasing"))
        // 确认白名单。
        XCTAssertTrue(p.contains("require user confirmation"))
        XCTAssertTrue(p.contains("headless runs deny them"))
    }

    func testBaseKeepsMergedToolNames() {
        for s in ["file_tools", "rag_tools", "skill_tools", "long_memory",
                  "update_working_checkpoint", "start_long_term_update",
                  "op=\"outline\"", "op=\"patch\"", "op=\"list\""] {
            XCTAssertTrue(BASE_SYSTEM_PROMPT.contains(s), "BASE 缺少合并后工具名：\(s)")
        }
    }

    /// 旧工具名与「全局禁令」措辞不得回流（历史事故：模型把它泛化成能力边界，
    /// 连 web_scan 都不敢调，反问用户「我是本地 agent，没有访问外网的权限」）。
    func testNoLegacyToolNamesOrGlobalBans() {
        let texts = [("base", BASE_SYSTEM_PROMPT),
                     ("hint", KNOWLEDGE_BASE_HINT),
                     ("l0", MEMORY_MANAGEMENT_SOP),
                     ("l1", DEFAULT_GLOBAL_MEM_INSIGHT),
                     ("rag result", ragUsageNote)]
        let legacy = ["file_read", "file_write", "file_patch", "file_list",
                      "rag_search", "rag_outline", "rag_status", "rag_ingest",
                      "web_render", "webexec_js", "web_execute_js",
                      "skill_search", "skill_load", "skill_save",
                      "memory_graph", "memory_search"]
        let bans = ["知识库问答只用", "只用 rag_", "禁止用文件工具", "不能改用文件工具绕过"]
        for (name, text) in texts {
            for bad in legacy + bans {
                XCTAssertFalse(text.contains(bad),
                               "\(name) 不应出现旧工具名/全局禁令：\(bad)")
            }
        }
    }

    /// 公理标题虽与 `looksLikeSystemPromptEcho` 的 markers 不完全同字，但 SOP 全文
    /// 复述会命中 >600 字的长度兜底；这里锁住两端都不失效。
    func testL0SOPStillRecognizedAsPromptEcho() {
        XCTAssertTrue(MEMORY_MANAGEMENT_SOP.contains("核心公理"))
        XCTAssertTrue(MEMORY_MANAGEMENT_SOP.contains("行动验证"))
        XCTAssertTrue(MEMORY_MANAGEMENT_SOP.contains("神圣不可删改"))
        XCTAssertTrue(KernelLoop.looksLikeSystemPromptEcho(MEMORY_MANAGEMENT_SOP))
    }

    func testL1KeepsRulesMarker() {
        XCTAssertTrue(DEFAULT_GLOBAL_MEM_INSIGHT.contains("[RULES]"))
        XCTAssertTrue(DEFAULT_GLOBAL_MEM_INSIGHT.contains("global_mem.txt"))
        XCTAssertTrue(DEFAULT_GLOBAL_MEM_INSIGHT.contains("raw_session"))
    }

    func testKnowledgeBaseHintKeepsNarrowWording() {
        for s in ["不是磁盘", "本说明只针对知识库文档",
                  "给网址或要实时信息请用 `web_scan`", "均不受限制"] {
            XCTAssertTrue(KNOWLEDGE_BASE_HINT.contains(s), "HINT 缺少窄规则措辞：\(s)")
        }
    }
}
