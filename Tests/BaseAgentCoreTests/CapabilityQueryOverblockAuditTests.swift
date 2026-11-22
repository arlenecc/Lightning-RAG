import XCTest
@testable import BaseAgentCore

/// 审查轻量轮拦截的所有可能误命中：除纯寒暄、明确能力问询外一律不得拦截。
final class CapabilityQueryOverblockAuditTests: XCTestCase {
    /// 这些是真实任务/内容问句，必须走 LLM，不能被本地直答吞掉。
    private let mustNotBlock = [
        // 知识库/分区内容（截图 bug 家族）
        "社会分区具体有什么", "知识库的社会分区具体有什么", "你知识库里有什么",
        "历史分区有哪些书", "自然分区有什么资料", "列出知识库文档清单",
        "有哪些知识分区", "知识库索引状态怎么样",
        // 工具相关但是任务（不是问能力）
        "列出工具调用记录", "看一下有哪些工具调用失败了",
        "介绍一下这个工具的用法", "shell_run 有什么参数",
        "工具调用历史里有什么错误", "有哪些工具支持联网",
        // 「会/能/可以」嵌在普通词里
        "社会变迁史有什么书", "会议纪要能整理一下吗", "这个可以转成md吗",
        "太阳能发电有哪些书", "能不能帮我查一下天气",
        // 普通问句/任务
        "为什么失败了", "怎么解决这个问题", "帮我分析一下",
        "把books-md里的文件分类", "读一下第一章",
        "什么是量子力学", "1+1等于几", "翻译这段话：hello",
        "写下今天的总结", "运行测试", "检查一下配置",
        // 更多「能/会/可以」嵌在普通词或任务里
        "会议记录能总结一下吗", "这本书会讲什么内容", "你能帮我改下这个文件吗",
        "可以给我列个计划吗", "这个技能能加载吗", "工具能联网吗",
        "有哪些工具可以处理图片", "功能怎么用", "能力评估怎么做",
        // 身份但不是问「你是谁」
        "你叫什么名字来着", "你今年多大了呢",
    ]

    /// 这些是纯寒暄或明确能力问询，必须走轻量直答。
    private let mustBlock = [
        "你好", "您好", "hi", "hello", "hey", "嗨", "谢谢", "多谢", "再见", "拜拜",
        "好的", "好", "行", "可以", "嗯", "哦", "ok", "okay",
        "你能做什么", "你可以干什么", "你都能做什么", "你会什么",
        "你有什么工具", "有哪些技能", "你支持哪些功能",
        "介绍一下你自己", "你是谁", "你是什么", "what can you do",
        "capabilities", "features", "你的能力有哪些",
    ]

    func testNoOverblockingOfRealTasks() {
        let blocked = mustNotBlock.filter {
            SkillManager.isGreeting($0) || SkillManager.isCapabilityQuery($0)
        }
        XCTAssertTrue(blocked.isEmpty, "以下真实任务/内容问句被误拦截: \(blocked)")
    }

    func testGreetingAndCapabilityStillBlocked() {
        let missed = mustBlock.filter {
            !SkillManager.isGreeting($0) && !SkillManager.isCapabilityQuery($0)
        }
        XCTAssertTrue(missed.isEmpty, "以下寒暄/能力问询未被轻量直答: \(missed)")
    }
}
