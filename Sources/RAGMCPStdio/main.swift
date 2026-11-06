import BaseAgentCore
import Foundation
import MCP

// rag-mcp-stdio：Lightning-RAG 知识库的 MCP stdio 服务。
//
// 供 Claude Desktop / Claude Code 等支持 stdio 接入的大模型应用以子进程方式
// 启动：应用配置 command 指向本可执行文件即可（协议实现来自官方 MCP Swift
// SDK 的 StdioTransport，initialize 握手、JSON-RPC 编解码均由 SDK 负责）。
//
// stdout 只承载 MCP 协议消息；所有日志走 stderr（AgentLog 已写 stderr）。
// 用法：
//   rag-mcp-stdio [--config <config.json>] [--kb <知识库目录>]
// 默认读取 ~/.base-agent/config.json（与 App 共用同一份配置与知识库）。

let arguments = Array(CommandLine.arguments.dropFirst())
var configPath = (NSHomeDirectory() as NSString)
    .appendingPathComponent(".base-agent/config.json")
var knowledgeBase: String?

var index = 0
while index < arguments.count {
    switch arguments[index] {
    case "--config" where index + 1 < arguments.count:
        index += 1
        configPath = arguments[index]
    case "--kb" where index + 1 < arguments.count:
        index += 1
        knowledgeBase = arguments[index]
    case "--help", "-h":
        FileHandle.standardError.write(Data("""
        用法：rag-mcp-stdio [--config <config.json>] [--kb <知识库目录>]

        从 stdin/stdout 提供 MCP stdio 服务（工具与 HTTP 版一致）。
        默认配置：\(configPath)
        """.utf8))
        exit(0)
    default:
        FileHandle.standardError.write(Data("未知参数：\(arguments[index])\n".utf8))
        exit(2)
    }
    index += 1
}

var config = AgentConfig.load(configPath)
if let knowledgeBase {
    config.knowledgeBase = knowledgeBase
}
config.ensureWorkspace()

let engine = RAGEngine(config: config)
let toolset = RAGMCPToolset(engine: engine) { line in
    AgentLog.info(line)
}
let server = await toolset.makeServer()

do {
    try await server.start(transport: MCP.StdioTransport())
} catch {
    AgentLog.warning("[rag-mcp-stdio] 启动失败：\(error.localizedDescription)")
    engine.close()
    exit(1)
}
AgentLog.info("[rag-mcp-stdio] 服务已启动（stdio，知识库：\(engine.knowledgeBase)）")

// stdin EOF（客户端退出）→ transport 流结束 → waitUntilCompleted 返回。
await server.waitUntilCompleted()
engine.close()
