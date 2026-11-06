# mcp-swift-sdk（vendored）

官方 [Model Context Protocol Swift SDK](https://github.com/modelcontextprotocol/swift-sdk)
的本地拷贝（对应 swift-sdk-main，MCP 2025-11-25 规范），用于：

- `RAGMCPServer`：RAG 知识库 MCP 服务（Streamable HTTP，`StatefulHTTPServerTransport`）
- `rag-mcp-stdio`：RAG 知识库 MCP 服务（stdio 接入）
- `MCPClient`：应用内 MCP 客户端（子进程 stdio，`StdioTransport` + `Client`）

为避免拉入 conformance 工具链依赖，只保留 `MCP` 库 target 与其三个依赖
（swift-system / swift-log / eventsource），去掉了 docc 插件与
MCPConformance 可执行示例；`Sources/MCP` 内的代码与上游保持一致，未做修改。

升级方式：用上游同版本目录覆盖 `Sources/MCP` 与本 Package.swift 的依赖声明即可。
