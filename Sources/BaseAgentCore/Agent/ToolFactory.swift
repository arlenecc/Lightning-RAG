import Foundation

/// 工具注册表工厂：集中装配所有原生工具、别名与 MCP 工具。
public enum AgentToolFactory {

    /// 创建已注册全部本地工具的注册表。RAG 引擎由 Agent 注入，保证工具与
    /// 系统提示看到同一实例。
    public static func build(config: AgentConfig,
                             callbacks: AgentCallbacks,
                             session: Session,
                             ragEngine: RAGEngine,
                             skillManager: SkillManager,
                             skillIndex: SkillIndex,
                             skillRetriever: SkillRetriever,
                             workMemory: WorkMemory? = nil,
                             sopTracker: SOPReadTracker = SOPReadTracker()) -> ToolRegistry {
        let registry = ToolRegistry(config: config, callbacks: callbacks)
        registry.session = session
        registry.ragEngine = ragEngine
        let sharedWorkMemory = workMemory ?? WorkMemory(workspace: config.workspace)
        registry.workMemory = sharedWorkMemory
        // 改进5：file_read 使用调用方（Agent）持有的 tracker，reset() 才能清空。
        let sharedSOPTracker = sopTracker

        var tools: [AgentTool] = [
            FileReadTool(config: config, callbacks: callbacks, registry: registry,
                         sopTracker: sharedSOPTracker),
            FileWriteTool(config: config, callbacks: callbacks, registry: registry),
            FileModifyTool(config: config, callbacks: callbacks, registry: registry),
            CodeRunTool(config: config, callbacks: callbacks, registry: registry),
            ShellRunTool(config: config, callbacks: callbacks, registry: registry),
            WebScanTool(config: config, callbacks: callbacks, registry: registry),
            AskUserTool(config: config, callbacks: callbacks, registry: registry),

            WorkMemoryTool(config: config, callbacks: callbacks, registry: registry,
                           workMemory: sharedWorkMemory),
            MemoryGraphTool(config: config, callbacks: callbacks, registry: registry),
            MemorySearchTool(config: config, callbacks: callbacks, registry: registry),

            SkillSearchTool(config: config, callbacks: callbacks, registry: registry,
                            index: skillIndex),
            SkillLoadTool(config: config, callbacks: callbacks, registry: registry,
                          retriever: skillRetriever, index: skillIndex),
            SkillSemanticSearchTool(config: config, callbacks: callbacks, registry: registry,
                                    retriever: skillRetriever),
            SkillSaveTool(config: config, callbacks: callbacks, registry: registry,
                          manager: skillManager, retriever: skillRetriever, index: skillIndex),

            UpdateWorkingCheckpointTool(config: config, callbacks: callbacks, registry: registry),
            StartLongTermUpdateTool(config: config, callbacks: callbacks, registry: registry),
            UpdatePlanTool(config: config, callbacks: callbacks, registry: registry),

            RagSearchTool(config: config, callbacks: callbacks, registry: registry,
                          engine: ragEngine),
            RagOutlineTool(config: config, callbacks: callbacks, registry: registry,
                           engine: ragEngine),
            RagStatusTool(config: config, callbacks: callbacks, registry: registry,
                          engine: ragEngine),
            RagIngestTool(config: config, callbacks: callbacks, registry: registry,
                          engine: ragEngine),
        ]
        if !config.browserEndpoint.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            tools.append(WebExecJsTool(config: config, callbacks: callbacks, registry: registry))
        }
        // 内置 WebKit 渲染：无需外部浏览器桥，动态页面兜底能力。
        tools.append(WebRenderTool(config: config, callbacks: callbacks, registry: registry))
        registry.registerAll(tools)

        // 原子工具名对齐 GenericAgent。
        if registry.get("file_modify") != nil {
            registry.register(AliasTool(
                targetName: "file_modify", alias: "file_patch",
                description: "Replace a unique block of text (old) with new. old must match exactly "
                    + "(whitespace/indentation). Use file_read first to get current content.",
                registry: registry))
        }
        if registry.get("webexec_js") != nil {
            registry.register(AliasTool(
                targetName: "webexec_js", alias: "web_execute_js",
                description: "Execute JavaScript in the attached browser for full page control.",
                registry: registry))
        }
        return registry
    }

    /// 连接配置中的 MCP 服务并注册其工具。失败只记录日志，不阻塞 Agent 启动。
    ///
    /// `onToolsChanged` 在服务端推送 `notifications/tools/list_changed` 后调用，
    /// 供上层失效工具 schema 缓存。
    @discardableResult
    public static func connectMCP(config: AgentConfig,
                                  callbacks: AgentCallbacks,
                                  registry: ToolRegistry,
                                  onToolsChanged: (() -> Void)? = nil) async -> [MCPClient] {
        var clients: [MCPClient] = []
        for server in config.mcpServers {
            let client = MCPClient(config: server)
            do {
                try await client.start()
                try await client.refreshTools()
                let names = MCPToolNameBox()
                let registered = installMCPTools(
                    client: client, serverName: server.name,
                    definitions: client.listTools(), registry: registry, names: names)
                client.onToolsChanged = { [weak client, weak registry, weak callbacks] definitions in
                    guard let client, let registry else { return }
                    let count = installMCPTools(
                        client: client, serverName: server.name,
                        definitions: definitions, registry: registry, names: names)
                    callbacks?.onLog("[mcp] \(server.name) 工具清单已更新，当前 \(count) 个工具")
                    onToolsChanged?()
                }
                clients.append(client)
                callbacks.onLog("[mcp] \(server.name) 已连接，注册 \(registered) 个工具")
            } catch {
                client.stop()
                callbacks.onLog("[mcp] \(server.name) 连接失败: \(error.localizedDescription)")
            }
        }
        return clients
    }

    /// 注册并同步 MCP 工具；刷新时移除服务端已删除的工具。
    @discardableResult
    private static func installMCPTools(client: MCPClient, serverName: String,
                                        definitions: [JSONValue], registry: ToolRegistry,
                                        names: MCPToolNameBox) -> Int {
        let previous = names.snapshot
        var newTools: [String: AgentTool] = [:]
        for definition in definitions {
            guard let plain = MCPTool(client: client, definition: definition,
                                      serverName: serverName, prefixName: false) else {
                continue
            }
            var chosen: AgentTool = plain
            // 与内置/其它服务重名时加前缀；如果冲突的是本服务上一版的同名工具，
            // 允许直接覆盖（这正是刷新场景）。
            if registry.get(plain.name) != nil, !previous.contains(plain.name) {
                guard let prefixed = MCPTool(client: client, definition: definition,
                                             serverName: serverName, prefixName: true),
                      registry.get(prefixed.name) == nil else {
                    AgentLog.warning("MCP \(serverName): 工具名冲突，已跳过 \(plain.name)")
                    continue
                }
                chosen = prefixed
            }
            newTools[chosen.name] = chosen
        }
        for stale in previous.subtracting(newTools.keys) {
            registry.unregister(stale)
        }
        registry.registerAll(Array(newTools.values))
        names.replace(Set(newTools.keys))
        return newTools.count
    }
}

/// 记录某个 MCP 服务上一版注册过的工具名（刷新时用于清理）。
private final class MCPToolNameBox: @unchecked Sendable {
    private let lock = NSLock()
    private var names: Set<String> = []

    var snapshot: Set<String> {
        lock.lock(); defer { lock.unlock() }
        return names
    }

    func replace(_ newNames: Set<String>) {
        lock.lock(); names = newNames; lock.unlock()
    }
}
