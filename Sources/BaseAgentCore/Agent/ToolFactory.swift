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

        var tools: [AgentTool] = []

        // 文件操作 → file_tools（op: read/write/modify/patch/list/search）。
        let fileModify = FileModifyTool(config: config, callbacks: callbacks, registry: registry)
        tools.append(CompositeTool(
            name: "file_tools",
            description:
                "Workspace file operations. Choose the action via `op`:\n"
                + "- op=\"read\": read a text file (path, offset, limit, line_numbers); returns "
                + "lines, optionally numbered for precise patching.\n"
                + "- op=\"write\": overwrite or create a file (path, content).\n"
                + "- op=\"modify\" (alias \"patch\"): replace the first occurrence of `old` with "
                + "`new` (path, old, new, replace_all).\n"
                + "- op=\"list\": list directory entries (path, glob, recursive, max_depth, max_entries).\n"
                + "- op=\"search\": regex-search file contents, returns \"path:line: text\" "
                + "(pattern, path, glob, case_sensitive, context, max_results).",
            subcommands: [
                ("read", FileReadTool(config: config, callbacks: callbacks, registry: registry,
                                      sopTracker: sharedSOPTracker)),
                ("write", FileWriteTool(config: config, callbacks: callbacks, registry: registry)),
                ("modify", fileModify),
                ("patch", fileModify),
                ("list", FileListTool(config: config, callbacks: callbacks, registry: registry)),
                ("search", FileSearchTool(config: config, callbacks: callbacks, registry: registry)),
            ]))

        // 网络 → web_scan（op: scan/render）；webexec_js 已删除。
        tools.append(CompositeTool(
            name: "web_scan",
            description:
                "Fetch or render a web page. Choose via `op`:\n"
                + "- op=\"scan\" (default): fetch a URL and return visible text (HTML stripped); "
                + "auto-renders JS-driven pages with the built-in WebKit (url, raw, render).\n"
                + "- op=\"render\": explicit WebKit render — wait for a CSS selector, run JS, "
                + "return text / rendered HTML / script result / screenshot "
                + "(url, wait_ms, wait_selector, script, raw_html, screenshot).",
            subcommands: [
                ("scan", WebScanTool(config: config, callbacks: callbacks, registry: registry)),
                ("render", WebRenderTool(config: config, callbacks: callbacks, registry: registry)),
            ],
            defaultOp: "scan"))

        // 长期记忆 → long_memory（op: graph/search）。
        tools.append(CompositeTool(
            name: "long_memory",
            description:
                "Long-term memory (knowledge graph + semantic recall). Choose via `op`:\n"
                + "- op=\"graph\": manipulate the entity/relation/observation graph "
                + "(graph_op: create_entity / add_observations / create_relation / list_entities / "
                + "get_entity / delete_entity / snapshot / clear).\n"
                + "- op=\"search\": returns <=1000 chars of JSON. Priority: the 5 most "
                + "recent matched observations (hits), then their entities and directly "
                + "connected relations; one-hop neighbor expansion only if budget remains, "
                + "otherwise expansion is dropped. Use the user's original wording for query.",
            subcommands: [
                ("graph", MemoryGraphTool(config: config, callbacks: callbacks, registry: registry)),
                ("search", MemorySearchTool(config: config, callbacks: callbacks, registry: registry)),
            ]))

        // 技能 → skill_tools（op: search/load/semantic_search/save/list）。
        tools.append(CompositeTool(
            name: "skill_tools",
            description:
                "Reusable skills. Choose via `op`:\n"
                + "- op=\"search\": keyword search of saved skills (query, top_k).\n"
                + "- op=\"semantic_search\": semantic/BM25 hybrid search of SKILL.md (query, top_k); "
                + "use the user's original wording for query.\n"
                + "- op=\"load\": load a skill's full instructions by directory (path, entry).\n"
                + "- op=\"save\": save a reusable workflow as a skill "
                + "(name, description, steps, when_to_use, pitfalls, tags).\n"
                + "- op=\"list\": list all saved skills (no arguments).",
            subcommands: [
                ("search", SkillSearchTool(config: config, callbacks: callbacks, registry: registry,
                                           index: skillIndex)),
                ("list", SkillListTool(config: config, callbacks: callbacks, registry: registry,
                                       index: skillIndex)),
                ("semantic_search", SkillSemanticSearchTool(
                    config: config, callbacks: callbacks, registry: registry,
                    retriever: skillRetriever)),
                ("load", SkillLoadTool(config: config, callbacks: callbacks, registry: registry,
                                       retriever: skillRetriever, index: skillIndex)),
                ("save", SkillSaveTool(config: config, callbacks: callbacks, registry: registry,
                                       manager: skillManager, retriever: skillRetriever,
                                       index: skillIndex)),
            ]))

        // 知识库 → rag_tools（op: search/outline/status/ingest）。
        tools.append(CompositeTool(
            name: "rag_tools",
            description:
                "Local knowledge-base (RAG). Choose via `op`:\n"
                + "- op=\"search\": retrieve relevant chunks (query, source, top_k); "
                + "use the user's original wording for query.\n"
                + "- op=\"outline\": view a document's chapter outline/summaries, or list all "
                + "documents (book_name).\n"
                + "- op=\"status\": knowledge-base indexing status.\n"
                + "- op=\"ingest\": build/update the index (force).",
            subcommands: [
                ("search", RagSearchTool(config: config, callbacks: callbacks, registry: registry,
                                         engine: ragEngine)),
                ("outline", RagOutlineTool(config: config, callbacks: callbacks, registry: registry,
                                           engine: ragEngine)),
                ("status", RagStatusTool(config: config, callbacks: callbacks, registry: registry,
                                         engine: ragEngine)),
                ("ingest", RagIngestTool(config: config, callbacks: callbacks, registry: registry,
                                         engine: ragEngine)),
            ]))

        // 未合并的独立工具。
        let standaloneTools: [AgentTool] = [
            CodeRunTool(config: config, callbacks: callbacks, registry: registry),
            ShellRunTool(config: config, callbacks: callbacks, registry: registry),
            AskUserTool(config: config, callbacks: callbacks, registry: registry),

            WorkMemoryTool(config: config, callbacks: callbacks, registry: registry,
                           workMemory: sharedWorkMemory),

            UpdateWorkingCheckpointTool(config: config, callbacks: callbacks, registry: registry),
            StartLongTermUpdateTool(config: config, callbacks: callbacks, registry: registry),
            UpdatePlanTool(config: config, callbacks: callbacks, registry: registry),
        ]
        tools.append(contentsOf: standaloneTools)

        registry.registerAll(tools)
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
