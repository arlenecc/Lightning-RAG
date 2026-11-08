import Foundation

// MARK: - code_run

/// `code_run`：把代码写入临时脚本后执行，捕获 stdout/stderr、退出码，
/// 超时/取消时终止进程并清理临时文件。
///
/// Python 版在内置命名空间 exec，Swift 无法复用解释器，因此统一走「临时脚本 +
/// 子进程」。支持 python / swift / bash / sh / zsh / node / ruby / perl / php。
///
/// 风险分级（`ToolRiskPolicy`）是第一道防线：sudo、管道下载执行、
/// `git push --force`、破坏性调用作用到工作区之外路径时弹人工确认。
public final class CodeRunTool: AgentTool {
    public let name = "code_run"
    public let description =
        "Execute code (multi-line). stdout/stderr captured. "
        + "Can import and pip install packages, call APIs, write files in workspace. "
        + "This is the capability core: anything not covered by a dedicated tool can "
        + "be done here. For system commands / CLI tools prefer shell_run "
        + "(or call it from here via subprocess). "
        + "Supported languages: python (default), swift, bash/sh/zsh, node, ruby, perl, php. "
        + "Killed after 30s unless overridden."
    public let destructive = true

    private let config: AgentConfig
    private let callbacks: AgentCallbacks?
    private let defaultTimeout: Double = 30

    public var parameters: JSONValue {
        ToolSupport.objectSchema(
            properties: [
                "code": ToolSupport.stringProperty("Source code to execute."),
                "language": ToolSupport.stringProperty(
                    "Language: python (default), swift, bash, sh, zsh, node, ruby, perl, php.",
                    enumValues: ["python", "swift", "bash", "sh", "zsh",
                                 "node", "ruby", "perl", "php"]),
                "timeout": ToolSupport.numberProperty(
                    "Timeout in seconds (default \(defaultTimeout))."),
            ],
            required: ["code"])
    }

    public init(config: AgentConfig, callbacks: AgentCallbacks?, registry: ToolRegistry? = nil) {
        self.config = config
        self.callbacks = callbacks
        _ = registry
    }

    public func shouldConfirm(_ args: [String: JSONValue]) -> Bool {
        if config.alwaysConfirmCode { return true }
        let code = args["code"].map(ToolSupport.coerceString) ?? ""
        let language = (ToolSupport.optionalString(args, "language") ?? "python")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        // bash/sh/zsh 的脚本本质是 shell 命令，用 shell 侧风险表判定。
        if ["bash", "sh", "zsh", "shell"].contains(language) {
            return ToolRiskPolicy.isRiskyCommand(code, workspace: config.workspace)
        }
        return ToolRiskPolicy.isRiskyCode(code, workspace: config.workspace)
    }

    public func run(_ args: [String: JSONValue]) async throws -> ToolResult {
        do {
            let code = try ToolSupport.requireString(args, "code")
            let language = (ToolSupport.optionalString(args, "language") ?? "python")
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            guard let spec = languageSpec(language) else {
                return .fail(
                    "Unsupported language: \(language). Supported: "
                    + "python, swift, bash, sh, zsh, node, ruby, perl, php")
            }

            let timeout = ToolSupport.clampTimeout(
                ToolSupport.optionalDouble(args, "timeout"),
                default: defaultTimeout)

            _ = config.ensureWorkspace()
            let workspace = ToolSupport.realPath(config.workspace)
            guard FileManager.default.fileExists(atPath: workspace) else {
                return .fail("workspace does not exist: \(config.workspace)")
            }

            // 临时脚本放在系统临时目录：不污染 workspace，退出时删除。
            let tempDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("base-agent-code-run", isDirectory: true)
            try FileManager.default.createDirectory(at: tempDir,
                                                    withIntermediateDirectories: true)
            let scriptURL = tempDir
                .appendingPathComponent("\(UUID().uuidString).\(spec.fileExtension)")
            defer { try? FileManager.default.removeItem(at: scriptURL) }
            try code.write(to: scriptURL, atomically: true, encoding: .utf8)

            let callbacks = self.callbacks
            // Python 代码通过 sitecustomize 注入「工作区外禁写 + L0/L1 记忆禁写」
            // 护栏。其他语言没有可注入的等价钩子，只依赖 shouldConfirm 的风险判定
            // 与文件工具侧的守卫（shell/MCP 属 OS 级能力，无法在此层拦截）。
            var environment: [String: String]? = nil
            var guardDirectory: URL? = nil
            if spec.prefixArguments.contains("python3") {
                let dir = tempDir.appendingPathComponent("guard-\(UUID().uuidString)")
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                try Self.pythonWriteGuard.write(to: dir.appendingPathComponent("sitecustomize.py"),
                                               atomically: true, encoding: .utf8)
                guardDirectory = dir
                var env = ToolSupport.sanitizedEnvironment()
                env["BASE_AGENT_WORKSPACE"] = workspace
                let previous = env["PYTHONPATH"] ?? ""
                env["PYTHONPATH"] = previous.isEmpty ? dir.path : dir.path + ":" + previous
                environment = env
            }
            defer {
                if let guardDirectory { try? FileManager.default.removeItem(at: guardDirectory) }
            }

            let outcome = await ToolProcessRunner.run(
                executable: spec.executable,
                arguments: spec.prefixArguments + [scriptURL.path],
                currentDirectory: workspace,
                environment: environment,
                timeout: timeout,
                shouldCancel: { callbacks?.isCancelled() ?? false })

            if let launchError = outcome.launchError {
                return .fail(launchError)
            }

            let output = ToolSupport.combineOutput(stdout: outcome.stdout,
                                                   stderr: outcome.stderr)

            if outcome.cancelled {
                var message = "用户取消了对话，代码执行被中断"
                if !output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    message += "\n[取消前已产生的输出]\n\(String(output.suffix(3000)))"
                }
                return .fail(message)
            }

            if outcome.timedOut {
                var message = "Code execution timed out after \(Int(timeout))s"
                if !output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    message += "\n[超时前已产生的输出]\n\(String(output.suffix(3000)))"
                }
                return .fail(message)
            }

            if outcome.exitCode == 0 {
                return .ok(output.isEmpty ? "(no output)" : output)
            }

            var detail = outcome.exitCode.map { "Exit code \($0)" }
                ?? "进程未能被回收（可能仍在运行或被信号终止）"
            // ToolResult.toMessage 失败时只渲染 error；把 traceback 带进 error，
            // 否则模型只看到“Exit code 1”而无从定位（对齐 Python 的做法）。
            if !outcome.stderr.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                detail += "\n" + String(outcome.stderr.suffix(3_000))
            }
            return ToolResult(success: false, output: output, error: detail)
        } catch {
            return ToolSupport.failure(error)
        }
    }

    // MARK: 语言映射

    /// sitecustomize 写护栏：拦截 builtins.open / io.open / os.open 的写模式，
    /// 阻止 Python 代码把工作区外的文件改写。允许 int fd（subprocess 管道需要）。
    /// 顶部预置资源限额（CPU / 单文件写大小）：macOS 对 RLIMIT_AS 不生效，
    /// 但 CPU 与 FSIZE 两个 rlimit 可用——死循环的 CPU 与写爆磁盘的文件
    /// 在内核超时兜底之外再多一层 OS 级闸门。
    private static let pythonWriteGuard = """
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024 * 1024,) * 2)
    except Exception:
        pass

    import builtins
    import io
    import os

    _WORKSPACE = os.path.realpath(os.environ.get("BASE_AGENT_WORKSPACE", ""))
    # L0/L1 会逐字进入 system prompt：即便位于工作区内，也不许脚本改写，
    # 否则一次提示词注入就能持久化到之后所有会话。
    _MEMORY_PROMPT_FILES = set([
        os.path.realpath(os.path.join(_WORKSPACE, "memory", "memory_management_sop.md")),
        os.path.realpath(os.path.join(_WORKSPACE, "memory", "global_mem_insight.txt")),
    ])

    def _inside(path):
        try:
            real = os.path.realpath(os.fspath(path))
        except Exception:
            return False
        if real in _MEMORY_PROMPT_FILES:
            raise PermissionError(
                "code_run write guard: refusing to rewrite system-prompt memory: %r" % (path,))
        return bool(_WORKSPACE) and (real == _WORKSPACE or real.startswith(_WORKSPACE + os.sep))

    def _write_mode(mode):
        if isinstance(mode, int):
            return bool(mode & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
        text = str(mode)
        return any(ch in text for ch in "wax+")

    _real_builtin_open = builtins.open
    _real_io_open = io.open
    _real_os_open = os.open

    def _deny(path):
        raise PermissionError(
            "code_run write guard: refusing to write outside workspace: %r" % (path,))

    def _guard_builtin_open(file, mode="r", *args, **kwargs):
        if isinstance(file, int):
            raise PermissionError("code_run write guard: int fd open is not allowed")
        if _write_mode(mode) and not _inside(file):
            _deny(file)
        return _real_builtin_open(file, mode, *args, **kwargs)

    def _guard_io_open(file, mode="r", *args, **kwargs):
        if not isinstance(file, int) and _write_mode(mode) and not _inside(file):
            _deny(file)
        return _real_io_open(file, mode, *args, **kwargs)

    def _guard_os_open(path, flags, *args, **kwargs):
        if _write_mode(flags) and not _inside(path):
            _deny(path)
        return _real_os_open(path, flags, *args, **kwargs)

    builtins.open = _guard_builtin_open
    io.open = _guard_io_open
    os.open = _guard_os_open
    """

    private struct LanguageSpec {
        let fileExtension: String
        let executable: String
        let prefixArguments: [String]
    }

    private func languageSpec(_ language: String) -> LanguageSpec? {
        switch language {
        case "python", "python3", "py":
            return LanguageSpec(fileExtension: "py",
                                executable: "/usr/bin/env",
                                prefixArguments: ["python3"])
        case "swift":
            return LanguageSpec(fileExtension: "swift",
                                executable: "/usr/bin/env",
                                prefixArguments: ["swift"])
        case "bash", "shell":
            return LanguageSpec(fileExtension: "sh",
                                executable: "/bin/bash",
                                prefixArguments: [])
        case "sh":
            return LanguageSpec(fileExtension: "sh",
                                executable: "/bin/sh",
                                prefixArguments: [])
        case "zsh":
            return LanguageSpec(fileExtension: "zsh",
                                executable: "/bin/zsh",
                                prefixArguments: [])
        case "node", "javascript", "js":
            return LanguageSpec(fileExtension: "js",
                                executable: "/usr/bin/env",
                                prefixArguments: ["node"])
        case "ruby":
            return LanguageSpec(fileExtension: "rb",
                                executable: "/usr/bin/env",
                                prefixArguments: ["ruby"])
        case "perl":
            return LanguageSpec(fileExtension: "pl",
                                executable: "/usr/bin/env",
                                prefixArguments: ["perl"])
        case "php":
            return LanguageSpec(fileExtension: "php",
                                executable: "/usr/bin/env",
                                prefixArguments: ["php"])
        default:
            return nil
        }
    }
}
