import Foundation
import Hub
import Tokenizers
import MLXLMCommon

/// 一批已 padding 的 token 序列。
public struct TokenizedBatch: Sendable {
    public var inputIDs: [[Int]]
    public var attentionMask: [[Int]]

    public init(inputIDs: [[Int]], attentionMask: [[Int]]) {
        self.inputIDs = inputIDs
        self.attentionMask = attentionMask
    }

    public var batchSize: Int { inputIDs.count }
    public var sequenceLength: Int { inputIDs.first?.count ?? 0 }
}

/// 本地 tokenizer 封装：encode/decode、截断、padding、pair 编码与 chat template。
///
/// `Tokenizer` 由 `swift-transformers` 的 `AutoTokenizer` 加载；同一模型目录只应加载一次。
public final class LocalTokenizer: @unchecked Sendable {
    public let base: any Tokenizers.Tokenizer
    public let directory: URL
    /// padding token；tokenizer_config.json 缺失时回退到 eos / 0。
    public let padTokenId: Int
    public let tokenizerClass: String
    public let modelType: String
    /// XLM-R 风格（`<s> A </s></s> B </s>`）的 pair 编码。
    public let usesXLMRPairFormat: Bool

    private init(
        base: any Tokenizers.Tokenizer,
        directory: URL,
        padTokenId: Int,
        tokenizerClass: String,
        modelType: String
    ) {
        self.base = base
        self.directory = directory
        self.padTokenId = padTokenId
        self.tokenizerClass = tokenizerClass
        self.modelType = modelType
        let lowerClass = tokenizerClass.lowercased()
        let lowerType = modelType.lowercased()
        self.usesXLMRPairFormat = lowerClass.contains("xlmroberta")
            || lowerClass.contains("xlm-roberta")
            || lowerType.contains("xlm-roberta")
            || lowerType.contains("xlm_roberta")
    }

    /// 从模型目录加载 tokenizer。
    ///
    /// `tokenizer.json` 必须存在；`tokenizer_config.json` 可选。部分转换产物
    /// （如 jina-v5 ONNX/MLX 目录）只带 tokenizer.json，此时用 tokenizer.json
    /// 自身的 `padding` 元数据与合成的最小配置构建 tokenizer。
    public static func load(directory: URL) async throws -> LocalTokenizer {
        let fm = FileManager.default
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
        let configURL = directory.appendingPathComponent("tokenizer_config.json")

        guard fm.fileExists(atPath: tokenizerURL.path) else {
            throw LocalModelError.missingTokenizer(directory: directory.path, file: "tokenizer.json")
        }

        let hasTokenizerConfig = fm.fileExists(atPath: configURL.path)
        let config: [String: Any]
        let base: any Tokenizers.Tokenizer
        do {
            if hasTokenizerConfig {
                // 非严格模式：tokenizer_config 里的 tokenizer_class 可能来自更新版本的
                // transformers（如 Qwen3_5Tokenizer）；只要 tokenizer.json 是标准 BPE，
                // 就应回退到 BPE 而不是直接拒绝加载整个模型。
                base = try await Tokenizers.AutoTokenizer.from(
                    modelFolder: directory, strict: false)
                config = readJSON(configURL) ?? [:]
            } else {
                let tokenizerData = try HubApi.shared.configuration(fileURL: tokenizerURL)
                // 最小合成配置：没有 tokenizer_config 时按通用 PreTrainedTokenizer
                // 构建；特殊 token 与 padding 由 tokenizer.json 提供。
                let synthetic = Config([
                    "tokenizer_class": Config("PreTrainedTokenizer"),
                    "model_max_length": Config(8192),
                ])
                base = try Tokenizers.AutoTokenizer.from(
                    tokenizerConfig: synthetic,
                    tokenizerData: tokenizerData,
                    strict: false)
                config = readJSON(tokenizerURL) ?? [:]
            }
        } catch {
            throw LocalModelError.runtime("tokenizer 加载失败（\(directory.path)）：\(error.localizedDescription)")
        }

        let tokenizerClass = (config["tokenizer_class"] as? String)
            ?? (config["model_type"] as? String)
            ?? "PreTrainedTokenizer"
        let modelType = (config["model_type"] as? String) ?? ""

        var padID: Int?
        if let value = config["pad_token_id"] as? Int, value >= 0 {
            padID = value
        } else if let token = config["pad_token"] as? String, !token.isEmpty {
            padID = base.convertTokenToId(token)
        }
        if padID == nil,
           let padding = config["padding"] as? [String: Any] {
            if let value = padding["pad_id"] as? Int, value >= 0 {
                padID = value
            } else if let token = padding["pad_token"] as? String, !token.isEmpty {
                padID = base.convertTokenToId(token)
            }
        }
        if padID == nil {
            padID = base.eosTokenId
        }

        return LocalTokenizer(
            base: base,
            directory: directory,
            padTokenId: padID ?? 0,
            tokenizerClass: tokenizerClass,
            modelType: modelType
        )
    }

    // MARK: - 单条编码 / 解码

    public func encode(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        base.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    /// 编码并按 maxLength 从右侧截断。
    public func encode(_ text: String, addSpecialTokens: Bool = true, maxLength: Int) -> [Int] {
        truncate(base.encode(text: text, addSpecialTokens: addSpecialTokens), maxLength: maxLength)
    }

    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) -> String {
        base.decode(tokens: ids, skipSpecialTokens: skipSpecialTokens)
    }

    // MARK: - 批量编码与 padding

    /// 对一组文本做右 padding，返回 input_ids 与 attention_mask。
    public func encodeBatch(
        _ texts: [String],
        maxLength: Int,
        addSpecialTokens: Bool = true
    ) -> TokenizedBatch {
        var sequences: [[Int]] = []
        sequences.reserveCapacity(texts.count)
        for text in texts {
            var ids = base.encode(text: text, addSpecialTokens: addSpecialTokens)
            ids = truncate(ids, maxLength: maxLength)
            if ids.isEmpty {
                // 空文本也要保留一个位置，避免 ONNX 收到 T=0 的张量。
                ids = [padTokenId]
            }
            sequences.append(ids)
        }
        return pad(sequences)
    }

    /// 对已编码序列做右 padding。
    public func pad(_ sequences: [[Int]]) -> TokenizedBatch {
        let width = sequences.map(\.count).max() ?? 0
        var ids: [[Int]] = []
        var masks: [[Int]] = []
        ids.reserveCapacity(sequences.count)
        masks.reserveCapacity(sequences.count)

        for sequence in sequences {
            let padding = max(0, width - sequence.count)
            ids.append(sequence + Array(repeating: padTokenId, count: padding))
            masks.append(Array(repeating: 1, count: sequence.count) + Array(repeating: 0, count: padding))
        }
        return TokenizedBatch(inputIDs: ids, attentionMask: masks)
    }

    // MARK: - 交叉编码器 pair 编码

    /// 将 query/document 编码成 reranker 需要的 pair 序列。
    ///
    /// - XLM-R（bge-reranker-v2-m3 / jina-reranker-v2）：`<s> query </s></s> document </s>`
    /// - BERT 系：`[CLS] query [SEP] document [SEP]`
    ///
    /// 部分 tokenizer（如 XLM-R）在 swift-transformers 里不会从 tokenizer_config
    /// 填充 `bosTokenId`，需要按 token 字符串回查；否则会误走 BERT 分支，
    /// 生成 `<unk> query </s> document </s>`，cross-encoder 分数会明显失真。
    public func encodePair(query: String, document: String, maxLength: Int) -> [Int] {
        let queryIDs = base.encode(text: query, addSpecialTokens: false)
        let documentIDs = base.encode(text: document, addSpecialTokens: false)

        var ids: [Int] = []
        if usesXLMRPairFormat,
           let bos = resolveSpecialTokenID(base.bosTokenId, token: base.bosToken,
                                          fallbacks: ["<s>"]),
           let eos = resolveSpecialTokenID(base.eosTokenId, token: base.eosToken,
                                          fallbacks: ["</s>"]) {
            ids.append(bos)
            ids.append(contentsOf: queryIDs)
            ids.append(eos)
            ids.append(eos)
            ids.append(contentsOf: documentIDs)
            ids.append(eos)
            // 截断必须保留末尾的 `</s>`，否则 document 段没有结束标记。
            if ids.count > maxLength {
                ids = Array(ids.prefix(max(1, maxLength - 1)))
                ids.append(eos)
            }
            return ids
        }

        let cls = resolveSpecialTokenID(base.convertTokenToId("[CLS]"), token: nil,
                                        fallbacks: ["[CLS]", "<s>"])
            ?? resolveSpecialTokenID(base.bosTokenId, token: base.bosToken,
                                     fallbacks: ["<s>"])
        let sep = resolveSpecialTokenID(base.convertTokenToId("[SEP]"), token: nil,
                                        fallbacks: ["[SEP]", "</s>"])
            ?? resolveSpecialTokenID(base.eosTokenId, token: base.eosToken,
                                     fallbacks: ["</s>"])
        if let cls { ids.append(cls) }
        ids.append(contentsOf: queryIDs)
        if let sep { ids.append(sep) }
        ids.append(contentsOf: documentIDs)
        if let sep { ids.append(sep) }
        if ids.count > maxLength {
            ids = Array(ids.prefix(maxLength))
        }
        return ids
    }

    /// 解析特殊 token id：显式 id 优先，其次 token 字符串，最后按约定别名回查。
    /// 未知 token（`<unk>`）不能当作特殊 token 使用。
    private func resolveSpecialTokenID(_ explicit: Int?, token: String?,
                                       fallbacks: [String] = []) -> Int? {
        if let explicit, explicit >= 0, explicit != base.unknownTokenId {
            return explicit
        }
        if let token, let id = base.convertTokenToId(token),
           id != base.unknownTokenId {
            return id
        }
        for fallback in fallbacks {
            if let id = base.convertTokenToId(fallback),
               id != base.unknownTokenId {
                return id
            }
        }
        return nil
    }

    // MARK: - Chat template

    /// 应用模型自带的 chat template。`enable_thinking` 传给 Qwen3 系模板。
    public func applyChatTemplate(
        messages: [ChatMessage],
        tools: [JSONValue]? = nil,
        enableThinking: Bool
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages, tools: tools,
                              enableThinking: enableThinking, addGenerationPrompt: true)
    }

    /// 显式控制是否追加 generation prompt；预热只需 system + tools 前缀。
    public func applyChatTemplate(
        messages: [ChatMessage],
        tools: [JSONValue]?,
        enableThinking: Bool,
        addGenerationPrompt: Bool
    ) throws -> [Int] {
        guard base.hasChatTemplate else {
            throw LocalModelError.invalidConfiguration(
                "tokenizer_config.json 未提供 chat_template（目录：\(directory.path)）"
            )
        }

        let formatted: [Tokenizers.Message] = messages.map { message in
            var dictionary: [String: any Sendable] = [
                "role": message.role.rawValue,
                "content": message.content,
            ]
            if let name = message.name, !name.isEmpty {
                dictionary["name"] = name
            }
            if let toolCallID = message.toolCallID, !toolCallID.isEmpty {
                dictionary["tool_call_id"] = toolCallID
            }
            if let toolCalls = message.toolCalls, !toolCalls.isEmpty {
                dictionary["tool_calls"] = toolCalls.map { call -> [String: any Sendable] in
                    let arguments = call.rawArguments.isEmpty
                        ? JSONValue.object(call.arguments).jsonString()
                        : call.rawArguments
                    return [
                        "id": call.id,
                        "type": "function",
                        "function": [
                            "name": call.name,
                            "arguments": arguments,
                        ] as [String: any Sendable],
                    ]
                }
            }
            return dictionary
        }

        let toolSpecs: [Tokenizers.ToolSpec]? = tools?.compactMap { tool -> Tokenizers.ToolSpec? in
            guard let object = tool.objectValue else { return nil }
            var result: [String: any Sendable] = [:]
            for (key, value) in object {
                result[key] = sendableValue(value)
            }
            return result
        }

        let context: [String: any Sendable] = ["enable_thinking": enableThinking]
        do {
            return try base.applyChatTemplate(
                messages: formatted,
                chatTemplate: nil,
                addGenerationPrompt: addGenerationPrompt,
                truncation: false,
                maxLength: nil,
                tools: toolSpecs,
                additionalContext: context
            )
        } catch {
            throw LocalModelError.runtime("chat template 应用失败：\(error.localizedDescription)")
        }
    }

    // MARK: - 内部工具

    private func truncate(_ ids: [Int], maxLength: Int) -> [Int] {
        guard maxLength > 0, ids.count > maxLength else { return ids }
        return Array(ids.prefix(maxLength))
    }

    private static func readJSON(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data),
              let dictionary = object as? [String: Any] else {
            return nil
        }
        return dictionary
    }
}

/// 将 `JSONValue` 递归转换为 Swift `Sendable` 容器，供 Jinja chat template 使用。
private func sendableValue(_ value: JSONValue) -> any Sendable {
    switch value {
    case .null:
        // 用 Optional<Int>.none 表示 null，避免依赖 NSNull 的 Sendable 标注。
        return Int?.none
    case .bool(let value):
        return value
    case .int(let value):
        return value
    case .double(let value):
        return value
    case .string(let value):
        return value
    case .array(let values):
        return values.map { sendableValue($0) }
    case .object(let object):
        var result: [String: any Sendable] = [:]
        for (key, value) in object {
            result[key] = sendableValue(value)
        }
        return result
    }
}

/// HF chat template 与 `swift-jinja` 的语法兼容层。
///
/// Spark-X2.6 的模板为了紧凑 JSON 写了
/// `tojson(separators=(',', ':'))` / `tojson(separators=(', ', ': '))`。
/// `swift-jinja` 的 `tojson` 只接受 `indent` 等参数，遇到 `separators`
/// 会抛 `Unexpected keyword argument 'separators' for filter.` ——
/// 只要 prompt 带 tools 就会命中，模型连第一次推理都跑不起来。
/// 补丁只去掉这个关键字参数，JSON 内容等价（仅空白不同）。
public enum ChatTemplateCompatibility {
    public static func strippingSeparators(_ template: String) -> String {
        template.replacingOccurrences(
            of: #"tojson\s*\(\s*separators\s*=\s*\([^)]*\)\s*\)"#,
            with: "tojson",
            options: .regularExpression)
    }

    /// 仅当模板确实含不兼容语法时返回覆盖用的字面量模板；其它模型
    /// 继续走 `tokenizer_config.json` 的原生模板，行为零变化。
    public static func overrideTemplate(in directory: URL) -> String? {
        let fm = FileManager.default
        var templates: [String] = []
        if let data = fm.contents(
            atPath: directory.appendingPathComponent("chat_template.jinja").path),
           let text = String(data: data, encoding: .utf8) {
            templates.append(text)
        }
        if let data = fm.contents(
            atPath: directory.appendingPathComponent("tokenizer_config.json").path),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let text = json["chat_template"] as? String {
                templates.append(text)
            } else if let list = json["chat_template"] as? [String] {
                templates.append(contentsOf: list)
            }
        }
        for template in templates {
            let patched = strippingSeparators(template)
            if patched != template { return patched }
        }
        return nil
    }
}

/// 把 `swift-transformers` 的 tokenizer 桥接给 `MLXLMCommon`。
public final class MLXTokenizerBridge: MLXLMCommon.Tokenizer, @unchecked Sendable {
    private let base: any Tokenizers.Tokenizer
    /// 非 nil 时用字面量模板覆盖 tokenizer_config 里的模板（兼容补丁）。
    private let chatTemplateOverride: String?

    public init(_ base: any Tokenizers.Tokenizer,
                chatTemplateOverride: String? = nil) {
        self.base = base
        self.chatTemplateOverride = chatTemplateOverride
    }

    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        base.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        base.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    public func convertTokenToId(_ token: String) -> Int? {
        base.convertTokenToId(token)
    }

    public func convertIdToToken(_ id: Int) -> String? {
        base.convertIdToToken(id)
    }

    public var bosToken: String? { base.bosToken }
    public var eosToken: String? { base.eosToken }
    public var unknownToken: String? { base.unknownToken }

    public func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try render(messages: messages, tools: tools,
                   additionalContext: additionalContext,
                   addGenerationPrompt: true)
    }

    /// 显式控制是否追加 generation prompt。
    ///
    /// 预热只缓存 system + tools 前缀，不能把 `<|im_start|>assistant` 头
    /// 也预填充进去，否则正式请求的 user 增量会接在 assistant 头后面。
    public func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?,
        addGenerationPrompt: Bool
    ) throws -> [Int] {
        try render(messages: messages, tools: tools,
                   additionalContext: additionalContext,
                   addGenerationPrompt: addGenerationPrompt)
    }

    private func render(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?,
        addGenerationPrompt: Bool
    ) throws -> [Int] {
        try base.applyChatTemplate(
            messages: messages,
            chatTemplate: chatTemplateOverride.map {
                Tokenizers.ChatTemplateArgument.literal($0)
            },
            addGenerationPrompt: addGenerationPrompt,
            truncation: false,
            maxLength: nil,
            tools: tools,
            additionalContext: additionalContext
        )
    }
}

/// 供 `MLXLLM` / `MLXVLM` 本地目录加载使用的 tokenizer loader。
public struct LocalTokenizerLoader: MLXLMCommon.TokenizerLoader {
    public init() {}

    public func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let tokenizer = try await LocalTokenizer.load(directory: directory)
        return MLXTokenizerBridge(
            tokenizer.base,
            chatTemplateOverride: ChatTemplateCompatibility.overrideTemplate(in: directory))
    }
}

/// 切片阶段使用的精确 token 计数器：只加载 tokenizer、不加载模型权重，按目录缓存。
public actor TokenizerCounterCache {
    public static let shared = TokenizerCounterCache()

    private var counters: [String: @Sendable (String) -> Int] = [:]

    /// 返回不含特殊 token 的同步计数器；模型目录缺少 tokenizer 时返回 nil。
    public func counter(for directory: URL) async -> (@Sendable (String) -> Int)? {
        let key = directory.standardizedFileURL.path
        if let cached = counters[key] { return cached }
        guard let tokenizer = try? await LocalTokenizer.load(directory: directory) else {
            return nil
        }
        let counter: @Sendable (String) -> Int = { text in
            tokenizer.encode(text, addSpecialTokens: false).count
        }
        counters[key] = counter
        return counter
    }
}
