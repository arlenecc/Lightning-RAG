import Foundation
import MCP

// MARK: - 应用 JSONValue 与官方 SDK Value 的桥接
//
// 两边都是 Codable，走 JSON 序列化互换即可；开销只在工具调用边界，
// 不在协议传输路径上（协议编解码完全由 SDK 承担）。

extension JSONValue {
    /// 官方 SDK `Value` → 应用 `JSONValue`。
    init(sdkValue: MCP.Value) throws {
        let data = try JSONEncoder().encode(sdkValue)
        self = try JSONDecoder().decode(JSONValue.self, from: data)
    }

    /// 应用 `JSONValue` → 官方 SDK `Value`。
    var sdkValue: MCP.Value {
        get throws {
            let data = Data(jsonString().utf8)
            return try JSONDecoder().decode(MCP.Value.self, from: data)
        }
    }

    /// 宽松版转换：非法值落为 `.null`（工具参数转换失败不应炸掉整次调用）。
    var sdkValueOrNull: MCP.Value {
        (try? sdkValue) ?? .null
    }
}

extension MCP.Value {
    /// 官方 SDK `Value` → 应用 `JSONValue`。
    var jsonValue: JSONValue {
        get throws {
            let data = try JSONEncoder().encode(self)
            return try JSONDecoder().decode(JSONValue.self, from: data)
        }
    }

    /// 宽松版转换：非法值落为 `.null`。
    var jsonValueOrNull: JSONValue {
        (try? jsonValue) ?? .null
    }
}
