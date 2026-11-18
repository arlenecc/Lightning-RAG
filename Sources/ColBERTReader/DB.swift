import Foundation
import SQLite3
import CSQLiteVec

enum DocSearchError: LocalizedError, Sendable {
    case storage(String)
    case model(String)
    case chunking(String)
    case quant(String)
    case argument(String)

    var errorDescription: String? {
        switch self {
        case .storage(let s): return "SQLite: \(s)"
        case .model(let s): return "ColBERT: \(s)"
        case .chunking(let s): return "切片: \(s)"
        case .quant(let s): return "残差量化: \(s)"
        case .argument(let s): return "参数: \(s)"
        }
    }
}

enum SQLValue {
    case null
    case int(Int64)
    case double(Double)
    case text(String)
    case blob(Data)

    init(_ v: Int) { self = .int(Int64(v)) }
    init(_ v: Int64) { self = .int(v) }
    init(_ v: Double) { self = .double(v) }
    init(_ v: String) { self = .text(v) }
    init(_ v: Data) { self = .blob(v) }

    static func floats(_ values: [Float]) -> SQLValue {
        .blob(values.withUnsafeBufferPointer { Data(buffer: $0) })
    }

    var intValue: Int64? {
        switch self {
        case .int(let v): return v
        case .double(let v): return Int64(v)
        case .text(let s): return Int64(s)
        default: return nil
        }
    }
    var doubleValue: Double? {
        switch self {
        case .double(let v): return v
        case .int(let v): return Double(v)
        case .text(let s): return Double(s)
        default: return nil
        }
    }
    var stringValue: String? {
        switch self {
        case .text(let s): return s
        case .int(let v): return String(v)
        case .double(let v): return String(v)
        default: return nil
        }
    }
    var dataValue: Data? {
        if case .blob(let d) = self { return d }
        return nil
    }
}

private let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

/// 轻量 SQLite 封装：WAL、事务、预编译语句、sqlite-vec 静态注册。
final class SQLiteDB {
    private var handle: OpaquePointer?
    private let lock = NSRecursiveLock()
    let path: String
    private(set) var vecLoaded = false

    init(path: String, readOnly: Bool = false) throws {
        self.path = path
        var db: OpaquePointer?
        let flags = readOnly ? SQLITE_OPEN_READONLY : (SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE)
        let rc = sqlite3_open_v2(path, &db, flags | SQLITE_OPEN_FULLMUTEX, nil)
        guard rc == SQLITE_OK, let db else {
            let msg = db.map { String(cString: sqlite3_errmsg($0)) } ?? "open failed"
            if let db { sqlite3_close(db) }
            throw DocSearchError.storage("无法打开 \(path): \(msg)")
        }
        handle = db
        sqlite3_busy_timeout(db, 5000)
        if !readOnly { _ = try? exec("PRAGMA journal_mode=WAL") }
        _ = try? exec("PRAGMA synchronous=NORMAL")
        _ = try? exec("PRAGMA foreign_keys=ON")
        _ = try? exec("PRAGMA temp_store=MEMORY")
        vecLoaded = sqlite3_vec_init(db, nil, nil) == SQLITE_OK
        if !vecLoaded { throw DocSearchError.storage("sqlite-vec 初始化失败，无法创建 vec0 表") }
    }

    deinit { close() }

    /// 显式关闭句柄（幂等）：供上层 shutdown 在对象仍被引用时提前释放连接。
    func close() {
        lock.lock(); defer { lock.unlock() }
        if let handle {
            sqlite3_close_v2(handle)
            self.handle = nil
        }
    }

    @discardableResult
    func exec(_ sql: String) throws -> Int {
        lock.lock(); defer { lock.unlock() }
        guard let handle else { throw DocSearchError.storage("数据库已关闭") }
        var err: UnsafeMutablePointer<CChar>?
        let rc = sqlite3_exec(handle, sql, nil, nil, &err)
        if rc != SQLITE_OK {
            let msg = err.map { String(cString: $0) } ?? "unknown"
            sqlite3_free(err)
            throw DocSearchError.storage("\(msg)\n\(sql.prefix(200))")
        }
        return Int(sqlite3_changes(handle))
    }

    @discardableResult
    func execute(_ sql: String, _ bindings: [SQLValue] = []) throws -> Int {
        lock.lock(); defer { lock.unlock() }
        guard let handle else { throw DocSearchError.storage("数据库已关闭") }
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(handle, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK, let stmt else {
            throw DocSearchError.storage("准备失败: \(String(cString: sqlite3_errmsg(handle)))")
        }
        defer { sqlite3_finalize(stmt) }
        try bind(stmt, bindings)
        let step = sqlite3_step(stmt)
        guard step == SQLITE_DONE || step == SQLITE_ROW else {
            throw DocSearchError.storage("执行失败: \(String(cString: sqlite3_errmsg(handle)))")
        }
        return Int(sqlite3_changes(handle))
    }

    func query(_ sql: String, _ bindings: [SQLValue] = []) throws -> [[String: SQLValue]] {
        lock.lock(); defer { lock.unlock() }
        guard let handle else { throw DocSearchError.storage("数据库已关闭") }
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(handle, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK, let stmt else {
            throw DocSearchError.storage("准备失败: \(String(cString: sqlite3_errmsg(handle)))")
        }
        defer { sqlite3_finalize(stmt) }
        try bind(stmt, bindings)
        var rows: [[String: SQLValue]] = []
        while true {
            let step = sqlite3_step(stmt)
            if step == SQLITE_DONE { break }
            guard step == SQLITE_ROW else {
                throw DocSearchError.storage("查询失败: \(String(cString: sqlite3_errmsg(handle)))")
            }
            var row: [String: SQLValue] = [:]
            for i in 0..<Int32(sqlite3_column_count(stmt)) {
                let name = String(cString: sqlite3_column_name(stmt, i))
                row[name] = column(stmt, i)
            }
            rows.append(row)
        }
        return rows
    }

    func transaction<T>(_ body: () throws -> T) throws -> T {
        lock.lock(); defer { lock.unlock() }
        try exec("BEGIN IMMEDIATE")
        do {
            let value = try body()
            try exec("COMMIT")
            return value
        } catch {
            try? exec("ROLLBACK")
            throw error
        }
    }

    private func bind(_ stmt: OpaquePointer?, _ values: [SQLValue]) throws {
        for (i, v) in values.enumerated() {
            let index = Int32(i + 1)
            switch v {
            case .null: sqlite3_bind_null(stmt, index)
            case .int(let x): sqlite3_bind_int64(stmt, index, x)
            case .double(let x): sqlite3_bind_double(stmt, index, x)
            case .text(let s): sqlite3_bind_text(stmt, index, s, -1, transient)
            case .blob(let d): d.withUnsafeBytes { sqlite3_bind_blob(stmt, index, $0.baseAddress, Int32(d.count), transient) }
            }
        }
    }

    private func column(_ stmt: OpaquePointer?, _ i: Int32) -> SQLValue {
        switch sqlite3_column_type(stmt, i) {
        case SQLITE_INTEGER: return .int(sqlite3_column_int64(stmt, i))
        case SQLITE_FLOAT: return .double(sqlite3_column_double(stmt, i))
        case SQLITE_TEXT:
            guard let p = sqlite3_column_text(stmt, i) else { return .null }
            return .text(String(cString: p))
        case SQLITE_BLOB:
            guard let p = sqlite3_column_blob(stmt, i) else { return .blob(Data()) }
            return .blob(Data(bytes: p, count: Int(sqlite3_column_bytes(stmt, i))))
        default: return .null
        }
    }
}
