// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "BaseAgent",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "BaseAgentCore", targets: ["BaseAgentCore"]),
        .executable(name: "Lightning-RAG", targets: ["BaseAgentApp"]),
    ],
    dependencies: [
        // Tokenizer：swift-transformers 的 Tokenizers 模块
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        // MLX 模型加载（LLM / VLM / Embedding）
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.32.0"),
        // ONNX Runtime（embedding / reranker）
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", from: "1.24.0"),
        // EPUB / OOXML 解包
        .package(url: "https://github.com/weichsel/ZipFoundation", from: "0.9.19"),
        // EPUB OPF/NCX 解析
        .package(url: "https://github.com/tadija/AEXML", from: "4.7.0"),
        // HTML/XML → 文本
        .package(url: "https://github.com/scinfu/SwiftSoup", from: "2.9.0"),
        // GGUF chat template 渲染（Jinja2）
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.5.2"),
        // 官方 MCP Swift SDK（本地 vendor，见 Vendor/mcp-swift-sdk/README.md）
        .package(path: "Vendor/mcp-swift-sdk"),
        // BaseAgentCore 自定义 Transport 需要实现官方协议的 logger 字段
        .package(url: "https://github.com/apple/swift-log.git", from: "1.5.0"),
    ],
    targets: [
        // sqlite-vec v0.1.9 amalgamation，静态编入进程（SQLITE_CORE 模式）
        .target(
            name: "CSQLiteVec",
            path: "Sources/CSQLiteVec",
            cSettings: [
                .define("SQLITE_CORE"),
                .define("SQLITE_VEC_STATIC"),
            ],
            linkerSettings: [.linkedLibrary("sqlite3")]
        ),
        // llama.cpp 官方 XCFramework：进程内 GGUF 加载（LLM / VLM / Embedding）
        .binaryTarget(
            name: "llama",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b11815/llama-b11815-xcframework.zip",
            checksum: "a7d14f886fc7d15edcf505b84ddba2e05c1bf2222a8dc6d7bcf57b7810aa7174"
        ),
        // CGPDF C 级 API 的 Swift shim：提取扫描页内嵌整页 JPEG（跳过栅格化）
        .target(
            name: "CPDFShim",
            path: "Sources/CPDFShim"
        ),
        .target(
            name: "BaseAgentCore",
            dependencies: [
                "CSQLiteVec",
                "CPDFShim",
                "llama",
                .product(name: "Jinja", package: "swift-jinja"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXRerankers", package: "mlx-swift-lm"),
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
                .product(name: "ZIPFoundation", package: "ZipFoundation"),
                .product(name: "AEXML", package: "AEXML"),
                .product(name: "SwiftSoup", package: "SwiftSoup"),
                .product(name: "MCP", package: "mcp-swift-sdk"),
                .product(name: "Logging", package: "swift-log"),
            ],
            path: "Sources/BaseAgentCore",
            resources: [
                // 语义切片专用：potion-multilingual-128M int4 ONNX（约 99 MB），
                // 静态词向量模型，无 512 token 位置上限，随 App 固化分发。
                .copy("Resources/potion-multilingual-128M-int4"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)],
            linkerSettings: [.linkedLibrary("sqlite3")]
        ),
        .executableTarget(
            name: "BaseAgentApp",
            dependencies: ["BaseAgentCore"],
            path: "Sources/BaseAgentApp",
            resources: [.process("Resources")],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "BaseAgentCoreTests",
            dependencies: [
                "BaseAgentCore",
                .product(name: "ZIPFoundation", package: "ZipFoundation"),
                .product(name: "MCP", package: "mcp-swift-sdk"),
            ],
            path: "Tests/BaseAgentCoreTests",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
