// swift-tools-version:6.1
// Vendored copy of the official MCP Swift SDK (modelcontextprotocol/swift-sdk).
// Trimmed to the `MCP` library target only: the docc plugin, swift-nio and the
// MCPConformance executables from upstream are not needed here.
import PackageDescription

let package = Package(
    name: "mcp-swift-sdk",
    platforms: [
        .macOS("13.0"),
        .macCatalyst("16.0"),
        .iOS("16.0"),
        .watchOS("9.0"),
        .tvOS("16.0"),
        .visionOS("1.0"),
    ],
    products: [
        .library(
            name: "MCP",
            targets: ["MCP"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-system.git", from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.5.0"),
        .package(url: "https://github.com/mattt/eventsource.git", from: "1.1.0"),
    ],
    targets: [
        .target(
            name: "MCP",
            dependencies: [
                .product(name: "SystemPackage", package: "swift-system"),
                .product(name: "Logging", package: "swift-log"),
                .product(
                    name: "EventSource", package: "eventsource",
                    condition: .when(platforms: [.macOS, .iOS, .tvOS, .visionOS, .watchOS, .macCatalyst])),
            ],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency")
            ]
        ),
    ]
)
