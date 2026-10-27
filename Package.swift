// swift-tools-version: 6.2

// Lashline's face tracker and liner shape as a Swift package, for apps on iPhone and Mac.
// The OBS plugin builds the same C++ files with CMake instead.

import PackageDescription

let package = Package(
    name: "Lashline",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "LashlineCore", targets: ["LashlineCore"]),
    ],
    targets: [
        .target(
            name: "LashlineCore",
            dependencies: ["onnxruntime"],
            path: "src",
            sources: ["lashline-core.cpp", "face-tracker.cpp", "one-euro-filter.cpp", "eyeliner-shape.cpp"],
            publicHeadersPath: "include",
            cxxSettings: [.define("LASHLINE_LINK_ONNXRUNTIME")],
            linkerSettings: [.linkedFramework("CoreML"), .linkedFramework("Foundation"), .linkedFramework("Network")]
        ),
        // ONNX Runtime's prebuilt iPhone/Mac framework, from its official CocoaPods archive.
        .binaryTarget(
            name: "onnxruntime",
            url: "https://download.onnxruntime.ai/pod-archive-onnxruntime-c-1.30.0.zip",
            checksum: "e6f1670c14406fd9f082bb400ab197a9b0a9646058ca6366e440642e2b54a2ea"
        ),
        .executableTarget(
            name: "lashline-track-image",
            dependencies: ["LashlineCore"],
            path: "tests/track-image"
        ),
    ],
    cxxLanguageStandard: .cxx17
)
