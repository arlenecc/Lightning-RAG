#!/bin/bash
# 把 SwiftPM 产物打包成可分发的 Lightning-RAG.app。
#
# 用法：scripts/package-app.sh [debug|release]（默认 release）
# 版本：VERSION 环境变量可覆盖，默认 1.4.15
# 说明：应用本体不含任何依赖模型（potion 语义切片 / embedding / reranker 均不打进包），
#       启动时通过模型自动下载流程补齐。
# 产物：dist/Lightning-RAG.app（含 App 图标、运行时资源 bundle 与 llama 框架）
set -euo pipefail

CONFIG="${1:-release}"
VERSION="${VERSION:-1.4.15}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PRODUCTS="$ROOT/.build/out/Products/$([[ "$CONFIG" == "release" ]] && echo Release || echo Debug)"
BINARY="$PRODUCTS/Lightning-RAG"
DIST="$ROOT/dist"
APP="$DIST/Lightning-RAG.app"

if [[ ! -x "$BINARY" ]]; then
  echo "未找到可执行文件：$BINARY（先运行 swift build -c $CONFIG --disable-sandbox）" >&2
  exit 1
fi

# 打包前强制重建，确保产物反映当前源码（避免复用 stale 的 Release 二进制
# 导致新功能未进包）。
echo "构建 $CONFIG 产物…"
swift build -c "$CONFIG" --disable-sandbox

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$APP/Contents/Frameworks"

cp "$BINARY" "$APP/Contents/MacOS/Lightning-RAG"

# 去掉 Swift 调试符号：Release 二进制默认带完整符号表（本次实测 102M），
# strip -x 保留动态链接所需的外部符号，仅移除本地/调试符号，可降到 ~59M。
# 完整符号仍在 .build/out/Products/Release/Lightning-RAG.dSYM，需要调试时可用。
echo " stripping 调试符号…"
strip -x "$APP/Contents/MacOS/Lightning-RAG"

# 三种运行时第三方资源 bundle（Cmlx metallib / Hub / Crypto 等）。
# BaseAgent_BaseAgentCore.bundle 只装 potion 语义切片模型，不打进包，
# 由启动时的模型自动下载流程补齐。
for bundle in "$PRODUCTS"/*.bundle; do
  [[ -e "$bundle" ]] || continue
  if [[ "$(basename "$bundle")" == "BaseAgent_BaseAgentCore.bundle" ]]; then
    continue
  fi
  cp -R "$bundle" "$APP/Contents/Resources/"
done

# 动态框架：llama（GGUF 进程内推理）。onnxruntime 在本项目中静态链接，
# 不需要单独拷贝。
if [[ -e "$PRODUCTS/llama.framework" ]]; then
  cp -R "$PRODUCTS/llama.framework" "$APP/Contents/Frameworks/"
fi

# 图标：优先用项目根目录的 icns
if [[ -f "$ROOT/BaseAgent.icns" ]]; then
  cp "$ROOT/BaseAgent.icns" "$APP/Contents/Resources/AppIcon.icns"
fi

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Lightning-RAG</string>
  <key>CFBundleDisplayName</key><string>Lightning-RAG</string>
  <key>CFBundleExecutable</key><string>Lightning-RAG</string>
  <key>CFBundleIdentifier</key><string>com.baseagent.desktop</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundleVersion</key><string>${VERSION}</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.productivity</string>
</dict>
</plist>
PLIST

# 依赖的框架通过 @rpath 解析：可执行文件与框架都在包内。
install_name_tool -add_rpath "@executable_path/../Frameworks" "$APP/Contents/MacOS/Lightning-RAG" 2>/dev/null || true

# 本地运行需要放宽沙箱（模型/知识库位于外部卷），如要上架需替换为正式签名。
codesign --force --deep --sign - "$APP" 2>&1 | tail -2

echo "已生成：$APP"
du -sh "$APP"
