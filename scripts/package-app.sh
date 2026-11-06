#!/bin/bash
# 把 SwiftPM 产物打包成可分发的 Lightning-RAG.app。
#
# 用法：scripts/package-app.sh [debug|release]（默认 release）
# 版本：VERSION 环境变量可覆盖，默认 1.2.16
# 版本形态：VARIANT=clean|lite（默认 clean）
#   clean     不包含任何依赖模型（potion 资源 bundle 也不打），应用启动时
#             通过模型自动下载流程补齐（语义切片 / embedding / reranker 等）
#   lite      clean + 内置 potion 语义切片（资源 bundle）
#             + jina embedding + jina reranker（三个基础模型），
#             主模型、辅助模型与可能需要的 OCR 由启动下载或用户自行指定
# 产物：dist/Lightning-RAG.app（含 App 图标、资源 bundle 与 llama 框架）
set -euo pipefail

CONFIG="${1:-release}"
VERSION="${VERSION:-1.2.16}"
VARIANT="${VARIANT:-clean}"
case "$VARIANT" in
  clean|lite) ;;
  *)
    echo "错误：VARIANT 必须是 clean 或 lite（当前：${VARIANT}）" >&2
    exit 1
    ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PRODUCTS="$ROOT/.build/out/Products/$([[ "$CONFIG" == "release" ]] && echo Release || echo Debug)"
BINARY="$PRODUCTS/Lightning-RAG"
DIST="$ROOT/dist"
# 两个版本的应用本体完全相同，只有内置模型不同；App 名保持统一。
APP="$DIST/Lightning-RAG.app"

EMBEDDING_NAME="jina-embeddings-v5-text-nano-mlx-int4"
RERANKER_NAME="jina-reranker-v2-base-multilingual-onnx"

# 只复制推理运行时实际读取的文件，避免把示例脚本、prompts、assets 与缓存打进包。
copy_embedding_model() {
  local src="$ROOT/models/$EMBEDDING_NAME"
  local dst="$APP/Contents/Resources/models/$EMBEDDING_NAME"
  if [[ ! -d "$src" ]]; then
    echo "错误：VARIANT=$VARIANT 需要内置 embedding，但缺少模型目录 $src" >&2
    exit 1
  fi
  mkdir -p "$dst"
  local file
  for file in config.json model.safetensors weights.safetensors \
              tokenizer.json tokenizer_config.json README.md; do
    if [[ -f "$src/$file" ]]; then
      cp "$src/$file" "$dst/"
    fi
  done
  if [[ ! -f "$dst/model.safetensors" && ! -f "$dst/weights.safetensors" ]]; then
    echo "错误：embedding 缺少 model.safetensors / weights.safetensors：$src" >&2
    exit 1
  fi
  echo "已打包默认 embedding：$EMBEDDING_NAME"
}

copy_reranker_model() {
  local src="$ROOT/models/$RERANKER_NAME"
  local dst="$APP/Contents/Resources/models/$RERANKER_NAME"
  if [[ ! -d "$src" ]]; then
    echo "错误：VARIANT=$VARIANT 需要内置 reranker，但缺少模型目录 $src" >&2
    exit 1
  fi
  mkdir -p "$dst"
  local file
  for file in model_quantized.onnx tokenizer.json tokenizer_config.json \
              special_tokens_map.json config.json; do
    if [[ -f "$src/$file" ]]; then
      cp "$src/$file" "$dst/"
    fi
  done
  if [[ ! -f "$dst/model_quantized.onnx" ]]; then
    echo "错误：reranker 缺少 model_quantized.onnx：$src" >&2
    exit 1
  fi
  echo "已打包默认 reranker：$RERANKER_NAME"
}


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

# 三种运行时第三方资源 bundle（Cmlx metallib / Hub / Crypto 等）两个版本都要；
# BaseAgent_BaseAgentCore.bundle 只装 potion 语义切片模型，clean 版跳过，
# 由启动时的模型自动下载流程补齐。
for bundle in "$PRODUCTS"/*.bundle; do
  [[ -e "$bundle" ]] || continue
  if [[ "$VARIANT" == "clean" && "$(basename "$bundle")" == "BaseAgent_BaseAgentCore.bundle" ]]; then
    continue
  fi
  cp -R "$bundle" "$APP/Contents/Resources/"
done

case "$VARIANT" in
  lite)
    copy_embedding_model
    copy_reranker_model
    ;;
  clean)
    echo "clean 版本不包含任何依赖模型；启动时通过自动下载流程补齐"
    ;;
esac

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

echo "已生成（${VARIANT}）：$APP"
du -sh "$APP"
