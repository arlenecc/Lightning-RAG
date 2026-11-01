#!/bin/bash
# 把 SwiftPM 产物打包成可分发的 Lightning-RAG.app。
#
# 用法：scripts/package-app.sh [debug|release]（默认 release）
# 版本：VERSION 环境变量可覆盖，默认 1.2.7
# 版本形态：VARIANT=clean|lite|all-in-one（默认 lite；ALL_IN_ONE=1 等价于 all-in-one）
#   clean     仅内置 potion-multilingual-128M int4（随 SwiftPM 资源 bundle 分发），
#             不含 embedding / reranker / 主模型 / 辅助模型 / OCR
#   lite      clean + jina embedding + jina reranker（三个基础模型），
#             主模型、辅助模型与可能需要的 OCR 由用户自行指定
#   all-in-one lite + 主模型 + 辅助模型 + OvisOCR3，开箱即用
# 产物：dist/Lightning-RAG.app（含 App 图标、资源 bundle 与 llama 框架）
set -euo pipefail

CONFIG="${1:-release}"
VERSION="${VERSION:-1.2.7}"
VARIANT="${VARIANT:-}"
if [[ -z "$VARIANT" && "${ALL_IN_ONE:-0}" == "1" ]]; then
  VARIANT="all-in-one"
fi
VARIANT="${VARIANT:-lite}"
case "$VARIANT" in
  clean|lite|all-in-one) ;;
  *)
    echo "错误：VARIANT 必须是 clean、lite 或 all-in-one（当前：${VARIANT}）" >&2
    exit 1
    ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PRODUCTS="$ROOT/.build/out/Products/$([[ "$CONFIG" == "release" ]] && echo Release || echo Debug)"
BINARY="$PRODUCTS/Lightning-RAG"
DIST="$ROOT/dist"
# 三种版本的应用本体完全相同，只有内置模型不同；App 名保持统一。
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

# 主模型 / 辅助模型 / OCR：只复制推理运行时读取的文件。
copy_runtime_model() {
  local name="$1"
  local src="$ROOT/models/$name"
  local dst="$APP/Contents/Resources/models/$name"
  if [[ ! -d "$src" ]]; then
    echo "错误：VARIANT=all-in-one 但缺少模型目录 $src" >&2
    exit 1
  fi
  mkdir -p "$dst"
  local files=(config.json generation_config.json chat_template.jinja \
    tokenizer.json tokenizer_config.json tokenizer.model special_tokens_map.json \
    vocab.json merges.txt model.safetensors model.safetensors.index.json \
    preprocessor_config.json processor_config.json video_preprocessor_config.json)
  local file
  for file in "${files[@]}"; do
    if [[ -f "$src/$file" ]]; then
      cp "$src/$file" "$dst/"
    fi
  done
  if [[ ! -f "$dst/model.safetensors" ]]; then
    echo "错误：模型缺少 model.safetensors：$src" >&2
    exit 1
  fi
  echo "已打包模型：$name ($(du -sh "$dst" | awk '{print $1}'))"
}

if [[ ! -x "$BINARY" ]]; then
  echo "未找到可执行文件：$BINARY（先运行 swift build -c $CONFIG --disable-sandbox）" >&2
  exit 1
fi

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$APP/Contents/Frameworks"

cp "$BINARY" "$APP/Contents/MacOS/Lightning-RAG"

# SwiftPM 资源 bundle：Bundle.module 会依次查找 Bundle.main.resourceURL
# （即 Contents/Resources）与 Bundle.main.bundleURL，因此放在 Resources 下。
# potion-multilingual-128M int4（语义切片）也在其中，三种版本都会带上。
for bundle in "$PRODUCTS"/*.bundle; do
  [[ -e "$bundle" ]] || continue
  cp -R "$bundle" "$APP/Contents/Resources/"
done

case "$VARIANT" in
  lite|all-in-one)
    copy_embedding_model
    copy_reranker_model
    ;;
  clean)
    echo "clean 版本仅内置 potion 语义切片模型；未打包 embedding / reranker / 主模型 / 辅助模型 / OCR"
    ;;
esac

if [[ "$VARIANT" == "all-in-one" ]]; then
  copy_runtime_model "qwen3.5-4b-agentic-coder-v4-4bit-mlx"
  copy_runtime_model "Qwen3.5-2B-MLX-4bit"
  copy_runtime_model "OvisOCR3-MLX-4bit-mixed46"
fi

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
