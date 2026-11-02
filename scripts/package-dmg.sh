#!/bin/bash
# 把 dist/Lightning-RAG.app 打包成可分发的 DMG。
#
# 用法：scripts/package-dmg.sh [版本]（默认读取 VERSION 环境变量，否则 1.2.10）
# 版本形态：VARIANT=clean|lite（默认 clean）
# 两个版本共用 dist/Lightning-RAG.app，只有 DMG 文件名带版本形态后缀。
# 产物：dist/Lightning-RAG-<版本>-<variant>.dmg（含 App 与 Applications 快捷方式）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${VERSION:-${1:-1.2.10}}"
VARIANT="${VARIANT:-clean}"
case "$VARIANT" in
  clean) LABEL="Clean" ;;
  lite) LABEL="Lite" ;;
  *)
    echo "错误：VARIANT 必须是 clean 或 lite（当前：${VARIANT}）" >&2
    exit 1
    ;;
esac

APP="$ROOT/dist/Lightning-RAG.app"
DMG="$ROOT/dist/Lightning-RAG-$VERSION-$VARIANT.dmg"
VOLUME_NAME="Lightning-RAG $LABEL $VERSION"

if [[ ! -d "$APP" ]]; then
  echo "未找到 App：$APP（先运行 VARIANT=$VARIANT scripts/package-app.sh release）" >&2
  exit 1
fi

STAGE="$(mktemp -d /tmp/lightning-rag-dmg.XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT

cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"

hdiutil create -volname "$VOLUME_NAME" -srcfolder "$STAGE" \
  -ov -format UDZO "$DMG"
hdiutil verify "$DMG"

echo "已生成：$DMG"
du -sh "$DMG"
