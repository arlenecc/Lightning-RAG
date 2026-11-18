#!/bin/bash
# 把 dist/Lightning-RAG.app 打包成可分发的 DMG。
#
# 用法：scripts/package-dmg.sh [版本]（默认读取 VERSION 环境变量，否则 1.3.30）
# 产物：dist/Lightning-RAG-<版本>.dmg（含 App 与 Applications 快捷方式）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${VERSION:-${1:-1.3.30}}"

APP="$ROOT/dist/Lightning-RAG.app"
DMG="$ROOT/dist/Lightning-RAG-$VERSION.dmg"
VOLUME_NAME="Lightning-RAG $VERSION"

if [[ ! -d "$APP" ]]; then
  echo "未找到 App：$APP（先运行 scripts/package-app.sh release）" >&2
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
