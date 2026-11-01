#!/usr/bin/env python3
"""Extract three chapters from 《印度佛教史》 EPUB for the summary benchmark."""
import json
import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import epub

warnings.filterwarnings("ignore")

SOURCE = Path("/Users/arlenecc/Downloads/test/印度佛教史.epub")
OUT = Path(__file__).parent / "data"
CHAPTERS = [
    ("index_split_013.html", "第一章 第四节 教理", "medium"),
    ("index_split_026.html", "第三章 第一节 阿育王以后的教团发展", "long"),
    ("index_split_036.html", "第四章 第五节 唯识的教理", "xlong"),
]

PAGE_MARKER = re.compile(r"^[（(]\s*\d+\s*[）)]\s*[上下]?$")
STOP_HEADINGS = {"参考书目", "注", "注释", "参考文献"}


def tidy(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v\u00a0]+", " ", text).strip()
    # 中文排版中，span/anchor 拆分产生的多余空格应去掉。
    text = re.sub(r"(?<=[\u3400-\u9fff，。；：、！？）」』】》])\s+", "", text)
    text = re.sub(r"\s+(?=[\u3400-\u9fff，。；：、！？）」』】》])", "", text)
    text = re.sub(r"\s+([，。；：、！？])", r"\1", text)
    return text


def clean(html: bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    paragraphs = []
    for p in soup.find_all("p"):
        cls = set(p.get("class", []))
        text = tidy(p.get_text(" ", strip=True))
        if not text or PAGE_MARKER.match(text):
            continue
        if text in STOP_HEADINGS:
            break
        # calibre_17/26 是页码标记，已在上面按内容过滤，这里再兜底。
        if "calibre_17" in cls or "calibre_26" in cls:
            continue
        paragraphs.append(text)
    return "\n\n".join(paragraphs)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    book = epub.read_epub(str(SOURCE))
    items = {item.get_name(): item for item in book.get_items_of_type(9)}
    manifest = []
    for name, title, bucket in CHAPTERS:
        text = clean(items[name].get_content())
        path = OUT / f"{Path(name).stem}.txt"
        path.write_text(f"# {title}\n\n{text}\n", encoding="utf-8")
        manifest.append(
            {
                "id": Path(name).stem,
                "title": title,
                "bucket": bucket,
                "chars": len(text),
                "source": str(SOURCE),
            }
        )
        print(f"{title}: {len(text)} chars -> {path}")

    (OUT / "chapters.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
