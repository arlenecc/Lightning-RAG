#!/usr/bin/env python3
"""Phase 1: extract a fact checklist from each chapter with the 4B judge model."""
import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

REFERENCE_TEMPLATE = """你是资深中文图书编辑。请从下面的章节原文中提取“审稿时必须核对”的关键信息清单。只依据原文，不要使用外部知识。

【章节标题】{title}

【章节原文】
{body}

要求：
1. key_points：提取 8 条关键信息。每条 20–60 字，必须包含具体的专名、术语、年代、数字或明确的因果关系；不要写“本章介绍了……”这类空话。
2. pitfalls：列出 3 个原文中容易被误读或混淆的细节（例如并列概念被错误归类、年代/人物对应关系、同一术语的不同层次）。只列原文确实存在且容易被写错的地方。
3. 所有内容必须能在原文中找到依据，不得补充原文之外的史实。
4. 严格输出一行 JSON，不要代码块，不要解释性前后缀：
{{"key_points":["..."],"pitfalls":["..."]}}
"""


def load_chapter(chapter_id: str) -> dict:
    manifest = json.loads((DATA / "chapters.json").read_text(encoding="utf-8"))
    chapter = next(c for c in manifest if c["id"] == chapter_id)
    raw = (DATA / f"{chapter_id}.txt").read_text(encoding="utf-8")
    lines = raw.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    chapter["body"] = "\n".join(lines).strip()
    return chapter


def parse_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object")
    data = json.loads(cleaned[start : end + 1])
    if len(data.get("key_points", [])) < 5 or len(data.get("pitfalls", [])) < 2:
        raise ValueError("incomplete checklist")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", required=True)
    parser.add_argument("--out", default="reference.json")
    parser.add_argument("--chapters", nargs="*", default=None)
    args = parser.parse_args()

    print(f"[reference] loading {args.judge}", flush=True)
    model, tokenizer = load(str(Path(args.judge).resolve()), lazy=False)
    mx.eval(model.parameters())

    manifest = json.loads((DATA / "chapters.json").read_text(encoding="utf-8"))
    chapter_ids = args.chapters or [c["id"] for c in manifest]
    payload = {
        "judge_model": str(Path(args.judge).resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chapters": [],
    }
    for chapter_id in chapter_ids:
        chapter = load_chapter(chapter_id)
        prompt = REFERENCE_TEMPLATE.format(
            title=chapter["title"], body=chapter["body"]
        )
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        print(
            f"[reference] {chapter_id} prompt={len(prompt_ids)} tokens", flush=True
        )
        mx.reset_peak_memory()
        started = time.perf_counter()
        parts = []
        last = None
        for response in stream_generate(
            model,
            tokenizer,
            prompt_ids,
            max_tokens=1200,
            sampler=make_sampler(temp=0),
            prefill_step_size=2048,
        ):
            if response.text:
                parts.append(response.text)
            last = response
        raw = "".join(parts)
        print(f"[reference] finish={last.finish_reason} out={last.generation_tokens}", flush=True)
        print(f"[reference] raw={raw[:800]!r}", flush=True)
        data = parse_json(raw)
        payload["chapters"].append(
            {
                "chapter_id": chapter_id,
                "chapter_title": chapter["title"],
                "reference": data,
                "meta": {
                    "seconds": time.perf_counter() - started,
                    "prompt_tokens": last.prompt_tokens,
                    "generation_tokens": last.generation_tokens,
                    "prefill_tps": last.prompt_tps,
                    "generation_tps": last.generation_tps,
                    "raw": raw,
                },
            }
        )
        print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)
        time.sleep(5)

    out = RESULTS / args.out
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[reference] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
