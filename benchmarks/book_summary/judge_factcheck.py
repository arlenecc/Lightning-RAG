#!/usr/bin/env python3
"""Sentence-level fact check of candidate summaries with the 4B judge model."""
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

FACTCHECK_TEMPLATE = """你是资深中文图书编辑，负责事实核查。下面是一章原文、一份关键信息清单和一份摘要。请逐句核对摘要中的事实性陈述。只依据原文，不要使用外部知识。

【原文关键信息清单】
{key_points}

【章节原文】
{body}

【摘要】
{summary}

核查方法：
1. 把摘要中的事实性陈述逐句拆开，忽略“本章……”这类纯评价句和格式标题。
2. 对每个陈述，在原文中找出支持它的原句：有依据标 "supported"；原文没有依据标 "unsupported"；与原文矛盾标 "contradicted"。
3. 特别检查并列清单：如果摘要把“四圣谛”“八正道”“五蕴”“十二缘起”等的组成项写错、替换或归类错误，必须标 contradicted，并引用原文的正确表述。
4. 只输出有问题的陈述（contradicted/unsupported），最多 8 条；没有则 issues 为空数组。不要为了凑数把有依据的陈述标成问题。

输出一行 JSON：
{{"issues":[{{"claim":"摘要原句","type":"contradicted","source_quote":"原文正确表述或“原文无此内容”","reason":"..."}}],"checked_count":12}}

只输出 JSON。"""


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
    if not isinstance(data.get("issues"), list):
        raise ValueError("missing issues array")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--tags", nargs="+", default=["0.8b", "2b"])
    parser.add_argument("--chapters", nargs="*", default=None)
    parser.add_argument("--out", default="factcheck.json")
    parser.add_argument("--max-tokens", type=int, default=1200)
    args = parser.parse_args()

    reference = json.loads((RESULTS / args.reference).read_text(encoding="utf-8"))
    ref_by_id = {c["chapter_id"]: c for c in reference["chapters"]}
    manifest = json.loads((DATA / "chapters.json").read_text(encoding="utf-8"))
    chapter_ids = args.chapters or [c["id"] for c in manifest]

    print(f"[factcheck] loading {args.judge}", flush=True)
    model, tokenizer = load(str(Path(args.judge).resolve()), lazy=False)
    mx.eval(model.parameters())

    runs = []
    for chapter_id in chapter_ids:
        chapter = load_chapter(chapter_id)
        ref = ref_by_id[chapter_id]["reference"]
        key_points = "\n".join(
            f"{i}. {p}" for i, p in enumerate(ref["key_points"], 1)
        )
        for tag in args.tags:
            summary = (RESULTS / tag / f"{chapter_id}.md").read_text(
                encoding="utf-8"
            ).strip()
            prompt = FACTCHECK_TEMPLATE.format(
                key_points=key_points,
                body=chapter["body"],
                summary=summary,
            )
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            print(
                f"[factcheck] {chapter_id} {tag} prompt={len(prompt_ids)}",
                flush=True,
            )
            mx.reset_peak_memory()
            started = time.perf_counter()
            parts = []
            last = None
            for response in stream_generate(
                model,
                tokenizer,
                prompt_ids,
                max_tokens=args.max_tokens,
                sampler=make_sampler(temp=0),
                prefill_step_size=2048,
            ):
                if response.text:
                    parts.append(response.text)
                last = response
            raw = "".join(parts)
            attempt = 0
            while True:
                try:
                    data = parse_json(raw)
                    break
                except Exception as error:
                    attempt += 1
                    if attempt >= 2:
                        print(f"[factcheck] raw={raw[:1500]!r}", flush=True)
                        raise
                    print(f"[factcheck] parse failed ({error}), retrying", flush=True)
                    parts = []
                    for response in stream_generate(
                        model,
                        tokenizer,
                        prompt_ids,
                        max_tokens=1000,
                        sampler=make_sampler(temp=0),
                        prefill_step_size=2048,
                    ):
                        if response.text:
                            parts.append(response.text)
                        last = response
                    raw = "".join(parts)
            runs.append(
                {
                    "chapter_id": chapter_id,
                    "chapter_title": chapter["title"],
                    "tag": tag,
                    "result": data,
                    "meta": {
                        "seconds": time.perf_counter() - started,
                        "prompt_tokens": last.prompt_tokens,
                        "generation_tokens": last.generation_tokens,
                        "prefill_tps": last.prompt_tps,
                        "generation_tps": last.generation_tps,
                        "peak_memory_gb": last.peak_memory,
                        "finish_reason": last.finish_reason,
                        "raw": raw,
                    },
                }
            )
            print(
                f"[factcheck] {tag}: issues={len(data['issues'])} "
                f"checked={data.get('checked_count')} "
                f"t={time.perf_counter() - started:.1f}s",
                flush=True,
            )
            for issue in data["issues"]:
                print(
                    f"  - [{issue.get('type')}] {issue.get('claim','')[:100]}",
                    flush=True,
                )
            time.sleep(5)

    payload = {
        "judge_model": str(Path(args.judge).resolve()),
        "reference_file": args.reference,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
    }
    out = RESULTS / args.out
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[factcheck] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
