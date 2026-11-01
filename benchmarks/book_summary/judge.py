#!/usr/bin/env python3
"""Use the 4B model as an anonymized pairwise judge for the chapter summaries."""
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

JUDGE_TEMPLATE = """你是资深中文图书编辑，正在审阅同一章节的两份摘要。请只依据下面的【章节原文】判断，不要使用任何外部知识，也不要因为摘要更长就给它更高分。

【章节原文】
{body}

【摘要A】
{summary_a}

【摘要B】
{summary_b}

请按以下维度分别给 A、B 打分（1–5 分，5 分最好）：
1. accuracy：事实准确性。人物、年代、概念、因果关系是否与原文一致，有无原文没有的内容。
2. coverage：内容覆盖。是否抓住章节主旨和关键论据，是否遗漏重要内容。
3. structure：结构表达。脉络是否清晰、层次是否合理、是否便于编辑阅读。
4. instruction：指令遵循。是否包含“一句话主旨 / 核心要点 / 关键词 / 本章结论”四部分，核心要点是否 4–6 条，关键词是否 3–5 个，总字数是否接近 400–700 字，有无多余说明或重复。
5. overall：综合质量。

要求：
- 必须同时给出 A 和 B 的分数。
- winner 只能是 "A"、"B" 或 "tie"。
- 指出每份摘要最具体的问题（例如编造、重复、遗漏、格式缺失），没有明显问题则写“无明显问题”。
- 严格输出一行 JSON，不要代码块，不要解释性前后缀：
{{"A":{{"accuracy":1,"coverage":1,"structure":1,"instruction":1,"overall":1}},"B":{{"accuracy":1,"coverage":1,"structure":1,"instruction":1,"overall":1}},"winner":"A","reason":"...","problems_A":"...","problems_B":"..."}}
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


def load_summary(tag: str, chapter_id: str) -> str:
    return (RESULTS / tag / f"{chapter_id}.md").read_text(encoding="utf-8").strip()


def parse_json(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in judge output")
    return json.loads(cleaned[start : end + 1])


def normalize_scores(data: dict) -> dict:
    for label in ("A", "B"):
        scores = data[label]
        for key in ("accuracy", "coverage", "structure", "instruction", "overall"):
            value = float(scores[key])
            if not 1 <= value <= 5:
                raise ValueError(f"{label}.{key} out of range: {value}")
            scores[key] = value
    if data["winner"] not in ("A", "B", "tie"):
        raise ValueError(f"invalid winner: {data['winner']}")
    return data


def judge_once(model, tokenizer, prompt: str, max_tokens: int) -> tuple[dict, dict]:
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    mx.reset_peak_memory()
    started = time.perf_counter()
    parts = []
    last = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=0),
        prefill_step_size=2048,
    ):
        if response.text:
            parts.append(response.text)
        last = response
    elapsed = time.perf_counter() - started
    raw = "".join(parts)
    parsed = normalize_scores(parse_json(raw))
    meta = {
        "prompt_tokens": last.prompt_tokens,
        "generation_tokens": last.generation_tokens,
        "seconds": elapsed,
        "prefill_tps": last.prompt_tps,
        "generation_tps": last.generation_tps,
        "peak_memory_gb": last.peak_memory,
        "finish_reason": last.finish_reason,
        "raw": raw,
    }
    return parsed, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", required=True)
    parser.add_argument("--out", default="judge.json")
    parser.add_argument("--chapters", nargs="*", default=None)
    parser.add_argument("--max-tokens", type=int, default=600)
    args = parser.parse_args()

    model_path = Path(args.judge).resolve()
    print(f"[judge] loading {model_path}", flush=True)
    load_started = time.perf_counter()
    model, tokenizer = load(str(model_path), lazy=False)
    mx.eval(model.parameters())
    print(f"[judge] loaded in {time.perf_counter() - load_started:.2f}s", flush=True)

    manifest = json.loads((DATA / "chapters.json").read_text(encoding="utf-8"))
    chapter_ids = args.chapters or [c["id"] for c in manifest]
    # 两个顺序各评一次，用于消除 A/B 位置偏差。
    orders = [("0.8b", "2b"), ("2b", "0.8b")]
    comparisons = []
    for chapter_id in chapter_ids:
        chapter = load_chapter(chapter_id)
        summaries = {tag: load_summary(tag, chapter_id) for tag in ("0.8b", "2b")}
        for first, second in orders:
            prompt = JUDGE_TEMPLATE.format(
                body=chapter["body"],
                summary_a=summaries[first],
                summary_b=summaries[second],
            )
            print(
                f"[judge] {chapter_id} A={first} B={second} "
                f"chars={len(prompt)}",
                flush=True,
            )
            attempt = 0
            while True:
                try:
                    parsed, meta = judge_once(
                        model, tokenizer, prompt, args.max_tokens
                    )
                    break
                except Exception as error:
                    attempt += 1
                    if attempt >= 2:
                        raise
                    print(f"[judge] parse/run failed ({error}), retrying once", flush=True)
            comparisons.append(
                {
                    "chapter_id": chapter_id,
                    "chapter_title": chapter["title"],
                    "A_tag": first,
                    "B_tag": second,
                    "scores": parsed,
                    "meta": meta,
                }
            )
            print(
                f"[judge] winner={parsed['winner']} "
                f"A_overall={parsed['A']['overall']} B_overall={parsed['B']['overall']} "
                f"t={meta['seconds']:.1f}s out={meta['generation_tokens']}",
                flush=True,
            )
            time.sleep(5)

    payload = {
        "judge_model": str(model_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparisons": comparisons,
    }
    out_path = RESULTS / args.out
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[judge] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
