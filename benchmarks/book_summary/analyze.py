#!/usr/bin/env python3
"""Compute speed and objective format metrics from the model runs."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TAGS = ["0.8b", "2b"]
CHAPTERS = json.loads((HERE / "data" / "chapters.json").read_text(encoding="utf-8"))

SECTION_RE = re.compile(r"^##\s+(一句话主旨|核心要点|关键词|本章结论)\s*$", re.M)
BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.、])\s+\S", re.M)
CJK_RE = re.compile(r"[\u3400-\u9fff]")


def section_text(text: str, name: str) -> str:
    match = re.search(
        rf"^##\s+{name}\s*$([\s\S]*?)(?=^##\s+|\Z)", text, re.M
    )
    return match.group(1).strip() if match else ""


def normalize_line(line: str) -> str:
    line = re.sub(r"^\s*(?:[-*]|\d+[.、])\s*", "", line)
    line = re.sub(r"\s+", "", line)
    return line


def repetition_stats(text: str) -> dict:
    lines = [normalize_line(line) for line in text.splitlines()]
    lines = [line for line in lines if len(line) >= 20 and not line.startswith("#")]
    counts = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    repeated = {line: count for line, count in counts.items() if count > 1}
    return {
        "repeated_groups": len(repeated),
        "max_repeat": max(repeated.values(), default=1),
        "repeated_chars": sum(len(line) * count for line, count in repeated.items()),
        "sample": next(iter(repeated), ""),
    }


def format_metrics(text: str, finish_reason: str) -> dict:
    sections = SECTION_RE.findall(text)
    core = section_text(text, "核心要点")
    bullets = len(BULLET_RE.findall(core))
    keyword_text = section_text(text, "关键词")
    keyword_items = [
        item
        for item in re.split(r"[、,，;；]\s*", keyword_text)
        if item.strip()
    ]
    chars = len(CJK_RE.findall(text))
    first_heading = text.find("## ")
    extra_preamble = first_heading > 0 and bool(text[:first_heading].strip())
    rep = repetition_stats(text)
    last_char = text.rstrip()[-1:] or ""
    truncated = finish_reason == "length" or last_char not in "。！？.!?）」』】”\""
    points = 0.0
    points += 2.0 if len(set(sections)) >= 4 else 1.5 if len(set(sections)) == 3 else 1.0
    points += 1.0 if 4 <= bullets <= 6 else 0.0
    points += 1.0 if 3 <= len(keyword_items) <= 5 else 0.0
    points += 1.0 if 400 <= chars <= 700 else 0.0
    points += 1.0 if not extra_preamble else 0.0
    points += 1.0 if rep["max_repeat"] == 1 and not truncated else 0.0
    instruction = round(1 + 4 * (points / 7), 2)
    return {
        "sections": sections,
        "section_count": len(set(sections)),
        "bullets": bullets,
        "keywords": len(keyword_items),
        "keyword_list": [item.strip() for item in keyword_items],
        "chars": chars,
        "extra_preamble": extra_preamble,
        "repetition": rep,
        "truncated": truncated,
        "instruction_score": instruction,
    }


def main() -> None:
    payload = {"models": {}, "runs": []}
    for tag in TAGS:
        data = json.loads((RESULTS / f"{tag}.json").read_text(encoding="utf-8"))
        model = {
            "tag": tag,
            "model_path": data["model_path"],
            "load_seconds": data["load_seconds"],
            "warmup_seconds": data["warmup_seconds"],
            "load_peak_memory_gb": data["load_peak_memory_gb"],
            "hardware": data["hardware"],
        }
        payload["models"][tag] = model
        for run in data["runs"]:
            metrics = format_metrics(run["text"], run["finish_reason"])
            payload["runs"].append(
                {
                    "tag": tag,
                    "chapter_id": run["chapter_id"],
                    "chapter_title": run["chapter_title"],
                    "bucket": run["bucket"],
                    "chapter_chars": run["chapter_chars"],
                    "prompt_tokens": run["prompt_tokens"],
                    "generation_tokens": run["generation_tokens"],
                    "ttft_seconds": run["ttft_seconds"],
                    "total_seconds": run["total_seconds"],
                    "prefill_tps": run["prefill_tps"],
                    "generation_tps": run["generation_tps"],
                    "peak_memory_gb": run["peak_memory_gb"],
                    "finish_reason": run["finish_reason"],
                    "output_chars": len(run["text"]),
                    "format": metrics,
                }
            )

    # 汇总速度
    for tag in TAGS:
        runs = [r for r in payload["runs"] if r["tag"] == tag]
        model = payload["models"][tag]
        model["avg_prefill_tps"] = round(
            sum(r["prefill_tps"] for r in runs) / len(runs), 1
        )
        model["avg_decode_tps"] = round(
            sum(r["generation_tps"] for r in runs) / len(runs), 1
        )
        model["avg_ttft_seconds"] = round(
            sum(r["ttft_seconds"] for r in runs) / len(runs), 2
        )
        model["total_generation_seconds"] = round(
            sum(r["total_seconds"] for r in runs), 2
        )
        model["total_with_load_seconds"] = round(
            model["load_seconds"]
            + model["warmup_seconds"]
            + model["total_generation_seconds"],
            2
        )
        model["prompt_tokens_total"] = sum(r["prompt_tokens"] for r in runs)
        model["generation_tokens_total"] = sum(
            r["generation_tokens"] for r in runs
        )
    a = payload["models"]["0.8b"]
    b = payload["models"]["2b"]
    payload["ratios"] = {
        "prefill_2b_over_0.8b": round(b["avg_prefill_tps"] / a["avg_prefill_tps"], 2),
        "decode_2b_over_0.8b": round(b["avg_decode_tps"] / a["avg_decode_tps"], 2),
        "ttft_2b_over_0.8b": round(b["avg_ttft_seconds"] / a["avg_ttft_seconds"], 2),
        "total_2b_over_0.8b": round(
            b["total_with_load_seconds"] / a["total_with_load_seconds"], 2
        ),
    }

    (RESULTS / "analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("== speed ==")
    for tag in TAGS:
        m = payload["models"][tag]
        print(
            tag,
            "load", m["load_seconds"],
            "avg_ttft", m["avg_ttft_seconds"],
            "prefill", m["avg_prefill_tps"],
            "decode", m["avg_decode_tps"],
            "total", m["total_with_load_seconds"],
        )
    print("ratios", payload["ratios"])
    print("== format ==")
    for run in payload["runs"]:
        f = run["format"]
        print(
            run["tag"],
            run["chapter_id"],
            "sections", f["section_count"],
            "bullets", f["bullets"],
            "kw", f["keywords"],
            "chars", f["chars"],
            "maxrep", f["repetition"]["max_repeat"],
            "trunc", f["truncated"],
            "instr", f["instruction_score"],
        )


if __name__ == "__main__":
    main()
