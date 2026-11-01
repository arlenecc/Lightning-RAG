#!/usr/bin/env python3
"""Phase 2: audit the two candidate summaries against the source + fact checklist."""
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

AUDIT_TEMPLATE = """你是资深中文图书编辑，负责审阅同一章节的两份摘要。只依据【章节原文】和【原文关键信息清单】判断，不要使用外部知识。

【章节标题】{title}

【章节原文】
{body}

【原文关键信息清单】
{key_points}

【原文易错细节】
{pitfalls}

【摘要A】
{summary_a}

【摘要B】
{summary_b}

请逐条核对，并严格输出一行 JSON：

1. point_status：对 A、B 各给 8 个关键信息点一个状态：
   - "covered"：实质内容正确且完整，换一种说法也算；
   - "partial"：提到了，但缺少关键要素、过于笼统或只覆盖一半；
   - "missing"：没有提到；
   - "contradicted"：提到了，但与原文冲突。
   特别注意：只写出术语、却把并列项写错（例如四圣谛的组成、八正道的组成、五蕴的组成）、把不同层级的概念混为一谈、年代/人物错配，都必须标 "contradicted"，不能因为出现了术语就标 covered。
2. errors：列出 A、B 中最重要的事实问题（最多各 5 条）。每条给 claim（摘要原话或紧贴原话的摘录）、type（"contradiction" 或 "unsupported"）、reason（原文如何说/为何无依据）。没有则空数组。
3. structure：给 A、B 的结构清晰度打 1–5 分。5=脉络清楚、层次合理；3=有层次但重点排布一般；1=杂乱难读。
4. flags：A、B 是否有 repetition（重复循环）、truncation（截断/句子未完）、extra_preamble（摘要前有多余说明），布尔值。
5. winner：综合两份摘要选 "A"、"B" 或 "tie"。
6. reason：2–3 句说明胜负关键，必须引用具体差异（覆盖了哪条、错了哪条、格式如何），不要空泛评价。

严格输出一行 JSON，不要代码块，不要解释性前后缀：
{{"point_status":{{"A":{{"1":"covered","2":"partial","3":"contradicted","4":"missing","5":"covered","6":"covered","7":"partial","8":"missing"}},"B":{{"1":"covered","2":"covered","3":"covered","4":"covered","5":"covered","6":"covered","7":"covered","8":"covered"}}}},"errors":{{"A":[{{"claim":"...","type":"contradiction","reason":"..."}}],"B":[]}},"structure":{{"A":3,"B":4}},"flags":{{"A":{{"repetition":false,"truncation":false,"extra_preamble":false}},"B":{{"repetition":false,"truncation":false,"extra_preamble":false}}}},"winner":"A","reason":"..."}}
"""

VALID_STATUS = {"covered", "partial", "missing", "contradicted"}


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
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object")
    data = json.loads(cleaned[start : end + 1])
    status = data.get("point_status", {})
    for label in ("A", "B"):
        mapping = status.get(label, {})
        if len(mapping) < 8:
            raise ValueError(f"incomplete point_status for {label}")
        for key, value in mapping.items():
            if value not in VALID_STATUS:
                raise ValueError(f"invalid status {label}.{key}={value}")
    if data.get("winner") not in ("A", "B", "tie"):
        raise ValueError("invalid winner")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", default="audit.json")
    parser.add_argument("--chapters", nargs="*", default=None)
    parser.add_argument("--max-tokens", type=int, default=1500)
    args = parser.parse_args()

    reference = json.loads((RESULTS / args.reference).read_text(encoding="utf-8"))
    ref_by_id = {c["chapter_id"]: c for c in reference["chapters"]}

    print(f"[audit] loading {args.judge}", flush=True)
    model, tokenizer = load(str(Path(args.judge).resolve()), lazy=False)
    mx.eval(model.parameters())

    chapter_ids = args.chapters or list(ref_by_id.keys())
    orders = [("0.8b", "2b"), ("2b", "0.8b")]
    comparisons = []
    for chapter_id in chapter_ids:
        chapter = load_chapter(chapter_id)
        ref = ref_by_id[chapter_id]["reference"]
        key_points = "\n".join(
            f"{i}. {point}" for i, point in enumerate(ref["key_points"], 1)
        )
        pitfalls = "\n".join(f"- {item}" for item in ref["pitfalls"])
        summaries = {tag: load_summary(tag, chapter_id) for tag in ("0.8b", "2b")}
        for first, second in orders:
            prompt = AUDIT_TEMPLATE.format(
                title=chapter["title"],
                body=chapter["body"],
                key_points=key_points,
                pitfalls=pitfalls,
                summary_a=summaries[first],
                summary_b=summaries[second],
            )
            print(
                f"[audit] {chapter_id} A={first} B={second} chars={len(prompt)}",
                flush=True,
            )
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
                        print(f"[audit] raw={raw[:1600]!r}", flush=True)
                        raise
                    print(f"[audit] parse failed ({error}), retrying once", flush=True)
                    parts = []
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
            comparisons.append(
                {
                    "chapter_id": chapter_id,
                    "chapter_title": chapter["title"],
                    "A_tag": first,
                    "B_tag": second,
                    "audit": data,
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
            statuses = data["point_status"]
            print(
                f"[audit] winner={data['winner']} "
                f"A={list(statuses['A'].values())} "
                f"B={list(statuses['B'].values())} "
                f"errors A={len(data.get('errors', {}).get('A', []))} "
                f"B={len(data.get('errors', {}).get('B', []))} "
                f"t={time.perf_counter() - started:.1f}s out={last.generation_tokens}",
                flush=True,
            )
            time.sleep(5)

    payload = {
        "judge_model": str(Path(args.judge).resolve()),
        "reference_file": args.reference,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparisons": comparisons,
    }
    out_path = RESULTS / args.out
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[audit] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
