#!/usr/bin/env python3
"""Run one MLX model over the benchmark chapters and record timing/output."""
import argparse
import json
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

from prompts import GEN_CONFIG, SYSTEM_PROMPT, USER_TEMPLATE

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def hardware() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    try:
        info["chip"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        info["memory_bytes"] = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        )
    except Exception:
        pass
    return info


def load_chapters() -> list[dict]:
    manifest = json.loads((DATA / "chapters.json").read_text(encoding="utf-8"))
    for chapter in manifest:
        path = DATA / f"{chapter['id']}.txt"
        raw = path.read_text(encoding="utf-8")
        # 去掉抽取时写入的标题行，避免与 prompt 中的标题重复。
        lines = raw.splitlines()
        if lines and lines[0].startswith("# "):
            lines = lines[1:]
        chapter["body"] = "\n".join(lines).strip()
        chapter["file"] = str(path)
    return manifest


def build_messages(chapter: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                title=chapter["title"], body=chapter["body"]
            ),
        },
    ]


def timed_generation(model, tokenizer, prompt_ids, max_tokens: int) -> dict:
    sampler = make_sampler(temp=GEN_CONFIG["temperature"])
    mx.reset_peak_memory()
    started = time.perf_counter()
    first_token_at = None
    text_parts = []
    last = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=sampler,
        prefill_step_size=GEN_CONFIG["prefill_step_size"],
    ):
        if first_token_at is None:
            first_token_at = time.perf_counter() - started
        if response.text:
            text_parts.append(response.text)
        last = response
    finished = time.perf_counter() - started
    if last is None:
        raise RuntimeError("generation produced no tokens")
    return {
        "text": "".join(text_parts),
        "prompt_tokens": last.prompt_tokens,
        "generation_tokens": last.generation_tokens,
        "ttft_seconds": first_token_at,
        "total_seconds": finished,
        "prefill_tps": last.prompt_tps,
        "generation_tps": last.generation_tps,
        "peak_memory_gb": last.peak_memory,
        "finish_reason": last.finish_reason,
        "peak_memory_bytes_after": int(mx.get_peak_memory()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-tokens", type=int, default=GEN_CONFIG["max_tokens"])
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    results_dir = HERE / "results" / args.tag
    results_dir.mkdir(parents=True, exist_ok=True)
    chapters = load_chapters()

    print(f"[{args.tag}] loading {model_path}", flush=True)
    load_started = time.perf_counter()
    model, tokenizer = load(str(model_path), lazy=False)
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - load_started
    load_peak_gb = mx.get_peak_memory() / 1e9
    print(
        f"[{args.tag}] loaded in {load_seconds:.2f}s, peak {load_peak_gb:.2f} GB",
        flush=True,
    )

    # 预热：触发 Metal kernel 编译，避免第一段计时被一次性编译成本污染。
    warm_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "用一句话说明什么是摘要。"}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    warm_started = time.perf_counter()
    for _ in stream_generate(
        model, tokenizer, warm_ids, max_tokens=16, sampler=make_sampler(temp=0)
    ):
        pass
    warmup_seconds = time.perf_counter() - warm_started
    print(f"[{args.tag}] warmup {warmup_seconds:.2f}s", flush=True)

    runs = []
    for index, chapter in enumerate(chapters, 1):
        messages = build_messages(chapter)
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=GEN_CONFIG["enable_thinking"],
        )
        print(
            f"[{args.tag}] ({index}/{len(chapters)}) {chapter['title']} "
            f"prompt={len(prompt_ids)} tokens",
            flush=True,
        )
        record = timed_generation(model, tokenizer, prompt_ids, args.max_tokens)
        record.update(
            {
                "chapter_id": chapter["id"],
                "chapter_title": chapter["title"],
                "bucket": chapter["bucket"],
                "chapter_chars": chapter["chars"],
            }
        )
        runs.append(record)
        (results_dir / f"{chapter['id']}.md").write_text(
            record["text"].strip() + "\n", encoding="utf-8"
        )
        print(
            f"[{args.tag}] done: ttft={record['ttft_seconds']:.2f}s "
            f"prefill={record['prefill_tps']:.1f} tok/s "
            f"decode={record['generation_tps']:.1f} tok/s "
            f"out={record['generation_tokens']} tokens "
            f"finish={record['finish_reason']} "
            f"peak={record['peak_memory_gb']:.2f} GB",
            flush=True,
        )
        time.sleep(5)

    payload = {
        "tag": args.tag,
        "model_path": str(model_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware(),
        "config": {
            **GEN_CONFIG,
            "max_tokens": args.max_tokens,
            "system_prompt": SYSTEM_PROMPT,
            "user_template": USER_TEMPLATE,
        },
        "load_seconds": load_seconds,
        "load_peak_memory_gb": load_peak_gb,
        "warmup_seconds": warmup_seconds,
        "process_max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "runs": runs,
    }
    out_file = HERE / "results" / f"{args.tag}.json"
    out_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[{args.tag}] wrote {out_file}", flush=True)


if __name__ == "__main__":
    main()
