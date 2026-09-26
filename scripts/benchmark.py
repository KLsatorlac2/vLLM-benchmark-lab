"""Unified benchmark controller and summary writer."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common import percentile, read_jsonl, write_jsonl


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [row for row in rows if row.get("type") != "metadata" and row.get("status") == "ok"]
    latency = [float(row["latency_ms"]) for row in samples if row.get("latency_ms") is not None]
    ttft = [float(row["ttft_ms"]) for row in samples if row.get("ttft_ms") is not None]
    output_tokens = sum(int(row.get("output_tokens", 0)) for row in samples)
    throughput = [float(row["output_tok_per_s"]) for row in samples if row.get("output_tok_per_s") is not None]
    return {
        "status": "ok" if samples else "failed",
        "requests": len(samples),
        "output_tokens": output_tokens,
        "mean_latency_ms": statistics.mean(latency) if latency else None,
        "p50_latency_ms": percentile(latency, 0.50),
        "p95_latency_ms": percentile(latency, 0.95),
        "p99_latency_ms": percentile(latency, 0.99),
        "mean_ttft_ms": statistics.mean(ttft) if ttft else None,
        "p95_ttft_ms": percentile(ttft, 0.95),
        "output_tok_per_s": statistics.mean(throughput) if throughput else None,
        "errors": [row.get("error") for row in rows if row.get("status") == "error"],
    }


def load_config(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Install pyyaml before using --config") from exc
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_command(config: dict[str, Any], output: str) -> list[str]:
    runner = config.get("runner", "vllm")
    script = "run_hf.py" if runner == "hf" else "run_vllm.py"
    command = [sys.executable, str(Path(__file__).with_name(script)), "--model", config["model"], "--output", output]
    aliases = {
        "max_model_len": "--max-model-len",
        "max_num_seqs": "--max-num-seqs",
        "max_tokens": "--max-tokens",
        "tensor_parallel_size": "--tensor-parallel-size",
        "gpu_memory_utilization": "--gpu-memory-utilization",
        "kv_cache_dtype": "--kv-cache-dtype",
    }
    boolean_flags = {"enable_prefix_caching": "--enable-prefix-caching", "enable_chunked_prefill": "--enable-chunked-prefill"}
    for key, value in config.items():
        if key in {"runner", "model", "repeats", "warmup", "output", "quantization", "revision", "dtype", "prompts"}:
            continue
        if key in boolean_flags and value:
            command.append(boolean_flags[key])
        elif key in aliases and value is not None:
            command.extend([aliases[key], str(value)])
    for key in ("revision", "dtype", "quantization", "prompts"):
        if config.get(key) is not None:
            command.extend([f"--{key.replace('_', '-')}", str(config[key])])
    command.extend(["--requests", str(config.get("requests", 4))])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(__file__).parents[1]
    run_id = time.strftime("%Y%m%d-%H%M%S")
    raw_dir = root / "results" / "raw" / run_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    repeats = int(config.get("repeats", 1))
    warmup = int(config.get("warmup", 1))

    for index in range(warmup):
        command = build_command(config, str(raw_dir / f"warmup-{index}.jsonl"))
        print("warmup:", " ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)

    summaries = []
    for index in range(repeats):
        output = raw_dir / f"repeat-{index}.jsonl"
        command = build_command(config, str(output))
        print("run:", " ".join(command))
        if args.dry_run:
            continue
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            summary = {"status": "failed", "error": completed.stderr[-2000:], "repeat": index}
        else:
            summary = summarize(read_jsonl(output))
            summary["repeat"] = index
        summaries.append(summary)

    if not args.dry_run:
        processed = root / "results" / "processed" / f"{run_id}.jsonl"
        write_jsonl(processed, [{"run_id": run_id, "config": config, **summary} for summary in summaries])
        print(json.dumps({"run_id": run_id, "processed": str(processed), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
